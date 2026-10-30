// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import com.google.common.collect.ImmutableMap;
import dev.vortex.api.DataSource;
import dev.vortex.api.Session;
import dev.vortex.spark.VortexFilePartition;
import dev.vortex.spark.VortexFilePartition.RowRange;
import dev.vortex.spark.VortexSparkSession;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.apache.spark.sql.connector.catalog.CatalogV2Util;
import org.apache.spark.sql.connector.catalog.Column;
import org.apache.spark.sql.connector.expressions.filter.Predicate;
import org.apache.spark.sql.connector.read.Batch;
import org.apache.spark.sql.connector.read.InputPartition;
import org.apache.spark.sql.connector.read.PartitionReaderFactory;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

/** Execution source for batch scans of Vortex file tables. */
public final class VortexBatchExec implements Batch {
    private final List<String> paths;
    private final StructType readSchema;
    private final StructType tableSchema;
    private final Map<String, String> formatOptions;
    private final Predicate[] pushedPredicates;
    private final PartitionPruner partitionPruner;
    private List<String> resolvedPaths;

    /**
     * Creates a new VortexBatchExec for scanning the specified Vortex files.
     *
     * @param paths the list of file paths to scan
     * @param columns the list of columns to read from the files
     * @param tableColumns the full table columns, which {@code columns} may prune nested fields from
     * @param pushedPredicates predicates pushed down by Spark; converted to a single Vortex filter expression at read
     *     time
     */
    public VortexBatchExec(
            List<String> paths,
            List<Column> columns,
            List<Column> tableColumns,
            Map<String, String> formatOptions,
            Predicate[] pushedPredicates) {
        this(paths, columns, tableColumns, formatOptions, pushedPredicates, null);
    }

    /**
     * Creates a new VortexBatchExec that skips files whose Hive-style partition values cannot satisfy
     * {@code partitionFilters}. Spark still evaluates those filters after the scan.
     */
    public VortexBatchExec(
            List<String> paths,
            List<Column> columns,
            List<Column> tableColumns,
            Map<String, String> formatOptions,
            Predicate[] pushedPredicates,
            Predicate[] partitionFilters) {
        this.paths = List.copyOf(paths);
        this.readSchema = CatalogV2Util.v2ColumnsToStructType(columns.toArray(new Column[0]));
        this.tableSchema = CatalogV2Util.v2ColumnsToStructType(tableColumns.toArray(new Column[0]));
        this.formatOptions = Map.copyOf(formatOptions);
        this.pushedPredicates = pushedPredicates == null ? new Predicate[0] : pushedPredicates.clone();
        this.partitionPruner = new PartitionPruner(partitionFilters, tableSchema);
    }

    /**
     * Plans the input partitions for this batch scan.
     *
     * <p>Directory-like entries are expanded to concrete {@code .vortex} files, and files whose partition values cannot
     * satisfy the partition filters are dropped. Each remaining file becomes one {@link VortexFilePartition}; the
     * partition carries the paths the reader should open, the requested schema, and any Hive-style partition values
     * parsed out of the path.
     *
     * <p>With {@code vortex.read.splitFiles=true} a file larger than the split size becomes several partitions covering
     * disjoint row ranges (see {@link FileSplitPlanner}). That reads every file's footer on the driver to learn its row
     * count and size, so it is off by default.
     */
    @Override
    public InputPartition[] planInputPartitions() {
        resolvedPaths = resolvePaths();
        Map<String, FileSplitPlanner.FileSize> sizes =
                FileSplitPlanner.enabled(formatOptions) ? measureFiles(resolvedPaths) : Map.of();
        long maxSplitBytes = sizes.isEmpty() ? 0 : FileSplitPlanner.maxSplitBytes(formatOptions, sizes.values());

        List<InputPartition> partitions = new ArrayList<>();
        for (String path : resolvedPaths) {
            Map<String, String> partVals = ImmutableMap.copyOf(PartitionPathUtils.parsePartitionValues(path));
            FileSplitPlanner.FileSize size = sizes.get(path);
            List<RowRange> ranges = size == null ? List.of() : FileSplitPlanner.splitRows(size, maxSplitBytes);
            if (ranges.isEmpty()) {
                partitions.add(new VortexFilePartition(List.of(path), readSchema, formatOptions, partVals));
            }
            for (RowRange range : ranges) {
                partitions.add(new VortexFilePartition(List.of(path), readSchema, formatOptions, partVals, range));
            }
        }
        return partitions.toArray(new InputPartition[0]);
    }

    /**
     * Opens each file to read its exact row count and on-storage size from the footer. Files whose count is not exact
     * or whose size is unknown are left out and read whole: a row range past the end of a file is a scan error.
     */
    private Map<String, FileSplitPlanner.FileSize> measureFiles(List<String> paths) {
        Session session = VortexSparkSession.get(formatOptions);
        Map<String, FileSplitPlanner.FileSize> sizes = new HashMap<>();
        for (String path : paths) {
            DataSource source = DataSource.open(session, path, formatOptions);
            OptionalLong bytes = source.byteSize().asOptional();
            if (source.rowCount() instanceof DataSource.RowCount.Exact exact && bytes.isPresent()) {
                sizes.put(path, new FileSplitPlanner.FileSize(exact.value(), bytes.getAsLong()));
            }
        }
        return sizes;
    }

    @Override
    public PartitionReaderFactory createReaderFactory() {
        List<String> files = resolvedPaths != null ? resolvedPaths : resolvePaths();
        Set<String> partitionColumns = collectPartitionColumnNames(files);
        StructType dataReadSchema = new StructType(Arrays.stream(readSchema.fields())
                .filter(field -> !partitionColumns.contains(field.name()))
                .toArray(StructField[]::new));
        // Only the read columns' table types are needed to build the projection.
        StructType dataTableSchema = new StructType(Arrays.stream(dataReadSchema.fieldNames())
                .map(tableSchema::apply)
                .toArray(StructField[]::new));
        return new VortexPartitionReaderFactory(dataReadSchema, dataTableSchema, formatOptions, pushedPredicates);
    }

    private List<String> resolvePaths() {
        return partitionPruner.prune(resolveVortexPaths(VortexSparkSession.get(formatOptions), paths, formatOptions));
    }

    /**
     * Expands directory-like entries to the data files beneath them, skipping Spark hidden files such as
     * {@code _SUCCESS} (see {@link VortexFileListing}); entries that already name a {@code .vortex} file are kept
     * as-is. Shared with {@link VortexScan#estimateStatistics()} so planning and execution resolve paths identically.
     */
    static List<String> resolveVortexPaths(Session session, List<String> paths, Map<String, String> formatOptions) {
        return paths.stream()
                .flatMap(path -> path.endsWith(".vortex")
                        ? Stream.of(path)
                        : VortexFileListing.listDataFiles(session, path, formatOptions).stream())
                .collect(Collectors.toList());
    }

    private static Set<String> collectPartitionColumnNames(List<String> files) {
        Set<String> all = new HashSet<>();
        for (String path : files) {
            all.addAll(PartitionPathUtils.parsePartitionValues(path).keySet());
        }
        return all;
    }
}
