// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import dev.vortex.spark.io.VortexIo;
import dev.vortex.spark.read.VortexBytesReadMetric;
import dev.vortex.spark.read.VortexPartitionReaderFactory;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.EnumSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;
import org.apache.hadoop.conf.Configuration;
import org.apache.spark.paths.SparkPath;
import org.apache.spark.sql.catalyst.FileSourceOptions;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.catalog.SupportsRead;
import org.apache.spark.sql.connector.catalog.Table;
import org.apache.spark.sql.connector.catalog.TableCapability;
import org.apache.spark.sql.connector.catalog.TableProvider;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.connector.metric.CustomMetric;
import org.apache.spark.sql.connector.metric.CustomTaskMetric;
import org.apache.spark.sql.connector.read.Batch;
import org.apache.spark.sql.connector.read.InputPartition;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.connector.read.PartitionReaderFactory;
import org.apache.spark.sql.connector.read.Scan;
import org.apache.spark.sql.connector.read.ScanBuilder;
import org.apache.spark.sql.execution.datasources.PartitionedFile;
import org.apache.spark.sql.sources.Filter;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;
import org.apache.spark.sql.vectorized.ColumnarBatch;

/**
 * A V2 source that reads every Vortex file under {@code path} in one task, with a new reader factory per file, the way
 * a caller that picks its own files and filters does.
 */
public final class OneFileAtATimeSource implements TableProvider {
    static final StructType SCHEMA = new StructType().add("id", DataTypes.LongType);

    @Override
    public StructType inferSchema(CaseInsensitiveStringMap options) {
        return SCHEMA;
    }

    @Override
    public Table getTable(StructType schema, Transform[] partitioning, Map<String, String> properties) {
        List<String> files = vortexFiles(properties.get("path"));
        return new SourceTable(files);
    }

    private static List<String> vortexFiles(String directory) {
        try (Stream<Path> listed = Files.list(Path.of(directory))) {
            return listed.map(Path::toString)
                    .filter(f -> f.endsWith(".vortex"))
                    .sorted()
                    .toList();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private record SourceTable(List<String> files) implements SupportsRead {
        @Override
        public String name() {
            return "one-file-at-a-time";
        }

        @Override
        public StructType schema() {
            return SCHEMA;
        }

        @Override
        public Set<TableCapability> capabilities() {
            return EnumSet.of(TableCapability.BATCH_READ);
        }

        @Override
        public ScanBuilder newScanBuilder(CaseInsensitiveStringMap options) {
            return () -> new SourceScan(files);
        }
    }

    private record SourceScan(List<String> files) implements Scan, Batch {
        @Override
        public StructType readSchema() {
            return SCHEMA;
        }

        @Override
        public Batch toBatch() {
            return this;
        }

        @Override
        public CustomMetric[] supportedCustomMetrics() {
            return new CustomMetric[] {new VortexBytesReadMetric()};
        }

        @Override
        public InputPartition[] planInputPartitions() {
            return new InputPartition[] {new FileList(files)};
        }

        @Override
        public PartitionReaderFactory createReaderFactory() {
            return partition -> new FilesReader(((FileList) partition).paths());
        }
    }

    private record FileList(List<String> paths) implements InputPartition {}

    private static final class FilesReader implements PartitionReader<InternalRow> {
        private final List<String> paths;
        private final VortexIo io;
        private final VortexOptions options = VortexOptions.of(Map.of());
        private int next;
        private PartitionReader<ColumnarBatch> file;
        private Iterator<InternalRow> rows;

        private FilesReader(List<String> paths) {
            this.paths = paths;
            Configuration conf = new Configuration();
            conf.set("fs.file.impl", VortexNestedSchemaPruningIoTest.CountingLocalFileSystem.class.getName());
            conf.set("fs.file.impl.disable.cache", "true");
            this.io = VortexIo.create(options, conf);
        }

        @Override
        public boolean next() throws IOException {
            while (rows == null || !rows.hasNext()) {
                if (file != null && file.next()) {
                    rows = file.get().rowIterator();
                    continue;
                }
                if (next == paths.size()) {
                    return false;
                }
                closeFile();
                file = open(paths.get(next++));
                rows = null;
            }
            return true;
        }

        private PartitionReader<ColumnarBatch> open(String path) {
            long length = Path.of(path).toFile().length();
            PartitionedFile partitioned = new PartitionedFile(
                    InternalRow.empty(),
                    SparkPath.fromPathString("file://" + path),
                    0,
                    length,
                    new String[0],
                    0,
                    length,
                    PartitionedFile.apply$default$8());
            VortexPartitionReaderFactory factory = new VortexPartitionReaderFactory(
                    new FileSourceOptions(scala.collection.immutable.Map$.MODULE$.empty()),
                    io,
                    options,
                    SCHEMA,
                    SCHEMA,
                    new StructType(),
                    new Filter[0],
                    false);
            return factory.buildColumnarReader(partitioned);
        }

        @Override
        public InternalRow get() {
            return rows.next().copy();
        }

        @Override
        public CustomTaskMetric[] currentMetricsValues() {
            return file == null ? new CustomTaskMetric[0] : file.currentMetricsValues();
        }

        @Override
        public void close() throws IOException {
            closeFile();
        }

        private void closeFile() throws IOException {
            if (file != null) {
                file.close();
            }
        }
    }
}
