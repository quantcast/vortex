// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.api.DataSource;
import dev.vortex.api.DataSource.RowCount;
import dev.vortex.api.Expression;
import dev.vortex.api.Partition;
import dev.vortex.api.Scan;
import dev.vortex.api.ScanOptions;
import dev.vortex.api.Session;
import dev.vortex.arrow.ArrowAllocation;
import dev.vortex.io.NativeReadable;
import dev.vortex.relocated.org.apache.arrow.memory.BufferAllocator;
import dev.vortex.relocated.org.apache.arrow.vector.VectorSchemaRoot;
import dev.vortex.relocated.org.apache.arrow.vector.ipc.ArrowReader;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.ArrowType;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Field;
import dev.vortex.spark.VortexOptions;
import dev.vortex.spark.VortexSparkSession;
import dev.vortex.spark.io.VortexFile;
import dev.vortex.spark.io.VortexIo;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.TreeMap;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.execution.datasources.PartitionedFile;
import org.apache.spark.sql.execution.vectorized.ConstantColumnVector;
import org.apache.spark.sql.sources.Filter;
import org.apache.spark.sql.types.ArrayType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarBatch;

/** Columnar reader over one Spark {@link PartitionedFile}. */
public final class VortexPartitionReader implements PartitionReader<ColumnarBatch> {
    private final PartitionedFile file;
    private final StructType readDataSchema;
    private final StructType readPartitionSchema;
    private final BufferAllocator allocator;

    /**
     * For each field of {@code readDataSchema}, its position among the columns this file returns, or {@code -1} for a
     * field the file does not carry. A merged dataset schema holds every field any file carries, so a file written
     * before a column was added returns fewer columns than the query asked for.
     */
    private final int[] vectorSlots;

    /** For each field of {@code readDataSchema}, how to lay it out as Spark reads it, or {@code null} as returned. */
    private final NestedSchemaPruning.Shape[] columnShapes;

    private NativeReadable readable;
    private Session session;
    private DataSource dataSource;
    private Scan scan;
    private Partition currentPartition;
    private ArrowReader currentReader;
    private ColumnarBatch currentBatch;
    private boolean batchLoaded;
    private boolean exhausted;

    public VortexPartitionReader(
            PartitionedFile file,
            StructType dataSchema,
            StructType readDataSchema,
            StructType readPartitionSchema,
            VortexIo io,
            VortexOptions formatOptions,
            Filter[] pushedFilters,
            boolean caseSensitive) {
        this.file = file;
        this.readDataSchema = readDataSchema;
        this.readPartitionSchema = readPartitionSchema;
        this.allocator = ArrowAllocation.rootAllocator();
        try {
            session = VortexSparkSession.get(formatOptions);
            readable = io.openReadable(new VortexFile(file.toPath().toString(), file.fileSize()));
            dataSource = DataSource.open(session, List.of(readable), io.readConcurrency());

            List<Field> arrowFields = dataSource.arrowSchema(allocator).getFields();
            StructType fileSchema = fileSchema(arrowFields);
            Map<String, String> fileFields = fieldNames(fileSchema, caseSensitive);
            List<StructField> projected = new ArrayList<>(readDataSchema.length());
            this.vectorSlots = new int[readDataSchema.length()];
            StructField[] fields = readDataSchema.fields();
            for (int i = 0; i < fields.length; i++) {
                if (fileFields.containsKey(fields[i].name())) {
                    vectorSlots[i] = projected.size();
                    projected.add(fields[i]);
                } else if (fields[i].nullable()) {
                    vectorSlots[i] = -1;
                } else {
                    throw new IllegalArgumentException(String.format(
                            Locale.ROOT,
                            "%s does not carry the non-nullable column %s that the query requires",
                            file.toPath(),
                            fields[i].name()));
                }
            }

            StructType projectedSchema = new StructType(projected.toArray(new StructField[0]));
            ReadTypes.check(
                    projectedSchema, arrowFields, caseSensitive, file.toPath().toString());

            var options = ScanOptions.builder();
            // Always project, an empty read schema included: a query that needs only partition columns or only a
            // row count must not pull every data column off storage. Struct columns that Spark pruned read only the
            // nested fields the query uses.
            NestedSchemaPruning.Projection projection = NestedSchemaPruning.projection(
                    projectedSchema, fileSchema, caseSensitive, file.toPath().toString());
            options.projection(projection.expression());
            this.columnShapes = new NestedSchemaPruning.Shape[fields.length];
            for (int i = 0; i < fields.length; i++) {
                if (vectorSlots[i] >= 0) {
                    columnShapes[i] = projection.shape(vectorSlots[i]);
                }
            }
            // Filters are converted against this file's own columns. Spark evaluates every data filter above the
            // scan anyway, so one that reads a column this file lacks, or stores in another type, is dropped rather
            // than pushed.
            buildFilterExpression(pushedFilters, ReadTypes.filterable(dataSchema, arrowFields))
                    .ifPresent(options::filter);
            Optional<SplitRows> rows = splitRows(file, dataSource.rowCount());
            if (rows.isPresent() && rows.get().count() == 0) {
                // The native scan reads a range of 0 to 0 as the whole file.
                exhausted = true;
            } else {
                rows.ifPresent(range -> options.rowRangeBegin(range.begin()).rowRangeEnd(range.end()));
                scan = dataSource.scan(options.build());
            }
        } catch (RuntimeException e) {
            closeReadableAfterFailure(e);
            throw e;
        }
    }

    /** The rows this split of {@code file} reads, or empty when it reads them all. */
    private static Optional<SplitRows> splitRows(PartitionedFile file, RowCount rowCount) {
        if (file.start() == 0 && file.length() >= file.fileSize()) {
            return Optional.empty();
        }
        if (!(rowCount instanceof RowCount.Exact exact)) {
            // Without an exact count no two splits can agree on a boundary, so the first reads everything.
            return file.start() == 0 ? Optional.empty() : Optional.of(new SplitRows(0, 0));
        }
        return Optional.of(SplitRows.of(file.start(), file.length(), file.fileSize(), exact.value()));
    }

    private static StructType fileSchema(List<Field> fields) {
        return new StructType(fields.stream()
                .map(field -> new StructField(field.getName(), sparkType(field), field.isNullable(), Metadata.empty()))
                .toArray(StructField[]::new));
    }

    private static DataType sparkType(Field field) {
        // Only names, nullability and struct and array shapes are matched against, so leaf types are left unconverted:
        // a file column Spark cannot read only fails a query that reads it.
        if (field.getType() instanceof ArrowType.List || field.getType() instanceof ArrowType.ListView) {
            Field element = field.getChildren().get(0);
            return new ArrayType(sparkType(element), element.isNullable());
        }
        if (!(field.getType() instanceof ArrowType.Struct)) {
            return DataTypes.NullType;
        }
        return new StructType(field.getChildren().stream()
                .map(child -> new StructField(child.getName(), sparkType(child), child.isNullable(), Metadata.empty()))
                .toArray(StructField[]::new));
    }

    private Map<String, String> fieldNames(StructType fileSchema, boolean caseSensitive) {
        Map<String, String> names = caseSensitive ? new HashMap<>() : new TreeMap<>(String.CASE_INSENSITIVE_ORDER);
        for (String name : fileSchema.fieldNames()) {
            String previous = names.putIfAbsent(name, name);
            if (previous != null) {
                throw new IllegalArgumentException(String.format(
                        Locale.ROOT,
                        "Ambiguous columns %s and %s in %s under Spark's case sensitivity setting",
                        previous,
                        name,
                        file.toPath()));
            }
        }
        return names;
    }

    private static Optional<Expression> buildFilterExpression(Filter[] filters, StructType dataSchema) {
        Expression combined = null;
        if (filters != null) {
            for (Filter filter : filters) {
                Optional<Expression> converted = SparkFilterToVortexExpression.convert(filter, dataSchema);
                if (converted.isPresent()) {
                    combined = combined == null ? converted.get() : Expression.and(combined, converted.get());
                }
            }
        }
        return Optional.ofNullable(combined);
    }

    @Override
    public boolean next() {
        closeCurrentBatch();
        batchLoaded = false;
        if (exhausted) {
            return false;
        }
        while (true) {
            if (currentReader != null) {
                try {
                    if (currentReader.loadNextBatch()) {
                        batchLoaded = true;
                        return true;
                    }
                } catch (IOException e) {
                    throw failure("load a batch from", e);
                }
                closeCurrentReader();
            }
            if (!scan.hasNext()) {
                exhausted = true;
                return false;
            }
            currentPartition = scan.next();
            currentReader = currentPartition.scanArrow(allocator);
        }
    }

    @Override
    public ColumnarBatch get() {
        if (!batchLoaded) {
            throw new IllegalStateException("no batch loaded; call next() first");
        }
        batchLoaded = false;
        VectorSchemaRoot root;
        try {
            root = currentReader.getVectorSchemaRoot();
        } catch (IOException e) {
            throw failure("read the loaded batch of", e);
        }

        int rowCount = root.getRowCount();
        StructField[] fields = readDataSchema.fields();
        ColumnVector[] dataVectors = new ColumnVector[fields.length];
        for (int i = 0; i < dataVectors.length; i++) {
            if (vectorSlots[i] < 0) {
                dataVectors[i] = nullColumn(rowCount, fields[i].dataType());
                continue;
            }
            ColumnVector vector =
                    new VortexArrowColumnVector(root.getFieldVectors().get(vectorSlots[i]));
            NestedSchemaPruning.Shape shape = columnShapes[i];
            dataVectors[i] =
                    shape == null ? vector : NullFilledStructVector.fill(vector, fields[i].dataType(), shape, rowCount);
        }
        ColumnVector[] partitionVectors =
                PartitionColumnVectors.create(rowCount, readPartitionSchema, file.partitionValues());
        ColumnVector[] vectors = Arrays.copyOf(dataVectors, dataVectors.length + partitionVectors.length);
        System.arraycopy(partitionVectors, 0, vectors, dataVectors.length, partitionVectors.length);
        currentBatch = new ColumnarBatch(vectors, rowCount);
        return currentBatch;
    }

    private static ColumnVector nullColumn(int rowCount, DataType type) {
        ConstantColumnVector vector = new ConstantColumnVector(rowCount, type);
        vector.setNull();
        return vector;
    }

    @Override
    public void close() {
        closeCurrentBatch();
        closeCurrentReader();
        scan = null;
        dataSource = null;
        session = null;
        if (readable != null) {
            try {
                readable.close();
            } catch (IOException e) {
                throw failure("close the readable for", e);
            } finally {
                readable = null;
            }
        }
    }

    private RuntimeException failure(String action, IOException cause) {
        return new UncheckedIOException(action + " " + file.toPath(), cause);
    }

    private void closeCurrentBatch() {
        if (currentBatch != null) {
            currentBatch.close();
            currentBatch = null;
        }
    }

    private void closeCurrentReader() {
        if (currentReader != null) {
            try {
                currentReader.close();
            } catch (IOException e) {
                throw failure("close the reader for", e);
            } finally {
                currentReader = null;
                currentPartition = null;
            }
        }
    }

    private void closeReadableAfterFailure(RuntimeException failure) {
        if (readable != null) {
            try {
                readable.close();
            } catch (IOException e) {
                failure.addSuppressed(e);
            }
            readable = null;
        }
    }
}
