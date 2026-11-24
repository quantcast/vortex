// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import dev.vortex.relocated.org.apache.arrow.memory.RootAllocator;
import dev.vortex.relocated.org.apache.arrow.vector.FieldVector;
import dev.vortex.relocated.org.apache.arrow.vector.VectorSchemaRoot;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

/**
 * The Java half of {@link VortexOutputWriter#write}: rows of the Parquet file at {@code -Dvortex.bench.parquet} into
 * Arrow batches through {@link ArrowFieldWriter}, checking the open batch against 16 MB with
 * {@link ArrowBatchMemory#reached} after each row when {@code measure} is set. A batch ends at 8,192 rows or 16 MB and is cleared rather than handed to native code,
 * with every vector's capacity hint back at 64 rows.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 3, time = 5)
@Measurement(iterations = 5, time = 5)
@Fork(
        value = 2,
        jvmArgsAppend = {"-Xmx4g", "-XX:MaxDirectMemorySize=4g"})
public class BatchFillBenchmark {
    private static final int ROWS = 65_536;
    private static final int BATCH_ROWS = 8192;
    private static final long BATCH_BYTES = 16 << 20;
    private static final int INITIAL_CAPACITY = 64;

    @Param({"true", "false"})
    private boolean measure;

    private List<InternalRow> rows;
    private ArrowBatchMemory memory;
    private RootAllocator allocator;
    private VectorSchemaRoot root;
    private ArrowFieldWriter<?>[] writers;

    @Setup(Level.Trial)
    public void setUp() {
        String parquet = System.getProperty("vortex.bench.parquet", "");
        if (parquet.isEmpty()) {
            throw new IllegalArgumentException("Set -Dvortex.bench.parquet to a Parquet file of wide nested rows");
        }
        SparkSession spark = SparkSession.builder()
                .master("local[1]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.ui.enabled", "false")
                .getOrCreate();
        StructType schema;
        try {
            var dataset = spark.read().parquet(parquet).limit(ROWS);
            schema = dataset.schema();
            rows = dataset.queryExecution().toRdd().toJavaRDD().map(InternalRow::copy).collect();
        } finally {
            spark.stop();
        }
        memory = new ArrowBatchMemory();
        allocator = new RootAllocator(memory, Long.MAX_VALUE);
        root = VectorSchemaRoot.create(SparkToArrowSchema.convert(schema), allocator);
        StructField[] fields = schema.fields();
        writers = new ArrowFieldWriter<?>[fields.length];
        for (int i = 0; i < fields.length; i++) {
            writers[i] = ArrowFieldWriter.create(root.getVector(i), fields[i].dataType());
        }
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        root.close();
        allocator.close();
    }

    @Benchmark
    public long fillBatches() {
        long batches = 0;
        int batchRows = 0;
        for (InternalRow row : rows) {
            if (batchRows == 0) {
                for (FieldVector vector : root.getFieldVectors()) {
                    setInitialCapacity(vector);
                }
                root.allocateNew();
            }
            for (int i = 0; i < writers.length; i++) {
                writers[i].write(row, i);
            }
            batchRows++;
            boolean full = batchRows >= BATCH_ROWS || (measure && memory.reached(root, BATCH_BYTES));
            if (full) {
                root.setRowCount(batchRows);
                root.clear();
                for (ArrowFieldWriter<?> writer : writers) {
                    writer.reset();
                }
                memory.reset();
                batchRows = 0;
                batches++;
            }
        }
        root.clear();
        for (ArrowFieldWriter<?> writer : writers) {
            writer.reset();
        }
        memory.reset();
        return batches;
    }

    private static void setInitialCapacity(FieldVector vector) {
        vector.setInitialCapacity(INITIAL_CAPACITY);
        for (FieldVector child : vector.getChildrenFromFields()) {
            setInitialCapacity(child);
        }
    }
}
