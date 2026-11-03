// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.bench;

import dev.vortex.api.Session;
import dev.vortex.api.VortexWriter;
import dev.vortex.relocated.org.apache.arrow.memory.BufferAllocator;
import dev.vortex.relocated.org.apache.arrow.memory.RootAllocator;
import dev.vortex.relocated.org.apache.arrow.vector.BigIntVector;
import dev.vortex.relocated.org.apache.arrow.vector.Float8Vector;
import dev.vortex.relocated.org.apache.arrow.vector.VarBinaryVector;
import dev.vortex.relocated.org.apache.arrow.vector.VectorSchemaRoot;
import dev.vortex.relocated.org.apache.arrow.vector.types.FloatingPointPrecision;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.ArrowType;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Field;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.FieldType;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Schema;
import dev.vortex.spark.io.HadoopWritable;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryPoolMXBean;
import java.lang.management.MemoryType;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.apache.hadoop.conf.Configuration;
import org.openjdk.jmh.annotations.AuxCounters;
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
 * Cost of writing a Vortex file through {@link HadoopWritable}, the sink Spark's file-format writer streams native
 * output into.
 *
 * <p>Run with {@code -prof gc} for heap allocated per file. {@code oldGenPeakMiB} reports the peak old-generation use
 * in each iteration, where G1 places arrays larger than half a region.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 5)
@Measurement(iterations = 10)
@Fork(
        value = 2,
        jvmArgsAppend = {"-Xmx2g", "-XX:+UseG1GC"})
public class HadoopWriteBenchmark {
    private static final int BATCH_ROWS = 8192;

    /** Rows per file: the larger file has bigger native output buffers. */
    @Param({"262144", "4194304"})
    private int rows;

    private Path directory;
    private Configuration conf;
    private Session session;
    private BufferAllocator allocator;
    private Schema schema;
    private VectorSchemaRoot batch;
    private int file;

    @Setup(Level.Trial)
    public void setUp() throws IOException {
        directory = Files.createTempDirectory("vortex-hadoop-write-bench");
        conf = new Configuration();
        session = Session.create();
        allocator = new RootAllocator();
        schema = new Schema(List.of(
                new Field("id", FieldType.notNullable(new ArrowType.Int(64, true)), null),
                new Field("payload", FieldType.notNullable(new ArrowType.Binary()), null),
                new Field(
                        "measure",
                        FieldType.notNullable(new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)),
                        null)));
        batch = VectorSchemaRoot.create(schema, allocator);
        // Random values, so encoded output is close to the input size rather than compressed away.
        Random random = new Random(0);
        BigIntVector id = (BigIntVector) batch.getVector("id");
        VarBinaryVector payload = (VarBinaryVector) batch.getVector("payload");
        Float8Vector measure = (Float8Vector) batch.getVector("measure");
        byte[] value = new byte[32];
        for (int i = 0; i < BATCH_ROWS; i++) {
            id.setSafe(i, random.nextLong());
            random.nextBytes(value);
            payload.setSafe(i, value);
            measure.setSafe(i, random.nextDouble());
        }
        batch.setRowCount(BATCH_ROWS);
    }

    @TearDown(Level.Trial)
    public void tearDown() throws IOException {
        batch.close();
        allocator.close();
        SparkScanBenchmark.deleteRecursively(directory);
    }

    /** Peak old-generation use per iteration, reset before each one. */
    @State(Scope.Thread)
    @AuxCounters(AuxCounters.Type.EVENTS)
    public static class OldGenPeak {
        private List<MemoryPoolMXBean> pools;
        public double oldGenPeakMiB;

        @Setup(Level.Iteration)
        public void reset() {
            System.gc();
            pools = ManagementFactory.getMemoryPoolMXBeans().stream()
                    .filter(pool ->
                            pool.getType() == MemoryType.HEAP && pool.getName().contains("Old Gen"))
                    .toList();
            pools.forEach(MemoryPoolMXBean::resetPeakUsage);
            oldGenPeakMiB = 0;
        }

        void record() {
            long peak = 0;
            for (MemoryPoolMXBean pool : pools) {
                peak += pool.getPeakUsage().getUsed();
            }
            oldGenPeakMiB = Math.max(oldGenPeakMiB, peak / (1024.0 * 1024.0));
        }
    }

    @Benchmark
    public long writeFile(OldGenPeak peak) throws IOException {
        String path = directory.resolve("file-" + (file++ % 4) + ".vortex").toString();
        long size;
        try (HadoopWritable sink = HadoopWritable.create(conf, path);
                VortexWriter writer =
                        VortexWriter.builder(session, sink, schema, allocator).build()) {
            for (int written = 0; written < rows; written += BATCH_ROWS) {
                writer.writeBatch(batch, allocator);
            }
            size = writer.finish().fileSize();
        }
        peak.record();
        return size;
    }
}
