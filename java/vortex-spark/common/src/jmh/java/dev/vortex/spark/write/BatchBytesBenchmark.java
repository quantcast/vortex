// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import dev.vortex.spark.VortexOptions;
import dev.vortex.spark.io.HadoopWritable;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.apache.hadoop.conf.Configuration;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.types.StructType;
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
 * Rows of a wide nested schema written through {@link VortexOutputWriter}, as a Spark task writes them: the rows of
 * the Parquet file at {@code -Dvortex.bench.parquet}, such as an hour's raw bid events, into one compact file.
 * {@code maxBatchBytes} 0 bounds batches by rows alone, which skips measuring the open batch after each row.
 *
 * <p>The native writer encodes on its own threads, so the time per file can hide the caller's share. {@code callerCpuMs}
 * and {@code files} sum over an iteration, so the caller's CPU per file is their ratio.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 3, time = 5)
@Measurement(iterations = 5, time = 5)
@Fork(
        value = 1,
        jvmArgsAppend = {"-Xmx4g", "-XX:MaxDirectMemorySize=4g"})
public class BatchBytesBenchmark {
    private static final int ROWS = 65_536;

    @Param({"16777216", "0"})
    private long maxBatchBytes;

    private StructType schema;
    private List<InternalRow> rows;
    private VortexOptions options;
    private Configuration conf;
    private Path directory;

    @Setup(Level.Trial)
    public void setUp() throws IOException {
        String parquet = System.getProperty("vortex.bench.parquet", "");
        if (parquet.isEmpty()) {
            throw new IllegalArgumentException("Set -Dvortex.bench.parquet to a Parquet file of wide nested rows");
        }
        SparkSession spark = SparkSession.builder()
                .master("local[1]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.ui.enabled", "false")
                .getOrCreate();
        try {
            var dataset = spark.read().parquet(parquet).limit(ROWS);
            schema = dataset.schema();
            rows = dataset.queryExecution().toRdd().toJavaRDD().map(InternalRow::copy).collect();
        } finally {
            spark.stop();
        }
        Map<String, String> settings = new HashMap<>(Map.of(
                VortexOutputWriter.WRITE_BATCH_SIZE_OPTION, "8192",
                VortexOutputWriter.COMPACT_OPTION, "true",
                VortexOutputWriter.WRITE_RESET_CAPACITY_OPTION, "auto",
                "vortex.workerThreads", "4"));
        if (maxBatchBytes > 0) {
            settings.put(VortexOutputWriter.WRITE_BATCH_BYTES_OPTION, Long.toString(maxBatchBytes));
        }
        options = VortexOptions.of(settings);
        conf = new Configuration();
        directory = Files.createTempDirectory("vortex-batch-bytes-bench");
    }

    @TearDown(Level.Trial)
    public void tearDown() throws IOException {
        try (var files = Files.walk(directory)) {
            files.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
        }
    }

    /** CPU time of the thread calling the writer, summed over an iteration with the files it wrote. */
    @State(Scope.Thread)
    @AuxCounters(AuxCounters.Type.EVENTS)
    public static class CallerCpu {
        private final ThreadMXBean threads = ManagementFactory.getThreadMXBean();
        public double callerCpuMs;
        public long files;

        @Setup(Level.Iteration)
        public void reset() {
            callerCpuMs = 0;
            files = 0;
        }
    }

    @Benchmark
    public long writeFile(CallerCpu cpu) {
        String path = directory.resolve("file.vortex").toString();
        long start = cpu.threads.getCurrentThreadCpuTime();
        VortexOutputWriter writer = new VortexOutputWriter(path, schema, options, HadoopWritable.create(conf, path));
        for (InternalRow row : rows) {
            writer.write(row);
        }
        writer.close();
        cpu.callerCpuMs += (cpu.threads.getCurrentThreadCpuTime() - start) / 1e6;
        cpu.files++;
        return writer.summary().fileSize();
    }
}
