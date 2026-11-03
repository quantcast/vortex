// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.management.ThreadMXBean;
import dev.vortex.api.VortexWriter;
import dev.vortex.io.NativeWritable;
import dev.vortex.spark.VortexOptions;
import dev.vortex.spark.VortexSparkSession;
import java.io.IOException;
import java.io.OutputStream;
import java.io.Serializable;
import java.lang.management.ManagementFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.types.StructType;
import scala.runtime.AbstractFunction1;
import scala.runtime.BoxedUnit;

/** Local measurement of Vortex scans and Vortex-to-Vortex rewrites, including native finish. */
public final class VortexMemoryBench {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final AtomicLong arrowPeak = new AtomicLong();
    private static final AtomicLong bufferedPeak = new AtomicLong();
    private static final AtomicLong rowCount = new AtomicLong();
    private static final AtomicLong outputBytes = new AtomicLong();
    private static final AtomicLong batchCount = new AtomicLong();

    static final class Sink implements NativeWritable {
        private final OutputStream output;

        Sink(String path) throws IOException {
            output = path.equals("discard")
                    ? OutputStream.nullOutputStream()
                    : Files.newOutputStream(Path.of(path));
        }

        @Override
        public void write(byte[] bytes, int offset, int length) throws IOException {
            output.write(bytes, offset, length);
        }

        @Override
        public void flush() throws IOException {
            output.flush();
        }

        @Override
        public void close() throws IOException {
            output.close();
        }
    }

    static final class Task extends AbstractFunction1<scala.collection.Iterator<InternalRow>, BoxedUnit>
            implements Serializable {
        private final StructType schema;
        private final Map<String, String> options;
        private final String output;
        private final boolean control;

        Task(StructType schema, Map<String, String> options, String output, boolean control) {
            this.schema = schema;
            this.options = options;
            this.output = output;
            this.control = control;
        }

        @Override
        public BoxedUnit apply(scala.collection.Iterator<InternalRow> rows) {
            try {
                VortexOutputWriter writer = control
                        ? null
                        : new VortexOutputWriter("peak-memory", schema, VortexOptions.of(options), new Sink(output));
                var allocator = control ? null : writer.allocator();
                var field = VortexOutputWriter.class.getDeclaredField("vortexWriter");
                field.setAccessible(true);
                VortexWriter nativeWriter = control ? null : (VortexWriter) field.get(writer);
                long count = 0;
                try {
                    while (rows.hasNext()) {
                        InternalRow row = rows.next();
                        if (writer != null) {
                            writer.write(row);
                        }
                        ++count;
                        if (writer != null && count % 256 == 0) {
                            bufferedPeak.accumulateAndGet(nativeWriter.bufferedBytes(), Math::max);
                        }
                        if (writer != null && count % 32768 == 0) {
                            System.out.printf(
                                    "SAMPLE rows=%d arrow=%d buffered=%d%n",
                                    count, allocator.getAllocatedMemory(), nativeWriter.bufferedBytes());
                        }
                    }
                } finally {
                    if (writer != null) {
                        writer.close();
                    }
                }
                rowCount.addAndGet(count);
                if (writer != null) {
                    arrowPeak.accumulateAndGet(allocator.getPeakMemoryAllocation(), Math::max);
                    outputBytes.addAndGet(writer.summary().fileSize());
                    batchCount.addAndGet(writer.summary().batchCount());
                    if (writer.summary().rowCount() != count) {
                        throw new AssertionError("Wrong native row count");
                    }
                }
                return BoxedUnit.UNIT;
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }

    public static void main(String[] args) throws Exception {
        // Arguments are supplied by run.py, which fixes the batch and worker settings.
        Map<String, String> options = new LinkedHashMap<>();
        options.put("vortex.write.batch.size", args[1]);
        if (!args[2].equals("0")) {
            options.put("vortex.write.batch.bytes", args[2]);
        }
        options.put("vortex.write.reset.capacity", args[3]);
        options.put("vortex.workerThreads", args[4]);
        if (args.length > 7 && !args[7].equals("0")) {
            options.put("vortex.splitConcurrency", args[7]);
        }
        options.put("vortex.splitFiles", "false");
        boolean control = Boolean.parseBoolean(args[6]);
        SparkSession spark = SparkSession.builder()
                .master("local[1]")
                .appName("vortex-peak-memory")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.ui.enabled", "false")
                .config("spark.sql.adaptive.enabled", "false")
                .config("spark.sql.codegen.wholeStage", "false")
                .config("spark.sql.debug.maxToStringFields", "5")
                .getOrCreate();
        spark.sparkContext().setLogLevel("ERROR");
        var data = spark.read().format("vortex").options(options).load(args[0]).coalesce(1);
        var schema = data.schema();
        VortexSparkSession.get(VortexOptions.of(options));
        System.gc();
        Thread.sleep(300);
        AtomicBoolean running = new AtomicBoolean(true);
        AtomicLong heapPeak = new AtomicLong();
        Thread sampler = new Thread(() -> {
            while (running.get()) {
                heapPeak.accumulateAndGet(
                        ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed(), Math::max);
                try {
                    Thread.sleep(10);
                } catch (InterruptedException e) {
                    return;
                }
            }
        });
        sampler.setDaemon(true);
        sampler.start();
        ThreadMXBean allocation = (ThreadMXBean) ManagementFactory.getThreadMXBean();
        long allocatedBefore = allocation.getTotalThreadAllocatedBytes();
        System.out.println("BENCH_BEGIN");
        System.out.flush();
        long begin = System.nanoTime();
        data.queryExecution().toRdd().foreachPartition(new Task(schema, options, args[5], control));
        double seconds = (System.nanoTime() - begin) / 1e9;
        running.set(false);
        sampler.join();
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("seconds", seconds);
        result.put("rows", rowCount.get());
        result.put("arrowPeakBytes", arrowPeak.get());
        result.put("bufferedPeakBytes", bufferedPeak.get());
        result.put("heapPeakBytes", heapPeak.get());
        result.put("jvmAllocatedBytes", allocation.getTotalThreadAllocatedBytes() - allocatedBefore);
        result.put("outputBytes", outputBytes.get());
        result.put("batchCount", batchCount.get());
        System.out.println("BENCH " + JSON.writeValueAsString(result));
        System.out.flush();
        if (control && args.length > 8 && args[8].equals("verify")) {
            String digest = "sum(cast(xxhash64(to_json(struct(*))) as decimal(38,0)))";
            System.out.println("VERIFIED_READ " + data.selectExpr("count(*)", digest).collectAsList());
        }
        if (!control && !args[5].equals("discard")) {
            var written = spark.read()
                    .format("vortex")
                    .options(Map.of("vortex.splitConcurrency", "4"))
                    .load(args[5])
                    .selectExpr(schema.fieldNames());
            String digest = "sum(cast(xxhash64(to_json(struct(*))) as decimal(38,0)))";
            var expected = data.selectExpr("count(*)", digest).collectAsList();
            var actual = written.selectExpr("count(*)", digest).collectAsList();
            if (!expected.equals(actual)) {
                throw new AssertionError("Readback mismatch: " + expected + " / " + actual);
            }
            System.out.println("VERIFIED " + expected);
        }
        spark.stop();
    }
}
