// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.management.ThreadMXBean;
import dev.vortex.io.NativeWritable;
import dev.vortex.spark.VortexOptions;
import java.lang.management.ManagementFactory;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow;
import org.apache.spark.sql.catalyst.util.ArrayBasedMapData;
import org.apache.spark.sql.catalyst.util.GenericArrayData;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.unsafe.types.UTF8String;

/** Synthetic writer workloads for the capacity policy benchmark. */
public final class CapacityBench {
    record Fixture(StructType schema, InternalRow[] rows, InternalRow outlier) {}

    static final ObjectMapper JSON = new ObjectMapper();

    static UTF8String text(int length, Random rng) {
        byte[] value = new byte[length];
        for (int j = 0; j < length; j++) value[j] = (byte) ('a' + rng.nextInt(26));
        return UTF8String.fromBytes(value);
    }

    static Fixture fixture(String shape) {
        Random rng = new Random(481516);
        StructType schema = new StructType();
        InternalRow[] rows = new InternalRow[256];
        InternalRow outlier = null;
        if (shape.equals("flat") || shape.equals("wide")) {
            int columns = shape.equals("wide") ? 64 : 8;
            for (int j = 0; j < columns; j++) schema = schema.add("c" + j, DataTypes.LongType, false);
            for (int i = 0; i < rows.length; i++) {
                Object[] v = new Object[columns];
                for (int j = 0; j < columns; j++) v[j] = rng.nextLong();
                rows[i] = new GenericInternalRow(v);
            }
        } else if (shape.equals("short-text") || shape.equals("long-text")) {
            int columns = shape.equals("short-text") ? 8 : 4, length = shape.equals("short-text") ? 32 : 4096;
            for (int j = 0; j < columns; j++) schema = schema.add("c" + j, DataTypes.StringType, true);
            for (int i = 0; i < rows.length; i++) {
                Object[] v = new Object[columns];
                for (int j = 0; j < columns; j++) v[j] = i % 17 == 0 ? null : text(length, rng);
                rows[i] = new GenericInternalRow(v);
            }
        } else if (shape.equals("outlier")) {
            schema = schema.add("values", DataTypes.createArrayType(DataTypes.LongType, false), true)
                    .add("label", DataTypes.StringType, true);
            for (int i = 0; i < rows.length; i++)
                rows[i] = new GenericInternalRow(
                        new Object[] {new GenericArrayData(new long[] {i, i + 1}), text(16, rng)});
            long[] big = new long[65536];
            Arrays.setAll(big, i -> i);
            outlier = new GenericInternalRow(new Object[] {new GenericArrayData(big), text(131072, rng)});
        } else if (shape.equals("nested")) {
            StructType element =
                    new StructType().add("id", DataTypes.LongType, false).add("label", DataTypes.StringType, true);
            StructType parent = new StructType()
                    .add("id", DataTypes.LongType, false)
                    .add("items", DataTypes.createArrayType(element, true), true);
            schema = schema.add("parent", parent, true);
            for (int i = 0; i < rows.length; i++) {
                Object[] items = new Object[16 + (i % 8)];
                for (int j = 0; j < items.length; j++)
                    items[j] = j % 11 == 0
                            ? null
                            : new GenericInternalRow(new Object[] {(long) (i * 32 + j), text(24, rng)});
                rows[i] = new GenericInternalRow(new Object[] {
                    i % 19 == 0 ? null : new GenericInternalRow(new Object[] {(long) i, new GenericArrayData(items)})
                });
            }
        } else if (shape.equals("map-null")) {
            StructType parent = new StructType()
                    .add("name", DataTypes.StringType, true)
                    .add("map", DataTypes.createMapType(DataTypes.StringType, DataTypes.LongType, true), true);
            schema = schema.add("parent", parent, true)
                    .add("empty", DataTypes.createArrayType(DataTypes.LongType, false), true);
            for (int i = 0; i < rows.length; i++) {
                Object[] keys = {text(12, rng), text(12, rng)}, values = {(long) i, null};
                var map = new ArrayBasedMapData(new GenericArrayData(keys), new GenericArrayData(values));
                rows[i] = new GenericInternalRow(new Object[] {
                    i % 3 == 0 ? null : new GenericInternalRow(new Object[] {text(24, rng), map}),
                    new GenericArrayData(new long[0])
                });
            }
        } else throw new IllegalArgumentException(shape);
        return new Fixture(schema, rows, outlier);
    }

    static Map<String, Object> run(Fixture fixture, int n, int batch, String reset, long batchBytes) throws Exception {
        Map<String, String> configured = new LinkedHashMap<>();
        configured.put(VortexOutputWriter.WRITE_BATCH_SIZE_OPTION, Integer.toString(batch));
        configured.put(VortexOutputWriter.WRITE_RESET_CAPACITY_OPTION, reset);
        if (batchBytes > 0) configured.put(VortexOutputWriter.WRITE_BATCH_BYTES_OPTION, Long.toString(batchBytes));
        var options = VortexOptions.of(configured);
        long arrowPeak = 0;
        long start = System.nanoTime();
        ThreadMXBean alloc = (ThreadMXBean) ManagementFactory.getThreadMXBean();
        long allocStart = alloc.getTotalThreadAllocatedBytes();
        long gcStart = ManagementFactory.getGarbageCollectorMXBeans().stream()
                .mapToLong(b -> Math.max(0, b.getCollectionTime()))
                .sum();
        long output;
        DiscardingWritable sink = new DiscardingWritable();
        VortexOutputWriter writer = new VortexOutputWriter("capacity-benchmark", fixture.schema(), options, sink);
        try {
            for (int i = 0; i < n; i++) {
                writer.write(
                        i == 0 && fixture.outlier() != null
                                ? fixture.outlier()
                                : fixture.rows()[i % fixture.rows().length]);
                if (i % 16 == 0 || i == n - 1)
                    arrowPeak = Math.max(arrowPeak, writer.allocator().getAllocatedMemory());
            }
        } finally {
            writer.close();
        }
        var completed = writer.summary();
        if (completed.rowCount() != n) throw new AssertionError("row count " + completed.rowCount());
        output = completed.fileSize();
        if (sink.bytes.get() != output) throw new AssertionError("output counter mismatch");
        long gcEnd = ManagementFactory.getGarbageCollectorMXBeans().stream()
                .mapToLong(b -> Math.max(0, b.getCollectionTime()))
                .sum();
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("seconds", (System.nanoTime() - start) / 1e9);
        result.put("arrowPeakBytes", arrowPeak);
        result.put("jvmAllocatedBytes", alloc.getTotalThreadAllocatedBytes() - allocStart);
        result.put("gcMillis", gcEnd - gcStart);
        result.put("outputBytes", output);
        result.put("rows", n);
        result.put("batch", batch);
        result.put("batchBytes", batchBytes);
        result.put("batchCount", completed.batchCount());
        result.put("reset", reset);
        return result;
    }

    private static final class DiscardingWritable implements NativeWritable {
        private final AtomicLong bytes = new AtomicLong();

        @Override
        public void write(byte[] buffer, int offset, int length) {
            bytes.addAndGet(length);
        }

        @Override
        public void flush() {}

        @Override
        public void close() {}
    }

    public static void main(String[] args) throws Exception {
        String shape = args[0];
        int n = Integer.parseInt(args[1]), batch = Integer.parseInt(args[2]);
        String reset = args[3];
        long batchBytes = Long.parseLong(args[4]);
        Fixture fixture = fixture(shape);
        run(fixture, Math.min(n, 8192), batch, reset, batchBytes);
        System.gc();
        Thread.sleep(100);
        AtomicBoolean active = new AtomicBoolean(true);
        AtomicLong heapPeak = new AtomicLong();
        Thread sampler = new Thread(() -> {
            while (active.get()) {
                heapPeak.accumulateAndGet(
                        ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed(), Math::max);
                try {
                    Thread.sleep(5);
                } catch (InterruptedException e) {
                    return;
                }
            }
        });
        sampler.setDaemon(true);
        sampler.start();
        System.out.println("BENCH_BEGIN");
        System.out.flush();
        Map<String, Object> result = run(fixture, n, batch, reset, batchBytes);
        active.set(false);
        sampler.join();
        result.put("shape", shape);
        result.put("heapPeakBytes", heapPeak.get());
        System.out.println("BENCH " + JSON.writeValueAsString(result));
        System.out.flush();
    }
}
