// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.api.DataSource;
import dev.vortex.api.Scan;
import dev.vortex.api.ScanOptions;
import dev.vortex.jni.NativeFiles;
import dev.vortex.relocated.org.apache.arrow.memory.ArrowBuf;
import dev.vortex.relocated.org.apache.arrow.memory.BufferAllocator;
import dev.vortex.relocated.org.apache.arrow.memory.RootAllocator;
import dev.vortex.relocated.org.apache.arrow.vector.FieldVector;
import dev.vortex.relocated.org.apache.arrow.vector.VectorSchemaRoot;
import dev.vortex.relocated.org.apache.arrow.vector.ipc.ArrowReader;
import dev.vortex.spark.VortexOptions;
import dev.vortex.spark.VortexSparkSession;
import dev.vortex.spark.io.HadoopWritable;
import java.io.IOException;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow;
import org.apache.spark.sql.catalyst.expressions.UnsafeProjection;
import org.apache.spark.sql.catalyst.expressions.UnsafeRow;
import org.apache.spark.sql.catalyst.util.ArrayBasedMapData;
import org.apache.spark.sql.catalyst.util.GenericArrayData;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Decimal;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.unsafe.Platform;
import org.apache.spark.unsafe.types.UTF8String;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.NullSource;
import org.junit.jupiter.params.provider.ValueSource;

final class VortexOutputWriterTest {
    private static final int DEFAULT = 8192;

    @TempDir
    Path tempDir;

    @Test
    void anUnsetBatchSizeFallsBackToTheDefault() {
        assertEquals(DEFAULT, VortexOutputWriter.configuredBatchSize(VortexOptions.empty()));
    }

    @Test
    void theGenericBatchSizeIsHonoured() {
        assertEquals(4096, VortexOutputWriter.configuredBatchSize(options(Map.of("batch.size", "4096"))));
    }

    @Test
    void theVortexBatchSizeOverridesTheGenericOne() {
        VortexOptions options = options(Map.of("batch.size", "4096", "vortex.write.batch.size", "512"));

        // A job that sets one batch size across formats can still say something different for Vortex.
        assertEquals(512, VortexOutputWriter.configuredBatchSize(options));
    }

    @Test
    void theOverrideStandsEvenWhenTheGenericValueIsUnusable() {
        VortexOptions options = options(Map.of("batch.size", "lots", "vortex.write.batch.size", "512"));

        assertEquals(512, VortexOutputWriter.configuredBatchSize(options));
    }

    @Test
    void eitherNameIsFoundUnderAnyCasing() {
        assertEquals(64, VortexOutputWriter.configuredBatchSize(options(Map.of("BATCH.SIZE", "64"))));
        assertEquals(64, VortexOutputWriter.configuredBatchSize(options(Map.of("Vortex.Write.Batch.Size", "64"))));
    }

    @Test
    void aBatchSizeOfTheWrongShapeNamesItsOption() {
        VortexOptions options = options(Map.of("vortex.write.batch.size", "lots"));

        assertTrue(assertThrows(IllegalArgumentException.class, () -> VortexOutputWriter.configuredBatchSize(options))
                .getMessage()
                .contains("vortex.write.batch.size"));
    }

    @Test
    void aBatchSizeOutsideItsRangeNamesItsOption() {
        for (String value : new String[] {"0", "65537"}) {
            VortexOptions options = options(Map.of("batch.size", value));

            assertTrue(
                    assertThrows(IllegalArgumentException.class, () -> VortexOutputWriter.configuredBatchSize(options))
                            .getMessage()
                            .contains("batch.size"));
        }
        assertEquals(65536, VortexOutputWriter.configuredBatchSize(options(Map.of("batch.size", "65536"))));
        assertEquals(1, VortexOutputWriter.configuredBatchSize(options(Map.of("vortex.write.batch.size", "1"))));
    }

    @Test
    void batchBytesAreOptionalPositiveLongs() {
        assertEquals(0, VortexOutputWriter.configuredBatchBytes(VortexOptions.empty()));
        assertEquals(1, VortexOutputWriter.configuredBatchBytes(options(Map.of("vortex.write.batch.bytes", "1"))));
        assertEquals(
                3_000_000_000L,
                VortexOutputWriter.configuredBatchBytes(options(Map.of("VORTEX.WRITE.BATCH.BYTES", " 3000000000 "))));
        assertEquals(
                Long.MAX_VALUE,
                VortexOutputWriter.configuredBatchBytes(
                        options(Map.of("vortex.write.batch.bytes", Long.toString(Long.MAX_VALUE)))));
        for (String value : new String[] {"0", "-1", "lots", "9223372036854775808"}) {
            assertTrue(assertThrows(
                            IllegalArgumentException.class,
                            () -> VortexOutputWriter.configuredBatchBytes(
                                    options(Map.of("vortex.write.batch.bytes", value))))
                    .getMessage()
                    .contains("vortex.write.batch.bytes"));
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void theRowLimitStillFlushesWithOrWithoutAByteLimit(boolean byteLimit) {
        StructType schema = new StructType().add("id", DataTypes.LongType, false);
        Map<String, String> values = byteLimit
                ? Map.of("batch.size", "3", "vortex.write.batch.bytes", Long.toString(Long.MAX_VALUE))
                : Map.of("batch.size", "3");
        VortexOutputWriter writer = writer("row-limit-" + byteLimit + ".vortex", schema, values);
        try {
            for (long i = 0; i < 7; i++) {
                writer.write(new GenericInternalRow(new Object[] {i}));
                assertEquals((i + 1) % 3, writer.batchRowCount());
            }
        } finally {
            writer.close();
        }
        assertEquals(List.of("0", "1", "2", "3", "4", "5", "6"), readRows(writer.path()));
    }

    @Test
    void aByteLimitFlushesNestedAndOversizedRowsBeforeTheRowLimit() {
        StructType schema = new StructType()
                .add("id", DataTypes.LongType, false)
                .add("nested", new StructType().add("s", DataTypes.StringType, false), false);
        VortexOutputWriter writer = writer("byte-limit.vortex", schema, Map.of("vortex.write.batch.bytes", "1000000"));
        String large = "x".repeat(2_000_000);
        try {
            writer.write(new GenericInternalRow(
                    new Object[] {0L, new GenericInternalRow(new Object[] {UTF8String.fromString("small")})}));
            assertEquals(1, writer.batchRowCount());
            writer.write(new GenericInternalRow(
                    new Object[] {1L, new GenericInternalRow(new Object[] {UTF8String.fromString(large)})}));
            assertEquals(0, writer.batchRowCount());
            writer.write(new GenericInternalRow(
                    new Object[] {2L, new GenericInternalRow(new Object[] {UTF8String.fromString("tail")})}));
            assertEquals(1, writer.batchRowCount());
            writer.write(new GenericInternalRow(
                    new Object[] {3L, new GenericInternalRow(new Object[] {UTF8String.fromString("last")})}));
            assertEquals(2, writer.batchRowCount());
        } finally {
            writer.close();
        }
        assertEquals(
                List.of(
                        "0\t{\"s\":\"small\"}",
                        "1\t{\"s\":\"" + large + "\"}",
                        "2\t{\"s\":\"tail\"}",
                        "3\t{\"s\":\"last\"}"),
                readRows(writer.path()));
    }

    @Test
    void aThresholdSmallerThanOneRowsAllocationStillMakesProgress() {
        StructType schema = new StructType().add("id", DataTypes.LongType, false);
        VortexOutputWriter writer =
                writer("single-row-batches.vortex", schema, Map.of("vortex.write.batch.bytes", "1"));
        try {
            for (long i = 0; i < 4; i++) {
                writer.write(new GenericInternalRow(new Object[] {i}));
                assertEquals(0, writer.batchRowCount());
            }
        } finally {
            writer.close();
        }
        assertEquals(List.of("0", "1", "2", "3"), readRows(writer.path()));
    }

    /**
     * The native writer keeps every batch it receives until a row block is complete, so the memory each batch takes
     * must be proportional to its rows, not to Arrow's default vector capacity of a few thousand values.
     */
    @Test
    void memoryGrowsWithRowsNotBatches() {
        StructType schema = new StructType()
                .add("id", DataTypes.LongType, false)
                .add("s", DataTypes.StringType, true)
                .add("a", DataTypes.createArrayType(DataTypes.LongType, false), true);

        VortexOutputWriter writer = writer("tiny-batches.vortex", schema, Map.of("vortex.write.batch.size", "1"));
        int rows = 400;
        for (long i = 0; i < rows; i++) {
            writer.write(new GenericInternalRow(
                    new Object[] {i, UTF8String.fromString("row-" + i), new GenericArrayData(new long[] {i, i + 1})}));
        }
        // Measured before close, while the native writer still holds all the batches.
        long held = writer.allocator().getAllocatedMemory();
        writer.close();

        assertTrue(held < 4L << 20, "the writer holds " + (held >> 10) + " KiB for " + rows + " rows");
        assertEquals(rows, readRows(writer.path()).size());
    }

    /** Every batch's buffers and the schema export are released by the time close returns. */
    @Test
    void closeReleasesEveryBatch() {
        StructType schema = new StructType()
                .add("id", DataTypes.LongType, false)
                .add("s", DataTypes.StringType, true)
                .add("st", new StructType().add("a", DataTypes.LongType).add("b", DataTypes.StringType), true)
                .add("m", DataTypes.createMapType(DataTypes.StringType, DataTypes.LongType, true), true);

        VortexOutputWriter writer = writer("released.vortex", schema, Map.of("batch.size", "3"));
        BufferAllocator allocator = writer.allocator();
        for (long i = 0; i < 20; i++) {
            writer.write(new GenericInternalRow(new Object[] {
                i,
                UTF8String.fromString("s" + i),
                new GenericInternalRow(new Object[] {i, UTF8String.fromString("b" + i)}),
                new ArrayBasedMapData(
                        new GenericArrayData(new Object[] {UTF8String.fromString("k" + i)}),
                        new GenericArrayData(new Object[] {i})),
            }));
        }
        writer.close();

        assertEquals(0, allocator.getAllocatedMemory());
        assertEquals(20, readRows(writer.path()).size());
    }

    /** Partitioned writes keep one writer open per partition value; none may hold memory once closed. */
    @Test
    void manySmallWritersReleaseTheirMemory() {
        StructType schema =
                new StructType().add("id", DataTypes.LongType, false).add("s", DataTypes.StringType);
        List<VortexOutputWriter> writers = new ArrayList<>();
        List<BufferAllocator> allocators = new ArrayList<>();
        for (int p = 0; p < 200; p++) {
            VortexOutputWriter writer = writer("p=" + p + "/part.vortex", schema, Map.of());
            writers.add(writer);
            allocators.add(writer.allocator());
        }
        for (long i = 0; i < 1000; i++) {
            writers.get((int) (i % writers.size()))
                    .write(new GenericInternalRow(new Object[] {i, UTF8String.fromString("v" + i)}));
        }
        writers.forEach(VortexOutputWriter::close);

        assertEquals(
                0L,
                allocators.stream()
                        .mapToLong(BufferAllocator::getAllocatedMemory)
                        .sum());
        assertEquals(5, readRows(writers.get(0).path()).size());
    }

    @ParameterizedTest
    @NullSource
    @ValueSource(strings = {"auto", "true", "false"})
    void capacityPolicyBoundsMemoryAfterAnOutlier(String policy) {
        StructType schema = new StructType().add("values", DataTypes.createArrayType(DataTypes.LongType, false), true);
        Map<String, String> values = policy == null
                ? Map.of(VortexOutputWriter.WRITE_BATCH_SIZE_OPTION, "1")
                : Map.of(
                        VortexOutputWriter.WRITE_BATCH_SIZE_OPTION,
                        "1",
                        VortexOutputWriter.WRITE_RESET_CAPACITY_OPTION,
                        policy);
        long[] large = new long[65536];
        Arrays.setAll(large, i -> i);
        List<String> expected = new ArrayList<>();
        expected.add(Arrays.toString(large).replace(" ", ""));
        VortexOutputWriter writer = writer("capacity.vortex", schema, values);
        BufferAllocator allocator = writer.allocator();
        long held;
        try {
            writer.write(new GenericInternalRow(new Object[] {new GenericArrayData(large)}));
            for (long i = 0; i < 64; i++) {
                writer.write(new GenericInternalRow(new Object[] {new GenericArrayData(new long[] {i})}));
                expected.add("[" + i + "]");
            }
            // All 65 outer rows remain below the native 8192-row block boundary.
            held = allocator.getAllocatedMemory();
        } finally {
            writer.close();
        }
        long budget = 8L * 1024 * 1024;
        assertTrue("false".equals(policy) ? held >= budget : held < budget, "retained " + held + " bytes");
        assertEquals(0, allocator.getAllocatedMemory());
        assertEquals(expected, readRows(writer.path()));
    }

    @ParameterizedTest
    @ValueSource(strings = {"auto", "true", "false"})
    void everyCapacityPolicyResetsAfterAByteTriggeredFlush(String policy) {
        StructType schema = new StructType().add("values", DataTypes.createArrayType(DataTypes.LongType, false), true);
        long[] large = new long[16384];
        VortexOutputWriter writer = writer(
                "byte-capacity.vortex",
                schema,
                Map.of(
                        VortexOutputWriter.WRITE_BATCH_SIZE_OPTION, "128",
                        VortexOutputWriter.WRITE_BATCH_BYTES_OPTION, "65536",
                        VortexOutputWriter.WRITE_RESET_CAPACITY_OPTION, policy));
        try {
            writer.write(new GenericInternalRow(new Object[] {new GenericArrayData(large)}));
            assertEquals(0, writer.batchRowCount());
            writer.write(new GenericInternalRow(new Object[] {new GenericArrayData(new long[] {1})}));
            assertEquals(1, writer.batchRowCount());
        } finally {
            writer.close();
        }
        assertEquals(List.of(Arrays.toString(large).replace(" ", ""), "[1]"), readRows(writer.path()));
    }

    @ParameterizedTest
    @CsvSource({"auto,0", "auto,1", "auto,2", "true,0", "true,1", "true,2", "false,0", "false,1", "false,2"})
    void capacityPolicyPreservesRowBoundedFixedWidthHints(String policy, int structDepth) {
        StructType schema = new StructType().add("value", DataTypes.LongType, false);
        GenericInternalRow row = new GenericInternalRow(new Object[] {7L});
        String expected = "7";
        for (int i = 0; i < structDepth; i++) {
            schema = new StructType().add("value", schema, true);
            row = new GenericInternalRow(new Object[] {row});
            expected = "{\"value\":" + expected + "}";
        }
        VortexOutputWriter writer = writer(
                "fixed-width-capacity.vortex",
                schema,
                Map.of(
                        VortexOutputWriter.WRITE_BATCH_SIZE_OPTION,
                        "1024",
                        VortexOutputWriter.WRITE_RESET_CAPACITY_OPTION,
                        policy));
        BufferAllocator allocator = writer.allocator();
        try {
            for (int i = 0; i < 1024; i++) {
                writer.write(row);
            }
            assertEquals(0, writer.batchRowCount());
            // Below a native block, the exported batch stays retained while the next batch allocates.
            long retained = allocator.getAllocatedMemory();
            writer.write(row);
            long allocated = allocator.getAllocatedMemory() - retained;
            assertTrue(
                    "true".equals(policy) ? allocated < 8192 : allocated >= 8192,
                    "next batch allocated " + allocated + " bytes");
        } finally {
            writer.close();
        }
        assertEquals(0, allocator.getAllocatedMemory());
        assertEquals(Collections.nCopies(1025, expected), readRows(writer.path()));
    }

    @ParameterizedTest
    @CsvSource({
        "auto,string", "true,string", "false,string",
        "auto,binary", "true,binary", "false,binary",
        "auto,list", "true,list", "false,list",
        "auto,map", "true,map", "false,map",
        "auto,struct", "true,struct", "false,struct"
    })
    void autoRetainsUsefulCapacityAndShrinksAfterValuesGetSmaller(String policy, String shape) {
        CapacityValue large = capacityValue(shape, 128);
        CapacityValue small = capacityValue(shape, 1);
        StructType schema = new StructType().add("value", large.type(), true);
        GenericInternalRow largeRow = new GenericInternalRow(new Object[] {large.value()});
        GenericInternalRow smallRow = new GenericInternalRow(new Object[] {small.value()});
        VortexOutputWriter writer = writer(
                "adaptive-capacity.vortex",
                schema,
                Map.of(
                        VortexOutputWriter.WRITE_BATCH_SIZE_OPTION,
                        "256",
                        VortexOutputWriter.WRITE_RESET_CAPACITY_OPTION,
                        policy));
        BufferAllocator allocator = writer.allocator();
        try {
            for (int i = 0; i < 256; i++) {
                writer.write(largeRow);
            }
            long retained = allocator.getAllocatedMemory();
            writer.write(smallRow);
            long allocated = allocator.getAllocatedMemory() - retained;
            assertTrue(
                    "true".equals(policy) ? allocated < 16384 : allocated >= 16384,
                    "after a full batch, the next batch allocated " + allocated + " bytes");
            for (int i = 1; i < 256; i++) {
                writer.write(smallRow);
            }
            retained = allocator.getAllocatedMemory();
            writer.write(smallRow);
            allocated = allocator.getAllocatedMemory() - retained;
            assertTrue(
                    "false".equals(policy) ? allocated >= 16384 : allocated < 16384,
                    "after a sparse batch, the next batch allocated " + allocated + " bytes");
        } finally {
            writer.close();
        }
        List<String> expected = new ArrayList<>(Collections.nCopies(256, large.expected()));
        expected.addAll(Collections.nCopies(257, small.expected()));
        assertEquals(expected, readRows(writer.path()));
        assertEquals(0, allocator.getAllocatedMemory());
    }

    private record CapacityValue(DataType type, Object value, String expected) {}

    private static CapacityValue capacityValue(String shape, int size) {
        long[] values = new long[size];
        Arrays.setAll(values, i -> i);
        return switch (shape) {
            case "string" ->
                new CapacityValue(DataTypes.StringType, UTF8String.fromString("x".repeat(size)), "x".repeat(size));
            case "binary" -> new CapacityValue(DataTypes.BinaryType, new byte[size], Arrays.toString(new byte[size]));
            case "list" ->
                new CapacityValue(
                        DataTypes.createArrayType(DataTypes.LongType, false),
                        new GenericArrayData(values),
                        Arrays.toString(values).replace(" ", ""));
            case "map" -> {
                List<String> entries = new ArrayList<>();
                for (long value : values) {
                    entries.add("{\"key\":" + value + ",\"value\":" + value + "}");
                }
                yield new CapacityValue(
                        DataTypes.createMapType(DataTypes.LongType, DataTypes.LongType, false),
                        new ArrayBasedMapData(new GenericArrayData(values), new GenericArrayData(values)),
                        "[" + String.join(",", entries) + "]");
            }
            case "struct" ->
                new CapacityValue(
                        new StructType().add("text", DataTypes.StringType, true),
                        new GenericInternalRow(new Object[] {UTF8String.fromString("x".repeat(size))}),
                        "{\"text\":\"" + "x".repeat(size) + "\"}");
            default -> throw new IllegalArgumentException(shape);
        };
    }

    @ParameterizedTest
    @ValueSource(strings = {"auto", "true", "false"})
    void byteTriggeredFlushesResetFixedWidthHintsOnSteadyData(String policy) {
        StructType schema = new StructType().add("id", DataTypes.LongType, false);
        VortexOutputWriter writer = writer(
                "steady-byte-capacity.vortex",
                schema,
                Map.of(
                        VortexOutputWriter.WRITE_BATCH_SIZE_OPTION, "8192",
                        VortexOutputWriter.WRITE_BATCH_BYTES_OPTION, "4096",
                        VortexOutputWriter.WRITE_RESET_CAPACITY_OPTION, policy));
        List<String> expected = new ArrayList<>();
        try {
            int firstBatchRows = 0;
            do {
                writer.write(new GenericInternalRow(new Object[] {(long) firstBatchRows}));
                expected.add(Integer.toString(firstBatchRows));
                firstBatchRows++;
            } while (writer.batchRowCount() != 0 && firstBatchRows < 8192);
            assertTrue(firstBatchRows > 1 && firstBatchRows < 8192);
            for (int i = 0; i < firstBatchRows; i++) {
                long value = (long) firstBatchRows + i;
                writer.write(new GenericInternalRow(new Object[] {value}));
                expected.add(Long.toString(value));
                assertEquals((i + 1) % firstBatchRows, writer.batchRowCount());
            }
        } finally {
            writer.close();
        }
        assertEquals(2, writer.summary().batchCount());
        assertEquals(expected, readRows(writer.path()));
    }

    @ParameterizedTest
    @ValueSource(strings = {"", "sometimes"})
    void invalidCapacityPoliciesAreRejectedOnTheDriver(String policy) {
        IllegalArgumentException error = assertThrows(
                IllegalArgumentException.class,
                () -> VortexOutputWriterFactory.validate(
                        options(Map.of(VortexOutputWriter.WRITE_RESET_CAPACITY_OPTION, policy))));
        assertTrue(error.getMessage().contains(VortexOutputWriter.WRITE_RESET_CAPACITY_OPTION));
    }

    @ParameterizedTest
    @CsvSource({"AUTO,AUTO", "' true ',ALWAYS", "FALSE,NEVER"})
    void capacityPolicyValuesAreTrimmedAndCaseInsensitive(String value, VortexOutputWriter.CapacityReset expected) {
        assertEquals(
                expected,
                VortexOutputWriter.configuredCapacityReset(options(Map.of("Vortex.Write.Reset.Capacity", value))));
    }

    /**
     * Spark hands the writer one UnsafeRow object, rewritten in place for every row, so the writer has to consume each
     * row before write() returns, including nulls and nested values, and across batch boundaries. The row's memory may
     * be off-heap, and a byte[] behind a binary value may be overwritten by the caller afterwards.
     */
    @Test
    void consumesReusedRowsAtWriteTime() {
        StructType schema = new StructType()
                .add("id", DataTypes.LongType, false)
                .add("s", DataTypes.StringType, true)
                .add(
                        "st",
                        new StructType().add("a", DataTypes.LongType, false).add("b", DataTypes.StringType, true),
                        true)
                .add("arr", DataTypes.createArrayType(DataTypes.LongType, false), true)
                .add("m", DataTypes.createMapType(DataTypes.StringType, DataTypes.LongType, false), true)
                .add("bin", DataTypes.BinaryType, false);
        UnsafeProjection projection = UnsafeProjection.create(schema);

        VortexOutputWriter writer = writer("reused-rows.vortex", schema, Map.of("vortex.write.batch.size", "4"));
        List<String> expected = new ArrayList<>();
        byte[] bin = new byte[2];
        for (long i = 0; i < 12; i++) {
            boolean odd = i % 2 == 1;
            long[] arr = new long[(int) (i % 3)];
            Arrays.setAll(arr, k -> k);
            bin[0] = (byte) i;
            bin[1] = (byte) -i;
            GenericInternalRow generic = new GenericInternalRow(new Object[] {
                i,
                odd ? null : UTF8String.fromString("s" + i),
                odd ? null : new GenericInternalRow(new Object[] {i, UTF8String.fromString("b" + i)}),
                new GenericArrayData(arr),
                odd
                        ? null
                        : new ArrayBasedMapData(
                                new GenericArrayData(new Object[] {UTF8String.fromString("k" + i)}),
                                new GenericArrayData(new long[] {i})),
                bin,
            });
            UnsafeRow unsafeRow = projection.apply(generic);
            switch ((int) (i % 3)) {
                case 0 -> writer.write(unsafeRow);
                case 1 -> {
                    // The same row in off-heap memory, overwritten and freed as soon as write() returns.
                    int size = unsafeRow.getSizeInBytes();
                    long address = Platform.allocateMemory(size);
                    try {
                        Platform.copyMemory(unsafeRow.getBaseObject(), unsafeRow.getBaseOffset(), null, address, size);
                        UnsafeRow offHeap = new UnsafeRow(schema.length());
                        offHeap.pointTo(null, address, size);
                        writer.write(offHeap);
                        Platform.setMemory(address, (byte) 0x5a, size);
                    } finally {
                        Platform.freeMemory(address);
                    }
                }
                default -> writer.write(generic);
            }
            Arrays.fill(bin, (byte) 0x5a);
            expected.add(String.join(
                    "\t",
                    Long.toString(i),
                    odd ? "null" : "s" + i,
                    odd ? "null" : "{\"a\":" + i + ",\"b\":\"b" + i + "\"}",
                    Arrays.toString(arr).replace(" ", ""),
                    odd ? "null" : "[{\"key\":\"k" + i + "\",\"value\":" + i + "}]",
                    "[" + i + ", " + -i + "]"));
        }
        writer.close();

        assertEquals(expected, readRows(writer.path()));
    }

    /** Nothing is written below a null struct, so its leaves keep their initial capacity however many rows arrive. */
    @Test
    void nullStructsAllocateNothingBelowThem() {
        StructType leaves = new StructType();
        for (int i = 0; i < 32; i++) {
            leaves = leaves.add("l" + i, DataTypes.LongType, true);
        }
        StructType schema =
                new StructType().add("id", DataTypes.LongType, false).add("st", leaves, true);
        VortexOutputWriter writer = writer("null-structs.vortex", schema, Map.of("vortex.write.batch.size", "4096"));
        int rows = 4000;
        for (long i = 0; i < rows; i++) {
            writer.write(new GenericInternalRow(new Object[] {i, null}));
        }
        long held = writer.allocator().getAllocatedMemory();
        writer.close();

        assertTrue(held < 262_144, "the writer holds " + (held / 1024) + " KiB for " + rows + " rows");
        assertEquals(rows, readRows(writer.path()).size());
    }

    /** A row that fails part-way leaves the batch misaligned, so the writer refuses later rows and drops the batch. */
    @Test
    void aRowThatFailsPartWayDiscardsTheBatch() {
        StructType schema =
                new StructType().add("id", DataTypes.LongType, false).add("s", DataTypes.StringType, true);
        VortexOutputWriter writer = writer("failed.vortex", schema, Map.of());
        BufferAllocator allocator = writer.allocator();
        writer.write(new GenericInternalRow(new Object[] {1L, UTF8String.fromString("a")}));
        // id is appended, then s fails: a java.lang.String is not a UTF8String.
        assertThrows(ClassCastException.class, () -> writer.write(new GenericInternalRow(new Object[] {2L, "b"})));
        assertThrows(
                IllegalStateException.class,
                () -> writer.write(new GenericInternalRow(new Object[] {3L, UTF8String.fromString("c")})));
        writer.close();

        assertEquals(0, allocator.getAllocatedMemory());
        assertEquals(List.of(), readRows(writer.path()));
    }

    /** A buffer released after close stands in for a late native release: close succeeds and the memory is freed. */
    @Test
    void aLateReleaseDoesNotFailCloseAndIsStillFreed() {
        StructType schema = new StructType().add("id", DataTypes.LongType, false);
        VortexOutputWriter writer = writer("late-release.vortex", schema, Map.of());
        BufferAllocator allocator = writer.allocator();
        writer.write(new GenericInternalRow(new Object[] {1L}));
        ArrowBuf held = allocator.buffer(64);
        writer.close();
        held.close();

        assertEquals(0, allocator.getAllocatedMemory());
        assertEquals(1, readRows(writer.path()).size());
    }

    @Test
    void decimalsRoundTrip() {
        StructType schema = new StructType()
                .add("small", DataTypes.createDecimalType(18, 4), true)
                .add("wide", DataTypes.createDecimalType(38, 10), true);
        UnsafeProjection projection = UnsafeProjection.create(schema);
        String[][] values = {
            {"0.0000", "0E-10"},
            {"-12345678901234.5678", "-1234567890123456789012345678.0123456789"},
            {"99999999999999.9999", "9999999999999999999999999999.9999999999"},
            {null, null},
        };

        VortexOutputWriter writer = writer("decimals.vortex", schema, Map.of());
        List<String> expected = new ArrayList<>();
        for (int i = 0; i < values.length * 2; i++) {
            String[] row = values[i % values.length];
            GenericInternalRow generic = new GenericInternalRow(new Object[] {
                row[0] == null ? null : Decimal.apply(new BigDecimal(row[0]), 18, 4),
                row[1] == null ? null : Decimal.apply(new BigDecimal(row[1]), 38, 10),
            });
            writer.write(i < values.length ? generic : projection.apply(generic));
            expected.add(row[0] + "\t" + row[1]);
        }
        writer.close();

        assertEquals(expected, readRows(writer.path()));
    }

    @Test
    void storesFooterMetadata() throws IOException {
        StructType schema = new StructType().add("id", DataTypes.LongType, false);
        String path = tempDir.resolve("metadata.vortex").toUri().toString();
        byte[] value = "v1".getBytes(StandardCharsets.UTF_8);
        VortexOutputWriter writer = new VortexOutputWriter(
                path,
                schema,
                VortexOptions.empty(),
                HadoopWritable.create(new Configuration(), path),
                Map.of("app.version", value));
        writer.write(new GenericInternalRow(new Object[] {7L}));
        writer.close();

        Map<String, byte[]> metadata = NativeFiles.readMetadata(VortexSparkSession.get(), path, Map.of());
        assertArrayEquals(value, metadata.get("app.version"));
        assertEquals(List.of("7"), readRows(path));
    }

    private VortexOutputWriter writer(String name, StructType schema, Map<String, String> values) {
        String path = tempDir.resolve(name).toUri().toString();
        return new VortexOutputWriter(path, schema, options(values), HadoopWritable.create(new Configuration(), path));
    }

    private static List<String> readRows(String fileUri) {
        List<String> rows = new ArrayList<>();
        try (BufferAllocator allocator = new RootAllocator()) {
            Scan scan = DataSource.open(VortexSparkSession.get(), fileUri)
                    .scan(ScanOptions.builder().ordered(true).build());
            while (scan.hasNext()) {
                try (ArrowReader reader = scan.next().scanArrow(allocator)) {
                    while (reader.loadNextBatch()) {
                        VectorSchemaRoot root = reader.getVectorSchemaRoot();
                        for (int i = 0; i < root.getRowCount(); i++) {
                            StringBuilder line = new StringBuilder();
                            for (FieldVector vector : root.getFieldVectors()) {
                                if (line.length() > 0) {
                                    line.append('\t');
                                }
                                Object value = vector.getObject(i);
                                line.append(value instanceof byte[] bytes ? Arrays.toString(bytes) : value);
                            }
                            rows.add(line.toString());
                        }
                    }
                } catch (IOException e) {
                    throw new AssertionError(e);
                }
            }
        }
        return rows;
    }

    private static VortexOptions options(Map<String, String> values) {
        return VortexOptions.of(values);
    }
}
