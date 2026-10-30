// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.api.DataSource;
import dev.vortex.api.Scan;
import dev.vortex.api.ScanOptions;
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
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow;
import org.apache.spark.sql.catalyst.expressions.UnsafeProjection;
import org.apache.spark.sql.catalyst.expressions.UnsafeRow;
import org.apache.spark.sql.catalyst.util.ArrayBasedMapData;
import org.apache.spark.sql.catalyst.util.GenericArrayData;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Decimal;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.unsafe.Platform;
import org.apache.spark.unsafe.types.UTF8String;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

final class VortexOutputWriterTest {
    private static final int DEFAULT = 2048;

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
                    // The same row in off-heap memory, freed as soon as write() returns.
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

    /** A failed write leaves the batch misaligned: the writer refuses more rows and close only releases. */
    @Test
    void closeAfterFailedWriteReleasesTheWriter() {
        StructType schema =
                new StructType().add("id", DataTypes.LongType, false).add("s", DataTypes.StringType, true);
        VortexOutputWriter writer = writer("failed.vortex", schema, Map.of());
        BufferAllocator allocator = writer.allocator();
        writer.write(new GenericInternalRow(new Object[] {1L, UTF8String.fromString("a")}));
        assertThrows(ClassCastException.class, () -> writer.write(new GenericInternalRow(new Object[] {2L, "a"})));

        assertThrows(
                IllegalStateException.class,
                () -> writer.write(new GenericInternalRow(new Object[] {3L, UTF8String.fromString("c")})));
        writer.close();
        assertEquals(0, allocator.getAllocatedMemory());
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

    private VortexOutputWriter writer(String name, StructType schema, Map<String, String> values) {
        String path = tempDir.resolve(name).toUri().toString();
        return new VortexOutputWriter(path, schema, options(values), HadoopWritable.create(new Configuration(), path));
    }

    private static List<String> readRows(String fileUri) {
        List<String> rows = new ArrayList<>();
        try (BufferAllocator allocator = new RootAllocator()) {
            Scan scan = DataSource.open(VortexSparkSession.get(), fileUri).scan(ScanOptions.of());
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
