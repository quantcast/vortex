// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.api.DataSource;
import dev.vortex.api.Scan;
import dev.vortex.api.ScanOptions;
import dev.vortex.relocated.org.apache.arrow.memory.BufferAllocator;
import dev.vortex.relocated.org.apache.arrow.memory.RootAllocator;
import dev.vortex.relocated.org.apache.arrow.vector.FieldVector;
import dev.vortex.relocated.org.apache.arrow.vector.VectorSchemaRoot;
import dev.vortex.relocated.org.apache.arrow.vector.ipc.ArrowReader;
import dev.vortex.spark.VortexSparkSession;
import java.io.IOException;
import java.math.BigDecimal;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow;
import org.apache.spark.sql.catalyst.expressions.UnsafeProjection;
import org.apache.spark.sql.catalyst.expressions.UnsafeRow;
import org.apache.spark.sql.catalyst.util.ArrayBasedMapData;
import org.apache.spark.sql.catalyst.util.GenericArrayData;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Decimal;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;
import org.apache.spark.unsafe.Platform;
import org.apache.spark.unsafe.types.UTF8String;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

final class VortexDataWriterTest {

    @TempDir
    Path tempDir;

    /**
     * The native writer keeps every batch it receives until a row block is complete, so the memory each batch takes
     * must be proportional to its rows, not to Arrow's default vector capacity of a few thousand values.
     */
    @Test
    void memoryGrowsWithRowsNotBatches() throws IOException {
        StructType schema = new StructType()
                .add("id", DataTypes.LongType, false)
                .add("s", DataTypes.StringType, true)
                .add("a", DataTypes.createArrayType(DataTypes.LongType, false), true);
        String fileUri = tempDir.resolve("tiny-batches.vortex").toUri().toString();
        CaseInsensitiveStringMap options = new CaseInsensitiveStringMap(Map.of("vortex.write.batch.size", "1"));

        VortexDataWriter writer = new VortexDataWriter(fileUri, schema, options);
        int rows = 400;
        for (long i = 0; i < rows; i++) {
            writer.write(new GenericInternalRow(
                    new Object[] {i, UTF8String.fromString("row-" + i), new GenericArrayData(new long[] {i, i + 1})}));
        }
        // Measured before commit, while the native writer still holds all the batches.
        long held = writer.allocatedBytes();
        VortexWriterCommitMessage message = (VortexWriterCommitMessage) writer.commit();

        assertEquals(rows, message.recordCount());
        assertTrue(held < 4L << 20, "the writer holds " + (held >> 10) + " KiB for " + rows + " rows");
    }

    /**
     * Spark hands the writer one UnsafeRow object, rewritten in place for every row, so the writer has to consume each
     * row before write() returns, including nulls and nested values, and across batch boundaries. The row's memory may
     * be off-heap, and a byte[] behind a binary value may be overwritten by the caller afterwards.
     */
    @Test
    void consumesReusedRowsAtWriteTime() throws IOException {
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
        String fileUri = tempDir.resolve("reused-rows.vortex").toUri().toString();
        CaseInsensitiveStringMap options = new CaseInsensitiveStringMap(Map.of("vortex.write.batch.size", "4"));

        VortexDataWriter writer = new VortexDataWriter(fileUri, schema, options);
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
        writer.commit();

        assertEquals(expected, readRows(fileUri, options));
    }

    /** A write that fails part-way leaves the batch misaligned; commit must then fail and release the writer. */
    @Test
    void commitAfterFailedWriteReleasesTheWriter() throws IOException {
        StructType schema =
                new StructType().add("id", DataTypes.LongType, false).add("s", DataTypes.StringType, true);
        String fileUri = tempDir.resolve("failed.vortex").toUri().toString();
        VortexDataWriter writer = new VortexDataWriter(fileUri, schema, new CaseInsensitiveStringMap(Map.of()));
        writer.write(new GenericInternalRow(new Object[] {1L, UTF8String.fromString("a")}));
        assertThrows(ClassCastException.class, () -> writer.write(new GenericInternalRow(new Object[] {2L, "a"})));

        assertThrows(IOException.class, writer::commit);
        assertEquals(0, writer.allocatedBytes());
    }

    @Test
    void decimalsRoundTrip() throws IOException {
        StructType schema = new StructType()
                .add("small", DataTypes.createDecimalType(18, 4), true)
                .add("wide", DataTypes.createDecimalType(38, 10), true);
        UnsafeProjection projection = UnsafeProjection.create(schema);
        String fileUri = tempDir.resolve("decimals.vortex").toUri().toString();
        CaseInsensitiveStringMap options = new CaseInsensitiveStringMap(Map.of());
        String[][] values = {
            {"0.0000", "0E-10"},
            {"-12345678901234.5678", "-1234567890123456789012345678.0123456789"},
            {"99999999999999.9999", "9999999999999999999999999999.9999999999"},
            {null, null},
        };

        VortexDataWriter writer = new VortexDataWriter(fileUri, schema, options);
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
        writer.commit();

        assertEquals(expected, readRows(fileUri, options));
    }

    private static List<String> readRows(String fileUri, CaseInsensitiveStringMap options) throws IOException {
        List<String> rows = new ArrayList<>();
        try (BufferAllocator allocator = new RootAllocator()) {
            Scan scan = DataSource.open(VortexSparkSession.get(options.asCaseSensitiveMap()), fileUri)
                    .scan(ScanOptions.of());
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
                }
            }
        }
        return rows;
    }
}
