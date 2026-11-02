// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.io.NativeWritable;
import dev.vortex.spark.VortexOptions;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.UncheckedIOException;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import org.apache.spark.SparkConf;
import org.apache.spark.serializer.KryoSerializer;
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow;
import org.apache.spark.sql.catalyst.util.GenericArrayData;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;
import scala.reflect.ClassTag$;

final class VortexOutputWriterSummaryTest {
    private static final StructType SCHEMA = new StructType().add("id", DataTypes.LongType, false);

    @ParameterizedTest
    @CsvSource({"0, 0", "1, 1", "4, 2", "5, 3"})
    void completedSummaryIncludesTheFinalPartialBatch(int rows, long batches) {
        CountingWritable sink = new CountingWritable();
        VortexOutputWriter writer = writer(SCHEMA, 2, sink);
        try {
            assertThrows(IllegalStateException.class, writer::summary);
            for (long i = 0; i < rows; i++) {
                writer.write(new GenericInternalRow(new Object[] {i}));
            }
        } finally {
            writer.close();
        }

        VortexOutputWriterSummary summary = writer.summary();
        assertEquals(sink.bytesWritten.get(), summary.fileSize());
        assertTrue(summary.fileSize() > 0);
        assertEquals(rows, summary.rowCount());
        assertEquals(batches, summary.batchCount());
        if (rows == 0) {
            assertEquals(0, summary.peakBatchArrowBytes());
            assertEquals(0, summary.peakBufferedBytes());
        } else {
            assertTrue(summary.peakBatchArrowBytes() >= Long.BYTES);
            assertTrue(summary.peakBufferedBytes() >= 0);
        }
        writer.close();
        assertSame(summary, writer.summary());
        assertEquals(1, sink.closes);
        assertThrows(IllegalStateException.class, () -> writer.write(new GenericInternalRow(new Object[] {0L})));
        assertSame(summary, writer.summary());
    }

    @Test
    void peakBatchBytesIncludeNestedBuffersWithoutAccumulatingOtherBatches() {
        StructType schema = new StructType().add("items", DataTypes.createArrayType(DataTypes.LongType, false), false);
        CountingWritable sink = new CountingWritable();
        VortexOutputWriter writer = writer(schema, 1, sink);
        int elements = 10_000;
        try {
            for (int i = 0; i < 16; i++) {
                writer.write(new GenericInternalRow(new Object[] {new GenericArrayData(new long[elements])}));
            }
        } finally {
            writer.close();
        }

        VortexOutputWriterSummary summary = writer.summary();
        assertEquals(16, summary.batchCount());
        assertTrue(summary.peakBatchArrowBytes() >= (long) elements * Long.BYTES);
        // Allow Arrow's capacity rounding, but not the sum of all sixteen batches.
        assertTrue(summary.peakBatchArrowBytes() < (long) elements * Long.BYTES * 4);
    }

    @Test
    void aFailedRowDoesNotProduceASuccessfulSummary() {
        VortexOutputWriter writer = writer(SCHEMA, 1, new CountingWritable());
        try {
            writer.write(new GenericInternalRow(new Object[] {1L}));
            assertThrows(ClassCastException.class, () -> writer.write(new GenericInternalRow(new Object[] {"invalid"})));
        } finally {
            writer.close();
        }
        assertThrows(IllegalStateException.class, writer::summary);
        writer.close();
        assertThrows(IllegalStateException.class, writer::summary);
    }

    @ParameterizedTest
    @ValueSource(strings = {"flush", "close"})
    void aFailedFinishOrSinkCloseDoesNotProduceASuccessfulSummary(String failAt) {
        CountingWritable sink = new CountingWritable();
        sink.failAt = failAt;
        VortexOutputWriter writer = writer(SCHEMA, 2, sink);
        writer.write(new GenericInternalRow(new Object[] {1L}));

        assertThrows(UncheckedIOException.class, writer::close);
        assertThrows(IllegalStateException.class, writer::summary);
        writer.close();
        assertEquals(1, sink.closes);
        assertThrows(IllegalStateException.class, writer::summary);
    }

    @Test
    void summaryCanBeCollectedWithJavaSerialization() throws IOException, ClassNotFoundException {
        VortexOutputWriterSummary summary = sampleSummary();
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (ObjectOutputStream output = new ObjectOutputStream(bytes)) {
            output.writeObject(summary);
        }
        try (ObjectInputStream input = new ObjectInputStream(new ByteArrayInputStream(bytes.toByteArray()))) {
            assertSummaryEquals(summary, (VortexOutputWriterSummary) input.readObject());
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void summaryCanBeCollectedWithSparkKryo(boolean unsafe) {
        VortexOutputWriterSummary summary = sampleSummary();
        SparkConf conf = new SparkConf(false).set("spark.kryo.unsafe", Boolean.toString(unsafe));
        var serializer = new KryoSerializer(conf).newInstance();
        var tag = ClassTag$.MODULE$.<VortexOutputWriterSummary>apply(VortexOutputWriterSummary.class);
        VortexOutputWriterSummary copy = serializer.deserialize(serializer.serialize(summary, tag), tag);
        assertSummaryEquals(summary, copy);
    }

    private static VortexOutputWriterSummary sampleSummary() {
        return new VortexOutputWriterSummary(1234, 42, 3, 4096, 8192);
    }

    private static void assertSummaryEquals(VortexOutputWriterSummary expected, VortexOutputWriterSummary actual) {
        assertEquals(expected.fileSize(), actual.fileSize());
        assertEquals(expected.rowCount(), actual.rowCount());
        assertEquals(expected.batchCount(), actual.batchCount());
        assertEquals(expected.peakBatchArrowBytes(), actual.peakBatchArrowBytes());
        assertEquals(expected.peakBufferedBytes(), actual.peakBufferedBytes());
    }

    private static VortexOutputWriter writer(StructType schema, int batchSize, NativeWritable sink) {
        return new VortexOutputWriter(
                "caller-owned.vortex", schema, VortexOptions.of(Map.of("batch.size", Integer.toString(batchSize))), sink);
    }

    private static final class CountingWritable implements NativeWritable {
        private final AtomicLong bytesWritten = new AtomicLong();
        private int closes;
        private String failAt;

        @Override
        public void write(byte[] buffer, int offset, int length) {
            bytesWritten.addAndGet(length);
        }

        @Override
        public void flush() throws IOException {
            if ("flush".equals(failAt)) {
                throw new IOException("injected flush failure");
            }
        }

        @Override
        public void close() throws IOException {
            closes++;
            if ("close".equals(failAt)) {
                throw new IOException("injected close failure");
            }
        }
    }
}
