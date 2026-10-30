// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.arrow.ArrowAllocation;
import dev.vortex.jni.NativeLoader;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.StructVector;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * {@link VortexWriter.Builder#copyBatches}: each batch's Arrow memory is free once {@code writeBatch} returns, though
 * the file isn't finished, and the file reads back the same.
 *
 * <p>Each batch is allocated from an allocator of its own, so what the native writer still holds of it shows as that
 * allocator's memory after the batch's vectors are closed. The import's own structs, a few hundred bytes, are released
 * on a native thread and may be outstanding for a moment, so a batch counts as freed once less than a kilobyte is left.
 */
public final class CopyBatchesTest {
    private static final int BATCHES = 8;
    private static final int ROWS = 2048;

    @TempDir
    static Path tempDir;

    private static final Schema SCHEMA = new Schema(List.of(new Field(
            "request",
            FieldType.nullable(ArrowType.Struct.INSTANCE),
            List.of(Field.nullable("id", new ArrowType.Int(32, true)), Field.nullable("name", new ArrowType.Utf8())))));

    @BeforeAll
    public static void loadLibrary() {
        NativeLoader.loadJni();
    }

    @Test
    public void eachBatchIsFreedAsItIsWritten() throws IOException {
        Session session = Session.create();
        String uri = tempDir.resolve("copied.vortex").toUri().toString();
        BufferAllocator root = ArrowAllocation.rootAllocator();

        List<BufferAllocator> allocators = new ArrayList<>();
        List<Long> held = new ArrayList<>();
        try (VortexWriter writer = VortexWriter.builder(session, uri, SCHEMA, root)
                .copyBatches(true)
                .build()) {
            for (int b = 0; b < BATCHES; b++) {
                BufferAllocator batchAllocator = root.newChildAllocator("batch-" + b, 0, Long.MAX_VALUE);
                allocators.add(batchAllocator);
                writeBatch(writer, batchAllocator, b);
                held.add(batchAllocator.getAllocatedMemory());
            }
            writer.finish();
        }
        allocators.forEach(BufferAllocator::close);

        assertTrue(held.stream().allMatch(bytes -> bytes < 1024), "bytes each batch still held: " + held);
        assertEquals(expectedIds(), readIds(session, uri));
    }

    /** Writes batch {@code b}: ids {@code b * ROWS} on, every seventh row a null request. */
    private static void writeBatch(VortexWriter writer, BufferAllocator allocator, int b) throws IOException {
        try (VectorSchemaRoot batch = VectorSchemaRoot.create(SCHEMA, allocator)) {
            StructVector request = (StructVector) batch.getVector("request");
            IntVector id = request.getChild("id", IntVector.class);
            VarCharVector name = request.getChild("name", VarCharVector.class);
            request.allocateNew();
            for (int i = 0; i < ROWS; i++) {
                if (i % 7 == 0) {
                    request.setNull(i);
                    continue;
                }
                request.setIndexDefined(i);
                id.setSafe(i, b * ROWS + i);
                name.setSafe(i, ("name-" + i).getBytes(UTF_8));
            }
            request.setValueCount(ROWS);
            batch.setRowCount(ROWS);
            try (ArrowArray array = ArrowArray.allocateNew(allocator);
                    ArrowSchema schema = ArrowSchema.allocateNew(allocator)) {
                Data.exportVectorSchemaRoot(allocator, batch, null, array, schema);
                writer.writeBatch(array.memoryAddress(), schema.memoryAddress());
                // The writer only reads the schema, so its export is ours to release.
                schema.release();
            }
        }
    }

    private static List<Integer> expectedIds() {
        List<Integer> ids = new ArrayList<>();
        for (int b = 0; b < BATCHES; b++) {
            for (int i = 0; i < ROWS; i++) {
                if (i % 7 != 0) {
                    ids.add(b * ROWS + i);
                }
            }
        }
        return ids;
    }

    /** The ids of the non-null requests, in file order. */
    private static List<Integer> readIds(Session session, String uri) throws IOException {
        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Scan scan = DataSource.open(session, uri)
                .scan(ScanOptions.builder().ordered(true).build());
        List<Integer> ids = new ArrayList<>();
        while (scan.hasNext()) {
            try (ArrowReader reader = scan.next().scanArrow(allocator)) {
                while (reader.loadNextBatch()) {
                    StructVector request =
                            (StructVector) reader.getVectorSchemaRoot().getVector("request");
                    IntVector id = request.getChild("id", IntVector.class);
                    for (int i = 0; i < request.getValueCount(); i++) {
                        if (!request.isNull(i)) {
                            ids.add(id.get(i));
                        }
                    }
                }
            }
        }
        return ids;
    }
}
