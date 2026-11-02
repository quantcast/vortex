// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.relocated.org.apache.arrow.memory.ArrowBuf;
import dev.vortex.relocated.org.apache.arrow.memory.RootAllocator;
import dev.vortex.relocated.org.apache.arrow.vector.BigIntVector;
import dev.vortex.relocated.org.apache.arrow.vector.VarBinaryVector;
import dev.vortex.relocated.org.apache.arrow.vector.VarCharVector;
import dev.vortex.relocated.org.apache.arrow.vector.VectorSchemaRoot;
import dev.vortex.relocated.org.apache.arrow.vector.complex.ListVector;
import dev.vortex.relocated.org.apache.arrow.vector.complex.StructVector;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.Test;

final class ArrowBatchMemoryTest {
    @Test
    void measuringAnUnfinishedBatchPreservesStringsAndBinaryAcrossBufferGrowth() {
        ArrowBatchMemory memory = new ArrowBatchMemory();
        StructType schema = new StructType().add("s", DataTypes.StringType).add("b", DataTypes.BinaryType);
        try (RootAllocator allocator = new RootAllocator(memory, Long.MAX_VALUE);
                VectorSchemaRoot root = VectorSchemaRoot.create(SparkToArrowSchema.convert(schema), allocator)) {
            root.allocateNew();
            VarCharVector strings = (VarCharVector) root.getVector("s");
            VarBinaryVector binary = (VarBinaryVector) root.getVector("b");
            strings.setSafe(0, "first".getBytes(StandardCharsets.UTF_8));
            binary.setSafe(0, new byte[] {1, 2, 3});
            memory.bytes(root);

            String largeString = "x".repeat(100_000);
            byte[] largeBinary = new byte[100_000];
            Arrays.fill(largeBinary, (byte) 7);
            strings.setSafe(1, largeString.getBytes(StandardCharsets.UTF_8));
            binary.setSafe(1, largeBinary);
            memory.bytes(root);
            root.setRowCount(2);

            assertEquals("first", strings.getObject(0).toString());
            assertEquals(largeString, strings.getObject(1).toString());
            assertArrayEquals(new byte[] {1, 2, 3}, binary.get(0));
            assertArrayEquals(largeBinary, binary.get(1));
        }
    }

    @Test
    void measuresNestedAllocationsOnceAndIgnoresOlderBatchReleases() {
        ArrowBatchMemory memory = new ArrowBatchMemory();
        StructType schema = new StructType()
                .add("id", DataTypes.LongType, false)
                .add("nested", new StructType().add("items", DataTypes.createArrayType(DataTypes.LongType, false)));
        try (RootAllocator allocator = new RootAllocator(memory, Long.MAX_VALUE);
                VectorSchemaRoot root = VectorSchemaRoot.create(SparkToArrowSchema.convert(schema), allocator)) {
            long batchBytes;
            try (ArrowBuf previousBatch = allocator.buffer(1_000_000)) {
                root.allocateNew();
                batchBytes = memory.bytes(root);
                assertEquals(
                        allocator.getAllocatedMemory()
                                - previousBatch.getReferenceManager().getSize(),
                        batchBytes);
                assertTrue(batchBytes > 0);
            }
            // A baseline subtraction would shrink here, although the open batch has not changed.
            assertEquals(batchBytes, memory.bytes(root));
            assertEquals(allocator.getAllocatedMemory(), memory.bytes(root));

            StructVector nested = (StructVector) root.getVector("nested");
            ListVector items = (ListVector) nested.getChild("items");
            ((BigIntVector) items.getDataVector()).setSafe(100_000, 1L);
            assertTrue(memory.bytes(root) > batchBytes);
            assertEquals(allocator.getAllocatedMemory(), memory.bytes(root));

            root.clear();
            memory.reset();
            assertEquals(0, memory.bytes(root));
            root.allocateNew();
            assertEquals(allocator.getAllocatedMemory(), memory.bytes(root));
        }
    }
}
