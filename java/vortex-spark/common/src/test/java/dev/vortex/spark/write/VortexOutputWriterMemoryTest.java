// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.relocated.org.apache.arrow.memory.BufferAllocator;
import dev.vortex.spark.VortexOptions;
import dev.vortex.spark.io.HadoopWritable;
import java.nio.file.Path;
import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow;
import org.apache.spark.sql.catalyst.util.GenericArrayData;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.unsafe.types.UTF8String;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

final class VortexOutputWriterMemoryTest {
    @TempDir
    Path tempDir;

    @Test
    void closingAWriterReleasesEveryArrowAllocation() {
        StructType schema = new StructType()
                .add("id", DataTypes.LongType, false)
                .add("name", DataTypes.StringType, true)
                .add("tags", DataTypes.createArrayType(DataTypes.StringType), true);
        String path = tempDir.resolve("part-00000.vortex").toUri().toString();
        VortexOutputWriter writer = new VortexOutputWriter(
                path,
                schema,
                VortexOptions.of(Map.of("vortex.write.batch.size", "8")),
                HadoopWritable.create(new Configuration(), path));
        BufferAllocator allocator = writer.allocator();

        for (long id = 0; id < 40; id++) {
            writer.write(new GenericInternalRow(new Object[] {
                id,
                id % 3 == 0 ? null : UTF8String.fromString("name-" + id),
                new org.apache.spark.sql.catalyst.util.GenericArrayData(new Object[] {UTF8String.fromString("t")})
            }));
        }
        writer.close();

        assertEquals(0, allocator.getAllocatedMemory());
    }

    @Test
    void writtenBatchesAreFreedBeforeClose() {
        StructType schema = new StructType()
                .add("id", DataTypes.LongType, false)
                .add(
                        "s",
                        new StructType()
                                .add("name", DataTypes.StringType, true)
                                .add("tags", DataTypes.createArrayType(DataTypes.StringType), true),
                        true);
        int batchSize = 8192;
        int batches = 64;
        String path = tempDir.resolve("part-00001.vortex").toUri().toString();
        VortexOutputWriter writer = new VortexOutputWriter(
                path,
                schema,
                VortexOptions.of(Map.of("vortex.write.batch.size", Integer.toString(batchSize))),
                HadoopWritable.create(new Configuration(), path));
        BufferAllocator allocator = writer.allocator();

        long oneBatch = 0;
        for (long id = 0; id < (long) batches * batchSize; id++) {
            if (id == batchSize - 1) {
                oneBatch = allocator.getAllocatedMemory();
            }
            UTF8String name = UTF8String.fromString("name-" + id);
            Object s = id % 2 == 0
                    ? null
                    : new GenericInternalRow(new Object[] {name, new GenericArrayData(new Object[] {name, name})});
            writer.write(new GenericInternalRow(new Object[] {id, s}));
        }
        long held = allocator.getAllocatedMemory();
        writer.close();

        // The writer keeps a few batches in flight, but must not hold every batch until the file finishes.
        assertTrue(held < oneBatch * batches / 2, held + " bytes held after " + batches + " batches of " + oneBatch);
        assertEquals(0, allocator.getAllocatedMemory());
    }
}
