// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import static org.junit.jupiter.api.Assertions.assertEquals;

import dev.vortex.relocated.org.apache.arrow.memory.BufferAllocator;
import dev.vortex.spark.VortexOptions;
import dev.vortex.spark.io.HadoopWritable;
import java.nio.file.Path;
import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.unsafe.types.UTF8String;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

final class VortexOutputWriterMemoryTest {
    @TempDir
    Path tempDir;

    @Test
    @Disabled("each batch's schema is exported and closed without being released, so its children stay allocated")
    void closingAWriterReleasesEveryArrowAllocation() throws Exception {
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
        // Spark exposes no allocator metrics, so look at the writer's own allocator.
        var field = VortexOutputWriter.class.getDeclaredField("allocator");
        field.setAccessible(true);
        BufferAllocator allocator = (BufferAllocator) field.get(writer);

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
}
