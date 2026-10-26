// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow;
import org.apache.spark.sql.catalyst.util.GenericArrayData;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;
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
}
