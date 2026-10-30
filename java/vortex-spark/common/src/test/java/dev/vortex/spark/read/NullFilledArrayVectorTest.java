// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.relocated.org.apache.arrow.memory.BufferAllocator;
import dev.vortex.relocated.org.apache.arrow.memory.RootAllocator;
import dev.vortex.relocated.org.apache.arrow.vector.complex.ListVector;
import dev.vortex.relocated.org.apache.arrow.vector.complex.impl.UnionListWriter;
import dev.vortex.relocated.org.apache.arrow.vector.complex.writer.BaseWriter.StructWriter;
import org.apache.spark.sql.types.ArrayType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarArray;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

final class NullFilledArrayVectorTest {
    @Test
    @DisplayName("A null array reads as null, and a present one with its missing element fields as null")
    void nullArray() {
        ArrayType read =
                new ArrayType(new StructType().add("b", DataTypes.LongType).add("a", DataTypes.LongType), true);
        StructType file =
                new StructType().add("arr", new ArrayType(new StructType().add("a", DataTypes.LongType), true));
        NestedSchemaPruning.Shape shape = NestedSchemaPruning.projection(
                        new StructType().add("arr", read), file, false, "f")
                .shape(0);

        try (BufferAllocator allocator = new RootAllocator();
                ListVector list = ListVector.empty("arr", allocator)) {
            UnionListWriter writer = list.getWriter();
            writer.allocate();
            writer.setPosition(0);
            writer.startList();
            StructWriter element = writer.struct();
            element.start();
            element.bigInt("a").writeBigInt(7);
            element.end();
            writer.endList();
            list.setValueCount(2);

            ColumnVector vector = NullFilledStructVector.fill(new VortexArrowColumnVector(list), read, shape, 2);

            assertTrue(vector.isNullAt(1));
            assertNull(vector.getArray(1));
            ColumnarArray array = vector.getArray(0);
            assertEquals(1, array.numElements());
            assertTrue(array.getStruct(0, 2).isNullAt(0));
            assertEquals(7, array.getStruct(0, 2).getLong(1));
            vector.close();
        }
    }
}
