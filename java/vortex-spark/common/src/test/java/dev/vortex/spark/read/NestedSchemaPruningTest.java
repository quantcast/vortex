// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.apache.spark.sql.types.ArrayType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

final class NestedSchemaPruningTest {
    private static final StructType FILE = new StructType()
            .add("s", new StructType().add("a", DataTypes.LongType).add("B", DataTypes.LongType));

    @Test
    @DisplayName("A non-nullable struct field the file does not carry fails, naming the field")
    void absentNonNullableField() {
        StructType read = new StructType()
                .add("s", new StructType().add("a", DataTypes.LongType).add("c", DataTypes.LongType, false));

        IllegalArgumentException error = assertThrows(
                IllegalArgumentException.class, () -> NestedSchemaPruning.projection(read, FILE, false, "f.vortex"));

        assertEquals("f.vortex does not carry the non-nullable field s.c that the query requires", error.getMessage());
    }

    @Test
    @DisplayName("Struct fields that differ only in case are ambiguous unless Spark is case sensitive")
    void ambiguousFields() {
        StructType file = new StructType()
                .add("s", new StructType().add("b", DataTypes.LongType).add("B", DataTypes.LongType));
        StructType read = new StructType().add("s", new StructType().add("b", DataTypes.LongType));

        assertThrows(IllegalArgumentException.class, () -> NestedSchemaPruning.projection(read, file, false, "f"));
        assertNull(NestedSchemaPruning.projection(read, file, true, "f").shape(0));
    }

    @Test
    @DisplayName("Fields the file lacks are filled in at the positions Spark reads them")
    void absentFieldSlots() {
        StructType read = new StructType()
                .add(
                        "s",
                        new StructType()
                                .add("c", DataTypes.LongType)
                                .add("b", DataTypes.LongType)
                                .add("d", DataTypes.LongType));

        NestedSchemaPruning.Shape shape =
                NestedSchemaPruning.projection(read, FILE, false, "f").shape(0);

        assertArrayEquals(new int[] {-1, 0, -1}, new int[] {shape.slot(0), shape.slot(1), shape.slot(2)});
    }

    @Test
    @DisplayName("A struct the file carries none of the read fields of is read for its nulls alone")
    void structWithoutReadFields() {
        StructType read = new StructType().add("s", new StructType().add("c", DataTypes.LongType));

        assertEquals(
                -1,
                NestedSchemaPruning.projection(read, FILE, false, "f").shape(0).slot(0));
    }

    @Test
    @DisplayName("Struct fields are pruned through arrays, keeping each array's element nullability")
    void pruneThroughArrays() {
        StructType inner = new StructType().add("h", DataTypes.LongType).add("x", DataTypes.StringType);
        StructType table = new StructType()
                .add("a", DataTypes.LongType)
                .add("g", new ArrayType(inner, true))
                .add("w", DataTypes.StringType);
        StructType required =
                new StructType().add("g", new ArrayType(new StructType().add("h", DataTypes.LongType), true));

        assertEquals(
                new ArrayType(required, false),
                NestedSchemaPruning.prune(new ArrayType(table, false), new ArrayType(required, true)));
    }

    @Test
    @DisplayName("Fields of array elements the file lacks are filled in at the positions Spark reads them")
    void absentArrayElementFieldSlots() {
        StructType file =
                new StructType().add("arr", new ArrayType(new StructType().add("a", DataTypes.LongType), true));
        StructType read = new StructType()
                .add(
                        "arr",
                        new ArrayType(
                                new StructType().add("b", DataTypes.LongType).add("a", DataTypes.LongType), true));

        NestedSchemaPruning.Shape element =
                NestedSchemaPruning.projection(read, file, false, "f").shape(0).element();

        assertArrayEquals(new int[] {-1, 0}, new int[] {element.slot(0), element.slot(1)});
    }
}
