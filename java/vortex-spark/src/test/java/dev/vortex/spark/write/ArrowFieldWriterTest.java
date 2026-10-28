// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

import dev.vortex.relocated.org.apache.arrow.memory.BufferAllocator;
import dev.vortex.relocated.org.apache.arrow.memory.RootAllocator;
import dev.vortex.relocated.org.apache.arrow.vector.VectorSchemaRoot;
import java.util.stream.Stream;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

final class ArrowFieldWriterTest {

    static Stream<DataType> convertedTypes() {
        StructType struct =
                new StructType().add("a", DataTypes.IntegerType, true).add("b", DataTypes.StringType, true);
        return Stream.of(
                DataTypes.BooleanType,
                DataTypes.ByteType,
                DataTypes.ShortType,
                DataTypes.IntegerType,
                DataTypes.LongType,
                DataTypes.FloatType,
                DataTypes.DoubleType,
                DataTypes.StringType,
                DataTypes.BinaryType,
                DataTypes.DateType,
                DataTypes.TimestampType,
                DataTypes.TimestampNTZType,
                DataTypes.createDecimalType(18, 4),
                DataTypes.createDecimalType(38, 10),
                struct,
                new StructType().add("inner", struct, true),
                DataTypes.createArrayType(DataTypes.LongType, true),
                DataTypes.createArrayType(struct, true),
                DataTypes.createMapType(DataTypes.StringType, DataTypes.LongType, true),
                DataTypes.createMapType(DataTypes.StringType, DataTypes.createArrayType(struct, true), true));
    }

    /** The writer opens the native file before building field writers, so the two type dispatchers must agree. */
    @ParameterizedTest
    @MethodSource("convertedTypes")
    void everyConvertedTypeHasAFieldWriter(DataType type) {
        StructType schema = new StructType().add("f", type, true);
        try (BufferAllocator allocator = new RootAllocator();
                VectorSchemaRoot root = VectorSchemaRoot.create(SparkToArrowSchema.convert(schema), allocator)) {
            assertDoesNotThrow(() -> ArrowFieldWriter.create(root.getVector(0), type));
        }
    }

    @Test
    void unsupportedTypesFailBeforeTheFileIsOpened() {
        StructType schema = new StructType().add("f", DataTypes.CalendarIntervalType, true);
        assertThrows(RuntimeException.class, () -> SparkToArrowSchema.convert(schema));
    }
}
