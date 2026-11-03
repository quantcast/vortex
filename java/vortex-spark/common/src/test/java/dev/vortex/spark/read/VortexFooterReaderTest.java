// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

final class VortexFooterReaderTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void equivalentNamesRetainFirstSpellingAndMergeNullability(boolean reverse) {
        StructType lower = new StructType().add("id", DataTypes.LongType, false);
        StructType upper = new StructType().add("ID", DataTypes.LongType, true);

        StructType merged = VortexFooterReader.mergeStructs(reverse ? upper : lower, reverse ? lower : upper, false);

        assertEquals(new StructType().add(reverse ? "ID" : "id", DataTypes.LongType, true), merged);
        assertFalse(VortexFooterReader.mergeStructs(lower, lower, false).apply("id").nullable());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void caseSensitiveNamesStaySeparateAndBecomeNullable(boolean reverse) {
        StructType lower = new StructType().add("id", DataTypes.LongType, false);
        StructType upper = new StructType().add("ID", DataTypes.LongType, false);

        StructType merged = VortexFooterReader.mergeStructs(reverse ? upper : lower, reverse ? lower : upper, true);

        assertEquals(
                new StructType()
                        .add(reverse ? "ID" : "id", DataTypes.LongType, true)
                        .add(reverse ? "id" : "ID", DataTypes.LongType, true),
                merged);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void equivalentNamesWithIncompatibleTypesAreRejected(boolean reverse) {
        StructType lower = new StructType().add("id", DataTypes.LongType, false);
        StructType upper = new StructType().add("ID", DataTypes.StringType, false);

        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class,
                () -> VortexFooterReader.mergeStructs(reverse ? upper : lower, reverse ? lower : upper, false));

        assertTrue(failure.getMessage().contains("field " + (reverse ? "ID" : "id") + " is"));
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void ambiguousNamesInEitherSchemaAreRejected(boolean reverse) {
        StructType ambiguous =
                new StructType().add("id", DataTypes.LongType, false).add("ID", DataTypes.LongType, false);
        StructType unique = new StructType().add("id", DataTypes.LongType, false);

        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class,
                () -> VortexFooterReader.mergeStructs(
                        reverse ? unique : ambiguous, reverse ? ambiguous : unique, false));

        assertTrue(failure.getMessage().contains("Ambiguous columns id and ID"));
        assertEquals(2, VortexFooterReader.mergeStructs(ambiguous, ambiguous, true).size());
    }
}
