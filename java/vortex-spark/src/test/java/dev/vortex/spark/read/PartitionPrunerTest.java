// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.Map;
import org.apache.spark.sql.connector.expressions.Expression;
import org.apache.spark.sql.connector.expressions.Expressions;
import org.apache.spark.sql.connector.expressions.LiteralValue;
import org.apache.spark.sql.connector.expressions.filter.And;
import org.apache.spark.sql.connector.expressions.filter.Not;
import org.apache.spark.sql.connector.expressions.filter.Or;
import org.apache.spark.sql.connector.expressions.filter.Predicate;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.unsafe.types.UTF8String;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/** Unit tests for {@link PartitionPruner}, which skips files by the Hive-style partition values in their paths. */
final class PartitionPrunerTest {

    private static final StructType SCHEMA = new StructType()
            .add("id", DataTypes.IntegerType)
            .add("year", DataTypes.IntegerType)
            .add("region", DataTypes.StringType)
            .add("score", DataTypes.DoubleType);

    private static final String NULL_PARTITION = "__HIVE_DEFAULT_PARTITION__";

    private static Predicate cmp(String op, String column, Object value, DataType type) {
        Object internal = value instanceof String s ? UTF8String.fromString(s) : value;
        return new Predicate(op, new Expression[] {Expressions.column(column), new LiteralValue<>(internal, type)});
    }

    private static Predicate yearIs(String op, int value) {
        return cmp(op, "year", value, DataTypes.IntegerType);
    }

    private static Predicate regionIs(String value) {
        return cmp("=", "region", value, DataTypes.StringType);
    }

    private static boolean mayMatch(Predicate filter, Map<String, String> partitionValues) {
        return new PartitionPruner(new Predicate[] {filter}, SCHEMA).mayMatch(partitionValues);
    }

    static List<Arguments> comparisons() {
        return List.of(
                Arguments.of("=", 2024, true),
                Arguments.of("=", 2025, false),
                Arguments.of("<>", 2024, false),
                Arguments.of("!=", 2025, true),
                Arguments.of(">", 2023, true),
                Arguments.of(">", 2024, false),
                Arguments.of(">=", 2024, true),
                Arguments.of("<", 2024, false),
                Arguments.of("<=", 2024, true),
                Arguments.of("<=>", 2024, true),
                Arguments.of("<=>", 2025, false));
    }

    @ParameterizedTest(name = "year {0} {1} on year=2024 -> {2}")
    @MethodSource("comparisons")
    @DisplayName("Comparisons against the parsed partition value")
    void comparisons(String op, int literal, boolean expected) {
        assertEquals(expected, mayMatch(yearIs(op, literal), Map.of("year", "2024")));
    }

    @Test
    @DisplayName("Comparisons with a null partition value are null, so the file is skipped")
    void nullPartitionValueComparison() {
        Map<String, String> nullYear = Map.of("year", NULL_PARTITION);
        assertEquals(false, mayMatch(yearIs("=", 2024), nullYear));
        assertEquals(false, mayMatch(new Not(yearIs("=", 2024)), nullYear));
        assertEquals(false, mayMatch(yearIs("<=>", 2024), nullYear));
        assertEquals(true, mayMatch(cmp("<=>", "year", null, DataTypes.IntegerType), nullYear));
    }

    @Test
    @DisplayName("IS_NULL and IS_NOT_NULL use the Hive default partition as null")
    void nullChecks() {
        Predicate isNull = new Predicate("IS_NULL", new Expression[] {Expressions.column("year")});
        Predicate isNotNull = new Predicate("IS_NOT_NULL", new Expression[] {Expressions.column("year")});
        assertEquals(true, mayMatch(isNull, Map.of("year", NULL_PARTITION)));
        assertEquals(false, mayMatch(isNull, Map.of("year", "2024")));
        assertEquals(false, mayMatch(isNotNull, Map.of("year", NULL_PARTITION)));
        assertEquals(true, mayMatch(isNotNull, Map.of("year", "2024")));
    }

    @Test
    @DisplayName("IN matches any listed value and is null when the list contains null")
    void in() {
        Predicate in = new Predicate("IN", new Expression[] {
            Expressions.column("year"),
            new LiteralValue<>(2023, DataTypes.IntegerType),
            new LiteralValue<>(2024, DataTypes.IntegerType)
        });
        assertEquals(true, mayMatch(in, Map.of("year", "2024")));
        assertEquals(false, mayMatch(in, Map.of("year", "2025")));

        Predicate inAfterNull = new Predicate("IN", new Expression[] {
            Expressions.column("year"),
            new LiteralValue<>(null, DataTypes.IntegerType),
            new LiteralValue<>(2024, DataTypes.IntegerType)
        });
        assertEquals(true, mayMatch(inAfterNull, Map.of("year", "2024")));

        Predicate notInWithNull = new Not(new Predicate("IN", new Expression[] {
            Expressions.column("year"),
            new LiteralValue<>(2023, DataTypes.IntegerType),
            new LiteralValue<>(null, DataTypes.IntegerType)
        }));
        assertEquals(false, mayMatch(notInWithNull, Map.of("year", "2025")));
    }

    @Test
    @DisplayName("String partition values support equality and STARTS_WITH")
    void strings() {
        assertEquals(true, mayMatch(regionIs("us east"), Map.of("region", "us east")));
        assertEquals(false, mayMatch(regionIs("eu"), Map.of("region", "us east")));
        Predicate startsWith = cmp("STARTS_WITH", "region", "us", DataTypes.StringType);
        assertEquals(true, mayMatch(startsWith, Map.of("region", "us east")));
        assertEquals(false, mayMatch(startsWith, Map.of("region", "eu")));
    }

    @Test
    @DisplayName("Double partition values compare NaN equal to NaN, as Spark does")
    void doubleNaN() {
        Predicate nan = cmp("=", "score", Double.NaN, DataTypes.DoubleType);
        assertEquals(true, mayMatch(nan, Map.of("score", "NaN")));
        assertEquals(false, mayMatch(nan, Map.of("score", "1.5")));
    }

    @Test
    @DisplayName("AND with a data column prunes on the partition side alone")
    void andWithDataColumn() {
        Predicate and = new And(yearIs("=", 2024), cmp(">", "id", 5, DataTypes.IntegerType));
        assertEquals(false, mayMatch(and, Map.of("year", "2025")));
        assertEquals(true, mayMatch(and, Map.of("year", "2024")));
    }

    @Test
    @DisplayName("OR with a data column keeps the file unless the partition side is the only way to match")
    void orWithDataColumn() {
        Predicate or = new Or(yearIs("=", 2024), cmp(">", "id", 5, DataTypes.IntegerType));
        assertEquals(true, mayMatch(or, Map.of("year", "2025")));

        Predicate partitionOnly = new Or(yearIs("=", 2024), regionIs("eu"));
        assertEquals(false, mayMatch(partitionOnly, Map.of("year", "2025", "region", "us")));
        assertEquals(true, mayMatch(partitionOnly, Map.of("year", "2025", "region", "eu")));
    }

    @Test
    @DisplayName("Unknown operators, mismatched types, missing or unparsable values keep the file")
    void unknownKeepsFile() {
        Predicate unknownOp = new Predicate("UPPER_EQ", new Expression[] {Expressions.column("region")});
        assertEquals(true, mayMatch(unknownOp, Map.of("region", "us")));
        assertEquals(true, mayMatch(cmp("=", "year", 2025L, DataTypes.LongType), Map.of("year", "2024")));
        assertEquals(true, mayMatch(yearIs("=", 2025), Map.of()));
        assertEquals(true, mayMatch(yearIs("=", 2025), Map.of("year", "not-a-number")));
    }

    @Test
    @DisplayName("prune keeps matching paths in order and requires every filter to allow the file")
    void prunePaths() {
        PartitionPruner pruner = new PartitionPruner(new Predicate[] {yearIs(">=", 2024), regionIs("us")}, SCHEMA);
        List<String> paths = List.of(
                "/t/year=2023/region=us/a.vortex",
                "/t/year=2024/region=us/b.vortex",
                "/t/year=2024/region=eu/c.vortex",
                "/t/year=2025/region=us/d.vortex");
        assertEquals(
                List.of("/t/year=2024/region=us/b.vortex", "/t/year=2025/region=us/d.vortex"), pruner.prune(paths));
        assertEquals(paths, new PartitionPruner(new Predicate[0], SCHEMA).prune(paths));
    }
}
