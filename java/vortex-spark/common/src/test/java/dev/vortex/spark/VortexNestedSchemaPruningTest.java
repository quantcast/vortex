// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/**
 * Tests that reading one field of a nested column returns that field.
 *
 * <p>With {@code spark.sql.optimizer.nestedSchemaPruning.enabled}, Spark asks the scan for only the nested fields a
 * query uses. Each row of the table holds {@code a = id + 100}, {@code b = id + 200} and {@code c = id + 300}, so a
 * field read from the wrong position shows up as the wrong hundred.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexNestedSchemaPruningTest {

    private static final List<Long> C_VALUES = List.of(300L, 301L, 302L, 303L, 304L);

    private SparkSession spark;
    private Dataset<Row> table;

    @BeforeAll
    public void setUp(@TempDir Path tempDir) {
        spark = SparkSession.builder()
                .appName("VortexNestedSchemaPruningTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.sql.adaptive.enabled", "false")
                .config("spark.sql.optimizer.nestedSchemaPruning.enabled", "true")
                .config("spark.sql.maxMetadataStringLength", "1000")
                .config("spark.ui.enabled", "false")
                .getOrCreate();

        Column abc = functions.struct(
                functions.col("id").plus(100).alias("a"),
                functions.col("id").plus(200).alias("b"),
                functions.col("id").plus(300).alias("c"));
        Dataset<Row> df = spark.range(5)
                .select(
                        functions.col("id"),
                        abc.alias("s"),
                        functions.array(abc).alias("arr"),
                        functions.map(functions.lit("k"), abc).alias("m"),
                        functions
                                .struct(functions.lit("x").alias("p"), abc.alias("inner"))
                                .alias("outer"),
                        nullWhen(
                                        functions.col("id").equalTo(1),
                                        functions.struct(
                                                functions.lit("x").alias("p"),
                                                nullWhen(functions.col("id").equalTo(2), abc)
                                                        .alias("inner")))
                                .alias("nullable_outer"),
                        nullWhen(
                                        functions.col("id").mod(2).equalTo(1),
                                        functions.struct(
                                                functions.col("id").plus(100).alias("a"),
                                                nullWhen(
                                                                functions
                                                                        .col("id")
                                                                        .equalTo(4),
                                                                functions
                                                                        .col("id")
                                                                        .plus(200))
                                                        .alias("b")))
                                .alias("n"));

        String path = tempDir.resolve("nested").toUri().toString();
        df.write()
                .format("vortex")
                .option("path", path)
                .mode(SaveMode.Overwrite)
                .save();
        table = spark.read().format("vortex").option("path", path).load();
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @Test
    @DisplayName("Selecting a struct field other than the first returns that field")
    public void structField() {
        assertEquals(C_VALUES, valuesOf(table, functions.col("s.c")));
    }

    @Test
    @DisplayName("Selecting a field of the structs in an array returns that field")
    public void arrayElementField() {
        assertEquals(C_VALUES, valuesOf(table, functions.element_at(functions.col("arr.c"), 1)));
    }

    @Test
    @DisplayName("Selecting a field of the structs in a map returns that field")
    public void mapValueField() {
        assertEquals(C_VALUES, valuesOf(table, functions.col("m").getItem("k").getField("c")));
    }

    @Test
    @DisplayName("Selecting a field of a struct inside a struct returns that field")
    public void structInStructField() {
        assertEquals(C_VALUES, valuesOf(table, functions.col("outer.inner.c")));
    }

    @Test
    @DisplayName("Null structs and null fields stay null when the struct is pruned")
    public void nullableStructField() {
        assertEquals(Arrays.asList(200L, null, 202L, null, null), valuesOf(table, functions.col("n.b")));
        assertEquals(
                List.of(false, true, false, true, false),
                valuesOf(table, functions.col("n").isNull()));
    }

    @Test
    @DisplayName("Nulls at each level of a struct inside a struct survive pruning both levels")
    public void nullableStructInStructField() {
        assertEquals(
                Arrays.asList(300L, null, null, 303L, 304L), valuesOf(table, functions.col("nullable_outer.inner.c")));
        assertEquals(
                List.of(false, true, false, false, false),
                valuesOf(table, functions.col("nullable_outer").isNull()));
        assertEquals(
                List.of(false, true, true, false, false),
                valuesOf(table, functions.col("nullable_outer.inner").isNull()));
    }

    @Test
    @DisplayName("A pushed filter on a field the query does not select still applies")
    public void filterOnPrunedField() {
        assertEquals(
                List.of(302L, 303L, 304L),
                valuesOf(table.where(functions.col("s.a").gt(101)), functions.col("s.c")));
    }

    @Test
    @DisplayName("Whole struct columns still read in full alongside pruned ones")
    public void wholeAndPrunedStructs() {
        Row row = table.where("id = 3")
                .select(functions.col("s"), functions.col("outer.inner.a"))
                .first();
        assertEquals(
                List.of(103L, 203L, 303L),
                List.of(
                        row.getStruct(0).getLong(0),
                        row.getStruct(0).getLong(1),
                        row.getStruct(0).getLong(2)));
        assertEquals(103L, row.getLong(1));
    }

    @Test
    @DisplayName("Struct columns are read pruned to the required fields, with columns and fields in table order")
    public void readSchemaPrunesStructFields() {
        assertReadSchema(
                "struct<id:bigint,s:struct<a:bigint,c:bigint>>",
                table.select(functions.col("s.c"), functions.col("id"), functions.col("s.a")));
    }

    @Test
    @DisplayName("Structs inside structs are read pruned at every level")
    public void readSchemaPrunesNestedStructs() {
        assertReadSchema("struct<outer:struct<inner:struct<b:bigint>>>", table.select(functions.col("outer.inner.b")));
    }

    @Test
    @DisplayName("Arrays of structs are read pruned, and maps of structs with their full table type")
    public void readSchemaPrunesArraysAndKeepsMapTypes() {
        assertReadSchema(
                "struct<arr:array<struct<c:bigint>>,m:map<string,struct<a:bigint,b:bigint,c:bigint>>>",
                table.select(
                        functions.element_at(functions.col("arr.c"), 1),
                        functions.col("m").getItem("k").getField("c")));
    }

    private static void assertReadSchema(String expected, Dataset<Row> query) {
        String plan = query.queryExecution().executedPlan().toString();
        int start = plan.indexOf("ReadSchema: ");
        assertTrue(start >= 0, plan);
        assertEquals(expected, plan.substring(start + "ReadSchema: ".length()).split("\\s", 2)[0], plan);
    }

    private static Column nullWhen(Column condition, Column value) {
        return functions.when(condition, functions.lit(null)).otherwise(value);
    }

    /** Returns {@code column} for each row of {@code rows}, ordered by id. */
    @SuppressWarnings("unchecked")
    private static <T> List<T> valuesOf(Dataset<Row> rows, Column column) {
        return rows.orderBy("id").select(column).collectAsList().stream()
                .map(row -> (T) row.get(0))
                .collect(Collectors.toList());
    }
}
