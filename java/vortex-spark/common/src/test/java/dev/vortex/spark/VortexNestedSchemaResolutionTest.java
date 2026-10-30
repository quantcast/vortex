// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Path;
import java.util.List;
import java.util.function.Function;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Tests that nested fields of a supplied schema resolve against each file by name, the way Parquet resolves them.
 *
 * <p>Every file holds {@code id} and {@code s = {a, b, t: {a, b}}}, with {@code s} null in one row and {@code s.t} in
 * another. Each query reads the same rows from Parquet and Vortex with the same supplied schema, with nested schema
 * pruning on and off.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexNestedSchemaResolutionTest {
    private static final String PRUNING = "spark.sql.optimizer.nestedSchemaPruning.enabled";
    private static final String CASE_SENSITIVE = "spark.sql.caseSensitive";

    private SparkSession spark;
    private String parquetPath;
    private String vortexPath;

    @BeforeAll
    public void setUp(@TempDir Path tempDir) {
        spark = SparkSession.builder()
                .appName("VortexNestedSchemaResolutionTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.sql.adaptive.enabled", "false")
                .config("spark.ui.enabled", "false")
                .getOrCreate();
        Dataset<Row> rows = spark.range(4)
                .selectExpr(
                        "id",
                        "IF(id = 1, NULL, named_struct('a', id + 10, 'b', id + 20, 't', IF(id = 2, NULL,"
                                + " named_struct('a', id + 30, 'b', id + 40)))) AS s");
        parquetPath = tempDir.resolve("parquet").toString();
        vortexPath = tempDir.resolve("vortex").toString();
        rows.write().parquet(parquetPath);
        rows.write().format("vortex").save(vortexPath);
    }

    @AfterEach
    public void resetConf() {
        spark.conf().unset(PRUNING);
        spark.conf().unset(CASE_SENSITIVE);
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @ParameterizedTest(name = "caseSensitive={0}")
    @ValueSource(booleans = {false, true})
    @DisplayName("A struct field named in another case resolves under Spark's case sensitivity")
    public void fieldInAnotherCase(boolean caseSensitive) {
        spark.conf().set(CASE_SENSITIVE, caseSensitive);
        String schema = "id BIGINT, s STRUCT<A: BIGINT, B: BIGINT, t: STRUCT<a: BIGINT, b: BIGINT>>";
        assertMatchesParquet(schema, data -> data.selectExpr("id", "s.B"));
        assertMatchesParquet(schema, data -> data.selectExpr("id", "s.B", "s.t.a"));
        assertMatchesParquet(schema, data -> data.selectExpr("id", "s"));
        assertMatchesParquet(schema, data -> data.where("s.B > 20").selectExpr("id", "s.t.b"));
    }

    @ParameterizedTest(name = "caseSensitive={0}")
    @ValueSource(booleans = {false, true})
    @DisplayName("A field of a struct inside a struct named in another case resolves under Spark's case sensitivity")
    public void nestedFieldInAnotherCase(boolean caseSensitive) {
        spark.conf().set(CASE_SENSITIVE, caseSensitive);
        String schema = "id BIGINT, s STRUCT<a: BIGINT, b: BIGINT, t: STRUCT<A: BIGINT, B: BIGINT>>";
        assertMatchesParquet(schema, data -> data.selectExpr("id", "s.t.B"));
        assertMatchesParquet(schema, data -> data.where("s.t.B > 40").selectExpr("id", "s.a"));
        if (caseSensitive) {
            // `t` keeps its nulls with none of its read fields in the file. Parquet does too from Spark 4.1.
            assertRows(
                    schema,
                    data -> data.selectExpr("id", "s.a", "s.t"),
                    "[[0,10,[null,null]], [1,null,null], [2,12,null], [3,13,[null,null]]]");
        } else {
            assertMatchesParquet(schema, data -> data.selectExpr("id", "s.a", "s.t"));
        }
    }

    @Test
    @DisplayName("A supplied struct with fewer fields than the file reads them by name")
    public void narrowerStruct() {
        String schema = "id BIGINT, s STRUCT<b: BIGINT, t: STRUCT<b: BIGINT>>";
        assertMatchesParquet(schema, data -> data.selectExpr("id", "s.b"));
        assertMatchesParquet(schema, data -> data.selectExpr("id", "s.t.b"));
        assertMatchesParquet(schema, data -> data.selectExpr("id", "s"));
    }

    @Test
    @DisplayName("A supplied struct with its fields in another order reads them by name")
    public void reorderedStruct() {
        String schema = "id BIGINT, s STRUCT<t: STRUCT<b: BIGINT, a: BIGINT>, b: BIGINT, a: BIGINT>";
        assertMatchesParquet(schema, data -> data.selectExpr("id", "s.b"));
        assertMatchesParquet(schema, data -> data.selectExpr("id", "s.t.b", "s.a"));
        assertMatchesParquet(schema, data -> data.selectExpr("id", "s"));
    }

    @Test
    @DisplayName("A nullable struct field the file does not carry reads as null")
    public void absentField() {
        String schema = "id BIGINT, s STRUCT<a: BIGINT, b: BIGINT, t: STRUCT<a: BIGINT, b: BIGINT, c: BIGINT>,"
                + " c: BIGINT, u: STRUCT<x: BIGINT>>";
        assertMatchesParquet(schema, data -> data.selectExpr("id", "s.c"));
        assertMatchesParquet(schema, data -> data.selectExpr("id", "s.t.c", "s.t.a"));
        assertMatchesParquet(schema, data -> data.selectExpr("id", "s.u.x", "s.u IS NULL", "s.b"));
        assertMatchesParquet(schema, data -> data.selectExpr("id", "s"));
        assertMatchesParquet(
                schema, data -> data.where("s.c IS NULL AND s.t.c IS NULL").selectExpr("id", "s.a"));
    }

    /** Asserts that {@code query} returns {@code expected} from Vortex, with pruning off and on. */
    private void assertRows(String schema, Function<Dataset<Row>, Dataset<Row>> query, String expected) {
        for (boolean pruning : new boolean[] {false, true}) {
            spark.conf().set(PRUNING, pruning);
            List<Row> actual = query.apply(
                            spark.read().schema(schema).format("vortex").load(vortexPath))
                    .orderBy("id")
                    .collectAsList();
            assertEquals(expected, actual.toString(), schema + " with pruning " + (pruning ? "on" : "off"));
        }
    }

    /** Asserts that {@code query} returns the same rows from Vortex as from Parquet, with pruning off and on. */
    private void assertMatchesParquet(String schema, Function<Dataset<Row>, Dataset<Row>> query) {
        for (boolean pruning : new boolean[] {false, true}) {
            spark.conf().set(PRUNING, pruning);
            List<Row> expected = query.apply(spark.read().schema(schema).parquet(parquetPath))
                    .orderBy("id")
                    .collectAsList();
            List<Row> actual = query.apply(
                            spark.read().schema(schema).format("vortex").load(vortexPath))
                    .orderBy("id")
                    .collectAsList();
            assertEquals(expected, actual, schema + " with pruning " + (pruning ? "on" : "off"));
        }
    }
}
