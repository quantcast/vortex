// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

/**
 * Nested schema pruning on catalog and bucketed tables, which Spark reads through the V1 {@link VortexFileFormat}.
 *
 * <p>Spark prunes nested fields on that path only for Parquet and ORC, so the connector's optimizer rule does it for
 * Vortex. The nullable struct {@code s = {a, b, w0..w7}} and the elements of {@code arr = array<struct<a, v0..v7>>}
 * hold eight wide string fields each, so a read that pulls in the whole struct shows up in the bytes read. {@code m} is
 * a map of structs.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexCatalogSchemaPruningTest {
    private static final int ROWS = 8192;
    private static final String PRUNING = "spark.sql.optimizer.nestedSchemaPruning.enabled";
    private static final String DATA = "SELECT id, CAST(id % 3 AS INT) AS p,"
            + " CASE WHEN id % 5 = 0 THEN NULL ELSE named_struct('a', id * 2 + 100, 'b', id * 3" + wide("w", "id")
            + ") END AS s,"
            + " CASE WHEN id % 7 = 0 THEN NULL ELSE transform(filter(sequence(0, 3), i -> i < id % 5),"
            + " i -> named_struct('a', id * 10 + i" + wide("v", "id, i") + ")) END AS arr,"
            + " named_struct('a', id % 16, 'b', sha2(CAST(id % 16 AS STRING), 256)) AS k,"
            + " map('k', named_struct('a', id, 'b', id * 2)) AS m"
            + " FROM range(" + ROWS + ")";

    private SparkSession spark;
    private Path warehouseDir;

    @BeforeAll
    public void setUp() throws IOException {
        warehouseDir = Files.createTempDirectory("vortex-catalog-pruning");
        spark = SparkSession.builder()
                .appName("VortexCatalogSchemaPruningTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config(
                        "spark.sql.warehouse.dir",
                        warehouseDir.resolve("warehouse").toUri().toString())
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.sql.adaptive.enabled", "false")
                .config("spark.sql.autoBroadcastJoinThreshold", "-1")
                .config(PRUNING, "true")
                // `arr[1]` fails on a short array in ANSI mode.
                .config("spark.sql.ansi.enabled", "false")
                .config("spark.ui.enabled", "false")
                .config(
                        "spark.hadoop.fs.file.impl",
                        VortexNestedSchemaPruningIoTest.CountingLocalFileSystem.class.getName())
                .config("spark.hadoop.fs.file.impl.disable.cache", "true")
                .getOrCreate();

        Dataset<Row> data = spark.sql(DATA);
        data.write().format("parquet").saveAsTable("expected");
        data.write().format("vortex").saveAsTable("saved");
        String located = warehouseDir.resolve("located").toUri().toString();
        data.write().format("vortex").option("vortex.listLayout", "true").save(located);
        spark.sql("CREATE TABLE located USING vortex LOCATION '" + located + "'");
        data.write().format("vortex").partitionBy("p").saveAsTable("partitioned");
        data.write().format("vortex").bucketBy(4, "id").sortBy("id").saveAsTable("bucketed");
        data.where("id % 2 = 0")
                .write()
                .format("vortex")
                .bucketBy(4, "id")
                .sortBy("id")
                .saveAsTable("bucketed_right");
        data.write().format("vortex").bucketBy(4, "k").sortBy("k").saveAsTable("bucketed_by_struct");
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @Test
    @DisplayName("A struct field of a saveAsTable table is read without the struct's other fields")
    public void savedTableReadsOnlyTheField() {
        assertPrunedRead("saved", "SELECT id, s.a FROM %s", "struct<id:bigint,s:struct<a:bigint>>");
        assertMatchesParquet("SELECT id, s.a, s.b FROM %s");
    }

    @Test
    @DisplayName("A struct field of a CREATE TABLE ... LOCATION table is read without the struct's other fields")
    public void locatedTableReadsOnlyTheField() {
        assertPrunedRead("located", "SELECT id, s.a FROM %s", "struct<id:bigint,s:struct<a:bigint>>");
        assertMatchesParquet("SELECT id, s.a FROM %s");
    }

    @Test
    @DisplayName("A field of the structs in an array is read without their other fields in the list layout")
    public void arrayElementFieldReadsOnlyThatField() {
        long whole = bytesRead("SELECT id, arr FROM located");
        long field = bytesRead("SELECT id, arr.a FROM located");
        System.out.printf(Locale.ROOT, "bytes read: located arr.a: %d; whole arr: %d%n", field, whole);
        assertTrue(field < whole / 10, "arr.a read " + field + " bytes; arr read " + whole);
        assertReadSchema("struct<id:bigint,arr:array<struct<a:bigint>>>", "SELECT id, arr.a FROM located");
        assertMatchesParquet("SELECT id, arr.a FROM %s");
        assertMatchesParquet("SELECT id, arr[1].a, arr.v3 FROM %s");
        assertMatchesParquet("SELECT id, arr.a FROM %s WHERE arr IS NULL OR size(arr) = 0");
    }

    @Test
    @DisplayName("Filters on nested fields read only the fields they and the projection use")
    public void filtersOnNestedFields() {
        String filterOnly = "SELECT id, s.a FROM %s WHERE s.b % 4 = 1";
        assertPrunedRead("saved", filterOnly, "struct<id:bigint,s:struct<a:bigint,b:bigint>>");
        assertTrue(assertMatchesParquet(filterOnly).size() > ROWS / 8);
        assertMatchesParquet("SELECT id, s.a FROM %s WHERE s.a > " + ROWS + " AND s.w3 IS NOT NULL");
    }

    @Test
    @DisplayName("A null check on a pruned struct sees the struct's own nulls")
    public void nullCheckOnPrunedStruct() {
        String nulls = "SELECT id, s.a FROM %s WHERE s IS NULL";
        assertPrunedRead("saved", nulls, "struct<id:bigint,s:struct<a:bigint>>");
        assertEquals((ROWS + 4) / 5, assertMatchesParquet(nulls).size());
        assertEquals(
                ROWS - (ROWS + 4) / 5,
                assertMatchesParquet("SELECT id, s.b FROM %s WHERE s IS NOT NULL")
                        .size());
        assertMatchesParquet("SELECT id, s IS NULL, s.b FROM %s");
    }

    @Test
    @DisplayName("Field names resolve case-insensitively, and the read schema keeps the table's names")
    public void caseInsensitiveNames() {
        assertPrunedRead("saved", "SELECT ID, S.A FROM %s", "struct<id:bigint,s:struct<a:bigint>>");
        assertMatchesParquet("SELECT ID, S.A, Arr.A FROM %s");
    }

    @Test
    @DisplayName("A partitioned table is pruned below its data columns and keeps its partition values")
    public void partitionedTable() {
        assertPrunedRead(
                "partitioned", "SELECT id, p, s.a FROM %s WHERE p = 1", "struct<id:bigint,s:struct<a:bigint>>");
        assertMatchesParquet("SELECT id, p, s.a FROM %s WHERE p = 1");
        assertMatchesParquet("SELECT id, p, s.b FROM %s WHERE s.a % 3 = p");
    }

    @Test
    @DisplayName("_metadata and input_file_name() read alongside pruned fields")
    public void metadataColumns() {
        String query = "SELECT id, s.a, _metadata.file_name, input_file_name() FROM saved";
        String plan = plan(query);
        assertTrue(plan.contains("s:struct<a:bigint>>"), plan);
        assertFalse(plan.contains("w0"), plan);
        List<Row> rows = assertMatchesUnpruned(query);
        assertEquals(ROWS, rows.size());
        for (Row row : rows) {
            assertTrue(row.getString(2).endsWith(".vortex"), row.toString());
            assertTrue(row.getString(3).endsWith(row.getString(2)), row.toString());
        }
    }

    @Test
    @DisplayName("The row reader returns pruned fields as the batch reader does")
    public void rowAndBatchReaders() {
        String query = "SELECT id, s.a, arr.a FROM %s WHERE s.b % 2 = 0";
        assertTrue(plan(on(query, "saved")).contains("Batched: true"));
        assertMatchesParquet(query);
        spark.conf().set("spark.sql.codegen.wholeStage", "false");
        try {
            assertTrue(plan(on(query, "saved")).contains("Batched: false"));
            assertPrunedRead("saved", "SELECT id, s.a FROM %s", "struct<id:bigint,s:struct<a:bigint>>");
            assertMatchesParquet(query);
        } finally {
            spark.conf().unset("spark.sql.codegen.wholeStage");
        }
    }

    @Test
    @DisplayName("A bucketed table is pruned, and a join on its bucket column still needs no shuffle")
    public void bucketedTable() {
        assertPrunedRead("bucketed", "SELECT id, s.a FROM %s", "struct<id:bigint,s:struct<a:bigint>>");
        assertMatchesParquet("SELECT id, s.a FROM %s");
        String join = "SELECT l.id, l.s.a, r.s.b FROM bucketed l JOIN bucketed_right r ON l.id = r.id";
        String plan = plan(join);
        assertFalse(plan.contains("Exchange"), "a bucketed join should not shuffle:\n" + plan);
        assertFalse(plan.contains("w0"), plan);
        assertTrue(plan.contains("ReadSchema: struct<id:bigint,s:struct<a:bigint>>"), plan);
        assertTrue(plan.contains("ReadSchema: struct<id:bigint,s:struct<b:bigint>>"), plan);
        assertEquals(ROWS / 2, assertMatchesUnpruned(join).size());
    }

    @Test
    @DisplayName("A struct bucket column keeps all its fields, and other structs are still pruned")
    public void structBucketColumnStaysWhole() {
        assertReadSchema(
                "struct<s:struct<a:bigint>,k:struct<a:bigint,b:string>>", "SELECT k.a, s.a FROM bucketed_by_struct");
        assertMatchesParquet("SELECT id, k.a, s.a FROM %s");
    }

    @Test
    @DisplayName("Structs inside maps keep their table type, which the reader returns")
    public void mapValuesStayWhole() {
        assertReadSchema(
                "struct<id:bigint,s:struct<a:bigint>,m:map<string,struct<a:bigint,b:bigint>>>",
                "SELECT id, s.a, m['k'].b FROM saved");
        assertMatchesParquet("SELECT id, s.a, m['k'].b FROM %s");
    }

    @Test
    @DisplayName("Turning nested schema pruning off reads the whole struct")
    public void pruningDisabled() {
        long whole = bytesRead("SELECT id, s FROM saved");
        spark.conf().set(PRUNING, "false");
        try {
            String plan = plan("SELECT id, s.a FROM saved");
            assertTrue(plan.contains("w0"), plan);
            long field = bytesRead("SELECT id, s.a FROM saved");
            assertTrue(field > whole / 2, "s.a read " + field + " bytes with pruning off; s read " + whole);
        } finally {
            spark.conf().set(PRUNING, "true");
        }
    }

    /**
     * Asserts {@code query} over {@code table} reads {@code readSchema}, and under a tenth of the bytes that reading
     * the whole of {@code s} does.
     */
    private void assertPrunedRead(String table, String query, String readSchema) {
        String pruned = on(query, table);
        long whole = bytesRead(on("SELECT id, s FROM %s", table));
        long field = bytesRead(pruned);
        System.out.printf(Locale.ROOT, "bytes read: %s: %d; whole s: %d%n", pruned, field, whole);
        assertReadSchema(readSchema, pruned);
        assertTrue(field < whole / 10, pruned + " read " + field + " bytes; s read " + whole);
    }

    /**
     * Asserts {@code query} returns the same rows from every Vortex table as from the Parquet table, with and without
     * nested schema pruning, and returns them.
     */
    private List<Row> assertMatchesParquet(String query) {
        List<Row> expected = sorted(on(query, "expected"));
        for (String table : List.of("saved", "located", "partitioned", "bucketed", "bucketed_by_struct")) {
            assertEquals(expected, assertMatchesUnpruned(on(query, table)), table);
        }
        return expected;
    }

    /** Asserts {@code query} returns the same rows with nested schema pruning off, and returns them. */
    private List<Row> assertMatchesUnpruned(String query) {
        List<Row> pruned = sorted(query);
        spark.conf().set(PRUNING, "false");
        try {
            assertEquals(sorted(query), pruned, query);
        } finally {
            spark.conf().set(PRUNING, "true");
        }
        return pruned;
    }

    /** {@code query} reading {@code table} in place of {@code %s}. */
    private static String on(String query, String table) {
        return query.replace("%s", table);
    }

    private List<Row> sorted(String query) {
        Dataset<Row> rows = spark.sql(query);
        return new ArrayList<>(rows.orderBy(rows.columns()[0]).collectAsList());
    }

    private void assertReadSchema(String expected, String query) {
        String plan = plan(query);
        int start = plan.indexOf("ReadSchema: ");
        assertTrue(start >= 0, plan);
        assertEquals(expected, plan.substring(start + "ReadSchema: ".length()).split("\\s", 2)[0], plan);
    }

    private String plan(String query) {
        return spark.sql(query).queryExecution().executedPlan().toString();
    }

    /** Hashes every selected value so the whole result is decoded, and returns the bytes read from storage. */
    private long bytesRead(String query) {
        VortexNestedSchemaPruningIoTest.BYTES_READ.set(0);
        spark.sql(query).selectExpr("bit_xor(xxhash64(*))").collectAsList();
        return VortexNestedSchemaPruningIoTest.BYTES_READ.get();
    }

    /** Eight wide string fields named {@code prefix0..prefix7}, as {@code named_struct} arguments. */
    private static String wide(String prefix, String values) {
        StringBuilder fields = new StringBuilder();
        for (int i = 0; i < 8; i++) {
            fields.append(String.format(
                    Locale.ROOT, ", '%s%d', sha2(concat_ws(':', '%s%d', %s), 256)", prefix, i, prefix, i, values));
        }
        return fields.toString();
    }
}
