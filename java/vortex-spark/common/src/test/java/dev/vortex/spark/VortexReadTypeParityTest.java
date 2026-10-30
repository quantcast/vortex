// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import dev.vortex.api.Session;
import dev.vortex.api.VortexWriter;
import dev.vortex.arrow.ArrowAllocation;
import dev.vortex.relocated.org.apache.arrow.c.ArrowArray;
import dev.vortex.relocated.org.apache.arrow.c.ArrowSchema;
import dev.vortex.relocated.org.apache.arrow.c.Data;
import dev.vortex.relocated.org.apache.arrow.memory.BufferAllocator;
import dev.vortex.relocated.org.apache.arrow.vector.BigIntVector;
import dev.vortex.relocated.org.apache.arrow.vector.DecimalVector;
import dev.vortex.relocated.org.apache.arrow.vector.TimeStampMicroTZVector;
import dev.vortex.relocated.org.apache.arrow.vector.TimeStampMilliTZVector;
import dev.vortex.relocated.org.apache.arrow.vector.TimeStampNanoTZVector;
import dev.vortex.relocated.org.apache.arrow.vector.TimeStampNanoVector;
import dev.vortex.relocated.org.apache.arrow.vector.TimeStampSecVector;
import dev.vortex.relocated.org.apache.arrow.vector.VarCharVector;
import dev.vortex.relocated.org.apache.arrow.vector.VectorSchemaRoot;
import dev.vortex.relocated.org.apache.arrow.vector.ViewVarCharVector;
import dev.vortex.relocated.org.apache.arrow.vector.complex.StructVector;
import dev.vortex.relocated.org.apache.arrow.vector.types.TimeUnit;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.ArrowType;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Field;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.FieldType;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Schema;
import java.io.IOException;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Stream;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Reading a column with a supplied type other than the one the file stores acts like Parquet.
 *
 * <p>Each case writes a column of one type to both formats and reads it back with another. Where Parquet fails, Vortex
 * must fail too, naming the column and the type asked for; where both return rows, the rows must match. Parquet's rules
 * change between Spark versions, so a case may let Vortex refuse a conversion Parquet makes. Every outcome is printed
 * on a {@code TYPE-MATRIX} line.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexReadTypeParityTest {
    private static final String USE_V1 = "spark.sql.sources.useV1SourceList";

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    void setUp() {
        spark = SparkSession.builder()
                .appName("VortexReadTypeParityTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.ui.enabled", "false")
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.sql.adaptive.enabled", "false")
                .config("spark.sql.session.timeZone", "UTC")
                // Microseconds, as Vortex stores timestamps, rather than the INT96 default.
                .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
                .getOrCreate();
    }

    @AfterAll
    void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    /**
     * One write and read. {@code write} selects {@code id} and {@code x}, and {@code readType} is the type {@code x} is
     * read with. A Vortex error must name {@code path}, the field whose type differs, and {@code pathType}, the type it
     * is read with.
     */
    record Case(
            String name,
            String write,
            String readType,
            String path,
            String pathType,
            String query,
            boolean vortexMayRefuse) {
        @Override
        public String toString() {
            return name;
        }
    }

    private static Case leaf(String writeType, String first, String second, String readType) {
        return leaf(writeType, first, second, readType, false);
    }

    private static Case leaf(String writeType, String first, String second, String readType, boolean mayRefuse) {
        return new Case(
                writeType + " as " + readType,
                values(writeType, first, second),
                readType,
                "x",
                readType,
                "SELECT id, x FROM t",
                mayRefuse);
    }

    private static String values(String writeType, String first, String second) {
        return String.format(
                Locale.ROOT,
                "SELECT CAST(id AS BIGINT) AS id, CAST(x AS %s) AS x FROM VALUES (0, %s), (1, %s), (2, NULL) AS v(id, x)",
                writeType,
                first,
                second);
    }

    static Stream<Case> cases() {
        Stream<Case> collated =
                org.apache.spark.package$.MODULE$.SPARK_VERSION().startsWith("3.")
                        ? Stream.empty()
                        : Stream.of(leaf("STRING", "'abc'", "'ABC'", "STRING COLLATE UTF8_LCASE"));
        return Stream.concat(common(), collated);
    }

    private static Stream<Case> common() {
        String longs = values("BIGINT", "1000000", "-5");
        return Stream.of(
                leaf("BIGINT", "1000000", "-5", "TIMESTAMP"),
                leaf("BIGINT", "1000000", "-5", "TIMESTAMP_NTZ"),
                leaf("BIGINT", "1", "5000000000", "INT"),
                leaf("INT", "1", "-2147483648", "BIGINT", true),
                leaf("INT", "1", "-2147483648", "DOUBLE", true),
                leaf("TINYINT", "1", "-128", "INT", true),
                leaf("INT", "7", "-1", "STRING"),
                leaf("STRING", "'12'", "'abc'", "INT"),
                leaf("FLOAT", "1.5", "0.1", "DOUBLE", true),
                leaf("DOUBLE", "0.1", "1e300", "FLOAT"),
                leaf("DECIMAL(10,2)", "12345678.91", "-0.01", "DECIMAL(12,2)"),
                leaf("DECIMAL(12,2)", "1.23", "9999999999.99", "DECIMAL(10,2)"),
                leaf("DECIMAL(10,2)", "12345678.91", "-0.01", "DECIMAL(12,4)"),
                leaf("DECIMAL(10,2)", "12345678.91", "-0.01", "DECIMAL(12,1)"),
                leaf("DATE", "DATE '2026-10-29'", "DATE '1969-12-31'", "TIMESTAMP"),
                leaf("DATE", "DATE '2026-10-29'", "DATE '1969-12-31'", "TIMESTAMP_NTZ", true),
                leaf("DATE", "DATE '2026-10-29'", "DATE '1969-12-31'", "INT"),
                leaf("INT", "20000", "-1", "DATE"),
                leaf("TIMESTAMP", "TIMESTAMP '2026-10-29 12:34:56.789'", "TIMESTAMP '1969-12-31 23:59:59'", "BIGINT"),
                leaf(
                        "TIMESTAMP_NTZ",
                        "TIMESTAMP_NTZ '2026-10-29 12:34:56.789'",
                        "TIMESTAMP_NTZ '1969-12-31 23:59:59'",
                        "BIGINT"),
                leaf(
                        "TIMESTAMP",
                        "TIMESTAMP '2026-10-29 12:34:56.789'",
                        "TIMESTAMP '1969-12-31 23:59:59'",
                        "TIMESTAMP_NTZ"),
                leaf(
                        "TIMESTAMP_NTZ",
                        "TIMESTAMP_NTZ '2026-10-29 12:34:56.789'",
                        "TIMESTAMP_NTZ '1969-12-31 23:59:59'",
                        "TIMESTAMP"),
                leaf("BOOLEAN", "true", "false", "INT"),
                leaf("BINARY", "X'616263'", "X''", "STRING", true),
                leaf("STRING", "'abc'", "''", "BINARY", true),
                new Case(
                        "struct field BIGINT as TIMESTAMP",
                        "SELECT id, named_struct('a', x, 'b', CAST(id AS STRING)) AS x FROM (" + longs + ")",
                        "STRUCT<a: TIMESTAMP, b: STRING>",
                        "x.a",
                        "TIMESTAMP",
                        "SELECT id, x FROM t",
                        false),
                new Case(
                        "selected struct field BIGINT as TIMESTAMP",
                        "SELECT id, named_struct('a', x, 'b', CAST(id AS STRING)) AS x FROM (" + longs + ")",
                        "STRUCT<a: TIMESTAMP, b: STRING>",
                        "x.a",
                        "TIMESTAMP",
                        "SELECT id, x.a FROM t",
                        false),
                new Case(
                        "BIGINT as STRUCT",
                        longs,
                        "STRUCT<a: BIGINT>",
                        "x",
                        "STRUCT<a: BIGINT>",
                        "SELECT id, x FROM t",
                        false),
                new Case(
                        "STRUCT as BIGINT",
                        "SELECT id, named_struct('a', x) AS x FROM (" + longs + ")",
                        "BIGINT",
                        "x",
                        "BIGINT",
                        "SELECT id, x FROM t",
                        false),
                new Case(
                        "ARRAY<BIGINT> as ARRAY<TIMESTAMP>",
                        "SELECT id, IF(x IS NULL, NULL, array(x, NULL)) AS x FROM (" + longs + ")",
                        "ARRAY<TIMESTAMP>",
                        "x.element",
                        "TIMESTAMP",
                        "SELECT id, x FROM t",
                        false),
                new Case(
                        "MAP<STRING, BIGINT> as MAP<STRING, TIMESTAMP>",
                        "SELECT id, IF(x IS NULL, NULL, map('k', x)) AS x FROM (" + longs + ")",
                        "MAP<STRING, TIMESTAMP>",
                        "x.value",
                        "TIMESTAMP",
                        "SELECT id, x FROM t",
                        false),
                new Case(
                        "array struct field BIGINT as TIMESTAMP",
                        "SELECT id, IF(x IS NULL, NULL, array(named_struct('a', x, 'b', CAST(id AS STRING)))) AS x"
                                + " FROM (" + longs + ")",
                        "ARRAY<STRUCT<a: TIMESTAMP, b: STRING>>",
                        "x.element.a",
                        "TIMESTAMP",
                        "SELECT id, x FROM t",
                        false),
                new Case(
                        "array struct fields in another order",
                        "SELECT id, IF(x IS NULL, NULL, array(named_struct('a', x, 'b', CAST(id AS STRING)))) AS x"
                                + " FROM (" + longs + ")",
                        "ARRAY<STRUCT<b: STRING, a: BIGINT>>",
                        "x.element",
                        "STRUCT<b: STRING, a: BIGINT>",
                        "SELECT id, x FROM t",
                        false),
                new Case(
                        "map value struct field BIGINT as TIMESTAMP",
                        "SELECT id, IF(x IS NULL, NULL, map('k', named_struct('a', x, 'b', CAST(id AS STRING)))) AS x"
                                + " FROM (" + longs + ")",
                        "MAP<STRING, STRUCT<a: TIMESTAMP, b: STRING>>",
                        "x.value.a",
                        "TIMESTAMP",
                        "SELECT id, x FROM t",
                        false),
                new Case(
                        "map value array struct field BIGINT as TIMESTAMP",
                        "SELECT id, IF(x IS NULL, NULL, map('k', array(named_struct('a', x)))) AS x" + " FROM (" + longs
                                + ")",
                        "MAP<STRING, ARRAY<STRUCT<a: TIMESTAMP>>>",
                        "x.value.element.a",
                        "TIMESTAMP",
                        "SELECT id, x FROM t",
                        false),
                // The vectors return a map's structs by position, so Vortex may refuse what Parquet matches by name.
                new Case(
                        "map value struct fields in another order",
                        "SELECT id, IF(x IS NULL, NULL, map('k', named_struct('a', x, 'b', CAST(id AS STRING)))) AS x"
                                + " FROM (" + longs + ")",
                        "MAP<STRING, STRUCT<b: STRING, a: BIGINT>>",
                        "x.value",
                        "STRUCT<b: STRING, a: BIGINT>",
                        "SELECT id, x FROM t",
                        true),
                new Case(
                        "map value struct with a field the file lacks",
                        "SELECT id, IF(x IS NULL, NULL, map('k', named_struct('a', x))) AS x FROM (" + longs + ")",
                        "MAP<STRING, STRUCT<a: BIGINT, c: INT>>",
                        "x.value",
                        "STRUCT<a: BIGINT, c: INT>",
                        "SELECT id, x FROM t",
                        true),
                new Case(
                        "filtered BIGINT as TIMESTAMP",
                        longs,
                        "TIMESTAMP",
                        "x",
                        "TIMESTAMP",
                        "SELECT id FROM t WHERE x > TIMESTAMP '1970-01-01 00:00:00.5'",
                        false),
                new Case(
                        "filtered INT as BIGINT",
                        values("INT", "1", "-2147483648"),
                        "BIGINT",
                        "x",
                        "BIGINT",
                        "SELECT id FROM t WHERE x > 0",
                        true));
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("cases")
    void aColumnReadAsAnotherTypeActsLikeParquet(Case c) {
        String slug = c.name().toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]+", "-");
        Outcome parquet = run(c, "parquet", slug, false);
        Outcome vortex = run(c, "vortex", slug, false);
        Outcome vortexV1 = run(c, "vortex", slug, true);
        for (Outcome outcome : List.of(vortex, vortexV1)) {
            if (parquet.failure() != null) {
                assertTrue(
                        outcome.failure() != null, "Parquet fails with " + parquet + " but Vortex returns " + outcome);
            } else if (outcome.failure() == null) {
                assertEquals(parquet.rows(), outcome.rows(), "rows");
            } else if (!c.vortexMayRefuse()) {
                fail("Parquet returns " + parquet + " but Vortex fails with " + outcome);
            }
            if (outcome.failure() != null) {
                String message = outcome.failure();
                String readType = spark.sql("SELECT CAST(NULL AS " + c.pathType() + ")")
                        .schema()
                        .fields()[0]
                        .dataType()
                        .catalogString();
                assertTrue(
                        message.contains("column " + c.path() + " ") && message.contains("read as " + readType),
                        "the error names column " + c.path() + " and type " + readType + ": " + message);
                assertTrue(message.contains(".vortex"), "the error names the file: " + message);
            }
        }
    }

    private record Outcome(List<String> rows, String failure) {
        @Override
        public String toString() {
            return failure != null ? "FAILS " + failure : rows.toString();
        }
    }

    private Outcome run(Case c, String format, String slug, boolean v1) {
        String root = tempDir.resolve(slug + "-" + format + (v1 ? "-v1" : "")).toString();
        spark.sql(c.write()).coalesce(1).write().format(format).save(root);
        String previous = spark.conf().get(USE_V1);
        if (v1) {
            spark.conf().set(USE_V1, previous + ",vortex");
        }
        Outcome outcome;
        try {
            spark.read()
                    .format(format)
                    .schema("id BIGINT, x " + c.readType())
                    .load(root)
                    .createOrReplaceTempView("t");
            outcome = new Outcome(spark.sql(c.query()).orderBy("id").toJSON().collectAsList(), null);
        } catch (Exception e) {
            // Spark rethrows a failed job's SparkException unchecked.
            outcome = new Outcome(null, describe(e));
        } finally {
            spark.conf().set(USE_V1, previous);
        }
        System.out.println(String.join(
                        "\t", "TYPE-MATRIX", spark.version(), c.name(), format + (v1 ? "-v1" : ""), outcome.toString())
                .replaceAll("\\s*\\R\\s*", " "));
        return outcome;
    }

    /** Every message in the cause chain, root cause last, so a check sees past Spark's wrappers. */
    private static String describe(Throwable failure) {
        StringBuilder messages = new StringBuilder();
        Throwable root = failure;
        for (Throwable t = failure; t != null; t = t.getCause()) {
            messages.append(t.getMessage()).append(" | ");
            root = t;
        }
        String rootMessage =
                String.valueOf(root.getMessage()).lines().findFirst().orElse("");
        return root.getClass().getSimpleName() + ": " + rootMessage + " || " + messages;
    }

    @Test
    void filesThatDifferOnlyInWaysTheReaderNormalizesReadTogether() throws IOException {
        Path dir = tempDir.resolve("normalized");
        writeFile(
                dir.resolve("a.vortex"),
                new Schema(List.of(
                        Field.notNullable("id", new ArrowType.Int(64, true)),
                        Field.notNullable("ts", new ArrowType.Timestamp(TimeUnit.MILLISECOND, "UTC")),
                        Field.notNullable("s", ArrowType.Utf8.INSTANCE),
                        Field.notNullable("d", new ArrowType.Decimal(10, 2, 128)),
                        struct("st", Field.notNullable("a", new ArrowType.Int(64, true))))),
                2,
                root -> {
                    BigIntVector id = (BigIntVector) root.getVector("id");
                    TimeStampMilliTZVector ts = (TimeStampMilliTZVector) root.getVector("ts");
                    VarCharVector s = (VarCharVector) root.getVector("s");
                    DecimalVector d = (DecimalVector) root.getVector("d");
                    StructVector st = (StructVector) root.getVector("st");
                    for (int i = 0; i < 2; i++) {
                        id.setSafe(i, i);
                        ts.setSafe(i, 1_000L * (i + 1));
                        s.setSafe(i, ("a" + i).getBytes(StandardCharsets.UTF_8));
                        d.setSafe(i, new BigDecimal(i + ".25"));
                        ((BigIntVector) st.getChild("a")).setSafe(i, 10 + i);
                        st.setIndexDefined(i);
                    }
                });
        writeFile(
                dir.resolve("b.vortex"),
                new Schema(List.of(
                        Field.nullable("id", new ArrowType.Int(64, true)),
                        Field.nullable("ts", new ArrowType.Timestamp(TimeUnit.NANOSECOND, "America/New_York")),
                        Field.nullable("s", ArrowType.Utf8View.INSTANCE),
                        Field.nullable("d", new ArrowType.Decimal(12, 2, 128)),
                        struct(
                                "st",
                                Field.nullable("a", new ArrowType.Int(64, true)),
                                Field.nullable("b", ArrowType.Utf8.INSTANCE)))),
                2,
                root -> {
                    ((BigIntVector) root.getVector("id")).setSafe(0, 2);
                    ((BigIntVector) root.getVector("id")).setSafe(1, 3);
                    ((TimeStampNanoTZVector) root.getVector("ts")).setSafe(0, 3_000_000_000L);
                    ((ViewVarCharVector) root.getVector("s")).setSafe(0, "b2".getBytes(StandardCharsets.UTF_8));
                    ((DecimalVector) root.getVector("d")).setSafe(0, new BigDecimal("1234567890.12"));
                    StructVector st = (StructVector) root.getVector("st");
                    ((BigIntVector) st.getChild("a")).setSafe(0, 12);
                    ((VarCharVector) st.getChild("b")).setSafe(0, "x".getBytes(StandardCharsets.UTF_8));
                    st.setIndexDefined(0);
                });
        writeFile(
                dir.resolve("c.vortex"),
                new Schema(List.of(
                        Field.nullable("id", new ArrowType.Int(64, true)),
                        Field.nullable("ts", new ArrowType.Timestamp(TimeUnit.MICROSECOND, "+01:00")))),
                1,
                root -> {
                    ((BigIntVector) root.getVector("id")).setSafe(0, 4);
                    ((TimeStampMicroTZVector) root.getVector("ts")).setSafe(0, 5_000_000L);
                });

        String schema = "id BIGINT, ts TIMESTAMP, s STRING, d DECIMAL(12,2), st STRUCT<a: BIGINT, b: STRING>";
        Function<Dataset<Row>, Dataset<Row>> columns =
                data -> data.selectExpr("id", "unix_micros(ts)", "s", "CAST(d AS STRING)", "st.a", "st.b");
        assertRows(
                dir,
                schema,
                columns,
                "[[0,1000000,a0,0.25,10,null], [1,2000000,a1,1.25,11,null], [2,3000000,b2,1234567890.12,12,x],"
                        + " [3,null,null,null,null,null], [4,5000000,null,null,null,null]]");
        assertRows(
                dir,
                schema,
                data -> columns.apply(data.where("ts > TIMESTAMP '1970-01-01 00:00:01.5'")),
                "[[1,2000000,a1,1.25,11,null], [2,3000000,b2,1234567890.12,12,x], [4,5000000,null,null,null,null]]");
        assertRows(
                dir,
                schema,
                data -> columns.apply(data.where("d > 1")),
                "[[1,2000000,a1,1.25,11,null], [2,3000000,b2,1234567890.12,12,x]]");
        assertRows(
                dir,
                schema,
                data -> columns.apply(data.where("s = 'b2' OR s = 'a0'")),
                "[[0,1000000,a0,0.25,10,null], [2,3000000,b2,1234567890.12,12,x]]");
        assertRows(dir, schema, data -> columns.apply(data.where("ts IS NULL")), "[[3,null,null,null,null,null]]");
        assertRows(dir, schema, data -> data.selectExpr("id", "st").where("st.a > 10"), "[[1,[11,null]], [2,[12,x]]]");
    }

    @Test
    void filesOfTimestampUnitsAndStringEncodingsReadWithTheInferredSchema() throws IOException {
        Path dir = tempDir.resolve("inferred");
        writeFile(
                dir.resolve("a.vortex"),
                new Schema(List.of(
                        Field.notNullable("id", new ArrowType.Int(64, true)),
                        Field.notNullable("ts", new ArrowType.Timestamp(TimeUnit.SECOND, null)),
                        Field.notNullable("s", ArrowType.Utf8.INSTANCE))),
                1,
                root -> {
                    ((BigIntVector) root.getVector("id")).setSafe(0, 0);
                    ((TimeStampSecVector) root.getVector("ts")).setSafe(0, 7);
                    ((VarCharVector) root.getVector("s")).setSafe(0, "a".getBytes(StandardCharsets.UTF_8));
                });
        writeFile(
                dir.resolve("b.vortex"),
                new Schema(List.of(
                        Field.nullable("id", new ArrowType.Int(64, true)),
                        Field.nullable("ts", new ArrowType.Timestamp(TimeUnit.NANOSECOND, null)),
                        Field.nullable("s", ArrowType.Utf8View.INSTANCE))),
                1,
                root -> {
                    ((BigIntVector) root.getVector("id")).setSafe(0, 1);
                    ((TimeStampNanoVector) root.getVector("ts")).setSafe(0, 8_000_000_000L);
                    ((ViewVarCharVector) root.getVector("s")).setSafe(0, "b".getBytes(StandardCharsets.UTF_8));
                });
        Dataset<Row> data = spark.read().format("vortex").load(dir.toString());
        assertEquals(
                "struct<id:bigint,ts:timestamp_ntz,s:string>", data.schema().catalogString());
        assertEquals(
                "[[0,7000000,a], [1,8000000,b]]",
                data.selectExpr("id", "unix_micros(CAST(ts AS TIMESTAMP))", "s")
                        .orderBy("id")
                        .collectAsList()
                        .toString());
    }

    @Test
    void withMergingOffAFileOfAnotherTypeFailsTheRead() throws IOException {
        Path dir = tempDir.resolve("unmerged");
        writeFile(
                dir.resolve("a.vortex"),
                new Schema(List.of(Field.nullable("x", new ArrowType.Int(64, true)))),
                1,
                root -> ((BigIntVector) root.getVector("x")).setSafe(0, 1));
        writeFile(
                dir.resolve("b.vortex"),
                new Schema(List.of(Field.nullable("x", ArrowType.Utf8.INSTANCE))),
                1,
                root -> ((VarCharVector) root.getVector("x")).setSafe(0, "1".getBytes(StandardCharsets.UTF_8)));
        try {
            List<Row> rows = spark.read()
                    .format("vortex")
                    .option("vortex.mergeSchema", "false")
                    .load(dir.toString())
                    .collectAsList();
            fail("read files of two types as one: " + rows);
        } catch (Exception e) {
            String message = describe(e);
            assertTrue(message.contains("column x "), message);
            assertTrue(message.contains(".vortex"), message);
        }
    }

    @Test
    void aTimestampInAnotherUnitIsNotReadAsItsMicroseconds() throws IOException {
        // Parquet reads a millisecond timestamp as BIGINT milliseconds, and the vector would return microseconds.
        Path dir = tempDir.resolve("millis");
        writeFile(
                dir.resolve("a.vortex"),
                new Schema(List.of(Field.nullable("x", new ArrowType.Timestamp(TimeUnit.MILLISECOND, "UTC")))),
                1,
                root -> ((TimeStampMilliTZVector) root.getVector("x")).setSafe(0, 1_000L));
        try {
            List<Row> rows = spark.read()
                    .format("vortex")
                    .schema("x BIGINT")
                    .load(dir.toString())
                    .collectAsList();
            fail("read a millisecond timestamp as BIGINT: " + rows);
        } catch (Exception e) {
            String message = describe(e);
            assertTrue(message.contains("column x as timestamp, which cannot be read as bigint"), message);
        }
    }

    private static Field struct(String name, Field... children) {
        return new Field(name, FieldType.nullable(ArrowType.Struct.INSTANCE), List.of(children));
    }

    private void assertRows(Path dir, String schema, Function<Dataset<Row>, Dataset<Row>> query, String expected) {
        List<Row> rows = query.apply(
                        spark.read().format("vortex").schema(schema).load(dir.toString()))
                .orderBy("id")
                .collectAsList();
        assertEquals(expected, rows.toString());
    }

    private static void writeFile(Path path, Schema schema, int rowCount, Consumer<VectorSchemaRoot> fill)
            throws IOException {
        Files.createDirectories(path.getParent());
        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator);
                VortexWriter writer = VortexWriter.builder(Session.create(), path.toString(), schema, allocator)
                        .build();
                ArrowArray array = ArrowArray.allocateNew(allocator);
                ArrowSchema arrowSchema = ArrowSchema.allocateNew(allocator)) {
            root.allocateNew();
            fill.accept(root);
            root.setRowCount(rowCount);
            Data.exportVectorSchemaRoot(allocator, root, null, array, arrowSchema);
            writer.writeBatch(array.memoryAddress(), arrowSchema.memoryAddress());
        }
    }
}
