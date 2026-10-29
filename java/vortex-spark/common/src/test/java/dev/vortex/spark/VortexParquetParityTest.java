// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * The same write and read give the same rows and schema through Vortex as through Parquet.
 *
 * <p>Each case runs one operation against both formats in fresh directories and compares what comes back, so the
 * expected behavior is whatever Spark's Parquet source does.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexParquetParityTest {
    private static final String NESTED = "SELECT id,"
            + " CASE WHEN id % 5 = 0 THEN NULL ELSE named_struct("
            + "   'a', CASE WHEN id % 3 = 0 THEN NULL ELSE id END,"
            + "   'b', CASE WHEN id % 4 = 1 THEN NULL"
            + "        ELSE named_struct('c', CASE WHEN id % 4 = 2 THEN NULL ELSE CAST(id AS STRING) END) END) END AS s,"
            + " CASE WHEN id % 6 = 0 THEN NULL WHEN id % 6 = 1 THEN array() ELSE array(id, NULL, id + 1) END AS arr,"
            + " CASE WHEN id % 7 = 0 THEN NULL WHEN id % 7 = 1 THEN map()"
            + "      ELSE map('k', CASE WHEN id % 2 = 0 THEN NULL ELSE id END, 'j', id) END AS m,"
            + " CASE WHEN id % 8 = 0 THEN NULL"
            + "      ELSE array(named_struct('f', CASE WHEN id % 3 = 1 THEN NULL ELSE id END), NULL) END AS arrs,"
            + " CASE WHEN id % 9 = 0 THEN NULL"
            + "      ELSE map('x', CASE WHEN id % 2 = 1 THEN NULL ELSE array(CAST(id AS STRING), NULL) END) END AS ma"
            + " FROM range(40)";

    private static final String FILTERABLE = "SELECT id,"
            + " CASE WHEN id % 5 = 0 THEN NULL ELSE CAST(id % 7 AS INT) END AS n,"
            + " CASE id % 3 WHEN 0 THEN concat('ab', id) WHEN 1 THEN concat('b', id) ELSE NULL END AS v,"
            + " CAST(id % 3 AS INT) AS pi,"
            + " CASE id % 4 WHEN 0 THEN 'C++' WHEN 1 THEN 'a/b' WHEN 2 THEN NULL ELSE 'Cx' END AS ps"
            + " FROM range(60)";

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    void setUp() {
        spark = SparkSession.builder()
                .appName("VortexParquetParityTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.ui.enabled", "false")
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.sql.adaptive.enabled", "false")
                .config("spark.sql.session.timeZone", "UTC")
                .getOrCreate();
        spark.conf().set("spark.sql.session.timeZone", "UTC");
    }

    @AfterAll
    void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"append", "overwrite", "errorifexists", "ignore"})
    void aSaveModeOnAnExistingPathActsLikeParquet(String mode) {
        assertMatchesParquet("mode-" + mode, (format, root) -> {
            spark.range(0, 10).write().format(format).save(root);
            String outcome = attempt(
                    () -> spark.range(10, 15).write().format(format).mode(mode).save(root));
            return withOutcome(read(format, root), outcome);
        });
    }

    @ParameterizedTest
    @ValueSource(strings = {"static", "dynamic"})
    void aPartitionOverwriteActsLikeParquet(String mode) {
        assertMatchesParquet("partition-overwrite-" + mode, (format, root) -> {
            spark.sql("SELECT id, CAST(id % 3 AS INT) AS p FROM range(12)")
                    .write()
                    .format(format)
                    .partitionBy("p")
                    .save(root);
            spark.conf().set("spark.sql.sources.partitionOverwriteMode", mode);
            try {
                spark.sql("SELECT 100L AS id, 1 AS p UNION ALL SELECT 101L, 5")
                        .write()
                        .format(format)
                        .partitionBy("p")
                        .mode("overwrite")
                        .save(root);
            } finally {
                spark.conf().unset("spark.sql.sources.partitionOverwriteMode");
            }
            return read(format, root);
        });
    }

    @Test
    void partitionColumnsOfEveryTypeRoundTripLikeParquet() {
        assertMatchesParquet("partition-types", (format, root) -> {
            spark.sql("SELECT id,"
                            + " CAST(id % 2 AS INT) AS pi,"
                            + " CASE id % 4 WHEN 0 THEN 'a/b' WHEN 1 THEN '100%' WHEN 2 THEN NULL ELSE 'C++ é' END AS ps,"
                            + " CASE WHEN id % 5 = 0 THEN NULL ELSE date_add(DATE '2026-10-29', CAST(id % 2 AS INT)) END"
                            + "   AS pd,"
                            + " CASE WHEN id % 6 = 0 THEN NULL ELSE timestamp_seconds(1793235723 + (id % 2) * 3600) END"
                            + "   AS pt,"
                            + " CASE WHEN id % 7 = 0 THEN NULL ELSE id % 3 = 0 END AS pb"
                            + " FROM range(24)")
                    .coalesce(1)
                    .write()
                    .format(format)
                    .partitionBy("pi", "ps", "pd", "pt", "pb")
                    .save(root);
            return read(format, root);
        });
        assertEquals(
                directories(tempDir.resolve("partition-types-parquet")),
                directories(tempDir.resolve("partition-types-vortex")));
    }

    @Test
    void nestedColumnsWithNullsAtEveryLevelRoundTripLikeParquet() {
        assertMatchesParquet("nested", (format, root) -> {
            spark.sql(NESTED).write().format(format).save(root);
            return read(format, root);
        });
    }

    @Test
    void selectingNestedFieldsActsLikeParquet() {
        assertMatchesParquet("nested-select", (format, root) -> {
            spark.sql(NESTED).write().format(format).save(root);
            return read(format, root)
                    .selectExpr("id", "s.a", "s.b.c", "arrs.f", "m['k']", "try_element_at(arr, 2)", "ma['x'][0]");
        });
    }

    @ParameterizedTest
    @ValueSource(
            strings = {
                "n = 3",
                "n < 3",
                "n IN (1, 2, 6)",
                "n IS NULL",
                "v LIKE 'ab%'",
                "v IS NULL",
                "pi = 1",
                "pi < 2",
                "pi IN (0, 2)",
                "ps IS NULL",
                "ps LIKE 'C+%'",
                "ps = 'a/b'",
                "n < 3 AND pi = 2",
                "n IS NULL OR ps = 'Cx'"
            })
    void aFilterSelectsTheSameRowsAsParquet(String filter) {
        assertMatchesParquet("filter", (format, root) -> {
            if (!Files.exists(Path.of(root))) {
                spark.sql(FILTERABLE)
                        .repartition(2)
                        .write()
                        .format(format)
                        .partitionBy("pi", "ps")
                        .save(root);
            }
            return read(format, root).filter(filter);
        });
    }

    @Test
    void inputFileNameSeesAsManyFilesAsParquet() {
        assertMatchesParquet("input-file", (format, root) -> {
            spark.sql("SELECT id, CAST(id % 2 AS INT) AS p FROM range(30)")
                    .repartition(3)
                    .write()
                    .format(format)
                    .partitionBy("p")
                    .save(root);
            return read(format, root)
                    .selectExpr("input_file_name() AS file")
                    .distinct()
                    .groupBy()
                    .count();
        });
    }

    @Test
    void aNarrowerUserSchemaReadsLikeParquet() {
        assertMatchesParquet("narrow-schema", (format, root) -> {
            writeWideTable(format, root);
            return spark.read()
                    .format(format)
                    .schema("b BIGINT, id BIGINT, missing INT")
                    .load(root);
        });
    }

    @Test
    @Disabled("the reader takes a struct's children by position, so a narrower struct in the read schema gets the"
            + " wrong fields")
    void aNarrowerStructInTheUserSchemaReadsLikeParquet() {
        assertMatchesParquet("narrow-struct", (format, root) -> {
            writeWideTable(format, root);
            return spark.read()
                    .format(format)
                    .schema("id BIGINT, c STRUCT<y: STRING>")
                    .load(root);
        });
    }

    private void writeWideTable(String format, String root) {
        spark.sql("SELECT id, CAST(id AS STRING) AS a, id * 2 AS b, named_struct('x', id, 'y', 'y') AS c"
                        + " FROM range(10)")
                .write()
                .format(format)
                .save(root);
    }

    @Test
    void twoAppendsWithReorderedColumnsReadLikeParquet() {
        assertMatchesParquet("reordered", (format, root) -> {
            spark.sql("SELECT id, CAST(id AS STRING) AS a, id * 2 AS b FROM range(5)")
                    .coalesce(1)
                    .write()
                    .format(format)
                    .save(root);
            spark.sql("SELECT id * 2 AS b, id, CAST(id AS STRING) AS a FROM range(5, 10)")
                    .coalesce(1)
                    .write()
                    .format(format)
                    .mode("append")
                    .save(root);
            // Either format takes its column order from whichever file it lists first, so compare by name.
            return read(format, root).select("id", "a", "b");
        });
    }

    @Test
    void readColumnsAreNullableLikeParquet() {
        assertMatchesParquet("nullability", (format, root) -> {
            spark.sql("SELECT id, CAST(id AS STRING) AS s, named_struct('x', id) AS n FROM range(3)")
                    .write()
                    .format(format)
                    .save(root);
            Dataset<Row> data = read(format, root);
            return withOutcome(data, data.schema().treeString());
        });
    }

    @FunctionalInterface
    private interface Operation {
        Dataset<Row> run(String format, String root);
    }

    private void assertMatchesParquet(String name, Operation operation) {
        Dataset<Row> parquet =
                operation.run("parquet", tempDir.resolve(name + "-parquet").toString());
        Dataset<Row> vortex =
                operation.run("vortex", tempDir.resolve(name + "-vortex").toString());
        assertEquals(parquet.schema().catalogString(), vortex.schema().catalogString(), "schema");
        assertEquals(sorted(parquet), sorted(vortex), "rows");
    }

    private Dataset<Row> read(String format, String root) {
        return spark.read().format(format).load(root);
    }

    private static Dataset<Row> withOutcome(Dataset<Row> data, String outcome) {
        return data.withColumn("outcome", org.apache.spark.sql.functions.lit(outcome));
    }

    private static String attempt(Runnable action) {
        try {
            action.run();
            return "succeeded";
        } catch (Exception e) {
            return e.getClass().getSimpleName();
        }
    }

    private static List<String> sorted(Dataset<Row> data) {
        return data.collectAsList().stream().map(Row::toString).sorted().toList();
    }

    private static List<String> directories(Path root) {
        try (Stream<Path> paths = Files.walk(root)) {
            return paths.filter(Files::isDirectory)
                    .map(path -> root.relativize(path).toString())
                    .sorted()
                    .collect(Collectors.toList());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
