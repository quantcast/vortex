// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.not;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.List;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

/** {@code IN} and {@code NOT IN} select the same rows from Vortex as from Parquet, through both read paths. */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexInFilterTest {
    private static final String V1_SOURCES = "spark.sql.sources.useV1SourceList";
    private static final String TABLE = "SELECT id,"
            + " CASE WHEN id % 11 = 0 THEN NULL ELSE concat('k', lpad(CAST(id AS STRING), 4, '0')) END AS s,"
            + " CASE WHEN id % 7 = 0 THEN NULL ELSE CAST(id % 50 AS INT) END AS i,"
            + " CASE WHEN id % 13 = 0 THEN NULL ELSE date_add(DATE '2026-01-01', CAST(id % 40 AS INT)) END AS d,"
            + " CASE WHEN id % 17 = 0 THEN NULL ELSE timestamp_seconds(1767225600 + (id % 30) * 3600) END AS ts,"
            + " CASE WHEN id % 19 = 0 THEN NULL ELSE CAST((id % 60) / 4 AS DECIMAL(10, 2)) END AS dec"
            + " FROM range(0, 400, 1, 4)";

    private SparkSession spark;

    @TempDir
    static Path tempDir;

    @BeforeAll
    void setUp() {
        spark = SparkSession.builder()
                .appName("VortexInFilterTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.ui.enabled", "false")
                .config("spark.sql.adaptive.enabled", "false")
                .config("spark.sql.session.timeZone", "UTC")
                .getOrCreate();
        spark.conf().set("spark.sql.session.timeZone", "UTC");
        for (String format : List.of("parquet", "vortex")) {
            spark.sql(TABLE).write().format(format).save(path(format));
        }
    }

    @AfterAll
    void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    static Stream<Arguments> filters() {
        String manyStrings = IntStream.range(0, 40)
                .mapToObj(n -> String.format("'k%04d'", n * 10 + 3))
                .collect(Collectors.joining(", "));
        String manyInts =
                IntStream.range(0, 21).mapToObj(n -> Integer.toString(n * 2)).collect(Collectors.joining(", "));
        String manyLongs =
                IntStream.range(0, 30).mapToObj(n -> Integer.toString(n * 13)).collect(Collectors.joining(", "));
        return Stream.of(
                        "s IN ('k0003', 'k0150', 'k0399')",
                        "s IN (" + manyStrings + ")",
                        "s IN ('k0003', NULL)",
                        "s IN ('k0003', 'k0003', 'k0150', 'k0150')",
                        "s IN ('absent', 'zzz')",
                        "s NOT IN ('k0003', 'k0150')",
                        "s NOT IN (" + manyStrings + ")",
                        "i IN (1, 2, 2, 49)",
                        "i IN (" + manyInts + ")",
                        "i IN (NULL, 4)",
                        "i NOT IN (1, 2, 3)",
                        "id IN (" + manyLongs + ")",
                        "d IN (DATE '2026-01-02', DATE '2026-01-20', DATE '2027-01-01')",
                        "d NOT IN (DATE '2026-01-02', DATE '2026-01-03')",
                        "ts IN (TIMESTAMP '2026-01-01 01:00:00', TIMESTAMP '2026-01-01 05:00:00')",
                        "dec IN (0.25, 1.50, 14.75, 99.99)",
                        "d IN (" + many(n -> String.format("DATE '2026-01-%02d'", n + 5)) + ")",
                        "ts IN (" + many(n -> String.format("TIMESTAMP '2026-01-01 %02d:00:00'", n + 3)) + ")",
                        "dec IN (" + many(n -> String.format("%d.%02d", n / 4 + 2, (n % 4) * 25)) + ")",
                        "s NOT IN (" + many(n -> String.format("'k%04d'", n * 7)) + ")",
                        "dec NOT IN (0.25, 1.50)",
                        "s IN ('k0003', 'k0004') OR i IN (5, 6)")
                .flatMap(filter -> Stream.of(Arguments.of("v1", filter), Arguments.of("v2", filter)));
    }

    /** Eighteen values, past the size at which the connector pushes a set instead of a disjunction. */
    private static String many(IntFunction<String> value) {
        return IntStream.range(0, 18).mapToObj(value).collect(Collectors.joining(", "));
    }

    @ParameterizedTest
    @MethodSource("filters")
    void aPushedInFilterSelectsTheSameRowsAsParquet(String readPath, String filter) {
        withReadPath(readPath, () -> {
            Dataset<Row> vortex = read("vortex").filter(filter);
            String plan = vortex.queryExecution().executedPlan().toString();
            assertTrue(plan.substring(plan.indexOf("PushedFilters")).contains("In("), plan);
            assertEquals(sorted(read("parquet").filter(filter)), sorted(vortex), filter);
        });
    }

    @ParameterizedTest
    @ValueSource(
            strings = {
                "s NOT IN ('k0003', NULL)",
                "i NOT IN (NULL, 4)",
                "ts NOT IN (TIMESTAMP '2026-01-01 01:00:00', NULL)"
            })
    void aNotInWithANullSelectsTheSameRowsAsParquet(String filter) {
        // Spark folds these to false before planning, so no scan sees them; the parity still holds.
        for (String readPath : List.of("v1", "v2")) {
            withReadPath(
                    readPath,
                    () -> assertEquals(
                            sorted(read("parquet").filter(filter)),
                            sorted(read("vortex").filter(filter)),
                            filter));
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"v1", "v2"})
    void anEmptyInListSelectsTheSameRowsAsParquet(String readPath) {
        withReadPath(readPath, () -> {
            for (Function<Column, Column> filter :
                    List.<Function<Column, Column>>of(column -> column.isin(), column -> not(column.isin()))) {
                assertEquals(
                        sorted(read("parquet").filter(filter.apply(col("i")))),
                        sorted(read("vortex").filter(filter.apply(col("i")))));
            }
        });
    }

    private void withReadPath(String readPath, Runnable test) {
        String previous = spark.conf().get(V1_SOURCES);
        spark.conf().set(V1_SOURCES, readPath.equals("v1") ? "vortex,parquet" : "parquet");
        try {
            test.run();
        } finally {
            spark.conf().set(V1_SOURCES, previous);
        }
    }

    private Dataset<Row> read(String format) {
        return spark.read().format(format).load(path(format));
    }

    private String path(String format) {
        return tempDir.resolve("in-" + format).toString();
    }

    private static List<String> sorted(Dataset<Row> data) {
        return data.collectAsList().stream().map(Row::toString).sorted().toList();
    }
}
