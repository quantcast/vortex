// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;
import java.util.stream.LongStream;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.api.io.TempDir;

/**
 * Times {@code key IN (...)} through the V1 file format on rows sorted by a 31-character string key, against the same
 * filter kept out of the scan and against Parquet.
 *
 * <p>Run with {@code ./gradlew -Dvortex.benchmark=true :vortex-spark-3.5_2.13:test --tests '*InFilterBenchmark*'};
 * {@code -Dvortex.benchmark.rows=N} changes the row count. The timings go to the test's standard output.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@EnabledIfSystemProperty(named = "vortex.benchmark", matches = "true")
public final class VortexInFilterBenchmarkTest {
    private static final int RUNS = 3;
    private static final String KEY =
            "concat('k', lpad(CAST(id AS STRING), 10, '0'), '-'," + " substr(sha2(CAST(id AS STRING), 256), 1, 19))";

    private final long rows = Long.getLong("vortex.benchmark.rows", 4_000_000L);
    private SparkSession spark;

    @TempDir
    static Path tempDir;

    @BeforeAll
    void setUp() {
        spark = SparkSession.builder()
                .appName("VortexInFilterBenchmarkTest")
                .master("local[4]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.ui.enabled", "false")
                .config("spark.sql.adaptive.enabled", "false")
                .config("spark.sql.sources.useV1SourceList", "vortex,parquet")
                .getOrCreate();
        spark.conf().set("spark.sql.sources.useV1SourceList", "vortex,parquet");
        for (String format : List.of("vortex", "parquet")) {
            spark.range(0, rows, 1, 8)
                    .selectExpr(KEY + " AS key", "id AS value")
                    .write()
                    .format(format)
                    .save(path(format));
        }
    }

    @AfterAll
    void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @Test
    void inFilterTimings() {
        long step = rows / 1000;
        List<String> report = new ArrayList<>();
        report.add(time(
                "1000 keys spread over the table",
                keys(LongStream.range(0, 1000).map(i -> i * step + 7))));
        report.add(time(
                "1000 keys in one stretch of the table",
                keys(LongStream.range(0, 1000).map(i -> rows / 2 + i * 3))));
        report.add(time(
                "10 keys spread over the table", keys(LongStream.range(0, 10).map(i -> i * (rows / 10) + 7))));
        System.out.println("IN filter benchmark, " + rows + " rows, best of " + RUNS + " after one warm-up\n"
                + String.join("\n", report));
    }

    private String time(String label, String keys) {
        double pushed = bestSeconds("vortex", "key IN (" + keys + ")", label);
        double unpushed = bestSeconds("vortex", "concat(key, '') IN (" + keys + ")", label);
        double parquet = bestSeconds("parquet", "key IN (" + keys + ")", label);
        return String.format(
                Locale.ROOT,
                "%s: vortex pushed %.2f s, vortex not pushed %.2f s, parquet %.2f s",
                label,
                pushed,
                unpushed,
                parquet);
    }

    private double bestSeconds(String format, String filter, String label) {
        double best = Double.MAX_VALUE;
        long count = -1;
        for (int run = 0; run <= RUNS; run++) {
            long start = System.nanoTime();
            count = spark.read()
                    .format(format)
                    .load(path(format))
                    .filter(filter)
                    .count();
            double seconds = (System.nanoTime() - start) / 1e9;
            if (run > 0) {
                best = Math.min(best, seconds);
            }
        }
        assertEquals(label.startsWith("10 ") ? 10 : 1000, count, format + " " + label);
        return best;
    }

    private String keys(LongStream ids) {
        String idList = Arrays.stream(ids.toArray()).mapToObj(Long::toString).collect(Collectors.joining(","));
        return spark
                .sql("SELECT " + KEY + " AS key FROM (SELECT explode(array(" + idList + ")) AS id)")
                .collectAsList()
                .stream()
                .map(row -> "'" + row.getString(0) + "'")
                .collect(Collectors.joining(", "));
    }

    private String path(String format) {
        return tempDir.resolve("in-filter-" + format).toString();
    }
}
