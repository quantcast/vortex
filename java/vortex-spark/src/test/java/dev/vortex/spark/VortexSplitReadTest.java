// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Stream;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/** A file larger than the split size is read by several Spark tasks that together return every row exactly once. */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
final class VortexSplitReadTest {
    private static final long ROWS = 100_000;

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    void setUp() {
        spark = SparkSession.builder()
                .appName("VortexSplitReadTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.ui.enabled", "false")
                .getOrCreate();
    }

    @AfterAll
    void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @Test
    @DisplayName("Row-range partitions of one file return every row once, with filters, nulls and partition columns")
    void splitFileIsReadExactlyOnce() throws IOException {
        Dataset<Row> df = spark.range(ROWS)
                .selectExpr(
                        "id",
                        "if(id % 7 = 0, null, concat('n', id)) as name",
                        "cast(id * 3 as double) as amount",
                        "if(id % 2 = 0, 'A', 'B') as grp");
        Path outputPath = tempDir.resolve("split");
        df.coalesce(1)
                .write()
                .format("vortex")
                .partitionBy("grp")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save();

        // Each partition directory holds one 50k-row file. Through Spark's own maxSplitBytes formula a budget of a
        // third of the largest file is 16666 rows, rounded down to two 8192-row blocks, so each file becomes four
        // ranges: three of 16384 rows and one of 848.
        long largestFile;
        try (Stream<Path> files = Files.walk(outputPath)) {
            largestFile = files.filter(p -> p.toString().endsWith(".vortex"))
                    .mapToLong(p -> p.toFile().length())
                    .max()
                    .orElseThrow();
        }
        spark.conf().set("spark.sql.files.maxPartitionBytes", largestFile / 3);
        spark.conf().set("spark.sql.files.openCostInBytes", 0);
        Dataset<Row> read = spark.read()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .option("vortex.read.splitFiles", "true")
                .load();
        assertEquals(8, read.rdd().getNumPartitions());

        assertEquals(
                df.orderBy("id").collectAsList(),
                read.select("id", "name", "amount", "grp").orderBy("id").collectAsList());
        assertEquals(ROWS, read.count());
        assertEquals(
                df.filter("amount > 60000 and grp = 'A'").count(),
                read.filter("amount > 60000 and grp = 'A'").count());
        assertEquals(
                df.filter("name is null").select("id").orderBy("id").collectAsList(),
                read.filter("name is null").select("id").orderBy("id").collectAsList());
    }
}
