// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.google.common.collect.ImmutableList;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.stream.Stream;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;
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

    @Test
    @DisplayName("Split options match in any case, and scan options override table options")
    void splitOptionsIgnoreCase() throws IOException {
        Path outputPath = tempDir.resolve("cased");
        spark.range(ROWS)
                .selectExpr("id", "concat('n', id) as name")
                .coalesce(1)
                .write()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save();
        long fileBytes;
        try (Stream<Path> files = Files.walk(outputPath)) {
            fileBytes = files.filter(p -> p.toString().endsWith(".vortex"))
                    .mapToLong(p -> p.toFile().length())
                    .sum();
        }
        // A third of the file is 33333 rows, rounded down to 32768: four ranges.
        spark.conf().set("spark.sql.files.maxPartitionBytes", fileBytes / 3);
        spark.conf().set("spark.sql.files.openCostInBytes", 0);
        var schema = spark.read()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .load()
                .schema();
        ImmutableList<String> paths = ImmutableList.of(outputPath.toUri().toString());

        VortexTable off = new VortexTable(paths, schema, Map.of("vortex.read.splitFiles", "false"), new Transform[0]);
        assertEquals(4, partitions(off, Map.of("vortex.read.splitfiles", "true")));

        VortexTable on = new VortexTable(paths, schema, Map.of("VORTEX.READ.SPLITFILES", "true"), new Transform[0]);
        assertEquals(1, partitions(on, Map.of("vortex.read.splitFiles", "false")));
        assertEquals(4, partitions(on, Map.of()));
        assertEquals(1, partitions(on, Map.of("Vortex.Read.MaxSplitBytes", Long.toString(Long.MAX_VALUE))));

        // Other options keep the table's spelling, which their case-sensitive lookups rely on.
        VortexTable threads = new VortexTable(paths, schema, Map.of("vortex.workerThreads", "2"), new Transform[0]);
        var partition = (VortexFilePartition)
                threads.newScanBuilder(new CaseInsensitiveStringMap(Map.of("vortex.workerThreads", "8")))
                        .build()
                        .toBatch()
                        .planInputPartitions()[0];
        assertEquals("8", partition.formatOptions().get("vortex.workerThreads"));
    }

    private static int partitions(VortexTable table, Map<String, String> scanOptions) {
        return table.newScanBuilder(new CaseInsensitiveStringMap(scanOptions))
                .build()
                .toBatch()
                .planInputPartitions()
                .length;
    }
}
