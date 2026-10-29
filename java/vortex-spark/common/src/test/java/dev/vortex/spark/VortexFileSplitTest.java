// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
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
 * Files larger than Spark's split size are read by several tasks, through the V2 scan and the V1 file format alike, and
 * together those tasks return every row exactly once.
 *
 * <p>Each test fixes {@code openCostInBytes} at 0 and {@code minPartitionNum} at 1, so the split size is
 * {@code maxPartitionBytes} capped at the dataset size and the expected partition count follows from the file sizes.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexFileSplitTest {
    // Large enough that every column holds several chunks.
    private static final long ROWS = 300_000;
    private static final int[] SPLITS_PER_FILE = {1, 2, 3, 7, 50};

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexFileSplitTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.sql.adaptive.enabled", "false")
                .config("spark.sql.optimizer.nestedSchemaPruning.enabled", "true")
                .config("spark.ui.enabled", "false")
                .getOrCreate();
    }

    @AfterEach
    public void resetConf() {
        for (String key : List.of(
                "spark.sql.files.maxPartitionBytes",
                "spark.sql.files.openCostInBytes",
                "spark.sql.files.minPartitionNum",
                "spark.sql.files.maxPartitionNum",
                "spark.sql.codegen.wholeStage",
                "spark.sql.sources.useV1SourceList")) {
            spark.conf().unset(key);
        }
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @ParameterizedTest(name = "{0}")
    @ValueSource(strings = {"v2", "v1", "v1-rows"})
    @DisplayName("Every split size returns each row of a many-chunk file once, from as many tasks as Spark plans")
    void everySplitSizeReadsEachRowOnce(String mode) {
        Path output = write(rows(ROWS).coalesce(1), "many_chunks", null);
        long fileSize = fileSizes(output).get(0);
        // Smaller than a chunk, unaligned to anything, and larger than the file.
        List<Long> splitSizes = new ArrayList<>(List.of(16_411L, 65_537L, 262_147L, 1_000_003L, fileSize + 1));
        for (int splits : SPLITS_PER_FILE) {
            splitSizes.add(fileSize / splits + splits % 2);
        }

        for (long splitSize : splitSizes) {
            Dataset<Row> read = read(output, mode, splitSize);
            String context = "mode=" + mode + " splitSize=" + splitSize + " fileSize=" + fileSize;
            assertEquals(
                    expectedPartitions(List.of(fileSize), splitSize), read.rdd().getNumPartitions(), context);
            assertEquals(
                    RowFactory.create(ROWS, ROWS, 0L, ROWS - 1),
                    read.selectExpr("count(id)", "count(distinct id)", "min(id)", "max(id)")
                            .first(),
                    context);
        }

        Dataset<Row> read = read(output, mode, 65_537);
        assertSameRows(rows(ROWS), read);
        assertSameRows(
                rows(ROWS).filter("amount > 400000 and name is not null"),
                read.filter("amount > 400000 and name is not null"));
    }

    @ParameterizedTest(name = "{0}")
    @ValueSource(strings = {"v2", "v1", "v1-rows"})
    @DisplayName("Each split reads its own contiguous rows, also when one task reads several splits of a file")
    void splitsSpreadRowsAcrossTasks(String mode) {
        Path output = write(rows(ROWS).coalesce(1), "spread", null);
        long splitSize = fileSizes(output).get(0) / 7 + 1;

        List<Row> splits = read(output, mode, splitSize)
                .selectExpr("input_file_block_start() as start", "id")
                .groupBy("start")
                .agg(functions.count("id"), functions.min("id"), functions.max("id"))
                .orderBy("start")
                .collectAsList();
        assertEquals(7, splits.size(), splits.toString());
        long next = 0;
        for (Row split : splits) {
            assertEquals(next, split.getLong(2), splits.toString());
            assertEquals(split.getLong(1), split.getLong(3) - split.getLong(2) + 1, splits.toString());
            next = split.getLong(3) + 1;
        }
        assertEquals(ROWS, next);

        spark.conf().set("spark.sql.files.maxPartitionNum", 1);
        Dataset<Row> packed = spark.read().format("vortex").load(output.toString());
        assertEquals(1, packed.rdd().getNumPartitions());
        assertEquals(
                RowFactory.create(ROWS, ROWS, 0L, ROWS - 1),
                packed.selectExpr("count(id)", "count(distinct id)", "min(id)", "max(id)")
                        .first());
    }

    @ParameterizedTest(name = "{0}")
    @ValueSource(strings = {"v2", "v1"})
    @DisplayName("Files of no rows, one row and one chunk split into tasks that return each row once")
    void smallFilesReadEachRowOnce(String mode) {
        for (long rows : new long[] {0, 1, 5000}) {
            Path output = write(rows(rows).coalesce(1), "small_" + rows, null);
            List<Long> sizes = fileSizes(output);
            assertEquals(1, sizes.size(), output.toString());
            long fileSize = sizes.get(0);
            for (int splits : SPLITS_PER_FILE) {
                long splitSize = fileSize / splits + splits % 2;
                Dataset<Row> read = read(output, mode, splitSize);
                String context = "mode=" + mode + " rows=" + rows + " splitSize=" + splitSize;
                assertEquals(expectedPartitions(sizes, splitSize), read.rdd().getNumPartitions(), context);
                assertSameRows(rows(rows), read);
                assertEquals(rows, read.count(), context);
            }
        }
    }

    @ParameterizedTest(name = "{0}")
    @ValueSource(strings = {"v2", "v1"})
    @DisplayName("Split files of a partitioned table keep partition values, filters and nested pruning")
    void partitionedSplitsKeepFiltersPartitionValuesAndPruning(String mode) {
        Dataset<Row> expected = rows(ROWS).selectExpr("*", "if(id % 2 = 0, 'A', 'B') as grp");
        Path output = write(expected.coalesce(1), "partitioned", "grp");
        List<Long> sizes = fileSizes(output);
        assertEquals(2, sizes.size(), output.toString());
        long splitSize = sizes.get(0) / 3 + 1;

        Dataset<Row> read = read(output, mode, splitSize);
        int partitions = read.rdd().getNumPartitions();
        assertEquals(expectedPartitions(sizes, splitSize), partitions);
        assertTrue(partitions >= 6, "partitions=" + partitions);

        assertSameRows(expected, read);
        String filter = "amount > 300000 and name is not null and grp = 'A'";
        assertSameRows(expected.filter(filter), read.filter(filter));
        assertSameRows(expected.select("id", "s.a", "grp"), read.select("id", "s.a", "grp"));
        if (mode.equals("v2")) {
            String plan =
                    read.select("id", "s.a").queryExecution().executedPlan().toString();
            assertTrue(plan.contains("ReadSchema: struct<id:bigint,s:struct<a:bigint>>"), plan);
        }
        assertSameRows(expected.groupBy("grp").count(), read.groupBy("grp").count());
        assertEquals(ROWS, read.count());
    }

    @Test
    @DisplayName("A pushed COUNT(*) counts each split file once, and so does the unpushed count")
    void countStarCountsEachFileOnce() {
        Path output = write(rows(ROWS).coalesce(1), "count", null);
        long splitSize = fileSizes(output).get(0) / 7;

        Dataset<Row> pushed = read(output, "v2", splitSize).selectExpr("count(*)");
        assertTrue(pushed.queryExecution().executedPlan().toString().contains("PushedAggregation: [COUNT(*)]"));
        // Splitting would only open the footer once per split.
        assertEquals(
                1,
                pushed.queryExecution()
                        .executedPlan()
                        .collectLeaves()
                        .head()
                        .executeColumnar()
                        .getNumPartitions());
        assertEquals(ROWS, pushed.first().getLong(0));

        Dataset<Row> unpushed = spark.read()
                .format("vortex")
                .option("vortex.aggregatePushdown", "false")
                .load(output.toString());
        assertTrue(unpushed.rdd().getNumPartitions() >= 7);
        assertEquals(ROWS, unpushed.selectExpr("count(*)").first().getLong(0));
    }

    @Test
    @DisplayName("Scan statistics read each split file's footer once")
    void statisticsCountEachSplitFileOnce() {
        Path output = write(rows(ROWS).selectExpr("*", "id % 2 as grp").coalesce(1), "stats", "grp");
        Dataset<Row> read = read(output, "v2", fileSizes(output).get(0) / 5);
        assertTrue(read.rdd().getNumPartitions() >= 10);

        var statistics =
                read.queryExecution().optimizedPlan().collectLeaves().head().stats();
        assertTrue(statistics.rowCount().isDefined());
        assertEquals(ROWS, statistics.rowCount().get().longValue());
    }

    @ParameterizedTest(name = "{0}")
    @ValueSource(strings = {"v2", "v1"})
    @DisplayName("vortex.splitFiles turns splitting off under any key case")
    void splitFilesOptionTurnsSplittingOff(String mode) {
        Path output = write(rows(ROWS).coalesce(1), "option", null);
        List<Long> sizes = fileSizes(output);
        long splitSize = sizes.get(0) / 4;
        assertEquals(
                expectedPartitions(sizes, splitSize),
                read(output, mode, splitSize).rdd().getNumPartitions());
        for (String key : List.of("vortex.splitFiles", "VORTEX.SPLITFILES", "vortex.splitfiles")) {
            Dataset<Row> read =
                    spark.read().format("vortex").option(key, "false").load(output.toString());
            assertEquals(1, read.rdd().getNumPartitions(), key);
            assertEquals(ROWS, read.count(), key);
        }
    }

    private Dataset<Row> rows(long count) {
        return spark.range(count)
                .selectExpr(
                        "id",
                        "if(id % 7 = 0, null, concat('n', id)) as name",
                        "cast(id * 3 as double) as amount",
                        "named_struct('a', id + 100, 'b', concat('b', id)) as s");
    }

    private Path write(Dataset<Row> data, String name, String partitionColumn) {
        Path output = tempDir.resolve(name);
        var writer = data.write().format("vortex").mode(SaveMode.Overwrite);
        if (partitionColumn != null) {
            writer = writer.partitionBy(partitionColumn);
        }
        writer.save(output.toString());
        return output;
    }

    /**
     * Reads through the V2 scan, the V1 file format, or the V1 format's row reader, with Spark splitting files at
     * {@code splitSize} bytes.
     */
    private Dataset<Row> read(Path output, String mode, long splitSize) {
        boolean v1 = !mode.equals("v2");
        spark.conf().set("spark.sql.sources.useV1SourceList", v1 ? "vortex" : "");
        // Whole-stage codegen off makes `FileSourceScanExec` ask the format for rows, not batches.
        spark.conf().set("spark.sql.codegen.wholeStage", !mode.equals("v1-rows"));
        spark.conf().set("spark.sql.files.maxPartitionBytes", splitSize);
        spark.conf().set("spark.sql.files.openCostInBytes", 0);
        spark.conf().set("spark.sql.files.minPartitionNum", 1);
        Dataset<Row> read = spark.read().format("vortex").load(output.toString());
        String plan = read.queryExecution().executedPlan().toString();
        assertTrue(plan.contains(v1 ? "FileScan vortex" : "BatchScan"), plan);
        return read;
    }

    private static List<Long> fileSizes(Path output) {
        try (Stream<Path> files = Files.walk(output)) {
            return files.filter(path -> path.toString().endsWith(".vortex"))
                    .map(path -> path.toFile().length())
                    .sorted(Comparator.reverseOrder())
                    .toList();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** Spark's split and packing rule ({@code FilePartition.getFilePartitions}) with no open cost. */
    private static int expectedPartitions(List<Long> fileSizes, long maxPartitionBytes) {
        long splitSize = Math.min(
                maxPartitionBytes, fileSizes.stream().mapToLong(Long::longValue).sum());
        List<Long> splits = new ArrayList<>();
        for (long fileSize : fileSizes) {
            for (long start = 0; start < fileSize; start += splitSize) {
                splits.add(Math.min(splitSize, fileSize - start));
            }
        }
        splits.sort(Comparator.reverseOrder());
        int partitions = 0;
        long current = 0;
        for (long split : splits) {
            if (current > 0 && current + split > splitSize) {
                partitions++;
                current = 0;
            }
            current += split;
        }
        return current > 0 ? partitions + 1 : partitions;
    }

    private static void assertSameRows(Dataset<Row> expected, Dataset<Row> actual) {
        Dataset<Row> projected = actual.select(expected.columns()[0], tail(expected.columns()));
        assertEquals(expected.count(), projected.count());
        assertEquals(0, expected.exceptAll(projected).count());
        assertEquals(0, projected.exceptAll(expected).count());
    }

    private static String[] tail(String[] columns) {
        String[] tail = new String[columns.length - 1];
        System.arraycopy(columns, 1, tail, 0, tail.length);
        return tail;
    }
}
