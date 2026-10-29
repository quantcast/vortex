// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.apache.spark.sql.functions.call_udf;
import static org.apache.spark.sql.functions.col;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.FilterOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.LongStream;
import java.util.stream.Stream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.LocalFileSystem;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.util.Progressable;
import org.apache.spark.TaskContext;
import org.apache.spark.internal.io.FileCommitProtocol.TaskCommitMessage;
import org.apache.spark.scheduler.SparkListener;
import org.apache.spark.scheduler.SparkListenerTaskEnd;
import org.apache.spark.sql.DataFrameWriter;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.execution.datasources.SQLHadoopMapReduceCommitProtocol;
import org.apache.spark.sql.types.DataTypes;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Writes follow Spark's commit protocol the way Parquet does: failed or retried attempts leave no rows behind, output
 * stays out of sight until the job commits, and separate applications never replace each other's files.
 */
public final class VortexWriteCommitTest {
    private static final AtomicInteger INJECTED = new AtomicInteger();
    private static final ConcurrentLinkedQueue<Integer> CLOSE_FAILURE_ATTEMPTS = new ConcurrentLinkedQueue<>();
    private static volatile CountDownLatch firstTaskDone;
    private static volatile CountDownLatch release;
    private static volatile Path failingOutput;
    private static volatile String failingFormat;

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeEach
    void setUp() {
        INJECTED.set(0);
        CLOSE_FAILURE_ATTEMPTS.clear();
        start(Map.of());
    }

    @AfterEach
    void tearDown() {
        if (release != null) {
            release.countDown();
        }
        stop();
    }

    // `local[2,2]` lets a failed task run a second attempt.
    private void start(Map<String, String> extra) {
        SparkSession.Builder builder = SparkSession.builder()
                .appName("VortexWriteCommitTest")
                .master("local[2,2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.ui.enabled", "false")
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.sql.adaptive.enabled", "false")
                .config("spark.sql.warehouse.dir", tempDir.resolve("warehouse").toString());
        extra.forEach(builder::config);
        spark = builder.getOrCreate();
    }

    private void stop() {
        if (spark != null) {
            spark.stop();
            spark = null;
        }
        SparkSession.clearActiveSession();
        SparkSession.clearDefaultSession();
    }

    @ParameterizedTest(name = "{0}, partitioned={1}")
    @CsvSource({"vortex,false", "vortex,true", "parquet,false", "parquet,true"})
    void aRetriedTaskWritesEachRowOnce(String format, boolean partitioned) throws IOException {
        Path output = tempDir.resolve("retry with space");
        failingOutput = output;
        failingFormat = format;
        spark.udf()
                .register(
                        "fail_first_attempt",
                        (UDF1<Long, Long>) id -> {
                            if (id == 80 && TaskContext.get().attemptNumber() == 0) {
                                assertFalse(
                                        dataFiles(failingOutput, failingFormat).isEmpty(),
                                        "the attempt must fail after it has written a file");
                                INJECTED.incrementAndGet();
                                throw new IOException("injected task failure");
                            }
                            return id;
                        },
                        DataTypes.LongType);
        Dataset<Row> input = spark.range(0, 128, 1, 1)
                .selectExpr("id", "CASE WHEN id < 64 THEN 'Côte' ELSE 'other' END AS p")
                // Sorted up front so the partitioned writer streams rows instead of sorting before it writes.
                .sortWithinPartitions("p")
                .select(call_udf("fail_first_attempt", col("id")).as("id"), col("p"));
        DataFrameWriter<Row> writer = input.write().format(format).option("vortex.write.batch.size", "8");
        if (partitioned) {
            writer = writer.partitionBy("p");
        }
        writer.save(output.toString());

        assertEquals(1, INJECTED.get());
        assertEquals(range(0, 128), ids(format, output));
        assertEquals(partitioned ? 2 : 1, dataFiles(output, format).size());
        assertCommitted(output);
    }

    @ParameterizedTest
    @ValueSource(strings = {"vortex", "parquet"})
    void aFileThatFailsToCloseIsRetriedWithoutItsSiblings(String format) throws IOException {
        Path output = tempDir.resolve("close-failure");
        spark.sparkContext()
                .hadoopConfiguration()
                .setClass("fs.file.impl", FailCloseFileSystem.class, LocalFileSystem.class);
        spark.sparkContext().hadoopConfiguration().setBoolean("fs.file.impl.disable.cache", true);
        writeRange(format, output, 0, 128, true, "append");

        assertEquals(1, INJECTED.get());
        // A second attempt runs only if the writer let the close failure fail the first one.
        assertEquals(
                List.of(0, 1),
                CLOSE_FAILURE_ATTEMPTS.stream().distinct().sorted().toList());
        assertEquals(range(0, 128), ids(format, output));
        assertEquals(2, dataFiles(output, format).size());
        assertCommitted(output);
    }

    @ParameterizedTest
    @ValueSource(strings = {"vortex", "parquet"})
    void aTaskThatFailsAfterCommittingIsRetriedWithoutDuplicates(String format) throws IOException {
        Path output = tempDir.resolve("commit-retry");
        spark.conf().set("spark.sql.sources.commitProtocolClass", FailFirstTaskCommit.class.getName());
        writeRange(format, output, 0, 128, true, "append");

        assertEquals(1, INJECTED.get());
        assertEquals(range(0, 128), ids(format, output));
        assertEquals(2, dataFiles(output, format).size());
        assertCommitted(output);
    }

    @ParameterizedTest
    @ValueSource(strings = {"vortex", "parquet"})
    void aFailedJobLeavesNoFilesBehind(String format) throws IOException {
        Path output = tempDir.resolve("failed-job");
        firstTaskDone = new CountDownLatch(1);
        spark.sparkContext().addSparkListener(new FirstTaskListener());
        spark.udf()
                .register(
                        "fail_after_sibling",
                        (UDF1<Long, Long>) id -> {
                            if (id == 128) {
                                if (!firstTaskDone.await(60, TimeUnit.SECONDS)) {
                                    throw new IOException("the sibling task never finished");
                                }
                                INJECTED.incrementAndGet();
                                throw new IOException("injected permanent failure");
                            }
                            return id;
                        },
                        DataTypes.LongType);
        Dataset<Row> input = spark.range(0, 256, 1, 2)
                .select(call_udf("fail_after_sibling", col("id")).as("id"));

        assertThrows(Exception.class, () -> input.write().format(format).save(output.toString()));
        assertEquals(2, INJECTED.get());
        assertEquals(List.of(), dataFiles(output, format));
        assertFalse(Files.exists(output.resolve("_SUCCESS")));
        assertFalse(Files.exists(output.resolve("_temporary")));
    }

    @ParameterizedTest
    @ValueSource(strings = {"vortex", "parquet"})
    void committedTaskOutputStaysHiddenUntilTheJobCommits(String format) throws Exception {
        Path output = tempDir.resolve("visibility");
        writeRange(format, output, -1, 0, false, "append");
        firstTaskDone = new CountDownLatch(1);
        release = new CountDownLatch(1);
        spark.sparkContext().addSparkListener(new FirstTaskListener());
        spark.udf()
                .register(
                        "pause",
                        (UDF1<Long, Long>) id -> {
                            if (id == 128 && !release.await(60, TimeUnit.SECONDS)) {
                                throw new IOException("the reader never released the write");
                            }
                            return id;
                        },
                        DataTypes.LongType);
        Dataset<Row> input =
                spark.range(0, 256, 1, 2).select(call_udf("pause", col("id")).as("id"));
        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<?> write = executor.submit(
                    () -> input.write().format(format).mode("append").save(output.toString()));
            assertTrue(firstTaskDone.await(60, TimeUnit.SECONDS), "the first task never committed");

            assertFalse(dataFiles(output.resolve("_temporary"), format).isEmpty());
            assertEquals(List.of(-1L), ids(format, output));

            release.countDown();
            write.get(60, TimeUnit.SECONDS);
            assertEquals(range(-1, 256), ids(format, output));
            assertCommitted(output);
        } finally {
            release.countDown();
            executor.shutdownNow();
        }
    }

    @ParameterizedTest(name = "{0}, partitioned={1}")
    @CsvSource({"vortex,false", "vortex,true", "parquet,false", "parquet,true"})
    void appendsFromSeparateApplicationsKeepBothRuns(String format, boolean partitioned) throws IOException {
        Path output = tempDir.resolve("append");
        String firstApplication = spark.sparkContext().applicationId();
        writeRange(format, output, 0, 10, partitioned, "append");
        List<Path> firstFiles = dataFiles(output, format);

        stop();
        start(Map.of());
        assertNotEquals(firstApplication, spark.sparkContext().applicationId());
        writeRange(format, output, 10, 20, partitioned, "append");

        assertEquals(range(0, 20), ids(format, output));
        assertTrue(dataFiles(output, format).containsAll(firstFiles));
        assertEquals(firstFiles.size() * 2, dataFiles(output, format).size());
    }

    // The version 2 committer moves each task's files into place at task commit, so only unique file names keep
    // the two runs apart.
    @ParameterizedTest
    @ValueSource(strings = {"vortex", "parquet"})
    void appendsFromSeparateApplicationsWithTheVersionTwoCommitterKeepBothRuns(String format) throws IOException {
        Map<String, String> conf = Map.of("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2");
        Path output = tempDir.resolve("append-v2");
        stop();
        start(conf);
        spark.range(0, 100, 1, 4).write().format(format).mode("append").save(output.toString());
        stop();
        start(conf);
        spark.range(100, 250, 1, 4).write().format(format).mode("append").save(output.toString());

        assertEquals(range(0, 250), ids(format, output));
        assertEquals(8, dataFiles(output, format).size());
    }

    @ParameterizedTest
    @ValueSource(strings = {"vortex", "parquet"})
    void unsafeKryoSerializationRoundTripsAMultiTaskWrite(String format) throws IOException {
        stop();
        start(Map.of(
                "spark.serializer", "org.apache.spark.serializer.KryoSerializer",
                "spark.kryo.unsafe", "true",
                "spark.kryo.registrationRequired", "false"));
        Path output = tempDir.resolve("kryo");
        spark.range(0, 400, 1, 4)
                .selectExpr("id", "CAST(id % 3 AS INT) AS p", "named_struct('s', CAST(id AS STRING)) AS n")
                .write()
                .format(format)
                .partitionBy("p")
                .save(output.toString());

        Dataset<Row> data = spark.read().format(format).load(output.toString());
        assertEquals(range(0, 400), ids(format, output));
        assertEquals(134L, data.filter("p = 0").count());
        assertEquals(
                List.of(RowFactory.create(7L, "7")),
                data.filter("n.s = '7'").select("id", "n.s").collectAsList());
        assertCommitted(output);
    }

    @ParameterizedTest(name = "{0}, {1}")
    @CsvSource({"vortex,static", "vortex,dynamic", "parquet,static", "parquet,dynamic"})
    void anOverwriteReplacesOnlyThePartitionsItsModeSelects(String format, String mode) {
        Path output = tempDir.resolve("save-overwrite");
        writeRange(format, output, 0, 10, true, "overwrite");
        spark.conf().set("spark.sql.sources.partitionOverwriteMode", mode);
        spark.range(100, 101)
                .selectExpr("id", "1 AS p")
                .write()
                .format(format)
                .partitionBy("p")
                .mode("overwrite")
                .save(output.toString());

        assertEquals(mode.equals("dynamic") ? List.of(0L, 2L, 4L, 6L, 8L, 100L) : List.of(100L), ids(format, output));
        assertCommitted(output);
    }

    @ParameterizedTest(name = "{0}, {1}")
    @CsvSource({"vortex,static", "vortex,dynamic", "parquet,static", "parquet,dynamic"})
    void insertOverwriteReplacesOnlyThePartitionsItsModeSelects(String format, String mode) {
        Path output = tempDir.resolve("sql-overwrite");
        spark.sql("CREATE TABLE commit_test (id BIGINT, p INT) USING " + format + " PARTITIONED BY (p) LOCATION '"
                + output + "'");
        try {
            spark.sql("INSERT INTO commit_test SELECT id, CAST(id % 2 AS INT) FROM range(10)");
            spark.conf().set("spark.sql.sources.partitionOverwriteMode", mode);
            spark.sql("INSERT OVERWRITE TABLE commit_test PARTITION (p) SELECT 100L, 1");

            List<Long> expected = mode.equals("dynamic") ? List.of(0L, 2L, 4L, 6L, 8L, 100L) : List.of(100L);
            assertEquals(expected, ids(format, output));
            assertEquals(
                    expected,
                    spark.table("commit_test").orderBy("id").collectAsList().stream()
                            .map(row -> row.getLong(0))
                            .toList());

            // A static partition spec replaces that partition alone in either mode.
            spark.sql("INSERT OVERWRITE TABLE commit_test PARTITION (p=1) SELECT 101L");
            assertEquals(
                    mode.equals("dynamic") ? List.of(0L, 2L, 4L, 6L, 8L, 101L) : List.of(101L), ids(format, output));
        } finally {
            spark.sql("DROP TABLE commit_test");
        }
    }

    private void writeRange(String format, Path output, long start, long end, boolean partitioned, String mode) {
        DataFrameWriter<Row> writer = spark.range(start, end, 1, 1)
                .selectExpr("id", "CAST(id % 2 AS INT) AS p")
                .write()
                .format(format)
                .mode(mode);
        if (partitioned) {
            writer = writer.partitionBy("p");
        }
        writer.save(output.toString());
    }

    private List<Long> ids(String format, Path output) {
        return spark.read().format(format).load(output.toString()).select("id").orderBy("id").collectAsList().stream()
                .map(row -> row.getLong(0))
                .toList();
    }

    private static List<Long> range(long start, long end) {
        return LongStream.range(start, end).boxed().toList();
    }

    private static List<Path> dataFiles(Path root, String format) throws IOException {
        if (!Files.exists(root)) {
            return List.of();
        }
        try (Stream<Path> paths = Files.walk(root)) {
            return paths.filter(Files::isRegularFile)
                    .filter(path -> path.getFileName().toString().endsWith("." + format))
                    .sorted()
                    .toList();
        }
    }

    private static void assertCommitted(Path output) {
        assertTrue(Files.exists(output.resolve("_SUCCESS")));
        assertFalse(Files.exists(output.resolve("_temporary")));
    }

    public static final class FirstTaskListener extends SparkListener {
        @Override
        public void onTaskEnd(SparkListenerTaskEnd event) {
            if (event.taskInfo().partitionId() == 0 && event.taskInfo().successful()) {
                firstTaskDone.countDown();
            }
        }
    }

    public static final class FailFirstTaskCommit extends SQLHadoopMapReduceCommitProtocol {
        public FailFirstTaskCommit(String jobId, String path, boolean dynamicPartitionOverwrite) {
            super(jobId, path, dynamicPartitionOverwrite);
        }

        @Override
        public TaskCommitMessage commitTask(TaskAttemptContext context) {
            TaskCommitMessage message = super.commitTask(context);
            if (context.getTaskAttemptID().getId() == 0) {
                INJECTED.incrementAndGet();
                throw new IllegalStateException("injected failure after the task committed its files");
            }
            return message;
        }
    }

    /** Fails closing the first attempt's file in partition {@code p=1}, after {@code p=0} has closed. */
    public static final class FailCloseFileSystem extends LocalFileSystem {
        @Override
        public FSDataOutputStream create(
                org.apache.hadoop.fs.Path path,
                FsPermission permission,
                boolean overwrite,
                int bufferSize,
                short replication,
                long blockSize,
                Progressable progress)
                throws IOException {
            FSDataOutputStream stream =
                    super.create(path, permission, overwrite, bufferSize, replication, blockSize, progress);
            TaskContext context = TaskContext.get();
            if (context == null || !path.toString().contains("/p=1/")) {
                return stream;
            }
            CLOSE_FAILURE_ATTEMPTS.add(context.attemptNumber());
            if (context.attemptNumber() != 0) {
                return stream;
            }
            return new FSDataOutputStream(
                    new FilterOutputStream(stream) {
                        @Override
                        public void close() throws IOException {
                            super.close();
                            if (INJECTED.compareAndSet(0, 1)) {
                                throw new IOException("injected close failure");
                            }
                        }
                    },
                    null);
        }
    }
}
