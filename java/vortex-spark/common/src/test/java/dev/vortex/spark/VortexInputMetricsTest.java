// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.spark.io.VortexIo;
import dev.vortex.spark.read.VortexPartitionReaderFactory;
import dev.vortex.spark.read.VortexReadMetrics;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Stream;
import org.apache.hadoop.conf.Configuration;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.paths.SparkPath;
import org.apache.spark.scheduler.SparkListener;
import org.apache.spark.scheduler.SparkListenerTaskEnd;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.FileSourceOptions;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.execution.SparkPlan;
import org.apache.spark.sql.execution.datasources.PartitionedFile;
import org.apache.spark.sql.sources.Filter;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.vectorized.ColumnarBatch;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/** Checks the final task metrics Spark publishes, after its own Hadoop metric update. */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
final class VortexInputMetricsTest {
    private static final StructType SCHEMA = new StructType().add("id", DataTypes.LongType);
    private SparkSession spark;
    private String table;
    private final InputBytesListener listener = new InputBytesListener();

    @BeforeAll
    void setUp(@TempDir Path tempDir) {
        spark = SparkSession.builder()
                .appName("VortexInputMetricsTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.ui.enabled", "false")
                .config("spark.sql.adaptive.enabled", "false")
                .config("spark.sql.sources.useV1SourceList", "")
                .config("spark.sql.files.openCostInBytes", "0")
                .config("spark.sql.files.minPartitionNum", "1")
                .config(
                        "spark.hadoop.fs.file.impl",
                        VortexNestedSchemaPruningIoTest.CountingLocalFileSystem.class.getName())
                .config("spark.hadoop.fs.file.impl.disable.cache", "true")
                .getOrCreate();
        table = tempDir.resolve("table").toString();
        spark.range(0, 16384, 1, 4).write().format("vortex").save(table);
        spark.sparkContext().addSparkListener(listener);
    }

    @AfterAll
    void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @Test
    void multipleFilesAccumulateWithoutDoubleCounting() throws Exception {
        Dataset<Row> read = read();
        assertEquals(1, read.rdd().getNumPartitions(), "several files must share one partition");
        verify(read, 16384);
    }

    @Test
    void coalescedTaskAccumulatesSeveralInputPartitions() throws Exception {
        spark.conf().set("spark.sql.files.maxPartitionBytes", "1024");
        try {
            Dataset<Row> read = read();
            assertTrue(read.rdd().getNumPartitions() > 1);
            verify(read.coalesce(1), 16384);
        } finally {
            spark.conf().unset("spark.sql.files.maxPartitionBytes");
        }
    }

    @Test
    void separateScansKeepIndependentCustomTotalsInOneTask() throws Exception {
        verify(read().filter("id % 3 = 0").union(read().filter("id % 3 != 0")).coalesce(1), 16384);
    }

    @Test
    void emptyResultsStillReportFooterAndFilterReads() throws Exception {
        verify(read().filter("id % 2 = -1"), 0);
    }

    @Test
    void pushedCountReportsFooterBytes() throws Exception {
        Dataset<Row> count = read().selectExpr("count(*)");
        assertTrue(count.queryExecution().executedPlan().toString().contains("PushedAggregation: [COUNT(*)]"));
        verify(count, 1);
    }

    @Test
    void earlyTerminationReportsInputBytes() throws Exception {
        verify(read().limit(1), 1);
    }

    @Test
    void filesReadOneByOneThroughTheFactoryReportTheirBytes() throws Exception {
        // Its own files: Vortex caches what it reads of a file, which would hide the footer reads other tests count.
        String own = table + "-one-by-one";
        spark.range(0, 16384, 1, 4).write().format("vortex").save(own);
        List<String> files;
        try (Stream<Path> listed = Files.list(Path.of(own))) {
            files = listed.map(Path::toString)
                    .filter(f -> f.endsWith(".vortex"))
                    .sorted()
                    .toList();
        }
        assertTrue(files.size() > 1, "the table must have several files");
        spark.sparkContext().listenerBus().waitUntilEmpty(10000);
        listener.bytes.set(0);
        VortexNestedSchemaPruningIoTest.BYTES_READ.set(0);

        List<long[]> results = JavaSparkContext.fromSparkContext(spark.sparkContext())
                .parallelize(List.of(0), 1)
                .map(ignored -> readOneByOne(files))
                .collect();
        spark.sparkContext().listenerBus().waitUntilEmpty(10000);

        long source = VortexNestedSchemaPruningIoTest.BYTES_READ.get();
        assertEquals(16384, results.get(0)[0]);
        assertTrue(source > 0, "the reads must read source bytes");
        assertEquals(source, results.get(0)[1], "the custom metric must hold every file's bytes");
        assertEquals(source, listener.bytes.get(), "task metrics must include each source byte once");
    }

    /** Reads each file with its own reader in this task, as a caller choosing its own files would: {rows, bytes}. */
    private static long[] readOneByOne(List<String> files) {
        VortexOptions options = VortexOptions.of(Map.of());
        Configuration conf = new Configuration();
        conf.set("fs.file.impl", VortexNestedSchemaPruningIoTest.CountingLocalFileSystem.class.getName());
        conf.set("fs.file.impl.disable.cache", "true");
        VortexPartitionReaderFactory factory = new VortexPartitionReaderFactory(
                new FileSourceOptions(scala.collection.immutable.Map$.MODULE$.empty()),
                VortexIo.create(options, conf),
                options,
                SCHEMA,
                SCHEMA,
                new StructType(),
                new Filter[0],
                false);
        VortexReadMetrics metrics = VortexReadMetrics.forTask(factory);
        long rows = 0;
        for (String path : files) {
            long length = Path.of(path).toFile().length();
            PartitionedFile file = new PartitionedFile(
                    InternalRow.empty(),
                    SparkPath.fromPathString("file://" + path),
                    0,
                    length,
                    new String[0],
                    0,
                    length,
                    PartitionedFile.apply$default$8());
            try (PartitionReader<ColumnarBatch> reader = factory.buildColumnarReader(file, metrics)) {
                while (reader.next()) {
                    rows += reader.get().numRows();
                }
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
            metrics.publish();
        }
        return new long[] {rows, metrics.currentMetricsValues()[0].value()};
    }

    private Dataset<Row> read() {
        return spark.read()
                .schema(SCHEMA)
                .format("vortex")
                .option("vortex.stats.rowCount", "false")
                .load(table);
    }

    private void verify(Dataset<Row> query, int rows) throws Exception {
        SparkPlan plan = query.queryExecution().executedPlan();
        assertTrue(plan.toString().contains("BatchScan"), plan.toString());
        spark.sparkContext().listenerBus().waitUntilEmpty(10000);
        listener.bytes.set(0);
        VortexNestedSchemaPruningIoTest.BYTES_READ.set(0);
        assertEquals(rows, query.collectAsList().size());
        spark.sparkContext().listenerBus().waitUntilEmpty(10000);
        long source = VortexNestedSchemaPruningIoTest.BYTES_READ.get();
        assertTrue(source > 0, "the scan must read source bytes");
        assertEquals(source, listener.bytes.get(), "task metrics must include each source byte once");
        assertEquals(source, customBytes(plan), "custom metrics must retain bytes from every file");
    }

    private static long customBytes(SparkPlan plan) {
        long bytes = plan.metrics().contains("vortexBytesRead")
                ? plan.metrics().apply("vortexBytesRead").value()
                : 0;
        var children = plan.children().iterator();
        while (children.hasNext()) {
            bytes += customBytes(children.next());
        }
        return bytes;
    }

    private static final class InputBytesListener extends SparkListener {
        private final AtomicLong bytes = new AtomicLong();

        @Override
        public void onTaskEnd(SparkListenerTaskEnd event) {
            bytes.addAndGet(event.taskMetrics().inputMetrics().bytesRead());
        }
    }
}
