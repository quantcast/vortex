// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicLong;
import org.apache.spark.scheduler.SparkListener;
import org.apache.spark.scheduler.SparkListenerTaskEnd;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.execution.SparkPlan;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
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
    void filesOpenedOneAtATimeReportInputBytes(@TempDir Path tempDir) throws Exception {
        // A fresh table: Vortex caches what it reads of a file, which would hide reads the other tests already made.
        String path = tempDir.resolve("one-at-a-time").toString();
        spark.range(0, 16384, 1, 4).write().format("vortex").save(path);
        Dataset<Row> read =
                spark.read().format(OneFileAtATimeSource.class.getName()).load(path);
        verify(read, 16384);
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
