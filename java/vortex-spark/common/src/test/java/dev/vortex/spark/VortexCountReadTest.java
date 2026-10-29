// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSInputStream;
import org.apache.hadoop.fs.LocalFileSystem;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** Counting rows reads row counts and filter columns, not every column of the table. */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexCountReadTest {
    private static final int ROWS = 4096;
    private static final AtomicLong BYTES_READ = new AtomicLong();

    private SparkSession spark;
    private String table;

    @BeforeAll
    void setUp(@TempDir Path tempDir) {
        spark = SparkSession.builder()
                .appName("VortexCountReadTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.ui.enabled", "false")
                .config("spark.sql.adaptive.enabled", "false")
                .config("spark.hadoop.fs.file.impl", CountingLocalFileSystem.class.getName())
                .config("spark.hadoop.fs.file.impl.disable.cache", "true")
                .getOrCreate();
        List<Column> columns = new ArrayList<>();
        columns.add(functions.col("id"));
        for (int i = 0; i < 8; i++) {
            columns.add(functions
                    .sha2(functions.concat_ws(":", functions.lit("c" + i), functions.col("id")), 256)
                    .alias("c" + i));
        }
        table = tempDir.resolve("wide").toString();
        spark.range(0, ROWS, 1, 2)
                .select(columns.toArray(new Column[0]))
                .write()
                .format("vortex")
                .save(table);
    }

    @AfterAll
    void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @ParameterizedTest(name = "v1={0}")
    @ValueSource(booleans = {false, true})
    void countStarReadsNoColumns(boolean v1) {
        withV1(v1, () -> {
            long full = bytesRead(spark.read().format("vortex").load(table).selectExpr("bit_xor(xxhash64(*))"));
            Dataset<Row> count = spark.read()
                    .format("vortex")
                    .option("vortex.aggregatePushdown", "false")
                    .load(table)
                    .selectExpr("count(*)");

            String plan = scanPlan(count, v1);
            assertTrue(plan.contains("ReadSchema: struct<>"), plan);
            long counted = bytesRead(count);
            assertEquals((long) ROWS, count.first().getLong(0));
            assertTrue(counted < full / 10, "count(*) read " + counted + " bytes, a full scan " + full);
        });
    }

    @ParameterizedTest(name = "v1={0}")
    @ValueSource(booleans = {false, true})
    void aFilteredCountReadsOnlyTheFilterColumn(boolean v1) {
        withV1(v1, () -> {
            long full = bytesRead(spark.read().format("vortex").load(table).selectExpr("bit_xor(xxhash64(*))"));
            Dataset<Row> count = spark.read()
                    .format("vortex")
                    .load(table)
                    .filter("id >= 100")
                    .selectExpr("count(*)");

            scanPlan(count, v1);
            long counted = bytesRead(count);
            assertEquals((long) ROWS - 100, count.first().getLong(0));
            assertTrue(counted < full / 10, "a filtered count read " + counted + " bytes, a full scan " + full);
        });
    }

    private void withV1(boolean v1, Runnable body) {
        String previous = spark.conf().get("spark.sql.sources.useV1SourceList");
        spark.conf().set("spark.sql.sources.useV1SourceList", v1 ? "vortex" : "");
        try {
            body.run();
        } finally {
            spark.conf().set("spark.sql.sources.useV1SourceList", previous);
        }
    }

    /** Returns the executed plan after checking it scans through the source the test asked for. */
    private static String scanPlan(Dataset<Row> query, boolean v1) {
        String plan = query.queryExecution().executedPlan().toString();
        // `FileScan vortex` is the V1 node, `BatchScan` the V2 one.
        assertTrue(plan.contains(v1 ? "FileScan vortex" : "BatchScan"), plan);
        assertFalse(plan.contains(v1 ? "BatchScan" : "FileScan"), plan);
        return plan;
    }

    private static long bytesRead(Dataset<Row> query) {
        BYTES_READ.set(0);
        query.collectAsList();
        return BYTES_READ.get();
    }

    // Native reads call back into Hadoop on worker threads, where Spark's task input metrics do not see them.
    public static final class CountingLocalFileSystem extends LocalFileSystem {
        @Override
        public FSDataInputStream open(org.apache.hadoop.fs.Path path, int bufferSize) throws IOException {
            return new FSDataInputStream(new CountingInputStream(super.open(path, bufferSize)));
        }
    }

    private static final class CountingInputStream extends FSInputStream {
        private final FSDataInputStream input;

        private CountingInputStream(FSDataInputStream input) {
            this.input = input;
        }

        @Override
        public void seek(long position) throws IOException {
            input.seek(position);
        }

        @Override
        public long getPos() throws IOException {
            return input.getPos();
        }

        @Override
        public boolean seekToNewSource(long target) throws IOException {
            return input.seekToNewSource(target);
        }

        @Override
        public int read() throws IOException {
            int value = input.read();
            if (value >= 0) {
                BYTES_READ.incrementAndGet();
            }
            return value;
        }

        @Override
        public int read(byte[] buffer, int offset, int length) throws IOException {
            int count = input.read(buffer, offset, length);
            if (count > 0) {
                BYTES_READ.addAndGet(count);
            }
            return count;
        }

        @Override
        public void close() throws IOException {
            input.close();
        }
    }
}
