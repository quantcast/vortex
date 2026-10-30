// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
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
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/**
 * Tests that reading a field below a nullable struct reads only that field from storage.
 *
 * <p>The nullable structs {@code s = {a, t, ...}}, {@code t = {x, ...}} and {@code x = {q, ...}} each hold eight more
 * wide string fields, so a read that pulls in the struct around a leaf shows up in the bytes read.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexNestedSchemaPruningIoTest {
    private static final int ROWS = 16384;
    private static final String PRUNING = "spark.sql.optimizer.nestedSchemaPruning.enabled";
    private static final AtomicLong BYTES_READ = new AtomicLong();

    private SparkSession spark;
    private Dataset<Row> table;

    @BeforeAll
    public void setUp(@TempDir Path tempDir) {
        spark = SparkSession.builder()
                .appName("VortexNestedSchemaPruningIoTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.sql.adaptive.enabled", "false")
                .config(PRUNING, "true")
                .config("spark.ui.enabled", "false")
                // The native scan reads through Hadoop on its own threads, outside Spark's input metrics.
                .config("spark.hadoop.fs.file.impl", CountingLocalFileSystem.class.getName())
                .config("spark.hadoop.fs.file.impl.disable.cache", "true")
                .getOrCreate();

        Column x = nullUnlessMultiple(
                wideStruct("q", functions.col("id").multiply(2).plus(300), "x"), 11);
        Column t = nullUnlessMultiple(wideStruct("x", x, "t"), 7);
        Column s = nullUnlessMultiple(
                wideStruct("a", functions.col("id").multiply(2).plus(100), "s").withField("t", t), 5);
        String path = tempDir.resolve("nested").toString();
        spark.range(0, ROWS, 1, 2)
                .select(functions.col("id"), s.alias("s"))
                .write()
                .format("vortex")
                .save(path);
        table = spark.read().format("vortex").load(path);
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @Test
    @DisplayName("A field of a nullable struct is read without the struct's other fields")
    public void nullableStructFieldReadsOnlyThatField() {
        long whole = bytesRead(table.select(functions.col("s")));
        long field = bytesRead(table.select(functions.col("s.a")));
        assertTrue(field < whole / 10, "s.a read " + field + " bytes; s read " + whole);
        assertMatchesUnpruned(table.select(functions.col("id"), functions.col("s.a")));
    }

    @Test
    @DisplayName("A field two structs below a nullable struct is read without the structs' other fields")
    public void deepNullableStructFieldReadsOnlyThatField() {
        long whole = bytesRead(table.select(functions.col("s.t.x")));
        long field = bytesRead(table.select(functions.col("s.t.x.q")));
        assertTrue(field < whole / 10, "s.t.x.q read " + field + " bytes; s.t.x read " + whole);
        assertMatchesUnpruned(table.select(functions.col("id"), functions.col("s.t.x.q")));
    }

    @Test
    @DisplayName("A null check on a pruned struct sees the struct's own nulls")
    public void nullCheckOnPrunedStruct() {
        // Spark prunes `s` to the fields the projection uses and evaluates the null check on what the scan returns.
        Dataset<Row> nulls =
                table.where(functions.col("s").isNull()).select(functions.col("id"), functions.col("s.t.x.q"));
        Dataset<Row> nonNulls =
                table.where(functions.col("s").isNotNull()).select(functions.col("id"), functions.col("s.a"));
        assertEquals((ROWS + 4) / 5, assertMatchesUnpruned(nulls).size());
        assertEquals(ROWS - (ROWS + 4) / 5, assertMatchesUnpruned(nonNulls).size());
        assertMatchesUnpruned(
                table.select(functions.col("id"), functions.col("s.t").isNull(), functions.col("s.t.x.q")));
        long whole = bytesRead(table.select(functions.col("s")));
        long pruned = bytesRead(nulls);
        assertTrue(pruned < whole / 10, "the null check read " + pruned + " bytes; s read " + whole);
    }

    @Test
    @DisplayName("A filter on one nested field and a projection of another match the unpruned read")
    public void filterAndProjectionOnDifferentNestedFields() {
        Dataset<Row> query = table.where(functions.col("s.t.x.q").gt(ROWS))
                .select(functions.col("id"), functions.col("s.a"), functions.col("s.t.x.q"));
        long pruned = bytesRead(query);
        long whole = bytesRead(table.select(functions.col("s")));
        assertTrue(pruned < whole / 10, "the filtered read read " + pruned + " bytes; s read " + whole);
        assertMatchesUnpruned(query);
    }

    /** Asserts {@code query} returns the same rows with nested schema pruning off, and returns them. */
    private List<Row> assertMatchesUnpruned(Dataset<Row> query) {
        List<Row> pruned = query.orderBy("id").collectAsList();
        spark.conf().set(PRUNING, "false");
        try {
            assertEquals(query.orderBy("id").collectAsList(), pruned);
        } finally {
            spark.conf().set(PRUNING, "true");
        }
        return pruned;
    }

    /** Hashes every selected value so the whole result is decoded, and returns the bytes read from storage. */
    private static long bytesRead(Dataset<Row> query) {
        BYTES_READ.set(0);
        query.selectExpr("bit_xor(xxhash64(*))").collectAsList();
        return BYTES_READ.get();
    }

    private static Column wideStruct(String name, Column value, String prefix) {
        List<Column> fields = new ArrayList<>();
        fields.add(value.alias(name));
        for (int i = 0; i < 8; i++) {
            fields.add(functions
                    .sha2(functions.concat_ws(":", functions.lit(prefix + i), functions.col("id")), 256)
                    .alias(prefix + i));
        }
        return functions.struct(fields.toArray(new Column[0]));
    }

    private static Column nullUnlessMultiple(Column value, int divisor) {
        return functions.when(functions.col("id").mod(divisor).notEqual(0), value);
    }

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
        public int read(byte[] bytes, int offset, int length) throws IOException {
            int count = input.read(bytes, offset, length);
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
