// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSInputStream;
import org.apache.hadoop.fs.LocalFileSystem;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.ArrayType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
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
 *
 * <p>{@code arr} is an {@code array<struct<a, w0..w7, g: array<struct<h, x0..x7>>>>} with nulls and empty arrays at
 * every level, written with and without {@code vortex.listLayout}, and as Parquet to compare against.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexNestedSchemaPruningIoTest {
    private static final int ROWS = 16384;
    private static final String PRUNING = "spark.sql.optimizer.nestedSchemaPruning.enabled";
    static final AtomicLong BYTES_READ = new AtomicLong();

    private SparkSession spark;
    private Dataset<Row> table;
    private Dataset<Row> listLayoutArrays;
    private Dataset<Row> flatArrays;
    private Dataset<Row> mixedArrays;
    private Dataset<Row> parquetArrays;
    private List<String> vortexArrayPaths;
    private String parquetArraysPath;

    @BeforeAll
    public void setUp(@TempDir Path tempDir) {
        spark = SparkSession.builder()
                .appName("VortexNestedSchemaPruningIoTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.sql.adaptive.enabled", "false")
                .config(PRUNING, "true")
                // Spark prunes below two arrays only through an index, as in `arr.g[0].h`, which ANSI mode fails on an
                // empty array.
                .config("spark.sql.ansi.enabled", "false")
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

        Dataset<Row> arrays = spark.range(0, ROWS, 1, 2).selectExpr("id", ARRAYS + " AS arr");
        String listLayoutPath = tempDir.resolve("list_layout").toString();
        arrays.write().format("vortex").option("vortex.listLayout", "true").save(listLayoutPath);
        String flatPath = tempDir.resolve("flat").toString();
        arrays.write().format("vortex").save(flatPath);
        String mixedPath = tempDir.resolve("mixed").toString();
        arrays.where("id < " + ROWS / 2)
                .write()
                .format("vortex")
                .option("vortex.listLayout", "true")
                .save(mixedPath);
        arrays.where("id >= " + ROWS / 2)
                .write()
                .format("vortex")
                .mode(SaveMode.Append)
                .save(mixedPath);
        String parquetPath = tempDir.resolve("parquet").toString();
        arrays.write().parquet(parquetPath);
        vortexArrayPaths = List.of(listLayoutPath, flatPath, mixedPath);
        parquetArraysPath = parquetPath;
        listLayoutArrays = spark.read().format("vortex").load(listLayoutPath);
        flatArrays = spark.read().format("vortex").load(flatPath);
        mixedArrays = spark.read().format("vortex").load(mixedPath);
        parquetArrays = spark.read().parquet(parquetPath);
    }

    private static final String ARRAYS = arraysSql();

    /** {@code arr.g[0].h}, which Spark reads as {@code arr: array<struct<g: array<struct<h>>>>}. */
    private static final Column INNER_H = functions.col("arr.g").getItem(0).getField("h");

    /** Lengths 0 to 4 and 0 to 3, with null arrays, null elements and empty arrays at both levels. */
    private static String arraysSql() {
        StringBuilder inner = new StringBuilder("named_struct('h', id * 1000 + i * 10 + j");
        StringBuilder outer = new StringBuilder("named_struct('a', id * 10 + i");
        for (int k = 0; k < 8; k++) {
            inner.append(String.format(Locale.ROOT, ", 'x%d', sha2(concat_ws(':', 'x%d', id, i, j), 256)", k, k));
            outer.append(String.format(Locale.ROOT, ", 'w%d', sha2(concat_ws(':', 'w%d', id, i), 256)", k, k));
        }
        inner.append(")");
        String g = "CASE WHEN (id + i) % 3 = 0 THEN NULL ELSE transform(filter(sequence(0, 2), j -> j < (id + i) % 4),"
                + " j -> CASE WHEN (id + i + j) % 5 = 0 THEN NULL ELSE " + inner + " END) END";
        outer.append(", 'g', ").append(g).append(")");
        return "CASE WHEN id % 7 = 0 THEN NULL ELSE transform(filter(sequence(0, 3), i -> i < id % 5),"
                + " i -> CASE WHEN (id + i) % 6 = 0 THEN NULL ELSE " + outer + " END) END";
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

    @Test
    @DisplayName("A field of the structs in an array is read without their other fields in the list layout")
    public void arrayElementFieldReadsOnlyThatField() {
        long whole = bytesRead(listLayoutArrays.select(functions.col("arr")));
        long field = bytesRead(listLayoutArrays.select(functions.col("arr.a")));
        report("list layout, arr", whole);
        report("list layout, arr.a", field);
        report("flat, arr", bytesRead(flatArrays.select(functions.col("arr"))));
        report("flat, arr.a", bytesRead(flatArrays.select(functions.col("arr.a"))));
        assertTrue(field < whole / 10, "arr.a read " + field + " bytes; arr read " + whole);
        assertMatchesParquet(arrays -> arrays.select(functions.col("id"), functions.col("arr.a")));
        assertMatchesParquet(arrays -> arrays.select(functions.col("id"), functions.col("arr")));
    }

    @Test
    @DisplayName("A field of the structs in an array inside array elements is read without their other fields")
    public void nestedArrayElementFieldReadsOnlyThatField() {
        long whole = bytesRead(listLayoutArrays.select(functions.col("arr")));
        long inner = bytesRead(listLayoutArrays.select(functions.col("arr.g")));
        long field = bytesRead(listLayoutArrays.select(INNER_H));
        report("list layout, arr.g", inner);
        report("list layout, arr.g[0].h", field);
        report("flat, arr.g", bytesRead(flatArrays.select(functions.col("arr.g"))));
        report("flat, arr.g[0].h", bytesRead(flatArrays.select(INNER_H)));
        // Spark prunes through `explode` only to the outer element fields, so this reads all of `g`.
        Function<Dataset<Row>, Dataset<Row>> exploded = arrays -> arrays.select(
                        functions.col("id"),
                        functions.explode(functions.col("arr")).as("e"))
                .select(functions.col("id"), functions.col("e.g.h"));
        report("list layout, explode(arr) then e.g.h", bytesRead(exploded.apply(listLayoutArrays)));
        assertTrue(field < whole / 10, "arr.g[0].h read " + field + " bytes; arr read " + whole);
        assertTrue(field < inner / 10, "arr.g[0].h read " + field + " bytes; arr.g read " + inner);
        assertMatchesParquet(arrays -> arrays.select(functions.col("id"), INNER_H));
        assertMatchesParquet(arrays -> arrays.select(functions.col("arr.g"), functions.col("id")));
        assertMatchesParquet(arrays -> arrays.select(
                functions.col("id"),
                functions.col("arr").getItem(1).getField("g").getField("h")));
        assertMatchesParquet(arrays -> arrays.select(functions.col("id"), INNER_H, functions.col("arr.a")));
        assertMatchesParquet(arrays -> arrays.select(
                functions.col("id"),
                functions.col("arr.g").getItem(1).getField("x3"),
                functions.col("arr.w5"),
                functions.col("arr.a")));
    }

    @Test
    @DisplayName("Filters combined with array element projections match Parquet")
    public void filtersWithArrayElementProjection() {
        Function<Dataset<Row>, Dataset<Row>> query = arrays -> arrays.where(functions
                        .col("id")
                        .mod(3)
                        .equalTo(0)
                        .and(functions.array_max(functions.col("arr.a")).gt(ROWS)))
                .select(functions.col("id"), INNER_H);
        long whole = bytesRead(listLayoutArrays.select(functions.col("arr")));
        long pruned = bytesRead(query.apply(listLayoutArrays));
        assertTrue(pruned < whole / 10, "the filtered read read " + pruned + " bytes; arr read " + whole);
        assertTrue(assertMatchesParquet(query).size() > 100);
        assertMatchesParquet(arrays ->
                arrays.where(functions.col("arr").isNull()).select(functions.col("id"), functions.col("arr.a")));
        assertMatchesParquet(arrays ->
                arrays.where(functions.size(functions.col("arr")).equalTo(0)).select(functions.col("id"), INNER_H));
    }

    @Test
    @DisplayName("Split reads of array element projections match Parquet")
    public void splitReadsOfArrayElementProjection() {
        spark.conf().set("spark.sql.files.maxPartitionBytes", "65537");
        spark.conf().set("spark.sql.files.openCostInBytes", "0");
        try {
            Dataset<Row> split = listLayoutArrays.select(functions.col("id"), INNER_H);
            assertTrue(
                    split.rdd().getNumPartitions() > 4,
                    "split into " + split.rdd().getNumPartitions());
            assertMatchesParquet(arrays -> arrays.select(functions.col("id"), INNER_H));
            assertMatchesParquet(arrays -> arrays.where(functions.col("id").gt(ROWS / 3))
                    .select(
                            functions.col("id"),
                            functions.col("arr.a"),
                            functions.col("arr.g").getItem(0).getField("x1")));
        } finally {
            spark.conf().unset("spark.sql.files.maxPartitionBytes");
            spark.conf().unset("spark.sql.files.openCostInBytes");
        }
    }

    @Test
    @DisplayName("Fields of array elements a file does not carry read as null, as from Parquet")
    public void absentArrayElementFieldsReadAsNull() {
        StructType schema = withField(parquetArrays.schema(), "z");
        List<Function<Dataset<Row>, Dataset<Row>>> queries = List.of(
                arrays -> arrays.select(functions.col("id"), functions.col("arr")),
                arrays -> arrays.select(functions.col("id"), functions.col("arr.z"), functions.col("arr.a")),
                arrays -> arrays.select(
                        functions.col("id"), functions.col("arr.g").getItem(0).getField("z"), INNER_H));
        for (Function<Dataset<Row>, Dataset<Row>> query : queries) {
            List<Row> expected = query.apply(spark.read().schema(schema).parquet(parquetArraysPath))
                    .orderBy("id")
                    .collectAsList();
            for (String path : vortexArrayPaths) {
                Dataset<Row> arrays =
                        spark.read().schema(schema).format("vortex").load(path);
                assertEquals(expected, assertMatchesUnpruned(query.apply(arrays)), path);
            }
        }
    }

    /** {@code schema} with a nullable long {@code name} added to the elements of {@code arr} and of {@code arr.g}. */
    private static StructType withField(StructType schema, String name) {
        StructField arr = schema.apply("arr");
        ArrayType arrType = (ArrayType) arr.dataType();
        StructType outer = (StructType) arrType.elementType();
        StructField g = outer.apply("g");
        ArrayType gType = (ArrayType) g.dataType();
        StructType inner = ((StructType) gType.elementType()).add(name, DataTypes.LongType);
        StructField[] outerFields = outer.fields().clone();
        outerFields[outer.fieldIndex("g")] =
                new StructField("g", new ArrayType(inner, gType.containsNull()), g.nullable(), g.metadata());
        StructType widened = new StructType(outerFields).add(name, DataTypes.LongType);
        StructField[] fields = schema.fields().clone();
        fields[schema.fieldIndex("arr")] =
                new StructField("arr", new ArrayType(widened, arrType.containsNull()), arr.nullable(), arr.metadata());
        return new StructType(fields);
    }

    /**
     * Asserts {@code query} returns the same rows from the list layout, flat and mixed tables, with and without nested
     * schema pruning, as from Parquet, and returns them.
     */
    private List<Row> assertMatchesParquet(Function<Dataset<Row>, Dataset<Row>> query) {
        List<Row> expected = query.apply(parquetArrays).orderBy("id").collectAsList();
        for (Dataset<Row> arrays : List.of(listLayoutArrays, flatArrays, mixedArrays)) {
            assertEquals(expected, assertMatchesUnpruned(query.apply(arrays)));
        }
        return expected;
    }

    private static void report(String read, long bytes) {
        System.out.printf(Locale.ROOT, "bytes read: %s: %d%n", read, bytes);
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
