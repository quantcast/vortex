// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.function.Function;
import java.util.stream.Stream;
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
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Tests that reading a leaf inside lists of structs nested in structs reads only that leaf from storage, through both
 * Spark read paths, and returns what Parquet returns.
 *
 * <p>The table nests lists the way wide event schemas do, one to four lists deep under structs:
 *
 * <pre>
 * id
 * r: struct
 *   q: struct
 *     imp: array&lt;struct
 *       id, w0..w7
 *       banner: struct
 *         w, h, b0..b7
 *         format: array&lt;struct&lt;w, h, f0..f7&gt;&gt;
 *       tags: array&lt;string&gt;
 *       seats: array&lt;struct
 *         name, s0..s7
 *         deals: array&lt;struct
 *           price, d0..d7
 *           segs: array&lt;struct&lt;v, g0..g7&gt;&gt;
 * </pre>
 *
 * <p>Every struct carries eight wide string siblings, so a read that pulls in a leaf's siblings shows up in the bytes
 * read. Each list level has null lists, null elements and empty lists. The table is written with
 * {@code vortex.listLayout}, without it, and as Parquet to compare against.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexListElementPruningIoTest {
    private static final int ROWS = 8192;
    private static final String PRUNING = "spark.sql.optimizer.nestedSchemaPruning.enabled";
    private static final String USE_V1 = "spark.sql.sources.useV1SourceList";

    private SparkSession spark;
    private String listLayoutPath;
    private String flatPath;
    private String parquetPath;

    @BeforeAll
    public void setUp(@TempDir Path tempDir) {
        spark = SparkSession.builder()
                .appName("VortexListElementPruningIoTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.sql.adaptive.enabled", "false")
                .config(PRUNING, "true")
                // Spark prunes below two arrays only through an index, which ANSI mode fails on an empty array.
                .config("spark.sql.ansi.enabled", "false")
                .config("spark.ui.enabled", "false")
                .config(
                        "spark.hadoop.fs.file.impl",
                        VortexNestedSchemaPruningIoTest.CountingLocalFileSystem.class.getName())
                .config("spark.hadoop.fs.file.impl.disable.cache", "true")
                .getOrCreate();

        Dataset<Row> rows = spark.range(0, ROWS, 1, 2)
                .selectExpr("id", "named_struct('q', named_struct('imp', " + imps() + ")) AS r");
        listLayoutPath = tempDir.resolve("list_layout").toString();
        rows.write().format("vortex").option("vortex.listLayout", "true").save(listLayoutPath);
        flatPath = tempDir.resolve("flat").toString();
        rows.write().format("vortex").save(flatPath);
        parquetPath = tempDir.resolve("parquet").toString();
        rows.write().parquet(parquetPath);
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
            SparkSession.clearActiveSession();
            SparkSession.clearDefaultSession();
        }
    }

    /** Eight wide string fields named {@code prefix0..7}, distinct per row and position. */
    private static String wide(String prefix, String key) {
        StringBuilder out = new StringBuilder();
        for (int k = 0; k < 8; k++) {
            out.append(String.format(
                    Locale.ROOT, ", '%s%d', sha2(concat_ws(':', '%s%d', %s), 256)", prefix, k, prefix, k, key));
        }
        return out.toString();
    }

    /**
     * A list of {@code element} over {@code sequence(0, max - 1)}, null when {@code seed % nullEvery == 0}, with a
     * length of {@code seed % (max + 1)}, so some lists are empty, and a null element when {@code (seed + i) % 7 == 0}.
     */
    private static String list(String var, String seed, int max, int nullEvery, String element) {
        return String.format(
                Locale.ROOT,
                "CASE WHEN (%s) %% %d = 0 THEN NULL ELSE transform(filter(sequence(0, %d), %s -> %s < (%s) %% %d),"
                        + " %s -> CASE WHEN (%s + %s) %% 7 = 0 THEN NULL ELSE %s END) END",
                seed,
                nullEvery,
                max - 1,
                var,
                var,
                seed,
                max + 1,
                var,
                seed,
                var,
                element);
    }

    private static String imps() {
        String format = list(
                "k",
                "id + i + 3",
                3,
                5,
                "named_struct('w', id + i * 10 + k, 'h', id + k" + wide("f", "id, i, k") + ")");
        String banner = "CASE WHEN (id + i) % 6 = 0 THEN NULL ELSE named_struct('w', id * 3 + i, 'h', id + i"
                + wide("b", "id, i") + ", 'format', " + format + ") END";
        String tags = list("t", "id + i + 1", 4, 9, "concat('tag', id % 13, '-', t)");
        String segs = list(
                "g",
                "id + i + c + d + 2",
                3,
                6,
                "named_struct('v', id + i + c + d + g" + wide("g", "id, i, c, d, g") + ")");
        String deals = list(
                "d",
                "id + i + c + 1",
                2,
                4,
                "named_struct('price', id * 7 + i + c + d" + wide("d", "id, i, c, d") + ", 'segs', " + segs + ")");
        String seats = list(
                "c",
                "id + i",
                3,
                8,
                "named_struct('name', concat('seat', c)" + wide("s", "id, i, c") + ", 'deals', " + deals + ")");
        String imp = "named_struct('id', id * 10 + i" + wide("w", "id, i") + ", 'banner', " + banner + ", 'tags', "
                + tags + ", 'seats', " + seats + ")";
        return list("i", "id", 4, 11, imp);
    }

    private static final Column IMP = functions.col("r.q.imp");

    /**
     * Leaves at one to four lists deep, each with the column read whole for comparison: the leaf's list, or the
     * innermost list around it.
     */
    static Stream<Arguments> leaves() {
        return Stream.of(
                Arguments.of("r.q.imp.id (1 list)", IMP.getField("id"), IMP),
                Arguments.of(
                        "r.q.imp.banner.w (1 list, struct below)",
                        IMP.getField("banner").getField("w"),
                        IMP),
                Arguments.of(
                        "r.q.imp[0].banner.format.w (2 lists)",
                        IMP.getItem(0).getField("banner").getField("format").getField("w"),
                        IMP.getItem(0).getField("banner").getField("format")),
                Arguments.of("r.q.imp.tags (list of strings in a list)", IMP.getField("tags"), IMP),
                Arguments.of(
                        "r.q.imp[0].seats.deals[0].price (3 lists)",
                        IMP.getItem(0)
                                .getField("seats")
                                .getField("deals")
                                .getItem(0)
                                .getField("price"),
                        IMP.getItem(0).getField("seats")),
                Arguments.of(
                        "r.q.imp[0].seats[0].deals[0].segs.v (4 lists)",
                        IMP.getItem(0)
                                .getField("seats")
                                .getItem(0)
                                .getField("deals")
                                .getItem(0)
                                .getField("segs")
                                .getField("v"),
                        IMP.getItem(0)
                                .getField("seats")
                                .getItem(0)
                                .getField("deals")
                                .getItem(0)
                                .getField("segs")));
    }

    static Stream<Arguments> leavesByPath() {
        return leaves().flatMap(
                        a -> Stream.of(true, false).map(v1 -> Arguments.of(a.get()[0], a.get()[1], a.get()[2], v1)));
    }

    @ParameterizedTest(name = "{0}, V1={3}")
    @MethodSource("leavesByPath")
    public void leafInsideListsReadsOnlyThatLeaf(String name, Column leaf, Column whole, boolean v1) {
        withReadPath(v1, () -> {
            Dataset<Row> table = spark.read().format("vortex").load(listLayoutPath);
            long leafBytes = bytesRead(table.select(leaf));
            long wholeBytes = bytesRead(table.select(whole));
            long flatLeaf =
                    bytesRead(spark.read().format("vortex").load(flatPath).select(leaf));
            System.out.printf(
                    Locale.ROOT,
                    "bytes read, V1=%s, %s: list layout %d (around it %d), flat %d%n",
                    v1,
                    name,
                    leafBytes,
                    wholeBytes,
                    flatLeaf);
            assertTrue(
                    leafBytes * 5 < wholeBytes,
                    name + " read " + leafBytes + " bytes; the list around it read " + wholeBytes);
            assertMatchesParquet(t -> t.select(functions.col("id"), leaf));
        });
    }

    @ParameterizedTest(name = "V1={0}")
    @MethodSource("readPaths")
    public void severalLeavesAndFiltersMatchParquet(boolean v1) {
        withReadPath(v1, () -> {
            assertMatchesParquet(t -> t.select(
                    functions.col("id"),
                    IMP.getField("id"),
                    IMP.getField("banner").getField("h"),
                    IMP.getItem(1).getField("seats").getField("name")));
            assertMatchesParquet(t -> t.where(functions.size(IMP).gt(2))
                    .select(
                            functions.col("id"),
                            IMP.getItem(0).getField("banner").getField("format").getField("h")));
            assertMatchesParquet(t -> t.where(IMP.isNull()).select(functions.col("id"), IMP.getField("id")));
            assertMatchesParquet(t -> t.where(functions.array_contains(
                            IMP.getField("id"), functions.col("id").multiply(10)))
                    .select(functions.col("id"), IMP.getField("tags")));
        });
    }

    static Stream<Boolean> readPaths() {
        return Stream.of(true, false);
    }

    private void withReadPath(boolean v1, Runnable body) {
        String previous = spark.conf().get(USE_V1, "");
        spark.conf().set(USE_V1, v1 ? "vortex" : "");
        try {
            body.run();
        } finally {
            spark.conf().set(USE_V1, previous);
        }
    }

    /** Asserts {@code query} returns Parquet's rows from both Vortex tables, with and without pruning. */
    private void assertMatchesParquet(Function<Dataset<Row>, Dataset<Row>> query) {
        List<Row> expected =
                query.apply(spark.read().parquet(parquetPath)).orderBy("id").collectAsList();
        for (String path : List.of(listLayoutPath, flatPath)) {
            Dataset<Row> read = query.apply(spark.read().format("vortex").load(path));
            assertEquals(expected, read.orderBy("id").collectAsList(), path);
            spark.conf().set(PRUNING, "false");
            try {
                assertEquals(expected, read.orderBy("id").collectAsList(), path + ", unpruned");
            } finally {
                spark.conf().set(PRUNING, "true");
            }
        }
    }

    /** Hashes every selected value so the whole result is decoded, and returns the bytes read from storage. */
    private static long bytesRead(Dataset<Row> query) {
        VortexNestedSchemaPruningIoTest.BYTES_READ.set(0);
        query.selectExpr("bit_xor(xxhash64(*))").collectAsList();
        return VortexNestedSchemaPruningIoTest.BYTES_READ.get();
    }
}
