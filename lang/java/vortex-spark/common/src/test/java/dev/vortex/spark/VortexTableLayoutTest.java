// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.apache.spark.sql.AnalysisException;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.catalog.ExternalCatalogUtils;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

/** Directory names, partition values and sidecar files on disk match what Spark and Parquet use. */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexTableLayoutTest {
    private static final StructType STRINGS =
            new StructType().add("id", DataTypes.IntegerType, false).add("p", DataTypes.StringType, true);
    private static final String HIVE_NULL = "__HIVE_DEFAULT_PARTITION__";

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    void setUp() {
        spark = SparkSession.builder()
                .appName("VortexTableLayoutTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.ui.enabled", "false")
                .config("spark.sql.shuffle.partitions", "1")
                .config("spark.sql.adaptive.enabled", "false")
                .config("spark.sql.session.timeZone", "UTC")
                .getOrCreate();
        spark.conf().set("spark.sql.session.timeZone", "UTC");
    }

    @AfterAll
    void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    static Stream<Arguments> partitionValues() {
        return Stream.of("100%", "a/b", "C++", "with space", "Côte", "x=y", "01:02", null)
                .flatMap(value -> Stream.of(Arguments.of(value, false), Arguments.of(value, true)));
    }

    @ParameterizedTest(name = "value={0}, uri={1}")
    @MethodSource("partitionValues")
    void aPartitionValueThatNeedsEscapingRoundTripsLikeParquet(String value, boolean uri) throws IOException {
        Path vortex = tempDir.resolve("escape-vortex");
        Path parquet = tempDir.resolve("escape-parquet");
        Dataset<Row> input = strings(RowFactory.create(1, value));
        write(input, "vortex", vortex, uri, "p");
        write(input, "parquet", parquet, uri, "p");

        String directory = "p=" + (value == null ? HIVE_NULL : ExternalCatalogUtils.escapePathName(value));
        assertEquals(Set.of(directory), directories(vortex));
        assertEquals(directories(parquet), directories(vortex));
        assertRows(input, read("vortex", vortex, uri));
        assertRows(input, read("vortex", vortex, !uri));

        Dataset<Row> replacement = strings(RowFactory.create(2, value));
        replacement
                .coalesce(1)
                .write()
                .format("vortex")
                .partitionBy("p")
                .mode("overwrite")
                .save(location(vortex, uri));
        assertRows(replacement, read("vortex", vortex, uri));
    }

    @Test
    void nullsInSeveralPartitionColumnsUseTheHiveDefaultPartition() throws IOException {
        StructType schema = STRINGS.add("q", DataTypes.IntegerType, true);
        Dataset<Row> input = spark.createDataFrame(
                List.of(
                        RowFactory.create(1, "a/b", 7),
                        RowFactory.create(2, null, 7),
                        RowFactory.create(3, "C++", null),
                        RowFactory.create(4, null, null)),
                schema);
        Path vortex = tempDir.resolve("nulls-vortex");
        Path parquet = tempDir.resolve("nulls-parquet");
        write(input, "vortex", vortex, false, "p", "q");
        write(input, "parquet", parquet, false, "p", "q");

        assertEquals(directories(parquet), directories(vortex));
        assertTrue(Files.isDirectory(vortex.resolve("p=" + HIVE_NULL).resolve("q=" + HIVE_NULL)));
        Dataset<Row> data = read("vortex", vortex, false);
        assertEquals(sorted(input), sorted(data.select("id", "p", "q")));
        assertEquals(
                List.of(RowFactory.create(4)),
                data.filter("p IS NULL AND q IS NULL").select("id").collectAsList());
    }

    @ParameterizedTest
    @ValueSource(strings = {"date", "timestamp"})
    void dateAndTimestampPartitionsAreNamedAndTypedLikeParquet(String type) throws IOException {
        String first = type.equals("date") ? "2026-10-29" : "2026-10-29 01:02:03";
        String second = type.equals("date") ? "2026-10-30" : "2026-10-30 04:05:06";
        Dataset<Row> input = spark.sql("SELECT 1 AS id, " + type + " '" + first + "' AS p UNION ALL SELECT 2 AS id, "
                + type + " '" + second + "' AS p");
        Path vortex = tempDir.resolve(type + "-vortex");
        Path parquet = tempDir.resolve(type + "-parquet");
        write(input, "vortex", vortex, false, "p");
        write(input, "parquet", parquet, false, "p");

        assertEquals(
                Set.of(
                        "p=" + ExternalCatalogUtils.escapePathName(first),
                        "p=" + ExternalCatalogUtils.escapePathName(second)),
                directories(vortex));
        assertEquals(directories(parquet), directories(vortex));
        Dataset<Row> data = read("vortex", vortex, false);
        assertEquals(
                input.schema().apply("p").dataType(), data.schema().apply("p").dataType());
        assertEquals(sorted(input), sorted(data.select("id", "p")));

        Dataset<Row> filtered = data.filter("p = " + type + " '" + first + "'");
        String plan = filtered.queryExecution().executedPlan().toString();
        assertTrue(plan.contains("PartitionFilters: [") && plan.contains("isnotnull(p"), plan);
        assertEquals(List.of(RowFactory.create(1)), filtered.select("id").collectAsList());
    }

    @ParameterizedTest(name = "v1={0}")
    @ValueSource(booleans = {false, true})
    void sidecarFilesAndHiddenDirectoriesAreNotRead(boolean v1) throws IOException {
        String previous = spark.conf().get("spark.sql.sources.useV1SourceList");
        spark.conf().set("spark.sql.sources.useV1SourceList", v1 ? "vortex" : "");
        try {
            Path root = tempDir.resolve("sidecars-" + v1);
            Dataset<Row> input = strings(RowFactory.create(1, "kept"));
            write(input, "vortex", root, false);
            Path data = dataFiles(root, "vortex").get(0);
            for (String sidecar : List.of("_SUCCESS", ".part.crc", ".hidden", "part.vortex._COPYING_")) {
                Files.writeString(root.resolve(sidecar), "not a vortex file");
            }
            // Spark lists summary files for Parquet's sake; they are still not data files.
            Files.writeString(root.resolve("_metadata"), "not a vortex file");
            Files.writeString(root.resolve("_common_metadata"), "not a vortex file");
            Files.copy(
                    data,
                    Files.createDirectories(root.resolve("_temporary/0/attempt_0"))
                            .resolve("staged.vortex"));
            Files.copy(
                    data,
                    Files.createDirectories(root.resolve(".hidden-directory")).resolve("copy.vortex"));

            Dataset<Row> loaded = read("vortex", root, false);
            String plan = loaded.queryExecution().executedPlan().toString();
            // `FileScan vortex` is the V1 node, `BatchScan` the V2 one.
            assertTrue(plan.contains(v1 ? "FileScan vortex" : "BatchScan"), plan);
            assertFalse(plan.contains(v1 ? "BatchScan" : "FileScan"), plan);
            assertRows(input, loaded);
            assertEquals(1L, loaded.count());
        } finally {
            spark.conf().set("spark.sql.sources.useV1SourceList", previous);
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"vortex", "parquet"})
    void aCommitMarkerAndStagingDirectoryAreIgnoredLikeParquet(String format) throws IOException {
        Path root = tempDir.resolve("marker-" + format);
        Dataset<Row> input = strings(RowFactory.create(1, "kept"));
        write(input, format, root, false);
        Path data = dataFiles(root, format).get(0);
        Files.writeString(root.resolve("_SUCCESS"), "");
        Files.copy(
                data,
                Files.createDirectories(root.resolve("_temporary/0/attempt_0")).resolve(data.getFileName()));

        assertRows(input, read(format, root, false));
    }

    @ParameterizedTest
    @ValueSource(strings = {"vortex", "parquet"})
    void aDirectoryNamedLikeADataFileReadsAsADataset(String format) {
        Path root = tempDir.resolve("table." + format);
        Dataset<Row> input = spark.range(0, 60, 1, 3).selectExpr("CAST(id AS INT) AS id", "CAST(id AS STRING) AS p");
        input.write().format(format).save(root.toString());

        assertEquals(
                3,
                Stream.of(root.toFile().list())
                        .filter(name -> name.endsWith("." + format))
                        .count());
        assertRows(input, read(format, root, false));
    }

    @ParameterizedTest
    @ValueSource(strings = {"vortex", "parquet"})
    void inputFileNameReportsTheFileEachRowCameFrom(String format) throws IOException {
        Path root = tempDir.resolve("input-file-" + format);
        spark.range(0, 90, 1, 3)
                .selectExpr("id", "CAST(id % 2 AS INT) AS p")
                .write()
                .format(format)
                .partitionBy("p")
                .save(root.toString());
        Set<String> written = dataFiles(root, format).stream()
                .map(path -> path.toUri().getPath())
                .collect(Collectors.toSet());

        List<Row> files = read(format, root, false)
                .selectExpr("input_file_name() AS file", "p")
                .distinct()
                .collectAsList();
        assertEquals(6, written.size());
        assertEquals(written.size(), files.size());
        Set<String> read = files.stream()
                .map(row -> java.net.URI.create(row.getString(0)).getPath())
                .collect(Collectors.toSet());
        assertEquals(written, read);
        for (Row row : files) {
            assertTrue(row.getString(0).contains("/p=" + row.getInt(1) + "/"), row.toString());
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"vortex", "parquet"})
    void saveWithAPathArgumentUsesTheDefaultMode(String format) {
        Path root = tempDir.resolve("default-mode-" + format);
        Dataset<Row> input = strings(RowFactory.create(1, "a"), RowFactory.create(2, "b"));
        input.write().format(format).save(root.toString());

        assertRows(input, spark.read().format(format).load(root.toString()));
        // The default mode is ErrorIfExists, the same for both formats.
        assertThrows(AnalysisException.class, () -> input.write().format(format).save(root.toString()));
        assertRows(input, spark.read().format(format).load(root.toString()));
    }

    private Dataset<Row> strings(Row... rows) {
        return spark.createDataFrame(List.of(rows), STRINGS);
    }

    private static String location(Path path, boolean uri) {
        return uri ? path.toUri().toString() : path.toString();
    }

    private static void write(Dataset<Row> input, String format, Path path, boolean uri, String... partitions) {
        input.coalesce(1).write().format(format).partitionBy(partitions).save(location(path, uri));
    }

    private Dataset<Row> read(String format, Path path, boolean uri) {
        return spark.read().format(format).load(location(path, uri));
    }

    private static void assertRows(Dataset<Row> expected, Dataset<Row> actual) {
        assertEquals(sorted(expected.select("id", "p")), sorted(actual.select("id", "p")));
    }

    private static List<String> sorted(Dataset<Row> data) {
        return data.collectAsList().stream().map(Row::toString).sorted().toList();
    }

    private static Set<String> directories(Path root) throws IOException {
        try (Stream<Path> paths = Files.walk(root)) {
            return paths.filter(Files::isDirectory)
                    .filter(path -> !path.equals(root))
                    .map(path -> root.relativize(path).toString())
                    .collect(Collectors.toSet());
        }
    }

    private static List<Path> dataFiles(Path root, String format) throws IOException {
        try (Stream<Path> paths = Files.walk(root)) {
            return paths.filter(Files::isRegularFile)
                    .filter(path -> path.getFileName().toString().endsWith("." + format))
                    .toList();
        }
    }
}
