// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static org.junit.jupiter.api.Assertions.assertEquals;

import dev.vortex.spark.VortexFilePartition;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.connector.expressions.Expression;
import org.apache.spark.sql.connector.expressions.Expressions;
import org.apache.spark.sql.connector.expressions.LiteralValue;
import org.apache.spark.sql.connector.expressions.filter.Predicate;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.vectorized.ColumnarBatch;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
final class VortexPartitionReaderTest {
    private SparkSession spark;
    private String file;

    @TempDir
    static Path tempDir;

    @BeforeAll
    void setUp() throws IOException {
        spark = SparkSession.builder()
                .appName("VortexPartitionReaderTest")
                .master("local[1]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.ui.enabled", "false")
                .getOrCreate();
        Path dir = tempDir.resolve("table");
        spark.range(10)
                .selectExpr("id", "concat('n', id) AS name", "CAST(id AS DOUBLE) AS value")
                .coalesce(1)
                .write()
                .format("vortex")
                .mode(SaveMode.Overwrite)
                .save(dir.toUri().toString());
        try (Stream<Path> files = Files.list(dir)) {
            file = files.filter(p -> p.toString().endsWith(".vortex"))
                    .findFirst()
                    .orElseThrow()
                    .toUri()
                    .toString();
        }
    }

    @AfterAll
    void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    /** With no required columns, as for {@code count(*)}, the scan must not materialize any column. */
    @Test
    @DisplayName("An empty required schema yields batches with no columns but the right row counts")
    void emptyProjectionReadsNoColumns() {
        assertEquals(10, countRows(null));

        Predicate idAboveFive = new Predicate(
                ">", new Expression[] {Expressions.column("id"), new LiteralValue<>(5L, DataTypes.LongType)});
        assertEquals(4, countRows(new Predicate[] {idAboveFive}));
    }

    private long countRows(Predicate[] predicates) {
        VortexFilePartition partition = new VortexFilePartition(List.of(file), new StructType(), Map.of(), Map.of());
        StructType tableSchema = new StructType()
                .add("id", DataTypes.LongType)
                .add("name", DataTypes.StringType)
                .add("value", DataTypes.DoubleType);
        VortexPartitionReader reader =
                new VortexPartitionReader(partition, new StructType(), tableSchema, Map.of(), predicates);
        long rows = 0;
        while (reader.next()) {
            ColumnarBatch batch = reader.get();
            assertEquals(0, batch.numCols(), "an empty projection must not materialize any column");
            rows += batch.numRows();
        }
        reader.close();
        return rows;
    }
}
