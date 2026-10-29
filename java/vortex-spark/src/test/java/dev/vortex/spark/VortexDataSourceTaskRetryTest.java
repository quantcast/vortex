// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Stream;
import org.apache.spark.TaskContext;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/** Writes whose first task attempt fails and is retried by Spark. */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexDataSourceTaskRetryTest {

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexDataSourceTaskRetryTest")
                .master("local[2,2]") // allow one retry per task
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.ui.enabled", "false")
                .getOrCreate();
        spark.udf()
                .register(
                        "failFirstAttempt",
                        (UDF1<Long, Long>) id -> {
                            if (TaskContext.get().attemptNumber() == 0 && id >= 6) {
                                throw new IllegalStateException("first attempt fails at id " + id);
                            }
                            return id;
                        },
                        DataTypes.LongType);
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    /**
     * The aborted attempt flushed three batches before failing. Its file must not survive, or the retry's rows are
     * duplicated. The output is a {@code file://} URL rather than a plain path, percent-encoded when it has a space.
     */
    @ParameterizedTest
    @CsvSource({"retried, false", "with space, false", "partitioned, true"})
    public void abortedAttemptLeavesNoFile(String dirName, boolean partitioned) throws IOException {
        Path dir = tempDir.resolve(dirName);
        String url = dir.toUri().toString();
        var writer = spark.range(10)
                .selectExpr("failFirstAttempt(id) as id", "0 as p")
                .coalesce(1)
                .write()
                .format("vortex")
                .mode("append")
                .option("vortex.write.batch.size", "2");
        (partitioned ? writer.partitionBy("p") : writer).save(url);

        try (Stream<Path> files = Files.walk(dir)) {
            assertEquals(1, files.filter(f -> f.toString().endsWith(".vortex")).count());
        }
        assertEquals(10, spark.read().format("vortex").load(url).count());
    }
}
