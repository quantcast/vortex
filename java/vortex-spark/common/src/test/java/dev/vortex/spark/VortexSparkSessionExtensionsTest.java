// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.SparkSessionExtensions;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

/** The Vortex optimizer rules, loaded from the classpath and listed in {@code spark.sql.extensions} as well. */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexSparkSessionExtensionsTest {
    private SparkSession spark;

    @BeforeAll
    public void setUp() throws IOException {
        Path warehouseDir = Files.createTempDirectory("vortex-extensions");
        spark = SparkSession.builder()
                .appName("VortexSparkSessionExtensionsTest")
                .master("local[1]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.warehouse.dir", warehouseDir.toUri().toString())
                .config("spark.sql.extensions", VortexSparkSessionExtensions.class.getName())
                .config("spark.ui.enabled", "false")
                .getOrCreate();
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @Test
    @DisplayName("Registering the extensions twice adds the pruning rule once")
    public void registersOnce() {
        SparkSessionExtensions extensions = new SparkSessionExtensions();
        new VortexSparkSessionExtensions().apply(extensions);
        new VortexSparkSessionExtensions().apply(extensions);
        assertEquals(1, extensions.buildPreCBORules(spark).size());
    }

    @Test
    @DisplayName("A catalog table is pruned when spark.sql.extensions also lists the extensions")
    public void prunesWithExtensionsListed() {
        spark.sql(
                "CREATE TABLE listed USING vortex AS SELECT id, named_struct('a', id, 'b', id * 2) AS s FROM range(10)");
        try {
            String plan = spark.sql("SELECT s.b FROM listed")
                    .queryExecution()
                    .executedPlan()
                    .toString();
            assertTrue(plan.contains("ReadSchema: struct<s:struct<b:bigint>>"), plan);
            List<Row> rows = spark.sql("SELECT s.b FROM listed ORDER BY s.b").collectAsList();
            assertEquals(10, rows.size());
            assertEquals(18L, rows.get(9).getLong(0));
        } finally {
            spark.sql("DROP TABLE listed");
        }
    }
}
