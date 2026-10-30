// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.util.HashMap;
import java.util.List;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Sessions built the way PySpark builds them, with the {@code SparkSession(SparkContext, options)} constructor. On
 * Spark 3.5 that constructor skips classpath extension providers, which the user guide documents.
 */
public final class VortexPySparkSessionTest {
    private static final String PRUNED = "ReadSchema: struct<s:struct<b:bigint>>";
    private static final String UNPRUNED = "ReadSchema: struct<s:struct<a:bigint,b:bigint>>";

    private SparkSession spark;

    @AfterEach
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
        SparkSession.clearActiveSession();
        SparkSession.clearDefaultSession();
    }

    @Test
    @DisplayName("Without spark.sql.extensions, the constructor registers the pruning rule only on Spark 4")
    public void classpathProviderOnlyOnSpark4() throws Exception {
        spark = pysparkSession(false);
        boolean loadsClasspathProviders = !spark.version().startsWith("3.");
        assertReadSchema(loadsClasspathProviders ? PRUNED : UNPRUNED);
    }

    @Test
    @DisplayName("With spark.sql.extensions, the constructor registers the pruning rule")
    public void extensionsConf() throws Exception {
        spark = pysparkSession(true);
        assertReadSchema(PRUNED);
    }

    private static SparkSession pysparkSession(boolean listExtensions) throws Exception {
        SparkConf conf = new SparkConf()
                .setMaster("local[1]")
                .setAppName("VortexPySparkSessionTest")
                .set("spark.driver.host", "127.0.0.1")
                .set("spark.ui.enabled", "false");
        if (listExtensions) {
            conf.set("spark.sql.extensions", VortexSparkSessionExtensions.class.getName());
        }
        SparkContext context = new SparkContext(conf);
        HashMap<String, String> options = new HashMap<>();
        options.put(
                "spark.sql.warehouse.dir",
                Files.createTempDirectory("vortex-pyspark").toUri().toString());
        SparkSession session = (SparkSession)
                sessionClass().getConstructor(SparkContext.class, HashMap.class).newInstance(context, options);
        SparkSession.setDefaultSession(session);
        SparkSession.setActiveSession(session);
        return session;
    }

    /** The JVM class PySpark instantiates: {@code classic.SparkSession} on Spark 4, {@code SparkSession} on 3.5. */
    private static Class<?> sessionClass() throws ClassNotFoundException {
        try {
            return Class.forName("org.apache.spark.sql.classic.SparkSession");
        } catch (ClassNotFoundException e) {
            return Class.forName("org.apache.spark.sql.SparkSession");
        }
    }

    private void assertReadSchema(String readSchema) {
        spark.sql("CREATE TABLE t USING vortex AS SELECT id, named_struct('a', id, 'b', id * 2) AS s FROM range(10)");
        List<Row> rows = spark.sql("SELECT s.b FROM t ORDER BY s.b").collectAsList();
        assertEquals(10, rows.size());
        assertEquals(18L, rows.get(9).getLong(0));
        String plan =
                spark.sql("SELECT s.b FROM t").queryExecution().executedPlan().toString();
        assertTrue(plan.contains(readSchema), plan);
    }
}
