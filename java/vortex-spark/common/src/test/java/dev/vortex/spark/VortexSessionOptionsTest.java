// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/** {@code vortex.*} settings in the Spark session are defaults beneath the options of each read and write. */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexSessionOptionsTest {
    private static final String READ_CONCURRENCY = "vortex.readConcurrency";
    private static final String USE_V1 = "spark.sql.sources.useV1SourceList";
    private static final String DATA_SOURCE_PREFIX = "spark.datasource.";

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexSessionOptionsTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.ui.enabled", "false")
                .getOrCreate();
    }

    @AfterEach
    public void resetConf() {
        for (String option : new String[] {READ_CONCURRENCY, VortexSparkSession.PROVIDER_OPTION}) {
            spark.conf().unset(option);
            spark.conf().unset(DATA_SOURCE_PREFIX + option);
        }
        spark.conf().unset(USE_V1);
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @Test
    void aSessionSettingAppliesToV2Reads() {
        spark.conf().set(USE_V1, "");
        assertSessionSettingAppliesToReads(write("v2"), "BatchScan", READ_CONCURRENCY);
    }

    @Test
    void aDataSourceSessionSettingAppliesToV2Reads() {
        spark.conf().set(USE_V1, "");
        assertSessionSettingAppliesToReads(write("v2_data_source"), "BatchScan", DATA_SOURCE_PREFIX + READ_CONCURRENCY);
    }

    @Test
    void aPlainSessionSettingBeatsTheDataSourceForm() {
        String path = write("both_forms");
        spark.conf().set(DATA_SOURCE_PREFIX + READ_CONCURRENCY, "-1");
        spark.conf().set(READ_CONCURRENCY, "0");

        assertEquals(10, spark.read().format("vortex").load(path).count());
    }

    @Test
    void aSessionSettingAppliesToV1Reads() {
        spark.conf().set(USE_V1, "vortex");
        assertSessionSettingAppliesToReads(write("v1"), "FileScan vortex", READ_CONCURRENCY);
    }

    @Test
    void aSessionSettingAppliesToWrites() {
        assertSessionSettingAppliesToWrites("written", VortexSparkSession.PROVIDER_OPTION);
    }

    @Test
    void aDataSourceSessionSettingAppliesToWrites() {
        assertSessionSettingAppliesToWrites(
                "written_data_source", DATA_SOURCE_PREFIX + VortexSparkSession.PROVIDER_OPTION);
    }

    private void assertSessionSettingAppliesToWrites(String name, String sessionKey) {
        Dataset<Row> data = spark.range(10).toDF();
        String output = tempDir.resolve(name).toString();
        spark.conf().set(sessionKey, "dev.vortex.spark.NoSuchProvider");

        Exception failure = assertThrows(
                Exception.class,
                () -> data.write().format("vortex").mode(SaveMode.Overwrite).save(output));
        assertTrue(mentions(failure, "dev.vortex.spark.NoSuchProvider"), failure::toString);

        // An empty provider selects the default session, and an explicit option beats the session's.
        data.write()
                .format("vortex")
                .option(VortexSparkSession.PROVIDER_OPTION, "")
                .mode(SaveMode.Overwrite)
                .save(output);
        spark.conf().unset(sessionKey);
        assertEquals(10, spark.read().format("vortex").load(output).count());
    }

    private void assertSessionSettingAppliesToReads(String path, String scanNode, String sessionKey) {
        spark.conf().set(sessionKey, "-1");

        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class,
                () -> spark.read().format("vortex").load(path).count());
        assertTrue(failure.getMessage().contains(READ_CONCURRENCY), failure::getMessage);

        Dataset<Row> read = spark.read()
                .format("vortex")
                .option("VORTEX.READCONCURRENCY", "0")
                .load(path);
        assertTrue(read.queryExecution().executedPlan().toString().contains(scanNode));
        assertEquals(10, read.count());
    }

    private String write(String name) {
        String output = tempDir.resolve(name).toString();
        spark.range(10).write().format("vortex").save(output);
        return output;
    }

    private static boolean mentions(Throwable failure, String text) {
        for (Throwable cause = failure; cause != null; cause = cause.getCause()) {
            if (String.valueOf(cause.getMessage()).contains(text)) {
                return true;
            }
        }
        return false;
    }
}
