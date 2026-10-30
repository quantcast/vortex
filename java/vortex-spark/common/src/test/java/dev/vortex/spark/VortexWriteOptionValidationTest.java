// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.api.Session;
import java.nio.file.Path;
import java.util.Locale;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * Write options the writers cannot use fail on the driver, before Spark launches a task.
 *
 * <p>Spark's V1 overwrite deletes the old files before any {@code FileFormat} hook sees the options, so this cannot
 * keep them, but it does stop the job from starting and reports the option directly rather than as a task failure.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexWriteOptionValidationTest {
    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexWriteOptionValidationTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.ui.enabled", "false")
                .getOrCreate();
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @ParameterizedTest
    @CsvSource({
        "batch.size, lots",
        "batch.size, 0",
        "vortex.write.batch.size, 65537",
        "vortex.listLayout, yes",
        "vortex.session.provider, dev.vortex.spark.NoSuchProvider",
        "vortex.session.provider, java.lang.String",
        "vortex.session.provider, dev.vortex.spark.VortexSessionProvider",
        "vortex.session.provider, dev.vortex.spark.VortexWriteOptionValidationTest$AbstractProvider",
        "vortex.session.provider, dev.vortex.spark.VortexWriteOptionValidationTest$PrivateConstructorProvider",
        "vortex.workerThreads, many",
        "vortex.workerThreads, -1"
    })
    void anUnusableWriteOptionFailsOnTheDriver(String option, String value) {
        String output = tempDir.resolve(option + "-" + value).toString();
        spark.range(10).write().format("vortex").save(output);
        // A first write fixes the JVM's worker pool, after which a later value used to go unchecked.
        spark.read().format("vortex").load(output).count();

        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class, () -> spark.range(10)
                .write()
                .format("vortex")
                .option(option, value)
                .mode(SaveMode.Overwrite)
                .save(output));
        assertTrue(
                failure.getMessage().toLowerCase(Locale.ROOT).contains(option.toLowerCase(Locale.ROOT)),
                failure::getMessage);
    }

    public abstract static class AbstractProvider implements VortexSessionProvider {}

    public static final class PrivateConstructorProvider implements VortexSessionProvider {
        private PrivateConstructorProvider() {}

        @Override
        public Session get() {
            return VortexSparkSession.get();
        }
    }
}
