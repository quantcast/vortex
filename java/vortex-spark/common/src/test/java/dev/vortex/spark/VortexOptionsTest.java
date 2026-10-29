// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

final class VortexOptionsTest {
    @ParameterizedTest
    @ValueSource(
            strings = {
                "vortex.readConcurrency",
                "vortex.readconcurrency",
                "VORTEX.READCONCURRENCY",
                "Vortex.ReadConcurrency"
            })
    void anyCasingFindsTheValue(String key) {
        VortexOptions options = VortexOptions.of(Map.of("vortex.readConcurrency", "4"));

        assertEquals(Optional.of("4"), options.get(key));
        assertEquals(4, options.getInt(key, 1));
    }

    @Test
    void anUnsetOptionFallsBack() {
        VortexOptions options = VortexOptions.empty();

        assertEquals(Optional.empty(), options.get("vortex.absent"));
        assertEquals("fallback", options.get("vortex.absent", "fallback"));
        assertEquals(7, options.getInt("vortex.absent", 7));
        assertTrue(options.getBoolean("vortex.absent", true));
    }

    @Test
    void booleansAreReadWithoutRegardToCase() {
        VortexOptions options = VortexOptions.of(Map.of("on", "TRUE", "off", " False "));

        assertTrue(options.getBoolean("on", false));
        assertFalse(options.getBoolean("off", true));
    }

    @Test
    void aValueOfTheWrongShapeNamesItsOption() {
        VortexOptions options = VortexOptions.of(Map.of("vortex.count", "many", "vortex.flag", "yes"));

        assertTrue(assertThrows(IllegalArgumentException.class, () -> options.getInt("vortex.count", 1))
                .getMessage()
                .contains("vortex.count"));
        assertTrue(assertThrows(IllegalArgumentException.class, () -> options.getBoolean("vortex.flag", true))
                .getMessage()
                .contains("vortex.flag"));
    }

    @Test
    void hadoopKeysKeepTheCasingTheyWereGivenIn() {
        Map<String, String> given = Map.of("fs.s3a.Endpoint", "https://storage.example");

        // Hadoop configuration keys are case-sensitive, so the map that feeds a Configuration must not be
        // lower-cased along with the vortex.* lookups.
        assertEquals(given, VortexOptions.of(given).asCaseSensitiveMap());
    }

    @Test
    void anOverlayReplacesEverySpellingOfItsKey() {
        Map<String, String> base = Map.of("vortex.x", "a", "VORTEX.X", "b", "vortex.y", "c");

        Map<String, String> merged = VortexOptions.overlay(base, Map.of("Vortex.X", "d", "vortex.z", "e"));

        assertEquals("d", VortexOptions.of(merged).get("vortex.x").orElseThrow());
        assertEquals(3, merged.size());
        assertEquals("c", merged.get("vortex.y"));
        assertEquals("e", merged.get("vortex.z"));
    }

    @Test
    void onlyVortexSessionSettingsAreDefaultsAndExplicitOptionsWin() {
        Map<String, String> session =
                Map.of("vortex.workerThreads", "8", "vortex.readConcurrency", "2", "spark.sql.shuffle.partitions", "4");

        VortexOptions resolved =
                VortexOptions.of(VortexOptions.withSessionDefaults(session, Map.of("VORTEX.WORKERTHREADS", "1")));

        assertEquals(1, resolved.getInt("vortex.workerThreads", 0));
        assertEquals(2, resolved.getInt("vortex.readConcurrency", 0));
        assertEquals(Optional.empty(), resolved.get("spark.sql.shuffle.partitions"));
    }

    @Test
    void sessionPrefixesMatchInAnyCase() {
        Map<String, String> session = Map.of(
                "VORTEX.READCONCURRENCY", "3",
                "Vortex.FooterParallelism", "5",
                "spark.datasource.VORTEX.write.batch.size", "512",
                "Spark.DataSource.Vortex.Stats.MaxFiles", "7");

        VortexOptions resolved = VortexOptions.of(VortexOptions.withSessionDefaults(session, Map.of()));

        assertEquals(3, resolved.getInt("vortex.readConcurrency", 0));
        assertEquals(5, resolved.getInt("vortex.footerParallelism", 0));
        assertEquals(512, resolved.getInt("vortex.write.batch.size", 0));
        assertEquals(7, resolved.getInt("vortex.stats.maxFiles", 0));
    }

    @Test
    void dataSourceSessionSettingsNameTheirOptionWithoutRepeatingVortex() {
        Map<String, String> session = Map.of(
                "spark.datasource.vortex.write.batch.size", "512",
                "spark.datasource.vortex.workerThreads", "8",
                "vortex.workerThreads", "2",
                "spark.datasource.parquet.mergeSchema", "true");

        VortexOptions resolved = VortexOptions.of(VortexOptions.withSessionDefaults(session, Map.of()));

        assertEquals(512, resolved.getInt("vortex.write.batch.size", 0));
        // A plain vortex.* setting is set by the application itself, so it beats the cluster-wide form.
        assertEquals(2, resolved.getInt("vortex.workerThreads", 0));
        assertEquals(Optional.empty(), resolved.get("vortex.mergeSchema"));
        assertEquals(Optional.empty(), resolved.get("mergeSchema"));
        assertEquals(
                3,
                VortexOptions.of(VortexOptions.withSessionDefaults(session, Map.of("vortex.workerthreads", "3")))
                        .getInt("vortex.workerThreads", 0));
    }

    @Test
    void optionsSurviveSerialization() throws IOException, ClassNotFoundException {
        VortexOptions options = VortexOptions.of(Map.of("vortex.workerThreads", "2"));

        VortexOptions shipped = roundTrip(options);

        assertEquals(2, shipped.getInt("vortex.workerthreads", 1));
        assertEquals(options, shipped);
    }

    private static VortexOptions roundTrip(VortexOptions options) throws IOException, ClassNotFoundException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (ObjectOutputStream out = new ObjectOutputStream(bytes)) {
            out.writeObject(options);
        }
        try (ObjectInputStream in = new ObjectInputStream(new ByteArrayInputStream(bytes.toByteArray()))) {
            return (VortexOptions) in.readObject();
        }
    }
}
