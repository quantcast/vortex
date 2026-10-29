// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;

import dev.vortex.spark.write.VortexWrite;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.connector.write.LogicalWriteInfo;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/** How a {@link VortexTable} combines its own options with those of each scan and write. */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexTableOptionsTest {
    private static final StructType SCHEMA = new StructType().add("id", DataTypes.LongType);

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexTableOptionsTest")
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

    @Test
    void aScanOptionReplacesTheTableOptionItMatchesInAnyCase() {
        VortexTable table = table(Map.of("vortex.mergeSchema", "true", "fs.s3a.Endpoint", "table"));

        Map<String, String> merged = table.newScanBuilder(
                        new CaseInsensitiveStringMap(Map.of("VORTEX.MERGESCHEMA", "false", "FS.S3A.ENDPOINT", "scan")))
                .build()
                .options()
                .asCaseSensitiveMap();

        // The table's spelling survives, because Hadoop configuration keys are looked up case-sensitively.
        assertEquals(Map.of("vortex.mergeSchema", "false", "fs.s3a.Endpoint", "scan"), merged);
    }

    @Test
    void aWriteOptionReplacesTheTableOptionItMatchesInAnyCase() {
        VortexTable table = table(Map.of("vortex.write.batch.size", "512"));

        VortexWrite write = (VortexWrite)
                table.newWriteBuilder(writeInfo(Map.of("Vortex.Write.Batch.Size", "1024", "batch.size", "64")))
                        .build();

        assertEquals(
                Map.of("vortex.write.batch.size", "1024", "batch.size", "64"),
                write.info().options().asCaseSensitiveMap());
    }

    private VortexTable table(Map<String, String> options) {
        Map<String, String> withPath = new HashMap<>(options);
        withPath.put("path", tempDir.toUri().toString());
        return (VortexTable) new VortexDataSourceV2().getTable(new CaseInsensitiveStringMap(withPath), SCHEMA);
    }

    private static LogicalWriteInfo writeInfo(Map<String, String> options) {
        return new LogicalWriteInfo() {
            @Override
            public String queryId() {
                return "query";
            }

            @Override
            public StructType schema() {
                return SCHEMA;
            }

            @Override
            public CaseInsensitiveStringMap options() {
                return new CaseInsensitiveStringMap(options);
            }
        };
    }
}
