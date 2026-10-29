// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import org.apache.spark.serializer.KryoSerializer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/**
 * Writes and reads Vortex files in a session whose serializer is {@link KryoSerializer}, with no registrator. Task
 * results, including the writers' commit messages, travel back to the driver through Kryo.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexKryoWriteTest {

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexKryoWriteTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.ui.enabled", "false")
                .config("spark.serializer", KryoSerializer.class.getName())
                .config("spark.kryo.unsafe", "true")
                .getOrCreate();
        // getOrCreate reuses a running context, which would silently keep its serializer.
        assertEquals(
                KryoSerializer.class.getName(),
                spark.sparkContext().getConf().get("spark.serializer"),
                "test needs a fresh SparkContext");
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @Test
    @DisplayName("A partitioned write and read round-trips under Kryo")
    public void partitionedWriteAndRead() {
        List<Row> rows = Arrays.asList(
                RowFactory.create(1, "alpha", "A"),
                RowFactory.create(2, "beta", "B"),
                RowFactory.create(3, "gamma", "A"),
                RowFactory.create(4, "delta", "B"));
        Dataset<Row> df = spark.createDataFrame(
                rows,
                DataTypes.createStructType(Arrays.asList(
                        DataTypes.createStructField("id", DataTypes.IntegerType, false),
                        DataTypes.createStructField("name", DataTypes.StringType, true),
                        DataTypes.createStructField("group", DataTypes.StringType, true))));
        String path = tempDir.resolve("kryo_partitioned").toUri().toString();

        df.repartition(2)
                .write()
                .format("vortex")
                .partitionBy("group")
                .option("path", path)
                .mode(SaveMode.Overwrite)
                .save();

        Dataset<Row> read = spark.read().format("vortex").option("path", path).load();
        assertEquals(rows, read.select("id", "name", "group").orderBy("id").collectAsList());
    }

    @Test
    @DisplayName("An unpartitioned write and read round-trips under Kryo")
    public void unpartitionedWriteAndRead() {
        List<Row> rows = Arrays.asList(RowFactory.create(1, "alpha"), RowFactory.create(2, "beta"));
        Dataset<Row> df = spark.createDataFrame(
                rows,
                DataTypes.createStructType(Arrays.asList(
                        DataTypes.createStructField("id", DataTypes.IntegerType, false),
                        DataTypes.createStructField("name", DataTypes.StringType, true))));
        String path = tempDir.resolve("kryo_flat").toUri().toString();

        df.repartition(2)
                .write()
                .format("vortex")
                .option("path", path)
                .mode(SaveMode.Overwrite)
                .save();

        Dataset<Row> read = spark.read().format("vortex").option("path", path).load();
        assertEquals(rows, read.orderBy("id").collectAsList());
    }
}
