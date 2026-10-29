// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.adobe.testing.s3mock.testcontainers.S3MockContainer;
import java.util.List;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

/** Writes through Spark's cloud commit protocol and the S3A magic committer, which uploads straight to S3. */
@Testcontainers(disabledWithoutDocker = true)
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexS3MagicCommitterTest {
    private static final String BUCKET = "magic-committer-bucket";

    @Container
    private static final S3MockContainer S3_MOCK = new S3MockContainer("4.11.0").withInitialBuckets(BUCKET);

    private SparkSession spark;

    @BeforeAll
    void setUp() {
        spark = SparkSession.builder()
                .appName("VortexS3MagicCommitterTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.ui.enabled", "false")
                .config("spark.sql.adaptive.enabled", "false")
                .config("spark.hadoop.fs.s3a.endpoint", S3_MOCK.getHttpEndpoint())
                .config("spark.hadoop.fs.s3a.access.key", "foo")
                .config("spark.hadoop.fs.s3a.secret.key", "bar")
                .config("spark.hadoop.fs.s3a.path.style.access", "true")
                .config("spark.hadoop.fs.s3a.change.detection.version.required", "false")
                .config("spark.hadoop.fs.s3a.change.detection.mode", "none")
                .config("spark.hadoop.fs.s3a.multiobjectdelete.enable", "false")
                .config("spark.hadoop.fs.s3a.committer.name", "magic")
                .config("spark.hadoop.fs.s3a.committer.magic.enabled", "true")
                .config(
                        "spark.hadoop.mapreduce.outputcommitter.factory.scheme.s3a",
                        "org.apache.hadoop.fs.s3a.commit.S3ACommitterFactory")
                .config(
                        "spark.sql.sources.commitProtocolClass",
                        "org.apache.spark.internal.io.cloud.PathOutputCommitProtocol")
                .config(
                        "spark.sql.parquet.output.committer.class",
                        "org.apache.spark.internal.io.cloud.BindingParquetOutputCommitter")
                .getOrCreate();
    }

    @AfterAll
    void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"vortex", "parquet"})
    void aPartitionedWriteCommitsThroughTheMagicCommitter(String format) {
        String path = "s3a://" + BUCKET + "/" + format;
        spark.range(0, 40, 1, 2)
                .selectExpr("id", "CAST(id % 2 AS INT) AS p")
                .write()
                .format(format)
                .partitionBy("p")
                .save(path);
        spark.range(40, 60, 1, 2)
                .selectExpr("id", "CAST(id % 2 AS INT) AS p")
                .write()
                .format(format)
                .partitionBy("p")
                .mode("append")
                .save(path);

        List<Row> counts = spark.read()
                .format(format)
                .load(path)
                .groupBy("p")
                .count()
                .orderBy("p")
                .collectAsList();
        assertEquals("[[0,30], [1,30]]", counts.toString());
    }
}
