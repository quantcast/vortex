// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.io;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import dev.vortex.spark.VortexOptions;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.apache.spark.SerializableWritable;
import org.apache.spark.SparkConf;
import org.apache.spark.serializer.JavaSerializer;
import org.apache.spark.serializer.KryoSerializer;
import org.apache.spark.serializer.SerializerInstance;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import scala.reflect.ClassTag;
import scala.reflect.ClassTag$;

final class VortexIoTest {
    @Test
    void concurrenciesDefaultToNativeDefaults() {
        VortexIo io = VortexIo.create(VortexOptions.empty(), new Configuration());

        assertEquals(0, io.readConcurrency());
        assertEquals(0, io.splitConcurrency());
    }

    @Test
    void splitConcurrencyIsParsedAndValidated() {
        Configuration conf = new Configuration();

        assertEquals(
                2,
                VortexIo.create(options(VortexIo.SPLIT_CONCURRENCY_OPTION, "2"), conf)
                        .splitConcurrency());
        assertEquals(
                2,
                VortexIo.create(options("VORTEX.SPLITCONCURRENCY", "2"), conf).splitConcurrency());
        assertThrows(
                IllegalArgumentException.class,
                () -> VortexIo.create(options(VortexIo.SPLIT_CONCURRENCY_OPTION, "-1"), conf));
        assertThrows(
                IllegalArgumentException.class,
                () -> VortexIo.create(options(VortexIo.SPLIT_CONCURRENCY_OPTION, "many"), conf));
    }

    @Test
    void readConcurrencyIsParsedAndValidated() {
        Configuration conf = new Configuration();

        assertEquals(
                8,
                VortexIo.create(options(VortexIo.READ_CONCURRENCY_OPTION, "8"), conf)
                        .readConcurrency());
        assertThrows(
                IllegalArgumentException.class,
                () -> VortexIo.create(options(VortexIo.READ_CONCURRENCY_OPTION, "-1"), conf));
        assertThrows(
                IllegalArgumentException.class,
                () -> VortexIo.create(options(VortexIo.READ_CONCURRENCY_OPTION, "many"), conf));
    }

    @Test
    void readConcurrencyIsFoundUnderAnyKeyCase() {
        Configuration conf = new Configuration();

        assertEquals(
                6, VortexIo.create(options("VORTEX.READCONCURRENCY", "6"), conf).readConcurrency());
        assertEquals(
                6, VortexIo.create(options("vortex.readconcurrency", "6"), conf).readConcurrency());
    }

    @Test
    void settingsAndConfigurationSurviveSerialization() throws IOException, ClassNotFoundException {
        Configuration conf = new Configuration();
        conf.set("fs.s3a.endpoint", "https://storage.example");
        VortexIo io = VortexIo.create(
                VortexOptions.of(Map.of(VortexIo.READ_CONCURRENCY_OPTION, "4", VortexIo.SPLIT_CONCURRENCY_OPTION, "3")),
                conf);

        VortexIo shipped = roundTrip(io);

        assertEquals(4, shipped.readConcurrency());
        assertEquals(3, shipped.splitConcurrency());
        assertEquals("https://storage.example", shipped.hadoopConf().get("fs.s3a.endpoint"));
    }

    private static VortexOptions options(String key, String value) {
        return VortexOptions.of(Map.of(key, value));
    }

    /**
     * A broadcast reaches remote executors through {@code spark.serializer}. Local mode hands tasks the driver's object
     * without serializing it, so only a round trip through each serializer shows what an executor gets.
     */
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void broadcastConfigurationSurvivesEachSparkSerializer(boolean kryo) {
        Configuration conf = new Configuration(false);
        conf.set("fs.s3a.endpoint", "https://storage.example");
        SerializerInstance serializer = (kryo ? new KryoSerializer(new SparkConf()) : new JavaSerializer(new SparkConf()))
                .newInstance();
        ClassTag<Object> tag = ClassTag$.MODULE$.apply(Object.class);

        SerializableWritable<?> shipped = (SerializableWritable<?>)
                serializer.deserialize(serializer.serialize(VortexIo.broadcastValue(conf), tag), tag);

        assertEquals("https://storage.example", ((Configuration) shipped.value()).get("fs.s3a.endpoint"));
    }

    private static VortexIo roundTrip(VortexIo io) throws IOException, ClassNotFoundException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (ObjectOutputStream out = new ObjectOutputStream(bytes)) {
            out.writeObject(io);
        }
        try (ObjectInputStream in = new ObjectInputStream(new ByteArrayInputStream(bytes.toByteArray()))) {
            return (VortexIo) in.readObject();
        }
    }
}
