// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import static org.junit.jupiter.api.Assertions.assertEquals;

import dev.vortex.spark.VortexFilePartition;
import dev.vortex.spark.write.PartitionedVortexDataWriter.PartitionedWriterCommitMessage;
import dev.vortex.spark.write.PartitionedVortexDataWriter.ResolvedTransform;
import java.util.List;
import java.util.Map;
import org.apache.spark.SparkConf;
import org.apache.spark.serializer.KryoSerializer;
import org.apache.spark.serializer.SerializerInstance;
import org.apache.spark.sql.connector.expressions.Expressions;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import scala.reflect.ClassTag$;

/**
 * Round-trips the connector's records and commit messages through Spark's {@link KryoSerializer} with no registrator,
 * the way a job with {@code spark.serializer=KryoSerializer} sends them. Kryo's {@code FieldSerializer} cannot handle
 * records, so these types must pick their serializer themselves rather than rely on a user-configured registrator.
 */
final class KryoSerializationTest {

    private static final StructType SCHEMA = new StructType(new StructField[] {
        new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
        new StructField("event_date", DataTypes.DateType, true, Metadata.empty())
    });

    private static SerializerInstance kryo(boolean unsafe) {
        return new KryoSerializer(new SparkConf().set("spark.kryo.unsafe", String.valueOf(unsafe))).newInstance();
    }

    /**
     * A serializer that rejects unregistered classes, with the connector's types registered by name only. Kryo then
     * looks up their serializer the same way it does for unregistered classes, so it must still find the annotation.
     */
    private static SerializerInstance strictKryo() {
        String classes = String.join(
                ",",
                VortexWriterCommitMessage.class.getName(),
                VortexFilePartition.class.getName(),
                ResolvedTransform.class.getName(),
                PartitionedWriterCommitMessage.class.getName());
        return new KryoSerializer(new SparkConf()
                        .set("spark.kryo.registrationRequired", "true")
                        .set("spark.kryo.classesToRegister", classes))
                .newInstance();
    }

    @SuppressWarnings("unchecked")
    private static <T> T roundTrip(SerializerInstance kryo, T value) {
        Class<T> cls = (Class<T>) value.getClass();
        return kryo.deserialize(kryo.serialize(value, ClassTag$.MODULE$.apply(cls)), ClassTag$.MODULE$.apply(cls));
    }

    @ParameterizedTest(name = "unsafe={0}")
    @ValueSource(booleans = {false, true})
    void writerCommitMessage(boolean unsafe) {
        VortexWriterCommitMessage message = new VortexWriterCommitMessage("file:///tmp/a.vortex", 10, 1024);
        assertEquals(message, roundTrip(kryo(unsafe), message));
    }

    @ParameterizedTest(name = "unsafe={0}")
    @ValueSource(booleans = {false, true})
    void filePartition(boolean unsafe) {
        VortexFilePartition partition = new VortexFilePartition(
                List.of("file:///tmp/a.vortex"),
                SCHEMA,
                Map.of("aws_region", "us-east-1"),
                Map.of("event_date", "2026-10-28"),
                new VortexFilePartition.RowRange(0, 100));
        assertEquals(partition, roundTrip(kryo(unsafe), partition));
    }

    @ParameterizedTest(name = "unsafe={0}")
    @ValueSource(booleans = {false, true})
    void resolvedTransforms(boolean unsafe) {
        ResolvedTransform[] resolved = PartitionedVortexDataWriter.resolveTransforms(
                new Transform[] {Expressions.identity("id"), Expressions.days("event_date")}, SCHEMA);
        for (ResolvedTransform transform : resolved) {
            assertEquals(transform, roundTrip(kryo(unsafe), transform));
        }
    }

    @ParameterizedTest(name = "unsafe={0}")
    @ValueSource(booleans = {false, true})
    void partitionedCommitMessage(boolean unsafe) {
        // Both commit() paths: the immutable List.of() returned after close, and the populated list.
        for (List<VortexWriterCommitMessage> messages : List.of(
                List.<VortexWriterCommitMessage>of(),
                List.of(new VortexWriterCommitMessage("file:///tmp/a.vortex", 10, 1024)))) {
            PartitionedWriterCommitMessage message = new PartitionedWriterCommitMessage(messages);
            assertEquals(messages, roundTrip(kryo(unsafe), message).getPartitionMessages());
        }
    }

    @Test
    void registeredByNameUnderRegistrationRequired() {
        SerializerInstance kryo = strictKryo();
        VortexWriterCommitMessage message = new VortexWriterCommitMessage("file:///tmp/a.vortex", 10, 1024);
        assertEquals(message, roundTrip(kryo, message));
        VortexFilePartition partition = new VortexFilePartition(
                List.of("file:///tmp/a.vortex"), SCHEMA, Map.of(), Map.of(), new VortexFilePartition.RowRange(0, 100));
        assertEquals(partition, roundTrip(kryo, partition));
        ResolvedTransform transform =
                PartitionedVortexDataWriter.resolveTransforms(new Transform[] {Expressions.identity("id")}, SCHEMA)[0];
        assertEquals(transform, roundTrip(kryo, transform));
        List<VortexWriterCommitMessage> messages = List.of(message);
        assertEquals(
                messages,
                roundTrip(kryo, new PartitionedWriterCommitMessage(messages)).getPartitionMessages());
    }
}
