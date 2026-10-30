// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import dev.vortex.jni.NativeRuntime;
import java.io.Serializable;
import java.util.Map;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.write.DataWriter;
import org.apache.spark.sql.connector.write.DataWriterFactory;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Factory for creating VortexDataWriter instances on Spark executors.
 *
 * <p>This factory is serialized and sent to executors where it creates data writers for each task. When partition
 * transforms are specified, it creates partitioned writers that organize output into Hive-style partition directories.
 */
public final class VortexDataWriterFactory implements DataWriterFactory, Serializable {

    private static final Logger log = LoggerFactory.getLogger(VortexDataWriterFactory.class);

    private final String outputUri;
    private final StructType schema;
    // Store options as a serializable Map instead of CaseInsensitiveStringMap
    private final Map<String, String> options;
    private final PartitionedVortexDataWriter.ResolvedTransform[] resolvedTransforms;
    private final int workerThreads;

    /**
     * Creates a new VortexDataWriterFactory.
     *
     * @param outputUri the base path where Vortex files will be written
     * @param schema the schema of the data to write
     * @param options additional write options
     * @param resolvedTransforms pre-resolved partition transforms (may be empty)
     */
    VortexDataWriterFactory(
            String outputUri,
            StructType schema,
            Map<String, String> options,
            PartitionedVortexDataWriter.ResolvedTransform[] resolvedTransforms) {
        this.outputUri = outputUri;
        this.schema = schema;
        this.options = options;
        this.resolvedTransforms = resolvedTransforms;
        // Options executors parse must fail here on the driver, before overwrite deletes the old files.
        CaseInsensitiveStringMap parsed = new CaseInsensitiveStringMap(options);
        this.workerThreads = parseWorkerThreads(parsed);
        parsed.getInt("vortex.write.batch.size", parsed.getInt("batch.size", 0));
    }

    private static int parseWorkerThreads(CaseInsensitiveStringMap options) {
        String value = options.getOrDefault("vortex.workerThreads", "4");
        int threads;
        try {
            threads = Integer.parseInt(value.trim());
        } catch (NumberFormatException e) {
            threads = -1;
        }
        if (threads < 0) {
            throw new IllegalArgumentException("vortex.workerThreads must be a non-negative integer, got: " + value);
        }
        return threads;
    }

    /**
     * Creates a new data writer for a specific partition and task.
     *
     * <p>Each task writes its data to a separate Vortex file to avoid conflicts. When partition transforms are
     * configured, returns a {@link PartitionedVortexDataWriter} that creates Hive-style partition directories.
     *
     * @param partitionId the partition ID
     * @param taskId the task ID
     * @return a new DataWriter instance
     */
    @Override
    public DataWriter<InternalRow> createWriter(int partitionId, long taskId) {
        log.debug("Creating writer for partition={} task={}", partitionId, taskId);

        CaseInsensitiveStringMap optionsMap = new CaseInsensitiveStringMap(options);
        // Without workers the native encoder only runs while this task thread blocks on it.
        NativeRuntime.setWorkerThreads(workerThreads);

        if (resolvedTransforms.length > 0) {
            log.debug("Creating partitioned writer with {} transforms", resolvedTransforms.length);
            return new PartitionedVortexDataWriter(
                    outputUri, schema, optionsMap, resolvedTransforms, partitionId, taskId);
        }

        // Non-partitioned write: single file per task
        String fileName = String.format("part-%05d-%d.vortex", partitionId, taskId);
        String fileUri;
        if (outputUri.endsWith("/")) {
            fileUri = outputUri + fileName;
        } else {
            fileUri = outputUri + "/" + fileName;
        }

        log.debug("Output file: {}", fileUri);
        return new VortexDataWriter(fileUri, schema, optionsMap);
    }
}
