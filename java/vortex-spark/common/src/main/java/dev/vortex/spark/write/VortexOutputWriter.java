// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import dev.vortex.api.Session;
import dev.vortex.api.VortexWriteSummary;
import dev.vortex.api.VortexWriter;
import dev.vortex.io.NativeWritable;
import dev.vortex.relocated.org.apache.arrow.memory.ArrowBuf;
import dev.vortex.relocated.org.apache.arrow.memory.BufferAllocator;
import dev.vortex.relocated.org.apache.arrow.memory.RootAllocator;
import dev.vortex.relocated.org.apache.arrow.vector.BaseFixedWidthVector;
import dev.vortex.relocated.org.apache.arrow.vector.FieldVector;
import dev.vortex.relocated.org.apache.arrow.vector.VectorSchemaRoot;
import dev.vortex.relocated.org.apache.arrow.vector.complex.StructVector;
import dev.vortex.spark.VortexOptions;
import dev.vortex.spark.VortexSparkSession;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import org.apache.spark.TaskContext;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.execution.datasources.OutputWriter;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Writes Spark InternalRow data to a Vortex file.
 *
 * <p>Each row is appended straight into a batch of Arrow vectors as it arrives; a full batch is handed to the Vortex
 * writer, which retains each column's buffers separately so it can free them one by one.
 */
public final class VortexOutputWriter extends OutputWriter {
    private static final Logger logger = LoggerFactory.getLogger(VortexOutputWriter.class);

    /** Option sizing the row batch converted to Arrow and handed to Vortex at a time. */
    public static final String BATCH_SIZE_OPTION = "batch.size";

    /** Option sizing the write batch for Vortex alone, overriding {@value #BATCH_SIZE_OPTION}. */
    public static final String WRITE_BATCH_SIZE_OPTION = "vortex.write.batch.size";

    /** Optional positive byte threshold for the open batch's Arrow allocations, checked after each row. */
    public static final String WRITE_BATCH_BYTES_OPTION = "vortex.write.batch.bytes";

    /**
     * Option storing list columns in the unstable list layout, so a read of some fields of array elements skips the
     * others. Off by default; future versions may not read files written with it.
     */
    public static final String LIST_LAYOUT_OPTION = "vortex.listLayout";

    /** Controls Arrow sizing resets after row-limit flushes; byte-triggered flushes always reset all hints. */
    public static final String WRITE_RESET_CAPACITY_OPTION = "vortex.write.reset.capacity";

    enum CapacityReset {
        AUTO,
        ALWAYS,
        NEVER
    }

    private static final int DEFAULT_BATCH_SIZE = 8192;
    private static final int MIN_BATCH_SIZE = 1;
    private static final int MAX_BATCH_SIZE = 65536; // 64K rows max per batch
    private static final int INITIAL_VECTOR_CAPACITY = 64;
    private static final long AUTO_CAPACITY_MULTIPLIER = 4;

    private final String filePath;
    private final VortexOptions options;
    private final int batchSize;
    private final long batchBytes;
    private final CapacityReset capacityReset;
    private final ArrowBatchMemory batchMemory;
    private NativeWritable writable;

    private Session session;
    private VortexWriter vortexWriter;
    private BufferAllocator allocator;
    private VectorSchemaRoot vectorSchemaRoot;
    private ArrowFieldWriter<?>[] fieldWriters;
    private int batchRowCount = 0;
    private long batchCount;
    private long peakBatchArrowBytes;
    private long peakBufferedBytes;
    private VortexOutputWriterSummary summary;
    private Exception writeFailure;
    private boolean closed = false;

    /**
     * Creates a writer for the task path assigned by Spark's commit protocol.
     *
     * @param filePath the path where the Vortex file will be written
     * @param schema the schema of the data to write
     * @param options additional write options
     */
    public VortexOutputWriter(String filePath, StructType schema, VortexOptions options, NativeWritable writable) {
        this(filePath, schema, options, writable, Map.of());
    }

    /**
     * Creates a writer that also stores {@code metadata} in the file footer, for callers that write files outside
     * Spark's commit protocol.
     *
     * @param metadata footer segments, read back verbatim by {@code NativeFiles.readMetadata}
     */
    public VortexOutputWriter(
            String filePath,
            StructType schema,
            VortexOptions options,
            NativeWritable writable,
            Map<String, byte[]> metadata) {
        this.filePath = filePath;
        this.options = options;
        this.writable = writable;

        this.batchSize = configuredBatchSize(options);
        this.batchBytes = configuredBatchBytes(options);
        this.capacityReset = configuredCapacityReset(options);
        this.batchMemory = batchBytes == 0 ? null : new ArrowBatchMemory();

        try {
            this.allocator = batchMemory == null ? new RootAllocator() : new RootAllocator(batchMemory, Long.MAX_VALUE);
            var arrowSchema = SparkToArrowSchema.convert(schema);

            this.vectorSchemaRoot = VectorSchemaRoot.create(arrowSchema, allocator);
            // Small start so writers that see few rows hold little memory; Arrow grows on demand.
            resetInitialCapacity(true);
            StructField[] fields = schema.fields();
            this.fieldWriters = new ArrowFieldWriter<?>[fields.length];
            for (int i = 0; i < fields.length; i++) {
                fieldWriters[i] = ArrowFieldWriter.create(vectorSchemaRoot.getVector(i), fields[i].dataType());
            }
            this.session = VortexSparkSession.get(options);
            this.vortexWriter = VortexWriter.builder(session, writable, arrowSchema, allocator)
                    .listLayout(listLayout(options))
                    .metadata(metadata)
                    .build();

            logger.debug("Initialized VortexOutputWriter for {}", filePath);

        } catch (IOException e) {
            closeAfterInitializationFailure(e);
            throw new UncheckedIOException("Failed to initialize VortexOutputWriter for " + filePath, e);
        } catch (RuntimeException e) {
            closeAfterInitializationFailure(e);
            throw e;
        }
    }

    /**
     * Whether {@value #LIST_LAYOUT_OPTION} is on.
     *
     * @throws IllegalArgumentException if the value is neither {@code true} nor {@code false}
     */
    static boolean listLayout(VortexOptions options) {
        return options.getBoolean(LIST_LAYOUT_OPTION, false);
    }

    /**
     * The batch size this writer was asked for.
     *
     * <p>{@value #BATCH_SIZE_OPTION} is the generic name, shared with whatever else a job writes.
     * {@value #WRITE_BATCH_SIZE_OPTION} overrides it, so a job that sets one batch size across formats can still say
     * something different for Vortex.
     *
     * @throws IllegalArgumentException if the value is not an integer from 1 to 65536
     */
    static int configuredBatchSize(VortexOptions options) {
        String key = options.get(WRITE_BATCH_SIZE_OPTION).isPresent() ? WRITE_BATCH_SIZE_OPTION : BATCH_SIZE_OPTION;
        int batchSize = options.getInt(key, DEFAULT_BATCH_SIZE);
        if (batchSize < MIN_BATCH_SIZE || batchSize > MAX_BATCH_SIZE) {
            throw new IllegalArgumentException(String.format(
                    Locale.ROOT, "%s must be from %d to %d, got %d", key, MIN_BATCH_SIZE, MAX_BATCH_SIZE, batchSize));
        }
        return batchSize;
    }

    /** Returns the positive batch byte threshold, or zero when unset. */
    static long configuredBatchBytes(VortexOptions options) {
        if (options.get(WRITE_BATCH_BYTES_OPTION).isEmpty()) {
            return 0;
        }
        long bytes = options.getLong(WRITE_BATCH_BYTES_OPTION, 0);
        if (bytes <= 0) {
            throw new IllegalArgumentException(
                    WRITE_BATCH_BYTES_OPTION + " must be a positive number of bytes, got " + bytes);
        }
        return bytes;
    }

    /** Resolves the capacity policy before allocating Arrow buffers or opening native resources. */
    static CapacityReset configuredCapacityReset(VortexOptions options) {
        String value = options.get(WRITE_RESET_CAPACITY_OPTION).orElse("auto");
        return switch (value.trim().toLowerCase(Locale.ROOT)) {
            case "auto" -> CapacityReset.AUTO;
            case "true" -> CapacityReset.ALWAYS;
            case "false" -> CapacityReset.NEVER;
            default ->
                throw new IllegalArgumentException(
                        WRITE_RESET_CAPACITY_OPTION + " must be auto, true, or false, got '" + value + "'");
        };
    }

    /**
     * Writes a single row to the Vortex file.
     *
     * <p>The row is consumed before this method returns: Spark reuses the row object for the next call.
     *
     * @param row the row to write
     */
    @Override
    public void write(InternalRow row) {
        if (closed) {
            throw new IllegalStateException("VortexOutputWriter for " + filePath + " is already closed");
        }
        if (writeFailure != null) {
            throw new IllegalStateException(
                    "VortexOutputWriter for " + filePath + " failed on an earlier write", writeFailure);
        }
        try {
            if (batchRowCount == 0) {
                // Fresh buffers for every batch: the native side keeps its own reference to each exported
                // buffer and releases it from its own thread once it is done with it, so they cannot be reused.
                // ArrowFieldWriter also needs them zeroed, since it leaves null slots untouched.
                vectorSchemaRoot.allocateNew();
            }
            for (int i = 0; i < fieldWriters.length; i++) {
                fieldWriters[i].write(row, i);
            }
            batchRowCount++;
            boolean reachedBytes = batchMemory != null && batchMemory.bytes(vectorSchemaRoot) >= batchBytes;
            if (batchRowCount >= batchSize || reachedBytes) {
                writeBatch(reachedBytes);
            }
        } catch (IOException e) {
            writeFailure = e;
            throw new UncheckedIOException("Failed to write a Vortex batch to " + filePath, e);
        } catch (RuntimeException e) {
            // A failure part-way through a row leaves the batch columns misaligned, so nothing
            // written after it may reach the file.
            writeFailure = e;
            throw e;
        }
    }

    private void resetInitialCapacity(boolean includeFixedWidth) {
        int initialCapacity = Math.min(batchSize, INITIAL_VECTOR_CAPACITY);
        for (FieldVector vector : vectorSchemaRoot.getFieldVectors()) {
            resetInitialCapacity(vector, initialCapacity, includeFixedWidth);
        }
    }

    private static void resetInitialCapacity(FieldVector vector, int initialCapacity, boolean includeFixedWidth) {
        if (!includeFixedWidth && vector instanceof StructVector) {
            // Structs preserve row counts. Evaluate list/map subtrees together because they multiply element counts.
            for (FieldVector child : vector.getChildrenFromFields()) {
                resetInitialCapacity(child, initialCapacity, false);
            }
        } else if (includeFixedWidth || (!(vector instanceof BaseFixedWidthVector) && capacityExcess(vector) > 0)) {
            vector.setInitialCapacity(initialCapacity);
        }
    }

    private static long capacityExcess(FieldVector vector) {
        // Keep ordinary allocation rounding; shrink only when the subtree uses less than a quarter of its buffers.
        long excess = 0;
        for (ArrowBuf buffer : vector.getFieldBuffers()) {
            excess += buffer.capacity() - AUTO_CAPACITY_MULTIPLIER * buffer.readableBytes();
        }
        for (FieldVector child : vector.getChildrenFromFields()) {
            excess += capacityExcess(child);
        }
        return excess;
    }

    /** Hands the rows appended since the last batch to the Vortex writer. */
    private void writeBatch(boolean reachedBytes) throws IOException {
        if (batchRowCount == 0) {
            return;
        }
        vectorSchemaRoot.setRowCount(batchRowCount);
        peakBatchArrowBytes = Math.max(peakBatchArrowBytes, arrowBytes(vectorSchemaRoot.getFieldVectors()));

        vortexWriter.writeBatch(vectorSchemaRoot, allocator);
        batchCount++;
        peakBufferedBytes = Math.max(peakBufferedBytes, vortexWriter.bufferedBytes());

        if (reachedBytes || capacityReset != CapacityReset.NEVER) {
            // A byte-triggered flush must also discard fixed-width hints to preserve its smaller batch budget.
            resetInitialCapacity(capacityReset == CapacityReset.ALWAYS || reachedBytes);
        }
        vectorSchemaRoot.clear();
        for (ArrowFieldWriter<?> writer : fieldWriters) {
            writer.reset();
        }
        batchRowCount = 0;
        if (batchMemory != null) {
            batchMemory.reset();
        }
    }

    private static long arrowBytes(List<FieldVector> vectors) {
        long bytes = 0;
        for (FieldVector vector : vectors) {
            for (ArrowBuf buffer : vector.getFieldBuffers()) {
                bytes += buffer.capacity();
            }
            bytes += arrowBytes(vector.getChildrenFromFields());
        }
        return bytes;
    }

    /**
     * Flushes the last batch and finalizes the file.
     *
     * <p>After a failed {@link #write}, or in a task that has already failed, the last batch is dropped because the
     * file is never committed. In both cases this releases everything and logs what fails instead of throwing, because
     * Spark's abort closes a task's writers one after another and stops at the first that throws.
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        boolean aborting = writeFailure != null || taskFailed();
        IOException failure = null;
        VortexWriteSummary fileSummary = null;
        if (!aborting) {
            try {
                writeBatch(false);
            } catch (IOException e) {
                failure = e;
            } catch (RuntimeException e) {
                failure = new IOException("Failed to write the last Vortex batch", e);
            }
        }
        if (vortexWriter != null) {
            try {
                fileSummary = vortexWriter.finish();
            } catch (IOException e) {
                failure = addFailure(failure, e);
            } finally {
                vortexWriter = null;
            }
        }

        failure = closeArrowResources(failure);
        if (writable != null) {
            try {
                writable.close();
            } catch (IOException e) {
                failure = addFailure(failure, e);
            } finally {
                writable = null;
            }
        }
        session = null;
        closed = true;
        if (aborting) {
            if (failure != null) {
                logger.warn("Failed to close the writer for {} in a failed task", filePath, failure);
            }
            return;
        }
        if (failure != null) {
            throw new UncheckedIOException("Failed to close Vortex output " + filePath, failure);
        }
        summary = new VortexOutputWriterSummary(
                fileSummary.fileSize(), fileSummary.rowCount(), batchCount, peakBatchArrowBytes, peakBufferedBytes);
    }

    /**
     * Returns the completed file's size, row count and observed batch memory usage without reopening it.
     *
     * @throws IllegalStateException unless {@link #close()} completed successfully outside a failed task
     */
    public VortexOutputWriterSummary summary() {
        if (summary == null) {
            throw new IllegalStateException(
                    "A write summary is only available after a successful close of " + filePath);
        }
        return summary;
    }

    @Override
    public String path() {
        return filePath;
    }

    /** Spark marks a task failed before its abort closes the task's writers. */
    private static boolean taskFailed() {
        TaskContext context = TaskContext.get();
        return context != null && context.isFailed();
    }

    /** Package-private so tests can see what close leaves allocated. */
    BufferAllocator allocator() {
        return allocator;
    }

    /** Package-private so tests can observe batch boundaries without depending on native scheduling. */
    int batchRowCount() {
        return batchRowCount;
    }

    private IOException closeArrowResources(IOException failure) {
        if (vectorSchemaRoot != null) {
            try {
                vectorSchemaRoot.close();
            } catch (RuntimeException e) {
                failure = addFailure(failure, new IOException("Failed to close VectorSchemaRoot", e));
            } finally {
                vectorSchemaRoot = null;
            }
        }
        if (allocator != null) {
            long leftover = allocator.getAllocatedMemory();
            if (leftover == 0) {
                allocator.close();
            } else {
                // Left open: a batch the native side releases late must still be freed, and must not fail the task.
                logger.warn(
                        "{} bytes of Arrow memory were still allocated when the writer for {} closed",
                        leftover,
                        filePath);
            }
            allocator = null;
        }
        return failure;
    }

    private void closeAfterInitializationFailure(Exception failure) {
        IOException closeFailure = closeArrowResources(null);
        if (closeFailure != null) {
            failure.addSuppressed(closeFailure);
        }
        try {
            writable.close();
        } catch (IOException e) {
            failure.addSuppressed(e);
        }
        writable = null;
    }

    private static IOException addFailure(IOException failure, IOException additional) {
        if (failure == null) {
            return additional;
        }
        failure.addSuppressed(additional);
        return failure;
    }
}
