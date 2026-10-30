// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import dev.vortex.api.Session;
import dev.vortex.api.VortexWriter;
import dev.vortex.jni.NativeFiles;
import dev.vortex.relocated.org.apache.arrow.c.ArrowArray;
import dev.vortex.relocated.org.apache.arrow.c.ArrowSchema;
import dev.vortex.relocated.org.apache.arrow.c.Data;
import dev.vortex.relocated.org.apache.arrow.memory.BufferAllocator;
import dev.vortex.relocated.org.apache.arrow.memory.RootAllocator;
import dev.vortex.relocated.org.apache.arrow.vector.FieldVector;
import dev.vortex.relocated.org.apache.arrow.vector.VectorSchemaRoot;
import dev.vortex.spark.VortexSparkSession;
import java.io.IOException;
import java.util.Map;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.write.DataWriter;
import org.apache.spark.sql.connector.write.WriterCommitMessage;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Writes Spark InternalRow data to a Vortex file.
 *
 * <p>Each row is appended straight into a batch of Arrow vectors as it arrives; a full batch is exported through the
 * Arrow C Data Interface and handed to the Vortex writer.
 */
public final class VortexDataWriter implements DataWriter<InternalRow>, AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(VortexDataWriter.class);

    private static final int DEFAULT_BATCH_SIZE = 2048;
    private static final int MIN_BATCH_SIZE = 1;
    private static final int MAX_BATCH_SIZE = 65536; // 64K rows max per batch
    private static final int INITIAL_VECTOR_CAPACITY = 64;

    private final String filePath;
    private final CaseInsensitiveStringMap options;
    private final int batchSize;

    private Session session;
    private VortexWriter vortexWriter;
    private BufferAllocator allocator;
    private VectorSchemaRoot vectorSchemaRoot;
    private ArrowSchema exportedSchema;
    private ArrowFieldWriter[] fieldWriters;
    private int batchRowCount = 0;
    private long recordCount = 0;
    private long bytesWritten = 0;
    private boolean failed = false;
    private boolean fileCreated = false;
    private boolean closed = false;

    /**
     * Creates a new VortexDataWriter.
     *
     * @param filePath the path where the Vortex file will be written
     * @param schema the schema of the data to write
     * @param options additional write options
     */
    VortexDataWriter(String filePath, StructType schema, CaseInsensitiveStringMap options) {
        this.filePath = filePath;
        this.options = options;

        // Get batch size from options with validation
        // Users can set this with: .option("vortex.write.batch.size", "4096")
        int configuredBatchSize =
                options.getInt("vortex.write.batch.size", options.getInt("batch.size", DEFAULT_BATCH_SIZE));
        if (configuredBatchSize < MIN_BATCH_SIZE || configuredBatchSize > MAX_BATCH_SIZE) {
            logger.warn(
                    "Batch size {} is out of valid range [{}, {}], using default: {}",
                    configuredBatchSize,
                    MIN_BATCH_SIZE,
                    MAX_BATCH_SIZE,
                    DEFAULT_BATCH_SIZE);
            this.batchSize = DEFAULT_BATCH_SIZE;
        } else {
            this.batchSize = configuredBatchSize;
            if (this.batchSize != DEFAULT_BATCH_SIZE) {
                logger.debug("Using configured batch size: {}", this.batchSize);
            }
        }

        try {
            this.allocator = new RootAllocator();
            var arrowSchema = SparkToArrowSchema.convert(schema);

            this.session = VortexSparkSession.get(options.asCaseSensitiveMap());
            this.vortexWriter = VortexWriter.builder(session, filePath, arrowSchema, allocator)
                    .options(options.asCaseSensitiveMap())
                    .build();
            this.fileCreated = true;
            this.vectorSchemaRoot = VectorSchemaRoot.create(arrowSchema, allocator);
            // Small start so writers that see few rows hold little memory; Arrow grows on demand.
            // It also keeps the largest capacity a batch needed, so one batch of huge values makes
            // later batches allocate as much.
            int initialCapacity = Math.min(batchSize, INITIAL_VECTOR_CAPACITY);
            for (FieldVector vector : vectorSchemaRoot.getFieldVectors()) {
                vector.setInitialCapacity(initialCapacity);
            }
            StructField[] fields = schema.fields();
            this.fieldWriters = new ArrowFieldWriter[fields.length];
            for (int i = 0; i < fields.length; i++) {
                fieldWriters[i] = ArrowFieldWriter.create(vectorSchemaRoot.getVector(i), fields[i].dataType());
            }
            // The native writer only reads the schema struct, so one export serves every batch.
            this.exportedSchema = ArrowSchema.allocateNew(allocator);
            Data.exportSchema(allocator, arrowSchema, null, exportedSchema);

            logger.debug("Initialized VortexDataWriter for {}", filePath);

        } catch (IOException | RuntimeException e) {
            logger.error("Failed to initialize VortexDataWriter for {}", filePath, e);
            // Spark gets no writer to abort, so release what was opened here, the file included.
            try {
                abort();
            } catch (IOException | RuntimeException cleanup) {
                e.addSuppressed(cleanup);
            }
            throw e instanceof RuntimeException runtime
                    ? runtime
                    : new RuntimeException("Failed to initialize VortexDataWriter", e);
        }
    }

    /**
     * Writes a single row to the Vortex file.
     *
     * <p>The row is consumed before this method returns: Spark reuses the row object for the next call.
     *
     * @param row the row to write
     * @throws IOException if writing fails
     */
    @Override
    public void write(InternalRow row) throws IOException {
        checkNotFailed();
        try {
            if (batchRowCount == 0) {
                // Fresh buffers for every batch: the native side imports the exported buffers zero-copy and
                // releases them from its own thread once a row block is complete, so they cannot be reused.
                vectorSchemaRoot.allocateNew();
            }
            for (int i = 0; i < fieldWriters.length; i++) {
                fieldWriters[i].write(row, i);
            }
            batchRowCount++;
            recordCount++;

            if (batchRowCount >= batchSize) {
                flushBatch();
            }
        } catch (IOException | RuntimeException e) {
            // A failure part-way through a row leaves the batch columns misaligned, so nothing
            // written after it may reach the file: the caller has to abort.
            failed = true;
            throw e;
        }
    }

    private void checkNotFailed() throws IOException {
        if (failed) {
            throw new IOException(
                    "VortexDataWriter for " + filePath + " failed on an earlier write and must be aborted");
        }
    }

    /** Hands the rows appended since the last flush to the Vortex writer. */
    private void flushBatch() throws IOException {
        if (batchRowCount == 0) {
            return;
        }
        vectorSchemaRoot.setRowCount(batchRowCount);

        try (ArrowArray arrowArray = ArrowArray.allocateNew(allocator)) {
            Data.exportVectorSchemaRoot(allocator, vectorSchemaRoot, null, arrowArray);
            try {
                vortexWriter.writeBatch(arrowArray.memoryAddress(), exportedSchema.memoryAddress());
            } catch (IOException | RuntimeException e) {
                // Native code takes the batch, and clears its release callback, only once it starts importing it.
                if (arrowArray.snapshot().release != 0) {
                    arrowArray.release();
                }
                throw e;
            }
        }

        vectorSchemaRoot.clear();
        for (ArrowFieldWriter writer : fieldWriters) {
            writer.reset();
        }
        batchRowCount = 0;
    }

    /** Bytes held by this writer's allocator, including batches the native writer has not released yet. */
    long allocatedBytes() {
        return allocator == null ? 0 : allocator.getAllocatedMemory();
    }

    /**
     * Commits the write operation and returns a commit message.
     *
     * <p>This flushes any remaining rows and closes the Vortex writer.
     *
     * @return a commit message with file information
     * @throws IOException if commit fails
     */
    @Override
    public WriterCommitMessage commit() throws IOException {
        if (failed) {
            // PartitionedVortexDataWriter never aborts a child whose commit threw, so release it here.
            abort();
            checkNotFailed();
        }
        if (!closed) {
            try {
                // Write any remaining rows
                flushBatch();

                // Finalize the file; the summary carries its physical size
                if (vortexWriter != null) {
                    bytesWritten = vortexWriter.finish().fileSize();
                    vortexWriter = null;
                }
                closeArrowResources();
            } catch (IOException | RuntimeException e) {
                // Spark aborts a task whose commit throws, but abort() would then find this writer closed.
                try {
                    abort();
                } catch (IOException | RuntimeException cleanup) {
                    e.addSuppressed(cleanup);
                }
                throw e;
            }

            closeAllocator();

            // Session is the JVM-wide singleton held by VortexSparkSession; we just
            // drop our local handle to it here.
            session = null;

            closed = true;
        }

        return new VortexWriterCommitMessage(filePath, recordCount, bytesWritten);
    }

    /**
     * Aborts the write operation and cleans up resources.
     *
     * <p>This deletes any partially written file.
     *
     * @throws IOException if abort fails
     */
    @Override
    public void abort() throws IOException {
        if (closed) {
            return;
        }
        RuntimeException failure = null;
        try {
            if (vortexWriter != null) {
                try {
                    vortexWriter.close();
                } catch (Exception e) {
                    // Ignore errors during abort
                } finally {
                    vortexWriter = null; // Always null out the reference
                }
            }
            failure = runGuarded(failure, this::closeArrowResources);
            failure = runGuarded(failure, this::closeAllocator);
            // Closing the native writer finalizes a valid file of the batches flushed so far, and a retried
            // attempt writes its own, so this one must go.
            if (fileCreated) {
                try {
                    deleteFile(filePath, options);
                } catch (RuntimeException e) {
                    logger.warn("Failed to delete the file of an aborted write attempt: {}", filePath, e);
                }
            }
        } finally {
            session = null;
            closed = true;
        }
        if (failure != null) {
            throw failure;
        }
    }

    /** Deletes a file a writer created. Only the native layer resolves {@code file://} and {@code s3://} URLs. */
    static void deleteFile(String path, CaseInsensitiveStringMap options) {
        Map<String, String> properties = options.asCaseSensitiveMap();
        NativeFiles.delete(VortexSparkSession.get(properties), new String[] {path}, properties);
    }

    private void closeArrowResources() {
        RuntimeException failure = runGuarded(null, () -> {
            if (vectorSchemaRoot != null) {
                VectorSchemaRoot root = vectorSchemaRoot;
                vectorSchemaRoot = null;
                root.close();
            }
        });
        failure = runGuarded(failure, () -> {
            if (exportedSchema != null) {
                ArrowSchema schema = exportedSchema;
                exportedSchema = null;
                // close() frees only the struct; the children the export allocated are freed by the release
                // callback, which the native side never calls because it only borrows the schema.
                schema.release();
                schema.close();
            }
        });
        if (failure != null) {
            throw failure;
        }
    }

    // Rust releases imported batches on its own thread, so some may still be outstanding here.
    private void closeAllocator() {
        if (allocator != null) {
            BufferAllocator closing = allocator;
            allocator = null;
            try {
                closing.close();
            } catch (IllegalStateException e) {
                logger.debug("Allocator closed with outstanding FFI allocations: {}", e.getMessage());
            }
        }
    }

    /** Runs {@code step}, adding its failure to {@code failure}, so that later cleanup steps still run. */
    private static RuntimeException runGuarded(RuntimeException failure, Runnable step) {
        try {
            step.run();
        } catch (RuntimeException e) {
            if (failure == null) {
                return e;
            }
            failure.addSuppressed(e);
        }
        return failure;
    }

    /**
     * Closes the writer and releases resources.
     *
     * <p>This method ensures resources are cleaned up even if commit() or abort() were not called, making the class
     * safe for use with try-with-resources.
     */
    @Override
    public void close() throws IOException {
        if (!closed) {
            logger.warn("VortexDataWriter.close() called without commit() or abort() - cleaning up");
            try {
                abort();
            } catch (IOException e) {
                logger.error("Error during cleanup in close()", e);
                // Suppress the exception as we're already in close()
            }
        }
    }
}
