// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import dev.vortex.api.Session;
import dev.vortex.api.VortexWriter;
import dev.vortex.io.NativeWritable;
import dev.vortex.relocated.org.apache.arrow.c.ArrowArray;
import dev.vortex.relocated.org.apache.arrow.c.ArrowSchema;
import dev.vortex.relocated.org.apache.arrow.c.Data;
import dev.vortex.relocated.org.apache.arrow.memory.BufferAllocator;
import dev.vortex.relocated.org.apache.arrow.memory.RootAllocator;
import dev.vortex.relocated.org.apache.arrow.vector.FieldVector;
import dev.vortex.relocated.org.apache.arrow.vector.VectorSchemaRoot;
import dev.vortex.spark.VortexOptions;
import dev.vortex.spark.VortexSparkSession;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Locale;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.execution.datasources.OutputWriter;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Writes Spark InternalRow data to a Vortex file.
 *
 * <p>Each row is appended straight into a batch of Arrow vectors as it arrives; a full batch is exported through the
 * Arrow C Data Interface and handed to the Vortex writer.
 */
public final class VortexOutputWriter extends OutputWriter {
    private static final Logger logger = LoggerFactory.getLogger(VortexOutputWriter.class);

    /** Option sizing the row batch converted to Arrow and handed to Vortex at a time. */
    public static final String BATCH_SIZE_OPTION = "batch.size";

    /** Option sizing the write batch for Vortex alone, overriding {@value #BATCH_SIZE_OPTION}. */
    public static final String WRITE_BATCH_SIZE_OPTION = "vortex.write.batch.size";

    /**
     * Option storing list columns in the unstable list layout, so a read of some fields of array elements skips the
     * others. Off by default; future versions may not read files written with it.
     */
    public static final String LIST_LAYOUT_OPTION = "vortex.listLayout";

    private static final int DEFAULT_BATCH_SIZE = 2048;
    private static final int MIN_BATCH_SIZE = 1;
    private static final int MAX_BATCH_SIZE = 65536; // 64K rows max per batch
    private static final int INITIAL_VECTOR_CAPACITY = 64;

    private final String filePath;
    private final VortexOptions options;
    private final int batchSize;
    private NativeWritable writable;

    private Session session;
    private VortexWriter vortexWriter;
    private BufferAllocator allocator;
    private VectorSchemaRoot vectorSchemaRoot;
    private ArrowSchema exportedSchema;
    private ArrowFieldWriter[] fieldWriters;
    private int batchRowCount = 0;
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
        this.filePath = filePath;
        this.options = options;
        this.writable = writable;

        this.batchSize = configuredBatchSize(options);

        try {
            this.allocator = new RootAllocator();
            var arrowSchema = SparkToArrowSchema.convert(schema);

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
            // The native writer only borrows the schema, so one export serves every batch.
            this.exportedSchema = ArrowSchema.allocateNew(allocator);
            Data.exportSchema(allocator, arrowSchema, null, exportedSchema);

            this.session = VortexSparkSession.get(options);
            this.vortexWriter = VortexWriter.builder(session, writable, arrowSchema, allocator)
                    .listLayout(listLayout(options))
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

    /**
     * Writes a single row to the Vortex file.
     *
     * <p>The row is consumed before this method returns: Spark reuses the row object for the next call.
     *
     * @param row the row to write
     */
    @Override
    public void write(InternalRow row) {
        if (writeFailure != null) {
            throw new IllegalStateException(
                    "VortexOutputWriter for " + filePath + " failed on an earlier write", writeFailure);
        }
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
            if (batchRowCount >= batchSize) {
                writeBatch();
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

    /** Hands the rows appended since the last batch to the Vortex writer. */
    private void writeBatch() throws IOException {
        if (batchRowCount == 0) {
            return;
        }
        vectorSchemaRoot.setRowCount(batchRowCount);

        try (ArrowArray arrowArray = ArrowArray.allocateNew(allocator)) {
            try {
                Data.exportVectorSchemaRoot(allocator, vectorSchemaRoot, null, arrowArray);
                vortexWriter.writeBatch(arrowArray.memoryAddress(), exportedSchema.memoryAddress());
            } finally {
                // The native writer takes the batch over; one it never reached is still ours to release.
                if (arrowArray.snapshot().release != 0) {
                    arrowArray.release();
                }
            }
        }

        vectorSchemaRoot.clear();
        for (ArrowFieldWriter writer : fieldWriters) {
            writer.reset();
        }
        batchRowCount = 0;
    }

    /**
     * Flushes the last batch and finalizes the file.
     *
     * <p>After a failed {@link #write}, the rows appended since the last batch are dropped and this throws once every
     * resource is released, so the incomplete file is never committed.
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        IOException failure = null;
        if (writeFailure == null) {
            try {
                writeBatch();
            } catch (IOException e) {
                failure = e;
            } catch (RuntimeException e) {
                failure = new IOException("Failed to write the last Vortex batch", e);
            }
        }
        if (vortexWriter != null) {
            try {
                vortexWriter.finish();
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
        if (writeFailure != null) {
            IllegalStateException incomplete = new IllegalStateException(
                    "VortexOutputWriter for " + filePath + " failed on an earlier write", writeFailure);
            if (failure != null) {
                incomplete.addSuppressed(failure);
            }
            throw incomplete;
        }
        if (failure != null) {
            throw new UncheckedIOException("Failed to close Vortex output " + filePath, failure);
        }
    }

    @Override
    public String path() {
        return filePath;
    }

    /** Package-private so tests can see what close leaves allocated. */
    BufferAllocator allocator() {
        return allocator;
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
        if (exportedSchema != null) {
            // close() frees only the struct; the children the export allocated are freed by the release
            // callback, which the native side never calls because it only borrows the schema.
            if (exportedSchema.snapshot().release != 0) {
                exportedSchema.release();
            }
            exportedSchema.close();
            exportedSchema = null;
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
