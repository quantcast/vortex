// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import java.io.Serializable;

/** Immutable statistics for a successfully closed {@link VortexOutputWriter}, serializable for collection from tasks. */
public final class VortexOutputWriterSummary implements Serializable {
    private static final long serialVersionUID = 1L;

    private final long fileSize;
    private final long rowCount;
    private final long batchCount;
    private final long peakBatchArrowBytes;
    private final long peakBufferedBytes;

    VortexOutputWriterSummary(
            long fileSize, long rowCount, long batchCount, long peakBatchArrowBytes, long peakBufferedBytes) {
        this.fileSize = fileSize;
        this.rowCount = rowCount;
        this.batchCount = batchCount;
        this.peakBatchArrowBytes = peakBatchArrowBytes;
        this.peakBufferedBytes = peakBufferedBytes;
    }

    /** Exact size of the completed file in bytes. */
    public long fileSize() {
        return fileSize;
    }

    /** Exact number of rows written. */
    public long rowCount() {
        return rowCount;
    }

    /** Number of nonempty Arrow batches submitted, including the final partial batch. */
    public long batchCount() {
        return batchCount;
    }

    /**
     * Maximum sum of Arrow buffer capacities in one submitted batch, including nested buffers.
     *
     * <p>Excludes other batches retained by native code and temporary export metadata.
     */
    public long peakBatchArrowBytes() {
        return peakBatchArrowBytes;
    }

    /**
     * Maximum native {@code bufferedBytes()} observed immediately after submitting each batch.
     *
     * <p>Sampling can miss intermediate peaks, and this counter does not measure all retained Arrow or native memory.
     */
    public long peakBufferedBytes() {
        return peakBufferedBytes;
    }
}
