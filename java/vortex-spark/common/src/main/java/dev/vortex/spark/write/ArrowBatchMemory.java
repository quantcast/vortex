// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import dev.vortex.relocated.org.apache.arrow.memory.AllocationListener;
import dev.vortex.relocated.org.apache.arrow.memory.ArrowBuf;
import dev.vortex.relocated.org.apache.arrow.memory.ReferenceManager;
import dev.vortex.relocated.org.apache.arrow.vector.BaseFixedWidthVector;
import dev.vortex.relocated.org.apache.arrow.vector.BaseVariableWidthVector;
import dev.vortex.relocated.org.apache.arrow.vector.FieldVector;
import dev.vortex.relocated.org.apache.arrow.vector.VectorSchemaRoot;
import dev.vortex.relocated.org.apache.arrow.vector.complex.ListVector;
import dev.vortex.relocated.org.apache.arrow.vector.complex.StructVector;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/** Measures the open batch independently of older batches retained or released by native code. */
final class ArrowBatchMemory implements AllocationListener {
    private final AtomicLong allocatedBytes = new AtomicLong();
    // -1 until the open batch is first measured, since no total of allocations is negative.
    private long allocatedBytesAtLastMeasure = -1;
    private VectorSchemaRoot root;
    private FieldVector[] vectors;
    private long bytes;

    @Override
    public void onAllocation(long size) {
        allocatedBytes.addAndGet(size);
    }

    /**
     * Whether the open batch holds at least {@code threshold} bytes, measuring it only once it might. Its buffers
     * only gain bytes through new allocations, and the listener hears each one's size as its buffer reports it, so the
     * batch holds at most the last measurement plus what was allocated since. That is doubled in case an allocation
     * manager holds more than it reports; it costs at most a measurement more a batch.
     */
    boolean reached(VectorSchemaRoot root, long threshold) {
        // The last measurement only bounds the root it was taken of.
        if (root == this.root && bytes + 2 * (allocatedBytes.get() - allocatedBytesAtLastMeasure) < threshold) {
            return false;
        }
        return bytes(root) >= threshold;
    }

    long bytes(VectorSchemaRoot root) {
        // Arrow allocates nothing for an empty request, so between allocations the batch holds the same buffers.
        long allocated = allocatedBytes.get();
        if (root != this.root) {
            this.root = root;
            this.vectors = flatten(root);
            allocatedBytesAtLastMeasure = -1;
        }
        if (allocated != allocatedBytesAtLastMeasure) {
            long total = 0;
            for (FieldVector vector : vectors) {
                total += bytes(vector);
            }
            bytes = total;
            allocatedBytesAtLastMeasure = allocated;
        }
        return bytes;
    }

    /** Starts the next batch, which is measured afresh. */
    void reset() {
        bytes = 0;
        allocatedBytesAtLastMeasure = -1;
    }

    /** Every vector under {@code root}, parents before children. A root's vectors stay the same across batches. */
    private static FieldVector[] flatten(VectorSchemaRoot root) {
        List<FieldVector> vectors = new ArrayList<>();
        for (FieldVector vector : root.getFieldVectors()) {
            flatten(vector, vectors);
        }
        return vectors.toArray(FieldVector[]::new);
    }

    private static void flatten(FieldVector vector, List<FieldVector> vectors) {
        vectors.add(vector);
        for (FieldVector child : vector.getChildrenFromFields()) {
            flatten(child, vectors);
        }
    }

    /**
     * The allocations {@code vector}'s own buffers hold. Arrow slices a fixed-width vector's validity and data, and a
     * variable-width vector's validity and offsets, from one allocation, so a buffer sharing the validity's is counted
     * once. Buffers never share an allocation across vectors.
     */
    private static long bytes(FieldVector vector) {
        // getFieldBuffers() finalizes variable-width offsets using valueCount, which is still zero mid-batch.
        ReferenceManager validity = vector.getValidityBuffer().getReferenceManager();
        long total = validity.getSize();
        if (vector instanceof BaseFixedWidthVector) {
            total += unlessShared(vector.getDataBuffer(), validity);
        } else if (vector instanceof BaseVariableWidthVector) {
            total += unlessShared(vector.getDataBuffer(), validity);
            total += unlessShared(vector.getOffsetBuffer(), validity);
        } else if (vector instanceof ListVector) {
            total += unlessShared(vector.getOffsetBuffer(), validity);
        } else if (!(vector instanceof StructVector)) {
            throw new IllegalArgumentException(
                    "Unsupported writer vector: " + vector.getClass().getName());
        }
        return total;
    }

    private static long unlessShared(ArrowBuf buffer, ReferenceManager validity) {
        ReferenceManager manager = buffer.getReferenceManager();
        return manager == validity ? 0 : manager.getSize();
    }
}
