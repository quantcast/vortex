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
import java.util.IdentityHashMap;
import java.util.concurrent.atomic.AtomicLong;

/** Measures the open batch independently of older batches retained or released by native code. */
final class ArrowBatchMemory implements AllocationListener {
    private final AtomicLong allocations = new AtomicLong();
    private final IdentityHashMap<ReferenceManager, Boolean> seen = new IdentityHashMap<>();
    private long measuredAllocations = -1;
    private long bytes;

    @Override
    public void onAllocation(long size) {
        allocations.incrementAndGet();
    }

    long bytes(VectorSchemaRoot root) {
        long currentAllocations = allocations.get();
        // Between buffer growths the batch owns the same allocations; do not walk a wide schema for every row.
        if (currentAllocations != measuredAllocations) {
            bytes = 0;
            try {
                for (FieldVector vector : root.getFieldVectors()) {
                    bytes += bytes(vector);
                }
            } finally {
                seen.clear();
            }
            measuredAllocations = currentAllocations;
        }
        return bytes;
    }

    void reset() {
        measuredAllocations = -1;
        bytes = 0;
    }

    private long bytes(FieldVector vector) {
        // getFieldBuffers() finalizes variable-width offsets using valueCount, which is still zero mid-batch.
        long total = bytes(vector.getValidityBuffer());
        if (vector instanceof BaseFixedWidthVector) {
            total += bytes(vector.getDataBuffer());
        } else if (vector instanceof BaseVariableWidthVector) {
            total += bytes(vector.getDataBuffer());
            total += bytes(vector.getOffsetBuffer());
        } else if (vector instanceof ListVector) {
            total += bytes(vector.getOffsetBuffer());
        } else if (!(vector instanceof StructVector)) {
            throw new IllegalArgumentException(
                    "Unsupported writer vector: " + vector.getClass().getName());
        }
        for (FieldVector child : vector.getChildrenFromFields()) {
            total += bytes(child);
        }
        return total;
    }

    private long bytes(ArrowBuf buffer) {
        ReferenceManager manager = buffer.getReferenceManager();
        // Fixed-width vectors can slice validity and data out of one allocation.
        return seen.put(manager, Boolean.TRUE) == null ? manager.getSize() : 0;
    }
}
