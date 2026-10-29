// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import java.math.BigInteger;

/**
 * The rows that one byte split of a Vortex file reads.
 *
 * <p>Spark cuts a file into contiguous byte ranges, but the native scan addresses rows. Each byte offset maps to a row
 * at the same fraction of the file, rounded down to {@link #ROW_BLOCK}, and a split reads the rows from its start
 * offset's row up to its end offset's. Neighbouring splits share an offset, so they share a row boundary, and every row
 * is read exactly once whatever offsets Spark picks.
 */
record SplitRows(long begin, long end) {
    /**
     * Row block of the default write strategy, which ends chunks on its multiples. A boundary elsewhere would cut a
     * chunk in every column, and both neighbouring splits would decode it. Correctness does not depend on it.
     */
    static final long ROW_BLOCK = 8192;

    /** The rows that bytes {@code [start, start + length)} of a {@code fileSize}-byte file read. */
    static SplitRows of(long start, long length, long fileSize, long rows) {
        return new SplitRows(rowAt(start, fileSize, rows), rowAt(start + length, fileSize, rows));
    }

    long count() {
        return end - begin;
    }

    private static long rowAt(long offset, long fileSize, long rows) {
        if (offset >= fileSize) {
            return rows;
        }
        if (offset <= 0) {
            return 0;
        }
        long row = BigInteger.valueOf(offset)
                .multiply(BigInteger.valueOf(rows))
                .divide(BigInteger.valueOf(fileSize))
                .longValueExact();
        return row / ROW_BLOCK * ROW_BLOCK;
    }
}
