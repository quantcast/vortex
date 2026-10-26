// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static com.google.common.base.Preconditions.checkArgument;

import dev.vortex.spark.VortexFilePartition.RowRange;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.internal.SQLConf;
import scala.Option;

/**
 * Splits a Vortex file into row ranges so that one large file can be read by several Spark tasks.
 *
 * <p>Spark's own file sources cut a file into byte ranges of at most {@code maxSplitBytes}, where that bound is
 * {@code min(spark.sql.files.maxPartitionBytes, max(spark.sql.files.openCostInBytes, totalBytes / parallelism))}
 * ({@code FilePartition.maxSplitBytes}). The native Vortex scan addresses rows rather than bytes, so the same byte
 * budget is converted into rows at the file's average bytes per row and rounded down to the writer's row block, which
 * is where chunk boundaries fall. A range that cuts through a chunk is still correct, the reader decodes the chunk and
 * slices it, so alignment is a cost optimisation only.
 */
final class FileSplitPlanner {
    /**
     * Set to {@code true} to split large files. Off by default because planning then reads every file's footer on the
     * driver, which is a round trip per file on object stores.
     */
    static final String SPLIT_FILES_OPTION = "vortex.read.splitFiles";

    /** Byte budget of one split; when absent Spark's {@code maxSplitBytes} formula is used. */
    static final String MAX_SPLIT_BYTES_OPTION = "vortex.read.maxSplitBytes";

    /**
     * Smallest accepted {@link #MAX_SPLIT_BYTES_OPTION}: below it a large file plans an unreasonable number of tasks.
     */
    static final long MIN_SPLIT_BYTES = 1L << 20;

    /** Row block of the default write strategy: the writer emits chunk boundaries only at multiples of it. */
    static final long ROW_BLOCK_SIZE = 8192;

    /** On-storage size of a file and its exact row count, as read from the footer. */
    record FileSize(long rows, long bytes) {}

    private FileSplitPlanner() {}

    static boolean enabled(Map<String, String> options) {
        String value = options.getOrDefault(SPLIT_FILES_OPTION, "false");
        if (value.equalsIgnoreCase("true")) {
            return true;
        }
        checkArgument(value.equalsIgnoreCase("false"), "%s must be true or false, got '%s'", SPLIT_FILES_OPTION, value);
        return false;
    }

    /** The byte budget of one split, from the option or Spark's file-source formula. */
    static long maxSplitBytes(Map<String, String> options, Collection<FileSize> files) {
        String configured = options.get(MAX_SPLIT_BYTES_OPTION);
        if (configured != null) {
            long bytes;
            try {
                bytes = Long.parseLong(configured);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(
                        MAX_SPLIT_BYTES_OPTION + " must be a byte count, got '" + configured + "'", e);
            }
            checkArgument(
                    bytes >= MIN_SPLIT_BYTES,
                    "%s must be at least %s bytes, got %s",
                    MAX_SPLIT_BYTES_OPTION,
                    MIN_SPLIT_BYTES,
                    bytes);
            return bytes;
        }
        SQLConf conf = SQLConf.get();
        long openCost = conf.filesOpenCostInBytes();
        long totalBytes = files.stream().mapToLong(f -> f.bytes() + openCost).sum();
        long bytesPerCore = totalBytes / Math.max(1, minPartitionNum(conf));
        return Math.min(conf.filesMaxPartitionBytes(), Math.max(openCost, bytesPerCore));
    }

    /**
     * {@code spark.sql.files.minPartitionNum}, else Spark's leaf-node default parallelism. Spelled out because Spark 4
     * moved {@code leafNodeDefaultParallelism} off the {@code SparkSession} API class.
     */
    private static int minPartitionNum(SQLConf conf) {
        Option<Object> configured = conf.filesMinPartitionNum();
        if (configured.isEmpty()) {
            configured = conf.getConf(SQLConf.LEAF_NODE_DEFAULT_PARALLELISM());
        }
        return configured.isDefined()
                ? ((Number) configured.get()).intValue()
                : SparkSession.active().sparkContext().defaultParallelism();
    }

    /**
     * Contiguous row ranges tiling {@code [0, rows)}, each covering about {@code maxSplitBytes}; empty when the file
     * fits in one split and should be read whole.
     */
    static List<RowRange> splitRows(FileSize file, long maxSplitBytes) {
        if (file.rows() == 0 || file.bytes() <= maxSplitBytes) {
            return List.of();
        }
        double rowsPerSplit = (double) file.rows() * maxSplitBytes / file.bytes();
        long aligned = Math.max(ROW_BLOCK_SIZE, (long) rowsPerSplit / ROW_BLOCK_SIZE * ROW_BLOCK_SIZE);
        if (aligned >= file.rows()) {
            return List.of();
        }
        List<RowRange> ranges = new ArrayList<>();
        for (long begin = 0; begin < file.rows(); begin += aligned) {
            ranges.add(new RowRange(begin, Math.min(begin + aligned, file.rows())));
        }
        return ranges;
    }
}
