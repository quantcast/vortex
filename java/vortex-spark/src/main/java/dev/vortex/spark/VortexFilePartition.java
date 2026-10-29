// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static com.google.common.base.Preconditions.checkArgument;

import com.esotericsoftware.kryo.DefaultSerializer;
import com.esotericsoftware.kryo.serializers.JavaSerializer;
import java.io.Serializable;
import java.util.List;
import java.util.Map;
import org.apache.spark.sql.connector.read.InputPartition;
import org.apache.spark.sql.types.StructType;

/**
 * An {@link InputPartition} describing a group of Vortex files, or a row range of one file, that a single reader should
 * handle together.
 *
 * <p>Each executor opens a single Vortex {@code Session}, {@code DataSource} and {@code Scan} over the partition's
 * {@link #paths()} and consumes every Vortex partition produced by that scan before moving on to the next Spark
 * {@code InputPartition}.
 *
 * <p>The requested output schema is carried as a {@link StructType} rather than a list of {@code Column} objects:
 * {@code StructType} is the stable serialization surface in Spark and survives shipping to executors reliably.
 *
 * @param paths the Vortex file paths (or globs) belonging to this input partition
 * @param readSchema the requested output schema (data columns + partition columns)
 * @param formatOptions object-store properties used to open the files
 * @param partitionValues Hive-style partition column values shared by all {@link #paths()}
 * @param rowRange the rows of the single file in {@link #paths()} this partition reads, or {@code null} to read the
 *     whole of every path
 */
// Kryo's FieldSerializer cannot rebuild records, so under KryoSerializer this uses Java serialization.
@DefaultSerializer(JavaSerializer.class)
public record VortexFilePartition(
        List<String> paths,
        StructType readSchema,
        Map<String, String> formatOptions,
        Map<String, String> partitionValues,
        RowRange rowRange)
        implements InputPartition, Serializable {

    public VortexFilePartition {
        checkArgument(rowRange == null || paths.size() == 1, "a row range applies to exactly one file, got %s", paths);
    }

    /** A partition reading the whole of every path. */
    public VortexFilePartition(
            List<String> paths,
            StructType readSchema,
            Map<String, String> formatOptions,
            Map<String, String> partitionValues) {
        this(paths, readSchema, formatOptions, partitionValues, null);
    }

    /**
     * A half-open range of row indices {@code [begin, end)} within one file. The native scan reads only these rows and
     * evaluates the projection and filter over them, so the ranges of one file partition it into disjoint tasks.
     */
    public record RowRange(long begin, long end) implements Serializable {
        public RowRange {
            checkArgument(begin >= 0 && begin < end, "invalid row range [%s, %s)", begin, end);
        }

        /** Number of rows in the range. */
        public long rows() {
            return end - begin;
        }
    }
}
