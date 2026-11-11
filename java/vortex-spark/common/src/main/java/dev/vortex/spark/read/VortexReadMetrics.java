// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.IdentityHashMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.LongConsumer;
import org.apache.hadoop.fs.FileSystem;
import org.apache.spark.TaskContext;
import org.apache.spark.sql.catalyst.FileSourceOptions;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.metric.CustomTaskMetric;
import org.apache.spark.sql.connector.read.InputPartition;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.execution.datasources.PartitionedFile;
import org.apache.spark.sql.execution.datasources.v2.FilePartitionReaderFactory;
import org.apache.spark.sql.vectorized.ColumnarBatch;
import org.apache.spark.util.TaskCompletionListener;

/** Cumulative successful source bytes, including footers and read-ahead, for one Spark task and scan. */
final class VortexReadMetrics implements LongConsumer {
    private static final ConcurrentMap<TaskContext, TaskBytes> TASK_BYTES = new ConcurrentHashMap<>();
    // The scan key of files opened one at a time with buildColumnarReader(file), so they share one task total.
    private static final Object SINGLE_FILES = new Object();
    // Spark reads this legacy statistics API. A private class key keeps this entry separate from storage statistics.
    @SuppressWarnings("deprecation")
    private static final FileSystem.Statistics NATIVE_STATISTICS =
            FileSystem.getStatistics("vortex-native", NativeReadStatistics.class);

    private abstract static class NativeReadStatistics extends FileSystem {}

    private final AtomicLong bytesRead;
    private final TaskBytes taskBytes;
    private final Thread taskThread = Thread.currentThread();

    VortexReadMetrics() {
        this(new AtomicLong(), new TaskBytes());
    }

    private VortexReadMetrics(AtomicLong bytesRead, TaskBytes taskBytes) {
        this.bytesRead = bytesRead;
        this.taskBytes = taskBytes;
    }

    @Override
    public void accept(long bytes) {
        bytesRead.addAndGet(bytes);
        if (Thread.currentThread() != taskThread) {
            taskBytes.workerBytes.addAndGet(bytes);
        }
    }

    private static TaskBytes taskBytes(TaskContext context) {
        if (context == null) {
            return new TaskBytes();
        }
        TaskBytes created = new TaskBytes();
        TaskBytes existing = TASK_BYTES.putIfAbsent(context, created);
        if (existing != null) {
            return existing;
        }
        // Registered outside the map update: Spark runs this immediately if the task has already completed.
        context.addTaskCompletionListener((TaskCompletionListener) ignored -> TASK_BYTES.remove(context));
        return created;
    }

    private void publishWorkerBytes(TaskContext context) {
        if (context == null) {
            return;
        }
        long completed = taskBytes.workerBytes.get();
        long delta = completed - taskBytes.publishedWorkerBytes;
        if (delta != 0) {
            // Spark overwrites input bytes from the task thread's Hadoop statistics, so worker-thread bytes must
            // land there too. Task-thread reads are already counted by the storage FileSystem's own statistics.
            NATIVE_STATISTICS.incrementBytesRead(delta);
            context.taskMetrics().inputMetrics().incBytesRead(delta);
            taskBytes.publishedWorkerBytes = completed;
        }
    }

    private static final class TaskBytes {
        private final AtomicLong workerBytes = new AtomicLong();
        private final IdentityHashMap<Object, AtomicLong> scanBytes = new IdentityHashMap<>();
        // Accessed only on the Spark task thread, shared by every reader/scan consumed by that task.
        private long publishedWorkerBytes;
    }

    CustomTaskMetric[] currentMetricsValues() {
        long snapshot = bytesRead.get();
        return new CustomTaskMetric[] {
            new CustomTaskMetric() {
                @Override
                public String name() {
                    return VortexBytesReadMetric.NAME;
                }

                @Override
                public long value() {
                    return snapshot;
                }
            }
        };
    }

    static PartitionReader<ColumnarBatch> createReader(
            InputPartition partition,
            Object factory,
            FileSourceOptions options,
            BiFunction<PartitionedFile, LongConsumer, PartitionReader<ColumnarBatch>> buildReader) {
        TaskContext context = TaskContext.get();
        // Several input partitions can run in one coalesced task, so each reader of this scan shares one total.
        VortexReadMetrics metrics = forScan(context, factory);
        PartitionReader<ColumnarBatch> files = new FilePartitionReaderFactory() {
            @Override
            public FileSourceOptions options() {
                return options;
            }

            @Override
            public PartitionReader<InternalRow> buildReader(PartitionedFile file) {
                throw new UnsupportedOperationException("row-based V2 reads are not supported");
            }

            @Override
            public PartitionReader<ColumnarBatch> buildColumnarReader(PartitionedFile file) {
                return buildReader.apply(file, metrics);
            }
        }.createColumnarReader(partition);

        // FilePartitionReader does not forward currentMetricsValues from individual files.
        return reporting(context, files, metrics);
    }

    /**
     * A reader of one file, opened outside a scan's {@code createColumnarReader}. Every such file in a task shares one
     * cumulative total, so a caller that forwards {@code currentMetricsValues} reports all of them.
     */
    static PartitionReader<ColumnarBatch> createFileReader(
            Function<LongConsumer, PartitionReader<ColumnarBatch>> buildReader) {
        TaskContext context = TaskContext.get();
        VortexReadMetrics metrics = forScan(context, SINGLE_FILES);
        return reporting(context, buildReader.apply(metrics), metrics);
    }

    // Spark sets (rather than adds) custom metrics, so every reader of one scan in a task must share its total.
    private static VortexReadMetrics forScan(TaskContext context, Object scan) {
        TaskBytes taskBytes = taskBytes(context);
        return new VortexReadMetrics(taskBytes.scanBytes.computeIfAbsent(scan, ignored -> new AtomicLong()), taskBytes);
    }

    private static PartitionReader<ColumnarBatch> reporting(
            TaskContext context, PartitionReader<ColumnarBatch> files, VortexReadMetrics metrics) {
        return new PartitionReader<>() {
            @Override
            public boolean next() throws IOException {
                try {
                    return files.next();
                } finally {
                    metrics.publishWorkerBytes(context);
                }
            }

            @Override
            public ColumnarBatch get() {
                return files.get();
            }

            @Override
            public CustomTaskMetric[] currentMetricsValues() {
                if (context != null && context.isCompleted()) {
                    // Spark asks for its final custom snapshot before closing the reader. Drain read-ahead first.
                    try {
                        files.close();
                    } catch (IOException e) {
                        throw new UncheckedIOException(e);
                    }
                }
                metrics.publishWorkerBytes(context);
                return metrics.currentMetricsValues();
            }

            @Override
            public void close() throws IOException {
                try {
                    files.close();
                } finally {
                    metrics.publishWorkerBytes(context);
                }
            }
        };
    }
}
