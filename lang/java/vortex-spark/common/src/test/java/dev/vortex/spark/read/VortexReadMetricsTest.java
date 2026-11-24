// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.apache.spark.TaskContext$;
import org.apache.spark.paths.SparkPath;
import org.apache.spark.sql.catalyst.FileSourceOptions;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.execution.datasources.FilePartition;
import org.apache.spark.sql.execution.datasources.PartitionedFile;
import org.apache.spark.sql.vectorized.ColumnarBatch;
import org.apache.spark.util.TaskCompletionListener;
import org.junit.jupiter.api.Test;
import scala.Option;
import scala.collection.immutable.Map$;

final class VortexReadMetricsTest {
    @Test
    void completionSnapshotIncludesReadsThatFinishDuringClose() throws Exception {
        var context = TaskContext$.MODULE$.empty();
        TaskContext$.MODULE$.setTaskContext(context);
        ExecutorService worker = Executors.newSingleThreadExecutor();
        CountDownLatch reading = new CountDownLatch(1);
        CountDownLatch finish = new CountDownLatch(1);
        AtomicReference<Future<?>> read = new AtomicReference<>();
        try {
            PartitionedFile file = new PartitionedFile(
                    InternalRow.empty(),
                    SparkPath.fromPathString("file:///memory.vortex"),
                    0,
                    23,
                    new String[0],
                    0,
                    23,
                    PartitionedFile.apply$default$8());
            try (PartitionReader<ColumnarBatch> reader = VortexReadMetrics.createReader(
                    new FilePartition(0, new PartitionedFile[] {file}),
                    new Object(),
                    new FileSourceOptions(Map$.MODULE$.empty()),
                    (ignored, metrics) -> new PartitionReader<>() {
                        private boolean closed;

                        @Override
                        public boolean next() {
                            read.set(worker.submit(() -> {
                                reading.countDown();
                                if (!finish.await(10, TimeUnit.SECONDS)) {
                                    throw new IOException("read timed out");
                                }
                                metrics.accept(23);
                                return null;
                            }));
                            return true;
                        }

                        @Override
                        public ColumnarBatch get() {
                            throw new UnsupportedOperationException();
                        }

                        @Override
                        public void close() throws IOException {
                            if (!closed) {
                                closed = true;
                                finish.countDown();
                                try {
                                    read.get().get(10, TimeUnit.SECONDS);
                                } catch (Exception e) {
                                    throw new IOException(e);
                                }
                            }
                        }
                    })) {
                assertTrue(reader.next());
                assertTrue(reading.await(10, TimeUnit.SECONDS));
                assertEquals(0, reader.currentMetricsValues()[0].value());
                context.addTaskCompletionListener((TaskCompletionListener) ignored -> {
                    assertEquals(23, reader.currentMetricsValues()[0].value());
                    assertEquals(23, reader.currentMetricsValues()[0].value());
                });
                context.markTaskCompleted(Option.empty());
                assertEquals(23, context.taskMetrics().inputMetrics().bytesRead());
            }
        } finally {
            finish.countDown();
            worker.shutdownNow();
            TaskContext$.MODULE$.unset();
        }
    }

    @Test
    void cumulativeSnapshotsIncludeConcurrentReads() throws Exception {
        VortexReadMetrics metrics = new VortexReadMetrics();
        ExecutorService workers = Executors.newFixedThreadPool(4);
        try {
            List<Future<?>> reads = new ArrayList<>();
            for (int i = 0; i < 100; i++) {
                reads.add(workers.submit(() -> metrics.accept(17)));
            }
            for (Future<?> read : reads) {
                read.get();
            }
        } finally {
            workers.shutdownNow();
        }
        var snapshot = metrics.currentMetricsValues()[0];
        assertEquals(1700, snapshot.value());
        assertEquals("vortexBytesRead", snapshot.name());
        metrics.accept(3);
        assertEquals(1700, snapshot.value());
        assertEquals(1703, metrics.currentMetricsValues()[0].value());
    }
}
