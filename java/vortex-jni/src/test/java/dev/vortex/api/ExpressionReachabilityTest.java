// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.management.ManagementFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Exercise compiled expression factories while the Cleaner processes their temporary inputs. */
public final class ExpressionReachabilityTest {
    private static final int LIST_SIZE = 2000;
    private static final int ITERATIONS = 2000;
    private static volatile Expression result;

    @TempDir
    Path directory;

    @Test
    void temporaryLiteralInputsSurviveNativeCalls() throws Exception {
        // A lifetime regression can crash native code, so keep the test runner in a separate JVM.
        Path log = directory.resolve("child.log");
        List<String> command = new ArrayList<>();
        command.add(Path.of(System.getProperty("java.home"), "bin", "java").toString());
        command.addAll(List.of(
                "-Xms32m",
                "-Xmx32m",
                "-Xbatch",
                "-XX:-TieredCompilation",
                "-XX:-DisableExplicitGC",
                "-XX:CompileThreshold=100",
                "-XX:CompileCommand=dontinline,dev.vortex.api.Expression::literalList",
                "-XX:ErrorFile=" + directory.resolve("hs_err_pid%p.log")));
        ManagementFactory.getRuntimeMXBean().getInputArguments().stream()
                .filter(arg -> arg.startsWith("--add-opens="))
                .forEach(command::add);
        command.addAll(
                List.of("-cp", System.getProperty("java.class.path"), getClass().getName(), directory.toString()));
        Process child = new ProcessBuilder(command)
                .redirectErrorStream(true)
                .redirectOutput(log.toFile())
                .start();
        boolean exited;
        try {
            exited = child.waitFor(120, TimeUnit.SECONDS);
        } finally {
            if (child.isAlive()) {
                child.destroyForcibly().waitFor();
            }
        }
        String childOutput = Files.readString(log);
        assertTrue(exited, "GC stress child timed out:\n" + childOutput);
        assertEquals(0, child.exitValue(), childOutput);
    }

    private static Expression buildList() {
        Expression[] elements = new Expression[LIST_SIZE];
        for (int i = 0; i < elements.length; i++) {
            elements[i] = Expression.literal((long) i);
        }
        return Expression.literalList(Expression.nullLiteral(Expression.DType.I64), false, elements);
    }

    /** Child entry point: validate native list values with a real filtered scan during GC pressure. */
    public static void main(String[] args) throws Exception {
        AtomicBoolean running = new AtomicBoolean(true);
        Thread gc = new Thread(
                () -> {
                    while (running.get()) {
                        System.gc();
                        try {
                            Thread.sleep(1);
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            return;
                        }
                    }
                },
                "expression-gc-pressure");
        gc.setDaemon(true);
        try (BufferAllocator allocator = new RootAllocator()) {
            Session session = Session.create();
            String uri = Path.of(args[0], "values.vortex").toString();
            Schema schema = new Schema(List.of(Field.notNullable("id", new ArrowType.Int(64, true))));
            try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator);
                    VortexWriter writer = VortexWriter.builder(session, uri, schema, allocator)
                            .build()) {
                BigIntVector ids = (BigIntVector) root.getVector(0);
                ids.allocateNew(LIST_SIZE + 1);
                for (int i = 0; i <= LIST_SIZE; i++) {
                    ids.set(i, i);
                }
                root.setRowCount(LIST_SIZE + 1);
                writer.writeBatch(root, allocator);
            }
            try (DataSource source = DataSource.open(session, uri)) {
                long collectionsBefore = collectionCount();
                gc.start();
                try {
                    for (int i = 0; i < ITERATIONS; i++) {
                        result = buildList();
                        if (i % 100 == 0 || i == ITERATIONS - 1) {
                            checkValues(source, allocator);
                        }
                    }
                } finally {
                    running.set(false);
                    gc.join();
                }
                if (collectionCount() <= collectionsBefore) {
                    throw new AssertionError("no garbage collection occurred during the stress loop");
                }
            }
        }
        System.out.println("completed=" + ITERATIONS);
    }

    private static long collectionCount() {
        return ManagementFactory.getGarbageCollectorMXBeans().stream()
                .mapToLong(collector -> Math.max(0, collector.getCollectionCount()))
                .sum();
    }

    private static void checkValues(DataSource source, BufferAllocator allocator) throws Exception {
        try (Scan scan = source.scan(ScanOptions.builder()
                .filter(Expression.in(Expression.column("id"), result))
                .ordered(true)
                .build())) {
            long expected = 0;
            while (scan.hasNext()) {
                try (ArrowReader reader = scan.next().scanArrow(allocator)) {
                    while (reader.loadNextBatch()) {
                        VectorSchemaRoot root = reader.getVectorSchemaRoot();
                        BigIntVector ids = (BigIntVector) root.getVector(0);
                        for (int row = 0; row < root.getRowCount(); row++) {
                            if (ids.isNull(row) || ids.get(row) != expected++) {
                                throw new AssertionError("incorrect list membership at row " + row);
                            }
                        }
                    }
                }
            }
            if (expected != LIST_SIZE) {
                throw new AssertionError("expected " + LIST_SIZE + " matches, got " + expected);
            }
        }
    }
}
