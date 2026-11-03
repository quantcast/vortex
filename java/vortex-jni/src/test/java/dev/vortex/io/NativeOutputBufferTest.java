// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.io;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.api.Session;
import dev.vortex.api.VortexWriter;
import dev.vortex.jni.NativeFiles;
import dev.vortex.jni.NativeLoader;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

final class NativeOutputBufferTest {
    @TempDir
    Path directory;

    @BeforeAll
    static void loadLibrary() {
        NativeLoader.loadJni();
    }

    /** Random bytes, so a chunk that was repeated or reordered changes the read-back value. */
    private static byte[] payload() {
        byte[] bytes = new byte[3 * 65_536 + 17];
        new Random(0).nextBytes(bytes);
        return bytes;
    }

    /** Writes an empty-schema file whose footer metadata holds {@code payload}, one large native buffer. */
    private static long writeMetadata(NativeWritable sink, byte[] payload) throws IOException {
        try (BufferAllocator allocator = new RootAllocator();
                VortexWriter writer = VortexWriter.builder(Session.create(), sink, new Schema(List.of()), allocator)
                        .putMetadata("payload", payload)
                        .build()) {
            return writer.finish().fileSize();
        }
    }

    @Test
    void writesReuseOneBoundedArrayAcrossLargeBuffers() throws IOException {
        Path path = directory.resolve("reuse.vortex");
        byte[] payload = payload();
        AtomicInteger calls = new AtomicInteger();
        try (OutputStream output = Files.newOutputStream(path)) {
            NativeWritable sink = new NativeWritable() {
                private byte[] firstArray;

                @Override
                public void write(byte[] buffer, int offset, int length) throws IOException {
                    if (firstArray == null) {
                        firstArray = buffer;
                    }
                    assertSame(firstArray, buffer, "reuse staging across upcalls and native buffers");
                    assertEquals(65_536, buffer.length);
                    assertTrue(length > 0 && length <= buffer.length);
                    calls.incrementAndGet();
                    output.write(buffer, offset, length);
                }

                @Override
                public void flush() throws IOException {
                    output.flush();
                }

                @Override
                public void close() {
                    throw new AssertionError("native code must not close the caller's sink");
                }
            };
            assertEquals(writeMetadata(sink, payload), Files.size(path));
            assertTrue(calls.get() > 4, "large native buffers must be split into bounded writes");
        }
        assertArrayEquals(
                payload,
                NativeFiles.readMetadata(Session.create(), path.toString(), Map.of())
                        .get("payload"));
    }

    @Test
    void writeExceptionsStopChunkingAndReachTheCaller() {
        AtomicInteger fullChunks = new AtomicInteger();
        AtomicInteger callsAfterFailure = new AtomicInteger();
        AtomicBoolean failed = new AtomicBoolean();
        NativeWritable sink = new NativeWritable() {
            @Override
            public void write(byte[] buffer, int offset, int length) throws IOException {
                if (failed.get()) {
                    callsAfterFailure.incrementAndGet();
                }
                // Fail inside the payload's chunk loop, after one full chunk succeeded.
                if (length == buffer.length && fullChunks.incrementAndGet() == 2) {
                    failed.set(true);
                    throw new IOException("injected write failure");
                }
            }

            @Override
            public void flush() {}

            @Override
            public void close() {}
        };
        IOException error = assertThrows(IOException.class, () -> writeMetadata(sink, payload()));
        assertTrue(failed.get());
        assertEquals(0, callsAfterFailure.get());
        assertTrue(error.getCause().getMessage().contains("injected write failure"), error.toString());
    }
}
