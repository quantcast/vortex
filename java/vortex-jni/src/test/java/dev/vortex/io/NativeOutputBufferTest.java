// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.io;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.api.Session;
import dev.vortex.api.VortexWriter;
import dev.vortex.jni.NativeFiles;
import dev.vortex.jni.NativeLoader;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ReadOnlyBufferException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Map;
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

    private static byte[] payload() {
        byte[] bytes = new byte[3 * 65_536 + 17];
        for (int i = 0; i < bytes.length; i++) {
            bytes[i] = (byte) i;
        }
        return bytes;
    }

    private static long writeMetadata(NativeWritable sink, byte[] payload) throws IOException {
        try (BufferAllocator allocator = new RootAllocator();
                VortexWriter writer = VortexWriter.builder(Session.create(), sink, new Schema(List.of()), allocator)
                        .putMetadata("payload", payload)
                        .build()) {
            return writer.finish().fileSize();
        }
    }

    @Test
    void byteArrayCallsRespectTheRequestedRange() throws IOException {
        ByteBuffer destination = ByteBuffer.allocate(4);
        ByteBufferWritable sink = new ByteBufferWritable() {
            @Override
            public void write(ByteBuffer buffer) {
                destination.put(buffer);
            }

            @Override
            public void flush() {}

            @Override
            public void close() {}
        };
        sink.write(new byte[] {0, 1, 2, 3, 4, 5}, 1, 4);
        sink.write(new byte[0], 0, 0);
        assertArrayEquals(new byte[] {1, 2, 3, 4}, destination.array());
    }

    @Test
    void directBuffersReachAFileChannelWithoutArrayWrites() throws IOException {
        Path path = directory.resolve("direct.vortex");
        byte[] payload = payload();
        AtomicInteger calls = new AtomicInteger();
        AtomicBoolean closed = new AtomicBoolean();
        AtomicBoolean flushed = new AtomicBoolean();
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
            ByteBufferWritable sink = new ByteBufferWritable() {
                @Override
                public void write(ByteBuffer buffer) throws IOException {
                    assertTrue(buffer.isDirect());
                    assertTrue(buffer.isReadOnly());
                    assertThrows(ReadOnlyBufferException.class, () -> buffer.put(0, (byte) 0));
                    calls.incrementAndGet();
                    while (buffer.hasRemaining()) {
                        channel.write(buffer);
                    }
                }

                @Override
                public void write(byte[] bytes, int offset, int length) {
                    throw new AssertionError("native output must use the direct-buffer overload");
                }

                @Override
                public void flush() {
                    flushed.set(true);
                }

                @Override
                public void close() {
                    closed.set(true);
                }
            };
            assertEquals(writeMetadata(sink, payload), channel.size());
            assertTrue(calls.get() > 1);
            assertTrue(flushed.get());
            assertFalse(closed.get(), "the caller owns the sink");
        }
        assertArrayEquals(
                payload,
                NativeFiles.readMetadata(Session.create(), path.toString(), Map.of())
                        .get("payload"));
    }

    @Test
    void legacyWritesReuseOneBoundedArrayAcrossLargeBuffers() throws IOException {
        Path path = directory.resolve("legacy.vortex");
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
    void directWritesMustConsumeTheWholeBuffer() {
        ByteBufferWritable sink = new ByteBufferWritable() {
            @Override
            public void write(ByteBuffer buffer) {
                buffer.get();
            }

            @Override
            public void flush() {}

            @Override
            public void close() {}
        };
        IOException error = assertThrows(IOException.class, () -> writeMetadata(sink, new byte[0]));
        assertTrue(error.getCause().getMessage().contains("bytes unconsumed"), error.toString());
    }

    @Test
    void directWriteExceptionsReachTheCaller() {
        ByteBufferWritable sink = new ByteBufferWritable() {
            @Override
            public void write(ByteBuffer buffer) throws IOException {
                throw new IOException("injected direct write failure");
            }

            @Override
            public void flush() {}

            @Override
            public void close() {}
        };
        IOException error = assertThrows(IOException.class, () -> writeMetadata(sink, new byte[0]));
        assertTrue(error.getCause().getMessage().contains("injected direct write failure"), error.toString());
    }

    @Test
    void legacyWriteExceptionsStopChunkingAndReachTheCaller() {
        AtomicInteger calls = new AtomicInteger();
        NativeWritable sink = new NativeWritable() {
            @Override
            public void write(byte[] buffer, int offset, int length) throws IOException {
                if (calls.incrementAndGet() == 3) {
                    throw new IOException("injected array write failure");
                }
            }

            @Override
            public void flush() {}

            @Override
            public void close() {}
        };
        IOException error = assertThrows(IOException.class, () -> writeMetadata(sink, payload()));
        assertEquals(3, calls.get());
        assertTrue(error.getCause().getMessage().contains("injected array write failure"), error.toString());
    }
}
