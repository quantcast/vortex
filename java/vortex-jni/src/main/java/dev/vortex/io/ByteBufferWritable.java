// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.io;

import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * A {@link NativeWritable} that consumes native output memory directly, without a staging {@code byte[]}.
 *
 * <p>Implement this for sinks with a real {@code ByteBuffer} API, such as a file channel. Wrapping an ordinary
 * {@code OutputStream} in a channel still copies through a byte array; it can use {@link NativeWritable} directly.
 */
public interface ByteBufferWritable extends NativeWritable {
    /**
     * Append all remaining bytes and advance {@code buffer.position()} to its limit before returning.
     *
     * <p>Native calls supply a read-only, direct buffer borrowing Rust-owned memory. Neither the buffer nor any derived
     * view may be retained or accessed after this call returns, including when it throws. An asynchronous sink must
     * copy into memory it owns before returning.
     */
    void write(ByteBuffer buffer) throws IOException;

    @Override
    default void write(byte[] buffer, int offset, int length) throws IOException {
        write(ByteBuffer.wrap(buffer, offset, length).asReadOnlyBuffer());
    }
}
