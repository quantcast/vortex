// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::io;
use std::sync::Arc;

use jni::JValue;
use jni::JavaVM;
use jni::objects::JByteArray;
use jni::objects::JObject;
use jni::refs::Global;
use vortex::error::VortexExpect;
use vortex::error::vortex_err;
use vortex::io::IoBuf;
use vortex::io::VortexWrite;

use crate::io::with_jvm;

/// Bound heap staging independently of the size of a native output buffer.
const WRITE_SCRATCH_BYTES: usize = 65_536;

enum WriteMode {
    Direct,
    Array(Option<Global<JByteArray<'static>>>),
}

/// A [`VortexWrite`] backed by a Java object implementing `dev.vortex.io.NativeWritable`.
///
/// Bytes are forwarded as blocking upcalls, either through a borrowed direct buffer or a
/// reusable heap array. `flush` and `shutdown` both map to `flush()`: the Java caller created the
/// underlying stream and remains responsible for closing it once the writer finishes.
/// Writes run inline on the runtime thread driving the write task, which is attached
/// to the JVM on first use.
pub(crate) struct JavaWrite {
    vm: JavaVM,
    writable: Arc<Global<JObject<'static>>>,
    mode: WriteMode,
}

impl JavaWrite {
    pub(crate) fn new(
        vm: JavaVM,
        writable: Arc<Global<JObject<'static>>>,
        direct_buffers: bool,
    ) -> Self {
        Self {
            vm,
            writable,
            mode: if direct_buffers {
                WriteMode::Direct
            } else {
                WriteMode::Array(None)
            },
        }
    }

    fn write_slice(&mut self, bytes: &[u8]) -> io::Result<()> {
        match &mut self.mode {
            WriteMode::Direct => {
                for chunk in bytes.chunks(i32::MAX as usize) {
                    with_jvm(&self.vm, |env| {
                        // SAFETY: `chunk` remains live through the synchronous upcall. Java sees
                        // only a read-only view and ByteBufferWritable forbids retaining any view.
                        let buffer = unsafe {
                            env.new_direct_byte_buffer(chunk.as_ptr().cast_mut(), chunk.len())?
                        };
                        let buffer = env
                            .call_method(
                                &buffer,
                                jni::jni_str!("asReadOnlyBuffer"),
                                jni::jni_sig!("()Ljava/nio/ByteBuffer;"),
                                &[],
                            )?
                            .l()?;
                        env.call_method(
                            self.writable.as_ref(),
                            jni::jni_str!("write"),
                            jni::jni_sig!("(Ljava/nio/ByteBuffer;)V"),
                            &[JValue::Object(&buffer)],
                        )?;
                        let remaining = env
                            .call_method(
                                &buffer,
                                jni::jni_str!("remaining"),
                                jni::jni_sig!("()I"),
                                &[],
                            )?
                            .i()?;
                        if remaining != 0 {
                            return Err(vortex_err!(
                                "write returned with {remaining} bytes unconsumed"
                            )
                            .into());
                        }
                        Ok(())
                    })
                    .map_err(io::Error::other)?;
                }
            }
            WriteMode::Array(scratch) => {
                if bytes.is_empty() {
                    return Ok(());
                }
                with_jvm(&self.vm, |env| {
                    let array = match scratch {
                        Some(array) => array,
                        None => {
                            let array = JByteArray::new(env, WRITE_SCRATCH_BYTES)?;
                            scratch.insert(env.new_global_ref(&array)?)
                        }
                    };
                    for chunk in bytes.chunks(WRITE_SCRATCH_BYTES) {
                        let jlen = i32::try_from(chunk.len()).vortex_expect("64 KiB fits in i32");
                        // SAFETY: u8 and i8 have the same size and alignment; only the bit
                        // representation is copied into Java's signed byte array.
                        let signed = unsafe {
                            std::slice::from_raw_parts(chunk.as_ptr().cast::<i8>(), chunk.len())
                        };
                        array.set_region(env, 0, signed)?;
                        env.call_method(
                            self.writable.as_ref(),
                            jni::jni_str!("write"),
                            jni::jni_sig!("([BII)V"),
                            &[
                                JValue::Object(array.as_ref()),
                                JValue::Int(0),
                                JValue::Int(jlen),
                            ],
                        )?;
                    }
                    Ok(())
                })
                .map_err(io::Error::other)?;
            }
        }
        Ok(())
    }

    fn flush_upcall(&self) -> io::Result<()> {
        with_jvm(&self.vm, |env| {
            env.call_method(
                self.writable.as_ref(),
                jni::jni_str!("flush"),
                jni::jni_sig!("()V"),
                &[],
            )?;
            Ok(())
        })
        .map_err(io::Error::other)
    }
}

impl VortexWrite for JavaWrite {
    async fn write_all<B: IoBuf>(&mut self, buffer: B) -> io::Result<B> {
        self.write_slice(buffer.as_slice())?;
        Ok(buffer)
    }

    async fn flush(&mut self) -> io::Result<()> {
        self.flush_upcall()
    }

    async fn shutdown(&mut self) -> io::Result<()> {
        self.flush_upcall()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use jni::JValue;
    use jni::objects::JByteArray;
    use vortex::error::VortexResult;
    use vortex::error::vortex_err;

    use super::JavaWrite;
    use crate::io::tests::assert_weak_ref_clears;
    use crate::io::tests::test_vm;
    use crate::io::with_jvm;

    /// Write upcalls from a fresh native thread must reach the Java object, and once
    /// the writer (and every other native reference) is dropped, its JNI global ref
    /// must be deleted so the Java object becomes collectible.
    #[test]
    fn test_write_upcalls_release_global_ref() -> VortexResult<()> {
        let Some(vm) = test_vm() else {
            return Ok(());
        };

        // `ByteArrayOutputStream` has the same `write([BII)V`/`flush()V` shape as
        // `dev.vortex.io.NativeWritable`, so it stands in for a caller-provided sink.
        let (writable, content, weak) = with_jvm(vm, |env| {
            let sink = env.new_object(
                jni::jni_str!("java/io/ByteArrayOutputStream"),
                jni::jni_sig!("()V"),
                &[],
            )?;
            let weak = env.new_object(
                jni::jni_str!("java/lang/ref/WeakReference"),
                jni::jni_sig!("(Ljava/lang/Object;)V"),
                &[JValue::Object(&sink)],
            )?;
            Ok((
                env.new_global_ref(&sink)?,
                env.new_global_ref(&sink)?,
                env.new_global_ref(&weak)?,
            ))
        })?;

        let thread_vm = vm.clone();
        std::thread::spawn(move || {
            let mut write = JavaWrite::new(thread_vm.clone(), Arc::new(writable), false);
            write.write_slice(b"vortex")?;
            write.flush_upcall()?;
            // `with_jvm` attaches permanently: the thread stays attached until it
            // exits, at which point the attachment is torn down automatically.
            let attached = thread_vm
                .is_thread_attached()
                .map_err(|e| vortex_err!("is_thread_attached failed: {e}"))?;
            assert!(attached, "write upcalls should leave the thread attached");
            VortexResult::Ok(())
        })
        .join()
        .map_err(|_| vortex_err!("writer thread panicked"))??;

        let written = with_jvm(vm, |env| {
            let array = env
                .call_method(
                    content.as_ref(),
                    jni::jni_str!("toByteArray"),
                    jni::jni_sig!("()[B"),
                    &[],
                )?
                .l()?;
            let array = env.cast_local::<JByteArray>(array)?;
            let mut bytes = vec![0i8; array.len(env)?];
            array.get_region(env, 0, &mut bytes)?;
            Ok(bytes.into_iter().map(|b| b as u8).collect::<Vec<u8>>())
        })?;
        assert_eq!(written, b"vortex");

        drop(content);
        assert_weak_ref_clears(vm, &weak)
    }
}
