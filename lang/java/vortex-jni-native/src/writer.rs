// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! JNI bindings for the Vortex file writer.
//!
//! Writes go through an in-flight queue of at most [`WRITE_CHANNEL_CAPACITY`] pending
//! batches on the same thread that drives the current-thread runtime.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use arrow_array::RecordBatch;
use arrow_array::StructArray;
use arrow_array::ffi::FFI_ArrowArray;
use arrow_array::ffi::FFI_ArrowSchema;
use arrow_data::layout;
use arrow_schema::DataType;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use async_fs::File;
use futures::SinkExt;
use futures::Stream;
use futures::StreamExt;
use futures::channel::mpsc;
use jni::Env;
use jni::EnvUnowned;
use jni::objects::JClass;
use jni::objects::JObject;
use jni::objects::JString;
use jni::objects::JValue;
use jni::sys::JNI_FALSE;
use jni::sys::JNI_TRUE;
use jni::sys::jboolean;
use jni::sys::jdouble;
use jni::sys::jlong;
use jni::sys::jobject;
use object_store::ObjectStore;
use object_store::path::Path as ObjectStorePath;
use vortex::array::ArrayRef;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::buffer::ByteBuffer;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::dtype::DType;
use vortex::editions::Edition;
use vortex::editions::EditionDeclaration;
use vortex::editions::EditionFamily;
use vortex::editions::EditionId;
use vortex::editions::EditionMember;
use vortex::editions::EditionSessionExt;
use vortex::editions::EnabledEditions;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::file::CountingVortexWrite;
use vortex::file::DeferredMetadata;
use vortex::file::VortexWriteOptions;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteSizeEstimator;
use vortex::file::WriteStrategyBuilder;
use vortex::file::WriteSummary;
use vortex::file::multi::parse_uri_or_path;
use vortex::io::VortexWrite;
use vortex::io::compat::Compat;
use vortex::io::object_store::ObjectStoreWrite;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::Task;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::BufferedBytesReservation;
use vortex::session::VortexSession;
use vortex::utils::aliases::hash_map::HashMap;
use vortex_arrow::ArrowSessionExt;

use crate::RUNTIME;
use crate::buffer::exported_buffer_addresses;
use crate::buffer::import_buffers;
use crate::errors::JNIError;
use crate::errors::try_or_throw;
use crate::file::extract_metadata;
use crate::file::extract_properties;
use crate::io::JavaWrite;
use crate::object_store::make_object_store;
use crate::session::session_ref;
use crate::statistics::checked_jlong;
use crate::statistics::column_statistics_to_java;
use crate::statistics::footer_column_statistics;

/// Capacity of the in-flight write queue. Small on purpose so that back-pressure from
/// the writer is felt on the Java thread producing batches.
const WRITE_CHANNEL_CAPACITY: usize = 4;

enum ResolvedStore {
    ObjectStore(Arc<dyn ObjectStore>, ObjectStorePath),
    Path(PathBuf),
}

fn resolve_store(
    url_or_path: &str,
    properties: &HashMap<String, String>,
) -> VortexResult<ResolvedStore> {
    let url = parse_uri_or_path(url_or_path)?;
    if url.scheme() == "file" {
        let path = url
            .to_file_path()
            .map_err(|_| vortex_err!("invalid file URL: {url_or_path}"))?;
        Ok(ResolvedStore::Path(path))
    } else {
        let (store, path) = make_object_store(&url, properties)?;
        Ok(ResolvedStore::ObjectStore(store, path))
    }
}

/// A batch in the in-flight write queue, with the reservation that counts it as uncompressed.
type QueuedBatch = (VortexResult<ArrayRef>, BufferedBytesReservation);

/// Creates the in-flight write queue and the array stream that the write task reads from it.
///
/// The write task drops the reservation of a batch when it takes the batch from the queue. From
/// then on, the layout strategies count the batch.
fn write_queue(
    write_schema: DType,
) -> (
    mpsc::Sender<QueuedBatch>,
    ArrayStreamAdapter<impl Stream<Item = VortexResult<ArrayRef>>>,
) {
    let (tx, rx) = mpsc::channel::<QueuedBatch>(WRITE_CHANNEL_CAPACITY);
    let rx = rx.map(|(batch, reservation)| {
        drop(reservation);
        batch
    });
    (tx, ArrayStreamAdapter::new(write_schema, rx))
}

/// Write options for `session`: its defaults, or the compact strategy, which trades read throughput
/// for file size as Python's `VortexWriteOptions.compact()` does.
fn session_write_options(session: &VortexSession, compact: bool) -> VortexWriteOptions {
    let options = session.write_options();
    if !compact {
        return options;
    }
    let compressor = BtrBlocksCompressorBuilder::from_session(session).with_compact();
    options.with_strategy(
        WriteStrategyBuilder::from_session(session)
            .with_btrblocks_builder(compressor)
            .build(),
    )
}

/// Native writer holding a write-task handle and a sender that Java pushes batches into.
pub struct NativeWriter {
    handle: Option<Task<VortexResult<WriteSummary>>>,
    session: VortexSession,
    arrow_schema: SchemaRef,
    write_schema: DType,
    bytes_written: Arc<AtomicU64>,
    size: WriteSizeEstimator,
    deferred_metadata: DeferredMetadata,
    sender: mpsc::Sender<QueuedBatch>,
}

impl NativeWriter {
    /// Spawn the write task into `write` and return the writer that sends batches to it.
    fn start<W: VortexWrite + Unpin + Send + 'static>(
        session: &VortexSession,
        arrow_schema: SchemaRef,
        write_schema: DType,
        write_options: VortexWriteOptions,
        write: W,
    ) -> Self {
        let (sender, stream) = write_queue(write_schema.clone());

        let size = write_options.size_estimator();
        let deferred_metadata = write_options.deferred_metadata();

        let mut write = CountingVortexWrite::new(write);
        let bytes_written = write.counter();
        let handle = session.handle().spawn(async move {
            let summary = write_options.write(&mut write, stream).await?;
            write.shutdown().await?;
            Ok(summary)
        });

        Self {
            handle: Some(handle),
            session: session.clone(),
            arrow_schema,
            write_schema,
            bytes_written,
            size,
            deferred_metadata,
            sender,
        }
    }

    pub fn into_raw(self: Box<Self>) -> jlong {
        Box::into_raw(self) as jlong
    }

    /// SAFETY: pointer must have been returned by [`Self::into_raw`].
    pub unsafe fn from_raw(pointer: jlong) -> Box<Self> {
        unsafe { Box::from_raw(pointer as *mut Self) }
    }

    /// SAFETY: pointer must have been returned by [`Self::into_raw`].
    pub unsafe fn from_ptr<'a>(pointer: jlong) -> &'a Self {
        debug_assert!(pointer != 0, "null writer pointer");
        unsafe { &*(pointer as *const Self) }
    }

    fn write_record_batch(&self, batch: RecordBatch) -> VortexResult<()> {
        let vortex_batch = self
            .session
            .arrow()
            .from_arrow_record_batch(batch, self.arrow_schema.as_ref())?;
        if !vortex_batch.dtype().eq(&self.write_schema) {
            return Err(vortex_err!(
                "write schema mismatch: expected {}, got {}",
                self.write_schema,
                vortex_batch.dtype()
            ));
        }
        let reservation = self.size.reserve_uncompressed(vortex_batch.nbytes());
        let mut sender = self.sender.clone();
        RUNTIME
            .block_on(async move { sender.send((Ok(vortex_batch), reservation)).await })
            .map_err(|e| vortex_err!("failed to send batch: {e}"))
    }

    fn bytes_written(&self) -> u64 {
        self.bytes_written.load(Ordering::Relaxed)
    }

    fn buffered_bytes(&self) -> u64 {
        self.size.buffered_bytes()
    }

    fn estimated_size(&self) -> u64 {
        self.size.estimated_size(self.bytes_written())
    }

    fn compression_ratio(&self) -> f64 {
        self.size.compression_ratio()
    }

    /// Close the input and wait for the write task. The task adds `metadata` to the footer.
    fn close(mut self, metadata: HashMap<String, ByteBuffer>) -> VortexResult<WriteSummary> {
        // The write task reads the metadata after the input ends, so add it first.
        self.deferred_metadata.extend(metadata);

        self.sender.disconnect();
        let handle = self
            .handle
            .take()
            .ok_or_else(|| vortex_err!("writer already closed"))?;
        RUNTIME.block_on(handle)
    }
}

fn write_summary_to_java<'local>(
    env: &mut Env<'local>,
    summary: &WriteSummary,
) -> Result<JObject<'local>, JNIError> {
    let statistics = footer_column_statistics(summary.footer());
    let columns = column_statistics_to_java(env, summary.footer(), statistics)?;

    Ok(env.new_object(
        jni::jni_str!("dev/vortex/api/VortexWriteSummary"),
        jni::jni_sig!("(JJ[Ldev/vortex/api/VortexColumnStatistics;)V"),
        &[
            JValue::Long(checked_jlong(summary.size(), "file size")?),
            JValue::Long(checked_jlong(summary.row_count(), "row count")?),
            JValue::Object(columns.as_ref()),
        ],
    )?)
}

/// Quantcast's editions. Named for the fork rather than the crate, so it can't collide with a
/// family upstream declares in `vortex-jni`.
static LIST_LAYOUT_FAMILY: EditionFamily = EditionFamily {
    name: "qc",
    origin: "vortex-jni",
    doc: "Components a JNI write can opt into without enabling the rest of the unstable edition, \
          maintained in Quantcast's fork.",
};

/// An edition holding only the list layout, which upstream keeps in its unstable edition. Frozen
/// in 0.91.0: every later reader reads its files. `list_layout_golden` holds its format fixed.
const LIST_LAYOUT_EDITION: EditionId = EditionId::new("qc", 2026, 11, 0);

static LIST_LAYOUT_DECLARATION: EditionDeclaration = EditionDeclaration {
    edition: Edition {
        id: LIST_LAYOUT_EDITION,
        immutable: true,
        min_library_version: Some("0.91.0"),
    },
    added: &[EditionMember::layout(&"vortex.list")],
};

pub(crate) fn register_list_layout_edition(session: &VortexSession) -> VortexResult<()> {
    session.editions().declare_family(&LIST_LAYOUT_FAMILY)?;
    session.register_edition(&LIST_LAYOUT_DECLARATION)
}

/// The session a write runs in.
///
/// The list layout is only written while an edition containing it is enabled, and enabled
/// editions are shared by every clone of a session. A list-layout write therefore runs in a fork
/// of the session, which keeps its registrations, with an enabled-editions set of its own.
pub(crate) fn write_session(
    session: &VortexSession,
    list_layout: bool,
) -> VortexResult<VortexSession> {
    if !list_layout {
        return Ok(session.clone());
    }
    let fork = session.fork();
    fork.register(EnabledEditions::default());
    for edition in session.enabled_editions().editions() {
        fork.enable_edition(edition)?;
    }
    fork.enable_edition(LIST_LAYOUT_EDITION)?;
    Ok(fork)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeWriter_create(
    mut env: EnvUnowned,
    _class: JClass,
    session_ptr: jlong,
    uri: JString,
    arrow_schema_addr: jlong,
    options: JObject,
    metadata: JObject,
    list_layout: jboolean,
    compact: jboolean,
) -> jlong {
    try_or_throw(&mut env, |env| {
        if session_ptr == 0 {
            throw_runtime!("null session pointer");
        }
        if arrow_schema_addr == 0 {
            throw_runtime!("null arrow schema address");
        }
        let session = &write_session(unsafe { session_ref(session_ptr) }, list_layout)?;

        // Owned, so dropping it releases the exporter's copy as the C data interface requires.
        let ffi_schema =
            unsafe { FFI_ArrowSchema::from_raw(arrow_schema_addr as *mut FFI_ArrowSchema) };
        let arrow_schema = Arc::new(Schema::try_from(&ffi_schema)?);
        let write_schema = session.arrow().from_arrow_schema(arrow_schema.as_ref())?;

        let file_path: String = uri.try_to_string(env)?;
        let properties: HashMap<String, String> = extract_properties(env, &options)?;
        let metadata = extract_metadata(env, &metadata)?;
        let resolved = resolve_store(&file_path, &properties)?;
        let write_options =
            session_write_options(session, compact).with_metadata_segments(metadata);
        // The same check runs inside `write`, but only once the write task is under way, where
        // it would surface as an opaque send failure on the first batch.
        write_options.validate_metadata()?;

        let writer = match resolved {
            ResolvedStore::Path(path) => {
                let file = RUNTIME.block_on(async {
                    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                        async_fs::create_dir_all(parent).await?;
                    }
                    Ok::<_, VortexError>(File::create(path).await?)
                })?;
                NativeWriter::start(session, arrow_schema, write_schema, write_options, file)
            }
            ResolvedStore::ObjectStore(store, path) => {
                let object_write =
                    RUNTIME.block_on(ObjectStoreWrite::new(Arc::new(Compat::new(store)), &path))?;
                NativeWriter::start(
                    session,
                    arrow_schema,
                    write_schema,
                    write_options,
                    object_write,
                )
            }
        };

        Ok(Box::new(writer).into_raw())
    })
}

/// Create a writer that streams the file into a caller-provided
/// `dev.vortex.io.NativeWritable` instead of a native storage client.
///
/// Bytes are pushed through blocking `write`/`flush` upcalls on the runtime thread
/// driving the write task. The Java caller owns the underlying stream and must close
/// it after `NativeWriter.close` returns; the native side only flushes.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeWriter_createStream(
    mut env: EnvUnowned,
    _class: JClass,
    session_ptr: jlong,
    writable: JObject,
    arrow_schema_addr: jlong,
    metadata: JObject,
    list_layout: jboolean,
    compact: jboolean,
) -> jlong {
    try_or_throw(&mut env, |env| {
        if session_ptr == 0 {
            throw_runtime!("null session pointer");
        }
        if arrow_schema_addr == 0 {
            throw_runtime!("null arrow schema address");
        }
        if writable.is_null() {
            throw_runtime!("null writable");
        }
        let session = &write_session(unsafe { session_ref(session_ptr) }, list_layout)?;

        // Owned, so dropping it releases the exporter's copy as the C data interface requires.
        let ffi_schema =
            unsafe { FFI_ArrowSchema::from_raw(arrow_schema_addr as *mut FFI_ArrowSchema) };
        let arrow_schema = Arc::new(Schema::try_from(&ffi_schema)?);
        let write_schema = session.arrow().from_arrow_schema(arrow_schema.as_ref())?;

        let metadata = extract_metadata(env, &metadata)?;
        let vm = env.get_java_vm()?;
        let direct_buffers =
            env.is_instance_of(&writable, jni::jni_str!("dev/vortex/io/ByteBufferWritable"))?;
        let writable = Arc::new(env.new_global_ref(&writable)?);
        let write_options =
            session_write_options(session, compact).with_metadata_segments(metadata);
        // See the note in `create`: validate before the write task can start.
        write_options.validate_metadata()?;

        let writer = NativeWriter::start(
            session,
            arrow_schema,
            write_schema,
            write_options,
            JavaWrite::new(env, vm, writable, direct_buffers)?,
        );

        Ok(Box::new(writer).into_raw())
    })
}

/// Write a batch to the Vortex file directly from Arrow C Data Interface pointers.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeWriter_writeBatch(
    mut env: EnvUnowned,
    _class: JClass,
    writer_ptr: jlong,
    arrow_array_addr: jlong,
    arrow_schema_addr: jlong,
) -> jboolean {
    if writer_ptr <= 0 {
        return JNI_FALSE;
    }

    try_or_throw(&mut env, |_env| {
        let writer = unsafe { NativeWriter::from_ptr(writer_ptr) };

        let ffi_array =
            unsafe { FFI_ArrowArray::from_raw(arrow_array_addr as *mut FFI_ArrowArray) };
        // Take the schema as well as the array. As the consumer, native code must release both, and dropping the
        // owned schema does that; borrowing it would leak the exporter's copy on every batch.
        let ffi_schema =
            unsafe { FFI_ArrowSchema::from_raw(arrow_schema_addr as *mut FFI_ArrowSchema) };

        let array_data = unsafe { arrow_array::ffi::from_ffi(ffi_array, &ffi_schema) }
            .map_err(|e| JNIError::Vortex(vortex_err!("failed to import Arrow FFI data: {e}")))?;

        let batch = RecordBatch::from(StructArray::from(array_data));
        writer.write_record_batch(batch)?;
        Ok(JNI_TRUE)
    })
}

/// Check that the buffer, child and dictionary counts of `array` agree with `data_type`.
///
/// The Arrow import does not validate, so a mismatch can panic or read out of bounds. This check
/// cannot find a different value width, for example `Int64` in the place of `Int32`.
fn check_ffi_layout(array: &FFI_ArrowArray, data_type: &DataType) -> VortexResult<()> {
    let expected = layout(data_type);
    let fixed_buffers = expected.buffers.len() + usize::from(expected.can_contain_null_mask);

    // View types have a variable count of data buffers.
    if expected.variadic {
        vortex_ensure!(
            array.num_buffers() > fixed_buffers,
            "Arrow array for {data_type} has {} buffers, expected more than {fixed_buffers}",
            array.num_buffers()
        );
    } else {
        vortex_ensure!(
            array.num_buffers() == fixed_buffers,
            "Arrow array for {data_type} has {} buffers, expected {fixed_buffers}",
            array.num_buffers()
        );
    }

    let children: Vec<&DataType> = match data_type {
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::FixedSizeList(field, _)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::Map(field, _) => vec![field.data_type()],
        DataType::Struct(fields) => fields.iter().map(|field| field.data_type()).collect(),
        DataType::Union(fields, _) => fields.iter().map(|(_, field)| field.data_type()).collect(),
        DataType::RunEndEncoded(run_ends, values) => {
            vec![run_ends.data_type(), values.data_type()]
        }
        _ => Vec::new(),
    };
    vortex_ensure!(
        array.num_children() == children.len(),
        "Arrow array for {data_type} has {} children, expected {}",
        array.num_children(),
        children.len()
    );

    for (index, child_type) in children.into_iter().enumerate() {
        check_ffi_layout(array.child(index), child_type)?;
    }

    match (data_type, array.dictionary()) {
        (DataType::Dictionary(_, value_type), Some(dictionary)) => {
            check_ffi_layout(dictionary, value_type)
        }
        (DataType::Dictionary(..), None) => {
            Err(vortex_err!("Arrow array for {data_type} has no dictionary"))
        }
        (_, Some(_)) => Err(vortex_err!(
            "Arrow array for {data_type} has a dictionary, expected none"
        )),
        (_, None) => Ok(()),
    }
}

/// Write a batch from an Arrow C Data Interface array that agrees with the writer schema.
///
/// Native code takes the array and releases it, also when the call fails.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeWriter_writeArray(
    mut env: EnvUnowned,
    _class: JClass,
    writer_ptr: jlong,
    arrow_array_addr: jlong,
) -> jboolean {
    if writer_ptr <= 0 {
        return JNI_FALSE;
    }

    try_or_throw(&mut env, |_env| {
        if arrow_array_addr == 0 {
            throw_runtime!("null arrow array address");
        }
        let writer = unsafe { NativeWriter::from_ptr(writer_ptr) };

        // Take the array first, so that each path below releases it.
        let ffi_array =
            unsafe { FFI_ArrowArray::from_raw(arrow_array_addr as *mut FFI_ArrowArray) };
        let dtype = DataType::Struct(writer.arrow_schema.fields().clone());
        check_ffi_layout(&ffi_array, &dtype)?;

        // SAFETY: `check_ffi_layout` passed, and the caller guarantees the value types.
        let array_data = unsafe { arrow_array::ffi::from_ffi_and_data_type(ffi_array, dtype) }
            .map_err(|e| JNIError::Vortex(vortex_err!("failed to import Arrow FFI data: {e}")))?;

        let batch = RecordBatch::from(StructArray::from(array_data));
        writer.write_record_batch(batch)?;
        Ok(JNI_TRUE)
    })
}

/// Import a Java Arrow batch with per-allocation ownership instead of a shared batch owner.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeWriter_writeBatchBuffers(
    mut env: EnvUnowned,
    _class: JClass,
    writer_ptr: jlong,
    arrow_array_addr: jlong,
    buffers: JObject,
) -> jboolean {
    if writer_ptr <= 0 {
        return JNI_FALSE;
    }
    try_or_throw(&mut env, |env| {
        // SAFETY: Java validates the root schema against the writer's schema and holds the
        // writer monitor for this call. Moving the array clears its release callback.
        let writer = unsafe { NativeWriter::from_ptr(writer_ptr) };
        let array = unsafe { FFI_ArrowArray::from_raw(arrow_array_addr as *mut FFI_ArrowArray) };
        let exported_addresses = exported_buffer_addresses(&array);
        let dtype = DataType::Struct(writer.arrow_schema.fields().clone());
        // SAFETY: the Java export keeps all buffers alive through the import, and the
        // caller validated its schema before exporting the array.
        let data = unsafe { arrow_array::ffi::from_ffi_and_data_type(array, dtype) }
            .map_err(|e| JNIError::Vortex(vortex_err!("failed to import Arrow FFI data: {e}")))?;
        let data = import_buffers(env, data, &exported_addresses, buffers)?;
        writer.write_record_batch(RecordBatch::from(StructArray::from(data)))?;
        Ok(JNI_TRUE)
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeWriter_bytesWritten(
    mut env: EnvUnowned,
    _class: JClass,
    writer_ptr: jlong,
) -> jlong {
    if writer_ptr <= 0 {
        return -1;
    }

    try_or_throw(&mut env, |_env| {
        let writer = unsafe { NativeWriter::from_ptr(writer_ptr) };
        Ok(checked_jlong(writer.bytes_written(), "bytes written")?)
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeWriter_bufferedBytes(
    mut env: EnvUnowned,
    _class: JClass,
    writer_ptr: jlong,
) -> jlong {
    if writer_ptr <= 0 {
        return -1;
    }

    try_or_throw(&mut env, |_env| {
        let writer = unsafe { NativeWriter::from_ptr(writer_ptr) };
        Ok(checked_jlong(writer.buffered_bytes(), "buffered bytes")?)
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeWriter_estimatedSize(
    mut env: EnvUnowned,
    _class: JClass,
    writer_ptr: jlong,
) -> jlong {
    if writer_ptr <= 0 {
        return -1;
    }

    try_or_throw(&mut env, |_env| {
        let writer = unsafe { NativeWriter::from_ptr(writer_ptr) };
        Ok(checked_jlong(writer.estimated_size(), "estimated size")?)
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeWriter_compressionRatio(
    _env: EnvUnowned,
    _class: JClass,
    writer_ptr: jlong,
) -> jdouble {
    if writer_ptr <= 0 {
        return -1.0;
    }

    let writer = unsafe { NativeWriter::from_ptr(writer_ptr) };
    writer.compression_ratio()
}

/// Finish the writer and return its summary.
///
/// `metadata` is a nullable `Map<String, byte[]>` that merges with the create-time segments.
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeWriter_finish(
    mut env: EnvUnowned,
    _class: JClass,
    writer_ptr: jlong,
    metadata: JObject,
) -> jobject {
    if writer_ptr <= 0 {
        return JObject::null().into_raw();
    }
    let writer = unsafe { NativeWriter::from_raw(writer_ptr) };
    try_or_throw(&mut env, |env| {
        let metadata = extract_metadata(env, &metadata)?;
        let summary = writer.close(metadata)?;
        Ok(write_summary_to_java(env, &summary)?.into_raw())
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeWriter_close(
    mut env: EnvUnowned,
    _class: JClass,
    writer_ptr: jlong,
) {
    if writer_ptr <= 0 {
        return;
    }
    let writer = unsafe { NativeWriter::from_raw(writer_ptr) };
    try_or_throw(&mut env, |_env| {
        writer.close(HashMap::new())?;
        Ok(())
    });
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::sync::Arc;

    use futures::StreamExt;
    use futures::future::BoxFuture;
    use parking_lot::Mutex;
    use vortex::array::IntoArray;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::ListArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::buffer::BufferHandle;
    use vortex::array::validity::Validity;
    use vortex::buffer::Alignment;
    use vortex::buffer::Buffer;
    use vortex::buffer::ByteBuffer;
    use vortex::buffer::ByteBufferMut;
    use vortex::editions::ComponentKind;
    use vortex::editions::EditionSessionExt;
    use vortex::editions::UNSTABLE;
    use vortex::error::VortexExpect;
    use vortex::error::VortexResult;
    use vortex::file::OpenOptionsSessionExt;
    use vortex::file::VortexOpenOptions;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::io::VortexReadAt;
    use vortex::io::runtime::BlockingRuntime;
    use vortex::layout::LayoutRef;
    use vortex::layout::VTable;
    use vortex::layout::layouts::flat::Flat;
    use vortex::layout::layouts::list::List;
    use vortex::scan::ScanRequest;
    use vortex::session::SessionExt;
    use vortex::session::SessionVar;
    use vortex::session::registry::Id;
    use vortex_arrow::ExecuteArrowExt;

    use super::LIST_LAYOUT_EDITION;
    use super::write_session;
    use crate::RUNTIME;
    use crate::data_source::file_open_options;
    use crate::session::new_session;

    #[derive(Clone, Debug)]
    struct Custom;

    impl SessionVar for Custom {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    #[test]
    fn list_layout_write_keeps_the_session_and_leaves_its_editions() -> VortexResult<()> {
        let session = *new_session();
        session.register(Custom);
        let mut editions = session.enabled_editions().editions();
        editions.sort();

        let write = write_session(&session, true)?;

        assert!(write.get_opt::<Custom>().is_some());
        let mut write_editions = write.enabled_editions().editions();
        write_editions.sort();
        let mut expected = editions.clone();
        expected.push(LIST_LAYOUT_EDITION);
        expected.sort();
        assert_eq!(write_editions, expected);
        let mut after = session.enabled_editions().editions();
        after.sort();
        assert_eq!(after, editions);
        Ok(())
    }

    const KINDS: [ComponentKind; 4] = [
        ComponentKind::Array,
        ComponentKind::Layout,
        ComponentKind::DType,
        ComponentKind::Aggregate,
    ];

    #[test]
    fn list_layout_edition_is_frozen() -> VortexResult<()> {
        let session = *new_session();
        session.editions().validate()?;
        let edition = session
            .editions()
            .find(&LIST_LAYOUT_EDITION)
            .vortex_expect("the list layout edition is registered");
        assert!(edition.immutable, "{edition:?}");
        assert!(!edition.is_draft(), "{edition:?}");
        Ok(())
    }

    #[test]
    fn list_layout_write_enables_only_the_list_layout() -> VortexResult<()> {
        let session = *new_session();
        session.editions().validate()?;
        let write = write_session(&session, true)?;
        for kind in KINDS {
            let mut expected = session.enabled_component_ids(kind);
            if kind == ComponentKind::Layout {
                expected.push(List.id());
                expected.sort_unstable();
            }
            assert_eq!(write.enabled_component_ids(kind), expected, "{kind}");
        }
        Ok(())
    }

    /// Layout ids in the tree, and every array id a flat leaf's segments may reference.
    fn file_ids(
        layout: &LayoutRef,
        layouts: &mut Vec<Id>,
        arrays: &mut Vec<Id>,
    ) -> VortexResult<()> {
        layouts.push(layout.encoding_id());
        if let Some(flat) = layout.as_opt::<Flat>() {
            arrays.extend_from_slice(flat.array_ctx().ids());
        }
        for child in layout.children()? {
            file_ids(&child, layouts, arrays)?;
        }
        Ok(())
    }

    #[test]
    fn list_layout_file_has_no_other_unstable_ids() -> VortexResult<()> {
        let session = write_session(&new_session(), true)?;
        let list = ListArray::try_new(
            Buffer::from_iter(0i64..65_536).into_array(),
            Buffer::from_iter((0u32..=8_192).map(|i| i * 8)).into_array(),
            Validity::NonNullable,
            &mut session.create_execution_ctx(),
        )?;
        let root = StructArray::from_fields(&[("arr", list.into_array())])?.into_array();
        let mut bytes = ByteBufferMut::empty();
        RUNTIME.block_on(
            session
                .write_options()
                .write(&mut bytes, root.to_array_stream()),
        )?;

        let file = session.open_options().open_buffer(bytes)?;
        let (mut layouts, mut arrays) = (Vec::new(), Vec::new());
        file_ids(file.footer().layout(), &mut layouts, &mut arrays)?;
        assert!(layouts.contains(&List.id()));
        let editions = session.editions();
        let unstable = |kind| -> Vec<Id> {
            editions
                .components_in(&UNSTABLE, kind)
                .into_iter()
                .map(|inclusion| inclusion.component_id)
                .filter(|id| *id != List.id())
                .collect()
        };
        let (unstable_layouts, unstable_arrays) = (
            unstable(ComponentKind::Layout),
            unstable(ComponentKind::Array),
        );
        let leaked: Vec<&Id> = layouts
            .iter()
            .filter(|id| unstable_layouts.contains(id))
            .chain(arrays.iter().filter(|id| unstable_arrays.contains(id)))
            .collect();
        assert!(leaked.is_empty(), "unstable ids in the file: {leaked:?}");
        Ok(())
    }

    /// Records the offset of every read, and doesn't merge reads, so each read is one segment.
    #[derive(Clone)]
    struct RecordingReads {
        bytes: ByteBuffer,
        offsets: Arc<Mutex<Vec<u64>>>,
    }

    impl VortexReadAt for RecordingReads {
        fn concurrency(&self) -> usize {
            8
        }

        fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
            let len = self.bytes.len() as u64;
            Box::pin(async move { Ok(len) })
        }

        fn read_at(
            &self,
            offset: u64,
            length: usize,
            alignment: Alignment,
        ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
            self.offsets.lock().push(offset);
            self.bytes.read_at(offset, length, alignment)
        }
    }

    /// Reads of a segment a scan of `bytes` in `splits` row ranges makes beyond the first.
    fn segments_read_twice(
        bytes: ByteBuffer,
        splits: u64,
        options: fn(VortexOpenOptions) -> VortexOpenOptions,
    ) -> VortexResult<usize> {
        let session = new_session();
        let offsets = Arc::new(Mutex::new(Vec::new()));
        let source = RecordingReads {
            bytes,
            offsets: Arc::clone(&offsets),
        };
        RUNTIME.block_on(async {
            let file = options(session.open_options())
                .open(Arc::new(source))
                .await?;
            offsets.lock().clear();
            let rows = file.row_count();
            let data_source = file.data_source()?;
            for split in 0..splits {
                let request = ScanRequest {
                    row_range: Some(rows * split / splits..rows * (split + 1) / splits),
                    ..Default::default()
                };
                let mut partitions = data_source.scan(request).await?.partitions();
                while let Some(partition) = partitions.next().await {
                    let mut chunks = partition?.execute()?;
                    while let Some(chunk) = chunks.next().await {
                        let mut ctx = session.create_execution_ctx();
                        chunk?.execute_arrow(None, &mut ctx)?;
                    }
                }
            }
            let mut offsets = offsets.lock().clone();
            offsets.sort_unstable();
            let total = offsets.len();
            offsets.dedup();
            Ok(total - offsets.len())
        })
    }

    /// A list column's elements are chunked apart from its rows, so a scan's row ranges share
    /// element segments, and each range asks for them after it has read its own offsets.
    #[test]
    fn list_layout_scan_reads_each_segment_once() -> VortexResult<()> {
        let session = write_session(&new_session(), true)?;
        let rows = 65_536u32;
        let elements = StructArray::from_fields(&[
            (
                "a",
                Buffer::from_iter(0i64..i64::from(rows) * 4).into_array(),
            ),
            (
                "b",
                Buffer::from_iter((0i64..i64::from(rows) * 4).map(|i| i % 7)).into_array(),
            ),
        ])?;
        let list = ListArray::try_new(
            elements.into_array(),
            Buffer::from_iter((0..=rows).map(|i| i * 4)).into_array(),
            Validity::NonNullable,
            &mut session.create_execution_ctx(),
        )?;
        let root = StructArray::from_fields(&[
            ("id", Buffer::from_iter(0i64..i64::from(rows)).into_array()),
            ("arr", list.into_array()),
        ])?
        .into_array();
        let mut bytes = ByteBufferMut::empty();
        RUNTIME.block_on(
            session
                .write_options()
                .write(&mut bytes, root.to_array_stream()),
        )?;
        let bytes = bytes.freeze();

        assert!(
            segments_read_twice(bytes.clone(), 4, |options| options)? > 0,
            "the test no longer reproduces re-reads"
        );
        assert_eq!(segments_read_twice(bytes, 4, file_open_options)?, 0);
        Ok(())
    }
}
