// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The session's segment IO service for opened files.
//!
//! Every file opened in a session registers with the session's [`FileSegmentIo`], which serves
//! the segment reads of planner-driven scans. A request is answered, in order of preference:
//!
//! 1. from the file's segment cache, which also holds the segments the footer read covered;
//! 2. by slicing the file, when it was opened from a buffer;
//! 3. by the file's coalescing read driver, the same one its segment source reads through.
//!
//! Reads through the driver are shared: every handle of a segment's byte range, from any session,
//! shares one read while any of them is alive, and the last one dropped cancels it if it has not
//! finished. A batch registers all of its reads with the driver before marking any of them wanted,
//! so the driver can coalesce them.

use std::fmt;
use std::fmt::Debug;
use std::sync::Arc;
use std::sync::Weak;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;

use futures::FutureExt;
use parking_lot::Mutex;
use parking_lot::RwLock;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::segments::FetchTracker;
use vortex_io::segments::IoIntent;
use vortex_io::segments::IoOwnerId;
use vortex_io::segments::RegisteredSource;
use vortex_io::segments::SegmentHandle;
use vortex_io::segments::SegmentIoService;
use vortex_io::segments::SegmentIoSession;
use vortex_io::segments::SegmentRequest;
use vortex_io::segments::SharedBytes;
use vortex_io::segments::SourceId;
use vortex_layout::segments::SegmentCache;
use vortex_layout::segments::SegmentId;
use vortex_session::SessionExt;
use vortex_session::SessionVar;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

use crate::SegmentSpec;
use crate::segment_io::driver::ReadId;
use crate::segment_io::driver::ReadRange;
pub(crate) use crate::segment_io::driver::SegmentDriver;

mod driver;

/// Serves the segment reads of every file registered in one session.
#[derive(Clone, Default)]
pub struct FileSegmentIo(Arc<Registry>);

#[derive(Default)]
struct Registry {
    files: RwLock<HashMap<SourceId, Weak<FileIo>>>,
    next: AtomicU32,
}

impl Debug for FileSegmentIo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FileSegmentIo")
            .field("files", &self.0.files.read().len())
            .finish()
    }
}

impl SessionVar for FileSegmentIo {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl FileSegmentIo {
    /// The service of `session`, created on first use.
    pub fn of(session: &VortexSession) -> Self {
        session.get::<Self>().clone()
    }

    /// Registers `file`. The service reads it while any clone of the registration is alive.
    pub(crate) fn register(&self, file: FileIo) -> RegisteredSource {
        let id = SourceId(self.0.next.fetch_add(1, Ordering::Relaxed));
        let file = Arc::new(file);
        self.0.files.write().insert(id, Arc::downgrade(&file));
        let registration = Arc::new(Registration {
            registry: Arc::downgrade(&self.0),
            id,
            _file: file,
        });
        RegisteredSource::new(Arc::new(self.clone()), id, registration)
    }
}

impl SegmentIoService for FileSegmentIo {
    fn session(&self) -> Arc<dyn SegmentIoSession> {
        Arc::new(FileSession {
            registry: Arc::clone(&self.0),
            files: Mutex::default(),
            fetches: Mutex::default(),
        })
    }
}

/// Keeps a file registered, and readable, while it lives.
struct Registration {
    registry: Weak<Registry>,
    id: SourceId,
    _file: Arc<FileIo>,
}

impl Drop for Registration {
    fn drop(&mut self) {
        if let Some(registry) = self.registry.upgrade() {
            registry.files.write().remove(&self.id);
        }
    }
}

/// How one registered file answers the segment reads its cache does not.
pub(crate) enum Backend {
    /// The file's own read driver, which coalesces and issues its reads.
    Driver(Arc<SegmentDriver>),
    /// The whole file, in memory.
    Buffer(ByteBuffer),
}

/// A registered file's segments and how to read them.
pub(crate) struct FileIo {
    specs: Arc<[SegmentSpec]>,
    cache: Arc<dyn SegmentCache>,
    backend: Backend,
    /// Reads in flight or finished and still held, by byte range. Segments that share a range
    /// share its read.
    reads: Mutex<HashMap<(u64, u32, Alignment), Weak<Read>>>,
}

impl FileIo {
    pub(crate) fn new(
        specs: Arc<[SegmentSpec]>,
        cache: Arc<dyn SegmentCache>,
        backend: Backend,
    ) -> Self {
        Self {
            specs,
            cache,
            backend,
            reads: Mutex::default(),
        }
    }

    /// A handle for `segment`. A read that must be marked wanted once the whole batch is
    /// registered is pushed to `wants`, with whether it is urgent.
    fn handle(
        &self,
        segment: SegmentId,
        want: Option<bool>,
        wants: &mut Vec<(Arc<Read>, bool)>,
    ) -> VortexResult<SegmentHandle> {
        let spec = self
            .specs
            .get(*segment as usize)
            .ok_or_else(|| vortex_err!("segment {segment} is not in the file"))?;
        if let Some(bytes) = self.cache.get_if_ready(segment) {
            return Ok(SegmentHandle::ready(Ok(BufferHandle::new_host(
                bytes.aligned(spec.alignment),
            ))));
        }
        let driver = match &self.backend {
            Backend::Buffer(buffer) => {
                let start = usize::try_from(spec.offset)?;
                let end = start + spec.length as usize;
                vortex_ensure_in_file(end, buffer.len(), segment)?;
                return Ok(SegmentHandle::ready(Ok(BufferHandle::new_host(
                    buffer.slice(start..end).aligned(spec.alignment),
                ))));
            }
            Backend::Driver(driver) => driver,
        };
        let key = (spec.offset, spec.length, spec.alignment);
        let mut reads = self.reads.lock();
        let read = match reads.get(&key).and_then(Weak::upgrade) {
            Some(read) => read,
            None => {
                let read = Arc::new(Read::register(
                    driver,
                    spec,
                    segment,
                    Arc::clone(&self.cache),
                )?);
                reads.retain(|_, read| read.strong_count() > 0);
                reads.insert(key, Arc::downgrade(&read));
                read
            }
        };
        if let Some(urgent) = want {
            wants.push((Arc::clone(&read), urgent));
        }
        Ok(SegmentHandle::new(read.bytes.clone(), read))
    }
}

fn vortex_ensure_in_file(end: usize, len: usize, segment: SegmentId) -> VortexResult<()> {
    if end > len {
        vortex_bail!(
            "segment {segment} ends at byte {end}, past the end of the file ({len} bytes)"
        );
    }
    Ok(())
}

/// One read registered with a file's driver, shared by every handle of its byte range.
struct Read {
    id: ReadId,
    driver: Arc<SegmentDriver>,
    /// Whether the driver was told the read is wanted, and whether urgently.
    wanted: AtomicBool,
    urgent: AtomicBool,
    bytes: SharedBytes,
}

impl Read {
    /// Registers a read of `spec` with the driver, without marking it wanted.
    fn register(
        driver: &Arc<SegmentDriver>,
        spec: &SegmentSpec,
        segment: SegmentId,
        cache: Arc<dyn SegmentCache>,
    ) -> VortexResult<Self> {
        let (id, receiver) = driver.register(ReadRange {
            offset: spec.offset,
            len: spec.length as usize,
            alignment: spec.alignment,
        })?;
        let bytes = async move {
            let bytes = receiver
                .await
                .map_err(|_| Arc::new(vortex_err!("the file's read driver dropped the read")))?
                .map_err(Arc::new)?;
            if let Some(buffer) = bytes.as_host_opt()
                && let Err(err) = cache.put(segment, buffer.clone()).await
            {
                tracing::warn!("Failed to store segment {segment} in cache: {err}");
            }
            Ok(bytes)
        }
        .boxed()
        .shared();
        Ok(Self {
            id,
            driver: Arc::clone(driver),
            wanted: AtomicBool::new(false),
            urgent: AtomicBool::new(false),
            bytes,
        })
    }

    /// Tells the driver the read is wanted, so it starts the read if it has not already. An
    /// urgent want upgrades an earlier background one.
    fn want(&self, urgent: bool) -> VortexResult<()> {
        let first = !self.wanted.swap(true, Ordering::Relaxed);
        let upgrade = urgent && !self.urgent.swap(true, Ordering::Relaxed);
        if first || upgrade {
            self.driver.want(self.id, urgent)?;
        }
        Ok(())
    }
}

impl Drop for Read {
    fn drop(&mut self) {
        if self.bytes.peek().is_none() {
            // The last handle is gone before the read finished.
            self.driver.withdraw(self.id);
        }
    }
}

/// One scope's reads through a [`FileSegmentIo`].
struct FileSession {
    registry: Arc<Registry>,
    /// The files this session has read, so later requests skip the registry.
    files: Mutex<HashMap<SourceId, Arc<FileIo>>>,
    fetches: Mutex<FetchTracker>,
}

impl FileSession {
    fn file(&self, source: SourceId) -> VortexResult<Arc<FileIo>> {
        let mut files = self.files.lock();
        if let Some(file) = files.get(&source) {
            return Ok(Arc::clone(file));
        }
        let file = self
            .registry
            .files
            .read()
            .get(&source)
            .and_then(Weak::upgrade)
            .ok_or_else(|| vortex_err!("{source:?} is not registered for segment reads"))?;
        files.insert(source, Arc::clone(&file));
        Ok(file)
    }
}

impl SegmentIoSession for FileSession {
    fn submit(
        &self,
        owner: IoOwnerId,
        batch: &[SegmentRequest],
    ) -> VortexResult<Vec<SegmentHandle>> {
        let mut wants = Vec::new();
        let handles = batch
            .iter()
            .map(|request| {
                if request.intent == IoIntent::Forget {
                    vortex_bail!("a Forget is withdrawn by dropping handles, not submitted");
                }
                let want = match request.intent {
                    IoIntent::Fetch => Some(true),
                    IoIntent::Prefetch => Some(false),
                    IoIntent::Announce | IoIntent::Forget => None,
                };
                self.file(request.key.source)?
                    .handle(request.key.segment, want, &mut wants)
            })
            .collect::<VortexResult<Vec<_>>>()?;
        // Every read of the batch is registered before any is wanted, so the driver can choose
        // physical reads with the whole batch in view.
        for (read, urgent) in wants {
            read.want(urgent)?;
        }
        let mut fetches = self.fetches.lock();
        for (request, handle) in batch.iter().zip(&handles) {
            if request.intent == IoIntent::Fetch && handle.try_get().is_none() {
                fetches.track(owner, handle);
            }
        }
        Ok(handles)
    }

    fn poll_ready(
        &self,
        cx: &mut Context<'_>,
        ready: &mut Vec<IoOwnerId>,
    ) -> Poll<VortexResult<()>> {
        self.fetches.lock().poll_ready(cx, ready)
    }
}

#[cfg(test)]
mod tests;
