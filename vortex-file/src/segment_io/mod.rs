// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The session's segment IO service, serving planner-driven scans over files' segment sources.
//!
//! A file is registered by its [`SegmentSource`], usually [`VortexFile::segment_source`], so
//! these reads go through the same segment cache, request sharing and coalescing read driver as
//! every other read of the file. The intents map onto the source's requests:
//!
//! - every request calls [`SegmentSource::request`], which for an opened file registers the read
//!   with its coalescing driver without starting it, so even an announcement can widen a read;
//! - a fetch or prefetch is then polled once, after the whole batch is registered, which starts
//!   the read;
//! - dropping every handle drops the source's request, which withdraws the read.
//!
//! [`VortexFile::segment_source`]: crate::VortexFile::segment_source

// TODO(joe): give this service a read driver of its own instead of reading through the files'
// v1 segment sources, so it can be optimised (fetches before prefetches, one scheduler across
// files) without changing v1.

use std::fmt;
use std::fmt::Debug;
use std::sync::Arc;
use std::sync::Weak;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;

use futures::FutureExt;
use futures::TryFutureExt;
use parking_lot::Mutex;
use parking_lot::RwLock;
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
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSource;
use vortex_session::SessionExt;
use vortex_session::SessionVar;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

/// Serves the segment reads of every source registered in one session.
#[derive(Clone, Default)]
pub struct FileSegmentIo(Arc<Registry>);

type Source = Arc<dyn SegmentSource>;

#[derive(Default)]
struct Registry {
    sources: RwLock<HashMap<SourceId, Weak<dyn SegmentSource>>>,
    next: AtomicU32,
}

impl Debug for FileSegmentIo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FileSegmentIo")
            .field("sources", &self.0.sources.read().len())
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

    /// Registers `source`. The service reads it while any clone of the registration is alive.
    pub fn register(&self, source: Source) -> RegisteredSource {
        let id = SourceId(self.0.next.fetch_add(1, Ordering::Relaxed));
        self.0.sources.write().insert(id, Arc::downgrade(&source));
        let registration = Arc::new(Registration {
            registry: Arc::downgrade(&self.0),
            id,
            _source: source,
        });
        RegisteredSource::new(Arc::new(self.clone()), id, registration)
    }
}

impl SegmentIoService for FileSegmentIo {
    fn session(&self) -> Arc<dyn SegmentIoSession> {
        Arc::new(FileSession {
            registry: Arc::clone(&self.0),
            sources: Mutex::default(),
            fetches: Mutex::default(),
        })
    }
}

/// Keeps a source registered, and readable, while it lives.
struct Registration {
    registry: Weak<Registry>,
    id: SourceId,
    _source: Source,
}

impl Drop for Registration {
    fn drop(&mut self) {
        if let Some(registry) = self.registry.upgrade() {
            registry.sources.write().remove(&self.id);
        }
    }
}

/// One scope's reads through a [`FileSegmentIo`].
struct FileSession {
    registry: Arc<Registry>,
    /// The sources this session has read, so later requests skip the registry.
    sources: Mutex<HashMap<SourceId, Source>>,
    fetches: Mutex<FetchTracker>,
}

impl FileSession {
    fn source(&self, id: SourceId) -> VortexResult<Source> {
        let mut sources = self.sources.lock();
        if let Some(source) = sources.get(&id) {
            return Ok(Arc::clone(source));
        }
        let source = self
            .registry
            .sources
            .read()
            .get(&id)
            .and_then(Weak::upgrade)
            .ok_or_else(|| vortex_err!("{id:?} is not registered for segment reads"))?;
        sources.insert(id, Arc::clone(&source));
        Ok(source)
    }
}

impl SegmentIoSession for FileSession {
    fn submit(
        &self,
        owner: IoOwnerId,
        batch: &[SegmentRequest],
    ) -> VortexResult<Vec<SegmentHandle>> {
        let mut wanted = Vec::new();
        let handles = batch
            .iter()
            .map(|request| {
                if request.intent == IoIntent::Forget {
                    vortex_bail!("a Forget is withdrawn by dropping handles, not submitted");
                }
                let source = self.source(request.key.source)?;
                let bytes: SharedBytes = source
                    .request(SegmentId::from(request.key.segment))
                    .map_err(Arc::new)
                    .boxed()
                    .shared();
                if request.intent != IoIntent::Announce {
                    wanted.push(bytes.clone());
                }
                // The handle keeps the source alive while its read may be in flight.
                Ok(SegmentHandle::new(bytes, Arc::new(source)))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        // Every read of the batch is registered before any is started, so the driver can choose
        // physical reads with the whole batch in view. The first poll starts a read; later polls
        // through the handles take its result.
        for bytes in wanted {
            drop(bytes.now_or_never());
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
