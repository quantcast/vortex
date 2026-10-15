// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The read driver of one registered file.
//!
//! The driver is one task. Sessions talk to it through three messages: a read is registered,
//! later marked wanted, and withdrawn if nothing needs it any more. Registering a read does no IO;
//! it only makes the range available for coalescing. The driver chooses physical reads only when
//! the reader has a free slot, after taking every message already queued, so a batch registered
//! together is seen together.
//!
//! A physical read starts at a wanted range, fetches before prefetches, and extends over the
//! registered ranges around it, wanted or not, while the gaps stay within the reader's coalescing
//! distance and the read within its maximum size.

use std::collections::BTreeSet;
use std::collections::VecDeque;
use std::future::poll_fn;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::task::Poll;

use futures::FutureExt;
use futures::StreamExt;
use futures::channel::mpsc;
use futures::channel::oneshot;
use futures::stream::FuturesUnordered;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_io::CoalesceConfig;
use vortex_io::VortexReadAt;
use vortex_io::runtime::Handle;
use vortex_io::runtime::Task;
use vortex_utils::aliases::hash_map::HashMap;

use crate::segments::RequestMetrics;

/// Identifies a read registered with a [`SegmentDriver`].
pub(crate) type ReadId = u64;

/// A byte range of the file and the alignment its bytes must have.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ReadRange {
    pub(crate) offset: u64,
    pub(crate) len: usize,
    pub(crate) alignment: Alignment,
}

impl ReadRange {
    fn end(&self) -> u64 {
        self.offset + self.len as u64
    }
}

/// The receiving end of a registered read.
pub(crate) type ReadReceiver = oneshot::Receiver<VortexResult<BufferHandle>>;

enum Command {
    Register {
        id: ReadId,
        range: ReadRange,
        reply: oneshot::Sender<VortexResult<BufferHandle>>,
    },
    Want {
        id: ReadId,
        urgent: bool,
    },
    Withdraw(ReadId),
}

/// The read driver of one file: a task that coalesces and issues the file's reads.
///
/// Dropping the driver stops its task.
pub(crate) struct SegmentDriver {
    commands: mpsc::UnboundedSender<Command>,
    next_id: AtomicU64,
    _task: Task<()>,
}

impl SegmentDriver {
    /// Starts the driver of a file read through `reader`, whose segments are aligned to at most
    /// `max_alignment`.
    pub(crate) fn spawn<R: VortexReadAt + 'static>(
        reader: R,
        max_alignment: Alignment,
        handle: &Handle,
        metrics: RequestMetrics,
    ) -> Self {
        let concurrency = reader.concurrency();
        if concurrency == 0 {
            vortex_panic!(
                "VortexReadAt::concurrency returned 0 (uri={:?}); this would stall I/O",
                reader.uri()
            );
        }
        let coalesce = reader.coalesce_config().map(|mut config| {
            // Aligning the coalesced start down can add up to (alignment - 1) bytes.
            // Increase max_size to keep the effective payload window consistent.
            let extra = (max_alignment.as_usize() as u64).saturating_sub(1);
            config.max_size = config.max_size.saturating_add(extra);
            config
        });
        let (commands, receiver) = mpsc::unbounded();
        let state = State {
            reader: Arc::new(reader),
            coalesce,
            max_alignment,
            concurrency,
            metrics,
            pending: HashMap::default(),
            by_offset: BTreeSet::new(),
            urgent: VecDeque::new(),
            background: VecDeque::new(),
            in_flight: FuturesUnordered::new(),
        };
        let task = handle.spawn_io(run(state, receiver));
        Self {
            commands,
            next_id: AtomicU64::new(0),
            _task: task,
        }
    }

    /// Registers a read of `range`, without wanting it yet.
    pub(crate) fn register(&self, range: ReadRange) -> VortexResult<(ReadId, ReadReceiver)> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (reply, receiver) = oneshot::channel();
        self.send(Command::Register { id, range, reply })?;
        Ok((id, receiver))
    }

    /// Marks a registered read wanted. Urgent reads are chosen before the others.
    pub(crate) fn want(&self, id: ReadId, urgent: bool) -> VortexResult<()> {
        self.send(Command::Want { id, urgent })
    }

    /// Withdraws a registered read that has not started.
    pub(crate) fn withdraw(&self, id: ReadId) {
        // A stopped driver has nothing left to withdraw.
        drop(self.send(Command::Withdraw(id)));
    }

    fn send(&self, command: Command) -> VortexResult<()> {
        self.commands
            .unbounded_send(command)
            .map_err(|_| vortex_err!("the file's read driver has stopped"))
    }
}

/// A registered read that no physical read has taken yet.
struct Pending {
    range: ReadRange,
    reply: oneshot::Sender<VortexResult<BufferHandle>>,
    /// `None` until wanted, then whether the want is urgent.
    wanted: Option<bool>,
}

/// One physical read and the registered reads it answers.
struct PhysicalRead {
    offset: u64,
    len: usize,
    members: Vec<(ReadRange, oneshot::Sender<VortexResult<BufferHandle>>)>,
}

type InFlight = FuturesUnordered<
    Pin<Box<dyn Future<Output = (PhysicalRead, VortexResult<BufferHandle>)> + Send>>,
>;

struct State {
    reader: Arc<dyn VortexReadAt>,
    coalesce: Option<CoalesceConfig>,
    max_alignment: Alignment,
    concurrency: usize,
    metrics: RequestMetrics,
    pending: HashMap<ReadId, Pending>,
    /// Registered reads no physical read has taken, in file order.
    by_offset: BTreeSet<(u64, ReadId)>,
    /// Wanted reads, oldest first. May name reads already taken or withdrawn.
    urgent: VecDeque<ReadId>,
    background: VecDeque<ReadId>,
    in_flight: InFlight,
}

async fn run(mut state: State, mut commands: mpsc::UnboundedReceiver<Command>) {
    let mut closed = false;
    poll_fn(|cx| {
        loop {
            // Take every queued message before choosing reads, so a batch is seen whole.
            while !closed {
                match commands.poll_next_unpin(cx) {
                    Poll::Ready(Some(command)) => state.apply(command),
                    Poll::Ready(None) => closed = true,
                    Poll::Pending => break,
                }
            }
            let mut progressed = false;
            while let Poll::Ready(Some((read, result))) = state.in_flight.poll_next_unpin(cx) {
                complete(read, result);
                progressed = true;
            }
            while state.in_flight.len() < state.concurrency
                && let Some(read) = state.choose()
            {
                state.start(read);
                progressed = true;
            }
            if closed && state.in_flight.is_empty() {
                return Poll::Ready(());
            }
            // New reads must be polled once to register for wake-ups.
            if !progressed {
                return Poll::Pending;
            }
        }
    })
    .await
}

impl State {
    fn apply(&mut self, command: Command) {
        match command {
            Command::Register { id, range, reply } => {
                self.by_offset.insert((range.offset, id));
                self.pending.insert(
                    id,
                    Pending {
                        range,
                        reply,
                        wanted: None,
                    },
                );
            }
            Command::Want { id, urgent } => {
                // A background read becomes urgent when fetched; its earlier queue entry is
                // skipped once the read is taken.
                if let Some(pending) = self.pending.get_mut(&id)
                    && pending
                        .wanted
                        .is_none_or(|was_urgent| urgent && !was_urgent)
                {
                    pending.wanted = Some(urgent);
                    if urgent {
                        self.urgent.push_back(id);
                    } else {
                        self.background.push_back(id);
                    }
                }
            }
            Command::Withdraw(id) => {
                if let Some(pending) = self.pending.remove(&id) {
                    self.by_offset.remove(&(pending.range.offset, id));
                }
            }
        }
    }

    /// The next wanted read no physical read has taken: an urgent one, if any.
    fn next_wanted(&mut self) -> Option<ReadId> {
        loop {
            let id = self
                .urgent
                .pop_front()
                .or_else(|| self.background.pop_front())?;
            if self.pending.contains_key(&id) {
                return Some(id);
            }
        }
    }

    fn take(
        &mut self,
        id: ReadId,
        members: &mut Vec<(ReadRange, oneshot::Sender<VortexResult<BufferHandle>>)>,
    ) {
        if let Some(pending) = self.pending.remove(&id) {
            self.by_offset.remove(&(pending.range.offset, id));
            members.push((pending.range, pending.reply));
        }
    }

    /// Chooses the physical read for the next wanted read, extended over its registered
    /// neighbours in file order while the gaps and the total stay within the coalescing window.
    fn choose(&mut self) -> Option<PhysicalRead> {
        let head = self.next_wanted()?;
        let range = self.pending.get(&head)?.range;
        let mut members = Vec::new();
        self.take(head, &mut members);
        let Some(window) = self.coalesce else {
            return Some(PhysicalRead {
                offset: range.offset,
                len: range.len,
                members,
            });
        };
        let align = self.max_alignment.as_usize() as u64;
        let fits = |start: u64, end: u64| end - (start - start % align) <= window.max_size;
        let (mut start, mut end) = (range.offset, range.end());
        loop {
            let mut grew = false;
            // Forwards first: scans read ahead of the ranges they want.
            if let Some(&(offset, id)) = self.by_offset.range((start, 0)..).next()
                && let Some(next) = self.pending.get(&id).map(|pending| pending.range)
                && offset <= end.saturating_add(window.distance)
                && fits(start, end.max(next.end()))
            {
                end = end.max(next.end());
                self.take(id, &mut members);
                grew = true;
            }
            if let Some(&(offset, id)) = self.by_offset.range(..(start, 0)).next_back()
                && let Some(previous) = self.pending.get(&id).map(|pending| pending.range)
                && previous.end().saturating_add(window.distance) >= start
                && fits(offset, end.max(previous.end()))
            {
                start = offset;
                end = end.max(previous.end());
                self.take(id, &mut members);
                grew = true;
            }
            if !grew {
                break;
            }
        }
        if members.len() == 1 {
            return Some(PhysicalRead {
                offset: range.offset,
                len: range.len,
                members,
            });
        }
        let start = start - start % align;
        Some(PhysicalRead {
            offset: start,
            len: usize::try_from(end - start).unwrap_or(usize::MAX),
            members,
        })
    }

    fn start(&mut self, read: PhysicalRead) {
        match read.members.len() {
            1 => self.metrics.individual_requests.add(1),
            members => {
                self.metrics.coalesced_requests.add(1);
                self.metrics.num_requests_coalesced.update(members as f64);
            }
        }
        self.metrics.read_ranges_calls.add(1);
        // A single member keeps its own alignment; a coalesced read is aligned for all of them.
        let alignment = match read.members.as_slice() {
            [(range, _)] => range.alignment,
            _ => self.max_alignment,
        };
        let bytes = self.reader.read_at(read.offset, read.len, alignment);
        self.in_flight
            .push(async move { (read, bytes.await) }.boxed());
    }
}

/// Hands each registered read its slice of a finished physical read.
fn complete(read: PhysicalRead, result: VortexResult<BufferHandle>) {
    let result = result.and_then(|bytes| {
        if bytes.len() != read.len {
            vortex_bail!(
                "expected {} bytes at {} but the reader returned {}",
                read.len,
                read.offset,
                bytes.len()
            );
        }
        Ok(bytes)
    });
    let bytes = match result {
        Ok(bytes) => bytes,
        Err(error) => {
            let error = Arc::new(error);
            for (_, reply) in read.members {
                drop(reply.send(Err(Arc::clone(&error).into())));
            }
            return;
        }
    };
    for (range, reply) in read.members {
        let start = usize::try_from(range.offset - read.offset).unwrap_or(usize::MAX);
        let slice = bytes
            .slice(start..start + range.len)
            .ensure_aligned(range.alignment);
        // A withdrawn reader dropped its receiver.
        drop(reply.send(slice));
    }
}

#[cfg(test)]
mod tests {
    use futures::future::BoxFuture;
    use parking_lot::Mutex;
    use rstest::rstest;
    use tokio::sync::Semaphore;
    use vortex_buffer::ByteBuffer;
    use vortex_io::runtime::tokio::TokioRuntime;
    use vortex_metrics::DefaultMetricsRegistry;

    use super::*;

    /// An in-memory file that records reads, and starts each only once a permit is released.
    #[derive(Clone)]
    struct GatedRead {
        bytes: ByteBuffer,
        coalesce: Option<CoalesceConfig>,
        concurrency: usize,
        fail: bool,
        reads: Arc<Mutex<Vec<(u64, usize)>>>,
        permits: Arc<Semaphore>,
    }

    impl VortexReadAt for GatedRead {
        fn coalesce_config(&self) -> Option<CoalesceConfig> {
            self.coalesce
        }

        fn concurrency(&self) -> usize {
            self.concurrency
        }

        fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
            self.bytes.size()
        }

        fn read_at(
            &self,
            offset: u64,
            length: usize,
            alignment: Alignment,
        ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
            self.reads.lock().push((offset, length));
            let permits = Arc::clone(&self.permits);
            let read = self.bytes.read_at(offset, length, alignment);
            let fail = self.fail;
            async move {
                permits
                    .acquire()
                    .await
                    .map_err(|err| vortex_err!("{err}"))?
                    .forget();
                if fail {
                    vortex_bail!("disk on fire");
                }
                read.await
            }
            .boxed()
        }
    }

    fn reader(coalesce: Option<(u64, u64)>, concurrency: usize, fail: bool) -> GatedRead {
        GatedRead {
            bytes: ByteBuffer::copy_from((0..=255u8).collect::<Vec<_>>()),
            coalesce: coalesce.map(|(distance, max_size)| CoalesceConfig { distance, max_size }),
            concurrency,
            fail,
            reads: Arc::default(),
            permits: Arc::new(Semaphore::new(0)),
        }
    }

    fn driver(reader: &GatedRead) -> SegmentDriver {
        SegmentDriver::spawn(
            reader.clone(),
            Alignment::none(),
            &TokioRuntime::current(),
            RequestMetrics::new(&DefaultMetricsRegistry::default(), vec![]),
        )
    }

    fn range(offset: u64, len: usize) -> ReadRange {
        ReadRange {
            offset,
            len,
            alignment: Alignment::none(),
        }
    }

    async fn bytes(receiver: ReadReceiver) -> VortexResult<Vec<u8>> {
        let bytes = receiver
            .await
            .map_err(|_| vortex_err!("dropped"))??
            .try_into_host()?
            .await?;
        Ok(bytes.as_slice().to_vec())
    }

    #[tokio::test]
    async fn fetches_are_read_before_prefetches() -> VortexResult<()> {
        // One slot, and the first read holds it until released, so the queue builds up.
        let reader = reader(None, 1, false);
        let driver = driver(&reader);
        let (first, first_rx) = driver.register(range(0, 4))?;
        driver.want(first, true)?;
        let (prefetch, prefetch_rx) = driver.register(range(100, 4))?;
        let (fetch, fetch_rx) = driver.register(range(200, 4))?;
        driver.want(prefetch, false)?;
        driver.want(fetch, true)?;
        reader.permits.add_permits(3);
        assert_eq!(bytes(first_rx).await?, vec![0, 1, 2, 3]);
        assert_eq!(bytes(fetch_rx).await?, vec![200, 201, 202, 203]);
        assert_eq!(bytes(prefetch_rx).await?, vec![100, 101, 102, 103]);
        assert_eq!(*reader.reads.lock(), vec![(0, 4), (200, 4), (100, 4)]);
        Ok(())
    }

    #[tokio::test]
    async fn a_prefetch_becomes_urgent_when_fetched() -> VortexResult<()> {
        let reader = reader(None, 1, false);
        let driver = driver(&reader);
        let (first, first_rx) = driver.register(range(0, 4))?;
        driver.want(first, true)?;
        let (early, early_rx) = driver.register(range(100, 4))?;
        let (late, late_rx) = driver.register(range(200, 4))?;
        driver.want(early, false)?;
        driver.want(late, false)?;
        driver.want(late, true)?;
        reader.permits.add_permits(3);
        for receiver in [first_rx, late_rx, early_rx] {
            bytes(receiver).await?;
        }
        assert_eq!(*reader.reads.lock(), vec![(0, 4), (200, 4), (100, 4)]);
        Ok(())
    }

    #[tokio::test]
    async fn withdrawn_reads_are_never_issued() -> VortexResult<()> {
        let reader = reader(Some((1024, 1024)), 1, false);
        let driver = driver(&reader);
        let (withdrawn, withdrawn_rx) = driver.register(range(8, 4))?;
        driver.withdraw(withdrawn);
        let (wanted, wanted_rx) = driver.register(range(0, 4))?;
        driver.want(wanted, true)?;
        reader.permits.add_permits(1);
        assert_eq!(bytes(wanted_rx).await?, vec![0, 1, 2, 3]);
        // The withdrawn neighbour neither widened the read nor was answered.
        assert_eq!(*reader.reads.lock(), vec![(0, 4)]);
        assert!(withdrawn_rx.await.is_err());
        Ok(())
    }

    #[rstest]
    #[case::within_window(Some((32, 64)), vec![(0, 40)])]
    #[case::gap_too_wide(Some((4, 64)), vec![(0, 4), (32, 8)])]
    #[case::too_large(Some((32, 16)), vec![(0, 4), (32, 8)])]
    #[case::no_coalescing(None, vec![(0, 4), (32, 8)])]
    #[tokio::test]
    async fn coalescing_stays_within_its_window(
        #[case] coalesce: Option<(u64, u64)>,
        #[case] expected: Vec<(u64, usize)>,
    ) -> VortexResult<()> {
        let reader = reader(coalesce, 4, false);
        let driver = driver(&reader);
        let (a, a_rx) = driver.register(range(0, 4))?;
        let (b, b_rx) = driver.register(range(32, 8))?;
        driver.want(a, true)?;
        driver.want(b, true)?;
        reader.permits.add_permits(2);
        assert_eq!(bytes(a_rx).await?, vec![0, 1, 2, 3]);
        assert_eq!(bytes(b_rx).await?, (32..40).collect::<Vec<u8>>());
        let mut reads = reader.reads.lock().clone();
        reads.sort_unstable();
        assert_eq!(reads, expected);
        Ok(())
    }

    #[tokio::test]
    async fn a_failed_read_fails_every_member() -> VortexResult<()> {
        let reader = reader(Some((1024, 1024)), 1, true);
        let driver = driver(&reader);
        let (a, a_rx) = driver.register(range(0, 4))?;
        let (_b, b_rx) = driver.register(range(8, 4))?;
        driver.want(a, true)?;
        reader.permits.add_permits(1);
        for receiver in [a_rx, b_rx] {
            let error = bytes(receiver).await.err().map(|error| error.to_string());
            assert!(error.is_some_and(|error| error.contains("disk on fire")));
        }
        assert_eq!(*reader.reads.lock(), vec![(0, 12)]);
        Ok(())
    }
}
