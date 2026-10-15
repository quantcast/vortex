// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::poll_fn;

use futures::future::BoxFuture;
use rstest::rstest;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexError;
use vortex_io::CoalesceConfig;
use vortex_io::VortexReadAt;
use vortex_io::runtime::tokio::TokioRuntime;
use vortex_io::segments::SegmentKey;
use vortex_layout::segments::NoOpSegmentCache;
use vortex_metrics::DefaultMetricsRegistry;

use super::*;
use crate::segments::FileSegmentSource;
use crate::segments::InitialReadSegmentCache;
use crate::segments::RequestMetrics;

const FILE_LEN: usize = 256;
/// Segments 0..3 sit next to each other; segment 3 is far from them.
const OFFSETS: [u64; 4] = [0, 16, 32, 128];

/// An in-memory file that records the physical reads made of it.
#[derive(Clone)]
struct RecordingRead {
    bytes: ByteBuffer,
    reads: Arc<Mutex<Vec<(u64, usize)>>>,
}

impl VortexReadAt for RecordingRead {
    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        Some(CoalesceConfig {
            distance: 1024,
            max_size: 1024,
        })
    }

    fn concurrency(&self) -> usize {
        4
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
        self.bytes.read_at(offset, length, alignment)
    }
}

fn bytes() -> ByteBuffer {
    let mut bytes = ByteBufferMut::with_capacity(FILE_LEN);
    for byte in 0..FILE_LEN {
        bytes.push(u8::try_from(byte % 251).unwrap_or_default());
    }
    bytes.freeze()
}

fn specs() -> Arc<[SegmentSpec]> {
    OFFSETS
        .iter()
        .map(|&offset| SegmentSpec {
            offset,
            length: 16,
            alignment: Alignment::none(),
        })
        .collect()
}

/// A service with one file registered over a recording reader.
fn service(cache: Arc<dyn SegmentCache>) -> (FileSegmentIo, RegisteredSource, RecordingRead) {
    let reader = RecordingRead {
        bytes: bytes(),
        reads: Arc::default(),
    };
    let source = FileSegmentSource::open(
        specs(),
        reader.clone(),
        TokioRuntime::current(),
        RequestMetrics::new(&DefaultMetricsRegistry::default(), vec![]),
    );
    let service = FileSegmentIo::default();
    let registration = service.register(FileIo::new(
        specs(),
        cache,
        Backend::Driver(source.driver_io()),
    ));
    (service, registration, reader)
}

fn request(source: &RegisteredSource, segment: u32, intent: IoIntent) -> SegmentRequest {
    SegmentRequest::new(
        SegmentKey::new(source.id(), SegmentId::from(segment)),
        intent,
    )
}

/// The bytes of `segment`, as the file holds them.
fn expected(segment: usize) -> ByteBuffer {
    let start = usize::try_from(OFFSETS[segment]).unwrap_or_default();
    bytes().slice(start..start + 16)
}

async fn read(handle: &SegmentHandle) -> VortexResult<ByteBuffer> {
    handle
        .bytes()
        .await
        .map_err(VortexError::from)?
        .try_into_host()?
        .await
}

#[tokio::test]
async fn a_batch_coalesces_with_its_announcements() -> VortexResult<()> {
    let (service, file, reader) = service(Arc::new(NoOpSegmentCache));
    let session = service.session();
    let handles = session.submit(
        IoOwnerId(7),
        &[
            request(&file, 0, IoIntent::Fetch),
            request(&file, 1, IoIntent::Fetch),
            request(&file, 2, IoIntent::Announce),
        ],
    )?;
    assert_eq!(read(&handles[0]).await?, expected(0));
    assert_eq!(read(&handles[1]).await?, expected(1));
    // The announced segment rode along in the same physical read.
    assert_eq!(read(&handles[2]).await?, expected(2));
    assert_eq!(*reader.reads.lock(), vec![(0, 48)]);

    let mut ready = Vec::new();
    poll_fn(|cx| session.poll_ready(cx, &mut ready)).await?;
    assert!(ready.iter().all(|owner| *owner == IoOwnerId(7)));
    Ok(())
}

#[tokio::test]
async fn sessions_share_one_read() -> VortexResult<()> {
    let (service, file, reader) = service(Arc::new(NoOpSegmentCache));
    let first = service
        .session()
        .submit(IoOwnerId(0), &[request(&file, 3, IoIntent::Fetch)])?;
    let second = service
        .session()
        .submit(IoOwnerId(0), &[request(&file, 3, IoIntent::Prefetch)])?;
    assert_eq!(read(&first[0]).await?, expected(3));
    assert_eq!(read(&second[0]).await?, expected(3));
    assert_eq!(*reader.reads.lock(), vec![(128, 16)]);
    Ok(())
}

#[rstest]
#[case::held(true, vec![(0, 144)])]
#[case::dropped(false, vec![(0, 16)])]
#[tokio::test]
async fn dropping_every_handle_withdraws_the_read(
    #[case] hold: bool,
    #[case] reads: Vec<(u64, usize)>,
) -> VortexResult<()> {
    let (service, file, reader) = service(Arc::new(NoOpSegmentCache));
    let session = service.session();
    let announced = session.submit(IoOwnerId(0), &[request(&file, 3, IoIntent::Announce)])?;
    if !hold {
        drop(announced);
    }
    let fetched = session.submit(IoOwnerId(0), &[request(&file, 0, IoIntent::Fetch)])?;
    assert_eq!(read(&fetched[0]).await?, expected(0));
    assert_eq!(*reader.reads.lock(), reads);
    Ok(())
}

#[tokio::test]
async fn cached_segments_are_ready_without_a_read() -> VortexResult<()> {
    let cache = InitialReadSegmentCache {
        initial: HashMap::from_iter([(SegmentId::from(1), expected(1))]),
        fallback: Arc::new(NoOpSegmentCache),
    };
    let (service, file, reader) = service(Arc::new(cache));
    let handles = service
        .session()
        .submit(IoOwnerId(0), &[request(&file, 1, IoIntent::Fetch)])?;
    let bytes = handles[0]
        .try_get()
        .ok_or_else(|| vortex_err!("a cached segment is ready at submit"))??;
    assert_eq!(bytes.try_into_host_sync()?, expected(1));
    assert!(reader.reads.lock().is_empty());
    Ok(())
}

#[test]
fn buffered_files_answer_at_submit() -> VortexResult<()> {
    let service = FileSegmentIo::default();
    let file = service.register(FileIo::new(
        specs(),
        Arc::new(NoOpSegmentCache),
        Backend::Buffer(bytes()),
    ));
    let handles = service
        .session()
        .submit(IoOwnerId(0), &[request(&file, 2, IoIntent::Fetch)])?;
    let bytes = handles[0]
        .try_get()
        .ok_or_else(|| vortex_err!("a buffered segment is ready at submit"))??;
    assert_eq!(bytes.try_into_host_sync()?, expected(2));
    Ok(())
}

#[test]
fn a_dropped_registration_is_unreadable() -> VortexResult<()> {
    let service = FileSegmentIo::default();
    let file = service.register(FileIo::new(
        specs(),
        Arc::new(NoOpSegmentCache),
        Backend::Buffer(bytes()),
    ));
    let segment = request(&file, 0, IoIntent::Fetch);
    drop(file);
    assert!(service.session().submit(IoOwnerId(0), &[segment]).is_err());
    Ok(())
}

#[test]
fn forget_is_never_submitted() {
    let service = FileSegmentIo::default();
    let file = service.register(FileIo::new(
        specs(),
        Arc::new(NoOpSegmentCache),
        Backend::Buffer(bytes()),
    ));
    assert!(
        service
            .session()
            .submit(IoOwnerId(0), &[request(&file, 0, IoIntent::Forget)])
            .is_err()
    );
}
