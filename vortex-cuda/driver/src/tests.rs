// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::c_void;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use crate::CudaContext;
use crate::CudaStream;
use crate::DevicePtr;
use crate::DevicePtrMut;
use crate::DriverError;
use crate::result::stream::launch_host_function;
use crate::sys;

/// A context and two streams, or `None` without a CUDA device.
fn streams() -> Option<(Arc<CudaContext>, Arc<CudaStream>, Arc<CudaStream>)> {
    // SAFETY: Only probes for the driver library.
    if !unsafe { sys::is_culib_present() } {
        return None;
    }
    let context = CudaContext::new(0).ok()?;
    let first = context.new_stream().ok()?;
    let second = context.new_stream().ok()?;
    Some((context, first, second))
}

#[test]
fn host_device_round_trip() -> Result<(), DriverError> {
    let Some((context, stream, _)) = streams() else {
        return Ok(());
    };
    let values: Vec<u32> = (0..100_000).collect();
    let device = stream.clone_htod(&values)?;
    assert_eq!(stream.clone_dtoh(&device)?, values);
    assert_eq!(stream.clone_dtoh(&device.slice(10..20))?, &values[10..20]);
    context.check_err()
}

/// A write on one stream is visible to reads on another, through views as well.
#[test]
fn cross_stream_reads_follow_writes() -> Result<(), DriverError> {
    let Some((context, writer, reader)) = streams() else {
        return Ok(());
    };
    for round in 0..50u8 {
        let mut device = writer.alloc_zeros::<u8>(64 << 20)?;
        {
            let bytes = device.num_bytes();
            let (pointer, _record) = device.device_ptr_mut(&writer);
            // SAFETY: The allocation holds `bytes` bytes.
            unsafe { sys::cuMemsetD8Async(pointer, round, bytes, writer.cu_stream()) }.result()?;
        }
        let tail = reader.clone_dtoh(&device.slice((64 << 20) - 16..))?;
        assert!(tail.iter().all(|value| *value == round));
    }
    context.check_err()
}

/// A free after a read on another stream waits for that read, and the memory a later
/// allocation reuses is not overwritten under it.
#[test]
fn frees_follow_reads_on_other_streams() -> Result<(), DriverError> {
    let Some((context, owner, reader)) = streams() else {
        return Ok(());
    };
    let source = owner.clone_htod(&vec![7u8; 32 << 20])?;
    let mut copies = Vec::new();
    for _ in 0..8 {
        // SAFETY: Fully written by the copy below.
        let mut target = unsafe { reader.alloc::<u8>(32 << 20) }?;
        reader.memcpy_dtod(&source, &mut target)?;
        copies.push(target);
    }
    drop(source);
    let reused = owner.clone_htod(&vec![9u8; 32 << 20])?;
    for copy in &copies {
        assert!(reader.clone_dtoh(copy)?.iter().all(|value| *value == 7));
    }
    assert!(owner.clone_dtoh(&reused)?.iter().all(|value| *value == 9));
    context.check_err()
}

/// Blocks the calling driver thread until the gate's sender sends or drops.
unsafe extern "C" fn wait_for_gate(gate: *mut c_void) {
    // SAFETY: The test leaks a boxed receiver for this one call and reclaims it here.
    let gate = unsafe { Box::from_raw(gate.cast::<mpsc::Receiver<()>>()) };
    let _opened = gate.recv();
}

/// Freeing an allocation another stream still reads does not hold up the allocation stream:
/// its later work runs while the read is blocked, as later file copies must while a decode
/// reads earlier segments.
#[test]
fn frees_do_not_stall_the_allocation_stream() -> Result<(), DriverError> {
    let Some((context, owner, reader)) = streams() else {
        return Ok(());
    };
    let (open, gate) = mpsc::channel::<()>();
    // SAFETY: The reader's work runs only after the gate opens; the host function makes no CUDA
    // calls and owns the boxed receiver.
    unsafe {
        launch_host_function(
            reader.cu_stream(),
            wait_for_gate,
            Box::into_raw(Box::new(gate)).cast(),
        )
    }?;
    let device = owner.clone_htod(&[3u8; 1 << 20])?;
    // SAFETY: Fully written by the copy below.
    let mut copy = unsafe { reader.alloc::<u8>(1 << 20) }?;
    reader.memcpy_dtod(&device, &mut copy)?;
    drop(device);
    // An asynchronous memset: a pageable upload could block this thread behind the gate.
    let later = owner.alloc_zeros::<u8>(1 << 20)?;

    let (done, finished) = mpsc::channel();
    let waiter = Arc::clone(&owner);
    let synchronized = std::thread::spawn(move || {
        let _sent = done.send(waiter.synchronize());
    });
    let unblocked = finished.recv_timeout(Duration::from_secs(10));
    drop(open);
    assert!(synchronized.join().is_ok());
    assert!(
        matches!(unblocked, Ok(Ok(()))),
        "the allocation stream waited for the blocked read: {unblocked:?}"
    );
    assert!(owner.clone_dtoh(&later)?.iter().all(|value| *value == 0));
    assert!(reader.clone_dtoh(&copy)?.iter().all(|value| *value == 3));
    context.check_err()
}

/// Same-stream accesses take no events; cross-stream ones reuse returned events.
#[test]
fn events_are_pooled() -> Result<(), DriverError> {
    let Some((context, first, second)) = streams() else {
        return Ok(());
    };
    for _ in 0..100 {
        let device = first.clone_htod(&[1u32; 1024])?;
        let (_pointer, record) = device.device_ptr(&second);
        drop(record);
    }
    // One write event and one read event, reused for every buffer.
    assert!(context.pooled_events() <= 2);
    context.check_err()
}
