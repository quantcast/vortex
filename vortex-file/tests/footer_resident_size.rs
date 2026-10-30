// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Checks that a footer's approximate byte size, which weighs it in the footer cache, tracks the
//! heap memory the parsed footer keeps alive.

#![expect(clippy::tests_outside_test_module)]

use std::alloc::GlobalAlloc;
use std::alloc::Layout;
use std::alloc::System;
use std::sync::LazyLock;
use std::sync::atomic::AtomicIsize;
use std::sync::atomic::Ordering;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::dtype::DType;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_file::Footer;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::session::RuntimeSession;
use vortex_layout::LayoutRef;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

mod common;

use common::enable_all_registered_array_encodings;

/// Counts the bytes currently allocated by the whole process.
struct CountingAllocator;

static LIVE_BYTES: AtomicIsize = AtomicIsize::new(0);

fn track(bytes: usize, sign: isize) {
    LIVE_BYTES.fetch_add(
        sign * isize::try_from(bytes).unwrap_or(isize::MAX),
        Ordering::Relaxed,
    );
}

// SAFETY: every call forwards to `System` with the caller's arguments.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            track(layout.size(), 1);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            track(layout.size(), 1);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        track(layout.size(), -1);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            track(layout.size(), -1);
            track(new_size, 1);
        }
        new_ptr
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    vortex_file::register_default_encodings(&session);
    enable_all_registered_array_encodings(&session);
    session
});

/// A struct of `groups` structs of `subgroups` structs of `leaves` mixed-type columns.
fn wide_array(groups: usize, subgroups: usize, leaves: usize) -> VortexResult<ArrayRef> {
    let columns = (0..groups)
        .map(|g| {
            let subs = (0..subgroups)
                .map(|s| {
                    let fields = (0..leaves)
                        .map(|l| {
                            let leaf = match l % 4 {
                                0 => buffer![1i64, 2, 3, 4].into_array(),
                                1 => buffer![1.5f64, 2.5, 3.5, 4.5].into_array(),
                                2 => VarBinArray::from(vec!["alpha", "beta", "gamma", "delta"])
                                    .into_array(),
                                _ => BoolArray::from_iter([true, false, true, true]).into_array(),
                            };
                            (format!("leaf_column_{l:03}"), leaf)
                        })
                        .collect::<Vec<_>>();
                    Ok((
                        format!("subgroup_{s:02}"),
                        StructArray::from_fields(&fields)?.into_array(),
                    ))
                })
                .collect::<VortexResult<Vec<_>>>()?;
            Ok((
                format!("group_{g:03}"),
                StructArray::from_fields(&subs)?.into_array(),
            ))
        })
        .collect::<VortexResult<Vec<_>>>()?;
    Ok(StructArray::from_fields(&columns)?.into_array())
}

async fn write(array: ArrayRef, file_statistics: bool) -> VortexResult<ByteBuffer> {
    let mut options = SESSION.write_options();
    if !file_statistics {
        options = options.with_file_statistics(Vec::new());
    }
    let mut buf = ByteBufferMut::empty();
    options.write(&mut buf, array.to_array_stream()).await?;
    Ok(ByteBuffer::from(buf))
}

fn open_footer(bytes: &ByteBuffer) -> VortexResult<Footer> {
    Ok(SESSION
        .open_options()
        .open_buffer(bytes.clone())?
        .footer()
        .clone())
}

/// Materializes every field of the dtype, as converting it to another schema does.
fn walk_dtype(dtype: &DType) -> usize {
    1 + dtype.as_struct_fields_opt().map_or(0, |fields| {
        fields.fields().map(|field| walk_dtype(&field)).sum()
    })
}

/// Materializes every node of the layout tree, as scanning every column does.
fn walk_layout(layout: &LayoutRef) -> VortexResult<usize> {
    let mut nodes = 1;
    for child in layout.children()? {
        nodes += walk_layout(&child)?;
    }
    Ok(nodes)
}

fn live_bytes() -> isize {
    LIVE_BYTES.load(Ordering::Relaxed)
}

struct Measurement {
    estimate: isize,
    /// Bytes the dtype walk allocated after the open, which are zero if the open materialized it.
    dtype_walk: isize,
    dtype_nodes: usize,
    resident: isize,
}

/// Measures the footer's approximate size against the bytes it holds once fully materialized.
fn measure(bytes: &ByteBuffer) -> VortexResult<Measurement> {
    // Warm up registries and interned ids so the measurement below counts only the footer.
    let warm = open_footer(bytes)?;
    walk_dtype(warm.dtype());
    walk_layout(warm.layout())?;
    drop(warm);

    let start = live_bytes();
    let footer = open_footer(bytes)?;
    let opened = live_bytes() - start;
    let dtype_nodes = walk_dtype(footer.dtype());
    let with_dtype = live_bytes() - start;
    let layout_nodes = walk_layout(footer.layout())?;
    let resident = live_bytes() - start;

    let estimate = footer
        .approx_byte_size()
        .and_then(|size| isize::try_from(size).ok())
        .unwrap_or(0);
    println!(
        "estimate {estimate}, resident after open {opened}, after dtype walk {with_dtype}, after \
         layout walk {resident}; {dtype_nodes} dtype nodes, {layout_nodes} layout nodes, {} stats \
         entries, {} aggregates, {} segments",
        footer.statistics().map_or(0, |stats| stats.paths().len()),
        footer.statistics().map_or(0, |stats| stats
            .aggregates()
            .iter()
            .map(|aggregates| aggregates.iter().count())
            .sum::<usize>()),
        footer.segment_map().len(),
    );
    Ok(Measurement {
        estimate,
        dtype_walk: with_dtype - opened,
        dtype_nodes,
        resident,
    })
}

// One test, so no other test allocates while the footer is measured.
#[tokio::test]
async fn footer_size_tracks_resident_memory() -> VortexResult<()> {
    let mut sizes = Vec::new();
    for file_statistics in [true, false] {
        // 2,400 leaves, a width at which footer caches were seen to grow far past their limit.
        let bytes = write(wide_array(40, 6, 10)?, file_statistics).await?;
        sizes.push((file_statistics, measure(&bytes)?));
    }
    for (file_statistics, m) in sizes {
        let (estimate, resident) = (m.estimate, m.resident);
        assert!(
            estimate >= resident / 2 && estimate <= resident * 2,
            "approximate size {estimate} should be within 2x of the {resident} bytes the footer \
             holds (file statistics: {file_statistics})"
        );
        // Reading file statistics resolves the dtype, but an open without them must leave it lazy.
        if !file_statistics {
            let floor = isize::try_from(m.dtype_nodes * 64)?;
            assert!(
                m.dtype_walk >= floor,
                "walking the dtype after the open allocated only {} bytes, so the open \
                 materialized it",
                m.dtype_walk
            );
        }
    }
    Ok(())
}
