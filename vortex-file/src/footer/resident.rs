// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Estimates of the heap memory a parsed footer keeps alive, which weigh it in footer caches.
//!
//! The layout tree is a view over a retained flatbuffer whose nodes materialize as they are visited
//! and then stay cached on the footer. The estimate counts every node as materialized, as it is
//! once a scan has read every column, but walks the flatbuffer so that estimating materializes
//! nothing. The dtype's share comes from [`DType::approx_resident_bytes`]. The per-node constants
//! were measured with a counting allocator, and `tests/footer_resident_size.rs` checks them
//! against one.
//!
//! [`DType::approx_resident_bytes`]: vortex_array::dtype::DType::approx_resident_bytes

use vortex_array::scalar::Scalar;
use vortex_layout::flatbuffers::layout as fbl;
use vortex_layout::layouts::file_stats::AggregateStat;

use crate::footer::FileStatistics;
use crate::footer::SegmentSpec;

/// A materialized layout node with its child adapter, segment ids and encoding-specific data.
const LAYOUT_NODE_BYTES: usize = 680;
/// A statistics entry without its aggregates: its dtype, its path twice and its path index slot.
const STATS_ENTRY_BYTES: usize = 160;

/// Estimates the bytes a parsed footer holds once every dtype and layout node is materialized.
///
/// `dtype_bytes` is the dtype's share, from [`DType::approx_resident_bytes`].
///
/// [`DType::approx_resident_bytes`]: vortex_array::dtype::DType::approx_resident_bytes
pub(super) fn footer_resident_bytes(
    dtype_bytes: usize,
    layout_flatbuffer: &[u8],
    segments: usize,
    statistics: Option<&FileStatistics>,
    metadata_bytes: usize,
) -> usize {
    // SAFETY: the caller has parsed `layout_flatbuffer` into a layout, which verified it.
    let layout = unsafe { flatbuffers::root_unchecked::<fbl::Layout<'_>>(layout_flatbuffer) };
    sum([
        dtype_bytes,
        layout_flatbuffer.len(),
        layout_nodes(layout).saturating_mul(LAYOUT_NODE_BYTES),
        segments.saturating_mul(size_of::<SegmentSpec>()),
        statistics.map_or(0, statistics_bytes),
        metadata_bytes,
    ])
}

fn layout_nodes(layout: fbl::Layout<'_>) -> usize {
    1usize.saturating_add(
        layout
            .children()
            .map_or(0, |children| sum(children.iter().map(layout_nodes))),
    )
}

fn statistics_bytes(statistics: &FileStatistics) -> usize {
    sum(statistics.aggregates().iter().map(|aggregates| {
        STATS_ENTRY_BYTES.saturating_add(sum(aggregates.iter().map(aggregate_bytes)))
    }))
}

fn aggregate_bytes(aggregate: &AggregateStat) -> usize {
    sum([
        size_of::<AggregateStat>(),
        aggregate.partial().map_or(0, Scalar::approx_nbytes),
        aggregate
            .value()
            .as_ref()
            .into_inner()
            .map_or(0, Scalar::approx_nbytes),
    ])
}

/// Sums without overflowing, since a malformed file can repeat references to shared nodes.
fn sum(values: impl IntoIterator<Item = usize>) -> usize {
    values.into_iter().fold(0, usize::saturating_add)
}
