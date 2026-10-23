//! How the draw cull packs the draws it keeps into one list per stream, so an
//! indirect draw executes only what its stream holds.
//!
//! A stream is a shader bucket's draws, or the skinned tail, which bucket 0
//! draws over its own vertex and index buffers. Three kernels build the lists
//! without atomics, so each keeps ascending record order: the cull counts each
//! 64-record group's draws per stream, a scan turns the counts into each
//! group's first place in its stream and writes the stream's total, and a
//! scatter writes every kept record's command at its group's place plus its
//! rank among the group's records of that stream. [`compact`] is the spec the
//! kernels are tested against.

use alloc::vec::Vec;

use concinnity_core::math::SaturatingFrom;

use super::{GpuBucketRegion, MAX_SHADER_BUCKETS};
use crate::gfx::cull_status::CullStatus;

/// Records one cull threadgroup covers, in the cull and in the scatter.
pub const CULL_GROUP_RECORDS: u32 = 64;
/// Streams a cull packs: one per bucket, then the skinned tail.
pub const DRAW_STREAMS: usize = MAX_SHADER_BUCKETS + 1;
/// The skinned tail's stream.
pub const SKINNED_STREAM: u32 = concinnity_core::math::saturating_u32(MAX_SHADER_BUCKETS);
/// Alignment of the count slots behind an indirect buffer's commands, the
/// widest storage-buffer offset alignment Vulkan allows.
pub const COUNT_SLOT_ALIGN: u64 = 256;

const _: () = assert!(DRAW_STREAMS <= 2 * CULL_GROUP_RECORDS as usize);

/// Matches `MTLIndirectCommandBufferExecutionRange`: the commands of a stream
/// in its bucket's indirect command buffer, which the scan writes and Metal
/// executes.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GpuExecutionRange {
    /// The stream's first command.
    pub location: u32,
    /// Commands the stream holds.
    pub length: u32,
}

/// Matches `CompactParams` in `cull_compact.hlsl`: what the scan and the
/// scatter pack, the same on every backend.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GpuCompactParams {
    /// Records the cull covered.
    pub object_count: u32,
    /// Shader buckets the cull routed between.
    pub bucket_count: u32,
    /// The first skinned record.
    pub skinned_base: u32,
    /// The outcome the phase keeps: [`CullStatus::DRAWN`] or
    /// [`CullStatus::REDRAW`].
    pub drawn: u32,
}

/// The records a cull of `live` records covers when its buffers hold
/// `capacity`, so a region's cull never writes past its own counts, status or
/// lists.
pub fn culled_records(live: usize, capacity: usize) -> u32 {
    u32::saturating_from(live.min(capacity))
}

/// Cull threadgroups over `records`.
pub fn cull_groups(records: u32) -> u32 {
    records.div_ceil(CULL_GROUP_RECORDS)
}

/// The words of group-count scratch a cull of `records` needs: one per stream
/// per group.
pub fn group_count_words(records: u32) -> usize {
    cull_groups(records) as usize * DRAW_STREAMS
}

/// Where group `group`'s count for `stream` lives in the scratch, stream by
/// stream so the scan reads its stream's counts back to back.
pub fn group_count_index(stream: u32, group: u32, groups: u32) -> usize {
    stream as usize * groups as usize + group as usize
}

/// The byte offset of the count slots behind indirect commands taking
/// `command_bytes`.
pub fn count_slot_offset(command_bytes: u64) -> u64 {
    command_bytes.div_ceil(COUNT_SLOT_ALIGN) * COUNT_SLOT_ALIGN
}

/// Bytes the count slots take: a `u32` per stream.
pub const COUNT_SLOT_BYTES: u64 = (DRAW_STREAMS * 4) as u64;

/// The records and the routing one cull packs.
#[derive(Copy, Clone, Debug)]
pub struct StreamLayout<'a> {
    /// The bucket regions the cull's records route into.
    pub regions: &'a [GpuBucketRegion; MAX_SHADER_BUCKETS],
    /// Buckets the dispatch routes between; 1 for a single stream.
    pub bucket_count: u32,
    /// The first skinned record.
    pub skinned_base: u32,
    /// Records culled.
    pub records: u32,
}

/// Where a stream's commands go: from `region` in a shared indirect buffer,
/// `first` places into its bucket's region, at most `capacity` of them.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct StreamSlots {
    /// The bucket region's first command in a shared indirect buffer.
    pub region: u32,
    /// The stream's first command within its bucket's region.
    pub first: u32,
    /// Commands the stream can hold.
    pub capacity: u32,
}

impl StreamLayout<'_> {
    /// The stream record `record`, routed to `bucket`, draws in, or `None`
    /// when the record falls outside the bucket's region. Mirrors
    /// `record_stream` in cull.hlsl.
    pub fn stream(&self, record: u32, bucket: u32) -> Option<u32> {
        if self.bucket_count <= 1 || bucket == 0 {
            return Some(if record >= self.skinned_base {
                SKINNED_STREAM
            } else {
                0
            });
        }
        let region = self.regions.get(bucket as usize)?;
        region.command(record).map(|_| bucket)
    }

    /// Where `stream`'s commands go. Mirrors `stream_slots` in
    /// `cull_compact.hlsl`.
    pub fn slots(&self, stream: u32) -> StreamSlots {
        let skinned_base = self.skinned_base.min(self.records);
        match stream {
            0 => StreamSlots {
                region: 0,
                first: 0,
                capacity: skinned_base,
            },
            SKINNED_STREAM => StreamSlots {
                region: 0,
                first: skinned_base,
                capacity: self.records - skinned_base,
            },
            b if b < self.bucket_count => {
                self.regions
                    .get(b as usize)
                    .map_or_else(StreamSlots::default, |r| StreamSlots {
                        region: r.offset,
                        first: 0,
                        capacity: r.len(),
                    })
            }
            _ => StreamSlots::default(),
        }
    }

    /// The streams the scan walks: each bucket's, then the skinned tail's.
    pub fn streams(&self) -> impl Iterator<Item = u32> {
        (0..self.bucket_count.min(SKINNED_STREAM)).chain(core::iter::once(SKINNED_STREAM))
    }
}

/// Each stream's records, in the order its commands execute, for a cull whose
/// status words are `status` drawing those with outcome `drawn`: the lists the
/// three kernels build, indexed by stream.
pub fn compact(status: &[u32], drawn: u32, layout: &StreamLayout<'_>) -> Vec<Vec<u32>> {
    let mut lists = alloc::vec![Vec::new(); DRAW_STREAMS];
    for (record, &word) in status.iter().enumerate().take(layout.records as usize) {
        let record = u32::saturating_from(record);
        if CullStatus::outcome(word) != drawn {
            continue;
        }
        let Some(stream) = layout.stream(record, CullStatus::bucket(word)) else {
            continue;
        };
        if let Some(list) = lists.get_mut(stream as usize)
            && u32::saturating_from(list.len()) < layout.slots(stream).capacity
        {
            list.push(record);
        }
    }
    lists
}

/// The group-count scratch the cull writes for `status`: each group's records
/// with outcome `drawn`, per stream, laid out by [`group_count_index`].
pub fn group_counts(status: &[u32], drawn: u32, layout: &StreamLayout<'_>) -> Vec<u32> {
    let groups = cull_groups(layout.records);
    let mut counts = alloc::vec![0u32; group_count_words(layout.records)];
    for (record, &word) in status.iter().enumerate().take(layout.records as usize) {
        let record = u32::saturating_from(record);
        if CullStatus::outcome(word) != drawn {
            continue;
        }
        if let Some(stream) = layout.stream(record, CullStatus::bucket(word))
            && let Some(c) = counts.get_mut(group_count_index(
                stream,
                record / CULL_GROUP_RECORDS,
                groups,
            ))
        {
            *c += 1;
        }
    }
    counts
}

/// What the scan leaves: `counts` turned in place into each group's first
/// place in its stream, and every stream's total, held to its capacity.
pub fn scan_group_counts(counts: &mut [u32], layout: &StreamLayout<'_>) -> [u32; DRAW_STREAMS] {
    let groups = cull_groups(layout.records);
    let mut totals = [0u32; DRAW_STREAMS];
    for stream in layout.streams() {
        let mut running = 0u32;
        for group in 0..groups {
            if let Some(c) = counts.get_mut(group_count_index(stream, group, groups)) {
                let n = *c;
                *c = running;
                running += n;
            }
        }
        if let Some(t) = totals.get_mut(stream as usize) {
            *t = running.min(layout.slots(stream).capacity);
        }
    }
    totals
}

#[cfg(test)]
mod tests;
