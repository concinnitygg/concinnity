//! The block the draw cull picks each record's shader bucket from: where every
//! bucket's commands live, how a record with a variant route chooses among its
//! Shader's variants, and the view the choice is made for.
//!
//! A variant drops terms from its Shader's fullest program. The cull picks it
//! for a draw only where those terms are exactly zero over the draw's whole
//! box, so a draw changes program only where both shade alike, and nothing pops
//! when it does.

use concinnity_core::math::{SaturatingFrom, sqrt};

use super::draw::DrawArgsFlags;
use super::{GpuDrawArgs, MAX_SHADER_BUCKETS};
use crate::render::uniforms::DaylightBlock;

/// Upper bound on the variant routes a world's records pick their buckets by:
/// one per Shader with grass variants, each taking at least two buckets.
pub const MAX_VARIANT_ROUTES: usize = MAX_SHADER_BUCKETS / 2;

/// Feature bit: the draw reaches where its grass hands over to the canopy.
pub const VARIANT_FAR: u32 = 1;
/// Feature bit: part of the draw may be lit from a planet's own sky.
pub const VARIANT_DAYLIGHT: u32 = 2;
/// How many feature combinations a route maps to buckets.
pub const VARIANT_FEATURE_COMBINATIONS: usize = 4;

// Every route fits the flag field one past "none", and a bucket one byte of a
// route.
const _: () = assert!(MAX_VARIANT_ROUTES < DrawArgsFlags::ROUTE_MASK as usize);
const _: () = assert!(MAX_SHADER_BUCKETS <= 0x100);

/// A cosine margin the daylight test keeps above the cone its variant needs,
/// so float rounding never picks the variant without the sky for a pixel the
/// sky would reach.
pub const DAYLIGHT_COS_MARGIN: f32 = 1e-6;

/// Where one shader bucket's indirect commands live: the build-time records it
/// can take, then the runtime reserve's, back to back from `offset`. A record
/// outside both spans never draws under the bucket.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GpuBucketRegion {
    /// First record of the build-time span.
    pub first: u32,
    /// Records in the build-time span.
    pub count: u32,
    /// First record of the runtime span.
    pub runtime_first: u32,
    /// Records in the runtime span.
    pub runtime_count: u32,
    /// The region's first command in the indirect buffer.
    pub offset: u32,
    /// Keeps the next region on a 16-byte row.
    pub _pad: [u32; 3],
}

impl GpuBucketRegion {
    /// Commands the region holds.
    pub fn len(&self) -> u32 {
        self.count + self.runtime_count
    }

    /// Whether the region holds no command.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The command `record` writes in this region, if it falls in a span.
    /// Mirrors `bucket_command` in cull.hlsl.
    pub fn command(&self, record: u32) -> Option<u32> {
        let k = record.wrapping_sub(self.first);
        if k < self.count {
            return Some(self.offset + k);
        }
        let k = record.wrapping_sub(self.runtime_first);
        (k < self.runtime_count).then(|| self.offset + self.count + k)
    }
}

/// How the records under one Shader's grass pick their bucket: the bucket for
/// each feature combination, and which features the route tests.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GpuVariantRoute {
    /// The bucket per feature combination, a byte each, combination 0 lowest.
    pub buckets: u32,
    /// The features ([`VARIANT_FAR`], [`VARIANT_DAYLIGHT`]) the route tests.
    pub features: u32,
    /// Keeps the next route on a 16-byte row.
    pub _pad: [u32; 2],
}

impl GpuVariantRoute {
    /// The route to `buckets`, indexed by feature combination, testing
    /// `features`.
    pub fn new(buckets: [u32; VARIANT_FEATURE_COMBINATIONS], features: u32) -> Self {
        let packed = buckets
            .iter()
            .enumerate()
            .fold(0, |acc, (i, &b)| acc | ((b.min(0xFF)) << (8 * i)));
        Self {
            buckets: packed,
            features,
            _pad: [0; 2],
        }
    }

    /// The bucket for feature combination `features`.
    pub fn bucket(&self, features: u32) -> u32 {
        (self.buckets >> (8 * (features & 3))) & 0xFF
    }
}

/// Matches `CullVariants` in cull.hlsl: the frame's view for the variant tests,
/// the routes and the bucket regions. One per cull view.
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GpuCullVariants {
    /// x: distance from the eye, in meters, where the grass hands over to the
    /// canopy; 0 sends every draw growing grass to its canopy variant.
    pub far: [f32; 4],
    /// [`DaylightBlock::center`] of the view.
    pub day_center: [f32; 4],
    /// [`DaylightBlock::camera_up`] of the view.
    pub day_camera_up: [f32; 4],
    /// [`DaylightBlock::reach`] of the view.
    pub day_reach: [f32; 4],
    /// Indexed by route.
    pub routes: [GpuVariantRoute; MAX_VARIANT_ROUTES],
    /// Indexed by bucket.
    pub regions: [GpuBucketRegion; MAX_SHADER_BUCKETS],
}

impl Default for GpuCullVariants {
    fn default() -> Self {
        Self {
            far: [0.0; 4],
            day_center: [0.0; 4],
            day_camera_up: [0.0; 4],
            day_reach: [0.0; 4],
            routes: [GpuVariantRoute::default(); MAX_VARIANT_ROUTES],
            regions: [GpuBucketRegion::default(); MAX_SHADER_BUCKETS],
        }
    }
}

impl GpuCullVariants {
    /// Set the view the variants are picked for: the eye's grass hand-over
    /// distance and its planet sky.
    pub fn set_view(&mut self, far_start: f32, daylight: &DaylightBlock) {
        self.far = [far_start, 0.0, 0.0, 0.0];
        self.day_center = daylight.center;
        self.day_camera_up = daylight.camera_up;
        self.day_reach = daylight.reach;
    }

    /// Give the default the first `records` records, and every bucket of the
    /// first `buckets` past it the runtime span `runtime`, each from past its
    /// build-time records; then pack the regions back to back again. For a
    /// record list that grows with the draws streamed in.
    pub fn cover_runtime(
        &mut self,
        buckets: usize,
        records: usize,
        runtime: core::ops::Range<usize>,
    ) {
        let mut offset = 0u32;
        for (b, r) in self.regions.iter_mut().enumerate().take(buckets) {
            if b == 0 {
                r.first = 0;
                r.count = u32::saturating_from(records);
            } else {
                let start = runtime.start.max((r.first + r.count) as usize);
                r.runtime_first = u32::saturating_from(start);
                r.runtime_count = u32::saturating_from(runtime.end.saturating_sub(start));
            }
            r.offset = offset;
            offset = offset.saturating_add(r.len());
        }
    }

    /// The buckets of the first `buckets`, a bit each, a record can draw
    /// under in a view lit by `daylight`: every bucket no route picks, and of
    /// a route's, those its features can reach. Aloft every routed record
    /// takes the planet's sky and without one none does, so the others draw
    /// nothing and need not execute.
    pub fn live_buckets(&self, buckets: usize, daylight: &DaylightBlock) -> u64 {
        let days: &[u32] = if !daylight.is_on() {
            &[0]
        } else if daylight.camera_up[3] > 0.0 {
            &[VARIANT_DAYLIGHT]
        } else {
            &[0, VARIANT_DAYLIGHT]
        };
        let mut routed = 0u64;
        let mut live = 0u64;
        for route in self.routes.iter().filter(|r| r.features != 0) {
            for features in 0..u32::saturating_from(VARIANT_FEATURE_COMBINATIONS) {
                routed |= 1u64.checked_shl(route.bucket(features)).unwrap_or(0);
            }
            for far in [0, VARIANT_FAR] {
                for &day in days {
                    live |= 1u64
                        .checked_shl(route.bucket((far | day) & route.features))
                        .unwrap_or(0);
                }
            }
        }
        let all = 1u64
            .checked_shl(u32::saturating_from(buckets))
            .map_or(u64::MAX, |b| b - 1);
        (!routed | live) & all
    }

    /// The features a draw with `args`'s flags and bounds `bb_min..bb_max`
    /// has seen from `cam_pos`, among those `route` tests; a draw the cull
    /// does not frustum-test counts as reaching everywhere. Mirrors
    /// `variant_features` in cull.hlsl.
    pub fn features(
        &self,
        route: &GpuVariantRoute,
        flags: u32,
        bb: ([f32; 3], [f32; 3]),
        cam_pos: [f32; 3],
    ) -> u32 {
        // Unbounded as the cull reads it: a record it does not frustum-test,
        // which always draws.
        let bounded = flags & DrawArgsFlags::CULLABLE != 0;
        let mut features = 0;
        if route.features & VARIANT_FAR != 0
            && flags & DrawArgsFlags::FAR_BAND != 0
            && (!bounded || reaches(bb, cam_pos, self.far[0]))
        {
            features |= VARIANT_FAR;
        }
        if route.features & VARIANT_DAYLIGHT != 0
            && self.day_center[3] > 0.5
            && (!bounded || self.daylight_reaches(bb))
        {
            features |= VARIANT_DAYLIGHT;
        }
        features
    }

    /// The bucket a record with `args` draws under, bounded by `bb` and seen
    /// from `cam_pos`. Mirrors `record_bucket` in cull.hlsl.
    pub fn bucket(&self, args: &GpuDrawArgs, bb: ([f32; 3], [f32; 3]), cam_pos: [f32; 3]) -> u32 {
        let route = (args.flags >> DrawArgsFlags::ROUTE_SHIFT) & DrawArgsFlags::ROUTE_MASK;
        match route
            .checked_sub(1)
            .and_then(|r| self.routes.get(r as usize))
        {
            Some(r) => r.bucket(self.features(r, args.flags, bb, cam_pos)),
            None => (args.flags >> DrawArgsFlags::BUCKET_SHIFT) & 0xFF,
        }
    }

    // Whether a planet's own sky may reach any point of `bb`: the camera is
    // high enough for it to reach everywhere, or a corner lies far enough
    // round the planet. The corners suffice: the points whose up lies within
    // the cone the sky does not reach form a convex cone, so a box whose
    // corners are all inside it is wholly inside.
    fn daylight_reaches(&self, bb: ([f32; 3], [f32; 3])) -> bool {
        if self.day_camera_up[3] > 0.0 {
            return true;
        }
        let center = [self.day_center[0], self.day_center[1], self.day_center[2]];
        let up = [
            self.day_camera_up[0],
            self.day_camera_up[1],
            self.day_camera_up[2],
        ];
        let near = self.day_reach[0] + DAYLIGHT_COS_MARGIN;
        (0..8).any(|i| {
            let corner = [
                if i & 1 != 0 { bb.1[0] } else { bb.0[0] },
                if i & 2 != 0 { bb.1[1] } else { bb.0[1] },
                if i & 4 != 0 { bb.1[2] } else { bb.0[2] },
            ];
            let d = [
                corner[0] - center[0],
                corner[1] - center[1],
                corner[2] - center[2],
            ];
            let len = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            len <= 0.0 || (d[0] * up[0] + d[1] * up[1] + d[2] * up[2]) / len < near
        })
    }
}

// Whether the farthest corner of `bb` lies past `start` from `cam_pos`.
fn reaches(bb: ([f32; 3], [f32; 3]), cam_pos: [f32; 3], start: f32) -> bool {
    let farthest_sq: f32 = (0..3)
        .map(|i| {
            let d = (cam_pos[i] - bb.0[i])
                .abs()
                .max((bb.1[i] - cam_pos[i]).abs());
            d * d
        })
        .sum();
    farthest_sq > start * start
}

/// The command regions for `spans`, one per bucket: each its build-time span
/// then its runtime span, back to back in bucket order.
pub fn bucket_regions(spans: &[BucketSpans]) -> [GpuBucketRegion; MAX_SHADER_BUCKETS] {
    let mut regions = [GpuBucketRegion::default(); MAX_SHADER_BUCKETS];
    let mut offset = 0u32;
    for (region, s) in regions.iter_mut().zip(spans) {
        *region = GpuBucketRegion {
            first: u32::saturating_from(s.records.start),
            count: u32::saturating_from(s.records.len()),
            runtime_first: u32::saturating_from(s.runtime.start),
            runtime_count: u32::saturating_from(s.runtime.len()),
            offset,
            _pad: [0; 3],
        };
        offset = offset.saturating_add(region.len());
    }
    regions
}

/// The records one bucket can take: a build-time span and a runtime one.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BucketSpans {
    /// The build-time records.
    pub records: core::ops::Range<usize>,
    /// The runtime records.
    pub runtime: core::ops::Range<usize>,
}

/// Commands every region of `regions` holds between them, the first `buckets`.
pub fn region_commands(regions: &[GpuBucketRegion], buckets: usize) -> usize {
    regions.iter().take(buckets).map(|r| r.len() as usize).sum()
}

#[cfg(test)]
mod tests;
