//! Slice assignment and light-space projections for the spot shadow map array.
//!
//! Local lights are static, so both the slice each shadowed spot owns and the
//! matrix it renders with are decided once per scene and never recomputed. Only
//! the depth contents refresh, on the schedule `SpotShadowScheduler` below
//! keeps, which mirrors the prime-then-round-robin policy the CSM cascades use.
//! Each refreshed slice draws only the casters its GPU cull keeps inside
//! `slice_frustum`.
//!
//! A spot's projection is a perspective frustum whose vertical FOV is the full
//! cone angle (2x the outer half-angle), so the cone inscribes the shadow slice's
//! square footprint. Right-handed with the engine's reversed [0, 1] depth
//! (near 1, the spot's range 0), matching `csm.rs`, so the same matrices are
//! valid on all three backends.

use crate::components::SpotLight;
use crate::gfx::frustum::Frustum;
use crate::gfx::projection::{look_at, up_for};
use crate::gfx::render_types::{MAX_SHADOWED_SPOTS, SpotShadowData};
use crate::math::vec3::{add, scale};
use crate::render::depth::shadow_perspective;
use crate::transform::mat4_mul;
use alloc::vec;
use alloc::vec::Vec;

// Near plane for a spot's shadow frustum. Fixed and small: the depth range is
// [SHADOW_NEAR, range], and pulling the near plane in costs precision while
// pushing it out clips casters close to the bulb.
const SHADOW_NEAR: f32 = 0.05;

// Depth compare offsets, in light-clip and world units respectively. Sized to
// clear the acne a 512-ish slice produces at grazing angles without detaching
// contact shadows.
const DEPTH_BIAS: f32 = 0.0015;
const NORMAL_BIAS: f32 = 0.035;

// Shortest range a shadowed spot may declare. A zero or near-zero range would
// collapse the frustum's depth span and make the projection degenerate.
const MIN_SHADOW_RANGE: f32 = 0.1;

// Per-spot slice assignment: `slices[i]` is the shadow map array slice spot `i`
// owns, or -1 when it casts no shadow (either `cast_shadows` is false or the
// slices ran out). The value is what `GpuLight.shadow_index` carries.
pub(crate) fn assign_spot_shadow_slices(spot_lights: &[SpotLight]) -> Vec<i32> {
    let mut next = 0_i32;
    let mut wanted = 0_usize;
    let slices: Vec<i32> = spot_lights
        .iter()
        .map(|l| {
            if !l.cast_shadows {
                return -1;
            }
            wanted += 1;
            if (next as usize) < MAX_SHADOWED_SPOTS {
                let slice = next;
                next += 1;
                slice
            } else {
                -1
            }
        })
        .collect();
    slices
}

// The `SpotShadowData` for each assigned slice, ordered by slice index. Pair
// with `assign_spot_shadow_slices` over the same slice: entry `slices[i]` of the
// result describes spot `i`.
pub(crate) fn build_spot_shadow_data(
    spot_lights: &[SpotLight],
    slices: &[i32],
) -> Vec<SpotShadowData> {
    let mut out = vec![SpotShadowData::ZERO; count_shadowed(slices)];
    for (light, &slice) in spot_lights.iter().zip(slices) {
        if slice >= 0 {
            out[slice as usize] = spot_shadow_data(light);
        }
    }
    out
}

// How many slices `assign_spot_shadow_slices` handed out.
pub(crate) fn count_shadowed(slices: &[i32]) -> usize {
    slices.iter().filter(|s| **s >= 0).count()
}

// One spot's light-space projection. The FOV is the full cone (2x the outer
// half-angle) so the lit cone fits inside the slice's square footprint; the
// validator caps the half-angle below 90 degrees, keeping the FOV under 180.
fn spot_shadow_data(light: &SpotLight) -> SpotShadowData {
    let dir = light.unit_direction();
    let far = light.range.max(MIN_SHADOW_RANGE);
    let view = look_at(
        light.position,
        add(light.position, scale(dir, far)),
        up_for(dir),
    );
    let fov = (2.0 * light.outer_angle).to_radians();
    let proj = shadow_perspective(fov, 1.0, SHADOW_NEAR, far);
    SpotShadowData {
        light_vp: mat4_mul(proj, view),
        depth_bias: DEPTH_BIAS,
        normal_bias: NORMAL_BIAS,
        _pad: [0.0; 2],
    }
}

/// Prime-then-round-robin refresh schedule over the assigned slices, mirroring
/// `ShadowCascadeScheduler`. Every slice renders once before it can be sampled;
/// after that `Hybrid` refreshes one slice per frame so N shadowed spots cost one
/// extra depth render per frame rather than N.
#[derive(Debug, Default)]
pub struct SpotShadowScheduler {
    clock: u32,
    primed: u32,
}

impl SpotShadowScheduler {
    /// Bit `i` set means slice `i` re-renders this frame. Advances the clock.
    pub fn next_mask(&mut self, every_frame: bool, shadowed: usize) -> u32 {
        let (mask, primed) = select_slice_mask(every_frame, self.clock, self.primed, shadowed);
        self.clock = self.clock.wrapping_add(1);
        self.primed = primed;
        mask
    }
}

// Pure selection step, split out so the policy is testable without renderer
// state. Returns `(render_mask, new_primed_mask)`. Any slice not yet primed is
// force-rendered so it never gets sampled before it holds valid depth.
fn select_slice_mask(every_frame: bool, clock: u32, primed: u32, shadowed: usize) -> (u32, u32) {
    let shadowed = shadowed.min(MAX_SHADOWED_SPOTS);
    if shadowed == 0 {
        return (0, primed);
    }
    let all = if shadowed >= 32 {
        u32::MAX
    } else {
        (1_u32 << shadowed) - 1
    };
    let scheduled = if every_frame {
        all
    } else {
        1_u32 << (clock as usize % shadowed)
    };
    let unprimed = all & !primed;
    let mask = (scheduled | unprimed) & all;
    (mask, primed | mask)
}

/// The slices a spot shadow pass re-renders this frame: the set bits of
/// `mask` below `count`, or every slice when no mask was set.
pub fn refreshed_slices(mask: u32, count: u32) -> impl Iterator<Item = u32> {
    let count = count.min(MAX_SHADOWED_SPOTS as u32);
    let mask = if mask == 0 { u32::MAX } else { mask };
    (0..count).filter(move |s| mask & (1 << s) != 0)
}

/// The world-space frustum a slice renders, which is also the volume its GPU
/// cull keeps casters from.
pub fn slice_frustum(data: &SpotShadowData) -> Frustum {
    Frustum::from_shadow(data.light_vp)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spot(cast: bool) -> SpotLight {
        SpotLight {
            cast_shadows: cast,
            ..SpotLight::default()
        }
    }

    fn transform(m: [[f32; 4]; 4], p: [f32; 3]) -> [f32; 4] {
        let mut out = [0.0_f32; 4];
        for row in 0..4 {
            out[row] = m[0][row] * p[0] + m[1][row] * p[1] + m[2][row] * p[2] + m[3][row];
        }
        out
    }

    #[test]
    fn slices_are_handed_out_in_declaration_order() {
        let lights = vec![spot(true), spot(true), spot(true)];
        assert_eq!(assign_spot_shadow_slices(&lights), vec![0, 1, 2]);
    }

    // A non-casting spot takes no slice and does not shift the ones after it.
    #[test]
    fn non_casting_spots_are_skipped_without_consuming_a_slice() {
        let lights = vec![spot(true), spot(false), spot(true)];
        assert_eq!(assign_spot_shadow_slices(&lights), vec![0, -1, 1]);
    }

    #[test]
    fn slices_past_the_cap_get_no_shadow() {
        let lights: Vec<SpotLight> = (0..MAX_SHADOWED_SPOTS + 3).map(|_| spot(true)).collect();
        let slices = assign_spot_shadow_slices(&lights);
        assert_eq!(count_shadowed(&slices), MAX_SHADOWED_SPOTS);
        assert_eq!(
            slices[MAX_SHADOWED_SPOTS - 1],
            MAX_SHADOWED_SPOTS as i32 - 1
        );
        assert!(slices[MAX_SHADOWED_SPOTS..].iter().all(|s| *s == -1));
    }

    #[test]
    fn shadow_data_is_indexed_by_slice_not_by_light() {
        let mut a = spot(false);
        a.range = 5.0;
        let mut b = spot(true);
        b.range = 33.0;
        let lights = vec![a, b];
        let slices = assign_spot_shadow_slices(&lights);
        assert_eq!(slices, vec![-1, 0]);
        let data = build_spot_shadow_data(&lights, &slices);
        // Only the casting light produced an entry, and it sits at slice 0.
        assert_eq!(data.len(), 1);
        // Its far plane is b's range: a point just inside it stays in front of
        // depth 0, one just past it falls behind.
        let clip = transform(data[0].light_vp, [0.0, 4.0 - 32.0, 0.0]);
        assert!(clip[3] > 0.0);
        assert!((clip[2] / clip[3]) > 0.0);
        let past = transform(data[0].light_vp, [0.0, 4.0 - 34.0, 0.0]);
        assert!((past[2] / past[3]) < 0.0);
        // The bulb's near plane holds depth 1, and nearer the bulb is nearer.
        let near = transform(data[0].light_vp, [0.0, 4.0 - SHADOW_NEAR, 0.0]);
        assert!(((near[2] / near[3]) - 1.0).abs() < 1e-5);
        let mid = transform(data[0].light_vp, [0.0, 4.0 - 10.0, 0.0]);
        assert!(mid[2] / mid[3] > clip[2] / clip[3]);
    }

    // The cone axis maps to the center of the slice, and the outer cone edge
    // lands on the NDC boundary -- i.e. the frustum exactly contains the cone.
    #[test]
    fn the_cone_inscribes_the_shadow_frustum() {
        let mut l = spot(true);
        l.position = [0.0, 10.0, 0.0];
        l.direction = [0.0, -1.0, 0.0];
        l.outer_angle = 30.0;
        l.range = 20.0;
        let d = spot_shadow_data(&l);

        // Straight down the axis: dead center.
        let center = transform(d.light_vp, [0.0, 0.0, 0.0]);
        assert!((center[0] / center[3]).abs() < 1e-4);
        assert!((center[1] / center[3]).abs() < 1e-4);

        // 10 units down, offset by tan(30 deg) * 10: exactly the cone edge.
        let edge_x = 30.0_f32.to_radians().tan() * 10.0;
        let edge = transform(d.light_vp, [edge_x, 0.0, 0.0]);
        assert!(((edge[0] / edge[3]).abs() - 1.0).abs() < 1e-3);
    }

    // A straight-down cone is the common case and the one where a naive +Y up
    // vector would collapse the basis into NaNs.
    #[test]
    fn a_straight_down_cone_produces_a_finite_matrix() {
        let mut l = spot(true);
        l.direction = [0.0, -1.0, 0.0];
        let d = spot_shadow_data(&l);
        assert!(d.light_vp.iter().flatten().all(|v| v.is_finite()));
    }

    // A zero range would collapse the depth span; it is floored instead.
    #[test]
    fn a_degenerate_range_still_produces_a_finite_matrix() {
        let mut l = spot(true);
        l.range = 0.0;
        let d = spot_shadow_data(&l);
        assert!(d.light_vp.iter().flatten().all(|v| v.is_finite()));
    }

    #[test]
    fn refreshed_slices_follow_the_mask() {
        let got: Vec<u32> = refreshed_slices(0b1010, 4).collect();
        assert_eq!(got, vec![1, 3]);
    }

    // Bits past the slice count name slices that do not exist.
    #[test]
    fn refreshed_slices_ignore_bits_past_the_count() {
        let got: Vec<u32> = refreshed_slices(0b1111_0001, 3).collect();
        assert_eq!(got, vec![0]);
    }

    // A pass that runs before any mask was set renders every slice rather than
    // leaving one unprimed.
    #[test]
    fn an_unset_mask_refreshes_every_slice() {
        let got: Vec<u32> = refreshed_slices(0, 3).collect();
        assert_eq!(got, vec![0, 1, 2]);
        assert_eq!(refreshed_slices(0, 0).count(), 0);
    }

    #[test]
    fn refreshed_slices_never_pass_the_array_capacity() {
        let n = refreshed_slices(u32::MAX, 64).count();
        assert_eq!(n, MAX_SHADOWED_SPOTS);
    }

    // The cull keeps what lies in the cone and drops what lies behind the bulb
    // or past the range.
    #[test]
    fn a_slice_frustum_keeps_only_what_the_cone_reaches() {
        let mut l = spot(true);
        l.position = [0.0, 10.0, 0.0];
        l.direction = [0.0, -1.0, 0.0];
        l.outer_angle = 30.0;
        l.range = 20.0;
        let f = slice_frustum(&spot_shadow_data(&l));
        let unit = |c: [f32; 3]| {
            (
                [c[0] - 0.5, c[1] - 0.5, c[2] - 0.5],
                [c[0] + 0.5, c[1] + 0.5, c[2] + 0.5],
            )
        };
        let (lo, hi) = unit([0.0, 0.0, 0.0]);
        assert!(f.intersects_aabb(lo, hi), "under the bulb");
        let (lo, hi) = unit([0.0, 15.0, 0.0]);
        assert!(!f.intersects_aabb(lo, hi), "behind the bulb");
        let (lo, hi) = unit([0.0, -15.0, 0.0]);
        assert!(!f.intersects_aabb(lo, hi), "past the range");
        let (lo, hi) = unit([12.0, 0.0, 0.0]);
        assert!(!f.intersects_aabb(lo, hi), "outside the cone");
        // The depth planes are exact: the range ends 20 m below the bulb and
        // the near plane sits SHADOW_NEAR below it.
        let (lo, hi) = unit([0.0, -9.45, 0.0]);
        assert!(f.intersects_aabb(lo, hi), "just inside the range");
        let (lo, hi) = unit([0.0, -10.55, 0.0]);
        assert!(!f.intersects_aabb(lo, hi), "just past the range");
        let gap = 0.5 * SHADOW_NEAR;
        assert!(!f.intersects_aabb([-0.01, 10.0 - gap, -0.01], [0.01, 9.999, 0.01]));
        assert!(f.intersects_aabb(
            [-0.01, 10.0 - 2.0 * SHADOW_NEAR, -0.01],
            [0.01, 9.999, 0.01]
        ));
    }

    #[test]
    fn no_shadowed_spots_renders_nothing() {
        let mut s = SpotShadowScheduler::default();
        assert_eq!(s.next_mask(false, 0), 0);
    }

    // Every slice is primed before the round-robin settles, so a slice is never
    // sampled holding stale depth.
    #[test]
    fn all_slices_prime_on_the_first_frame() {
        let mut s = SpotShadowScheduler::default();
        assert_eq!(s.next_mask(false, 4), 0b1111);
    }

    #[test]
    fn hybrid_settles_into_one_slice_per_frame() {
        let mut s = SpotShadowScheduler::default();
        s.next_mask(false, 4);
        assert_eq!(s.next_mask(false, 4), 0b0010);
        assert_eq!(s.next_mask(false, 4), 0b0100);
        assert_eq!(s.next_mask(false, 4), 0b1000);
        assert_eq!(s.next_mask(false, 4), 0b0001);
    }

    #[test]
    fn every_frame_refreshes_all_slices() {
        let mut s = SpotShadowScheduler::default();
        s.next_mask(true, 3);
        assert_eq!(s.next_mask(true, 3), 0b111);
    }

    // A slice that appears after priming (a larger shadowed count) is primed on
    // the frame it appears rather than waiting for its round-robin turn.
    #[test]
    fn a_newly_appearing_slice_is_primed_immediately() {
        let mut s = SpotShadowScheduler::default();
        s.next_mask(false, 2);
        let mask = s.next_mask(false, 4);
        assert!(mask & 0b1100 == 0b1100, "the two new slices prime at once");
    }

    #[test]
    fn the_mask_never_exceeds_the_shadowed_count() {
        let mut s = SpotShadowScheduler::default();
        for _ in 0..40 {
            assert_eq!(s.next_mask(false, 3) & !0b111, 0);
        }
    }
}
