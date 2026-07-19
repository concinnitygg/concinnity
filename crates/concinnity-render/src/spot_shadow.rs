// src/spot_shadow.rs
//
// Slice assignment and light-space projections for the spot shadow map array.
//
// Local lights are static, so both the slice each shadowed spot owns and the
// matrix it renders with are decided once per scene and never recomputed. Only
// the depth contents need refreshing, and only when a caster moves -- that
// schedule is `SpotShadowScheduler` below, which mirrors the prime-then-
// round-robin policy the CSM cascades use.
//
// A spot's projection is a perspective frustum whose vertical FOV is the full
// cone angle (2x the outer half-angle), so the cone inscribes the shadow slice's
// square footprint. Right-handed with [0, 1] depth, matching `csm.rs`, so the
// same matrices are valid on all three backends.

use crate::assets::{SpotLight, SpotLightGeometry};
use crate::mat::{add3, look_at, mat4_mul, perspective_rh, scale3, up_for};
use crate::render_types::{MAX_SHADOWED_SPOTS, SpotShadowData};

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
pub fn assign_spot_shadow_slices(spot_lights: &[SpotLight]) -> Vec<i32> {
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
    if wanted > MAX_SHADOWED_SPOTS {
        tracing::warn!(
            "GraphicsSystem: {} spot lights request shadows; only {} slices exist -- the rest light without casting",
            wanted,
            MAX_SHADOWED_SPOTS
        );
    }
    slices
}

// The `SpotShadowData` for each assigned slice, ordered by slice index. Pair
// with `assign_spot_shadow_slices` over the same slice: entry `slices[i]` of the
// result describes spot `i`.
pub fn build_spot_shadow_data(spot_lights: &[SpotLight], slices: &[i32]) -> Vec<SpotShadowData> {
    let mut out = vec![SpotShadowData::ZERO; count_shadowed(slices)];
    for (light, &slice) in spot_lights.iter().zip(slices) {
        if slice >= 0 {
            out[slice as usize] = spot_shadow_data(light);
        }
    }
    out
}

// How many slices `assign_spot_shadow_slices` handed out.
pub fn count_shadowed(slices: &[i32]) -> usize {
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
        add3(light.position, scale3(dir, far)),
        up_for(dir),
    );
    let fov = (2.0 * light.outer_angle).to_radians();
    let proj = perspective_rh(fov, 1.0, SHADOW_NEAR, far);
    SpotShadowData {
        light_vp: mat4_mul(proj, view),
        depth_bias: DEPTH_BIAS,
        normal_bias: NORMAL_BIAS,
        _pad: [0.0; 2],
    }
}

// Prime-then-round-robin refresh schedule over the assigned slices, mirroring
// `ShadowCascadeScheduler`. Every slice renders once before it can be sampled;
// after that `Hybrid` refreshes one slice per frame so N shadowed spots cost one
// extra depth render per frame rather than N.
#[derive(Debug, Default)]
pub struct SpotShadowScheduler {
    clock: u32,
    primed: u32,
}

impl SpotShadowScheduler {
    // Bit `i` set means slice `i` re-renders this frame. Advances the clock.
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
        // Its far plane is b's range: a point just inside it stays within depth 1.
        let clip = transform(data[0].light_vp, [0.0, 4.0 - 32.0, 0.0]);
        assert!(clip[3] > 0.0);
        assert!((clip[2] / clip[3]) < 1.0);
    }

    // The cone axis maps to the centre of the slice, and the outer cone edge
    // lands on the NDC boundary -- i.e. the frustum exactly contains the cone.
    #[test]
    fn the_cone_inscribes_the_shadow_frustum() {
        let mut l = spot(true);
        l.position = [0.0, 10.0, 0.0];
        l.direction = [0.0, -1.0, 0.0];
        l.outer_angle = 30.0;
        l.range = 20.0;
        let d = spot_shadow_data(&l);

        // Straight down the axis: dead centre.
        let centre = transform(d.light_vp, [0.0, 0.0, 0.0]);
        assert!((centre[0] / centre[3]).abs() < 1e-4);
        assert!((centre[1] / centre[3]).abs() < 1e-4);

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
