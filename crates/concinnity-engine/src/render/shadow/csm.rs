//! Cascaded shadow map cascade computation. Produces a `ShadowUniforms` carrying
//! one orthographic light-view-projection matrix per cascade, the view-space
//! far depth for each cascade (the fragment shader uses these to select which
//! cascade slice to sample), and the depth where the cascades start fading out.
//!
//! Algorithm (per-frame, called from each backend's draw loop):
//!
//!   1. Split the camera's [near, `shadow_distance`] depth range into N cascade
//!      sub-ranges using the practical PSSM blend
//!      (lambda * logarithmic + (1 - lambda) * linear). The blend leans
//!      logarithmic, so the nearest cascade stays small as the reach grows and
//!      shadows at the camera's feet keep their sharpness.
//!   2. For each cascade, compute the 8 frustum corners at its near/far depths,
//!      then bound the corners with a sphere. The sphere bound makes the
//!      orthographic light frustum rotation-invariant, eliminating shimmer
//!      when the camera rotates.
//!   3. Snap the sphere center, in world space along the light's right/up axes,
//!      to a per-cascade texel grid so the grid stays anchored in the world and
//!      individual texels don't crawl as the camera translates.
//!   4. Build a RH `look_at` from outside the sphere along the light direction
//!      and an ortho projection that exactly encloses the sphere.
//!
//! The math is shared across all three backends: Metal, Vulkan, and DirectX
//! all use RH view matrices and the engine's reversed [0, 1] depth in their
//! orthographic projections, so the same VPs are valid for every backend's
//! shadow sampling.

use crate::gfx::projection::look_at;
use crate::gfx::render_types::{CloudShadowLanes, NUM_SHADOW_CASCADES, ShadowUniforms};
use crate::render::depth::shadow_ortho;
use crate::transform::{IDENTITY, mat4_mul};
use concinnity_core::math::SaturatingFrom;
use concinnity_core::math::vec3::{add, cross, dot, length_sq, normalize_clamped, scale, sub};
use concinnity_core::math::{powf, round, sqrt, tan};

const SPLIT_LAMBDA: f32 = 0.85;

// The share of the reach, at its far end, over which the cascaded shadow fades
// out instead of cutting off.
const CASCADE_FADE_FRACTION: f32 = 0.15;

/// The share of each cascade's depth range, at its far end, over which the
/// shader blends it into the next cascade.
pub(crate) const CASCADE_BLEND_FRACTION: f32 = 0.15;

// The view depth from which the shader samples `cascade` of `splits`: zero for
// the first, and for any other the start of the band where the one before it
// blends into it.
fn cascade_blend_in(splits: &[f32; NUM_SHADOW_CASCADES], cascade: usize) -> f32 {
    if cascade == 0 {
        return 0.0;
    }
    let far = splits[cascade - 1];
    let near = if cascade == 1 {
        0.0
    } else {
        splits[cascade - 2]
    };
    far - CASCADE_BLEND_FRACTION * (far - near)
}

/// The live cascades of `uniforms` the shader samples somewhere nearer than
/// `depth`, one bit each. The cascades are sampled in depth order, so the set
/// is always a run from cascade 0.
pub(crate) fn cascades_sampled_within(uniforms: &ShadowUniforms, depth: f32) -> u32 {
    let live = (uniforms.active_cascades as usize).clamp(1, NUM_SHADOW_CASCADES);
    (0..live)
        .take_while(|&i| cascade_blend_in(&uniforms.cascade_splits, i) < depth)
        .fold(0, |mask, i| mask | (1 << i))
}

/// Fallback uniforms used when no shadow pass is active: identity VPs and a
/// single split at +inf so the fragment shader always picks cascade 0 with a
/// 1x1 fallback texture (returns "fully lit"). Nothing fades.
pub fn empty_shadow_uniforms() -> ShadowUniforms {
    ShadowUniforms {
        light_vps: [IDENTITY; NUM_SHADOW_CASCADES],
        cascade_splits: [f32::INFINITY; NUM_SHADOW_CASCADES],
        active_cascades: u32::saturating_from(NUM_SHADOW_CASCADES),
        fade_start: f32::INFINITY,
        _pad: [0; 2],
        clouds: CloudShadowLanes::OFF,
    }
}

// The view depth at which cascades reaching `reach` start fading out: the last
// `CASCADE_FADE_FRACTION` of it, so the shadow is whole up to here and gone at
// the reach.
fn cascade_fade_start(reach: f32) -> f32 {
    reach * (1.0 - CASCADE_FADE_FRACTION)
}

/// Camera, light, and shadow-configuration inputs to
/// [`compute_shadow_uniforms`].
#[derive(Clone, Copy)]
pub struct ShadowUniformInputs {
    /// Camera view matrix (column-major, RH, same convention as `look_at`).
    pub view: [[f32; 4]; 4],
    /// World-space camera position.
    pub cam_pos: [f32; 3],
    /// Vertical FOV in radians.
    pub fov_y_rad: f32,
    /// Viewport aspect ratio (width / height).
    pub aspect: f32,
    /// Camera near plane.
    pub near: f32,
    /// Far end of the last cascade. Cascades cover [near, `shadow_distance`].
    pub shadow_distance: f32,
    /// Unit vector pointing TOWARD the light. Same convention as
    /// `DirectionalLight.direction`; renormalized internally.
    pub light_dir_to_source: [f32; 3],
    /// Per-cascade texture resolution; used for texel snapping.
    pub shadow_map_size: u32,
    /// How many of the `NUM_SHADOW_CASCADES` slots are live (1..=4); only the
    /// first `active` are split + projected, the rest hold a negative split
    /// sentinel and an identity VP so the shader never selects them.
    pub active_cascades: u32,
}

/// Compute cascade VPs + split depths from camera + light parameters.
pub fn compute_shadow_uniforms(inputs: ShadowUniformInputs) -> ShadowUniforms {
    let ShadowUniformInputs {
        view,
        cam_pos,
        fov_y_rad,
        aspect,
        near,
        shadow_distance,
        light_dir_to_source,
        shadow_map_size,
        active_cascades,
    } = inputs;
    let shadow_far = shadow_distance.max(near + 1.0);
    let active = active_cascades.clamp(1, u32::saturating_from(NUM_SHADOW_CASCADES)) as usize;

    // Practical PSSM splits over the `active` cascades. The unused tail keeps a
    // negative sentinel so the fragment shader's `view_depth < split` test never
    // selects a cascade the CPU did not render.
    let cascade_count = active as f32;
    let mut splits = [-1.0_f32; NUM_SHADOW_CASCADES];
    for (i, split) in splits.iter_mut().take(active).enumerate() {
        let p = (i + 1) as f32 / cascade_count;
        let log = near * powf(shadow_far / near, p);
        let lin = near + (shadow_far - near) * p;
        *split = SPLIT_LAMBDA * log + (1.0 - SPLIT_LAMBDA) * lin;
    }

    let l_to = normalize_clamped(light_dir_to_source, 1e-6);

    // Camera basis from view matrix (column-major; look_at fills row 0 = right,
    // row 1 = up, row 2 = -forward into view[*][0], view[*][1], view[*][2]).
    let right = [view[0][0], view[1][0], view[2][0]];
    let up = [view[0][1], view[1][1], view[2][1]];
    let forward = [-view[0][2], -view[1][2], -view[2][2]];

    let tan_half_v = tan(fov_y_rad * 0.5);
    let tan_half_h = tan_half_v * aspect;

    let mut light_vps = [IDENTITY; NUM_SHADOW_CASCADES];
    let mut prev_split = near;
    for i in 0..active {
        let near_d = prev_split;
        let far_d = splits[i];
        prev_split = far_d;

        // 8 frustum corners at near_d and far_d in world space.
        let h_near = near_d * tan_half_v;
        let w_near = near_d * tan_half_h;
        let h_far = far_d * tan_half_v;
        let w_far = far_d * tan_half_h;
        let cn = add(cam_pos, scale(forward, near_d));
        let cf = add(cam_pos, scale(forward, far_d));
        let corners: [[f32; 3]; 8] = [
            add(add(cn, scale(right, -w_near)), scale(up, -h_near)),
            add(add(cn, scale(right, w_near)), scale(up, -h_near)),
            add(add(cn, scale(right, w_near)), scale(up, h_near)),
            add(add(cn, scale(right, -w_near)), scale(up, h_near)),
            add(add(cf, scale(right, -w_far)), scale(up, -h_far)),
            add(add(cf, scale(right, w_far)), scale(up, -h_far)),
            add(add(cf, scale(right, w_far)), scale(up, h_far)),
            add(add(cf, scale(right, -w_far)), scale(up, h_far)),
        ];

        // Bounding sphere of the corners.
        let mut center = [0.0_f32; 3];
        for c in &corners {
            center[0] += c[0];
            center[1] += c[1];
            center[2] += c[2];
        }
        center = scale(center, 1.0 / 8.0);
        let mut r2 = 0.0_f32;
        for c in &corners {
            let d = sub(*c, center);
            let dd = length_sq(d);
            if dd > r2 {
                r2 = dd;
            }
        }
        let radius = sqrt(r2).max(1e-3);

        // Stable up axis: avoid parallel to light direction.
        let up_l = if l_to[1].abs() > 0.95 {
            [1.0_f32, 0.0, 0.0]
        } else {
            [0.0_f32, 1.0, 0.0]
        };

        // Light-space basis matching `look_at` below: f points from the light
        // toward the scene, r and u span the shadow texel grid.
        let f = scale(l_to, -1.0);
        let r = normalize_clamped(cross(f, up_l), 1e-6);
        let u = cross(r, f);

        // Texel-grid snap. Quantize the cascade center along the light's right
        // and up axes to whole shadow texels so the texel grid stays anchored in
        // world space; the texels then stop crawling under camera translation
        // (the shadow stops chasing the camera). The snap must happen in world
        // space, before look_at: snapping the center's light-space xy afterwards
        // is a no-op, because look_at always maps the center onto the optical
        // axis (its light-space xy is identically zero).
        let texel_size = 2.0 * radius / shadow_map_size.max(1) as f32;
        let cx = dot(r, center);
        let cy = dot(u, center);
        let snap_dx = round(cx / texel_size) * texel_size - cx;
        let snap_dy = round(cy / texel_size) * texel_size - cy;
        let center = add(add(center, scale(r, snap_dx)), scale(u, snap_dy));

        // Build the light view from the snapped center and an ortho projection
        // enclosing the sphere. The eye sits at +radius along the light
        // direction so the sphere center maps to the middle of the depth range.
        //
        // The near plane is pushed back toward the light by `caster_extent` so
        // casters ABOVE this cascade's volume (tree canopies, tall building
        // tops, anything between the light and the sphere) still render into the
        // shadow map. Without it the near cascades, whose ortho boxes are only
        // `radius` deep along the light, clip any caster taller than the cascade;
        // as the camera moves, which casters fall inside each cascade changes,
        // so elevated shadows pop in and out and their edges slide with the
        // camera. The extension grows only the depth range, not the XY
        // footprint, so shadow-map resolution is unchanged.
        let caster_extent = shadow_far;
        let light_eye = add(center, scale(l_to, radius));
        let light_view = look_at(light_eye, center, up_l);
        let proj = shadow_ortho(
            -radius,
            radius,
            -radius,
            radius,
            -caster_extent,
            2.0 * radius,
        );
        light_vps[i] = mat4_mul(proj, light_view);
    }

    ShadowUniforms {
        light_vps,
        cascade_splits: splits,
        active_cascades: u32::saturating_from(active),
        fade_start: cascade_fade_start(splits[active - 1]),
        _pad: [0; 2],
        clouds: CloudShadowLanes::OFF,
    }
}

// Right-handed look-at producing a column-major view matrix matching the
// per-backend look_at helpers (Metal, Vulkan, DX all use the same convention).
#[cfg(test)]
mod tests {
    use super::*;

    fn ident_view() -> [[f32; 4]; 4] {
        // Camera at origin looking down -Z, up = +Y.
        look_at([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0])
    }

    #[test]
    fn empty_uniforms_have_infinite_splits() {
        let u = empty_shadow_uniforms();
        for s in &u.cascade_splits {
            assert!(s.is_infinite());
        }
    }

    fn uniforms_reaching(reach: f32, active_cascades: u32) -> ShadowUniforms {
        compute_shadow_uniforms(ShadowUniformInputs {
            view: ident_view(),
            cam_pos: [0.0, 0.0, 0.0],
            fov_y_rad: core::f32::consts::FRAC_PI_2,
            aspect: 1.0,
            near: 0.1,
            shadow_distance: reach,
            light_dir_to_source: [0.0, 1.0, 0.0],
            shadow_map_size: 2048,
            active_cascades,
        })
    }

    // How much of the cascaded shadow remains at `view_depth`: the reference
    // for `cascade_coverage` in `shadow_cascade.hlsl`.
    fn cascade_coverage(view_depth: f32, fade_start: f32, reach: f32) -> f32 {
        if view_depth <= fade_start {
            return 1.0;
        }
        ((reach - view_depth) / (reach - fade_start).max(1e-4)).clamp(0.0, 1.0)
    }

    const REACHES: [f32; 6] = [10.0, 80.0, 160.0, 320.0, 500.0, 2000.0];

    // The weight of each cascade in the shadow at `view_depth`: the reference
    // for the pick and cross-cascade blend of `cascade_shadow` in
    // `shadow_cascade.hlsl`.
    fn cascade_weights(u: &ShadowUniforms, view_depth: f32) -> [f32; NUM_SHADOW_CASCADES] {
        let live = u.active_cascades as usize;
        let mut weights = [0.0; NUM_SHADOW_CASCADES];
        let Some(c) = (0..live).find(|&i| view_depth < u.cascade_splits[i]) else {
            return weights;
        };
        weights[c] = 1.0;
        if c + 1 < live {
            let far = u.cascade_splits[c];
            let near = if c == 0 { 0.0 } else { u.cascade_splits[c - 1] };
            let band = (far - near) * CASCADE_BLEND_FRACTION;
            let t = (view_depth - (far - band)) / band.max(1e-4);
            if t > 0.0 {
                let t = t.clamp(0.0, 1.0);
                weights[c] = 1.0 - t;
                weights[c + 1] = t;
            }
        }
        weights
    }

    #[test]
    fn the_blend_fraction_matches_the_shader() {
        use crate::render::shader::{consts, embedded};
        assert_eq!(
            consts::float(embedded::SHADOW_CASCADE, "CASCADE_BLEND_FRACTION"),
            CASCADE_BLEND_FRACTION
        );
    }

    // Every cascade the shader samples nearer than a depth is in the set for
    // that depth, and nothing else is.
    #[test]
    fn the_sampled_set_holds_exactly_the_cascades_the_shader_reads() {
        for reach in REACHES {
            for count in 1..=4u32 {
                let u = uniforms_reaching(reach, count);
                let mut seen = 0u32;
                let steps = 4000;
                for i in 0..=steps {
                    let d = reach * i as f32 / steps as f32;
                    let within = cascades_sampled_within(&u, d + 1e-4 * reach);
                    for (c, w) in cascade_weights(&u, d).iter().enumerate() {
                        if *w > 0.0 {
                            seen |= 1 << c;
                        }
                    }
                    assert_eq!(seen & !within, 0, "{reach} m, {count}: {d} m");
                }
                assert_eq!(seen, cascades_sampled_within(&u, reach * 2.0));
            }
        }
    }

    // At 80 m the third cascade blends in before its near split, so a depth
    // just short of that split already reads it.
    #[test]
    fn the_sampled_set_counts_the_blend_band() {
        let u = uniforms_reaching(80.0, 4);
        let split = u.cascade_splits[1];
        assert!(split > 8.0 && split < 9.0, "{split}");
        assert_eq!(cascades_sampled_within(&u, split - 0.5), 0b111);
        assert_eq!(cascades_sampled_within(&u, 2.0), 0b1);
        assert_eq!(cascades_sampled_within(&u, 0.0), 0);
        assert_eq!(cascades_sampled_within(&u, 1.0e6), 0b1111);
    }

    #[test]
    fn splits_increase_and_end_at_the_reach() {
        for reach in REACHES {
            for count in 1..=4u32 {
                let u = uniforms_reaching(reach, count);
                let live = &u.cascade_splits[..count as usize];
                assert!(live[0] > 0.1, "{reach} m, {count}: {live:?}");
                assert!(
                    live.windows(2).all(|w| w[1] > w[0]),
                    "{reach} m, {count}: {live:?}"
                );
                let last = live[live.len() - 1];
                assert!(
                    (last - reach).abs() < 1e-3 * reach,
                    "{reach} m, {count}: {live:?}"
                );
            }
        }
    }

    // The nearest cascade, where the camera's feet are, grows far slower than
    // the reach, so a longer reach costs little sharpness up close.
    #[test]
    fn the_first_split_stays_small_as_the_reach_grows() {
        let first = |reach| uniforms_reaching(reach, 4).cascade_splits[0];
        for pair in REACHES.windows(2) {
            let (a, b) = (first(pair[0]), first(pair[1]));
            assert!(b > a, "{pair:?}: {a} then {b}");
            assert!(b / a < pair[1] / pair[0], "{pair:?}: {a} then {b}");
        }
        assert!(first(80.0) < 5.0, "{}", first(80.0));
        assert!(first(320.0) < 15.0, "{}", first(320.0));
    }

    #[test]
    fn the_fade_starts_inside_the_last_cascade() {
        for reach in REACHES {
            for count in 1..=4u32 {
                let u = uniforms_reaching(reach, count);
                let last = count as usize - 1;
                let last_near = if last == 0 {
                    0.1
                } else {
                    u.cascade_splits[last - 1]
                };
                assert!(
                    u.fade_start > last_near && u.fade_start < u.cascade_splits[last],
                    "{reach} m, {count}: fade at {} in {:?}",
                    u.fade_start,
                    u.cascade_splits
                );
                assert!((u.fade_start - cascade_fade_start(reach)).abs() < 1e-3 * reach);
            }
        }
    }

    #[test]
    fn coverage_is_whole_inside_the_fade_and_none_at_the_reach() {
        let u = uniforms_reaching(80.0, 4);
        let (start, reach) = (u.fade_start, u.cascade_splits[3]);
        assert_eq!(cascade_coverage(0.0, start, reach), 1.0);
        assert_eq!(cascade_coverage(start, start, reach), 1.0);
        assert_eq!(cascade_coverage(reach, start, reach), 0.0);
        assert_eq!(cascade_coverage(reach * 2.0, start, reach), 0.0);
        let mid = cascade_coverage(0.5 * (start + reach), start, reach);
        assert!((mid - 0.5).abs() < 1e-4, "{mid}");
        let mut last = 1.0;
        for i in 0..=32 {
            let d = start + (reach - start) * i as f32 / 32.0;
            let c = cascade_coverage(d, start, reach);
            assert!(c <= last, "coverage rose at {d}: {c} after {last}");
            last = c;
        }
    }

    #[test]
    fn empty_uniforms_never_fade() {
        let u = empty_shadow_uniforms();
        for d in [0.0, 80.0, 1.0e6] {
            assert_eq!(cascade_coverage(d, u.fade_start, u.cascade_splits[3]), 1.0);
        }
    }

    #[test]
    fn fewer_active_cascades_fill_only_the_live_slots() {
        // With active_cascades = 2 the first two slots cover [near, shadow_far]
        // (last split == shadow distance) and the unused tail holds the negative
        // sentinel + identity VPs, so the shader never selects an unrendered slot.
        let u = compute_shadow_uniforms(ShadowUniformInputs {
            view: ident_view(),
            cam_pos: [0.0, 0.0, 0.0],
            fov_y_rad: core::f32::consts::FRAC_PI_2,
            aspect: 1.0,
            near: 0.1,
            shadow_distance: 80.0,
            light_dir_to_source: [0.0, 1.0, 0.0],
            shadow_map_size: 2048,
            active_cascades: 2,
        });
        assert_eq!(u.active_cascades, 2);
        assert!(u.cascade_splits[0] > 0.1);
        assert!(u.cascade_splits[1] > u.cascade_splits[0]);
        assert!((u.cascade_splits[1] - 80.0).abs() < 1e-3);
        // Unused tail: negative split sentinel + untouched identity VP.
        assert!(u.cascade_splits[2] < 0.0);
        assert!(u.cascade_splits[3] < 0.0);
        assert_eq!(u.light_vps[2], IDENTITY);
        assert_eq!(u.light_vps[3], IDENTITY);
        // Out-of-range counts clamp into 1..=4.
        let one = compute_shadow_uniforms(ShadowUniformInputs {
            view: ident_view(),
            cam_pos: [0.0, 0.0, 0.0],
            fov_y_rad: core::f32::consts::FRAC_PI_2,
            aspect: 1.0,
            near: 0.1,
            shadow_distance: 80.0,
            light_dir_to_source: [0.0, 1.0, 0.0],
            shadow_map_size: 2048,
            active_cascades: 0,
        });
        assert_eq!(one.active_cascades, 1);
        assert!((one.cascade_splits[0] - 80.0).abs() < 1e-3);
    }

    #[test]
    fn near_clamped_to_avoid_degenerate_log() {
        // shadow_distance smaller than near is clamped to near + 1.0 so the
        // logarithmic split term stays finite.
        let u = compute_shadow_uniforms(ShadowUniformInputs {
            view: ident_view(),
            cam_pos: [0.0, 0.0, 0.0],
            fov_y_rad: core::f32::consts::FRAC_PI_2,
            aspect: 1.0,
            near: 5.0,
            shadow_distance: 1.0,
            light_dir_to_source: [0.0, 1.0, 0.0],
            shadow_map_size: 2048,
            active_cascades: 4,
        });
        for s in &u.cascade_splits {
            assert!(s.is_finite() && *s > 0.0);
        }
    }

    #[test]
    fn cascade_vps_finite_for_typical_inputs() {
        let u = compute_shadow_uniforms(ShadowUniformInputs {
            view: ident_view(),
            cam_pos: [10.0, 5.0, -3.0],
            fov_y_rad: core::f32::consts::FRAC_PI_4,
            aspect: 16.0 / 9.0,
            near: 0.1,
            shadow_distance: 80.0,
            light_dir_to_source: [-0.4, 0.7, 0.3],
            shadow_map_size: 2048,
            active_cascades: 4,
        });
        for vp in &u.light_vps {
            for col in vp {
                for v in col {
                    assert!(v.is_finite(), "non-finite element in light_vp");
                }
            }
        }
    }

    #[test]
    fn point_inside_first_cascade_projects_into_unit_box() {
        // A point a few meters in front of the camera should project into the
        // first cascade's light NDC, inside the [-1, 1] xy box (depth [0, 1]).
        let u = compute_shadow_uniforms(ShadowUniformInputs {
            view: ident_view(),
            cam_pos: [0.0, 0.0, 0.0],
            fov_y_rad: core::f32::consts::FRAC_PI_4,
            aspect: 16.0 / 9.0,
            near: 0.1,
            shadow_distance: 80.0,
            light_dir_to_source: [0.0, 1.0, 0.0],
            shadow_map_size: 2048,
            active_cascades: 4,
        });
        // World point 2m in front of camera (looking down -Z).
        let p = [0.0_f32, 0.0, -2.0, 1.0];
        let vp = u.light_vps[0];
        let mut clip = [0.0_f32; 4];
        for row in 0..4 {
            clip[row] =
                vp[0][row] * p[0] + vp[1][row] * p[1] + vp[2][row] * p[2] + vp[3][row] * p[3];
        }
        let ndc = [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]];
        assert!(ndc[0].abs() <= 1.0, "x out of range: {}", ndc[0]);
        assert!(ndc[1].abs() <= 1.0, "y out of range: {}", ndc[1]);
        assert!(
            ndc[2] >= -0.05 && ndc[2] <= 1.05,
            "depth out of range: {}",
            ndc[2]
        );
    }

    // Project a fixed world point into cascade 0's shadow map and return its
    // texel coordinates for a camera at `cam` (looking down -Z).
    fn cascade0_texel(cam: [f32; 3], world_p: [f32; 3], light: [f32; 3], size: u32) -> (f32, f32) {
        let view = look_at(cam, [cam[0], cam[1], cam[2] - 1.0], [0.0, 1.0, 0.0]);
        let u = compute_shadow_uniforms(ShadowUniformInputs {
            view,
            cam_pos: cam,
            fov_y_rad: core::f32::consts::FRAC_PI_4,
            aspect: 16.0 / 9.0,
            near: 0.1,
            shadow_distance: 80.0,
            light_dir_to_source: light,
            shadow_map_size: size,
            active_cascades: 4,
        });
        let vp = u.light_vps[0];
        let p = [world_p[0], world_p[1], world_p[2], 1.0];
        let mut clip = [0.0_f32; 4];
        for row in 0..4 {
            clip[row] =
                vp[0][row] * p[0] + vp[1][row] * p[1] + vp[2][row] * p[2] + vp[3][row] * p[3];
        }
        let uvx = (clip[0] / clip[3] * 0.5 + 0.5) * size as f32;
        let uvy = (-clip[1] / clip[3] * 0.5 + 0.5) * size as f32;
        (uvx, uvy)
    }

    #[test]
    fn texels_do_not_crawl_under_camera_translation() {
        // The chasing-shadow regression. A fixed world point must land on the
        // SAME sub-texel of the shadow map regardless of camera position: with
        // the texel grid anchored in world space, translating the camera shifts
        // the projected point by a whole number of texels only, so its
        // fractional texel position is invariant. (Before the world-space snap,
        // the center was snapped after look_at, which is a no-op, and the grid
        // slid continuously with the camera -- the shadow "chased" it.)
        let light = [-0.4, 0.78, 0.5];
        let size = 2048u32;
        let world_p = [3.0_f32, 0.0, -5.0];

        // Two camera positions a fraction of a texel apart in world space.
        let a = cascade0_texel([0.0, 2.0, 0.0], world_p, light, size);
        let b = cascade0_texel([0.137, 2.0, 0.091], world_p, light, size);

        // The texel delta must be (within float error) a whole number of
        // texels: the fractional residual is the per-texel crawl.
        let dx = a.0 - b.0;
        let dy = a.1 - b.1;
        let rx = dx - dx.round();
        let ry = dy - dy.round();
        assert!(
            rx.abs() < 0.05,
            "shadow x crawls within a texel: residual {rx}"
        );
        assert!(
            ry.abs() < 0.05,
            "shadow y crawls within a texel: residual {ry}"
        );
    }

    // Project a world point through cascade 0's VP and return its light NDC.
    fn cascade0_ndc(cam: [f32; 3], world_p: [f32; 3], light: [f32; 3]) -> [f32; 3] {
        let view = look_at(cam, [cam[0], cam[1], cam[2] - 1.0], [0.0, 1.0, 0.0]);
        let u = compute_shadow_uniforms(ShadowUniformInputs {
            view,
            cam_pos: cam,
            fov_y_rad: core::f32::consts::FRAC_PI_4,
            aspect: 16.0 / 9.0,
            near: 0.1,
            shadow_distance: 80.0,
            light_dir_to_source: light,
            shadow_map_size: 2048,
            active_cascades: 4,
        });
        let vp = u.light_vps[0];
        let p = [world_p[0], world_p[1], world_p[2], 1.0];
        let mut clip = [0.0_f32; 4];
        for row in 0..4 {
            clip[row] =
                vp[0][row] * p[0] + vp[1][row] * p[1] + vp[2][row] * p[2] + vp[3][row] * p[3];
        }
        [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]]
    }

    #[test]
    fn tall_casters_above_cascade_are_not_clipped() {
        // The disappearing-shadow regression. A caster high above the near
        // cascade (a tree canopy, a building top) must still fall inside the
        // cascade's light frustum so it renders into the shadow map; otherwise
        // its shadow vanishes the moment the receiver drops into a small near
        // cascade, and the clip boundary slides across the world as the camera
        // moves. The ortho near plane is extended toward the light for exactly
        // this. A caster 30m up the light from a receiver in the cascade sits
        // well beyond the few-meter cascade-0 sphere radius, so without the
        // extension it projects past the near plane, to ndc.z > 1 (clipped).
        let cam = [0.0_f32, 0.0, 0.0];
        let light = [0.0_f32, 0.85, 0.3]; // mostly overhead
        let receiver = [0.0_f32, 0.0, -2.0];
        let caster = add(receiver, scale(normalize_clamped(light, 1e-6), 30.0));
        let ndc = cascade0_ndc(cam, caster, light);
        assert!(
            ndc[0].abs() <= 1.0,
            "caster x outside footprint: {}",
            ndc[0]
        );
        assert!(
            ndc[1].abs() <= 1.0,
            "caster y outside footprint: {}",
            ndc[1]
        );
        assert!(
            (0.0..=1.0).contains(&ndc[2]),
            "tall caster clipped from shadow map: ndc.z = {}",
            ndc[2]
        );
        // Nearer the light is nearer in the map, so the caster wins the depth
        // test against the ground below it.
        let ground = cascade0_ndc(cam, receiver, light);
        assert!(ndc[2] > ground[2], "{} vs {}", ndc[2], ground[2]);
    }

    // Every cascade's depth rises toward the light at exactly its z row's
    // length per meter: the scale the sample bias converts meters with.
    #[test]
    fn each_cascade_maps_its_sphere_into_reversed_depth() {
        let light = normalize_clamped([-0.4, 0.78, 0.5], 1e-6);
        let u = compute_shadow_uniforms(ShadowUniformInputs {
            view: ident_view(),
            cam_pos: [0.0, 0.0, 0.0],
            fov_y_rad: core::f32::consts::FRAC_PI_4,
            aspect: 16.0 / 9.0,
            near: 0.1,
            shadow_distance: 80.0,
            light_dir_to_source: light,
            shadow_map_size: 2048,
            active_cascades: 4,
        });
        for vp in u.light_vps {
            let depth = |p: [f32; 3]| {
                let z = vp[0][2] * p[0] + vp[1][2] * p[1] + vp[2][2] * p[2] + vp[3][2];
                let w = vp[0][3] * p[0] + vp[1][3] * p[1] + vp[2][3] * p[2] + vp[3][3];
                z / w
            };
            // The z row's length is the depth scale per meter along the light.
            let row2 = [vp[0][2], vp[1][2], vp[2][2]];
            let per_meter = sqrt(length_sq(row2));
            let probe = [0.0, 0.0, -5.0];
            let toward_light = depth(add(probe, scale(light, 1.0))) - depth(probe);
            assert!(toward_light > 0.0, "{toward_light}");
            assert!((toward_light - per_meter).abs() < 1e-3 * per_meter.max(1e-6) + 1e-6);
        }
    }
}
