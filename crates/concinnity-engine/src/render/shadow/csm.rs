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
use crate::gfx::render_types::{
    CloudShadowLanes, FarShadowLanes, NUM_SHADOW_CASCADES, ShadowUniforms,
};
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

/// The view depth from which the shader samples `cascade` of `splits`: zero
/// for the first, and for any other the start of the band where the one before
/// it blends into it.
pub(crate) fn cascade_blend_in(splits: &[f32; NUM_SHADOW_CASCADES], cascade: usize) -> f32 {
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
        far: FarShadowLanes::OFF,
    }
}

// The view depth at which cascades reaching `reach` start fading out: the last
// `CASCADE_FADE_FRACTION` of it, so the shadow is whole up to here and gone at
// the reach.
fn cascade_fade_start(reach: f32) -> f32 {
    reach * (1.0 - CASCADE_FADE_FRACTION)
}

/// How far back toward the light a shadow view covering `shadow_far` takes its
/// casters from.
pub(crate) fn caster_extent(shadow_far: f32) -> f32 {
    shadow_far
}

/// The axes a shadow view lit along `to_light` lays its texel grid on, matching
/// the `look_at` it renders through.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LightBasis {
    /// The grid's first axis.
    pub right: [f32; 3],
    /// The grid's second axis.
    pub up: [f32; 3],
    /// The up vector the view's `look_at` takes: never parallel to the light.
    pub up_hint: [f32; 3],
}

impl LightBasis {
    /// The basis of a view lit along the unit vector `to_light`.
    pub(crate) fn toward(to_light: [f32; 3]) -> Self {
        let up_hint = if to_light[1].abs() > 0.95 {
            [1.0_f32, 0.0, 0.0]
        } else {
            [0.0_f32, 1.0, 0.0]
        };
        let forward = scale(to_light, -1.0);
        let right = normalize_clamped(cross(forward, up_hint), 1e-6);
        Self {
            right,
            up: cross(right, forward),
            up_hint,
        }
    }
}

/// How far the farthest corner of a shadow view's light box lies from the
/// center of the sphere it is fit to: the box spans the sphere's `radius`
/// across and reaches `extent` past it toward the light.
pub(crate) fn light_box_corner(radius: f32, extent: f32) -> f32 {
    let side = radius + extent;
    sqrt(side * side + 2.0 * radius * radius)
}

// Practical PSSM splits of [near, `shadow_far`] over `active` cascades. The
// unused tail keeps a negative sentinel so the fragment shader's
// `view_depth < split` test never selects a cascade the CPU did not render.
fn cascade_splits(near: f32, shadow_far: f32, active: usize) -> [f32; NUM_SHADOW_CASCADES] {
    let cascade_count = active as f32;
    let mut splits = [-1.0_f32; NUM_SHADOW_CASCADES];
    for (i, split) in splits.iter_mut().take(active).enumerate() {
        let p = (i + 1) as f32 / cascade_count;
        let log = near * powf(shadow_far / near, p);
        let lin = near + (shadow_far - near) * p;
        *split = SPLIT_LAMBDA * log + (1.0 - SPLIT_LAMBDA) * lin;
    }
    splits
}

/// Where a camera sits and which way it faces.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CameraBasis {
    position: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
    forward: [f32; 3],
}

impl CameraBasis {
    /// The basis of a camera at `position` with the column-major world-to-view
    /// matrix `view`: `look_at` fills row 0 with right, row 1 with up and row
    /// 2 with -forward.
    pub(crate) fn of_view(view: &[[f32; 4]; 4], position: [f32; 3]) -> Self {
        Self {
            position,
            right: [view[0][0], view[1][0], view[2][0]],
            up: [view[0][1], view[1][1], view[2][1]],
            forward: [-view[0][2], -view[1][2], -view[2][2]],
        }
    }
}

/// The 8 corners of the camera frustum's slice between view depths `near_d`
/// and `far_d`, for half-angle tangents `tan_half_v` and `tan_half_h`.
pub(crate) fn slice_corners(
    camera: &CameraBasis,
    near_d: f32,
    far_d: f32,
    tan_half_v: f32,
    tan_half_h: f32,
) -> [[f32; 3]; 8] {
    let CameraBasis {
        position,
        right,
        up,
        forward,
    } = *camera;
    let h_near = near_d * tan_half_v;
    let w_near = near_d * tan_half_h;
    let h_far = far_d * tan_half_v;
    let w_far = far_d * tan_half_h;
    let cn = add(position, scale(forward, near_d));
    let cf = add(position, scale(forward, far_d));
    [
        add(add(cn, scale(right, -w_near)), scale(up, -h_near)),
        add(add(cn, scale(right, w_near)), scale(up, -h_near)),
        add(add(cn, scale(right, w_near)), scale(up, h_near)),
        add(add(cn, scale(right, -w_near)), scale(up, h_near)),
        add(add(cf, scale(right, -w_far)), scale(up, -h_far)),
        add(add(cf, scale(right, w_far)), scale(up, -h_far)),
        add(add(cf, scale(right, w_far)), scale(up, h_far)),
        add(add(cf, scale(right, -w_far)), scale(up, h_far)),
    ]
}

// The sphere bounding the 8 corners of the camera frustum's slice between view
// depths `near_d` and `far_d`: its center and radius.
fn slice_sphere(
    camera: &CameraBasis,
    near_d: f32,
    far_d: f32,
    tan_half_v: f32,
    tan_half_h: f32,
) -> ([f32; 3], f32) {
    let corners = slice_corners(camera, near_d, far_d, tan_half_v, tan_half_h);
    let mut center = [0.0_f32; 3];
    for c in &corners {
        center = add(center, *c);
    }
    center = scale(center, 1.0 / 8.0);
    let r2 = corners
        .iter()
        .map(|c| length_sq(sub(*c, center)))
        .fold(0.0_f32, f32::max);
    (center, sqrt(r2).max(1e-3))
}

/// The camera the cascades are fit to, as far as their extent depends on it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CascadeView {
    /// Camera near plane.
    pub near: f32,
    /// Vertical FOV in radians.
    pub fov_y_rad: f32,
    /// Viewport aspect ratio (width / height).
    pub aspect: f32,
    /// How far the camera sees, which caps every shadow's reach; `None` for
    /// no limit.
    pub view_distance: Option<f32>,
}

/// The cascaded shadows a renderer runs, as far as where they take their
/// casters from depends on them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CascadeReach {
    /// Far end of the last cascade, in world units.
    pub distance: f32,
    /// How many cascades split that range.
    pub cascades: u32,
    /// The shadow map size the renderer runs; 0 runs no cascades.
    pub map_size: u32,
}

impl CascadeReach {
    /// How far from the camera a caster can lie and still render into a
    /// cascade fit to `view`, whatever the light's direction: the farthest
    /// corner of any cascade's light box, which spans its slice's bounding
    /// sphere across and reaches back toward the light past it. The cascades
    /// reach no farther than the camera sees. 0 when no cascades run.
    pub fn caster_reach(&self, view: CascadeView) -> f32 {
        if self.map_size == 0 {
            return 0.0;
        }
        let distance = view
            .view_distance
            .map_or(self.distance, |d| self.distance.min(d));
        let shadow_far = distance.max(view.near + 1.0);
        let active = self
            .cascades
            .clamp(1, u32::saturating_from(NUM_SHADOW_CASCADES)) as usize;
        let splits = cascade_splits(view.near, shadow_far, active);
        let extent = caster_extent(shadow_far);
        let tan_half_v = tan(view.fov_y_rad * 0.5);
        let camera = CameraBasis {
            position: [0.0; 3],
            right: [1.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, -1.0],
        };
        let mut near_d = view.near;
        let mut reach = 0.0_f32;
        for &far_d in &splits[..active] {
            let (center, radius) =
                slice_sphere(&camera, near_d, far_d, tan_half_v, tan_half_v * view.aspect);
            near_d = far_d;
            // The texel snap moves the box by up to half a texel on two axes.
            let snap = radius * core::f32::consts::SQRT_2 / self.map_size as f32;
            reach = reach.max(sqrt(length_sq(center)) + snap + light_box_corner(radius, extent));
        }
        reach
    }
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

    let splits = cascade_splits(near, shadow_far, active);

    let l_to = normalize_clamped(light_dir_to_source, 1e-6);

    let camera = CameraBasis::of_view(&view, cam_pos);

    let tan_half_v = tan(fov_y_rad * 0.5);
    let tan_half_h = tan_half_v * aspect;

    let mut light_vps = [IDENTITY; NUM_SHADOW_CASCADES];
    let mut prev_split = near;
    for (vp, &far_d) in light_vps.iter_mut().zip(&splits[..active]) {
        let near_d = prev_split;
        prev_split = far_d;
        let (center, radius) = slice_sphere(&camera, near_d, far_d, tan_half_v, tan_half_h);

        let LightBasis {
            right: r,
            up: u,
            up_hint: up_l,
        } = LightBasis::toward(l_to);

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
        let caster_extent = caster_extent(shadow_far);
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
        *vp = mat4_mul(proj, light_view);
    }

    ShadowUniforms {
        light_vps,
        cascade_splits: splits,
        active_cascades: u32::saturating_from(active),
        fade_start: cascade_fade_start(splits[active - 1]),
        _pad: [0; 2],
        clouds: CloudShadowLanes::OFF,
        far: FarShadowLanes::OFF,
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

    // Every corner of every cascade's light box, the volume its casters render
    // from, lies within the caster reach, whichever way the camera faces and
    // the light falls; and the reach is not loose, since a light low ahead of
    // the camera pushes a box corner close to it.
    #[test]
    fn every_cascade_box_lies_within_the_caster_reach() {
        use crate::transform::mat4_inverse;
        use concinnity_core::math::{cos, sin};
        let near = 0.1;
        let cam = [12.0, 40.0, -7.0];
        for fov_deg in [40.0_f32, 70.0, 100.0] {
            for aspect in [1.0_f32, 16.0 / 9.0, 21.0 / 9.0] {
                for distance in [40.0_f32, 250.0, 800.0] {
                    for cascades in 1..=4_u32 {
                        let reach = CascadeReach {
                            distance,
                            cascades,
                            map_size: 2048,
                        }
                        .caster_reach(CascadeView {
                            near,
                            fov_y_rad: fov_deg.to_radians(),
                            aspect,
                            view_distance: None,
                        });
                        let mut farthest = 0.0_f32;
                        for yaw in [0.0_f32, 1.1, 2.6, 4.0] {
                            let ahead = [sin(yaw), -0.2, -cos(yaw)];
                            let view = look_at(cam, add(cam, ahead), [0.0, 1.0, 0.0]);
                            for (az, elev) in [(0.0_f32, 0.09), (1.7, 0.3), (3.4, 1.2), (5.0, 0.02)]
                            {
                                let light = [cos(elev) * sin(az), sin(elev), -cos(elev) * cos(az)];
                                for light in [light, add(ahead, [0.0, 0.25, 0.0])] {
                                    let u = compute_shadow_uniforms(ShadowUniformInputs {
                                        view,
                                        cam_pos: cam,
                                        fov_y_rad: fov_deg.to_radians(),
                                        aspect,
                                        near,
                                        shadow_distance: distance,
                                        light_dir_to_source: light,
                                        shadow_map_size: 2048,
                                        active_cascades: cascades,
                                    });
                                    for vp in &u.light_vps[..cascades as usize] {
                                        let inv = mat4_inverse(*vp);
                                        for corner in 0..8 {
                                            let ndc = [
                                                if corner & 1 == 0 { -1.0 } else { 1.0 },
                                                if corner & 2 == 0 { -1.0 } else { 1.0 },
                                                if corner & 4 == 0 { 0.0 } else { 1.0 },
                                                1.0,
                                            ];
                                            let p = mat4_mul_vec4(inv, ndc);
                                            let world = scale([p[0], p[1], p[2]], 1.0 / p[3]);
                                            let d = sqrt(length_sq(sub(world, cam)));
                                            farthest = farthest.max(d);
                                            assert!(
                                                d <= reach * 1.001 + 0.01,
                                                "{d} > {reach}: fov {fov_deg} aspect {aspect} \
                                                 reach {distance} cascades {cascades}"
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        assert!(
                            farthest > 0.9 * reach,
                            "{farthest} of {reach}: fov {fov_deg} aspect {aspect} \
                             reach {distance} cascades {cascades}"
                        );
                    }
                }
            }
        }
    }

    // Column-major `m` times `v`.
    fn mat4_mul_vec4(m: [[f32; 4]; 4], v: [f32; 4]) -> [f32; 4] {
        let mut out = [0.0; 4];
        for (col, &x) in m.iter().zip(&v) {
            for (o, &c) in out.iter_mut().zip(col) {
                *o += c * x;
            }
        }
        out
    }

    #[test]
    fn no_shadow_maps_take_no_casters() {
        let reach = CascadeReach {
            distance: 250.0,
            cascades: 4,
            map_size: 0,
        };
        let view = CascadeView {
            near: 0.1,
            fov_y_rad: 1.2,
            aspect: 1.5,
            view_distance: None,
        };
        assert_eq!(reach.caster_reach(view), 0.0);
    }

    // The cascades reach no farther than the camera sees, so they take their
    // casters no farther either.
    #[test]
    fn the_caster_reach_stops_at_the_view_distance() {
        let view = CascadeView {
            near: 0.1,
            fov_y_rad: 1.2,
            aspect: 1.5,
            view_distance: None,
        };
        let reach = |distance| CascadeReach {
            distance,
            cascades: 4,
            map_size: 2048,
        };
        let short = CascadeView {
            view_distance: Some(100.0),
            ..view
        };
        assert_eq!(
            reach(250.0).caster_reach(short),
            reach(100.0).caster_reach(view)
        );
        let beyond = CascadeView {
            view_distance: Some(900.0),
            ..view
        };
        assert_eq!(
            reach(250.0).caster_reach(beyond),
            reach(250.0).caster_reach(view)
        );
    }
}
