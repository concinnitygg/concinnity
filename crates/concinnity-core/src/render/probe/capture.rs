//! Reflection probe capture math: the six cube-face view-projection matrices a
//! probe renders the scene through, plus the load-time conversion of the
//! captured faces into the prefiltered IBL payload the environment sampler
//! consumes. Backend-agnostic; each backend renders the scene into the faces.
//!
//! Face order and orientation come from [`crate::gfx::cubemap`], the one home of
//! the engine's cube convention. Each face's view-projection is built so that
//! `face_dir(face, u, v)` projects to NDC `(u, -v)` (screen-down is +v), which
//! the orientation test pins exactly.

use crate::bake::environment_map as em;
use crate::components::ReflectionProbe;
use crate::gfx::cubemap::FACE_BASIS;
use crate::gfx::frustum::Frustum;
use crate::gfx::projection::view_from_basis;
use crate::render::depth::camera_projection;
use crate::render::probe::radiance::PROBE_STORAGE_SCALE;
use crate::render::uniforms::ProbePrefilterParams;
use crate::transform::mat4_mul;
use core::f32::consts::FRAC_PI_2;

// One cube face: a 90-degree vertical field of view at a square aspect, with
// the camera's infinite depth.
fn perspective_90(near: f32) -> [[f32; 4]; 4] {
    camera_projection(FRAC_PI_2, 1.0, near)
}

/// Near plane of every probe face. The capture is independent of the live
/// camera's near plane and view distance, so a probe sees the same scene
/// whatever the camera's are.
const CAPTURE_NEAR: f32 = 0.05;

/// The view-projection for cube face `face` (0..6) captured from `eye`, from
/// `CAPTURE_NEAR` out to infinity.
pub fn face_view_projection(eye: [f32; 3], face: usize) -> [[f32; 4]; 4] {
    let b = FACE_BASIS[face];
    let view = view_from_basis(eye, b[0], b[1], b[2]);
    mat4_mul(perspective_90(CAPTURE_NEAR), view)
}

/// The culling frustum of cube face `face` captured from `eye`. With a
/// `capture_distance` it keeps only objects reaching nearer than that along the
/// face's view axis; `None` reaches without limit.
pub fn face_frustum(eye: [f32; 3], face: usize, capture_distance: Option<f32>) -> Frustum {
    Frustum::from_camera(face_view_projection(eye, face), capture_distance)
}

/// The world->view matrix alone for cube face `face`, captured from `eye`. The
/// main pass needs both the combined view-projection (vertex clip transform) and
/// the bare view matrix (some shaders reconstruct view-space data), so the probe
/// capture builds a `ViewUniforms` from this plus `face_view_projection`.
pub fn face_view_matrix(eye: [f32; 3], face: usize) -> [[f32; 4]; 4] {
    let b = FACE_BASIS[face];
    view_from_basis(eye, b[0], b[1], b[2])
}

/// Where one reflection probe is captured from and the influence box it serves.
/// Backend-agnostic: the renderer bakes a cube at `position` and, for a surface
/// inside [`box_min`, `box_max`], selects this probe and parallax-corrects against the
/// box. Derived from a declared `ReflectionProbe` asset or `auto_seed_probes`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbePlacement {
    /// World-space position.
    pub position: [f32; 3],
    /// Lower corner of the probe's parallax box.
    pub box_min: [f32; 3],
    /// Upper corner of the probe's parallax box.
    pub box_max: [f32; 3],
    /// How far each cube face's capture reaches along its view axis. `None`
    /// captures without limit.
    pub capture_distance: Option<f32>,
    /// Distance along each axis the probe's blend weight fades over across
    /// the box surface.
    pub blend_margin: [f32; 3],
    /// Whether a point outside every probe's influence fades from this capture
    /// to the environment with distance from the box, or keeps the capture.
    pub sky_fallback: bool,
}

/// Fraction of a box's half-extent a probe's blend margin spans when nothing
/// sets it.
const BLEND_MARGIN_FRACTION: f32 = 0.2;

/// Floor of a blend margin, which keeps a zero margin from dividing by zero in
/// the blend weight.
const MIN_BLEND_MARGIN: f32 = 1e-4;

/// A blend margin the same on every axis: `blend_distance`, or a fraction of
/// the box's smallest half-extent, so a declared probe reaches only a little
/// past its walls whichever way it is long.
fn uniform_blend_margin(half_extents: [f32; 3], blend_distance: Option<f32>) -> [f32; 3] {
    let smallest = half_extents[0].min(half_extents[1]).min(half_extents[2]);
    let margin = blend_distance.unwrap_or(BLEND_MARGIN_FRACTION * smallest);
    [margin.max(MIN_BLEND_MARGIN); 3]
}

/// A blend margin that scales with the box's extent along each axis, so a wide,
/// shallow seeded cell fades across its width as gradually as a cube of that
/// width would.
fn per_axis_blend_margin(box_min: [f32; 3], box_max: [f32; 3]) -> [f32; 3] {
    core::array::from_fn(|i| {
        (BLEND_MARGIN_FRACTION * 0.5 * (box_max[i] - box_min[i])).max(MIN_BLEND_MARGIN)
    })
}

impl From<&ReflectionProbe> for ProbePlacement {
    fn from(probe: &ReflectionProbe) -> Self {
        Self {
            capture_distance: probe.capture_distance,
            blend_margin: uniform_blend_margin(probe.half_extents, probe.blend_distance),
            sky_fallback: probe.sky_fallback,
            ..Self::from_center_extents(probe.position, probe.half_extents)
        }
    }
}

impl ProbePlacement {
    /// A placement captured from `position` without limit, whose box is
    /// `position` plus or minus `half_extents`, blending as a declared probe
    /// with no `blend_distance` does.
    pub fn from_center_extents(position: [f32; 3], half_extents: [f32; 3]) -> Self {
        Self {
            position,
            box_min: core::array::from_fn(|i| position[i] - half_extents[i]),
            box_max: core::array::from_fn(|i| position[i] + half_extents[i]),
            capture_distance: None,
            blend_margin: uniform_blend_margin(half_extents, None),
            sky_fallback: true,
        }
    }

    /// An auto-seeded placement captured from `position` without limit over the
    /// box `box_min..box_max`, blending over a margin per axis.
    pub(super) fn seeded(position: [f32; 3], box_min: [f32; 3], box_max: [f32; 3]) -> Self {
        Self {
            position,
            box_min,
            box_max,
            capture_distance: None,
            blend_margin: per_axis_blend_margin(box_min, box_max),
            sky_fallback: true,
        }
    }

    /// The record the shaders read for this placement once its cube is baked:
    /// the box with parallax enabled (`box_min.w = 1`), the sky fallback
    /// (`box_max.w`), the capture point and the blend margin.
    pub fn uniforms(&self) -> crate::render::uniforms::ProbeUniforms {
        let [x0, y0, z0] = self.box_min;
        let [x1, y1, z1] = self.box_max;
        let [px, py, pz] = self.position;
        let [mx, my, mz] = self.blend_margin;
        crate::render::uniforms::ProbeUniforms {
            box_min: [x0, y0, z0, 1.0],
            box_max: [x1, y1, z1, if self.sky_fallback { 1.0 } else { 0.0 }],
            probe_pos: [px, py, pz, 0.0],
            blend_margin: [mx, my, mz, 0.0],
        }
    }
}

/// The dispatch plan of one runtime probe prefilter: which kernel runs at which
/// destination mip, and the parameters each one pushes.
///
/// The three backends drive identical dispatches, so the sizes, the roughness per
/// mip, the sample count and the firefly clamp are decided once here rather than
/// in three copies that could drift and make a probe look different per platform.
/// The roughness comes from the same [`em::prefilter_roughness`] the build-time
/// CPU convolution uses, so a captured probe and an imported HDR agree.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrefilterPlan {
    face_size: u32,
    mips: u32,
    samples: u32,
    clamp: f32,
    storage_scale: f32,
}

impl PrefilterPlan {
    /// The plan a runtime capture bakes with.
    ///
    /// 512px faces match the `EnvironmentMap` asset default, so a captured probe
    /// resolves reflections as sharply as an imported HDR. 128 GGX samples per
    /// texel is well short of the importer's 1024, which the solid-angle mip
    /// selection in `probe_prefilter.hlsl` pays for: each sample reads the
    /// pyramid level matching its own footprint, so a sparse set still integrates
    /// a wide lobe without aliasing. The clamp matches the asset default, so a
    /// captured probe and an imported HDR suppress the same bright-square fringe.
    /// The faces are stored at [`PROBE_STORAGE_SCALE`].
    pub const RUNTIME: Self = Self {
        face_size: 512,
        mips: em::max_mip_count(512),
        samples: 128,
        clamp: 12.0,
        storage_scale: PROBE_STORAGE_SCALE,
    };

    /// A plan over `face_size` faces at `samples` GGX samples and firefly cap
    /// `clamp`. `face_size` must be a power of two of at least four.
    pub const fn new(face_size: u32, samples: u32, clamp: f32) -> Self {
        Self {
            face_size,
            mips: em::max_mip_count(face_size),
            samples,
            clamp,
            storage_scale: 1.0,
        }
    }

    /// This plan over a cube whose radiance is stored at `scale`: the firefly
    /// cap, given in absolute radiance, is applied at the same scale.
    #[must_use]
    pub const fn stored_at(self, scale: f32) -> Self {
        Self {
            storage_scale: scale,
            ..self
        }
    }

    /// Cube-face edge of the capture, in texels: mip 0 of both cubes.
    pub const fn face_size(&self) -> u32 {
        self.face_size
    }

    /// Mip levels in the probe cube, and in the capture pyramid feeding it.
    pub const fn mips(&self) -> u32 {
        self.mips
    }

    /// Cube-face edge of mip `mip`, in texels.
    pub const fn mip_face_size(&self, mip: u32) -> u32 {
        self.face_size >> mip
    }

    /// Params for the clamped copy of the capture into probe-cube mip 0.
    pub fn mip0_params(&self) -> ProbePrefilterParams {
        ProbePrefilterParams {
            dst_size: self.face_size,
            ..self.base_params()
        }
    }

    /// Params for the box reduction producing capture-pyramid mip `dst_mip`,
    /// which reads mip `dst_mip - 1`. Only valid for `1 <= dst_mip < mips`.
    pub fn downsample_params(&self, dst_mip: u32) -> ProbePrefilterParams {
        ProbePrefilterParams {
            dst_size: self.mip_face_size(dst_mip),
            src_mip: dst_mip - 1,
            ..self.base_params()
        }
    }

    /// Params for the GGX convolution producing probe-cube mip `dst_mip`. Only
    /// valid for `1 <= dst_mip < mips`; mip 0 is the clamped copy instead.
    pub fn ggx_params(&self, dst_mip: u32) -> ProbePrefilterParams {
        ProbePrefilterParams {
            dst_size: self.mip_face_size(dst_mip),
            roughness: em::prefilter_roughness(dst_mip, self.mips),
            ..self.base_params()
        }
    }

    fn base_params(&self) -> ProbePrefilterParams {
        ProbePrefilterParams {
            dst_size: self.face_size,
            src_size: self.face_size,
            sample_count: self.samples,
            src_mip: 0,
            roughness: 0.0,
            clamp_lum: self.clamp * self.storage_scale,
            src_mip_count: self.mips as f32,
            _pad: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::cubemap::face_dir;

    fn project(vp: [[f32; 4]; 4], p: [f32; 3]) -> (f32, f32, f32) {
        let mut c = [0.0f32; 4];
        let pv = [p[0], p[1], p[2], 1.0];
        for row in 0..4 {
            for k in 0..4 {
                c[row] += vp[k][row] * pv[k];
            }
        }
        (c[0] / c[3], c[1] / c[3], c[3])
    }

    // Each face's view-projection must map face_dir(face, u, v) to NDC
    // (u, -v): screen-right is +u, screen-down is +v, exactly the layout the
    // capture cube stores and the prefilter samples. A flipped or rotated face
    // would break this and is the classic cube-capture bug.
    #[test]
    fn face_view_projection_matches_cube_convention() {
        let eye = [3.0, -1.5, 2.0];
        let samples = [
            (0.0f32, 0.0f32),
            (0.5, 0.0),
            (0.0, 0.5),
            (-0.6, 0.3),
            (0.7, -0.4),
        ];
        for face in 0..6 {
            let vp = face_view_projection(eye, face);
            for &(u, v) in &samples {
                let d = face_dir(face, u, v);
                let p = [eye[0] + d[0], eye[1] + d[1], eye[2] + d[2]];
                let (nx, ny, w) = project(vp, p);
                assert!(
                    w > 0.0,
                    "face {face} sample ({u},{v}) behind camera (w={w})"
                );
                assert!(
                    (nx - u).abs() < 1e-4 && (ny - (-v)).abs() < 1e-4,
                    "face {face} ({u},{v}) -> ndc ({nx},{ny}), expected ({u},{})",
                    -v
                );
            }
        }
    }

    // Device depth of the point `distance` along face `face`'s axis.
    fn face_depth(face: usize, distance: f32) -> f32 {
        let eye = [4.0, 1.0, -2.0];
        let d = face_dir(face, 0.0, 0.0);
        let p = [
            eye[0] + d[0] * distance,
            eye[1] + d[1] * distance,
            eye[2] + d[2] * distance,
        ];
        let vp = face_view_projection(eye, face);
        let pv = [p[0], p[1], p[2], 1.0];
        let (mut z, mut w) = (0.0, 0.0);
        for k in 0..4 {
            z += vp[k][2] * pv[k];
            w += vp[k][3] * pv[k];
        }
        z / w
    }

    // The capture keeps geometry from just past the near plane out to any
    // distance, and clips only nearer than its near plane.
    #[test]
    fn every_face_captures_from_its_near_plane_out() {
        for face in 0..6 {
            let mut prev = 1.0;
            for distance in [CAPTURE_NEAR * 1.5, 1.0, 250.0, 2.0e3, 1.0e5, 1.0e8] {
                let z = face_depth(face, distance);
                assert!(z > 0.0 && z < prev, "face {face} at {distance}: z {z}");
                prev = z;
            }
            // Reversed depth: nearer than the near plane rises above 1.
            assert!(face_depth(face, CAPTURE_NEAR * 0.5) > 1.0, "face {face}");
        }
    }

    #[test]
    fn face_view_matrix_composes_to_face_vp() {
        // The exposed bare view matrix, pre-multiplied by the 90-degree
        // projection, must reproduce `face_view_projection` exactly (the probe
        // capture builds ViewUniforms from the two separately).
        let eye = [1.0, 2.0, -3.0];
        for face in 0..6 {
            let vp = face_view_projection(eye, face);
            let comp = mat4_mul(perspective_90(CAPTURE_NEAR), face_view_matrix(eye, face));
            for c in 0..4 {
                for r in 0..4 {
                    assert!(
                        (vp[c][r] - comp[c][r]).abs() < 1e-5,
                        "face {face} [{c}][{r}] mismatch"
                    );
                }
            }
        }
    }

    #[test]
    fn a_placement_record_enables_parallax_over_its_box() {
        let u = ProbePlacement::from_center_extents([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]).uniforms();
        assert_eq!(u.box_min, [-3.0, -3.0, -3.0, 1.0]);
        assert_eq!(u.box_max, [5.0, 7.0, 9.0, 1.0]);
        assert_eq!(u.probe_pos, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(u.blend_margin, [0.8, 0.8, 0.8, 0.0]);
    }

    // A long room reaches as far past its end walls as past its side walls, so
    // it does not leak into the room beyond either.
    #[test]
    fn a_declared_probe_blends_over_its_smallest_half_extent() {
        let probe = ReflectionProbe {
            position: [0.0; 3],
            half_extents: [10.0, 1.5, 3.0],
            ..Default::default()
        };
        let u = ProbePlacement::from(&probe).uniforms();
        let near = |a: [f32; 3], b: [f32; 3]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5);
        assert!(near(u.blend_margin(), [0.3; 3]), "{:?}", u.blend_margin());
        let (lo, hi) = u.influence_bounds();
        assert!(near(lo, [-10.3, -1.8, -3.3]), "{lo:?}");
        assert!(near(hi, [10.3, 1.8, 3.3]), "{hi:?}");
    }

    #[test]
    fn a_declared_blend_distance_is_used_on_every_axis() {
        let probe = ReflectionProbe {
            position: [0.0; 3],
            half_extents: [10.0, 1.5, 3.0],
            blend_distance: Some(2.0),
            ..Default::default()
        };
        let u = ProbePlacement::from(&probe).uniforms();
        assert_eq!(u.blend_margin(), [2.0; 3]);
        assert_eq!(
            u.influence_bounds(),
            ([-12.0, -3.5, -5.0], [12.0, 3.5, 5.0])
        );
    }

    // A probe validated to no blend still divides by a positive margin.
    #[test]
    fn a_zero_blend_distance_keeps_a_positive_margin() {
        let probe = ReflectionProbe {
            blend_distance: Some(0.0),
            ..Default::default()
        };
        let margin = ProbePlacement::from(&probe).blend_margin;
        assert!(margin.iter().all(|&m| m > 0.0 && m <= 1e-3), "{margin:?}");
    }

    #[test]
    fn a_declared_probe_carries_its_sky_fallback_into_its_record() {
        let on = ProbePlacement::from(&ReflectionProbe::default());
        assert!(on.sky_fallback);
        assert_eq!(on.uniforms().box_max[3], 1.0);
        let off = ProbePlacement::from(&ReflectionProbe {
            sky_fallback: false,
            ..Default::default()
        });
        assert!(!off.sky_fallback);
        assert_eq!(off.uniforms().box_max[3], 0.0);
    }

    #[test]
    fn placement_from_center_extents_builds_box() {
        let p = ProbePlacement::from_center_extents([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
        assert_eq!(p.box_min, [-3.0, -3.0, -3.0]);
        assert_eq!(p.box_max, [5.0, 7.0, 9.0]);
        assert_eq!(p.position, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn a_declared_probe_carries_its_capture_distance() {
        let probe = ReflectionProbe {
            position: [1.0, 2.0, 3.0],
            half_extents: [4.0, 5.0, 6.0],
            capture_distance: Some(7.5),
            ..Default::default()
        };
        let p = ProbePlacement::from(&probe);
        assert_eq!(p.capture_distance, Some(7.5));
        assert_eq!(p.box_min, [-3.0, -3.0, -3.0]);
        assert_eq!(p.box_max, [5.0, 7.0, 9.0]);
        assert_eq!(p.position, [1.0, 2.0, 3.0]);
        let unlimited = ProbePlacement::from(&ReflectionProbe::default());
        assert_eq!(unlimited.capture_distance, None);
    }

    // A box of half-size `half` centered `along` the face axis and `across` it
    // (toward the face's +u edge) from `eye`.
    fn box_on_face(eye: [f32; 3], face: usize, along: f32, across: f32, half: f32) -> bool {
        let axis = face_dir(face, 0.0, 0.0);
        let side = FACE_BASIS[face][0];
        let c: [f32; 3] = core::array::from_fn(|i| eye[i] + axis[i] * along + side[i] * across);
        face_frustum(eye, face, Some(10.0))
            .intersects_aabb(c.map(|x| x - half), c.map(|x| x + half))
    }

    // The distance is measured along each face's own axis, so across the six
    // faces it bounds a cube: an object off to the side but nearer than the
    // distance along the axis stays even when its straight-line distance
    // exceeds it, and an object straddling the distance is kept whole.
    #[test]
    fn a_capture_distance_bounds_every_face_along_its_axis() {
        let eye = [3.0, -1.5, 2.0];
        for face in 0..6 {
            assert!(
                box_on_face(eye, face, 9.0, 0.0, 0.25),
                "face {face}: nearer"
            );
            assert!(
                !box_on_face(eye, face, 11.0, 0.0, 0.25),
                "face {face}: farther"
            );
            assert!(
                box_on_face(eye, face, 10.0, 0.0, 1.0),
                "face {face}: straddling"
            );
            assert!(
                box_on_face(eye, face, 9.0, 8.0, 0.25),
                "face {face}: a cube corner"
            );
            let far = face_dir(face, 0.0, 0.0).map(|a| a * 1.0e4);
            let far: [f32; 3] = core::array::from_fn(|i| eye[i] + far[i]);
            let unlimited = face_frustum(eye, face, None);
            assert!(
                unlimited.intersects_aabb(far, far.map(|x| x + 1.0)),
                "face {face}"
            );
        }
    }

    #[test]
    fn face_centers_look_down_their_axis() {
        // The center texel of each face projects to the NDC origin.
        let eye = [0.0, 0.0, 0.0];
        for face in 0..6 {
            let vp = face_view_projection(eye, face);
            let d = face_dir(face, 0.0, 0.0);
            let (nx, ny, w) = project(vp, d);
            assert!(w > 0.0);
            assert!(
                nx.abs() < 1e-5 && ny.abs() < 1e-5,
                "face {face} center off-origin"
            );
        }
    }

    #[test]
    fn the_runtime_prefilter_plan_halves_each_mip_down_to_four_texels() {
        let plan = PrefilterPlan::RUNTIME;
        assert_eq!(plan.face_size(), 512);
        // 512, 256, 128, 64, 32, 16, 8, 4: the convolution stops at 4x4 faces,
        // where a rougher lobe has nothing left to integrate.
        assert_eq!(plan.mips(), 8);
        assert_eq!(plan.mip_face_size(0), 512);
        assert_eq!(plan.mip_face_size(plan.mips() - 1), 4);
        for mip in 1..plan.mips() {
            assert_eq!(plan.mip_face_size(mip), plan.mip_face_size(mip - 1) / 2);
        }
    }

    #[test]
    fn the_prefilter_plan_matches_the_cpu_convolution_it_replaces() {
        // The GPU kernels have to reproduce what the build-time CPU convolution
        // produces for the same capture, so the roughness ramp is the shared one:
        // 0 at the mirror mip, 1 at the roughest.
        let plan = PrefilterPlan::RUNTIME;
        for mip in 1..plan.mips() {
            let p = plan.ggx_params(mip);
            assert_eq!(p.roughness, em::prefilter_roughness(mip, plan.mips()));
            assert_eq!(p.dst_size, plan.mip_face_size(mip));
            // Every GGX dispatch reads the whole pyramid, from mip 0.
            assert_eq!(p.src_size, plan.face_size());
            assert_eq!(p.src_mip_count, plan.mips() as f32);
        }
        assert_eq!(plan.ggx_params(plan.mips() - 1).roughness, 1.0);
        // The mirror mip is a clamped copy, not a convolution.
        assert_eq!(plan.mip0_params().dst_size, plan.face_size());
        assert_eq!(plan.mip0_params().roughness, 0.0);
    }

    #[test]
    fn a_downsample_reads_the_level_above_the_one_it_writes() {
        let plan = PrefilterPlan::RUNTIME;
        for mip in 1..plan.mips() {
            let p = plan.downsample_params(mip);
            assert_eq!(p.src_mip, mip - 1);
            assert_eq!(p.dst_size, plan.mip_face_size(mip));
            // A 2x2 reduction, so the source is exactly twice the destination.
            assert_eq!(plan.mip_face_size(p.src_mip), 2 * p.dst_size);
        }
    }

    // The firefly cap is authored in absolute radiance and applied where the
    // cube stores it.
    #[test]
    fn the_firefly_clamp_follows_the_storage_scale() {
        assert_eq!(
            PrefilterPlan::RUNTIME.mip0_params().clamp_lum,
            12.0 * PROBE_STORAGE_SCALE
        );
        let plain = PrefilterPlan::new(64, 16, 4.0);
        assert_eq!(plain.mip0_params().clamp_lum, 4.0);
        assert_eq!(plain.stored_at(8.0).ggx_params(1).clamp_lum, 32.0);
    }

    #[test]
    fn every_prefilter_dispatch_carries_the_same_firefly_clamp() {
        // The clamp bites in three places -- the mirror copy, each pyramid level,
        // and each GGX sample -- and a level that lost it would leak a firefly's
        // energy back into the coarse taps.
        let plan = PrefilterPlan::RUNTIME;
        let clamp = plan.mip0_params().clamp_lum;
        assert!(clamp > 0.0);
        for mip in 1..plan.mips() {
            assert_eq!(plan.downsample_params(mip).clamp_lum, clamp);
            assert_eq!(plan.ggx_params(mip).clamp_lum, clamp);
        }
    }
}
