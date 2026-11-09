//! The camera blocks: what the forward pass and the G-buffer pre-pass need to
//! place a fragment, and the per-draw model pair the pre-pass differentiates for
//! motion vectors.

use super::daylight::{DaylightBlock, VIEW_BLOCK_LIMIT};
use crate::gfx::render_types::TileMorphView;

/// The camera a screen-space pass reconstructs world positions from, as the
/// per-frame view blocks of the raymarch and transparent passes take it.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PassCamera {
    /// The view-projection the pass's depth was rasterized with, column-major.
    pub vp: [[f32; 4]; 4],
    /// Inverse of `vp`.
    pub inv_vp: [[f32; 4]; 4],
    /// World-space camera position.
    pub cam_pos: [f32; 3],
    /// Render-target size in pixels.
    pub viewport: [f32; 2],
    /// Seconds since the world started.
    pub time: f32,
    /// Mip count of the bound IBL prefilter cube; 0 when none is bound.
    pub prefilter_mip_count: f32,
    /// 1 when the irradiance cube holds diffuse light, from an
    /// `EnvironmentMap` or an `Atmosphere`; 0 otherwise.
    pub irradiance: f32,
    /// Rows of the rotation taking a world direction into the environment
    /// cubemaps' baked frame.
    pub sky_rot: [[f32; 4]; 3],
    /// The scale the frame's HDR chain is written at
    /// ([`crate::gfx::auto_exposure::PreExposure`]).
    pub pre_exposure: f32,
}

/// Per-frame view-projection uniforms, uploaded once per frame and shared across
/// every draw in it. `view` is the standalone view matrix the vertex shader uses
/// to compute view-space depth for cascade selection in the fragment shader.
///
/// The shader source declares the same bytes with its own spelling
/// (`view_mat`, and `cam_x`/`cam_y`/`cam_z` in place of
/// `cam_pos`), which is why the two are checked as byte ranges rather than by
/// name.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct ViewUniforms {
    /// Combined view-projection matrix (column-major).
    pub vp: [[f32; 4]; 4],
    /// Camera view matrix (column-major). Used to compute view-space depth
    /// in the vertex shader for shadow cascade selection.
    pub view: [[f32; 4]; 4],
    /// Elapsed seconds, available to shaders for animation.
    pub elapsed: f32,
    /// 1.0 when a screen-space / ray-traced reflection resolve composites this
    /// frame, else 0.0. The forward fragment shader uses it to yield the sharp
    /// specular for glossy surfaces to that resolve (whose miss-fallback samples
    /// the same probe set), so a glossy surface does not show both the
    /// parallax-approximate forward probe reflection and the exact resolved one.
    pub reflections_enabled: f32,
    /// World-space camera position (`packed_float3` in the MSL contract, alignment 4).
    pub cam_pos: [f32; 3],
    /// Number of mip levels in the bound IBL prefilter cubemap. 0 means
    /// "no `EnvironmentMap` bound": the fragment shader uses this as the IBL
    /// enable flag and falls back to a flat ambient placeholder.
    pub prefilter_mip_count: f32,
    /// 1.0 while the unlit view mode is active: `shade_surface` returns the base
    /// color before lighting. Occupies what was pad space, so the offsets in
    /// the user-shader binding contract are unchanged.
    pub shade_mode: f32,
    /// 1.0 when the screen-space ambient occlusion describes this view, so the
    /// indirect term samples it; 0.0 for a probe or mirror face, which renders
    /// another viewpoint than the one the occlusion was computed for. Occupies
    /// what was pad space, so the offsets in the user-shader binding contract
    /// are unchanged.
    pub ambient_occlusion: f32,
    /// Rows of the rotation that takes a world-space direction into the
    /// environment cubemaps' baked frame: the inverse of the sky's current
    /// rotation (`SkyOrientation::sample_rows`), identity when the world
    /// declares no `SkyRotation`. Every cubemap sample helper applies it, so
    /// the sky, the ambient fill and the glossy reflections turn together.
    /// One `float4` per row; row 0's `w` is the environment prefilter cube's
    /// storage scale as a base-2 log (`radiance_storage::with_sky_cube_scale`).
    pub sky_rot: [[f32; 4]; 3],
    /// 1.0 when the bound irradiance cube holds the environment's diffuse
    /// light (an environment map's, or an atmosphere's), so surfaces take
    /// their ambient light from it; 0.0 falls back to a flat ambient.
    pub irradiance: f32,
    /// The scale the pass's color target is written at
    /// ([`crate::gfx::auto_exposure::PreExposure`]): the frame's pre-exposure
    /// for the main pass, the probes' storage scale for a probe face
    /// ([`crate::render::probe::radiance::PROBE_STORAGE_SCALE`]), the frame's
    /// pre-exposure again for a planar mirror
    /// ([`crate::render::probe::radiance::FaceStorage`]). Occupies what was pad space, so the
    /// offsets in the user-shader binding contract are unchanged.
    pub pre_exposure: f32,
    /// 1.0 when the screen-space indirect target the main pass samples carries
    /// the bounce of nearby surfaces beside the occlusion (screen-space GI is
    /// on); 0.0 when it holds the occlusion alone, and for a probe or mirror
    /// face. Occupies what was pad space, so the offsets in the user-shader
    /// binding contract are unchanged.
    pub screen_bounce: f32,
    /// What the view's ground is shaded as in place of what the view does
    /// not draw ([`StandIns`]), as a lane: 0 for every view that draws the
    /// whole scene. Occupies what was pad space, so the offsets in the
    /// user-shader binding contract are unchanged.
    pub stand_ins: f32,
    /// The planet's per-point sky, [`DaylightBlock::OFF`] for a world
    /// without one.
    pub daylight: DaylightBlock,
    /// The camera a planet's tiles morph by: the main camera's in every view,
    /// so a reflection shows the tiles as the screen does.
    pub tile_morph: TileMorphView,
    /// 1.0 for a probe or mirror face: the view is a reflection, so its sky
    /// leaves out the sun's and the moon's discs, which those lights'
    /// specular terms draw. 0.0 for the main view.
    pub reflection: f32,
    /// Pads the block to a whole float4 row.
    pub _pad: [f32; 3],
}

const _: () = assert!(size_of::<ViewUniforms>() <= VIEW_BLOCK_LIMIT);

/// The camera of a view rendered from another viewpoint than the main camera:
/// a reflection-probe face or a planar mirror.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FaceCamera {
    /// View-projection of the face, column-major.
    pub vp: [[f32; 4]; 4],
    /// View matrix of the face, column-major.
    pub view: [[f32; 4]; 4],
    /// World-space eye position.
    pub cam_pos: [f32; 3],
    /// Elapsed seconds the face's shaders animate at.
    pub elapsed: f32,
    /// Mip count of the bound IBL prefilter cube; 0 when none is bound.
    pub prefilter_mip_count: f32,
    /// Rows of the rotation taking a world direction into the environment
    /// cubemaps' baked frame.
    pub sky_rot: [[f32; 4]; 3],
    /// 1 when the irradiance cube holds diffuse light, else 0.
    pub irradiance: f32,
    /// The scale the face's color target is written at.
    pub pre_exposure: f32,
    /// The planet's per-point sky the face is lit by.
    pub daylight: DaylightBlock,
    /// The camera a planet's tiles morph by.
    pub tile_morph: TileMorphView,
}

impl ViewUniforms {
    /// The block for a probe face or a planar mirror. Such a view is always
    /// lit whatever the viewport shows, and no reflection resolve, occlusion
    /// or screen-space bounce describes it, so the forward probe specular is
    /// its only reflection source.
    pub fn face(face: FaceCamera) -> Self {
        Self {
            vp: face.vp,
            view: face.view,
            elapsed: face.elapsed,
            reflections_enabled: 0.0,
            cam_pos: face.cam_pos,
            prefilter_mip_count: face.prefilter_mip_count,
            shade_mode: 0.0,
            ambient_occlusion: 0.0,
            sky_rot: face.sky_rot,
            irradiance: face.irradiance,
            pre_exposure: face.pre_exposure,
            screen_bounce: 0.0,
            stand_ins: 0.0,
            daylight: face.daylight,
            tile_morph: face.tile_morph,
            reflection: 1.0,
            _pad: [0.0; 3],
        }
    }
}

/// What a view that leaves out part of the scene shades its ground as instead,
/// so a capture without the grass blades or the sea's surface still shows a
/// meadow and the sea.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum StandIns {
    /// The view draws the whole scene.
    None,
    /// Grassy ground is shaded as the far canopy all the way in.
    Canopy,
    /// As `Canopy`, and the ground under a planet's sea is shaded as the
    /// sea's calm surface over it.
    CanopyAndSea,
}

impl StandIns {
    /// The lane the shaders read.
    pub fn lane(self) -> f32 {
        match self {
            Self::None => 0.0,
            Self::Canopy => 1.0,
            Self::CanopyAndSea => 2.0,
        }
    }
}

impl ViewUniforms {
    /// This block with its ground standing in for `stand_ins`.
    #[must_use]
    pub fn with_stand_ins(self, stand_ins: StandIns) -> Self {
        Self {
            stand_ins: stand_ins.lane(),
            ..self
        }
    }
}

/// Per-frame view inputs to the unified G-buffer pre-pass. The jittered current
/// VP drives the sky's rasterized position (the surfaces take theirs from the
/// main pass's view block); `view` takes the normal into view space (where
/// SSR / SSAO / SSGI / RT work); the un-jittered cur/prev VPs derive a
/// jitter-free motion vector, `prev_model_carry` moves the previous frame's
/// model transforms into this frame, and `prev_elapsed` / `prev_cam_pos`
/// rewind a vertex hook driven by the clock or the camera to where the
/// previous frame drew it. Matches `GbView` in `shaders/gbuffer_common.hlsl`.
/// 352 bytes: five float4x4 and two float4 rows.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct GBufferView {
    /// View-projection with this frame's TAA jitter applied.
    pub jittered_vp: [[f32; 4]; 4],
    /// This frame's unjittered view-projection.
    pub cur_vp: [[f32; 4]; 4],
    /// The previous frame's unjittered view-projection.
    pub prev_vp: [[f32; 4]; 4],
    /// View matrix, column-major.
    pub view: [[f32; 4]; 4],
    /// The move that carries the model transforms the previous frame
    /// recorded into this frame: identity unless the world was rebased
    /// between them.
    pub prev_model_carry: [[f32; 4]; 4],
    /// The main pass's `elapsed` as the previous frame saw it.
    pub prev_elapsed: f32,
    /// The camera position as the previous frame saw it.
    pub prev_cam_pos: [f32; 3],
    /// 1 when a consumer reads motion this frame, else 0: the previous frame is
    /// then this one, and the surfaces skip reprojecting.
    pub motion: u32,
    /// Pads the block to a whole float4 row.
    pub _pad: [u32; 3],
}

impl GBufferView {
    /// The block for a frame rasterized through `jittered_vp` and viewed
    /// through `view`, whose motion reprojects from `cur` to `prev`, the
    /// previous frame's model transforms moved into this frame by
    /// `prev_model_carry`. Without `motion` the previous frame is `cur`
    /// itself, so every motion vector is zero and the surfaces skip
    /// reprojecting.
    pub fn new(
        jittered_vp: [[f32; 4]; 4],
        view: [[f32; 4]; 4],
        cur: crate::render::temporal::view_history::ViewFrame,
        prev: crate::render::temporal::view_history::ViewFrame,
        prev_model_carry: [[f32; 4]; 4],
        motion: bool,
    ) -> Self {
        let (prev, prev_model_carry) = if motion {
            (prev, prev_model_carry)
        } else {
            (cur, crate::transform::IDENTITY)
        };
        Self {
            jittered_vp,
            cur_vp: cur.vp,
            prev_vp: prev.vp,
            view,
            prev_model_carry,
            prev_elapsed: prev.elapsed,
            prev_cam_pos: prev.cam_pos,
            motion: u32::from(motion),
            _pad: [0; 3],
        }
    }
}

// A camera whose every lane holds a distinct value, for the view-block tests.
#[cfg(test)]
pub(super) fn test_camera() -> PassCamera {
    let m = |k: f32| [[k; 4]; 4];
    PassCamera {
        vp: m(1.0),
        inv_vp: m(2.0),
        cam_pos: [3.0, 4.0, 5.0],
        viewport: [640.0, 480.0],
        time: 6.5,
        prefilter_mip_count: 7.0,
        irradiance: 1.0,
        sky_rot: [[8.0; 4]; 3],
        pre_exposure: 9.5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};

    // A face carries its own camera and lighting inputs and none of the main
    // view's screen-space terms.
    #[test]
    fn a_face_view_is_lit_and_has_no_screen_space_terms() {
        let u = ViewUniforms::face(FaceCamera {
            vp: [[2.0; 4]; 4],
            view: [[3.0; 4]; 4],
            cam_pos: [1.0, 2.0, 3.0],
            elapsed: 4.0,
            prefilter_mip_count: 5.0,
            sky_rot: [[6.0; 4]; 3],
            irradiance: 1.0,
            pre_exposure: 0.5,
            daylight: DaylightBlock::OFF,
            tile_morph: TileMorphView {
                level0_range: 7.0,
                ..TileMorphView::default()
            },
        });
        assert_eq!(u.tile_morph.level0_range, 7.0);
        assert_eq!(
            (u.vp, u.view, u.cam_pos),
            ([[2.0; 4]; 4], [[3.0; 4]; 4], [1.0, 2.0, 3.0])
        );
        assert_eq!(
            (u.elapsed, u.prefilter_mip_count, u.sky_rot),
            (4.0, 5.0, [[6.0; 4]; 3])
        );
        assert_eq!((u.irradiance, u.pre_exposure), (1.0, 0.5));
        assert_eq!(u.reflection, 1.0, "a face is a reflection");
        assert_eq!(
            (
                u.reflections_enabled,
                u.shade_mode,
                u.ambient_occlusion,
                u.screen_bounce
            ),
            (0.0, 0.0, 0.0, 0.0)
        );
    }

    // Every offset here is a byte position the main pass reads through.
    // `sky_rot` lands past the block's own end padding, so nothing before it
    // moved.
    #[test]
    fn view_uniforms_layout_matches_msl() {
        assert_eq!(size_of::<ViewUniforms>(), 224 + 1664 + 32 + 16);
        assert_eq!(offset_of!(ViewUniforms, vp), 0);
        assert_eq!(offset_of!(ViewUniforms, view), 64);
        assert_eq!(offset_of!(ViewUniforms, elapsed), 128);
        assert_eq!(offset_of!(ViewUniforms, reflections_enabled), 132);
        assert_eq!(offset_of!(ViewUniforms, cam_pos), 136);
        assert_eq!(offset_of!(ViewUniforms, prefilter_mip_count), 148);
        assert_eq!(offset_of!(ViewUniforms, shade_mode), 152);
        assert_eq!(offset_of!(ViewUniforms, ambient_occlusion), 156);
        // A float4 array is 16-byte aligned on every target.
        assert_eq!(offset_of!(ViewUniforms, sky_rot), 160);
        assert_eq!(offset_of!(ViewUniforms, irradiance), 208);
        assert_eq!(offset_of!(ViewUniforms, pre_exposure), 212);
        assert_eq!(offset_of!(ViewUniforms, screen_bounce), 216);
        assert_eq!(offset_of!(ViewUniforms, stand_ins), 220);
        assert_eq!(offset_of!(ViewUniforms, daylight), 224);
        assert_eq!(offset_of!(ViewUniforms, tile_morph), 224 + 1664);
        assert_eq!(offset_of!(ViewUniforms, reflection), 224 + 1664 + 32);
        assert_eq!(offset_of!(ViewUniforms, _pad), 224 + 1664 + 36);
        assert_eq!(size_of::<ViewUniforms>() % 16, 0);
    }

    // Five float4x4s, then the previous clock and camera position, then the
    // motion flag padded to a whole float4 row.
    #[test]
    fn gbuffer_view_layout_matches_the_shader() {
        assert_eq!(size_of::<GBufferView>(), 352);
        assert_eq!(offset_of!(GBufferView, jittered_vp), 0);
        assert_eq!(offset_of!(GBufferView, cur_vp), 64);
        assert_eq!(offset_of!(GBufferView, prev_vp), 128);
        assert_eq!(offset_of!(GBufferView, view), 192);
        assert_eq!(offset_of!(GBufferView, prev_model_carry), 256);
        assert_eq!(offset_of!(GBufferView, prev_elapsed), 320);
        assert_eq!(offset_of!(GBufferView, prev_cam_pos), 324);
        assert_eq!(offset_of!(GBufferView, motion), 336);
        assert_eq!(offset_of!(GBufferView, _pad), 340);
    }

    #[test]
    #[expect(
        clippy::used_underscore_binding,
        reason = "the test checks the padding stays zero"
    )]
    fn gbuffer_view_takes_the_previous_frame_from_its_history() {
        use crate::render::temporal::view_history::ViewFrame;
        let m = |k: f32| [[k; 4]; 4];
        let cur = ViewFrame {
            vp: m(2.0),
            elapsed: 1.5,
            cam_pos: [1.0, 2.0, 3.0],
        };
        let prev = ViewFrame {
            vp: m(3.0),
            elapsed: 1.25,
            cam_pos: [0.5, 2.0, 3.0],
        };
        let block = GBufferView::new(m(1.0), m(4.0), cur, prev, m(5.0), true);
        assert_eq!(block.jittered_vp, m(1.0));
        assert_eq!(block.cur_vp, m(2.0));
        assert_eq!(block.prev_vp, m(3.0));
        assert_eq!(block.view, m(4.0));
        assert_eq!(block.prev_model_carry, m(5.0));
        assert_eq!(block.prev_elapsed, 1.25);
        assert_eq!(block.prev_cam_pos, [0.5, 2.0, 3.0]);
        assert_eq!(block.motion, 1);
        assert_eq!(block._pad, [0; 3]);
    }

    // With nothing reading motion the previous frame collapses onto this one.
    #[test]
    fn gbuffer_view_without_motion_reprojects_onto_itself() {
        use crate::render::temporal::view_history::ViewFrame;
        let m = |k: f32| [[k; 4]; 4];
        let cur = ViewFrame {
            vp: m(2.0),
            elapsed: 1.5,
            cam_pos: [1.0, 2.0, 3.0],
        };
        let prev = ViewFrame {
            vp: m(3.0),
            elapsed: 1.25,
            cam_pos: [0.5, 2.0, 3.0],
        };
        let block = GBufferView::new(m(1.0), m(4.0), cur, prev, m(5.0), false);
        assert_eq!(block.prev_vp, block.cur_vp);
        assert_eq!(block.prev_model_carry, crate::transform::IDENTITY);
        assert_eq!(block.prev_elapsed, 1.5);
        assert_eq!(block.prev_cam_pos, [1.0, 2.0, 3.0]);
        assert_eq!(block.motion, 0);
    }
}
