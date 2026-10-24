//! The passes that draw their own geometry into the main target: projected
//! decals, world-space lines, and the GPU particle system.

use super::daylight::DaylightBlock;
use crate::gfx::render_types::LightUniforms;
use crate::render::light::lights::glint_sun;

/// Per-frame view inputs to the projected-decal pass. Matches `DecalView` in
/// `shaders/decal.hlsl`. 144 bytes.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct DecalView {
    /// View-projection matrix used by the main pass (jittered when TAA is on).
    pub vp: [[f32; 4]; 4],
    /// Inverse of `vp`. The fragment shader uses it to reconstruct world space
    /// from the depth attachment at each pixel.
    pub inv_vp: [[f32; 4]; 4],
    /// HDR target dimensions in pixels: drives the screen->NDC conversion.
    pub viewport: [f32; 2],
    /// The scale the HDR chain is written at; the stamped color is multiplied
    /// by it.
    pub pre_exposure: f32,
    /// Padding so the field layout matches the shader-side struct.
    pub _pad: f32,
}

/// Per-decal uniforms uploaded before each draw. Matches `DecalParams` in
/// `shaders/decal.hlsl`. 160 bytes (two float4x4s, a float4 tint, four scalars).
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct DecalParams {
    /// Model matrix, column-major.
    pub model: [[f32; 4]; 4],
    /// Inverse model matrix, column-major.
    pub inv_model: [[f32; 4]; 4],
    /// Linear RGBA tint multiplied into the sampled image.
    pub tint: [f32; 4],
    /// Exponent of the edge fade curve.
    pub fade_pow: f32,
    /// Padding so the field layout matches the shader-side struct.
    pub _pad0: f32,
    /// Padding so the field layout matches the shader-side struct.
    pub _pad1: f32,
    /// Padding so the field layout matches the shader-side struct.
    pub _pad2: f32,
}

/// Per-frame view inputs to the line pass. Matches `LineView` in
/// `shaders/line.hlsl`. 80 bytes.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct LineView {
    /// View-projection matrix used by the main pass (jittered when TAA is on),
    /// so a line lands on the same pixel the geometry it sits on did.
    pub vp: [[f32; 4]; 4],
    /// Alpha multiplier applied where a line falls behind scene geometry.
    pub occluded_alpha: f32,
    /// The scale the HDR chain is written at; the line color is multiplied by
    /// it.
    pub pre_exposure: f32,
    /// Padding so the field layout matches the shader-side struct.
    pub _pad: [f32; 2],
}

/// Per-frame view inputs to the particle render pass. Matches `ParticleView` in
/// `shaders/particle.hlsl`. 144 bytes.
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::NoUninit)]
#[repr(C)]
pub struct ParticleView {
    /// View-projection matrix used by the main pass.
    pub vp: [[f32; 4]; 4],
    /// World-space camera right vector: drives the first billboard axis. The
    /// shader reads it as the `xyz` of a float4 whose `w` is `pre_exposure`.
    pub cam_right: [f32; 3],
    /// The scale the HDR chain is written at; the sprite color is multiplied
    /// by it.
    pub pre_exposure: f32,
    /// World-space camera up vector: drives the second billboard axis.
    pub cam_up: [f32; 3],
    /// Padding so the field layout matches the shader-side struct.
    pub _pad1: f32,
    /// `[x, y, z, _]`: unit direction toward the sun, zero without one.
    pub sun_dir: [f32; 4],
    /// `[r, g, b, _]`: the light a white surface facing the sun reflects
    /// where the camera is, which lit particles take.
    pub sun_light: [f32; 4],
    /// `[r, g, b, _]`: the light a white surface under the open sky reflects
    /// from it where the camera is.
    pub sky_light: [f32; 4],
}

/// The light a white surface reflects from the sky where there is no
/// planet's sky to read: the flat ambient fill the water takes too.
const PARTICLE_FLAT_SKY: f32 = 0.03;

impl ParticleView {
    /// The view block for camera `view` (column-major) seen through `vp`,
    /// written at `pre_exposure`, lit by the sun `lights` glints with and,
    /// on a planet, the sky its `daylight` curves give where the camera is.
    pub fn new(
        vp: [[f32; 4]; 4],
        view: &[[f32; 4]; 4],
        pre_exposure: f32,
        lights: &LightUniforms,
        daylight: &DaylightBlock,
    ) -> Self {
        let (sun_dir, sun_color) = glint_sun(lights);
        let inv_pi = core::f32::consts::FRAC_1_PI;
        let ambient = lights.ambient_intensity;
        let sky = if daylight.is_on() {
            daylight.camera_irradiance().map(|e| e * inv_pi * ambient)
        } else {
            [PARTICLE_FLAT_SKY * ambient; 3]
        };
        Self {
            vp,
            cam_right: [view[0][0], view[1][0], view[2][0]],
            pre_exposure,
            cam_up: [view[0][1], view[1][1], view[2][1]],
            _pad1: 0.0,
            sun_dir,
            sun_light: [
                sun_color[0] * inv_pi,
                sun_color[1] * inv_pi,
                sun_color[2] * inv_pi,
                0.0,
            ],
            sky_light: [sky[0], sky[1], sky[2], 0.0],
        }
    }
}

/// One particle slot in the pool the simulation kernel writes and the render
/// pair reads. Matches `Particle` in `shaders/particle_types.hlsl`, which
/// spells the same 48 bytes as three float4 lanes.
#[derive(Copy, Clone, Default)]
#[repr(C)]
pub struct GpuParticle {
    /// World-space position.
    pub position: [f32; 3],
    /// Seconds since the particle spawned.
    pub age: f32,
    /// World-space velocity in units per second.
    pub velocity: [f32; 3],
    /// Total lifetime in seconds.
    pub lifetime: f32,
    /// The up of the plane the particle dies falling back through, or zero
    /// for none.
    pub floor_up: [f32; 3],
    /// The plane's height along `floor_up`.
    pub floor_height: f32,
}

/// Per-dispatch parameters for the `rt_skin` compute kernel, which deforms one
/// skinned object's bind-pose vertices into the shared deformed buffer the RT
/// acceleration structure traces. Matches `SkinParams` in `rt_skin.hlsl`.
///
/// Metal writes it to buffer(3), Vulkan pushes it, DirectX takes it as root
/// constants at b0.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit)]
pub struct SkinParams {
    /// First vertex of this object's region in the shared vertex buffer.
    pub vertex_base: u32,
    /// Vertices this dispatch deforms.
    pub vertex_count: u32,
    /// Palette size; joint indices are clamped below it.
    pub joint_count: u32,
    /// Morph targets in the delta buffer; zero when the object has none.
    pub target_count: u32,
}

/// Per-dispatch parameters for the `model_history` compute kernel, which
/// snapshots this frame's model matrices out of the bindless object buffer into
/// the frame's model-history ring slot. Matches `ModelHistoryParams` in
/// `model_history.hlsl`.
///
/// Every backend binds it at the kernel's first slot: Metal at buffer(0),
/// Vulkan as set 0 binding 0, DirectX as root constants at b0.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit)]
pub struct ModelHistoryParams {
    /// Cull records to snapshot: the object buffer's whole length.
    pub record_count: u32,
    /// Pads the block to the 16-byte floor a constant buffer binds at.
    pub _pad: [u32; 3],
}

#[cfg(test)]
mod view_layout_tests {
    use super::*;
    use core::mem::{offset_of, size_of};

    // Two float4x4s, then the viewport and the pre-exposure in one lane.
    #[test]
    fn decal_view_layout_matches_the_shader() {
        assert_eq!(size_of::<DecalView>(), 144);
        assert_eq!(offset_of!(DecalView, vp), 0);
        assert_eq!(offset_of!(DecalView, inv_vp), 64);
        assert_eq!(offset_of!(DecalView, viewport), 128);
        assert_eq!(offset_of!(DecalView, pre_exposure), 136);
        assert_eq!(offset_of!(DecalView, _pad), 140);
    }

    #[test]
    fn line_view_layout_matches_the_shader() {
        assert_eq!(size_of::<LineView>(), 80);
        assert_eq!(offset_of!(LineView, vp), 0);
        assert_eq!(offset_of!(LineView, occluded_alpha), 64);
        assert_eq!(offset_of!(LineView, pre_exposure), 68);
        assert_eq!(offset_of!(LineView, _pad), 72);
    }

    // The two axes are float4 lanes; the pre-exposure is the first one's `w`.
    #[test]
    fn particle_view_layout_matches_the_shader() {
        assert_eq!(size_of::<ParticleView>(), 144);
        assert_eq!(offset_of!(ParticleView, vp), 0);
        assert_eq!(offset_of!(ParticleView, cam_right), 64);
        assert_eq!(offset_of!(ParticleView, pre_exposure), 76);
        assert_eq!(offset_of!(ParticleView, cam_up), 80);
        assert_eq!(offset_of!(ParticleView, _pad1), 92);
        assert_eq!(offset_of!(ParticleView, sun_dir), 96);
        assert_eq!(offset_of!(ParticleView, sun_light), 112);
        assert_eq!(offset_of!(ParticleView, sky_light), 128);
    }

    #[test]
    fn gpu_particle_layout_matches_the_shader() {
        assert_eq!(size_of::<GpuParticle>(), 48);
        assert_eq!(offset_of!(GpuParticle, position), 0);
        assert_eq!(offset_of!(GpuParticle, age), 12);
        assert_eq!(offset_of!(GpuParticle, velocity), 16);
        assert_eq!(offset_of!(GpuParticle, lifetime), 28);
        assert_eq!(offset_of!(GpuParticle, floor_up), 32);
        assert_eq!(offset_of!(GpuParticle, floor_height), 44);
    }

    // The billboard axes are the view's first two rows; the sun is the first
    // directional light over pi, and the sky the planet's where the camera
    // is, else the flat fill, both scaled by the ambient intensity.
    #[test]
    fn the_particle_view_takes_the_camera_axes_and_its_light() {
        let mut view = [[0.0; 4]; 4];
        view[0] = [1.0, 2.0, 3.0, 0.0];
        view[1] = [4.0, 5.0, 6.0, 0.0];
        view[2] = [7.0, 8.0, 9.0, 0.0];
        let mut lights = LightUniforms::DEFAULT;
        lights.ambient_intensity = 2.0;
        lights.num_directional = 0;
        let unlit = ParticleView::new([[0.0; 4]; 4], &view, 0.5, &lights, &DaylightBlock::OFF);
        assert_eq!(unlit.cam_right, [1.0, 4.0, 7.0]);
        assert_eq!(unlit.cam_up, [2.0, 5.0, 8.0]);
        assert_eq!(unlit.pre_exposure, 0.5);
        assert_eq!(unlit.sun_light, [0.0; 4]);
        assert_eq!(unlit.sky_light, [0.06, 0.06, 0.06, 0.0]);
        lights.num_directional = 1;
        lights.directional[0] = crate::gfx::render_types::DirectionalLightData {
            direction: [0.0, 2.0, 0.0],
            intensity: core::f32::consts::PI,
            color: [1.0, 0.5, 0.25],
            _pad: 0.0,
        };
        let lit = ParticleView::new([[0.0; 4]; 4], &view, 1.0, &lights, &DaylightBlock::OFF);
        assert_eq!(lit.sun_dir, [0.0, 1.0, 0.0, 0.0]);
        for (got, want) in lit.sun_light.iter().zip([1.0, 0.5, 0.25, 0.0]) {
            assert!((got - want).abs() < 1e-6, "{:?}", lit.sun_light);
        }
    }
}

#[cfg(test)]
mod model_history_tests {
    use super::*;
    use core::mem::{offset_of, size_of};

    // One uint padded to the 16-byte constant-buffer floor.
    #[test]
    fn model_history_params_layout_matches_the_shader() {
        assert_eq!(size_of::<ModelHistoryParams>(), 16);
        assert_eq!(offset_of!(ModelHistoryParams, record_count), 0);
        assert_eq!(offset_of!(ModelHistoryParams, _pad), 4);
    }
}
