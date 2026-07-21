// src/vulkan/uniforms.rs
//
// repr(C) uniform / push-constant structs shared between the Vulkan frame
// encoders and the GLSL shaders (std140 / std430 / push-constant layouts). Each
// struct is mirrored field-for-field in a `.glsl`/`.vert`/`.frag`/`.comp` shader
// and locked by a layout test asserting its `size_of` and every `offset_of!`.
//
// These are GPU-free (plain repr(C) types, no ash/vk), so they live in
// concinnity-render and their layout tests count toward coverage; the Vulkan
// backend re-exports this module under `crate::vulkan::uniforms` and each pass
// file re-exports the struct(s) it fills so their existing paths are unchanged.

use crate::assets::sdf_volume::SDF_PARAMS_LEN;

// The auto-exposure push-constant block: the three luminance-mapping scalars
// then a pad rounding to 16 bytes. Mirrors the HLSL / Metal struct of the same
// name and the block in both auto-exposure compute shaders.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct AutoExposureParams {
    pub lum_log2_min: f32,
    pub lum_log2_range: f32,
    pub lum_to_bin_scale: f32,
    pub _pad: f32,
}

// Byte size of the auto-exposure push-constant range. Pins the struct size to
// what the pipeline layout declares.
pub const AUTO_EXPOSURE_PUSH_BYTES: u32 = 16;

// The composite text push constant (text.vert): the window dimensions then two
// pads rounding the block to 16 bytes.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct TextPush {
    pub win_width: f32,
    pub win_height: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

// The main-pass push constant (std430): the model matrix, roughness/metallic
// with two pads, then tint and emissive vec3s each followed by a pad (112 B
// total). Matches ModelUniforms(64) + MaterialUniforms(48) packed together.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct MainPush {
    pub model: [[f32; 4]; 4],
    pub roughness: f32,
    pub metallic: f32,
    pub _mpad0: f32,
    pub _mpad1: f32,
    pub tint: [f32; 3],
    pub _mpad2: f32,
    pub emissive: [f32; 3],
    pub _mpad3: f32,
}

// The TAA resolve push constant (taa.frag): a single history-valid flag (4 B).
#[derive(Copy, Clone)]
#[repr(C)]
pub struct TaaPush {
    pub history_valid: f32,
}

// The GPU-cull push constant (cull.comp, std430): six already-normalised frustum
// planes (xyz = normal, w = d), then the camera position sharing its 16-byte slot
// with the build-time object count (112 B total).
#[derive(Copy, Clone)]
#[repr(C)]
pub struct CullParams {
    pub planes: [[f32; 4]; 6],
    pub cam_pos: [f32; 3],
    pub object_count: u32,
}

// The main-pass std140 `ViewBlock` UBO: two mat4 (VP + view) then elapsed/pad and
// the camera position as three scalars, the IBL prefilter mip count, and two end
// pads (160 B total). cam_pos is three individual floats to avoid std140 vec3
// alignment bumping subsequent fields.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct ViewUniforms {
    pub vp: [[f32; 4]; 4],
    pub view_mat: [[f32; 4]; 4],
    pub elapsed: f32,
    // 1.0 when an SSR / RT reflection composite owns the sharp specular this frame,
    // so the forward bindless shader fades its glossy-dielectric probe specular to
    // avoid double-counting; 0.0 keeps the full forward reflection (and at probe
    // bakes, where no resolve runs). Repurposes the former offset-132 pad.
    pub reflections_enabled: f32,
    pub cam_x: f32,
    pub cam_y: f32,
    pub cam_z: f32,
    // Number of mip levels in the bound IBL prefilter cubemap. 0 = IBL off.
    pub prefilter_mip_count: f32,
    pub _ep0: f32,
    pub _ep1: f32,
}

// The Hi-Z build push constant (hiz_init.comp / hiz_downsample.comp): four
// tightly-packed uints (16 bytes).
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct HizParams {
    pub dst_width: u32,
    pub dst_height: u32,
    pub src_mip: u32,
    pub sample_count: u32,
}

// Cull-side Hi-Z uniforms (cull.comp, std140, 80 bytes): the previous frame's
// un-jittered view-projection, the Hi-Z mip-0 dimensions, the mip count, and an
// enable flag. Mirrors the Metal / DirectX CullUniforms tail.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct CullHizParams {
    // Previous frame's un-jittered view-projection. Projects each AABB into the
    // depth space the Hi-Z pyramid was reduced from (`M * v`).
    pub prev_view_proj: [[f32; 4]; 4],
    // Hi-Z mip-0 dimensions (in texels).
    pub hiz_size: [f32; 2],
    // How many mip levels live in the bound texture.
    pub hiz_mip_count: u32,
    // 0 skips the Hi-Z test entirely (first frame / after a resize, before a
    // valid pyramid exists).
    pub hiz_enabled: u32,
}

// The G-buffer pre-pass std140 `GbView` UBO (set 0, binding 0): the jittered VP
// rasterises, the un-jittered cur/prev VPs drive the motion vector, the view
// matrix transforms the normal + depth. Four column-major mat4 (256 B).
#[derive(Copy, Clone)]
#[repr(C)]
pub struct GbViewUniforms {
    pub jittered_vp: [[f32; 4]; 4],
    pub cur_vp: [[f32; 4]; 4],
    pub prev_vp: [[f32; 4]; 4],
    pub view_mat: [[f32; 4]; 4],
}

// The G-buffer pre-pass push constant (shared GLSL `PushBlock`): cur_model then
// prev_model (two column-major mat4) then roughness, plus a trailing pad to
// 16-byte alignment. The motion vector reads cur/prev model; the fragment reads
// roughness.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct GbModelPush {
    pub cur_model: [[f32; 4]; 4],
    pub prev_model: [[f32; 4]; 4],
    pub roughness: f32,
    pub _pad: [f32; 3],
}

// Byte size of the G-buffer pre-pass push-constant range (cur_model 64 +
// prev_model 64 + roughness 4 + 12 pad). Pins the struct size.
pub const GBUFFER_PREPASS_PUSH_BYTES: u32 = 144;

// The transparent (glass) per-frame view UBO (glass.{vert,frag}
// `TransparentViewBlock`, std140, 160 bytes). Mirrors the DirectX / Metal
// `TransparentView`.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct TransparentView {
    pub vp: [[f32; 4]; 4],
    pub inv_vp: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub viewport: [f32; 2],
    pub time: f32,
    // Mips in the sky prefilter cube; 0 = no EnvironmentMap bound. The glass
    // reflection keeps the white rim where no probe covers and no env cube exists.
    pub prefilter_mip_count: f32,
}

// The per-panel glass UBO (glass `GlassParamsBlock`, std140, 64 bytes). Vec3
// fields ride in vec4s (.w unused) so the layout is byte-identical regardless of
// std140 packing. Mirrors the DirectX `GlassParamsGpu`.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct GlassParams {
    pub centre: [f32; 4],
    pub normal: [f32; 4],
    pub tint: [f32; 4],
    pub opacity: f32,
    pub refraction_strength: f32,
    pub fresnel_power: f32,
    // 1.0 when this pane was assigned a planar reflection slot (sample the sharp
    // mirror render), 0.0 keeps the probe / sky reflection path.
    pub planar: f32,
}

// One particle slot in the simulation pool (particle_simulate.comp / particle.vert
// `Particle`, std430): a (vec3, float) position/age then a (vec3, float)
// velocity/lifetime, 32 bytes.
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct GpuParticle {
    pub position: [f32; 3],
    pub age: f32,
    pub velocity: [f32; 3],
    pub lifetime: f32,
}

// The particle render pass per-frame view UBO (particle.vert `ParticleView`,
// std140): a mat4 then two (vec3, pad) camera-basis slots, 96 bytes.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct ParticleView {
    pub vp: [[f32; 4]; 4],
    pub cam_right: [f32; 3],
    pub _pad0: f32,
    pub cam_up: [f32; 3],
    pub _pad1: f32,
}

// The raymarch pass per-frame view UBO (raymarch_helpers.glsl
// `RaymarchViewBlock`, std140, 160 bytes). Mirrors the DirectX / Metal
// `RaymarchView`.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct RaymarchView {
    pub vp: [[f32; 4]; 4],
    pub inv_vp: [[f32; 4]; 4],
    pub cam_pos: [f32; 3],
    pub _pad0: f32,
    pub viewport: [f32; 2],
    pub time: f32,
    pub prefilter_mip_count: f32,
}

// The per-volume SDF raymarch UBO (`SdfVolumeBlock`, std140, 176 bytes). Mirrors
// the DirectX `RaymarchVolumeUniforms`.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct RaymarchVolumeUniforms {
    pub centre: [f32; 3],
    pub _pad0: f32,
    pub extent: [f32; 3],
    pub _pad1: f32,
    pub cone_ratio: f32,
    pub max_distance: f32,
    pub max_steps: i32,
    pub receive_shadows: i32,
    pub params: [f32; SDF_PARAMS_LEN],
}

// The RT skinning compute push constant (rt_skin.comp `SkinParams`): four
// tightly-packed uints (16 bytes). `target_count` is the morph-target count in
// the delta buffer (0 = no morphing).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SkinParams {
    pub vertex_base: u32,
    pub vertex_count: u32,
    pub joint_count: u32,
    pub target_count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    // AutoExposureParams must match the `AutoExposureParams` push-constant block
    // in both auto-exposure compute shaders: the three luminance-mapping scalars
    // then a pad rounding to 16 bytes. Pinned by AUTO_EXPOSURE_PUSH_BYTES.
    #[test]
    fn auto_exposure_params_layout_matches_glsl() {
        assert_eq!(size_of::<AutoExposureParams>(), 16);
        assert_eq!(
            size_of::<AutoExposureParams>() as u32,
            AUTO_EXPOSURE_PUSH_BYTES
        );
        assert_eq!(offset_of!(AutoExposureParams, lum_log2_min), 0);
        assert_eq!(offset_of!(AutoExposureParams, lum_log2_range), 4);
        assert_eq!(offset_of!(AutoExposureParams, lum_to_bin_scale), 8);
        assert_eq!(offset_of!(AutoExposureParams, _pad), 12);
    }

    // TextPush must match the `TextPush` push constant in text.vert: the window
    // dimensions then two pads rounding the block to 16 bytes.
    #[test]
    fn text_push_layout_matches_glsl() {
        assert_eq!(size_of::<TextPush>(), 16);
        assert_eq!(offset_of!(TextPush, win_width), 0);
        assert_eq!(offset_of!(TextPush, win_height), 4);
        assert_eq!(offset_of!(TextPush, _pad0), 8);
        assert_eq!(offset_of!(TextPush, _pad1), 12);
    }

    // MainPush must match the `PushBlock` push constant in the main-pass shaders
    // (std430): the model matrix, roughness/metallic with two pads, then tint and
    // emissive vec3s each followed by a pad (112 B total).
    #[test]
    fn main_push_layout_matches_glsl() {
        assert_eq!(size_of::<MainPush>(), 112);
        assert_eq!(offset_of!(MainPush, model), 0);
        assert_eq!(offset_of!(MainPush, roughness), 64);
        assert_eq!(offset_of!(MainPush, metallic), 68);
        assert_eq!(offset_of!(MainPush, _mpad0), 72);
        assert_eq!(offset_of!(MainPush, _mpad1), 76);
        assert_eq!(offset_of!(MainPush, tint), 80);
        assert_eq!(offset_of!(MainPush, _mpad2), 92);
        assert_eq!(offset_of!(MainPush, emissive), 96);
        assert_eq!(offset_of!(MainPush, _mpad3), 108);
    }

    // TaaPush must match the `TaaBlock` push constant in taa.frag: a single
    // history-valid flag (4 bytes).
    #[test]
    fn taa_push_layout_matches_glsl() {
        assert_eq!(size_of::<TaaPush>(), 4);
        assert_eq!(offset_of!(TaaPush, history_valid), 0);
    }

    // CullParams must match the `CullParams` push-constant block in cull.comp
    // (std430): six frustum planes, then cam_pos sharing its 16-byte slot with
    // object_count (112 B total).
    #[test]
    fn cull_params_layout_matches_glsl() {
        assert_eq!(size_of::<CullParams>(), 112);
        assert_eq!(offset_of!(CullParams, planes), 0);
        assert_eq!(offset_of!(CullParams, cam_pos), 96);
        assert_eq!(offset_of!(CullParams, object_count), 108);
    }

    // ViewUniforms must match the std140 `ViewBlock` UBO in the main-pass
    // shaders: two mat4 then elapsed/pad and the camera position as three
    // scalars, prefilter mip count, and two end pads (160 B total).
    #[test]
    fn view_uniforms_layout_matches_glsl() {
        assert_eq!(size_of::<ViewUniforms>(), 160);
        assert_eq!(offset_of!(ViewUniforms, vp), 0);
        assert_eq!(offset_of!(ViewUniforms, view_mat), 64);
        assert_eq!(offset_of!(ViewUniforms, elapsed), 128);
        assert_eq!(offset_of!(ViewUniforms, reflections_enabled), 132);
        assert_eq!(offset_of!(ViewUniforms, cam_x), 136);
        assert_eq!(offset_of!(ViewUniforms, cam_y), 140);
        assert_eq!(offset_of!(ViewUniforms, cam_z), 144);
        assert_eq!(offset_of!(ViewUniforms, prefilter_mip_count), 148);
        assert_eq!(offset_of!(ViewUniforms, _ep0), 152);
        assert_eq!(offset_of!(ViewUniforms, _ep1), 156);
    }

    // GLSL HizParams push block: four tightly packed uints (16 bytes).
    #[test]
    fn hiz_params_layout() {
        assert_eq!(size_of::<HizParams>(), 16);
        assert_eq!(offset_of!(HizParams, dst_width), 0);
        assert_eq!(offset_of!(HizParams, dst_height), 4);
        assert_eq!(offset_of!(HizParams, src_mip), 8);
        assert_eq!(offset_of!(HizParams, sample_count), 12);
    }

    // std140 CullHizParams in cull.comp: mat4 (64) + vec2 (8, 8-aligned) + two
    // uints. Total 80 bytes, tightly packed after the mat4.
    #[test]
    fn cull_hiz_params_layout_matches_glsl() {
        assert_eq!(size_of::<CullHizParams>(), 80);
        assert_eq!(offset_of!(CullHizParams, prev_view_proj), 0);
        assert_eq!(offset_of!(CullHizParams, hiz_size), 64);
        assert_eq!(offset_of!(CullHizParams, hiz_mip_count), 72);
        assert_eq!(offset_of!(CullHizParams, hiz_enabled), 76);
    }

    // GbViewUniforms must match the `GbView` UBO (set 0, binding 0) in every
    // pre-pass VS: four std140 column-major mat4 at offsets 0, 64, 128, 192
    // (256 B total).
    #[test]
    fn gb_view_uniforms_layout_matches_glsl() {
        assert_eq!(size_of::<GbViewUniforms>(), 256);
        assert_eq!(offset_of!(GbViewUniforms, jittered_vp), 0);
        assert_eq!(offset_of!(GbViewUniforms, cur_vp), 64);
        assert_eq!(offset_of!(GbViewUniforms, prev_vp), 128);
        assert_eq!(offset_of!(GbViewUniforms, view_mat), 192);
    }

    // GbModelPush is pushed as the shared `PushBlock`: cur_model then prev_model
    // (two column-major mat4) then roughness at offset 128, plus pad. The total
    // must match the push-constant range size.
    #[test]
    fn gb_model_push_layout_matches_glsl() {
        assert_eq!(size_of::<GbModelPush>(), 144);
        assert_eq!(offset_of!(GbModelPush, cur_model), 0);
        assert_eq!(offset_of!(GbModelPush, prev_model), 64);
        assert_eq!(offset_of!(GbModelPush, roughness), 128);
        assert_eq!(size_of::<GbModelPush>() as u32, GBUFFER_PREPASS_PUSH_BYTES);
    }

    // The GLSL `TransparentViewBlock` std140 layout is 160 bytes.
    #[test]
    fn transparent_view_layout_matches_glsl() {
        assert_eq!(size_of::<TransparentView>(), 160);
        assert_eq!(offset_of!(TransparentView, vp), 0);
        assert_eq!(offset_of!(TransparentView, inv_vp), 64);
        assert_eq!(offset_of!(TransparentView, camera_pos), 128);
        assert_eq!(offset_of!(TransparentView, viewport), 144);
        assert_eq!(offset_of!(TransparentView, time), 152);
        assert_eq!(offset_of!(TransparentView, prefilter_mip_count), 156);
    }

    // The GLSL `GlassParamsBlock` std140 layout is 64 bytes.
    #[test]
    fn glass_params_layout_matches_glsl() {
        assert_eq!(size_of::<GlassParams>(), 64);
        assert_eq!(offset_of!(GlassParams, centre), 0);
        assert_eq!(offset_of!(GlassParams, normal), 16);
        assert_eq!(offset_of!(GlassParams, tint), 32);
        assert_eq!(offset_of!(GlassParams, opacity), 48);
        assert_eq!(offset_of!(GlassParams, refraction_strength), 52);
        assert_eq!(offset_of!(GlassParams, fresnel_power), 56);
        assert_eq!(offset_of!(GlassParams, planar), 60);
    }

    // Mirrors the `Particle` struct in particle_simulate.comp: std430 packs
    // (vec3, float) into a 16-byte block, so the struct is 32 bytes total.
    #[test]
    fn gpu_particle_layout_matches_glsl() {
        assert_eq!(size_of::<GpuParticle>(), 32);
        assert_eq!(offset_of!(GpuParticle, position), 0);
        assert_eq!(offset_of!(GpuParticle, age), 12);
        assert_eq!(offset_of!(GpuParticle, velocity), 16);
        assert_eq!(offset_of!(GpuParticle, lifetime), 28);
    }

    // Mirrors the `ParticleView` uniform block in particle.vert: mat4 (64) +
    // (vec3 + pad) + (vec3 + pad) = 96.
    #[test]
    fn particle_view_layout_matches_glsl() {
        assert_eq!(size_of::<ParticleView>(), 96);
        assert_eq!(offset_of!(ParticleView, vp), 0);
        assert_eq!(offset_of!(ParticleView, cam_right), 64);
        assert_eq!(offset_of!(ParticleView, cam_up), 80);
    }

    // The GLSL `RaymarchViewBlock` std140 layout is 160 bytes.
    #[test]
    fn raymarch_view_layout_matches_glsl() {
        assert_eq!(size_of::<RaymarchView>(), 160);
        assert_eq!(offset_of!(RaymarchView, vp), 0);
        assert_eq!(offset_of!(RaymarchView, inv_vp), 64);
        assert_eq!(offset_of!(RaymarchView, cam_pos), 128);
        assert_eq!(offset_of!(RaymarchView, viewport), 144);
        assert_eq!(offset_of!(RaymarchView, time), 152);
        assert_eq!(offset_of!(RaymarchView, prefilter_mip_count), 156);
    }

    // The GLSL `SdfVolumeBlock` std140 layout is 176 bytes.
    #[test]
    fn sdf_volume_uniforms_layout_matches_glsl() {
        assert_eq!(size_of::<RaymarchVolumeUniforms>(), 176);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, centre), 0);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, extent), 16);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, cone_ratio), 32);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, max_distance), 36);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, max_steps), 40);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, receive_shadows), 44);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, params), 48);
    }

    // GLSL `SkinParams` push-constant block in rt_skin.comp: four tightly packed
    // uints (16 bytes).
    #[test]
    fn skin_params_layout_matches_glsl() {
        assert_eq!(size_of::<SkinParams>(), 16);
        assert_eq!(offset_of!(SkinParams, vertex_base), 0);
        assert_eq!(offset_of!(SkinParams, vertex_count), 4);
        assert_eq!(offset_of!(SkinParams, joint_count), 8);
        assert_eq!(offset_of!(SkinParams, target_count), 12);
    }
}
