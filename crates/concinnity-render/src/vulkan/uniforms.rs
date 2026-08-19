// src/vulkan/uniforms.rs
//
// repr(C) uniform / push-constant structs only the Vulkan frame encoders bind
// (std140 / std430 / push-constant layouts). Each is mirrored field-for-field
// in a `.glsl`/`.vert`/`.frag`/`.comp` shader under `vulkan/shaders/`, or in
// the one `.slang` block Vulkan alone declares.
//
// Blocks whose shader counterpart is a single-source `.slang` declaration are
// declared once for every backend in `crate::uniforms`; what is left here is
// what only this backend binds. Their layouts are checked by `shader_layout` in
// concinnity-device, which reads the expected offsets out of slangc's
// reflection per target. The hand-written asserts below are for the families
// whose shaders are still per backend -- the cull kernel, the skinning and
// morph kernels, the raymarch SDF templates, the legacy per-draw main and
// velocity passes, and Metal's water / glass_mesh_rt.

use crate::assets::sdf_volume::SDF_PARAMS_LEN;

// Byte size of the auto-exposure push-constant range. Pins the struct size to
// what the pipeline layout declares.
pub const AUTO_EXPOSURE_PUSH_BYTES: u32 = 16;

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

// The GPU-cull push constant (cull.comp, std430): six already-normalised frustum
// planes (xyz = normal, w = d), the camera position sharing its 16-byte slot with
// the build-time object count, then the shader-bucket routing (120 B total).
#[derive(Copy, Clone)]
#[repr(C)]
pub struct CullParams {
    pub planes: [[f32; 4]; 6],
    pub cam_pos: [f32; 3],
    pub object_count: u32,
    // Shader-bucket command regions in the indirect buffer. The kernel writes
    // every record's slot in all `bucket_count` regions (a draw in the record's
    // own bucket, a no-op everywhere else); region `b` starts at command
    // `b * bucket_stride`. `bucket_count = 1` degenerates to a single region.
    pub bucket_count: u32,
    pub bucket_stride: u32,
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

    // CullParams must match the `CullParams` push-constant block in cull.comp
    // (std430): six frustum planes, then cam_pos sharing its 16-byte slot with
    // object_count (112 B total).
    #[test]
    fn cull_params_layout_matches_glsl() {
        assert_eq!(size_of::<CullParams>(), 120);
        assert_eq!(offset_of!(CullParams, planes), 0);
        assert_eq!(offset_of!(CullParams, cam_pos), 96);
        assert_eq!(offset_of!(CullParams, object_count), 108);
        assert_eq!(offset_of!(CullParams, bucket_count), 112);
        assert_eq!(offset_of!(CullParams, bucket_stride), 116);
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
