// src/directx/uniforms.rs
//
// repr(C) uniform / root-constant structs only the DirectX frame encoders bind
// (cbuffer / root-constant layouts). Each is mirrored field-for-field in an
// `.hlsl` shader under `directx/shaders/`.
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

// The GPU-cull `CullParams` cbuffer (b0, 208 bytes): six already-normalised
// frustum planes, the camera position sharing its row with the object count, the
// previous frame's view-projection, the Hi-Z metadata (dims, mip count, enable
// flag), then the shader-bucket routing. DirectX fuses the cull + Hi-Z uniforms
// into one cbuffer.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct CullParams {
    pub planes: [[f32; 4]; 6],
    pub cam_pos: [f32; 3],
    pub object_count: u32,
    pub prev_view_proj: [[f32; 4]; 4],
    pub hiz_size: [f32; 2],
    pub hiz_mip_count: u32,
    pub hiz_enabled: u32,
    // Shader-bucket command regions in the indirect buffer. The kernel writes
    // every record's slot in all `bucket_count` regions (a draw in the record's
    // own bucket, a no-op everywhere else); region `b` starts at command
    // `b * bucket_stride`. `bucket_count = 1` degenerates to a single region.
    pub bucket_count: u32,
    pub bucket_stride: u32,
    pub _pad: [u32; 2],
}

// The raymarch pass per-frame `RaymarchView` cbuffer (b0, 160 bytes; aligned to
// 256 for the D3D12 cbuffer). Mirrors the Metal `RaymarchView`.
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

// The per-volume `SdfVolumeUniforms` cbuffer (b1, 176 bytes; aligned to 256 in
// the cbuffer allocation).
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

// The RT skinning compute root-constant block (rt_skin.hlsl `SkinParams`): four
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

    // CullParams must match the `CullParams` cbuffer (b0) in cull.hlsl: six
    // frustum planes, cam_pos sharing its row with object_count, the previous
    // view-projection, the Hi-Z metadata, then the bucket routing pair opening
    // a fresh 16-byte row (208 B total).
    #[test]
    fn cull_params_layout_matches_hlsl() {
        assert_eq!(size_of::<CullParams>(), 208);
        assert_eq!(offset_of!(CullParams, planes), 0);
        assert_eq!(offset_of!(CullParams, cam_pos), 96);
        assert_eq!(offset_of!(CullParams, object_count), 108);
        assert_eq!(offset_of!(CullParams, prev_view_proj), 112);
        assert_eq!(offset_of!(CullParams, hiz_size), 176);
        assert_eq!(offset_of!(CullParams, hiz_mip_count), 184);
        assert_eq!(offset_of!(CullParams, hiz_enabled), 188);
        assert_eq!(offset_of!(CullParams, bucket_count), 192);
        assert_eq!(offset_of!(CullParams, bucket_stride), 196);
    }

    // RaymarchView must match the `RaymarchView` cbuffer (b0) in
    // raymarch_helpers.hlsl: two column-major float4x4 then the packed
    // cam_pos/pad/viewport/time/prefilter scalars (160 B total).
    #[test]
    fn raymarch_view_layout_matches_hlsl() {
        assert_eq!(size_of::<RaymarchView>(), 160);
        assert_eq!(offset_of!(RaymarchView, vp), 0);
        assert_eq!(offset_of!(RaymarchView, inv_vp), 64);
        assert_eq!(offset_of!(RaymarchView, cam_pos), 128);
        assert_eq!(offset_of!(RaymarchView, _pad0), 140);
        assert_eq!(offset_of!(RaymarchView, viewport), 144);
        assert_eq!(offset_of!(RaymarchView, time), 152);
        assert_eq!(offset_of!(RaymarchView, prefilter_mip_count), 156);
    }

    // RaymarchVolumeUniforms must match the `SdfVolumeUniforms` cbuffer (b1):
    // centre/pad, extent/pad, the four scalars, then 32 floats of params packed
    // as 8 float4 rows (176 B total).
    #[test]
    fn raymarch_volume_uniforms_layout_matches_hlsl() {
        assert_eq!(size_of::<RaymarchVolumeUniforms>(), 176);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, centre), 0);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, _pad0), 12);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, extent), 16);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, _pad1), 28);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, cone_ratio), 32);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, max_distance), 36);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, max_steps), 40);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, receive_shadows), 44);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, params), 48);
    }

    // HLSL `SkinParams` cbuffer in rt_skin.hlsl: four tightly packed uints
    // (16 bytes).
    #[test]
    fn skin_params_layout_matches_hlsl() {
        assert_eq!(size_of::<SkinParams>(), 16);
        assert_eq!(offset_of!(SkinParams, vertex_base), 0);
        assert_eq!(offset_of!(SkinParams, vertex_count), 4);
        assert_eq!(offset_of!(SkinParams, joint_count), 8);
        assert_eq!(offset_of!(SkinParams, target_count), 12);
    }
}
