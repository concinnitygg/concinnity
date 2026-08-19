// src/directx/uniforms.rs
//
// repr(C) uniform / root-constant structs shared between the DirectX frame
// encoders and the HLSL shaders (cbuffer / root-constant layouts). Each struct
// is mirrored field-for-field in a shader declaration -- an `.hlsl` under
// `directx/shaders/`, or a single-source `.slang` under `shaders/` -- and
// locked by a layout test asserting its `size_of` and every `offset_of!`.
//
// These are GPU-free (plain repr(C) types, no D3D12), so they live in
// concinnity-render and their layout tests count toward coverage; the DirectX
// backend re-exports this module under `crate::directx::uniforms` and each pass
// file re-exports the struct(s) it fills so their existing paths are unchanged.

use crate::assets::sdf_volume::SDF_PARAMS_LEN;

// The main-pass `ViewBlock` cbuffer (b1, 160 bytes): two column-major float4x4
// (VP + view) then elapsed/pad and the camera position as three scalars, the IBL
// prefilter mip count, and two end pads. cam_pos is three individual floats to
// avoid HLSL packing surprises.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct ViewUniforms {
    pub vp: [[f32; 4]; 4],
    pub view_mat: [[f32; 4]; 4],
    pub elapsed: f32,
    // 1.0 when a reflection resolve (SSR resolve or RT reflections) composites
    // over this frame's HDR scene, so the forward probe specular yields to it
    // below the reflection roughness cut; 0.0 keeps the full forward specular.
    pub reflections_enabled: f32,
    pub cam_x: f32,
    pub cam_y: f32,
    pub cam_z: f32,
    // Number of mip levels in the bound IBL prefilter cubemap. 0 = IBL off.
    pub prefilter_mip_count: f32,
    // 1.0 while the unlit view mode is active: the main fragment stage returns
    // the surface base color before lighting. Repurposed pad space, so the
    // struct size is unchanged.
    pub shade_mode: f32,
    pub _ep1: f32,
}

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

// The decal pass per-frame `DecalView` cbuffer (b0, 144 bytes): two column-major
// float4x4 (VP + inverse VP) then the viewport and a 2-float pad.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct DecalView {
    pub vp: [[f32; 4]; 4],
    pub inv_vp: [[f32; 4]; 4],
    pub viewport: [f32; 2],
    pub _pad: [f32; 2],
}

// The per-decal `DecalParams` cbuffer (b1, 160 bytes): the model + inverse model,
// the float4 tint, the fade power, then three end pads.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct DecalParams {
    pub model: [[f32; 4]; 4],
    pub inv_model: [[f32; 4]; 4],
    pub tint: [f32; 4],
    pub fade_pow: f32,
    pub _p0: f32,
    pub _p1: f32,
    pub _p2: f32,
}

// The line pass per-frame `LineView` cbuffer (b0, 80 bytes): the column-major
// float4x4 view-projection then the occluded-alpha multiplier and three pads.
// Mirrors `metal::uniforms::LineView`.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct LineView {
    pub vp: [[f32; 4]; 4],
    pub occluded_alpha: f32,
    pub _pad: [f32; 3],
}

// The transparent (glass) per-frame `TransparentView` cbuffer (160 bytes).
// Mirrors `metal::uniforms::TransparentView`.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct TransparentViewGpu {
    pub vp: [[f32; 4]; 4],
    pub inv_vp: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub viewport: [f32; 2],
    pub time: f32,
    // Mips in the sky prefilter cube; 0 = no EnvironmentMap bound. A per-frame
    // "has env" gate for the glass reflection fallback (DX keeps it here rather
    // than in the static per-panel GlassParams CBV).
    pub prefilter_mip_count: f32,
}

// The per-panel `GlassParams` cbuffer (64 bytes). Vec3 fields ride in float4s
// (.w unused) so the layout is byte-identical regardless of HLSL packing.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct GlassParamsGpu {
    pub centre: [f32; 4],
    pub normal: [f32; 4],
    pub tint: [f32; 4],
    pub opacity: f32,
    pub refraction_strength: f32,
    pub fresnel_power: f32,
    // 1.0 when this pane was assigned a planar reflection slot (the shader then
    // samples the sharp mirror render at t3); 0.0 keeps the probe / sky path.
    pub planar: f32,
}

// One particle slot in the simulation pool (`Particle` in
// shaders/particle_simulate.slang, 32 bytes: `float3 + float` twice).
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct GpuParticle {
    pub position: [f32; 3],
    pub age: f32,
    pub velocity: [f32; 3],
    pub lifetime: f32,
}

// The particle render pass per-frame `ParticleView` cbuffer
// (shaders/particle.slang, 96 bytes: float4x4 + two (float3, pad) camera-basis
// slots).
#[derive(Copy, Clone)]
#[repr(C)]
pub struct ParticleView {
    pub vp: [[f32; 4]; 4],
    pub cam_right: [f32; 3],
    pub _pad0: f32,
    pub cam_up: [f32; 3],
    pub _pad1: f32,
}

// The G-buffer pre-pass `GbView` cbuffer (b0, 256 bytes): four column-major
// float4x4 (jittered VP rasterises, the un-jittered cur/prev VPs drive the
// motion vector, the view matrix transforms the normal + depth).
#[derive(Copy, Clone)]
#[repr(C)]
pub struct GbViewUniforms {
    pub jittered_vp: [[f32; 4]; 4],
    pub cur_vp: [[f32; 4]; 4],
    pub prev_vp: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
}

// The G-buffer pre-pass per-draw `GbModel` root-constant block (b1, 32 root
// constants = 128 bytes): the current and previous model matrices for the motion
// vector.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct GbModelPush {
    pub cur_model: [[f32; 4]; 4],
    pub prev_model: [[f32; 4]; 4],
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

    // ViewUniforms must match the `ViewBlock` cbuffer (b1) in the main-pass
    // shaders: two column-major float4x4 then elapsed/pad and the camera
    // position as three scalars, prefilter mip count, and two end pads (160 B).
    #[test]
    fn view_uniforms_layout_matches_hlsl() {
        assert_eq!(size_of::<ViewUniforms>(), 160);
        assert_eq!(offset_of!(ViewUniforms, vp), 0);
        assert_eq!(offset_of!(ViewUniforms, view_mat), 64);
        assert_eq!(offset_of!(ViewUniforms, elapsed), 128);
        assert_eq!(offset_of!(ViewUniforms, reflections_enabled), 132);
        assert_eq!(offset_of!(ViewUniforms, cam_x), 136);
        assert_eq!(offset_of!(ViewUniforms, cam_y), 140);
        assert_eq!(offset_of!(ViewUniforms, cam_z), 144);
        assert_eq!(offset_of!(ViewUniforms, prefilter_mip_count), 148);
        assert_eq!(offset_of!(ViewUniforms, shade_mode), 152);
        assert_eq!(offset_of!(ViewUniforms, _ep1), 156);
    }

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

    // DecalView must match the `DecalView` cbuffer (b0) in the decal shaders:
    // two column-major float4x4 then viewport and a 2-float pad (144 B total).
    #[test]
    fn decal_view_layout_matches_hlsl() {
        assert_eq!(size_of::<DecalView>(), 144);
        assert_eq!(offset_of!(DecalView, vp), 0);
        assert_eq!(offset_of!(DecalView, inv_vp), 64);
        assert_eq!(offset_of!(DecalView, viewport), 128);
        assert_eq!(offset_of!(DecalView, _pad), 136);
    }

    // DecalParams must match the `DecalParams` cbuffer (b1): model and
    // inv_model, the float4 tint, fade_pow, then three end pads (160 B total).
    #[test]
    fn decal_params_layout_matches_hlsl() {
        assert_eq!(size_of::<DecalParams>(), 160);
        assert_eq!(offset_of!(DecalParams, model), 0);
        assert_eq!(offset_of!(DecalParams, inv_model), 64);
        assert_eq!(offset_of!(DecalParams, tint), 128);
        assert_eq!(offset_of!(DecalParams, fade_pow), 144);
        assert_eq!(offset_of!(DecalParams, _p0), 148);
        assert_eq!(offset_of!(DecalParams, _p1), 152);
        assert_eq!(offset_of!(DecalParams, _p2), 156);
    }

    // LineView must match the `LineView` cbuffer (b0) in the line shaders: a
    // column-major float4x4 then the occluded-alpha scalar and three pads
    // (80 B total).
    #[test]
    fn line_view_layout_matches_hlsl() {
        assert_eq!(size_of::<LineView>(), 80);
        assert_eq!(offset_of!(LineView, vp), 0);
        assert_eq!(offset_of!(LineView, occluded_alpha), 64);
        assert_eq!(offset_of!(LineView, _pad), 68);
    }

    // The HLSL `TransparentView` cbuffer std layout is 160 bytes.
    #[test]
    fn transparent_view_layout_matches_hlsl() {
        assert_eq!(size_of::<TransparentViewGpu>(), 160);
        assert_eq!(offset_of!(TransparentViewGpu, vp), 0);
        assert_eq!(offset_of!(TransparentViewGpu, inv_vp), 64);
        assert_eq!(offset_of!(TransparentViewGpu, camera_pos), 128);
        assert_eq!(offset_of!(TransparentViewGpu, viewport), 144);
        assert_eq!(offset_of!(TransparentViewGpu, time), 152);
        assert_eq!(offset_of!(TransparentViewGpu, prefilter_mip_count), 156);
    }

    // The HLSL `GlassParams` cbuffer std layout is 64 bytes.
    #[test]
    fn glass_params_layout_matches_hlsl() {
        assert_eq!(size_of::<GlassParamsGpu>(), 64);
        assert_eq!(offset_of!(GlassParamsGpu, centre), 0);
        assert_eq!(offset_of!(GlassParamsGpu, normal), 16);
        assert_eq!(offset_of!(GlassParamsGpu, tint), 32);
        assert_eq!(offset_of!(GlassParamsGpu, opacity), 48);
        assert_eq!(offset_of!(GlassParamsGpu, refraction_strength), 52);
        assert_eq!(offset_of!(GlassParamsGpu, fresnel_power), 56);
        assert_eq!(offset_of!(GlassParamsGpu, planar), 60);
    }

    // Mirrors the `Particle` struct in shaders/particle_simulate.slang:
    // float3 + float, twice = 32 bytes, layout 0/12/16/28.
    #[test]
    fn gpu_particle_layout_matches_hlsl() {
        assert_eq!(size_of::<GpuParticle>(), 32);
        assert_eq!(offset_of!(GpuParticle, position), 0);
        assert_eq!(offset_of!(GpuParticle, age), 12);
        assert_eq!(offset_of!(GpuParticle, velocity), 16);
        assert_eq!(offset_of!(GpuParticle, lifetime), 28);
    }

    // Mirrors the `ParticleView` cbuffer in shaders/particle.slang: float4x4 (64)
    // + (float3 + pad) + (float3 + pad) = 96.
    #[test]
    fn particle_view_layout_matches_shaders() {
        assert_eq!(size_of::<ParticleView>(), 96);
        assert_eq!(offset_of!(ParticleView, vp), 0);
        assert_eq!(offset_of!(ParticleView, cam_right), 64);
        assert_eq!(offset_of!(ParticleView, cam_up), 80);
    }

    // GbViewUniforms must match the `GbView` cbuffer (b0) in every pre-pass VS:
    // four column-major float4x4 at offsets 0, 64, 128, 192 (256 B total).
    #[test]
    fn gb_view_uniforms_layout_matches_hlsl() {
        assert_eq!(size_of::<GbViewUniforms>(), 256);
        assert_eq!(offset_of!(GbViewUniforms, jittered_vp), 0);
        assert_eq!(offset_of!(GbViewUniforms, cur_vp), 64);
        assert_eq!(offset_of!(GbViewUniforms, prev_vp), 128);
        assert_eq!(offset_of!(GbViewUniforms, view), 192);
    }

    // GbModelPush is pushed as 32 root constants at b1, matching the `GbModel`
    // cbuffer: cur_model then prev_model (two column-major float4x4).
    #[test]
    fn gb_model_push_layout_matches_hlsl() {
        assert_eq!(size_of::<GbModelPush>(), 128);
        assert_eq!(size_of::<GbModelPush>() / 4, 32);
        assert_eq!(offset_of!(GbModelPush, cur_model), 0);
        assert_eq!(offset_of!(GbModelPush, prev_model), 64);
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
