// src/shader_layout/mirrors/raymarch.rs
//
// The raymarched SDF volume pass. Both blocks are bound by every backend, so
// both mirror everywhere.
//
// `SdfVolumeUniforms` is the reason this file exists. Its centre and extent were
// spelled as a `float3` beside a `float` pad, which is what the CPU uploads and
// what SPIR-V and DXIL lay out -- but Metal sizes a constant-buffer `float3` at
// 16 bytes, so every field after the first pair sat four bytes late there and
// the volume marched a box it was never given. Reflection is what says so.

use concinnity_core::render::uniforms::{
    RaymarchShadowCascade, RaymarchView, RaymarchVolumeUniforms,
};

use crate::shader_layout::mirror::{Case, everywhere, mirror};

pub(in crate::shader_layout) fn surface() -> Vec<Case> {
    vec![
        everywhere(mirror!(RaymarchView => "RaymarchView" {
            vp,
            inv_vp,
            cam_pos,
            viewport,
            time,
            prefilter_mip_count,
            sky_rot,
        })),
        // The two float4 lanes carry an xyz and a pad each; the Rust side keeps
        // them spelled as the three-component value plus the pad it uploads.
        everywhere(mirror!(RaymarchVolumeUniforms => "SdfVolumeUniforms" {
            [centre, _pad0] => ["centre"],
            [extent, _pad1] => ["extent"],
            cone_ratio,
            max_distance,
            max_steps,
            receive_shadows,
            params,
        })),
    ]
}

pub(in crate::shader_layout) fn shadow() -> Vec<Case> {
    vec![everywhere(
        mirror!(RaymarchShadowCascade => "RaymarchShadowCascade" {
            cascade_idx,
            [_pad] => ["_pad0", "_pad1", "_pad2"],
        }),
    )]
}
