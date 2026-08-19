// src/shader_layout/mirrors/geometry.rs
//
// The G-buffer pre-pass, the shadow pass, and the passes that draw their own
// geometry: decals, world-space lines, particles and the text overlay.
//
// The pre-pass and the shadow pass are the two places where the hosts still
// diverge: Vulkan carries the per-draw constants in one push-constant block,
// while Metal and DirectX split them across separate buffers. Each leg mirrors
// the struct its own host binds; everything else is one declaration for all
// three.

use concinnity_core::gfx::render_types::{ParticleParams, ShadowPassPush, TextUniforms};
use concinnity_render::metal::uniforms::{ModelUniforms, SsrPrepassMat};
use concinnity_render::uniforms::{
    DecalParams, DecalView, GBufferModel, GBufferView, GpuParticle, LineView, ParticleView,
};
use concinnity_render::vulkan::uniforms::GbModelPush;

use crate::shader_layout::mirror::{Case, everywhere, mirror, on};
use crate::shader_layout::programs::Target;

const METAL: &[Target] = &[Target::Metal];
const VULKAN: &[Target] = &[Target::Vulkan];
const METAL_AND_DIRECTX: &[Target] = &[Target::Metal, Target::DirectX];

pub(in crate::shader_layout) fn gbuffer_vertex() -> Vec<Case> {
    vec![
        everywhere(mirror!(GBufferView => "GbView" {
            jittered_vp,
            cur_vp,
            prev_vp,
            [view] => ["view_mat"],
        })),
        // The model pair is a constant buffer on Metal and DirectX; Vulkan puts
        // it in the push-constant block alongside the roughness.
        on(
            METAL_AND_DIRECTX,
            mirror!(GBufferModel => "GbModel" { cur_model, prev_model, }),
        ),
        on(VULKAN, vulkan_model_push()),
    ]
}

pub(in crate::shader_layout) fn gbuffer_fragment() -> Vec<Case> {
    vec![
        on(
            METAL,
            mirror!(SsrPrepassMat => "GbMat" {
                roughness,
                [_pad] => ["_pad0", "_pad1", "_pad2"],
            }),
        ),
        on(VULKAN, vulkan_model_push()),
    ]
}

// Both pre-pass stages see the whole Vulkan push block: the vertex reads the
// model pair, the fragment only the roughness.
fn vulkan_model_push() -> crate::shader_layout::mirror::Mirror {
    mirror!(GbModelPush => "GbModelPush" {
        cur_model,
        prev_model,
        roughness,
        [_pad] => ["_pad0", "_pad1", "_pad2"],
    })
}

pub(in crate::shader_layout) fn shadow() -> Vec<Case> {
    vec![
        on(METAL, mirror!(ModelUniforms => "ModelUniforms" { model, })),
        on(
            METAL,
            mirror!(ShadowPassPush => "ShadowPassPush" {
                cascade_idx,
                [_pad] => ["_pad0", "_pad1", "_pad2"],
            }),
        ),
    ]
}

pub(in crate::shader_layout) fn decal() -> Vec<Case> {
    vec![
        on(
            METAL_AND_DIRECTX,
            mirror!(DecalView => "DecalView" { vp, inv_vp, viewport, _pad, }),
        ),
        on(
            METAL_AND_DIRECTX,
            mirror!(DecalParams => "DecalParams" {
                model,
                inv_model,
                tint,
                fade_pow,
                _pad0,
                _pad1,
                _pad2,
            }),
        ),
    ]
}

pub(in crate::shader_layout) fn line() -> Vec<Case> {
    vec![everywhere(mirror!(LineView => "LineView" {
        vp,
        occluded_alpha,
        [_pad] => ["_pad0", "_pad1", "_pad2"],
    }))]
}

pub(in crate::shader_layout) fn particle() -> Vec<Case> {
    vec![
        everywhere(mirror!(ParticleView => "ParticleView" {
            vp,
            [cam_right, _pad0] => ["cam_right"],
            [cam_up, _pad1] => ["cam_up"],
        })),
        everywhere(mirror!(GpuParticle => "Particle" { position, age, velocity, lifetime, })),
        everywhere(mirror!(ParticleParams => "ParticleParams" {
            [position, spread_cos] => ["position_spread"],
            [direction, speed_min] => ["direction_speed_min"],
            [gravity, speed_max] => ["gravity_speed_max"],
            color_start,
            color_end,
            lifetime_min,
            lifetime_max,
            size_start,
            size_end,
            dt,
            spawn_budget,
            random_seed,
            max_particles,
        })),
    ]
}

pub(in crate::shader_layout) fn text() -> Vec<Case> {
    vec![everywhere(mirror!(TextUniforms => "TextUniforms" {
        win_width,
        win_height,
        [_pad] => ["_pad0", "_pad1"],
    }))]
}
