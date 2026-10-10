//! The G-buffer pre-pass, the shadow pass, and the passes that draw their own
//! geometry: decals, world-space lines, particles and the text overlay.
//!
//! The shadow pass is where the hosts still diverge: Vulkan carries the per-draw
//! constants in one push-constant block, while Metal and DirectX split them
//! across separate buffers. Each leg mirrors the struct its own host binds;
//! everything else is one declaration for all three.

use concinnity_core::gfx::render_types::{
    ParticleParams, ShadowPassPush, ShadowUniforms, TextUniforms,
};
use concinnity_core::render::uniforms::directx::CullParams as DxCullParams;
use concinnity_core::render::uniforms::grass::{GpuGrassBlade, GrassParams};
use concinnity_core::render::uniforms::metal::CullUniforms as MetalCullParams;
use concinnity_core::render::uniforms::vulkan::{CullHizParams, CullParams as VkCullParams};
use concinnity_core::render::uniforms::{
    DecalParams, DecalView, GBufferView, GpuParticle, LineView, ParticleView, SkinParams,
};

use crate::shader_layout::mirror::{Case, everywhere, mirror, on};
use concinnity_core::platform::Platform;

const METAL: &[Platform] = &[Platform::Metal];
const VULKAN: &[Platform] = &[Platform::Vulkan];
const DIRECTX: &[Platform] = &[Platform::DirectX];
const METAL_AND_DIRECTX: &[Platform] = &[Platform::Metal, Platform::DirectX];

pub(in crate::shader_layout) fn gbuffer_vertex() -> Vec<Case> {
    vec![everywhere(mirror!(GBufferView => "GbView" {
        jittered_vp,
        cur_vp,
        prev_vp,
        [view] => ["view_mat"],
        prev_elapsed,
        [prev_cam_pos] => ["prev_cam_x", "prev_cam_y", "prev_cam_z"],
        motion,
        [_pad] => ["_pad0", "_pad1", "_pad2"],
    }))]
}

pub(in crate::shader_layout) fn shadow() -> Vec<Case> {
    vec![
        everywhere(mirror!(ShadowUniforms => "ShadowUniforms" {
            light_vps,
            cascade_splits,
            active_cascades,
            [_pad] => ["_pad0", "_pad1", "_pad2"],
        })),
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

pub(in crate::shader_layout) fn grass() -> Vec<Case> {
    vec![
        everywhere(mirror!(GrassParams => "GrassParams" {
            patch_rect,
            ground_y,
            tile_size,
            cell_size,
            cells_per_side,
            tile_origin,
            tile_count,
            [cam_pos, draw_distance] => ["cam_pos_distance"],
            frustum,
            height,
            height_variance,
            width,
            clump_size,
            stiffness,
            color_variation,
            capacity,
            args_slot,
            root_color,
            tip_color,
            wind,
            wind_gust,
        })),
        everywhere(mirror!(GpuGrassBlade => "GrassBlade" {
            root_facing,
            shape,
        })),
    ]
}

pub(in crate::shader_layout) fn particle() -> Vec<Case> {
    vec![
        everywhere(mirror!(ParticleView => "ParticleView" {
            vp,
            [cam_right, _pad0] => ["cam_right"],
            [cam_up, _pad1] => ["cam_up"],
        })),
        everywhere(mirror!(GpuParticle => "Particle" {
            [position, age] => ["position_age"],
            [velocity, lifetime] => ["velocity_lifetime"],
        })),
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
            spawn_count,
            spawn_first,
            random_seed,
            max_particles,
            [_pad] => ["_pad0", "_pad1", "_pad2"],
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

pub(in crate::shader_layout) fn rt_skin() -> Vec<Case> {
    vec![everywhere(mirror!(SkinParams => "SkinParams" {
        vertex_base,
        vertex_count,
        joint_count,
        target_count,
    }))]
}

// The GPU draw cull. The three hosts group the same fields differently: Vulkan
// splits the frustum and bucket routing (a push constant) from the Hi-Z
// reprojection (a set-1 uniform buffer, owned by `vulkan/hiz.rs`), DirectX
// fuses both into one b0 root-constant block, and Metal's one block also
// carries what its ICB encode dispatch reads. Each leg mirrors the struct its
// own host binds.
pub(in crate::shader_layout) fn cull() -> Vec<Case> {
    vec![
        on(
            METAL,
            mirror!(MetalCullParams => "CullParams" {
                planes,
                cam_pos,
                prev_view_proj,
                hiz_size,
                hiz_mip_count,
                hiz_enabled,
                object_count,
                skinned_base,
                cascade_base,
                bucket_count,
            }),
        ),
        on(
            VULKAN,
            mirror!(VkCullParams => "CullParams" {
                planes,
                cam_pos,
                object_count,
                bucket_count,
                bucket_stride,
            }),
        ),
        on(
            VULKAN,
            mirror!(CullHizParams => "CullHizParams" {
                prev_view_proj,
                hiz_size,
                hiz_mip_count,
                hiz_enabled,
            }),
        ),
        on(
            DIRECTX,
            mirror!(DxCullParams => "CullParams" {
                planes,
                cam_pos,
                object_count,
                prev_view_proj,
                hiz_size,
                hiz_mip_count,
                hiz_enabled,
                bucket_count,
                bucket_stride,
                [_pad] => ["_pad0", "_pad1"],
            }),
        ),
    ]
}
