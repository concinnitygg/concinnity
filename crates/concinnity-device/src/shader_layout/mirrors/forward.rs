// src/shader_layout/mirrors/forward.rs
//
// The bindless forward pass and the clustered light-binning kernel. Both files
// declare `ClusterParams` and `GpuLight`; they are separate declarations that
// can drift apart, so each is checked against the same Rust mirror.

use concinnity_core::gfx::render_types::{
    AreaLightData, ClusterParams, DirectionalLightData, GpuLight, GpuObjectData, LightUniforms,
    PointLightData, ShadowUniforms, SpotShadowData,
};
use concinnity_core::render::uniforms::ViewUniforms;

use crate::shader_layout::mirror::{Case, everywhere, mirror};

pub(in crate::shader_layout) fn main_bindless() -> Vec<Case> {
    let mut cases = vec![
        // The shader spells the same bytes under its own names.
        everywhere(mirror!(ViewUniforms => "ViewUniforms" {
            vp,
            [view] => ["view_mat"],
            elapsed,
            reflections_enabled,
            [cam_pos] => ["cam_x", "cam_y", "cam_z"],
            prefilter_mip_count,
            shade_mode,
            [_end_pad] => ["_ep1"],
            sky_rot,
        })),
        everywhere(mirror!(LightUniforms => "LightUniforms" {
            [directional] => ["dir"],
            [point] => ["pt"],
            [num_directional] => ["num_dir"],
            [num_point] => ["num_pt"],
            ambient_intensity,
            num_local_lights,
        })),
        everywhere(mirror!(DirectionalLightData => "DirLight" {
            [direction, intensity] => ["dir_i"],
            [color, _pad] => ["col"],
        })),
        everywhere(mirror!(PointLightData => "PointLight" {
            [position, range] => ["pos_r"],
            [color, intensity] => ["col_i"],
        })),
        // The forward pass reads the cascade matrices and splits plus the live
        // count; the trailing pad the CPU uploads is not declared here.
        everywhere(mirror!(ShadowUniforms => "ShadowUniforms" {
            light_vps,
            cascade_splits,
            active_cascades,
            [_pad] => [],
        })),
        everywhere(mirror!(SpotShadowData => "SpotShadowData" {
            light_vp,
            depth_bias,
            normal_bias,
            _pad,
        })),
        everywhere(mirror!(AreaLightData => "AreaLightData" {
            [right, two_sided] => ["right_two_sided"],
            [up, _pad] => ["up_pad"],
        })),
        everywhere(mirror!(GpuObjectData => "GpuObjectData" {
            model,
            [tint, roughness] => ["tint_roughness"],
            [emissive, metallic] => ["emissive_metallic"],
            albedo_index,
            normal_index,
            emissive_map_index,
            orm_map_index,
            [bb_min, cull_distance] => ["bb_min_cull_distance"],
            [bb_max, alpha_cutoff] => ["bb_max_alpha_cutoff"],
        })),
    ];
    cases.extend(light_cull());
    cases
}

pub(in crate::shader_layout) fn light_cull() -> Vec<Case> {
    vec![
        everywhere(mirror!(ClusterParams => "ClusterParams" {
            inv_view_proj,
            [cam_pos, z_near] => ["cam_pos_znear"],
            [view_forward, z_far] => ["view_forward_zfar"],
            grid_x,
            grid_y,
            grid_z,
            num_lights,
            screen_w,
            screen_h,
            use_clusters,
            _pad,
        })),
        everywhere(mirror!(GpuLight => "GpuLight" {
            [position, range] => ["position_range"],
            [color, intensity] => ["color_intensity"],
            [direction, kind] => ["direction_kind"],
            cos_inner,
            cos_outer,
            shadow_index,
            data_index,
        })),
    ]
}
