//! The bindless forward pass and the clustered light-binning kernel. Both files
//! declare `ClusterParams` and `GpuLight`; they are separate declarations that
//! can drift apart, so each is checked against the same Rust mirror.

use concinnity_engine::gfx::render_types::{
    AreaLightData, ClusterParams, DirectionalLightData, GpuLight, GpuMaterialParams, GpuObjectData,
    LightUniforms, PointLightData, ShadowUniforms, SpotShadowData, TileMorphView,
};
use concinnity_engine::render::uniforms::canopy::{CanopyLayerGpu, CanopyParams, ShoreSwashGpu};
use concinnity_engine::render::uniforms::ground::{GroundCoverGpu, GroundRuleGpu};
use concinnity_engine::render::uniforms::{DaylightBlock, ViewUniforms};

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
            ambient_occlusion,
            sky_rot,
            irradiance,
            pre_exposure,
            screen_bounce,
            [_pad] => ["_pad_view2"],
            daylight,
            tile_morph,
        })),
        everywhere(mirror!(TileMorphView => "TileMorphView" {
            [eye, level0_range] => ["eye_range0"],
            [prev_eye, coarsest_level] => ["prev_eye_coarsest"],
        })),
        everywhere(mirror!(DaylightBlock => "DaylightBlock" {
            center,
            camera_up,
            reach,
            body_dir,
            body_light,
            _pad,
            knots,
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
            [_pad] => ["_pad0", "_pad1", "_pad2"],
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
            params_index,
            [far_field] => ["far_canopy", "_far_pad0", "_far_pad1"],
            tile_morph,
        })),
    ];
    cases.extend(light_cull());
    cases
}

// The far-field canopy's block and layers. Only the canopy variant of the main
// fragment compiles them, so they are checked against it.
pub(in crate::shader_layout) fn canopy() -> Vec<Case> {
    vec![
        everywhere(mirror!(CanopyParams => "CanopyParams" {
            planet_rows,
            planet_radius,
            meadow_scale,
            meadow_mean,
            layer_count,
            layers,
            ground_turn,
            ground_salts,
            ground_frequency,
            ground_height_scale,
            ground_cell,
            ground_octaves,
            continent_salts,
            continent_frequency,
            continent_scale,
            continent_offset,
            continent_octaves,
            beach_width,
            beach_rise,
            hill_rise,
            ocean_depth,
            shelf_length,
            seabed_lift,
            cover_count,
            [_pad] => ["_pad0"],
            ground_origin,
            ground_up,
            covers,
            band,
            swash,
        })),
        everywhere(mirror!(ShoreSwashGpu => "ShoreSwash" {
            sea_level,
            runup,
            reach,
            _pad,
            omegas,
            phases,
            amplitudes,
            noise_salts,
            noise,
        })),
        everywhere(mirror!(CanopyLayerGpu => "CanopyLayer" {
            [root_color, color_variation] => ["root_color_variation"],
            [tip_color, macro_variation] => ["tip_color_macro"],
            [clump_size, height_variance, leaf_area, bend] => ["shape"],
            seed,
            sky_projection,
            [_pad] => ["_pad0", "_pad1"],
            [litter_color, clumping] => ["litter_clumping"],
            rule,
        })),
        everywhere(mirror!(GroundRuleGpu => "GroundRule" {
            altitude,
            slope,
        })),
        everywhere(mirror!(GroundCoverGpu => "GroundCover" {
            rule,
            tint,
            albedo_index,
            roughness,
            metallic,
            [_pad] => ["_pad0", "_pad1"],
        })),
    ]
}

// The parameter table's row. Only a variant whose hooks read a parameter
// compiles the table, so it is checked against one.
pub(in crate::shader_layout) fn material_params() -> Vec<Case> {
    vec![everywhere(
        mirror!(GpuMaterialParams => "GpuMaterialParams" {
            values,
        }),
    )]
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
            num_probes,
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
