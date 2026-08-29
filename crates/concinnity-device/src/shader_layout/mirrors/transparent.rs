// src/shader_layout/mirrors/transparent.rs
//
// The transparent pass's three producers, the ray-traced reflection resolve, and
// the fog pair. `TransparentView` is declared by glass.slang, glass_mesh.slang
// and water.slang alike, so all three are mirrored: they are separate
// declarations that can drift apart. The fog froxel kernel carries the third declaration of
// `ShadowUniforms` -- the only one that spells out the trailing pad the CPU
// uploads.

use concinnity_core::gfx::render_types::{
    FogFroxelParams, FogParams, RtGeomEntry, RtParams, ShadowUniforms,
};
use concinnity_core::render::uniforms::{
    GlassMeshParams, GlassParams, ProbeSet, ProbeUniforms, TransparentView, WaterParams,
    WaterWaveGpu,
};

use crate::shader_layout::mirror::{Case, everywhere, mirror, on};
use crate::shader_layout::programs::Target;

// The ray-traced reflection resolve builds only where inline ray query does.
// slangc rejects `TraceRayInline` on the Metal target in every stage, so the
// MSL variant of `rt_reflections.slang` cannot be compiled and its structs
// cannot be reflected there; the Metal renderer traces through its own
// `.metal` sources instead.
const VULKAN_AND_DIRECTX: &[Target] = &[Target::Vulkan, Target::DirectX];

pub(in crate::shader_layout) fn glass() -> Vec<Case> {
    vec![
        everywhere(mirror!(TransparentView => "TransparentView" {
            vp,
            inv_vp,
            camera_pos,
            viewport,
            time,
            prefilter_mip_count,
        })),
        everywhere(mirror!(GlassParams => "GlassParams" {
            centre,
            normal,
            tint,
            opacity,
            refraction_strength,
            fresnel_power,
            planar,
        })),
        everywhere(mirror!(ProbeUniforms => "ProbeUniforms" { box_min, box_max, probe_pos, })),
        everywhere(mirror!(ProbeSet => "ProbeSet" {
            count,
            [_pad] => ["_pad0", "_pad1", "_pad2"],
            probes,
        })),
    ]
}

pub(in crate::shader_layout) fn glass_mesh() -> Vec<Case> {
    vec![
        everywhere(mirror!(TransparentView => "TransparentView" {
            vp,
            inv_vp,
            camera_pos,
            viewport,
            time,
            prefilter_mip_count,
        })),
        everywhere(mirror!(GlassMeshParams => "GlassMeshParams" {
            model,
            tint,
            opacity,
            refraction_strength,
            fresnel_power,
            prefilter_mip_count,
        })),
    ]
}

pub(in crate::shader_layout) fn water() -> Vec<Case> {
    vec![
        everywhere(mirror!(TransparentView => "TransparentView" {
            vp,
            inv_vp,
            camera_pos,
            viewport,
            time,
            prefilter_mip_count,
        })),
        everywhere(mirror!(WaterWaveGpu => "WaterWave" { dir_amp_wave, speed_steep_pad, })),
        everywhere(mirror!(WaterParams => "WaterParams" {
            centre,
            deep_colour,
            shallow_colour,
            depth_falloff,
            foam_width,
            foam_intensity,
            fresnel_power,
            roughness,
            refraction_strength,
            wave_count,
            _pad,
            waves,
            planar,
        })),
    ]
}

pub(in crate::shader_layout) fn rt_reflections() -> Vec<Case> {
    vec![
        on(
            VULKAN_AND_DIRECTX,
            mirror!(RtParams => "RtParams" {
                intensity,
                max_distance,
                tan_half_fov_y,
                aspect,
                prefilter_mip_count,
                _pad0,
                _pad1,
                _pad2,
                cam_pos,
                sun_dir,
                sun_color,
                inv_view,
            }),
        ),
        on(
            VULKAN_AND_DIRECTX,
            mirror!(RtGeomEntry => "RtGeomEntry" {
                index_offset,
                base_vertex,
                albedo_index,
                normal_index,
                [tint] => ["tint_r", "tint_g", "tint_b"],
                roughness,
                metallic,
                [emissive] => ["emissive_r", "emissive_g", "emissive_b"],
                model,
                emissive_map_index,
                [_pad] => ["_pad0", "_pad1", "_pad2"],
            }),
        ),
    ]
}

pub(in crate::shader_layout) fn fog() -> Vec<Case> {
    vec![
        everywhere(mirror!(FogParams => "FogParams" {
            inv_vp,
            color,
            [cam_pos, _pad0] => ["cam_pos"],
            [sun_dir, _pad1] => ["sun_dir"],
            [sun_color, _pad2] => ["sun_color"],
            density,
            height_falloff,
            height_reference,
            max_distance,
            phase_g,
            ambient,
            viewport,
            inv_max_distance,
            [_pad3] => ["_pad3a", "_pad3b", "_pad3c"],
        })),
        everywhere(mirror!(FogFroxelParams => "FogFroxelParams" {
            view,
            [froxel_dims, _pad_align] => ["froxel_dims"],
            z_near,
            z_far,
            [_pad] => ["_pad0", "_pad1"],
        })),
        everywhere(mirror!(ShadowUniforms => "ShadowUniforms" {
            light_vps,
            cascade_splits,
            active_cascades,
            [_pad] => ["_pad0", "_pad1", "_pad2"],
        })),
    ]
}
