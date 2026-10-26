use super::light::{
    CLUSTER_GRID_X, CLUSTER_GRID_Y, CLUSTER_GRID_Z, CLUSTER_LIGHT_LIST_STRIDE,
    CLUSTER_PROBE_MASK_WORDS, MAX_CLUSTERED_PROBES, MAX_LIGHTS_PER_CLUSTER,
};
use super::post::DEPTH_VIEW_UNLIMITED_SPAN;
use super::*;
use crate::render::shader::consts;
use crate::test_support::draw_object;
use alloc::vec;
use alloc::vec::Vec;
use concinnity_core::math::SaturatingFrom;
use core::mem::{offset_of, size_of};

// The two reserved entries sit past the real textures in a fixed order, so
// the resolvers and every backend's pool construction agree.
#[test]
fn the_fallbacks_follow_the_real_textures_in_order() {
    let texture_count = 4;
    assert_eq!(normal_pool_index(NO_NORMAL_MAP_SLOT, texture_count), 4);
    assert_eq!(albedo_pool_index(NO_ALBEDO_SLOT, texture_count), 5);
    assert_eq!(
        FALLBACK_TEXTURE_COUNT, 2,
        "the pool reserves exactly the flat-normal and white entries"
    );
}

#[test]
fn draw_indices_round_trip_through_usize() {
    assert_eq!(DrawIndex::from_usize(7), DrawIndex(7));
    assert_eq!(DrawIndex(7).index(), 7);
    assert_eq!(SkinnedIndex::from_usize(3).index(), 3);
    let max = u32::MAX as usize;
    assert_eq!(DrawIndex::from_usize(max).0, u32::MAX);
    assert_eq!(alloc::format!("{}", SkinnedIndex(12)), "12");
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "DrawIndex does not fit in a u32")]
fn a_draw_index_past_u32_is_caught_in_debug() {
    let _ = DrawIndex::from_usize(u32::MAX as usize + 1);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "SkinnedIndex does not fit in a u32")]
fn a_skinned_index_past_u32_is_caught_in_debug() {
    let _ = SkinnedIndex::from_usize(u32::MAX as usize + 1);
}

// A clone copies an existing draw object, so a world with none reserves no
// clone records; that is what keeps the GPU-driven path off for a
// menu-only world.
#[test]
fn the_clone_reserve_is_empty_without_build_time_objects() {
    assert_eq!(clone_reserve(0), 0);
    assert_eq!(clone_reserve(1), MAX_CLONE_DRAWS);
    assert_eq!(clone_reserve(500), MAX_CLONE_DRAWS);
}

// Only resident objects past the build-time count hold a runtime record, so
// a retired slot frees its record and build-time objects never count.
#[test]
fn the_runtime_reserve_counts_resident_runtime_objects_only() {
    let resident = draw_object();
    let mut retired = draw_object();
    retired.resident = false;
    let n_objects = 2;
    let objects = vec![
        draw_object(),
        draw_object(),
        resident,
        retired,
        draw_object(),
    ];
    assert!(!runtime_reserve_full(&objects, n_objects, 3));
    assert!(runtime_reserve_full(&objects, n_objects, 2));
    assert!(runtime_reserve_full(&objects, n_objects, 0));
    assert!(!runtime_reserve_full(&objects[..n_objects], n_objects, 1));
}

// The regression this fixes: slot 0 is a real handle, so an unset albedo
// encoded as 0 sampled whichever texture the world declared first. Only the
// sentinel may reach the white entry.
#[test]
fn an_unset_albedo_resolves_to_white_and_slot_zero_stays_a_real_texture() {
    let texture_count = 4;
    assert_eq!(albedo_pool_index(0, texture_count), 0);
    assert_ne!(
        albedo_pool_index(NO_ALBEDO_SLOT, texture_count),
        albedo_pool_index(0, texture_count)
    );
}

// A stale slot past the pool still has to land on a real texture rather
// than run off the end or be mistaken for the sentinel.
#[test]
fn an_out_of_range_albedo_clamps_into_the_real_textures() {
    assert_eq!(albedo_pool_index(9, 4), 3);
    assert_eq!(albedo_pool_index(0, 0), 0);
}

// Three (vec3, scalar) lanes, the two colors, then twelve scalars, the last
// three the emitter's bursts; 128 bytes, Vulkan's whole push range.
#[test]
fn particle_params_layout_matches_the_shader() {
    assert_eq!(size_of::<ParticleParams>(), 128);
    assert_eq!(offset_of!(ParticleParams, position), 0);
    assert_eq!(offset_of!(ParticleParams, spread_cos), 12);
    assert_eq!(offset_of!(ParticleParams, direction), 16);
    assert_eq!(offset_of!(ParticleParams, speed_min), 28);
    assert_eq!(offset_of!(ParticleParams, gravity), 32);
    assert_eq!(offset_of!(ParticleParams, speed_max), 44);
    assert_eq!(offset_of!(ParticleParams, color_start), 48);
    assert_eq!(offset_of!(ParticleParams, color_end), 64);
    assert_eq!(offset_of!(ParticleParams, lifetime_min), 80);
    assert_eq!(offset_of!(ParticleParams, lifetime_max), 84);
    assert_eq!(offset_of!(ParticleParams, size_start), 88);
    assert_eq!(offset_of!(ParticleParams, size_end), 92);
    assert_eq!(offset_of!(ParticleParams, dt), 96);
    assert_eq!(offset_of!(ParticleParams, spawn_count), 100);
    assert_eq!(offset_of!(ParticleParams, spawn_first), 104);
    assert_eq!(offset_of!(ParticleParams, random_seed), 108);
    assert_eq!(offset_of!(ParticleParams, max_particles), 112);
    assert_eq!(offset_of!(ParticleParams, burst_first), 116);
    assert_eq!(offset_of!(ParticleParams, burst_count), 120);
    assert_eq!(offset_of!(ParticleParams, burst_spawns), 124);
}

// Three float4 lanes of scalars after the four vector lanes; the
// pre-exposure opens the last one.
#[test]
fn fog_params_layout_matches_the_shader() {
    assert_eq!(size_of::<FogParams>(), 176);
    assert_eq!(offset_of!(FogParams, color), 64);
    assert_eq!(offset_of!(FogParams, cam_pos), 80);
    assert_eq!(offset_of!(FogParams, sun_dir), 96);
    assert_eq!(offset_of!(FogParams, sun_color), 112);
    assert_eq!(offset_of!(FogParams, density), 128);
    assert_eq!(offset_of!(FogParams, ambient), 148);
    assert_eq!(offset_of!(FogParams, viewport), 152);
    assert_eq!(offset_of!(FogParams, inv_max_distance), 160);
    assert_eq!(offset_of!(FogParams, pre_exposure), 164);
    assert_eq!(offset_of!(FogParams, _pad3), 168);
}

// Sixteen scalars: the post tunables, the fade, the view selector, the
// depth view's two distances, the reactive flag, and the night-vision weight
// and luminance scale.
#[test]
fn composite_params_layout_matches_shaders() {
    assert_eq!(size_of::<CompositeParams>(), 64);
    assert_eq!(offset_of!(CompositeParams, post), 0);
    assert_eq!(size_of::<PostProcessParams>(), 36);
    assert_eq!(offset_of!(PostProcessParams, exposure), 12);
    assert_eq!(offset_of!(PostProcessParams, fxaa), 32);
    assert_eq!(offset_of!(CompositeParams, fade), 36);
    assert_eq!(offset_of!(CompositeParams, view_mode), 40);
    assert_eq!(offset_of!(CompositeParams, depth_near), 44);
    assert_eq!(offset_of!(CompositeParams, depth_far), 48);
    assert_eq!(offset_of!(CompositeParams, reactive_valid), 52);
    assert_eq!(offset_of!(CompositeParams, night_vision), 56);
    assert_eq!(offset_of!(CompositeParams, night_nits), 60);
}

#[test]
fn the_depth_view_spans_near_to_the_view_distance() {
    assert_eq!(
        CompositeParams::depth_view_range(0.05, Some(400.0)),
        (0.05, 400.0)
    );
    assert_eq!(
        CompositeParams::depth_view_range(0.05, None),
        (0.05, DEPTH_VIEW_UNLIMITED_SPAN)
    );
    // Degenerate inputs still give the log ramp a span.
    assert_eq!(
        CompositeParams::depth_view_range(0.0, Some(0.0)),
        (1e-3, 2e-3)
    );
    assert_eq!(
        CompositeParams::depth_view_range(1.0, Some(0.5)),
        (1.0, 2.0)
    );
}

// Every (vec3, scalar) pair shares one 16-byte lane, as the shader's float4
// spelling does.
#[test]
fn cluster_params_layout_matches_shaders() {
    assert_eq!(size_of::<ClusterParams>(), 128);
    assert_eq!(offset_of!(ClusterParams, inv_view_proj), 0);
    assert_eq!(offset_of!(ClusterParams, cam_pos), 64);
    assert_eq!(offset_of!(ClusterParams, z_near), 76);
    assert_eq!(offset_of!(ClusterParams, view_forward), 80);
    assert_eq!(offset_of!(ClusterParams, z_far), 92);
    assert_eq!(offset_of!(ClusterParams, grid_x), 96);
    assert_eq!(offset_of!(ClusterParams, grid_y), 100);
    assert_eq!(offset_of!(ClusterParams, grid_z), 104);
    assert_eq!(offset_of!(ClusterParams, num_lights), 108);
    assert_eq!(offset_of!(ClusterParams, screen_w), 112);
    assert_eq!(offset_of!(ClusterParams, screen_h), 116);
    assert_eq!(offset_of!(ClusterParams, use_clusters), 120);
    assert_eq!(offset_of!(ClusterParams, num_probes), 124);
}

#[test]
fn cluster_params_for_camera_fills_every_field() {
    let mut view = crate::transform::IDENTITY;
    // Camera looking down -X: view row 2 (the backward axis) is +X.
    view[0][2] = 1.0;
    view[2][2] = 0.0;
    let camera = ClusterCamera {
        view,
        proj: crate::transform::IDENTITY,
        position: [1.0, 2.0, 3.0],
        near: 0.0,
        range: 500.0,
        width: 1920,
        height: 1080,
    };
    let p = ClusterParams::for_camera(&camera, 7, 3);
    assert_eq!(
        p.inv_view_proj,
        crate::transform::mat4_inverse(crate::transform::mat4_mul(
            crate::transform::IDENTITY,
            view
        ))
    );
    assert_eq!(p.cam_pos, [1.0, 2.0, 3.0]);
    assert_eq!(p.z_near, 1e-3, "near clamps off zero");
    assert_eq!(p.view_forward, [-1.0, 0.0, 0.0]);
    assert_eq!(p.z_far, 500.0);
    assert_eq!(
        (p.grid_x, p.grid_y, p.grid_z),
        (CLUSTER_GRID_X, CLUSTER_GRID_Y, CLUSTER_GRID_Z)
    );
    assert_eq!((p.num_lights, p.num_probes, p.use_clusters), (7, 3, 1));
    assert_eq!((p.screen_w, p.screen_h), (1920.0, 1080.0));

    let off = ClusterParams::for_camera(&camera, -1, 0);
    assert_eq!((off.num_lights, off.use_clusters), (0, 0));
    let probes_only = ClusterParams::for_camera(&camera, 0, 1);
    assert_eq!((probes_only.num_lights, probes_only.use_clusters), (0, 1));
}

#[test]
fn cluster_params_with_camera_keeps_the_counts_and_moves_the_grid() {
    let main = ClusterCamera {
        view: crate::transform::IDENTITY,
        proj: crate::transform::IDENTITY,
        position: [0.0; 3],
        near: 0.1,
        range: 100.0,
        width: 1920,
        height: 1080,
    };
    let mut other_view = crate::transform::IDENTITY;
    other_view[0][2] = 1.0;
    other_view[2][2] = 0.0;
    let other = ClusterCamera {
        view: other_view,
        position: [4.0, 5.0, 6.0],
        near: 0.2,
        range: 40.0,
        width: 960,
        height: 540,
        ..main
    };
    let base = ClusterParams::for_camera(&main, 9, 2);
    let moved = base.with_camera(&other);
    let direct = ClusterParams::for_camera(&other, 9, 2);
    assert_eq!(bytemuck::bytes_of(&moved), bytemuck::bytes_of(&direct));
    assert_eq!(
        (moved.num_lights, moved.num_probes, moved.use_clusters),
        (9, 2, 1)
    );
}

// `cluster_types.hlsl` hardcodes the list layout every cluster reader and
// the binning kernel share, so it must track the values the CPU sizes the
// buffer with.
#[test]
fn cluster_list_constants_match_shaders() {
    let src = crate::render::shader::embedded::CLUSTER_TYPES;
    for (name, value) in [
        ("CLUSTER_LIGHT_LIST_STRIDE", CLUSTER_LIGHT_LIST_STRIDE),
        ("MAX_LIGHTS_PER_CLUSTER", MAX_LIGHTS_PER_CLUSTER),
        ("CLUSTER_PROBE_MASK_WORDS", CLUSTER_PROBE_MASK_WORDS),
        ("MAX_CLUSTERED_PROBES", MAX_CLUSTERED_PROBES),
    ] {
        assert_eq!(
            consts::uint(src, name),
            value as usize,
            "cluster_types.hlsl {name} drifted from render_types"
        );
    }
}

// The light and shadow blocks' array lengths, which every pass that splices
// `light_types.hlsl` sizes them by.
#[test]
fn light_limits_match_shaders() {
    let src = crate::render::shader::embedded::LIGHT_TYPES;
    for (name, value) in [
        ("MAX_DIRECTIONAL_LIGHTS", MAX_DIRECTIONAL_LIGHTS),
        ("MAX_POINT_LIGHTS", MAX_POINT_LIGHTS),
        ("NUM_SHADOW_CASCADES", NUM_SHADOW_CASCADES),
    ] {
        assert_eq!(consts::uint(src, name), value, "{name}");
    }
}

#[test]
fn draw_flags_match_shaders() {
    let src = crate::render::shader::embedded::OBJECT_COMMON;
    for (name, value) in [
        ("DRAW_ENABLED", DrawArgsFlags::ENABLED),
        ("DRAW_CULLABLE", DrawArgsFlags::CULLABLE),
        ("DRAW_NO_HISTORY", DrawArgsFlags::NO_HISTORY),
        ("DRAW_BUCKET_SHIFT", DrawArgsFlags::BUCKET_SHIFT),
    ] {
        assert_eq!(consts::uint(src, name), value as usize, "{name}");
    }
}

// The SSR resolve's push block. `sky_rot` is a float4 array, so it needs
// the 16-byte boundary the matrix ahead of it already lands on.
#[test]
fn ssr_params_layout_matches_shaders() {
    assert_eq!(size_of::<SsrParams>(), 144);
    assert_eq!(offset_of!(SsrParams, intensity), 0);
    assert_eq!(offset_of!(SsrParams, max_distance), 4);
    assert_eq!(offset_of!(SsrParams, tan_half_fov_y), 8);
    assert_eq!(offset_of!(SsrParams, aspect), 12);
    assert_eq!(offset_of!(SsrParams, stride), 16);
    assert_eq!(offset_of!(SsrParams, thickness), 20);
    assert_eq!(offset_of!(SsrParams, prefilter_mip_count), 24);
    assert_eq!(offset_of!(SsrParams, pre_exposure), 28);
    assert_eq!(offset_of!(SsrParams, inv_view), 32);
    assert_eq!(offset_of!(SsrParams, sky_rot), 96);
    assert_eq!(size_of::<SsrParams>() % 16, 0);
}

#[test]
fn ssao_params_layout_matches_shaders() {
    assert_eq!(size_of::<SsaoParams>(), 32);
    assert_eq!(offset_of!(SsaoParams, radius), 0);
    assert_eq!(offset_of!(SsaoParams, intensity), 4);
    assert_eq!(offset_of!(SsaoParams, tan_half_fov_y), 8);
    assert_eq!(offset_of!(SsaoParams, aspect), 12);
    assert_eq!(offset_of!(SsaoParams, reach), 16);
    assert_eq!(offset_of!(SsaoParams, fade_scale), 20);
    assert_eq!(offset_of!(SsaoParams, fade_bias), 24);
    assert_eq!(offset_of!(SsaoParams, _pad), 28);
}

#[test]
fn ssgi_params_layout_matches_shaders() {
    assert_eq!(size_of::<SsgiParams>(), 48);
    assert_eq!(offset_of!(SsgiParams, intensity), 0);
    assert_eq!(offset_of!(SsgiParams, max_distance), 4);
    assert_eq!(offset_of!(SsgiParams, tan_half_fov_y), 8);
    assert_eq!(offset_of!(SsgiParams, aspect), 12);
    assert_eq!(offset_of!(SsgiParams, thickness), 16);
    assert_eq!(offset_of!(SsgiParams, history_valid), 20);
    assert_eq!(offset_of!(SsgiParams, rays), 24);
    assert_eq!(offset_of!(SsgiParams, frame), 28);
    assert_eq!(offset_of!(SsgiParams, levels), 32);
    assert_eq!(offset_of!(SsgiParams, gi_scale), 36);
    assert_eq!(offset_of!(SsgiParams, history_scale), 40);
    assert_eq!(offset_of!(SsgiParams, _pad), 44);
}

// The RT kernel's parameter block, with every vec3 padded to a float4.
#[test]
fn rt_params_layout_matches_shaders() {
    assert_eq!(size_of::<RtParams>(), 192);
    assert_eq!(offset_of!(RtParams, intensity), 0);
    assert_eq!(offset_of!(RtParams, max_distance), 4);
    assert_eq!(offset_of!(RtParams, tan_half_fov_y), 8);
    assert_eq!(offset_of!(RtParams, aspect), 12);
    assert_eq!(offset_of!(RtParams, prefilter_mip_count), 16);
    assert_eq!(offset_of!(RtParams, sun_shadows), 20);
    assert_eq!(offset_of!(RtParams, trace_divisor), 24);
    assert_eq!(offset_of!(RtParams, pre_exposure), 28);
    assert_eq!(offset_of!(RtParams, cam_pos), 32);
    assert_eq!(offset_of!(RtParams, sun_dir), 48);
    assert_eq!(offset_of!(RtParams, sun_color), 64);
    assert_eq!(offset_of!(RtParams, inv_view), 80);
    assert_eq!(offset_of!(RtParams, sky_rot), 144);
    assert_eq!(size_of::<RtParams>() % 16, 0);
}

#[test]
fn gpu_object_data_layout_matches_shaders() {
    // `object_common.hlsl` spells each (vec3, scalar)
    // pair as one 16-byte lane; the four indices fill the lane between the
    // material block and the cull bounds.
    assert_eq!(size_of::<GpuObjectData>(), 176);
    assert_eq!(offset_of!(GpuObjectData, model), 0);
    assert_eq!(offset_of!(GpuObjectData, tint), 64);
    assert_eq!(offset_of!(GpuObjectData, roughness), 76);
    assert_eq!(offset_of!(GpuObjectData, emissive), 80);
    assert_eq!(offset_of!(GpuObjectData, metallic), 92);
    assert_eq!(offset_of!(GpuObjectData, albedo_index), 96);
    assert_eq!(offset_of!(GpuObjectData, normal_index), 100);
    assert_eq!(offset_of!(GpuObjectData, emissive_map_index), 104);
    assert_eq!(offset_of!(GpuObjectData, orm_map_index), 108);
    assert_eq!(offset_of!(GpuObjectData, bb_min), 112);
    assert_eq!(offset_of!(GpuObjectData, cull_distance), 124);
    assert_eq!(offset_of!(GpuObjectData, bb_max), 128);
    assert_eq!(offset_of!(GpuObjectData, alpha_cutoff), 140);
    assert_eq!(offset_of!(GpuObjectData, params_index), 144);
    assert_eq!(offset_of!(GpuObjectData, far_field), 148);
    assert_eq!(offset_of!(FarFieldTint, canopy), 0);
    assert_eq!(offset_of!(FarFieldTint, _pad), 4);
    assert_eq!(size_of::<FarFieldTint>(), 12);
    assert_eq!(offset_of!(GpuObjectData, tile_morph), 160);
    assert_eq!(offset_of!(GpuTileMorph, vertex_base), 0);
    assert_eq!(offset_of!(GpuTileMorph, seams), 4);
    assert_eq!(offset_of!(GpuTileMorph, level), 8);
    assert_eq!(offset_of!(GpuTileMorph, planet_tile), 12);
    assert_eq!(size_of::<GpuTileMorph>(), 16);
}

#[test]
fn tile_morph_view_layout_matches_shaders() {
    // `tile_morph_types.hlsl` declares it as two float4s.
    assert_eq!(size_of::<TileMorphView>(), 32);
    assert_eq!(offset_of!(TileMorphView, eye), 0);
    assert_eq!(offset_of!(TileMorphView, level0_range), 12);
    assert_eq!(offset_of!(TileMorphView, prev_eye), 16);
    assert_eq!(offset_of!(TileMorphView, coarsest_level), 28);
}

#[test]
fn shadow_pass_push_layout_matches_shaders() {
    // `shadow.hlsl` declares the view slot padded to a float4, then the morph view.
    assert_eq!(size_of::<ShadowPassPush>(), 48);
    assert_eq!(offset_of!(ShadowPassPush, cascade_idx), 0);
    assert_eq!(offset_of!(ShadowPassPush, _pad), 4);
    assert_eq!(offset_of!(ShadowPassPush, tile_morph), 16);
}

#[test]
fn gpu_material_params_layout_matches_shaders() {
    // `main_bindless.hlsl` declares the row as two float4s.
    assert_eq!(size_of::<GpuMaterialParams>(), 32);
    assert_eq!(offset_of!(GpuMaterialParams, values), 0);
}

#[test]
fn text_vertex_layout_matches_shaders() {
    // The text vertex buffer is consumed through a vertex descriptor whose
    // attributes sit at offsets 0 (pos), 8 (uv), 16 (color) with a 32-byte
    // stride, matching the `TextVertexIn` attribute slots in
    // `shaders/text.hlsl`.
    assert_eq!(size_of::<TextVertex>(), 32);
    assert_eq!(offset_of!(TextVertex, pos), 0);
    assert_eq!(offset_of!(TextVertex, uv), 8);
    assert_eq!(offset_of!(TextVertex, color), 16);
    assert_eq!(offset_of!(TextVertex, mode), 28);
}

#[test]
fn line_vertex_layout_matches_shaders() {
    // Consumed through a vertex descriptor whose attributes sit at offsets
    // 0 (pos), 12 (edge), 16 (color) with a 32-byte stride, matching the
    // `LineVertexIn` attribute slots in `shaders/line.hlsl`.
    assert_eq!(size_of::<LineVertex>(), 32);
    assert_eq!(offset_of!(LineVertex, pos), 0);
    assert_eq!(offset_of!(LineVertex, edge), 12);
    assert_eq!(offset_of!(LineVertex, color), 16);
}

#[test]
fn active_lod_returns_lod0_when_no_alternates() {
    let obj = draw_object();
    assert_eq!(obj.active_lod(0.0), (obj.index_offset, obj.index_count));
    assert_eq!(obj.active_lod(1000.0), (obj.index_offset, obj.index_count));
}

#[test]
fn instance_object_records_expand_clamp_and_transform_bounds() {
    use crate::gfx::frustum::transform_aabb;
    let translate = |x: f32| {
        let mut m = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        m[3][0] = x;
        m
    };
    // Cluster A: 2 instances, in-range texture/normal slots, unit-cube mesh.
    let mut a = sample_cluster(vec![translate(0.0), translate(10.0)], Vec::new());
    a.texture_slot = 3;
    a.normal_map_slot = 2;
    a.local_bb_min = [0.0, 0.0, 0.0];
    a.local_bb_max = [1.0, 1.0, 1.0];
    // Cluster B: 1 instance, out-of-range slots that must clamp to the pool.
    let mut b = sample_cluster(vec![translate(5.0)], Vec::new());
    b.texture_slot = 99;
    b.normal_map_slot = 99;
    b.local_bb_min = [0.0, 0.0, 0.0];
    b.local_bb_max = [2.0, 2.0, 2.0];

    // 8 real textures in the shared pool (flat-normal fallback at slot 8).
    let recs = instance_object_records(&[a, b], 8);
    // Cluster-then-instance order: 2 from A, then 1 from B.
    assert_eq!(recs.len(), 3);
    // Shared pool addressing: albedo and a real normal map both index the pool
    // by their own handle.
    assert_eq!(recs[0].albedo_index, 3);
    assert_eq!(recs[0].normal_index, 2);
    assert_eq!(recs[1].albedo_index, 3);
    // The second instance carries its own model + translated world AABB.
    assert_eq!(recs[1].model[3][0], 10.0);
    let (exp_min, exp_max) = transform_aabb([0.0; 3], [1.0, 1.0, 1.0], translate(10.0));
    assert_eq!(recs[1].bb_min, exp_min);
    assert_eq!(recs[1].bb_max, exp_max);
    // Out-of-range slots clamp to the last real texture (7).
    assert_eq!(recs[2].albedo_index, 7);
    assert_eq!(recs[2].normal_index, 7);
}

#[test]
fn instance_object_records_map_a_missing_normal_to_the_fallback() {
    // A cluster with no normal map addresses the flat-normal fallback slot
    // (one past the last real texture), so the shader samples flat (0,0,1).
    let mut c = sample_cluster(vec![[[0.0; 4]; 4]], Vec::new());
    c.texture_slot = 1;
    c.normal_map_slot = NO_NORMAL_MAP_SLOT;
    let recs = instance_object_records(&[c], 5);
    assert_eq!(recs[0].albedo_index, 1);
    assert_eq!(recs[0].normal_index, 5);
}

#[test]
fn pack_skinned_record_pads_and_transforms_bounds() {
    use crate::gfx::frustum::transform_aabb;
    let mut model = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    model[3][0] = 10.0;
    let obj = SkinnedDrawObject {
        vertex_base: 0,
        vertex_count: 10,
        index_offset: 0,
        index_count: 30,
        model,
        texture_slot: 5,
        normal_map_slot: 2,
        material: MaterialUniforms::DEFAULT,
        visible: true,
        joint_count: 4,
        local_bb_min: [-1.0, -1.0, -1.0],
        local_bb_max: [1.0, 1.0, 1.0],
        lod_alternates: Vec::new(),
    };
    let rec = pack_skinned_record(&obj, 5, 8 + 2);
    // Model is applied after skinning (the kernel deforms into model space).
    assert_eq!(rec.model, model);
    assert_eq!(rec.albedo_index, 5);
    assert_eq!(rec.normal_index, 8 + 2);
    // No distance cutoff for skinned (frustum + Hi-Z only).
    assert_eq!(rec.cull_distance, 0.0);
    // Bind-pose half-extent 1.0, padded x2 -> 2.0, then translated +10 on x.
    let (exp_min, exp_max) = transform_aabb([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0], model);
    assert_eq!(rec.bb_min, exp_min);
    assert_eq!(rec.bb_max, exp_max);
    assert_eq!(rec.bb_min[0], 8.0);
    assert_eq!(rec.bb_max[0], 12.0);
}

#[test]
fn active_lod_picks_highest_passing_threshold() {
    let mut obj = draw_object();
    obj.lod_alternates = vec![
        LodSlice {
            index_offset: 100,
            index_count: 18,
            switch_distance: 10.0,
        },
        LodSlice {
            index_offset: 200,
            index_count: 9,
            switch_distance: 25.0,
        },
        LodSlice {
            index_offset: 300,
            index_count: 3,
            switch_distance: 60.0,
        },
    ];
    // Below the first threshold → LOD0.
    assert_eq!(obj.active_lod(5.0), (obj.index_offset, obj.index_count));
    // Just past the first → LOD1.
    assert_eq!(obj.active_lod(15.0), (100, 18));
    // Between LOD2 and LOD3 thresholds → LOD2.
    assert_eq!(obj.active_lod(40.0), (200, 9));
    // Past everything → LOD3.
    assert_eq!(obj.active_lod(120.0), (300, 3));
}

fn sample_cluster(
    instances: Vec<[[f32; 4]; 4]>,
    lod_alternates: Vec<LodSlice>,
) -> InstancedCluster {
    InstancedCluster {
        vertex_offset: 0,
        vertex_count: 0,
        index_offset: 0,
        index_count: 60,
        texture_slot: 0,
        normal_map_slot: 0,
        material: MaterialUniforms::DEFAULT,
        cluster_bb_min: [f32::NAN; 3],
        cluster_bb_max: [f32::NAN; 3],
        local_bb_min: [0.0; 3],
        local_bb_max: [0.0; 3],
        cull_distance: 0.0,
        instances,
        lod_alternates,
    }
}

#[test]
fn pack_object_record_stores_caller_pool_indices() {
    // The texture-pool indices are caller-supplied (per-backend addressing)
    // and copied verbatim into the record the shader fetches.
    let obj = draw_object();
    let rec = pack_object_record(&obj, 14, 15);
    assert_eq!(rec.albedo_index, 14);
    assert_eq!(rec.normal_index, 15);
}

#[test]
fn pack_object_record_copies_transform_material_and_bounds() {
    let obj = draw_object();
    let rec = pack_object_record(&obj, 6, 7);
    assert_eq!(rec.model, obj.model);
    assert_eq!(rec.tint, obj.material.tint);
    assert_eq!(rec.roughness, obj.material.roughness);
    assert_eq!(rec.emissive, obj.material.emissive);
    assert_eq!(rec.metallic, obj.material.metallic);
    assert_eq!(rec.alpha_cutoff, obj.material.alpha_cutoff);
    assert_eq!(rec.bb_min, obj.bb_min);
    assert_eq!(rec.bb_max, obj.bb_max);
    assert_eq!(rec.cull_distance, obj.cull_distance);
    assert_eq!(rec.params_index, obj.material.params_index);
    assert_eq!(rec.far_field, obj.material.far_field);
}

#[test]
fn pack_object_record_leaves_unset_maps_and_band_empty() {
    // The trailing index slots come straight from the material; the
    // sample material leaves the emissive + ORM map indices unset (0),
    // which the shader reads as "no map", and carries no far-field band.
    let rec = pack_object_record(&draw_object(), 0, 1);
    assert_eq!(rec.emissive_map_index, 0);
    assert_eq!(rec.orm_map_index, 0);
    assert_eq!(rec.far_field, FarFieldTint::NONE);
}

#[test]
fn a_terrain_material_carries_its_band_into_the_record() {
    let mut obj = draw_object();
    obj.material.far_field = FarFieldTint::of(0x0001_0201);
    assert_eq!(
        pack_object_record(&obj, 0, 1).far_field,
        obj.material.far_field
    );
}

#[test]
fn gpu_draw_args_layout_matches_shaders() {
    // The `GpuDrawArgs` struct in `cull.hlsl` and `cull_encode.metal` is
    // four tightly packed uints; the kernel reads garbage if the Rust
    // record drifts from it.
    assert_eq!(size_of::<GpuDrawArgs>(), 16);
    assert_eq!(offset_of!(GpuDrawArgs, index_count), 0);
    assert_eq!(offset_of!(GpuDrawArgs, index_offset), 4);
    assert_eq!(offset_of!(GpuDrawArgs, base_vertex), 8);
    assert_eq!(offset_of!(GpuDrawArgs, flags), 12);
}

#[test]
fn draw_args_flags_packs_cull_decision() {
    // A visible, resident, cullable object: both bits set.
    assert_eq!(
        draw_args_flags(true, true, true),
        DrawArgsFlags::ENABLED | DrawArgsFlags::CULLABLE
    );
    // Non-cullable (e.g. a room): enabled but never frustum-tested.
    assert_eq!(draw_args_flags(true, true, false), DrawArgsFlags::ENABLED);
    // Hidden or not-yet-streamed objects clear ENABLED so the kernel
    // resets their indirect command.
    assert_eq!(draw_args_flags(false, true, true), DrawArgsFlags::CULLABLE);
    assert_eq!(draw_args_flags(true, false, true), DrawArgsFlags::CULLABLE);
    assert_eq!(draw_args_flags(false, false, false), 0);
    // The history bit is packed by the caller, never by the cull-decision
    // helper, and collides with neither decision bit nor the bucket field.
    assert_eq!(draw_args_no_history(), 4);
    let flags =
        draw_args_flags(true, true, true) | draw_args_no_history() | draw_args_bucket_bits(3);
    assert_eq!(flags & DrawArgsFlags::NO_HISTORY, DrawArgsFlags::NO_HISTORY);
    assert_eq!(flags & DrawArgsFlags::ENABLED, DrawArgsFlags::ENABLED);
    assert_eq!(flags & DrawArgsFlags::CULLABLE, DrawArgsFlags::CULLABLE);
    assert_eq!((flags >> DrawArgsFlags::BUCKET_SHIFT) & 0xFF, 3);
}

// The bucket rides bits 8..16 and never collides with the cull-decision bits,
// so the two ORed together survive a round trip through the cull kernels'
// shift + mask.
#[test]
fn bucket_bits_ride_above_the_cull_decision_bits() {
    assert_eq!(draw_args_bucket_bits(0), 0);
    assert_eq!(draw_args_bucket_bits(1), 1 << 8);
    assert_eq!(draw_args_bucket_bits(3), 3 << 8);
    // A bucket past the cap clamps to the last routable one.
    let last = u32::saturating_from(MAX_SHADER_BUCKETS - 1);
    assert_eq!(
        draw_args_bucket_bits(u32::saturating_from(MAX_SHADER_BUCKETS)),
        last << 8
    );
    assert_eq!(draw_args_bucket_bits(u32::MAX), last << 8);
    // Packed alongside the decision bits, each half reads back unchanged.
    let flags = draw_args_flags(true, true, true) | draw_args_bucket_bits(2);
    assert_eq!(flags & DrawArgsFlags::ENABLED, DrawArgsFlags::ENABLED);
    assert_eq!(flags & DrawArgsFlags::CULLABLE, DrawArgsFlags::CULLABLE);
    assert_eq!((flags >> DrawArgsFlags::BUCKET_SHIFT) & 0xFF, 2);
}

// One authored quality knob scales both shadow kinds: a spot slice is a
// quarter of the directional map, bounded so a very low or very high
// setting still allocates a usable array.
#[test]
fn a_spot_shadow_slice_is_a_bounded_quarter_of_the_directional_map() {
    assert_eq!(spot_shadow_slice_size(4096), 1024);
    assert_eq!(spot_shadow_slice_size(2048), 512);
    assert_eq!(spot_shadow_slice_size(1024), 256);
    // Clamped at both ends rather than scaling without limit.
    assert_eq!(spot_shadow_slice_size(64), 256, "floored");
    assert_eq!(spot_shadow_slice_size(u32::MAX), 1024, "ceilinged");
}

// Shadows off still allocates one slice, and views must not claim the
// cascade count the disabled array never got.
#[test]
fn an_empty_shadow_array_is_one_texel_and_one_slice() {
    let cascades = u32::saturating_from(NUM_SHADOW_CASCADES);
    let one = ShadowArrayExtent { size: 1, layers: 1 };
    assert_eq!(shadow_array_extent(0, cascades), one, "shadows off");
    assert_eq!(shadow_array_extent(512, 0), one, "no slices");
    assert_eq!(
        shadow_array_extent(2048, cascades),
        ShadowArrayExtent {
            size: 2048,
            layers: cascades
        }
    );
}

// A degenerate cluster AABB disables culling rather than culling
// everything: a non-finite bound cannot be compared against a frustum.
#[test]
fn a_cluster_is_cullable_only_with_finite_bounds() {
    let cluster = |min: [f32; 3], max: [f32; 3]| InstancedCluster {
        vertex_offset: 0,
        vertex_count: 0,
        index_offset: 0,
        index_count: 0,
        texture_slot: 0,
        normal_map_slot: 0,
        material: MaterialUniforms::DEFAULT,
        cluster_bb_min: min,
        cluster_bb_max: max,
        local_bb_min: [-1.0; 3],
        local_bb_max: [1.0; 3],
        cull_distance: 0.0,
        instances: Vec::new(),
        lod_alternates: Vec::new(),
    };
    assert!(cluster([-1.0; 3], [1.0; 3]).cullable());
    assert!(!cluster([f32::NAN; 3], [1.0; 3]).cullable());
    assert!(!cluster([-1.0; 3], [f32::INFINITY; 3]).cullable());
}

// With no alternates every distance picks the object's own slice, so a
// skinned mesh that declared one LOD renders the same however far it is.
#[test]
fn a_skinned_object_without_alternates_keeps_its_own_slice() {
    let obj = crate::test_support::skinned_draw_object();
    let expected = (obj.index_offset, obj.index_count);
    assert_eq!(obj.active_lod(0.0), expected);
    assert_eq!(obj.active_lod(1000.0), expected);
}
