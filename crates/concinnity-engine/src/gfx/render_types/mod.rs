//! The `repr(C)` GPU layouts every backend and the render prep share with the shaders.

mod draw;
mod light;
mod material;
mod overlay;
mod post;
mod shadow;
mod streams;
mod variants;

#[cfg(test)]
pub(crate) use draw::DrawArgsFlags;
pub use draw::{
    DrawIndex, DrawObject, GpuDrawArgs, GpuObjectData, GpuTileMorph, InstancedCluster, LodSlice,
    MAX_CLONE_DRAWS, MAX_JOINTS, MAX_SHADER_BUCKETS, PlanetTileDraw, SkinnedDrawObject,
    SkinnedIndex, TileMorphView, clone_reserve, draw_args_bucket_bits, draw_args_flags,
    draw_args_no_history, draw_args_planet_tile, draw_args_route_bits, instance_object_records,
    pack_instance_record, pack_object_record, pack_skinned_record, runtime_reserve_full,
};
pub use light::{
    AreaLightData, CLUSTER_COUNT, CLUSTER_LIST_LEN, ClusterCamera, ClusterParams,
    DirectionalLightData, GpuLight, LIGHT_KIND_AREA, LIGHT_KIND_POINT, LIGHT_KIND_SPOT,
    LightUniforms, MAX_AREA_LIGHTS, MAX_DIRECTIONAL_LIGHTS, MAX_LOCAL_LIGHTS, MAX_POINT_LIGHTS,
    PointLightData,
};
pub use material::{
    FALLBACK_TEXTURE_COUNT, FarFieldTint, GpuMaterialParams, MATERIAL_PARAM_COUNT,
    MaterialUniforms, NO_ALBEDO_SLOT, NO_NORMAL_MAP_SLOT, albedo_pool_index, normal_pool_index,
};
pub use overlay::{LineVertex, TextDrawCall, TextUniforms, TextVertex};
pub use post::{
    CompositeParams, FogFroxelParams, FogParams, ParticleParams, PostProcessParams,
    PostProcessTunables, RtGeomEntry, RtParams, SsaoParams, SsgiParams, SsrParams,
};
pub use shadow::{
    MAX_SHADOWED_SPOTS, NUM_SHADOW_CASCADES, ShadowArrayExtent, ShadowPassPush, ShadowUniforms,
    SpotShadowData, shadow_array_extent, spot_shadow_slice_size,
};
pub use streams::{
    COUNT_SLOT_ALIGN, COUNT_SLOT_BYTES, CULL_GROUP_RECORDS, DRAW_STREAMS, GpuCompactParams,
    GpuExecutionRange, SKINNED_STREAM, StreamLayout, StreamSlots, compact, count_slot_offset,
    cull_groups, culled_records, group_count_index, group_count_words, group_counts,
    scan_group_counts,
};
pub use variants::{
    BucketSpans, DAYLIGHT_COS_MARGIN, GpuBucketRegion, GpuCullVariants, GpuVariantRoute,
    MAX_VARIANT_ROUTES, VARIANT_DAYLIGHT, VARIANT_FAR, VARIANT_FEATURE_COMBINATIONS,
    bucket_regions, region_commands,
};

#[cfg(test)]
mod tests;
