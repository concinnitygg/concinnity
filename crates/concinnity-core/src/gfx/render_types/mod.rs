//! The `repr(C)` GPU layouts every backend and the render prep share with the shaders.

mod draw;
mod light;
mod material;
mod overlay;
mod post;
mod shadow;

#[cfg(test)]
pub(crate) use draw::DrawArgsFlags;
pub use draw::{
    DrawIndex, DrawObject, GpuDrawArgs, GpuObjectData, GpuTileMorph, InstancedCluster, LodSlice,
    MAX_CLONE_DRAWS, MAX_DRAW_BUCKETS, MAX_JOINTS, MAX_SHADER_BUCKETS, PlanetTileDraw,
    SkinnedDrawObject, SkinnedIndex, clone_reserve, draw_args_bucket_bits, draw_args_flags,
    draw_args_no_history, instance_object_records, pack_instance_record, pack_object_record,
    pack_skinned_record, runtime_reserve_full,
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

#[cfg(test)]
mod tests;
