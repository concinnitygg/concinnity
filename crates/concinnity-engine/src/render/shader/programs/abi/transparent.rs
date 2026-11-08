//! The transparent producers' bindings, glass and water, across the base and
//! ray-traced root signatures.

/// The DirectX registers.
pub mod dx {
    use crate::render::shader::programs::abi::register::{Cbv, Register, Sampler, Srv};

    /// The transparent view block.
    pub const VIEW: Register<Cbv> = Register::new("view", 0);
    /// The record's block, which water's vertex stage reads too.
    pub const PARAMS: Register<Cbv> = Register::new("params", 1);
    /// The cascaded shadow block.
    pub const SHADOW: Register<Cbv> = Register::new("shadow_cb", 3);
    /// The live probe count.
    pub const PROBE_SET: Register<Cbv> = Register::new("probe_set", 4);
    /// The ray-traced producers' block.
    pub const RT_PARAMS: Register<Cbv> = Register::new("rt_params", 5);
    /// The main camera's cluster grid parameters.
    pub const CLUSTER: Register<Cbv> = Register::new("cluster", 6);
    /// The scene copy behind the surface.
    pub const SCENE_COLOR: Register<Srv> = Register::new("scene_color", 0);
    /// The scene depth.
    pub const SCENE_DEPTH: Register<Srv> = Register::new("scene_depth", 1);
    /// The sky prefilter cube, where no probe covers.
    pub const PREFILTER_CUBE: Register<Srv> = Register::new("prefilter_cube", 2);
    /// The record's planar mirror resolve.
    pub const PLANAR_REFLECTION: Register<Srv> = Register::new("planar_reflection", 3);
    /// The scene's acceleration structure.
    pub const SCENE_TLAS: Register<Srv> = Register::new("scene_tlas", 4);
    /// The static vertices.
    pub const VERTICES: Register<Srv> = Register::new("verts", 5);
    /// The static indices.
    pub const INDICES: Register<Srv> = Register::new("indices", 6);
    /// The deformed skinned vertices.
    pub const SKINNED_VERTICES: Register<Srv> = Register::new("sverts", 8);
    /// The skinned indices.
    pub const SKINNED_INDICES: Register<Srv> = Register::new("sidx", 9);
    /// The geometry table.
    pub const GEOMETRY: Register<Srv> = Register::new("geom", 10);
    /// The front glass reflection layer, with the back one in the register after.
    pub const GLASS_REFLECTION: Register<Srv> = Register::new("glass_reflection", 11);
    /// The back glass reflection layer.
    pub const GLASS_REFLECTION_BACK: Register<Srv> = Register::new("glass_reflection_back", 12);
    /// The cascaded shadow map, with the irradiance cube in the register after.
    pub const SHADOW_MAP: Register<Srv> = Register::new("shadow_map", 13);
    /// The sky irradiance cube.
    pub const IRRADIANCE_CUBE: Register<Srv> = Register::new("irradiance_cube", 14);
    /// The oceans' displacement arrays, with the slopes and a sea's wake in the registers after.
    pub const OCEAN_DISPLACEMENT: Register<Srv> = Register::new("ocean_displacement", 15);
    /// The oceans' slope arrays.
    pub const OCEAN_SLOPES: Register<Srv> = Register::new("ocean_slopes", 16);
    /// A planet sea's wake field.
    pub const WAKE_FIELD: Register<Srv> = Register::new("wake_field", 17);
    /// The clouds' shadow map.
    pub const CLOUD_SHADOW: Register<Srv> = Register::new("cloud_shadow", 18);
    /// The reflection-probe cube array.
    pub const PROBE_CUBES: Register<Srv> = Register::new("probe_cubes", 20);
    /// The reflection-probe records.
    pub const PROBE_RECORDS: Register<Srv> = Register::new("probe_records", 21);
    /// The per-cluster lists.
    pub const CLUSTER_LIST: Register<Srv> = Register::new("cluster_list", 22);
    /// The bindless texture pool, for the textured producers.
    pub const TEX_POOL: Register<Srv> = Register::new("tex_pool", 0).in_space(1);
    /// The scene copy's and depth's sampler.
    pub const POST_SAMPLER: Register<Sampler> = Register::new("post_samp", 0);
    /// The pool's sampler.
    pub const POOL_SAMPLER: Register<Sampler> = Register::new("pool_sampler", 1);
    /// The cubes' sampler.
    pub const CUBE_SAMPLER: Register<Sampler> = Register::new("cube_sampler", 2);
    /// The shadow comparison sampler.
    pub const SHADOW_SAMPLER: Register<Sampler> = Register::new("shadow_sampler", 3);
    /// The oceans' sampler.
    pub const OCEAN_SAMPLER: Register<Sampler> = Register::new("ocean_sampler", 4);
}

/// The Metal slots.
pub mod metal {
    use crate::render::shader::programs::abi::slot::{Buffer, Sampler, Slot, Texture};

    /// The ray-traced producers' block.
    pub const RT_PARAMS: Slot<Buffer> = Slot::new("rt_params", 0);
    /// The static vertices.
    pub const VERTICES: Slot<Buffer> = Slot::new("verts", 1);
    /// The static indices.
    pub const INDICES: Slot<Buffer> = Slot::new("indices", 2);
    /// The geometry table.
    pub const GEOMETRY: Slot<Buffer> = Slot::new("geom", 3);
    /// The scene's acceleration structure.
    pub const SCENE_TLAS: Slot<Buffer> = Slot::new("scene_tlas", 4);
    /// The transparent view block.
    pub const VIEW: Slot<Buffer> = Slot::new("view", 5);
    /// The record's block, on both stages.
    pub const PARAMS: Slot<Buffer> = Slot::new("params", 6);
    /// The live probe count.
    pub const PROBE_SET: Slot<Buffer> = Slot::new("probe_set", 7);
    /// The deformed skinned vertices.
    pub const SKINNED_VERTICES: Slot<Buffer> = Slot::new("sverts", 8);
    /// The skinned indices.
    pub const SKINNED_INDICES: Slot<Buffer> = Slot::new("sidx", 9);
    /// The bindless pool's argument buffer, for the textured producers.
    pub const TEXTURES: Slot<Buffer> = Slot::new("spvDescriptorSet6", 10);
    /// The reflection-probe records.
    pub const PROBE_RECORDS: Slot<Buffer> = Slot::new("probe_records", 11);
    /// The main camera's cluster grid parameters.
    pub const CLUSTER: Slot<Buffer> = Slot::new("cluster", 12);
    /// The per-cluster lists.
    pub const CLUSTER_LIST: Slot<Buffer> = Slot::new("cluster_list", 13);
    /// The cascaded shadow block.
    pub const SHADOW: Slot<Buffer> = Slot::new("shadow_cb", 14);
    /// The scene copy behind the surface.
    pub const SCENE_COLOR: Slot<Texture> = Slot::new("scene_color", 0);
    /// The scene depth.
    pub const SCENE_DEPTH: Slot<Texture> = Slot::new("scene_depth", 1);
    /// The sky prefilter cube.
    pub const PREFILTER_CUBE: Slot<Texture> = Slot::new("prefilter_cube", 2);
    /// The record's planar mirror resolve.
    pub const PLANAR_REFLECTION: Slot<Texture> = Slot::new("planar_reflection", 3);
    /// The front glass reflection layer.
    pub const GLASS_REFLECTION: Slot<Texture> = Slot::new("glass_reflection", 4);
    /// The back glass reflection layer.
    pub const GLASS_REFLECTION_BACK: Slot<Texture> = Slot::new("glass_reflection_back", 5);
    /// The reflection-probe cube array.
    pub const PROBE_CUBES: Slot<Texture> = Slot::new("probe_cubes", 6);
    /// The cascaded shadow map.
    pub const SHADOW_MAP: Slot<Texture> = Slot::new("shadow_map", 7);
    /// The sky irradiance cube.
    pub const IRRADIANCE_CUBE: Slot<Texture> = Slot::new("irradiance_cube", 8);
    /// The oceans' displacement arrays.
    pub const OCEAN_DISPLACEMENT: Slot<Texture> = Slot::new("ocean_displacement", 9);
    /// The oceans' slope arrays.
    pub const OCEAN_SLOPES: Slot<Texture> = Slot::new("ocean_slopes", 10);
    /// A planet sea's wake field.
    pub const WAKE_FIELD: Slot<Texture> = Slot::new("wake_field", 11);
    /// The clouds' shadow map.
    pub const CLOUD_SHADOW: Slot<Texture> = Slot::new("cloud_shadow", 12);
    /// The scene copy's sampler.
    pub const SCENE_COLOR_SAMPLER: Slot<Sampler> = Slot::new("scene_color_sampler", 0);
    /// The prefilter cube's sampler, with the probe array's after it.
    pub const PREFILTER_CUBE_SAMPLER: Slot<Sampler> = Slot::new("prefilter_cube_sampler", 1);
    /// The probe cube array's sampler.
    pub const PROBE_CUBE_SAMPLER: Slot<Sampler> = Slot::new("probe_cube_sampler", 2);
    /// The planar resolve's sampler.
    pub const PLANAR_REFLECTION_SAMPLER: Slot<Sampler> = Slot::new("planar_reflection_sampler", 3);
    /// The pool's sampler.
    pub const POOL_SAMPLER: Slot<Sampler> = Slot::new("pool_sampler", 4);
    /// The shadow comparison sampler.
    pub const SHADOW_SAMPLER: Slot<Sampler> = Slot::new("shadow_sampler", 5);
    /// The oceans' sampler.
    pub const OCEAN_SAMPLER: Slot<Sampler> = Slot::new("ocean_sampler", 6);
}
