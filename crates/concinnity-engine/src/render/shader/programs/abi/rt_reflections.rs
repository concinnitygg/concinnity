//! The ray-traced reflections' bindings: the classify and the trace, which
//! share one root signature.

use super::group::ThreadGroup;
use crate::render::post::rt_reflections as rt;

/// The classify's group.
pub const CLASSIFY_GROUP: ThreadGroup = ThreadGroup::new(rt::CLASSIFY_GROUP, rt::CLASSIFY_GROUP, 1);
/// The trace's group.
pub const TRACE_GROUP: ThreadGroup = ThreadGroup::new(rt::TRACE_GROUP, 1, 1);

/// The DirectX registers.
pub mod dx {
    use crate::render::shader::programs::abi::register::{Cbv, Register, Sampler, Srv, Uav};

    /// The pass's block.
    pub const PARAMS: Register<Cbv> = Register::new("rt_params", 0);
    /// The live probe count.
    pub const PROBE_SET: Register<Cbv> = Register::new("probe_set", 4);
    /// The cluster grid's parameters.
    pub const CLUSTER: Register<Cbv> = Register::new("cluster", 5);
    /// The far field's canopy block.
    pub const CANOPY: Register<Cbv> = Register::new("canopy_cb", 6);
    /// The scene's acceleration structure.
    pub const SCENE_TLAS: Register<Srv> = Register::new("scene_tlas", 0);
    /// The static vertices.
    pub const VERTICES: Register<Srv> = Register::new("verts", 1);
    /// The static indices.
    pub const INDICES: Register<Srv> = Register::new("indices", 2);
    /// The geometry table.
    pub const GEOMETRY: Register<Srv> = Register::new("geom", 3);
    /// The scene color.
    pub const SCENE: Register<Srv> = Register::new("scene_tex", 4);
    /// The G-buffer normal and depth.
    pub const GBUFFER: Register<Srv> = Register::new("gbuffer", 5);
    /// The roughness.
    pub const ROUGHNESS: Register<Srv> = Register::new("rough_tex", 6);
    /// The sky prefilter cube.
    pub const PREFILTER: Register<Srv> = Register::new("prefilter", 7);
    /// The deformed skinned vertices.
    pub const SKINNED_VERTICES: Register<Srv> = Register::new("sverts", 8);
    /// The skinned indices.
    pub const SKINNED_INDICES: Register<Srv> = Register::new("sidx", 9);
    /// The reflection-probe cube array.
    pub const PROBE_CUBES: Register<Srv> = Register::new("probe_cubes", 10);
    /// The reflection-probe records.
    pub const PROBE_RECORDS: Register<Srv> = Register::new("probe_records", 11);
    /// The per-cluster lists.
    pub const CLUSTER_LIST: Register<Srv> = Register::new("cluster_list", 12);
    /// The canopy's view tables.
    pub const CANOPY_TABLES: Register<Srv> = Register::new("canopy_tables_sb", 13);
    /// SSGI's pyramid, which the classify reads.
    pub const PYRAMID: Register<Srv> = Register::new("pyramid", 14);
    /// The ray list, which the trace reads.
    pub const RAY_LIST: Register<Srv> = Register::new("ray_list", 15);
    /// The ray arguments, which the trace reads.
    pub const RAY_ARGS: Register<Srv> = Register::new("ray_args", 16);
    /// The waves' slope cascades, which the trace reads.
    pub const WAVE_SLOPES: Register<Srv> = Register::new("wave_slopes", 18);
    /// The rays the trace lists for the sky's own pass.
    pub const MISSES_OUT: Register<Uav> = Register::new("reflection_misses", 3);
    /// The listed rays' indirect arguments.
    pub const MISS_ARGS_OUT: Register<Uav> = Register::new("reflection_miss_args", 4);
    /// The bindless texture pool, for the textured trace.
    pub const TEX_POOL: Register<Srv> = Register::new("tex_pool", 0).in_space(1);
    /// The reflections written.
    pub const OUTPUT: Register<Uav> = Register::new("output", 0);
    /// The reflected hits' distances and deferred skies.
    pub const HITS_OUT: Register<Uav> = Register::new("hit_out", 5);
    /// The ray list, which the classify writes.
    pub const RAY_LIST_OUT: Register<Uav> = Register::new("ray_list", 1);
    /// The ray arguments, which the classify writes.
    pub const RAY_ARGS_OUT: Register<Uav> = Register::new("ray_args", 2);
    /// The screen sources' sampler.
    pub const SCREEN_SAMPLER: Register<Sampler> = Register::new("screen_sampler", 0);
    /// The prefilter cube's sampler.
    pub const CUBE_SAMPLER: Register<Sampler> = Register::new("cube_sampler", 1);
    /// The pool's sampler.
    pub const POOL_SAMPLER: Register<Sampler> = Register::new("pool_sampler", 2);
    /// The probe cube array's sampler.
    pub const PROBE_CUBE_SAMPLER: Register<Sampler> = Register::new("probe_cube_sampler", 3);
    /// The waves' sampler.
    pub const WAVE_SAMPLER: Register<Sampler> = Register::new("wave_sampler", 4);
}

/// The Metal slots.
pub mod metal {
    use crate::render::shader::programs::abi::slot::{Buffer, Sampler, Slot, Texture};

    /// The pass's block.
    pub const PARAMS: Slot<Buffer> = Slot::new("rt_params", 0);
    /// The static vertices.
    pub const VERTICES: Slot<Buffer> = Slot::new("verts", 1);
    /// The static indices.
    pub const INDICES: Slot<Buffer> = Slot::new("indices", 2);
    /// The geometry table.
    pub const GEOMETRY: Slot<Buffer> = Slot::new("geom", 3);
    /// The scene's acceleration structure.
    pub const SCENE_TLAS: Slot<Buffer> = Slot::new("scene_tlas", 4);
    /// The deformed skinned vertices.
    pub const SKINNED_VERTICES: Slot<Buffer> = Slot::new("sverts", 5);
    /// The skinned indices.
    pub const SKINNED_INDICES: Slot<Buffer> = Slot::new("sidx", 6);
    /// The bindless pool's argument buffer, for the textured trace.
    pub const TEXTURES: Slot<Buffer> = Slot::new("spvDescriptorSet3", 7);
    /// The live probe count.
    pub const PROBE_SET: Slot<Buffer> = Slot::new("probe_set", 8);
    /// The cluster grid's parameters.
    pub const CLUSTER: Slot<Buffer> = Slot::new("cluster", 9);
    /// The per-cluster lists.
    pub const CLUSTER_LIST: Slot<Buffer> = Slot::new("cluster_list", 10);
    /// The reflection-probe records.
    pub const PROBE_RECORDS: Slot<Buffer> = Slot::new("probe_records", 11);
    /// The far field's canopy block.
    pub const CANOPY: Slot<Buffer> = Slot::new("canopy_cb", 12);
    /// The canopy's view tables.
    pub const CANOPY_TABLES: Slot<Buffer> = Slot::new("canopy_tables_sb", 13);
    /// The ray list between the kernels.
    pub const RAY_LIST: Slot<Buffer> = Slot::new("ray_list", 14);
    /// The ray arguments between the kernels.
    pub const RAY_ARGS: Slot<Buffer> = Slot::new("ray_args", 15);
    /// The rays the trace lists for the sky's own pass.
    pub const MISSES_OUT: Slot<Buffer> = Slot::new("reflection_misses", 16);
    /// The listed rays' indirect arguments.
    pub const MISS_ARGS_OUT: Slot<Buffer> = Slot::new("reflection_miss_args", 17);
    /// The scene color.
    pub const SCENE: Slot<Texture> = Slot::new("scene_tex", 0);
    /// The G-buffer normal and depth.
    pub const GBUFFER: Slot<Texture> = Slot::new("gbuffer", 1);
    /// The roughness.
    pub const ROUGHNESS: Slot<Texture> = Slot::new("rough_tex", 2);
    /// The sky prefilter cube.
    pub const PREFILTER: Slot<Texture> = Slot::new("prefilter", 3);
    /// The reflection-probe cube array.
    pub const PROBE_CUBES: Slot<Texture> = Slot::new("probe_cubes", 4);
    /// SSGI's pyramid, which the classify reads.
    pub const PYRAMID: Slot<Texture> = Slot::new("pyramid", 5);
    /// The reflections written.
    pub const OUTPUT: Slot<Texture> = Slot::new("_output", 6);
    /// The waves' slope cascades, which the trace reads.
    pub const WAVE_SLOPES: Slot<Texture> = Slot::new("wave_slopes", 7);
    /// The reflected hits' distances and deferred skies.
    pub const HITS_OUT: Slot<Texture> = Slot::new("hit_out", 8);
    /// The scene's sampler.
    pub const SCENE_SAMPLER: Slot<Sampler> = Slot::new("scene_tex_sampler", 0);
    /// The G-buffer's sampler.
    pub const GBUFFER_SAMPLER: Slot<Sampler> = Slot::new("gbuffer_sampler", 1);
    /// The roughness's sampler.
    pub const ROUGHNESS_SAMPLER: Slot<Sampler> = Slot::new("rough_tex_sampler", 2);
    /// The prefilter cube's sampler.
    pub const PREFILTER_SAMPLER: Slot<Sampler> = Slot::new("prefilter_sampler", 3);
    /// The probe cube array's sampler.
    pub const PROBE_CUBE_SAMPLER: Slot<Sampler> = Slot::new("probe_cube_sampler", 4);
    /// The pool's sampler.
    pub const POOL_SAMPLER: Slot<Sampler> = Slot::new("pool_sampler", 5);
    /// The waves' sampler.
    pub const WAVE_SAMPLER: Slot<Sampler> = Slot::new("wave_sampler", 6);
}
