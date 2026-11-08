//! The ray-traced reflections' bindings: the classify and the trace, which
//! share one root signature.

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
    /// The bindless texture pool, for the textured trace.
    pub const TEX_POOL: Register<Srv> = Register::new("tex_pool", 0).in_space(1);
    /// The reflections written.
    pub const OUTPUT: Register<Uav> = Register::new("output", 0);
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
}
