//! The GPU-driven main pass's bindings, which its G-buffer pre-pass and the
//! grass draws share.

/// signatures.
pub mod dx {
    use crate::render::shader::programs::abi::register::{Cbv, Register, Sampler, Srv};

    /// The per-draw object id, a root constant the cull's commands set.
    pub const OBJECT_ID: Register<Cbv> = Register::new("objid_cb", 0);
    /// The view block.
    pub const VIEW: Register<Cbv> = Register::new("view_cb", 1);
    /// The light block.
    pub const LIGHTS: Register<Cbv> = Register::new("lights_cb", 2);
    /// The cascaded shadow block.
    pub const SHADOW: Register<Cbv> = Register::new("shadow_cb", 3);
    /// The live reflection-probe count.
    pub const PROBE_SET: Register<Cbv> = Register::new("probe_set_cb", 4);
    /// The cluster grid's parameters.
    pub const CLUSTER: Register<Cbv> = Register::new("cluster_cb", 5);
    /// The pre-pass's own view block.
    pub const GBUFFER_VIEW: Register<Cbv> = Register::new("gb_view", 6);
    /// The grass block.
    pub const GRASS: Register<Cbv> = Register::new("grass", 7);
    /// The far field's canopy block.
    pub const CANOPY: Register<Cbv> = Register::new("canopy_cb", 8);

    /// The cascaded shadow map array.
    pub const SHADOW_MAP: Register<Srv> = Register::new("shadow_map", 0);
    /// The scene's local lights.
    pub const LOCAL_LIGHTS: Register<Srv> = Register::new("local_lights_sb", 1);
    /// The per-cluster light lists.
    pub const CLUSTER_LIST: Register<Srv> = Register::new("cluster_list_sb", 2);
    /// The frame's records.
    pub const OBJECTS: Register<Srv> = Register::new("objects_sb", 3);
    /// The screen-space indirect light.
    pub const SCREEN_INDIRECT: Register<Srv> = Register::new("screen_indirect", 4);
    /// The sky irradiance cube, with the prefilter cube in the register after.
    pub const IRRADIANCE_CUBE: Register<Srv> = Register::new("irradiance_cube", 5);
    /// The sky prefilter cube.
    pub const PREFILTER_CUBE: Register<Srv> = Register::new("prefilter_cube", 6);
    /// The reflection-probe cube array.
    pub const PROBE_CUBES: Register<Srv> = Register::new("probe_cubes", 7);
    /// The reflection-probe records.
    pub const PROBE_RECORDS: Register<Srv> = Register::new("probe_records_sb", 8);
    /// The shared vertex stream.
    pub const MESH_VERTICES: Register<Srv> = Register::new("mesh_vertices", 9);
    /// The spot shadow projections.
    pub const SPOT_SHADOWS: Register<Srv> = Register::new("spot_shadows_sb", 15);
    /// The spot shadow depth array.
    pub const SPOT_SHADOW_MAP: Register<Srv> = Register::new("spot_shadow_map", 16);
    /// The rect area lights.
    pub const AREA_LIGHTS: Register<Srv> = Register::new("area_lights_sb", 17);
    /// The LTC matrix table, with the magnitude table in the register after.
    pub const LTC_MATRIX: Register<Srv> = Register::new("ltc_matrix", 18);
    /// The LTC magnitude table.
    pub const LTC_MAGNITUDE: Register<Srv> = Register::new("ltc_magnitude", 19);
    /// The material parameter table.
    pub const MATERIAL_PARAMS: Register<Srv> = Register::new("material_params_sb", 20);
    /// The previous frame's model history, which the pre-pass reads.
    pub const PREV_MODELS: Register<Srv> = Register::new("prev_models", 21);
    /// This frame's draw arguments, which the pre-pass reads.
    pub const DRAW_ARGS: Register<Srv> = Register::new("draw_args", 22);
    /// The visible grass blades.
    pub const GRASS_BLADES: Register<Srv> = Register::new("grass_blades", 23);
    /// The canopy's view tables.
    pub const CANOPY_TABLES: Register<Srv> = Register::new("canopy_tables_sb", 24);
    /// The clouds' shadow map.
    pub const CLOUD_SHADOW: Register<Srv> = Register::new("cloud_shadow", 25);
    /// The far shadow map past the cascades.
    pub const FAR_SHADOW_MAP: Register<Srv> = Register::new("far_shadow_map", 10);
    /// The unbounded bindless texture pool.
    pub const TEX_POOL: Register<Srv> = Register::new("tex_pool", 0).in_space(1);

    /// The shadow comparison sampler.
    pub const SHADOW_SAMPLER: Register<Sampler> = Register::new("shadow_sampler", 0);
    /// The linear repeat sampler, with the cube sampler in the register after.
    pub const LINEAR_SAMPLER: Register<Sampler> = Register::new("linear_sampler", 1);
    /// The cube sampler.
    pub const CUBE_SAMPLER: Register<Sampler> = Register::new("cube_sampler", 2);
}

/// The Metal slots the main pass, its pre-pass and the grass draws bind,
/// encoder-bound so the ICB-executed draws inherit them.
pub mod metal {
    use crate::render::shader::programs::abi::slot::{Buffer, Slot};

    /// The view block.
    pub const VIEW: Slot<Buffer> = Slot::new("view_cb", 0);
    /// The pre-pass's own view block.
    pub const GBUFFER_VIEW: Slot<Buffer> = Slot::new("gb_view", 3);
    /// The light block.
    pub const LIGHTS: Slot<Buffer> = Slot::new("lights_cb", 4);
    /// The cascaded shadow block.
    pub const SHADOW: Slot<Buffer> = Slot::new("shadow_cb", 5);
    /// The live reflection-probe count.
    pub const PROBE_SET: Slot<Buffer> = Slot::new("probe_set_cb", 6);
    /// The bindless texture argument buffer.
    pub const TEXTURES: Slot<Buffer> = Slot::new("spvDescriptorSet1", 7);
    /// The scene's local lights.
    pub const LOCAL_LIGHTS: Slot<Buffer> = Slot::new("local_lights_sb", 8);
    /// The frame's records.
    pub const OBJECTS: Slot<Buffer> = Slot::new("objects_sb", 9);
    /// The bindless sampler argument buffer.
    pub const SAMPLERS: Slot<Buffer> = Slot::new("spvDescriptorSet2", 10);
    /// The cluster grid's parameters.
    pub const CLUSTER: Slot<Buffer> = Slot::new("cluster_cb", 11);
    /// The per-cluster light lists.
    pub const CLUSTER_LIST: Slot<Buffer> = Slot::new("cluster_list_sb", 12);
    /// The spot shadow projections.
    pub const SPOT_SHADOWS: Slot<Buffer> = Slot::new("spot_shadows_sb", 13);
    /// The rect area lights.
    pub const AREA_LIGHTS: Slot<Buffer> = Slot::new("area_lights_sb", 14);
    /// The reflection-probe records.
    pub const PROBE_RECORDS: Slot<Buffer> = Slot::new("probe_records_sb", 15);
    /// The previous frame's model history, which the pre-pass reads.
    pub const PREV_MODELS: Slot<Buffer> = Slot::new("prev_models", 17);
    /// This frame's draw arguments, which the pre-pass reads.
    pub const DRAW_ARGS: Slot<Buffer> = Slot::new("draw_args", 18);
    /// The far field's canopy block.
    pub const CANOPY: Slot<Buffer> = Slot::new("canopy_cb", 21);
    /// The canopy's view tables.
    pub const CANOPY_TABLES: Slot<Buffer> = Slot::new("canopy_tables_sb", 22);
    /// The shared vertex stream.
    pub const MESH_VERTICES: Slot<Buffer> = Slot::new("mesh_vertices", 23);
}
