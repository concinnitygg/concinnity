//! The slots each Metal program's host encoder writes, which the device build
//! checks against the MSL the precompile emits.

use super::MetalAbi;
use crate::render::shader::programs::metal::bindless_textures;
use crate::render::shader::programs::{metal, shared};

// The bindless main fragment's discrete slots, every variant.
const MAIN_BINDLESS_SLOTS: &[(&str, &str)] = &[
    ("view_cb", "buffer(0)"),
    ("lights_cb", "buffer(4)"),
    ("shadow_cb", "buffer(5)"),
    ("probe_set_cb", "buffer(6)"),
    ("probe_records_sb", "buffer(15)"),
    ("spvDescriptorSet1", "buffer(7)"),
    ("local_lights_sb", "buffer(8)"),
    ("objects_sb", "buffer(9)"),
    ("spvDescriptorSet2", "buffer(10)"),
    ("cluster_cb", "buffer(11)"),
    ("cluster_list_sb", "buffer(12)"),
    ("spot_shadows_sb", "buffer(13)"),
    ("area_lights_sb", "buffer(14)"),
];

// The far-field canopy's block and view tables, read by the canopy variant.
const CANOPY_SLOTS: &[(&str, &str)] = &[
    ("canopy_cb", "buffer(21)"),
    ("canopy_tables_sb", "buffer(22)"),
];

// The ground under grass reads the canopy block but none of its view tables.
const GRASS_GROUND_SLOTS: &[(&str, &str)] = &[("canopy_cb", "buffer(21)")];

const MAIN_BINDLESS_ARGUMENT_IDS: &[(&str, usize)] = &[
    ("shadow_map", bindless_textures::SHADOW_MAP),
    ("irradiance_cube", bindless_textures::IRRADIANCE_CUBE),
    ("prefilter_cube", bindless_textures::PREFILTER_CUBE),
    ("screen_indirect", bindless_textures::SCREEN_INDIRECT),
    ("probe_cubes", bindless_textures::PROBE_CUBES),
    ("spot_shadow_map", bindless_textures::SPOT_SHADOW_MAP),
    ("ltc_matrix", bindless_textures::LTC_MATRIX),
    ("ltc_magnitude", bindless_textures::LTC_MAGNITUDE),
    ("cloud_shadow", bindless_textures::CLOUD_SHADOW),
    ("tex_pool", bindless_textures::pool(0)),
    ("tex_sampler", 0),
    ("shadow_sampler", 1),
    ("cube_sampler", 2),
];

/// The Metal programs whose slots their host encoders write by hand, every
/// resource each binds.
pub const METAL_ENTRY_ABI: &[MetalAbi] = &[
    // The draw cull's decision half. Its buffers are pinned by register() and
    // the Hi-Z texture takes texture(0) from declaration order; the slots are
    // what `metal/cull.rs` binds once for this dispatch and the scan and
    // scatter that follow it on the same encoder.
    MetalAbi {
        program: &shared::CULL_EARLY,
        slots: &[&[
            ("objects", "buffer(0)"),
            ("draw_args", "buffer(1)"),
            ("cull", "buffer(2)"),
            ("cull_status", "buffer(5)"),
            ("variants", "buffer(8)"),
            ("group_counts", "buffer(9)"),
            ("hiz_tex", "texture(0)"),
        ]],
        argument_ids: &[],
    },
    // The late pass re-tests candidates only, so it never reads `draw_args`; the
    // encode dispatch after it does, from the same binding.
    MetalAbi {
        program: &shared::CULL_LATE,
        slots: &[&[
            ("objects", "buffer(0)"),
            ("cull", "buffer(2)"),
            ("cull_status", "buffer(5)"),
            ("variants", "buffer(8)"),
            ("group_counts", "buffer(9)"),
            ("hiz_tex", "texture(0)"),
        ]],
        argument_ids: &[],
    },
    // The scan between the decision and the encode, on the same encoder.
    MetalAbi {
        program: &shared::CULL_SCAN,
        slots: &[&[
            ("variants", "buffer(8)"),
            ("group_counts", "buffer(9)"),
            ("stream_counts", "buffer(10)"),
            ("params", "buffer(11)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::CULL_SHADOW,
        slots: &[&[
            ("objects", "buffer(0)"),
            ("draw_args", "buffer(1)"),
            ("cull", "buffer(2)"),
            ("cull_status", "buffer(5)"),
            ("group_counts", "buffer(9)"),
        ]],
        argument_ids: &[],
    },
    // The model-history snapshot: params, the object buffer it reads and the
    // ring slot it writes take buffer(0..2) from declaration order, which is
    // what `metal/model_history.rs` binds.
    MetalAbi {
        program: &shared::MODEL_HISTORY,
        slots: &[&[
            ("params", "buffer(0)"),
            ("objects", "buffer(1)"),
            ("history", "buffer(2)"),
        ]],
        argument_ids: &[],
    },
    // The G-buffer pre-pass, from the main pass's file: the main pass's view
    // block, records, parameter table, texture argument buffer and samplers at
    // their main-pass slots, plus the pre-pass's own view block, model history
    // and draw args, pinned clear of the vertex streams at buffer(1) and
    // buffer(2). The slots are what `metal/post/gbuffer.rs` binds once for every
    // ICB-executed draw.
    MetalAbi {
        program: &shared::MAIN_PREPASS_VERT,
        slots: &[&[
            ("view_cb", "buffer(0)"),
            ("gb_view", "buffer(3)"),
            ("objects_sb", "buffer(9)"),
            ("prev_models", "buffer(17)"),
            ("draw_args", "buffer(18)"),
            ("mesh_vertices", "buffer(23)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::MAIN_PREPASS_FRAG,
        slots: &[&[
            ("gb_view", "buffer(3)"),
            ("spvDescriptorSet1", "buffer(7)"),
            ("objects_sb", "buffer(9)"),
            ("spvDescriptorSet2", "buffer(10)"),
        ]],
        argument_ids: &[("tex_pool", bindless_textures::pool(0)), ("tex_sampler", 0)],
    },
    // The grass kernel, from `metal/grass.rs`: its block, the blades it
    // appends, the draw arguments, and the terrain heights and mask texels it
    // reads, at buffer(0..4) by register, last frame's depth pyramid at
    // texture(5), the bend field, the planet tiles, the shared vertex stream
    // and the tile records at buffer(6..9).
    MetalAbi {
        program: &shared::GRASS_GENERATE,
        slots: &[&[
            ("grass", "buffer(0)"),
            ("blades_out", "buffer(1)"),
            ("draw_args", "buffer(2)"),
            ("terrain_heights", "buffer(3)"),
            ("grass_masks", "buffer(4)"),
            ("grass_hiz", "texture(5)"),
            ("grass_bend_field", "buffer(6)"),
            ("planet_tiles", "buffer(7)"),
            ("mesh_vertices", "buffer(8)"),
            ("grass_tiles", "buffer(9)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::GRASS_BEND,
        slots: &[&[("bend", "buffer(0)"), ("bend_field", "buffer(1)")]],
        argument_ids: &[],
    },
    // The simulated ocean's kernels, from `metal/water_sim.rs`: the constants
    // at buffer(0), the spectrum and row transforms at buffer(5) and (6), and
    // the source and destination mips at texture(0..3), all by register.
    MetalAbi {
        program: &shared::WATER_FFT_ROWS,
        slots: &[&[
            ("params", "buffer(0)"),
            ("spectrum", "buffer(5)"),
            ("rows", "buffer(6)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_WAKE_ROWS,
        slots: &[&[
            ("params", "buffer(0)"),
            ("state", "buffer(1)"),
            ("work", "buffer(2)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_WAKE_COLUMNS,
        slots: &[&[("params", "buffer(0)"), ("work", "buffer(2)")]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_WAKE_INVERSE,
        slots: &[&[
            ("params", "buffer(0)"),
            ("state", "buffer(1)"),
            ("work", "buffer(2)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_WAKE_SHADE,
        slots: &[&[
            ("params", "buffer(0)"),
            ("state", "buffer(1)"),
            ("dst_field", "texture(3)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_WAKE_MIP,
        slots: &[&[("dst_field", "texture(3)"), ("src_field", "texture(4)")]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_FFT_COLUMNS,
        slots: &[&[
            ("params", "buffer(0)"),
            ("rows", "buffer(6)"),
            ("dst_displacement", "texture(2)"),
            ("dst_slopes", "texture(3)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_FFT_COLUMNS_FOAM,
        slots: &[&[
            ("params", "buffer(0)"),
            ("rows", "buffer(6)"),
            ("dst_displacement", "texture(2)"),
            ("dst_slopes", "texture(3)"),
            ("foam", "buffer(7)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_MIP,
        slots: &[&[
            ("params", "buffer(0)"),
            ("src_displacement", "texture(0)"),
            ("src_slopes", "texture(1)"),
            ("dst_displacement", "texture(2)"),
            ("dst_slopes", "texture(3)"),
        ]],
        argument_ids: &[],
    },
    // The cascade's blades draw in the shadow pass under their own pipeline,
    // which binds only the shadow block and the cascade's blades.
    MetalAbi {
        program: &shared::GRASS_SHADOW_VERT,
        slots: &[&[("grass", "buffer(0)"), ("grass_blades", "buffer(1)")]],
        argument_ids: &[],
    },
    // The grass draws at the tail of the pre-pass and the main pass: the
    // pass's view blocks where the surfaces left them, and the grass block and
    // blades at buffer(19) and buffer(20), clear of every main-pass slot.
    MetalAbi {
        program: &shared::GRASS_PREPASS_VERT,
        slots: &[&[
            ("view_cb", "buffer(0)"),
            ("gb_view", "buffer(3)"),
            ("grass", "buffer(19)"),
            ("grass_blades", "buffer(20)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::GRASS_PREPASS_FRAG,
        slots: &[&[("gb_view", "buffer(3)")]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::GRASS_PREPASS_CROSSFADE_FRAG,
        slots: &[&[("gb_view", "buffer(3)")]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::GRASS_VERT,
        slots: &[&[
            ("view_cb", "buffer(0)"),
            ("grass", "buffer(19)"),
            ("grass_blades", "buffer(20)"),
        ]],
        argument_ids: &[],
    },
    // The lit blades read the main pass's lights, shadows, clusters and
    // environment where the surfaces left them, through the same two argument
    // buffers.
    MetalAbi {
        program: &shared::GRASS_FRAG,
        slots: &[&[
            ("view_cb", "buffer(0)"),
            ("lights_cb", "buffer(4)"),
            ("shadow_cb", "buffer(5)"),
            ("spvDescriptorSet1", "buffer(7)"),
            ("local_lights_sb", "buffer(8)"),
            ("spvDescriptorSet2", "buffer(10)"),
            ("cluster_cb", "buffer(11)"),
            ("cluster_list_sb", "buffer(12)"),
            ("spot_shadows_sb", "buffer(13)"),
        ]],
        argument_ids: &[
            ("shadow_map", bindless_textures::SHADOW_MAP),
            ("irradiance_cube", bindless_textures::IRRADIANCE_CUBE),
            ("screen_indirect", bindless_textures::SCREEN_INDIRECT),
            ("spot_shadow_map", bindless_textures::SPOT_SHADOW_MAP),
            ("tex_pool", bindless_textures::pool(0)),
            ("tex_sampler", 0),
            ("shadow_sampler", 1),
            ("cube_sampler", 2),
        ],
    },
    MetalAbi {
        program: &shared::GRASS_CROSSFADE_FRAG,
        slots: &[&[
            ("view_cb", "buffer(0)"),
            ("lights_cb", "buffer(4)"),
            ("shadow_cb", "buffer(5)"),
            ("spvDescriptorSet1", "buffer(7)"),
            ("local_lights_sb", "buffer(8)"),
            ("spvDescriptorSet2", "buffer(10)"),
            ("cluster_cb", "buffer(11)"),
            ("cluster_list_sb", "buffer(12)"),
            ("spot_shadows_sb", "buffer(13)"),
        ]],
        argument_ids: &[
            ("shadow_map", bindless_textures::SHADOW_MAP),
            ("irradiance_cube", bindless_textures::IRRADIANCE_CUBE),
            ("screen_indirect", bindless_textures::SCREEN_INDIRECT),
            ("spot_shadow_map", bindless_textures::SPOT_SHADOW_MAP),
            ("tex_pool", bindless_textures::pool(0)),
            ("tex_sampler", 0),
            ("shadow_sampler", 1),
            ("cube_sampler", 2),
        ],
    },
    // The sky's motion, at the tail of the same pre-pass encoder. buffer(0) is
    // the surfaces' main-pass view block until `metal/sky.rs` rebinds the
    // sky's own there.
    MetalAbi {
        program: &shared::GBUFFER_SKY_VERT,
        slots: &[&[("gb_view", "buffer(0)")]],
        argument_ids: &[],
    },
    // The sky, from `metal/sky.rs`: the pass's view block, the prefilter cube
    // and the cube sampler, each at slot 0.
    MetalAbi {
        program: &shared::SKY_VERT,
        slots: &[&[("view_cb", "buffer(0)")]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::SKY_FRAG,
        slots: &[&[
            ("view_cb", "buffer(0)"),
            ("prefilter_cube", "texture(0)"),
            ("cube_sampler", "sampler(0)"),
        ]],
        argument_ids: &[],
    },
    // The shadow pass (cascades and spot slices): depth-only, its model out of
    // the object record at buffer(9) like the pre-pass, its view index at
    // buffer(7).
    MetalAbi {
        program: &shared::SHADOW_VERT_BINDLESS,
        slots: &[&[
            ("shadow_cb", "buffer(0)"),
            ("mesh_vertices", "buffer(2)"),
            ("cascade_cb", "buffer(7)"),
            ("objects", "buffer(9)"),
        ]],
        argument_ids: &[],
    },
    // The bindless main pass. Its two argument buffers are whole descriptor
    // sets, each at the index `[[cn::metal_argument_buffer(n)]]` names, and the
    // ids inside the texture one are where `metal/cull.rs` writes each member:
    // the fixed shadow / IBL / screen-indirect / probe / spot / LTC members first, then
    // the unsized pool. `EngineSamplers` is written 0..2 by
    // `build_bindless_sampler_args`. Only the grass variants read the canopy
    // block, and only the canopy variant its view tables.
    MetalAbi {
        program: &shared::MAIN_BINDLESS_FRAG,
        slots: &[MAIN_BINDLESS_SLOTS],
        argument_ids: MAIN_BINDLESS_ARGUMENT_IDS,
    },
    MetalAbi {
        program: &shared::MAIN_BINDLESS_CANOPY_FRAG,
        slots: &[MAIN_BINDLESS_SLOTS, CANOPY_SLOTS],
        argument_ids: MAIN_BINDLESS_ARGUMENT_IDS,
    },
    MetalAbi {
        program: &shared::MAIN_BINDLESS_CANOPY_DAYLIGHT_FRAG,
        slots: &[MAIN_BINDLESS_SLOTS, CANOPY_SLOTS],
        argument_ids: MAIN_BINDLESS_ARGUMENT_IDS,
    },
    MetalAbi {
        program: &shared::MAIN_BINDLESS_GROUND_FRAG,
        slots: &[MAIN_BINDLESS_SLOTS, GRASS_GROUND_SLOTS],
        argument_ids: MAIN_BINDLESS_ARGUMENT_IDS,
    },
    MetalAbi {
        program: &shared::MAIN_BINDLESS_GROUND_DAYLIGHT_FRAG,
        slots: &[MAIN_BINDLESS_SLOTS, GRASS_GROUND_SLOTS],
        argument_ids: MAIN_BINDLESS_ARGUMENT_IDS,
    },
    MetalAbi {
        program: &shared::MAIN_BINDLESS_VERT,
        slots: &[&[
            ("view_cb", "buffer(0)"),
            ("objects_sb", "buffer(9)"),
            ("mesh_vertices", "buffer(23)"),
        ]],
        argument_ids: &[],
    },
    // The SSR resolve: the probe cube array, its records and the cluster grid
    // are discrete bindings beside the screen sources, which is what
    // `metal/post/post_device.rs` binds.
    MetalAbi {
        program: &shared::SSR_RESOLVE,
        slots: &[&[
            ("params", "buffer(0)"),
            ("probe_set", "buffer(1)"),
            ("scene", "texture(0)"),
            ("scene_samp", "sampler(0)"),
            ("gbuffer", "texture(1)"),
            ("gbuffer_samp", "sampler(1)"),
            ("rough_tex", "texture(2)"),
            ("rough_tex_samp", "sampler(2)"),
            ("prefilter", "texture(3)"),
            ("prefilter_samp", "sampler(3)"),
            ("probe_records", "buffer(5)"),
            ("probe_cubes", "texture(4)"),
            ("probe_cube_sampler", "sampler(4)"),
            ("cluster", "buffer(2)"),
            ("cluster_list", "buffer(6)"),
        ]],
        argument_ids: &[],
    },
    // The RT resolve. The textured variant's one argument buffer is the
    // bindless pool at buffer(7): the main pass's texture block, bound at the
    // pool's offset, so the unsized array's id 0 is the pool's first texture.
    // Everything else, the probe set included, is pinned by its register().
    MetalAbi {
        program: &shared::RT_REFLECTIONS_CLASSIFY,
        slots: &[RT_SCREEN_SLOTS, RT_CLASSIFY_SLOTS],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::RT_REFLECTIONS_TRACE,
        slots: &[RT_SCREEN_SLOTS, RT_REFLECTIONS_SLOTS],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::RT_REFLECTIONS_TRACE_TEXTURED,
        slots: &[
            RT_SCREEN_SLOTS,
            RT_REFLECTIONS_SLOTS,
            RT_REFLECTIONS_POOL_SLOTS,
        ],
        argument_ids: &[("tex_pool", 0)],
    },
    MetalAbi {
        program: &shared::GLASS_FRAG,
        slots: GLASS_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::GLASS_FRAG_RT,
        slots: GLASS_RT_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::GLASS_REFLECTION_FRAG,
        slots: GLASS_REFLECTION_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::GLASS_FRAG_RT_TEXTURED,
        slots: GLASS_RT_TEXTURED_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::GLASS_REFLECTION_FRAG_TEXTURED,
        slots: GLASS_REFLECTION_TEXTURED_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::GLASS_MESH_FRAG_RT,
        slots: GLASS_RT_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::GLASS_MESH_REFLECTION_FRAG,
        slots: GLASS_REFLECTION_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::GLASS_MESH_FRAG_RT_TEXTURED,
        slots: GLASS_RT_TEXTURED_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::GLASS_MESH_REFLECTION_FRAG_TEXTURED,
        slots: GLASS_REFLECTION_TEXTURED_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_VERT,
        slots: &[&[
            ("view", "buffer(5)"),
            ("params", "buffer(6)"),
            ("ocean_displacement", "texture(9)"),
            ("ocean_sampler", "sampler(6)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_FRAG,
        slots: WATER_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_FRAG_RT,
        slots: WATER_RT_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_FRAG_RT_TEXTURED,
        slots: WATER_RT_TEXTURED_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_SEA_VERT,
        slots: &[&[
            ("view", "buffer(5)"),
            ("params", "buffer(6)"),
            ("ocean_displacement", "texture(9)"),
            ("ocean_sampler", "sampler(6)"),
            ("wake_field", "texture(11)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_SEA_FRAG,
        slots: WATER_SEA_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_SEA_DAYLIGHT_FRAG,
        slots: WATER_SEA_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_SEA_FRAG_RT,
        slots: WATER_SEA_RT_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_SEA_DAYLIGHT_FRAG_RT,
        slots: WATER_SEA_RT_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_SEA_FRAG_RT_TEXTURED,
        slots: WATER_SEA_RT_TEXTURED_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_SEA_DAYLIGHT_FRAG_RT_TEXTURED,
        slots: WATER_SEA_RT_TEXTURED_SLOTS,
        argument_ids: &[],
    },
    // The HLSL families. Their Metal indices come from the `register()`
    // annotations in the source rather than from declaration order, so these
    // rows check the generated binding table against the slots the encoders in
    // `metal/{post/bloom,text_upload,decal,particle}.rs` write.
    MetalAbi {
        program: &shared::BLOOM_PREFILTER,
        slots: &[&[
            ("src", "texture(0)"),
            ("src_sampler", "sampler(0)"),
            ("post", "buffer(0)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::BLOOM_DOWNSAMPLE,
        slots: &[&[("src", "texture(0)"), ("src_sampler", "sampler(0)")]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::BLOOM_UPSAMPLE,
        slots: &[&[("src", "texture(0)"), ("src_sampler", "sampler(0)")]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::TEXT_VERT,
        slots: &[&[("uni", "buffer(0)")]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::TEXT_FRAG,
        slots: &[&[("atlas", "texture(0)"), ("atlas_sampler", "sampler(0)")]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::DECAL_VERT,
        slots: &[&[("view", "buffer(0)"), ("params", "buffer(1)")]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::DECAL_FRAG,
        slots: &[&[
            ("view", "buffer(0)"),
            ("params", "buffer(1)"),
            ("scene_depth", "texture(0)"),
            ("decal_tex", "texture(1)"),
            ("decal_tex_sampler", "sampler(0)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::PARTICLE_VERT,
        slots: &[&[
            ("pool", "buffer(0)"),
            ("view", "buffer(1)"),
            ("params", "buffer(2)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::PARTICLE_FRAG,
        slots: &[&[
            ("albedo", "texture(0)"),
            ("albedo_sampler", "sampler(0)"),
            ("scene_depth", "texture(2)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::PARTICLE_LIT_VERT,
        slots: &[&[
            ("pool", "buffer(0)"),
            ("view", "buffer(1)"),
            ("params", "buffer(2)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::PARTICLE_LIT_FRAG,
        slots: &[&[
            ("albedo", "texture(0)"),
            ("albedo_sampler", "sampler(0)"),
            ("scene_depth", "texture(2)"),
        ]],
        argument_ids: &[],
    },
    // The fullscreen post fragments. Each source is one texture/sampler pair at
    // the index its register names, and the constants ride buffer(0), which is
    // what `metal/post/{post_device,ssao}.rs` bind.
    MetalAbi {
        program: &shared::TAA_FRAG,
        slots: &[&[
            ("params", "buffer(0)"),
            ("scene_tex", "texture(0)"),
            ("scene_samp", "sampler(0)"),
            ("velocity_tex", "texture(1)"),
            ("velocity_samp", "sampler(1)"),
            ("history_tex", "texture(2)"),
            ("history_samp", "sampler(2)"),
            ("reactive_tex", "texture(3)"),
            ("reactive_samp", "sampler(3)"),
        ]],
        argument_ids: &[],
    },
    // The depth copy reads the G-buffer by point load, so its sampler is
    // compiled out.
    MetalAbi {
        program: &shared::SSAO_DEPTH,
        slots: &[&[("params", "buffer(0)"), ("gbuffer", "texture(0)")]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::SSAO_KERNEL,
        slots: &[&[
            ("params", "buffer(0)"),
            ("gbuffer", "texture(0)"),
            ("gbuffer_samp", "sampler(0)"),
            ("depth", "texture(1)"),
            ("depth_samp", "sampler(1)"),
        ]],
        argument_ids: &[],
    },
    // The blur binds no constants at all, so its only slots are the two
    // sources.
    MetalAbi {
        program: &shared::SSAO_BLUR,
        slots: &[&[
            ("ao_raw", "texture(0)"),
            ("ao_raw_samp", "sampler(0)"),
            ("depth", "texture(1)"),
            ("depth_samp", "sampler(1)"),
        ]],
        argument_ids: &[],
    },
    // The disocclusion stages read by point load, so their samplers are
    // compiled out.
    MetalAbi {
        program: &shared::DISOCCLUSION_MARK,
        slots: &[&[
            ("gbuffer", "texture(0)"),
            ("velocity", "texture(1)"),
            ("previous", "texture(2)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::DISOCCLUSION_SNAPSHOT,
        slots: &[&[("gbuffer", "texture(0)"), ("velocity", "texture(1)")]],
        argument_ids: &[],
    },
    // The SSGI stages read every source but the hit radiance and the occlusion
    // by point load, so the samplers those sources declare are compiled out and
    // only the scene's and the occlusion's survive.
    MetalAbi {
        program: &shared::SSGI_DEPTH,
        slots: &[&[("params", "buffer(0)"), ("gbuffer", "texture(0)")]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::SSGI_REDUCE,
        slots: &[&[("finer", "texture(0)")]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::SSGI_AGE,
        slots: &[&[
            ("params", "buffer(0)"),
            ("gbuffer", "texture(0)"),
            ("velocity", "texture(1)"),
            ("pyramid", "texture(2)"),
            ("prev_pyramid", "texture(3)"),
            ("prev_age", "texture(4)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::SSGI_TRACE,
        slots: &[&[
            ("params", "buffer(0)"),
            ("scene", "texture(0)"),
            ("scene_samp", "sampler(0)"),
            ("gbuffer", "texture(1)"),
            ("velocity", "texture(2)"),
            ("pyramid", "texture(3)"),
            ("prev_pyramid", "texture(4)"),
            ("history", "texture(5)"),
            ("age", "texture(6)"),
            ("prev_age", "texture(7)"),
            ("reflectance", "texture(8)"),
        ]],
        argument_ids: &[],
    },
    // The volumetric-fog pair. The froxel kernel is the one place the two ABIs
    // disagree: its destination volume is a UAV, which D3D numbers in a space of
    // its own and Metal folds into the texture namespace behind the cascades, so
    // the non-DXIL leg declares it `u1` and lands on texture(1) -- which is what
    // `metal/fog.rs` binds, and the opposite of where the emitter would place it
    // unaided.
    MetalAbi {
        program: &shared::FOG_FROXEL,
        slots: &[&[
            ("fog", "buffer(0)"),
            ("froxel", "buffer(1)"),
            ("shadow_uni", "buffer(2)"),
            ("shadow_map", "texture(0)"),
            ("shadow_samp", "sampler(0)"),
            ("fog_volume", "texture(1)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::FOG_FRAG,
        slots: &[&[
            ("fog", "buffer(0)"),
            ("froxel", "buffer(1)"),
            ("scene_depth", "texture(0)"),
            ("fog_volume", "texture(1)"),
            ("fog_volume_samp", "sampler(0)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::SSGI_ACCUMULATE,
        slots: &[&[
            ("params", "buffer(0)"),
            ("fast", "texture(0)"),
            ("gbuffer", "texture(1)"),
            ("velocity", "texture(2)"),
            ("pyramid", "texture(3)"),
            ("prev_pyramid", "texture(4)"),
            ("history", "texture(5)"),
            ("age", "texture(6)"),
            ("prev_age", "texture(7)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::SSGI_RESOLVE,
        slots: &[&[
            ("params", "buffer(0)"),
            ("history", "texture(0)"),
            ("prev_pyramid", "texture(1)"),
            ("prev_age", "texture(2)"),
            ("gbuffer", "texture(3)"),
            ("velocity", "texture(4)"),
            ("occlusion", "texture(5)"),
            ("occlusion_samp", "sampler(5)"),
        ]],
        argument_ids: &[],
    },
    // The reflection pair and the final composite, the heaviest source counts
    // in the tree. `metal/post/ssr.rs` binds each source at its declaration
    // index and repeats one sampler across the same range, and
    // `metal/post/post_device.rs` puts the composite constants on buffer(0).
    MetalAbi {
        program: &shared::REFLECTION_BLUR,
        slots: &[&[
            ("reflection", "texture(0)"),
            ("reflection_samp", "sampler(0)"),
            ("rough_tex", "texture(1)"),
            ("rough_tex_samp", "sampler(1)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::REFLECTION_COMPOSITE,
        slots: &[&[
            ("reflection", "texture(0)"),
            ("reflection_samp", "sampler(0)"),
            ("scene", "texture(1)"),
            ("scene_samp", "sampler(1)"),
            ("gbuffer", "texture(2)"),
            ("gbuffer_samp", "sampler(2)"),
            ("rough_tex", "texture(3)"),
            ("rough_tex_samp", "sampler(3)"),
            ("blur_tex", "texture(4)"),
            ("blur_tex_samp", "sampler(4)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::COMPOSITE_FRAG,
        slots: &[&[
            ("post", "buffer(0)"),
            ("hdr_tex", "texture(0)"),
            ("hdr_tex_samp", "sampler(0)"),
            ("bloom_tex", "texture(1)"),
            ("bloom_tex_samp", "sampler(1)"),
            ("lut_tex", "texture(2)"),
            ("lut_tex_samp", "sampler(2)"),
            ("gbuf_nd_tex", "texture(3)"),
            ("gbuf_nd_tex_samp", "sampler(3)"),
            ("gbuf_rough_tex", "texture(4)"),
            ("gbuf_rough_tex_samp", "sampler(4)"),
            ("ao_tex", "texture(5)"),
            ("ao_tex_samp", "sampler(5)"),
            ("gbuf_motion_tex", "texture(6)"),
            ("gbuf_motion_tex_samp", "sampler(6)"),
            ("reactive_tex", "texture(7)"),
            ("reactive_tex_samp", "sampler(7)"),
        ]],
        argument_ids: &[],
    },
    // The compute families and the world-space line pair. Their Metal indices
    // come from the `register()` annotations too, and the slots below are what
    // `metal/{auto_exposure,light_cull,particle,line}.rs` and the two skin
    // dispatches in `metal/raytrace.rs` write.
    MetalAbi {
        program: &shared::AUTO_EXPOSURE_BUILD,
        slots: &[&[
            ("hdr_texture", "texture(0)"),
            ("histogram", "buffer(0)"),
            ("params", "buffer(1)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::AUTO_EXPOSURE_AVERAGE,
        slots: &[&[
            ("histogram", "buffer(0)"),
            ("measurement", "buffer(1)"),
            ("params", "buffer(2)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::LIGHT_CULL,
        slots: &[&[
            ("cluster", "buffer(0)"),
            ("lights", "buffer(1)"),
            ("cluster_list", "buffer(2)"),
            ("probe_records", "buffer(3)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::PARTICLE_SIMULATE,
        slots: &[&[
            ("pool", "buffer(0)"),
            ("frame_rows", "buffer(1)"),
            ("params", "buffer(2)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::RT_SKIN,
        slots: &[&[
            ("src", "buffer(0)"),
            ("dst", "buffer(1)"),
            ("palette", "buffer(2)"),
            ("params", "buffer(3)"),
            ("morph_data", "buffer(4)"),
            ("morph_weights", "buffer(5)"),
        ]],
        argument_ids: &[],
    },
    // The Hi-Z pyramid's per-mip chain, which is Metal's half of the builder
    // (Vulkan and DirectX run the SPD trio instead). The slots are what
    // `metal/hiz.rs` writes for every dispatch of the chain.
    MetalAbi {
        program: &metal::HIZ_INIT_MSAA,
        slots: HIZ_INIT_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &metal::HIZ_INIT_SINGLE,
        slots: HIZ_INIT_SLOTS,
        argument_ids: &[],
    },
    MetalAbi {
        program: &metal::HIZ_DOWNSAMPLE,
        slots: &[&[
            ("params", "buffer(0)"),
            ("src_hiz", "texture(0)"),
            ("dst_mip", "texture(1)"),
        ]],
        argument_ids: &[],
    },
    // The reflection-probe prefilter, whose three kernels `metal/probe_prefilter.rs`
    // binds the same way: params, source, destination. The GGX kernel's
    // destination is the one slot D3D and Metal disagree on, so its `u`
    // register moves behind `CN_BACKEND_DIRECTX` and Metal keeps texture(1).
    MetalAbi {
        program: &shared::PROBE_MIP0,
        slots: &[&[
            ("params", "buffer(0)"),
            ("src_mip", "texture(0)"),
            ("dst_mip", "texture(1)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::PROBE_DOWNSAMPLE,
        slots: &[&[
            ("params", "buffer(0)"),
            ("src_mip", "texture(0)"),
            ("dst_mip", "texture(1)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::PROBE_GGX,
        slots: &[&[
            ("params", "buffer(0)"),
            ("src_cube", "texture(0)"),
            ("src_sampler", "sampler(0)"),
            ("dst_mip", "texture(1)"),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::LINE_VERT,
        slots: &[&[("view", "buffer(0)")]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::LINE_FRAG,
        slots: &[&[("view", "buffer(0)"), ("scene_depth", "texture(0)")]],
        argument_ids: &[],
    },
];

// Both Hi-Z init variants: the params, the depth they read and the first mip.
const HIZ_INIT_SLOTS: &[&[(&str, &str)]] = &[&[
    ("params", "buffer(0)"),
    ("src_depth", "texture(0)"),
    ("dst_mip", "texture(1)"),
]];

// What every transparent fragment binds for its own draw: the view and
// surface blocks, the scene depth and the sky prefilter cube.
const TRANSPARENT_SLOTS: &[(&str, &str)] = &[
    ("view", "buffer(5)"),
    ("params", "buffer(6)"),
    ("scene_depth", "texture(1)"),
    ("prefilter_cube", "texture(2)"),
    ("prefilter_cube_sampler", "sampler(1)"),
];

// The transparent pass's probe set and the cluster grid that bins it.
const TRANSPARENT_PROBE_SLOTS: &[(&str, &str)] = &[
    ("probe_set", "buffer(7)"),
    ("probe_records", "buffer(11)"),
    ("cluster", "buffer(12)"),
    ("cluster_list", "buffer(13)"),
    ("probe_cubes", "texture(6)"),
    ("probe_cube_sampler", "sampler(2)"),
];

// The refraction snapshot every see-through transparent fragment reads.
const SCENE_COLOR_SLOTS: &[(&str, &str)] = &[
    ("scene_color", "texture(0)"),
    ("scene_color_sampler", "sampler(0)"),
];

// The planar mirror resolve.
const PLANAR_SLOTS: &[(&str, &str)] = &[
    ("planar_reflection", "texture(3)"),
    ("planar_reflection_sampler", "sampler(3)"),
];

// The ray-traced transparent fragments' trace inputs.
const TRANSPARENT_TRACE_SLOTS: &[(&str, &str)] = &[
    ("rt_params", "buffer(0)"),
    ("verts", "buffer(1)"),
    ("indices", "buffer(2)"),
    ("geom", "buffer(3)"),
    ("scene_tlas", "buffer(4)"),
    ("sverts", "buffer(8)"),
    ("sidx", "buffer(9)"),
];

// Both reflection layers the reduced pre-pass wrote.
const REFLECTION_LAYER_SLOTS: &[(&str, &str)] = &[
    ("glass_reflection", "texture(4)"),
    ("glass_reflection_back", "texture(5)"),
];

// The reflection layer a pre-pass entry peels behind.
const REFLECTION_PEEL_SLOTS: &[(&str, &str)] = &[("glass_reflection", "texture(4)")];

// A textured transparent variant's bindless pool argument buffer and sampler.
const TRANSPARENT_POOL_SLOTS: &[(&str, &str)] = &[
    ("spvDescriptorSet6", "buffer(10)"),
    ("pool_sampler", "sampler(4)"),
];

// Every slot both RT reflection kernels bind, the probe set included, pinned
// by its register(): the screen sources, the probe fallback, the output and the
// ray list and arguments.
const RT_SCREEN_SLOTS: &[(&str, &str)] = &[
    ("rt_params", "buffer(0)"),
    ("probe_set", "buffer(8)"),
    ("scene_tex", "texture(0)"),
    ("scene_tex_sampler", "sampler(0)"),
    ("gbuffer", "texture(1)"),
    ("gbuffer_sampler", "sampler(1)"),
    ("rough_tex", "texture(2)"),
    ("rough_tex_sampler", "sampler(2)"),
    ("prefilter", "texture(3)"),
    ("prefilter_sampler", "sampler(3)"),
    ("probe_records", "buffer(11)"),
    ("probe_cubes", "texture(4)"),
    ("probe_cube_sampler", "sampler(4)"),
    ("cluster", "buffer(9)"),
    ("cluster_list", "buffer(10)"),
    ("_output", "texture(6)"),
    ("ray_list", "buffer(14)"),
    ("ray_args", "buffer(15)"),
];

// The classify's own: SSGI's pyramid.
const RT_CLASSIFY_SLOTS: &[(&str, &str)] = &[("pyramid", "texture(5)")];

// The trace's own: the acceleration structure and the geometry it fetches hits
// from.
const RT_REFLECTIONS_SLOTS: &[(&str, &str)] = &[
    ("verts", "buffer(1)"),
    ("indices", "buffer(2)"),
    ("geom", "buffer(3)"),
    ("scene_tlas", "buffer(4)"),
    ("sverts", "buffer(5)"),
    ("sidx", "buffer(6)"),
];

// The textured RT resolve's bindless pool (the main pass's texture block, bound
// at the pool's offset) and the canopy block its grass-covered hits shade from.
const RT_REFLECTIONS_POOL_SLOTS: &[(&str, &str)] = &[
    ("spvDescriptorSet3", "buffer(7)"),
    ("pool_sampler", "sampler(5)"),
    ("canopy_cb", "buffer(12)"),
    ("canopy_tables_sb", "buffer(13)"),
];

const GLASS_SLOTS: &[&[(&str, &str)]] = &[
    TRANSPARENT_SLOTS,
    TRANSPARENT_PROBE_SLOTS,
    SCENE_COLOR_SLOTS,
    PLANAR_SLOTS,
];
const GLASS_RT_SLOTS: &[&[(&str, &str)]] = &[
    TRANSPARENT_SLOTS,
    TRANSPARENT_PROBE_SLOTS,
    SCENE_COLOR_SLOTS,
    TRANSPARENT_TRACE_SLOTS,
    REFLECTION_LAYER_SLOTS,
];
const GLASS_RT_TEXTURED_SLOTS: &[&[(&str, &str)]] = &[
    TRANSPARENT_SLOTS,
    TRANSPARENT_PROBE_SLOTS,
    SCENE_COLOR_SLOTS,
    TRANSPARENT_TRACE_SLOTS,
    REFLECTION_LAYER_SLOTS,
    TRANSPARENT_POOL_SLOTS,
];
const GLASS_REFLECTION_SLOTS: &[&[(&str, &str)]] = &[
    TRANSPARENT_SLOTS,
    TRANSPARENT_PROBE_SLOTS,
    TRANSPARENT_TRACE_SLOTS,
    REFLECTION_PEEL_SLOTS,
];
const GLASS_REFLECTION_TEXTURED_SLOTS: &[&[(&str, &str)]] = &[
    TRANSPARENT_SLOTS,
    TRANSPARENT_PROBE_SLOTS,
    TRANSPARENT_TRACE_SLOTS,
    REFLECTION_PEEL_SLOTS,
    TRANSPARENT_POOL_SLOTS,
];
// The light the water body is lit by: the cascaded shadow and the sky
// irradiance cube.
const WATER_BODY_LIGHT_SLOTS: &[(&str, &str)] = &[
    ("shadow_cb", "buffer(14)"),
    ("shadow_map", "texture(7)"),
    ("irradiance_cube", "texture(8)"),
    ("cloud_shadow", "texture(12)"),
    ("shadow_sampler", "sampler(5)"),
];
// The simulated oceans' cascades, whose slopes and foam every water fragment
// shades from.
const WATER_OCEAN_SLOTS: &[(&str, &str)] = &[
    ("ocean_displacement", "texture(9)"),
    ("ocean_slopes", "texture(10)"),
    ("ocean_sampler", "sampler(6)"),
];
const WATER_SLOTS: &[&[(&str, &str)]] = &[
    TRANSPARENT_SLOTS,
    TRANSPARENT_PROBE_SLOTS,
    SCENE_COLOR_SLOTS,
    PLANAR_SLOTS,
    WATER_BODY_LIGHT_SLOTS,
    WATER_OCEAN_SLOTS,
];
const WATER_RT_SLOTS: &[&[(&str, &str)]] = &[
    TRANSPARENT_SLOTS,
    TRANSPARENT_PROBE_SLOTS,
    SCENE_COLOR_SLOTS,
    PLANAR_SLOTS,
    TRANSPARENT_TRACE_SLOTS,
    WATER_BODY_LIGHT_SLOTS,
    WATER_OCEAN_SLOTS,
];
const WATER_RT_TEXTURED_SLOTS: &[&[(&str, &str)]] = &[
    TRANSPARENT_SLOTS,
    TRANSPARENT_PROBE_SLOTS,
    SCENE_COLOR_SLOTS,
    PLANAR_SLOTS,
    TRANSPARENT_TRACE_SLOTS,
    TRANSPARENT_POOL_SLOTS,
    WATER_BODY_LIGHT_SLOTS,
    WATER_OCEAN_SLOTS,
];
// A planet sea's wake field, which its fragments shade from too.
const WATER_WAKE_SLOTS: &[(&str, &str)] = &[("wake_field", "texture(11)")];
const WATER_SEA_SLOTS: &[&[(&str, &str)]] = &[
    TRANSPARENT_SLOTS,
    TRANSPARENT_PROBE_SLOTS,
    SCENE_COLOR_SLOTS,
    PLANAR_SLOTS,
    WATER_BODY_LIGHT_SLOTS,
    WATER_OCEAN_SLOTS,
    WATER_WAKE_SLOTS,
];
const WATER_SEA_RT_SLOTS: &[&[(&str, &str)]] = &[
    TRANSPARENT_SLOTS,
    TRANSPARENT_PROBE_SLOTS,
    SCENE_COLOR_SLOTS,
    PLANAR_SLOTS,
    TRANSPARENT_TRACE_SLOTS,
    WATER_BODY_LIGHT_SLOTS,
    WATER_OCEAN_SLOTS,
    WATER_WAKE_SLOTS,
];
const WATER_SEA_RT_TEXTURED_SLOTS: &[&[(&str, &str)]] = &[
    TRANSPARENT_SLOTS,
    TRANSPARENT_PROBE_SLOTS,
    SCENE_COLOR_SLOTS,
    PLANAR_SLOTS,
    TRANSPARENT_TRACE_SLOTS,
    TRANSPARENT_POOL_SLOTS,
    WATER_BODY_LIGHT_SLOTS,
    WATER_OCEAN_SLOTS,
    WATER_WAKE_SLOTS,
];
