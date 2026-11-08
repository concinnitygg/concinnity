//! The slots each Metal program's host encoder writes, which the device build
//! checks against the MSL the precompile emits.

use super::MetalAbi;
use super::cull::metal as cull;
use super::main::metal as main;
use super::slot::{MetalBinding, buffer, sampler, texture};
use crate::render::shader::programs::metal::bindless_textures;
use crate::render::shader::programs::{metal, shared};

// The bindless main fragment's discrete slots, every variant.
const MAIN_BINDLESS_SLOTS: &[MetalBinding] = &[
    main::VIEW.binding(),
    main::LIGHTS.binding(),
    main::SHADOW.binding(),
    main::PROBE_SET.binding(),
    main::PROBE_RECORDS.binding(),
    main::TEXTURES.binding(),
    main::LOCAL_LIGHTS.binding(),
    main::OBJECTS.binding(),
    main::SAMPLERS.binding(),
    main::CLUSTER.binding(),
    main::CLUSTER_LIST.binding(),
    main::SPOT_SHADOWS.binding(),
    main::AREA_LIGHTS.binding(),
];

// The far-field canopy's block and view tables, read by the canopy variant.
const CANOPY_SLOTS: &[MetalBinding] = &[main::CANOPY.binding(), main::CANOPY_TABLES.binding()];

// The ground under grass reads the canopy block but none of its view tables.
const GRASS_GROUND_SLOTS: &[MetalBinding] = &[main::CANOPY.binding()];

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
    // the Hi-Z texture takes texture(0) from declaration order; the cull's
    // encoder binds the slots once for this dispatch and the scan and scatter
    // that follow it.
    MetalAbi {
        program: &shared::CULL_EARLY,
        slots: &[&[
            cull::OBJECTS.binding(),
            cull::DRAW_ARGS.binding(),
            cull::CULL.binding(),
            cull::CULL_STATUS.binding(),
            cull::VARIANTS.binding(),
            cull::GROUP_COUNTS.binding(),
            cull::HIZ.binding(),
        ]],
        argument_ids: &[],
    },
    // The late pass re-tests candidates only, so it never reads `draw_args`; the
    // encode dispatch after it does, from the same binding.
    MetalAbi {
        program: &shared::CULL_LATE,
        slots: &[&[
            cull::OBJECTS.binding(),
            cull::CULL.binding(),
            cull::CULL_STATUS.binding(),
            cull::VARIANTS.binding(),
            cull::GROUP_COUNTS.binding(),
            cull::HIZ.binding(),
        ]],
        argument_ids: &[],
    },
    // The scan between the decision and the encode, on the same encoder.
    MetalAbi {
        program: &shared::CULL_SCAN,
        slots: &[&[
            cull::VARIANTS.binding(),
            cull::GROUP_COUNTS.binding(),
            cull::STREAM_COUNTS.binding(),
            cull::SCAN_PARAMS.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::CULL_SHADOW,
        slots: &[&[
            cull::OBJECTS.binding(),
            cull::DRAW_ARGS.binding(),
            cull::CULL.binding(),
            cull::CULL_STATUS.binding(),
            cull::GROUP_COUNTS.binding(),
        ]],
        argument_ids: &[],
    },
    // The model-history snapshot: params, the object buffer it reads and the
    // ring slot it writes take buffer(0..2) from declaration order, which is
    // what `metal/model_history.rs` binds.
    MetalAbi {
        program: &shared::MODEL_HISTORY,
        slots: &[&[
            buffer("params", 0),
            buffer("objects", 1),
            buffer("history", 2),
        ]],
        argument_ids: &[],
    },
    // The G-buffer pre-pass, from the main pass's file: the main pass's view
    // block, records, parameter table, texture argument buffer and samplers at
    // their main-pass slots, plus the pre-pass's own view block, model history
    // and draw args, pinned clear of the vertex streams at buffer(1) and
    // buffer(2). The pre-pass encoder binds them once for every ICB-executed
    // draw.
    MetalAbi {
        program: &shared::MAIN_PREPASS_VERT,
        slots: &[&[
            main::VIEW.binding(),
            main::GBUFFER_VIEW.binding(),
            main::OBJECTS.binding(),
            main::PREV_MODELS.binding(),
            main::DRAW_ARGS.binding(),
            main::MESH_VERTICES.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::MAIN_PREPASS_FRAG,
        slots: &[&[
            main::GBUFFER_VIEW.binding(),
            main::TEXTURES.binding(),
            main::OBJECTS.binding(),
            main::SAMPLERS.binding(),
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
            buffer("grass", 0),
            buffer("blades_out", 1),
            buffer("draw_args", 2),
            buffer("terrain_heights", 3),
            buffer("grass_masks", 4),
            texture("grass_hiz", 5),
            buffer("grass_bend_field", 6),
            buffer("planet_tiles", 7),
            buffer("mesh_vertices", 8),
            buffer("grass_tiles", 9),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::GRASS_BEND,
        slots: &[&[buffer("bend", 0), buffer("bend_field", 1)]],
        argument_ids: &[],
    },
    // The simulated ocean's kernels, from `metal/water_sim.rs`: the constants
    // at buffer(0), the spectrum and row transforms at buffer(5) and (6), and
    // the source and destination mips at texture(0..3), all by register.
    MetalAbi {
        program: &shared::WATER_FFT_ROWS,
        slots: &[&[
            buffer("params", 0),
            buffer("spectrum", 5),
            buffer("rows", 6),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_WAKE_ROWS,
        slots: &[&[buffer("params", 0), buffer("state", 1), buffer("work", 2)]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_WAKE_COLUMNS,
        slots: &[&[buffer("params", 0), buffer("work", 2)]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_WAKE_INVERSE,
        slots: &[&[buffer("params", 0), buffer("state", 1), buffer("work", 2)]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_WAKE_SHADE,
        slots: &[&[
            buffer("params", 0),
            buffer("state", 1),
            texture("dst_field", 3),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_WAKE_MIP,
        slots: &[&[texture("dst_field", 3), texture("src_field", 4)]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_FFT_COLUMNS,
        slots: &[&[
            buffer("params", 0),
            buffer("rows", 6),
            texture("dst_displacement", 2),
            texture("dst_slopes", 3),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_FFT_COLUMNS_FOAM,
        slots: &[&[
            buffer("params", 0),
            buffer("rows", 6),
            texture("dst_displacement", 2),
            texture("dst_slopes", 3),
            buffer("foam", 7),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_MIP,
        slots: &[&[
            buffer("params", 0),
            texture("src_displacement", 0),
            texture("src_slopes", 1),
            texture("dst_displacement", 2),
            texture("dst_slopes", 3),
        ]],
        argument_ids: &[],
    },
    // The cascade's blades draw in the shadow pass under their own pipeline,
    // which binds only the shadow block and the cascade's blades.
    MetalAbi {
        program: &shared::GRASS_SHADOW_VERT,
        slots: &[&[buffer("grass", 0), buffer("grass_blades", 1)]],
        argument_ids: &[],
    },
    // The grass draws at the tail of the pre-pass and the main pass: the
    // pass's view blocks where the surfaces left them, and the grass block and
    // blades at buffer(19) and buffer(20), clear of every main-pass slot.
    MetalAbi {
        program: &shared::GRASS_PREPASS_VERT,
        slots: &[&[
            buffer("view_cb", 0),
            buffer("gb_view", 3),
            buffer("grass", 19),
            buffer("grass_blades", 20),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::GRASS_PREPASS_FRAG,
        slots: &[&[buffer("gb_view", 3)]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::GRASS_PREPASS_CROSSFADE_FRAG,
        slots: &[&[buffer("gb_view", 3)]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::GRASS_VERT,
        slots: &[&[
            buffer("view_cb", 0),
            buffer("grass", 19),
            buffer("grass_blades", 20),
        ]],
        argument_ids: &[],
    },
    // The lit blades read the main pass's lights, shadows, clusters and
    // environment where the surfaces left them, through the same two argument
    // buffers.
    MetalAbi {
        program: &shared::GRASS_FRAG,
        slots: &[&[
            buffer("view_cb", 0),
            buffer("lights_cb", 4),
            buffer("shadow_cb", 5),
            buffer("spvDescriptorSet1", 7),
            buffer("local_lights_sb", 8),
            buffer("spvDescriptorSet2", 10),
            buffer("cluster_cb", 11),
            buffer("cluster_list_sb", 12),
            buffer("spot_shadows_sb", 13),
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
            buffer("view_cb", 0),
            buffer("lights_cb", 4),
            buffer("shadow_cb", 5),
            buffer("spvDescriptorSet1", 7),
            buffer("local_lights_sb", 8),
            buffer("spvDescriptorSet2", 10),
            buffer("cluster_cb", 11),
            buffer("cluster_list_sb", 12),
            buffer("spot_shadows_sb", 13),
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
        slots: &[&[buffer("gb_view", 0)]],
        argument_ids: &[],
    },
    // The sky, from `metal/sky.rs`: the pass's view block, the prefilter cube
    // and the cube sampler, each at slot 0.
    MetalAbi {
        program: &shared::SKY_VERT,
        slots: &[&[buffer("view_cb", 0)]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::SKY_FRAG,
        slots: &[&[
            buffer("view_cb", 0),
            texture("prefilter_cube", 0),
            sampler("cube_sampler", 0),
        ]],
        argument_ids: &[],
    },
    // The shadow pass (cascades and spot slices): depth-only, its model out of
    // the object record at buffer(9) like the pre-pass, its view index at
    // buffer(7).
    MetalAbi {
        program: &shared::SHADOW_VERT_BINDLESS,
        slots: &[&[
            buffer("shadow_cb", 0),
            buffer("mesh_vertices", 2),
            buffer("cascade_cb", 7),
            buffer("objects", 9),
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
            main::VIEW.binding(),
            main::OBJECTS.binding(),
            main::MESH_VERTICES.binding(),
        ]],
        argument_ids: &[],
    },
    // The SSR resolve: the probe cube array, its records and the cluster grid
    // are discrete bindings beside the screen sources, which is what
    // `metal/post/post_device.rs` binds.
    MetalAbi {
        program: &shared::SSR_RESOLVE,
        slots: &[&[
            buffer("params", 0),
            buffer("probe_set", 1),
            texture("scene", 0),
            sampler("scene_samp", 0),
            texture("gbuffer", 1),
            sampler("gbuffer_samp", 1),
            texture("rough_tex", 2),
            sampler("rough_tex_samp", 2),
            texture("prefilter", 3),
            sampler("prefilter_samp", 3),
            buffer("probe_records", 5),
            texture("probe_cubes", 4),
            sampler("probe_cube_sampler", 4),
            buffer("cluster", 2),
            buffer("cluster_list", 6),
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
            buffer("view", 5),
            buffer("params", 6),
            texture("ocean_displacement", 9),
            sampler("ocean_sampler", 6),
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
            buffer("view", 5),
            buffer("params", 6),
            texture("ocean_displacement", 9),
            sampler("ocean_sampler", 6),
            texture("wake_field", 11),
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
            texture("src", 0),
            sampler("src_sampler", 0),
            buffer("post", 0),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::BLOOM_DOWNSAMPLE,
        slots: &[&[texture("src", 0), sampler("src_sampler", 0)]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::BLOOM_UPSAMPLE,
        slots: &[&[texture("src", 0), sampler("src_sampler", 0)]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::TEXT_VERT,
        slots: &[&[buffer("uni", 0)]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::TEXT_FRAG,
        slots: &[&[texture("atlas", 0), sampler("atlas_sampler", 0)]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::DECAL_VERT,
        slots: &[&[buffer("view", 0), buffer("params", 1)]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::DECAL_FRAG,
        slots: &[&[
            buffer("view", 0),
            buffer("params", 1),
            texture("scene_depth", 0),
            texture("decal_tex", 1),
            sampler("decal_tex_sampler", 0),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::PARTICLE_VERT,
        slots: &[&[buffer("pool", 0), buffer("view", 1), buffer("params", 2)]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::PARTICLE_FRAG,
        slots: &[&[
            texture("albedo", 0),
            sampler("albedo_sampler", 0),
            texture("scene_depth", 2),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::PARTICLE_LIT_VERT,
        slots: &[&[buffer("pool", 0), buffer("view", 1), buffer("params", 2)]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::PARTICLE_LIT_FRAG,
        slots: &[&[
            texture("albedo", 0),
            sampler("albedo_sampler", 0),
            texture("scene_depth", 2),
        ]],
        argument_ids: &[],
    },
    // The fullscreen post fragments. Each source is one texture/sampler pair at
    // the index its register names, and the constants ride buffer(0), which is
    // what `metal/post/{post_device,ssao}.rs` bind.
    MetalAbi {
        program: &shared::TAA_FRAG,
        slots: &[&[
            buffer("params", 0),
            texture("scene_tex", 0),
            sampler("scene_samp", 0),
            texture("velocity_tex", 1),
            sampler("velocity_samp", 1),
            texture("history_tex", 2),
            sampler("history_samp", 2),
            texture("reactive_tex", 3),
            sampler("reactive_samp", 3),
        ]],
        argument_ids: &[],
    },
    // The depth copy reads the G-buffer by point load, so its sampler is
    // compiled out.
    MetalAbi {
        program: &shared::SSAO_DEPTH,
        slots: &[&[buffer("params", 0), texture("gbuffer", 0)]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::SSAO_KERNEL,
        slots: &[&[
            buffer("params", 0),
            texture("gbuffer", 0),
            sampler("gbuffer_samp", 0),
            texture("depth", 1),
            sampler("depth_samp", 1),
        ]],
        argument_ids: &[],
    },
    // The blur binds no constants at all, so its only slots are the two
    // sources.
    MetalAbi {
        program: &shared::SSAO_BLUR,
        slots: &[&[
            texture("ao_raw", 0),
            sampler("ao_raw_samp", 0),
            texture("depth", 1),
            sampler("depth_samp", 1),
        ]],
        argument_ids: &[],
    },
    // The disocclusion stages read by point load, so their samplers are
    // compiled out.
    MetalAbi {
        program: &shared::DISOCCLUSION_MARK,
        slots: &[&[
            texture("gbuffer", 0),
            texture("velocity", 1),
            texture("previous", 2),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::DISOCCLUSION_SNAPSHOT,
        slots: &[&[texture("gbuffer", 0), texture("velocity", 1)]],
        argument_ids: &[],
    },
    // The SSGI stages read every source but the hit radiance and the occlusion
    // by point load, so the samplers those sources declare are compiled out and
    // only the scene's and the occlusion's survive.
    MetalAbi {
        program: &shared::SSGI_DEPTH,
        slots: &[&[buffer("params", 0), texture("gbuffer", 0)]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::SSGI_REDUCE,
        slots: &[&[texture("finer", 0)]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::SSGI_AGE,
        slots: &[&[
            buffer("params", 0),
            texture("gbuffer", 0),
            texture("velocity", 1),
            texture("pyramid", 2),
            texture("prev_pyramid", 3),
            texture("prev_age", 4),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::SSGI_TRACE,
        slots: &[&[
            buffer("params", 0),
            texture("scene", 0),
            sampler("scene_samp", 0),
            texture("gbuffer", 1),
            texture("velocity", 2),
            texture("pyramid", 3),
            texture("prev_pyramid", 4),
            texture("history", 5),
            texture("age", 6),
            texture("prev_age", 7),
            texture("reflectance", 8),
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
            buffer("fog", 0),
            buffer("froxel", 1),
            buffer("shadow_uni", 2),
            texture("shadow_map", 0),
            sampler("shadow_samp", 0),
            texture("fog_volume", 1),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::FOG_FRAG,
        slots: &[&[
            buffer("fog", 0),
            buffer("froxel", 1),
            texture("scene_depth", 0),
            texture("fog_volume", 1),
            sampler("fog_volume_samp", 0),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::SSGI_ACCUMULATE,
        slots: &[&[
            buffer("params", 0),
            texture("fast", 0),
            texture("gbuffer", 1),
            texture("velocity", 2),
            texture("pyramid", 3),
            texture("prev_pyramid", 4),
            texture("history", 5),
            texture("age", 6),
            texture("prev_age", 7),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::SSGI_RESOLVE,
        slots: &[&[
            buffer("params", 0),
            texture("history", 0),
            texture("prev_pyramid", 1),
            texture("prev_age", 2),
            texture("gbuffer", 3),
            texture("velocity", 4),
            texture("occlusion", 5),
            sampler("occlusion_samp", 5),
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
            texture("reflection", 0),
            sampler("reflection_samp", 0),
            texture("rough_tex", 1),
            sampler("rough_tex_samp", 1),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::REFLECTION_COMPOSITE,
        slots: &[&[
            texture("reflection", 0),
            sampler("reflection_samp", 0),
            texture("scene", 1),
            sampler("scene_samp", 1),
            texture("gbuffer", 2),
            sampler("gbuffer_samp", 2),
            texture("rough_tex", 3),
            sampler("rough_tex_samp", 3),
            texture("blur_tex", 4),
            sampler("blur_tex_samp", 4),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::COMPOSITE_FRAG,
        slots: &[&[
            buffer("post", 0),
            texture("hdr_tex", 0),
            sampler("hdr_tex_samp", 0),
            texture("bloom_tex", 1),
            sampler("bloom_tex_samp", 1),
            texture("lut_tex", 2),
            sampler("lut_tex_samp", 2),
            texture("gbuf_nd_tex", 3),
            sampler("gbuf_nd_tex_samp", 3),
            texture("gbuf_rough_tex", 4),
            sampler("gbuf_rough_tex_samp", 4),
            texture("ao_tex", 5),
            sampler("ao_tex_samp", 5),
            texture("gbuf_motion_tex", 6),
            sampler("gbuf_motion_tex_samp", 6),
            texture("reactive_tex", 7),
            sampler("reactive_tex_samp", 7),
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
            texture("hdr_texture", 0),
            buffer("histogram", 0),
            buffer("params", 1),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::AUTO_EXPOSURE_AVERAGE,
        slots: &[&[
            buffer("histogram", 0),
            buffer("measurement", 1),
            buffer("params", 2),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::LIGHT_CULL,
        slots: &[&[
            buffer("cluster", 0),
            buffer("lights", 1),
            buffer("cluster_list", 2),
            buffer("probe_records", 3),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::PARTICLE_SIMULATE,
        slots: &[&[
            buffer("pool", 0),
            buffer("frame_rows", 1),
            buffer("params", 2),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::RT_SKIN,
        slots: &[&[
            buffer("src", 0),
            buffer("dst", 1),
            buffer("palette", 2),
            buffer("params", 3),
            buffer("morph_data", 4),
            buffer("morph_weights", 5),
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
            buffer("params", 0),
            texture("src_hiz", 0),
            texture("dst_mip", 1),
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
            buffer("params", 0),
            texture("src_mip", 0),
            texture("dst_mip", 1),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::PROBE_DOWNSAMPLE,
        slots: &[&[
            buffer("params", 0),
            texture("src_mip", 0),
            texture("dst_mip", 1),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::PROBE_GGX,
        slots: &[&[
            buffer("params", 0),
            texture("src_cube", 0),
            sampler("src_sampler", 0),
            texture("dst_mip", 1),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::LINE_VERT,
        slots: &[&[buffer("view", 0)]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::LINE_FRAG,
        slots: &[&[buffer("view", 0), texture("scene_depth", 0)]],
        argument_ids: &[],
    },
];

// Both Hi-Z init variants: the params, the depth they read and the first mip.
const HIZ_INIT_SLOTS: &[&[MetalBinding]] = &[&[
    buffer("params", 0),
    texture("src_depth", 0),
    texture("dst_mip", 1),
]];

// What every transparent fragment binds for its own draw: the view and
// surface blocks, the scene depth and the sky prefilter cube.
const TRANSPARENT_SLOTS: &[MetalBinding] = &[
    buffer("view", 5),
    buffer("params", 6),
    texture("scene_depth", 1),
    texture("prefilter_cube", 2),
    sampler("prefilter_cube_sampler", 1),
];

// The transparent pass's probe set and the cluster grid that bins it.
const TRANSPARENT_PROBE_SLOTS: &[MetalBinding] = &[
    buffer("probe_set", 7),
    buffer("probe_records", 11),
    buffer("cluster", 12),
    buffer("cluster_list", 13),
    texture("probe_cubes", 6),
    sampler("probe_cube_sampler", 2),
];

// The refraction snapshot every see-through transparent fragment reads.
const SCENE_COLOR_SLOTS: &[MetalBinding] =
    &[texture("scene_color", 0), sampler("scene_color_sampler", 0)];

// The planar mirror resolve.
const PLANAR_SLOTS: &[MetalBinding] = &[
    texture("planar_reflection", 3),
    sampler("planar_reflection_sampler", 3),
];

// The ray-traced transparent fragments' trace inputs.
const TRANSPARENT_TRACE_SLOTS: &[MetalBinding] = &[
    buffer("rt_params", 0),
    buffer("verts", 1),
    buffer("indices", 2),
    buffer("geom", 3),
    buffer("scene_tlas", 4),
    buffer("sverts", 8),
    buffer("sidx", 9),
];

// Both reflection layers the reduced pre-pass wrote.
const REFLECTION_LAYER_SLOTS: &[MetalBinding] = &[
    texture("glass_reflection", 4),
    texture("glass_reflection_back", 5),
];

// The reflection layer a pre-pass entry peels behind.
const REFLECTION_PEEL_SLOTS: &[MetalBinding] = &[texture("glass_reflection", 4)];

// A textured transparent variant's bindless pool argument buffer and sampler.
const TRANSPARENT_POOL_SLOTS: &[MetalBinding] =
    &[buffer("spvDescriptorSet6", 10), sampler("pool_sampler", 4)];

// Every slot both RT reflection kernels bind, the probe set included, pinned
// by its register(): the screen sources, the probe fallback, the output and the
// ray list and arguments.
const RT_SCREEN_SLOTS: &[MetalBinding] = &[
    buffer("rt_params", 0),
    buffer("probe_set", 8),
    texture("scene_tex", 0),
    sampler("scene_tex_sampler", 0),
    texture("gbuffer", 1),
    sampler("gbuffer_sampler", 1),
    texture("rough_tex", 2),
    sampler("rough_tex_sampler", 2),
    texture("prefilter", 3),
    sampler("prefilter_sampler", 3),
    buffer("probe_records", 11),
    texture("probe_cubes", 4),
    sampler("probe_cube_sampler", 4),
    buffer("cluster", 9),
    buffer("cluster_list", 10),
    texture("_output", 6),
    buffer("ray_list", 14),
    buffer("ray_args", 15),
];

// The classify's own: SSGI's pyramid.
const RT_CLASSIFY_SLOTS: &[MetalBinding] = &[texture("pyramid", 5)];

// The trace's own: the acceleration structure and the geometry it fetches hits
// from.
const RT_REFLECTIONS_SLOTS: &[MetalBinding] = &[
    buffer("verts", 1),
    buffer("indices", 2),
    buffer("geom", 3),
    buffer("scene_tlas", 4),
    buffer("sverts", 5),
    buffer("sidx", 6),
];

// The textured RT resolve's bindless pool (the main pass's texture block, bound
// at the pool's offset) and the canopy block its grass-covered hits shade from.
const RT_REFLECTIONS_POOL_SLOTS: &[MetalBinding] = &[
    buffer("spvDescriptorSet3", 7),
    sampler("pool_sampler", 5),
    buffer("canopy_cb", 12),
    buffer("canopy_tables_sb", 13),
];

const GLASS_SLOTS: &[&[MetalBinding]] = &[
    TRANSPARENT_SLOTS,
    TRANSPARENT_PROBE_SLOTS,
    SCENE_COLOR_SLOTS,
    PLANAR_SLOTS,
];
const GLASS_RT_SLOTS: &[&[MetalBinding]] = &[
    TRANSPARENT_SLOTS,
    TRANSPARENT_PROBE_SLOTS,
    SCENE_COLOR_SLOTS,
    TRANSPARENT_TRACE_SLOTS,
    REFLECTION_LAYER_SLOTS,
];
const GLASS_RT_TEXTURED_SLOTS: &[&[MetalBinding]] = &[
    TRANSPARENT_SLOTS,
    TRANSPARENT_PROBE_SLOTS,
    SCENE_COLOR_SLOTS,
    TRANSPARENT_TRACE_SLOTS,
    REFLECTION_LAYER_SLOTS,
    TRANSPARENT_POOL_SLOTS,
];
const GLASS_REFLECTION_SLOTS: &[&[MetalBinding]] = &[
    TRANSPARENT_SLOTS,
    TRANSPARENT_PROBE_SLOTS,
    TRANSPARENT_TRACE_SLOTS,
    REFLECTION_PEEL_SLOTS,
];
const GLASS_REFLECTION_TEXTURED_SLOTS: &[&[MetalBinding]] = &[
    TRANSPARENT_SLOTS,
    TRANSPARENT_PROBE_SLOTS,
    TRANSPARENT_TRACE_SLOTS,
    REFLECTION_PEEL_SLOTS,
    TRANSPARENT_POOL_SLOTS,
];
// The light the water body is lit by: the cascaded shadow and the sky
// irradiance cube.
const WATER_BODY_LIGHT_SLOTS: &[MetalBinding] = &[
    buffer("shadow_cb", 14),
    texture("shadow_map", 7),
    texture("irradiance_cube", 8),
    texture("cloud_shadow", 12),
    sampler("shadow_sampler", 5),
];
// The simulated oceans' cascades, whose slopes and foam every water fragment
// shades from.
const WATER_OCEAN_SLOTS: &[MetalBinding] = &[
    texture("ocean_displacement", 9),
    texture("ocean_slopes", 10),
    sampler("ocean_sampler", 6),
];
const WATER_SLOTS: &[&[MetalBinding]] = &[
    TRANSPARENT_SLOTS,
    TRANSPARENT_PROBE_SLOTS,
    SCENE_COLOR_SLOTS,
    PLANAR_SLOTS,
    WATER_BODY_LIGHT_SLOTS,
    WATER_OCEAN_SLOTS,
];
const WATER_RT_SLOTS: &[&[MetalBinding]] = &[
    TRANSPARENT_SLOTS,
    TRANSPARENT_PROBE_SLOTS,
    SCENE_COLOR_SLOTS,
    PLANAR_SLOTS,
    TRANSPARENT_TRACE_SLOTS,
    WATER_BODY_LIGHT_SLOTS,
    WATER_OCEAN_SLOTS,
];
const WATER_RT_TEXTURED_SLOTS: &[&[MetalBinding]] = &[
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
const WATER_WAKE_SLOTS: &[MetalBinding] = &[texture("wake_field", 11)];
const WATER_SEA_SLOTS: &[&[MetalBinding]] = &[
    TRANSPARENT_SLOTS,
    TRANSPARENT_PROBE_SLOTS,
    SCENE_COLOR_SLOTS,
    PLANAR_SLOTS,
    WATER_BODY_LIGHT_SLOTS,
    WATER_OCEAN_SLOTS,
    WATER_WAKE_SLOTS,
];
const WATER_SEA_RT_SLOTS: &[&[MetalBinding]] = &[
    TRANSPARENT_SLOTS,
    TRANSPARENT_PROBE_SLOTS,
    SCENE_COLOR_SLOTS,
    PLANAR_SLOTS,
    TRANSPARENT_TRACE_SLOTS,
    WATER_BODY_LIGHT_SLOTS,
    WATER_OCEAN_SLOTS,
    WATER_WAKE_SLOTS,
];
const WATER_SEA_RT_TEXTURED_SLOTS: &[&[MetalBinding]] = &[
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
