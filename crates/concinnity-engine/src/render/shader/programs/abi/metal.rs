//! The slots each Metal program's host encoder writes, which the device build
//! checks against the MSL the precompile emits.

use super::MetalAbi;
use super::atmosphere::metal as atmosphere;
use super::clouds::metal as clouds;
use super::composite::metal as composite;
use super::cull::metal as cull;
use super::decal::metal as decal;
use super::exposure::metal as exposure;
use super::fog::metal as fog;
use super::gbuffer_sky::metal as gbuffer_sky;
use super::grass::metal as grass;
use super::history::metal as history;
use super::hiz::metal as hiz;
use super::light_cull::metal as light_cull;
use super::line::metal as line;
use super::main::metal as main;
use super::ocean::metal as ocean;
use super::particle::metal as particle;
use super::probe::metal as probe;
use super::rt_reflections::metal as rt;
use super::shadow::metal as shadow;
use super::skin::metal as skin;
use super::sky::metal as sky;
use super::slot::{MetalBinding, buffer, sampler, texture};
use super::text::metal as text;
use super::transparent::metal as transparent;
use super::wake::metal as wake;
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
    // The atmosphere's table kernels on one encoder, the sky drawn from the
    // sky-view table, and the haze.
    MetalAbi {
        program: &shared::ATMOSPHERE_TRANSMITTANCE,
        slots: &[&[
            atmosphere::kernel::ATMOSPHERE.binding(),
            atmosphere::kernel::OUTPUT.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::ATMOSPHERE_MULTISCATTER,
        slots: &[&[
            atmosphere::kernel::ATMOSPHERE.binding(),
            atmosphere::kernel::TRANSMITTANCE.binding(),
            atmosphere::kernel::OUTPUT.binding(),
            atmosphere::kernel::LUT_SAMPLER.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::ATMOSPHERE_SKY_VIEW,
        slots: &[&[
            atmosphere::kernel::ATMOSPHERE.binding(),
            atmosphere::kernel::TRANSMITTANCE.binding(),
            atmosphere::kernel::MULTISCATTER.binding(),
            atmosphere::kernel::OUTPUT.binding(),
            atmosphere::kernel::LUT_SAMPLER.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::ATMOSPHERE_AERIAL,
        slots: &[&[
            atmosphere::kernel::ATMOSPHERE.binding(),
            atmosphere::kernel::TRANSMITTANCE.binding(),
            atmosphere::kernel::MULTISCATTER.binding(),
            atmosphere::kernel::OUTPUT.binding(),
            atmosphere::kernel::LUT_SAMPLER.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::ATMOSPHERE_IRRADIANCE,
        slots: &[&[
            atmosphere::kernel::ATMOSPHERE.binding(),
            atmosphere::kernel::TRANSMITTANCE.binding(),
            atmosphere::kernel::SKY_VIEW.binding(),
            atmosphere::kernel::CLOUD_PANORAMA.binding(),
            atmosphere::kernel::OUTPUT.binding(),
            atmosphere::kernel::LUT_SAMPLER.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::ATMOSPHERE_SKY_CUBE,
        slots: &[&[
            atmosphere::kernel::ATMOSPHERE.binding(),
            atmosphere::kernel::TRANSMITTANCE.binding(),
            atmosphere::kernel::SKY_VIEW.binding(),
            atmosphere::kernel::CLOUD_PANORAMA.binding(),
            atmosphere::kernel::OUTPUT.binding(),
            atmosphere::kernel::LUT_SAMPLER.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::SKY_ATMOSPHERE_FRAG,
        slots: &[&[
            atmosphere::sky::VIEW.binding(),
            atmosphere::sky::ATMOSPHERE.binding(),
            atmosphere::sky::SKY_VIEW.binding(),
            atmosphere::sky::LUT_SAMPLER.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::AERIAL_PERSPECTIVE_FRAG,
        slots: &[&[
            atmosphere::aerial::ATMOSPHERE.binding(),
            atmosphere::aerial::SCENE_DEPTH.binding(),
            atmosphere::aerial::AERIAL_VOLUME.binding(),
            atmosphere::aerial::TRANSMITTANCE.binding(),
            atmosphere::aerial::MULTISCATTER.binding(),
            atmosphere::aerial::LUT_SAMPLER.binding(),
        ]],
        argument_ids: &[],
    },
    // The clouds: the noise, the march kernels, the reconstruction that also
    // reads the march and last frame, and the composite.
    MetalAbi {
        program: &shared::CLOUD_NOISE,
        slots: &[&[clouds::noise::NOISE_OUT.binding()]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::CLOUD_MARCH,
        slots: &[&[
            clouds::march::ATMOSPHERE.binding(),
            clouds::march::CLOUDS.binding(),
            clouds::march::SCENE_DEPTH.binding(),
            clouds::march::AERIAL_VOLUME.binding(),
            clouds::march::TRANSMITTANCE.binding(),
            clouds::march::MULTISCATTER.binding(),
            clouds::march::NOISE.binding(),
            clouds::march::OUTPUT.binding(),
            clouds::march::LUT_SAMPLER.binding(),
            clouds::march::NOISE_SAMPLER.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::CLOUD_PANORAMA,
        slots: &[&[
            clouds::march::ATMOSPHERE.binding(),
            clouds::march::CLOUDS.binding(),
            clouds::march::TRANSMITTANCE.binding(),
            clouds::march::MULTISCATTER.binding(),
            clouds::march::NOISE.binding(),
            clouds::march::OUTPUT.binding(),
            clouds::march::LUT_SAMPLER.binding(),
            clouds::march::NOISE_SAMPLER.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::CLOUD_SHADOW,
        slots: &[&[
            clouds::march::CLOUDS.binding(),
            clouds::march::NOISE.binding(),
            clouds::march::OUTPUT.binding(),
            clouds::march::NOISE_SAMPLER.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::CLOUD_RECONSTRUCT,
        slots: &[&[
            clouds::march::ATMOSPHERE.binding(),
            clouds::march::CLOUDS.binding(),
            clouds::march::SCENE_DEPTH.binding(),
            clouds::march::OUTPUT.binding(),
            clouds::march::LUT_SAMPLER.binding(),
            clouds::march::MARCHED.binding(),
            clouds::march::PREVIOUS.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::CLOUD_COMPOSITE_FRAG,
        slots: &[&[
            clouds::composite::CLOUDS.binding(),
            clouds::composite::SCENE_DEPTH.binding(),
            clouds::composite::TARGET.binding(),
        ]],
        argument_ids: &[],
    },
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
    // what its encoder binds.
    MetalAbi {
        program: &shared::MODEL_HISTORY,
        slots: &[&[
            history::PARAMS.binding(),
            history::OBJECTS.binding(),
            history::HISTORY.binding(),
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
    // The grass kernel: its block, the blades it
    // appends, the draw arguments, and the terrain heights and mask texels it
    // reads, at buffer(0..4) by register, last frame's depth pyramid at
    // texture(5), the bend field, the planet tiles, the shared vertex stream
    // and the tile records at buffer(6..9).
    MetalAbi {
        program: &shared::GRASS_GENERATE,
        slots: &[&[
            grass::generate::GRASS.binding(),
            grass::generate::BLADES_OUT.binding(),
            grass::generate::DRAW_ARGS.binding(),
            grass::generate::TERRAIN_HEIGHTS.binding(),
            grass::generate::GRASS_MASKS.binding(),
            grass::generate::HIZ.binding(),
            grass::generate::BEND_FIELD.binding(),
            grass::generate::PLANET_TILES.binding(),
            grass::generate::MESH_VERTICES.binding(),
            grass::generate::TILES.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::GRASS_BEND,
        slots: &[&[
            grass::bend::BEND.binding(),
            grass::bend::BEND_FIELD.binding(),
        ]],
        argument_ids: &[],
    },
    // The simulated ocean's kernels: the constants
    // at buffer(0), the spectrum and row transforms at buffer(5) and (6), and
    // the source and destination mips at texture(0..3), all by register.
    MetalAbi {
        program: &shared::WATER_FFT_ROWS,
        slots: &[&[
            ocean::PARAMS.binding(),
            ocean::SPECTRUM.binding(),
            ocean::ROWS.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_WAKE_ROWS,
        slots: &[&[
            wake::PARAMS.binding(),
            wake::STATE.binding(),
            wake::WORK.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_WAKE_COLUMNS,
        slots: &[&[wake::PARAMS.binding(), wake::WORK.binding()]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_WAKE_INVERSE,
        slots: &[&[
            wake::PARAMS.binding(),
            wake::STATE.binding(),
            wake::WORK.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_WAKE_SHADE,
        slots: &[&[
            wake::PARAMS.binding(),
            wake::STATE.binding(),
            wake::DST_FIELD.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_WAKE_MIP,
        slots: &[&[wake::DST_FIELD.binding(), wake::SRC_FIELD.binding()]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_FFT_COLUMNS,
        slots: &[&[
            ocean::PARAMS.binding(),
            ocean::ROWS.binding(),
            ocean::DST_DISPLACEMENT.binding(),
            ocean::DST_SLOPES.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_FFT_COLUMNS_FOAM,
        slots: &[&[
            ocean::PARAMS.binding(),
            ocean::ROWS.binding(),
            ocean::DST_DISPLACEMENT.binding(),
            ocean::DST_SLOPES.binding(),
            ocean::FOAM.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::WATER_MIP,
        slots: &[&[
            ocean::PARAMS.binding(),
            ocean::SRC_DISPLACEMENT.binding(),
            ocean::SRC_SLOPES.binding(),
            ocean::DST_DISPLACEMENT.binding(),
            ocean::DST_SLOPES.binding(),
        ]],
        argument_ids: &[],
    },
    // The cascade's blades draw in the shadow pass under their own pipeline,
    // which binds only the shadow block and the cascade's blades.
    MetalAbi {
        program: &shared::GRASS_SHADOW_VERT,
        slots: &[&[
            grass::shadow::GRASS.binding(),
            grass::shadow::BLADES.binding(),
        ]],
        argument_ids: &[],
    },
    // The grass draws at the tail of the pre-pass and the main pass: the
    // pass's view blocks where the surfaces left them, and the grass block and
    // blades at buffer(19) and buffer(20), clear of every main-pass slot.
    MetalAbi {
        program: &shared::GRASS_PREPASS_VERT,
        slots: &[&[
            main::VIEW.binding(),
            main::GBUFFER_VIEW.binding(),
            grass::draw::GRASS.binding(),
            grass::draw::BLADES.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::GRASS_PREPASS_FRAG,
        slots: &[&[main::GBUFFER_VIEW.binding()]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::GRASS_PREPASS_CROSSFADE_FRAG,
        slots: &[&[main::GBUFFER_VIEW.binding()]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::GRASS_VERT,
        slots: &[&[
            main::VIEW.binding(),
            grass::draw::GRASS.binding(),
            grass::draw::BLADES.binding(),
        ]],
        argument_ids: &[],
    },
    // The lit blades read the main pass's lights, shadows, clusters and
    // environment where the surfaces left them, through the same two argument
    // buffers.
    MetalAbi {
        program: &shared::GRASS_FRAG,
        slots: &[&[
            main::VIEW.binding(),
            main::LIGHTS.binding(),
            main::SHADOW.binding(),
            main::TEXTURES.binding(),
            main::LOCAL_LIGHTS.binding(),
            main::SAMPLERS.binding(),
            main::CLUSTER.binding(),
            main::CLUSTER_LIST.binding(),
            main::SPOT_SHADOWS.binding(),
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
            main::VIEW.binding(),
            main::LIGHTS.binding(),
            main::SHADOW.binding(),
            main::TEXTURES.binding(),
            main::LOCAL_LIGHTS.binding(),
            main::SAMPLERS.binding(),
            main::CLUSTER.binding(),
            main::CLUSTER_LIST.binding(),
            main::SPOT_SHADOWS.binding(),
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
    // the surfaces' main-pass view block until the sky's encoder rebinds the
    // sky's own there.
    MetalAbi {
        program: &shared::GBUFFER_SKY_VERT,
        slots: &[&[gbuffer_sky::GBUFFER_VIEW.binding()]],
        argument_ids: &[],
    },
    // The sky: the pass's view block, the prefilter cube
    // and the cube sampler, each at slot 0.
    MetalAbi {
        program: &shared::SKY_VERT,
        slots: &[&[sky::VIEW.binding()]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::SKY_FRAG,
        slots: &[&[
            sky::VIEW.binding(),
            sky::PREFILTER_CUBE.binding(),
            sky::CUBE_SAMPLER.binding(),
        ]],
        argument_ids: &[],
    },
    // The shadow pass (cascades and spot slices): depth-only, its model out of
    // the object record at buffer(9) like the pre-pass, its view index at
    // buffer(7).
    MetalAbi {
        program: &shared::SHADOW_VERT_BINDLESS,
        slots: &[&[
            shadow::SHADOW.binding(),
            shadow::MESH_VERTICES.binding(),
            shadow::CASCADE.binding(),
            shadow::OBJECTS.binding(),
        ]],
        argument_ids: &[],
    },
    // The bindless main pass. Its two argument buffers are whole descriptor
    // sets, each at the index `[[cn::metal_argument_buffer(n)]]` names, and the
    // ids inside the texture one are where the cull writes each member:
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
    // are discrete bindings beside the screen sources, which is what the post
    // encoder binds.
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
            transparent::VIEW.binding(),
            transparent::PARAMS.binding(),
            transparent::OCEAN_DISPLACEMENT.binding(),
            transparent::OCEAN_SAMPLER.binding(),
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
            transparent::VIEW.binding(),
            transparent::PARAMS.binding(),
            transparent::OCEAN_DISPLACEMENT.binding(),
            transparent::OCEAN_SAMPLER.binding(),
            transparent::WAKE_FIELD.binding(),
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
    // rows check the generated binding table against the slots the bloom, text,
    // decal and particle encoders write.
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
        slots: &[&[text::UNIFORMS.binding()]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::TEXT_FRAG,
        slots: &[&[text::ATLAS.binding(), text::ATLAS_SAMPLER.binding()]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::DECAL_VERT,
        slots: &[&[decal::VIEW.binding(), decal::PARAMS.binding()]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::DECAL_FRAG,
        slots: &[&[
            decal::VIEW.binding(),
            decal::PARAMS.binding(),
            decal::SCENE_DEPTH.binding(),
            decal::DECAL_TEX.binding(),
            decal::DECAL_TEX_SAMPLER.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::PARTICLE_VERT,
        slots: &[&[
            particle::draw::POOL.binding(),
            particle::draw::VIEW.binding(),
            particle::draw::PARAMS.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::PARTICLE_FRAG,
        slots: &[&[
            particle::draw::ALBEDO.binding(),
            particle::draw::ALBEDO_SAMPLER.binding(),
            particle::draw::SCENE_DEPTH.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::PARTICLE_LIT_VERT,
        slots: &[&[
            particle::draw::POOL.binding(),
            particle::draw::VIEW.binding(),
            particle::draw::PARAMS.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::PARTICLE_LIT_FRAG,
        slots: &[&[
            particle::draw::ALBEDO.binding(),
            particle::draw::ALBEDO_SAMPLER.binding(),
            particle::draw::SCENE_DEPTH.binding(),
        ]],
        argument_ids: &[],
    },
    // The fullscreen post fragments. Each source is one texture/sampler pair at
    // the index its register names, and the constants ride buffer(0), which is
    // what the post encoders bind.
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
    // the fog encoder binds, and the opposite of where the emitter would place
    // it unaided.
    MetalAbi {
        program: &shared::FOG_FROXEL,
        slots: &[&[
            fog::froxel::FOG.binding(),
            fog::froxel::FROXEL.binding(),
            fog::froxel::SHADOW.binding(),
            fog::froxel::SHADOW_MAP.binding(),
            fog::froxel::SHADOW_SAMPLER.binding(),
            fog::froxel::VOLUME.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::FOG_FRAG,
        slots: &[&[
            fog::draw::FOG.binding(),
            fog::draw::FROXEL.binding(),
            fog::draw::SCENE_DEPTH.binding(),
            fog::draw::VOLUME.binding(),
            fog::draw::VOLUME_SAMPLER.binding(),
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
    // in the tree. The reflection encoder binds each source at its declaration
    // index and repeats one sampler across the same range, and
    // the post encoder puts the composite constants on buffer(0).
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
            composite::PARAMS.binding(),
            composite::HDR.binding(),
            composite::HDR_SAMPLER.binding(),
            composite::BLOOM.binding(),
            composite::BLOOM_SAMPLER.binding(),
            composite::LUT.binding(),
            composite::LUT_SAMPLER.binding(),
            composite::GBUFFER_NORMAL_DEPTH.binding(),
            composite::GBUFFER_NORMAL_DEPTH_SAMPLER.binding(),
            composite::GBUFFER_ROUGHNESS.binding(),
            composite::GBUFFER_ROUGHNESS_SAMPLER.binding(),
            composite::AO.binding(),
            composite::AO_SAMPLER.binding(),
            composite::GBUFFER_MOTION.binding(),
            composite::GBUFFER_MOTION_SAMPLER.binding(),
            composite::REACTIVE.binding(),
            composite::REACTIVE_SAMPLER.binding(),
        ]],
        argument_ids: &[],
    },
    // The compute families and the world-space line pair. Their Metal indices
    // come from the `register()` annotations too, and the slots below are what
    // the exposure, light-cull, particle, line and skin encoders write.
    MetalAbi {
        program: &shared::AUTO_EXPOSURE_BUILD,
        slots: &[&[
            exposure::build::HDR_TEXTURE.binding(),
            exposure::build::HISTOGRAM.binding(),
            exposure::build::PARAMS.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::AUTO_EXPOSURE_AVERAGE,
        slots: &[&[
            exposure::average::HISTOGRAM.binding(),
            exposure::average::MEASUREMENT.binding(),
            exposure::average::PARAMS.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::LIGHT_CULL,
        slots: &[&[
            light_cull::CLUSTER.binding(),
            light_cull::LIGHTS.binding(),
            light_cull::CLUSTER_LIST.binding(),
            light_cull::PROBE_RECORDS.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::PARTICLE_SIMULATE,
        slots: &[&[
            particle::simulate::POOL.binding(),
            particle::simulate::FRAME_ROWS.binding(),
            particle::simulate::PARAMS.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::RT_SKIN,
        slots: &[&[
            skin::SRC.binding(),
            skin::DST.binding(),
            skin::PALETTE.binding(),
            skin::PARAMS.binding(),
            skin::MORPH_DATA.binding(),
            skin::MORPH_WEIGHTS.binding(),
        ]],
        argument_ids: &[],
    },
    // The Hi-Z pyramid's per-mip chain, which is Metal's half of the builder
    // (Vulkan and DirectX run the SPD trio instead). The slots are what its
    // encoder writes for every dispatch of the chain.
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
            hiz::downsample::PARAMS.binding(),
            hiz::downsample::SRC_HIZ.binding(),
            hiz::downsample::DST_MIP.binding(),
        ]],
        argument_ids: &[],
    },
    // The reflection-probe prefilter, whose three kernels its encoder binds the
    // same way: params, source, destination. The GGX kernel's
    // destination is the one slot D3D and Metal disagree on, so its `u`
    // register moves behind `CN_BACKEND_DIRECTX` and Metal keeps texture(1).
    MetalAbi {
        program: &shared::PROBE_MIP0,
        slots: &[&[
            probe::mip::PARAMS.binding(),
            probe::mip::SRC_MIP.binding(),
            probe::mip::DST_MIP.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::PROBE_DOWNSAMPLE,
        slots: &[&[
            probe::mip::PARAMS.binding(),
            probe::mip::SRC_MIP.binding(),
            probe::mip::DST_MIP.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::PROBE_GGX,
        slots: &[&[
            probe::ggx::PARAMS.binding(),
            probe::ggx::SRC_CUBE.binding(),
            probe::ggx::SRC_SAMPLER.binding(),
            probe::ggx::DST_MIP.binding(),
        ]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::LINE_VERT,
        slots: &[&[line::VIEW.binding()]],
        argument_ids: &[],
    },
    MetalAbi {
        program: &shared::LINE_FRAG,
        slots: &[&[line::VIEW.binding(), line::SCENE_DEPTH.binding()]],
        argument_ids: &[],
    },
];

// Both Hi-Z init variants: the params, the depth they read and the first mip.
const HIZ_INIT_SLOTS: &[&[MetalBinding]] = &[&[
    hiz::init::PARAMS.binding(),
    hiz::init::SRC_DEPTH.binding(),
    hiz::init::DST_MIP.binding(),
]];

// What every transparent fragment binds for its own draw: the view and
// surface blocks, the scene depth and the sky prefilter cube.
const TRANSPARENT_SLOTS: &[MetalBinding] = &[
    transparent::VIEW.binding(),
    transparent::PARAMS.binding(),
    transparent::SCENE_DEPTH.binding(),
    transparent::PREFILTER_CUBE.binding(),
    transparent::PREFILTER_CUBE_SAMPLER.binding(),
];

// The transparent pass's probe set and the cluster grid that bins it.
const TRANSPARENT_PROBE_SLOTS: &[MetalBinding] = &[
    transparent::PROBE_SET.binding(),
    transparent::PROBE_RECORDS.binding(),
    transparent::CLUSTER.binding(),
    transparent::CLUSTER_LIST.binding(),
    transparent::PROBE_CUBES.binding(),
    transparent::PROBE_CUBE_SAMPLER.binding(),
];

// The refraction snapshot every see-through transparent fragment reads.
const SCENE_COLOR_SLOTS: &[MetalBinding] = &[
    transparent::SCENE_COLOR.binding(),
    transparent::SCENE_COLOR_SAMPLER.binding(),
];

// The planar mirror resolve.
const PLANAR_SLOTS: &[MetalBinding] = &[
    transparent::PLANAR_REFLECTION.binding(),
    transparent::PLANAR_REFLECTION_SAMPLER.binding(),
];

// The ray-traced transparent fragments' trace inputs.
const TRANSPARENT_TRACE_SLOTS: &[MetalBinding] = &[
    transparent::RT_PARAMS.binding(),
    transparent::VERTICES.binding(),
    transparent::INDICES.binding(),
    transparent::GEOMETRY.binding(),
    transparent::SCENE_TLAS.binding(),
    transparent::SKINNED_VERTICES.binding(),
    transparent::SKINNED_INDICES.binding(),
];

// Both reflection layers the reduced pre-pass wrote.
const REFLECTION_LAYER_SLOTS: &[MetalBinding] = &[
    transparent::GLASS_REFLECTION.binding(),
    transparent::GLASS_REFLECTION_BACK.binding(),
];

// The reflection layer a pre-pass entry peels behind.
const REFLECTION_PEEL_SLOTS: &[MetalBinding] = &[transparent::GLASS_REFLECTION.binding()];

// A textured transparent variant's bindless pool argument buffer and sampler.
const TRANSPARENT_POOL_SLOTS: &[MetalBinding] = &[
    transparent::TEXTURES.binding(),
    transparent::POOL_SAMPLER.binding(),
];

// Every slot both RT reflection kernels bind, the probe set included, pinned
// by its register(): the screen sources, the probe fallback, the output and the
// ray list and arguments.
const RT_SCREEN_SLOTS: &[MetalBinding] = &[
    rt::PARAMS.binding(),
    rt::PROBE_SET.binding(),
    rt::SCENE.binding(),
    rt::SCENE_SAMPLER.binding(),
    rt::GBUFFER.binding(),
    rt::GBUFFER_SAMPLER.binding(),
    rt::ROUGHNESS.binding(),
    rt::ROUGHNESS_SAMPLER.binding(),
    rt::PREFILTER.binding(),
    rt::PREFILTER_SAMPLER.binding(),
    rt::PROBE_RECORDS.binding(),
    rt::PROBE_CUBES.binding(),
    rt::PROBE_CUBE_SAMPLER.binding(),
    rt::CLUSTER.binding(),
    rt::CLUSTER_LIST.binding(),
    rt::OUTPUT.binding(),
    rt::RAY_LIST.binding(),
    rt::RAY_ARGS.binding(),
];

// The classify's own: SSGI's pyramid.
const RT_CLASSIFY_SLOTS: &[MetalBinding] = &[rt::PYRAMID.binding()];

// The trace's own: the acceleration structure and the geometry it fetches hits
// from.
const RT_REFLECTIONS_SLOTS: &[MetalBinding] = &[
    rt::VERTICES.binding(),
    rt::INDICES.binding(),
    rt::GEOMETRY.binding(),
    rt::SCENE_TLAS.binding(),
    rt::SKINNED_VERTICES.binding(),
    rt::SKINNED_INDICES.binding(),
];

// The textured RT resolve's bindless pool (the main pass's texture block, bound
// at the pool's offset) and the canopy block its grass-covered hits shade from.
const RT_REFLECTIONS_POOL_SLOTS: &[MetalBinding] = &[
    rt::TEXTURES.binding(),
    rt::POOL_SAMPLER.binding(),
    rt::CANOPY.binding(),
    rt::CANOPY_TABLES.binding(),
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
    transparent::SHADOW.binding(),
    transparent::SHADOW_MAP.binding(),
    transparent::IRRADIANCE_CUBE.binding(),
    transparent::CLOUD_SHADOW.binding(),
    transparent::SHADOW_SAMPLER.binding(),
];
// The simulated oceans' cascades, whose slopes and foam every water fragment
// shades from.
const WATER_OCEAN_SLOTS: &[MetalBinding] = &[
    transparent::OCEAN_DISPLACEMENT.binding(),
    transparent::OCEAN_SLOPES.binding(),
    transparent::OCEAN_SAMPLER.binding(),
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
const WATER_WAKE_SLOTS: &[MetalBinding] = &[transparent::WAKE_FIELD.binding()];
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
