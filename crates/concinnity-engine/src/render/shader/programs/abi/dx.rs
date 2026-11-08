//! The registers each DirectX program's root signature binds, which the device
//! build checks against the registers the HLSL declares.

use super::DxilAbi;
use super::cull::dx as cull;
use super::main::dx as main;
use super::register::{DxBinding, cbv, sampler, srv, uav};
use crate::render::shader::programs::{commands, shared, spd};

// Every register the bindless main root signature declares.
const MAIN_BINDLESS_REGISTERS: &[DxBinding] = &[
    main::OBJECT_ID.binding(),
    main::VIEW.binding(),
    main::LIGHTS.binding(),
    main::SHADOW.binding(),
    main::PROBE_SET.binding(),
    main::CLUSTER.binding(),
    main::SHADOW_MAP.binding(),
    main::LOCAL_LIGHTS.binding(),
    main::CLUSTER_LIST.binding(),
    main::OBJECTS.binding(),
    main::SCREEN_INDIRECT.binding(),
    main::IRRADIANCE_CUBE.binding(),
    main::PREFILTER_CUBE.binding(),
    main::PROBE_CUBES.binding(),
    main::PROBE_RECORDS.binding(),
    main::SPOT_SHADOWS.binding(),
    main::SPOT_SHADOW_MAP.binding(),
    main::AREA_LIGHTS.binding(),
    main::LTC_MATRIX.binding(),
    main::LTC_MAGNITUDE.binding(),
    main::MATERIAL_PARAMS.binding(),
    main::CLOUD_SHADOW.binding(),
    main::CANOPY.binding(),
    main::CANOPY_TABLES.binding(),
    main::MESH_VERTICES.binding(),
    main::TEX_POOL.binding(),
    main::SHADOW_SAMPLER.binding(),
    main::LINEAR_SAMPLER.binding(),
    main::CUBE_SAMPLER.binding(),
];

// The G-buffer pre-pass root signature: the
// main pass's object id, view block, records, parameter table, pool and
// samplers at their main-pass registers, plus the pre-pass's own view block,
// model history and draw args. Both stages read the one declaration set.
const MAIN_PREPASS_REGISTERS: &[DxBinding] = &[
    main::OBJECT_ID.binding(),
    main::VIEW.binding(),
    main::GBUFFER_VIEW.binding(),
    main::OBJECTS.binding(),
    main::MATERIAL_PARAMS.binding(),
    main::PREV_MODELS.binding(),
    main::DRAW_ARGS.binding(),
    main::MESH_VERTICES.binding(),
    main::TEX_POOL.binding(),
    main::LINEAR_SAMPLER.binding(),
    main::CUBE_SAMPLER.binding(),
];

// The grass block and the visible blades, which the grass draws add to the
// main and pre-pass root signatures.
const GRASS_DRAW_REGISTERS: &[DxBinding] = &[main::GRASS.binding(), main::GRASS_BLADES.binding()];

// Both view-cull phases.
const CULL_REGISTERS: &[DxBinding] = &[
    cull::CULL.binding(),
    cull::OBJECTS.binding(),
    cull::DRAW_ARGS.binding(),
    cull::HIZ.binding(),
    cull::VARIANTS.binding(),
    cull::GROUP_COUNTS.binding(),
    cull::CULL_STATUS.binding(),
];

// The scan and the scatter that pack a cull's draws: one root signature, the
// list lengths and the commands sharing its last slot.
const CULL_SCAN_REGISTERS: &[DxBinding] = &[
    cull::compact::PARAMS.binding(),
    cull::compact::VARIANTS.binding(),
    cull::compact::GROUP_COUNTS.binding(),
    cull::compact::STREAM_COUNTS.binding(),
];
const CULL_SCATTER_REGISTERS: &[DxBinding] = &[
    cull::compact::PARAMS.binding(),
    cull::compact::VARIANTS.binding(),
    cull::compact::DRAW_ARGS.binding(),
    cull::compact::GROUP_COUNTS.binding(),
    cull::compact::CULL_STATUS.binding(),
    cull::compact::COMMANDS.binding(),
];

/// The root signatures the DirectX backend builds. They are pinned because the
/// shader files are shared
/// with Metal and Vulkan, whose hosts bind the same declarations at entirely
/// different slots, so an edit made on either of those platforms cannot see a
/// DirectX root signature at all. A DirectX build checks them, which is where
/// such an edit gets caught.
pub const DXIL_ENTRY_ABI: &[DxilAbi] = &[
    // The bindless main pair, whose layout is a contract with world Shaders as
    // well: each builds its own PSO against the same root signature (see
    // directx/world_shaders.rs), so a register that moves misbinds every one.
    // Both stages read the one declaration set.
    DxilAbi {
        program: &shared::MAIN_BINDLESS_FRAG,
        registers: MAIN_BINDLESS_REGISTERS,
    },
    DxilAbi {
        program: &shared::MAIN_BINDLESS_CANOPY_FRAG,
        registers: MAIN_BINDLESS_REGISTERS,
    },
    DxilAbi {
        program: &shared::MAIN_BINDLESS_CANOPY_DAYLIGHT_FRAG,
        registers: MAIN_BINDLESS_REGISTERS,
    },
    DxilAbi {
        program: &shared::MAIN_BINDLESS_GROUND_FRAG,
        registers: MAIN_BINDLESS_REGISTERS,
    },
    DxilAbi {
        program: &shared::MAIN_BINDLESS_GROUND_DAYLIGHT_FRAG,
        registers: MAIN_BINDLESS_REGISTERS,
    },
    // The draw cull. Its registers are pinned in the source because declaration
    // order would hand `cull_status` u0 and `group_counts` u1, the reverse of
    // what the cull's root signature binds; the shadow variant declares no Hi-Z or
    // variant block.
    DxilAbi {
        program: &shared::CULL_EARLY,
        registers: CULL_REGISTERS,
    },
    DxilAbi {
        program: &shared::CULL_LATE,
        registers: CULL_REGISTERS,
    },
    DxilAbi {
        program: &shared::CULL_SHADOW,
        registers: &[
            cull::CULL.binding(),
            cull::OBJECTS.binding(),
            cull::DRAW_ARGS.binding(),
            cull::GROUP_COUNTS.binding(),
            cull::CULL_STATUS.binding(),
        ],
    },
    DxilAbi {
        program: &shared::CULL_SCAN,
        registers: CULL_SCAN_REGISTERS,
    },
    DxilAbi {
        program: &commands::CULL_SCATTER,
        registers: CULL_SCATTER_REGISTERS,
    },
    DxilAbi {
        program: &shared::MODEL_HISTORY,
        registers: &[cbv("params", 0), srv("objects", 0), uav("history", 0)],
    },
    // The Hi-Z pyramid's two SPD dispatches, from `directx/hiz.rs`: the params
    // as root constants at b0, the main depth at t0 (phase 1 only) and the
    // per-mip UAV table at u0.
    DxilAbi {
        program: &spd::HIZ_SPD_SINGLE,
        registers: &[cbv("params", 0), srv("src_depth", 0), uav("spd_mips", 0)],
    },
    DxilAbi {
        program: &spd::HIZ_SPD_MSAA,
        registers: &[cbv("params", 0), srv("src_depth", 0), uav("spd_mips", 0)],
    },
    DxilAbi {
        program: &spd::HIZ_SPD_TAIL,
        registers: &[cbv("params", 0), uav("spd_mips", 0)],
    },
    // The reflection-probe prefilter, from `directx/probe_prefilter.rs`. The
    // mirror copy and the pyramid reduction share a signature whose UAV table
    // is u0 the source and u1 the destination; the GGX kernel binds one UAV, so
    // its destination is u0 and the capture pyramid takes t0 with the static
    // sampler at s0.
    DxilAbi {
        program: &shared::PROBE_MIP0,
        registers: &[cbv("params", 0), uav("src_mip", 0), uav("dst_mip", 1)],
    },
    DxilAbi {
        program: &shared::PROBE_DOWNSAMPLE,
        registers: &[cbv("params", 0), uav("src_mip", 0), uav("dst_mip", 1)],
    },
    DxilAbi {
        program: &shared::PROBE_GGX,
        registers: &[
            cbv("params", 0),
            srv("src_cube", 0),
            sampler("src_sampler", 0),
            uav("dst_mip", 0),
        ],
    },
    // The pre-pass compiles from the main pass's file, so a world Shader's
    // pre-pass PSO builds against the same root signature as the engine's.
    DxilAbi {
        program: &shared::MAIN_PREPASS_VERT,
        registers: MAIN_PREPASS_REGISTERS,
    },
    // The grass kernel, under its own root signature in `directx/grass.rs`.
    DxilAbi {
        program: &shared::GRASS_GENERATE,
        registers: &[
            cbv("grass", 0),
            uav("blades_out", 1),
            uav("draw_args", 2),
            srv("terrain_heights", 3),
            srv("grass_masks", 4),
            srv("grass_hiz", 5),
            srv("grass_bend_field", 6),
            srv("planet_tiles", 7),
            srv("mesh_vertices", 8),
            srv("grass_tiles", 9),
        ],
    },
    // The bend pass and the cascade's depth-only draw, each under its own root
    // signature in `directx/grass.rs`.
    DxilAbi {
        program: &shared::GRASS_BEND,
        registers: &[cbv("bend", 0), uav("bend_field", 1)],
    },
    DxilAbi {
        program: &shared::GRASS_SHADOW_VERT,
        registers: &[cbv("grass", 0), srv("grass_blades", 1)],
    },
    // The grass draws compile from the main pass's resources, so each stage
    // states those registers and the grass pair on top of them.
    DxilAbi {
        program: &shared::GRASS_VERT,
        registers: MAIN_BINDLESS_REGISTERS,
    },
    DxilAbi {
        program: &shared::GRASS_VERT,
        registers: GRASS_DRAW_REGISTERS,
    },
    DxilAbi {
        program: &shared::GRASS_FRAG,
        registers: MAIN_BINDLESS_REGISTERS,
    },
    DxilAbi {
        program: &shared::GRASS_CROSSFADE_FRAG,
        registers: MAIN_BINDLESS_REGISTERS,
    },
    DxilAbi {
        program: &shared::GRASS_PREPASS_VERT,
        registers: MAIN_PREPASS_REGISTERS,
    },
    DxilAbi {
        program: &shared::GRASS_PREPASS_VERT,
        registers: GRASS_DRAW_REGISTERS,
    },
    // The sky's motion, under its own root signature.
    DxilAbi {
        program: &shared::GBUFFER_SKY_VERT,
        registers: &[cbv("gb_view", 1)],
    },
    DxilAbi {
        program: &shared::GBUFFER_SKY_FRAG,
        registers: &[],
    },
    // The sky, from `directx/sky.rs`: the pass's view block as a root CBV, the
    // prefilter cube in a one-entry table, and a static sampler.
    DxilAbi {
        program: &shared::SKY_VERT,
        registers: &[cbv("view_cb", 0)],
    },
    DxilAbi {
        program: &shared::SKY_FRAG,
        registers: &[
            cbv("view_cb", 0),
            srv("prefilter_cube", 0),
            sampler("cube_sampler", 0),
        ],
    },
    DxilAbi {
        program: &shared::SHADOW_VERT_BINDLESS,
        registers: &[
            cbv("objid_cb", 0),
            cbv("shadow_cb", 1),
            cbv("push", 2),
            srv("objects", 0),
            srv("mesh_vertices", 1),
        ],
    },
    DxilAbi {
        program: &shared::RT_REFLECTIONS_CLASSIFY,
        registers: RT_CLASSIFY_REGISTERS,
    },
    DxilAbi {
        program: &shared::RT_REFLECTIONS_TRACE,
        registers: RT_REFLECTIONS_REGISTERS,
    },
    DxilAbi {
        program: &shared::RT_REFLECTIONS_TRACE_TEXTURED,
        registers: RT_REFLECTIONS_TEXTURED_REGISTERS,
    },
    DxilAbi {
        program: &shared::GLASS_VERT,
        registers: &[cbv("view", 0)],
    },
    DxilAbi {
        program: &shared::GLASS_FRAG,
        registers: GLASS_REGISTERS,
    },
    DxilAbi {
        program: &shared::GLASS_FRAG_RT,
        registers: GLASS_RT_REGISTERS,
    },
    DxilAbi {
        program: &shared::GLASS_FRAG_RT_TEXTURED,
        registers: GLASS_RT_TEXTURED_REGISTERS,
    },
    DxilAbi {
        program: &shared::GLASS_REFLECTION_FRAG,
        registers: GLASS_REFLECTION_REGISTERS,
    },
    DxilAbi {
        program: &shared::GLASS_REFLECTION_FRAG_TEXTURED,
        registers: GLASS_RT_TEXTURED_REGISTERS,
    },
    // The see-through glass MESH producer. Ray-traced only, so it declares the
    // RT register set unconditionally; its vertex stage reads the model matrix
    // out of b1, which is why b1 is visible to every stage in the root
    // signature.
    DxilAbi {
        program: &shared::GLASS_MESH_VERT,
        registers: &[cbv("view", 0), cbv("params", 1)],
    },
    DxilAbi {
        program: &shared::GLASS_MESH_FRAG_RT,
        registers: GLASS_MESH_RT_REGISTERS,
    },
    DxilAbi {
        program: &shared::GLASS_MESH_FRAG_RT_TEXTURED,
        registers: GLASS_RT_TEXTURED_REGISTERS,
    },
    DxilAbi {
        program: &shared::GLASS_MESH_REFLECTION_FRAG,
        registers: GLASS_REFLECTION_REGISTERS,
    },
    DxilAbi {
        program: &shared::GLASS_MESH_REFLECTION_FRAG_TEXTURED,
        registers: GLASS_RT_TEXTURED_REGISTERS,
    },
    // Water shares every register glass declares, which is what lets one root
    // signature per path serve both producers of the transparent pass. The RT
    // pair additionally reads the planar resolve at t3 (see
    // WATER_RT_REGISTERS), which the shared RT signature has a table for, and
    // every water fragment the body's light (WATER_BODY_LIGHT_REGISTERS).
    DxilAbi {
        program: &shared::WATER_VERT,
        registers: &[cbv("view", 0), cbv("params", 1)],
    },
    DxilAbi {
        program: &shared::WATER_VERT,
        registers: WATER_OCEAN_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_FRAG,
        registers: WATER_OCEAN_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_FRAG_RT,
        registers: WATER_OCEAN_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_FRAG_RT_TEXTURED,
        registers: WATER_OCEAN_REGISTERS,
    },
    // A planet sea compiles the same entries under its own gate and binds
    // exactly what the rectangle's do.
    DxilAbi {
        program: &shared::WATER_SEA_VERT,
        registers: &[cbv("view", 0), cbv("params", 1)],
    },
    DxilAbi {
        program: &shared::WATER_SEA_VERT,
        registers: WATER_OCEAN_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_SEA_VERT,
        registers: WATER_WAKE_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_SEA_FRAG,
        registers: WATER_OCEAN_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_SEA_FRAG,
        registers: WATER_WAKE_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_SEA_DAYLIGHT_FRAG,
        registers: WATER_OCEAN_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_SEA_DAYLIGHT_FRAG,
        registers: WATER_WAKE_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_SEA_FRAG_RT,
        registers: WATER_OCEAN_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_SEA_FRAG_RT,
        registers: WATER_WAKE_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_SEA_DAYLIGHT_FRAG_RT,
        registers: WATER_OCEAN_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_SEA_DAYLIGHT_FRAG_RT,
        registers: WATER_WAKE_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_SEA_FRAG_RT_TEXTURED,
        registers: WATER_OCEAN_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_SEA_FRAG_RT_TEXTURED,
        registers: WATER_WAKE_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_SEA_DAYLIGHT_FRAG_RT_TEXTURED,
        registers: WATER_OCEAN_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_SEA_DAYLIGHT_FRAG_RT_TEXTURED,
        registers: WATER_WAKE_REGISTERS,
    },
    // The simulated ocean's kernels, under their own root signature in
    // `directx/water_sim.rs`: the root constants, the spectrum and row
    // transforms as root views, and one table of the four mip UAVs.
    // The wake's kernels, under their own root signature in
    // `directx/water_sim.rs`: the block as a root CBV, the state and the
    // transform as root views, and a table of the field's two mip views.
    DxilAbi {
        program: &shared::WATER_WAKE_ROWS,
        registers: &[cbv("params", 0), uav("state", 1), uav("work", 2)],
    },
    DxilAbi {
        program: &shared::WATER_WAKE_COLUMNS,
        registers: &[cbv("params", 0), uav("work", 2)],
    },
    DxilAbi {
        program: &shared::WATER_WAKE_INVERSE,
        registers: &[cbv("params", 0), uav("state", 1), uav("work", 2)],
    },
    DxilAbi {
        program: &shared::WATER_WAKE_SHADE,
        registers: &[cbv("params", 0), uav("state", 1), uav("dst_field", 3)],
    },
    DxilAbi {
        program: &shared::WATER_WAKE_MIP,
        registers: &[uav("dst_field", 3), srv("src_field", 4)],
    },
    DxilAbi {
        program: &shared::WATER_FFT_ROWS,
        registers: &[cbv("params", 0), srv("spectrum", 5), uav("rows", 6)],
    },
    DxilAbi {
        program: &shared::WATER_FFT_COLUMNS,
        registers: &[
            cbv("params", 0),
            uav("rows", 6),
            uav("dst_displacement", 2),
            uav("dst_slopes", 3),
        ],
    },
    DxilAbi {
        program: &shared::WATER_FFT_COLUMNS_FOAM,
        registers: &[
            cbv("params", 0),
            uav("rows", 6),
            uav("dst_displacement", 2),
            uav("dst_slopes", 3),
            uav("foam", 7),
        ],
    },
    DxilAbi {
        program: &shared::WATER_MIP,
        registers: &[
            cbv("params", 0),
            srv("src_displacement", 0),
            srv("src_slopes", 1),
            uav("dst_displacement", 2),
            uav("dst_slopes", 3),
        ],
    },
    DxilAbi {
        program: &shared::WATER_FRAG,
        registers: GLASS_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_FRAG,
        registers: WATER_BODY_LIGHT_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_FRAG_RT,
        registers: WATER_BODY_LIGHT_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_FRAG_RT,
        registers: WATER_RT_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_FRAG_RT_TEXTURED,
        registers: WATER_RT_TEXTURED_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_SEA_FRAG,
        registers: GLASS_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_SEA_DAYLIGHT_FRAG,
        registers: GLASS_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_SEA_FRAG,
        registers: WATER_BODY_LIGHT_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_SEA_DAYLIGHT_FRAG,
        registers: WATER_BODY_LIGHT_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_SEA_FRAG_RT,
        registers: WATER_BODY_LIGHT_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_SEA_DAYLIGHT_FRAG_RT,
        registers: WATER_BODY_LIGHT_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_SEA_FRAG_RT,
        registers: WATER_RT_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_SEA_DAYLIGHT_FRAG_RT,
        registers: WATER_RT_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_SEA_FRAG_RT_TEXTURED,
        registers: WATER_RT_TEXTURED_REGISTERS,
    },
    DxilAbi {
        program: &shared::WATER_SEA_DAYLIGHT_FRAG_RT_TEXTURED,
        registers: WATER_RT_TEXTURED_REGISTERS,
    },
    // The compute families whose root signatures live in
    // `directx/{raytrace,light_cull,auto_exposure,particle}.rs`. Each
    // states its registers in a `CN_BACKEND_DIRECTX` branch, so these rows are the
    // check that the branch and the root signature still name the same slots.
    DxilAbi {
        program: &shared::RT_SKIN,
        registers: &[
            srv("src", 0),
            srv("palette", 1),
            uav("dst", 0),
            srv("morph_data", 2),
            srv("morph_weights", 3),
            cbv("params", 0),
        ],
    },
    DxilAbi {
        program: &shared::LIGHT_CULL,
        registers: &[
            cbv("cluster", 0),
            srv("lights", 0),
            uav("cluster_list", 0),
            srv("probe_records", 1),
        ],
    },
    DxilAbi {
        program: &shared::AUTO_EXPOSURE_BUILD,
        registers: &[cbv("params", 0), srv("hdr_texture", 0), uav("histogram", 0)],
    },
    DxilAbi {
        program: &shared::AUTO_EXPOSURE_AVERAGE,
        registers: &[cbv("params", 0), uav("histogram", 0), uav("measurement", 1)],
    },
    DxilAbi {
        program: &shared::PARTICLE_SIMULATE,
        registers: &[cbv("params", 0), uav("pool", 0), srv("frame_rows", 1)],
    },
    // The raster remainder, from `directx/{particle,decal,line}.rs` and
    // `pipeline.rs`. Only the particle pair has a `CN_BACKEND_DIRECTX` branch, and only
    // to swap the two constant buffers; the rest are here so an edit to their
    // declarations cannot move a register the root signature binds without
    // failing the build.
    DxilAbi {
        program: &shared::PARTICLE_VERT,
        registers: &[cbv("view", 0), cbv("params", 1), srv("pool", 0)],
    },
    DxilAbi {
        program: &shared::PARTICLE_FRAG,
        registers: &[
            srv("albedo", 1),
            sampler("albedo_sampler", 0),
            srv("scene_depth", 2),
        ],
    },
    DxilAbi {
        program: &shared::PARTICLE_LIT_VERT,
        registers: &[cbv("view", 0), cbv("params", 1), srv("pool", 0)],
    },
    DxilAbi {
        program: &shared::PARTICLE_LIT_FRAG,
        registers: &[
            srv("albedo", 1),
            sampler("albedo_sampler", 0),
            srv("scene_depth", 2),
        ],
    },
    DxilAbi {
        program: &shared::DECAL_VERT,
        registers: &[cbv("view", 0), cbv("params", 1)],
    },
    DxilAbi {
        program: &shared::DECAL_FRAG,
        registers: &[
            cbv("view", 0),
            cbv("params", 1),
            srv("scene_depth", 0),
            srv("decal_tex", 1),
            sampler("decal_tex_sampler", 0),
        ],
    },
    DxilAbi {
        program: &shared::LINE_VERT,
        registers: &[cbv("view", 0)],
    },
    DxilAbi {
        program: &shared::LINE_FRAG,
        registers: &[cbv("view", 0), srv("scene_depth", 0)],
    },
    DxilAbi {
        program: &shared::TEXT_VERT,
        registers: &[cbv("uni", 0)],
    },
    DxilAbi {
        program: &shared::TEXT_FRAG,
        registers: &[srv("atlas", 0), sampler("atlas_sampler", 0)],
    },
];

// The ray-traced reflection resolve's root signature, from
// `directx/post/rt_reflections.rs`. The probe cube array, its records and
// the cluster lists binning them sit clear of the screen-space SRVs.
const RT_REFLECTIONS_REGISTERS: &[DxBinding] = &[
    cbv("rt_params", 0),
    uav("output", 0),
    srv("ray_list", 15),
    srv("ray_args", 16),
    cbv("probe_set", 4),
    cbv("cluster", 5),
    srv("scene_tlas", 0),
    srv("verts", 1),
    srv("indices", 2),
    srv("geom", 3),
    srv("scene_tex", 4),
    srv("gbuffer", 5),
    srv("rough_tex", 6),
    srv("prefilter", 7),
    srv("sverts", 8),
    srv("sidx", 9),
    srv("probe_cubes", 10),
    srv("probe_records", 11),
    srv("cluster_list", 12),
    sampler("screen_sampler", 0),
    sampler("cube_sampler", 1),
    sampler("probe_cube_sampler", 3),
];

// The classify shares the trace's root signature: the screen sources, the probe
// fallback and the output at the trace's registers, plus SSGI's pyramid and the
// ray list and arguments it writes.
const RT_CLASSIFY_REGISTERS: &[DxBinding] = &[
    cbv("rt_params", 0),
    cbv("probe_set", 4),
    cbv("cluster", 5),
    uav("output", 0),
    uav("ray_list", 1),
    uav("ray_args", 2),
    srv("scene_tex", 4),
    srv("gbuffer", 5),
    srv("rough_tex", 6),
    srv("prefilter", 7),
    srv("probe_cubes", 10),
    srv("probe_records", 11),
    srv("cluster_list", 12),
    srv("pyramid", 14),
    sampler("screen_sampler", 0),
    sampler("cube_sampler", 1),
    sampler("probe_cube_sampler", 3),
];

const RT_REFLECTIONS_TEXTURED_REGISTERS: &[DxBinding] = &[
    srv("tex_pool", 0).in_space(1),
    sampler("pool_sampler", 2),
    cbv("canopy_cb", 6),
    srv("canopy_tables_sb", 13),
];

// The transparent pass's root signatures, from `directx/transparent.rs`,
// which glass and water both draw under. The probe cube array, its records and
// the cluster lists sit at t20 / t21 / t22 in every variant, clear of the
// trace's SRVs at t4..t12, with the cluster params at b6.
const GLASS_REGISTERS: &[DxBinding] = &[
    cbv("view", 0),
    cbv("params", 1),
    cbv("probe_set", 4),
    srv("scene_color", 0),
    srv("scene_depth", 1),
    srv("prefilter_cube", 2),
    srv("planar_reflection", 3),
    srv("probe_cubes", 20),
    srv("probe_records", 21),
    cbv("cluster", 6),
    srv("cluster_list", 22),
    sampler("post_samp", 0),
    sampler("cube_sampler", 2),
];

const GLASS_RT_REGISTERS: &[DxBinding] = &[
    cbv("view", 0),
    cbv("params", 1),
    cbv("rt_params", 5),
    cbv("probe_set", 4),
    srv("scene_color", 0),
    srv("scene_depth", 1),
    srv("prefilter_cube", 2),
    srv("scene_tlas", 4),
    srv("verts", 5),
    srv("indices", 6),
    srv("sverts", 8),
    srv("sidx", 9),
    srv("geom", 10),
    srv("probe_cubes", 20),
    srv("probe_records", 21),
    cbv("cluster", 6),
    srv("cluster_list", 22),
    srv("glass_reflection", 11),
    srv("glass_reflection_back", 12),
];

// The reduced reflection pre-pass entries of both glass producers: the RT set,
// without the refraction snapshot or the second layer, reading the layer it
// peels behind at t11.
const GLASS_REFLECTION_REGISTERS: &[DxBinding] = &[
    cbv("view", 0),
    cbv("params", 1),
    cbv("rt_params", 5),
    cbv("probe_set", 4),
    srv("scene_depth", 1),
    srv("prefilter_cube", 2),
    srv("scene_tlas", 4),
    srv("verts", 5),
    srv("indices", 6),
    srv("sverts", 8),
    srv("sidx", 9),
    srv("geom", 10),
    srv("glass_reflection", 11),
    srv("probe_cubes", 20),
    srv("probe_records", 21),
    cbv("cluster", 6),
    srv("cluster_list", 22),
];

const GLASS_RT_TEXTURED_REGISTERS: &[DxBinding] =
    &[srv("tex_pool", 0).in_space(1), sampler("pool_sampler", 1)];

// Water's RT fragments, which are glass's set plus the planar resolve at t3: a
// water surface with a mirror plane samples it in place of tracing, so unlike
// glass the register survives into the RT variant and the RT root signature
// carries a table for it.
const WATER_RT_REGISTERS: &[DxBinding] = &[
    cbv("view", 0),
    cbv("params", 1),
    cbv("rt_params", 5),
    cbv("probe_set", 4),
    srv("scene_color", 0),
    srv("scene_depth", 1),
    srv("prefilter_cube", 2),
    srv("planar_reflection", 3),
    srv("scene_tlas", 4),
    srv("verts", 5),
    srv("indices", 6),
    srv("sverts", 8),
    srv("sidx", 9),
    srv("geom", 10),
    srv("probe_cubes", 20),
    srv("probe_records", 21),
    cbv("cluster", 6),
    srv("cluster_list", 22),
];

// The simulated oceans' cascade arrays and their sampler, a table and a static
// sampler visible to every stage in both signatures.
const WATER_OCEAN_REGISTERS: &[DxBinding] = &[
    srv("ocean_displacement", 15),
    srv("ocean_slopes", 16),
    sampler("ocean_sampler", 4),
];

// A planet sea's wake field, the third SRV of the oceans' table.
const WATER_WAKE_REGISTERS: &[DxBinding] = &[srv("wake_field", 17)];

// The light every water fragment's body is lit by: the cascaded shadow and the
// sky irradiance cube, a table and a CBV past the probe set in both signatures.
const WATER_BODY_LIGHT_REGISTERS: &[DxBinding] = &[
    cbv("shadow_cb", 3),
    srv("shadow_map", 13),
    srv("irradiance_cube", 14),
    srv("cloud_shadow", 18),
    sampler("shadow_sampler", 3),
];

const WATER_RT_TEXTURED_REGISTERS: &[DxBinding] = &[
    srv("planar_reflection", 3),
    srv("tex_pool", 0).in_space(1),
    sampler("pool_sampler", 1),
];

// The see-through glass mesh fragment: the RT set above, minus the planar
// resolve at t3 (a curved mesh has no mirror plane) and with the same t20 probe
// cube base, so the pass's RT root signature covers it unchanged.
const GLASS_MESH_RT_REGISTERS: &[DxBinding] = &[
    cbv("view", 0),
    cbv("params", 1),
    cbv("rt_params", 5),
    cbv("probe_set", 4),
    srv("scene_color", 0),
    srv("scene_depth", 1),
    srv("prefilter_cube", 2),
    srv("scene_tlas", 4),
    srv("verts", 5),
    srv("indices", 6),
    srv("sverts", 8),
    srv("sidx", 9),
    srv("geom", 10),
    srv("probe_cubes", 20),
    srv("probe_records", 21),
    cbv("cluster", 6),
    srv("cluster_list", 22),
    sampler("post_samp", 0),
    sampler("cube_sampler", 2),
    srv("glass_reflection", 11),
    srv("glass_reflection_back", 12),
];
