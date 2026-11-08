//! The registers each DirectX program's root signature binds, which the device
//! build checks against the registers the HLSL declares.

use super::DxilAbi;
use crate::render::shader::programs::{commands, shared, spd};

// Every register the bindless main root signature in
// `directx/init/pipelines.rs` declares, as (parameter, HLSL register).
const MAIN_BINDLESS_REGISTERS: &[(&str, &str)] = &[
    ("objid_cb", "b0"),
    ("view_cb", "b1"),
    ("lights_cb", "b2"),
    ("shadow_cb", "b3"),
    ("probe_set_cb", "b4"),
    ("cluster_cb", "b5"),
    ("shadow_map", "t0"),
    ("local_lights_sb", "t1"),
    ("cluster_list_sb", "t2"),
    ("objects_sb", "t3"),
    ("screen_indirect", "t4"),
    ("irradiance_cube", "t5"),
    ("prefilter_cube", "t6"),
    ("probe_cubes", "t7"),
    ("probe_records_sb", "t8"),
    ("spot_shadows_sb", "t15"),
    ("spot_shadow_map", "t16"),
    ("area_lights_sb", "t17"),
    ("ltc_matrix", "t18"),
    ("ltc_magnitude", "t19"),
    ("material_params_sb", "t20"),
    ("cloud_shadow", "t25"),
    ("canopy_cb", "b8"),
    ("canopy_tables_sb", "t24"),
    ("mesh_vertices", "t9"),
    ("tex_pool", "t0, space1"),
    ("shadow_sampler", "s0"),
    ("linear_sampler", "s1"),
    ("cube_sampler", "s2"),
];

// The G-buffer pre-pass root signature in `directx/post/gbuffer.rs`: the
// main pass's object id, view block, records, parameter table, pool and
// samplers at their main-pass registers, plus the pre-pass's own view block,
// model history and draw args. Both stages read the one declaration set.
const MAIN_PREPASS_REGISTERS: &[(&str, &str)] = &[
    ("objid_cb", "b0"),
    ("view_cb", "b1"),
    ("gb_view", "b6"),
    ("objects_sb", "t3"),
    ("material_params_sb", "t20"),
    ("prev_models", "t21"),
    ("draw_args", "t22"),
    ("mesh_vertices", "t9"),
    ("tex_pool", "t0, space1"),
    ("linear_sampler", "s1"),
    ("cube_sampler", "s2"),
];

// The grass block and the visible blades, which the grass draws add to the
// main and pre-pass root signatures (`directx/grass.rs`).
const GRASS_DRAW_REGISTERS: &[(&str, &str)] = &[("grass", "b7"), ("grass_blades", "t23")];

// Both view-cull phases, from `directx/cull.rs`.
const CULL_REGISTERS: &[(&str, &str)] = &[
    ("cull", "b0"),
    ("objects", "t0"),
    ("draw_args", "t1"),
    ("hiz_tex", "t2"),
    ("variants", "t3"),
    ("group_counts", "u0"),
    ("cull_status", "u1"),
];

// The scan and the scatter that pack a cull's draws, from `directx/cull.rs`:
// one root signature, the list lengths and the commands sharing its last slot.
const CULL_SCAN_REGISTERS: &[(&str, &str)] = &[
    ("params", "b0"),
    ("variants", "t0"),
    ("group_counts", "u0"),
    ("stream_counts", "u2"),
];
const CULL_SCATTER_REGISTERS: &[(&str, &str)] = &[
    ("params", "b0"),
    ("variants", "t0"),
    ("draw_args", "t1"),
    ("group_counts", "u0"),
    ("cull_status", "u1"),
    ("commands", "u2"),
];

/// The DirectX root signatures the device backend builds, in its
/// `directx/init/pipelines.rs`, `cull.rs`, `hiz.rs`, `probe_prefilter.rs`,
/// `post/gbuffer.rs`, `post/rt_reflections.rs`, `transparent.rs` and the other
/// modules the rows name. They are pinned because the shader files are shared
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
    // what `directx/cull.rs` binds; the shadow variant declares no Hi-Z or
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
            ("cull", "b0"),
            ("objects", "t0"),
            ("draw_args", "t1"),
            ("group_counts", "u0"),
            ("cull_status", "u1"),
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
        registers: &[("params", "b0"), ("objects", "t0"), ("history", "u0")],
    },
    // The Hi-Z pyramid's two SPD dispatches, from `directx/hiz.rs`: the params
    // as root constants at b0, the main depth at t0 (phase 1 only) and the
    // per-mip UAV table at u0.
    DxilAbi {
        program: &spd::HIZ_SPD_SINGLE,
        registers: &[("params", "b0"), ("src_depth", "t0"), ("spd_mips", "u0")],
    },
    DxilAbi {
        program: &spd::HIZ_SPD_MSAA,
        registers: &[("params", "b0"), ("src_depth", "t0"), ("spd_mips", "u0")],
    },
    DxilAbi {
        program: &spd::HIZ_SPD_TAIL,
        registers: &[("params", "b0"), ("spd_mips", "u0")],
    },
    // The reflection-probe prefilter, from `directx/probe_prefilter.rs`. The
    // mirror copy and the pyramid reduction share a signature whose UAV table
    // is u0 the source and u1 the destination; the GGX kernel binds one UAV, so
    // its destination is u0 and the capture pyramid takes t0 with the static
    // sampler at s0.
    DxilAbi {
        program: &shared::PROBE_MIP0,
        registers: &[("params", "b0"), ("src_mip", "u0"), ("dst_mip", "u1")],
    },
    DxilAbi {
        program: &shared::PROBE_DOWNSAMPLE,
        registers: &[("params", "b0"), ("src_mip", "u0"), ("dst_mip", "u1")],
    },
    DxilAbi {
        program: &shared::PROBE_GGX,
        registers: &[
            ("params", "b0"),
            ("src_cube", "t0"),
            ("src_sampler", "s0"),
            ("dst_mip", "u0"),
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
            ("grass", "b0"),
            ("blades_out", "u1"),
            ("draw_args", "u2"),
            ("terrain_heights", "t3"),
            ("grass_masks", "t4"),
            ("grass_hiz", "t5"),
            ("grass_bend_field", "t6"),
            ("planet_tiles", "t7"),
            ("mesh_vertices", "t8"),
            ("grass_tiles", "t9"),
        ],
    },
    // The bend pass and the cascade's depth-only draw, each under its own root
    // signature in `directx/grass.rs`.
    DxilAbi {
        program: &shared::GRASS_BEND,
        registers: &[("bend", "b0"), ("bend_field", "u1")],
    },
    DxilAbi {
        program: &shared::GRASS_SHADOW_VERT,
        registers: &[("grass", "b0"), ("grass_blades", "t1")],
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
        registers: &[("gb_view", "b1")],
    },
    DxilAbi {
        program: &shared::GBUFFER_SKY_FRAG,
        registers: &[],
    },
    // The sky, from `directx/sky.rs`: the pass's view block as a root CBV, the
    // prefilter cube in a one-entry table, and a static sampler.
    DxilAbi {
        program: &shared::SKY_VERT,
        registers: &[("view_cb", "b0")],
    },
    DxilAbi {
        program: &shared::SKY_FRAG,
        registers: &[
            ("view_cb", "b0"),
            ("prefilter_cube", "t0"),
            ("cube_sampler", "s0"),
        ],
    },
    DxilAbi {
        program: &shared::SHADOW_VERT_BINDLESS,
        registers: &[
            ("objid_cb", "b0"),
            ("shadow_cb", "b1"),
            ("push", "b2"),
            ("objects", "t0"),
            ("mesh_vertices", "t1"),
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
        registers: &[("view", "b0")],
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
        registers: &[("view", "b0"), ("params", "b1")],
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
        registers: &[("view", "b0"), ("params", "b1")],
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
        registers: &[("view", "b0"), ("params", "b1")],
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
        registers: &[("params", "b0"), ("state", "u1"), ("work", "u2")],
    },
    DxilAbi {
        program: &shared::WATER_WAKE_COLUMNS,
        registers: &[("params", "b0"), ("work", "u2")],
    },
    DxilAbi {
        program: &shared::WATER_WAKE_INVERSE,
        registers: &[("params", "b0"), ("state", "u1"), ("work", "u2")],
    },
    DxilAbi {
        program: &shared::WATER_WAKE_SHADE,
        registers: &[("params", "b0"), ("state", "u1"), ("dst_field", "u3")],
    },
    DxilAbi {
        program: &shared::WATER_WAKE_MIP,
        registers: &[("dst_field", "u3"), ("src_field", "t4")],
    },
    DxilAbi {
        program: &shared::WATER_FFT_ROWS,
        registers: &[("params", "b0"), ("spectrum", "t5"), ("rows", "u6")],
    },
    DxilAbi {
        program: &shared::WATER_FFT_COLUMNS,
        registers: &[
            ("params", "b0"),
            ("rows", "u6"),
            ("dst_displacement", "u2"),
            ("dst_slopes", "u3"),
        ],
    },
    DxilAbi {
        program: &shared::WATER_FFT_COLUMNS_FOAM,
        registers: &[
            ("params", "b0"),
            ("rows", "u6"),
            ("dst_displacement", "u2"),
            ("dst_slopes", "u3"),
            ("foam", "u7"),
        ],
    },
    DxilAbi {
        program: &shared::WATER_MIP,
        registers: &[
            ("params", "b0"),
            ("src_displacement", "t0"),
            ("src_slopes", "t1"),
            ("dst_displacement", "u2"),
            ("dst_slopes", "u3"),
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
            ("src", "t0"),
            ("palette", "t1"),
            ("dst", "u0"),
            ("morph_data", "t2"),
            ("morph_weights", "t3"),
            ("params", "b0"),
        ],
    },
    DxilAbi {
        program: &shared::LIGHT_CULL,
        registers: &[
            ("cluster", "b0"),
            ("lights", "t0"),
            ("cluster_list", "u0"),
            ("probe_records", "t1"),
        ],
    },
    DxilAbi {
        program: &shared::AUTO_EXPOSURE_BUILD,
        registers: &[("params", "b0"), ("hdr_texture", "t0"), ("histogram", "u0")],
    },
    DxilAbi {
        program: &shared::AUTO_EXPOSURE_AVERAGE,
        registers: &[("params", "b0"), ("histogram", "u0"), ("measurement", "u1")],
    },
    DxilAbi {
        program: &shared::PARTICLE_SIMULATE,
        registers: &[("params", "b0"), ("pool", "u0"), ("frame_rows", "t1")],
    },
    // The raster remainder, from `directx/{particle,decal,line}.rs` and
    // `pipeline.rs`. Only the particle pair has a `CN_BACKEND_DIRECTX` branch, and only
    // to swap the two constant buffers; the rest are here so an edit to their
    // declarations cannot move a register the root signature binds without
    // failing the build.
    DxilAbi {
        program: &shared::PARTICLE_VERT,
        registers: &[("view", "b0"), ("params", "b1"), ("pool", "t0")],
    },
    DxilAbi {
        program: &shared::PARTICLE_FRAG,
        registers: &[
            ("albedo", "t1"),
            ("albedo_sampler", "s0"),
            ("scene_depth", "t2"),
        ],
    },
    DxilAbi {
        program: &shared::PARTICLE_LIT_VERT,
        registers: &[("view", "b0"), ("params", "b1"), ("pool", "t0")],
    },
    DxilAbi {
        program: &shared::PARTICLE_LIT_FRAG,
        registers: &[
            ("albedo", "t1"),
            ("albedo_sampler", "s0"),
            ("scene_depth", "t2"),
        ],
    },
    DxilAbi {
        program: &shared::DECAL_VERT,
        registers: &[("view", "b0"), ("params", "b1")],
    },
    DxilAbi {
        program: &shared::DECAL_FRAG,
        registers: &[
            ("view", "b0"),
            ("params", "b1"),
            ("scene_depth", "t0"),
            ("decal_tex", "t1"),
            ("decal_tex_sampler", "s0"),
        ],
    },
    DxilAbi {
        program: &shared::LINE_VERT,
        registers: &[("view", "b0")],
    },
    DxilAbi {
        program: &shared::LINE_FRAG,
        registers: &[("view", "b0"), ("scene_depth", "t0")],
    },
    DxilAbi {
        program: &shared::TEXT_VERT,
        registers: &[("uni", "b0")],
    },
    DxilAbi {
        program: &shared::TEXT_FRAG,
        registers: &[("atlas", "t0"), ("atlas_sampler", "s0")],
    },
];

// The ray-traced reflection resolve's root signature, from
// `directx/post/rt_reflections.rs`. The probe cube array, its records and
// the cluster lists binning them sit clear of the screen-space SRVs.
const RT_REFLECTIONS_REGISTERS: &[(&str, &str)] = &[
    ("rt_params", "b0"),
    ("output", "u0"),
    ("ray_list", "t15"),
    ("ray_args", "t16"),
    ("probe_set", "b4"),
    ("cluster", "b5"),
    ("scene_tlas", "t0"),
    ("verts", "t1"),
    ("indices", "t2"),
    ("geom", "t3"),
    ("scene_tex", "t4"),
    ("gbuffer", "t5"),
    ("rough_tex", "t6"),
    ("prefilter", "t7"),
    ("sverts", "t8"),
    ("sidx", "t9"),
    ("probe_cubes", "t10"),
    ("probe_records", "t11"),
    ("cluster_list", "t12"),
    ("screen_sampler", "s0"),
    ("cube_sampler", "s1"),
    ("probe_cube_sampler", "s3"),
];

// The classify shares the trace's root signature: the screen sources, the probe
// fallback and the output at the trace's registers, plus SSGI's pyramid and the
// ray list and arguments it writes.
const RT_CLASSIFY_REGISTERS: &[(&str, &str)] = &[
    ("rt_params", "b0"),
    ("probe_set", "b4"),
    ("cluster", "b5"),
    ("output", "u0"),
    ("ray_list", "u1"),
    ("ray_args", "u2"),
    ("scene_tex", "t4"),
    ("gbuffer", "t5"),
    ("rough_tex", "t6"),
    ("prefilter", "t7"),
    ("probe_cubes", "t10"),
    ("probe_records", "t11"),
    ("cluster_list", "t12"),
    ("pyramid", "t14"),
    ("screen_sampler", "s0"),
    ("cube_sampler", "s1"),
    ("probe_cube_sampler", "s3"),
];

const RT_REFLECTIONS_TEXTURED_REGISTERS: &[(&str, &str)] = &[
    ("tex_pool", "t0, space1"),
    ("pool_sampler", "s2"),
    ("canopy_cb", "b6"),
    ("canopy_tables_sb", "t13"),
];

// The transparent pass's root signatures, from `directx/transparent.rs`,
// which glass and water both draw under. The probe cube array, its records and
// the cluster lists sit at t20 / t21 / t22 in every variant, clear of the
// trace's SRVs at t4..t12, with the cluster params at b6.
const GLASS_REGISTERS: &[(&str, &str)] = &[
    ("view", "b0"),
    ("params", "b1"),
    ("probe_set", "b4"),
    ("scene_color", "t0"),
    ("scene_depth", "t1"),
    ("prefilter_cube", "t2"),
    ("planar_reflection", "t3"),
    ("probe_cubes", "t20"),
    ("probe_records", "t21"),
    ("cluster", "b6"),
    ("cluster_list", "t22"),
    ("post_samp", "s0"),
    ("cube_sampler", "s2"),
];

const GLASS_RT_REGISTERS: &[(&str, &str)] = &[
    ("view", "b0"),
    ("params", "b1"),
    ("rt_params", "b5"),
    ("probe_set", "b4"),
    ("scene_color", "t0"),
    ("scene_depth", "t1"),
    ("prefilter_cube", "t2"),
    ("scene_tlas", "t4"),
    ("verts", "t5"),
    ("indices", "t6"),
    ("sverts", "t8"),
    ("sidx", "t9"),
    ("geom", "t10"),
    ("probe_cubes", "t20"),
    ("probe_records", "t21"),
    ("cluster", "b6"),
    ("cluster_list", "t22"),
    ("glass_reflection", "t11"),
    ("glass_reflection_back", "t12"),
];

// The reduced reflection pre-pass entries of both glass producers: the RT set,
// without the refraction snapshot or the second layer, reading the layer it
// peels behind at t11.
const GLASS_REFLECTION_REGISTERS: &[(&str, &str)] = &[
    ("view", "b0"),
    ("params", "b1"),
    ("rt_params", "b5"),
    ("probe_set", "b4"),
    ("scene_depth", "t1"),
    ("prefilter_cube", "t2"),
    ("scene_tlas", "t4"),
    ("verts", "t5"),
    ("indices", "t6"),
    ("sverts", "t8"),
    ("sidx", "t9"),
    ("geom", "t10"),
    ("glass_reflection", "t11"),
    ("probe_cubes", "t20"),
    ("probe_records", "t21"),
    ("cluster", "b6"),
    ("cluster_list", "t22"),
];

const GLASS_RT_TEXTURED_REGISTERS: &[(&str, &str)] =
    &[("tex_pool", "t0, space1"), ("pool_sampler", "s1")];

// Water's RT fragments, which are glass's set plus the planar resolve at t3: a
// water surface with a mirror plane samples it in place of tracing, so unlike
// glass the register survives into the RT variant and the RT root signature
// carries a table for it.
const WATER_RT_REGISTERS: &[(&str, &str)] = &[
    ("view", "b0"),
    ("params", "b1"),
    ("rt_params", "b5"),
    ("probe_set", "b4"),
    ("scene_color", "t0"),
    ("scene_depth", "t1"),
    ("prefilter_cube", "t2"),
    ("planar_reflection", "t3"),
    ("scene_tlas", "t4"),
    ("verts", "t5"),
    ("indices", "t6"),
    ("sverts", "t8"),
    ("sidx", "t9"),
    ("geom", "t10"),
    ("probe_cubes", "t20"),
    ("probe_records", "t21"),
    ("cluster", "b6"),
    ("cluster_list", "t22"),
];

// The simulated oceans' cascade arrays and their sampler, a table and a static
// sampler visible to every stage in both signatures.
const WATER_OCEAN_REGISTERS: &[(&str, &str)] = &[
    ("ocean_displacement", "t15"),
    ("ocean_slopes", "t16"),
    ("ocean_sampler", "s4"),
];

// A planet sea's wake field, the third SRV of the oceans' table.
const WATER_WAKE_REGISTERS: &[(&str, &str)] = &[("wake_field", "t17")];

// The light every water fragment's body is lit by: the cascaded shadow and the
// sky irradiance cube, a table and a CBV past the probe set in both signatures.
const WATER_BODY_LIGHT_REGISTERS: &[(&str, &str)] = &[
    ("shadow_cb", "b3"),
    ("shadow_map", "t13"),
    ("irradiance_cube", "t14"),
    ("cloud_shadow", "t18"),
    ("shadow_sampler", "s3"),
];

const WATER_RT_TEXTURED_REGISTERS: &[(&str, &str)] = &[
    ("planar_reflection", "t3"),
    ("tex_pool", "t0, space1"),
    ("pool_sampler", "s1"),
];

// The see-through glass mesh fragment: the RT set above, minus the planar
// resolve at t3 (a curved mesh has no mirror plane) and with the same t20 probe
// cube base, so the pass's RT root signature covers it unchanged.
const GLASS_MESH_RT_REGISTERS: &[(&str, &str)] = &[
    ("view", "b0"),
    ("params", "b1"),
    ("rt_params", "b5"),
    ("probe_set", "b4"),
    ("scene_color", "t0"),
    ("scene_depth", "t1"),
    ("prefilter_cube", "t2"),
    ("scene_tlas", "t4"),
    ("verts", "t5"),
    ("indices", "t6"),
    ("sverts", "t8"),
    ("sidx", "t9"),
    ("geom", "t10"),
    ("probe_cubes", "t20"),
    ("probe_records", "t21"),
    ("cluster", "b6"),
    ("cluster_list", "t22"),
    ("post_samp", "s0"),
    ("cube_sampler", "s2"),
    ("glass_reflection", "t11"),
    ("glass_reflection_back", "t12"),
];
