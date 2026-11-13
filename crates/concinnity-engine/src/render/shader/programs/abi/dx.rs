//! The registers each DirectX program's root signature binds, which the device
//! build checks against the registers the HLSL declares.

use super::DxilAbi;
use super::atmosphere::dx as atmosphere;
use super::clouds::dx as clouds;
use super::composite::dx as composite;
use super::cull::dx as cull;
use super::decal::dx as decal;
use super::exposure::dx as exposure;
use super::fog::dx as fog;
use super::gbuffer_sky::dx as gbuffer_sky;
use super::grass::dx as grass;
use super::history::dx as history;
use super::hiz::dx as hiz;
use super::light_cull::dx as light_cull;
use super::line::dx as line;
use super::main::dx as main;
use super::ocean::dx as ocean;
use super::particle::dx as particle;
use super::probe::dx as probe;
use super::receivers::dx as receivers;
use super::register::DxBinding;
use super::rt_reflections::dx as rt;
use super::shadow::dx as shadow;
use super::skin::dx as skin;
use super::sky::dx as sky;
use super::text::dx as text;
use super::transparent::dx as transparent;
use super::wake::dx as wake;
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
    main::FAR_SHADOW_MAP.binding(),
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
    // The clouds: the noise, the march kernels under one root signature, the
    // reconstruction that also reads the march and last frame, and the composite.
    DxilAbi {
        program: &shared::CLOUD_NOISE,
        registers: &[clouds::noise::NOISE_OUT.binding()],
    },
    DxilAbi {
        program: &shared::CLOUD_MARCH,
        registers: &[
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
        ],
    },
    DxilAbi {
        program: &shared::CLOUD_PANORAMA,
        registers: &[
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
        ],
    },
    DxilAbi {
        program: &shared::CLOUD_SHADOW,
        registers: &[
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
        ],
    },
    DxilAbi {
        program: &shared::CLOUD_RECONSTRUCT,
        registers: &[
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
            clouds::march::MARCHED.binding(),
            clouds::march::PREVIOUS.binding(),
        ],
    },
    DxilAbi {
        program: &shared::CLOUD_COMPOSITE_FRAG,
        registers: &[
            clouds::composite::CLOUDS.binding(),
            clouds::composite::SCENE_DEPTH.binding(),
            clouds::composite::TARGET.binding(),
        ],
    },
    // The atmosphere's table kernels, one root signature declaring what each
    // reads, the sky drawn from the sky-view table, and the haze.
    DxilAbi {
        program: &shared::ATMOSPHERE_TRANSMITTANCE,
        registers: &[
            atmosphere::kernel::ATMOSPHERE.binding(),
            atmosphere::kernel::OUTPUT.binding(),
            atmosphere::kernel::LUT_SAMPLER.binding(),
        ],
    },
    DxilAbi {
        program: &shared::ATMOSPHERE_MULTISCATTER,
        registers: &[
            atmosphere::kernel::ATMOSPHERE.binding(),
            atmosphere::kernel::TRANSMITTANCE.binding(),
            atmosphere::kernel::OUTPUT.binding(),
            atmosphere::kernel::LUT_SAMPLER.binding(),
        ],
    },
    DxilAbi {
        program: &shared::ATMOSPHERE_SKY_VIEW,
        registers: &[
            atmosphere::kernel::ATMOSPHERE.binding(),
            atmosphere::kernel::TRANSMITTANCE.binding(),
            atmosphere::kernel::MULTISCATTER.binding(),
            atmosphere::kernel::OUTPUT.binding(),
            atmosphere::kernel::LUT_SAMPLER.binding(),
        ],
    },
    DxilAbi {
        program: &shared::ATMOSPHERE_AERIAL,
        registers: &[
            atmosphere::kernel::ATMOSPHERE.binding(),
            atmosphere::kernel::TRANSMITTANCE.binding(),
            atmosphere::kernel::MULTISCATTER.binding(),
            atmosphere::kernel::OUTPUT.binding(),
            atmosphere::kernel::LUT_SAMPLER.binding(),
        ],
    },
    DxilAbi {
        program: &shared::ATMOSPHERE_IRRADIANCE,
        registers: &[
            atmosphere::kernel::ATMOSPHERE.binding(),
            atmosphere::kernel::TRANSMITTANCE.binding(),
            atmosphere::kernel::SKY_VIEW.binding(),
            atmosphere::kernel::CLOUD_PANORAMA.binding(),
            atmosphere::kernel::OUTPUT.binding(),
            atmosphere::kernel::LUT_SAMPLER.binding(),
        ],
    },
    DxilAbi {
        program: &shared::ATMOSPHERE_SKY_CUBE,
        registers: &[
            atmosphere::kernel::ATMOSPHERE.binding(),
            atmosphere::kernel::TRANSMITTANCE.binding(),
            atmosphere::kernel::SKY_VIEW.binding(),
            atmosphere::kernel::CLOUD_PANORAMA.binding(),
            atmosphere::kernel::OUTPUT.binding(),
            atmosphere::kernel::LUT_SAMPLER.binding(),
        ],
    },
    DxilAbi {
        program: &shared::SKY_ATMOSPHERE_FRAG,
        registers: &[
            atmosphere::sky::VIEW.binding(),
            atmosphere::sky::ATMOSPHERE.binding(),
            atmosphere::sky::SKY_VIEW.binding(),
            atmosphere::sky::LUT_SAMPLER.binding(),
        ],
    },
    DxilAbi {
        program: &shared::AERIAL_PERSPECTIVE_FRAG,
        registers: &[
            atmosphere::aerial::ATMOSPHERE.binding(),
            atmosphere::aerial::SCENE_DEPTH.binding(),
            atmosphere::aerial::AERIAL_VOLUME.binding(),
            atmosphere::aerial::TRANSMITTANCE.binding(),
            atmosphere::aerial::MULTISCATTER.binding(),
            atmosphere::aerial::LUT_SAMPLER.binding(),
        ],
    },
    // The composite, every source with its own sampler.
    DxilAbi {
        program: &shared::COMPOSITE_FRAG,
        registers: &[
            composite::PARAMS.binding(),
            composite::HDR.binding(),
            composite::BLOOM.binding(),
            composite::LUT.binding(),
            composite::GBUFFER_NORMAL_DEPTH.binding(),
            composite::GBUFFER_ROUGHNESS.binding(),
            composite::AO.binding(),
            composite::GBUFFER_MOTION.binding(),
            composite::REACTIVE.binding(),
            composite::HDR_SAMPLER.binding(),
            composite::BLOOM_SAMPLER.binding(),
            composite::LUT_SAMPLER.binding(),
            composite::GBUFFER_NORMAL_DEPTH_SAMPLER.binding(),
            composite::GBUFFER_ROUGHNESS_SAMPLER.binding(),
            composite::AO_SAMPLER.binding(),
            composite::GBUFFER_MOTION_SAMPLER.binding(),
            composite::REACTIVE_SAMPLER.binding(),
        ],
    },
    // The volumetric fog's froxel kernel and fullscreen pass.
    DxilAbi {
        program: &shared::FOG_FROXEL,
        registers: &[
            fog::froxel::FOG.binding(),
            fog::froxel::FROXEL.binding(),
            fog::froxel::SHADOW.binding(),
            fog::froxel::SHADOW_MAP.binding(),
            fog::froxel::SHADOW_SAMPLER.binding(),
            fog::froxel::VOLUME.binding(),
            fog::froxel::FAR_SHADOW_MAP.binding(),
        ],
    },
    DxilAbi {
        program: &shared::FOG_FRAG,
        registers: &[
            fog::draw::FOG.binding(),
            fog::draw::FROXEL.binding(),
            fog::draw::SCENE_DEPTH.binding(),
            fog::draw::VOLUME.binding(),
            fog::draw::VOLUME_SAMPLER.binding(),
        ],
    },
    // The bindless main pair, whose layout is a contract with world Shaders as
    // well: each builds its own PSO against the same root signature, so a
    // register that moves misbinds every one.
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
        registers: &[
            history::PARAMS.binding(),
            history::OBJECTS.binding(),
            history::HISTORY.binding(),
        ],
    },
    // The Hi-Z pyramid's two SPD dispatches: the params as root constants, the
    // main depth (phase 1 only) and the per-mip UAV table.
    DxilAbi {
        program: &spd::HIZ_SPD_SINGLE,
        registers: &[
            hiz::PARAMS.binding(),
            hiz::SRC_DEPTH.binding(),
            hiz::SPD_MIPS.binding(),
        ],
    },
    DxilAbi {
        program: &spd::HIZ_SPD_MSAA,
        registers: &[
            hiz::PARAMS.binding(),
            hiz::SRC_DEPTH.binding(),
            hiz::SPD_MIPS.binding(),
        ],
    },
    DxilAbi {
        program: &spd::HIZ_SPD_TAIL,
        registers: &[hiz::PARAMS.binding(), hiz::SPD_MIPS.binding()],
    },
    // The reflection-probe prefilter. The
    // mirror copy and the pyramid reduction share a signature whose UAV table
    // is u0 the source and u1 the destination; the GGX kernel binds one UAV, so
    // its destination is u0 and the capture pyramid takes t0 with the static
    // sampler at s0.
    DxilAbi {
        program: &shared::PROBE_MIP0,
        registers: &[
            probe::mip::PARAMS.binding(),
            probe::mip::SRC_MIP.binding(),
            probe::mip::DST_MIP.binding(),
        ],
    },
    DxilAbi {
        program: &shared::PROBE_DOWNSAMPLE,
        registers: &[
            probe::mip::PARAMS.binding(),
            probe::mip::SRC_MIP.binding(),
            probe::mip::DST_MIP.binding(),
        ],
    },
    DxilAbi {
        program: &shared::PROBE_GGX,
        registers: &[
            probe::ggx::PARAMS.binding(),
            probe::ggx::SRC_CUBE.binding(),
            probe::ggx::SRC_SAMPLER.binding(),
            probe::ggx::DST_MIP.binding(),
        ],
    },
    // The pre-pass compiles from the main pass's file, so a world Shader's
    // pre-pass PSO builds against the same root signature as the engine's.
    DxilAbi {
        program: &shared::MAIN_PREPASS_VERT,
        registers: MAIN_PREPASS_REGISTERS,
    },
    // The grass kernel, under its own root signature.
    DxilAbi {
        program: &shared::GRASS_GENERATE,
        registers: &[
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
        ],
    },
    // The bend pass and the cascade's depth-only draw, each under its own root
    // signature.
    DxilAbi {
        program: &shared::GRASS_BEND,
        registers: &[
            grass::bend::BEND.binding(),
            grass::bend::BEND_FIELD.binding(),
        ],
    },
    DxilAbi {
        program: &shared::GRASS_SHADOW_VERT,
        registers: &[
            grass::shadow::GRASS.binding(),
            grass::shadow::BLADES.binding(),
            grass::shadow::CASCADE.binding(),
        ],
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
        registers: &[gbuffer_sky::GBUFFER_VIEW.binding()],
    },
    DxilAbi {
        program: &shared::GBUFFER_SKY_FRAG,
        registers: &[],
    },
    // The sky: the pass's view block as a root CBV, the
    // prefilter cube in a one-entry table, and a static sampler.
    DxilAbi {
        program: &shared::SKY_VERT,
        registers: &[sky::VIEW.binding()],
    },
    DxilAbi {
        program: &shared::SKY_FRAG,
        registers: &[
            sky::VIEW.binding(),
            sky::PREFILTER_CUBE.binding(),
            sky::CUBE_SAMPLER.binding(),
        ],
    },
    DxilAbi {
        program: &shared::SHADOW_VERT_BINDLESS,
        registers: &[
            shadow::OBJECT_ID.binding(),
            shadow::SHADOW.binding(),
            shadow::PUSH.binding(),
            shadow::OBJECTS.binding(),
            shadow::MESH_VERTICES.binding(),
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
        registers: &[transparent::VIEW.binding()],
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
        registers: &[transparent::VIEW.binding(), transparent::PARAMS.binding()],
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
        registers: &[transparent::VIEW.binding(), transparent::PARAMS.binding()],
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
        registers: &[transparent::VIEW.binding(), transparent::PARAMS.binding()],
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
    // The simulated ocean's kernels, under their own root signature: the root
    // constants, the spectrum and row
    // transforms as root views, and one table of the four mip UAVs.
    // The wake's kernels, under their own root signature: the block as a root
    // CBV, the state and the
    // transform as root views, and a table of the field's two mip views.
    DxilAbi {
        program: &shared::WATER_WAKE_ROWS,
        registers: &[
            wake::PARAMS.binding(),
            wake::STATE.binding(),
            wake::WORK.binding(),
        ],
    },
    DxilAbi {
        program: &shared::WATER_WAKE_COLUMNS,
        registers: &[wake::PARAMS.binding(), wake::WORK.binding()],
    },
    DxilAbi {
        program: &shared::WATER_WAKE_INVERSE,
        registers: &[
            wake::PARAMS.binding(),
            wake::STATE.binding(),
            wake::WORK.binding(),
        ],
    },
    DxilAbi {
        program: &shared::WATER_WAKE_SHADE,
        registers: &[
            wake::PARAMS.binding(),
            wake::STATE.binding(),
            wake::DST_FIELD.binding(),
        ],
    },
    DxilAbi {
        program: &shared::WATER_WAKE_MIP,
        registers: &[wake::DST_FIELD.binding(), wake::SRC_FIELD.binding()],
    },
    DxilAbi {
        program: &shared::WATER_FFT_ROWS,
        registers: &[
            ocean::PARAMS.binding(),
            ocean::SPECTRUM.binding(),
            ocean::ROWS.binding(),
        ],
    },
    DxilAbi {
        program: &shared::WATER_FFT_COLUMNS,
        registers: &[
            ocean::PARAMS.binding(),
            ocean::ROWS.binding(),
            ocean::DST_DISPLACEMENT.binding(),
            ocean::DST_SLOPES.binding(),
        ],
    },
    DxilAbi {
        program: &shared::WATER_FFT_COLUMNS_FOAM,
        registers: &[
            ocean::PARAMS.binding(),
            ocean::ROWS.binding(),
            ocean::DST_DISPLACEMENT.binding(),
            ocean::DST_SLOPES.binding(),
            ocean::FOAM.binding(),
        ],
    },
    DxilAbi {
        program: &shared::WATER_MIP,
        registers: &[
            ocean::PARAMS.binding(),
            ocean::SRC_DISPLACEMENT.binding(),
            ocean::SRC_SLOPES.binding(),
            ocean::DST_DISPLACEMENT.binding(),
            ocean::DST_SLOPES.binding(),
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
    // The skinning, light-cull, auto-exposure, shadow receiver and particle
    // compute families. Each states its registers in a `CN_BACKEND_DIRECTX`
    // branch, so these rows are the check that the branch and the root
    // signature still name the same slots.
    DxilAbi {
        program: &shared::RT_SKIN,
        registers: &[
            skin::SRC.binding(),
            skin::PALETTE.binding(),
            skin::DST.binding(),
            skin::MORPH_DATA.binding(),
            skin::MORPH_WEIGHTS.binding(),
            skin::PARAMS.binding(),
        ],
    },
    DxilAbi {
        program: &shared::LIGHT_CULL,
        registers: &[
            light_cull::CLUSTER.binding(),
            light_cull::LIGHTS.binding(),
            light_cull::CLUSTER_LIST.binding(),
            light_cull::PROBE_RECORDS.binding(),
        ],
    },
    DxilAbi {
        program: &shared::AUTO_EXPOSURE_BUILD,
        registers: &[
            exposure::build::PARAMS.binding(),
            exposure::build::HDR_TEXTURE.binding(),
            exposure::build::HISTOGRAM.binding(),
        ],
    },
    DxilAbi {
        program: &shared::AUTO_EXPOSURE_AVERAGE,
        registers: &[
            exposure::average::PARAMS.binding(),
            exposure::average::HISTOGRAM.binding(),
            exposure::average::MEASUREMENT.binding(),
        ],
    },
    DxilAbi {
        program: &shared::SHADOW_RECEIVERS_REDUCE,
        registers: &[
            receivers::reduce::PARAMS.binding(),
            receivers::reduce::DEPTH.binding(),
            receivers::reduce::PARTIALS.binding(),
        ],
    },
    DxilAbi {
        program: &shared::SHADOW_RECEIVERS_MERGE,
        registers: &[
            receivers::merge::PARAMS.binding(),
            receivers::merge::PARTIALS.binding(),
            receivers::merge::BOUNDS.binding(),
        ],
    },
    DxilAbi {
        program: &shared::PARTICLE_SIMULATE,
        registers: &[
            particle::simulate::PARAMS.binding(),
            particle::simulate::POOL.binding(),
            particle::simulate::FRAME_ROWS.binding(),
        ],
    },
    // The raster remainder: particles, decals, lines and text. Only the particle
    // pair has a `CN_BACKEND_DIRECTX` branch, and only
    // to swap the two constant buffers; the rest are here so an edit to their
    // declarations cannot move a register the root signature binds without
    // failing the build.
    DxilAbi {
        program: &shared::PARTICLE_VERT,
        registers: &[
            particle::draw::VIEW.binding(),
            particle::draw::PARAMS.binding(),
            particle::draw::POOL.binding(),
        ],
    },
    DxilAbi {
        program: &shared::PARTICLE_FRAG,
        registers: &[
            particle::draw::ALBEDO.binding(),
            particle::draw::ALBEDO_SAMPLER.binding(),
            particle::draw::SCENE_DEPTH.binding(),
        ],
    },
    DxilAbi {
        program: &shared::PARTICLE_LIT_VERT,
        registers: &[
            particle::draw::VIEW.binding(),
            particle::draw::PARAMS.binding(),
            particle::draw::POOL.binding(),
        ],
    },
    DxilAbi {
        program: &shared::PARTICLE_LIT_FRAG,
        registers: &[
            particle::draw::ALBEDO.binding(),
            particle::draw::ALBEDO_SAMPLER.binding(),
            particle::draw::SCENE_DEPTH.binding(),
        ],
    },
    DxilAbi {
        program: &shared::DECAL_VERT,
        registers: &[decal::VIEW.binding(), decal::PARAMS.binding()],
    },
    DxilAbi {
        program: &shared::DECAL_FRAG,
        registers: &[
            decal::VIEW.binding(),
            decal::PARAMS.binding(),
            decal::SCENE_DEPTH.binding(),
            decal::DECAL_TEX.binding(),
            decal::DECAL_TEX_SAMPLER.binding(),
        ],
    },
    DxilAbi {
        program: &shared::LINE_VERT,
        registers: &[line::VIEW.binding()],
    },
    DxilAbi {
        program: &shared::LINE_FRAG,
        registers: &[line::VIEW.binding(), line::SCENE_DEPTH.binding()],
    },
    DxilAbi {
        program: &shared::TEXT_VERT,
        registers: &[text::UNIFORMS.binding()],
    },
    DxilAbi {
        program: &shared::TEXT_FRAG,
        registers: &[text::ATLAS.binding(), text::ATLAS_SAMPLER.binding()],
    },
];

// The ray-traced reflection resolve's root signature. The probe cube array, its records and
// the cluster lists binning them sit clear of the screen-space SRVs.
const RT_REFLECTIONS_REGISTERS: &[DxBinding] = &[
    rt::PARAMS.binding(),
    rt::OUTPUT.binding(),
    rt::RAY_LIST.binding(),
    rt::RAY_ARGS.binding(),
    rt::PROBE_SET.binding(),
    rt::CLUSTER.binding(),
    rt::SCENE_TLAS.binding(),
    rt::VERTICES.binding(),
    rt::INDICES.binding(),
    rt::GEOMETRY.binding(),
    rt::SCENE.binding(),
    rt::GBUFFER.binding(),
    rt::ROUGHNESS.binding(),
    rt::PREFILTER.binding(),
    rt::SKINNED_VERTICES.binding(),
    rt::SKINNED_INDICES.binding(),
    rt::PROBE_CUBES.binding(),
    rt::PROBE_RECORDS.binding(),
    rt::CLUSTER_LIST.binding(),
    rt::WAVE_SLOPES.binding(),
    rt::SCREEN_SAMPLER.binding(),
    rt::CUBE_SAMPLER.binding(),
    rt::PROBE_CUBE_SAMPLER.binding(),
    rt::WAVE_SAMPLER.binding(),
];

// The classify shares the trace's root signature: the screen sources, the probe
// fallback and the output at the trace's registers, plus SSGI's pyramid and the
// ray list and arguments it writes.
const RT_CLASSIFY_REGISTERS: &[DxBinding] = &[
    rt::PARAMS.binding(),
    rt::PROBE_SET.binding(),
    rt::CLUSTER.binding(),
    rt::OUTPUT.binding(),
    rt::RAY_LIST_OUT.binding(),
    rt::RAY_ARGS_OUT.binding(),
    rt::SCENE.binding(),
    rt::GBUFFER.binding(),
    rt::ROUGHNESS.binding(),
    rt::PREFILTER.binding(),
    rt::PROBE_CUBES.binding(),
    rt::PROBE_RECORDS.binding(),
    rt::CLUSTER_LIST.binding(),
    rt::PYRAMID.binding(),
    rt::SCREEN_SAMPLER.binding(),
    rt::CUBE_SAMPLER.binding(),
    rt::PROBE_CUBE_SAMPLER.binding(),
];

const RT_REFLECTIONS_TEXTURED_REGISTERS: &[DxBinding] = &[
    rt::TEX_POOL.binding(),
    rt::POOL_SAMPLER.binding(),
    rt::CANOPY.binding(),
    rt::CANOPY_TABLES.binding(),
];

// The transparent pass's root signatures, which glass and water both draw
// under. The probe cube array, its records and the cluster lists sit at the
// same registers in every variant, clear of the trace's SRVs.
const GLASS_REGISTERS: &[DxBinding] = &[
    transparent::VIEW.binding(),
    transparent::PARAMS.binding(),
    transparent::PROBE_SET.binding(),
    transparent::SCENE_COLOR.binding(),
    transparent::SCENE_DEPTH.binding(),
    transparent::PREFILTER_CUBE.binding(),
    transparent::PLANAR_REFLECTION.binding(),
    transparent::PROBE_CUBES.binding(),
    transparent::PROBE_RECORDS.binding(),
    transparent::CLUSTER.binding(),
    transparent::CLUSTER_LIST.binding(),
    transparent::POST_SAMPLER.binding(),
    transparent::CUBE_SAMPLER.binding(),
];

const GLASS_RT_REGISTERS: &[DxBinding] = &[
    transparent::VIEW.binding(),
    transparent::PARAMS.binding(),
    transparent::RT_PARAMS.binding(),
    transparent::PROBE_SET.binding(),
    transparent::SCENE_COLOR.binding(),
    transparent::SCENE_DEPTH.binding(),
    transparent::PREFILTER_CUBE.binding(),
    transparent::SCENE_TLAS.binding(),
    transparent::VERTICES.binding(),
    transparent::INDICES.binding(),
    transparent::SKINNED_VERTICES.binding(),
    transparent::SKINNED_INDICES.binding(),
    transparent::GEOMETRY.binding(),
    transparent::PROBE_CUBES.binding(),
    transparent::PROBE_RECORDS.binding(),
    transparent::CLUSTER.binding(),
    transparent::CLUSTER_LIST.binding(),
    transparent::GLASS_REFLECTION.binding(),
    transparent::GLASS_REFLECTION_BACK.binding(),
];

// The reduced reflection pre-pass entries of both glass producers: the RT set,
// without the refraction snapshot or the second layer, reading the layer it
// peels behind at t11.
const GLASS_REFLECTION_REGISTERS: &[DxBinding] = &[
    transparent::VIEW.binding(),
    transparent::PARAMS.binding(),
    transparent::RT_PARAMS.binding(),
    transparent::PROBE_SET.binding(),
    transparent::SCENE_DEPTH.binding(),
    transparent::PREFILTER_CUBE.binding(),
    transparent::SCENE_TLAS.binding(),
    transparent::VERTICES.binding(),
    transparent::INDICES.binding(),
    transparent::SKINNED_VERTICES.binding(),
    transparent::SKINNED_INDICES.binding(),
    transparent::GEOMETRY.binding(),
    transparent::GLASS_REFLECTION.binding(),
    transparent::PROBE_CUBES.binding(),
    transparent::PROBE_RECORDS.binding(),
    transparent::CLUSTER.binding(),
    transparent::CLUSTER_LIST.binding(),
];

const GLASS_RT_TEXTURED_REGISTERS: &[DxBinding] = &[
    transparent::TEX_POOL.binding(),
    transparent::POOL_SAMPLER.binding(),
];

// Water's RT fragments, which are glass's set plus the planar resolve at t3: a
// water surface with a mirror plane samples it in place of tracing, so unlike
// glass the register survives into the RT variant and the RT root signature
// carries a table for it.
const WATER_RT_REGISTERS: &[DxBinding] = &[
    transparent::VIEW.binding(),
    transparent::PARAMS.binding(),
    transparent::RT_PARAMS.binding(),
    transparent::PROBE_SET.binding(),
    transparent::SCENE_COLOR.binding(),
    transparent::SCENE_DEPTH.binding(),
    transparent::PREFILTER_CUBE.binding(),
    transparent::PLANAR_REFLECTION.binding(),
    transparent::SCENE_TLAS.binding(),
    transparent::VERTICES.binding(),
    transparent::INDICES.binding(),
    transparent::SKINNED_VERTICES.binding(),
    transparent::SKINNED_INDICES.binding(),
    transparent::GEOMETRY.binding(),
    transparent::PROBE_CUBES.binding(),
    transparent::PROBE_RECORDS.binding(),
    transparent::CLUSTER.binding(),
    transparent::CLUSTER_LIST.binding(),
];

// The simulated oceans' cascade arrays and their sampler, a table and a static
// sampler visible to every stage in both signatures.
const WATER_OCEAN_REGISTERS: &[DxBinding] = &[
    transparent::OCEAN_DISPLACEMENT.binding(),
    transparent::OCEAN_SLOPES.binding(),
    transparent::OCEAN_SAMPLER.binding(),
];

// A planet sea's wake field, the third SRV of the oceans' table.
const WATER_WAKE_REGISTERS: &[DxBinding] = &[transparent::WAKE_FIELD.binding()];

// The light every water fragment's body is lit by: the cascaded shadow and the
// sky irradiance cube, a table and a CBV past the probe set in both signatures.
const WATER_BODY_LIGHT_REGISTERS: &[DxBinding] = &[
    transparent::SHADOW.binding(),
    transparent::SHADOW_MAP.binding(),
    transparent::IRRADIANCE_CUBE.binding(),
    transparent::CLOUD_SHADOW.binding(),
    transparent::FAR_SHADOW_MAP.binding(),
    transparent::SHADOW_SAMPLER.binding(),
];

const WATER_RT_TEXTURED_REGISTERS: &[DxBinding] = &[
    transparent::PLANAR_REFLECTION.binding(),
    transparent::TEX_POOL.binding(),
    transparent::POOL_SAMPLER.binding(),
];

// The see-through glass mesh fragment: the RT set above, minus the planar
// resolve at t3 (a curved mesh has no mirror plane) and with the same t20 probe
// cube base, so the pass's RT root signature covers it unchanged.
const GLASS_MESH_RT_REGISTERS: &[DxBinding] = &[
    transparent::VIEW.binding(),
    transparent::PARAMS.binding(),
    transparent::RT_PARAMS.binding(),
    transparent::PROBE_SET.binding(),
    transparent::SCENE_COLOR.binding(),
    transparent::SCENE_DEPTH.binding(),
    transparent::PREFILTER_CUBE.binding(),
    transparent::SCENE_TLAS.binding(),
    transparent::VERTICES.binding(),
    transparent::INDICES.binding(),
    transparent::SKINNED_VERTICES.binding(),
    transparent::SKINNED_INDICES.binding(),
    transparent::GEOMETRY.binding(),
    transparent::PROBE_CUBES.binding(),
    transparent::PROBE_RECORDS.binding(),
    transparent::CLUSTER.binding(),
    transparent::CLUSTER_LIST.binding(),
    transparent::POST_SAMPLER.binding(),
    transparent::CUBE_SAMPLER.binding(),
    transparent::GLASS_REFLECTION.binding(),
    transparent::GLASS_REFLECTION_BACK.binding(),
];
