//! How many threads each compute kernel's group runs. A host sizes its
//! dispatches from the same constant the kernel's `[numthreads]` is checked
//! against, so the two cannot disagree.

use super::{
    atmosphere, clouds, cull, exposure, fog, grass, history, hiz, light_cull, ocean, particle,
    probe, rt_reflections, skin, wake,
};
use crate::render::shader::programs::{
    ShaderProgram, commands, metal as metal_programs, shared, spd,
};

/// A compute kernel's thread-group size, as its `[numthreads(x, y, z)]`
/// declares it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThreadGroup {
    /// Threads along each axis.
    pub size: [u32; 3],
}

impl ThreadGroup {
    /// A group `x` by `y` by `z` threads.
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { size: [x, y, z] }
    }

    /// The groups a dispatch of `threads` threads takes, rounded up on each
    /// axis.
    pub const fn groups(self, threads: [u32; 3]) -> [u32; 3] {
        [
            threads[0].div_ceil(self.size[0]),
            threads[1].div_ceil(self.size[1]),
            threads[2].div_ceil(self.size[2]),
        ]
    }

    /// [`Self::groups`] for a flat grid `threads` across and down.
    pub const fn groups_2d(self, threads: [u32; 2]) -> [u32; 3] {
        self.groups([threads[0], threads[1], 1])
    }

    /// [`Self::groups`] for a line of `threads` threads.
    pub const fn groups_1d(self, threads: u32) -> [u32; 3] {
        self.groups([threads, 1, 1])
    }
}

/// One compute program and the group size its host dispatches it at.
pub struct KernelGroup {
    /// The program.
    pub program: &'static ShaderProgram,
    /// Its group size.
    pub group: ThreadGroup,
}

const fn row(program: &'static ShaderProgram, group: ThreadGroup) -> KernelGroup {
    KernelGroup { program, group }
}

/// Every compute program any backend compiles, with its group size. A test
/// checks each against the `[numthreads]` its entry point declares.
pub const KERNEL_GROUPS: &[KernelGroup] = &[
    row(&shared::CULL_EARLY, cull::CULL_GROUP),
    row(&shared::CULL_LATE, cull::CULL_GROUP),
    row(&shared::CULL_SHADOW, cull::CULL_GROUP),
    row(&shared::CULL_SCAN, cull::SCAN_GROUP),
    row(&commands::CULL_SCATTER, cull::SCATTER_GROUP),
    row(&shared::MODEL_HISTORY, history::SNAPSHOT_GROUP),
    row(&shared::LIGHT_CULL, light_cull::CLUSTER_GROUP),
    row(&shared::RT_SKIN, skin::SKIN_GROUP),
    row(&shared::PROBE_MIP0, probe::PREFILTER_GROUP),
    row(&shared::PROBE_DOWNSAMPLE, probe::PREFILTER_GROUP),
    row(&shared::PROBE_GGX, probe::PREFILTER_GROUP),
    row(&shared::FOG_FROXEL, fog::FROXEL_GROUP),
    row(
        &shared::ATMOSPHERE_TRANSMITTANCE,
        atmosphere::TRANSMITTANCE_GROUP,
    ),
    row(
        &shared::ATMOSPHERE_MULTISCATTER,
        atmosphere::MULTISCATTER_GROUP,
    ),
    row(&shared::ATMOSPHERE_SKY_VIEW, atmosphere::SKY_VIEW_GROUP),
    row(&shared::ATMOSPHERE_AERIAL, atmosphere::AERIAL_GROUP),
    row(&shared::ATMOSPHERE_IRRADIANCE, atmosphere::IRRADIANCE_GROUP),
    row(&shared::ATMOSPHERE_SKY_CUBE, atmosphere::SKY_CUBE_GROUP),
    row(&shared::CLOUD_NOISE, clouds::NOISE_GROUP),
    row(&shared::CLOUD_MARCH, clouds::MARCH_GROUP),
    row(&shared::CLOUD_PANORAMA, clouds::PANORAMA_GROUP),
    row(&shared::CLOUD_SHADOW, clouds::SHADOW_GROUP),
    row(&shared::CLOUD_RECONSTRUCT, clouds::RECONSTRUCT_GROUP),
    row(&shared::AUTO_EXPOSURE_BUILD, exposure::BUILD_GROUP),
    row(&shared::AUTO_EXPOSURE_AVERAGE, exposure::AVERAGE_GROUP),
    row(&shared::PARTICLE_SIMULATE, particle::SIMULATE_GROUP),
    row(
        &shared::RT_REFLECTIONS_CLASSIFY,
        rt_reflections::CLASSIFY_GROUP,
    ),
    row(&shared::RT_REFLECTIONS_TRACE, rt_reflections::TRACE_GROUP),
    row(
        &shared::RT_REFLECTIONS_TRACE_TEXTURED,
        rt_reflections::TRACE_GROUP,
    ),
    row(&shared::WATER_FFT_ROWS, ocean::FFT_GROUP),
    row(&shared::WATER_FFT_COLUMNS, ocean::FFT_GROUP),
    row(&shared::WATER_FFT_COLUMNS_FOAM, ocean::FFT_GROUP),
    row(&shared::WATER_MIP, ocean::MIP_GROUP),
    row(&shared::WATER_WAKE_ROWS, wake::FFT_GROUP),
    row(&shared::WATER_WAKE_COLUMNS, wake::FFT_GROUP),
    row(&shared::WATER_WAKE_INVERSE, wake::FFT_GROUP),
    row(&shared::WATER_WAKE_SHADE, wake::SHADE_GROUP),
    row(&shared::WATER_WAKE_MIP, wake::MIP_GROUP),
    row(&shared::GRASS_GENERATE, grass::GENERATE_GROUP),
    row(&shared::GRASS_BEND, grass::BEND_GROUP),
    row(&spd::HIZ_SPD_MSAA, hiz::SPD_GROUP),
    row(&spd::HIZ_SPD_SINGLE, hiz::SPD_GROUP),
    row(&spd::HIZ_SPD_TAIL, hiz::SPD_GROUP),
    row(&metal_programs::HIZ_INIT_MSAA, hiz::INIT_GROUP),
    row(&metal_programs::HIZ_INIT_SINGLE, hiz::INIT_GROUP),
    row(&metal_programs::HIZ_DOWNSAMPLE, hiz::DOWNSAMPLE_GROUP),
];

#[cfg(test)]
mod tests;
