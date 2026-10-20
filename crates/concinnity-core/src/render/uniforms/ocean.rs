//! What each dispatch of the ocean simulation reads: the simulation's grid and
//! cascades, how far into the repeat period the frame is, where its cascades
//! whitecap and how much of last frame's foam survives, and, for the mip
//! reduction, which level it writes.

use crate::math::SaturatingFrom;
use crate::render::environment::water::ocean::clock::time_fraction;
use crate::render::environment::water::ocean::{OCEAN_CHOPPINESS, OceanSimulation};

/// Threads in one FFT group: one butterfly per pair of points of the largest
/// cascade. Mirrors `WATER_FFT_THREADS` in `shaders/water_fft.hlsl`.
pub const OCEAN_FFT_THREADS: u32 = 128;

/// The edge of the square group the mip reduction runs in. Mirrors
/// `WATER_MIP_GROUP` in `shaders/water_fft.hlsl`.
pub const OCEAN_MIP_GROUP: u32 = 8;

/// One dispatch's constants. Vulkan pushes them, DirectX hands them over as
/// root constants and Metal writes them to buffer(0). Matches
/// `WaterFftParams` in `shaders/water_fft.hlsl`. 80 bytes.
#[derive(Copy, Clone, Default, Debug, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct OceanFftParams {
    /// `[texels per side, log2 of it, first layer, cascades]`.
    pub grid: [u32; 4],
    /// Each cascade's side, in meters; zero past the cascades.
    pub lengths: [f32; 4],
    /// The Jacobian below which each cascade injects foam
    /// ([`FoamCalibration`](crate::render::environment::water::ocean::foam::FoamCalibration)).
    pub foam_thresholds: [f32; 4],
    /// One over the Jacobian span each cascade's injection ramps over; zero
    /// for a cascade that never foams.
    pub foam_inv_ramps: [f32; 4],
    /// How far into the repeat period the frame is, in `[0, 1)`.
    pub time_fraction: f32,
    /// Horizontal displacement per meter of height
    /// ([`OCEAN_CHOPPINESS`]).
    pub choppiness: f32,
    /// The side of the mip level a reduction writes; unused by the FFT.
    pub mip_size: u32,
    /// The share of last frame's foam that survives this one
    /// ([`FoamClock::advance`](crate::render::environment::water::ocean::foam::FoamClock::advance));
    /// zero restarts it from clear water.
    pub foam_keep: f32,
}

impl OceanFftParams {
    /// The FFT's constants for `sim` at `time` seconds into the world, with
    /// `foam_keep` of last frame's foam surviving.
    pub fn simulate(sim: &OceanSimulation, time: f64, foam_keep: f32) -> Self {
        let c = &sim.cascades;
        let size = c.size();
        let foam = sim.foam.unwrap_or_default();
        Self {
            grid: [
                size,
                size.trailing_zeros(),
                sim.first_layer,
                u32::saturating_from(c.count()),
            ],
            lengths: c.padded_lengths(),
            foam_thresholds: foam.thresholds,
            foam_inv_ramps: foam.inv_ramps,
            time_fraction: time_fraction(time),
            choppiness: OCEAN_CHOPPINESS,
            mip_size: size,
            foam_keep,
        }
    }

    /// The same simulation's reduction into mip `level`.
    #[must_use]
    pub fn reduce(&self, level: u32) -> Self {
        Self {
            mip_size: (self.grid[0] >> level).max(1),
            ..*self
        }
    }
}

/// Mip levels of a cascade `size` texels across, down to one texel.
pub fn ocean_mip_levels(size: u32) -> u32 {
    size.max(1).trailing_zeros() + 1
}

/// Groups along each side of the reduction into a mip `mip_size` across.
pub fn ocean_mip_groups(mip_size: u32) -> u32 {
    mip_size.div_ceil(OCEAN_MIP_GROUP)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::environment::water::ocean::cascades::Cascades;
    use crate::render::environment::water::ocean::clock::OCEAN_REPEAT_SECONDS;
    use crate::render::environment::water::ocean::foam::FoamCalibration;
    use crate::render::environment::water::ocean::{OceanQuality, OceanSea};
    use core::mem::{offset_of, size_of};

    #[test]
    fn ocean_fft_params_layout_matches_shader() {
        assert_eq!(size_of::<OceanFftParams>(), 80);
        assert_eq!(offset_of!(OceanFftParams, grid), 0);
        assert_eq!(offset_of!(OceanFftParams, lengths), 16);
        assert_eq!(offset_of!(OceanFftParams, foam_thresholds), 32);
        assert_eq!(offset_of!(OceanFftParams, foam_inv_ramps), 48);
        assert_eq!(offset_of!(OceanFftParams, time_fraction), 64);
        assert_eq!(offset_of!(OceanFftParams, choppiness), 68);
        assert_eq!(offset_of!(OceanFftParams, mip_size), 72);
        assert_eq!(offset_of!(OceanFftParams, foam_keep), 76);
    }

    #[test]
    fn the_constants_match_the_shader() {
        let src = crate::render::shader::embedded::WATER_FFT;
        assert_eq!(
            crate::render::shader::consts::uint(src, "WATER_FFT_THREADS"),
            OCEAN_FFT_THREADS as usize
        );
        assert_eq!(
            crate::render::shader::consts::uint(src, "WATER_MIP_GROUP"),
            OCEAN_MIP_GROUP as usize
        );
        assert_eq!(2 * OCEAN_FFT_THREADS, OceanQuality::MAX_SIZE);
    }

    #[test]
    fn a_simulation_fills_its_dispatch() {
        let sea = OceanSea {
            wind_speed: 9.0,
            direction: [1.0, 0.0],
            fetch: 1.0e5,
        };
        let cascades = Cascades::new(56.0, OceanQuality::FULL);
        let foam = FoamCalibration::new(&sea, &cascades);
        let mut sim = OceanSimulation {
            foam: Some(foam),
            sea,
            cascades,
            first_layer: 3,
        };
        let p = OceanFftParams::simulate(&sim, 2.0 * OCEAN_REPEAT_SECONDS + 64.0, 0.5);
        assert_eq!(p.grid, [256, 8, 3, 3]);
        assert_eq!(p.lengths, sim.cascades.padded_lengths());
        assert_eq!(p.time_fraction, 0.25);
        assert_eq!(p.choppiness, OCEAN_CHOPPINESS);
        assert_eq!(p.foam_thresholds, foam.thresholds);
        assert_eq!(p.foam_inv_ramps, foam.inv_ramps);
        assert_eq!(p.foam_keep, 0.5);
        let r = p.reduce(3);
        assert_eq!(r.mip_size, 32);
        assert_eq!(r.grid, p.grid);
        assert_eq!(p.reduce(20).mip_size, 1);

        // Without whitecaps no cascade foams.
        sim.foam = None;
        let p = OceanFftParams::simulate(&sim, 0.0, 0.5);
        assert_eq!(p.foam_inv_ramps, [0.0; 4]);
    }

    #[test]
    fn the_mip_chain_runs_down_to_one_texel() {
        assert_eq!(ocean_mip_levels(256), 9);
        assert_eq!(ocean_mip_levels(128), 8);
        assert_eq!(ocean_mip_groups(256), 32);
        assert_eq!(ocean_mip_groups(4), 1);
    }
}
