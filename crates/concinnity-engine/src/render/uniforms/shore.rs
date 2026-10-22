//! What a planet sea's shader reads its shore waves from. Matches
//! `WaterShore` in `shaders/water.hlsl`.

use crate::planet::{CONTINENT_OCTAVES, octave_weight_sum};
use crate::render::environment::water::sea::SeaShoreMotion;
use crate::render::environment::water::sea::shore::{
    PHASE_NOISE_RADIANS, SHORE_NODES, SHORE_OCTAVES, SHORE_TRAINS, SURF_LACE_PERIOD,
    SWELL_VARIATION, ShoreWaves,
};
use concinnity_core::math::narrow_f32;

/// A planet sea's shore waves: where the coast lies (the continent field the
/// planet's tiles are built from and the coast's profile), the trains and
/// their phase table, and this frame's frame-to-planet rotation and phases.
/// Zero for a sea without coasts, which `continent_octaves` 0 marks. 1584
/// bytes.
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct WaterShoreGpu {
    /// The rotation from the simulated frame into the planet's axes, one row
    /// per axis, each `w` the anchor's direction from the center along it.
    pub planet_rows: [[f32; 4]; 3],
    /// Each continent octave's hash salt, four to a row.
    pub continent_salts: [[u32; 4]; 2],
    /// Lattice cells of the broadest continent octave per unit of direction.
    pub continent_frequency: f32,
    /// Meters inland per unit of the continent octaves' weighted sum.
    pub continent_scale: f32,
    /// Meters inland at a weighted sum of 0, negated.
    pub continent_offset: f32,
    /// Continent octaves summed; 0 for a sea without coasts.
    pub continent_octaves: u32,
    /// The phase noise's octave salts; `zw` unused.
    pub noise_salts: [u32; 4],
    /// `[lattice cells per unit of direction, radians either way, 1 / planet
    /// radius, 1 / sea radius]`.
    pub noise: [f32; 4],
    /// `[sea level, waterline, ocean depth, shelf]`, in meters
    /// ([`CoastProfile`](crate::render::environment::water::sea::shore::CoastProfile)).
    pub coast: [f32; 4],
    /// `[beach width, beach rise, table reach, peak wavelength]`, in meters.
    pub beach: [f32; 4],
    /// `[quiet inland, _, _, _]`: seaward of
    /// [`ShoreWaves::quiet_inland`] the shader skips the finer continent
    /// octaves.
    pub bounds: [f32; 4],
    /// `[lattice cells per unit of direction, share either way, _, _]`: how
    /// the swell's height varies along the shore.
    pub swell: [f32; 4],
    /// `[x, y, z, period]`: the anchor's place on the sea in the planet's
    /// axes, wrapped into one period of meters, which the surf's foam is laid
    /// out from.
    pub lace: [f32; 4],
    /// Each train's angular frequency, in radians per second; `w` unused.
    pub omegas: [f32; 4],
    /// Each train's deep-water height, in meters; `w` unused.
    pub amplitudes: [f32; 4],
    /// Each train's phase at the waterline this frame; `w` unused.
    pub phases: [f32; 4],
    /// The wavelength each cascade fades out at in shallow water; 0 keeps it.
    pub cascade_wavelengths: [f32; 4],
    /// Per node, the trains' phases then their wavenumbers, `w` unused.
    pub table: [[f32; 4]; 2 * SHORE_NODES],
    /// Per noise octave the shore reads (the continents', the phase noise's,
    /// then the swell's), the lattice cell the anchor's direction lies in;
    /// `w` unused.
    pub lattice_cells: [[i32; 4]; SHORE_OCTAVES],
    /// Where in that cell the anchor's direction lies; `w` unused.
    pub lattice_places: [[f32; 4]; SHORE_OCTAVES],
}

impl Default for WaterShoreGpu {
    fn default() -> Self {
        bytemuck::Zeroable::zeroed()
    }
}

impl WaterShoreGpu {
    /// The block for `shore`, before any frame moves it.
    pub fn from_shore(shore: &ShoreWaves) -> Self {
        let c = shore.continents();
        let r = shore.planet_radius();
        let mut salts = [[0; 4]; 2];
        for octave in 0..CONTINENT_OCTAVES {
            salts[octave as usize / 4][octave as usize % 4] = c.octave_salt(octave);
        }
        let per_sum = c.coast_scale() / octave_weight_sum(CONTINENT_OCTAVES);
        let coast = shore.coast();
        let lane =
            |v: [f64; SHORE_TRAINS]| [narrow_f32(v[0]), narrow_f32(v[1]), narrow_f32(v[2]), 0.0];
        let [s0, s1, s2] = shore.noise_salts();
        let mut table = [[0.0; 4]; 2 * SHORE_NODES];
        for j in 0..SHORE_NODES {
            table[2 * j] = lane(shore.table().phases(j));
            table[2 * j + 1] = lane(shore.table().wavenumbers(j));
        }
        Self {
            planet_rows: [[0.0; 4]; 3],
            continent_salts: salts,
            continent_frequency: narrow_f32(c.base_frequency(r)),
            continent_scale: narrow_f32(per_sum),
            continent_offset: narrow_f32(c.threshold() * c.coast_scale()),
            continent_octaves: CONTINENT_OCTAVES,
            noise_salts: [s0, s1, s2, 0],
            noise: [
                narrow_f32(shore.noise_frequency()),
                narrow_f32(PHASE_NOISE_RADIANS),
                narrow_f32(1.0 / r),
                narrow_f32(1.0 / shore.radius()),
            ],
            coast: [
                narrow_f32(coast.sea_level),
                narrow_f32(coast.waterline),
                narrow_f32(coast.ocean_depth),
                narrow_f32(coast.shelf),
            ],
            beach: [
                narrow_f32(coast.beach_width),
                narrow_f32(coast.beach_rise),
                narrow_f32(shore.table().reach()),
                narrow_f32(shore.peak_wavelength()),
            ],
            bounds: [narrow_f32(shore.quiet_inland()), 0.0, 0.0, 0.0],
            swell: [
                narrow_f32(shore.swell_frequency()),
                narrow_f32(SWELL_VARIATION),
                0.0,
                0.0,
            ],
            lace: [0.0, 0.0, 0.0, narrow_f32(SURF_LACE_PERIOD)],
            omegas: lane(shore.trains().map(|t| t.omega)),
            amplitudes: lane(shore.trains().map(|t| t.amplitude)),
            phases: [0.0; 4],
            cascade_wavelengths: shore.cascade_wavelengths().map(narrow_f32),
            table,
            lattice_cells: [[0; 4]; SHORE_OCTAVES],
            lattice_places: [[0.0; 4]; SHORE_OCTAVES],
        }
    }

    /// The block moved to the instant and frame `motion` describes.
    pub fn advance(&mut self, motion: &SeaShoreMotion) {
        self.planet_rows = motion.rows;
        let [x, y, z] = motion.lace_origin;
        self.lace = [x, y, z, self.lace[3]];
        let [a, b, c] = motion.phases;
        self.phases = [a, b, c, 0.0];
        for (i, place) in motion.lattice.iter().enumerate() {
            let [x, y, z] = place.cell;
            self.lattice_cells[i] = [x, y, z, 0];
            let [x, y, z] = place.within.map(narrow_f32);
            self.lattice_places[i] = [x, y, z, 0.0];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};

    // Mirrors `WaterShore` in shaders/water.hlsl.
    #[test]
    fn water_shore_matches_hlsl() {
        assert_eq!(size_of::<WaterShoreGpu>(), 1584);
        assert_eq!(offset_of!(WaterShoreGpu, planet_rows), 0);
        assert_eq!(offset_of!(WaterShoreGpu, continent_salts), 48);
        assert_eq!(offset_of!(WaterShoreGpu, continent_frequency), 80);
        assert_eq!(offset_of!(WaterShoreGpu, continent_scale), 84);
        assert_eq!(offset_of!(WaterShoreGpu, continent_offset), 88);
        assert_eq!(offset_of!(WaterShoreGpu, continent_octaves), 92);
        assert_eq!(offset_of!(WaterShoreGpu, noise_salts), 96);
        assert_eq!(offset_of!(WaterShoreGpu, noise), 112);
        assert_eq!(offset_of!(WaterShoreGpu, coast), 128);
        assert_eq!(offset_of!(WaterShoreGpu, beach), 144);
        assert_eq!(offset_of!(WaterShoreGpu, bounds), 160);
        assert_eq!(offset_of!(WaterShoreGpu, swell), 176);
        assert_eq!(offset_of!(WaterShoreGpu, lace), 192);
        assert_eq!(offset_of!(WaterShoreGpu, omegas), 208);
        assert_eq!(offset_of!(WaterShoreGpu, amplitudes), 224);
        assert_eq!(offset_of!(WaterShoreGpu, phases), 240);
        assert_eq!(offset_of!(WaterShoreGpu, cascade_wavelengths), 256);
        assert_eq!(offset_of!(WaterShoreGpu, table), 272);
        assert_eq!(offset_of!(WaterShoreGpu, lattice_cells), 1296);
        assert_eq!(offset_of!(WaterShoreGpu, lattice_places), 1440);
    }
}
