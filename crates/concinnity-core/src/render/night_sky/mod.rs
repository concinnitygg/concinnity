//! The night sky: stars, the Milky Way, distant galaxies and the moon on the
//! celestial sphere, drawn behind the atmosphere and turned with the sky.
//!
//! The sky's shaders draw it procedurally (`shaders/night_sky.hlsl`) from a
//! few lanes of the atmosphere's constant block: a star per cell of a cube
//! grid laid over the sphere, a band of noise around the galactic plane, a
//! handful of galaxies placed here on the CPU, and the moon's lit face. The
//! air in front of it dims it by the transmittance of each view ray, and the
//! daylight sky drawn over it outshines it, so night falls without a switch.
//!
//! This module holds the CPU halves: the hash every placement draws from,
//! the star grid and magnitude distribution, the galaxies, the Milky Way's
//! envelope, the moon's path and phase, and the handover of the shadowed
//! directional light from the sun to the moon.

mod celestial;
mod frame;
mod galaxies;
mod handover;
mod hash;
mod lanes;
mod milky_way;
mod moon;
mod settings;
mod stars;

pub use celestial::CelestialBasis;
pub use frame::{MoonFrame, NightFrame};
pub use galaxies::{
    ELLIPTICAL_NORM, GALAXY_MAGNITUDE, GALAXY_RADIUS_DEG, Galaxy, MAX_GALAXIES, SPIRAL_NORM,
    galaxy, galaxy_coordinates, galaxy_lanes, galaxy_profile,
};
pub use handover::{MOON_FADE_DEG, moon_share, shadowed_first};
pub use hash::{night_hash, night_unit, seed_lane};
pub use lanes::NightLanes;
pub use milky_way::milky_way_envelope;
pub use moon::{MoonPath, face_light, face_up, lambert_phase};
pub use settings::{
    EARTHSHINE, GALAXY_GAIN, MILKY_WAY_RADIANCE, MoonSettings, NightSkySettings,
    STAR_ZERO_ILLUMINANCE,
};
pub use stars::{
    BRIGHTEST_MAGNITUDE, COUNT_SLOPE, FACE_MARGIN, LIMIT_RANGE, NAKED_EYE_LIMIT, NAKED_EYE_STARS,
    STAR_GRID, STAR_SATURATION, STAR_SIGMA_PX, Star, blackbody_color, cell_solid_angle, cube_face,
    face_direction, footprint, luminance, magnitude_of, star_in_cell, stars_per_steradian,
    temperature_of,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::shader_consts::{float, uint};

    #[test]
    fn the_constants_match_the_shader() {
        let src = crate::render::shaders::NIGHT_SKY;
        assert_eq!(uint(src, "NIGHT_STAR_GRID"), STAR_GRID as usize);
        assert_eq!(float(src, "NIGHT_STAR_SIGMA_PX"), STAR_SIGMA_PX);
        assert_eq!(float(src, "NIGHT_FACE_MARGIN"), FACE_MARGIN);
        assert_eq!(float(src, "NIGHT_STAR_SATURATION"), STAR_SATURATION);
        assert_eq!(float(src, "NIGHT_COUNT_SLOPE"), COUNT_SLOPE);
        assert_eq!(float(src, "NIGHT_BRIGHTEST_MAGNITUDE"), BRIGHTEST_MAGNITUDE);
        let types = crate::render::shaders::ATMOSPHERE_TYPES;
        assert_eq!(uint(types, "NIGHT_MAX_GALAXIES"), MAX_GALAXIES);
    }
}
