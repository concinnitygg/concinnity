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
#[cfg(test)]
mod milky_way;
mod moon;
mod settings;
mod stars;

pub(crate) use frame::NightFrame;
pub(crate) use galaxies::MAX_GALAXIES;
pub(crate) use handover::{moon_share, shadowed_first};
pub(crate) use lanes::NightLanes;
pub(crate) use settings::NightSkySettings;
pub(crate) use stars::LIMIT_RANGE;

#[cfg(test)]
mod tests {
    use super::stars::{
        BRIGHTEST_MAGNITUDE, COUNT_SLOPE, FACE_MARGIN, STAR_GRID, STAR_SATURATION, STAR_SIGMA_PX,
    };
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
