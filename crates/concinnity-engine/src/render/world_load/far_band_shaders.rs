//! Which world Shaders a far band draws under, so the backend compiles the
//! grass variants for only those Shaders.

use alloc::vec;
use alloc::vec::Vec;

use crate::gfx::render_types::FarFieldTint;
use crate::render::shader::world_pipelines::GrassVariants;

/// Per `ShaderHandle` value (`shader_count` of them, the default at 0): the
/// grass variants the Shader needs, any when one of `draws` (each its shader
/// bucket and far band) has a band under it. A planet whose ground shades from
/// the canopy block streams its tiles under the default. On a `planet` world
/// each variant comes with and without the planet's own sky.
pub fn far_band_shaders(
    shader_count: usize,
    draws: impl IntoIterator<Item = (u32, FarFieldTint)>,
    planet_far_field: bool,
    planet: bool,
) -> Vec<GrassVariants> {
    let grass = if planet {
        GrassVariants::PlanetSky
    } else {
        GrassVariants::Grass
    };
    let mut far = vec![GrassVariants::None; shader_count];
    if planet_far_field && let Some(default) = far.first_mut() {
        *default = grass;
    }
    for (bucket, band) in draws {
        if band.shades_ground()
            && let Some(flag) = far.get_mut(bucket as usize)
        {
            *flag = grass;
        }
    }
    far
}

#[cfg(test)]
mod tests {
    use super::*;
    use GrassVariants::{Grass, None as NoGrass, PlanetSky};

    fn draw(bucket: u32, far: bool) -> (u32, FarFieldTint) {
        let band = if far {
            FarFieldTint::of(1 << 8)
        } else {
            FarFieldTint::NONE
        };
        (bucket, band)
    }

    #[test]
    fn only_shaders_under_a_far_band_are_flagged() {
        let draws = [
            draw(0, false),
            draw(1, true),
            draw(2, false),
            draw(2, false),
        ];
        assert_eq!(
            far_band_shaders(3, draws, false, false),
            [NoGrass, Grass, NoGrass]
        );
        assert_eq!(far_band_shaders(3, [], false, false), [NoGrass; 3]);
    }

    #[test]
    fn a_grassy_planet_flags_the_default_with_its_sky() {
        assert_eq!(far_band_shaders(2, [], true, true), [PlanetSky, NoGrass]);
        assert_eq!(
            far_band_shaders(2, [draw(1, true)], false, true),
            [NoGrass, PlanetSky]
        );
        assert!(far_band_shaders(0, [], true, true).is_empty());
    }

    #[test]
    fn a_bucket_past_the_shaders_is_ignored() {
        assert_eq!(
            far_band_shaders(1, [draw(4, true)], false, false),
            [NoGrass]
        );
    }
}
