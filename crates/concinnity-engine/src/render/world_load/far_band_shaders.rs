//! Which world Shaders a far band draws under, so the backend compiles the
//! grass canopy into only those buckets.

use alloc::vec;
use alloc::vec::Vec;

use crate::gfx::render_types::FarFieldTint;

/// Per `ShaderHandle` value (`shader_count` of them, the default at 0): whether
/// one of `draws` (each its shader bucket and far band) has a band under that
/// Shader. A planet whose ground shades from the canopy block streams its
/// tiles under the default.
pub fn far_band_shaders(
    shader_count: usize,
    draws: impl IntoIterator<Item = (u32, FarFieldTint)>,
    planet_far_field: bool,
) -> Vec<bool> {
    let mut far = vec![false; shader_count];
    if planet_far_field && let Some(default) = far.first_mut() {
        *default = true;
    }
    for (bucket, band) in draws {
        if band.shades_ground()
            && let Some(flag) = far.get_mut(bucket as usize)
        {
            *flag = true;
        }
    }
    far
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(far_band_shaders(3, draws, false), [false, true, false]);
        assert_eq!(far_band_shaders(3, [], false), [false; 3]);
    }

    #[test]
    fn a_grassy_planet_flags_the_default() {
        assert_eq!(far_band_shaders(2, [], true), [true, false]);
        assert!(far_band_shaders(0, [], true).is_empty());
    }

    #[test]
    fn a_bucket_past_the_shaders_is_ignored() {
        assert_eq!(far_band_shaders(1, [draw(4, true)], false), [false]);
    }
}
