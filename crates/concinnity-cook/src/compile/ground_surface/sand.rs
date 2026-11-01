// Sand: wind ripples with a gentle windward slope and a steep lee, wandering
// and fading in patches, over loose grains, dark mineral grains settled in the
// troughs, and scattered shell fragments.

use super::field::{cells, ellipse_inside, fbm, hash, nearest, smoothstep, texel, unit};
use super::{Surface, TILE_METERS};
use concinnity_engine::components::SandLook;

// Where along a ripple's wavelength its crest lies, from the windward trough.
const CREST: f32 = 0.72;
// How far the crests wander, in wavelengths.
const WANDER: f32 = 1.1;
// Relief of the dunelets the ripples ride on and of the loose grains, in
// meters at full `grain`.
const UNDULATION_M: f32 = 0.004;
const GRAIN_M: f32 = 0.0007;
// Shell fragments: lattice cell width in meters, their radius as a share of
// a cell, and how far they stand proud.
const SHELL_CELL_M: f32 = 0.035;
const SHELL_RADIUS: f32 = 0.16;
const SHELL_RISE_M: f32 = 0.0012;

pub(super) fn surface(look: &SandLook, size: u32) -> Surface {
    let seed = look.seed.wrapping_mul(0x2545_f491);
    let (m, n) = ripple_wave(look.ripple_spacing, look.ripple_direction_deg);
    let shell_cells = cells(SHELL_CELL_M);
    let mut surface = Surface::sampled(size, |u, v| {
        let warp = (fbm(u, v, 3, 3, seed ^ 1) - 0.5) * 2.0 * WANDER
            + (fbm(u, v, 9, 2, seed ^ 2) - 0.5) * 0.35;
        let phase = (m * u + n * v + warp).rem_euclid(1.0);
        let profile = ripple_profile(phase);
        let strength = 0.3 + 0.7 * smoothstep(0.3, 0.7, fbm(u, v, 2, 2, seed ^ 3));
        let mut h = look.ripple_height * strength * (profile - 0.5);
        h += UNDULATION_M * (fbm(u, v, 4, 3, seed ^ 4) - 0.5);
        let px = hash(texel(u, size), texel(v, size), seed ^ 5);
        let grain = look.grain.clamp(0.0, 1.0);
        h += GRAIN_M * grain * (fbm(u, v, 160, 2, seed ^ 6) - 0.5 + 0.5 * (unit(px) - 0.5));

        // The sand's mottling, darker in the troughs where heavy grains settle,
        // and speckled with dark and clear grains.
        let mottle = 1.0 + 0.12 * (fbm(u, v, 3, 3, seed ^ 7) - 0.5);
        let dark = look.dark_grains.clamp(0.0, 1.0);
        let trough = 1.0 - 0.45 * dark * strength * (1.0 - profile).powi(2);
        let roll = unit(hash(px, 1, seed));
        let speck = if roll < 0.18 * dark {
            0.55
        } else if roll > 1.0 - 0.15 * grain {
            1.25
        } else {
            1.0 + 0.1 * grain * (unit(hash(px, 2, seed)) - 0.5)
        };
        let k = mottle * trough * speck;
        let mut albedo = look.color.map(|c| c * k);

        // A shell fragment where one lies.
        let near = nearest(u, v, shell_cells, seed ^ 8);
        if unit(near.id) < look.shells.clamp(0.0, 1.0) * 0.6 {
            let size = 0.6 + 0.8 * unit(hash(near.id, 3, seed));
            let inside = ellipse_inside(&near, SHELL_RADIUS * size, 0.55);
            if inside > 0.0 {
                let cover = smoothstep(0.0, 0.25, inside);
                let shade = 0.8 + 0.3 * unit(hash(near.id, 4, seed));
                albedo = core::array::from_fn(|c| {
                    albedo[c] + (look.shell_color[c] * shade - albedo[c]) * cover
                });
                h += SHELL_RISE_M * size * inside.sqrt();
            }
        }
        (h, albedo)
    });
    surface.darken_cavities();
    surface.match_mean(look.color);
    surface
}

// The ripples' wave as whole cycles across the tile along u and v, nearest the
// spacing and direction asked for, so the crests meet themselves at its edges.
fn ripple_wave(spacing: f32, direction_deg: f32) -> (f32, f32) {
    let cycles = TILE_METERS / spacing.max(0.01);
    let (s, c) = direction_deg.to_radians().sin_cos();
    let (m, n) = ((c * cycles).round(), (s * cycles).round());
    if m == 0.0 && n == 0.0 {
        (1.0, 0.0)
    } else {
        (m, n)
    }
}

// A ripple's height in [0, 1] at `phase` along its wavelength: a long rise to
// the crest and a short drop down the lee.
fn ripple_profile(phase: f32) -> f32 {
    let t = if phase < CREST {
        phase / CREST
    } else {
        (1.0 - phase) / (1.0 - CREST)
    };
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ripples_take_whole_cycles_across_the_tile() {
        assert_eq!(ripple_wave(0.1, 0.0), (10.0, 0.0));
        let (m, n) = ripple_wave(0.1, 90.0);
        assert_eq!((m.abs(), n), (0.0, 10.0));
        assert_eq!(ripple_wave(50.0, 0.0), (1.0, 0.0));
    }

    #[test]
    fn a_ripple_peaks_at_its_crest_and_bottoms_at_its_trough() {
        assert_eq!(ripple_profile(0.0), 0.0);
        assert_eq!(ripple_profile(CREST), 1.0);
        assert!(ripple_profile(0.99) < 0.05);
        // The windward rise is the longer side.
        assert!(ripple_profile(CREST * 0.5) > ripple_profile(CREST + (1.0 - CREST) * 0.5) - 1e-6);
    }

    #[test]
    fn unrippled_sand_is_far_flatter() {
        let flat = SandLook {
            ripple_height: 0.0,
            ..SandLook::default()
        };
        let range = |s: &Surface| {
            let (lo, hi) = s.height_range();
            hi - lo
        };
        assert!(range(&surface(&flat, 64)) < range(&surface(&SandLook::default(), 64)));
    }

    #[test]
    fn seeds_draw_different_sand() {
        let a = surface(&SandLook::default(), 64);
        let b = surface(
            &SandLook {
                seed: 7,
                ..SandLook::default()
            },
            64,
        );
        assert_ne!(a.height, b.height);
    }
}
