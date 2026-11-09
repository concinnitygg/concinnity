// Dirt: crumbs and clods of soil with grooves between them, stones bedded in
// it, darker humus in patches, and polygons of dried cracks.

use super::Surface;
use super::field::{cells, ellipse_inside, fbm, hash, nearest, smoothstep, texel, unit};
use concinnity_engine::components::DirtLook;

// The crumbs between the clods, as a share of a clod's width.
const CRUMB_SCALE: f32 = 0.3;
// How far a clod stands proud at full `clods`, as a share of its width.
const CLOD_RISE: f32 = 0.3;
// Relief of the ground the clods lie on, in meters.
const UNDULATION_M: f32 = 0.008;
// A stone's radius as a share of its lattice cell, and how far it stands
// proud as a share of its radius.
const STONE_RADIUS: f32 = 0.22;
const STONE_RISE: f32 = 0.5;
// The dried crack polygons' width in meters, and the cracks' depth.
const CRACK_CELL_M: f32 = 0.18;
const CRACK_DEPTH_M: f32 = 0.008;

pub(super) fn surface(look: &DirtLook, size: u32) -> Surface {
    let seed = look.seed.wrapping_mul(0x9e37_79b9) ^ 0x7f4a_7c15;
    let clod_m = look.clod_size.max(0.005);
    let clod_cells = cells(clod_m);
    let crumb_cells = cells(clod_m * CRUMB_SCALE);
    let stone_m = look.stone_size.max(0.005);
    let stone_cells = cells(stone_m * 2.2);
    let crack_cells = cells(CRACK_CELL_M);
    let mut surface = Surface::sampled(size, |u, v| {
        let clods = look.clods.clamp(0.0, 1.0);
        let (cu, cv) = warped(u, v, clod_cells, seed ^ 9);
        let clod = nearest(cu, cv, clod_cells, seed ^ 1);
        let (gu, gv) = warped(u, v, crumb_cells, seed ^ 0xa);
        let crumb = nearest(gu, gv, crumb_cells, seed ^ 2);
        let lump = dome(clod.f1, 0.35 + 0.35 * unit(clod.id));
        let grit = dome(crumb.f1, 0.3 + 0.4 * unit(crumb.id));
        let mut h = clods
            * CLOD_RISE
            * (clod_m * lump * (0.5 + 0.5 * unit(hash(clod.id, 5, seed)))
                + clod_m * CRUMB_SCALE * grit);
        h += UNDULATION_M * (fbm(u, v, 4, 4, seed ^ 3) - 0.5);
        h += 0.15 * clod_m * (fbm(u, v, clod_cells, 3, seed ^ 0xb) - 0.5);

        // Humus darkens the soil in patches; the clods' crowns dry lighter
        // and each clod takes a slightly different hue.
        let humus = look.humus.clamp(0.0, 1.0) * smoothstep(0.35, 0.7, fbm(u, v, 3, 4, seed ^ 4));
        let crown = 1.0 + 0.15 * clods * lump;
        let hue = 0.08 * (unit(hash(clod.id, 1, seed)) - 0.5);
        let px = hash(texel(u, size), texel(v, size), seed ^ 5);
        let speck = 1.0 + 0.16 * (unit(px) - 0.5);
        let k = (1.0 - 0.4 * humus) * crown * speck;
        let mut albedo = [
            look.color[0] * k * (1.0 + hue),
            look.color[1] * k,
            look.color[2] * k * (1.0 - hue),
        ];

        // Dried cracks along the polygons' edges.
        let cracks = look.cracks.clamp(0.0, 1.0);
        if cracks > 0.0 {
            let cell = nearest(u, v, crack_cells, seed ^ 6);
            let crack = 1.0 - smoothstep(0.0, 0.03 + 0.05 * cracks, cell.f2 - cell.f1);
            h -= cracks * CRACK_DEPTH_M * crack;
            albedo = albedo.map(|c| c * (1.0 - 0.5 * cracks * crack));
        }

        // A stone where one is bedded.
        let near = nearest(u, v, stone_cells, seed ^ 7);
        if unit(near.id) < look.stones.clamp(0.0, 1.0) * 0.7 {
            let scale = 0.55 + 0.8 * unit(hash(near.id, 2, seed));
            let inside = ellipse_inside(&near, STONE_RADIUS * scale, 0.7);
            if inside > 0.0 {
                let dome = stone_m * scale * STONE_RISE * inside.sqrt();
                let shade = 0.6 + 0.4 * unit(hash(near.id, 3, seed));
                let tone = 0.06 * (unit(hash(near.id, 4, seed)) - 0.5);
                let stone = [
                    look.stone_color[0] * shade * (1.0 + tone),
                    look.stone_color[1] * shade,
                    look.stone_color[2] * shade * (1.0 - tone),
                ];
                let cover = smoothstep(0.0, 0.15, inside);
                albedo = core::array::from_fn(|c| albedo[c] + (stone[c] - albedo[c]) * cover);
                h = h.max(dome * 0.6) + dome * 0.4;
            }
        }
        (h, albedo)
    });
    surface.darken_cavities();
    surface.match_mean(look.color);
    surface
}

// (`u`, `v`) pushed about by a noise a few cells wide, so features scattered
// over a lattice `cells` across lose their straight edges.
fn warped(u: f32, v: f32, cells: u32, seed: u32) -> (f32, f32) {
    let reach = 0.7 / cells.max(1) as f32;
    let broad = (cells / 3).max(1);
    (
        u + reach * (fbm(u, v, broad, 2, seed) - 0.5),
        v + reach * (fbm(u, v, broad, 2, seed ^ 1) - 0.5),
    )
}

// A rounded lump over a scattered point, `radius` wide in cell widths: 1 at
// its crown and 0 from `radius` out.
fn dome(distance: f32, radius: f32) -> f32 {
    let t = (distance / radius).min(1.0);
    (1.0 - t * t).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cracks_deepen_the_relief() {
        let cracked = DirtLook {
            cracks: 1.0,
            ..DirtLook::default()
        };
        let low = |s: &Surface| s.height_range().0;
        assert!(low(&surface(&cracked, 64)) < low(&surface(&DirtLook::default(), 64)));
    }

    #[test]
    fn stones_raise_the_relief() {
        let bare = DirtLook {
            stones: 0.0,
            ..DirtLook::default()
        };
        let stony = DirtLook {
            stones: 1.0,
            ..DirtLook::default()
        };
        let high = |s: &Surface| s.height_range().1;
        assert!(high(&surface(&stony, 128)) > high(&surface(&bare, 128)));
    }
}
