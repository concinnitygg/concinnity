//! Vector icons built from the HUD's own rounded rectangles. A glyph is a few
//! parts laid out on a unit square and scaled into whatever cell it is drawn
//! in, so it is resolution-independent and stays sharp at any display scale.

use concinnity_engine::ecs::World;
use concinnity_engine::ecs::asset_id::AssetId;

use super::widget;

/// The most parts a glyph has: the sprites one icon reserves.
pub(crate) const MAX_PARTS: usize = 4;

/// One rounded rectangle of a glyph, in units of the cell's side.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Part {
    /// `[x, y, w, h]` from the cell's top-left corner.
    pub rect: [f32; 4],
    /// Corner radius as a share of the part's shorter side, up to 0.5.
    pub round: f32,
    /// Outline width as a share of the cell's side; 0 fills the part.
    pub ring: f32,
}

/// An icon: its parts, back to front.
pub(crate) type Glyph = &'static [Part];

// A filled part on a 16-unit grid.
const fn fill(x: f32, y: f32, w: f32, h: f32, round: f32) -> Part {
    Part {
        rect: [x / 16.0, y / 16.0, w / 16.0, h / 16.0],
        round,
        ring: 0.0,
    }
}

// An outlined part on a 16-unit grid, its stroke `width` units wide.
const fn outline(x: f32, y: f32, w: f32, h: f32, round: f32, width: f32) -> Part {
    Part {
        ring: width / 16.0,
        ..fill(x, y, w, h, round)
    }
}

/// An arrow rising from the ground.
pub(crate) const RAISE: Glyph = &[
    fill(1.5, 13.0, 13.0, 2.0, 0.5),
    fill(7.0, 5.5, 2.0, 6.5, 0.5),
    fill(4.5, 4.5, 7.0, 2.0, 0.5),
    fill(6.5, 2.0, 3.0, 3.0, 0.5),
];

/// An arrow sinking under the ground.
pub(crate) const LOWER: Glyph = &[
    fill(1.5, 1.0, 13.0, 2.0, 0.5),
    fill(7.0, 4.0, 2.0, 6.5, 0.5),
    fill(4.5, 9.5, 7.0, 2.0, 0.5),
    fill(6.5, 11.0, 3.0, 3.0, 0.5),
];

/// A soft dome on the ground.
pub(crate) const SMOOTH: Glyph = &[
    fill(1.0, 13.0, 14.0, 2.0, 0.5),
    fill(2.5, 6.0, 11.0, 8.0, 0.5),
];

/// A plateau pressed flat from above.
pub(crate) const FLATTEN: Glyph = &[
    fill(1.5, 10.0, 13.0, 5.0, 0.15),
    fill(7.0, 1.0, 2.0, 5.0, 0.5),
    fill(4.5, 5.0, 7.0, 2.0, 0.5),
];

/// Blades of grass.
pub(crate) const GRASS: Glyph = &[
    fill(1.0, 13.0, 14.0, 2.0, 0.5),
    fill(3.0, 6.0, 2.0, 8.0, 0.5),
    fill(7.0, 2.5, 2.0, 11.5, 0.5),
    fill(11.0, 4.5, 2.0, 9.5, 0.5),
];

/// A paint brush.
pub(crate) const COVER: Glyph = &[
    fill(7.0, 1.0, 2.0, 7.0, 0.5),
    fill(5.5, 7.0, 5.0, 2.5, 0.2),
    fill(5.0, 9.0, 6.0, 6.0, 0.4),
];

/// A drop of water.
pub(crate) const WATER: Glyph = &[
    fill(3.5, 6.5, 9.0, 8.5, 0.5),
    fill(5.5, 3.5, 5.0, 5.0, 0.5),
    fill(7.0, 1.0, 2.0, 4.0, 0.5),
];

/// An eraser, half worn.
pub(crate) const ERASE: Glyph = &[
    outline(1.5, 4.5, 13.0, 7.0, 0.25, 1.5),
    fill(1.5, 4.5, 6.0, 7.0, 0.25),
    fill(1.5, 13.5, 13.0, 1.5, 0.5),
];

/// Part `part`'s rectangle and corner radius in the square `cell` (`[x, y,
/// side]` in window pixels).
pub(crate) fn part_rect(cell: [f32; 3], part: &Part) -> ([f32; 4], f32) {
    let [x, y, side] = cell;
    let r = part.rect;
    let rect = [x + r[0] * side, y + r[1] * side, r[2] * side, r[3] * side];
    let radius = part.round.clamp(0.0, 0.5) * rect[2].min(rect[3]);
    (rect, radius)
}

/// Draw `glyph` in the square `cell` with `color`, part `i` on sprite
/// `ids[i]`; the sprites the glyph leaves over are hidden.
pub(crate) fn place(
    world: &mut World,
    ids: &[AssetId; MAX_PARTS],
    cell: [f32; 3],
    glyph: Glyph,
    color: [f32; 4],
) {
    for (i, &id) in ids.iter().enumerate() {
        let Some(part) = glyph.get(i) else {
            widget::set_sprite_visible(world, id, false);
            continue;
        };
        let (rect, radius) = part_rect(cell, part);
        if part.ring > 0.0 {
            widget::place_rounded(world, id, rect, [0.0; 4], radius, true);
            if let Some(s) = widget::sprite_mut(world, id) {
                s.border_color = color;
                s.border_width = part.ring * cell[2];
            }
        } else {
            widget::place_rounded(world, id, rect, color, radius, true);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Glyph; 8] = [RAISE, LOWER, SMOOTH, FLATTEN, GRASS, COVER, WATER, ERASE];

    #[test]
    fn every_glyph_fits_its_cell_and_its_sprites() {
        for glyph in ALL {
            assert!(!glyph.is_empty() && glyph.len() <= MAX_PARTS);
            for part in glyph {
                let [x, y, w, h] = part.rect;
                assert!(x >= 0.0 && y >= 0.0 && w > 0.0 && h > 0.0);
                assert!(x + w <= 1.0 && y + h <= 1.0, "{part:?}");
                assert!((0.0..=0.5).contains(&part.round));
            }
        }
    }

    // A glyph scales with its cell: twice the cell, twice every rect and
    // radius, so it reads the same at any size.
    #[test]
    fn parts_scale_with_the_cell() {
        for part in RAISE.iter().chain(ERASE) {
            let (small, r_small) = part_rect([10.0, 20.0, 16.0], part);
            let (big, r_big) = part_rect([10.0, 20.0, 32.0], part);
            assert_eq!((big[2], big[3]), (small[2] * 2.0, small[3] * 2.0));
            assert_eq!(big[0] - 10.0, (small[0] - 10.0) * 2.0);
            assert_eq!(r_big, r_small * 2.0);
        }
    }

    #[test]
    fn a_full_round_makes_a_pill() {
        let (rect, radius) = part_rect([0.0, 0.0, 16.0], &RAISE[0]);
        assert_eq!(radius, rect[3] * 0.5);
    }
}
