// A vector icon: a few rounded rectangles on a unit square, scaled into the
// cell a widget draws it in.

use alloc::vec::Vec;

/// One rounded rectangle of an [Icon](#icon), in units of the icon's side.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct IconPart {
    /// `[x, y, width, height]` from the icon's top-left corner, each from 0
    /// to 1.
    pub rect: [f32; 4],
    /// Corner rounding as a share of the part's shorter side, up to 0.5 for a
    /// pill or a circle.
    pub round: f32,
    /// Outline width as a share of the icon's side; 0 fills the part.
    pub ring: f32,
}

impl IconPart {
    /// A filled part at `[x, y, width, height]` on a 16-unit grid.
    pub const fn fill(rect: [f32; 4], round: f32) -> Self {
        Self {
            rect: [
                rect[0] / 16.0,
                rect[1] / 16.0,
                rect[2] / 16.0,
                rect[3] / 16.0,
            ],
            round,
            ring: 0.0,
        }
    }

    /// An outlined part at `[x, y, width, height]` on a 16-unit grid, its
    /// stroke `width` units wide.
    pub const fn outline(rect: [f32; 4], round: f32, width: f32) -> Self {
        Self {
            ring: width / 16.0,
            ..Self::fill(rect, round)
        }
    }

    /// The part's rectangle and corner radius in the square `cell`, `[x, y,
    /// side]`.
    pub fn placed(&self, cell: [f32; 3]) -> ([f32; 4], f32) {
        let [x, y, side] = cell;
        let r = self.rect;
        let rect = [x + r[0] * side, y + r[1] * side, r[2] * side, r[3] * side];
        let radius = self.round.clamp(0.0, 0.5) * rect[2].min(rect[3]);
        (rect, radius)
    }
}

/// A small vector icon drawn in one color: rounded rectangles, filled or
/// outlined, laid out on a unit square and scaled to whatever cell shows it,
/// so it stays sharp at any size. Parts draw back to front.
///
/// ```rust
/// # use concinnity_engine::components::{Icon, IconPart};
/// // A padlock: an outlined shackle behind a filled body.
/// let lock = Icon {
///     parts: vec![
///         IconPart::outline([4.5, 2.0, 7.0, 8.0], 0.5, 1.5),
///         IconPart::fill([2.5, 7.0, 11.0, 8.0], 0.25),
///     ],
/// };
/// assert!(!lock.is_empty());
/// ```
#[derive(
    Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields,
)]
#[serde(default)]
pub struct Icon {
    /// The icon's parts, back to front.
    pub parts: Vec<IconPart>,
}

impl Icon {
    /// Whether the icon draws nothing.
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_part_scales_with_its_cell_and_rounds_to_its_shorter_side() {
        let part = IconPart::fill([4.0, 0.0, 8.0, 4.0], 0.5);
        let (rect, radius) = part.placed([10.0, 20.0, 32.0]);
        assert_eq!(rect, [18.0, 20.0, 16.0, 8.0]);
        assert_eq!(radius, 4.0);
        let ring = IconPart::outline([0.0, 0.0, 16.0, 16.0], 2.0, 2.0);
        assert_eq!(ring.ring, 0.125);
        assert_eq!(ring.placed([0.0, 0.0, 16.0]).1, 8.0, "round caps at 0.5");
    }
}
