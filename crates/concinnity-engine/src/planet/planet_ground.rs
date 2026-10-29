// The ground sculpted into one square of a planet: as a world authors it, and
// as a loaded world keeps it until the planet reads it.

use alloc::vec::Vec;

use super::edits::{EditPage, PAGE_SAMPLES, PAGE_SIDE};
use super::shape::CubeFace;
use super::tile::{MAX_TILE_LEVEL, TileId};
use crate::ecs::PayloadLocator;

/// The ground sculpted into one square of a [`Planet`](#planet): how far it
/// is raised or lowered, and how much of the generated hills it keeps, at 33
/// by 33 points spread evenly across the square.
///
/// The square is `tile`, `[face, level, column, row]` of the planet's
/// squares: each `face` (0 to 5: `+X`, `-X`, `+Y`, `-Y`, `+Z`, `-Z`) is one
/// side of the cube the sphere is seen through, `level` 0 is the whole face
/// and each level up to 20 halves the side, and `column` and `row` count the
/// squares across the face at that level. The points run along the column
/// first, row by row, the first and last of each row on the square's edges,
/// so a square's edge points are its neighbor's too.
///
/// `height` raises (or, below 0, lowers) the ground at each point, in meters.
/// `detail`, in `[0, 1]`, scales the generated hills there: 1 keeps them
/// whole, 0 leaves only the smooth continents and coasts. An empty `detail`
/// keeps the hills whole everywhere. Between points the ground blends
/// smoothly.
///
/// A place on the planet takes its ground from the smallest square over it,
/// never from several at once, so finer sculpting in a small area sits on a
/// larger square sculpted around it. A world holds at most one square per
/// tile.
///
/// ```rust
/// # use concinnity_engine::components::cook::PlanetGround;
/// PlanetGround {
///     tile: [2, 8, 128, 127],
///     height: vec![1.5; 33 * 33],
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct PlanetGroundArgs {
    /// The square: `[face, level, column, row]`.
    pub tile: [u32; 4],
    /// The ground raised at each point, in meters; 33 x 33 values.
    pub height: Vec<f32>,
    /// How much of the generated hills each point keeps, in `[0, 1]`; 33 x 33
    /// values, or none to keep them whole.
    pub detail: Vec<f32>,
}

/// The runtime `PlanetGround`: the square it covers and its compiled page's
/// place in the blob. A loaded world reads every one into its planet and
/// keeps none.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PlanetGround {
    /// The square: `[face, level, column, row]`.
    pub tile: [u32; 4],
    /// The compiled page's place in the blob, injected at load.
    pub locator: Option<PayloadLocator>,
}

/// Why a `PlanetGround`'s args do not make a page.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum PlanetGroundError {
    /// The tile names no square of the planet.
    #[error(
        "`tile` {0:?} names no square: face 0 to 5, level 0 to {MAX_TILE_LEVEL}, and column \
         and row under 2 to the power of the level"
    )]
    Tile([u32; 4]),
    /// `height` does not hold a value per point.
    #[error("`height` holds {0} values; a square holds {PAGE_SIDE} x {PAGE_SIDE} = {PAGE_SAMPLES}")]
    Height(usize),
    /// `detail` holds values, but not one per point.
    #[error(
        "`detail` holds {0} values; a square holds {PAGE_SIDE} x {PAGE_SIDE} = {PAGE_SAMPLES}, \
         or none to keep the hills whole"
    )]
    Detail(usize),
    /// A value is not a finite number, or a detail lies outside `[0, 1]`.
    #[error("`{0}` value {1} is out of range: heights must be finite, details in [0, 1]")]
    Value(&'static str, f32),
}

impl PlanetGroundArgs {
    /// The square `tile` names, if it names one.
    pub fn tile_id(&self) -> Option<TileId> {
        tile_id(self.tile)
    }

    /// The square and its page.
    ///
    /// # Errors
    ///
    /// Returns why the args do not make a page.
    pub fn page(&self) -> Result<(TileId, EditPage), PlanetGroundError> {
        let tile = self.tile_id().ok_or(PlanetGroundError::Tile(self.tile))?;
        if self.height.len() != PAGE_SAMPLES {
            return Err(PlanetGroundError::Height(self.height.len()));
        }
        if !self.detail.is_empty() && self.detail.len() != PAGE_SAMPLES {
            return Err(PlanetGroundError::Detail(self.detail.len()));
        }
        if let Some(&v) = self.height.iter().find(|v| !v.is_finite()) {
            return Err(PlanetGroundError::Value("height", v));
        }
        if let Some(&v) = self.detail.iter().find(|v| !(0.0..=1.0).contains(*v)) {
            return Err(PlanetGroundError::Value("detail", v));
        }
        let e: Vec<f64> = self.height.iter().map(|&v| f64::from(v)).collect();
        let w: Vec<f64> = if self.detail.is_empty() {
            alloc::vec![1.0; PAGE_SAMPLES]
        } else {
            self.detail.iter().map(|&v| f64::from(v)).collect()
        };
        let page = EditPage::from_samples(&e, &w).ok_or(PlanetGroundError::Height(e.len()))?;
        Ok((tile, page))
    }
}

impl PlanetGround {
    /// The runtime form of `args`: its square, its page compiled apart.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "the cook bakes every type whose authored form differs from it by value"
    )]
    pub fn bake(args: PlanetGroundArgs) -> Self {
        Self {
            tile: args.tile,
            locator: None,
        }
    }

    /// The square `tile` names, if it names one.
    pub fn tile_id(&self) -> Option<TileId> {
        tile_id(self.tile)
    }
}

// The square `[face, level, column, row]` names, if it names one.
fn tile_id([face, level, x, y]: [u32; 4]) -> Option<TileId> {
    let tile = TileId {
        face: CubeFace(u8::try_from(face).ok()?),
        level: u8::try_from(level).ok()?,
        x,
        y,
    };
    tile.is_valid().then_some(tile)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> PlanetGroundArgs {
        PlanetGroundArgs {
            tile: [2, 8, 128, 127],
            height: alloc::vec![2.0; PAGE_SAMPLES],
            detail: Vec::new(),
        }
    }

    #[test]
    fn args_make_the_page_they_describe() {
        let (tile, page) = args().page().unwrap();
        assert_eq!(
            (tile.face, tile.level, tile.x, tile.y),
            (CubeFace(2), 8, 128, 127)
        );
        assert_eq!(page.e_at(4, 30), 2.0);
        assert!(!page.has_w());
        let detailed = PlanetGroundArgs {
            detail: alloc::vec![0.5; PAGE_SAMPLES],
            ..args()
        };
        assert!((detailed.page().unwrap().1.w_at(0, 0) - 0.5).abs() < 1.0 / 255.0);
        assert_eq!(PlanetGround::bake(args()).tile_id(), Some(tile));
    }

    #[test]
    fn bad_args_say_what_is_wrong() {
        let off = PlanetGroundArgs {
            tile: [2, 3, 8, 0],
            ..args()
        };
        assert_eq!(
            off.page().unwrap_err(),
            PlanetGroundError::Tile([2, 3, 8, 0])
        );
        for tile in [[6, 0, 0, 0], [0, 21, 0, 0], [300, 1, 0, 0]] {
            assert!(
                PlanetGroundArgs { tile, ..args() }.page().is_err(),
                "{tile:?}"
            );
        }
        let short = PlanetGroundArgs {
            height: alloc::vec![0.0; 10],
            ..args()
        };
        assert_eq!(short.page().unwrap_err(), PlanetGroundError::Height(10));
        let detail = PlanetGroundArgs {
            detail: alloc::vec![1.0; 3],
            ..args()
        };
        assert_eq!(detail.page().unwrap_err(), PlanetGroundError::Detail(3));
        let mut nan = args();
        nan.height[7] = f32::NAN;
        assert!(matches!(
            nan.page(),
            Err(PlanetGroundError::Value("height", _))
        ));
        let over = PlanetGroundArgs {
            detail: alloc::vec![1.5; PAGE_SAMPLES],
            ..args()
        };
        assert_eq!(
            over.page().unwrap_err(),
            PlanetGroundError::Value("detail", 1.5)
        );
        assert!(
            alloc::string::ToString::to_string(&off.page().unwrap_err()).contains("face 0 to 5")
        );
    }
}
