// Where drawn planet tiles of different levels meet: keeping neighbors at most
// a level apart, and naming the sides each tile meets a coarser or a finer one
// on, the sides its mesh must match its neighbor's along.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use super::tile::{TileId, TileSide};

/// The sides of a drawn tile that meet a drawn tile a level coarser, and the
/// sides that meet drawn tiles a level finer, each a mask of
/// [`TileSide::bit`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TileSeams {
    /// Sides beside a coarser tile.
    pub coarser: u8,
    /// Sides beside finer tiles.
    pub finer: u8,
}

/// A tile to draw and its seams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrawnTile {
    /// The tile.
    pub id: TileId,
    /// Where it meets tiles of other levels.
    pub seams: TileSeams,
}

// How a square of the quadtree is covered: drawn as one tile, or split into
// drawn tiles below it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cover {
    Drawn,
    Split,
}

// Every drawn tile, and every ancestor of one.
struct CoverMap(hashbrown::HashMap<TileId, Cover>);

impl CoverMap {
    fn new(drawn: &BTreeSet<TileId>) -> Self {
        let mut map = hashbrown::HashMap::with_capacity(drawn.len() * 2);
        for &t in drawn {
            map.insert(t, Cover::Drawn);
        }
        for &t in drawn {
            let mut up = t.parent();
            while let Some(p) = up {
                if map.insert(p, Cover::Split).is_some() {
                    break;
                }
                up = p.parent();
            }
        }
        Self(map)
    }

    // How many levels coarser than `tile` the drawn tile over its square is:
    // 0 when `tile` is drawn itself, `None` when finer tiles cover it or
    // nothing does.
    fn coarser_by(&self, tile: TileId) -> Option<u8> {
        let mut at = Some(tile);
        while let Some(t) = at {
            match self.0.get(&t) {
                Some(Cover::Drawn) => return Some(tile.level - t.level),
                Some(Cover::Split) => return None,
                None => at = t.parent(),
            }
        }
        None
    }

    fn split(&self, tile: TileId) -> bool {
        self.0.get(&tile) == Some(&Cover::Split)
    }
}

// Whether `tile` descends from `ancestor` (or is it).
fn descends_from(tile: TileId, ancestor: TileId) -> bool {
    tile.face == ancestor.face
        && tile.level >= ancestor.level
        && (
            tile.x >> (tile.level - ancestor.level),
            tile.y >> (tile.level - ancestor.level),
        ) == (ancestor.x, ancestor.y)
}

/// Coarsen the cover `drawn` until no two neighboring tiles are more than a
/// level apart: a tile beside one two or more levels coarser gives way, with
/// everything else under its parent, to its parent.
pub(super) fn balance(drawn: &mut BTreeSet<TileId>) {
    loop {
        let map = CoverMap::new(drawn);
        let too_fine = drawn.iter().find_map(|&t| {
            TileSide::ALL
                .iter()
                .any(|&side| map.coarser_by(t.neighbor(side)).is_some_and(|c| c >= 2))
                .then(|| t.parent())
                .flatten()
        });
        let Some(parent) = too_fine else {
            return;
        };
        drawn.retain(|&t| !descends_from(t, parent));
        drawn.insert(parent);
    }
}

/// Each tile of the cover `drawn`, in order, with the sides it meets a
/// coarser or a finer tile on.
pub(super) fn seams(drawn: &BTreeSet<TileId>) -> Vec<DrawnTile> {
    let map = CoverMap::new(drawn);
    drawn
        .iter()
        .map(|&id| {
            let mut seams = TileSeams::default();
            for side in TileSide::ALL {
                let across = id.neighbor(side);
                match map.coarser_by(across) {
                    Some(c) if c > 0 => seams.coarser |= side.bit(),
                    None if map.split(across) => seams.finer |= side.bit(),
                    _ => {}
                }
            }
            DrawnTile { id, seams }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planet::shape::CubeFace;

    // Every tile of `face` at `level`.
    fn whole_face(face: CubeFace, level: u8) -> impl Iterator<Item = TileId> {
        let n = 1u32 << level;
        (0..n * n).map(move |k| TileId {
            face,
            level,
            x: k % n,
            y: k / n,
        })
    }

    // The top face at `level`, every other face whole.
    fn top_at(level: u8) -> BTreeSet<TileId> {
        CubeFace::ALL
            .into_iter()
            .flat_map(|f| {
                let level = if f == CubeFace(2) { level } else { 0 };
                whole_face(f, level)
            })
            .collect()
    }

    fn covers_every_face_once(drawn: &BTreeSet<TileId>) {
        for face in CubeFace::ALL {
            let area: f64 = drawn
                .iter()
                .filter(|t| t.face == face)
                .map(|t| 1.0 / f64::from(1u32 << (2 * u32::from(t.level))))
                .sum();
            assert!((area - 1.0).abs() < 1e-12, "face {face:?} covered {area}");
        }
    }

    // A face two levels finer than the faces around it gives way until it is
    // one level finer, still covering itself once.
    #[test]
    fn balance_coarsens_tiles_two_levels_finer_than_a_neighbor() {
        let mut drawn = top_at(2);
        balance(&mut drawn);
        covers_every_face_once(&drawn);
        assert_eq!(drawn, top_at(1));
        let map = CoverMap::new(&drawn);
        for &t in &drawn {
            for side in TileSide::ALL {
                assert!(map.coarser_by(t.neighbor(side)).is_none_or(|c| c <= 1));
            }
        }
    }

    #[test]
    fn a_balanced_cover_is_left_alone() {
        let mut drawn = top_at(1);
        balance(&mut drawn);
        assert_eq!(drawn, top_at(1));
    }

    // The top face a level finer: its tiles meet coarser faces on their outer
    // sides, and the faces around it meet finer tiles on the side toward it.
    #[test]
    fn seams_name_the_sides_meeting_other_levels() {
        let drawn = top_at(1);
        let seams: hashbrown::HashMap<TileId, TileSeams> =
            seams(&drawn).into_iter().map(|d| (d.id, d.seams)).collect();
        let corner = TileId {
            face: CubeFace(2),
            level: 1,
            x: 0,
            y: 0,
        };
        assert_eq!(
            seams[&corner],
            TileSeams {
                coarser: TileSide::MinS.bit() | TileSide::MinT.bit(),
                finer: 0,
            }
        );
        // The top face's -s side looks down -X, onto -X's +t side.
        assert_eq!(
            seams[&TileId::root(CubeFace(1))],
            TileSeams {
                coarser: 0,
                finer: TileSide::MaxT.bit(),
            }
        );
        // The bottom face touches no finer tile.
        assert_eq!(seams[&TileId::root(CubeFace(3))], TileSeams::default());
    }

    #[test]
    fn descent_follows_the_quadtree() {
        let root = TileId::root(CubeFace(4));
        let child = root.children()[3];
        let grandchild = child.children()[0];
        assert!(descends_from(grandchild, root));
        assert!(descends_from(grandchild, child));
        assert!(descends_from(child, child));
        assert!(!descends_from(child, grandchild));
        assert!(!descends_from(root.children()[0].children()[0], child));
        assert!(!descends_from(grandchild, TileId::root(CubeFace(5))));
    }
}
