// Where on a planet its edits changed: a cap of the sphere, the area a brush
// stroke covers, tested against the tiles and patches built over it.

use concinnity_core::math::dvec3;
use concinnity_core::math::dvec3::DVec3;

use super::super::tile::TileId;

/// A round area of a planet's surface: every direction within `angle`
/// radians of the unit direction `center`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EditArea {
    /// The area's middle, a unit direction from the planet's center.
    pub center: DVec3,
    /// How far the area reaches from its middle, in radians.
    pub angle: f64,
}

// Slack every overlap test gives, in radians: well under a millimeter on any
// planet, so a tile meeting the area along an edge counts as touched.
const SLACK: f64 = 1e-9;

impl EditArea {
    /// The whole surface.
    pub const EVERYWHERE: Self = Self {
        center: [0.0, 1.0, 0.0],
        angle: core::f64::consts::PI,
    };

    /// The area within `radius` meters along the ground of unit direction
    /// `center`, on a planet of radius `planet_radius`.
    pub fn around(center: DVec3, radius: f64, planet_radius: f64) -> Self {
        Self {
            center: dvec3::normalize_or(center, [0.0, 1.0, 0.0]),
            angle: (radius / planet_radius.max(1e-3)).max(0.0),
        }
    }

    /// The smallest area holding both this one and `other`.
    #[must_use]
    pub fn merged(self, other: Self) -> Self {
        let between = angle_between(self.center, other.center);
        if between + other.angle <= self.angle {
            return self;
        }
        if between + self.angle <= other.angle {
            return other;
        }
        let angle = (between + self.angle + other.angle) * 0.5;
        let across = dvec3::sub(
            other.center,
            dvec3::scale(self.center, dvec3::dot(self.center, other.center)),
        );
        if angle >= core::f64::consts::PI || dvec3::length(across) < 1e-12 {
            return Self::EVERYWHERE;
        }
        // The middle lies along the arc from this middle to the other's.
        let toward = angle - self.angle;
        let side = dvec3::normalize_or(across, self.center);
        Self {
            center: dvec3::add(
                dvec3::scale(self.center, libm::cos(toward)),
                dvec3::scale(side, libm::sin(toward)),
            ),
            angle,
        }
    }

    /// Whether the area reaches unit direction `dir`.
    pub fn contains(&self, dir: DVec3) -> bool {
        angle_between(self.center, dir) <= self.angle + SLACK
    }

    /// The smallest area of this kind holding all of `tile`.
    pub fn of_tile(tile: TileId) -> Self {
        let center = tile.direction(0.5, 0.5);
        // The edges bulge past the corners by little: the middles of the
        // sides join the corners in bounding the tile.
        let reach = [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (0.5, 0.0),
            (0.0, 0.5),
            (1.0, 0.5),
            (0.5, 1.0),
        ]
        .into_iter()
        .map(|(s, t)| angle_between(center, tile.direction(s, t)))
        .fold(0.0, f64::max);
        Self {
            center,
            angle: reach * 1.01,
        }
    }

    /// Whether the area and `other` share any of the surface.
    pub fn overlaps(&self, other: &Self) -> bool {
        angle_between(self.center, other.center) <= self.angle + other.angle + SLACK
    }

    /// Whether the area reaches any of `tile`.
    pub fn overlaps_tile(&self, tile: TileId) -> bool {
        self.overlaps(&Self::of_tile(tile))
    }
}

// The angle between unit directions `a` and `b`, exact for small angles.
fn angle_between(a: DVec3, b: DVec3) -> f64 {
    libm::atan2(dvec3::length(dvec3::cross(a, b)), dvec3::dot(a, b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planet::CubeFace;

    #[test]
    fn an_area_holds_what_lies_within_its_reach() {
        let a = EditArea::around([0.0, 1.0, 0.0], 100.0, 50_000.0);
        assert!((a.angle - 0.002).abs() < 1e-15);
        assert!(a.contains(dvec3::normalize_or([0.0019, 1.0, 0.0], [0.0; 3])));
        assert!(!a.contains(dvec3::normalize_or([0.0021, 1.0, 0.0], [0.0; 3])));
    }

    // A tile is touched where the area reaches any of it, its corners and
    // its edges, and not where the area lies wholly beside it.
    #[test]
    fn tiles_under_the_area_are_touched_and_others_not() {
        let tile = TileId {
            face: CubeFace(2),
            level: 10,
            x: 512,
            y: 511,
        };
        let corner = tile.direction(1.0, 1.0);
        assert!(
            EditArea {
                center: corner,
                angle: 1e-7
            }
            .overlaps_tile(tile)
        );
        let edge = tile.direction(0.5, 0.0);
        assert!(
            EditArea {
                center: edge,
                angle: 0.0
            }
            .overlaps_tile(tile)
        );
        let far = tile
            .neighbor(crate::planet::TileSide::MaxS)
            .neighbor(crate::planet::TileSide::MaxS);
        let beside = EditArea {
            center: far.direction(0.9, 0.5),
            angle: 1e-6,
        };
        assert!(!beside.overlaps_tile(tile));
        assert!(beside.overlaps_tile(TileId::root(CubeFace(2))));
    }

    #[test]
    fn a_merged_area_holds_both() {
        let a = EditArea::around([0.0, 1.0, 0.0], 50.0, 50_000.0);
        let b = EditArea::around(
            dvec3::normalize_or([0.01, 1.0, 0.002], [0.0; 3]),
            20.0,
            50_000.0,
        );
        let m = a.merged(b);
        for area in [a, b] {
            for k in 0..16 {
                let turn = f64::from(k) * core::f64::consts::TAU / 16.0;
                let side = dvec3::normalize_or(
                    dvec3::cross(area.center, [libm::cos(turn), 0.0, libm::sin(turn)]),
                    [1.0, 0.0, 0.0],
                );
                let rim = dvec3::add(
                    dvec3::scale(area.center, libm::cos(area.angle)),
                    dvec3::scale(side, libm::sin(area.angle)),
                );
                assert!(m.contains(rim), "{k}");
            }
        }
        assert!(m.angle < a.angle + b.angle + angle_between(a.center, b.center));
        assert_eq!(a.merged(EditArea { angle: 1e-5, ..a }), a);
        let opposite = EditArea::around([0.0, -1.0, 0.0], 1.0, 50_000.0);
        assert_eq!(a.merged(opposite), EditArea::EVERYWHERE);
    }
}
