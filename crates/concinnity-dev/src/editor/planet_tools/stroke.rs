//! A stroke: a drag across the ground laid out as points a set distance apart
//! along it, whatever the cursor's speed, so a tool works the ground evenly.
//! Points are unit directions from the planet's center, spaced by the arc
//! they make on its datum sphere.

use concinnity_core::math::dvec3;
use concinnity_engine::planet::DVec3;

/// The spacing between a stroke's points, as a share of the brush radius.
pub(crate) const SPACING: f64 = 0.25;

// The most points one move lays; a longer jump restarts the stroke at the
// cursor instead of filling the gap.
const MAX_POINTS_PER_MOVE: usize = 256;

/// Lays a stroke's points as its cursor moves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Spacer {
    // The angle between points, in radians.
    step: f64,
    // The last point laid; `None` until the stroke reaches the ground.
    last: Option<DVec3>,
}

impl Spacer {
    /// A stroke whose points lie `spacing` meters apart on a planet of
    /// `radius` meters.
    pub(crate) fn new(spacing: f64, radius: f64) -> Self {
        Self {
            step: (spacing / radius.max(1.0)).max(1e-12),
            last: None,
        }
    }

    /// Move the stroke's cursor to unit direction `to`, calling `emit` with
    /// each point laid on the way, in order: the first point where the stroke
    /// reaches the ground, then one every step from the last one laid.
    pub(crate) fn advance(&mut self, to: DVec3, mut emit: impl FnMut(DVec3)) {
        let Some(mut from) = self.last else {
            self.last = Some(to);
            emit(to);
            return;
        };
        let mut gap = angle(from, to);
        if gap > self.step * MAX_POINTS_PER_MOVE as f64 {
            self.last = Some(to);
            emit(to);
            return;
        }
        while gap >= self.step {
            from = toward(from, to, self.step);
            gap -= self.step;
            emit(from);
        }
        self.last = Some(from);
    }

    /// The cursor left the ground: the stroke picks up again, without
    /// bridging the gap, wherever the cursor next reaches it.
    pub(crate) fn lift(&mut self) {
        self.last = None;
    }
}

// The angle between unit directions, precise for small angles.
fn angle(a: DVec3, b: DVec3) -> f64 {
    dvec3::length(dvec3::cross(a, b)).atan2(dvec3::dot(a, b))
}

// The unit direction `step` radians from `from` along the great circle to
// `to`.
fn toward(from: DVec3, to: DVec3, step: f64) -> DVec3 {
    let along = dvec3::sub(to, dvec3::scale(from, dvec3::dot(from, to)));
    let along = dvec3::normalize_or(along, from);
    dvec3::normalize_or(
        dvec3::add(
            dvec3::scale(from, step.cos()),
            dvec3::scale(along, step.sin()),
        ),
        from,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const RADIUS: f64 = 50_000.0;

    // The unit direction `meters` east of the top along the equator.
    fn east(meters: f64) -> DVec3 {
        let a = meters / RADIUS;
        [a.sin(), a.cos(), 0.0]
    }

    fn arc(a: DVec3, b: DVec3) -> f64 {
        angle(a, b) * RADIUS
    }

    fn drag(spacer: &mut Spacer, to: DVec3) -> Vec<DVec3> {
        let mut out = Vec::new();
        spacer.advance(to, |p| out.push(p));
        out
    }

    #[test]
    fn the_first_point_is_where_the_stroke_starts() {
        let mut s = Spacer::new(2.0, RADIUS);
        assert_eq!(drag(&mut s, east(0.0)), vec![east(0.0)]);
    }

    // However the drag is cut into moves, points land every spacing along it.
    #[test]
    fn points_fall_one_spacing_apart_whatever_the_moves() {
        for moves in [1, 3, 17, 400] {
            let mut s = Spacer::new(2.0, RADIUS);
            let mut points = drag(&mut s, east(0.0));
            for k in 1..=moves {
                points.extend(drag(&mut s, east(21.0 * f64::from(k) / f64::from(moves))));
            }
            assert_eq!(points.len(), 11, "{moves} moves");
            for (i, p) in points.iter().enumerate() {
                let along = arc(east(0.0), *p);
                assert!((along - 2.0 * i as f64).abs() < 1e-6, "{along}");
            }
        }
    }

    // Short moves lay nothing until the cursor has gone a whole spacing.
    #[test]
    fn short_moves_wait_for_a_whole_spacing() {
        let mut s = Spacer::new(2.0, RADIUS);
        drag(&mut s, east(0.0));
        assert!(drag(&mut s, east(0.9)).is_empty());
        assert!(drag(&mut s, east(1.8)).is_empty());
        let laid = drag(&mut s, east(2.1));
        assert_eq!(laid.len(), 1);
        assert!((arc(east(0.0), laid[0]) - 2.0).abs() < 1e-6);
    }

    // The points follow the drag's direction, staying on the great circle
    // toward the cursor.
    #[test]
    fn points_head_toward_the_cursor() {
        let mut s = Spacer::new(1.0, RADIUS);
        let start = [0.0, 1.0, 0.0];
        drag(&mut s, start);
        let a = 10.0 / RADIUS;
        let north = [0.0, a.cos(), -a.sin()];
        let laid = drag(&mut s, north);
        assert_eq!(laid.len(), 10);
        for p in laid {
            assert!(p[0].abs() < 1e-12);
            assert!((dvec3::length(p) - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn a_lifted_stroke_restarts_without_bridging() {
        let mut s = Spacer::new(2.0, RADIUS);
        drag(&mut s, east(0.0));
        s.lift();
        assert_eq!(drag(&mut s, east(30.0)), vec![east(30.0)]);
        assert_eq!(drag(&mut s, east(34.5)).len(), 2);
    }

    // A jump across half the planet restarts at the cursor rather than
    // laying thousands of points in one frame.
    #[test]
    fn a_huge_jump_restarts_at_the_cursor() {
        let mut s = Spacer::new(2.0, RADIUS);
        drag(&mut s, east(0.0));
        assert_eq!(drag(&mut s, east(60_000.0)), vec![east(60_000.0)]);
    }

    #[test]
    fn spacing_scales_with_the_brush() {
        let mut s = Spacer::new(8.0 * SPACING, RADIUS);
        drag(&mut s, east(0.0));
        assert_eq!(drag(&mut s, east(20.0)).len(), 10);
    }
}
