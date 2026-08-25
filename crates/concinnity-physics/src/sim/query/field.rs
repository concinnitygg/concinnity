// concinnity-physics/src/sim/query/field.rs
//
// Rays and swept shapes against terrain.
//
// Neither question goes through the iterative distance query the convex shapes
// use, and for the same reason the narrow phase does not: the nearest feature
// of a surface made of triangles is often the edge two of them share, and an
// edge normal points sideways out of a surface a shape was sliding along. Both
// answers here come from a triangle's own plane instead, so a capsule crossing
// a cell boundary meets two planes that agree rather than an edge that does
// not.
//
// The sweep is exact because it is a plane. A translated shape's furthest
// point along a fixed direction moves with the shape, so the distance to a
// plane changes linearly with the sweep and the moment of contact is one
// division rather than a search. What the clip then decides is whether that
// contact happened over this triangle or over its neighbour.
//
// The ray walks the grid cell by cell along its own direction rather than
// testing everything under its bounds, because a ray long enough to cross a
// terrain has bounds that cover most of it.

use crate::ColliderShape;

use crate::sim::aabb::Aabb;
use crate::sim::collide::Pose;
use crate::sim::collide::heightfield::{Heightfield, Heightfields, MAX_CANDIDATE_TRIANGLES};
use crate::sim::collide::triangle::{
    MAX_TRIANGLE_POINTS, Triangle, TriangleContact, contacts, support_point,
};
use crate::sim::math::Vec3;

use super::ray::{Ray, RayImpact};
use super::sweep::{SweepImpact, TOUCH_GAP};

/// Slack the impact pose is re-measured with, so a shape stopped exactly on a
/// plane still reads as touching it.
const IMPACT_MARGIN: f32 = 1.0e-3;

/// Directions flatter than this on the sweep axis are treated as parallel, so
/// a grid walk divides by nothing.
const PARALLEL: f32 = 1.0e-8;

/// How far outside a triangle's own area a ray may land and still be counted
/// as having hit it. A ray dropped exactly on the boundary between two cells
/// is on the outer edge of one of them and only that cell is walked, so
/// without the slack it would pass through the surface.
const BARY_SLACK: f32 = 1.0e-5;

/// Where a ray first meets a grid, within `max_dist`.
pub(crate) fn raycast(
    fields: &Heightfields,
    index: u32,
    ray: Ray,
    max_dist: f32,
) -> Option<RayImpact> {
    let field = fields.get(index)?;
    let (enter, exit) = clip_to(field.bounds(), ray, max_dist)?;

    let mut walk = GridWalk::new(field, ray, enter, exit)?;
    let mut best: Option<(u32, RayImpact)> = None;
    while let Some((row, col, cell_entry)) = walk.next_cell() {
        if best.is_some_and(|(_, hit)| cell_entry > hit.distance) {
            break;
        }
        for (half, triangle) in field.triangles_at(row, col).into_iter().enumerate() {
            let Some(triangle) = triangle else { continue };
            let Some(distance) = pierce(&triangle, ray, max_dist) else {
                continue;
            };
            let key = field.triangle_key(row, col, half);
            if best.is_none_or(|(kept, hit)| {
                distance < hit.distance || (distance == hit.distance && key < kept)
            }) {
                best = Some((
                    key,
                    RayImpact {
                        distance,
                        // Whichever side the ray came from, the normal faces
                        // back along it.
                        normal: if triangle.normal.dot(ray.direction) > 0.0 {
                            -triangle.normal
                        } else {
                            triangle.normal
                        },
                    },
                ));
            }
        }
    }
    if walk.cut_off {
        fields.note_overflow();
    }
    best.map(|(_, hit)| hit)
}

/// The nearest triangle a swept shape runs into.
pub(crate) fn sweep(
    fields: &Heightfields,
    index: u32,
    shape: &ColliderShape,
    pose: Pose,
    motion: Vec3,
    swept: Aabb,
) -> Option<SweepImpact> {
    let field = fields.get(index)?;
    let mut found = [TriangleContact {
        point: Vec3::ZERO,
        separation: 0.0,
        feature: 0,
    }; MAX_TRIANGLE_POINTS];

    let mut best: Option<(u32, SweepImpact)> = None;
    let mut candidates = field.candidates(swept);
    for (key, triangle) in &mut candidates {
        let deepest = support_point(shape, pose, -triangle.normal);
        let start_gap = triangle.height_of(deepest);
        let closing = -motion.dot(triangle.normal);

        let toi = if start_gap <= TOUCH_GAP {
            0.0
        } else if closing > 0.0 {
            let reached = start_gap / closing;
            if reached > 1.0 {
                continue;
            }
            reached
        } else {
            continue;
        };
        if best.is_some_and(|(kept, hit)| toi > hit.toi || (toi == hit.toi && key > kept)) {
            continue;
        }

        // The plane says when; the triangle's own extent says whether it was
        // this triangle the shape met or the one next to it.
        let count = contacts(
            &triangle,
            shape,
            pose,
            motion * toi,
            IMPACT_MARGIN,
            &mut found,
        );
        let Some(contact) = found[..count]
            .iter()
            .min_by(|a, b| a.separation.total_cmp(&b.separation))
        else {
            continue;
        };
        best = Some((
            key,
            SweepImpact {
                toi,
                point: contact.point - triangle.normal * contact.separation,
                normal: triangle.normal,
                gap: start_gap - toi * closing,
                started_touching: toi <= 0.0,
            },
        ));
    }
    fields.note(&candidates);
    best.map(|(_, hit)| hit)
}

/// Where a ray enters and leaves a box, within `max_dist`.
fn clip_to(bounds: Aabb, ray: Ray, max_dist: f32) -> Option<(f32, f32)> {
    let mut near = 0.0f32;
    let mut far = max_dist;
    for axis in 0..3 {
        let direction = ray.direction.get(axis);
        let origin = ray.origin.get(axis);
        if libm::fabsf(direction) < PARALLEL {
            if origin < bounds.min.get(axis) || origin > bounds.max.get(axis) {
                return None;
            }
            continue;
        }
        let inverse = 1.0 / direction;
        let mut low = (bounds.min.get(axis) - origin) * inverse;
        let mut high = (bounds.max.get(axis) - origin) * inverse;
        if low > high {
            core::mem::swap(&mut low, &mut high);
        }
        near = near.max(low);
        far = far.min(high);
        if near > far {
            return None;
        }
    }
    Some((near, far))
}

/// Where a ray meets one triangle, from either side.
fn pierce(triangle: &Triangle, ray: Ray, max_dist: f32) -> Option<f32> {
    let [v0, v1, v2] = triangle.corners;
    let (edge1, edge2) = (v1 - v0, v2 - v0);
    let across = ray.direction.cross(edge2);
    let determinant = edge1.dot(across);
    if libm::fabsf(determinant) < PARALLEL {
        return None;
    }
    let inverse = 1.0 / determinant;
    let to_origin = ray.origin - v0;
    let u = to_origin.dot(across) * inverse;
    if !(-BARY_SLACK..=1.0 + BARY_SLACK).contains(&u) {
        return None;
    }
    let along = to_origin.cross(edge1);
    let v = ray.direction.dot(along) * inverse;
    if v < -BARY_SLACK || u + v > 1.0 + BARY_SLACK {
        return None;
    }
    let distance = edge2.dot(along) * inverse;
    (0.0..=max_dist).contains(&distance).then_some(distance)
}

/// A walk over the cells a ray crosses, in the order it crosses them.
struct GridWalk<'a> {
    field: &'a Heightfield,
    /// Cell the walk is about to hand back.
    row: isize,
    col: isize,
    /// Which way each index moves, and how far along the ray one whole cell is.
    row_step: isize,
    col_step: isize,
    row_delta: f32,
    col_delta: f32,
    /// Distance along the ray to the next boundary on each axis.
    row_next: f32,
    col_next: f32,
    /// Distance to the cell about to be handed back, and where the walk ends.
    entry: f32,
    exit: f32,
    budget: usize,
    cut_off: bool,
}

impl<'a> GridWalk<'a> {
    fn new(field: &'a Heightfield, ray: Ray, enter: f32, exit: f32) -> Option<Self> {
        let (rows, cols) = field.cells();
        let (cell_x, cell_z) = field.cell_size();
        let (min_x, min_z) = field.grid_min();
        let (row, col) = field.cell_at(ray.origin + ray.direction * enter);

        let axis = |position: f32,
                    direction: f32,
                    start: f32,
                    size: f32,
                    cell: isize|
         -> (isize, f32, f32) {
            if libm::fabsf(direction) < PARALLEL {
                return (0, f32::INFINITY, f32::INFINITY);
            }
            let step = if direction > 0.0 { 1 } else { -1 };
            let boundary = start + (cell + isize::from(direction > 0.0)) as f32 * size;
            (
                step,
                libm::fabsf(size / direction),
                enter + (boundary - position) / direction,
            )
        };
        let at = ray.origin + ray.direction * enter;
        let (col_step, col_delta, col_next) =
            axis(at.x, ray.direction.x, min_x, cell_x, col as isize);
        let (row_step, row_delta, row_next) =
            axis(at.z, ray.direction.z, min_z, cell_z, row as isize);

        (rows > 0 && cols > 0).then_some(GridWalk {
            field,
            row: row as isize,
            col: col as isize,
            row_step,
            col_step,
            row_delta,
            col_delta,
            row_next,
            col_next,
            entry: enter,
            exit,
            budget: MAX_CANDIDATE_TRIANGLES / 2,
            cut_off: false,
        })
    }

    /// The next cell the ray crosses, with how far along the ray it starts.
    fn next_cell(&mut self) -> Option<(usize, usize, f32)> {
        let (rows, cols) = self.field.cells();
        if self.row < 0 || self.col < 0 || self.row >= rows as isize || self.col >= cols as isize {
            return None;
        }
        if self.entry > self.exit {
            return None;
        }
        if self.budget == 0 {
            self.cut_off = true;
            return None;
        }
        self.budget -= 1;
        let cell = (self.row as usize, self.col as usize, self.entry);
        if self.col_next < self.row_next {
            self.entry = self.col_next;
            self.col_next += self.col_delta;
            self.col += self.col_step;
        } else {
            self.entry = self.row_next;
            self.row_next += self.row_delta;
            self.row += self.row_step;
        }
        Some(cell)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::math::{Quat, vec3};
    use alloc::vec;

    /// Flat terrain at `y = 0`, ten units square, four cells on a side.
    fn flat_table() -> (Heightfields, u32) {
        let mut fields = Heightfields::new();
        let field = Heightfield::new(5, 5, vec![0.0; 25], vec3(10.0, 1.0, 10.0), Vec3::ZERO)
            .expect("a real grid");
        let index = fields.push(field);
        (fields, index)
    }

    /// Terrain rising one unit per unit of `+x`.
    fn slope_table() -> (Heightfields, u32) {
        let mut heights = vec::Vec::new();
        for _ in 0..5 {
            for col in 0..5 {
                heights.push(col as f32 * 2.5 - 5.0);
            }
        }
        let mut fields = Heightfields::new();
        let field = Heightfield::new(5, 5, heights, vec3(10.0, 1.0, 10.0), Vec3::ZERO)
            .expect("a real grid");
        let index = fields.push(field);
        (fields, index)
    }

    fn down(origin: Vec3) -> Ray {
        Ray {
            origin,
            direction: -Vec3::Y,
        }
    }

    #[test]
    fn a_ray_dropped_onto_flat_terrain_lands_on_it_facing_up() {
        let (fields, index) = flat_table();
        for x in [-4.5f32, -1.3, 0.0, 2.7, 4.4] {
            for z in [-4.1f32, -0.2, 3.3] {
                let hit = raycast(&fields, index, down(vec3(x, 6.0, z)), 20.0)
                    .unwrap_or_else(|| panic!("nothing under ({x}, {z})"));
                assert!((hit.distance - 6.0).abs() < 1.0e-4, "({x}, {z}): {hit:?}");
                assert!(hit.normal.y > 0.999, "({x}, {z}): {hit:?}");
            }
        }
    }

    #[test]
    fn a_ray_that_misses_the_grid_or_stops_short_reports_nothing() {
        let (fields, index) = flat_table();
        assert!(raycast(&fields, index, down(vec3(20.0, 6.0, 0.0)), 40.0).is_none());
        assert!(raycast(&fields, index, down(vec3(0.0, 6.0, 0.0)), 5.0).is_none());
        assert!(raycast(&fields, index, down(vec3(0.0, 6.0, 0.0)), 6.1).is_some());
        assert!(raycast(&fields, index, down(Vec3::ZERO), 20.0).is_some());
        assert!(raycast(&fields, index, down(vec3(0.0, -6.0, 0.0)), 20.0).is_none());
    }

    // A ray across the terrain has to walk the cells it crosses, and stop at
    // the first surface it meets rather than the nearest one to its origin.
    #[test]
    fn a_ray_along_a_slope_meets_the_first_face_that_rises_into_it() {
        let (fields, index) = slope_table();
        let ray = Ray {
            origin: vec3(-4.9, 0.0, 0.1),
            direction: Vec3::X,
        };
        let hit = raycast(&fields, index, ray, 20.0).expect("the slope is in the way");
        // The surface passes y = 0 halfway across, so a ray held at y = 0 from
        // the low end meets it around the middle.
        assert!((hit.distance - 4.9).abs() < 0.3, "{hit:?}");
        assert!(hit.normal.x < -0.3, "it faces back down the slope: {hit:?}");
    }

    #[test]
    fn a_ray_from_under_the_surface_is_answered_with_a_normal_facing_it() {
        let (fields, index) = flat_table();
        let hit = raycast(
            &fields,
            index,
            Ray {
                origin: vec3(0.0, -3.0, 0.0),
                direction: Vec3::Y,
            },
            20.0,
        )
        .expect("the surface is above");
        assert!((hit.distance - 3.0).abs() < 1.0e-4, "{hit:?}");
        assert!(hit.normal.y < -0.999, "{hit:?}");
    }

    fn at(position: Vec3) -> Pose {
        Pose {
            position,
            rotation: Quat::IDENTITY,
        }
    }

    #[test]
    fn a_shape_dropped_onto_flat_terrain_stops_on_it() {
        let (fields, index) = flat_table();
        let capsule = ColliderShape::Capsule {
            half_height: 0.6,
            radius: 0.3,
        };
        let hit = sweep(
            &fields,
            index,
            &capsule,
            at(vec3(-1.0, 4.0, 1.0)),
            vec3(0.0, -8.0, 0.0),
            Aabb {
                min: vec3(-2.0, -5.0, 0.0),
                max: vec3(0.0, 5.0, 2.0),
            },
        )
        .expect("the ground is down there");
        let landed = 4.0 - hit.toi * 8.0;
        assert!((landed - 0.9).abs() < 0.01, "landed at {landed}");
        assert!(hit.normal.y > 0.999, "{hit:?}");
        assert!(!hit.started_touching);
    }

    #[test]
    fn a_shape_already_on_the_surface_says_so() {
        let (fields, index) = flat_table();
        let ball = ColliderShape::Ball { radius: 0.5 };
        let hit = sweep(
            &fields,
            index,
            &ball,
            at(vec3(0.0, 0.5, 0.0)),
            vec3(1.0, 0.0, 0.0),
            Aabb {
                min: vec3(-1.0, -1.0, -1.0),
                max: vec3(2.0, 1.0, 1.0),
            },
        )
        .expect("it is standing on it");
        assert_eq!(hit.toi, 0.0);
        assert!(hit.started_touching);
        assert!(hit.gap.abs() < 1.0e-3, "{hit:?}");
    }

    // The property the whole module exists for: a capsule driven across the
    // boundary between two cells must not be stopped by the edge they share.
    #[test]
    fn a_capsule_driven_across_a_cell_boundary_is_not_caught_by_it() {
        let (fields, index) = flat_table();
        let capsule = ColliderShape::Capsule {
            half_height: 0.6,
            radius: 0.3,
        };
        // Cell boundaries sit every 2.5 units; this walk crosses several.
        let mut x = -4.0f32;
        for _ in 0..200 {
            let from = vec3(x, 0.9, 0.15);
            let motion = vec3(0.05, 0.0, 0.0);
            let bounds = Aabb {
                min: from - Vec3::splat(1.0),
                max: from + motion + Vec3::splat(1.0),
            };
            let blocked = sweep(&fields, index, &capsule, at(from), motion, bounds)
                .filter(|hit| hit.toi < 0.999 && hit.normal.y < 0.9);
            assert!(
                blocked.is_none(),
                "caught crossing a cell boundary at x = {x}: {blocked:?}"
            );
            x += 0.05;
        }
    }

    #[test]
    fn a_sweep_that_reaches_nothing_reports_nothing() {
        let (fields, index) = flat_table();
        let ball = ColliderShape::Ball { radius: 0.2 };
        assert!(
            sweep(
                &fields,
                index,
                &ball,
                at(vec3(0.0, 5.0, 0.0)),
                vec3(2.0, 0.0, 0.0),
                Aabb {
                    min: vec3(-1.0, 4.0, -1.0),
                    max: vec3(3.0, 6.0, 1.0),
                },
            )
            .is_none()
        );
    }

    // A shape driven into a rising slope has to be stopped by it, with the
    // slope's own normal to slide along.
    #[test]
    fn a_shape_driven_into_a_slope_is_stopped_by_its_face() {
        let (fields, index) = slope_table();
        let ball = ColliderShape::Ball { radius: 0.4 };
        let from = vec3(-4.0, -1.5, 0.0);
        let motion = vec3(4.0, 0.0, 0.0);
        let hit = sweep(
            &fields,
            index,
            &ball,
            at(from),
            motion,
            Aabb {
                min: from - Vec3::splat(1.0),
                max: from + motion + Vec3::splat(1.0),
            },
        )
        .expect("the slope is in the way");
        assert!(hit.toi > 0.0 && hit.toi < 1.0, "{hit:?}");
        assert!(hit.normal.x < -0.3 && hit.normal.y > 0.3, "{hit:?}");
    }
}
