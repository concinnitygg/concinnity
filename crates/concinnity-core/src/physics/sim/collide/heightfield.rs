// Terrain: a grid of heights, and the triangles a query has to look at.
//
// A height grid is not a convex shape and cannot be made into one, so it is
// stored apart from the bodies and referenced by them. What a body holds is an
// index into the table below; the grid itself is fixed, unrotated, and built
// once, which is what lets every vertex be kept in world space and every
// lookup be arithmetic rather than a transform.
//
// The cost that has to be bounded is how many triangles a question touches. A
// query names an axis-aligned box, the box names a rectangle of cells, and the
// cells hand back their triangles in row-major order -- but a large enough box
// names more triangles than any query should be allowed to spend, so the walk
// stops at a fixed count and records that it did. Declining and saying so is
// the house rule; growing a buffer to fit would put an allocation on the step
// path, and silently answering with part of the surface would read as an
// answer.

use crate::math::floor;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, Ordering};

use crate::physics::ColliderShape;
use crate::physics::sim::aabb::Aabb;
use crate::physics::sim::contact::{Manifold, ManifoldPoint};
use crate::physics::sim::math::{Vec3, vec3};

use super::support::{Pose, reduce_to_quad};
use super::triangle::{MAX_TRIANGLE_POINTS, Triangle, TriangleContact, contacts};

/// Triangles one query may look at before it is cut off. Sixteen cells square
/// is more surface than a body the size of a character ever spans, and a
/// question that wants more is asking the wrong structure.
pub(crate) const MAX_CANDIDATE_TRIANGLES: usize = 512;

/// Contact planes one body may lean on across a whole grid. A body resting in
/// a fold touches two, and one straddling a cell corner four; past that the
/// extra planes repeat what the deepest ones already say.
const MAX_FIELD_MANIFOLDS: usize = 4;

/// A grid of world-space heights, and the extent it covers.
///
/// `heights` is row-major with `rows * cols` entries. Rows run along `z` and
/// columns along `x`, which is the order the terrain generator writes.
pub(crate) struct Heightfield {
    rows: usize,
    cols: usize,
    heights: Vec<f32>,
    /// World position of the grid's centre.
    origin: Vec3,
    /// Half the footprint on `x` and `z`, and the multiplier on a stored
    /// height.
    half_width: f32,
    half_depth: f32,
    height_scale: f32,
    /// Footprint of one cell.
    cell_x: f32,
    cell_z: f32,
    bounds: Aabb,
}

impl Heightfield {
    /// Build a grid, or `None` when it names no surface: fewer than two rows
    /// or columns, the wrong number of heights, or a footprint of nothing.
    pub(crate) fn new(
        rows: usize,
        cols: usize,
        heights: Vec<f32>,
        scale: Vec3,
        origin: Vec3,
    ) -> Option<Self> {
        if rows < 2 || cols < 2 || heights.len() != rows * cols {
            return None;
        }
        if !scale.is_finite() || !origin.is_finite() {
            return None;
        }
        let (width, depth) = (scale.x.abs(), scale.z.abs());
        if width <= 0.0 || depth <= 0.0 {
            return None;
        }
        let (mut lowest, mut highest) = (f32::INFINITY, f32::NEG_INFINITY);
        for height in &heights {
            if !height.is_finite() {
                return None;
            }
            let scaled = height * scale.y;
            lowest = lowest.min(scaled);
            highest = highest.max(scaled);
        }
        let (half_width, half_depth) = (width * 0.5, depth * 0.5);
        let bounds = Aabb {
            min: origin + vec3(-half_width, lowest, -half_depth),
            max: origin + vec3(half_width, highest, half_depth),
        };
        Some(Heightfield {
            rows,
            cols,
            heights,
            origin,
            half_width,
            half_depth,
            height_scale: scale.y,
            cell_x: width / (cols - 1) as f32,
            cell_z: depth / (rows - 1) as f32,
            bounds,
        })
    }

    pub(crate) fn bounds(&self) -> Aabb {
        self.bounds
    }

    pub(crate) fn reserved_bytes(&self) -> u64 {
        (self.heights.capacity() * size_of::<f32>() + size_of::<Heightfield>()) as u64
    }

    /// The world-space grid vertex at a row and column.
    pub(crate) fn vertex(&self, row: usize, col: usize) -> Vec3 {
        vec3(
            self.origin.x - self.half_width + col as f32 * self.cell_x,
            self.origin.y + self.heights[row * self.cols + col] * self.height_scale,
            self.origin.z - self.half_depth + row as f32 * self.cell_z,
        )
    }

    /// The two triangles of one cell, split along the diagonal that runs from
    /// its low corner. `None` where the corners are collinear.
    pub(crate) fn triangles_at(&self, row: usize, col: usize) -> [Option<Triangle>; 2] {
        if row + 1 >= self.rows || col + 1 >= self.cols {
            return [None, None];
        }
        let near_low = self.vertex(row, col);
        let near_high = self.vertex(row, col + 1);
        let far_low = self.vertex(row + 1, col);
        let far_high = self.vertex(row + 1, col + 1);
        [
            Triangle::new([near_low, far_low, near_high]),
            Triangle::new([far_low, far_high, near_high]),
        ]
    }

    /// Cells the grid is divided into, as rows by columns.
    pub(crate) fn cells(&self) -> (usize, usize) {
        (self.rows - 1, self.cols - 1)
    }

    /// The footprint of one cell on `x` and on `z`.
    pub(crate) fn cell_size(&self) -> (f32, f32) {
        (self.cell_x, self.cell_z)
    }

    /// The grid's low corner on `x` and `z`.
    pub(crate) fn grid_min(&self) -> (f32, f32) {
        (
            self.origin.x - self.half_width,
            self.origin.z - self.half_depth,
        )
    }

    /// The cell a world point stands over, clamped to the grid.
    pub(crate) fn cell_at(&self, point: Vec3) -> (usize, usize) {
        let (min_x, min_z) = self.grid_min();
        let (rows, cols) = self.cells();
        let clamp = |value: f32, start: f32, step: f32, count: usize| {
            floor((value - start) / step).clamp(0.0, (count - 1) as f32) as usize
        };
        (
            clamp(point.z, min_z, self.cell_z, rows),
            clamp(point.x, min_x, self.cell_x, cols),
        )
    }

    /// A number naming one triangle of the grid, so a contact on it can be
    /// recognised again next step.
    pub(crate) fn triangle_key(&self, row: usize, col: usize, half: usize) -> u32 {
        ((row * (self.cols - 1) + col) * 2 + half) as u32
    }

    /// The cells an axis-aligned box reaches, or `None` when it reaches none.
    pub(crate) fn cell_range(&self, bounds: Aabb) -> Option<CellRange> {
        if !self.bounds.overlaps(bounds) {
            return None;
        }
        let span = |low: f32, high: f32, start: f32, step: f32, count: usize| {
            let first = floor((low - start) / step);
            let last = floor((high - start) / step);
            let limit = (count - 2) as f32;
            (
                first.clamp(0.0, limit) as usize,
                last.clamp(0.0, limit) as usize,
            )
        };
        let (col0, col1) = span(
            bounds.min.x,
            bounds.max.x,
            self.origin.x - self.half_width,
            self.cell_x,
            self.cols,
        );
        let (row0, row1) = span(
            bounds.min.z,
            bounds.max.z,
            self.origin.z - self.half_depth,
            self.cell_z,
            self.rows,
        );
        Some(CellRange {
            row0,
            row1,
            col0,
            col1,
        })
    }

    /// The triangles under an axis-aligned box, in row-major order, up to the
    /// cap.
    pub(crate) fn candidates(&self, bounds: Aabb) -> Candidates<'_> {
        Candidates {
            field: self,
            range: self.cell_range(bounds),
            row: 0,
            col: 0,
            half: 0,
            budget: MAX_CANDIDATE_TRIANGLES,
            started: false,
            cut_off: false,
        }
    }
}

/// An inclusive rectangle of cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CellRange {
    pub(crate) row0: usize,
    pub(crate) row1: usize,
    pub(crate) col0: usize,
    pub(crate) col1: usize,
}

/// A bounded walk over the triangles of a rectangle of cells.
pub(crate) struct Candidates<'a> {
    field: &'a Heightfield,
    range: Option<CellRange>,
    row: usize,
    col: usize,
    half: usize,
    budget: usize,
    /// Whether the first cell has been taken up yet.
    started: bool,
    cut_off: bool,
}

impl Candidates<'_> {
    /// Whether the walk stopped at the cap with surface still to look at.
    pub(crate) fn cut_off(&self) -> bool {
        self.cut_off
    }
}

impl Iterator for Candidates<'_> {
    type Item = (u32, Triangle);

    fn next(&mut self) -> Option<Self::Item> {
        let range = self.range?;
        if !self.started {
            self.started = true;
            self.row = range.row0;
            self.col = range.col0;
        }
        loop {
            if self.row > range.row1 {
                return None;
            }
            if self.budget == 0 {
                self.cut_off = true;
                return None;
            }
            let (row, col, half) = (self.row, self.col, self.half);
            self.half += 1;
            if self.half > 1 {
                self.half = 0;
                self.col += 1;
                if self.col > range.col1 {
                    self.col = range.col0;
                    self.row += 1;
                }
            }
            self.budget -= 1;
            if let Some(triangle) = self.field.triangles_at(row, col)[half] {
                return Some((self.field.triangle_key(row, col, half), triangle));
            }
        }
    }
}

/// Every height grid in the simulation, and how often a query has had to give
/// up on one.
pub(crate) struct Heightfields {
    fields: Vec<Heightfield>,
    /// Queries cut off at the candidate cap since the count was last cleared.
    /// Diagnostic only: it never changes an answer, so a relaxed counter is
    /// enough and it keeps a query shareable between threads.
    overflows: AtomicU32,
}

impl Heightfields {
    pub(crate) fn new() -> Self {
        Heightfields {
            fields: Vec::new(),
            overflows: AtomicU32::new(0),
        }
    }

    /// Store a grid and return the index a body names it by.
    pub(crate) fn push(&mut self, field: Heightfield) -> u32 {
        self.fields.push(field);
        (self.fields.len() - 1) as u32
    }

    pub(crate) fn get(&self, index: u32) -> Option<&Heightfield> {
        self.fields.get(index as usize)
    }

    #[cfg(test)]
    pub(crate) fn overflows(&self) -> u32 {
        self.overflows.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn clear_overflows(&self) {
        self.overflows.store(0, Ordering::Relaxed);
    }

    /// Record that a walk ran out of candidates, if it did.
    pub(crate) fn note(&self, candidates: &Candidates<'_>) {
        if candidates.cut_off() {
            self.note_overflow();
        }
    }

    /// Record that a query gave up with surface still to look at.
    pub(crate) fn note_overflow(&self) {
        self.overflows.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn reserved_bytes(&self) -> u64 {
        self.fields
            .iter()
            .map(Heightfield::reserved_bytes)
            .sum::<u64>()
            + (self.fields.capacity() * size_of::<Heightfield>()) as u64
    }
}

/// The convex body a grid is being asked about, and where it reaches.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Incoming<'a> {
    pub(crate) shape: &'a ColliderShape,
    pub(crate) pose: Pose,
    /// World bounds of the shape, already widened by the contact margin.
    pub(crate) bounds: Aabb,
}

/// Which bodies a grid's contacts belong to, and what the pair is made of.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FieldPair {
    pub(crate) a: u32,
    pub(crate) b: u32,
    /// Whether the grid is the second body, in which case every normal has to
    /// be turned round to keep pointing from the first toward the second.
    pub(crate) reversed: bool,
    pub(crate) friction: f32,
    pub(crate) restitution: f32,
}

/// Build the manifolds a convex shape makes with a grid, appending them to
/// `out`. Returns how many were added.
///
/// One manifold per triangle rather than one for the pair: each triangle has
/// its own plane, and folding two planes into one normal would push a body out
/// of a fold sideways.
pub(crate) fn collide_into(
    fields: &Heightfields,
    index: u32,
    incoming: Incoming<'_>,
    margin: f32,
    pair: FieldPair,
    out: &mut Vec<Manifold>,
) -> usize {
    let Some(field) = fields.get(index) else {
        return 0;
    };
    let mut kept = [(u32::MAX, Manifold::new(0, 0)); MAX_FIELD_MANIFOLDS];
    let mut count = 0usize;
    let mut shallowest = 0usize;

    let mut candidates = field.candidates(incoming.bounds);
    for (key, triangle) in &mut candidates {
        let mut found = [TriangleContact {
            point: Vec3::ZERO,
            separation: 0.0,
            feature: 0,
        }; MAX_TRIANGLE_POINTS];
        let found_count = contacts(
            &triangle,
            incoming.shape,
            incoming.pose,
            Vec3::ZERO,
            margin,
            &mut found,
        );
        if found_count == 0 {
            continue;
        }
        let manifold = manifold_for(&triangle, &found[..found_count], key, &pair);
        let depth = deepest(&manifold);
        if count < MAX_FIELD_MANIFOLDS {
            kept[count] = (key, manifold);
            count += 1;
            shallowest = shallowest_of(&kept[..count]);
        } else if depth < deepest(&kept[shallowest].1) {
            kept[shallowest] = (key, manifold);
            shallowest = shallowest_of(&kept[..count]);
        }
    }
    fields.note(&candidates);

    // Emitted in grid order whichever order they were found in, so the solve
    // sees the same list twice for the same scene.
    kept[..count].sort_unstable_by_key(|(key, _)| *key);
    for (_, manifold) in &kept[..count] {
        out.push(*manifold);
    }
    count
}

fn deepest(manifold: &Manifold) -> f32 {
    manifold
        .points()
        .iter()
        .fold(f32::INFINITY, |low, point| low.min(point.separation))
}

fn shallowest_of(kept: &[(u32, Manifold)]) -> usize {
    let mut at = 0usize;
    for (index, (_, manifold)) in kept.iter().enumerate() {
        if deepest(manifold) > deepest(&kept[at].1) {
            at = index;
        }
    }
    at
}

/// Turn one triangle's contacts into a manifold for the pair.
fn manifold_for(
    triangle: &Triangle,
    found: &[TriangleContact],
    key: u32,
    pair: &FieldPair,
) -> Manifold {
    let mut manifold = Manifold::new(pair.a, pair.b);
    manifold.normal = if pair.reversed {
        -triangle.normal
    } else {
        triangle.normal
    };
    manifold.friction = pair.friction;
    manifold.restitution = pair.restitution;

    let mut points = [Vec3::ZERO; MAX_TRIANGLE_POINTS];
    let mut separations = [0.0f32; MAX_TRIANGLE_POINTS];
    for (index, contact) in found.iter().enumerate() {
        points[index] = contact.point;
        separations[index] = contact.separation;
    }
    let mut keep = [0usize; 4];
    let kept = reduce_to_quad(
        &points[..found.len()],
        &separations[..found.len()],
        triangle.normal,
        &mut keep,
    );
    for &index in &keep[..kept] {
        let contact = found[index];
        manifold.push(ManifoldPoint {
            // Midway between the two surfaces, the same as every other pair.
            point: contact.point - triangle.normal * (contact.separation * 0.5),
            separation: contact.separation,
            id: (key << 6) | contact.feature,
            normal_impulse: 0.0,
            tangent_impulse: [0.0; 2],
        });
    }
    manifold
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::sim::math::Quat;
    use alloc::vec;

    /// A flat grid two cells square, centred on the origin, four units wide.
    fn flat() -> Heightfield {
        Heightfield::new(3, 3, vec![0.0; 9], vec3(4.0, 1.0, 4.0), Vec3::ZERO).expect("a real grid")
    }

    /// A grid rising along `+x`: every column is a step higher.
    fn ramp() -> Heightfield {
        let heights = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        Heightfield::new(3, 3, heights, vec3(4.0, 1.0, 4.0), Vec3::ZERO).expect("a real grid")
    }

    fn at(position: Vec3) -> Pose {
        Pose {
            position,
            rotation: Quat::IDENTITY,
        }
    }

    #[test]
    fn a_grid_that_names_no_surface_is_refused() {
        assert!(Heightfield::new(1, 3, vec![0.0; 3], Vec3::splat(1.0), Vec3::ZERO).is_none());
        assert!(Heightfield::new(3, 3, vec![0.0; 8], Vec3::splat(1.0), Vec3::ZERO).is_none());
        assert!(Heightfield::new(3, 3, vec![0.0; 9], vec3(0.0, 1.0, 1.0), Vec3::ZERO).is_none());
        assert!(
            Heightfield::new(
                2,
                2,
                vec![0.0, f32::NAN, 0.0, 0.0],
                Vec3::splat(1.0),
                Vec3::ZERO
            )
            .is_none()
        );
    }

    // Rows run along z and columns along x, and the grid is centred on its
    // origin: everything downstream reads the surface through this.
    #[test]
    fn the_grid_spans_its_footprint_with_rows_along_z_and_columns_along_x() {
        let field = ramp();
        assert_eq!(field.vertex(0, 0), vec3(-2.0, 0.0, -2.0));
        assert_eq!(field.vertex(0, 2), vec3(2.0, 2.0, -2.0));
        assert_eq!(field.vertex(2, 0), vec3(-2.0, 0.0, 2.0));
        assert_eq!(field.bounds().min, vec3(-2.0, 0.0, -2.0));
        assert_eq!(field.bounds().max, vec3(2.0, 2.0, 2.0));
        assert!(field.reserved_bytes() > 0);
    }

    #[test]
    fn a_cell_hands_back_two_triangles_that_both_face_up() {
        let field = ramp();
        let [first, second] = field.triangles_at(0, 0);
        for triangle in [first.expect("a triangle"), second.expect("a triangle")] {
            assert!(triangle.normal.y > 0.0, "{triangle:?}");
            assert!((triangle.normal.length() - 1.0).abs() < 1.0e-5);
        }
        // Past the last cell there is nothing.
        assert_eq!(field.triangles_at(2, 0), [None, None]);
        assert_eq!(field.triangles_at(0, 2), [None, None]);
        // Every triangle of the grid is named by its own key.
        let mut keys = alloc::vec::Vec::new();
        for row in 0..2 {
            for col in 0..2 {
                for half in 0..2 {
                    keys.push(field.triangle_key(row, col, half));
                }
            }
        }
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), 8);
    }

    #[test]
    fn a_box_names_the_cells_it_covers_and_nothing_outside_the_grid() {
        let field = flat();
        let over_one_cell = field
            .cell_range(Aabb {
                min: vec3(-1.5, -1.0, -1.5),
                max: vec3(-1.0, 1.0, -1.0),
            })
            .expect("inside the grid");
        assert_eq!(
            over_one_cell,
            CellRange {
                row0: 0,
                row1: 0,
                col0: 0,
                col1: 0
            }
        );
        let over_all = field
            .cell_range(Aabb {
                min: Vec3::splat(-10.0),
                max: Vec3::splat(10.0),
            })
            .expect("covering the grid");
        assert_eq!(
            over_all,
            CellRange {
                row0: 0,
                row1: 1,
                col0: 0,
                col1: 1
            }
        );
        assert!(
            field
                .cell_range(Aabb {
                    min: vec3(20.0, 0.0, 20.0),
                    max: vec3(21.0, 1.0, 21.0),
                })
                .is_none()
        );
    }

    #[test]
    fn a_walk_over_the_whole_grid_visits_every_triangle_once() {
        let field = flat();
        let all = Aabb {
            min: Vec3::splat(-10.0),
            max: Vec3::splat(10.0),
        };
        let mut candidates = field.candidates(all);
        let keys: alloc::vec::Vec<u32> = (&mut candidates).map(|(key, _)| key).collect();
        assert!(!candidates.cut_off());
        assert_eq!(keys.len(), 8);
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 8, "{keys:?}");
    }

    // The cap has to be reported rather than hidden: a query that gave up on
    // part of the surface answered a different question.
    #[test]
    fn a_walk_past_the_cap_stops_and_says_so() {
        let side = 40;
        let field = Heightfield::new(
            side,
            side,
            vec![0.0; side * side],
            vec3(100.0, 1.0, 100.0),
            Vec3::ZERO,
        )
        .expect("a real grid");
        let all = Aabb {
            min: Vec3::splat(-100.0),
            max: Vec3::splat(100.0),
        };
        let mut candidates = field.candidates(all);
        let seen = (&mut candidates).count();
        assert_eq!(seen, MAX_CANDIDATE_TRIANGLES);
        assert!(candidates.cut_off(), "the cap has to be surfaced");

        let table = Heightfields::new();
        assert_eq!(table.overflows(), 0);
        table.note(&candidates);
        assert_eq!(table.overflows(), 1);
        table.clear_overflows();
        assert_eq!(table.overflows(), 0);
    }

    #[test]
    fn the_table_hands_a_grid_back_by_the_index_it_was_given() {
        let mut table = Heightfields::new();
        let index = table.push(flat());
        assert_eq!(index, 0);
        assert!(table.get(0).is_some());
        assert!(table.get(1).is_none());
        assert!(table.reserved_bytes() > 0);
    }

    // The resting case, and the one that has to be right before anything else
    // is: a box over flat terrain holds on the whole face it sits on.
    #[test]
    fn a_box_resting_on_flat_terrain_contacts_through_its_lowest_face() {
        let mut table = Heightfields::new();
        let index = table.push(flat());
        let cube = ColliderShape::Cuboid {
            half_extents: [0.5, 0.5, 0.5],
        };
        let pose = at(vec3(-1.0, 0.5, -1.0));
        let bounds = Aabb::from_center_half_extents(pose.position, Vec3::splat(0.5));
        let mut out = alloc::vec::Vec::new();
        let manifolds = collide_into(
            &table,
            index,
            Incoming {
                shape: &cube,
                pose,
                bounds,
            },
            0.02,
            FieldPair {
                a: 0,
                b: 1,
                reversed: false,
                friction: 0.5,
                restitution: 0.0,
            },
            &mut out,
        );
        assert!(manifolds > 0, "the box has to find the ground");
        let mut points = 0usize;
        for manifold in &out {
            assert!(manifold.normal.y > 0.99, "{:?}", manifold.normal);
            assert_eq!((manifold.a, manifold.b), (0, 1));
            for point in manifold.points() {
                assert!(point.separation.abs() < 1.0e-4, "{point:?}");
                points += 1;
            }
        }
        assert!(points >= 4, "a face has four corners: {points}");
    }

    // Whichever way round the pair arrives, the normal points from the first
    // body toward the second.
    #[test]
    fn a_reversed_pair_turns_every_normal_round() {
        let mut table = Heightfields::new();
        let index = table.push(flat());
        let ball = ColliderShape::Ball { radius: 0.5 };
        let pose = at(vec3(-1.0, 0.4, -1.0));
        let bounds = Aabb::from_center_half_extents(pose.position, Vec3::splat(0.5));
        let mut out = alloc::vec::Vec::new();
        collide_into(
            &table,
            index,
            Incoming {
                shape: &ball,
                pose,
                bounds,
            },
            0.02,
            FieldPair {
                a: 3,
                b: 4,
                reversed: true,
                friction: 0.5,
                restitution: 0.0,
            },
            &mut out,
        );
        assert!(!out.is_empty());
        for manifold in &out {
            assert!(manifold.normal.y < -0.99, "{:?}", manifold.normal);
            assert!(manifold.points().iter().all(|p| p.separation < 0.0));
        }
    }

    #[test]
    fn a_shape_clear_of_the_terrain_makes_no_manifold() {
        let mut table = Heightfields::new();
        let index = table.push(flat());
        let ball = ColliderShape::Ball { radius: 0.5 };
        let pose = at(vec3(0.0, 5.0, 0.0));
        let bounds = Aabb::from_center_half_extents(pose.position, Vec3::splat(0.5));
        let mut out = alloc::vec::Vec::new();
        assert_eq!(
            collide_into(
                &table,
                index,
                Incoming {
                    shape: &ball,
                    pose,
                    bounds,
                },
                0.02,
                FieldPair {
                    a: 0,
                    b: 1,
                    reversed: false,
                    friction: 0.5,
                    restitution: 0.0
                },
                &mut out,
            ),
            0
        );
        assert!(out.is_empty());
    }

    // A slope's contact normal has to lean, or nothing ever rolls downhill.
    #[test]
    fn a_ball_on_a_slope_is_pushed_along_the_slopes_own_normal() {
        let mut table = Heightfields::new();
        let index = table.push(ramp());
        let ball = ColliderShape::Ball { radius: 0.3 };
        // The ramp rises one unit per two along x, so the surface under
        // (-1, .., -1) sits at y = 0.5.
        let pose = at(vec3(-1.0, 0.75, -1.0));
        let bounds = Aabb::from_center_half_extents(pose.position, Vec3::splat(0.3));
        let mut out = alloc::vec::Vec::new();
        assert!(
            collide_into(
                &table,
                index,
                Incoming {
                    shape: &ball,
                    pose,
                    bounds,
                },
                0.05,
                FieldPair {
                    a: 0,
                    b: 1,
                    reversed: false,
                    friction: 0.5,
                    restitution: 0.0
                },
                &mut out,
            ) > 0
        );
        for manifold in &out {
            assert!(manifold.normal.y > 0.5, "{:?}", manifold.normal);
            assert!(
                manifold.normal.x < -0.3,
                "the normal has to lean downhill: {:?}",
                manifold.normal
            );
        }
    }
}
