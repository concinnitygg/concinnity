// The point of a simplex closest to the origin, and which of its vertices
// carry that point.
//
// This is the inner loop of the distance query, split out because it is pure
// geometry over at most four points and can be checked against answers worked
// out by hand. It returns barycentric weights rather than just a point, which
// is what lets the caller carry the answer back to witness points on the two
// original shapes.
//
// The tetrahedron case reports the origin as enclosed only when the simplex
// has volume to enclose it with. A flat simplex has no inside, so its hull is
// the union of its faces and every face is examined.

use crate::physics::sim::math::Vec3;

/// The closest point of a simplex to the origin, and the vertices behind it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Closest {
    /// Vector from the origin to the closest point.
    pub(crate) point: Vec3,
    /// Indices of the input vertices that carry the point.
    pub(crate) keep: [usize; 4],
    /// Their barycentric weights, aligned with `keep`.
    pub(crate) weights: [f32; 4],
    /// How many of `keep` and `weights` are meaningful.
    pub(crate) count: usize,
    /// Whether the origin lies inside the simplex, which only a tetrahedron
    /// with volume can report.
    pub(crate) encloses_origin: bool,
}

impl Closest {
    fn of(point: Vec3, keep: &[usize], weights: &[f32]) -> Self {
        let mut out = Closest {
            point,
            keep: [0; 4],
            weights: [0.0; 4],
            count: keep.len(),
            encloses_origin: false,
        };
        out.keep[..keep.len()].copy_from_slice(keep);
        out.weights[..weights.len()].copy_from_slice(weights);
        out
    }
}

/// A tetrahedron flatter than this has no interior to enclose the origin with.
const FLAT_VOLUME: f32 = 1.0e-12;

/// The point of `points`' convex hull closest to the origin.
///
/// `points` holds one to four vertices; more than four is a caller error and
/// only the first four are read.
pub(crate) fn closest_to_origin(points: &[Vec3]) -> Closest {
    match points {
        [] => Closest::of(Vec3::ZERO, &[], &[]),
        [a] => Closest::of(*a, &[0], &[1.0]),
        [a, b] => segment(*a, *b),
        [a, b, c] => triangle([*a, *b, *c], [0, 1, 2]),
        [a, b, c, d, ..] => tetrahedron([*a, *b, *c, *d]),
    }
}

fn segment(a: Vec3, b: Vec3) -> Closest {
    let ab = b - a;
    let length_squared = ab.length_squared();
    if length_squared <= f32::MIN_POSITIVE {
        return Closest::of(a, &[0], &[1.0]);
    }
    let t = (-a.dot(ab)) / length_squared;
    if t <= 0.0 {
        return Closest::of(a, &[0], &[1.0]);
    }
    if t >= 1.0 {
        return Closest::of(b, &[1], &[1.0]);
    }
    Closest::of(a + ab * t, &[0, 1], &[1.0 - t, t])
}

/// The Voronoi-region walk over a triangle: each region names the feature
/// that carries the closest point, so the answer arrives already reduced.
///
/// `remap` gives the caller's index for each of the triangle's own vertices,
/// which is what lets the tetrahedron case reuse this for its faces.
fn triangle(v: [Vec3; 3], remap: [usize; 3]) -> Closest {
    let [a, b, c] = v;
    let (ab, ac) = (b - a, c - a);

    let d1 = ab.dot(-a);
    let d2 = ac.dot(-a);
    if d1 <= 0.0 && d2 <= 0.0 {
        return Closest::of(a, &[remap[0]], &[1.0]);
    }

    let d3 = ab.dot(-b);
    let d4 = ac.dot(-b);
    if d3 >= 0.0 && d4 <= d3 {
        return Closest::of(b, &[remap[1]], &[1.0]);
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let t = d1 / (d1 - d3);
        return Closest::of(a + ab * t, &[remap[0], remap[1]], &[1.0 - t, t]);
    }

    let d5 = ab.dot(-c);
    let d6 = ac.dot(-c);
    if d6 >= 0.0 && d5 <= d6 {
        return Closest::of(c, &[remap[2]], &[1.0]);
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let t = d2 / (d2 - d6);
        return Closest::of(a + ac * t, &[remap[0], remap[2]], &[1.0 - t, t]);
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return Closest::of(b + (c - b) * t, &[remap[1], remap[2]], &[1.0 - t, t]);
    }

    let total = va + vb + vc;
    if total <= f32::MIN_POSITIVE {
        // Degenerate: the three points are collinear, so the longest edge
        // carries whatever the hull has.
        return segment_of(v, remap);
    }
    let inv = 1.0 / total;
    let (beta, gamma) = (vb * inv, vc * inv);
    Closest::of(
        a + ab * beta + ac * gamma,
        &remap,
        &[1.0 - beta - gamma, beta, gamma],
    )
}

/// The best of a collinear triangle's three edges.
fn segment_of(v: [Vec3; 3], remap: [usize; 3]) -> Closest {
    let mut best: Option<Closest> = None;
    for (i, j) in [(0, 1), (0, 2), (1, 2)] {
        let mut edge = segment(v[i], v[j]);
        for slot in &mut edge.keep[..edge.count] {
            *slot = remap[[i, j][*slot]];
        }
        if best.is_none_or(|current| edge.point.length_squared() < current.point.length_squared()) {
            best = Some(edge);
        }
    }
    best.unwrap_or_else(|| Closest::of(v[0], &[remap[0]], &[1.0]))
}

/// The four triangular faces of a tetrahedron, each with the vertex opposite
/// it, wound so the opposite vertex is on the inside of the face's plane.
const FACES: [([usize; 3], usize); 4] = [
    ([0, 1, 2], 3),
    ([0, 2, 3], 1),
    ([0, 3, 1], 2),
    ([1, 3, 2], 0),
];

fn tetrahedron(v: [Vec3; 4]) -> Closest {
    let volume = (v[1] - v[0]).dot((v[2] - v[0]).cross(v[3] - v[0]));

    let mut best: Option<Closest> = None;
    let mut outside_any = false;
    for (face, opposite) in FACES {
        let [a, b, c] = [v[face[0]], v[face[1]], v[face[2]]];
        let normal = (b - a).cross(c - a);
        // The origin is outside this face when it and the remaining vertex
        // fall on opposite sides of the face's plane.
        if (-a).dot(normal) * (v[opposite] - a).dot(normal) >= 0.0 {
            continue;
        }
        outside_any = true;
        keep_nearest(&mut best, triangle([a, b, c], face));
    }

    if let Some(found) = best {
        return found;
    }
    if outside_any || volume.abs() > FLAT_VOLUME {
        let mut enclosed = Closest::of(Vec3::ZERO, &[0, 1, 2, 3], &[0.0; 4]);
        enclosed.encloses_origin = true;
        return enclosed;
    }

    // Flat: no inside to be in, so the hull is the faces and all four count.
    for (face, _) in FACES {
        let [a, b, c] = [v[face[0]], v[face[1]], v[face[2]]];
        keep_nearest(&mut best, triangle([a, b, c], face));
    }
    best.unwrap_or_else(|| Closest::of(v[0], &[0], &[1.0]))
}

fn keep_nearest(best: &mut Option<Closest>, candidate: Closest) {
    if best.is_none_or(|current| candidate.point.length_squared() < current.point.length_squared())
    {
        *best = Some(candidate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::sim::math::vec3;

    fn recombine(points: &[Vec3], closest: &Closest) -> Vec3 {
        let mut sum = Vec3::ZERO;
        for i in 0..closest.count {
            sum += points[closest.keep[i]] * closest.weights[i];
        }
        sum
    }

    fn check(points: &[Vec3], expected: Vec3) -> Closest {
        let closest = closest_to_origin(points);
        assert!(
            (closest.point - expected).length() < 1.0e-5,
            "{points:?} -> {:?}, wanted {expected:?}",
            closest.point
        );
        // The weights have to reproduce the point, or the witness points the
        // caller builds from them land somewhere else entirely.
        assert!(
            (recombine(points, &closest) - closest.point).length() < 1.0e-5,
            "weights do not rebuild the point: {closest:?}"
        );
        let sum: f32 = closest.weights[..closest.count].iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-5, "{closest:?}");
        closest
    }

    #[test]
    fn a_lone_point_is_its_own_answer() {
        let closest = check(&[vec3(1.0, 2.0, 3.0)], vec3(1.0, 2.0, 3.0));
        assert_eq!(closest.count, 1);
        assert!(!closest.encloses_origin);
    }

    #[test]
    fn a_segment_answers_from_its_interior_or_from_an_end() {
        // Straddling the origin: the foot of the perpendicular.
        let closest = check(
            &[vec3(-1.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0)],
            vec3(0.0, 1.0, 0.0),
        );
        assert_eq!(closest.count, 2);
        // Entirely to one side: the nearer end, and only that end.
        let end = check(
            &[vec3(1.0, 0.0, 0.0), vec3(3.0, 0.0, 0.0)],
            vec3(1.0, 0.0, 0.0),
        );
        assert_eq!(end.count, 1);
        assert_eq!(end.keep[0], 0);
    }

    #[test]
    fn a_degenerate_segment_collapses_to_its_endpoint() {
        let p = vec3(2.0, 0.0, 0.0);
        let closest = check(&[p, p], p);
        assert_eq!(closest.count, 1);
    }

    #[test]
    fn a_triangle_answers_from_its_face_when_the_origin_is_over_it() {
        let closest = check(
            &[
                vec3(-1.0, 2.0, -1.0),
                vec3(1.0, 2.0, -1.0),
                vec3(0.0, 2.0, 1.0),
            ],
            vec3(0.0, 2.0, 0.0),
        );
        assert_eq!(closest.count, 3, "the whole face carries it");
    }

    #[test]
    fn a_triangle_answers_from_an_edge_or_a_corner_when_the_origin_is_past_it() {
        // Past the edge between the first two vertices.
        let edge = check(
            &[
                vec3(-1.0, 1.0, 1.0),
                vec3(1.0, 1.0, 1.0),
                vec3(0.0, 1.0, 3.0),
            ],
            vec3(0.0, 1.0, 1.0),
        );
        assert_eq!(edge.count, 2);
        // Past a corner.
        let corner = check(
            &[
                vec3(1.0, 1.0, 1.0),
                vec3(3.0, 1.0, 1.0),
                vec3(1.0, 1.0, 3.0),
            ],
            vec3(1.0, 1.0, 1.0),
        );
        assert_eq!(corner.count, 1);
        assert_eq!(corner.keep[0], 0);
    }

    #[test]
    fn a_collinear_triangle_answers_from_its_longest_reach() {
        let closest = check(
            &[
                vec3(-2.0, 1.0, 0.0),
                vec3(0.0, 1.0, 0.0),
                vec3(2.0, 1.0, 0.0),
            ],
            vec3(0.0, 1.0, 0.0),
        );
        assert!(!closest.encloses_origin);
    }

    #[test]
    fn a_tetrahedron_around_the_origin_reports_it_enclosed() {
        let closest = closest_to_origin(&[
            vec3(1.0, 1.0, 1.0),
            vec3(-1.0, -1.0, 1.0),
            vec3(-1.0, 1.0, -1.0),
            vec3(1.0, -1.0, -1.0),
        ]);
        assert!(closest.encloses_origin);
        assert_eq!(closest.point, Vec3::ZERO);
    }

    #[test]
    fn a_tetrahedron_beside_the_origin_answers_from_its_nearest_feature() {
        // Lifted clear of the origin along y: the bottom face carries it.
        let closest = check(
            &[
                vec3(-1.0, 2.0, -1.0),
                vec3(1.0, 2.0, -1.0),
                vec3(0.0, 2.0, 1.0),
                vec3(0.0, 4.0, 0.0),
            ],
            vec3(0.0, 2.0, 0.0),
        );
        assert!(!closest.encloses_origin);
        assert_eq!(closest.count, 3);
    }

    // A support point that lands on the plane of the existing triangle gives
    // a flat four-point simplex, which has no inside to report.
    #[test]
    fn a_flat_tetrahedron_never_claims_to_enclose_the_origin() {
        let closest = check(
            &[
                vec3(-1.0, 1.0, -1.0),
                vec3(1.0, 1.0, -1.0),
                vec3(1.0, 1.0, 1.0),
                vec3(-1.0, 1.0, 1.0),
            ],
            vec3(0.0, 1.0, 0.0),
        );
        assert!(!closest.encloses_origin);
    }

    #[test]
    fn a_fully_degenerate_tetrahedron_still_answers() {
        let p = vec3(0.0, 3.0, 0.0);
        let closest = check(&[p, p, p, p], p);
        assert!(!closest.encloses_origin);
    }

    // The same points must give the same answer whatever order they arrive
    // in, up to which vertices are named: a query that depended on argument
    // order would not be reproducible.
    #[test]
    fn the_answer_does_not_depend_on_the_order_the_vertices_arrive_in() {
        let v = [
            vec3(-1.0, 2.0, -1.0),
            vec3(1.0, 2.0, -1.0),
            vec3(0.0, 2.0, 1.0),
            vec3(0.0, 4.0, 0.5),
        ];
        let base = closest_to_origin(&v).point;
        for rotation in 1..4 {
            let rotated: [Vec3; 4] = core::array::from_fn(|i| v[(i + rotation) % 4]);
            let point = closest_to_origin(&rotated).point;
            assert!((point - base).length() < 1.0e-5, "{rotation}: {point:?}");
        }
    }

    // Each of a triangle's three vertex regions has to answer with that
    // vertex alone. A and B are covered above; C is the arm the walk reaches
    // last, and a triangle placed so the origin sits past it must reduce to
    // one point rather than carrying an edge the caller would then search.
    #[test]
    fn a_triangle_reduces_to_the_vertex_the_origin_sits_past() {
        // C is the near corner; the origin lies beyond it, outside both
        // edges that meet there.
        let a = vec3(3.0, 2.0, 0.0);
        let b = vec3(3.0, -2.0, 0.0);
        let c = vec3(1.0, 0.0, 0.0);
        let closest = check(&[a, b, c], c);
        assert_eq!(closest.count, 1, "{closest:?}");
        assert_eq!(closest.keep[0], 2, "the answer is C's own index");
    }

    // The edge between the two far vertices: the origin is outside the
    // triangle across BC, so the answer is the foot of the perpendicular on
    // that edge and the two vertices carrying it.
    #[test]
    fn a_triangle_reduces_to_the_edge_the_origin_faces() {
        let a = vec3(1.0, 0.0, 2.0);
        let b = vec3(1.0, -1.0, 0.0);
        let c = vec3(1.0, 1.0, 0.0);
        let closest = check(&[a, b, c], vec3(1.0, 0.0, 0.0));
        assert_eq!(closest.count, 2, "{closest:?}");
        let mut kept = [closest.keep[0], closest.keep[1]];
        kept.sort();
        assert_eq!(kept, [1, 2], "B and C carry the closest point");
    }

    // Three collinear points enclose no area, so the barycentric split has
    // nothing to divide by. The longest edge carries whatever the hull has,
    // and the answer still has to rebuild from its weights.
    #[test]
    fn a_collinear_triangle_answers_from_an_edge() {
        let closest = check(
            &[
                vec3(1.0, -2.0, 0.0),
                vec3(1.0, 0.0, 0.0),
                vec3(1.0, 2.0, 0.0),
            ],
            vec3(1.0, 0.0, 0.0),
        );
        assert!(closest.count >= 1, "{closest:?}");
        assert!(!closest.encloses_origin);
    }

    // A triangle with area small enough that the barycentric denominators
    // underflow: the region guards do not fire, and the split has nothing to
    // divide by. Exactly-collinear points never get this far -- one of the
    // vertex or edge regions always claims them first -- so this is the shape
    // the fallback actually exists for. The answer still has to be finite and
    // rebuild from its weights.
    #[test]
    fn a_triangle_too_thin_to_divide_by_still_answers() {
        let closest = closest_to_origin(&[
            vec3(1.0, -2.0, 0.0),
            vec3(1.0, 2.0, 0.0),
            vec3(1.0, 0.0, 1.0e-24),
        ]);
        assert!(
            closest.point.x.is_finite()
                && closest.point.y.is_finite()
                && closest.point.z.is_finite(),
            "{closest:?}"
        );
        assert!(closest.count >= 1, "{closest:?}");
        let sum: f32 = closest.weights[..closest.count].iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-5, "{closest:?}");
    }

    // Collinear and coincident at once: every edge is degenerate, so the
    // walk falls all the way back to a single vertex.
    #[test]
    fn a_triangle_of_one_repeated_point_collapses_to_that_point() {
        let p = vec3(2.0, 1.0, 0.0);
        let closest = check(&[p, p, p], p);
        assert_eq!(closest.count, 1, "{closest:?}");
    }
}
