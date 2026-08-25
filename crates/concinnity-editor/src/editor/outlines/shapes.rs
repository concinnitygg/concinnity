// src/editor/outlines/shapes.rs
//
// Pure line-segment generators for the extent outlines: each function appends
// a shape's wireframe to a caller-owned buffer, so a frame's outlines build
// into one reused allocation. All shapes are world-space `Line`s for the
// renderer's line pass; oriented shapes take a column-major model matrix
// (`Transform::model_matrix` convention, `m[col][row]`).

use concinnity_cpu::geometry::glass_quad::plane_basis;
use concinnity_cpu::gfx::lines::Line;

// Full-circle resolution. High enough that a large range sphere still reads
// as a curve, cheap enough to ignore (a circle is 32 segments).
pub(crate) const CIRCLE_SEGMENTS: usize = 32;
// Capsule cap arcs are semicircles at half the circle resolution.
pub(crate) const ARC_SEGMENTS: usize = 16;
// Edges of a box / frustum wireframe.
pub(crate) const BOX_EDGES: usize = 12;
// Side lines connecting a cone's apex to its base circle.
pub(crate) const CONE_SIDES: usize = 4;

// Per-shape draw style: one colour and pixel width for every segment.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct Stroke {
    pub color: [f32; 4],
    pub width_px: f32,
}

// Box corners as sign masks (bit 0 = +x, bit 1 = +y, bit 2 = +z) and the 12
// edges as corner-index pairs. Shared with the billboard drag ghost's dotted
// screen-space outline.
pub(crate) const EDGES: [(usize, usize); BOX_EDGES] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn transform_point(m: &[[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
        m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
        m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2],
    ]
}

fn seg(out: &mut Vec<Line>, a: [f32; 3], b: [f32; 3], s: Stroke) {
    out.push(Line {
        start: a,
        end: b,
        start_color: s.color,
        end_color: s.color,
        width_px: s.width_px,
    });
}

// A closed polyline through `points` (last connects back to first).
fn closed_polyline(out: &mut Vec<Line>, points: &[[f32; 3]], s: Stroke) {
    for i in 0..points.len() {
        seg(out, points[i], points[(i + 1) % points.len()], s);
    }
}

// The point at angle `theta` on the circle spanned by unit axes `u`, `v`.
fn on_circle(center: [f32; 3], u: [f32; 3], v: [f32; 3], radius: f32, theta: f32) -> [f32; 3] {
    add(
        center,
        add(
            scale(u, radius * theta.cos()),
            scale(v, radius * theta.sin()),
        ),
    )
}

// A closed ring: `point_at(theta)` sampled over a full turn.
fn push_ring(out: &mut Vec<Line>, s: Stroke, point_at: impl Fn(f32) -> [f32; 3]) {
    let mut points = [[0.0f32; 3]; CIRCLE_SEGMENTS];
    for (i, p) in points.iter_mut().enumerate() {
        *p = point_at(i as f32 / CIRCLE_SEGMENTS as f32 * core::f32::consts::TAU);
    }
    closed_polyline(out, &points, s);
}

// A circle of `radius` around `center` in the plane perpendicular to `normal`.
// A non-positive radius appends nothing.
pub(crate) fn push_circle(
    out: &mut Vec<Line>,
    center: [f32; 3],
    normal: [f32; 3],
    radius: f32,
    s: Stroke,
) {
    if radius <= 0.0 {
        return;
    }
    let (u, v) = plane_basis(normal);
    push_ring(out, s, |theta| on_circle(center, u, v, radius, theta));
}

// A sphere as its three axis-aligned great circles.
pub(crate) fn push_sphere(out: &mut Vec<Line>, center: [f32; 3], radius: f32, s: Stroke) {
    for normal in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
        push_circle(out, center, normal, radius, s);
    }
}

// The world-space corners of an oriented box, indexed by the `EDGES` sign
// masks. Shared with the billboard ghost's screen-space projection.
pub(crate) fn box_corners(model: &[[f32; 4]; 4], half_extents: [f32; 3]) -> [[f32; 3]; 8] {
    let mut corners = [[0.0f32; 3]; 8];
    for (i, corner) in corners.iter_mut().enumerate() {
        let local = [
            if i & 1 == 0 {
                -half_extents[0]
            } else {
                half_extents[0]
            },
            if i & 2 == 0 {
                -half_extents[1]
            } else {
                half_extents[1]
            },
            if i & 4 == 0 {
                -half_extents[2]
            } else {
                half_extents[2]
            },
        ];
        *corner = transform_point(model, local);
    }
    corners
}

// An oriented box: the 12 edges of `half_extents` through `model`.
pub(crate) fn push_box(
    out: &mut Vec<Line>,
    model: &[[f32; 4]; 4],
    half_extents: [f32; 3],
    s: Stroke,
) {
    let corners = box_corners(model, half_extents);
    for (a, b) in EDGES {
        seg(out, corners[a], corners[b], s);
    }
}

// A local-space circle pushed through `model`: cap rings of the capsule.
fn push_local_circle(
    out: &mut Vec<Line>,
    model: &[[f32; 4]; 4],
    center_y: f32,
    radius: f32,
    s: Stroke,
) {
    push_ring(out, s, |theta| {
        transform_point(
            model,
            [radius * theta.cos(), center_y, radius * theta.sin()],
        )
    });
}

// A local-space semicircular cap arc in the plane spanned by local `axis`
// (x or z) and y, centered at local (0, `center_y`, 0), pushed through `model`.
// `up` picks the top (+y) or bottom (-y) half.
fn push_cap_arc(
    out: &mut Vec<Line>,
    model: &[[f32; 4]; 4],
    axis: usize,
    center_y: f32,
    radius: f32,
    up: bool,
    s: Stroke,
) {
    let mut prev: Option<[f32; 3]> = None;
    for i in 0..=ARC_SEGMENTS {
        let theta = i as f32 / ARC_SEGMENTS as f32 * core::f32::consts::PI;
        let y_off = if up { theta.sin() } else { -theta.sin() };
        let mut local = [0.0f32; 3];
        local[axis] = radius * theta.cos();
        local[1] = center_y + radius * y_off;
        let p = transform_point(model, local);
        if let Some(prev) = prev {
            seg(out, prev, p, s);
        }
        prev = Some(p);
    }
}

// A Y-axis capsule (`radius`, cylinder `half_height`, hemispherical caps)
// through `model`: two cap rings, four side lines, four cap arcs. A
// non-positive radius appends nothing.
pub(crate) fn push_capsule(
    out: &mut Vec<Line>,
    model: &[[f32; 4]; 4],
    radius: f32,
    half_height: f32,
    s: Stroke,
) {
    if radius <= 0.0 {
        return;
    }
    let h = half_height.max(0.0);
    push_local_circle(out, model, h, radius, s);
    push_local_circle(out, model, -h, radius, s);
    for local_dir in [[radius, 0.0], [-radius, 0.0], [0.0, radius], [0.0, -radius]] {
        let top = transform_point(model, [local_dir[0], h, local_dir[1]]);
        let bottom = transform_point(model, [local_dir[0], -h, local_dir[1]]);
        seg(out, top, bottom, s);
    }
    for axis in [0usize, 2] {
        push_cap_arc(out, model, axis, h, radius, true, s);
        push_cap_arc(out, model, axis, -h, radius, false, s);
    }
}

// A spot cone: apex at `apex`, opening along unit `dir` with half-angle
// `half_angle_rad`, slant length `range` (the light's attenuation reach), so
// the base circle sits on the range sphere. Non-positive range or angle
// appends nothing.
pub(crate) fn push_cone(
    out: &mut Vec<Line>,
    apex: [f32; 3],
    dir: [f32; 3],
    half_angle_rad: f32,
    range: f32,
    s: Stroke,
) {
    if range <= 0.0 || half_angle_rad <= 0.0 {
        return;
    }
    let base_center = add(apex, scale(dir, range * half_angle_rad.cos()));
    let base_radius = range * half_angle_rad.sin();
    let (u, v) = plane_basis(dir);
    for i in 0..CONE_SIDES {
        let theta = i as f32 / CONE_SIDES as f32 * core::f32::consts::TAU;
        seg(
            out,
            apex,
            on_circle(base_center, u, v, base_radius, theta),
            s,
        );
    }
    push_circle(out, base_center, dir, base_radius, s);
}

// A camera frustum: its apex pose (an orthonormal world basis) and projection
// parameters.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Frustum {
    pub origin: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub forward: [f32; 3],
    pub fov_y_rad: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

// A camera frustum wireframe: near rect, far rect, and the four connecting
// edges. Appends nothing for a degenerate projection (`far <= near` or a
// non-positive fov / aspect).
pub(crate) fn push_frustum(out: &mut Vec<Line>, f: &Frustum, s: Stroke) {
    let tan_half = (f.fov_y_rad * 0.5).tan();
    if f.far <= f.near
        || f.near <= 0.0
        || tan_half <= 0.0
        || !tan_half.is_finite()
        || f.aspect <= 0.0
    {
        return;
    }
    let corners = |d: f32| -> [[f32; 3]; 4] {
        let hh = d * tan_half;
        let hw = hh * f.aspect;
        let center = add(f.origin, scale(f.forward, d));
        [
            add(center, add(scale(f.right, -hw), scale(f.up, -hh))),
            add(center, add(scale(f.right, hw), scale(f.up, -hh))),
            add(center, add(scale(f.right, hw), scale(f.up, hh))),
            add(center, add(scale(f.right, -hw), scale(f.up, hh))),
        ]
    };
    let near = corners(f.near);
    let far = corners(f.far);
    closed_polyline(out, &near, s);
    closed_polyline(out, &far, s);
    for i in 0..4 {
        seg(out, near[i], far[i], s);
    }
}

// A rect-area light: the emitting rectangle plus a normal ray of length
// `range`. The tangent frame is `plane_basis`, the same one the renderer
// packs into `AreaLightData`, so the outline is the lit rectangle. Appends
// nothing for a degenerate half size.
pub(crate) fn push_rect(
    out: &mut Vec<Line>,
    centre: [f32; 3],
    normal: [f32; 3],
    half_size: [f32; 2],
    range: f32,
    s: Stroke,
) {
    if half_size[0] <= 0.0 || half_size[1] <= 0.0 {
        return;
    }
    let (tangent, bitangent) = plane_basis(normal);
    let corner = |su: f32, sv: f32| {
        add(
            centre,
            add(
                scale(tangent, su * half_size[0]),
                scale(bitangent, sv * half_size[1]),
            ),
        )
    };
    closed_polyline(
        out,
        &[
            corner(-1.0, -1.0),
            corner(1.0, -1.0),
            corner(1.0, 1.0),
            corner(-1.0, 1.0),
        ],
        s,
    );
    if range > 0.0 {
        seg(out, centre, add(centre, scale(normal, range)), s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STROKE: Stroke = Stroke {
        color: [1.0, 0.5, 0.0, 1.0],
        width_px: 2.0,
    };

    fn len3(v: [f32; 3]) -> f32 {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
        len3([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
    }

    fn identity() -> [[f32; 4]; 4] {
        crate::components::Transform::default().model_matrix()
    }

    // Every endpoint of every emitted segment, for on-surface assertions.
    fn endpoints(lines: &[Line]) -> Vec<[f32; 3]> {
        lines.iter().flat_map(|l| [l.start, l.end]).collect()
    }

    #[test]
    fn circle_points_sit_on_the_radius_in_the_plane() {
        let mut out = Vec::new();
        push_circle(&mut out, [1.0, 2.0, 3.0], [0.0, 1.0, 0.0], 5.0, STROKE);
        assert_eq!(out.len(), CIRCLE_SEGMENTS);
        for p in endpoints(&out) {
            assert!((dist(p, [1.0, 2.0, 3.0]) - 5.0).abs() < 1e-4, "{p:?}");
            assert!((p[1] - 2.0).abs() < 1e-5, "in the y-normal plane: {p:?}");
        }
        // The polyline closes: every endpoint is shared by two segments.
        assert_eq!(out[0].start, out[CIRCLE_SEGMENTS - 1].end);
    }

    #[test]
    fn degenerate_circle_and_sphere_emit_nothing() {
        let mut out = Vec::new();
        push_circle(&mut out, [0.0; 3], [0.0, 1.0, 0.0], 0.0, STROKE);
        push_circle(&mut out, [0.0; 3], [0.0, 1.0, 0.0], -1.0, STROKE);
        push_sphere(&mut out, [0.0; 3], 0.0, STROKE);
        assert!(out.is_empty());
    }

    #[test]
    fn sphere_is_three_great_circles_on_the_radius() {
        let mut out = Vec::new();
        push_sphere(&mut out, [0.0, 4.0, 0.0], 2.0, STROKE);
        assert_eq!(out.len(), 3 * CIRCLE_SEGMENTS);
        for p in endpoints(&out) {
            assert!((dist(p, [0.0, 4.0, 0.0]) - 2.0).abs() < 1e-4, "{p:?}");
        }
    }

    #[test]
    fn box_edges_span_the_transformed_extents() {
        let mut out = Vec::new();
        let model = crate::components::Transform {
            position: [10.0, 0.0, 0.0],
            rotation_deg: [0.0, 90.0, 0.0],
            scale: [1.0; 3],
        }
        .model_matrix();
        push_box(&mut out, &model, [1.0, 2.0, 3.0], STROKE);
        assert_eq!(out.len(), BOX_EDGES);
        // A 90-degree yaw swaps the x and z extents around the new center.
        for p in endpoints(&out) {
            assert!((p[0] - 10.0).abs() < 3.0 + 1e-3, "{p:?}");
            assert!(p[1].abs() < 2.0 + 1e-3, "{p:?}");
            assert!(p[2].abs() < 1.0 + 1e-3, "{p:?}");
        }
        let max_x = endpoints(&out)
            .iter()
            .map(|p| p[0])
            .fold(f32::MIN, f32::max);
        assert!(
            (max_x - 13.0).abs() < 1e-3,
            "yawed z extent reaches +13: {max_x}"
        );
    }

    #[test]
    fn capsule_covers_rings_sides_and_arcs() {
        let mut out = Vec::new();
        push_capsule(&mut out, &identity(), 0.5, 1.0, STROKE);
        assert_eq!(out.len(), 2 * CIRCLE_SEGMENTS + 4 + 4 * ARC_SEGMENTS);
        // Every point stays inside the capsule's bounding box and outside the
        // cylinder core's waist.
        for p in endpoints(&out) {
            assert!(p[1].abs() <= 1.5 + 1e-4, "{p:?}");
            assert!((p[0] * p[0] + p[2] * p[2]).sqrt() <= 0.5 + 1e-4, "{p:?}");
        }
        // The arcs reach the cap poles.
        let max_y = endpoints(&out)
            .iter()
            .map(|p| p[1])
            .fold(f32::MIN, f32::max);
        assert!((max_y - 1.5).abs() < 1e-4, "cap pole at h + r: {max_y}");
        let mut none = Vec::new();
        push_capsule(&mut none, &identity(), 0.0, 1.0, STROKE);
        assert!(none.is_empty());
    }

    #[test]
    fn cone_base_sits_on_the_range_sphere() {
        let mut out = Vec::new();
        let apex = [0.0, 5.0, 0.0];
        push_cone(
            &mut out,
            apex,
            [0.0, -1.0, 0.0],
            30f32.to_radians(),
            10.0,
            STROKE,
        );
        assert_eq!(out.len(), CONE_SIDES + CIRCLE_SEGMENTS);
        // Side lines start at the apex; base circle points are `range` from it.
        for l in &out[..CONE_SIDES] {
            assert_eq!(l.start, apex);
            assert!((dist(l.end, apex) - 10.0).abs() < 1e-3, "{:?}", l.end);
        }
        for l in &out[CONE_SIDES..] {
            assert!((dist(l.start, apex) - 10.0).abs() < 1e-3);
            // The base plane sits at range * cos(angle) below the apex.
            assert!((l.start[1] - (5.0 - 10.0 * 30f32.to_radians().cos())).abs() < 1e-3);
        }
        let mut none = Vec::new();
        push_cone(&mut none, apex, [0.0, -1.0, 0.0], 0.0, 10.0, STROKE);
        push_cone(&mut none, apex, [0.0, -1.0, 0.0], 0.5, 0.0, STROKE);
        assert!(none.is_empty());
    }

    #[test]
    fn frustum_has_twelve_edges_between_its_planes() {
        let frustum = Frustum {
            origin: [0.0; 3],
            right: [1.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            forward: [0.0, 0.0, -1.0],
            fov_y_rad: 90f32.to_radians(),
            aspect: 16.0 / 9.0,
            near: 1.0,
            far: 10.0,
        };
        let mut out = Vec::new();
        push_frustum(&mut out, &frustum, STROKE);
        assert_eq!(out.len(), BOX_EDGES);
        // fov 90: half height equals the plane distance; width is aspect times
        // that. Every corner sits on its plane.
        for p in endpoints(&out) {
            let d = -p[2];
            assert!((d - 1.0).abs() < 1e-4 || (d - 10.0).abs() < 1e-4, "{p:?}");
            assert!((p[1].abs() - d).abs() < 1e-3, "{p:?}");
            assert!((p[0].abs() - d * 16.0 / 9.0).abs() < 1e-2, "{p:?}");
        }
        let mut none = Vec::new();
        push_frustum(
            &mut none,
            &Frustum {
                near: 10.0,
                ..frustum
            },
            STROKE,
        );
        assert!(none.is_empty(), "far <= near is degenerate");
    }

    #[test]
    fn rect_outline_matches_the_lit_rectangle() {
        let mut out = Vec::new();
        // A downward panel: plane_basis puts the rect in the xz plane.
        push_rect(
            &mut out,
            [0.0, 3.0, 0.0],
            [0.0, -1.0, 0.0],
            [2.0, 1.0],
            18.0,
            STROKE,
        );
        assert_eq!(out.len(), 4 + 1);
        for l in &out[..4] {
            assert!((l.start[1] - 3.0).abs() < 1e-5, "rect stays in its plane");
        }
        // The normal ray runs the attenuation range along the normal.
        let ray = out[4];
        assert_eq!(ray.start, [0.0, 3.0, 0.0]);
        assert!((ray.end[1] - (3.0 - 18.0)).abs() < 1e-4);
        let mut none = Vec::new();
        push_rect(
            &mut none,
            [0.0; 3],
            [0.0, -1.0, 0.0],
            [0.0, 1.0],
            18.0,
            STROKE,
        );
        assert!(none.is_empty());
    }
}
