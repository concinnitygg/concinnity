//! The clapper-board mark from the editor's icon, laid on every face of a box
//! as raw geometry: eight slats in two rows over the board below them. The
//! polygons are the icon's own, in its units, and are re-laid per face onto
//! the box's surface, along with a frame that traces the box's edges.

use concinnity::bake::{Mesh, VertexData};

// The mark's polygons in the icon's units, y down: the upper and lower rows
// of slats, then the board. Every polygon is convex.
const SLATS: [[[f32; 2]; 4]; 8] = [
    [[14.0, 92.0], [37.0, 102.0], [23.0, 109.0], [0.0, 99.0]],
    [[47.0, 77.0], [70.0, 87.0], [56.0, 94.0], [33.0, 84.0]],
    [[80.0, 62.0], [103.0, 72.0], [89.0, 79.0], [66.0, 69.0]],
    [[113.0, 47.0], [136.0, 57.0], [122.0, 64.0], [99.0, 54.0]],
    [[14.0, 128.0], [37.0, 118.0], [23.0, 111.0], [0.0, 121.0]],
    [[47.0, 143.0], [70.0, 133.0], [56.0, 126.0], [33.0, 136.0]],
    [[80.0, 158.0], [103.0, 148.0], [89.0, 141.0], [66.0, 151.0]],
    [
        [113.0, 173.0],
        [136.0, 163.0],
        [122.0, 156.0],
        [99.0, 166.0],
    ],
];
const BOARD: [[f32; 2]; 3] = [[0.0, 124.64], [0.0, 174.64], [110.0, 174.64]];
// The box the polygons above fit in, in the same units.
const MARK_MIN: [f32; 2] = [0.0, 46.0];
const MARK_MAX: [f32; 2] = [136.0, 174.64];

// One face of a box: its outward normal and the two in-plane axes geometry is
// laid along, right-handed so that `right x up = normal`.
struct Face {
    normal: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
}

const FACES: [Face; 6] = [
    Face {
        normal: [0.0, 0.0, 1.0],
        right: [1.0, 0.0, 0.0],
        up: [0.0, 1.0, 0.0],
    },
    Face {
        normal: [0.0, 0.0, -1.0],
        right: [-1.0, 0.0, 0.0],
        up: [0.0, 1.0, 0.0],
    },
    Face {
        normal: [1.0, 0.0, 0.0],
        right: [0.0, 0.0, -1.0],
        up: [0.0, 1.0, 0.0],
    },
    Face {
        normal: [-1.0, 0.0, 0.0],
        right: [0.0, 0.0, 1.0],
        up: [0.0, 1.0, 0.0],
    },
    Face {
        normal: [0.0, 1.0, 0.0],
        right: [1.0, 0.0, 0.0],
        up: [0.0, 0.0, -1.0],
    },
    Face {
        normal: [0.0, -1.0, 0.0],
        right: [1.0, 0.0, 0.0],
        up: [0.0, 0.0, 1.0],
    },
];

/// The mark on every face of a box of `half_extent`, spanning `span` of the
/// face's width and lifted `lift` off the surface so it draws over the box.
pub(crate) fn mark_on_box(half_extent: f32, span: f32, lift: f32) -> Mesh {
    let scale = span * 2.0 * half_extent / (MARK_MAX[0] - MARK_MIN[0]);
    let centre = [
        (MARK_MIN[0] + MARK_MAX[0]) * 0.5,
        (MARK_MIN[1] + MARK_MAX[1]) * 0.5,
    ];
    // The icon's y runs down the page; the face's runs up it.
    let on_face = |p: [f32; 2]| [(p[0] - centre[0]) * scale, (centre[1] - p[1]) * scale];

    let mut sheet = FaceSheet::new(half_extent, lift);
    for face in &FACES {
        for slat in &SLATS {
            sheet.polygon(face, &slat.map(on_face));
        }
        sheet.polygon(face, &BOARD.map(on_face));
    }
    sheet.into_mesh()
}

/// A frame tracing every edge of a box of `half_extent`: a band `width` wide
/// inside each face's border, lifted `lift` off the surface.
pub(crate) fn edge_frame(half_extent: f32, width: f32, lift: f32) -> Mesh {
    let outer = half_extent;
    let inner = half_extent - width;
    let mut sheet = FaceSheet::new(half_extent, lift);
    for face in &FACES {
        // Four bands, each running the full length of one side.
        sheet.polygon(
            face,
            &[
                [-outer, inner],
                [outer, inner],
                [outer, outer],
                [-outer, outer],
            ],
        );
        sheet.polygon(
            face,
            &[
                [-outer, -outer],
                [outer, -outer],
                [outer, -inner],
                [-outer, -inner],
            ],
        );
        sheet.polygon(
            face,
            &[
                [-outer, -inner],
                [-inner, -inner],
                [-inner, inner],
                [-outer, inner],
            ],
        );
        sheet.polygon(
            face,
            &[
                [inner, -inner],
                [outer, -inner],
                [outer, inner],
                [inner, inner],
            ],
        );
    }
    sheet.into_mesh()
}

// Geometry accumulated over the faces of one box, each polygon placed in a
// face's plane just off its surface.
struct FaceSheet {
    half_extent: f32,
    lift: f32,
    vertices: Vec<VertexData>,
    indices: Vec<u16>,
}

impl FaceSheet {
    fn new(half_extent: f32, lift: f32) -> Self {
        Self {
            half_extent,
            lift,
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }

    // Lay a convex polygon, given in the face's `(right, up)` coordinates, on
    // `face`, fanned from its first point and wound to face outward.
    fn polygon(&mut self, face: &Face, points: &[[f32; 2]]) {
        let mut points = points.to_vec();
        if signed_area(&points) < 0.0 {
            points.reverse();
        }
        let first = self.vertices.len() as u16;
        let height = self.half_extent + self.lift;
        for [u, v] in &points {
            let pos = std::array::from_fn(|k| {
                face.normal[k] * height + face.right[k] * u + face.up[k] * v
            });
            self.vertices.push(VertexData {
                pos,
                color: [1.0; 3],
                uv: [
                    u / (2.0 * self.half_extent) + 0.5,
                    v / (2.0 * self.half_extent) + 0.5,
                ],
            });
        }
        for i in 1..points.len() as u16 - 1 {
            self.indices.extend([first, first + i, first + i + 1]);
        }
    }

    fn into_mesh(self) -> Mesh {
        Mesh {
            vertices: self.vertices,
            indices: self.indices,
            ..Default::default()
        }
    }
}

// Twice the signed area of a polygon: positive when its points run
// counter-clockwise.
fn signed_area(points: &[[f32; 2]]) -> f32 {
    points
        .iter()
        .zip(points.iter().cycle().skip(1))
        .map(|(a, b)| a[0] * b[1] - b[0] * a[1])
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity::bake;

    fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    // Every triangle winds counter-clockwise seen from outside the box: its
    // face normal points the way the box face it sits on does.
    fn assert_faces_outward(mesh: &Mesh) {
        for tri in mesh.indices.chunks_exact(3) {
            let [a, b, c] = [
                mesh.vertices[tri[0] as usize].pos,
                mesh.vertices[tri[1] as usize].pos,
                mesh.vertices[tri[2] as usize].pos,
            ];
            let normal = cross(sub(b, a), sub(c, a));
            let centroid: [f32; 3] = std::array::from_fn(|k| (a[k] + b[k] + c[k]) / 3.0);
            assert!(
                dot(normal, centroid) > 0.0,
                "triangle {tri:?} at {centroid:?} winds inward"
            );
        }
    }

    #[test]
    fn each_face_axis_frame_is_right_handed() {
        for face in &FACES {
            assert_eq!(cross(face.right, face.up), face.normal);
        }
    }

    #[test]
    fn the_mark_covers_every_face_and_faces_outward() {
        let mesh = mark_on_box(0.7, 0.6, 0.01);
        // Eight slats of four points and a board of three, per face.
        assert_eq!(mesh.vertices.len(), 6 * (8 * 4 + 3));
        assert_eq!(mesh.indices.len(), 6 * (8 * 2 + 1) * 3);
        assert_faces_outward(&mesh);
        // Lifted off the surface, and within the face's width.
        for v in &mesh.vertices {
            let out = v.pos.iter().fold(0.0f32, |m, c| m.max(c.abs()));
            assert!((out - 0.71).abs() < 1e-5, "{:?}", v.pos);
            assert!(v.pos.iter().all(|c| c.abs() <= 0.71 + 1e-5));
            assert!(v.uv.iter().all(|t| (0.0..=1.0).contains(t)), "{:?}", v.uv);
        }
    }

    #[test]
    fn the_mark_spans_the_requested_fraction_of_a_face() {
        let mesh = mark_on_box(1.0, 0.5, 0.0);
        let front: Vec<_> = mesh
            .vertices
            .iter()
            .filter(|v| v.pos[2] > 0.99)
            .map(|v| v.pos[0])
            .collect();
        let width = front.iter().fold(f32::MIN, |m, x| m.max(*x))
            - front.iter().fold(f32::MAX, |m, x| m.min(*x));
        assert!((width - 1.0).abs() < 1e-5, "spans {width}");
    }

    #[test]
    fn the_frame_lines_every_edge_and_faces_outward() {
        let mesh = edge_frame(0.7, 0.05, 0.01);
        assert_eq!(mesh.vertices.len(), 6 * 4 * 4);
        assert_eq!(mesh.indices.len(), 6 * 4 * 2 * 3);
        assert_faces_outward(&mesh);
    }

    #[test]
    fn a_clockwise_polygon_is_turned_around() {
        let mut sheet = FaceSheet::new(1.0, 0.0);
        sheet.polygon(&FACES[0], &[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]);
        let mesh = sheet.into_mesh();
        assert_faces_outward(&mesh);
        assert!(signed_area(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]) > 0.0);
    }

    #[test]
    fn both_meshes_bake() {
        bake::mesh(&mark_on_box(0.7, 0.6, 0.004)).expect("the mark bakes");
        bake::mesh(&edge_frame(0.7, 0.03, 0.004)).expect("the frame bakes");
    }
}
