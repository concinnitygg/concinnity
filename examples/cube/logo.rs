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
    let center = [
        (MARK_MIN[0] + MARK_MAX[0]) * 0.5,
        (MARK_MIN[1] + MARK_MAX[1]) * 0.5,
    ];
    // The icon's y runs down the page; the face's runs up it.
    let on_face = |p: [f32; 2]| [(p[0] - center[0]) * scale, (center[1] - p[1]) * scale];

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
