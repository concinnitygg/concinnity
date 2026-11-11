//! Where a shadow view's casters can lie: a caster shadows only what lies
//! farther from the light along the same line, so a view drawn for one frame's
//! receivers need hold only the casters across the light from them. The
//! receivers are what every view shading with the sun's cascades this frame
//! can see: the camera's, a prop reflection probe's cube and each planar
//! mirror's, each picking a cascade by its own view depth.

use crate::render::planar_reflection::MAX_PLANAR_PLANES;
use crate::transform::{Mat4, mat4_inverse};
use concinnity_core::math::sqrt;
use concinnity_core::math::vec3::{add, dot, length_sq, scale, sub};

/// How far past its receivers' span across the light a cut box reaches, in
/// texels of the view it culls for: the 5x5 rotated filter reaches about 2.83
/// texels from a receiver, and its hash rotation a little more.
pub(crate) const RECEIVER_PAD_TEXELS: f32 = 3.0;

/// A view that shades with the sun's cascades, picking each by its own view
/// depth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ShadowReader {
    /// A perspective view: its eye and the rays through its four corners,
    /// each scaled to reach a view depth of 1.
    Frustum {
        /// Where the view looks from.
        eye: [f32; 3],
        /// The corner rays.
        rays: [[f32; 3]; 4],
    },
    /// The six 90 degree faces of a cube capture about one point.
    Cube {
        /// Where the capture looks from.
        center: [f32; 3],
    },
}

impl ShadowReader {
    /// The perspective view at `eye` with the world-to-view matrix `view` and
    /// the view-projection `view_proj`, which may crop it or tilt its near
    /// plane.
    pub(crate) fn through(view: &Mat4, view_proj: &Mat4, eye: [f32; 3]) -> Self {
        let inverse = mat4_inverse(*view_proj);
        let forward = [-view[0][2], -view[1][2], -view[2][2]];
        let corners = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        let rays = corners.map(|[x, y]| {
            let p: [f32; 4] = core::array::from_fn(|r| {
                inverse[0][r] * x + inverse[1][r] * y + inverse[2][r] * 0.5 + inverse[3][r]
            });
            let w = if p[3].abs() > 1e-12 { p[3] } else { 1e-12 };
            let dir = sub([p[0] / w, p[1] / w, p[2] / w], eye);
            let depth = dot(dir, forward);
            let depth = if depth.abs() > 1e-6 { depth } else { 1e-6 };
            scale(dir, 1.0 / depth)
        });
        Self::Frustum { eye, rays }
    }

    /// The corners of what this view sees between view depths `near_d` and
    /// `far_d`.
    pub(crate) fn corners(&self, near_d: f32, far_d: f32) -> [[f32; 3]; 8] {
        match *self {
            Self::Frustum { eye, rays } => core::array::from_fn(|i| {
                let d = if i < 4 { near_d } else { far_d };
                add(eye, scale(rays[i % 4], d))
            }),
            // A face sees out to `far_d` along its axis and less across it.
            Self::Cube { center } => core::array::from_fn(|i| {
                let s = |bit: usize| if i & bit == 0 { -far_d } else { far_d };
                [center[0] + s(1), center[1] + s(2), center[2] + s(4)]
            }),
        }
    }
}

/// Every view shading with the sun's cascades this frame.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ShadowReaders {
    camera: Option<ShadowReader>,
    probe: Option<ShadowReader>,
    mirrors: [Option<ShadowReader>; MAX_PLANAR_PLANES],
}

/// Room for the corners of every reader's view of one depth band.
pub(crate) type ReceiverCorners = [[f32; 3]; 8 * (2 + MAX_PLANAR_PLANES)];

impl ShadowReaders {
    /// Set the camera's view.
    pub(crate) fn set_camera(&mut self, camera: ShadowReader) {
        self.camera = Some(camera);
    }

    /// Set the prop reflection probe capturing this frame, if any.
    pub(crate) fn set_probe(&mut self, probe: Option<ShadowReader>) {
        self.probe = probe;
    }

    /// Set the planar mirrors rendering this frame, as many as there is room
    /// for.
    pub(crate) fn set_mirrors(&mut self, mirrors: impl IntoIterator<Item = ShadowReader>) {
        self.mirrors = [None; MAX_PLANAR_PLANES];
        for (slot, mirror) in self.mirrors.iter_mut().zip(mirrors) {
            *slot = Some(mirror);
        }
    }

    /// The corners of every reader's view between view depths `near_d` and
    /// `far_d`, written into `out`; returns how many.
    pub(crate) fn corners(&self, near_d: f32, far_d: f32, out: &mut ReceiverCorners) -> usize {
        let readers = [self.camera, self.probe]
            .into_iter()
            .chain(self.mirrors)
            .flatten();
        let mut n = 0;
        for reader in readers {
            out[n..n + 8].copy_from_slice(&reader.corners(near_d, far_d));
            n += 8;
        }
        n
    }
}

// The range of `dot(axis, p)` one orthographic row maps onto `ndc`, with the
// row's unit axis.
fn row_range(vp: &[[f32; 4]; 4], row: usize, ndc: [f32; 2]) -> ([f32; 3], [f32; 2]) {
    let xyz = [vp[0][row], vp[1][row], vp[2][row]];
    let len = sqrt(length_sq(xyz)).max(1e-12);
    let w = vp[3][row];
    let a = (ndc[0] - w) / len;
    let b = (ndc[1] - w) / len;
    (scale(xyz, 1.0 / len), [a.min(b), a.max(b)])
}

/// The receivers a cut box keeps the casters of, and how its views are
/// sampled.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ReceiverCut<'a> {
    /// The corners of the space the receivers fill.
    pub receivers: &'a [[f32; 3]],
    /// The texels across each view's map, which the cut is padded by
    /// [`RECEIVER_PAD_TEXELS`] of.
    pub map_size: u32,
}

/// The orthographic view-projection whose box holds the union of the boxes of
/// `vps`, cascades of one light sharing its axes. With a `cut` it holds only
/// the casters of its receivers: cut across the light to their span, padded
/// by the filter's reach, and cut off past the receiver farthest from the
/// light. `None` when there is no view, or no caster can reach a receiver. It
/// maps the box onto the same clip ranges each cascade uses: x and y over
/// [-1, 1], and reversed depth over [0, 1] rising toward the light.
pub(crate) fn cast_box(
    vps: impl IntoIterator<Item = [[f32; 4]; 4]>,
    cut: Option<ReceiverCut<'_>>,
) -> Option<[[f32; 4]; 4]> {
    const RANGES: [[f32; 2]; 3] = [[-1.0, 1.0], [-1.0, 1.0], [0.0, 1.0]];
    let mut axes = [[0.0_f32; 3]; 3];
    let mut bounds = [[f32::INFINITY, f32::NEG_INFINITY]; 3];
    // The widest view's span across the light on each row.
    let mut widest = [0.0_f32; 2];
    let mut any = false;
    for vp in vps {
        for (row, ndc) in RANGES.iter().enumerate() {
            let (axis, [lo, hi]) = row_range(&vp, row, *ndc);
            if !any {
                axes[row] = axis;
            }
            if let Some(w) = widest.get_mut(row) {
                *w = w.max(hi - lo);
            }
            // Cascades of one light share their axes, so the range along the
            // first one's axis is the cascade's own.
            let along = dot(axis, axes[row]);
            let (lo, hi) = if along < 0.0 { (-hi, -lo) } else { (lo, hi) };
            bounds[row] = [bounds[row][0].min(lo), bounds[row][1].max(hi)];
        }
        any = true;
    }
    if !any {
        return None;
    }
    if let Some(cut) = cut {
        for (row, bound) in bounds.iter_mut().enumerate() {
            let (lo, hi) =
                cut.receivers
                    .iter()
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), p| {
                        let d = dot(axes[row], *p);
                        (lo.min(d), hi.max(d))
                    });
            // Casters stand toward the light from what they shadow, so depth
            // is only cut on the side away from it, and unpadded.
            if let Some(&w) = widest.get(row) {
                let pad = RECEIVER_PAD_TEXELS * w / cut.map_size.max(1) as f32;
                bound[0] = bound[0].max(lo - pad);
                bound[1] = bound[1].min(hi + pad);
            } else {
                bound[0] = bound[0].max(lo);
            }
        }
        if bounds.iter().any(|[lo, hi]| lo >= hi) {
            return None;
        }
    }
    // Row r maps `dot(axis, p)` over its bounds linearly onto RANGES[r].
    let mut out = [[0.0_f32; 4]; 4];
    for (row, ndc) in RANGES.iter().enumerate() {
        let [lo, hi] = bounds[row];
        let k = (ndc[1] - ndc[0]) / (hi - lo).max(1e-6);
        for (col, c) in axes[row].iter().enumerate() {
            out[col][row] = c * k;
        }
        out[3][row] = ndc[0] - lo * k;
    }
    out[3][3] = 1.0;
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::projection::look_at;
    use crate::render::depth::camera_projection;
    use crate::transform::mat4_mul;

    fn view_depth(view: &Mat4, p: [f32; 3]) -> f32 {
        -(view[0][2] * p[0] + view[1][2] * p[1] + view[2][2] * p[2] + view[3][2])
    }

    // A view's corners lie on its frustum's edges at the depths asked for.
    #[test]
    fn a_view_s_corners_lie_at_its_depths_on_its_edges() {
        let eye = [3.0, 2.0, -1.0];
        let view = look_at(eye, [0.0, 1.0, -9.0], [0.0, 1.0, 0.0]);
        let vp = mat4_mul(camera_projection(1.1, 1.6, 0.1), view);
        let corners = ShadowReader::through(&view, &vp, eye).corners(2.0, 30.0);
        for (i, c) in corners.iter().enumerate() {
            let want = if i < 4 { 2.0 } else { 30.0 };
            assert!((view_depth(&view, *c) - want).abs() < 1e-3, "{i} {c:?}");
            let clip: [f32; 4] = core::array::from_fn(|r| {
                vp[0][r] * c[0] + vp[1][r] * c[1] + vp[2][r] * c[2] + vp[3][r]
            });
            for axis in 0..2 {
                assert!(
                    ((clip[axis] / clip[3]).abs() - 1.0).abs() < 1e-3,
                    "{i} {clip:?}"
                );
            }
        }
    }

    // A cube face sees a point at the depth of its largest offset, so the
    // cube of that half-size holds everything the capture shades out to it.
    #[test]
    fn a_cube_s_corners_bound_its_faces_out_to_the_depth() {
        let center = [1.0, 2.0, 3.0];
        let corners = ShadowReader::Cube { center }.corners(0.0, 5.0);
        for axis in 0..3 {
            let lo = corners
                .iter()
                .map(|c| c[axis])
                .fold(f32::INFINITY, f32::min);
            let hi = corners
                .iter()
                .map(|c| c[axis])
                .fold(f32::NEG_INFINITY, f32::max);
            assert_eq!([lo, hi], [center[axis] - 5.0, center[axis] + 5.0]);
        }
    }

    #[test]
    fn readers_give_the_corners_of_every_view_set() {
        let mut readers = ShadowReaders::default();
        let mut out = [[0.0; 3]; 8 * (2 + MAX_PLANAR_PLANES)];
        assert_eq!(readers.corners(0.0, 1.0, &mut out), 0);
        readers.set_camera(ShadowReader::Cube { center: [0.0; 3] });
        readers.set_probe(Some(ShadowReader::Cube { center: [1.0; 3] }));
        readers.set_mirrors([ShadowReader::Cube { center: [2.0; 3] }; 9]);
        assert_eq!(
            readers.corners(0.0, 1.0, &mut out),
            8 * (2 + MAX_PLANAR_PLANES)
        );
        readers.set_mirrors([]);
        readers.set_probe(None);
        assert_eq!(readers.corners(0.0, 1.0, &mut out), 8);
    }
}
