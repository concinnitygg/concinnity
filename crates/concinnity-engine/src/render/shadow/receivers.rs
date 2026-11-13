//! Where a shadow view's casters can lie: a caster shadows only what lies
//! farther from the light along the same line, so a view drawn for one frame's
//! receivers need hold only the casters across the light from them. The
//! receivers are what every view shading with the sun's cascades this frame
//! can see: the camera's, a prop reflection probe's cube and each planar
//! mirror's, each picking a cascade by its own view depth.

use crate::environment::planet::Rebase;
use crate::render::planar_reflection::MAX_PLANAR_PLANES;
use crate::render::shadow::motion::{ShadowSweep, span};
use crate::transform::{Mat4, mat4_inverse};
use concinnity_core::math::sqrt;
use concinnity_core::math::vec3::{add, cross, dot, length_sq, scale, sub};

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

    /// The same view in the frame `rebase` moves the world to.
    #[must_use]
    pub(crate) fn rebased(self, rebase: &Rebase) -> Self {
        match self {
            Self::Frustum { eye, rays } => Self::Frustum {
                eye: rebase.apply_point(eye),
                rays: rays.map(|r| rebase.apply_vector(r)),
            },
            Self::Cube { center } => Self::Cube {
                center: rebase.apply_point(center),
            },
        }
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

    /// Whether this view can see, between view depths `near_d` and `far_d`,
    /// anything inside `sweep`: where a moved caster's shadow can fall.
    pub(crate) fn may_receive(&self, near_d: f32, far_d: f32, sweep: &ShadowSweep) -> bool {
        match *self {
            Self::Frustum { .. } => {
                let corners = self.corners(near_d, far_d);
                let overlaps = (0..3).all(|a| {
                    let [lo, hi] = span(&corners, sweep.axes[a]);
                    let [s_lo, s_hi] = sweep.ranges[a];
                    lo <= s_hi && s_lo <= hi
                });
                overlaps
                    && !slab_planes(&corners)
                        .iter()
                        .any(|plane| sweep.corners.iter().all(|p| plane_side(*plane, *p) < 0.0))
            }
            // A face sees what lies between the depths along its axis, a shell
            // about the center.
            Self::Cube { center } => {
                let mut lo = [f32::INFINITY; 3];
                let mut hi = [f32::NEG_INFINITY; 3];
                for p in &sweep.corners {
                    for a in 0..3 {
                        lo[a] = lo[a].min(p[a] - center[a]);
                        hi[a] = hi[a].max(p[a] - center[a]);
                    }
                }
                let reaches_outer = (0..3).all(|a| lo[a] <= far_d && -far_d <= hi[a]);
                let inside_inner = (0..3).all(|a| -near_d < lo[a] && hi[a] < near_d);
                reaches_outer && !inside_inner
            }
        }
    }
}

// The inward planes (normal, offset) of the slab whose corners `corners`
// lists as `ShadowReader::corners` gives them; a degenerate face, such as a
// near face shrunk to the eye, is left out.
fn slab_planes(corners: &[[f32; 3]; 8]) -> [[f32; 4]; 6] {
    let centroid = scale(corners.iter().fold([0.0; 3], |a, c| add(a, *c)), 1.0 / 8.0);
    let faces = [
        [0, 1, 2],
        [4, 6, 5],
        [0, 7, 4],
        [1, 5, 6],
        [0, 4, 5],
        [3, 6, 7],
    ];
    faces.map(|[a, b, c]| {
        let n = cross(sub(corners[b], corners[a]), sub(corners[c], corners[a]));
        let len = sqrt(length_sq(n));
        if len < 1e-6 {
            return [0.0, 0.0, 0.0, 1.0];
        }
        let n = scale(n, 1.0 / len);
        let n = if dot(n, sub(centroid, corners[a])) < 0.0 {
            scale(n, -1.0)
        } else {
            n
        };
        [n[0], n[1], n[2], -dot(n, corners[a])]
    })
}

// How far inside the plane `plane` the point `p` lies; negative outside.
fn plane_side(plane: [f32; 4], p: [f32; 3]) -> f32 {
    plane[0] * p[0] + plane[1] * p[1] + plane[2] * p[2] + plane[3]
}

/// Every view shading with the sun's cascades this frame.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ShadowReaders {
    camera: Option<ShadowReader>,
    probe: Option<ShadowReader>,
    // How far from the probe its views take the cascades.
    probe_reach: f32,
    mirrors: [Option<ShadowReader>; MAX_PLANAR_PLANES],
}

/// Room for the corners of every reader's view of one depth band.
pub(crate) type ReceiverCorners = [[f32; 3]; 8 * (2 + MAX_PLANAR_PLANES)];

impl ShadowReaders {
    /// Carry every view into the frame `rebase` moves the world to.
    pub(crate) fn rebase(&mut self, rebase: &Rebase) {
        let carry = |r: &mut Option<ShadowReader>| *r = r.map(|v| v.rebased(rebase));
        carry(&mut self.camera);
        carry(&mut self.probe);
        self.mirrors.iter_mut().for_each(carry);
    }

    /// Set the camera's view.
    pub(crate) fn set_camera(&mut self, camera: ShadowReader) {
        self.camera = Some(camera);
    }

    /// Set the prop reflection probe capturing this frame, if any, and how
    /// far from it its views take the cascades.
    pub(crate) fn set_probe(&mut self, probe: Option<ShadowReader>, reach: f32) {
        self.probe = probe;
        self.probe_reach = reach;
    }

    // Every reader whose view takes the cascades somewhere between view
    // depths `near_d` and `far_d`, with the far depth it takes them to.
    fn reading(&self, near_d: f32, far_d: f32) -> impl Iterator<Item = (ShadowReader, f32)> + '_ {
        let probe = self
            .probe
            .filter(|_| near_d < self.probe_reach)
            .map(|p| (p, far_d.min(self.probe_reach)));
        let views = [self.camera].into_iter().chain(self.mirrors).flatten();
        views.map(move |v| (v, far_d)).chain(probe)
    }

    /// Set the planar mirrors rendering this frame, as many as there is room
    /// for.
    pub(crate) fn set_mirrors(&mut self, mirrors: impl IntoIterator<Item = ShadowReader>) {
        self.mirrors = [None; MAX_PLANAR_PLANES];
        for (slot, mirror) in self.mirrors.iter_mut().zip(mirrors) {
            *slot = Some(mirror);
        }
    }

    /// Whether any reader can see, between view depths `near_d` and `far_d`,
    /// anything inside `sweep`.
    pub(crate) fn may_receive(&self, near_d: f32, far_d: f32, sweep: &ShadowSweep) -> bool {
        self.reading(near_d, far_d)
            .any(|(r, far)| r.may_receive(near_d, far, sweep))
    }

    /// The corners of every reader's view between view depths `near_d` and
    /// `far_d`, written into `out`; returns how many.
    pub(crate) fn corners(&self, near_d: f32, far_d: f32, out: &mut ReceiverCorners) -> usize {
        let mut n = 0;
        for (reader, far) in self.reading(near_d, far_d) {
            out[n..n + 8].copy_from_slice(&reader.corners(near_d, far));
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

    fn camera_reader() -> (Mat4, ShadowReader) {
        let eye = [0.0, 2.0, 0.0];
        let view = look_at(eye, [0.0, 1.5, -10.0], [0.0, 1.0, 0.0]);
        let vp = mat4_mul(camera_projection(1.0, 16.0 / 9.0, 0.1), view);
        (view, ShadowReader::through(&view, &vp, eye))
    }

    fn sweep_at(center: [f32; 3], to_light: [f32; 3]) -> ShadowSweep {
        let lo = [center[0] - 0.5, center[1], center[2] - 0.5];
        let hi = [center[0] + 0.5, center[1] + 1.0, center[2] + 0.5];
        ShadowSweep::new((lo, hi), hi[1] - lo[1], to_light)
    }

    // A caster on the ground in view can shadow the band it stands in, and not
    // one far past it.
    #[test]
    fn a_view_receives_the_shadow_of_what_stands_in_its_band() {
        let (_, reader) = camera_reader();
        let sweep = sweep_at([0.0, 0.0, -8.0], [0.0, 0.8, 0.6]);
        assert!(reader.may_receive(4.0, 12.0, &sweep));
        assert!(!reader.may_receive(50.0, 250.0, &sweep));
    }

    // Under a low light behind it, a caster's long shadow reaches a band
    // past it; overhead, the same caster's does not.
    #[test]
    fn a_low_light_s_long_shadow_reaches_farther_bands() {
        let (_, reader) = camera_reader();
        let low = sweep_at([0.0, 0.0, -8.0], [0.0, 0.1, 0.995]);
        let overhead = sweep_at([0.0, 0.0, -8.0], [0.0, 1.0, 0.0]);
        assert!(reader.may_receive(12.0, 30.0, &low));
        assert!(!reader.may_receive(12.0, 30.0, &overhead));
    }

    // A caster behind the camera, lit from in front, shadows nothing in view.
    #[test]
    fn a_shadow_falling_away_from_the_view_is_not_received() {
        let (_, reader) = camera_reader();
        let sweep = sweep_at([0.0, 0.0, 6.0], [0.0, 0.5, -0.866]);
        assert!(!reader.may_receive(0.0, 250.0, &sweep));
    }

    // A cube capture sees a shell about its center: what lies inside its near
    // depth on every axis it does not see in that band.
    #[test]
    fn a_cube_receives_only_what_its_shell_holds() {
        let cube = ShadowReader::Cube { center: [0.0; 3] };
        let near = sweep_at([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]);
        assert!(cube.may_receive(0.0, 4.0, &near));
        assert!(!cube.may_receive(4.0, 20.0, &near));
        let mid = sweep_at([0.0, 0.0, -6.0], [0.0, 1.0, 0.0]);
        assert!(cube.may_receive(4.0, 20.0, &mid));
        assert!(!cube.may_receive(20.0, 60.0, &mid));
    }

    // A view carried through a rebase sees the same space, carried; a cube
    // capture keeps its faces on the frame's axes, so only its center moves.
    #[test]
    fn a_rebased_view_sees_the_carried_space() {
        let rebase = Rebase {
            rotation: concinnity_core::math::quat_from_axis_angle([0.0, 1.0, 0.0], 0.3),
            translation: [-900.0, 4.0, 10.0],
        };
        let close = |a: [f32; 3], b: [f32; 3]| (0..3).all(|k| (a[k] - b[k]).abs() < 1e-3);
        let (_, frustum) = camera_reader();
        let before = frustum.corners(2.0, 30.0);
        let after = frustum.rebased(&rebase).corners(2.0, 30.0);
        for (b, a) in before.iter().zip(&after) {
            assert!(close(rebase.apply_point(*b), *a), "{b:?} {a:?}");
        }
        let center = [3.0, 1.0, -2.0];
        let cube = ShadowReader::Cube { center }.rebased(&rebase);
        assert_eq!(
            cube,
            ShadowReader::Cube {
                center: rebase.apply_point(center)
            }
        );
    }

    #[test]
    fn readers_give_the_corners_of_every_view_set() {
        let mut readers = ShadowReaders::default();
        let mut out = [[0.0; 3]; 8 * (2 + MAX_PLANAR_PLANES)];
        assert_eq!(readers.corners(0.0, 1.0, &mut out), 0);
        readers.set_camera(ShadowReader::Cube { center: [0.0; 3] });
        readers.set_probe(Some(ShadowReader::Cube { center: [1.0; 3] }), 5.0);
        readers.set_mirrors([ShadowReader::Cube { center: [2.0; 3] }; 9]);
        assert_eq!(
            readers.corners(0.0, 1.0, &mut out),
            8 * (2 + MAX_PLANAR_PLANES)
        );
        readers.set_mirrors([]);
        readers.set_probe(None, 5.0);
        assert_eq!(readers.corners(0.0, 1.0, &mut out), 8);
    }

    // The probe takes the cascades only out to its reach: a band past it
    // leaves the probe out, and one across it stops the probe there.
    #[test]
    fn the_probe_reads_only_out_to_its_reach() {
        let mut readers = ShadowReaders::default();
        let mut out = [[0.0; 3]; 8 * (2 + MAX_PLANAR_PLANES)];
        readers.set_probe(Some(ShadowReader::Cube { center: [0.0; 3] }), 5.0);
        assert_eq!(readers.corners(6.0, 9.0, &mut out), 0);
        assert_eq!(readers.corners(2.0, 9.0, &mut out), 8);
        assert!(out[..8].iter().all(|p| p.iter().all(|v| v.abs() <= 5.0)));
        let far = ShadowSweep::new(([7.0, 0.0, 0.0], [8.0, 1.0, 1.0]), 1.0, [0.0, 1.0, 0.0]);
        assert!(!readers.may_receive(2.0, 9.0, &far));
        let near = ShadowSweep::new(([3.0, 0.0, 0.0], [4.0, 1.0, 1.0]), 1.0, [0.0, 1.0, 0.0]);
        assert!(readers.may_receive(2.0, 9.0, &near));
    }
}
