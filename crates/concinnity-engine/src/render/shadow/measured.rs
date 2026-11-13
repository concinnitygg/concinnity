//! What the camera saw of each cascade's depth band, measured on the GPU from
//! the frame's depth and read back a few frames later without waiting: the
//! surfaces a moved caster's shadow can land on, bounded by what is there
//! rather than by the whole volume each band of the view spans.
//!
//! A measurement is stamped with the camera and light it was taken through and
//! carried through every rebase since. It stands for what the camera sees now
//! only while it is recent, the camera has not turned far, its projection is
//! the same and the bands are split as they were; it is then padded by how
//! far the camera moved and turned since, and widened by every caster that
//! moved since, so a receiver that moved is still held.

use crate::environment::planet::Rebase;
use crate::gfx::frustum::{Aabb, transform_aabb};
use crate::gfx::render_types::{
    NUM_SHADOW_CASCADES, ReceiverBand, ReceiverBounds, ReceiverBoundsParams, receiver_tiles,
};
use crate::render::shadow::csm::LightBasis;
use crate::render::shadow::motion::span;
use crate::transform::Mat4;
use concinnity_core::math::vec3::{add, dot, length_sq, scale, sub};
use concinnity_core::math::{acos, sqrt};

/// How far the camera may have turned since a measurement, in radians, for it
/// to still stand for what the camera sees once padded by the turn: about 14
/// degrees, a follow camera's swing over the few frames a readback takes, while
/// the surfaces a turn uncovers still border what was measured. A wider turn
/// drops the measurement.
pub(crate) const MAX_MEASURED_TURN: f32 = 0.25;

/// How many frames old a measurement may be and still be used: more than the
/// frames any backend keeps in flight.
pub(crate) const MAX_MEASURED_AGE: u32 = 6;

// How many frames' stamps are kept: enough for the oldest usable measurement
// and every frame since.
const STAMPS: usize = MAX_MEASURED_AGE as usize + 2;

/// The camera and light a frame's receivers are measured through.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MeasureView {
    /// World-to-view matrix, column-major.
    pub view: Mat4,
    /// The camera's position.
    pub eye: [f32; 3],
    /// The inverse of the view-projection the depth was drawn through.
    pub depth_inv_vp: Mat4,
    /// Vertical field of view, in radians.
    pub fov_y: f32,
    /// Viewport width over height.
    pub aspect: f32,
    /// Unit vector toward the light.
    pub to_light: [f32; 3],
    /// Each cascade's depth band, from where it blends in to its split; a
    /// band not in use starts past where it ends.
    pub bands: [[f32; 2]; NUM_SHADOW_CASCADES],
    /// The depth's size in texels.
    pub size: [u32; 2],
}

impl MeasureView {
    /// The kernel's parameters for this view, numbered `serial`: the matrix
    /// that takes a depth texel's NDC and device depth to the light's right,
    /// up and toward-the-light coordinates and the view depth.
    pub(crate) fn params(&self, serial: u32) -> ReceiverBoundsParams {
        let [right, up, toward] = LightBasis::axes(self.to_light);
        let forward = camera_basis(&self.view)[2];
        let row = |axis: [f32; 3], offset: f32| [axis[0], axis[1], axis[2], offset];
        let rows = [
            row(right, 0.0),
            row(up, 0.0),
            row(toward, 0.0),
            row(forward, -dot(forward, self.eye)),
        ];
        let inv = &self.depth_inv_vp;
        let to_light_space = core::array::from_fn(|col| {
            core::array::from_fn(|r| (0..4).map(|k| rows[r][k] * inv[col][k]).sum())
        });
        let in_use = |c: usize| self.bands[c][0] <= self.bands[c][1];
        ReceiverBoundsParams {
            to_light_space,
            w_row: core::array::from_fn(|col| inv[col][3]),
            band_near: core::array::from_fn(|c| if in_use(c) { self.bands[c][0] } else { 0.0 }),
            band_far: core::array::from_fn(|c| if in_use(c) { self.bands[c][1] } else { -1.0 }),
            width: self.size[0],
            height: self.size[1],
            groups_x: receiver_tiles(self.size[0], self.size[1])[0],
            serial,
        }
    }
}

/// One cascade band's measured receivers as they stand now: between view
/// depths `depths`, and within `ranges` along the light's right, up and
/// toward-the-light axes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MeasuredBand {
    /// The view depths the receivers lie between.
    pub depths: [f32; 2],
    /// Their range along each of the light's axes.
    pub ranges: [[f32; 2]; 3],
}

/// The camera's measured receivers in each cascade's band, `None` where there
/// is no usable measurement and the band's whole view volume stands in.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CameraBands {
    /// Each cascade's band.
    pub bands: [Option<MeasuredBand>; NUM_SHADOW_CASCADES],
}

impl CameraBands {
    /// Cascade `cascade`'s measured band, if any.
    pub(crate) fn band(&self, cascade: usize) -> Option<MeasuredBand> {
        self.bands.get(cascade).copied().flatten()
    }
}

/// Where casters moved in one frame, merged into one box.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) enum MovedSince {
    /// Nothing moved.
    #[default]
    Nothing,
    /// Everything that moved lies in this box.
    Within(Aabb),
    /// Something moved with no box to say where.
    Anywhere,
}

impl MovedSince {
    /// The merge of every box in `boxes`, or [`Self::Anywhere`] when `anywhere`.
    pub(crate) fn of(anywhere: bool, boxes: impl IntoIterator<Item = Aabb>) -> Self {
        if anywhere {
            return Self::Anywhere;
        }
        boxes.into_iter().fold(Self::Nothing, Self::with)
    }

    fn with(self, b: Aabb) -> Self {
        match self {
            Self::Nothing => Self::Within(b),
            Self::Within((lo, hi)) => Self::Within((
                core::array::from_fn(|i| lo[i].min(b.0[i])),
                core::array::from_fn(|i| hi[i].max(b.1[i])),
            )),
            Self::Anywhere => Self::Anywhere,
        }
    }

    fn merged(self, other: Self) -> Self {
        match other {
            Self::Nothing => self,
            Self::Within(b) => self.with(b),
            Self::Anywhere => Self::Anywhere,
        }
    }

    fn rebased(self, rebase: &Rebase) -> Self {
        match self {
            Self::Within((lo, hi)) => Self::Within(transform_aabb(lo, hi, rebase.matrix())),
            other => other,
        }
    }
}

// What a measurement was taken through, carried into the frame the plan is in
// now.
#[derive(Clone, Copy, Debug)]
struct Stamp {
    serial: u32,
    // The light's right, up and toward-the-light axes.
    axes: [[f32; 3]; 3],
    eye: [f32; 3],
    // The camera's right, up and forward.
    basis: [[f32; 3]; 3],
    bands: [[f32; 2]; NUM_SHADOW_CASCADES],
    // The vertical field of view and aspect the camera saw through.
    fov_y: f32,
    aspect: f32,
    // Where casters moved before the frame was drawn, so the depth holds them.
    moved: MovedSince,
    // Every rebase since, applied to everything above.
    carry: Rebase,
}

impl Stamp {
    fn axes(&self) -> [[f32; 3]; 3] {
        self.axes.map(|a| self.carry.apply_vector(a))
    }

    fn eye(&self) -> [f32; 3] {
        self.carry.apply_point(self.eye)
    }

    fn basis(&self) -> [[f32; 3]; 3] {
        self.basis.map(|a| self.carry.apply_vector(a))
    }

    fn moved(&self) -> MovedSince {
        self.moved.rebased(&self.carry)
    }
}

// A measurement the GPU returned, with the stamp it was taken through.
#[derive(Clone, Copy, Debug)]
struct Measured {
    stamp: Stamp,
    bands: [ReceiverBand; NUM_SHADOW_CASCADES],
}

/// Numbers each frame's measurement, keeps what it was taken through, and
/// holds the newest one the GPU returned.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ReceiverMeasure {
    next: u32,
    stamps: [Option<Stamp>; STAMPS],
    held: Option<Measured>,
}

impl Default for ReceiverMeasure {
    fn default() -> Self {
        Self {
            next: 1,
            stamps: [None; STAMPS],
            held: None,
        }
    }
}

impl ReceiverMeasure {
    /// Stamp this frame's measurement through `view`, with `moved` where
    /// casters moved before the frame is drawn, and return its kernel's
    /// parameters.
    pub(crate) fn begin(&mut self, view: &MeasureView, moved: MovedSince) -> ReceiverBoundsParams {
        let serial = self.next;
        self.next = after(serial);
        self.stamps[slot(serial)] = Some(Stamp {
            serial,
            axes: LightBasis::axes(view.to_light),
            eye: view.eye,
            basis: camera_basis(&view.view),
            bands: view.bands,
            fov_y: view.fov_y,
            aspect: view.aspect,
            moved,
            carry: Rebase::IDENTITY,
        });
        view.params(serial)
    }

    /// Take a measurement the GPU finished: kept when its stamp is still known
    /// and it is newer than the one held. Serial 0 is never stamped, so a
    /// readback slot nothing was written to is never taken.
    pub(crate) fn accept(&mut self, bounds: &ReceiverBounds) {
        if bounds.serial == 0 {
            return;
        }
        let Some(stamp) = self.stamps[slot(bounds.serial)].filter(|s| s.serial == bounds.serial)
        else {
            return;
        };
        if self
            .held
            .is_some_and(|h| !newer(bounds.serial, h.stamp.serial))
        {
            return;
        }
        self.held = Some(Measured {
            stamp,
            bands: bounds.bands,
        });
    }

    /// Carry every stamp and the held measurement into the frame `rebase`
    /// moves the world to.
    pub(crate) fn rebase(&mut self, rebase: &Rebase) {
        let carry = |s: &mut Stamp| s.carry = s.carry.then(rebase);
        self.stamps.iter_mut().flatten().for_each(carry);
        if let Some(h) = &mut self.held {
            carry(&mut h.stamp);
        }
    }

    /// Forget every stamp and measurement, for a view that does not continue
    /// the last. Numbering carries on, so a measurement still in flight is
    /// not taken for a later one.
    pub(crate) fn reset(&mut self) {
        self.stamps = [None; STAMPS];
        self.held = None;
    }

    /// What the held measurement says the camera of `view` sees in each band:
    /// `reach[c]` is how far from the camera band `c` reaches, and `moved`
    /// where casters moved this frame. Empty when there is no measurement
    /// recent enough, the bands are split differently, the field of view or
    /// aspect changed, the camera turned too far, or a caster moved with no
    /// box to say where.
    pub(crate) fn camera_bands(
        &self,
        view: &MeasureView,
        reach: [f32; NUM_SHADOW_CASCADES],
        moved: MovedSince,
    ) -> CameraBands {
        let axes = LightBasis::axes(view.to_light);
        let none = CameraBands::default();
        let Some(held) = self.held else {
            return none;
        };
        let stamp = &held.stamp;
        if steps(stamp.serial, self.next) > MAX_MEASURED_AGE
            || !same_bands(&stamp.bands, &view.bands)
            || !close(stamp.fov_y, view.fov_y)
            || !close(stamp.aspect, view.aspect)
        {
            return none;
        }
        let turn = turn_between(&stamp.basis(), &camera_basis(&view.view));
        if turn > MAX_MEASURED_TURN {
            return none;
        }
        let since = core::iter::successors(Some(after(stamp.serial)), |s| Some(after(*s)))
            .take_while(|s| *s != self.next)
            .take(STAMPS)
            .filter_map(|s| self.stamps[slot(s)].filter(|t| t.serial == s))
            .fold(moved, |m, s| m.merged(s.moved()));
        if since == MovedSince::Anywhere {
            return none;
        }
        let moved_eye = sqrt(length_sq(sub(view.eye, stamp.eye())));
        let measured_axes = stamp.axes();
        let shift = measured_axes.map(|a| dot(a, stamp.carry.translation));
        let forward = camera_basis(&view.view)[2];
        let bands = core::array::from_fn(|c| {
            let raw = held.bands[c];
            let [near, far] = view.bands[c];
            if raw.is_empty() || near > far {
                return None;
            }
            let pad = moved_eye + turn * reach[c];
            let held_ranges: [[f32; 2]; 3] =
                core::array::from_fn(|k| [raw.lo[k] + shift[k], raw.hi[k] + shift[k]]);
            let mut ranges = project_box(&measured_axes, &held_ranges, &axes)
                .map(|[lo, hi]| [lo - pad, hi + pad]);
            let mut depths = [near.max(raw.lo[3] - pad), far.min(raw.hi[3] + pad)];
            if let MovedSince::Within(b) = since {
                let corners = box_corners(b);
                let [lo, hi] = span(&corners, forward).map(|d| d - dot(forward, view.eye));
                if lo <= far && near <= hi {
                    for (k, range) in ranges.iter_mut().enumerate() {
                        let [b_lo, b_hi] = span(&corners, axes[k]);
                        *range = [range[0].min(b_lo), range[1].max(b_hi)];
                    }
                    depths = [depths[0].min(lo.max(near)), depths[1].max(hi.min(far))];
                }
            }
            Some(MeasuredBand { depths, ranges })
        });
        CameraBands { bands }
    }
}

fn slot(serial: u32) -> usize {
    serial as usize % STAMPS
}

// The serial numbered after `serial`, skipping 0, which marks a readback slot
// nothing was written to.
fn after(serial: u32) -> u32 {
    match serial.wrapping_add(1) {
        0 => 1,
        next => next,
    }
}

// How many measurements were numbered from `from` to `to`.
fn steps(from: u32, to: u32) -> u32 {
    to.wrapping_sub(from) - u32::from(to < from)
}

// Whether serial `a` was numbered after `b`, across the counter wrapping.
fn newer(a: u32, b: u32) -> bool {
    a != b && a.wrapping_sub(b) < u32::MAX / 2
}

fn same_bands(a: &[[f32; 2]; NUM_SHADOW_CASCADES], b: &[[f32; 2]; NUM_SHADOW_CASCADES]) -> bool {
    a.iter()
        .zip(b)
        .all(|(x, y)| (0..2).all(|i| close(x[i], y[i])))
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-4 * a.abs().max(b.abs()).max(1.0)
}

// The camera's right, up and forward in the world, read from its view matrix.
fn camera_basis(view: &Mat4) -> [[f32; 3]; 3] {
    [
        [view[0][0], view[1][0], view[2][0]],
        [view[0][1], view[1][1], view[2][1]],
        [-view[0][2], -view[1][2], -view[2][2]],
    ]
}

// The angle of the rotation taking orthonormal basis `a` to `b`.
fn turn_between(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> f32 {
    let trace: f32 = (0..3).map(|k| dot(a[k], b[k])).sum();
    acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))
}

// The range along each of `onto` of the box spanning `ranges` along the
// orthonormal `axes`.
fn project_box(
    axes: &[[f32; 3]; 3],
    ranges: &[[f32; 2]; 3],
    onto: &[[f32; 3]; 3],
) -> [[f32; 2]; 3] {
    let center = (0..3).fold([0.0; 3], |c, k| {
        add(c, scale(axes[k], (ranges[k][0] + ranges[k][1]) * 0.5))
    });
    onto.map(|b| {
        let mid = dot(b, center);
        let half: f32 = (0..3)
            .map(|k| dot(b, axes[k]).abs() * (ranges[k][1] - ranges[k][0]) * 0.5)
            .sum();
        [mid - half, mid + half]
    })
}

fn box_corners((lo, hi): Aabb) -> [[f32; 3]; 8] {
    core::array::from_fn(|i| {
        core::array::from_fn(|a| if i & (1 << a) == 0 { lo[a] } else { hi[a] })
    })
}

#[cfg(test)]
pub(crate) mod tests;
