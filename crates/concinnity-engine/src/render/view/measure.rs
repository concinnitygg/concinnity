//! The view's measurement from the frame's depth: each frame's samples are
//! numbered and stamped with the camera and subjects they were taken through,
//! and placed on what they show once the GPU hands them back.

use super::facts::{Seen, Sighting, VIEW_SUBJECTS};
use super::subjects::{Hit, ViewSubjects};
use crate::gfx::render_types::{ViewSampleCell, ViewSampleGrid, ViewSampleParams};
use crate::render::depth::DEPTH_FAR;
use crate::transform::Mat4;
use concinnity_core::math::vec3::{normalize_or, scale, sub};

/// How many frames old a measurement may be and still stand for what the
/// camera sees: more than the frames any backend keeps in flight.
pub(crate) const MAX_MEASURED_AGE: u32 = 6;

// How many frames' stamps are kept: enough for the oldest usable measurement
// and every frame since.
const STAMPS: usize = MAX_MEASURED_AGE as usize + 2;

// Smallest homogeneous w a reconstructed depth point may have:
// `DEPTH_RECONSTRUCT_MIN_W` in `depth_convention.hlsl`.
const RECONSTRUCT_MIN_W: f32 = 1e-7;

/// The camera and subjects a frame's depth is sampled through.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MeasureView {
    /// The inverse of the view-projection the depth was drawn through.
    pub depth_inv_vp: Mat4,
    /// The camera's position.
    pub eye: [f32; 3],
    /// The depth's size in texels.
    pub size: [u32; 2],
    /// The subjects in the frame.
    pub subjects: ViewSubjects,
}

#[derive(Clone, Copy, Debug)]
struct Stamp {
    serial: u32,
    view: MeasureView,
    grid: ViewSampleGrid,
    spot: [u32; 2],
}

/// What one measurement saw.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Measured {
    /// The share of the screen showing sky.
    pub sky: f32,
    /// The surfaces no subject holds.
    pub other: Sighting,
    /// Each subject, by slot.
    pub subjects: [Sighting; VIEW_SUBJECTS],
}

/// Numbers each frame's samples, keeps what they were taken through, and
/// holds the newest measurement the GPU returned.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ViewMeasure {
    next: u32,
    stamps: [Option<Stamp>; STAMPS],
    held: Option<(u32, Measured)>,
}

impl Default for ViewMeasure {
    fn default() -> Self {
        Self {
            next: 1,
            stamps: [None; STAMPS],
            held: None,
        }
    }
}

impl ViewMeasure {
    /// Stamp this frame's samples through `view` and return their pass's
    /// parameters.
    pub(crate) fn begin(&mut self, view: &MeasureView) -> ViewSampleParams {
        let serial = self.next;
        self.next = after(serial);
        let [width, height] = view.size;
        let grid = ViewSampleGrid::over(width, height);
        let params = ViewSampleParams::new(width, height, grid, serial);
        self.stamps[slot(serial)] = Some(Stamp {
            serial,
            view: *view,
            grid,
            spot: params.spot,
        });
        params
    }

    /// Take samples the GPU finished: measured when their stamp is still
    /// known, every cell of its grid carries its number, and they are newer
    /// than the measurement held. Serial 0 is never stamped, so a readback
    /// nothing was written to is never taken.
    pub(crate) fn accept(&mut self, cells: &[ViewSampleCell]) {
        let Some(serial) = cells.first().map(|c| c.serial).filter(|s| *s != 0) else {
            return;
        };
        let Some(stamp) = self.stamps[slot(serial)].filter(|s| s.serial == serial) else {
            return;
        };
        if self.held.is_some_and(|(h, _)| !newer(serial, h)) {
            return;
        }
        let Some(cells) = cells.get(..stamp.grid.cells()) else {
            return;
        };
        if cells.iter().any(|c| c.serial != serial) {
            return;
        }
        self.held = Some((serial, measure(&stamp, cells)));
    }

    /// Forget every stamp and measurement, for a view that does not continue
    /// the last. Numbering carries on, so samples still in flight are not
    /// taken for later ones.
    pub(crate) fn reset(&mut self) {
        self.stamps = [None; STAMPS];
        self.held = None;
    }

    /// The held measurement and how many frames before the next one it was
    /// taken; `None` when there is none recent enough.
    pub(crate) fn latest(&self) -> Option<(u32, &Measured)> {
        let (serial, measured) = self.held.as_ref()?;
        let age = steps(*serial, self.next);
        (age <= MAX_MEASURED_AGE).then_some((age, measured))
    }

    /// What the held measurement saw beyond the subjects.
    pub(crate) fn seen(&self) -> Option<Seen> {
        self.latest().map(|(age, m)| Seen {
            age,
            drift_px: 0.0,
            sky: m.sky,
            other: m.other,
        })
    }
}

// Place every cell of `cells` on what it shows: its sample counts toward the
// shares, and its nearest texel too toward each nearest distance.
fn measure(stamp: &Stamp, cells: &[ViewSampleCell]) -> Measured {
    let view = &stamp.view;
    let rays = Rays::of(view);
    let mut tally = Tally::default();
    for (i, cell) in cells.iter().enumerate() {
        let at = stamp.grid.sample(i, stamp.spot, view.size[0], view.size[1]);
        let hit = rays.hit(view, at, cell.depth);
        tally.count(hit);
        tally.near(hit);
        let nearest = cell.nearest_texel();
        if cell.nearest > DEPTH_FAR && nearest != at {
            tally.near(rays.hit(view, nearest, cell.nearest));
        }
    }
    tally.measured(cells.len())
}

// The camera rays through a depth's texels: a texel's homogeneous point at
// device depth `d` is `x * across + y * down + start + d * depth`, the inverse
// view-projection's columns folded with the texel-to-NDC step once.
struct Rays {
    across: [f32; 4],
    down: [f32; 4],
    start: [f32; 4],
    depth: [f32; 4],
}

impl Rays {
    fn of(view: &MeasureView) -> Self {
        let [width, height] = view.size.map(|s| s.max(1) as f32);
        let [c0, c1, c2, c3] = view.depth_inv_vp;
        Self {
            across: c0.map(|v| v * 2.0 / width),
            down: c1.map(|v| v * -2.0 / height),
            start: core::array::from_fn(|k| {
                c0[k] * (1.0 / width - 1.0) + c1[k] * (1.0 - 1.0 / height) + c3[k]
            }),
            depth: c2,
        }
    }

    // What the depth sample `depth` at `texel` shows.
    fn hit(&self, view: &MeasureView, [x, y]: [u32; 2], depth: f32) -> Hit {
        let (x, y) = (x as f32, y as f32);
        let at = |d: f32| -> [f32; 4] {
            core::array::from_fn(|k| {
                x * self.across[k] + y * self.down[k] + self.start[k] + d * self.depth[k]
            })
        };
        let away = |h: [f32; 4]| sub([h[0], h[1], h[2]], scale(view.eye, h[3]));
        let dir = normalize_or(away(at(DEPTH_FAR)), 1e-12, [0.0, 0.0, -1.0]);
        let opaque = (depth > DEPTH_FAR).then(|| {
            let h = at(depth);
            scale(away(h), 1.0 / h[3].max(RECONSTRUCT_MIN_W))
        });
        view.subjects.hit(view.eye, dir, opaque)
    }
}

#[derive(Default)]
struct Tally {
    sky: u32,
    other: u32,
    subjects: [u32; VIEW_SUBJECTS],
    other_nearest: Option<f32>,
    nearest: [Option<f32>; VIEW_SUBJECTS],
}

impl Tally {
    fn count(&mut self, hit: Hit) {
        match hit {
            Hit::Sky => self.sky += 1,
            Hit::Subject(slot, _) => self.subjects[slot] += 1,
            Hit::Other(_) => self.other += 1,
        }
    }

    fn near(&mut self, hit: Hit) {
        let (nearest, d) = match hit {
            Hit::Sky => return,
            Hit::Subject(slot, d) => (&mut self.nearest[slot], d),
            Hit::Other(d) => (&mut self.other_nearest, d),
        };
        *nearest = Some(nearest.map_or(d, |n| n.min(d)));
    }

    fn measured(&self, cells: usize) -> Measured {
        let share = |n: u32| n as f32 / cells.max(1) as f32;
        Measured {
            sky: share(self.sky),
            other: Sighting {
                coverage: share(self.other),
                nearest: self.other_nearest,
            },
            subjects: core::array::from_fn(|s| Sighting {
                coverage: share(self.subjects[s]),
                nearest: self.nearest[s],
            }),
        }
    }
}

fn slot(serial: u32) -> usize {
    serial as usize % STAMPS
}

// The serial numbered after `serial`, skipping 0, which marks a readback no
// measurement wrote.
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
