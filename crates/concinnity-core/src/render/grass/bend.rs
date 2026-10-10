//! The bend field trampled blades spring back from: a square window of cells
//! around the camera, each holding how far the blades rooted in it are pressed
//! over, `[x, z]` per meter of blade. Characters and bodies standing in the
//! grass stamp their footprints into it every frame; between stamps every cell
//! relaxes toward upright over a few seconds.
//!
//! The window follows the camera a whole cell at a time and is addressed
//! toroidally: a cell's slot is its world cell coordinates wrapped by the
//! window's size, so a cell the window keeps stays in the slot it was written
//! to, and only the cells it newly covers, whose slots last held cells it just
//! left, start upright. The GPU keeps two copies, this frame's and last
//! frame's, so a blade's previous bend, and with it its motion vector, is
//! always at hand. The bend pass and the blade kernel in `grass.hlsl` mirror
//! the addressing and the stamp below.

use alloc::vec::Vec;

use crate::math::{exp, floor, sqrt};
use crate::render::uniforms::grass::MAX_GRASS_STAMPS;

/// Cells along each side of the window. A power of two, so wrapping a cell
/// coordinate is a mask.
pub const GRASS_BEND_RESOLUTION: u32 = 256;

/// Edge of one cell, in meters.
pub const GRASS_BEND_CELL_SIZE: f32 = 0.25;

/// Seconds over which a pressed blade recovers all but `1/e` of its bend.
pub const GRASS_BEND_RECOVERY: f32 = 1.0;

/// The longest frame the field relaxes over at once, in seconds, so a stall
/// does not stand the whole field up in one frame.
pub const GRASS_BEND_MAX_STEP: f32 = 0.25;

/// How far past its radius a footprint presses blades, as a multiple of it.
pub const GRASS_STAMP_REACH: f32 = 1.5;

/// Where a footprint's press starts to fade, as a multiple of its radius.
pub const GRASS_STAMP_CORE: f32 = 0.8;

/// The bend a footprint at full strength stores, per meter of blade; a blade
/// pressed this far lies flat.
pub const GRASS_TRAMPLE_BEND: f32 = 0.85;

/// Words in the bend buffer: two halves of one packed bend per cell.
pub const GRASS_BEND_WORDS: usize = 2 * (GRASS_BEND_RESOLUTION * GRASS_BEND_RESOLUTION) as usize;

/// Something that tramples grass: an upright shape standing on or over the
/// ground, as wide as `radius` around its axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrassBender {
    /// Its lowest point, on its vertical axis, in world space.
    pub base: [f32; 3],
    /// Horizontal radius, in meters.
    pub radius: f32,
    /// Height from `base` to its top, in meters.
    pub height: f32,
}

/// The window of cells the field covers: [`GRASS_BEND_RESOLUTION`] cells
/// along each axis from the cell at `origin`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BendWindow {
    /// The world cell at the window's minimum corner.
    pub origin: [i32; 2],
}

impl BendWindow {
    /// The window centered, to the cell, on world `position`.
    pub fn around(position: [f32; 3]) -> Self {
        let half = (GRASS_BEND_RESOLUTION / 2) as i32;
        let [x, z] = cell_of([position[0], position[2]]);
        Self {
            origin: [x - half, z - half],
        }
    }

    /// Whether world cell `cell` lies inside the window.
    pub fn contains(&self, cell: [i32; 2]) -> bool {
        let n = GRASS_BEND_RESOLUTION as i32;
        (0..2).all(|i| cell[i] >= self.origin[i] && cell[i] < self.origin[i] + n)
    }

    /// The world cell the window holds in slot `slot`.
    pub fn cell_at_slot(&self, slot: usize) -> [i32; 2] {
        let n = GRASS_BEND_RESOLUTION as i32;
        let s = [slot as i32 % n, slot as i32 / n];
        core::array::from_fn(|i| self.origin[i] + (s[i] - self.origin[i]).rem_euclid(n))
    }

    /// The window's ground rectangle, `[min x, min z, max x, max z]`.
    pub fn rect(&self) -> [f32; 4] {
        let n = GRASS_BEND_RESOLUTION as i32;
        let at = |c: i32| c as f32 * GRASS_BEND_CELL_SIZE;
        [
            at(self.origin[0]),
            at(self.origin[1]),
            at(self.origin[0] + n),
            at(self.origin[1] + n),
        ]
    }
}

/// The world cell world `xz` falls in.
pub fn cell_of(xz: [f32; 2]) -> [i32; 2] {
    xz.map(|v| floor(v / GRASS_BEND_CELL_SIZE) as i32)
}

/// The slot world cell `cell` is kept in, whichever window holds it: its
/// coordinates wrapped by the window's size, row-major.
pub fn slot_of(cell: [i32; 2]) -> usize {
    let mask = GRASS_BEND_RESOLUTION as i32 - 1;
    ((cell[0] & mask) + (cell[1] & mask) * GRASS_BEND_RESOLUTION as i32) as usize
}

/// What a bend is scaled by as it springs back over `dt` seconds.
pub fn decay(dt: f32) -> f32 {
    exp(-dt.clamp(0.0, GRASS_BEND_MAX_STEP) / GRASS_BEND_RECOVERY)
}

/// The bend a footprint `stamp` (center `x`, `z`, radius, strength) presses
/// into the blades at world `xz`: outward from its center, full within its
/// core, fading to nothing at its reach. Mirrors `grass_stamp_bend`.
pub fn stamp_bend(stamp: [f32; 4], xz: [f32; 2]) -> [f32; 2] {
    let [cx, cz, radius, strength] = stamp;
    let d = [xz[0] - cx, xz[1] - cz];
    let dist = sqrt(d[0] * d[0] + d[1] * d[1]);
    let inner = radius * GRASS_STAMP_CORE;
    let outer = radius * GRASS_STAMP_REACH;
    let t = ((dist - inner) / (outer - inner).max(1e-4)).clamp(0.0, 1.0);
    let falloff = 1.0 - t * t * (3.0 - 2.0 * t);
    let dir = if dist > 1e-4 {
        [d[0] / dist, d[1] / dist]
    } else {
        [1.0, 0.0]
    };
    let k = GRASS_TRAMPLE_BEND * strength * falloff;
    [dir[0] * k, dir[1] * k]
}

/// The footprint `bender` stamps where the ground under it is at `ground_y`
/// and the tallest blades are `blade_height` tall: center, radius and
/// strength, the strength falling from 1 standing on the ground to nothing
/// once it clears the blades. `None` when it presses nothing.
pub fn stamp_of(bender: &GrassBender, ground_y: f32, blade_height: f32) -> Option<[f32; 4]> {
    let gap = bender.base[1] - ground_y;
    let top = gap + bender.height;
    if bender.radius <= 0.0 || top < 0.0 || blade_height <= 0.0 {
        return None;
    }
    let strength = 1.0 - (gap / blade_height).clamp(0.0, 1.0);
    (strength > 0.0).then_some([bender.base[0], bender.base[2], bender.radius, strength])
}

/// The footprints this frame stamps into `window`: each bender's, where
/// `ground` finds terrain under it, whose reach touches the window, nearest
/// `camera` first, at most [`MAX_GRASS_STAMPS`] of them.
pub fn gather_stamps(
    benders: &[GrassBender],
    window: &BendWindow,
    camera: [f32; 3],
    blade_height: f32,
    ground: impl Fn([f32; 2]) -> Option<f32>,
) -> Vec<[f32; 4]> {
    let [x0, z0, x1, z1] = window.rect();
    let mut stamps: Vec<(f32, [f32; 4])> = benders
        .iter()
        .filter_map(|b| {
            let ground_y = ground([b.base[0], b.base[2]])?;
            let stamp = stamp_of(b, ground_y, blade_height)?;
            let reach = stamp[2] * GRASS_STAMP_REACH;
            let inside = stamp[0] + reach > x0
                && stamp[0] - reach < x1
                && stamp[1] + reach > z0
                && stamp[1] - reach < z1;
            let dx = stamp[0] - camera[0];
            let dz = stamp[1] - camera[2];
            inside.then_some((dx * dx + dz * dz, stamp))
        })
        .collect();
    stamps.sort_by(|a, b| a.0.total_cmp(&b.0));
    stamps
        .into_iter()
        .take(MAX_GRASS_STAMPS)
        .map(|(_, s)| s)
        .collect()
}

/// One frame's step of the field: where its window lies now and lay last
/// frame, how far it relaxes, and which half of the buffer it writes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BendStep {
    /// This frame's window.
    pub window: BendWindow,
    /// Last frame's window, `None` on the first frame, when the other half
    /// holds nothing.
    pub prev: Option<BendWindow>,
    /// What last frame's bend is scaled by.
    pub decay: f32,
    /// The half of the buffer this frame writes.
    pub write_half: u32,
}

/// What the field remembers between frames.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BendHistory {
    last: Option<(BendWindow, f32)>,
    half: u32,
}

impl BendHistory {
    /// Advance to a frame at `elapsed` seconds with the camera at `position`.
    pub fn advance(&mut self, position: [f32; 3], elapsed: f32) -> BendStep {
        let window = BendWindow::around(position);
        let (prev, decay) = match self.last {
            Some((prev, at)) => (Some(prev), decay(elapsed - at)),
            None => (None, 1.0),
        };
        let write_half = self.half;
        self.half ^= 1;
        self.last = Some((window, elapsed));
        BendStep {
            window,
            prev,
            decay,
            write_half,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: i32 = GRASS_BEND_RESOLUTION as i32;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn the_window_centers_on_the_camera_cell() {
        let w = BendWindow::around([0.1, 5.0, -0.1]);
        assert_eq!(w.origin, [-N / 2, -1 - N / 2]);
        assert!(w.contains([0, -1]));
        assert!(w.contains(w.origin));
        assert!(!w.contains([w.origin[0] + N, 0]));
        assert!(!w.contains([w.origin[0] - 1, 0]));
        let [x0, z0, x1, z1] = w.rect();
        assert!(close(x1 - x0, N as f32 * GRASS_BEND_CELL_SIZE));
        assert!(close(z1 - z0, N as f32 * GRASS_BEND_CELL_SIZE));
        assert!(x0 < 0.1 && 0.1 < x1 && z0 < -0.1 && -0.1 < z1);
    }

    // Toroidal addressing: every slot of a window holds one of its cells, and
    // each cell of the window sits in exactly the slot `slot_of` names.
    #[test]
    fn every_slot_holds_one_cell_of_the_window() {
        let w = BendWindow { origin: [-300, 17] };
        let mut cells = alloc::collections::BTreeSet::new();
        for slot in 0..(N * N) as usize {
            let cell = w.cell_at_slot(slot);
            assert!(w.contains(cell), "slot {slot} -> {cell:?}");
            assert_eq!(slot_of(cell), slot);
            assert!(cells.insert(cell), "slot {slot} repeats {cell:?}");
        }
    }

    // A cell the window keeps as it scrolls stays in its slot, and a slot whose
    // cell scrolled out now holds a cell the old window did not cover, which is
    // what the bend pass zeroes.
    #[test]
    fn scrolling_keeps_shared_cells_in_place() {
        let a = BendWindow::around([0.0, 0.0, 0.0]);
        let b = BendWindow::around([3.3, 0.0, -10.1]);
        assert_eq!(b.origin, [a.origin[0] + 13, a.origin[1] - 41]);
        let mut fresh = 0;
        for slot in 0..(N * N) as usize {
            let now = b.cell_at_slot(slot);
            if a.contains(now) {
                assert_eq!(a.cell_at_slot(slot), now);
            } else {
                fresh += 1;
            }
        }
        assert_eq!(fresh, N * N - (N - 13) * (N - 41));
    }

    #[test]
    fn negative_cells_wrap_into_the_window() {
        assert_eq!(cell_of([-0.01, -0.26]), [-1, -2]);
        assert_eq!(slot_of([-1, 0]), (N - 1) as usize);
        assert_eq!(slot_of([0, -1]), ((N - 1) * N) as usize);
        assert_eq!(slot_of([N, N]), 0);
    }

    #[test]
    fn the_field_relaxes_over_the_recovery_time() {
        assert_eq!(decay(0.0), 1.0);
        assert_eq!(decay(-1.0), 1.0);
        assert!(close(decay(0.1), exp(-0.1 / GRASS_BEND_RECOVERY)));
        // Halving the step twice is the same as one whole step.
        assert!(close(decay(0.05) * decay(0.05), decay(0.1)));
        // A stall relaxes the field one capped step, not all the way.
        assert_eq!(decay(5.0), decay(GRASS_BEND_MAX_STEP));
        // A few seconds of frames stands it all but upright.
        let mut bend = GRASS_TRAMPLE_BEND;
        for _ in 0..(60 * 4) {
            bend *= decay(1.0 / 60.0);
        }
        assert!(bend < 0.02, "{bend}");
    }

    #[test]
    fn the_history_alternates_halves_and_times_the_decay() {
        let mut h = BendHistory::default();
        let first = h.advance([0.0; 3], 2.0);
        assert_eq!(first.prev, None);
        assert_eq!(first.decay, 1.0);
        assert_eq!(first.write_half, 0);
        let second = h.advance([1.0, 0.0, 0.0], 2.1);
        assert_eq!(second.prev, Some(first.window));
        assert!(close(second.decay, decay(0.1)));
        assert_eq!(second.write_half, 1);
        assert_eq!(h.advance([1.0, 0.0, 0.0], 2.2).write_half, 0);
    }

    #[test]
    fn a_footprint_presses_outward_and_fades_at_its_reach() {
        let stamp = [2.0, -1.0, 0.5, 1.0];
        let inside = stamp_bend(stamp, [2.2, -1.0]);
        assert!(close(inside[0], GRASS_TRAMPLE_BEND));
        assert!(close(inside[1], 0.0));
        let edge = stamp_bend(stamp, [2.0, -1.0 - 0.5 * GRASS_STAMP_REACH]);
        assert!(close(edge[0], 0.0) && close(edge[1], 0.0));
        let between = stamp_bend(stamp, [2.0, -1.0 + 0.5]);
        assert!(between[1] > 0.0 && between[1] < GRASS_TRAMPLE_BEND);
        let half = stamp_bend([2.0, -1.0, 0.5, 0.5], [2.2, -1.0]);
        assert!(close(half[0], 0.5 * GRASS_TRAMPLE_BEND));
    }

    #[test]
    fn a_bender_presses_as_hard_as_it_is_low() {
        let on = GrassBender {
            base: [1.0, 3.0, 2.0],
            radius: 0.4,
            height: 1.8,
        };
        assert_eq!(stamp_of(&on, 3.0, 0.5), Some([1.0, 2.0, 0.4, 1.0]));
        let hover = GrassBender {
            base: [1.0, 3.25, 2.0],
            ..on
        };
        assert_eq!(stamp_of(&hover, 3.0, 0.5).map(|s| s[3]), Some(0.5));
        let clear = GrassBender {
            base: [1.0, 3.5, 2.0],
            ..on
        };
        assert_eq!(stamp_of(&clear, 3.0, 0.5), None);
        // Sunk into the ground it still presses, until it is wholly under it.
        assert_eq!(stamp_of(&on, 4.0, 0.5).map(|s| s[3]), Some(1.0));
        assert_eq!(stamp_of(&on, 5.0, 0.5), None);
    }

    #[test]
    fn stamps_are_culled_to_the_window_and_nearest_first() {
        let window = BendWindow::around([0.0; 3]);
        let at = |x: f32, z: f32| GrassBender {
            base: [x, 0.0, z],
            radius: 0.3,
            height: 1.0,
        };
        let far = N as f32 * GRASS_BEND_CELL_SIZE;
        let benders = [at(5.0, 0.0), at(far, 0.0), at(1.0, 1.0), at(-3.0, 50.0)];
        // The last has no ground under it.
        let ground = |xz: [f32; 2]| (xz[1] < 40.0).then_some(0.0);
        let stamps = gather_stamps(&benders, &window, [0.0; 3], 0.5, ground);
        assert_eq!(stamps.len(), 2);
        assert_eq!(&stamps[0][..2], &[1.0, 1.0]);
        assert_eq!(&stamps[1][..2], &[5.0, 0.0]);
    }

    #[test]
    fn at_most_the_stamp_limit_is_gathered() {
        let window = BendWindow::around([0.0; 3]);
        let benders: Vec<GrassBender> = (0..MAX_GRASS_STAMPS + 10)
            .map(|i| GrassBender {
                base: [i as f32 * 0.1, 0.0, 0.0],
                radius: 0.3,
                height: 1.0,
            })
            .collect();
        let stamps = gather_stamps(&benders, &window, [0.0; 3], 0.5, |_| Some(0.0));
        assert_eq!(stamps.len(), MAX_GRASS_STAMPS);
        assert_eq!(stamps[0][0], 0.0);
        assert!(stamps.iter().all(|s| s[0] < MAX_GRASS_STAMPS as f32 * 0.1));
    }
}
