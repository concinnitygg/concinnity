//! The square of cells the wake covers around the camera, addressed
//! toroidally: a cell's slot is its coordinates wrapped by the window's size,
//! so a cell the window keeps stays in the slot it was written to as the
//! window moves, and only the cells it newly covers start still.

use super::{WAKE_CELL_SIZE, WAKE_RESOLUTION};
use concinnity_core::math::{SaturatingFrom, floor};

/// The window of cells the wake covers: [`WAKE_RESOLUTION`] cells along
/// each axis from the cell at `origin`, in the simulated frame's `x` and `z`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WakeWindow {
    /// The cell at the window's minimum corner.
    pub origin: [i32; 2],
}

impl WakeWindow {
    /// The window centered, to the cell, on `xz`.
    pub fn around(xz: [f32; 2]) -> Self {
        let half = i32::saturating_from(WAKE_RESOLUTION / 2);
        let [x, z] = cell_of(xz);
        Self {
            origin: [x - half, z - half],
        }
    }

    /// Whether cell `cell` lies inside the window.
    pub fn contains(&self, cell: [i32; 2]) -> bool {
        let n = i32::saturating_from(WAKE_RESOLUTION);
        (0..2).all(|i| cell[i] >= self.origin[i] && cell[i] < self.origin[i] + n)
    }

    /// The window's center, in meters.
    pub fn center(&self) -> [f32; 2] {
        let half = (WAKE_RESOLUTION / 2) as f32;
        self.origin.map(|o| (o as f32 + half) * WAKE_CELL_SIZE)
    }

    /// Whether a disc at `xz` of radius `reach` meets the window.
    pub fn touches(&self, xz: [f32; 2], reach: f32) -> bool {
        let half = 0.5 * WAKE_RESOLUTION as f32 * WAKE_CELL_SIZE + reach;
        let c = self.center();
        (xz[0] - c[0]).abs() < half && (xz[1] - c[1]).abs() < half
    }
}

/// The cell `xz` falls in.
pub fn cell_of(xz: [f32; 2]) -> [i32; 2] {
    xz.map(|v| i32::saturating_from(floor(v / WAKE_CELL_SIZE)))
}

/// The slot cell `cell` is kept in, whichever window holds it: its
/// coordinates wrapped by the window's size, row-major. Mirrors
/// `wake_slot` in `water_wake.hlsl`.
pub fn slot_of(cell: [i32; 2]) -> usize {
    let mask = i32::saturating_from(WAKE_RESOLUTION) - 1;
    usize::saturating_from((cell[0] & mask) + (cell[1] & mask) * (mask + 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: i32 = WAKE_RESOLUTION.cast_signed();

    #[test]
    fn the_window_centers_on_its_point() {
        let w = WakeWindow::around([0.1, -0.1]);
        assert_eq!(w.origin, [-N / 2, -N / 2 - 1]);
        assert!(w.contains([0, -1]));
        assert!(!w.contains([N / 2, 0]));
        assert!(w.contains([N / 2 - 1, 0]));
        let c = w.center();
        assert_eq!(c, [0.0, -WAKE_CELL_SIZE]);
        assert!(w.touches([31.0, 0.0], 0.0));
        assert!(!w.touches([33.0, 0.0], 0.0));
        assert!(w.touches([33.0, 0.0], 2.0));
    }

    // Every cell of a window has a slot of its own, and a cell keeps its slot
    // when the window moves on.
    #[test]
    fn slots_wrap_and_stay_put() {
        let w = WakeWindow::around([3.0, 7.0]);
        let mut seen = alloc::vec![false; (N * N) as usize];
        for z in 0..N {
            for x in 0..N {
                let slot = slot_of([w.origin[0] + x, w.origin[1] + z]);
                assert!(!seen[slot]);
                seen[slot] = true;
            }
        }
        assert_eq!(slot_of([5, -3]), slot_of([5 + N, -3 - 2 * N]));
        assert_eq!(slot_of([-1, 0]), (N - 1) as usize);
    }
}
