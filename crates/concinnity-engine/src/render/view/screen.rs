//! Where on the screen a thing in the world can show.

use crate::gfx::frustum::Aabb;
use crate::transform::Mat4;

// The least clip-space w a box corner may have and still be placed on the
// screen: a box reaching to the camera's plane can cover all of it.
const MIN_W: f32 = 1e-6;

/// A rectangle of the screen in texture coordinates: `u` across from the
/// left edge, `v` down from the top, each in `[0, 1]` on screen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenRect {
    /// The least corner.
    pub min: [f32; 2],
    /// The greatest corner.
    pub max: [f32; 2],
}

impl ScreenRect {
    /// The whole screen.
    pub const FULL: Self = Self {
        min: [0.0; 2],
        max: [1.0; 2],
    };

    /// No part of the screen.
    pub const EMPTY: Self = Self {
        min: [1.0; 2],
        max: [0.0; 2],
    };

    /// The part of the screen the box `bounds` can cover, seen through the
    /// view-projection `vp`: the whole screen for a box reaching to the
    /// camera's plane.
    pub fn of_box(vp: &Mat4, (lo, hi): Aabb) -> Self {
        let mut rect = Self::EMPTY;
        for corner in 0..8u8 {
            let p: [f32; 3] =
                core::array::from_fn(|k| if corner >> k & 1 == 0 { lo[k] } else { hi[k] });
            let clip: [f32; 4] = core::array::from_fn(|r| {
                vp[0][r] * p[0] + vp[1][r] * p[1] + vp[2][r] * p[2] + vp[3][r]
            });
            if clip[3] <= MIN_W {
                return Self::FULL;
            }
            let uv = [clip[0] / clip[3] * 0.5 + 0.5, 0.5 - clip[1] / clip[3] * 0.5];
            rect = rect.union(Self { min: uv, max: uv });
        }
        rect.clamped()
    }

    /// The least rectangle holding both.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        Self {
            min: core::array::from_fn(|k| self.min[k].min(other.min[k])),
            max: core::array::from_fn(|k| self.max[k].max(other.max[k])),
        }
    }

    /// Whether it holds no part of the screen.
    pub fn is_empty(&self) -> bool {
        (0..2).any(|k| self.min[k] >= self.max[k])
    }

    // The part of it on screen.
    fn clamped(self) -> Self {
        Self {
            min: self.min.map(|v| v.clamp(0.0, 1.0)),
            max: self.max.map(|v| v.clamp(0.0, 1.0)),
        }
    }
}
