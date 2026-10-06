//! The per-frame camera and timing inputs an upscale dispatch takes.

use std::cell::Cell;

/// One frame's temporal and camera parameters, shared with the jittered
/// projection so the rasterized scene and the reconstruction agree.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct UpscaleCamera {
    /// Sub-pixel jitter of this frame's projection, in render pixels.
    pub(crate) jitter_offset: [f32; 2],
    /// The frame's elapsed-seconds stamp.
    pub(crate) elapsed: f32,
    /// Distance to the near plane, at least a millimeter.
    pub(crate) near: f32,
    pub(crate) fov_y_radians: f32,
}

impl UpscaleCamera {
    pub(crate) fn new(
        jitter_offset: [f32; 2],
        elapsed: f32,
        near: f32,
        fov_y_radians: f32,
    ) -> Self {
        Self {
            jitter_offset,
            elapsed,
            near: near.max(1e-3),
            fov_y_radians,
        }
    }
}

/// The time between consecutive dispatches of one upscaler context.
#[derive(Debug, Default)]
pub(super) struct FrameClock {
    prev_elapsed: Cell<f32>,
}

impl FrameClock {
    /// Milliseconds since the previous call, clamped to `[1, 100]`. The first
    /// call measures from zero, and a stalled frame can be arbitrarily long,
    /// while the temporal heuristics expect frame-time-sized values.
    pub(super) fn delta_ms(&self, now: f32) -> f32 {
        let last = self.prev_elapsed.replace(now);
        ((now - last) * 1000.0).clamp(1.0, 100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_delta_is_clamped() {
        let clock = FrameClock::default();
        assert!((clock.delta_ms(10.0) - 100.0).abs() < 1e-3);
        assert!((clock.delta_ms(10.016) - 16.0).abs() < 1e-2);
        assert!((clock.delta_ms(10.016) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn near_plane_is_kept_off_zero() {
        assert_eq!(UpscaleCamera::new([0.0; 2], 0.0, 0.0, 1.0).near, 1e-3);
        assert_eq!(UpscaleCamera::new([0.0; 2], 0.0, 0.5, 1.0).near, 0.5);
    }
}
