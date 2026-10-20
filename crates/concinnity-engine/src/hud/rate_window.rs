//! The frame-counting window the HUD rate readouts average over.

// Counts frames and accumulates their frame time until an interval has passed.
#[derive(Debug, Default)]
pub(crate) struct RateWindow {
    frames: u32,
    secs: f32,
}

impl RateWindow {
    // Count one frame of `dt` seconds. Once `interval` seconds have accumulated,
    // returns the window's frame count and length and starts a new window.
    pub(crate) fn tick(&mut self, dt: f32, interval: f32) -> Option<(u32, f32)> {
        self.frames += 1;
        self.secs += dt.max(0.0);
        if self.secs < interval {
            return None;
        }
        let window = (self.frames, self.secs);
        *self = Self::default();
        Some(window)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closes_once_the_interval_accumulates() {
        let mut window = RateWindow::default();
        assert_eq!(window.tick(0.25, 1.0), None);
        assert_eq!(window.tick(0.25, 1.0), None);
        assert_eq!(window.tick(0.25, 1.0), None);
        assert_eq!(window.tick(0.25, 1.0), Some((4, 1.0)));
    }

    #[test]
    fn a_closed_window_starts_over() {
        let mut window = RateWindow::default();
        assert_eq!(window.tick(2.0, 1.0), Some((1, 2.0)));
        assert_eq!(window.tick(0.5, 1.0), None);
        assert_eq!(window.tick(0.5, 1.0), Some((2, 1.0)));
    }

    #[test]
    fn a_negative_dt_adds_no_time() {
        let mut window = RateWindow::default();
        assert_eq!(window.tick(-5.0, 1.0), None);
        assert_eq!(window.tick(1.0, 1.0), Some((2, 1.0)));
    }
}
