//! The pre-exposure handed to a temporal upscaler that divides its input by
//! one before accumulating it.
//!
//! Handing over the frame's full [`PreExposure`] would take the upscaler's
//! history back to absolute radiance, which leaves FP16's normal range in a
//! dim scene. Instead the upscaler is handed the frame's pre-exposure relative
//! to a reference that moves in large steps: frame-to-frame exposure changes
//! still reach it exactly, while what it accumulates stays within a few stops
//! of the pre-exposed chain. A step invalidates its history, so the caller
//! resets the upscaler whenever [`UpscalerExposure::advance`] reports one.

use super::PreExposure;

// How far the handed-over pre-exposure may drift from 1 before the reference
// steps (4 stops either way).
const WINDOW: f32 = 16.0;
// Where a step lands the handed-over value, on the side the exposure came
// from, so it can keep moving the same way for 6 stops before the next step.
const LANDING: f32 = 4.0;

/// A reference pre-exposure that follows the frame's in large steps, and the
/// ratio between the two an upscaler is handed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UpscalerExposure {
    reference: Option<f32>,
    value: f32,
}

impl Default for UpscalerExposure {
    fn default() -> Self {
        Self {
            reference: None,
            value: 1.0,
        }
    }
}

impl UpscalerExposure {
    /// Moves to the frame written at `pre`. Returns whether the reference
    /// stepped, which invalidates the upscaler's history.
    pub fn advance(&mut self, pre: &PreExposure) -> bool {
        let current = pre.value();
        let (reference, stepped) = match self.reference {
            None => (current, false),
            Some(r) if current > r * WINDOW => (current * LANDING, true),
            Some(r) if current * WINDOW < r => (current / LANDING, true),
            Some(r) => (r, false),
        };
        self.reference = Some(reference);
        self.value = current / reference;
        stepped
    }

    /// The pre-exposure the upscaler divides this frame's input by.
    pub fn value(&self) -> f32 {
        self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stepper(start: f32) -> (UpscalerExposure, PreExposure) {
        let pre = PreExposure::new(start);
        let mut up = UpscalerExposure::default();
        assert!(!up.advance(&pre));
        (up, pre)
    }

    #[test]
    fn nothing_advanced_hands_over_unit() {
        assert_eq!(UpscalerExposure::default().value(), 1.0);
    }

    #[test]
    fn the_first_frame_is_its_own_reference() {
        let (up, _) = stepper(2f32.powi(16));
        assert_eq!(up.value(), 1.0);
    }

    #[test]
    fn small_changes_reach_the_upscaler_without_a_step() {
        let (mut up, mut pre) = stepper(8.0);
        pre.advance(16.0);
        assert!(!up.advance(&pre));
        assert_eq!(up.value(), 2.0);
        pre.advance(1.0);
        assert!(!up.advance(&pre));
        assert_eq!(up.value(), 0.125);
    }

    #[test]
    fn leaving_the_window_steps_and_lands_on_the_side_it_came_from() {
        let (mut up, mut pre) = stepper(1.0);
        pre.advance(WINDOW * 2.0);
        assert!(up.advance(&pre));
        assert_eq!(up.value(), 1.0 / LANDING);
        let (mut up, mut pre) = stepper(1.0);
        pre.advance(1.0 / (WINDOW * 2.0));
        assert!(up.advance(&pre));
        assert_eq!(up.value(), LANDING);
    }

    #[test]
    fn a_day_to_night_ramp_stays_in_range_with_few_steps() {
        // 29 stops from a sunlit EV to a night one, a twentieth of a stop
        // per frame, and back.
        let (mut up, mut pre) = stepper(2f32.powi(-13));
        let mut steps = 0;
        let frames = (0..580)
            .map(|i| i as f32)
            .chain((0..580).rev().map(|i| i as f32));
        for i in frames {
            pre.advance(2f32.powf(-13.0 + i / 20.0));
            steps += usize::from(up.advance(&pre));
            assert!(
                (1.0 / WINDOW..=WINDOW).contains(&up.value()),
                "{}",
                up.value()
            );
        }
        assert!(steps <= 12, "{steps} steps");
    }
}
