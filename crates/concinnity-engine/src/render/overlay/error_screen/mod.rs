//! A screen that reports a fatal startup error and offers a Quit button: its
//! layout, the input that dismisses it, and the text-only frame it draws.

mod layout;

use crate::gfx::view_modes::ShowFlags;
use crate::gfx::view_modes::ViewMode;
pub use layout::{Layout, build};

use crate::components::{InputKey, KeyEvent};
use crate::input::snapshot::InputSnapshot;
use crate::render::backend::FrameParams;

/// The frame drawing `text_calls` alone over a hidden world, `elapsed`
/// seconds into the screen.
pub fn frame_params(
    text_calls: &[crate::gfx::render_types::TextDrawCall],
    elapsed: f64,
) -> FrameParams<'_> {
    FrameParams {
        input_sampled_us: None,
        elapsed,
        water_time: 0.0,
        fov_y_radians: 1.0,
        near: 0.1,
        view_distance: None,
        cam_pos: [0.0, 0.0, 0.0],
        text_calls,
        lines: &[],
        world_hidden: true,
        view_mode: ViewMode::default(),
        show: ShowFlags::default(),
        sky_rot: crate::sky::SkyOrientation::IDENTITY_ROWS,
        history_reset: false,
        rebase: None,
        grass_benders: &[],
        water_stirs: &[],
        particle_bursts: &[],
        atmosphere: None,
        daylight: &crate::render::uniforms::DaylightBlock::OFF,
        sunlight: 1.0,
        prop_probe: None,
        capture: None,
    }
}

/// Whether this frame's input dismisses the screen: Escape, Return, or a click
/// while the cursor is over Quit.
pub fn dismissed(input: &InputSnapshot, hovered: bool) -> bool {
    input.escape
        || (input.left_click && hovered)
        || input
            .key_events
            .iter()
            .any(|e| matches!(e, KeyEvent::Press(p) if p.key == InputKey::Enter))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_dismisses() {
        let input = InputSnapshot {
            escape: true,
            ..Default::default()
        };
        assert!(dismissed(&input, false));
    }

    #[test]
    fn a_click_over_quit_dismisses() {
        let input = InputSnapshot {
            left_click: true,
            ..Default::default()
        };
        assert!(dismissed(&input, true));
    }

    #[test]
    fn a_click_away_from_quit_does_not_dismiss() {
        let input = InputSnapshot {
            left_click: true,
            ..Default::default()
        };
        assert!(!dismissed(&input, false));
    }

    #[test]
    fn enter_dismisses() {
        let input = InputSnapshot {
            key_events: alloc::vec![KeyEvent::press(InputKey::Enter)],
            ..Default::default()
        };
        assert!(dismissed(&input, false));
    }

    #[test]
    fn a_frame_without_a_dismissal_does_not_dismiss() {
        assert!(!dismissed(&InputSnapshot::default(), true));
    }
}
