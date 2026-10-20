//! The merge of a frame's window input and gamepad snapshot into the published
//! `FrameInput`, under the gameplay gate.

use crate::components::{FrameInput, GamepadButton, GamepadMap, NavDirection};
use crate::ecs::FrameTime;

use super::pad::PadSnapshot;
use super::snapshot::InputSnapshot;

// Frame dt ceiling for the nav auto-repeat, so a hitch or debugger pause never
// counts as a long hold.
const NAV_DT_MAX: f32 = 0.25;

/// The nav auto-repeat's share of this frame's time.
pub fn nav_dt(frame: FrameTime) -> f32 {
    frame.dt.clamp(0.0, NAV_DT_MAX)
}

/// Merge the window's keyboard/mouse snapshot with the gamepad snapshot under
/// the gameplay gate. The caller fills in the window metrics.
pub fn compose_frame_input(
    raw: &InputSnapshot,
    pad: &PadSnapshot,
    map: GamepadMap,
    nav: Option<NavDirection>,
    gameplay: bool,
) -> FrameInput {
    let axis = |v: [f32; 2]| if gameplay { v } else { [0.0, 0.0] };
    FrameInput {
        forward: (raw.forward || pad.held(GamepadButton::DpadUp)) && gameplay,
        backward: (raw.backward || pad.held(GamepadButton::DpadDown)) && gameplay,
        left: (raw.left || pad.held(GamepadButton::DpadLeft)) && gameplay,
        right: (raw.right || pad.held(GamepadButton::DpadRight)) && gameplay,
        sprint: (raw.sprint || pad.held(map.sprint)) && gameplay,
        interact: (raw.interact || pad.pressed(map.interact)) && gameplay,
        jump: (raw.jump || pad.pressed(map.jump)) && gameplay,
        move_axis: axis(pad.move_axis),
        look_axis: axis(pad.look_axis),
        mouse_dx: if gameplay { raw.mouse_dx } else { 0.0 },
        mouse_dy: if gameplay { raw.mouse_dy } else { 0.0 },
        // Not gated by `gameplay`: a scrollable menu still scrolls
        // while it is open (the camera is what freezes behind it).
        scroll_delta: raw.scroll_delta,
        mouse_x: raw.mouse_x,
        mouse_y: raw.mouse_y,
        left_click: raw.left_click,
        left_button_down: raw.left_button_down,
        // Not gated by `gameplay` either: like `left_click`, the press stays
        // false at the backend while the cursor is captured.
        right_click: raw.right_click,
        viewport: [0.0, 0.0],
        top_inset: 0.0,
        hud_toggle: raw.hud_toggle,
        // Start mirrors Escape (pause / back), so like `raw.escape` it stays
        // live while a menu is open.
        escape: raw.escape || pad.pressed(GamepadButton::Start),
        // Not gated by `gameplay`: a story's Ctrl fast-forward works
        // while its stage (a view) is up, like the rebind capture below.
        ctrl: raw.ctrl,
        // Not gated by `gameplay`: the editor's additive selection needs the
        // modifier while it holds the cursor. `raw.sprint` is the Shift key.
        shift: raw.sprint,
        // Not gated by `gameplay`: the editor's orbit drag needs the modifier
        // while it holds the cursor, like `shift`.
        alt: raw.alt,
        // Not gated by `gameplay` either: it carries the same UI shortcuts
        // Ctrl does on the platforms that use it.
        cmd: raw.cmd,
        // Not gated by `gameplay`: the rebind captures work while the
        // settings menu is open (the camera is what freezes behind it), and
        // text-input fields type while a menu (or the in-engine editor) is up.
        key_events: raw.key_events.clone(),
        captured_button: pad.first_pressed(),
        // Not gated by `gameplay`: menu focus movement + confirm/back consume
        // these while a screen is active; during play UI ignores them and the
        // buttons keep their merged meanings above.
        nav,
        confirm: pad.pressed(GamepadButton::South),
        back: pad.pressed(GamepadButton::East),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::GamepadAction;
    use crate::input::pad::{PadAxis, PadEvent, PadState};

    fn pad_snapshot(events: &[PadEvent]) -> PadSnapshot {
        let mut state = PadState::default();
        state.ingest(PadEvent::Connected { pad: 0 });
        for &e in events {
            state.ingest(e);
        }
        state.snapshot(0.15)
    }

    fn press(button: GamepadButton) -> PadEvent {
        PadEvent::Button {
            pad: 0,
            button,
            pressed: true,
        }
    }

    // A hitch counts as at most NAV_DT_MAX of hold, and a negative frame as none.
    #[test]
    fn nav_dt_clamps_the_frame_time() {
        let frame = |dt| FrameTime { dt, elapsed: 0.0 };
        assert_eq!(nav_dt(frame(0.1)), 0.1);
        assert_eq!(nav_dt(frame(3.0)), NAV_DT_MAX);
        assert_eq!(nav_dt(frame(-1.0)), 0.0);
    }

    #[test]
    fn gamepad_buttons_merge_into_the_boolean_fields() {
        let pad = pad_snapshot(&[
            press(GamepadButton::DpadUp),
            press(GamepadMap::DEFAULT.jump),
            press(GamepadMap::DEFAULT.sprint),
            press(GamepadMap::DEFAULT.interact),
            press(GamepadButton::Start),
        ]);
        let input = compose_frame_input(
            &InputSnapshot::default(),
            &pad,
            GamepadMap::DEFAULT,
            None,
            true,
        );
        assert!(input.forward, "d-pad up drives forward");
        assert!(input.jump && input.sprint && input.interact);
        assert!(input.escape, "Start mirrors Escape");
        assert!(!input.backward && !input.left && !input.right);
    }

    #[test]
    fn keyboard_fields_pass_through_unchanged() {
        let raw = InputSnapshot {
            forward: true,
            sprint: true,
            mouse_dx: 3.0,
            mouse_dy: -2.0,
            scroll_delta: 1.5,
            ..Default::default()
        };
        let input = compose_frame_input(
            &raw,
            &PadSnapshot::default(),
            GamepadMap::DEFAULT,
            None,
            true,
        );
        assert!(input.forward && input.sprint);
        assert_eq!(input.mouse_dx, 3.0);
        assert_eq!(input.mouse_dy, -2.0);
        assert_eq!(input.scroll_delta, 1.5);
        assert_eq!(input.move_axis, [0.0, 0.0]);
        assert_eq!(input.captured_button, None);
    }

    // The UI modifiers pass the menu gate: they carry shortcuts that must work
    // while a menu (or the in-engine editor) holds the screen.
    #[test]
    fn ui_modifiers_stay_live_behind_the_menu_gate() {
        let raw = InputSnapshot {
            ctrl: true,
            alt: true,
            cmd: true,
            ..Default::default()
        };
        for gameplay in [true, false] {
            let input = compose_frame_input(
                &raw,
                &PadSnapshot::default(),
                GamepadMap::DEFAULT,
                None,
                gameplay,
            );
            assert!(input.ctrl && input.alt && input.cmd, "gameplay={gameplay}");
        }
    }

    #[test]
    fn rebound_map_routes_buttons_to_their_actions() {
        let mut map = GamepadMap::DEFAULT;
        map.rebind(GamepadAction::Jump, GamepadButton::North);
        let pad = pad_snapshot(&[press(GamepadButton::North)]);
        let input = compose_frame_input(&InputSnapshot::default(), &pad, map, None, true);
        assert!(input.jump, "jump follows the rebound button");
        let default_pad = pad_snapshot(&[press(GamepadButton::South)]);
        let input = compose_frame_input(&InputSnapshot::default(), &default_pad, map, None, true);
        assert!(!input.jump, "the old button no longer jumps");
    }

    #[test]
    fn menu_gate_freezes_gameplay_but_keeps_capture_and_escape() {
        let pad = pad_snapshot(&[
            press(GamepadButton::DpadUp),
            press(GamepadMap::DEFAULT.jump),
            press(GamepadButton::Start),
            PadEvent::Axis {
                pad: 0,
                axis: PadAxis::LeftY,
                value: 1.0,
            },
            PadEvent::Axis {
                pad: 0,
                axis: PadAxis::RightX,
                value: 1.0,
            },
        ]);
        let raw = InputSnapshot {
            forward: true,
            sprint: true,
            mouse_dx: 5.0,
            ..Default::default()
        };
        let input = compose_frame_input(&raw, &pad, GamepadMap::DEFAULT, None, false);
        // Movement, jump, axes, and mouse deltas all freeze behind the menu.
        assert!(!input.forward && !input.jump);
        assert!(!input.sprint, "sprint freezes behind the menu");
        assert!(input.shift, "the raw Shift modifier stays live for UI");
        assert_eq!(input.move_axis, [0.0, 0.0]);
        assert_eq!(input.look_axis, [0.0, 0.0]);
        assert_eq!(input.mouse_dx, 0.0);
        // The rebind capture and Escape stay live for the menu itself. South
        // precedes Start in declaration order, so it is the captured button.
        assert_eq!(input.captured_button, Some(GamepadMap::DEFAULT.jump));
        assert!(input.escape);
    }

    #[test]
    fn mouse_clicks_pass_the_menu_gate() {
        // Both click pulses serve UI, so neither freezes behind a menu: the
        // backend already withholds them while the cursor is captured.
        let raw = InputSnapshot {
            left_click: true,
            right_click: true,
            ..Default::default()
        };
        let pad = pad_snapshot(&[]);
        let input = compose_frame_input(&raw, &pad, GamepadMap::DEFAULT, None, false);
        assert!(input.left_click);
        assert!(input.right_click);
    }

    #[test]
    fn stick_axes_publish_when_gameplay_is_live() {
        let pad = pad_snapshot(&[
            PadEvent::Axis {
                pad: 0,
                axis: PadAxis::LeftY,
                value: 1.0,
            },
            PadEvent::Axis {
                pad: 0,
                axis: PadAxis::RightY,
                value: 1.0,
            },
        ]);
        let input = compose_frame_input(
            &InputSnapshot::default(),
            &pad,
            GamepadMap::DEFAULT,
            None,
            true,
        );
        assert!(
            (input.move_axis[1] - 1.0).abs() < 1e-6,
            "{:?}",
            input.move_axis
        );
        // Stick up looks up: negative Y in the mouse-delta convention.
        assert!(input.look_axis[1] < -0.9, "{:?}", input.look_axis);
    }

    #[test]
    fn nav_confirm_and_back_pass_the_menu_gate() {
        let pad = pad_snapshot(&[press(GamepadButton::South), press(GamepadButton::East)]);
        let nav = Some(NavDirection::Down);
        let input = compose_frame_input(
            &InputSnapshot::default(),
            &pad,
            GamepadMap::DEFAULT,
            nav,
            false,
        );
        assert_eq!(input.nav, nav);
        assert!(input.confirm, "South edge surfaces behind the menu");
        assert!(input.back, "East edge surfaces behind the menu");
        // The same fields surface during play too; UI just ignores them.
        let input = compose_frame_input(
            &InputSnapshot::default(),
            &pad,
            GamepadMap::DEFAULT,
            nav,
            true,
        );
        assert!(input.confirm && input.back);
    }
}
