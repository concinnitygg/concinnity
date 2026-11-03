//! The merge of a frame's window input and gamepad snapshot into the published
//! `FrameInput`, under the gameplay gate.

use crate::components::{FrameInput, GamepadButton, GamepadMap, InputKey, KeyEvent, NavDirection};
use crate::ecs::FrameTime;

use super::keymap::KeyMap;
use super::pad::PadSnapshot;
use super::snapshot::InputSnapshot;

/// What drives each gameplay action: the key map and the gamepad map.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Bindings {
    /// The keyboard key bound to each action.
    pub keys: KeyMap,
    /// The gamepad button bound to each action.
    pub pad: GamepadMap,
}

impl Bindings {
    /// The default keys and buttons.
    pub const DEFAULT: Self = Self {
        keys: KeyMap::DEFAULT,
        pad: GamepadMap::DEFAULT,
    };
}

// Whether `key` went down this frame: a fresh press, not an auto-repeat, and
// not part of a Command shortcut.
fn pressed_now(raw: &InputSnapshot, key: InputKey) -> bool {
    raw.key_events.iter().any(|e| match e {
        KeyEvent::Press(p) => p.key == key && !p.repeat && !p.mods.cmd,
        KeyEvent::Text(_) => false,
    })
}

// Frame dt ceiling for the nav auto-repeat, so a hitch or debugger pause never
// counts as a long hold.
const NAV_DT_MAX: f32 = 0.25;

/// The nav auto-repeat's share of this frame's time.
pub fn nav_dt(frame: FrameTime) -> f32 {
    frame.dt.clamp(0.0, NAV_DT_MAX)
}

/// Merge the window's keyboard/mouse snapshot with the gamepad snapshot under
/// the gameplay gate, reading each action's key and button from `bindings`.
/// The caller fills in the window metrics.
pub fn compose_frame_input(
    raw: &InputSnapshot,
    pad: &PadSnapshot,
    bindings: Bindings,
    nav: Option<NavDirection>,
    gameplay: bool,
) -> FrameInput {
    let axis = |v: [f32; 2]| if gameplay { v } else { [0.0, 0.0] };
    let (keys, map) = (bindings.keys, bindings.pad);
    let held = |key: InputKey| raw.held.contains(key);
    FrameInput {
        forward: (held(keys.forward) || pad.held(GamepadButton::DpadUp)) && gameplay,
        backward: (held(keys.backward) || pad.held(GamepadButton::DpadDown)) && gameplay,
        left: (held(keys.left) || pad.held(GamepadButton::DpadLeft)) && gameplay,
        right: (held(keys.right) || pad.held(GamepadButton::DpadRight)) && gameplay,
        sprint: (held(keys.sprint) || pad.held(map.sprint)) && gameplay,
        interact: (pressed_now(raw, keys.interact) || pad.pressed(map.interact)) && gameplay,
        jump: (pressed_now(raw, keys.jump) || pad.pressed(map.jump)) && gameplay,
        jump_held: (held(keys.jump) || pad.held(map.jump)) && gameplay,
        descend: (held(keys.descend) || pad.held(map.descend)) && gameplay,
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
        ctrl: held(InputKey::Control),
        // Not gated by `gameplay`: the editor's additive selection needs the
        // modifier while it holds the cursor.
        shift: held(InputKey::Shift),
        // Not gated by `gameplay`: the editor's orbit drag needs the modifier
        // while it holds the cursor, like `shift`.
        alt: held(InputKey::Alt),
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
    use crate::components::{GamepadAction, KeyMods, KeyPress};
    use crate::input::held_keys::HeldKeys;
    use crate::input::pad::{PadAxis, PadEvent, PadState};

    fn press_key(key: InputKey, repeat: bool) -> KeyEvent {
        KeyEvent::Press(KeyPress {
            key,
            mods: KeyMods::NONE,
            repeat,
        })
    }

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
            Bindings::DEFAULT,
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
            held: HeldKeys::NONE.with(InputKey::W).with(InputKey::Shift),
            mouse_dx: 3.0,
            mouse_dy: -2.0,
            scroll_delta: 1.5,
            ..Default::default()
        };
        let input =
            compose_frame_input(&raw, &PadSnapshot::default(), Bindings::DEFAULT, None, true);
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
            held: HeldKeys::NONE.with(InputKey::Control).with(InputKey::Alt),
            cmd: true,
            ..Default::default()
        };
        for gameplay in [true, false] {
            let input = compose_frame_input(
                &raw,
                &PadSnapshot::default(),
                Bindings::DEFAULT,
                None,
                gameplay,
            );
            assert!(input.ctrl && input.alt && input.cmd, "gameplay={gameplay}");
        }
    }

    // Each action reads the key the map binds to it: held keys drive the held
    // actions, and a fresh press (not a repeat, not a Command shortcut) the
    // one-frame ones.
    #[test]
    fn keys_drive_the_actions_the_key_map_binds_them_to() {
        let mut keys = KeyMap::DEFAULT;
        keys.rebind(crate::input::keymap::Bindable::Jump, InputKey::J);
        let bindings = Bindings {
            keys,
            ..Bindings::DEFAULT
        };
        let raw = InputSnapshot {
            held: HeldKeys::NONE
                .with(InputKey::J)
                .with(InputKey::D)
                .with(InputKey::Control),
            key_events: alloc::vec![press_key(InputKey::J, false), press_key(InputKey::E, true)],
            ..Default::default()
        };
        let pad = PadSnapshot::default();
        let input = compose_frame_input(&raw, &pad, bindings, None, true);
        assert!(input.jump && input.jump_held, "J jumps once rebound");
        assert!(input.right && !input.forward);
        assert!(input.descend, "Control descends by default");
        assert!(!input.interact, "an auto-repeat is no fresh press");
        let space = InputSnapshot {
            held: HeldKeys::NONE.with(InputKey::Space),
            key_events: alloc::vec![press_key(InputKey::Space, false)],
            ..Default::default()
        };
        let input = compose_frame_input(&space, &pad, bindings, None, true);
        assert!(!input.jump && !input.jump_held, "Space no longer jumps");
        let shortcut = InputSnapshot {
            key_events: alloc::vec![KeyEvent::Press(KeyPress {
                key: InputKey::J,
                mods: KeyMods {
                    cmd: true,
                    ..KeyMods::NONE
                },
                repeat: false,
            })],
            ..Default::default()
        };
        let input = compose_frame_input(&shortcut, &pad, bindings, None, true);
        assert!(!input.jump, "a Command shortcut drives no action");
    }

    // A one-frame action bound to Control fires from its press, as any key's
    // would: rebinding interact onto Control keeps it working.
    #[test]
    fn a_press_action_bound_to_control_fires() {
        let mut keys = KeyMap::DEFAULT;
        keys.rebind(crate::input::keymap::Bindable::Interact, InputKey::Control);
        let bindings = Bindings {
            keys,
            ..Bindings::DEFAULT
        };
        let raw = InputSnapshot {
            held: HeldKeys::NONE.with(InputKey::Control),
            key_events: alloc::vec![KeyEvent::Press(KeyPress {
                key: InputKey::Control,
                mods: KeyMods::CTRL,
                repeat: false,
            })],
            ..Default::default()
        };
        let input = compose_frame_input(&raw, &PadSnapshot::default(), bindings, None, true);
        assert!(input.interact, "interact fires from Control");
        assert!(!input.descend, "descend moved to E in the swap");
    }

    #[test]
    fn rebound_map_routes_buttons_to_their_actions() {
        let mut map = GamepadMap::DEFAULT;
        map.rebind(GamepadAction::Jump, GamepadButton::North);
        let pad = pad_snapshot(&[press(GamepadButton::North)]);
        let bindings = Bindings {
            pad: map,
            ..Bindings::DEFAULT
        };
        let input = compose_frame_input(&InputSnapshot::default(), &pad, bindings, None, true);
        assert!(input.jump, "jump follows the rebound button");
        let default_pad = pad_snapshot(&[press(GamepadButton::South)]);
        let input = compose_frame_input(
            &InputSnapshot::default(),
            &default_pad,
            bindings,
            None,
            true,
        );
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
            held: HeldKeys::NONE.with(InputKey::W).with(InputKey::Shift),
            key_events: alloc::vec![press_key(InputKey::Space, false)],
            mouse_dx: 5.0,
            ..Default::default()
        };
        let input = compose_frame_input(&raw, &pad, Bindings::DEFAULT, None, false);
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
        let input = compose_frame_input(&raw, &pad, Bindings::DEFAULT, None, false);
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
            Bindings::DEFAULT,
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
            Bindings::DEFAULT,
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
            Bindings::DEFAULT,
            nav,
            true,
        );
        assert!(input.confirm && input.back);
    }
}
