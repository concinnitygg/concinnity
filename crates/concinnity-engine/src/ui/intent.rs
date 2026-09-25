// The menu pulses one frame of input carries, derived before any stage runs so
// every stage reads the same verdict.

use concinnity_core::components::{FrameInput, InputKey, NavDirection};

// How far (Manhattan, window pixels) the cursor must travel between frames to
// count as a deliberate move that dismisses the focus cursor.
const CURSOR_MOVE_THRESHOLD: f32 = 2.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct UiIntent {
    // The mouse moved since last frame, which dismisses the focus cursor.
    pub(super) cursor_moved: bool,
    // A focus-navigation pulse from the pad or the arrow keys.
    pub(super) nav: Option<NavDirection>,
    // Enter was pressed while a screen is up and no field is typing.
    pub(super) enter_pressed: bool,
    // Enter that confirms the focused control rather than reaching KeyBindings.
    pub(super) enter_confirm: bool,
    // Fire the focused control: the pad's South button, or `enter_confirm`.
    pub(super) confirm: bool,
    // Escape, or the pad's East button while a screen is up.
    pub(super) ui_escape: bool,
    // The key name KeyBindings and screen toggles match this frame: the
    // frame's last fresh press, so a held key toggles or fires once.
    pub(super) pressed_key: Option<&'static str>,
}

// Derive this frame's intent. `has_focus` is whether the focus cursor was set
// before this frame, and `last_cursor` last frame's cursor position.
pub(super) fn frame_intent(
    input: &FrameInput,
    typing: bool,
    screen_active: bool,
    has_focus: bool,
    last_cursor: Option<(f32, f32)>,
) -> UiIntent {
    let cursor_moved = last_cursor.is_some_and(|(px, py)| {
        (input.mouse_x - px).abs() + (input.mouse_y - py).abs() > CURSOR_MOVE_THRESHOLD
    });
    let menu_keys = screen_active && !typing;
    let nav = if menu_keys {
        input
            .nav
            .or_else(|| input.pressed_keys().filter_map(arrow_nav).last())
    } else {
        None
    };
    // An unfocused Enter still reaches the KeyBindings (e.g. a story's advance
    // binding); a cursor move this frame has already dismissed the focus. A
    // held Enter confirms once, like a binding fires once.
    let enter_pressed = menu_keys && input.pressed_fresh(InputKey::Enter);
    let enter_confirm = enter_pressed && has_focus && !cursor_moved;
    let confirm = (menu_keys && input.confirm) || enter_confirm;
    let ui_escape = input.escape || (input.back && screen_active);
    // The pad's back pulse rides the Escape name, so it pops toggled screens and
    // fires Escape bindings exactly like the key.
    let pressed_key = if ui_escape {
        Some("Escape")
    } else {
        input.fresh_keys().last().map(InputKey::name)
    };
    UiIntent {
        cursor_moved,
        nav,
        enter_pressed,
        enter_confirm,
        confirm,
        ui_escape,
        pressed_key,
    }
}

// The focus step an arrow key asks for.
fn arrow_nav(key: InputKey) -> Option<NavDirection> {
    match key {
        InputKey::Up => Some(NavDirection::Up),
        InputKey::Down => Some(NavDirection::Down),
        InputKey::Left => Some(NavDirection::Left),
        InputKey::Right => Some(NavDirection::Right),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::{KeyEvent, KeyMods, KeyPress};

    fn at(mx: f32, my: f32) -> FrameInput {
        FrameInput {
            mouse_x: mx,
            mouse_y: my,
            ..Default::default()
        }
    }

    fn key(k: InputKey) -> FrameInput {
        FrameInput {
            key_events: vec![KeyEvent::press(k)],
            ..Default::default()
        }
    }

    fn held(k: InputKey) -> FrameInput {
        let press = KeyPress {
            repeat: true,
            ..KeyPress::new(k, KeyMods::NONE)
        };
        FrameInput {
            key_events: vec![KeyEvent::Press(press)],
            ..Default::default()
        }
    }

    // A menu screen is up, nothing is typing, and the cursor has not moved.
    fn menu(input: &FrameInput, has_focus: bool) -> UiIntent {
        frame_intent(input, false, true, has_focus, Some((0.0, 0.0)))
    }

    #[test]
    fn cursor_moved_needs_a_last_position_and_the_threshold() {
        let input = at(1.5, 1.5);
        assert!(!frame_intent(&input, false, true, false, None).cursor_moved);
        assert!(frame_intent(&input, false, true, false, Some((0.0, 0.0))).cursor_moved);
        let still = at(1.0, 1.0);
        assert!(!frame_intent(&still, false, true, false, Some((0.0, 0.0))).cursor_moved);
    }

    #[test]
    fn arrow_keys_navigate_only_on_an_untyped_screen() {
        let input = key(InputKey::Down);
        assert_eq!(menu(&input, false).nav, Some(NavDirection::Down));
        assert_eq!(frame_intent(&input, true, true, false, None).nav, None);
        assert_eq!(frame_intent(&input, false, false, false, None).nav, None);
    }

    #[test]
    fn pad_nav_wins_over_an_arrow_key() {
        let input = FrameInput {
            nav: Some(NavDirection::Left),
            key_events: vec![KeyEvent::press(InputKey::Right)],
            ..Default::default()
        };
        assert_eq!(menu(&input, false).nav, Some(NavDirection::Left));
    }

    #[test]
    fn enter_confirms_only_with_focus_and_a_still_cursor() {
        let input = key(InputKey::Enter);
        let unfocused = menu(&input, false);
        assert!(unfocused.enter_pressed);
        assert!(!unfocused.enter_confirm);
        assert!(!unfocused.confirm);

        let focused = menu(&input, true);
        assert!(focused.enter_confirm);
        assert!(focused.confirm);

        let moved = FrameInput {
            mouse_x: 10.0,
            ..input.clone()
        };
        let moved = frame_intent(&moved, false, true, true, Some((0.0, 0.0)));
        assert!(moved.enter_pressed);
        assert!(!moved.enter_confirm);
    }

    // Several presses in one frame: the last one names the binding, and an
    // arrow earlier in the frame still navigates.
    #[test]
    fn the_last_press_names_the_key_and_arrows_still_navigate() {
        let input = FrameInput {
            key_events: vec![
                KeyEvent::press(InputKey::Down),
                KeyEvent::press(InputKey::Shift),
                KeyEvent::press(InputKey::Space),
            ],
            ..Default::default()
        };
        let intent = menu(&input, false);
        assert_eq!(intent.pressed_key, Some("Space"));
        assert_eq!(intent.nav, Some(NavDirection::Down));
    }

    // A held key's auto-repeat names no key for bindings or toggles and
    // confirms nothing, but a held arrow keeps stepping the focus.
    #[test]
    fn repeats_navigate_but_never_fire() {
        let space = menu(&held(InputKey::Space), true);
        assert_eq!(space.pressed_key, None);
        let enter = menu(&held(InputKey::Enter), true);
        assert!(!enter.enter_pressed && !enter.confirm);
        assert_eq!(
            menu(&held(InputKey::Down), false).nav,
            Some(NavDirection::Down)
        );
        assert_eq!(
            menu(&key(InputKey::Space), false).pressed_key,
            Some("Space"),
            "a fresh press still names its key"
        );
    }

    #[test]
    fn enter_is_not_a_menu_press_while_typing() {
        let input = key(InputKey::Enter);
        let typing = frame_intent(&input, true, true, true, None);
        assert!(!typing.enter_pressed);
        assert!(!typing.confirm);
        assert_eq!(typing.pressed_key, Some("Enter"));
    }

    #[test]
    fn pad_confirm_needs_an_untyped_screen() {
        let input = FrameInput {
            confirm: true,
            ..Default::default()
        };
        assert!(menu(&input, false).confirm);
        assert!(!frame_intent(&input, true, true, false, None).confirm);
        assert!(!frame_intent(&input, false, false, false, None).confirm);
    }

    #[test]
    fn back_mirrors_escape_only_while_a_screen_is_active() {
        let back = FrameInput {
            back: true,
            ..Default::default()
        };
        let on_screen = menu(&back, false);
        assert!(on_screen.ui_escape);
        assert_eq!(on_screen.pressed_key, Some("Escape"));

        let in_play = frame_intent(&back, false, false, false, None);
        assert!(!in_play.ui_escape);
        assert_eq!(in_play.pressed_key, None);
    }

    #[test]
    fn escape_overrides_the_pressed_key_name() {
        let input = FrameInput {
            escape: true,
            key_events: vec![KeyEvent::press(InputKey::Space)],
            ..Default::default()
        };
        let intent = frame_intent(&input, true, false, false, None);
        assert!(intent.ui_escape);
        assert_eq!(intent.pressed_key, Some("Escape"));
        assert_eq!(
            menu(&key(InputKey::Space), false).pressed_key,
            Some("Space")
        );
    }
}
