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
    // The key name KeyBindings and screen toggles match this frame.
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
        input.nav.or(match input.captured_key {
            Some(InputKey::Up) => Some(NavDirection::Up),
            Some(InputKey::Down) => Some(NavDirection::Down),
            Some(InputKey::Left) => Some(NavDirection::Left),
            Some(InputKey::Right) => Some(NavDirection::Right),
            _ => None,
        })
    } else {
        None
    };
    // An unfocused Enter still reaches the KeyBindings (e.g. a story's advance
    // binding); a cursor move this frame has already dismissed the focus.
    let enter_pressed = menu_keys && input.captured_key == Some(InputKey::Enter);
    let enter_confirm = enter_pressed && has_focus && !cursor_moved;
    let confirm = (menu_keys && input.confirm) || enter_confirm;
    let ui_escape = input.escape || (input.back && screen_active);
    // The pad's back pulse rides the Escape name, so it pops toggled screens and
    // fires Escape bindings exactly like the key.
    let pressed_key = if ui_escape {
        Some("Escape")
    } else {
        input.captured_key.map(InputKey::name)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn at(mx: f32, my: f32) -> FrameInput {
        FrameInput {
            mouse_x: mx,
            mouse_y: my,
            ..Default::default()
        }
    }

    fn key(k: InputKey) -> FrameInput {
        FrameInput {
            captured_key: Some(k),
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
            captured_key: Some(InputKey::Right),
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
    fn escape_overrides_the_captured_key_name() {
        let input = FrameInput {
            escape: true,
            captured_key: Some(InputKey::Space),
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
