// src/text_input_system.rs
//
// Drives editable TextInput fields: click-to-focus, character entry from the
// frame's typed character, and caret editing / movement from the Backspace /
// Delete / Left / Right keys. Present whenever the world has any `TextInput`
// (see `World::build_text_input`). Focus and caret live on the component itself
// (runtime-only fields), so the renderer and any reader see the edited text in
// place.

use crate::assets::{FrameInput, Key, SpriteFit, TextInput};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, StepResult, System};
use concinnity_core::gfx::overlay::OverlayTransform;

// One text edit applied at the caret in a single frame.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Edit {
    Insert(char),
    Backspace,
    Delete,
    Left,
    Right,
}

// The caret edit a non-printable key requests, if any. Printable characters
// arrive via `FrameInput.typed_char` and take precedence (see `step`).
fn command_from_key(key: Key) -> Option<Edit> {
    match key {
        Key::Backspace => Some(Edit::Backspace),
        Key::Delete => Some(Edit::Delete),
        Key::Left => Some(Edit::Left),
        Key::Right => Some(Edit::Right),
        _ => None,
    }
}

// Byte offset of the character at `char_idx`, or the string length when the
// index is at (or past) the end. Keeps all edits on a UTF-8 char boundary.
fn char_byte(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .nth(char_idx)
        .map(|(b, _)| b)
        .unwrap_or(s.len())
}

// Apply one edit to `content` at the caret (a character index). `max_len` of 0
// means no limit. The caret is clamped into range first, so an externally set
// caret (e.g. a pre-filled field) stays valid.
fn apply_edit(content: &mut String, caret: &mut usize, edit: Edit, max_len: usize) {
    let n = content.chars().count();
    let c = (*caret).min(n);
    match edit {
        Edit::Insert(ch) => {
            if max_len != 0 && n >= max_len {
                return;
            }
            let byte = char_byte(content, c);
            content.insert(byte, ch);
            *caret = c + 1;
        }
        Edit::Backspace => {
            if c > 0 {
                let start = char_byte(content, c - 1);
                let end = char_byte(content, c);
                content.replace_range(start..end, "");
                *caret = c - 1;
            }
        }
        Edit::Delete => {
            if c < n {
                let start = char_byte(content, c);
                let end = char_byte(content, c + 1);
                content.replace_range(start..end, "");
                *caret = c;
            }
        }
        Edit::Left => *caret = c.saturating_sub(1),
        Edit::Right => *caret = (c + 1).min(n),
    }
}

// Whether the cursor is inside a field's rect. A screen-owned field is authored in
// the reference canvas, so the cursor is mapped back through the screen's `fit`
// (matching the region hit-test in `UiInputSystem`); a HUD field (no screen) uses
// window pixels directly.
fn cursor_in_field(ti: &TextInput, mx: f32, my: f32, viewport: [f32; 2]) -> bool {
    let (qx, qy) = if ti.screen.is_none() {
        (mx, my)
    } else {
        let overlay = match ti.fit {
            SpriteFit::Bottom => OverlayTransform::bottom_anchored_from_viewport(viewport),
            SpriteFit::Cover => OverlayTransform::cover_from_viewport(viewport),
            SpriteFit::Fit => OverlayTransform::from_viewport(viewport),
        };
        overlay.inverse(mx, my)
    };
    qx >= ti.x && qx < ti.x + ti.width && qy >= ti.y && qy < ti.y + ti.height
}

#[derive(Debug, Default)]
pub struct TextInputSystem;

impl TextInputSystem {
    pub fn new() -> Self {
        Self
    }
}

impl System for TextInputSystem {
    fn access(&self) -> crate::ecs::Access {
        crate::ecs::Access::new()
            .reads_components(crate::component_mask![crate::assets::FrameInput])
            .writes_components(crate::component_mask![crate::assets::TextInput])
    }

    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        let input = match ctx.query::<FrameInput>().last().cloned() {
            Some(i) => i,
            None => return StepResult::Continue,
        };

        // A printable character types; otherwise a Backspace / Delete / arrow
        // moves or edits at the caret. Printable input wins (control keys never
        // produce a `typed_char`).
        let edit = if let Some(ch) = input.typed_char {
            Some(Edit::Insert(ch))
        } else {
            input.captured_key.and_then(command_from_key)
        };

        // On a click, focus moves to the top-most visible field under the cursor
        // (last one wins, matching draw order); a click that misses every field
        // blurs them all. Between clicks the focus flag on the component stands,
        // so another system (or the editor) can pre-focus a field.
        let hit_id: Option<AssetId> = if input.left_click {
            let mut hit = None;
            for ti in ctx.query::<TextInput>() {
                if ti.visible && cursor_in_field(ti, input.mouse_x, input.mouse_y, input.viewport) {
                    hit = Some(ti.asset_id);
                }
            }
            hit
        } else {
            None
        };

        for ti in ctx.query_mut::<TextInput>() {
            if input.left_click {
                ti.focused = ti.visible && Some(ti.asset_id) == hit_id;
            }
            // Only the focused field's caret is edited or drawn, so only it needs
            // the per-frame char-count walk. A click moves the caret to the end;
            // otherwise clamp it in case `content` changed out from under us.
            if ti.focused {
                let n = ti.content.chars().count();
                ti.caret = if input.left_click { n } else { ti.caret.min(n) };
                if ti.visible
                    && let Some(edit) = edit
                {
                    apply_edit(&mut ti.content, &mut ti.caret, edit, ti.max_len as usize);
                }
            }
        }

        StepResult::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_appends_and_advances_caret() {
        let mut s = String::new();
        let mut c = 0;
        apply_edit(&mut s, &mut c, Edit::Insert('h'), 0);
        apply_edit(&mut s, &mut c, Edit::Insert('i'), 0);
        assert_eq!(s, "hi");
        assert_eq!(c, 2);
    }

    #[test]
    fn insert_in_the_middle() {
        let mut s = "ac".to_string();
        let mut c = 1;
        apply_edit(&mut s, &mut c, Edit::Insert('b'), 0);
        assert_eq!(s, "abc");
        assert_eq!(c, 2);
    }

    #[test]
    fn backspace_removes_char_before_caret() {
        let mut s = "abc".to_string();
        let mut c = 2;
        apply_edit(&mut s, &mut c, Edit::Backspace, 0);
        assert_eq!(s, "ac");
        assert_eq!(c, 1);
        // Backspace at the start is a no-op.
        let mut c0 = 0;
        let mut s0 = "x".to_string();
        apply_edit(&mut s0, &mut c0, Edit::Backspace, 0);
        assert_eq!(s0, "x");
        assert_eq!(c0, 0);
    }

    #[test]
    fn delete_removes_char_at_caret() {
        let mut s = "abc".to_string();
        let mut c = 1;
        apply_edit(&mut s, &mut c, Edit::Delete, 0);
        assert_eq!(s, "ac");
        assert_eq!(c, 1);
        // Delete at the end is a no-op.
        let mut c2 = 2;
        apply_edit(&mut s, &mut c2, Edit::Delete, 0);
        assert_eq!(s, "ac");
    }

    #[test]
    fn max_len_clamps_inserts() {
        let mut s = "ab".to_string();
        let mut c = 2;
        apply_edit(&mut s, &mut c, Edit::Insert('c'), 2);
        assert_eq!(s, "ab");
        assert_eq!(c, 2);
    }

    #[test]
    fn arrows_move_and_clamp_caret() {
        let mut s = "abc".to_string();
        let mut c = 1;
        apply_edit(&mut s, &mut c, Edit::Left, 0);
        assert_eq!(c, 0);
        apply_edit(&mut s, &mut c, Edit::Left, 0); // clamps at 0
        assert_eq!(c, 0);
        apply_edit(&mut s, &mut c, Edit::Right, 0);
        assert_eq!(c, 1);
        c = 3;
        apply_edit(&mut s, &mut c, Edit::Right, 0); // clamps at len
        assert_eq!(c, 3);
    }

    #[test]
    fn edits_stay_on_char_boundaries() {
        // A multi-byte char must not panic or split.
        let mut s = "aé".to_string(); // 'é' is two bytes
        let mut c = 2;
        apply_edit(&mut s, &mut c, Edit::Backspace, 0);
        assert_eq!(s, "a");
        assert_eq!(c, 1);
    }

    #[test]
    fn command_keys_map_to_edits() {
        assert_eq!(command_from_key(Key::Backspace), Some(Edit::Backspace));
        assert_eq!(command_from_key(Key::Delete), Some(Edit::Delete));
        assert_eq!(command_from_key(Key::Left), Some(Edit::Left));
        assert_eq!(command_from_key(Key::Right), Some(Edit::Right));
        assert_eq!(command_from_key(Key::A), None);
        assert_eq!(command_from_key(Key::Enter), None);
    }

    #[test]
    fn hud_field_hit_tests_in_window_pixels() {
        let ti = TextInput {
            x: 100.0,
            y: 50.0,
            width: 200.0,
            height: 40.0,
            screen: None,
            ..Default::default()
        };
        assert!(cursor_in_field(&ti, 150.0, 60.0, [1280.0, 720.0]));
        assert!(!cursor_in_field(&ti, 50.0, 60.0, [1280.0, 720.0]));
        assert!(!cursor_in_field(&ti, 150.0, 100.0, [1280.0, 720.0]));
    }

    use crate::ecs::World;

    #[test]
    fn focused_field_types_through_the_schedule() {
        // A world with just a TextInput builds only TextInputSystem; a typed
        // character lands in the focused field.
        let mut world = World::new();
        world.add_component(TextInput {
            asset_id: AssetId(1),
            focused: true,
            screen: None,
            ..Default::default()
        });
        world.start().unwrap();
        world.add_component(FrameInput {
            typed_char: Some('h'),
            ..Default::default()
        });
        world.step();
        assert_eq!(world.query::<TextInput>().next().unwrap().content, "h");
    }

    #[test]
    fn click_focuses_the_field_under_the_cursor() {
        let mut world = World::new();
        world.add_component(TextInput {
            asset_id: AssetId(1),
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 40.0,
            screen: None,
            ..Default::default()
        });
        world.add_component(TextInput {
            asset_id: AssetId(2),
            x: 200.0,
            y: 0.0,
            width: 100.0,
            height: 40.0,
            screen: None,
            ..Default::default()
        });
        world.start().unwrap();
        // Click inside field 2.
        world.add_component(FrameInput {
            mouse_x: 250.0,
            mouse_y: 20.0,
            left_click: true,
            viewport: [1280.0, 720.0],
            ..Default::default()
        });
        world.step();
        let focus: Vec<(u32, bool)> = world
            .query::<TextInput>()
            .map(|t| (t.asset_id.0, t.focused))
            .collect();
        assert!(focus.contains(&(1, false)));
        assert!(focus.contains(&(2, true)));
    }
}
