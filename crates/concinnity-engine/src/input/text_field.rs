//! Drives editable `TextField` fields: character entry from the frame's typed
//! characters, and caret editing / movement from the Backspace /
//! Delete / Left / Right / Home / End keys, every event of the frame applied in
//! the order it arrived. Present whenever the world has any `TextField`
//! (the engine's schedule gates it). The UI input system gives a field focus on
//! a click; focus and caret live on the component itself (runtime-only fields),
//! so the renderer and any reader see the edited text in place. A bound field
//! keeps its binding and its text in step.

use crate::ecs::{Access, PipelineContext, StepResult, System, World};
use alloc::string::String;
use alloc::vec::Vec;

use crate::components::{
    Binding, FrameInput, InputKey, KeyEvent, TextField, UiState, UiValue, ValueOp, VariableCommand,
    VariableValues,
};
use crate::presentation::ui::apply_value;
use crate::presentation::ui::bound::Values;
use crate::settings::SettingValues;

// One text edit applied at the caret in a single frame.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Edit {
    Insert(char),
    Backspace,
    Delete,
    Left,
    Right,
    Home,
    End,
}

// The caret edit a non-printable key requests, if any. A printable key's
// character arrives as its own text event (see `edit_from_event`).
fn command_from_key(key: InputKey) -> Option<Edit> {
    match key {
        InputKey::Backspace => Some(Edit::Backspace),
        InputKey::Delete => Some(Edit::Delete),
        InputKey::Left => Some(Edit::Left),
        InputKey::Right => Some(Edit::Right),
        InputKey::Home => Some(Edit::Home),
        InputKey::End => Some(Edit::End),
        _ => None,
    }
}

// The edit one keyboard event requests, if any.
fn edit_from_event(event: KeyEvent) -> Option<Edit> {
    match event {
        KeyEvent::Text(ch) => Some(Edit::Insert(ch)),
        KeyEvent::Press(press) => command_from_key(press.key),
    }
}

// Byte offset of the character at `char_idx`, or the string length when the
// index is at (or past) the end. Keeps all edits on a UTF-8 char boundary.
fn char_byte(s: &str, char_idx: usize) -> usize {
    s.char_indices().nth(char_idx).map_or(s.len(), |(b, _)| b)
}

// Apply one edit to `content` at the caret (a character index). `max_len` of 0
// means no limit. The caret is clamped into range first, so an externally set
// caret (e.g. a pre-filled field) stays valid.
fn apply_edit(content: &mut String, caret: &mut usize, edit: Edit, max_len: usize) -> bool {
    let n = content.chars().count();
    let c = (*caret).min(n);
    match edit {
        Edit::Insert(ch) => {
            if max_len != 0 && n >= max_len {
                return false;
            }
            let byte = char_byte(content, c);
            content.insert(byte, ch);
            *caret = c + 1;
            return true;
        }
        Edit::Backspace => {
            if c > 0 {
                let start = char_byte(content, c - 1);
                let end = char_byte(content, c);
                content.replace_range(start..end, "");
                *caret = c - 1;
                return true;
            }
        }
        Edit::Delete => {
            if c < n {
                let start = char_byte(content, c);
                let end = char_byte(content, c + 1);
                content.replace_range(start..end, "");
                *caret = c;
                return true;
            }
        }
        Edit::Left => *caret = c.saturating_sub(1),
        Edit::Right => *caret = (c + 1).min(n),
        Edit::Home => *caret = 0,
        Edit::End => *caret = n,
    }
    false
}

/// Edits the focused `TextField` field from the frame's key events.
#[derive(Debug, Default)]
pub struct TextFieldSystem {
    // This frame's edits, reused across frames.
    edits: Vec<Edit>,
    // The bound fields edited this frame, written back once the fields are
    // released.
    edited: Vec<(Binding, String)>,
}

impl TextFieldSystem {
    /// The system for a world that declares any `TextField`.
    pub fn for_world(world: &World) -> Option<Self> {
        world.query::<TextField>().next().map(|_| Self::default())
    }
}

impl System for TextFieldSystem {
    fn access(&self) -> Access {
        Access::new()
            .reads_components(crate::component_mask![FrameInput])
            .writes_components(crate::component_mask![TextField])
            .reads_resources(crate::resource_mask![SettingValues])
            .writes_resources(crate::resource_mask![
                UiState,
                VariableValues,
                VariableCommand
            ])
    }

    fn step(&mut self, ctx: &mut PipelineContext<'_>) -> StepResult {
        let Some(input) = ctx.query::<FrameInput>().last() else {
            return StepResult::Continue;
        };
        // Every character typed and editing key pressed this frame, in order,
        // so fast typing at a low frame rate loses nothing.
        self.edits.clear();
        self.edits
            .extend(input.key_events.iter().copied().filter_map(edit_from_event));

        // A bound field shows its binding's text while it is not focused.
        let state = ctx.take_resource::<UiState>();
        let vars = ctx.take_resource::<VariableValues>();
        let values = Values {
            state: state.as_ref(),
            vars: vars.as_ref(),
            settings: None,
        };
        for ti in ctx.query_mut::<TextField>() {
            if !ti.focused
                && let Some(bound) = ti.text.as_ref().and_then(|b| values.text(b))
                && ti.content != bound
            {
                ti.content.clear();
                ti.content.push_str(bound);
            }
            // Only the focused field's caret is edited or drawn, so only it needs
            // the per-frame char-count walk. A click moves the caret to the end;
            // otherwise clamp it in case `content` changed out from under us.
            if ti.focused {
                let n = ti.content.chars().count();
                ti.caret = ti.caret.min(n);
                if ti.visible && !self.edits.is_empty() {
                    let mut changed = false;
                    for &edit in &self.edits {
                        changed |=
                            apply_edit(&mut ti.content, &mut ti.caret, edit, ti.max_len as usize);
                    }
                    if changed && let Some(binding) = &ti.text {
                        self.edited.push((binding.clone(), ti.content.clone()));
                    }
                }
            }
        }
        if let Some(state) = state {
            ctx.insert_resource(state);
        }
        if let Some(vars) = vars {
            ctx.insert_resource(vars);
        }
        for (binding, text) in self.edited.drain(..) {
            apply_value(&binding, &ValueOp::Set(UiValue::Text(text)), ctx);
        }

        StepResult::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::Phase;
    use crate::ecs::asset_id::AssetId;
    use crate::test_support::start_with;
    use alloc::string::ToString;
    use alloc::vec;

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
        assert_eq!(command_from_key(InputKey::Backspace), Some(Edit::Backspace));
        assert_eq!(command_from_key(InputKey::Delete), Some(Edit::Delete));
        assert_eq!(command_from_key(InputKey::Left), Some(Edit::Left));
        assert_eq!(command_from_key(InputKey::Right), Some(Edit::Right));
        assert_eq!(command_from_key(InputKey::Home), Some(Edit::Home));
        assert_eq!(command_from_key(InputKey::End), Some(Edit::End));
        assert_eq!(command_from_key(InputKey::A), None);
        assert_eq!(command_from_key(InputKey::Enter), None);
    }

    #[test]
    fn home_and_end_jump_the_caret() {
        let mut s = "abc".to_string();
        let mut c = 1;
        apply_edit(&mut s, &mut c, Edit::End, 0);
        assert_eq!(c, 3);
        apply_edit(&mut s, &mut c, Edit::Home, 0);
        assert_eq!(c, 0);
    }

    use crate::ecs::World;

    #[test]
    fn focused_field_types_through_the_schedule() {
        // A typed character lands in the focused field.
        let mut world = World::new();
        world.push_identified(
            AssetId(1),
            TextField {
                focused: true,
                screen: None,
                ..Default::default()
            },
        );
        start_with(
            &mut world,
            Phase::Late,
            "TextFieldSystem",
            TextFieldSystem::default(),
        );
        world.add_component(FrameInput {
            key_events: vec![KeyEvent::Text('h')],
            ..Default::default()
        });
        world.step();
        assert_eq!(world.query::<TextField>().next().unwrap().content, "h");
    }

    // Several keystrokes landing in one frame all apply, in arrival order: the
    // fast-typing case a one-per-frame pulse used to drop.
    #[test]
    fn every_event_of_a_frame_applies_in_order() {
        let mut world = World::new();
        world.push_identified(
            AssetId(1),
            TextField {
                focused: true,
                screen: None,
                ..Default::default()
            },
        );
        start_with(
            &mut world,
            Phase::Late,
            "TextFieldSystem",
            TextFieldSystem::default(),
        );
        world.add_component(FrameInput {
            key_events: vec![
                KeyEvent::press(InputKey::A),
                KeyEvent::Text('a'),
                KeyEvent::press(InputKey::B),
                KeyEvent::Text('b'),
                KeyEvent::press(InputKey::Backspace),
                KeyEvent::press(InputKey::C),
                KeyEvent::Text('c'),
                KeyEvent::press(InputKey::Home),
                KeyEvent::Text('>'),
            ],
            ..Default::default()
        });
        world.step();
        let field = world.query::<TextField>().next().unwrap();
        assert_eq!(field.content, ">ac");
        assert_eq!(field.caret, 1);
    }

    // A bound field shows its binding while unfocused and writes each edit
    // back to it.
    #[test]
    fn a_bound_field_reads_and_writes_its_binding() {
        let mut world = World::new();
        let mut state = UiState::default();
        state.set("name", UiValue::Text("Ada".into()));
        world.insert_resource(state);
        world.push_identified(
            AssetId(1),
            TextField {
                text: Some(Binding::State("name".into())),
                screen: None,
                ..Default::default()
            },
        );
        start_with(
            &mut world,
            Phase::Late,
            "TextFieldSystem",
            TextFieldSystem::default(),
        );
        world.add_component(FrameInput::default());
        world.step();
        assert_eq!(world.query::<TextField>().next().unwrap().content, "Ada");

        for field in world.query_mut::<TextField>() {
            field.focused = true;
            field.caret = 3;
        }
        world.add_component(FrameInput {
            key_events: vec![KeyEvent::Text('!')],
            ..Default::default()
        });
        world.step();
        let name = world
            .resource::<UiState>()
            .and_then(|s| s.get("name").cloned());
        assert_eq!(name, Some(UiValue::Text("Ada!".into())));
    }
}
