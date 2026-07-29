// src/editor/hook/behavior_asset.rs
//
// EditorHook: the open Behavior as a whole asset -- naming it and taking it out
// of the world. Editing what a behavior does lives in `behavior_edit.rs`; these
// act on the authored line itself, so both go through `mark_changed` and land in
// the undo history like any other entry edit.
//
// Removing one takes two presses. `mark_changed` does record the pre-edit list,
// so Undo brings a removed behavior back, but an undo restores the entry list
// alone: the checker's verdict is left standing from whatever was open before
// the jump, and the history is bounded, so the recovery is neither immediate nor
// indefinite. Arming the chip keeps a stray press from needing it at all.

use serde_json::Value;

use super::*;

impl EditorHook {
    // Arm the removal, then carry it out. Any other press on the panel disarms
    // it, so the second press has to be a deliberate one.
    pub(super) fn remove_behavior(&mut self, armed: bool, world: &mut World) {
        let Some(idx) = self.behavior_entry() else {
            return;
        };
        if !armed {
            self.behavior_remove_armed = true;
            return;
        }
        self.remove_entry_at(idx);
        // Reopening clamps the ordinal, so the panel lands on whatever has slid
        // into it -- or on the empty-world prompt, once the last one is gone.
        self.open_behavior(world);
    }

    // Take the keyboard for the name field. Re-pressing a focused field leaves
    // what is typed there alone; arriving at it fresh seeds it from the entry.
    pub(super) fn focus_behavior_name(&mut self, world: &mut World) {
        if self.behavior_entry().is_none() || self.behavior_name_focus {
            return;
        }
        self.behavior_focus = false;
        self.behavior_picking = false;
        self.seed_behavior_name(world);
        self.behavior_name_focus = true;
    }

    // Give up the name field without committing, reverting an abandoned edit
    // rather than leaving it in the field to be committed by a later Enter.
    pub(super) fn blur_behavior_name(&mut self, world: &mut World) {
        if !self.behavior_name_focus {
            return;
        }
        self.behavior_name_focus = false;
        self.seed_behavior_name(world);
    }

    pub(super) fn seed_behavior_name(&mut self, world: &mut World) {
        let name = self
            .behavior_entry()
            .and_then(|i| entry_name(&self.entries[i]))
            .unwrap_or("")
            .to_string();
        widget::seed_field(world, behavior_panel::NAME_INPUT, &name);
    }

    // Commit the name field onto the open behavior. A blank name is refused
    // rather than written, and one already taken is suffixed until it is free,
    // so the entry list keeps its one-name-one-asset rule. The field is re-seeded
    // from what actually landed, so the panel never shows a name the world does
    // not hold.
    pub(super) fn rename_behavior(&mut self, world: &mut World) {
        let Some(idx) = self.behavior_entry() else {
            return;
        };
        let typed = widget::field_text(world, behavior_panel::NAME_INPUT);
        if typed.trim().is_empty() {
            self.behavior_status = Some(Status::message("a behavior needs a name"));
            self.blur_behavior_name(world);
            return;
        }
        let name = self.finalize_rename(&typed, idx, "Behavior");
        if let Some(entry) = self.entries[idx].as_object_mut() {
            entry.insert("name".to_string(), Value::String(name));
        }
        self.mark_changed();
        self.behavior_name_focus = false;
        // The checker's messages carry the behavior's name, so its verdict is
        // re-read under the new one rather than left quoting the old.
        self.refresh_behavior_status();
        self.seed_behavior_name(world);
    }
}
