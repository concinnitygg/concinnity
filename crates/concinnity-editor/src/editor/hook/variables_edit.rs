// src/editor/hook/variables_edit.rs
//
// EditorHook: the Variables panel's actions. The panel edits the world's one
// `Variables` entry the way the Behavior panel edits one behavior -- its authored
// args directly, committing as each change is made, so the live preview rebuilds
// and SAVE persists it like any other panel edit.
//
// Declaring the table is what makes it authoritative, so creating the asset is
// never a blank act: the first declaration creates it holding that variable, and
// a name the behaviors already use is declared with the type it is given rather
// than being left out of a table that now has to account for it.
//
// The keyboard is Enter and Escape only: Enter commits the field holding it, and
// Escape gives it up. There is no navigation to add, because the panel is one
// list and the arrows are already what the text fields use.

use serde_json::Value;

use super::*;
use crate::assets::Key;
use crate::editor::behavior::edit;
use crate::editor::behavior::palette;
use crate::editor::behavior::relations;
use crate::editor::variables::Row;

// Owned per-tick data backing a `VariablesView`.
pub(super) struct VariablesData {
    pub rows: Vec<Row>,
    pub authoritative: bool,
    pub status: Option<String>,
    // Whether a live session's values are showing in place of the declared
    // starting values (see `hook/trace_drive.rs`).
    pub live: bool,
}

// The type words a variable steps through, in the order the palette lists them.
fn literal_verbs() -> Vec<&'static str> {
    palette::LITERALS.iter().map(|e| e.verb).collect()
}

impl EditorHook {
    // The `entries` index of the world's `Variables` singleton, when it has one.
    pub(super) fn variables_entry(&self) -> Option<usize> {
        self.entries
            .iter()
            .position(|e| entry_type(e) == Some("Variables"))
    }

    pub(super) fn variables_data(&self) -> VariablesData {
        let args = self.variables_args();
        let used = relations::variables_used(&self.behavior_pairs());
        let mut rows = variables::rows(args.as_ref(), &used);
        // While a session runs, the value column carries the runtime's current
        // values, and the selected entity's behavior locals join the list for
        // inspection. Both revert to the declarations on Stop (the trace drive
        // clears them).
        let live = !self.live_vars.is_empty();
        if live {
            for row in &mut rows {
                if let Some((_, ty, value)) = self.live_vars.iter().find(|(n, _, _)| *n == row.name)
                {
                    row.ty = ty.clone();
                    row.value = value.clone();
                }
            }
            // A runtime variable no authored row covers still shows: it holds
            // state right now, whatever the table says.
            for (name, ty, value) in &self.live_vars {
                if rows.iter().all(|r| &r.name != name) {
                    rows.push(Row {
                        name: name.clone(),
                        at: None,
                        ty: ty.clone(),
                        value: value.clone(),
                        local: false,
                    });
                }
            }
            rows.extend(self.live_locals.iter().map(|(name, ty, value)| Row {
                name: name.clone(),
                at: None,
                ty: ty.clone(),
                value: value.clone(),
                local: true,
            }));
        }
        VariablesData {
            authoritative: variables::authoritative(args.as_ref()),
            status: self.variables_status(&rows),
            rows,
            live,
        }
    }

    // What the panel warns about. The checker's own complaint comes first, since
    // it is what the build will say; otherwise a declared table that leaves out a
    // name its behaviors use is the thing worth saying, because that is a build
    // error the panel can fix.
    fn variables_status(&self, rows: &[Row]) -> Option<String> {
        if let Some(idx) = self.variables_entry() {
            let name = entry_name(&self.entries[idx]).unwrap_or("");
            let args = self.variables_args().unwrap_or(Value::Null);
            if let Err(e) = concinnity_cook::check::behavior::check_variables(name, &args) {
                return Some(e.lines().next().unwrap_or(&e).to_string());
            }
        }
        let missing: Vec<&str> = rows
            .iter()
            .filter(|r| !r.declared() && !r.local)
            .map(|r| r.name.as_str())
            .collect();
        if missing.is_empty() || !variables::authoritative(self.variables_args().as_ref()) {
            return None;
        }
        // Two lines of banner, so the sentence has to fit in one breath: what to
        // do first, then why.
        Some(format!(
            "declare {}: this table is authoritative, so the build rejects what it leaves out",
            missing.join(", "),
        ))
    }

    pub(super) fn make_variables_view<'a>(
        &'a self,
        data: &'a VariablesData,
        mouse: [f32; 2],
    ) -> VariablesView<'a> {
        let frontmost = self.panel_order.last() == Some(&PanelKey::Variables);
        let on_declared = self
            .variables_row
            .and_then(|i| data.rows.get(i))
            .is_some_and(Row::declared);
        VariablesView {
            rows: &data.rows,
            scroll: self.variables_scroll,
            selected: self.variables_row,
            authoritative: data.authoritative,
            name_focus: self.variables_name_focus && on_declared && frontmost,
            value_focus: self.variables_value_focus && on_declared && frontmost,
            status: data.status.as_deref(),
            live: data.live,
            mouse,
        }
    }

    // (Re)open the panel: drop any stale selection and focus, and seed the fields
    // from whatever the table now holds.
    pub(super) fn open_variables(&mut self, world: &mut World) {
        self.variables_row = None;
        self.variables_scroll = 0;
        self.variables_name_focus = false;
        self.variables_value_focus = false;
        self.seed_variables_fields(world);
    }

    // Open the panel with `name` selected, for the callers that arrive from
    // somewhere else in the editor already knowing which variable they mean.
    pub(super) fn open_variable_named(&mut self, name: &str, world: &mut World) {
        if !self.variables_open {
            registry::panel(PanelKey::Variables).toggle(self, world);
        }
        self.focus_panel(PanelKey::Variables);
        let at = self
            .variables_data()
            .rows
            .iter()
            .position(|r| r.name == name);
        self.variables_row = at;
        self.variables_name_focus = false;
        self.variables_value_focus = false;
        self.ensure_variables_row_visible();
        self.seed_variables_fields(world);
    }

    pub(super) fn apply_variables_action(&mut self, action: VariablesAction, world: &mut World) {
        // A focused name field lasts only until the next press, so the keyboard
        // cannot stay in a field the user has clicked away from.
        if action != VariablesAction::FocusName {
            self.blur_variables_name(world);
        }
        match action {
            VariablesAction::New => self.add_variable(world),
            VariablesAction::Select(i) => self.select_variable(i, world),
            VariablesAction::Declare => self.declare_variable(world),
            VariablesAction::Retype => self.retype_variable(world),
            VariablesAction::Remove => self.remove_variable(world),
            VariablesAction::FocusName => self.variables_name_focus = true,
            VariablesAction::FocusValue => self.variables_value_focus = true,
            VariablesAction::Consume => {
                self.variables_value_focus = false;
            }
        }
    }

    fn select_variable(&mut self, i: usize, world: &mut World) {
        self.variables_row = Some(i);
        self.variables_value_focus = false;
        self.seed_variables_fields(world);
    }

    // Declare a fresh variable and select it. The first one creates the table,
    // which is also the moment it becomes authoritative.
    fn add_variable(&mut self, world: &mut World) {
        let name = variables::unique_name(self.variables_args().as_ref(), "variable");
        self.declare(&name, "int", world);
    }

    // Declare the selected name the behaviors already use, keeping that name so
    // the behaviors resolve against it.
    fn declare_variable(&mut self, world: &mut World) {
        let Some(row) = self
            .variables_row
            .and_then(|i| self.variables_data().rows.get(i).cloned())
        else {
            return;
        };
        if row.declared() {
            return;
        }
        self.declare(&row.name, "int", world);
    }

    // Append a declaration of `name` with `verb`'s default value, creating the
    // table when the world has none, then select what landed.
    fn declare(&mut self, name: &str, verb: &str, world: &mut World) {
        let decl = serde_json::json!({"name": name, "value": palette::literal_default(verb)});
        match self.variables_entry() {
            Some(idx) => {
                let mut args = self
                    .variables_args()
                    .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
                let vars = args
                    .as_object_mut()
                    .and_then(|m| {
                        m.entry("vars")
                            .or_insert_with(|| Value::Array(Vec::new()))
                            .as_array_mut()
                    })
                    .map(|a| {
                        a.push(decl);
                        a.len() - 1
                    });
                if vars.is_none() {
                    return;
                }
                self.write_variables(idx, args, world);
            }
            None => {
                let asset = self.unique_name("world_vars");
                self.entries.push(serde_json::json!({
                    "name": asset, "type": "Variables", "args": {"vars": [decl]},
                }));
                self.mark_changed();
                self.after_variables_change(world);
            }
        }
        self.variables_row = self
            .variables_data()
            .rows
            .iter()
            .position(|r| r.name == name && r.declared());
        self.ensure_variables_row_visible();
        self.seed_variables_fields(world);
    }

    // Step the selected declaration's type, carrying its value across where the
    // shapes allow it (the same rule the Behavior panel's literal palette uses).
    fn retype_variable(&mut self, world: &mut World) {
        let Some((idx, at)) = self.selected_declaration() else {
            return;
        };
        let mut args = self.variables_args().unwrap_or(Value::Null);
        let Some(slot) = decl_mut(&mut args, at) else {
            return;
        };
        let current = slot.get("value").cloned().unwrap_or(Value::Null);
        let verbs = literal_verbs();
        let next = verbs
            .iter()
            .position(|v| *v == palette::verb_of(&current))
            .map(|i| (i + 1) % verbs.len())
            .unwrap_or(0);
        let value = palette::swap_literal(&current, verbs[next]);
        let Some(map) = slot.as_object_mut() else {
            return;
        };
        map.insert("value".to_string(), value);
        self.write_variables(idx, args, world);
        self.seed_variables_fields(world);
    }

    fn remove_variable(&mut self, world: &mut World) {
        let Some((idx, at)) = self.selected_declaration() else {
            return;
        };
        let mut args = self.variables_args().unwrap_or(Value::Null);
        let Some(vars) = args.get_mut("vars").and_then(Value::as_array_mut) else {
            return;
        };
        if at >= vars.len() {
            return;
        }
        vars.remove(at);
        self.write_variables(idx, args, world);
        // The removed row is gone; whatever slid into its place is not what was
        // selected, so the selection is dropped rather than retargeted.
        self.variables_row = None;
        self.variables_value_focus = false;
        self.seed_variables_fields(world);
    }

    // The per-frame editing key, while the panel is frontmost: Enter commits the
    // field holding the keyboard, Escape gives it up.
    pub(super) fn variables_keys(&mut self, world: &mut World, input: &FrameInput) {
        if input.escape {
            self.blur_variables_name(world);
            self.variables_value_focus = false;
            return;
        }
        if input.captured_key != Some(Key::Enter) {
            return;
        }
        if self.variables_name_focus {
            self.rename_variable(world);
            return;
        }
        self.commit_variable_value(world);
    }

    // Rename the selected variable. Behaviors name variables as plain strings, so
    // nothing in the world holds a reference this could dangle: a rename that
    // orphans a `set` shows up as the checker's own complaint about that behavior.
    fn rename_variable(&mut self, world: &mut World) {
        let Some((idx, at)) = self.selected_declaration() else {
            return;
        };
        let typed = widget::field_text(world, variables_panel::NAME_INPUT)
            .trim()
            .to_string();
        self.blur_variables_name(world);
        if typed.is_empty() {
            return;
        }
        let mut args = self.variables_args().unwrap_or(Value::Null);
        let Some(map) = decl_mut(&mut args, at).and_then(Value::as_object_mut) else {
            return;
        };
        map.insert("name".to_string(), Value::String(typed.clone()));
        self.write_variables(idx, args, world);
        self.variables_row = self
            .variables_data()
            .rows
            .iter()
            .position(|r| r.name == typed && r.declared());
        self.seed_variables_fields(world);
    }

    fn commit_variable_value(&mut self, world: &mut World) {
        if !self.variables_value_focus {
            return;
        }
        let Some((idx, at)) = self.selected_declaration() else {
            return;
        };
        let text = widget::field_text(world, variables_panel::VALUE_INPUT);
        let mut args = self.variables_args().unwrap_or(Value::Null);
        let Some(slot) = decl_mut(&mut args, at) else {
            return;
        };
        let current = slot.get("value").cloned().unwrap_or(Value::Null);
        let Ok(value) = edit::literal(palette::verb_of(&current), &text) else {
            // The checker would not see a malformed payload (it never lands), so
            // re-seeding the field is what says the typing was refused.
            self.seed_variables_fields(world);
            return;
        };
        let Some(map) = slot.as_object_mut() else {
            return;
        };
        map.insert("value".to_string(), value);
        self.write_variables(idx, args, world);
        self.seed_variables_fields(world);
    }

    // The entry holding the table and the selected declaration's place in it.
    fn selected_declaration(&self) -> Option<(usize, usize)> {
        let at = self
            .variables_row
            .and_then(|i| self.variables_data().rows.get(i).cloned())
            .and_then(|r| r.at)?;
        Some((self.variables_entry()?, at))
    }

    fn write_variables(&mut self, idx: usize, args: Value, world: &mut World) {
        let Some(entry) = self.entries[idx].as_object_mut() else {
            return;
        };
        entry.insert("args".to_string(), args);
        self.mark_changed();
        self.after_variables_change(world);
    }

    // A variable's type or starting value changes what the behaviors reading it
    // type-check against, so the Behavior panel's verdict is re-taken too.
    fn after_variables_change(&mut self, world: &mut World) {
        self.refresh_behavior_status();
        let _ = world;
    }

    fn seed_variables_fields(&mut self, world: &mut World) {
        let row = self
            .variables_row
            .and_then(|i| self.variables_data().rows.get(i).cloned());
        let (name, value) = match row.filter(Row::declared) {
            Some(r) => (r.name, r.value),
            None => (String::new(), String::new()),
        };
        widget::seed_field(world, variables_panel::NAME_INPUT, &name);
        widget::seed_field(world, variables_panel::VALUE_INPUT, &value);
    }

    // Give the name field up, putting back what the table holds so an abandoned
    // rename does not leave the field disagreeing with the world.
    fn blur_variables_name(&mut self, world: &mut World) {
        if !self.variables_name_focus {
            return;
        }
        self.variables_name_focus = false;
        self.seed_variables_fields(world);
    }

    fn variables_rows_shown(&self) -> usize {
        variables_panel::visible_rows(self.effective_size(PanelKey::Variables)[1])
    }

    fn ensure_variables_row_visible(&mut self) {
        let Some(row) = self.variables_row else {
            return;
        };
        let shown = self.variables_rows_shown();
        self.variables_scroll =
            crate::editor::behavior::navigate::scroll_to(row, self.variables_scroll, shown);
    }

    pub(super) fn scroll_variables(&mut self, delta: f32) {
        let max = self
            .variables_data()
            .rows
            .len()
            .saturating_sub(self.variables_rows_shown());
        self.variables_scroll = scroll_step(self.variables_scroll, delta, max);
    }
}

// The declaration at `at` inside a table's args.
fn decl_mut(args: &mut Value, at: usize) -> Option<&mut Value> {
    args.get_mut("vars")?.as_array_mut()?.get_mut(at)
}
