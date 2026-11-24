//! `EditorHook`: the Behavior panel's actions. The panel opens one `Behavior`
//! entry at a time and edits its authored args directly, so every change is an
//! ordinary world edit -- the live preview rebuilds from the in-memory entries
//! and SAVE persists them, like any other panel. What the keyboard does with
//! these actions is `behavior_keys.rs`.
//!
//! A behavior body has no unbounded loop and no recursion, so it always
//! terminates; that is what makes running an edited body in the live world safe
//! and why edits commit as they are made rather than behind an Apply button.
//!
//! Validation is not repeated here: the world's own checker runs against the
//! same args after each commit and its message goes straight to the status line,
//! with the world's `Variables` table supplied so a misspelled variable is
//! caught the way the build would catch it. Cross-asset names (a spawn template,
//! a clip) resolve against the whole world, so those stay a build-time check.

use concinnity_cook::authoring::registry::RegisteredType;
use concinnity_cook::authoring::world::{
    args_without_id, entry_handle, entry_handles, find_entry, replace_args,
};
use concinnity_core::math::SaturatingFrom;
use concinnity_engine::components::{Canvas, TextField, UiEvent};
use concinnity_engine::ecs::World;
use concinnity_engine::ecs::asset_id::AssetId;
use serde_json::Value;
use std::sync::OnceLock;

use crate::editor::behavior::chart::{self, ChartMarks};
use crate::editor::behavior::edit::{self, Pick};
use crate::editor::behavior::graph::{self, CardKind, Chart};
use crate::editor::behavior::outline::{self, Row};
use crate::editor::behavior::panel::{self, BehaviorAction, BehaviorView, Status, ViewMode};
use crate::editor::behavior::{clip, fault, fields, filter, pulse, relations};
use crate::editor::hook::edit::behavior_state::Shown;
use crate::editor::hook::{EditorHook, entry_type};
use crate::editor::panels::registry::PanelKey;
use crate::editor::ui::shapes::Shapes;
use crate::editor::ui::{BuiltPanel, field, state};

// Owned data backing a `BehaviorView`: the open behavior's outline and the
// palette its selected row offers.
#[derive(Debug)]
pub(in crate::editor::hook) struct BehaviorData {
    pub name: String,
    pub index: usize,
    pub total: usize,
    pub rows: Vec<Row>,
    pub chart: Chart,
    // The world's behaviors and how they reach each other. Built only while the
    // overview is showing, since it walks every behavior in the world.
    pub overview: Chart,
    // The card the selection belongs to, and that node's own rows: what the
    // chart lights up and what the inspector lists.
    pub card: Option<usize>,
    pub fields: Vec<usize>,
    pub picks: Vec<Pick>,
    // The options the palette's filter keeps, best first, as indices into
    // `picks`. `picks` stays whole: whether the row offers anything at all is a
    // different question from what the query answers.
    pub matches: Vec<usize>,
    pub editable: bool,
}

// Every registered component name, sorted, for the `scope` and query palettes.
// Offering the vocabulary directly is what keeps "unknown component" out of the
// status line in the first place.
fn component_names() -> &'static [&'static str] {
    static NAMES: OnceLock<Vec<&'static str>> = OnceLock::new();
    NAMES.get_or_init(|| {
        let mut names: Vec<&'static str> =
            RegisteredType::all().iter().map(|t| t.as_str()).collect();
        names.sort_unstable();
        names
    })
}

impl EditorHook {
    // The `entries` indices of every Behavior, in authored order.
    pub(in crate::editor::hook) fn behavior_entries(&self) -> Vec<usize> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, e)| entry_type(e) == Some(RegisteredType::Behavior))
            .map(|(i, _)| i)
            .collect()
    }

    // The `entries` index of the open behavior. The panel holds an ordinal into
    // the list above rather than an entry index, so an unrelated add / delete /
    // undo cannot silently retarget it at another asset.
    pub(in crate::editor::hook) fn behavior_entry(&self) -> Option<usize> {
        let all = self.behavior_entries();
        all.get(self.behavior.index.min(all.len().saturating_sub(1)))
            .copied()
    }

    pub(in crate::editor::hook) fn behavior_args(&self) -> Value {
        self.behavior_entry()
            .and_then(|i| self.entries[i].get("args"))
            .filter(|a| a.is_object())
            .cloned()
            .map_or_else(|| Value::Object(serde_json::Map::new()), args_without_id)
    }

    pub(in crate::editor::hook) fn behavior_rows(&self) -> Vec<Row> {
        match self.behavior_entry() {
            Some(_) => outline::rows(&self.behavior_args()),
            None => Vec::new(),
        }
    }

    // The args of the world's `Variables` singleton, when it declares one.
    pub(super) fn variables_args(&self) -> Option<Value> {
        self.entries
            .iter()
            .find(|e| entry_type(e) == Some(RegisteredType::Variables))
            .and_then(|e| e.get("args"))
            .cloned()
            .map(args_without_id)
    }

    pub(in crate::editor::hook) fn behavior_data(&self) -> BehaviorData {
        let all = self.behavior_entries();
        let rows = self.behavior_rows();
        let selected = self.behavior.row.and_then(|i| rows.get(i));
        let chart = match self.behavior_entry() {
            Some(_) => graph::chart(&self.behavior_args()),
            None => Chart::default(),
        };
        let card = selected.and_then(|r| fields::owning_card(&chart.cards, &r.path));
        let picks = selected.map_or_else(Vec::new, |r| edit::picks(&r.kind, component_names()));
        let name = self
            .behavior_entry()
            .and_then(|i| entry_handle(&self.entries, i))
            .unwrap_or_default();
        BehaviorData {
            name,
            index: self.behavior.index.min(all.len().saturating_sub(1)),
            total: all.len(),
            matches: filter::matching(&picks, &self.behavior.filter),
            picks,
            editable: selected
                .is_some_and(|r| edit::text_value(&self.behavior_args(), r).is_some()),
            fields: card
                .map(|i| fields::own_rows(&rows, &chart.cards, i))
                .unwrap_or_default(),
            overview: match self.behavior.mode {
                ViewMode::Overview => {
                    relations::map(&self.behavior_pairs(), &self.declared_assets())
                }
                _ => Chart::default(),
            },
            card,
            chart,
            rows,
        }
    }

    // Every behavior's name and authored args, in the order the panel steps
    // through them, for the overview to map.
    pub(super) fn behavior_pairs(&self) -> Vec<(String, Value)> {
        self.behavior_entries()
            .into_iter()
            .map(|i| {
                (
                    entry_handle(&self.entries, i).unwrap_or_default(),
                    self.entries[i].get("args").cloned().unwrap_or(Value::Null),
                )
            })
            .collect()
    }

    // Every entry the world declares, beside the key the editor addresses it
    // by, so the overview can tell an asset a behavior reaches from a name the
    // world never declares -- and select the one it does. Mapping the entry
    // shape is the hook's job, as it is for the map.
    fn declared_assets(&self) -> Vec<relations::Declared> {
        let handles = entry_handles(&self.entries);
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(i, entry)| {
                Some(relations::Declared {
                    name: handles.get(i)?.clone()?,
                    asset_type: entry_type(entry)?,
                    key: self.entries.key_at(i)?,
                })
            })
            .collect()
    }

    // Bring the panel's data up to date with the model, rebuilding it only when
    // what it is built from changed, then this frame's live marks and the
    // chart canvas's shapes.
    pub(in crate::editor::hook) fn prepare_behavior(&mut self, world: &mut World) {
        self.shown_behavior();
        self.behavior.reveal_now = self.behavior.reveal.take();
        self.behavior.reveal_pick_now = std::mem::take(&mut self.behavior.reveal_pick);
        self.mark_behavior();
        self.write_behavior_chart(world);
    }

    // The panel's data as the user sees it, rebuilt first only when what it is
    // built from changed. Actions resolve what was pressed against this rather
    // than against a fresh build, so they answer the panel on screen.
    pub(in crate::editor::hook) fn shown_behavior(&mut self) -> &BehaviorData {
        let stale = self.behavior.shown.as_ref().is_none_or(|s| {
            s.entries != self.entries.revision()
                || s.index != self.behavior.index
                || s.row != self.behavior.row
                || s.mode != self.behavior.mode
                || s.filter != self.behavior.filter
        });
        let shown = if let Some(shown) = self.behavior.shown.take().filter(|_| !stale) {
            shown
        } else {
            self.behavior.revision = self.behavior.revision.wrapping_add(1);
            Shown {
                entries: self.entries.revision(),
                index: self.behavior.index,
                row: self.behavior.row,
                mode: self.behavior.mode,
                filter: self.behavior.filter.clone(),
                data: self.behavior_data(),
            }
        };
        &self.behavior.shown.insert(shown).data
    }

    // This frame's pulses and breakpoints over the shown body.
    fn mark_behavior(&mut self) {
        let Some(shown) = self.behavior.shown.as_ref() else {
            return;
        };
        let data = &shown.data;
        let marks = &mut self.behavior.marks;
        marks.pulse_cards.clear();
        marks.pulse_rows.clear();
        marks.break_cards.clear();
        for p in &self.behavior.pulses {
            let alpha = pulse::alpha(p.age);
            if alpha <= 0.0 {
                continue;
            }
            if let Some(c) = fields::owning_card(&data.chart.cards, &p.path) {
                marks.pulse_cards.push((c, alpha));
            }
            if let Some(r) = data.rows.iter().position(|r| r.path == p.path) {
                marks.pulse_rows.push((r, alpha));
            }
        }
        for (name, path) in &self.behavior.breakpoints {
            if *name == data.name
                && let Some(c) = data.chart.cards.iter().position(|c| &c.path == path)
            {
                marks.break_cards.push(c);
            }
        }
    }

    // Write the chart showing as the canvas's shapes. Pulses and breakpoints
    // address nodes of the open body, so the overview (whole behaviors) draws
    // none, and keeps a selection of its own.
    fn write_behavior_chart(&mut self, world: &mut World) {
        let Some(shown) = self.behavior.shown.as_ref() else {
            return;
        };
        let data = &shown.data;
        let overview = self.behavior.mode == ViewMode::Overview;
        let fault_row = self.behavior_fault_row(data);
        let marks = &mut self.behavior.marks;
        let view = if overview {
            ChartMarks {
                chart: &data.overview,
                selected: self.behavior.overview_card,
                faulted: None,
                hovered: self.behavior.hovered,
                pulses: &[],
                breakpoints: &[],
            }
        } else {
            ChartMarks {
                chart: &data.chart,
                selected: data.card,
                faulted: panel::fault_card(&data.chart, &data.rows, fault_row),
                hovered: self.behavior.hovered,
                pulses: &marks.pulse_cards,
                breakpoints: &marks.break_cards,
            }
        };
        let taken = &mut marks.taken;
        state::with_state(world, |s| {
            s.edit_shapes(panel::SHAPES, |list| {
                let mut shapes = Shapes::new(list);
                chart::push(&mut shapes, &view, taken);
                shapes.finish();
            });
        });
    }

    pub(in crate::editor::hook) fn behavior_fault_row(&self, data: &BehaviorData) -> Option<usize> {
        self.behavior
            .status
            .as_ref()
            .and_then(|s| fault::row_of(&data.rows, s.at()))
    }

    // What the panel shows this frame, once its data is prepared.
    pub(in crate::editor::hook) fn make_behavior_view(&self) -> Option<BehaviorView<'_>> {
        let data = &self.behavior.shown.as_ref()?.data;
        // Focus is asserted only while frontmost, so a buried panel's field
        // cannot steal the keyboard.
        let front = self.panel_order.last() == Some(&PanelKey::Behavior);
        let picking = self.behavior.picking && !data.picks.is_empty();
        Some(BehaviorView {
            index: data.index,
            total: data.total,
            rows: &data.rows,
            selected: self.behavior.row,
            mode: self.behavior.mode,
            chart: &data.chart,
            card: data.card,
            fields: &data.fields,
            picks: &data.picks,
            matches: &data.matches,
            picking,
            pick: self.behavior.pick,
            editable: data.editable,
            focus: self.behavior.focus && data.editable && front,
            name_focus: self.behavior.name_focus && front,
            filter_focus: picking && front,
            remove_armed: self.behavior.remove_armed,
            status: self.behavior.status.as_ref(),
            fault_row: self.behavior_fault_row(data),
            pulse_rows: &self.behavior.marks.pulse_rows,
            revision: self.behavior.revision,
            reveal: self.behavior.reveal_now,
            reveal_pick: self.behavior.reveal_pick_now,
            pan: self.behavior.pan,
        })
    }

    // The panel's three fields: the name, the value, the filter.
    pub(in crate::editor::hook) fn behavior_fields(&self) -> [Option<AssetId>; 3] {
        self.behavior.fields
    }

    // Take in what the user did to the panel's widgets that no event reports:
    // a field a press focused or left, what is typed into the filter, where
    // the canvas was panned to, and the card under the cursor.
    pub(in crate::editor::hook) fn observe_behavior(
        &mut self,
        world: &mut World,
        built: &BuiltPanel,
    ) {
        let [name, value, _] = panel::focused(world, built);
        let front = self.panel_order.last() == Some(&PanelKey::Behavior);
        if name != (self.behavior.name_focus && front) {
            if name {
                self.focus_behavior_name(world);
            } else {
                self.blur_behavior_name(world);
            }
        }
        let editable = self
            .behavior
            .shown
            .as_ref()
            .is_some_and(|s| s.data.editable);
        if value != (self.behavior.focus && editable && front) {
            self.behavior.focus = value;
            if value {
                self.blur_behavior_name(world);
            }
        }
        let [name, value, _] = panel::fields(built);
        for (typed, id) in self.behavior.typed.iter_mut().zip([name, value]) {
            if let Some(f) = id.and_then(|id| world.get_by_id::<TextField>(id))
                && *typed != f.content
            {
                typed.clone_from(&f.content);
            }
        }
        self.sample_behavior_filter(world);
        if let Some(c) = panel::canvas_of(built).and_then(|id| world.get_by_id::<Canvas>(id)) {
            self.behavior.pan = c.pan;
            self.behavior.hovered = c.hovered.and_then(|t| usize::try_from(t).ok());
        }
    }

    // Read the palette's filter off its field. A narrowed palette is a different
    // list, so the highlight starts again at its best answer rather than
    // keeping a place that may no longer be in it.
    pub(in crate::editor::hook) fn sample_behavior_filter(&mut self, world: &World) {
        let [_, _, filter] = self.behavior_fields();
        let typed = filter
            .filter(|_| self.behavior.picking)
            .and_then(|id| world.get_by_id::<TextField>(id))
            .map_or("", |f| f.content.as_str());
        if typed != self.behavior.filter {
            self.behavior.pick = 0;
            self.behavior.reveal_pick = true;
            self.behavior.filter = typed.to_string();
        }
    }

    // Give a world the panel was just built into its fields: what is being
    // typed into one holding the keyboard, and the model's own text otherwise.
    pub(in crate::editor::hook) fn restore_behavior(
        &mut self,
        world: &mut World,
        built: &BuiltPanel,
    ) {
        self.behavior.fields = panel::fields(built);
        let [name, value, _] = self.behavior_fields();
        if self.behavior.name_focus {
            field::set_text(world, name, &self.behavior.typed[0]);
        } else {
            self.seed_behavior_name(world);
        }
        if self.behavior.focus {
            field::set_text(world, value, &self.behavior.typed[1]);
        } else {
            self.seed_behavior_value(world);
        }
        self.behavior.shown = None;
    }

    // A press on one of the panel's controls.
    pub(in crate::editor::hook) fn on_behavior_event(
        &mut self,
        world: &mut World,
        event: &UiEvent,
    ) -> bool {
        let Some(action) = panel::action(event) else {
            return false;
        };
        self.apply_behavior_action(action, world);
        true
    }

    // (Re)open the panel on the behavior the ordinal points at: drop any stale
    // selection and palette, seed the value field, and re-run the checker.
    pub(in crate::editor::hook) fn open_behavior(&mut self, world: &mut World) {
        let all = self.behavior_entries();
        self.behavior.index = self.behavior.index.min(all.len().saturating_sub(1));
        self.behavior.row = None;
        self.behavior.picking = false;
        self.clear_behavior_filter(world);
        self.behavior.focus = false;
        self.behavior.name_focus = false;
        self.behavior.remove_armed = false;
        self.refresh_behavior_status();
        self.seed_behavior_value(world);
        self.seed_behavior_name(world);
    }

    // Run the world checker over the open behavior as it now stands. The
    // message is the checker's own, so the panel never disagrees with the build.
    pub(in crate::editor::hook) fn refresh_behavior_status(&mut self) {
        let Some(idx) = self.behavior_entry() else {
            self.behavior.status = None;
            return;
        };
        let name = entry_handle(&self.entries, idx).unwrap_or_default();
        let args = self.behavior_args();
        let vars = self.variables_args();
        self.behavior.status = Some(
            match concinnity_cook::check::behavior::check_with_variables(
                &name,
                &args,
                vars.as_ref(),
            ) {
                Ok(()) => Status::Ok,
                // The banner is two lines, so it carries the checker's first
                // one; where the complaint is about survives whole.
                Err(e) => Status::Error {
                    message: e.message.lines().next().unwrap_or(&e.message).to_string(),
                    at: fault::to_path(&e.at),
                },
            },
        );
    }

    // Seed the value field from the selected row (blank when the row carries no
    // typed value).
    pub(super) fn seed_behavior_value(&self, world: &mut World) {
        let args = self.behavior_args();
        let text = self
            .behavior
            .row
            .and_then(|i| self.behavior_rows().get(i).cloned())
            .and_then(|r| edit::text_value(&args, &r))
            .unwrap_or_default();
        let [_, value, _] = self.behavior_fields();
        field::set_text(world, value, &text);
    }

    // Write the edited args back onto the open entry, refresh the checker's
    // verdict, and re-seed the value field. `mark_changed` swaps the live
    // preview world so the edited body runs immediately.
    fn commit_behavior(&mut self, args: Value, world: &mut World) {
        let Some(idx) = self.behavior_entry() else {
            return;
        };
        replace_args(&mut self.entries[idx], args);
        self.mark_changed();
        let prev_fault = self.behavior_fault_message();
        self.refresh_behavior_status();
        self.notify_behavior_fault(prev_fault.as_deref());
        self.seed_behavior_value(world);
    }

    // Carry out what the panel or a key asked for.
    pub(in crate::editor::hook) fn apply_behavior_action(
        &mut self,
        action: BehaviorAction,
        world: &mut World,
    ) {
        // An armed removal and a focused name field last only until the next
        // press, taken before the action runs.
        let armed = std::mem::take(&mut self.behavior.remove_armed);
        self.blur_behavior_name(world);
        match action {
            BehaviorAction::Step(delta) => self.step_behavior(delta, world),
            BehaviorAction::New => self.add_behavior(world),
            BehaviorAction::Remove => self.remove_behavior(armed, world),
            BehaviorAction::Select(i) => self.select_behavior_row(i, world),
            BehaviorAction::SelectField(slot) => {
                if let Some(&row) = self.shown_behavior().fields.get(slot) {
                    self.select_behavior_row(row, world);
                }
            }
            BehaviorAction::Palette => {
                self.behavior.picking = !self.behavior.picking;
                self.clear_behavior_filter(world);
                self.behavior.focus = false;
            }
            BehaviorAction::Choose(slot) => {
                if let Some(&i) = self.shown_behavior().matches.get(slot) {
                    self.choose_behavior_pick(i, world);
                }
            }
            BehaviorAction::Dismiss => {
                self.behavior.picking = false;
                self.clear_behavior_filter(world);
            }
            BehaviorAction::Delete => self.delete_behavior_row(world),
            BehaviorAction::Move(delta) => self.move_behavior_row(delta as isize, world),
            BehaviorAction::ToggleView => self.toggle_behavior_view(),
            BehaviorAction::Card(i) => self.press_behavior_card(i, world),
            BehaviorAction::SelectCard(i) => self.select_behavior_card(i, world),
            BehaviorAction::OpenCard(i) => self.open_behavior_card(i, world),
            BehaviorAction::OpenVariable(i) => self.open_overview_variable(i, world),
            BehaviorAction::SelectAsset(i) => self.select_overview_asset(i),
            BehaviorAction::GoToFault => self.select_behavior_fault(world),
            BehaviorAction::Copy => self.copy_behavior_row(),
            BehaviorAction::Paste => self.paste_behavior_row(world),
            BehaviorAction::Duplicate => self.duplicate_behavior_row(world),
        }
    }

    // A press on a card of the chart showing: the open body's node, or what an
    // overview card stands for.
    fn press_behavior_card(&mut self, i: usize, world: &mut World) {
        self.behavior.focus = false;
        let action = match self.behavior.mode {
            ViewMode::Overview => self
                .shown_behavior()
                .overview
                .cards
                .get(i)
                .and_then(|card| panel::overview_action(card, i)),
            _ => Some(BehaviorAction::SelectCard(i)),
        };
        if let Some(action) = action {
            self.apply_behavior_action(action, world);
        }
    }

    // Open the previous / next behavior, wrapping at either end.
    fn step_behavior(&mut self, delta: i32, world: &mut World) {
        let total = self.behavior_entries().len();
        if total == 0 {
            return;
        }
        let at = i32::saturating_from(self.behavior.index.min(total - 1));
        self.behavior.index = (at + delta).rem_euclid(i32::saturating_from(total)) as usize;
        self.open_behavior(world);
    }

    // Append a blank Behavior and open it. Nothing about a blank one is inert
    // now that the panel builds its body, which is why it is also offered by
    // the Assets panel's type picker.
    fn add_behavior(&mut self, world: &mut World) {
        let name = self.unique_name("behavior");
        self.entries.push(serde_json::json!({
            "type": "Behavior", "args": {"$id": name, "on": "start", "do": []},
        }));
        self.mark_changed();
        self.behavior.index = self.behavior_entries().len().saturating_sub(1);
        self.open_behavior(world);
    }

    // Select a row. Selecting never edits: a row that offers options lights the
    // Pick button, and a row that takes typed text is ready to type into
    // straight away.
    pub(in crate::editor::hook) fn select_behavior_row(&mut self, i: usize, world: &mut World) {
        self.behavior.row = Some(i);
        self.behavior.picking = false;
        let Some(row) = self.behavior_rows().get(i).cloned() else {
            return;
        };
        self.seed_behavior_value(world);
        self.behavior.focus = edit::text_value(&self.behavior_args(), &row).is_some();
    }

    // A palette opens on the whole vocabulary: a query is about the pick being
    // made, so it never outlives it.
    fn clear_behavior_filter(&mut self, world: &mut World) {
        self.behavior.filter.clear();
        self.behavior.pick = 0;
        self.behavior.reveal_pick = true;
        let [_, _, filter] = self.behavior_fields();
        field::set_text(world, filter, "");
    }

    fn choose_behavior_pick(&mut self, i: usize, world: &mut World) {
        self.behavior.picking = false;
        self.clear_behavior_filter(world);
        let data = self.behavior_data();
        let (Some(row), Some(pick)) = (
            self.behavior.row.and_then(|r| data.rows.get(r)),
            data.picks.get(i),
        ) else {
            return;
        };
        let (row, verb) = (row.clone(), pick.verb);
        let mut args = self.behavior_args();
        if edit::apply_pick(&mut args, &row, verb) {
            self.commit_behavior(args, world);
        }
    }

    fn delete_behavior_row(&mut self, world: &mut World) {
        let Some(row) = self
            .behavior
            .row
            .and_then(|i| self.behavior_rows().get(i).cloned())
        else {
            return;
        };
        let mut args = self.behavior_args();
        if edit::remove(&mut args, &row) {
            self.commit_behavior(args, world);
            // The removed row is gone; whatever slid into its place is not what
            // the user had selected, so the selection is dropped rather than
            // silently retargeted.
            self.behavior.row = None;
            self.behavior.focus = false;
        }
    }

    fn move_behavior_row(&mut self, delta: isize, world: &mut World) {
        let Some(row) = self
            .behavior
            .row
            .and_then(|i| self.behavior_rows().get(i).cloned())
        else {
            return;
        };
        let mut args = self.behavior_args();
        let Some(moved) = edit::shift(&mut args, &row, delta) else {
            return;
        };
        self.commit_behavior(args, world);
        // Follow the member to wherever it landed, so repeated moves keep
        // acting on the same node rather than on whatever took its row.
        self.behavior.row = self
            .behavior_rows()
            .iter()
            .position(|r| r.element.as_ref() == Some(&moved));
        self.ensure_behavior_visible();
    }

    pub(in crate::editor::hook) fn commit_behavior_value(&mut self, world: &mut World) {
        if !self.behavior.focus {
            return;
        }
        let Some(row) = self
            .behavior
            .row
            .and_then(|i| self.behavior_rows().get(i).cloned())
        else {
            return;
        };
        let [_, value, _] = self.behavior_fields();
        let text = field::text(world, value);
        let mut args = self.behavior_args();
        match edit::apply_text(&mut args, &row, &text) {
            Ok(()) => self.commit_behavior(args, world),
            Err(e) => self.behavior.status = Some(Status::message(e)),
        }
    }

    // Step to the next view. The selection survives, because the outline and the
    // chart are over the same rows: a card selected in the chart is the row the
    // outline opens on. The pan does not, because each chart is its own shape.
    fn toggle_behavior_view(&mut self) {
        self.behavior.toggle_view();
        if self.behavior.mode == ViewMode::Overview {
            // The map opens on the behavior that was showing, so it says where
            // the panel already is rather than starting from nothing.
            let at = self.behavior_data().index;
            self.point_overview_at(at);
        }
        self.ensure_behavior_visible();
    }

    // Put the overview's own cursor on the card standing for behavior `at`,
    // which is what the keyboard steps from and the canvas pans to.
    fn point_overview_at(&mut self, at: usize) {
        self.behavior.overview_card = self
            .behavior_data()
            .overview
            .cards
            .iter()
            .position(|c| c.behavior == Some(at));
    }

    // Open the behavior an overview card stands for, in the chart view, so
    // clicking through the map lands on the body it named. The card stands for
    // a world asset too, so it is selected like one on the way: the Assets tree
    // and the places map then agree on what the panel is showing.
    fn open_behavior_card(&mut self, i: usize, world: &mut World) {
        let card = self.shown_behavior().overview.cards.get(i).cloned();
        let Some(at) = card.as_ref().and_then(|c| c.behavior) else {
            return;
        };
        self.behavior.overview_card = Some(i);
        if let Some(handle) = card.and_then(|c| c.handle) {
            self.select_handle(&handle);
        }
        self.behavior.index = at;
        self.behavior.mode = ViewMode::Chart;
        self.behavior.pan = [0.0, 0.0];
        self.open_behavior(world);
    }

    // Select the world asset an overview card stands for. The overview says
    // which behaviors reach a scene or a screen; where that place sits in the
    // world is the Map's question, and the shared selection is what carries it
    // there.
    fn select_overview_asset(&mut self, i: usize) {
        let Some(handle) = self
            .shown_behavior()
            .overview
            .cards
            .get(i)
            .and_then(|c| c.handle.clone())
        else {
            return;
        };
        self.behavior.overview_card = Some(i);
        self.select_handle(&handle);
    }

    // Open the Behavior panel on the named behavior in its current view. A
    // generated behavior falls back to the edit form.
    pub(in crate::editor::hook) fn open_behavior_named(&mut self, name: &str, world: &mut World) {
        let target = find_entry(&self.entries, name);
        let ordinal = self
            .behavior_entries()
            .iter()
            .position(|&i| Some(i) == target);
        let Some(ordinal) = ordinal else {
            self.select_named(name);
            self.focus_ui_on(name);
            return;
        };
        self.behavior.index = ordinal;
        self.open[PanelKey::Behavior] = true;
        self.open_behavior(world);
        self.focus_panel(PanelKey::Behavior);
        if self.behavior.mode == ViewMode::Overview {
            self.point_overview_at(ordinal);
        }
        self.ensure_behavior_visible();
    }

    // Hold the selected member. Nothing is written, so this is not an edit and
    // takes no history snapshot.
    fn copy_behavior_row(&mut self) {
        let data = self.behavior_data();
        let Some(row) = self.behavior.row.and_then(|i| data.rows.get(i)) else {
            return;
        };
        if let Some(held) = clip::of(&self.behavior_args(), &data.rows, row) {
            self.behavior.clip = Some(held);
        }
    }

    fn paste_behavior_row(&mut self, world: &mut World) {
        let Some(held) = self.behavior.clip.clone() else {
            return;
        };
        self.place_behavior_clip(&held, world);
    }

    // A duplicate is a paste of the selection itself, so it leaves whatever is
    // held for a later paste alone.
    fn duplicate_behavior_row(&mut self, world: &mut World) {
        let data = self.behavior_data();
        let Some(row) = self.behavior.row.and_then(|i| data.rows.get(i)) else {
            return;
        };
        let Some(held) = clip::of(&self.behavior_args(), &data.rows, row) else {
            return;
        };
        self.place_behavior_clip(&held, world);
    }

    // Put `held` where the selection says, then follow the copy: the next press
    // acts on what just landed rather than on what it came from.
    fn place_behavior_clip(&mut self, held: &clip::Clip, world: &mut World) {
        let data = self.behavior_data();
        let Some(row) = self.behavior.row.and_then(|i| data.rows.get(i)) else {
            return;
        };
        let mut args = self.behavior_args();
        let Some(landed) = clip::paste(&mut args, &data.rows, row, held) else {
            return;
        };
        self.commit_behavior(args, world);
        self.behavior.row = self
            .behavior_rows()
            .iter()
            .position(|r| r.element.as_ref() == Some(&landed));
        self.seed_behavior_value(world);
        self.ensure_behavior_visible();
    }

    // Go to what the checker is complaining about. The overview maps the world
    // rather than the open behavior's body, so it steps to the chart on the way:
    // a complaint is about a place inside one behavior, and the chart is the view
    // that shows one.
    pub(in crate::editor::hook) fn select_behavior_fault(&mut self, world: &mut World) {
        let data = self.behavior_data();
        let Some(row) = self
            .behavior
            .status
            .as_ref()
            .and_then(|s| fault::row_of(&data.rows, s.at()))
        else {
            return;
        };
        if self.behavior.mode == ViewMode::Overview {
            self.behavior.mode = ViewMode::Chart;
            self.behavior.pan = [0.0, 0.0];
        }
        self.select_behavior_row(row, world);
        self.ensure_behavior_visible();
    }

    // Open the world's variable table on the variable an overview card stands
    // for. Behaviors reach each other through world state, so the map's variable
    // cards are the natural way into the table that declares it.
    fn open_overview_variable(&mut self, i: usize, world: &mut World) {
        let Some(name) = self
            .shown_behavior()
            .overview
            .cards
            .get(i)
            .map(|c| c.title.clone())
        else {
            return;
        };
        self.open_variable_named(&name, world);
    }

    // Select the row the card at `i` stands for. Cards cover the body and the
    // source; the declarations are reached from the outline. A Ctrl+click on
    // a node card toggles its breakpoint instead: the run pauses when that
    // node next executes.
    fn select_behavior_card(&mut self, i: usize, world: &mut World) {
        let ctrl = self.ctrl_held;
        let data = self.shown_behavior();
        let Some(card) = data.chart.cards.get(i) else {
            return;
        };
        let path = card.path.clone();
        if ctrl && card.kind == CardKind::Node {
            self.toggle_behavior_breakpoint(&path);
            return;
        }
        let Some(row) = data.rows.iter().position(|r| r.path == path) else {
            return;
        };
        self.select_behavior_row(row, world);
    }

    fn behavior_canvas(&self) -> [f32; 2] {
        panel::chart_canvas(self.effective_size(PanelKey::Behavior), self.behavior.mode)
    }

    // Bring what is selected into view, whichever view is showing it: the
    // outline scrolls to its row, and either chart pans to its card.
    pub(in crate::editor::hook) fn ensure_behavior_visible(&mut self) {
        match self.behavior.mode {
            ViewMode::Overview => self.pan_to_overview_card(),
            ViewMode::Chart => {
                if let Some(row) = self.behavior.row {
                    self.pan_to_behavior_row(row);
                }
            }
            ViewMode::Outline => self.behavior.reveal = self.behavior.row,
        }
    }

    fn pan_to_overview_card(&mut self) {
        let data = self.behavior_data();
        let Some(card) = self
            .behavior
            .overview_card
            .and_then(|i| data.overview.cards.get(i))
        else {
            return;
        };
        self.behavior.pan = chart::pan_to(
            card,
            self.behavior_canvas(),
            self.behavior.pan,
            &data.overview,
        );
    }

    // Bring the card `row` belongs to into the canvas, so selecting one of a
    // node's fields brings the node itself into view. A row no card owns (a
    // declaration) leaves the pan alone.
    fn pan_to_behavior_row(&mut self, row: usize) {
        let data = self.behavior_data();
        let Some(card) = data
            .rows
            .get(row)
            .and_then(|r| fields::owning_card(&data.chart.cards, &r.path))
            .and_then(|i| data.chart.cards.get(i))
        else {
            return;
        };
        self.behavior.pan =
            chart::pan_to(card, self.behavior_canvas(), self.behavior.pan, &data.chart);
    }
}
