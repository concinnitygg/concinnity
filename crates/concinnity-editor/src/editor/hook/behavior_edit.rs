// src/editor/hook/behavior_edit.rs
//
// EditorHook: the Behavior panel's actions. The panel opens one `Behavior`
// entry at a time and edits its authored args directly, so every change is an
// ordinary world edit -- the live preview rebuilds from the in-memory entries
// and SAVE persists them, like any other panel. What the keyboard does with
// these actions is `behavior_keys.rs`.
//
// A behavior body has no unbounded loop and no recursion, so it always
// terminates; that is what makes running an edited body in the live world safe
// and why edits commit as they are made rather than behind an Apply button.
//
// Validation is not repeated here: the world's own checker runs against the
// same args after each commit and its message goes straight to the status line,
// with the world's `Variables` table supplied so a misspelled variable is
// caught the way the build would catch it. Cross-asset names (a spawn template,
// a clip) resolve against the whole world, so those stay a build-time check.

use std::sync::OnceLock;

use concinnity_cook::ComponentType;
use serde_json::Value;

use super::*;
use crate::editor::behavior::clip;
use crate::editor::behavior::edit::{self, Pick};
use crate::editor::behavior::fault;
use crate::editor::behavior::fields;
use crate::editor::behavior::filter;
use crate::editor::behavior::graph::{self, CardKind, Chart};
use crate::editor::behavior::navigate;
use crate::editor::behavior::outline::{self, Row};
use crate::editor::behavior::pulse;
use crate::editor::behavior::relations;
use crate::editor::behavior_chart;

// Owned per-tick data backing a `BehaviorView`: the open behavior's outline and
// the palette its selected row offers.
pub(super) struct BehaviorData {
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
    // Live-debug marks over the body: cards / rows whose node just executed
    // (with each pulse's remaining strength), and the cards holding a
    // breakpoint. All empty outside a live session.
    pub pulse_cards: Vec<(usize, f32)>,
    pub pulse_rows: Vec<(usize, f32)>,
    pub break_cards: Vec<usize>,
}

// How far one wheel notch pans the chart.
const WHEEL_PAN: f32 = 40.0;

// Every registered component name, sorted, for the `scope` and query palettes.
// Offering the vocabulary directly is what keeps "unknown component" out of the
// status line in the first place.
fn component_names() -> &'static [&'static str] {
    static NAMES: OnceLock<Vec<&'static str>> = OnceLock::new();
    NAMES.get_or_init(|| {
        let mut names: Vec<&'static str> =
            ComponentType::all().iter().map(|t| t.as_str()).collect();
        names.sort_unstable();
        names
    })
}

impl EditorHook {
    // The `entries` indices of every Behavior, in authored order.
    pub(super) fn behavior_entries(&self) -> Vec<usize> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, e)| entry_type(e) == Some("Behavior"))
            .map(|(i, _)| i)
            .collect()
    }

    // The `entries` index of the open behavior. The panel holds an ordinal into
    // the list above rather than an entry index, so an unrelated add / delete /
    // undo cannot silently retarget it at another asset.
    pub(super) fn behavior_entry(&self) -> Option<usize> {
        let all = self.behavior_entries();
        all.get(self.behavior_index.min(all.len().saturating_sub(1)))
            .copied()
    }

    pub(super) fn behavior_args(&self) -> Value {
        self.behavior_entry()
            .and_then(|i| self.entries[i].get("args"))
            .filter(|a| a.is_object())
            .cloned()
            .unwrap_or_else(|| Value::Object(serde_json::Map::new()))
    }

    pub(super) fn behavior_rows(&self) -> Vec<Row> {
        match self.behavior_entry() {
            Some(_) => outline::rows(&self.behavior_args()),
            None => Vec::new(),
        }
    }

    // The args of the world's `Variables` singleton, when it declares one.
    pub(super) fn variables_args(&self) -> Option<Value> {
        self.entries
            .iter()
            .find(|e| entry_type(e) == Some("Variables"))
            .and_then(|e| e.get("args"))
            .cloned()
    }

    pub(super) fn behavior_data(&self) -> BehaviorData {
        let all = self.behavior_entries();
        let rows = self.behavior_rows();
        let selected = self.behavior_row.and_then(|i| rows.get(i));
        let chart = match self.behavior_entry() {
            Some(_) => graph::chart(&self.behavior_args()),
            None => Chart::default(),
        };
        let card = selected.and_then(|r| fields::owning_card(&chart.cards, &r.path));
        let picks = selected.map_or_else(Vec::new, |r| edit::picks(&r.kind, component_names()));
        let name = self
            .behavior_entry()
            .and_then(|i| entry_name(&self.entries[i]))
            .unwrap_or("")
            .to_string();
        let mut pulse_cards = Vec::new();
        let mut pulse_rows = Vec::new();
        for p in &self.behavior_pulses {
            let alpha = pulse::alpha(p.at.elapsed().as_secs_f32());
            if alpha <= 0.0 {
                continue;
            }
            if let Some(c) = fields::owning_card(&chart.cards, &p.path) {
                pulse_cards.push((c, alpha));
            }
            if let Some(r) = rows.iter().position(|r| r.path == p.path) {
                pulse_rows.push((r, alpha));
            }
        }
        let break_cards = self
            .behavior_breakpoints
            .iter()
            .filter(|(n, _)| *n == name)
            .filter_map(|(_, path)| chart.cards.iter().position(|c| &c.path == path))
            .collect();
        BehaviorData {
            name,
            index: self.behavior_index.min(all.len().saturating_sub(1)),
            total: all.len(),
            matches: filter::matching(&picks, &self.behavior_filter),
            picks,
            editable: selected
                .is_some_and(|r| edit::text_value(&self.behavior_args(), r).is_some()),
            fields: card
                .map(|i| fields::own_rows(&rows, &chart.cards, i))
                .unwrap_or_default(),
            overview: match self.behavior_mode {
                ViewMode::Overview => {
                    relations::map(&self.behavior_pairs(), &self.declared_assets())
                }
                _ => Chart::default(),
            },
            card,
            chart,
            rows,
            pulse_cards,
            pulse_rows,
            break_cards,
        }
    }

    // Every behavior's name and authored args, in the order the panel steps
    // through them, for the overview to map.
    pub(super) fn behavior_pairs(&self) -> Vec<(String, Value)> {
        self.behavior_entries()
            .into_iter()
            .map(|i| {
                let e = &self.entries[i];
                (
                    entry_name(e).unwrap_or("").to_string(),
                    e.get("args").cloned().unwrap_or(Value::Null),
                )
            })
            .collect()
    }

    // Every entry's name and type, so the overview can tell an asset a behavior
    // reaches from a name the world never declares. Mapping the entry shape is
    // the hook's job; the map itself only ever sees names and types.
    fn declared_assets(&self) -> Vec<(&str, &str)> {
        self.entries
            .iter()
            .filter_map(|e| Some((entry_name(e)?, entry_type(e)?)))
            .collect()
    }

    pub(super) fn make_behavior_view<'a>(
        &'a self,
        data: &'a BehaviorData,
        mouse: [f32; 2],
    ) -> BehaviorView<'a> {
        BehaviorView {
            name: &data.name,
            index: data.index,
            total: data.total,
            rows: &data.rows,
            scroll: self.behavior_scroll,
            selected: self.behavior_row,
            chart: &data.chart,
            overview: &data.overview,
            mode: self.behavior_mode,
            pan: self.behavior_pan,
            card: data.card,
            fields: &data.fields,
            picks: &data.picks,
            matches: &data.matches,
            picking: self.behavior_picking && !data.picks.is_empty(),
            pick_scroll: self.behavior_pick_scroll,
            pick: self.behavior_pick,
            filter_focus: self.behavior_picking
                && !data.picks.is_empty()
                && self.panel_order.last() == Some(&PanelKey::Behavior),
            overview_card: self.behavior_overview_card,
            editable: data.editable,
            // Focus is asserted only while frontmost, so a buried panel's field
            // cannot steal the keyboard.
            focus: self.behavior_focus
                && data.editable
                && self.panel_order.last() == Some(&PanelKey::Behavior),
            name_focus: self.behavior_name_focus
                && self.panel_order.last() == Some(&PanelKey::Behavior),
            remove_armed: self.behavior_remove_armed,
            fault_row: self
                .behavior_status
                .as_ref()
                .and_then(|s| fault::row_of(&data.rows, s.at())),
            status: self.behavior_status.as_ref(),
            pulse_cards: &data.pulse_cards,
            pulse_rows: &data.pulse_rows,
            break_cards: &data.break_cards,
            mouse,
        }
    }

    // (Re)open the panel on the behavior the ordinal points at: drop any stale
    // selection and palette, seed the value field, and re-run the checker.
    pub(super) fn open_behavior(&mut self, world: &mut World) {
        let all = self.behavior_entries();
        self.behavior_index = self.behavior_index.min(all.len().saturating_sub(1));
        self.behavior_row = None;
        self.behavior_scroll = 0;
        self.behavior_picking = false;
        self.clear_behavior_filter(world);
        self.behavior_focus = false;
        self.behavior_name_focus = false;
        self.behavior_remove_armed = false;
        self.refresh_behavior_status();
        self.seed_behavior_value(world);
        self.seed_behavior_name(world);
    }

    // Run the world checker over the open behavior as it now stands. The
    // message is the checker's own, so the panel never disagrees with the build.
    pub(super) fn refresh_behavior_status(&mut self) {
        let Some(idx) = self.behavior_entry() else {
            self.behavior_status = None;
            return;
        };
        let name = entry_name(&self.entries[idx]).unwrap_or("").to_string();
        let args = self.behavior_args();
        let vars = self.variables_args();
        self.behavior_status = Some(
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
    pub(super) fn seed_behavior_value(&mut self, world: &mut World) {
        let args = self.behavior_args();
        let text = self
            .behavior_row
            .and_then(|i| self.behavior_rows().get(i).cloned())
            .and_then(|r| edit::text_value(&args, &r))
            .unwrap_or_default();
        widget::seed_field(world, behavior_panel::VALUE_INPUT, &text);
    }

    // Write the edited args back onto the open entry, refresh the checker's
    // verdict, and re-seed the value field. `mark_changed` swaps the live
    // preview world so the edited body runs immediately.
    fn commit_behavior(&mut self, args: Value, world: &mut World) {
        let Some(idx) = self.behavior_entry() else {
            return;
        };
        let Some(entry) = self.entries[idx].as_object_mut() else {
            return;
        };
        entry.insert("args".to_string(), args);
        self.mark_changed();
        self.refresh_behavior_status();
        self.seed_behavior_value(world);
    }

    // Route a resolved Behavior-panel click.
    pub(super) fn apply_behavior_action(
        &mut self,
        action: BehaviorAction,
        world: &mut World,
        mouse: [f32; 2],
    ) {
        // An armed removal and a focused name field both last only until the
        // next press: the chip cannot sit armed behind whatever the user does
        // next, and the keyboard cannot stay in a field they have clicked away
        // from. Both are taken before the action runs, so the press that arms
        // the chip is not also the press that disarms it.
        let armed = std::mem::take(&mut self.behavior_remove_armed);
        if action != BehaviorAction::FocusName {
            self.blur_behavior_name(world);
        }
        match action {
            BehaviorAction::Step(delta) => self.step_behavior(delta, world),
            BehaviorAction::New => self.add_behavior(world),
            BehaviorAction::Remove => self.remove_behavior(armed, world),
            BehaviorAction::FocusName => self.focus_behavior_name(world),
            BehaviorAction::Select(i) => self.select_behavior_row(i, world),
            BehaviorAction::Palette => {
                self.behavior_picking = !self.behavior_picking;
                self.clear_behavior_filter(world);
                self.behavior_focus = false;
            }
            BehaviorAction::Choose(i) => self.choose_behavior_pick(i, world),
            BehaviorAction::Dismiss => {
                self.behavior_picking = false;
                self.clear_behavior_filter(world);
            }
            BehaviorAction::Delete => self.delete_behavior_row(world),
            BehaviorAction::Move(delta) => self.move_behavior_row(delta as isize, world),
            BehaviorAction::FocusValue => self.behavior_focus = true,
            BehaviorAction::ToggleView => self.toggle_behavior_view(),
            BehaviorAction::SelectCard(i) => self.select_behavior_card(i, world),
            BehaviorAction::OpenCard(i) => self.open_behavior_card(i, world),
            BehaviorAction::OpenVariable(i) => self.open_overview_variable(i, world),
            BehaviorAction::PanStart => self.start_behavior_pan(mouse),
            BehaviorAction::GoToFault => self.select_behavior_fault(world),
            BehaviorAction::Copy => self.copy_behavior_row(),
            BehaviorAction::Paste => self.paste_behavior_row(world),
            BehaviorAction::Duplicate => self.duplicate_behavior_row(world),
            BehaviorAction::Consume => self.behavior_focus = false,
        }
    }

    // Open the previous / next behavior, wrapping at either end.
    fn step_behavior(&mut self, delta: i32, world: &mut World) {
        let total = self.behavior_entries().len();
        if total == 0 {
            return;
        }
        let at = self.behavior_index.min(total - 1) as i32;
        self.behavior_index = (at + delta).rem_euclid(total as i32) as usize;
        self.open_behavior(world);
    }

    // Append a blank Behavior and open it. Nothing about a blank one is inert
    // now that the panel builds its body, which is why it is also offered by
    // the Assets panel's type picker.
    fn add_behavior(&mut self, world: &mut World) {
        let name = self.unique_name("behavior");
        self.entries.push(serde_json::json!({
            "name": name, "type": "Behavior", "args": {"on": "start", "do": []},
        }));
        self.mark_changed();
        self.behavior_index = self.behavior_entries().len().saturating_sub(1);
        self.open_behavior(world);
    }

    // Select a row. Selecting never edits: a row that offers options lights the
    // Pick button, and a row that takes typed text is ready to type into
    // straight away.
    pub(super) fn select_behavior_row(&mut self, i: usize, world: &mut World) {
        self.behavior_row = Some(i);
        self.behavior_picking = false;
        let Some(row) = self.behavior_rows().get(i).cloned() else {
            return;
        };
        self.seed_behavior_value(world);
        self.behavior_focus = edit::text_value(&self.behavior_args(), &row).is_some();
    }

    // A palette opens on the whole vocabulary: a query is about the pick being
    // made, so it never outlives it.
    fn clear_behavior_filter(&mut self, world: &mut World) {
        self.behavior_filter.clear();
        self.behavior_pick = 0;
        self.behavior_pick_scroll = 0;
        widget::seed_field(world, behavior_panel::FILTER_INPUT, "");
    }

    fn choose_behavior_pick(&mut self, i: usize, world: &mut World) {
        self.behavior_picking = false;
        self.clear_behavior_filter(world);
        let data = self.behavior_data();
        let (Some(row), Some(pick)) = (
            self.behavior_row.and_then(|r| data.rows.get(r)),
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
            .behavior_row
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
            self.behavior_row = None;
            self.behavior_focus = false;
        }
    }

    fn move_behavior_row(&mut self, delta: isize, world: &mut World) {
        let Some(row) = self
            .behavior_row
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
        self.behavior_row = self
            .behavior_rows()
            .iter()
            .position(|r| r.element.as_ref() == Some(&moved));
        self.ensure_behavior_visible();
    }

    pub(super) fn commit_behavior_value(&mut self, world: &mut World) {
        if !self.behavior_focus {
            return;
        }
        let Some(row) = self
            .behavior_row
            .and_then(|i| self.behavior_rows().get(i).cloned())
        else {
            return;
        };
        let text = widget::field_text(world, behavior_panel::VALUE_INPUT);
        let mut args = self.behavior_args();
        match edit::apply_text(&mut args, &row, &text) {
            Ok(()) => self.commit_behavior(args, world),
            Err(e) => self.behavior_status = Some(Status::message(e)),
        }
    }

    // Step to the next view. The selection survives, because the outline and the
    // chart are over the same rows: a card selected in the chart is the row the
    // outline opens on. The pan does not, because each chart is its own shape.
    fn toggle_behavior_view(&mut self) {
        self.behavior_mode = self.behavior_mode.other();
        self.behavior_picking = false;
        self.behavior_pan_drag = None;
        self.behavior_pan = [0.0, 0.0];
        if self.behavior_mode == ViewMode::Overview {
            // The map opens on the behavior that was showing, so it says where
            // the panel already is rather than starting from nothing.
            let data = self.behavior_data();
            self.behavior_overview_card = data
                .overview
                .cards
                .iter()
                .position(|c| c.behavior == Some(data.index));
        }
        self.ensure_behavior_visible();
    }

    // Open the behavior an overview card stands for, in the chart view, so
    // clicking through the map lands on the body it named.
    fn open_behavior_card(&mut self, i: usize, world: &mut World) {
        let Some(at) = self
            .behavior_data()
            .overview
            .cards
            .get(i)
            .and_then(|c| c.behavior)
        else {
            return;
        };
        self.behavior_index = at;
        self.behavior_mode = ViewMode::Chart;
        self.behavior_pan = [0.0, 0.0];
        self.open_behavior(world);
    }

    // Hold the selected member. Nothing is written, so this is not an edit and
    // takes no history snapshot.
    fn copy_behavior_row(&mut self) {
        let data = self.behavior_data();
        let Some(row) = self.behavior_row.and_then(|i| data.rows.get(i)) else {
            return;
        };
        if let Some(held) = clip::of(&self.behavior_args(), &data.rows, row) {
            self.behavior_clip = Some(held);
        }
    }

    fn paste_behavior_row(&mut self, world: &mut World) {
        let Some(held) = self.behavior_clip.clone() else {
            return;
        };
        self.place_behavior_clip(&held, world);
    }

    // A duplicate is a paste of the selection itself, so it leaves whatever is
    // held for a later paste alone.
    fn duplicate_behavior_row(&mut self, world: &mut World) {
        let data = self.behavior_data();
        let Some(row) = self.behavior_row.and_then(|i| data.rows.get(i)) else {
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
        let Some(row) = self.behavior_row.and_then(|i| data.rows.get(i)) else {
            return;
        };
        let mut args = self.behavior_args();
        let Some(landed) = clip::paste(&mut args, &data.rows, row, held) else {
            return;
        };
        self.commit_behavior(args, world);
        self.behavior_row = self
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
    fn select_behavior_fault(&mut self, world: &mut World) {
        let data = self.behavior_data();
        let Some(row) = self
            .behavior_status
            .as_ref()
            .and_then(|s| fault::row_of(&data.rows, s.at()))
        else {
            return;
        };
        if self.behavior_mode == ViewMode::Overview {
            self.behavior_mode = ViewMode::Chart;
            self.behavior_pan = [0.0, 0.0];
        }
        self.select_behavior_row(row, world);
        self.ensure_behavior_visible();
    }

    // Open the world's variable table on the variable an overview card stands
    // for. Behaviors reach each other through world state, so the map's variable
    // cards are the natural way into the table that declares it.
    fn open_overview_variable(&mut self, i: usize, world: &mut World) {
        let Some(name) = self
            .behavior_data()
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
        let data = self.behavior_data();
        let Some(card) = data.chart.cards.get(i) else {
            return;
        };
        if self.ctrl_held && card.kind == CardKind::Node {
            let path = card.path.clone();
            self.toggle_behavior_breakpoint(&path);
            return;
        }
        let path = card.path.clone();
        let Some(row) = data.rows.iter().position(|r| r.path == path) else {
            return;
        };
        self.select_behavior_row(row, world);
    }

    // The anchor is the pan plus the cursor, so `anchor - cursor` keeps the
    // point grabbed under the cursor for as long as the button is held.
    fn start_behavior_pan(&mut self, mouse: [f32; 2]) {
        self.behavior_focus = false;
        self.behavior_pan_drag = Some([
            self.behavior_pan[0] + mouse[0],
            self.behavior_pan[1] + mouse[1],
        ]);
    }

    // While a canvas pan is held the chart tracks the cursor; releasing the
    // button ends it.
    pub(super) fn drive_behavior_pan(&mut self, input: &FrameInput) {
        let Some(anchor) = self.behavior_pan_drag else {
            return;
        };
        if !input.left_button_down {
            self.behavior_pan_drag = None;
            return;
        }
        let want = [anchor[0] - input.mouse_x, anchor[1] - input.mouse_y];
        let chart = self.behavior_shown_chart();
        self.behavior_pan = behavior_chart::clamp_pan(want, &chart, self.behavior_canvas());
    }

    // The chart the panel is drawing: the open behavior's body, or the world's
    // behaviors mapped. Panning acts on whichever is on screen.
    fn behavior_shown_chart(&self) -> Chart {
        let data = self.behavior_data();
        match self.behavior_mode {
            ViewMode::Overview => data.overview,
            _ => data.chart,
        }
    }

    fn behavior_canvas(&self) -> [f32; 2] {
        behavior_panel::chart_canvas(self.effective_size(PanelKey::Behavior), self.behavior_mode)
    }

    fn behavior_rows_shown(&self) -> usize {
        behavior_panel::visible_rows(self.effective_size(PanelKey::Behavior)[1])
    }

    // Bring what is selected into view, whichever view is showing it: the
    // outline scrolls to its row, and either chart pans to its card.
    pub(super) fn ensure_behavior_visible(&mut self) {
        match self.behavior_mode {
            ViewMode::Overview => self.pan_to_overview_card(),
            ViewMode::Chart => {
                if let Some(row) = self.behavior_row {
                    self.pan_to_behavior_row(row);
                }
            }
            ViewMode::Outline => {
                let Some(row) = self.behavior_row else {
                    return;
                };
                let shown = self.behavior_rows_shown();
                self.behavior_scroll = navigate::scroll_to(row, self.behavior_scroll, shown);
            }
        }
    }

    fn pan_to_overview_card(&mut self) {
        let data = self.behavior_data();
        let Some(card) = self
            .behavior_overview_card
            .and_then(|i| data.overview.cards.get(i))
        else {
            return;
        };
        self.behavior_pan = behavior_chart::pan_to(
            card,
            self.behavior_canvas(),
            self.behavior_pan,
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
        self.behavior_pan =
            behavior_chart::pan_to(card, self.behavior_canvas(), self.behavior_pan, &data.chart);
    }

    // The wheel scrolls the open palette while it is up, pans the chart in chart
    // view, and scrolls the outline otherwise.
    pub(super) fn scroll_behavior(&mut self, delta: f32) {
        if self.behavior_picking {
            let total = self.behavior_data().matches.len();
            let max = total.saturating_sub(behavior_panel::PICK_POOL);
            self.behavior_pick_scroll = scroll_step(self.behavior_pick_scroll, delta, max);
            return;
        }
        if self.behavior_mode.drawn_as_chart() {
            let chart = self.behavior_shown_chart();
            let canvas = self.behavior_canvas();
            let step = if delta > 0.0 { WHEEL_PAN } else { -WHEEL_PAN };
            // The wheel moves along whichever axis has anywhere to go, so a
            // chart that is wide and one row tall scrolls sideways rather than
            // not at all.
            let pan = self.behavior_pan;
            let want = if behavior_chart::max_pan(&chart, canvas)[1] > 0.0 {
                [pan[0], pan[1] + step]
            } else {
                [pan[0] + step, pan[1]]
            };
            self.behavior_pan = behavior_chart::clamp_pan(want, &chart, canvas);
            return;
        }
        let max = self
            .behavior_rows()
            .len()
            .saturating_sub(self.behavior_rows_shown());
        self.behavior_scroll = scroll_step(self.behavior_scroll, delta, max);
    }
}
