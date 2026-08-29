// src/editor/hook/behavior_keys.rs
//
// EditorHook: the Behavior panel's keyboard. `frame_keys` reaches the frontmost
// open panel only, so these bindings never compete with another panel's, and a
// key is one press rather than a held state -- one press is one step.
//
// The palette, while it is open, is modal over everything below it:
//   typing       narrows it to what the query answers, best match first
//   Up / Down    move the highlighted option, bringing it into the window
//   Enter        insert the highlighted option
//   Escape       close the palette without picking
// Characters reach the palette's own field, so the keys below act on the options
// the query keeps rather than on the whole vocabulary. Narrowing puts the
// highlight back at the top, which is what makes typing and then committing land
// on the best answer. Escape closes outright rather than first clearing the
// query: one press, one meaning, and Backspace already clears.
//
// Escape (its own `FrameInput` pulse rather than a `InputKey`) otherwise answers
// whichever state is waiting on a press, most consequential first: an armed
// removal is cancelled, and failing that the field holding the keyboard gives
// it up, the name field reverting to what the world holds.
//
// Enter commits the field holding the keyboard -- the name onto the open
// behavior, or the value into the selected row. With no field focused it acts on
// the selection instead: opening its palette in the outline and the chart, and
// in the overview opening the behavior a card stands for (a trigger, variable,
// or asset card has none).
//
// Tab steps to the next view, the same cycle the header's own button walks.
//
// Ctrl carries a list member between places, whole subtree and all:
//   Ctrl+C       hold the selected member
//   Ctrl+V       put it into the list the selection addresses, after the
//                selected member or at the end of the list itself
//   Ctrl+D       duplicate the selected member in place, leaving what is held
// What is held outlives the open behavior, so a node can be carried to another
// one. These act only while no field holds the keyboard, since a copy is about
// the selected node rather than the text being typed.
//
// The arrows move the selection through whichever view is showing:
//   Outline      Up / Down step one row
//   Chart        Left / Right follow the chain, Up / Down cross between a
//                branching node's stacked branches
//   Overview     the same, over the map's cards
// The name field is the asset's rather than the selection's, so it holds the
// keyboard -- arrows and Tab alike -- until Enter or Escape gives it up. The
// value field is the selection's and follows it from view to view, so it holds
// only Left and Right, which are the caret's (`text_input_system`).

use super::*;
use crate::components::InputKey;
use crate::editor::behavior::graph::Card;
use crate::editor::behavior::navigate::{self, Dir};
use crate::editor::behavior_chart::CARD_POOL;

fn direction(key: InputKey) -> Option<Dir> {
    Some(match key {
        InputKey::Up => Dir::Up,
        InputKey::Down => Dir::Down,
        InputKey::Left => Dir::Left,
        InputKey::Right => Dir::Right,
        _ => return None,
    })
}

// The cards a step can reach: the ones the chart draws, so the keyboard lands
// only where a click could (`behavior_chart::hit_card`).
fn reachable(cards: &[Card]) -> &[Card] {
    &cards[..cards.len().min(CARD_POOL)]
}

impl EditorHook {
    pub(super) fn behavior_keys(&mut self, world: &mut World, input: &FrameInput) {
        if input.escape {
            self.cancel_behavior_key(world);
            return;
        }
        let Some(key) = input.captured_key else {
            return;
        };
        if self.behavior_palette_open() {
            self.behavior_palette_key(key, world);
            return;
        }
        // Ctrl carries the clipboard, and only while no field holds the keyboard:
        // a copy is about the selected node, not about the text being typed.
        if input.ctrl {
            if !self.behavior_focus && !self.behavior_name_focus {
                self.clipboard_behavior_key(key, world);
            }
            return;
        }
        match key {
            InputKey::Enter => self.commit_behavior_key(world),
            _ if self.behavior_name_focus => {}
            InputKey::Tab => {
                self.apply_behavior_action(BehaviorAction::ToggleView, world, [0.0, 0.0])
            }
            // Left and Right belong to the caret while the value field holds
            // the keyboard.
            _ => {
                let step = direction(key).filter(|d| !(self.behavior_focus && d.horizontal()));
                if let Some(dir) = step {
                    self.step_behavior_selection(dir, world);
                }
            }
        }
    }

    fn clipboard_behavior_key(&mut self, key: InputKey, world: &mut World) {
        let action = match key {
            InputKey::C => BehaviorAction::Copy,
            InputKey::V => BehaviorAction::Paste,
            InputKey::D => BehaviorAction::Duplicate,
            _ => return,
        };
        self.apply_behavior_action(action, world, [0.0, 0.0]);
    }

    // Whether the palette is not just up but showing something to pick, which is
    // the same test the panel draws and hit-tests by.
    fn behavior_palette_open(&self) -> bool {
        self.behavior_picking && !self.behavior_data().picks.is_empty()
    }

    fn behavior_palette_key(&mut self, key: InputKey, world: &mut World) {
        match key {
            InputKey::Enter => {
                // The highlight counts kept options, so it resolves through the
                // filter before it can name one.
                let Some(&at) = self.behavior_data().matches.get(self.behavior_pick) else {
                    return;
                };
                self.apply_behavior_action(BehaviorAction::Choose(at), world, [0.0, 0.0]);
            }
            InputKey::Up | InputKey::Down => {
                let total = self.behavior_data().matches.len();
                let delta = if key == InputKey::Up { -1 } else { 1 };
                let Some(at) = navigate::step(Some(self.behavior_pick), delta, total) else {
                    return;
                };
                self.behavior_pick = at;
                self.behavior_pick_scroll =
                    navigate::scroll_to(at, self.behavior_pick_scroll, behavior_panel::PICK_POOL);
            }
            _ => {}
        }
    }

    fn cancel_behavior_key(&mut self, world: &mut World) {
        if self.behavior_palette_open() {
            self.apply_behavior_action(BehaviorAction::Dismiss, world, [0.0, 0.0]);
            return;
        }
        if std::mem::take(&mut self.behavior_remove_armed) {
            return;
        }
        if self.behavior_name_focus {
            self.blur_behavior_name(world);
            return;
        }
        self.behavior_focus = false;
    }

    fn commit_behavior_key(&mut self, world: &mut World) {
        if self.behavior_name_focus {
            self.rename_behavior(world);
            return;
        }
        if self.behavior_focus {
            self.commit_behavior_value(world);
            return;
        }
        let action = match self.behavior_mode {
            ViewMode::Overview => match self.behavior_overview_card {
                Some(card) => BehaviorAction::OpenCard(card),
                None => return,
            },
            // A row offering nothing has no palette to open, so the press is
            // left alone rather than toggling one that never shows.
            _ if self.behavior_row.is_some() && !self.behavior_data().picks.is_empty() => {
                BehaviorAction::Palette
            }
            _ => return,
        };
        self.apply_behavior_action(action, world, [0.0, 0.0]);
    }

    fn step_behavior_selection(&mut self, dir: Dir, world: &mut World) {
        match self.behavior_mode {
            ViewMode::Outline => {
                let Some(delta) = dir.list_step() else {
                    return;
                };
                let total = self.behavior_rows().len();
                let Some(row) = navigate::step(self.behavior_row, delta, total) else {
                    return;
                };
                self.apply_behavior_action(BehaviorAction::Select(row), world, [0.0, 0.0]);
            }
            ViewMode::Chart => {
                let data = self.behavior_data();
                let Some(card) = navigate::nearest(reachable(&data.chart.cards), data.card, dir)
                else {
                    return;
                };
                self.apply_behavior_action(BehaviorAction::SelectCard(card), world, [0.0, 0.0]);
            }
            ViewMode::Overview => {
                let data = self.behavior_data();
                let cards = reachable(&data.overview.cards);
                let from = self.behavior_overview_card.filter(|&i| i < cards.len());
                let Some(card) = navigate::nearest(cards, from, dir) else {
                    return;
                };
                self.behavior_overview_card = Some(card);
            }
        }
        self.ensure_behavior_visible();
    }
}
