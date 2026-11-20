//! `EditorHook`: the Map panel's actions. The map shows where a world can be and
//! how it moves between those places, and changes nothing, so the hook's job is
//! turning the working entry list into the canvas's shapes, carrying the
//! canvas's pan, and reaching what a place card stands for.
//!
//! The entry shape is the hook's to map, as it is for the behavior overview:
//! `editor/map/` only ever sees typed assets and the keys they are addressed
//! by. An entry the registry cannot type is left out, because a build would not
//! place one either.

use concinnity_cook::authoring::world::{WorldJsonlAsset, args_without_id, entry_handles};
use concinnity_engine::components::{UiEvent, UiValue};
use concinnity_engine::ecs::World;
use serde_json::Value;

use crate::editor::asset_handle::AssetHandle;
use crate::editor::behavior::chart::{self, ChartMarks};
use crate::editor::behavior::graph::Chart;
use crate::editor::hook::{EditorHook, entry_type};
use crate::editor::map::{self, Entries};
use crate::editor::panels::registry::PanelKey;
use crate::editor::ui::BuiltPanel;
use crate::editor::ui::shapes::Shapes;
use crate::editor::ui::state;

impl EditorHook {
    fn map_entries(&self) -> Entries {
        let handles = entry_handles(&self.entries);
        Entries::new(self.entries.iter().enumerate().filter_map(|(i, entry)| {
            Some((
                self.entries.key_at(i)?,
                WorldJsonlAsset {
                    id: handles.get(i)?.clone()?,
                    asset_type: entry_type(entry)?,
                    args: args_without_id(entry.get("args").cloned().unwrap_or(Value::Null)),
                },
            ))
        }))
    }

    pub(in crate::editor::hook) fn map_chart(&self) -> Chart {
        map::map(&self.map_entries())
    }

    // The card standing for what is selected, wherever the selection was made.
    pub(in crate::editor::hook) fn map_selected(&self, chart: &Chart) -> Option<usize> {
        self.selection
            .active()
            .and_then(|handle| map::card_of(chart, handle))
    }

    // Take in where the user panned the canvas to and the card under the
    // cursor.
    pub(in crate::editor::hook) fn observe_map(&mut self, world: &World, built: &BuiltPanel) {
        if let Some((pan, hovered)) = map::panel::observed(world, built) {
            self.map.pan = pan;
            self.map.hovered = hovered;
        }
    }

    // Bring the canvas onto what it should show and write the chart as its
    // shapes.
    pub(in crate::editor::hook) fn prepare_map(&mut self, world: &mut World) {
        // The chart is the entries' alone, so it is rebuilt only when they
        // change.
        let revision = self.entries.revision();
        let chart = match self.map.chart.take() {
            Some((seen, chart)) if seen == revision => chart,
            _ => self.map_chart(),
        };
        self.drive_map(&chart);
        let marks = ChartMarks {
            chart: &chart,
            selected: self.map_selected(&chart),
            faulted: None,
            hovered: self.map.hovered,
            // A card stands for a place rather than for a node a session can
            // stop in, so the Behavior panel's live marks have nothing to point
            // at here.
            pulses: &[],
            breakpoints: &[],
        };
        let taken = &mut self.map.taken;
        state::with_state(world, |s| {
            s.edit_shapes(map::panel::SHAPES, |list| {
                let mut shapes = Shapes::new(list);
                chart::push(&mut shapes, &marks, taken);
                shapes.finish();
            });
        });
        self.map.chart = Some((revision, chart));
    }

    // A press on the card at the index the event carries.
    pub(in crate::editor::hook) fn on_map_event(
        &mut self,
        world: &mut World,
        event: &UiEvent,
    ) -> bool {
        if event.name != map::panel::CARD_EVENT {
            return false;
        }
        if let Some(UiValue::Int(card)) = event.value
            && let Ok(card) = usize::try_from(card)
        {
            self.select_map_card(card, world);
        }
        true
    }

    // Select a card like its asset elsewhere: plain replaces and opens, shift
    // toggles, and the tree reveals the row. Ctrl asks what sends the world
    // there.
    pub(in crate::editor::hook) fn select_map_card(&mut self, card: usize, world: &mut World) {
        let chart = self.map_chart();
        let Some(handle) = chart.cards.get(card).and_then(|c| c.handle.clone()) else {
            return;
        };
        if self.ctrl_held {
            self.select_movers_into(&handle, world);
        } else {
            self.select_handle(&handle);
        }
        // The card that was acted on is on the canvas already, so the follow
        // below has nothing left to bring into view.
        self.map.shown = self.selection.active().cloned();
    }

    // Select the behaviors that move the world to a place, opening the first.
    fn select_movers_into(&mut self, handle: &AssetHandle, world: &mut World) {
        let Some(place) = self.handle_name(handle) else {
            return;
        };
        let movers = map::behaviors_into(&self.map_entries(), &place);
        let Some(first) = movers.first().cloned() else {
            self.notifier
                .info(&format!("no behavior sends the world to {place}"));
            return;
        };
        let handles = movers.iter().map(|name| self.handle_for(name)).collect();
        self.selection.set(handles);
        self.follow_active();
        self.open_behavior_named(&first, world);
    }

    // Keep the canvas on what it should be showing, once a frame: where the
    // world starts when it has not been rooted in this one, then whatever is
    // selected.
    pub(in crate::editor::hook) fn drive_map(&mut self, chart: &Chart) {
        if !self.open[PanelKey::Map] {
            return;
        }
        let canvas = self.map_canvas();
        if !self.map.rooted {
            self.map.rooted = true;
            self.map.pan = map::panel::root_pan(chart, canvas);
        }
        let active = self.selection.active().cloned();
        if active == self.map.shown {
            return;
        }
        self.map.shown.clone_from(&active);
        // A selection standing for a place is brought into view, moving no
        // further than it must; one naming no place leaves the canvas alone.
        if let Some(card) = active
            .and_then(|handle| map::card_of(chart, &handle))
            .and_then(|i| chart.cards.get(i))
        {
            self.map.pan = chart::pan_to(card, canvas, self.map.pan, chart);
        }
    }

    pub(in crate::editor::hook) fn map_canvas(&self) -> [f32; 2] {
        map::panel::canvas(self.effective_size(PanelKey::Map))
    }
}
