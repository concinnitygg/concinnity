//! EditorHook: the Map panel's actions. The map shows where a world can be and
//! how it moves between those places, and changes nothing, so the whole of the
//! hook's job is turning the working entry list into what the map reads and
//! carrying the canvas's pan.
//!
//! The entry shape is the hook's to map, as it is for the behavior overview:
//! `editor/map/` only ever sees typed assets and the keys they are addressed
//! by. An entry the registry cannot type is left out, because a build would not
//! place one either.

use concinnity_cook::authoring::registry::RegisteredType;
use concinnity_cook::authoring::world::{WorldJsonlAsset, args_without_id, entry_handles};
use concinnity_core::components::FrameInput;
use concinnity_core::ecs::World;
use serde_json::Value;

use crate::editor::behavior::chart;
use crate::editor::behavior::graph::Chart;
use crate::editor::hook::{EditorHook, entry_type};
use crate::editor::map::panel::{MapAction, MapView};
use crate::editor::map::{self, Entries};
use crate::editor::panels::registry::PanelKey;

// How far one wheel notch pans the canvas.
const WHEEL_PAN: f32 = 40.0;

impl EditorHook {
    fn map_entries(&self) -> Entries {
        let handles = entry_handles(&self.entries);
        Entries::new(self.entries.iter().enumerate().filter_map(|(i, entry)| {
            Some((
                self.entries.key_at(i)?,
                WorldJsonlAsset {
                    id: handles.get(i)?.clone()?,
                    asset_type: RegisteredType::parse(entry_type(entry)?)?,
                    args: args_without_id(entry.get("args").cloned().unwrap_or(Value::Null)),
                },
            ))
        }))
    }

    pub(in crate::editor::hook) fn map_chart(&self) -> Chart {
        map::map(&self.map_entries())
    }

    pub(in crate::editor::hook) fn make_map_view<'a>(
        &self,
        chart: &'a Chart,
        mouse: [f32; 2],
    ) -> MapView<'a> {
        MapView {
            chart,
            selected: self
                .selection
                .active()
                .and_then(|handle| map::card_of(chart, handle)),
            pan: self.map.pan,
            mouse,
        }
    }

    // The anchor is the pan plus the cursor, so `anchor - cursor` keeps the
    // place grabbed under the cursor for as long as the button is held.
    pub(in crate::editor::hook) fn apply_map_action(
        &mut self,
        action: MapAction,
        mouse: [f32; 2],
        world: &mut World,
    ) {
        match action {
            MapAction::Select(card) => self.select_map_card(card, world),
            MapAction::PanStart => {
                self.map.pan_drag = Some([self.map.pan[0] + mouse[0], self.map.pan[1] + mouse[1]]);
            }
            MapAction::Consume => {}
        }
    }

    // A card is selected the way its asset is selected anywhere else: plain
    // replaces the selection and opens the asset's editing surface, shift
    // toggles membership, and the Assets tree reveals the row either way. A
    // place the build generates has no authored line, so it opens seeded from
    // what the expansion produced, exactly as its own row does.
    fn select_map_card(&mut self, card: usize, world: &mut World) {
        let chart = self.map_chart();
        let Some(name) = chart
            .cards
            .get(card)
            .and_then(|c| c.handle.as_ref())
            .and_then(|handle| self.handle_name(handle))
        else {
            return;
        };
        if self.shift_held {
            if self.toggle_named(&name) {
                self.open_asset_form(&name, world);
            } else {
                self.follow_active(world);
            }
        } else {
            self.select_named(&name);
            self.open_asset_form(&name, world);
        }
        self.reveal_in_tree(&name, world);
        self.pick_last = None;
        // The card that was clicked is on the canvas already, so the follow
        // below has nothing left to bring into view.
        self.map.shown = self.selection.active().cloned();
    }

    // Keep the canvas on what it should be showing, once a frame: where the
    // world starts when it has not been rooted in this one, then whatever is
    // selected. An in-flight pan owns the canvas until the button comes up.
    pub(in crate::editor::hook) fn drive_map(&mut self) {
        if !self.map.open || self.map.pan_drag.is_some() {
            return;
        }
        let chart = self.map_chart();
        let canvas = self.map_canvas();
        if !self.map.rooted {
            self.map.rooted = true;
            self.map.pan = map::panel::root_pan(&chart, canvas);
        }
        // An edit can shrink the map under the canvas, which would otherwise
        // strand it past the last place.
        self.map.pan = chart::clamp_pan(self.map.pan, &chart, canvas);
        let active = self.selection.active().cloned();
        if active == self.map.shown {
            return;
        }
        self.map.shown = active.clone();
        // A selection standing for a place is brought into view, moving no
        // further than it must; one naming no place leaves the canvas alone.
        if let Some(card) = active
            .and_then(|handle| map::card_of(&chart, &handle))
            .and_then(|i| chart.cards.get(i))
        {
            self.map.pan = chart::pan_to(card, canvas, self.map.pan, &chart);
        }
    }

    // While a pan is held the map tracks the cursor; releasing the button ends
    // it.
    pub(in crate::editor::hook) fn drive_map_pan(&mut self, input: &FrameInput) {
        let Some(anchor) = self.map.pan_drag else {
            return;
        };
        if !input.left_button_down {
            self.map.pan_drag = None;
            return;
        }
        let want = [anchor[0] - input.mouse_x, anchor[1] - input.mouse_y];
        self.map.pan = chart::clamp_pan(want, &self.map_chart(), self.map_canvas());
    }

    // The wheel pans along whichever axis has anywhere to go, so a map that is
    // wide and one row tall scrolls sideways rather than not at all.
    pub(in crate::editor::hook) fn scroll_map(&mut self, delta: f32) {
        let chart = self.map_chart();
        let canvas = self.map_canvas();
        let step = if delta > 0.0 { WHEEL_PAN } else { -WHEEL_PAN };
        let pan = self.map.pan;
        let want = match chart::max_pan(&chart, canvas)[1] > 0.0 {
            true => [pan[0], pan[1] + step],
            false => [pan[0] + step, pan[1]],
        };
        self.map.pan = chart::clamp_pan(want, &chart, canvas);
    }

    fn map_canvas(&self) -> [f32; 2] {
        map::panel::canvas(self.effective_size(PanelKey::Map))
    }
}
