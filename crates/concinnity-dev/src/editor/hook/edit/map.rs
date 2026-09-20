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
            pan: self.map.pan,
            mouse,
        }
    }

    // The anchor is the pan plus the cursor, so `anchor - cursor` keeps the
    // place grabbed under the cursor for as long as the button is held.
    pub(in crate::editor::hook) fn apply_map_action(&mut self, action: MapAction, mouse: [f32; 2]) {
        match action {
            MapAction::PanStart => {
                self.map.pan_drag = Some([self.map.pan[0] + mouse[0], self.map.pan[1] + mouse[1]]);
            }
            MapAction::Consume => {}
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
