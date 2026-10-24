//! `EditorHook`: the Planet panel's actions. The panel edits the world's first
//! `Planet` entry (`planet.rs` lays out its rows); every commit is validated
//! through `form::validate` first, so nothing invalid reaches the entries. A
//! reference picker, a button or an add row commits on the click; typed boxes
//! commit together on Apply.

use concinnity_cook::authoring::world::replace_args;
use concinnity_engine::ecs::World;
use serde_json::{Map, Value, json};

use crate::editor::hook::{EditorHook, declared_id, names_of_types, short_status};
use crate::editor::panels::form::FieldKind;
use crate::editor::panels::planet::{self, Button, List, Row, RowKind};
use crate::editor::panels::planet_panel::{self, PlanetAction, PlanetView};
use crate::editor::panels::registry::PanelKey;
use crate::editor::widget;

// The open water a new sea's wind blows across, in meters.
const SEA_FETCH_METERS: f64 = 50_000.0;

impl EditorHook {
    // The planet entry's args with every default filled in.
    fn planet_args(&self) -> Option<Map<String, Value>> {
        let idx = self.entry_index_of(planet::TY)?;
        Some(planet::working(&self.entry_args(idx)))
    }

    // The panel's rows, each reference picker offering the world's assets of
    // its target types.
    pub(in crate::editor::hook) fn planet_rows(&self) -> Vec<Row> {
        let mut rows = planet::rows(self.planet_args().as_ref());
        for row in &mut rows {
            if let RowKind::Field { field, nullable } = &mut row.kind
                && let FieldKind::Ref { targets } = field.kind
            {
                let names = names_of_types(&self.entries, targets);
                planet::set_ref_options(field, *nullable, &names);
            }
        }
        rows
    }

    pub(in crate::editor::hook) fn make_planet_view<'a>(
        &'a self,
        rows: &'a [Row],
        mouse: [f32; 2],
    ) -> PlanetView<'a> {
        PlanetView {
            rows,
            scroll: self.planet.scroll.min(planet_panel::max_scroll(rows.len())),
            // Focus only while frontmost, so the inputs never fight another
            // panel's focused field for typed keys.
            focus: self
                .planet
                .focus
                .filter(|_| self.panel_order.last() == Some(&PanelKey::Planet)),
            status: self.planet.status.as_deref(),
            dirty: self.planet.touched,
            mouse,
        }
    }

    // Seed every number box from the planet as committed: on open, after a
    // structural edit and after Apply. Not on every entry change, so a
    // half-typed edit is never clobbered.
    pub(in crate::editor::hook) fn seed_planet(&mut self, world: &mut World) {
        self.planet.touched = false;
        for (r, row) in self.planet_rows().iter().enumerate() {
            if let Some(f) = row.field()
                && f.kind.has_text_input()
            {
                widget::seed_field(world, planet_panel::input(r), &f.initial);
            }
        }
    }

    pub(in crate::editor::hook) fn scroll_planet(&mut self, delta: f32) {
        let max = planet_panel::max_scroll(self.planet_rows().len());
        let cur = self.planet.scroll.min(max);
        self.planet.scroll = super::super::scroll_step(cur, delta, max);
    }

    pub(in crate::editor::hook) fn apply_planet_action(
        &mut self,
        action: PlanetAction,
        world: &mut World,
    ) {
        match action {
            PlanetAction::Focus(r) => self.planet.focus = Some(r),
            PlanetAction::CycleRef(r) => self.cycle_planet_ref(r),
            PlanetAction::Button(b) => self.press_planet_button(b, world),
            PlanetAction::AddPlanet => self.add_planet(world),
            PlanetAction::AddSea => self.add_planet_sea(world),
            PlanetAction::Apply => self.apply_planet(world),
            PlanetAction::Consume => self.planet.focus = None,
        }
    }

    // Validate the edit that took the working args to `edited` and write it
    // over the planet entry; whether it took.
    fn commit_planet(&mut self, edited: &Map<String, Value>) -> bool {
        self.planet.status = None;
        let Some(idx) = self.entry_index_of(planet::TY) else {
            return false;
        };
        let authored = self.entry_args(idx);
        let args = planet::authored_edit(&authored, &planet::working(&authored), edited);
        let name = self
            .entries
            .get(idx)
            .and_then(declared_id)
            .unwrap_or(planet::TY);
        if let Err(e) = crate::editor::panels::form::validate(planet::TY, name, &args) {
            self.planet.status = Some(short_status(&e));
            return false;
        }
        if let Some(entry) = self.entries.get_mut(idx) {
            replace_args(entry, Value::Object(args));
        }
        self.mark_changed();
        true
    }

    // Write every number box into the planet together.
    pub(in crate::editor::hook) fn apply_planet(&mut self, world: &mut World) {
        let Some(args) = self.planet_args() else {
            return;
        };
        let rows = planet::rows(Some(&args));
        let texts: Vec<String> = (0..rows.len())
            .map(|r| widget::field_text(world, planet_panel::input(r)))
            .collect();
        if self.commit_planet(&planet::assemble(&args, &rows, &texts)) {
            self.seed_planet(world);
        }
    }

    // Move row `r`'s picker to its next option and commit it, leaving any
    // typed box pending.
    fn cycle_planet_ref(&mut self, r: usize) {
        let rows = self.planet_rows();
        let Some(field) = rows.get(r).and_then(Row::field) else {
            return;
        };
        let (Some(next), Some(mut args)) = (planet::next_ref(field), self.planet_args()) else {
            return;
        };
        if planet::set_value(&mut args, &field.key, next) {
            self.commit_planet(&args);
        }
    }

    fn press_planet_button(&mut self, button: Button, world: &mut World) {
        let grass = match button {
            Button::Add(List::Layers) => Some(self.first_or_new("Grass", &json!({}))),
            _ => None,
        };
        let Some(args) = self.planet_args() else {
            return;
        };
        if let Some(next) = planet::press(&args, button, grass.as_deref())
            && self.commit_planet(&next)
        {
            self.seed_planet(world);
        }
    }

    // The name of the world's first asset of type `ty`, or of a new one with
    // `args` added for the edit to reference.
    fn first_or_new(&mut self, ty: &str, args: &Value) -> String {
        if let Some(name) = names_of_types(&self.entries, &[ty]).into_iter().next() {
            return name;
        }
        self.add_named(ty, args)
    }

    // Append an asset of type `ty` with `args` under a fresh name.
    fn add_named(&mut self, ty: &str, args: &Value) -> String {
        let name = self.unique_name(ty);
        let mut args = args.as_object().cloned().unwrap_or_default();
        args.insert(
            concinnity_cook::authoring::world::ID_KEY.to_string(),
            json!(name),
        );
        self.entries.push(json!({"type": ty, "args": args}));
        name
    }

    fn add_planet(&mut self, world: &mut World) {
        if self.entry_index_of(planet::TY).is_some() {
            return;
        }
        self.entries.push(json!({"type": planet::TY, "args": {}}));
        self.mark_changed();
        self.seed_planet(world);
    }

    // Give the planet a sea of its own new water.
    fn add_planet_sea(&mut self, world: &mut World) {
        let Some(args) = self.planet_args() else {
            return;
        };
        let water = self.add_named("WaterSurface", &json!({"fetch_meters": SEA_FETCH_METERS}));
        if self.commit_planet(&planet::with_sea(&args, &water)) {
            self.seed_planet(world);
        } else if let Some(i) = self
            .entries
            .iter()
            .rposition(|e| declared_id(e) == Some(&water))
        {
            self.entries.remove(i);
        }
    }
}
