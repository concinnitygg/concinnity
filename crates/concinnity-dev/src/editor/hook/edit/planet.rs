//! `EditorHook`: the Planet panel's actions. The panel edits the world's first
//! `Planet` entry (`panels/planet/` lays out its rows); every commit is validated
//! through `form::validate` first, so nothing invalid reaches the entries. A
//! reference picker, a button or an add row commits on the click; typed boxes
//! commit together on Apply.

use concinnity_cook::authoring::world::replace_args;
use concinnity_engine::ecs::World;
use serde_json::{Map, Value, json};

use crate::editor::hook::{EditorHook, declared_id, names_of_types, short_status};
use crate::editor::panels::form::FieldKind;
use crate::editor::panels::planet::panel::{PlanetAction, PlanetView};
use crate::editor::panels::planet::tool_strip::StripView;
use crate::editor::panels::planet::{self, Button, List, Row, RowKind};
use crate::editor::panels::registry::PanelKey;
use crate::editor::planet_tools::Tool;
use crate::editor::ui::BuiltPanel;

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

    // Bring what the panel shows up to date with the entries, and build its
    // rows again when what they lay out changed.
    pub(in crate::editor::hook) fn prepare_planet(
        &mut self,
        world: &mut World,
        built: &mut BuiltPanel,
    ) {
        let revision = self.entries.revision();
        if self
            .planet
            .view
            .as_ref()
            .is_none_or(|(r, _)| *r != revision)
        {
            self.planet.view = Some((revision, self.planet_rows()));
        }
        // Texts that no longer match the rows, such as none at all in a fresh
        // world, are seeded from the planet.
        if self
            .planet
            .view
            .as_ref()
            .is_some_and(|(_, rows)| rows.len() != self.planet.texts.len())
        {
            self.seed_planet();
        }
        let tool = self.planet_tool.tools.active;
        let pick = tool
            .and_then(Tool::list)
            .map_or(0, |l| self.planet_tool.tools.pick(l));
        if self
            .planet
            .pick
            .as_ref()
            .is_none_or(|(r, t, p, _)| (*r, *t, *p) != (revision, tool, pick))
        {
            self.planet.pick = Some((revision, tool, pick, self.planet_pick_label()));
        }
        let Some((_, rows)) = &self.planet.view else {
            return;
        };
        if self
            .planet
            .shown
            .as_ref()
            .is_none_or(|ids| !ids.lays_out(rows))
        {
            self.planet.shown = Some(planet::panel::build_rows(world, built, rows));
        }
    }

    pub(in crate::editor::hook) fn make_planet_view(&self) -> Option<PlanetView<'_>> {
        let (_, rows) = self.planet.view.as_ref()?;
        Some(PlanetView {
            rows,
            texts: &self.planet.texts,
            // Focus only while frontmost, so the inputs never fight another
            // panel's focused field for typed keys.
            focus: self
                .planet
                .focus
                .filter(|_| self.panel_order.last() == Some(&PanelKey::Planet)),
            status: self.planet.status.as_deref(),
            dirty: self.planet.touched,
            strip: StripView {
                tools: &self.planet_tool.tools,
                pick: self.planet.pick.as_ref().map_or("", |p| p.3.as_str()),
            },
        })
    }

    // Take in what was typed into the panel and the box holding the keyboard.
    pub(in crate::editor::hook) fn observe_planet(&mut self, world: &World) {
        let Some(ids) = &self.planet.shown else {
            return;
        };
        let focus = planet::panel::observe(world, ids, &mut self.planet.texts);
        let shown = self.make_planet_view().and_then(|v| v.focus);
        if focus != shown {
            self.planet.focus = focus;
        }
    }

    // Seed every number box from the planet as committed: on open, after a
    // structural edit and after Apply. Not on every entry change, so a
    // half-typed edit is never clobbered.
    pub(in crate::editor::hook) fn seed_planet(&mut self) {
        self.planet.touched = false;
        self.planet.texts = self
            .planet_rows()
            .iter()
            .map(|row| {
                row.field()
                    .filter(|f| f.kind.has_text_field())
                    .map(|f| f.initial.clone())
                    .unwrap_or_default()
            })
            .collect();
    }

    pub(in crate::editor::hook) fn apply_planet_action(&mut self, action: PlanetAction) {
        match action {
            PlanetAction::CycleRef(r) => self.cycle_planet_ref(r),
            PlanetAction::RowButton { row, slot } => {
                let button = self
                    .planet
                    .view
                    .as_ref()
                    .and_then(|(_, rows)| rows.get(row))
                    .and_then(|r| r.buttons.get(slot).copied().flatten());
                if let Some(b) = button {
                    self.press_planet_button(b);
                }
            }
            PlanetAction::AddPlanet => self.add_planet(),
            PlanetAction::AddSea => self.add_planet_sea(),
            PlanetAction::Apply => self.apply_planet(),
            PlanetAction::Strip(a) => self.apply_strip_action(a),
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
    pub(in crate::editor::hook) fn apply_planet(&mut self) {
        let Some(args) = self.planet_args() else {
            return;
        };
        let rows = planet::rows(Some(&args));
        if self.commit_planet(&planet::assemble(&args, &rows, &self.planet.texts)) {
            self.seed_planet();
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

    fn press_planet_button(&mut self, button: Button) {
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
            self.seed_planet();
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

    fn add_planet(&mut self) {
        if self.entry_index_of(planet::TY).is_some() {
            return;
        }
        self.entries.push(json!({"type": planet::TY, "args": {}}));
        self.mark_changed();
        self.seed_planet();
    }

    // Give the planet a sea of its own new water.
    fn add_planet_sea(&mut self) {
        let Some(args) = self.planet_args() else {
            return;
        };
        let water = self.add_named("WaterSurface", &json!({"fetch_meters": SEA_FETCH_METERS}));
        if self.commit_planet(&planet::with_sea(&args, &water)) {
            self.seed_planet();
        } else if let Some(i) = self
            .entries
            .iter()
            .rposition(|e| declared_id(e) == Some(&water))
        {
            self.entries.remove(i);
        }
    }
}
