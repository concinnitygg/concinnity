//! `EditorHook`: the Planet panel's tools at work. While a tool is picked and
//! the panel is shown, the ground under the cursor is found by a ray test
//! against the planet's height function, the brush ring is drawn on it, and a
//! press over the viewport goes to the tool instead of selection and the
//! gizmo: its drag lays a stroke of points along the ground for the tool.

use concinnity_engine::components::FrameInput;
use concinnity_engine::ecs::World;
use concinnity_engine::gfx::lines::Line;
use concinnity_engine::planet::{PlanetFrame, geo_direction};
use serde_json::Value;

use super::planet_tool_state::Stroke;
use crate::editor::hook::EditorHook;
use crate::editor::hook::pick::camera_ray;
use crate::editor::hud;
use crate::editor::panels::planet::List;
use crate::editor::panels::planet_panel;
use crate::editor::panels::planet_tool_strip::{self, StripAction};
use crate::editor::panels::registry::{self, PanelKey};
use crate::editor::planet_tools::ring::{self, RING_POINTS};
use crate::editor::planet_tools::stroke::{SPACING, Spacer};
use crate::editor::planet_tools::{self, Tool};
use crate::editor::widget::{self, point_in};

const RING_COLOR: [f32; 4] = [0.95, 0.72, 0.22, 1.0];
const INNER_COLOR: [f32; 4] = [0.95, 0.72, 0.22, 0.45];
const RING_WIDTH_PX: f32 = 2.0;
const INNER_WIDTH_PX: f32 = 1.5;
// How far the ring floats over the ground, as a share of the distance to it:
// enough to clear a coarse tile's facets far off.
const RING_LIFT_SHARE: f64 = 0.002;
const MIN_RING_LIFT: f64 = 0.05;
// The center tick's length, as a share of the brush radius.
const TICK_SHARE: f32 = 0.2;
// The falloff below which the brush's inner ring sits on its edge, unshown.
const MIN_SHOWN_FALLOFF: f32 = 0.02;

impl EditorHook {
    // The tool working the viewport: picked, with the Planet panel shown.
    fn planet_tool(&self) -> Option<Tool> {
        self.planet_tool
            .tools
            .active
            .filter(|_| self.panel_shown(PanelKey::Planet) && !self.sim.playing())
    }

    // Each frame, before the press routing: follow a dragged brush slider,
    // find the ground under the cursor, lay the rings around it, and carry a
    // stroke on to it.
    pub(in crate::editor::hook) fn drive_planet_tool(
        &mut self,
        input: &FrameInput,
        vp: [f32; 2],
        world: &World,
    ) {
        self.drive_brush_slider(input, vp);
        if self.planet_tool().is_none() {
            self.planet_tool.hover = None;
            self.planet_tool.stroke = None;
            return;
        }
        let mouse = [input.mouse_x, input.mouse_y];
        let stroking = self.planet_tool.stroke.is_some();
        let planet = world.resource::<PlanetFrame>().copied();
        self.planet_tool.hover = planet
            .filter(|_| stroking || !self.pointer_over_ui(vp, mouse))
            .and_then(|p| {
                let ray = camera_ray(world, input.viewport, mouse)?;
                p.pick_ground(ray.origin, ray.dir, f64::INFINITY)
            });
        if let (Some(p), Some(hit)) = (planet, self.planet_tool.hover) {
            self.lay_rings(&p, hit.geo.latitude, hit.geo.longitude, hit.distance);
        }
        if stroking {
            self.follow_stroke(input.left_button_down, planet.as_ref());
        }
    }

    fn lay_rings(&mut self, planet: &PlanetFrame, latitude: f64, longitude: f64, distance: f64) {
        let center = geo_direction(latitude, longitude);
        let lift = (distance * RING_LIFT_SHARE).max(MIN_RING_LIFT);
        let brush = self.planet_tool.tools.brush;
        let state = &mut self.planet_tool;
        ring::lay(
            planet,
            center,
            f64::from(brush.radius),
            lift,
            &mut state.ring,
        );
        let inner = brush.radius * (1.0 - brush.falloff);
        if brush.falloff > MIN_SHOWN_FALLOFF && inner > 0.0 {
            let out = state.inner.get_or_insert([[0.0; 3]; RING_POINTS]);
            ring::lay(planet, center, f64::from(inner), lift, out);
        } else {
            state.inner = None;
        }
    }

    // Carry the stroke on to the ground under the cursor, handing the points
    // it lays to the tool; a release ends it.
    fn follow_stroke(&mut self, held: bool, planet: Option<&PlanetFrame>) {
        if !held {
            self.planet_tool.stroke = None;
            return;
        }
        let state = &mut self.planet_tool;
        let (Some(stroke), Some(planet)) = (state.stroke.as_mut(), planet) else {
            return;
        };
        state.points.clear();
        match state.hover {
            Some(hit) => {
                let shape = &planet.shape;
                let points = &mut state.points;
                let to = geo_direction(hit.geo.latitude, hit.geo.longitude);
                stroke
                    .spacer
                    .advance(to, |d| points.push(shape.geo_at(shape.surface(d))));
            }
            None => stroke.spacer.lift(),
        }
        if !state.points.is_empty() {
            planet_tools::work(stroke.tool, &state.tools.brush, &state.points);
        }
    }

    // A press the panels and the top bar left: with a tool picked it starts
    // a stroke there, whether or not the cursor is over the ground.
    pub(in crate::editor::hook) fn try_planet_tool_press(&mut self, world: &World) -> bool {
        let Some(tool) = self.planet_tool() else {
            return false;
        };
        let planet = world.resource::<PlanetFrame>().copied();
        let spacing = f64::from(self.planet_tool.tools.brush.radius) * SPACING;
        let radius = planet.map_or(1.0, |p| p.shape.radius);
        self.planet_tool.stroke = Some(Stroke {
            tool,
            spacer: Spacer::new(spacing, radius),
        });
        self.follow_stroke(true, planet.as_ref());
        true
    }

    // Whether the cursor is over the top bar, a shown panel, or a menu or
    // dialog that would take a press there.
    fn pointer_over_ui(&self, vp: [f32; 2], mouse: [f32; 2]) -> bool {
        if self.modal.is_some() || self.create_menu.is_some() || self.display_menu_open {
            return true;
        }
        if mouse[1] < hud::BAR_H {
            return true;
        }
        registry::all().any(|p| {
            let key = p.key();
            self.panel_shown(key)
                && point_in(
                    mouse[0],
                    mouse[1],
                    widget::outer_rect(self.origin(key, vp), self.effective_size(key)),
                )
        })
    }

    // The brush ring, its falloff ring and a tick along the ground's normal,
    // over the ground the tool is on.
    pub(in crate::editor::hook) fn push_brush_lines(&self, out: &mut Vec<Line>) {
        let state = &self.planet_tool;
        let Some(hit) = state.hover.filter(|_| self.planet_tool().is_some()) else {
            return;
        };
        push_ring(out, &state.ring, RING_COLOR, RING_WIDTH_PX);
        if let Some(inner) = &state.inner {
            push_ring(out, inner, INNER_COLOR, INNER_WIDTH_PX);
        }
        let reach = state.tools.brush.radius * TICK_SHARE;
        out.push(Line {
            start: hit.point,
            end: core::array::from_fn(|i| hit.point[i] + hit.normal[i] * reach),
            start_color: RING_COLOR,
            end_color: RING_COLOR,
            width_px: RING_WIDTH_PX,
        });
    }

    // While a brush slider is held, its setting follows the cursor.
    fn drive_brush_slider(&mut self, input: &FrameInput, vp: [f32; 2]) {
        let Some(setting) = self.planet_tool.sliding else {
            return;
        };
        if !input.left_button_down {
            self.planet_tool.sliding = None;
            return;
        }
        let o = self.origin(PanelKey::Planet, vp);
        let w = self.effective_size(PanelKey::Planet)[0];
        let top = planet_panel::strip_top(o);
        let t = planet_tool_strip::slide_to(o, top, w, setting, input.mouse_x);
        self.planet_tool.tools.brush.set_position(setting, t);
    }

    pub(in crate::editor::hook) fn apply_strip_action(&mut self, action: StripAction) {
        let tools = &mut self.planet_tool.tools;
        match action {
            StripAction::Toggle(tool) => {
                tools.toggle(tool);
                self.planet_tool.stroke = None;
            }
            StripAction::Slide(setting, t) => {
                tools.brush.set_position(setting, t);
                self.planet_tool.sliding = Some(setting);
            }
            StripAction::Cycle(list) => {
                let count = self.planet_list(list).map_or(0, |e| e.len());
                self.planet_tool.tools.cycle(list, count);
            }
            StripAction::Consume => {}
        }
        self.planet.focus = None;
    }

    // The tool keys, while the Planet panel is frontmost and none of its
    // boxes is being typed into.
    pub(in crate::editor::hook) fn planet_tool_keys(&mut self, input: &FrameInput) {
        if self.planet.focus.is_some() || self.text_focus_active() || self.sim.playing() {
            return;
        }
        for press in input.key_presses() {
            if self.planet_tool.tools.key(press) && self.planet_tool.tools.active.is_none() {
                self.planet_tool.put_down();
            }
        }
    }

    // The entries of the planet's `list`.
    fn planet_list(&self, list: List) -> Option<Vec<Value>> {
        let idx = self.entry_index_of(crate::editor::panels::planet::TY)?;
        self.entry_args(idx)
            .get(list.key())
            .and_then(Value::as_array)
            .cloned()
    }

    /// What the paint tool's pick shows: the picked entry's grass or material
    /// and its place in the list.
    pub(in crate::editor::hook) fn planet_pick_label(&self) -> String {
        let Some(list) = self.planet_tool.tools.active.and_then(Tool::list) else {
            return String::new();
        };
        let entries = self.planet_list(list).unwrap_or_default();
        if entries.is_empty() {
            return format!("(no {})", list.key());
        }
        let i = self.planet_tool.tools.pick(list).min(entries.len() - 1);
        let field = match list {
            List::Layers => "grass",
            List::Covers => "material",
        };
        let name = entries[i]
            .get(field)
            .and_then(Value::as_str)
            .unwrap_or("(none)");
        format!("{name}   {}/{}", i + 1, entries.len())
    }
}

fn push_ring(out: &mut Vec<Line>, points: &[[f32; 3]; RING_POINTS], color: [f32; 4], width: f32) {
    for k in 0..RING_POINTS {
        out.push(Line {
            start: points[k],
            end: points[(k + 1) % RING_POINTS],
            start_color: color,
            end_color: color,
            width_px: width,
        });
    }
}
