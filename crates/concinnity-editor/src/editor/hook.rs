// src/editor/hook.rs
//
// The editor's per-frame drive. Implements the run loop's `DebugHook` seam: each
// frame it hit-tests the editor HUD's controls against the live input, mutates
// the working authored entry list, persists on SAVE, drives the world's cursor /
// freeze state, and re-anchors + recolours the HUD. This is the whole editor: it
// lives in the editor crate (never linked by the shipped runtime), so no editor
// code is compiled into a shipped game.
//
// The top bar (`hud.rs`) owns SAVE and the Templates dropdown. The Assets button
// opens the browse panel (`panel.rs`): a combo (dropdown) that filters the
// browse list by type, a "+" that opens a typed autocomplete of the addable
// types, and a browse list grouped by type. Clicking a name (or picking a type
// from the "+" picker) opens the add / edit form in its own floating panel
// (`form_panel.rs`); the browse row of the edited entry stays highlighted. The
// combo's filter field and the form's name heading are real `TextInput` assets
// edited by the engine's text-input system; the hook reads them back. All three
// panels (Assets, edit form, Preview) are floating: holding their title bars
// drags them (the hook owns each origin, clamped so a panel can never leave the
// screen).
//
// Cursor control: the editor holds the cursor by default (edit mode -- cursor
// free, world frozen), publishing `MenuOverride(Some(true))`. Ticking the
// Preview panel's capture checkbox hands the cursor to the world (`Some(false)`);
// Escape takes it back. F1 hides / shows the whole HUD.
//
// SAVE re-serializes the authored entry list to world.jsonl and recompiles the
// blobs through the validated cook tail (`build_world_to_disk`). A successful
// SAVE then applies the edit to the running world without recreating the window:
// the live render backend is transplanted out of the world and `apply_world_swap`
// rebuilds the recompiled world onto it (carried in as a `PendingBackend`).

use super::form::{self, FormField};
use super::form_panel::{self, FormAction, FormFocus, FormView};
use super::hud::{self, HudAction, HudState};
use super::lighting;
use super::lighting_panel::{self, LightingAction, LightingView};
use super::list_panel::Row;
use super::panel::{self, Combo, ListRow, PanelAction, PanelView};
use super::preview::{self, PreviewAction};
use super::registry::{self, PANEL_COUNT, PanelKey};
use super::template_panel::{self, TemplateAction, TemplateView};
use super::templates::{self, TemplatesAction};
use super::view::{self, ViewAction};
use super::widget::{self, point_in};
// Re-exported for the hook's submodules (they reach these editor-level items as
// `super::asset_list` / `super::seeded_content`).
use super::asset_list;
use super::seeded_content;
use crate::app::state::App;
use crate::assets::FrameInput;
use crate::debug_hook::DebugHook;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{HudLayers, MenuOverride, PendingBackend, World};

// Draw layer for the top bar: far above the floating panels' layers (which are a
// small 1..=6 rank), so the bar always sits on top even under a dragged panel.
const TOP_BAR_LAYER: i32 = 1_000;

// An active title-bar drag: the grabbed panel and the cursor's offset from its
// origin at the press, so the panel follows without snapping to the cursor.
#[derive(Debug, Clone, Copy)]
struct Drag {
    key: PanelKey,
    grab: [f32; 2],
}

pub(crate) struct EditorHook {
    // Path to the world.jsonl the edits are written back to.
    world_path: String,
    // The authored entry list (names live here, unlike the compiled blob). Edits
    // mutate this; SAVE serializes it back to `world_path`.
    entries: Vec<serde_json::Value>,
    // Whether `entries` has changes not yet written to disk.
    dirty: bool,
    // Whether the world currently holds the cursor (play mode). Starts false:
    // the editor owns the cursor at launch so the HUD is immediately usable.
    world_capture: bool,
    // Whether the whole HUD is shown (F1 toggle). Starts shown.
    hud_visible: bool,
    // Set whenever the authored entries change (add / edit / delete / template);
    // consumed by `apply_world_swap` to rebuild the live preview world from the
    // in-memory entries. SAVE does not set this -- the preview is already current.
    rebuild_preview: bool,
    // Whether the Templates panel is shown (toggled from the View panel).
    templates_open: bool,
    // The template whose detail panel is open (index into the templates
    // registry); `None` means the detail panel is closed.
    open_template: Option<usize>,
    // First visible row of the Template detail panel's asset list.
    template_list_scroll: usize,
    // Whether the Lighting panel is shown (toggled from the View panel), which
    // text binding holds keyboard focus, and the message from the last rejected
    // Apply.
    lighting_open: bool,
    lighting_focus: Option<usize>,
    lighting_status: Option<String>,
    // Whether the Assets panel is shown (toggled from the View panel).
    panel_open: bool,
    // Whether the Preview panel is shown (starts shown; toggled from the View
    // panel).
    preview_open: bool,
    // Whether the View panel itself is shown (the top-bar View button toggles it).
    view_open: bool,
    // The header combo (dropdown) state: closed, filtering, or picking a type.
    combo: Combo,
    // Active type filter for the browse list, or `None` for "all".
    type_filter: Option<String>,
    // First visible row of the (grouped) browse list / the combo option list.
    list_scroll: usize,
    combo_scroll: usize,
    // The type of the open add / edit form; `None` means the form panel is
    // closed.
    selected_type: Option<String>,
    // When the form is editing an existing entry, its `entries` index (else a new
    // asset is being added).
    editing: Option<usize>,
    // The editable arg fields of the open form (derived from the type's default
    // args). Empty while the form is closed.
    form_fields: Vec<FormField>,
    // First visible field of the form's scroll window (its physical control pool is
    // fixed size, so a form wider than `form::FIELD_POOL` scrolls). Reset on open /
    // structural change.
    form_scroll: usize,
    // Which form input has keyboard focus.
    form_focus: FormFocus,
    // A validation message from the last rejected Add, shown under the form.
    form_error: Option<String>,
    // The `entries` index whose Delete menu is open, if any.
    row_menu: Option<usize>,
    // The form arg field whose value dropdown is open (a large enum / ref set),
    // and its scroll offset. `None` outside an open dropdown.
    field_dropdown: Option<usize>,
    field_dropdown_scroll: usize,
    // The open form's working args tree: the fields are derived from it, and it is
    // mutated by add / remove (structure) and, on capture, by the controls. Empty
    // outside AddForm.
    form_args: serde_json::Map<String, serde_json::Value>,
    // The paths of the form's non-colour vector fields currently disclosed into
    // per-element leaves. Cleared when the form opens / closes.
    vec_expanded: std::collections::HashSet<String>,
    // The floating panels' dragged origins, indexed by `PanelKey`; `None` means
    // the panel still sits at its default anchor. Always clamped fully on screen
    // before use.
    positions: [Option<[f32; 2]>; PANEL_COUNT],
    // The title-bar drag in progress, if any.
    drag: Option<Drag>,
    // The floating panels back-to-front: the last entry is the frontmost (drawn on
    // top + first to receive clicks). Dragging or clicking a panel moves it to the
    // end. Its position drives the per-frame `HudLayers` publish so overlapping
    // panels occlude cleanly instead of merging.
    panel_order: Vec<PanelKey>,
}

// Owned per-tick data backing a `PanelView` (computed from the entries + the live
// filter field, then borrowed for both hit-testing and layout).
struct PanelData {
    filter_label: String,
    combo_options: Vec<String>,
    combo_selected: Option<usize>,
    list_rows: Vec<ListRow>,
    form_title: String,
}

// Owned per-tick data backing a `TemplateView` (the open template's title,
// description, and grouped asset rows).
#[derive(Default)]
struct TemplateDetailData {
    title: String,
    description: String,
    rows: Vec<ListRow>,
}

// Owned per-tick data backing a `LightingView` (the row list and the
// per-binding fields derived from the current entries).
struct LightingData {
    rows: Vec<lighting::Row>,
    fields: Vec<Option<FormField>>,
}

// Move a scroll offset one row toward the wheel direction, clamped to `max`.
fn scroll_step(cur: usize, delta: f32, max: usize) -> usize {
    if delta > 0.0 {
        (cur + 1).min(max)
    } else {
        cur.saturating_sub(1)
    }
}

// The physical control slot showing logical form field `j` under scroll offset
// `scroll`, or `None` when the field is outside the visible window. The panel's
// control pool is slot-indexed, so seeding / reading a field goes through its slot.
fn visible_slot(j: usize, scroll: usize) -> Option<usize> {
    (j >= scroll && j < scroll + form::FIELD_POOL).then(|| j - scroll)
}

// The first line of a validation error, clipped to fit the panel's status line.
fn short_error(e: &str) -> String {
    let line = e.lines().next().unwrap_or(e);
    let clipped: String = line.chars().take(44).collect();
    if clipped.len() < line.len() {
        format!("{clipped}...")
    } else {
        clipped
    }
}

// The `name` string of an entry, if present.
fn entry_name(e: &serde_json::Value) -> Option<&str> {
    e.get("name").and_then(|v| v.as_str())
}
fn entry_type(e: &serde_json::Value) -> Option<&str> {
    e.get("type").and_then(|v| v.as_str())
}

// The names of the working entries whose type is `ty` (the reference options a
// field targeting that type can pick from).
fn names_of_type(entries: &[serde_json::Value], ty: &str) -> Vec<String> {
    entries
        .iter()
        .filter(|e| entry_type(e) == Some(ty))
        .filter_map(|e| entry_name(e).map(String::from))
        .collect()
}

mod browse;
mod editing;
mod edits;
mod layout;
// Named to avoid colliding with the `use super::lighting` module import.
mod lighting_edit;
// The per-panel `Panel` impls, reachable by the registry (`editor/registry.rs`).
pub(super) mod panels;
mod routing;
#[cfg(test)]
mod tests;

impl EditorHook {
    pub(crate) fn new(world_path: String, entries: Vec<serde_json::Value>) -> Self {
        Self {
            world_path,
            entries,
            dirty: false,
            world_capture: false,
            hud_visible: true,
            rebuild_preview: false,
            templates_open: false,
            open_template: None,
            template_list_scroll: 0,
            lighting_open: false,
            lighting_focus: None,
            lighting_status: None,
            panel_open: false,
            preview_open: true,
            view_open: false,
            combo: Combo::Closed,
            type_filter: None,
            list_scroll: 0,
            combo_scroll: 0,
            selected_type: None,
            editing: None,
            form_fields: Vec::new(),
            form_scroll: 0,
            form_focus: FormFocus::Name,
            form_error: None,
            row_menu: None,
            field_dropdown: None,
            field_dropdown_scroll: 0,
            form_args: serde_json::Map::new(),
            vec_expanded: std::collections::HashSet::new(),
            positions: [None; PANEL_COUNT],
            drag: None,
            // Back-to-front, matching the injected draw order (registry order:
            // the Template detail panel frontmost, over the Templates list it
            // spawns from).
            panel_order: PanelKey::ALL.to_vec(),
        }
    }
}

impl DebugHook for EditorHook {
    fn tick(&mut self, world: &mut World) {
        let input = world.query::<FrameInput>().last().cloned();
        if let Some(input) = &input {
            // Escape hands the cursor back to the editor (leaves play mode).
            if input.escape {
                self.world_capture = false;
            }
            // F1 (an edge pulse) toggles the whole HUD.
            if input.hud_toggle {
                self.hud_visible = !self.hud_visible;
            }
            let vp = input.viewport;
            if self.hud_visible && vp[0] > 0.0 {
                // An active title-bar drag follows the cursor; a fresh press (only
                // when no drag is running -- the press that starts one must not
                // also resolve to a control) routes to the bar and the panels.
                self.drive_drag(input, vp);
                if input.left_click && self.drag.is_none() {
                    self.route_click(input, vp, world);
                }
                // Wheel routing: the frontmost scrollable panel under the cursor
                // takes the wheel. An open value dropdown is modal and can extend
                // past the form panel, so it scrolls the form from anywhere while
                // open.
                if input.scroll_delta.abs() > 0.5 {
                    let (mx, my) = (input.mouse_x, input.mouse_y);
                    let form_shown = registry::panel(PanelKey::Edit).is_open(self);
                    if form_shown && self.field_dropdown.is_some() {
                        self.scroll_form(input.scroll_delta, world);
                    } else {
                        let front_to_back: Vec<PanelKey> =
                            self.panel_order.iter().rev().copied().collect();
                        for key in front_to_back {
                            let p = registry::panel(key);
                            if !p.is_open(self) {
                                continue;
                            }
                            let o = self.origin(key, vp);
                            if p.wheel_over(self, world, mx, my, o) {
                                p.scroll(self, world, input.scroll_delta);
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Drive the world's cursor / freeze state: edit mode (`Some(true)`) frees
        // the cursor and freezes the world; play mode (`Some(false)`) runs it.
        world.insert_resource(MenuOverride(Some(!self.world_capture)));

        // Re-anchor + recolour the top bar, then lay out (or hide) the panels.
        hud::apply_layout(world, self.hud_state());
        let vp = input.as_ref().map(|i| i.viewport).unwrap_or([0.0, 0.0]);
        let mouse = input
            .as_ref()
            .map(|i| [i.mouse_x, i.mouse_y])
            .unwrap_or([0.0, 0.0]);
        // The panels lay out only once the HUD is shown and a real viewport exists
        // (frame 0 keeps the injected-hidden placeholders).
        let shown = self.hud_visible && vp[0] > 0.0;
        // Publish the panels' draw layers (focus stack) so the renderer occludes
        // overlaps by focus this frame. Empty while the HUD is hidden, so the
        // renderer skips the overlay sort entirely.
        if shown {
            self.publish_layers(world);
        } else {
            world.insert_resource(HudLayers::default());
        }
        // Lay out every open panel (hiding keeps its state, so toggling back
        // restores the same view). Compound gates live on each panel's
        // `is_open` -- e.g. the edit form shows only while the assets UI is on.
        for p in registry::all() {
            if shown && p.is_open(self) {
                let o = self.origin(p.key(), vp);
                p.draw(self, world, o, mouse);
            } else {
                p.hide(world);
            }
        }
    }

    // Rebuild the live preview world from the in-memory entries and swap it under
    // the running render backend, so any authored change (add / edit / delete /
    // template apply) shows immediately without recreating the OS window or
    // touching disk (SAVE owns persistence). Run once per frame by the run loop
    // right after `tick`, whenever an edit flagged `rebuild_preview`.
    //
    // The recompiled world is built FIRST, in a throwaway App; only once that
    // succeeds is the backend transplanted out of the live world. So a rebuild
    // failure leaves the live world -- and its window -- fully intact; the next
    // edit retries. The backend is never dropped on an error path.
    fn apply_world_swap(&mut self, app: &mut App) {
        if !self.rebuild_preview {
            return;
        }
        self.rebuild_preview = false;

        let world = match self.build_preview_world() {
            Ok(w) => w,
            Err(e) => {
                tracing::error!("editor: live preview rebuild failed, keeping current world: {e}");
                return;
            }
        };
        let mut staged = App::new();
        staged.load_world(world);
        // Carry the editor's typed text (an open form's name + fields, the combo
        // filter) across the fresh HUD injection so it is not blanked.
        let fields = Self::field_snapshot(app.world());
        super::inject::editor_hud(staged.world_mut());
        Self::restore_fields(staged.world_mut(), &fields);
        staged
            .world_mut()
            .insert_resource(MenuOverride(Some(!self.world_capture)));

        let Some(backend) = app.world_mut().take_render_backend() else {
            return;
        };
        staged.world_mut().insert_resource(PendingBackend(backend));
        let new_world = std::mem::replace(staged.world_mut(), World::new_empty());
        app.load_world(new_world);
        if let Err(e) = app.start() {
            tracing::error!("editor: live preview start failed: {e:?}");
        }
    }
}
