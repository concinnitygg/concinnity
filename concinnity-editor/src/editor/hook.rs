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
use super::panel::{self, Combo, ListRow, PanelAction, PanelView};
use super::preview::{self, PreviewAction};
use super::template_panel::{self, TemplateAction, TemplateView};
use super::templates::{self, TemplatesAction};
use super::view::{self, ViewAction, ViewState};
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
// small 1..=3 rank), so the bar always sits on top even under a dragged panel.
const TOP_BAR_LAYER: i32 = 1_000;

// Which floating panel a title-bar drag is moving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DragTarget {
    Assets,
    Edit,
    Preview,
    View,
    Templates,
    // The Template detail panel spawned by picking a Templates-list row.
    TemplateDetail,
}

// Which scroll region a wheel event lands in (decided by the tick from the
// cursor position, since the browse and form panels can be open at once).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScrollTarget {
    // The edit form's field window (or its open value dropdown).
    Form,
    // The Assets panel's browse list / combo options.
    List,
    // The Template detail panel's asset list.
    TemplateList,
}

// An active title-bar drag: the grabbed panel and the cursor's offset from its
// origin at the press, so the panel follows without snapping to the cursor.
#[derive(Debug, Clone, Copy)]
struct Drag {
    target: DragTarget,
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
    // The floating panels' dragged origins; `None` means the panel still sits at
    // its default anchor. Always clamped fully on screen before use.
    panel_pos: Option<[f32; 2]>,
    edit_pos: Option<[f32; 2]>,
    preview_pos: Option<[f32; 2]>,
    view_pos: Option<[f32; 2]>,
    templates_pos: Option<[f32; 2]>,
    template_detail_pos: Option<[f32; 2]>,
    // The title-bar drag in progress, if any.
    drag: Option<Drag>,
    // The floating panels back-to-front: the last entry is the frontmost (drawn on
    // top + first to receive clicks). Dragging or clicking a panel moves it to the
    // end. Its position drives the per-frame `HudLayers` publish so overlapping
    // panels occlude cleanly instead of merging.
    panel_order: Vec<DragTarget>,
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
            panel_pos: None,
            edit_pos: None,
            preview_pos: None,
            view_pos: None,
            templates_pos: None,
            template_detail_pos: None,
            drag: None,
            // Back-to-front, matching the injected draw order (the Template detail
            // panel frontmost, over the Templates list it spawns from).
            panel_order: vec![
                DragTarget::Assets,
                DragTarget::Edit,
                DragTarget::Preview,
                DragTarget::View,
                DragTarget::Templates,
                DragTarget::TemplateDetail,
            ],
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
                // takes the wheel (the form's field window or the Assets list). An
                // open value dropdown is modal and can extend past the panel, so it
                // scrolls the form from anywhere while open.
                if input.scroll_delta.abs() > 0.5 {
                    let (mx, my) = (input.mouse_x, input.mouse_y);
                    let form_shown = self.form_open() && self.panel_open;
                    if form_shown && self.field_dropdown.is_some() {
                        self.scroll(input.scroll_delta, ScrollTarget::Form, world);
                    } else {
                        let front_to_back: Vec<DragTarget> =
                            self.panel_order.iter().rev().copied().collect();
                        for target in front_to_back {
                            let hit = match target {
                                DragTarget::Edit if form_shown => {
                                    let data = self.panel_data(world);
                                    let view = self.make_form_view(&data, [mx, my]);
                                    form_panel::cursor_over(&view, mx, my, self.edit_origin(vp))
                                        .then_some(ScrollTarget::Form)
                                }
                                DragTarget::Assets if self.panel_open => {
                                    panel::cursor_over_body(mx, my, self.panel_origin(vp))
                                        .then_some(ScrollTarget::List)
                                }
                                DragTarget::TemplateDetail => self
                                    .open_template
                                    .filter(|_| self.templates_open)
                                    .and_then(|i| {
                                        let data = self.template_detail_data(i);
                                        let view = self.make_template_view(&data, [mx, my]);
                                        template_panel::cursor_over(
                                            &view,
                                            mx,
                                            my,
                                            self.template_detail_origin(i, vp),
                                        )
                                        .then_some(ScrollTarget::TemplateList)
                                    }),
                                _ => None,
                            };
                            if let Some(t) = hit {
                                self.scroll(input.scroll_delta, t, world);
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
        // Preview panel (toggled from the View panel; shown by default).
        if shown && self.preview_open {
            preview::apply(world, self.preview_origin(vp), self.world_capture, mouse);
        } else {
            preview::hide_all(world);
        }
        // View panel (toggled from the top-bar View button).
        if shown && self.view_open {
            view::apply(world, self.view_origin(vp), self.view_state(), mouse);
        } else {
            view::hide_all(world);
        }
        // Templates panel (toggled from the View panel); the picked template's row
        // stays highlighted while its detail panel is open.
        if shown && self.templates_open {
            templates::apply(world, self.templates_origin(vp), self.open_template, mouse);
        } else {
            templates::hide_all(world);
        }
        // Template detail panel: shown only while the Templates list is open and a
        // template is picked; hiding it keeps `open_template` so a later toggle-on
        // restores it.
        let detail = self.open_template.filter(|_| shown && self.templates_open);
        if let Some(i) = detail {
            let data = self.template_detail_data(i);
            let view = self.make_template_view(&data, mouse);
            template_panel::apply(world, Some(&view), self.template_detail_origin(i, vp));
        } else {
            template_panel::apply(world, None, [0.0, 0.0]);
        }
        // Assets panel (toggled from the View panel).
        if shown && self.panel_open {
            let data = self.panel_data(world);
            let view = self.make_view(&data, mouse);
            panel::apply(world, Some(&view), self.panel_origin(vp));
        } else {
            panel::apply(world, None, [0.0, 0.0]);
        }
        // The edit form shows only while the assets UI is on (panel_open); hiding
        // it keeps its state so a later toggle-on restores it.
        let show_form = shown && self.panel_open && self.form_open();
        if show_form {
            let data = self.panel_data(world);
            let view = self.make_form_view(&data, mouse);
            form_panel::apply(world, Some(&view), self.edit_origin(vp));
        } else {
            form_panel::apply(world, None, [0.0, 0.0]);
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
        staged.world_mut().remove_all::<crate::assets::DebugHud>();
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
