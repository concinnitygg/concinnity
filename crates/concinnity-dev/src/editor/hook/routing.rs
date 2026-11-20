//! `EditorHook`: per-frame pointer routing -- the in-flight viewport gestures,
//! the press that reaches the world once the top bar and the panels (which take
//! their own) leave it, and the editor's keyboard shortcuts.

use concinnity_engine::components::{FrameInput, Identity, InputKey, Layout};
use concinnity_engine::ecs::World;
use concinnity_engine::ecs::asset_id::AssetId;

use super::{EditorHook, bookmarks};
use crate::editor::hud::{self, HudState};
use crate::editor::panels::registry::{self, PanelKey};
use crate::editor::ui::BuiltUi;
use crate::editor::ui::rect::point_in;
use crate::editor::viewport::gizmo;

impl EditorHook {
    // Everything a frame's input drives while a world is open: the in-flight
    // viewport gestures, the press that reaches the world once the top bar and
    // the panels (which take their own) leave it, and the editor's keyboard
    // shortcuts. The start screen routes none of it: its panel drives itself.
    pub(super) fn drive_session_input(
        &mut self,
        input: &FrameInput,
        vp: [f32; 2],
        world: &mut World,
    ) {
        // An in-flight gizmo drag follows the cursor, cancels on
        // Escape, and commits on release, before any new press routes.
        if self.gizmo_drag.is_some() {
            self.drive_gizmo_drag(input, vp, world);
        }
        // A shape slider drag likewise: follow + live preview,
        // cancel, or commit once.
        if self.shape.drag.is_some() {
            self.drive_shape_drag(input, world);
        }
        // An in-flight marquee likewise: follow, cancel, or select.
        if self.marquee.is_some() {
            self.drive_marquee(input, vp, world);
        }
        // An in-flight orbit tumble follows the cursor and ends on
        // release.
        if self.orbit.is_some() {
            self.drive_orbit(input, world);
        }
        // A drag-out placement from the Content panel: ghost follow,
        // cancel, or commit.
        if self.content.drag.is_some() {
            self.drive_content_drag(input, vp, world);
        }
        // A picked Planet tool finds the ground under the cursor and carries
        // its stroke on.
        self.drive_planet_tool(input, vp, world);
        if input.left_click
            && self.gizmo_drag.is_none()
            && self.shape.drag.is_none()
            && self.marquee.is_none()
            && self.content.drag.is_none()
            && self.orbit.is_none()
            && !self.planet_tool.busy()
        {
            // An open dialog or menu takes the press itself, so nothing here
            // routes it. Otherwise an Alt+press over the viewport starts an
            // orbit tumble, and anything else routes to the bar, the panels,
            // and the world.
            let overlay_open =
                self.modal.is_some() || self.display_menu_open || self.create_menu.is_some();
            // A press away from an open transient layout (a palette) is the
            // engine's to dismiss it with, so it reaches nothing here.
            if !overlay_open && !self.press_dismisses(world, input) {
                if input.alt && self.try_begin_orbit(input, vp, world) {
                    self.editor_took_press = true;
                } else {
                    self.route_click(input, vp, world);
                }
            }
        }
        // An unclaimed viewport right press opens the create menu at
        // the cursor, under the same not-mid-gesture guards; the
        // confirmation dialog swallows it like every other press, and one
        // away from an open palette only dismisses it.
        if input.right_click
            && self.modal.is_none()
            && !self.press_dismisses(world, input)
            && self.gizmo_drag.is_none()
            && self.shape.drag.is_none()
            && self.marquee.is_none()
            && self.content.drag.is_none()
            && self.orbit.is_none()
            && !self.planet_tool.busy()
        {
            self.open_create_menu(input, vp, world);
        }
        // T / R / S pick the gizmo's mode (translate / rotate /
        // scale), under the same guards as the history shortcuts.
        if !input.ctrl
            && !self.sim.playing()
            && !self.text_focus_active()
            && self.gizmo_drag.is_none()
        {
            for press in input.key_presses() {
                match press.key {
                    // F and H toggle, so a held key acts once.
                    InputKey::F | InputKey::H if press.repeat => {}
                    InputKey::T => self.gizmo_mode = gizmo::GizmoMode::Translate,
                    InputKey::R => self.gizmo_mode = gizmo::GizmoMode::Rotate,
                    InputKey::S => self.gizmo_mode = gizmo::GizmoMode::Scale,
                    // F frames the selection; Shift+F keeps the old fly
                    // toggle one modifier away.
                    InputKey::F => {
                        if press.mods.shift {
                            self.toggle_fly();
                        } else {
                            self.frame_selection(input.viewport, world);
                        }
                    }
                    // H hides the selection; Shift+H isolates it.
                    InputKey::H => {
                        if press.mods.shift {
                            self.toggle_isolate();
                        } else {
                            self.hide_selected();
                        }
                    }
                    // 1..9 glide back to a saved camera bookmark.
                    key => {
                        if let Some(slot) = bookmarks::slot_for(key) {
                            self.recall_bookmark(slot, world);
                        }
                    }
                }
            }
        }
        // Backtick toggles the console. The flag cleared here is the
        // one-frame focus blur a backtick open sets, so the text
        // system never types that backtick into the command line.
        self.console.blur = false;
        self.drive_console_toggle(input, world);
        // Ctrl+K toggles the palette, with the same one-frame blur.
        self.palette.blur = false;
        self.drive_palette_toggle(input);
        // Undo/redo, duplicate and drop-to-floor, unless the world, a text field
        // or a gizmo drag owns the input. A frontmost Behavior panel keeps its
        // own Ctrl+D.
        if input.ctrl
            && !self.sim.playing()
            && !self.text_focus_active()
            && self.gizmo_drag.is_none()
            && self.shape.drag.is_none()
        {
            for key in input.pressed_keys() {
                match key {
                    InputKey::Z => self.undo(),
                    InputKey::Y => self.redo(),
                    InputKey::D if self.frontmost_open_panel() != Some(PanelKey::Behavior) => {
                        self.duplicate_selection();
                    }
                    InputKey::Down => {
                        self.drop_selection_to_floor(world);
                    }
                    // Ctrl+H makes everything visible again.
                    InputKey::H => self.unhide_all(),
                    // Ctrl+1..9 save the camera pose to a bookmark.
                    key => {
                        if let Some(slot) = bookmarks::slot_for(key) {
                            self.save_bookmark(slot, world);
                        }
                    }
                }
            }
        }
        // The transport shortcuts, live in every state (pausing a
        // running world is their whole point).
        self.sim_keys(input);
        // Per-frame editing keys go to the frontmost open panel.
        if let Some(key) = self.frontmost_open_panel() {
            registry::panel(key).frame_keys(self, world, input);
        }
    }

    pub(super) fn hud_state(&self) -> HudState {
        HudState {
            dirty: self.dirty,
            undo: self.can_undo(),
            redo: self.can_redo(),
            view_open: self.open[PanelKey::View],
            display_open: self.display_menu_open,
            sim: self.sim.state,
            // The start screen is the whole window: the bar it would sit under
            // is not drawn, and never resolves a press.
            visible: self.hud_visible && !self.start_mode,
        }
    }

    // Route a press: the top bar first (it draws over the panels), then the panels
    // front-to-back so the frontmost claims a press in an overlap. Whichever panel
    // claims it comes to the front.
    pub(super) fn route_click(&mut self, input: &FrameInput, vp: [f32; 2], world: &World) {
        let (mx, my) = (input.mouse_x, input.mouse_y);
        // The toast stack draws above everything and takes its own presses.
        if Self::over_toasts(world, mx, my, vp) {
            return;
        }
        // The top bar takes its own presses, and one between its chips
        // reaches nothing behind it.
        if my < hud::BAR_H && mx >= 0.0 && mx < vp[0] {
            return;
        }
        // Front-to-back: the frontmost shown panel to claim the press handles it and
        // rises to the front (so clicking an exposed sliver of a buried panel brings
        // it forward). `panel_order`'s tail is frontmost.
        let front_to_back: Vec<PanelKey> = self.panel_order.iter().rev().copied().collect();
        for key in front_to_back {
            if self.try_panel_press(key, mx, my, world) {
                return;
            }
        }
        // An unclaimed press goes to the Planet tool, then gizmo handles, then
        // billboards, then the 3D pick. Edit mode only.
        if self.try_planet_tool_press(world) {
            return;
        }
        if self.try_gizmo_press(input, vp, world) {
            return;
        }
        if self.try_billboard_press(input, vp, world) {
            return;
        }
        self.click_world(input, world);
    }

    // Whether the shown panel registered at `key` lies under a press, which
    // brings it to the front: `false` sends the caller on to the next panel back.
    pub(super) fn try_panel_press(
        &mut self,
        key: PanelKey,
        mx: f32,
        my: f32,
        world: &World,
    ) -> bool {
        if !self.panel_shown(key) {
            return false;
        }
        // The panel takes its own presses: the hook only keeps them from
        // reaching what is behind it.
        let hit = Self::built_rect(world, key).is_some_and(|r| point_in(mx, my, r));
        if hit {
            self.focus_panel(key);
        }
        hit
    }

    // Whether a press lands outside every shown layout of the editor's that
    // names an `on_dismiss` event, where each was last laid out: the press the
    // engine takes to dismiss them, so it reaches nothing here. As in the
    // engine, a press on a panel shown in front of a layout's own is outside it.
    pub(super) fn press_dismisses(&self, world: &World, input: &FrameInput) -> bool {
        let Some(built) = world.resource::<BuiltUi>() else {
            return false;
        };
        let Some(screen) = built.panels_iter().find_map(|p| p.screen) else {
            return false;
        };
        let (mx, my) = (input.mouse_x, input.mouse_y);
        let mut any = false;
        for (entity, layout) in world.query_with_entity::<Layout>() {
            let ours = layout.screen.is_some_and(|s| s.id() == screen);
            let Some(rect) = layout
                .rect
                .filter(|_| ours && layout.visible && !layout.on_dismiss.is_empty())
            else {
                continue;
            };
            any = true;
            let id = world.get::<Identity>(entity).map(|i| i.id());
            if point_in(mx, my, rect) && !self.covered_in_front(world, built, id, mx, my) {
                return false;
            }
        }
        any
    }

    // Whether a panel shown in front of the one holding `id` lies under the
    // press; nothing covers a layout no panel holds.
    fn covered_in_front(
        &self,
        world: &World,
        built: &BuiltUi,
        id: Option<AssetId>,
        mx: f32,
        my: f32,
    ) -> bool {
        let owner = id.and_then(|id| {
            PanelKey::ALL.into_iter().find(|&k| {
                built
                    .panel(k)
                    .is_some_and(|p| p.root == id || p.elements.contains(&id))
            })
        });
        let Some(at) = owner.and_then(|k| self.panel_order.iter().position(|&o| o == k)) else {
            return false;
        };
        self.panel_order[at + 1..].iter().any(|&front| {
            self.panel_shown(front)
                && Self::built_rect(world, front).is_some_and(|r| point_in(mx, my, r))
        })
    }
}
