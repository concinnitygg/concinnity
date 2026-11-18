//! `EditorHook`: the right-click "Create here" menu flow. An unclaimed viewport
//! right press opens the menu at the cursor, capturing the world position under
//! it (surface hit, ground-plane fallback -- the drag-out landing rules) at
//! open time; picking a row creates one entry at that point through the shared
//! entries + `mark_changed` path (one undo step) and selects it. The menu
//! (`editor/create_menu.rs`) closes itself on a click away or Escape; a history
//! jump closes it too.

use concinnity_cook::authoring::registry::RegisteredType;
use concinnity_engine::components::{FrameInput, UiEvent};
use concinnity_engine::ecs::World;

use crate::editor::asset_handle::AssetHandle;
use crate::editor::create_menu::{self, MenuItem};
use crate::editor::hook::{EditorHook, drag, names_of_types};
use crate::editor::hud;
use crate::editor::panels::registry::{self, PanelKey};
use crate::editor::ui::BuiltUi;
use crate::editor::widget::{self, point_in};

// An open "Create here" menu.
pub(in crate::editor::hook) struct CreateMenu {
    // The world position captured from under the right-click.
    pub(in crate::editor::hook) pos: [f32; 3],
    // Whether the Prefab section is unfolded.
    pub(in crate::editor::hook) prefabs_open: bool,
}

// The rows the menu lists, read at an entries revision and fold state.
pub(in crate::editor::hook) struct CreateItems {
    key: (u64, bool),
    items: Vec<MenuItem>,
}

impl EditorHook {
    // The menu rows for the current entry list and fold state.
    pub(in crate::editor::hook) fn create_menu_items(&self) -> Vec<MenuItem> {
        let prefabs = names_of_types(&self.entries, &["Prefab"]);
        let open = self.create_menu.as_ref().is_some_and(|m| m.prefabs_open);
        create_menu::items(&prefabs, open)
    }

    // Open (or move) the menu at a right press, unless the press belongs to
    // the top bar, a floating panel, or the open menu itself, or the world
    // offers no landing point (no camera).
    pub(in crate::editor::hook) fn open_create_menu(
        &mut self,
        input: &FrameInput,
        vp: [f32; 2],
        world: &World,
    ) {
        let (mx, my) = (input.mouse_x, input.mouse_y);
        if self.sim.playing() || my <= hud::BAR_H || self.over_open_panel(mx, my, vp) {
            self.create_menu = None;
            return;
        }
        if self.over_create_menu(world, mx, my) {
            return;
        }
        let Some(pos) = self.drop_point(world, vp, [mx, my], input.ctrl) else {
            return;
        };
        if create_menu::items(&names_of_types(&self.entries, &["Prefab"]), false).is_empty() {
            return;
        }
        self.create_menu = Some(CreateMenu {
            pos,
            prefabs_open: false,
        });
        self.display_menu_open = false;
    }

    // Whether `(mx, my)` lands on the open menu.
    fn over_create_menu(&self, world: &World, mx: f32, my: f32) -> bool {
        let rows = self.create_items.as_ref().map_or(0, |c| c.items.len());
        self.create_menu.is_some()
            && world
                .resource::<BuiltUi>()
                .and_then(|b| b.create_menu)
                .is_some_and(|id| create_menu::covers(world, id, mx, my, rows))
    }

    // Whether `(mx, my)` lands on any open floating panel.
    pub(super) fn over_open_panel(&self, mx: f32, my: f32, vp: [f32; 2]) -> bool {
        PanelKey::ALL.iter().any(|&key| {
            registry::panel(key).is_open(self)
                && point_in(
                    mx,
                    my,
                    widget::outer_rect(self.origin(key, vp), self.effective_size(key)),
                )
        })
    }

    // A pick from the menu; whether `event` was one.
    pub(in crate::editor::hook) fn on_create_menu_event(&mut self, event: &UiEvent) -> bool {
        let Some(row) = create_menu::pick(event) else {
            return false;
        };
        let item = self
            .create_items
            .as_ref()
            .and_then(|c| c.items.get(row))
            .cloned();
        let Some(pos) = self.create_menu.as_ref().map(|m| m.pos) else {
            return true;
        };
        match item {
            Some(MenuItem::Type(ty)) => {
                self.create_type_at(ty, pos);
                self.create_menu = None;
            }
            Some(MenuItem::PrefabHeader { open }) => {
                if let Some(m) = &mut self.create_menu {
                    m.prefabs_open = !open;
                }
            }
            Some(MenuItem::Prefab(name)) => {
                self.create_prefab_at(&name, pos);
                self.create_menu = None;
            }
            None => {}
        }
        true
    }

    // Take in a close the menu made itself: a click away or Escape. A world
    // the menu was just built into has shown it open nowhere yet.
    pub(in crate::editor::hook) fn observe_create_menu(&mut self, world: &World) {
        let fresh = world.resource::<BuiltUi>().is_some_and(|b| b.fresh);
        if !fresh && self.create_menu.is_some() && !create_menu::is_open(world) {
            self.create_menu = None;
        }
    }

    // Create one anonymous entry of `ty` at `pos`: default args with the
    // position overridden, one undo step, selected. A placed asset is addressed
    // by its label until the user gives it a `$id`.
    fn create_type_at(&mut self, ty: &str, pos: [f32; 3]) {
        let arg = create_menu::position_key(ty);
        let args = serde_json::json!({ arg: pos.map(crate::editor::round::round3) });
        let key = self
            .entries
            .push(serde_json::json!({ "type": ty, "args": args }));
        self.mark_changed();
        self.selection.replace(AssetHandle::Entry(key));
    }

    // Create an anonymous Prop instancing `prefab` at `pos` (the drag-out
    // mapping).
    fn create_prefab_at(&mut self, prefab: &str, pos: [f32; 3]) {
        let Some(args) = drag::content::placement_args(
            RegisteredType::Prefab,
            prefab,
            pos.map(crate::editor::round::round3),
        ) else {
            return;
        };
        let key = self
            .entries
            .push(serde_json::json!({ "type": "Prop", "args": args }));
        self.mark_changed();
        self.selection.replace(AssetHandle::Entry(key));
    }

    // Bring the menu in line with the hook: its rows when they change, and
    // whether it is open. It closes while the viewport overlays are hidden.
    pub(in crate::editor::hook) fn drive_create_menu(&mut self, world: &mut World, shown: bool) {
        if !shown {
            self.create_menu = None;
        }
        let key = (
            self.entries.revision(),
            self.create_menu.as_ref().is_some_and(|m| m.prefabs_open),
        );
        let stale = self.create_items.as_ref().is_none_or(|c| c.key != key);
        if self.create_menu.is_some() && stale {
            let items = self.create_menu_items();
            create_menu::list(world, &items);
            self.create_items = Some(CreateItems { key, items });
        }
        create_menu::show(world, self.create_menu.is_some());
    }
}
