// src/editor/hook/create_menu_drive.rs
//
// EditorHook: the right-click "Create here" menu flow. An unclaimed viewport
// right press opens the menu at the cursor, capturing the world position under
// it (surface hit, ground-plane fallback -- the drag-out landing rules) at
// open time; picking a row creates one entry at that point through the shared
// entries + mark_changed path (one undo step) and selects it; any other press,
// Escape, or a history jump dismisses it.

use super::*;

// An open "Create here" menu.
pub(super) struct CreateMenu {
    // Top-left, clamped on screen at open; the menu does not track the cursor.
    pub(super) origin: [f32; 2],
    // The world position captured from under the right-click.
    pub(super) pos: [f32; 3],
    // Whether the Prefab section is unfolded.
    pub(super) prefabs_open: bool,
}

impl EditorHook {
    // The menu rows for the current entry list and fold state.
    pub(super) fn create_menu_items(&self) -> Vec<create_menu::MenuItem> {
        let prefabs = names_of_type(&self.entries, "Prefab");
        let open = self.create_menu.as_ref().is_some_and(|m| m.prefabs_open);
        create_menu::items(&prefabs, open)
    }

    // Open (or move) the menu at a right press, unless the press belongs to
    // the top bar or a floating panel, or the world offers no landing point
    // (no camera).
    pub(super) fn open_create_menu(&mut self, input: &FrameInput, vp: [f32; 2], world: &World) {
        let (mx, my) = (input.mouse_x, input.mouse_y);
        if self.sim.playing() || my <= hud::BAR_H || self.over_open_panel(mx, my, vp) {
            self.create_menu = None;
            return;
        }
        let Some(pos) = self.drop_point(world, vp, [mx, my], input.ctrl) else {
            return;
        };
        let rows = create_menu::items(&names_of_type(&self.entries, "Prefab"), false).len();
        if rows == 0 {
            return;
        }
        self.create_menu = Some(CreateMenu {
            origin: create_menu::origin([mx, my], rows, vp),
            pos,
            prefabs_open: false,
        });
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

    // Resolve a left press while the menu is open: a row acts, a press on the
    // menu's chrome is swallowed, anything else dismisses. Returns whether the
    // menu was open (the press is consumed either way -- a click-away only
    // dismisses, like the panel overlays).
    pub(super) fn route_create_menu_click(&mut self, input: &FrameInput, vp: [f32; 2]) -> bool {
        let Some(menu) = &self.create_menu else {
            return false;
        };
        let (o, pos) = (menu.origin, menu.pos);
        let items = self.create_menu_items();
        let (mx, my) = (input.mouse_x, input.mouse_y);
        match create_menu::hit_row(mx, my, o, items.len()) {
            Some(row) => match &items[row] {
                create_menu::MenuItem::Type(ty) => {
                    self.create_type_at(ty, pos);
                    self.create_menu = None;
                }
                create_menu::MenuItem::PrefabHeader { open } => {
                    let unfold = !open;
                    if let Some(m) = &mut self.create_menu {
                        m.prefabs_open = unfold;
                    }
                    // The unfolded list is taller; keep it fully on screen.
                    let rows = self.create_menu_items().len();
                    if let Some(m) = &mut self.create_menu {
                        m.origin = create_menu::origin(m.origin, rows, vp);
                    }
                }
                create_menu::MenuItem::Prefab(name) => {
                    self.create_prefab_at(&name.clone(), pos);
                    self.create_menu = None;
                }
            },
            None if create_menu::over(mx, my, o, items.len()) => {}
            None => self.create_menu = None,
        }
        true
    }

    // Create one entry of `ty` at `pos`: default args with the position
    // overridden, a unique generated name, one undo step, selected.
    fn create_type_at(&mut self, ty: &str, pos: [f32; 3]) {
        let name = self.unique_name(ty);
        let key = create_menu::position_key(ty);
        let args = serde_json::json!({ key: pos.map(gizmo_drag::round3) });
        self.entries.push(serde_json::json!({
            "name": name, "type": ty, "args": args
        }));
        self.mark_changed();
        self.selection.replace(name);
    }

    // Create a Prop instancing `prefab` at `pos` (the drag-out mapping).
    fn create_prefab_at(&mut self, prefab: &str, pos: [f32; 3]) {
        let Some(args) =
            content_drag::placement_args("Prefab", prefab, pos.map(gizmo_drag::round3))
        else {
            return;
        };
        let name = self.unique_from(prefab);
        self.entries.push(serde_json::json!({
            "name": name, "type": "Prop", "args": args
        }));
        self.mark_changed();
        self.selection.replace(name);
    }

    // Lay out (or hide) the menu this frame.
    pub(super) fn drive_create_menu_draw(&self, world: &mut World, shown: bool, mouse: [f32; 2]) {
        match (&self.create_menu, shown) {
            (Some(menu), true) => {
                let items = self.create_menu_items();
                create_menu::apply(world, menu.origin, &items, mouse);
            }
            _ => create_menu::hide(world),
        }
    }
}
