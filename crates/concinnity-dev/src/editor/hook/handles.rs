//! EditorHook: resolving an `AssetHandle` against the world as it stands this
//! frame. Everything the editor addresses -- the selection, the open form, the
//! gizmo, the panels that follow the active member -- holds handles, and passes
//! through here to reach an authored entry, a live `AssetId`, or the name a row
//! is drawn under.
//!
//! The name the world knows an asset by is its handle: the `$id` it declares,
//! or for an anonymous entry the `<Type>#<ordinal>` label the build records
//! for it. This is the one place that reads a handle as identity, which is
//! what makes the interner's per-build reset harmless: a handle that no longer
//! resolves yields `None` for the frame rather than addressing whatever asset
//! took its place.

use concinnity_cook::authoring::world::{entry_handle, find_entry};
use concinnity_core::ecs::World;
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_host::thread::asset_id;

use super::EditorHook;
use crate::editor::asset_handle::AssetHandle;
use crate::editor::entry_list::EntryId;
use crate::editor::selection::SelectedNames;

impl EditorHook {
    // The handle for the asset the world knows as `name`: its authored entry
    // when the world declares one, else the generated asset of that name.
    pub(in crate::editor) fn handle_for(&self, name: &str) -> AssetHandle {
        match self.entry_key_named(name) {
            Some(key) => AssetHandle::Entry(key),
            None => AssetHandle::Generated(name.to_string()),
        }
    }

    // The session key of the authored entry the world knows as `name`: the
    // one declaring it as its `$id`, or the anonymous entry it labels.
    pub(in crate::editor) fn entry_key_named(&self, name: &str) -> Option<EntryId> {
        self.entries.key_at(find_entry(&self.entries, name)?)
    }

    // The name the world knows a handle's asset by, or `None` once the handle
    // addresses nothing (its entry was deleted, its expansion stopped).
    pub(in crate::editor) fn handle_name(&self, handle: &AssetHandle) -> Option<String> {
        match handle {
            AssetHandle::Entry(key) => entry_handle(&self.entries, self.entries.index_of(*key)?),
            AssetHandle::Generated(name) => Some(name.clone()),
        }
    }

    // The authored entry a handle addresses, as an index into the working
    // list. `None` for a generated asset, which has no line of its own.
    pub(in crate::editor) fn handle_index(&self, handle: &AssetHandle) -> Option<usize> {
        self.entries.index_of(handle.entry()?)
    }

    // The dense id this build gave a handle's asset, for the world-side joins
    // (the pick index, `EntityById`). The build records every asset under
    // its handle, an anonymous one under its label, so one lookup serves both.
    pub(in crate::editor) fn handle_asset_id(&self, handle: &AssetHandle) -> Option<AssetId> {
        asset_id::lookup(&self.handle_name(handle)?)
    }

    // The handle behind a picked `AssetId`, or `None` when the id resolves to
    // no name this build interned.
    pub(in crate::editor) fn handle_of_asset_id(&self, id: AssetId) -> Option<AssetHandle> {
        Some(self.handle_for(&asset_id::name_of(id)?))
    }

    // A copy of the selection, for the drives that walk it while mutating the
    // entries it addresses.
    pub(in crate::editor) fn selected(&self) -> Vec<AssetHandle> {
        self.selection.iter().cloned().collect()
    }

    // Replace the selection with the asset the world knows as `name`.
    pub(in crate::editor) fn select_named(&mut self, name: &str) {
        let handle = self.handle_for(name);
        self.selection.replace(handle);
    }

    // Select the asset a chart card stands for, the way that asset is selected
    // anywhere else: plain replaces the selection and opens its editing
    // surface, shift toggles membership, and the Assets tree reveals the row
    // either way. A handle addressing nothing this frame selects nothing.
    pub(in crate::editor::hook) fn select_handle(
        &mut self,
        handle: &AssetHandle,
        world: &mut World,
    ) {
        let Some(name) = self.handle_name(handle) else {
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
    }

    // Toggle that asset's membership, returning whether it is selected after.
    pub(in crate::editor) fn toggle_named(&mut self, name: &str) -> bool {
        let handle = self.handle_for(name);
        self.selection.toggle(handle)
    }

    // The selection resolved to this frame's names, for the row and icon
    // draws. Members the current build does not produce drop out.
    pub(in crate::editor) fn selected_names(&self) -> SelectedNames {
        let names = self
            .selection
            .iter()
            .filter_map(|h| self.handle_name(h))
            .collect();
        let active = self.selection.active().and_then(|h| self.handle_name(h));
        SelectedNames::new(names, active)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn hook_with(entries: Vec<serde_json::Value>) -> EditorHook {
        EditorHook::new("unused.jsonl".to_string(), entries)
    }

    fn entries() -> Vec<serde_json::Value> {
        vec![
            json!({"type": "Prop", "args": {"$id": "floor"}}),
            json!({"type": "PointLight", "args": {"$id": "lamp"}}),
        ]
    }

    #[test]
    fn an_authored_name_resolves_to_its_entry_and_back() {
        let h = hook_with(entries());
        let handle = h.handle_for("lamp");
        assert_eq!(h.handle_index(&handle), Some(1));
        assert_eq!(h.handle_name(&handle).as_deref(), Some("lamp"));
    }

    #[test]
    fn a_name_no_entry_declares_resolves_to_a_generated_handle() {
        let h = hook_with(entries());
        let handle = h.handle_for("bistro_prop_7");
        assert_eq!(handle, AssetHandle::Generated("bistro_prop_7".into()));
        assert_eq!(h.handle_index(&handle), None, "no authored line to edit");
        assert_eq!(h.handle_name(&handle).as_deref(), Some("bistro_prop_7"));
    }

    #[test]
    fn an_entry_handle_follows_its_entry_through_a_rename() {
        let mut h = hook_with(entries());
        let handle = h.handle_for("lamp");
        h.entries[1]["args"]["$id"] = json!("lantern");
        assert_eq!(h.handle_name(&handle).as_deref(), Some("lantern"));
        assert_eq!(h.handle_index(&handle), Some(1));
    }

    #[test]
    fn an_entry_handle_follows_its_entry_past_a_removal_above_it() {
        let mut h = hook_with(entries());
        let handle = h.handle_for("lamp");
        h.entries.remove(0);
        assert_eq!(h.handle_index(&handle), Some(0));
        assert_eq!(h.handle_name(&handle).as_deref(), Some("lamp"));
    }

    #[test]
    fn a_deleted_entrys_handle_resolves_to_nothing() {
        let mut h = hook_with(entries());
        let handle = h.handle_for("lamp");
        h.entries.remove(1);
        assert_eq!(h.handle_name(&handle), None);
        assert_eq!(h.handle_index(&handle), None);
        // A fresh entry reusing the name is a different asset, not this one.
        h.entries
            .push(json!({"type": "PointLight", "args": {"$id": "lamp"}}));
        assert_eq!(h.handle_index(&handle), None);
        assert_eq!(h.handle_index(&h.handle_for("lamp")), Some(1));
    }

    // An anonymous entry is known by its label, which follows its position
    // among the anonymous entries of its type.
    #[test]
    fn an_anonymous_entry_resolves_through_its_label() {
        let mut h = hook_with(vec![
            json!({"type": "Prop", "args": {}}),
            json!({"type": "Prop", "args": {"$id": "named"}}),
            json!({"type": "Prop", "args": {}}),
        ]);
        let second = h.handle_for("Prop#1");
        assert_eq!(h.handle_index(&second), Some(2));
        assert_eq!(h.handle_name(&second).as_deref(), Some("Prop#1"));
        // Removing the first anonymous Prop relabels the second.
        h.entries.remove(0);
        assert_eq!(h.handle_index(&second), Some(1));
        assert_eq!(h.handle_name(&second).as_deref(), Some("Prop#0"));
    }

    // The build records an anonymous asset under its label, so the entry's
    // live id is one lookup away like a declared one's.
    #[test]
    fn an_anonymous_entry_reaches_the_id_its_build_recorded() {
        let h = hook_with(vec![
            json!({"type": "Prop", "args": {"$id": "floor"}}),
            json!({"type": "Prop", "args": {}}),
        ]);
        asset_id::reset_interner();
        asset_id::prime_name_table(&[(0, "floor".to_string()), (1, "Prop#0".to_string())]);
        let handle = h.handle_for("Prop#0");
        assert_eq!(h.handle_asset_id(&handle), Some(AssetId(1)));
        assert_eq!(h.handle_of_asset_id(AssetId(1)), Some(handle));
    }

    #[test]
    fn selected_names_drops_the_members_the_build_no_longer_produces() {
        let mut h = hook_with(entries());
        let floor = h.handle_for("floor");
        let lamp = h.handle_for("lamp");
        h.selection.set(vec![floor, lamp]);
        h.entries.remove(0);

        let names = h.selected_names();
        assert!(!names.contains("floor"), "its entry is gone");
        assert!(names.contains("lamp"));
        assert!(names.is_active("lamp"));
    }
}
