//! EditorHook: the /select console command. Resolution is pure
//! (`editor/select_related.rs`); this dispatch feeds it the working entries
//! (or, for origin, a fresh cook so the grouping matches the outliner) and
//! replaces the selection with what comes back.

use concinnity_core::ecs::World;

use crate::editor::asset_handle::AssetHandle;
use crate::editor::hook::EditorHook;
use crate::editor::panels::asset_tree;
use crate::editor::panels::console;
use crate::editor::select_related;

impl EditorHook {
    pub(super) fn console_select(&mut self, cmd: console::SelectCmd, world: &mut World) {
        let handles = match cmd {
            console::SelectCmd::Origin => {
                let Some(active) = self.selection.active().and_then(|h| self.handle_name(h)) else {
                    self.console_sink.error("nothing selected");
                    return;
                };
                match self.cook_working_entries() {
                    Ok(loaded) => {
                        let groups = asset_tree::groups_from(&loaded);
                        match select_related::same_group(&groups, &active) {
                            Some(names) => {
                                names.iter().map(|n| self.handle_for(n)).collect::<Vec<_>>()
                            }
                            None => {
                                self.console_sink
                                    .error(&format!("no origin group lists {active}"));
                                return;
                            }
                        }
                    }
                    Err(e) => {
                        self.console_sink.error(&e);
                        return;
                    }
                }
            }
            console::SelectCmd::Using(target) => {
                let found =
                    self.entry_handles(select_related::entries_using(&self.entries, &target));
                if found.is_empty() {
                    self.console_sink
                        .info(&format!("nothing references {target}"));
                    return;
                }
                found
            }
            console::SelectCmd::Type(ty) => {
                let found = self.entry_handles(select_related::entries_of_type(&self.entries, &ty));
                if found.is_empty() {
                    self.console_sink.info(&format!("no assets of type {ty}"));
                    return;
                }
                found
            }
        };
        let n = handles.len();
        self.selection.set(handles);
        self.follow_active(world);
        self.console_sink.info(&format!("selected {n}"));
    }

    // The handles addressing a run of working-entry positions.
    fn entry_handles(&self, positions: Vec<usize>) -> Vec<AssetHandle> {
        positions
            .into_iter()
            .filter_map(|i| self.entries.key_at(i).map(AssetHandle::Entry))
            .collect()
    }
}
