// src/editor/hook/select_edit.rs
//
// EditorHook: the /select console command. Resolution is pure
// (`editor/select_related.rs`); this dispatch feeds it the working entries
// (or, for origin, a fresh cook so the grouping matches the outliner) and
// replaces the selection with what comes back.

use super::super::select_related;
use super::*;

impl EditorHook {
    pub(super) fn console_select(&mut self, cmd: console::SelectCmd, world: &mut World) {
        let names = match cmd {
            console::SelectCmd::Origin => {
                let Some(active) = self.selection.active().map(String::from) else {
                    self.console_sink.error("nothing selected");
                    return;
                };
                match self.cook_working_entries() {
                    Ok(loaded) => {
                        let groups = asset_tree::groups_from(&loaded);
                        match select_related::same_group(&groups, &active) {
                            Some(names) => names,
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
                let names = select_related::names_using(&self.entries, &target);
                if names.is_empty() {
                    self.console_sink
                        .info(&format!("nothing references {target}"));
                    return;
                }
                names
            }
            console::SelectCmd::Type(ty) => {
                let names = select_related::names_of_type(&self.entries, &ty);
                if names.is_empty() {
                    self.console_sink.info(&format!("no assets of type {ty}"));
                    return;
                }
                names
            }
        };
        let n = names.len();
        self.selection.set(names);
        self.follow_active(world);
        self.console_sink.info(&format!("selected {n}"));
    }
}
