// Persisting the settings-menu choices: core's SettingsSystem edits the
// `StoredSettings` resource and, when its revision moves, writes it through this
// file store on a background thread.

pub(crate) mod store;

use concinnity_core::settings::StoredSettings;
use concinnity_core::settings::system::SettingsStore;
use concinnity_host::store::paths::StateTree;
use std::path::PathBuf;

use crate::file_writer::FileWriter;

#[derive(Debug)]
pub(crate) struct FileStore {
    path: PathBuf,
    // Its thread starts on the first change, so an unchanged session never
    // starts one. A burst of changes coalesces to the newest.
    writer: FileWriter,
}

impl FileStore {
    // The store at `tree`'s settings file.
    pub(crate) fn new(tree: &StateTree) -> Self {
        Self {
            path: tree.settings_path(),
            writer: FileWriter::new("cn-settings-writer"),
        }
    }
}

impl SettingsStore for FileStore {
    fn write(&self, stored: &StoredSettings) {
        self.writer
            .write_cbor(self.path.clone(), stored, "settings");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::ecs::{Phase, PipelineContext, StepResult, System, SystemTable, World};
    use concinnity_core::settings::system::SettingsSystem;

    // A written store reads back from the settings file once the writer drains.
    #[test]
    fn a_write_reaches_the_settings_file() {
        let dir = concinnity_testing::TempTree::new();
        let tree = StateTree::at(dir.path());
        let store = FileStore::new(&tree);
        let mut stored = StoredSettings::default();
        stored.graphics.vsync = Some(true);
        store.write(&stored);
        drop(store);
        assert_eq!(store::load(Some(&tree)), stored);
    }

    // A change made on a tick that stops later is still written: the settings
    // system writes in the step that made it, before the systems that can stop
    // the run.
    #[test]
    fn a_change_on_a_stopping_tick_is_written() {
        #[derive(Debug)]
        struct EditThenStop;
        impl System for EditThenStop {
            fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
                let mut stored = StoredSettings::default();
                stored.graphics.vsync = Some(false);
                stored.store_changed(ctx.resources);
                StepResult::Continue
            }
        }
        #[derive(Debug)]
        struct Stop;
        impl System for Stop {
            fn step(&mut self, _ctx: &mut PipelineContext) -> StepResult {
                StepResult::Stop
            }
        }

        let dir = concinnity_testing::TempTree::new();
        let tree = StateTree::at(dir.path());
        let mut world = World::new();
        world.add_system(Phase::PreRender, "EditThenStop", EditThenStop);
        world.add_system(
            Phase::PreRender,
            "SettingsSystem",
            SettingsSystem::new().with_store(Box::new(FileStore::new(&tree))),
        );
        world.add_system(Phase::PreRender, "Stop", Stop);
        world.start(&SystemTable::EMPTY).unwrap();
        assert_eq!(world.step(), StepResult::Stop);
        drop(world);
        assert_eq!(store::load(Some(&tree)).graphics.vsync, Some(false));
    }
}
