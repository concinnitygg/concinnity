//! The settings-menu choices on disk: core's `StoredSettings` at the state
//! root's `settings`, the mutable sibling of the build-regenerated `data`.
//! Stored as CBOR, binary like the data blobs but self-describing; bincode,
//! which the data blobs use, is positional and safe only for files regenerated
//! each build. Core's SettingsSystem writes through [`FileStore`] on a
//! background thread whenever its revision moves.

use concinnity_core::settings::StoredSettings;
use concinnity_core::settings::system::SettingsStore;
use concinnity_host::store::paths::StateTree;
use concinnity_host::store::writer::FileWriter;
use std::path::{Path, PathBuf};

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

// Load from `tree`'s `settings` file. Returns defaults when nothing is stored or
// the file is unreadable.
pub(crate) fn load(tree: Option<&StateTree>) -> StoredSettings {
    tree.map(|tree| load_from(&tree.settings_path()))
        .unwrap_or_default()
}

// Persist to `tree`'s `settings` file, creating the directory as needed. An app
// with no state tree has nowhere to persist to, which is not an error: the
// choices apply for the rest of the run.
pub(crate) fn save(settings: &StoredSettings, tree: Option<&StateTree>) -> std::io::Result<()> {
    match tree {
        Some(tree) => concinnity_host::store::cbor::write(&tree.settings_path(), settings),
        None => Ok(()),
    }
}

// Read settings from `path`. Split from `load` so the read path can be tested
// against a sandbox file, never the developer's real one. No file yet, or a
// truncated / incompatible one, starts from defaults.
fn load_from(path: &Path) -> StoredSettings {
    concinnity_host::store::cbor::read(path, "settings store").unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::UpscaleQuality;
    use concinnity_core::ecs::{Phase, PipelineContext, StepResult, System, SystemTable, World};
    use concinnity_core::settings::UserGraphicsSettings;
    use concinnity_core::settings::system::SettingsSystem;

    // The on-disk path must stay sandboxable so a test can never clobber the
    // developer's real `settings` file. A non-default `render_scale` mirrors a
    // real persisted choice and proves a populated field survives the trip.
    #[test]
    fn settings_save_load_roundtrip_is_sandboxed() {
        let s = StoredSettings {
            graphics: UserGraphicsSettings {
                render_scale: Some(UpscaleQuality::Performance),
                vsync: Some(true),
                exposure_ev: Some(-1.5),
                ..Default::default()
            },
            ..Default::default()
        };

        let dir = concinnity_testing::TempTree::new();
        let tree = StateTree::at(dir.path());

        save(&s, Some(&tree)).unwrap();
        assert!(tree.settings_path().exists());
        assert_eq!(load_from(&tree.settings_path()), s);
    }

    // Settings load and save against the tree they are handed, and an app with
    // no tree loads defaults and persists nothing rather than guessing a file.
    #[test]
    fn settings_follow_the_tree_they_are_given_and_no_tree_writes_nothing() {
        let dir = concinnity_testing::TempTree::new();
        let tree = StateTree::at(dir.path());

        let stored = StoredSettings {
            graphics: UserGraphicsSettings {
                vsync: Some(true),
                ..Default::default()
            },
            ..Default::default()
        };
        save(&stored, Some(&tree)).unwrap();
        assert!(tree.settings_path().exists());
        assert_eq!(load(Some(&tree)), stored);

        assert_eq!(load(None), StoredSettings::default());
        assert!(save(&stored, None).is_ok());
    }

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
        assert_eq!(load(Some(&tree)), stored);
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
        assert_eq!(load(Some(&tree)).graphics.vsync, Some(false));
    }
}
