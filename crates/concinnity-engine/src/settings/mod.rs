// Persisting the settings-menu choices: core's SettingsSystem edits the
// `StoredSettings` resource and bumps `SettingsRevision`; this system writes a
// snapshot to disk on a background thread whenever the revision moves.

pub(crate) mod store;

use concinnity_core::ecs::{PipelineContext, StepResult, System};
use concinnity_core::settings::{SettingsRevision, StoredSettings};
use concinnity_host::store::paths::StateTree;

use crate::file_writer::FileWriter;

#[derive(Debug)]
pub(crate) struct SettingsWriterSystem {
    // The revision last written; a world starts at zero, unchanged.
    written: SettingsRevision,
    // Its thread starts on the first change, so an unchanged session never
    // starts one. A burst of changes coalesces to the newest.
    writer: FileWriter,
}

impl SettingsWriterSystem {
    pub(crate) fn new() -> Self {
        Self {
            written: SettingsRevision::default(),
            writer: FileWriter::new("cn-settings-writer"),
        }
    }
}

impl System for SettingsWriterSystem {
    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        let Some(&revision) = ctx.resource::<SettingsRevision>() else {
            return StepResult::Continue;
        };
        if revision == self.written {
            return StepResult::Continue;
        }
        let Some(stored) = ctx.resource::<StoredSettings>() else {
            return StepResult::Continue;
        };
        // An app with no state tree has nowhere to persist to; the choices
        // apply for the rest of the run.
        if let Some(tree) = ctx.resource::<StateTree>() {
            self.writer
                .write_cbor(tree.settings_path(), stored, "settings");
        }
        self.written = revision;
        StepResult::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::ecs::World;

    fn world_with_store(dir: &concinnity_testing::TempTree) -> (World, StateTree) {
        let tree = StateTree::at(dir.path());
        let mut world = World::new();
        world.insert_resource(tree.clone());
        (world, tree)
    }

    // Each new revision writes the store; an unchanged one writes nothing.
    #[test]
    fn a_new_revision_writes_the_store() {
        let dir = concinnity_testing::TempTree::new();
        let (mut world, tree) = world_with_store(&dir);
        let mut system = SettingsWriterSystem::new();
        let mut stored = StoredSettings::default();
        stored.graphics.vsync = Some(true);
        world.insert_resource(stored.clone());
        world.insert_resource(SettingsRevision::default());
        system.step(&mut world.context());
        assert!(!tree.settings_path().exists(), "revision zero is unchanged");

        world.resource_mut::<SettingsRevision>().unwrap().bump();
        system.step(&mut world.context());
        system.step(&mut world.context());
        drop(system);
        assert_eq!(store::load(Some(&tree)), stored);
    }

    // A revision seen before the store is published is written once the
    // store appears, not skipped.
    #[test]
    fn a_revision_waits_for_its_store() {
        let dir = concinnity_testing::TempTree::new();
        let (mut world, tree) = world_with_store(&dir);
        let mut system = SettingsWriterSystem::new();
        world.insert_resource(SettingsRevision(1));
        system.step(&mut world.context());
        let mut stored = StoredSettings::default();
        stored.graphics.vsync = Some(false);
        world.insert_resource(stored.clone());
        system.step(&mut world.context());
        drop(system);
        assert_eq!(store::load(Some(&tree)), stored);
    }

    // A change made on a tick that stops later is still written: the writer
    // runs right after the settings drain, before the systems that can stop
    // the run.
    #[test]
    fn a_change_on_a_stopping_tick_is_written() {
        use concinnity_core::ecs::{Phase, PipelineContext, StepResult, System};

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
        let (mut world, tree) = world_with_store(&dir);
        world.add_system(Phase::PreRender, "EditThenStop", EditThenStop);
        world.add_system(
            Phase::PreRender,
            "SettingsWriterSystem",
            SettingsWriterSystem::new(),
        );
        world.add_system(Phase::PreRender, "Stop", Stop);
        world
            .start(&concinnity_core::ecs::SystemTable::EMPTY)
            .unwrap();
        assert_eq!(world.step(), StepResult::Stop);
        drop(world);
        assert_eq!(store::load(Some(&tree)).graphics.vsync, Some(false));
    }
}
