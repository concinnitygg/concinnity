// Persisting the settings-menu choices: core's SettingsSystem edits the
// `StoredSettings` resource and bumps `SettingsRevision`; this system writes a
// snapshot to disk on a background thread whenever the revision moves.

pub(crate) mod store;
pub(crate) mod writer;

use concinnity_core::ecs::{PipelineContext, StepResult, System};
use concinnity_core::settings::{SettingsRevision, StoredSettings};
use concinnity_host::store::paths::StateTree;

use writer::SettingsWriter;

#[derive(Default)]
pub(crate) struct SettingsWriterSystem {
    // The revision last written; a world starts at zero, unchanged.
    written: SettingsRevision,
    // Spawned on the first change, so an unchanged session never starts the
    // thread.
    writer: Option<SettingsWriter>,
}

impl std::fmt::Debug for SettingsWriterSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SettingsWriterSystem")
            .field("written", &self.written)
            .finish_non_exhaustive()
    }
}

impl SettingsWriterSystem {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    fn with_writer(writer: SettingsWriter) -> Self {
        Self {
            writer: Some(writer),
            ..Self::default()
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
        let Some(stored) = ctx.resource::<StoredSettings>().cloned() else {
            return StepResult::Continue;
        };
        let tree = ctx.resource::<StateTree>().cloned();
        self.writer
            .get_or_insert_with(|| SettingsWriter::spawn(tree))
            .save(stored);
        self.written = revision;
        StepResult::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::ecs::World;
    use std::sync::{Arc, Mutex};

    // Each new revision writes the store once; an unchanged one writes nothing.
    #[test]
    fn a_new_revision_writes_the_store_once() {
        let written: Arc<Mutex<Vec<StoredSettings>>> = Arc::default();
        let log = Arc::clone(&written);
        let mut system = SettingsWriterSystem::with_writer(SettingsWriter::with_sink(move |s| {
            log.lock().unwrap().push(s.clone());
            Ok(())
        }));
        let mut world = World::new();
        let mut stored = StoredSettings::default();
        stored.graphics.vsync = Some(true);
        world.insert_resource(stored.clone());
        world.insert_resource(SettingsRevision::default());
        system.step(&mut world.context());

        world.resource_mut::<SettingsRevision>().unwrap().bump();
        system.step(&mut world.context());
        system.step(&mut world.context());
        drop(system);
        assert_eq!(*written.lock().unwrap(), [stored]);
    }

    // A revision seen before the store is published is written once the
    // store appears, not skipped.
    #[test]
    fn a_revision_waits_for_its_store() {
        let written: Arc<Mutex<Vec<StoredSettings>>> = Arc::default();
        let log = Arc::clone(&written);
        let mut system = SettingsWriterSystem::with_writer(SettingsWriter::with_sink(move |s| {
            log.lock().unwrap().push(s.clone());
            Ok(())
        }));
        let mut world = World::new();
        world.insert_resource(SettingsRevision(1));
        system.step(&mut world.context());
        world.insert_resource(StoredSettings::default());
        system.step(&mut world.context());
        drop(system);
        assert_eq!(*written.lock().unwrap(), [StoredSettings::default()]);
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

        let written: Arc<Mutex<Vec<StoredSettings>>> = Arc::default();
        let log = Arc::clone(&written);
        let writer = SettingsWriterSystem::with_writer(SettingsWriter::with_sink(move |s| {
            log.lock().unwrap().push(s.clone());
            Ok(())
        }));
        let mut world = World::new();
        world.add_system(Phase::PreRender, "EditThenStop", EditThenStop);
        world.add_system(Phase::PreRender, "SettingsWriterSystem", writer);
        world.add_system(Phase::PreRender, "Stop", Stop);
        world
            .start(&concinnity_core::ecs::SystemTable::EMPTY)
            .unwrap();
        assert_eq!(world.step(), StepResult::Stop);
        drop(world);
        let saved = written.lock().unwrap();
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0].graphics.vsync, Some(false));
    }
}
