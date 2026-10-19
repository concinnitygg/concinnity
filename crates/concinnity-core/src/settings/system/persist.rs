// Writing the settings store back as it changes.

use super::SettingsSystem;
use crate::ecs::PipelineContext;
use crate::settings::{SettingsRevision, StoredSettings};

/// Where a host keeps the settings store. A host with none applies changes for
/// the run without persisting them.
pub trait SettingsStore: core::fmt::Debug + Send {
    /// Replace what is kept with `stored`. Reporting a failed write is the
    /// implementor's job.
    fn write(&self, stored: &StoredSettings);
}

impl SettingsSystem {
    // Write the store when its revision moved since the last write. A revision
    // seen before the store is published waits for it.
    pub(super) fn persist(&mut self, ctx: &PipelineContext) {
        let Some(&revision) = ctx.resource::<SettingsRevision>() else {
            return;
        };
        if revision == self.written {
            return;
        }
        let Some(stored) = ctx.resource::<StoredSettings>() else {
            return;
        };
        if let Some(store) = &self.store {
            store.write(stored);
        }
        self.written = revision;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::{StepResult, System, World};
    use alloc::boxed::Box;
    use alloc::sync::Arc;
    use alloc::vec::Vec;

    #[derive(Debug, Default, Clone)]
    struct Recorded(Arc<spin::Mutex<Vec<StoredSettings>>>);

    impl SettingsStore for Recorded {
        fn write(&self, stored: &StoredSettings) {
            self.0.lock().push(stored.clone());
        }
    }

    fn system() -> (SettingsSystem, Recorded) {
        let store = Recorded::default();
        (
            SettingsSystem::new().with_store(Box::new(store.clone())),
            store,
        )
    }

    // Each new revision writes the store once; an unchanged one writes nothing.
    #[test]
    fn a_new_revision_writes_the_store_once() {
        let (mut system, store) = system();
        let mut world = World::new();
        let mut stored = StoredSettings::default();
        stored.graphics.vsync = Some(true);
        world.insert_resource(stored.clone());
        world.insert_resource(SettingsRevision::default());
        system.step(&mut world.context());
        assert!(store.0.lock().is_empty(), "revision zero is unchanged");

        world.resource_mut::<SettingsRevision>().unwrap().bump();
        assert_eq!(system.step(&mut world.context()), StepResult::Continue);
        system.step(&mut world.context());
        assert_eq!(*store.0.lock(), [stored]);
    }

    // A revision seen before the store is published is written once the store
    // appears, not skipped.
    #[test]
    fn a_revision_waits_for_its_store() {
        let (mut system, store) = system();
        let mut world = World::new();
        world.insert_resource(SettingsRevision(1));
        system.step(&mut world.context());
        let mut stored = StoredSettings::default();
        stored.graphics.vsync = Some(false);
        world.insert_resource(stored.clone());
        system.step(&mut world.context());
        assert_eq!(*store.0.lock(), [stored]);
    }
}
