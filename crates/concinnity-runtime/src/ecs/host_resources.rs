// The resources core systems read that only this host can produce: what the
// OS reports, what the build compiled in, and what the settings file holds.
// Published at world start, before any system's init reads them. A value an
// embedder already inserted is kept.

use concinnity_core::budget::ResidentMemory;
use concinnity_core::ecs::{Clock, PipelineContext, ScheduleMode};
use concinnity_core::platform::CompiledPlatforms;
use concinnity_core::settings::{SettingsRevision, StoredSettings};
use concinnity_host::store::paths::StateTree;

pub(crate) fn publish(ctx: &mut PipelineContext) {
    insert_absent(ctx, || Clock(concinnity_host::clock::monotonic_micros));
    // Systems fan out on the job pool unless the launch chose a serial schedule.
    insert_absent(ctx, || ScheduleMode::Parallel);
    insert_absent(ctx, || {
        ResidentMemory(concinnity_host::sys::memory::process_resident_bytes)
    });
    insert_absent(ctx, || CompiledPlatforms(crate::platform::compiled()));
    insert_absent(ctx, crate::platform::api_status);
    if ctx.resource::<StoredSettings>().is_none() {
        let stored = crate::settings::store::load(ctx.resource::<StateTree>());
        ctx.insert_resource(stored);
    }
    insert_absent(ctx, SettingsRevision::default);
}

fn insert_absent<T: core::any::Any + Send>(ctx: &mut PipelineContext, value: impl FnOnce() -> T) {
    if ctx.resource::<T>().is_none() {
        ctx.insert_resource(value());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::ecs::World;
    use concinnity_core::platform::GraphicsApiStatus;

    #[test]
    fn a_world_without_a_state_tree_stores_nothing() {
        let mut world = World::new();
        publish(&mut world.context());
        assert_eq!(
            world.resource::<StoredSettings>(),
            Some(&StoredSettings::default())
        );
        assert_eq!(
            world.resource::<CompiledPlatforms>().map(|p| p.0),
            Some(crate::platform::compiled())
        );
        assert!(world.resource::<ResidentMemory>().is_some());
        assert!(world.resource::<Clock>().is_some());
        assert_eq!(
            world.resource::<ScheduleMode>(),
            Some(&ScheduleMode::Parallel)
        );
        assert_eq!(
            world.resource::<GraphicsApiStatus>().map(|s| s.current),
            Some(crate::platform::current())
        );
        assert_eq!(
            world.resource::<SettingsRevision>(),
            Some(&SettingsRevision::default())
        );
    }

    // An embedder's own values survive the publish.
    #[test]
    fn values_already_present_are_kept() {
        fn frozen() -> u64 {
            9
        }
        fn none() -> Option<u64> {
            None
        }
        let mut chosen = StoredSettings::default();
        chosen.graphics.fov = Some(60.0);
        let mut world = World::new();
        world.insert_resource(Clock(frozen));
        world.insert_resource(ResidentMemory(none));
        world.insert_resource(CompiledPlatforms(&[]));
        world.insert_resource(chosen.clone());
        world.insert_resource(ScheduleMode::Serial);
        publish(&mut world.context());
        assert_eq!(
            world.resource::<ScheduleMode>(),
            Some(&ScheduleMode::Serial)
        );
        assert_eq!(world.resource::<Clock>().map(Clock::read), Some(9));
        assert_eq!(
            world
                .resource::<ResidentMemory>()
                .and_then(ResidentMemory::read),
            None
        );
        assert_eq!(
            world.resource::<CompiledPlatforms>().map(|p| p.0.len()),
            Some(0)
        );
        assert_eq!(world.resource::<StoredSettings>(), Some(&chosen));
    }
}
