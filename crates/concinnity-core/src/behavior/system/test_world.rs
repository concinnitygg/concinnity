// The hand-assembled world the unit tests and the benchmarks drive: a
// component storage plus the four other pieces a `PipelineContext` borrows.
//
// Building the context directly is what lets a test hand the tick an explicit
// dt, which is what makes timers, delays, and cooldowns deterministic.

use alloc::vec::Vec;

use crate::components::Behavior;
use crate::ecs::{Arena, ComponentStorage, FrameContext, NoPayloads, PipelineContext, Resources};
use crate::gfx::profile::FrameProfile;

pub(super) struct TestWorld {
    pub(super) components: ComponentStorage,
    blob: NoPayloads,
    profile: FrameProfile,
    pub(super) resources: Resources,
    scratch: Arena,
    // Simulated time accumulated by the tests' `tick` helper.
    pub(super) elapsed: f32,
}

impl TestWorld {
    pub(super) fn ctx(&mut self) -> PipelineContext<'_> {
        PipelineContext {
            components: &mut self.components,
            blob: &mut self.blob,
            profile: &mut self.profile,
            resources: &mut self.resources,
            frame: FrameContext::new(&self.scratch),
        }
    }
}

pub(super) fn world_with(behaviors: Vec<Behavior>) -> TestWorld {
    let mut world = TestWorld {
        components: ComponentStorage::default(),
        blob: NoPayloads,
        profile: FrameProfile::default(),
        resources: Resources::default(),
        scratch: Arena::with_capacity(64 * 1024),
        elapsed: 0.0,
    };
    for b in behaviors {
        world.components.push_typed(b);
    }
    world
}
