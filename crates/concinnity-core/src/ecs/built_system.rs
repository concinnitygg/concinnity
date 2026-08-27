// src/ecs/built_system.rs
//
// What a world holds once a table gate has run: the constructed system behind a
// `dyn System` pointer, paired with the name of the table entry that built it.
// A trait object carries no name of its own, and the name is what the profile,
// the log, and the schedule's ordering edges all key on.

use alloc::boxed::Box;

use crate::ecs::{Access, PipelineContext, StepResult, System};

/// A constructed system and the table entry name it was built from.
#[derive(Debug)]
pub struct BuiltSystem {
    name: &'static str,
    system: Box<dyn System>,
}

impl BuiltSystem {
    // Pair a gate's system with its table entry's name.
    pub(crate) fn new(name: &'static str, system: Box<dyn System>) -> Self {
        Self { name, system }
    }

    /// Stable display name used for profiling and logging: the system's entry
    /// name in the table, which is also what its ordering edges name.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Run the system's `init`.
    pub fn init(&mut self, ctx: &mut PipelineContext) {
        self.system.init(ctx);
    }

    /// Run the system's `step`.
    pub fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        self.system.step(ctx)
    }

    /// The system's declared data access, consulted at schedule build (after
    /// init). Defaults to exclusive via the `System` trait.
    pub fn access(&self) -> Access {
        self.system.access()
    }

    /// Borrow the system as `S`, or `None` when it is a different system.
    pub fn downcast_ref<S: System>(&self) -> Option<&S> {
        (&*self.system as &dyn core::any::Any).downcast_ref::<S>()
    }

    /// Mutably borrow the system as `S`, or `None` when it is a different
    /// system. The `DebugHook::tick` drive reaches the GraphicsSystem's
    /// hot-reload bookkeeping and the AnimationSystem's clip table through
    /// this, from outside the per-system step.
    pub fn downcast_mut<S: System>(&mut self) -> Option<&mut S> {
        (&mut *self.system as &mut dyn core::any::Any).downcast_mut::<S>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Tick(u32);

    impl System for Tick {
        fn step(&mut self, _ctx: &mut PipelineContext) -> StepResult {
            self.0 += 1;
            StepResult::Continue
        }
    }

    #[derive(Debug)]
    struct Other;

    impl System for Other {
        fn step(&mut self, _ctx: &mut PipelineContext) -> StepResult {
            StepResult::Continue
        }
    }

    // The pair carries the table name a trait object cannot.
    #[test]
    fn keeps_the_table_name() {
        let built = BuiltSystem::new("Tick", Box::new(Tick(0)));
        assert_eq!(built.name(), "Tick");
    }

    // Stepping runs the boxed system's own body.
    #[test]
    fn steps_the_boxed_system() {
        let mut world = crate::ecs::World::new();
        let mut ctx = world.context();
        let mut built = BuiltSystem::new("Tick", Box::new(Tick(0)));
        assert_eq!(built.step(&mut ctx), StepResult::Continue);
        assert_eq!(built.step(&mut ctx), StepResult::Continue);
        assert_eq!(built.downcast_ref::<Tick>().expect("a Tick").0, 2);
    }

    // A downcast to the wrong system yields nothing rather than the wrong body.
    #[test]
    fn downcast_answers_only_for_its_own_type() {
        let mut built = BuiltSystem::new("Tick", Box::new(Tick(7)));
        assert!(built.downcast_ref::<Other>().is_none());
        assert!(built.downcast_mut::<Other>().is_none());
        built.downcast_mut::<Tick>().expect("a Tick").0 = 9;
        assert_eq!(built.downcast_ref::<Tick>().expect("a Tick").0, 9);
    }
}
