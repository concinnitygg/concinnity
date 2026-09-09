//! Writing a system: code that runs every tick over a [`World`](crate::World).
//!
//! A world's own behaviour is data -- components describe what exists, and the
//! engine's systems run over them. A system is the other half: Rust that runs
//! on the same tick as the engine's own, over the same world, with the same
//! borrow of it.
//!
//! Implement [`System`] and register it with
//! [`World::add_system`](crate::World::add_system), naming the [`Phase`] it
//! runs in:
//!
//! ```
//! use concinnity::components::TextLabel;
//! use concinnity::system::{Phase, PipelineContext, StepResult, System};
//! use concinnity::World;
//!
//! #[derive(Debug)]
//! struct Ticker {
//!     ticks: u32,
//! }
//!
//! impl System for Ticker {
//!     fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
//!         self.ticks += 1;
//!         for label in ctx.query_mut::<TextLabel>() {
//!             label.content = format!("tick {}", self.ticks);
//!         }
//!         StepResult::Continue
//!     }
//! }
//!
//! let mut world = World::new();
//! world.add_component(TextLabel::default());
//! world.add_system(Phase::Late, "Ticker", Ticker { ticks: 0 });
//! ```
//!
//! # State
//!
//! A system owns whatever it needs: the value registered is the value stepped,
//! so per-system state is ordinary struct fields. State several systems share
//! belongs in a resource -- any `Send` type, published with
//! [`PipelineContext::insert_resource`] and read back by type.
//!
//! Per-entity state belongs in a component of your own. The
//! [`components`](crate::components) vocabulary is the engine's and does not
//! grow, but an application declares its own types beside it with
//! [`declare_components!`](crate::declare_components), and they behave like any
//! other component from there on:
//!
//! ```
//! use concinnity::components::Transform;
//! use concinnity::system::{PipelineContext, StepResult, System};
//! use concinnity::{World, declare_components};
//!
//! #[derive(Debug)]
//! struct Velocity([f32; 3]);
//!
//! declare_components!(Velocity);
//!
//! #[derive(Debug)]
//! struct Motion;
//!
//! impl System for Motion {
//!     fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
//!         let moving: Vec<_> = ctx
//!             .join2::<Velocity, Transform>()
//!             .map(|(entity, v, _)| (entity, v.0))
//!             .collect();
//!         for (entity, velocity) in moving {
//!             if let Some(t) = ctx.get_mut::<Transform>(entity) {
//!                 for axis in 0..3 {
//!                     t.position[axis] += velocity[axis];
//!                 }
//!             }
//!         }
//!         StepResult::Continue
//!     }
//! }
//!
//! let mut world = World::new();
//! let mover = world.spawn();
//! world.insert(mover, Velocity([1.0, 0.0, 0.0]));
//! world.insert(mover, Transform::default());
//! ```
//!
//! A declared component is runtime-only: it has no authoring name and no blob
//! record, so it cannot be written in a world file or survive a save. Seed it
//! from Rust, with [`World::spawn`] and [`World::insert`] before the run or
//! [`PipelineContext::push`] during it.
//!
//! # Declaring access
//!
//! [`System::access`] says what a step touches. The default claims everything,
//! which is always correct and always safe; declaring narrower access is what
//! turns on the debug-build check that a step touched only what it said it
//! would. Build the masks with [`component_mask!`](crate::component_mask):
//!
//! ```
//! # use concinnity::components::{TextLabel, Transform};
//! # use concinnity::component_mask;
//! # use concinnity::system::Access;
//! # fn declare() -> Access {
//! Access::new()
//!     .reads_components(component_mask![Transform])
//!     .writes_components(component_mask![TextLabel])
//! # }
//! ```

pub use concinnity_core::ecs::{
    Access, ComponentId, ComponentMask, ComponentSlot, Entity, Phase, PipelineContext, StepResult,
    System,
};

#[cfg(test)]
mod tests {
    use alloc::sync::Arc;
    use alloc::vec::Vec;
    use core::sync::atomic::{AtomicU32, Ordering};

    use super::{Phase, PipelineContext, StepResult, System};
    use crate::components::TextLabel;
    use crate::{App, World};

    // Counts its steps into shared state, then finishes. A world whose only
    // system is this one ends the run when it reports Done, so the count the
    // caller reads back is the whole run.
    #[derive(Debug)]
    struct Counting {
        steps: Arc<AtomicU32>,
        stop_after: u32,
    }

    impl System for Counting {
        fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
            let seen = self.steps.fetch_add(1, Ordering::Relaxed) + 1;
            for label in ctx.query_mut::<TextLabel>() {
                label.content = alloc::format!("step {seen}");
            }
            if seen >= self.stop_after {
                StepResult::Done
            } else {
                StepResult::Continue
            }
        }
    }

    // The whole point: code registered from outside the engine runs on the same
    // loop the engine's own systems run on, over the same world, and ends the
    // run when it is finished.
    #[test]
    fn a_registered_system_runs_on_the_shipped_loop() {
        let steps = Arc::new(AtomicU32::new(0));

        let mut world = World::new();
        world.add_component(TextLabel::default());
        world.add_system(
            Phase::Late,
            "Counting",
            Counting {
                steps: Arc::clone(&steps),
                stop_after: 3,
            },
        );

        assert_eq!(App::from_world(world).into_headless().run(), Ok(()));
        assert_eq!(steps.load(Ordering::Relaxed), 3);
    }

    // An application's own component type, through the whole path: declared,
    // seeded on a world beside a vocabulary component, and read and written by
    // a registered system on the shipped loop. Nothing in the engine knows this
    // type exists.
    #[test]
    fn a_declared_component_drives_a_registered_system() {
        use crate::components::Transform;
        use crate::declare_components;

        #[derive(Debug)]
        struct Countdown(u32);

        declare_components!(Countdown);

        // Ticks every Countdown down, moving its Transform as it goes, and
        // finishes once none is left above zero. `moved` is what the run is
        // read back through: the world itself is consumed by `run`, and a step
        // that found nothing to join would end the run just as quietly as one
        // that drained the countdown.
        #[derive(Debug)]
        struct Tick {
            moved: Arc<AtomicU32>,
        }

        impl System for Tick {
            fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
                let live: Vec<_> = ctx
                    .join2::<Countdown, Transform>()
                    .filter(|(_, c, _)| c.0 > 0)
                    .map(|(entity, _, _)| entity)
                    .collect();
                if live.is_empty() {
                    return StepResult::Done;
                }
                for entity in live {
                    if let Some(c) = ctx.get_mut::<Countdown>(entity) {
                        c.0 -= 1;
                    }
                    if let Some(t) = ctx.get_mut::<Transform>(entity) {
                        t.position[0] += 1.0;
                    }
                    self.moved.fetch_add(1, Ordering::Relaxed);
                }
                StepResult::Continue
            }
        }

        let moved = Arc::new(AtomicU32::new(0));
        let mut world = World::new();
        let mover = world.spawn();
        world.insert(mover, Countdown(3));
        world.insert(mover, Transform::default());

        let app = App::from_world(world).into_headless().with_system(
            Phase::Late,
            "Tick",
            Tick {
                moved: Arc::clone(&moved),
            },
        );
        assert_eq!(app.run(), Ok(()));
        assert_eq!(
            moved.load(Ordering::Relaxed),
            3,
            "the join reached the entity on each of the three ticks its countdown ran"
        );
    }

    // The same registration reaches an app rather than a world, which is the
    // path a compiled world takes.
    #[test]
    fn an_app_takes_a_registration_too() {
        let steps = Arc::new(AtomicU32::new(0));

        let app = App::from_world(World::new()).into_headless().with_system(
            Phase::Early,
            "Counting",
            Counting {
                steps: Arc::clone(&steps),
                stop_after: 2,
            },
        );

        assert_eq!(app.run(), Ok(()));
        assert_eq!(steps.load(Ordering::Relaxed), 2);
    }
}
