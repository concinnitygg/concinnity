//! The windowed loop as a [`Driver`]: what a host holds when the loop it runs
//! is a runtime value rather than a compile-time type.

use concinnity_core::Driver;
use concinnity_core::ecs::World;
use concinnity_core::error::WorldError;

use crate::app::runtime::Runtime;

impl Driver for Runtime {
    fn start(&mut self) -> Result<(), WorldError> {
        Runtime::start(self)
    }

    fn run(self: Box<Self>) -> Result<(), WorldError> {
        (*self).run()
    }

    fn into_world(self: Box<Self>) -> World {
        (*self).into_world()
    }

    fn world_mut(&mut self) -> &mut World {
        Runtime::world_mut(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::AppConfig;

    // Starting through the trait reaches the windowed loop's own start, budgets
    // and all, and the second call is refused the same way the inherent one is.
    // A world with no GraphicsConfig starts without building a GPU.
    #[test]
    fn a_driver_starts_the_world_it_holds() {
        let mut driver: Box<dyn Driver> = Box::new(Runtime::new());
        assert!(driver.start().is_ok());
        assert!(matches!(driver.start(), Err(WorldError::AlreadyStarted)));
    }

    // The other way out: the world comes back as it was handed over, so a
    // caller can put it on a different loop.
    #[test]
    fn a_driver_hands_its_world_back_unrun() {
        let mut runtime = Runtime::new();
        runtime.world_mut().add_component(AppConfig {
            home: String::new(),
            max_memory_mb: 512,
            job_threads: 2,
        });

        let driver: Box<dyn Driver> = Box::new(runtime);
        let world = driver.into_world();
        assert!(
            world.query::<AppConfig>().next().is_some(),
            "the world keeps its content"
        );
    }
}
