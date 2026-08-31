//! The windowed loop as a [`Driver`]: what a host holds when the loop it runs
//! is a runtime value rather than a compile-time type.

use concinnity_core::Driver;
use concinnity_core::ecs::World;

use crate::app::state::App;
use crate::result::CnResult;

impl Driver for App {
    fn start(&mut self) -> Result<(), CnResult> {
        App::start(self)
    }

    fn run(self: Box<Self>) -> Result<(), CnResult> {
        (*self).run()
    }

    fn into_world(self: Box<Self>) -> World {
        (*self).into_world()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::AppConfig;

    // Starting through the trait reaches the windowed loop's own start, budgets
    // and all, and the second call is refused the same way the inherent one is.
    // A world with no GraphicsConfig starts without building a GPU.
    #[test]
    fn a_driver_starts_the_world_it_holds() {
        let mut driver: Box<dyn Driver> = Box::new(App::new());
        assert_eq!(driver.start(), Ok(()));
        assert_eq!(driver.start(), Err(CnResult::InvalidState));
    }

    // The other way out: the world comes back as it was handed over, so a
    // caller can put it on a different loop.
    #[test]
    fn a_driver_hands_its_world_back_unrun() {
        let mut app = App::new();
        app.world_mut().add_component(AppConfig {
            home: String::new(),
            max_memory_mb: 512,
            job_threads: 2,
        });

        let driver: Box<dyn Driver> = Box::new(app);
        let world = driver.into_world();
        assert!(
            world.query::<AppConfig>().next().is_some(),
            "the world keeps its content"
        );
    }
}
