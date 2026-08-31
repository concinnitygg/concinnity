//! Choosing the loop a world runs on.

use alloc::boxed::Box;

use concinnity_core::Driver;
use concinnity_core::ecs::HEADLESS_SYSTEMS;

use crate::world::Inner;

// The loop a world runs on unless the caller asks for another: the engine's
// windowed one where there is an operating system to drive, the headless one
// where there is not. This is the whole of what the two tiers disagree about;
// everything above holds the result behind the trait.
#[cfg(feature = "std")]
pub(crate) fn select(world: Inner) -> Box<dyn Driver> {
    adopt(concinnity_engine::App::from_world(world))
}

#[cfg(not(feature = "std"))]
pub(crate) fn select(world: Inner) -> Box<dyn Driver> {
    headless(world)
}

// The loop an already-loaded engine app runs on. A build with no backend
// feature has no renderer for the windowed loop to drive, so the world comes
// straight back off it onto the headless one.
#[cfg(feature = "std")]
pub(crate) fn adopt(app: concinnity_engine::App) -> Box<dyn Driver> {
    if concinnity_engine::HAS_RENDER_BACKEND {
        Box::new(app)
    } else {
        headless(app.into_world())
    }
}

// The headless loop, on every tier: the simulation systems this framework owns,
// stepped on a fixed virtual timestep with no window and no renderer behind
// them. A world's `GraphicsConfig` is inert here, since there is no render
// system for it to gate in.
pub(crate) fn headless(world: Inner) -> Box<dyn Driver> {
    Box::new(concinnity_core::App::with_systems(world, HEADLESS_SYSTEMS))
}

#[cfg(test)]
mod tests {
    use crate::components::{GraphicsConfig, TextLabel};
    use crate::{App, World};

    // The whole point of the driver being a value: a `std` build can run a
    // world in process. On the default driver this call would stand up a window
    // and wait for it, so the test would never return.
    #[test]
    fn a_headless_run_returns_in_process() {
        let mut world = World::new();
        world.add_component(TextLabel {
            content: "Hello, world!".into(),
            ..Default::default()
        });

        assert_eq!(App::from_world(world).into_headless().run(), Ok(()));
    }

    // A world authored to be seen runs headless too: the GraphicsConfig that
    // gates the render stack into a windowed run has no system to gate in here.
    #[test]
    fn a_world_that_would_render_runs_headless_as_well() {
        let mut world = World::new();
        world.add_component(GraphicsConfig::default());

        assert_eq!(App::from_world(world).into_headless().run(), Ok(()));
    }

    // With no backend feature there is no renderer to drive, so the loop a
    // world runs on by default IS the headless one: this plain `run` returns in
    // process rather than standing up a window and waiting for it. Under a
    // backend the same call would never return, which is why the assertion is
    // the backend-free build's alone.
    #[cfg(not(any(
        feature = "native",
        feature = "metal",
        feature = "directx",
        feature = "vulkan"
    )))]
    #[test]
    fn the_default_loop_is_headless_with_no_backend_compiled_in() {
        let mut world = World::new();
        world.add_component(GraphicsConfig::default());

        assert_eq!(App::from_world(world).run(), Ok(()));
    }
}
