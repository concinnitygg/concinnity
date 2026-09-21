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
    adopt(concinnity_engine::Runtime::from_world(world))
}

#[cfg(not(feature = "std"))]
pub(crate) fn select(world: Inner) -> Box<dyn Driver> {
    headless(world)
}

// The loop an already-loaded engine runtime runs on, which is the runtime's own
// answer: it resolves a world with nothing to draw, a world that asked for
// headless, a build with no backend, and a machine with no GPU all to the same
// place. Anything else keeps the windowed loop.
#[cfg(feature = "std")]
pub(crate) fn adopt(mut runtime: concinnity_engine::Runtime) -> Box<dyn Driver> {
    if runtime.render_mode().renders() {
        Box::new(runtime)
    } else {
        headless(runtime.into_world())
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
    use crate::components::{AppConfig, GraphicsConfig, PhysicsConfig, TextLabel};
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

        App::from_world(world)
            .into_headless()
            .run()
            .expect("the run ends");
    }

    // A world authored to be seen runs headless too: what would gate the render
    // stack into a windowed run has no system to gate in here.
    #[test]
    fn a_world_that_would_render_runs_headless_as_well() {
        let mut world = World::new();
        world.add_component(GraphicsConfig::default());
        world.add_component(TextLabel {
            content: "Hello, world!".into(),
            ..Default::default()
        });

        App::from_world(world)
            .into_headless()
            .run()
            .expect("the run ends");
    }

    // Start `world` on whatever loop the default selection picks, with windows
    // banned first: a selection that regressed to the windowed loop panics
    // naming the backend instead of blocking on an event loop the harness
    // cannot end. Asserting on `start` rather than `run` because the headless
    // loop runs until its last system finishes, which a world carrying a
    // never-ending system never does.
    fn assert_default_loop_is_headless(world: World) {
        concinnity_testing::forbid_windows();
        let mut app = App::from_world(world);
        app.driver_mut().start().expect("the world starts");
    }

    // The opt-out, through the default selection rather than `into_headless`: a
    // world with plenty to draw stays off the windowed loop because it asked
    // to.
    #[test]
    fn a_world_asking_for_headless_takes_the_headless_loop_by_default() {
        let mut world = World::new();
        world.add_component(TextLabel {
            content: "Hello, world!".into(),
            ..Default::default()
        });
        world.add_component(AppConfig {
            headless: true,
            ..Default::default()
        });

        assert_default_loop_is_headless(world);
    }

    // A world holding nothing to draw takes the headless loop as well, which is
    // what keeps a simulation-only world from opening an empty window.
    #[test]
    fn a_world_with_nothing_to_draw_takes_the_headless_loop_by_default() {
        let mut world = World::new();
        world.add_component(PhysicsConfig::default());

        assert_default_loop_is_headless(world);
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

        App::from_world(world).run().expect("the run ends");
    }
}
