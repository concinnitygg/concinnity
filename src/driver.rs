//! Choosing the loop a world runs on.

use alloc::boxed::Box;

use concinnity_engine::Driver;
use concinnity_engine::ecs::HEADLESS_SYSTEMS;

use crate::world::Inner;

// The loop an app's world runs on, chosen when the app runs rather than when
// it is built, so what the app adds first is in the world by then: its launch
// arguments name the graphics API the choice probes the machine with.
pub(crate) enum Loop {
    #[cfg(feature = "std")]
    Unchosen(Box<concinnity_runtime::Runtime>),
    Chosen(Box<dyn Driver>),
}

impl Loop {
    // The loop a world runs on unless the caller asks for another: the
    // engine's windowed one where there is an operating system to drive, the
    // headless one where there is not. This is the whole of what the two tiers
    // disagree about; everything above holds the result behind the trait.
    #[cfg(feature = "std")]
    pub(crate) fn new(world: Inner) -> Self {
        Self::runtime(concinnity_runtime::Runtime::from_world(world))
    }

    #[cfg(not(feature = "std"))]
    pub(crate) fn new(world: Inner) -> Self {
        Self::Chosen(headless(world))
    }

    // The loop an already-loaded engine runtime runs on, chosen as `adopt`
    // chooses it once the app runs.
    #[cfg(feature = "std")]
    pub(crate) fn runtime(runtime: concinnity_runtime::Runtime) -> Self {
        Self::Unchosen(Box::new(runtime))
    }

    pub(crate) fn world_mut(&mut self) -> &mut Inner {
        match self {
            #[cfg(feature = "std")]
            Self::Unchosen(runtime) => runtime.world_mut(),
            Self::Chosen(driver) => driver.world_mut(),
        }
    }

    pub(crate) fn into_world(self) -> Inner {
        match self {
            #[cfg(feature = "std")]
            Self::Unchosen(runtime) => runtime.into_world(),
            Self::Chosen(driver) => driver.into_world(),
        }
    }

    // Choose the loop now.
    pub(crate) fn chosen(self) -> Box<dyn Driver> {
        match self {
            #[cfg(feature = "std")]
            Self::Unchosen(runtime) => adopt(*runtime),
            Self::Chosen(driver) => driver,
        }
    }
}

// The loop an already-loaded engine runtime runs on, which is the runtime's own
// answer: a world with nothing to draw and a world that asked for headless run
// headless. Anything else keeps the engine's loop, whose start refuses a world
// with something to draw on a build or machine that cannot draw it.
#[cfg(feature = "std")]
fn adopt(mut runtime: concinnity_runtime::Runtime) -> Box<dyn Driver> {
    use concinnity_engine::render::backend::render_mode::Resolution;
    match runtime.render_resolution() {
        Resolution::Headless(_) => headless(runtime.into_world()),
        Resolution::Rendered | Resolution::Unavailable(_) => Box::new(runtime),
    }
}

// The headless loop, on every tier: the simulation systems this framework owns,
// stepped on a fixed virtual timestep with no window and no renderer behind
// them. A world's `Window` is inert here, since there is no render system for
// it to gate in.
pub(crate) fn headless(world: Inner) -> Box<dyn Driver> {
    Box::new(concinnity_engine::App::with_systems(
        world,
        HEADLESS_SYSTEMS,
    ))
}

#[cfg(test)]
mod tests {
    use crate::components::{AppConfig, PhysicsConfig, TextLabel, Window};
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
        world.add_component(Window::default());
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
        let mut driver = App::from_world(world).into_driver();
        driver.start().expect("the world starts");
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
}
