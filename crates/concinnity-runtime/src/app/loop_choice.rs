//! An application: a world plus the loop that runs it, chosen when it runs.

use std::path::Path;

use concinnity_engine::Driver;
use concinnity_engine::ecs::{Phase, System, World};
use concinnity_engine::error::WorldError;
use concinnity_engine::launch::LaunchArgs;
use concinnity_engine::render::backend::render_mode::Resolution;

use crate::{Runtime, StartupError};

/// A runnable application, built from a world assembled in process or from a
/// world already compiled into a blob file.
///
/// The loop is chosen when the app runs rather than when it is built, so what
/// the app adds first is in the world by then: its launch arguments name the
/// graphics API the choice probes the machine with.
pub struct App {
    driver: Loop,
}

enum Loop {
    Unchosen(Box<Runtime>),
    Chosen(Box<dyn Driver>),
}

impl core::fmt::Debug for App {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("App").finish_non_exhaustive()
    }
}

impl App {
    /// An app that runs `world`.
    pub fn from_world(world: World) -> Self {
        Self::unchosen(Runtime::from_world(world))
    }

    /// An app that runs the world compiled into the blob file at `path`.
    /// Overflow payload blobs are that file's siblings named by index, so a
    /// world written to `data/0` reads `data/1`, `data/2`, ... beside it.
    ///
    /// What the app writes at runtime (its settings, save files, crash
    /// reports, and shader caches) lands beside the world it read, stepping
    /// out of a directory named `data`, so `from_blob("mygame/data/0")` keeps
    /// them under `mygame/`. A world that declares an `AppConfig` with a `home`
    /// chooses the location itself.
    ///
    /// # Errors
    ///
    /// Returns an error if no blob exists at `path`, it does not load, or it is
    /// a single-file package of a world that needs overflow blobs.
    pub fn from_blob(path: impl AsRef<Path>) -> Result<Self, StartupError> {
        Runtime::from_blob(path.as_ref()).map(Self::unchosen)
    }

    fn unchosen(runtime: Runtime) -> Self {
        Self {
            driver: Loop::Unchosen(Box::new(runtime)),
        }
    }

    /// Register a system on the app's world, to run in `phase` under `name`.
    ///
    /// The same registration [`World::add_system`] makes, reachable on an app
    /// whose world came from a blob. A name that repeats an engine system's or
    /// an earlier registration's makes [`run`](App::run) return an error.
    #[must_use]
    pub fn with_system<S: System>(mut self, phase: Phase, name: &'static str, system: S) -> Self {
        self.world_mut().add_system(phase, name, system);
        self
    }

    /// The same app, run as `args` ask: benchmarked, with vsync, under a
    /// frame-rate cap or a quality preset, on a chosen graphics API, or for a
    /// fixed number of frames.
    #[must_use]
    pub fn with_args(mut self, args: LaunchArgs) -> Self {
        self.world_mut().insert_resource(args);
        self
    }

    /// The same app on the headless loop: the simulation systems stepped on a
    /// fixed virtual timestep, with no window and no renderer. A world
    /// authored to be seen keeps everything it holds and draws none of it.
    #[must_use]
    pub fn into_headless(self) -> Self {
        Self {
            driver: Loop::Chosen(headless(self.into_world())),
        }
    }

    /// Run the app until its window closes, a system stops the world, its last
    /// system finishes, or the process is interrupted.
    ///
    /// A world with nothing to draw, a world that asked for headless, or a
    /// machine that cannot render has no window to close: the world steps on a
    /// fixed virtual timestep, as fast as the host can step it.
    ///
    /// # Errors
    ///
    /// Returns an error when the world fails to start, or a world meant to be
    /// seen could not build its renderer on a machine that has a GPU.
    pub fn run(self) -> Result<(), WorldError> {
        self.into_driver().run()
    }

    /// Choose the loop now and hand it over without running it.
    pub fn into_driver(self) -> Box<dyn Driver> {
        match self.driver {
            Loop::Unchosen(runtime) => adopt(*runtime),
            Loop::Chosen(driver) => driver,
        }
    }

    fn world_mut(&mut self) -> &mut World {
        match &mut self.driver {
            Loop::Unchosen(runtime) => runtime.world_mut(),
            Loop::Chosen(driver) => driver.world_mut(),
        }
    }

    fn into_world(self) -> World {
        match self.driver {
            Loop::Unchosen(runtime) => runtime.into_world(),
            Loop::Chosen(driver) => driver.into_world(),
        }
    }
}

// The runtime's own answer: a world with nothing to draw and a world that
// asked for headless run headless. Anything else keeps the windowed loop, whose
// start refuses a world with something to draw on a build or machine that
// cannot draw it.
fn adopt(mut runtime: Runtime) -> Box<dyn Driver> {
    match runtime.render_resolution() {
        Resolution::Headless(_) => headless(runtime.into_world()),
        Resolution::Rendered | Resolution::Unavailable(_) => Box::new(runtime),
    }
}

fn headless(world: World) -> Box<dyn Driver> {
    Box::new(concinnity_engine::App::from_world(world))
}

#[cfg(test)]
mod tests {
    use super::App;
    use concinnity_engine::components::{AppConfig, PhysicsConfig, TextLabel, Window};
    use concinnity_engine::ecs::World;

    fn hello() -> TextLabel {
        TextLabel {
            content: "Hello, world!".into(),
            ..Default::default()
        }
    }

    // A std build can run a world in process. On the default loop this world
    // would stand up a window and wait for it, so the test would never return.
    #[test]
    fn a_headless_run_returns_in_process() {
        let mut world = World::new();
        world.add_component(hello());

        App::from_world(world)
            .into_headless()
            .run()
            .expect("the run ends");
    }

    // What would gate the render stack into a windowed run has no system to
    // gate in on the headless loop.
    #[test]
    fn a_world_that_would_render_runs_headless_as_well() {
        let mut world = World::new();
        world.add_component(Window::default());
        world.add_component(hello());

        App::from_world(world)
            .into_headless()
            .run()
            .expect("the run ends");
    }

    // Windows are banned first, so a selection that regressed to the windowed
    // loop panics naming the backend instead of blocking on an event loop the
    // harness cannot end. `start` rather than `run`, since a world carrying a
    // never-ending system never finishes.
    fn assert_default_loop_is_headless(world: World) {
        concinnity_testing::forbid_windows();
        let mut driver = App::from_world(world).into_driver();
        driver.start().expect("the world starts");
    }

    #[test]
    fn a_world_asking_for_headless_takes_the_headless_loop_by_default() {
        let mut world = World::new();
        world.add_component(hello());
        world.add_component(AppConfig {
            headless: true,
            ..Default::default()
        });

        assert_default_loop_is_headless(world);
    }

    // What keeps a simulation-only world from opening an empty window.
    #[test]
    fn a_world_with_nothing_to_draw_takes_the_headless_loop_by_default() {
        let mut world = World::new();
        world.add_component(PhysicsConfig::default());

        assert_default_loop_is_headless(world);
    }
}
