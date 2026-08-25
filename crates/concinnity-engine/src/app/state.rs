//! The `App` value: a world plus the loop state that drives it.

use crate::blob;
use crate::ecs::{StepResult, World};
use crate::result::CnResult;
use crate::shutdown::ShutdownToken;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AppStatus {
    Created,
    Started,
}

#[derive(Debug)]
/// The application: a world plus the loop state that drives it.
pub struct App {
    status: AppStatus,
    world: World,
    shutdown: ShutdownToken,
    // FPS-cap pacer, run before each world step so no system pays the sleep
    // inside its own step time (see `app::pacing`).
    pacer: crate::app::pacing::FramePacer,
    // Fixed-timestep accumulator; publishes the frame's `SimTiming` resource
    // before each world step (see `app::clock`).
    clock: crate::app::clock::SimClock,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    /// An app holding an empty world.
    pub fn new() -> Self {
        Self {
            status: AppStatus::Created,
            world: World::new(),
            shutdown: ShutdownToken::new(),
            pacer: Default::default(),
            clock: Default::default(),
        }
    }

    /// An app holding an already-built world, ready to start or run.
    pub fn from_world(world: World) -> Self {
        let mut app = Self::new();
        app.load_world(world);
        app
    }

    /// An app holding the world compiled into the blob file at `path`.
    /// Overflow payload blobs are its siblings named by index, so a world
    /// written to `data/0` reads `data/1`, `data/2`, ... beside it.
    pub fn from_blob(path: &std::path::Path) -> Result<Self, CnResult> {
        let mut app = Self::new();
        app.install(blob::load_at(path)?);
        Ok(app)
    }

    /// load assets and blob payload data from the primary blob and
    /// populate the world. Replaces any previously loaded world
    pub fn load_blob(&mut self) -> Result<(), CnResult> {
        self.install(blob::load()?);
        Ok(())
    }

    // Populate the world from an already-decoded blob, replacing whatever the
    // app held.
    fn install(&mut self, loaded: blob::LoadedBlob) {
        let (assets, mut resources, scene_groups, mesh_bounds, physics_budget, manifest, blob_data) = (
            loaded.components,
            loaded.resources,
            loaded.scene_groups,
            loaded.mesh_bounds,
            loaded.physics_budget,
            loaded.manifest,
            loaded.blob,
        );

        let mut world = World::from_blob(blob_data);
        // The manifest's per-type counts size each column once up front, so
        // the bulk load below never reallocates mid-push.
        world.reserve_components(&manifest.component_counts);
        // Index every named component's entity as it is minted, so name
        // references resolve for any type (the decompose pass merges the
        // Prop-derived entries into this same map).
        let mut by_name = std::collections::BTreeMap::new();
        for (name, asset) in assets {
            let entity = world.add(asset);
            if let Some(id) = name {
                by_name.insert(id, entity);
            }
        }
        world.insert_resource(crate::ecs::decompose::EntityByName(by_name));
        world.insert_resource(crate::ecs::BlobSceneGroups(scene_groups));
        world.insert_resource(crate::ecs::BlobMeshBounds(mesh_bounds));
        // Absent for a world with no physics content, which is also a world
        // with no PhysicsSystem to read it.
        if let Some(budget) = physics_budget {
            world.insert_resource(concinnity_core::ecs::WorldPhysicsBudget(budget));
        }
        // Load the blob's resource stream into the per-kind tables the systems
        // read by handle. AudioSystem reads the AudioClipTable at init; the
        // renderer reads the TextureTable to build its shared texture pool.
        crate::resource::install_resource_tables(&mut world, &mut resources);
        self.world = world;
    }

    /// Borrow the app's world.
    pub fn world(&self) -> &World {
        &self.world
    }

    /// Mutably borrow the app's world.
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    /// clone of the root cancellation token. Pass this to systems or the
    /// ctrl+c handler so they all share a single cancellation source
    pub fn shutdown_token(&self) -> ShutdownToken {
        self.shutdown.clone()
    }

    /// Build the world's systems and run their `init`. Must run once, before
    /// the first step.
    pub fn start(&mut self) -> Result<(), CnResult> {
        if self.status != AppStatus::Created {
            tracing::error!("App must be in Created state to start");
            return Err(CnResult::InvalidState);
        }
        self.install_budgets();
        self.world.start()?;
        self.status = AppStatus::Started;
        Ok(())
    }

    // Compute the process thread + memory budgets from the host machine and the
    // world's `Application` limits, size the shared job pool, and publish both
    // as world resources (read by the debug server and, later, the streaming
    // budget enforcement). Runs before `world.start()` so the pool is sized
    // before the first system uses it. Idempotent: a second start (the editor's
    // live rebuild) recomputes the same values and the pool sizing no-ops.
    fn install_budgets(&mut self) {
        use crate::app::{budget, sysmem};

        let limits = self
            .world
            .query::<crate::components::Application>()
            .next()
            .map(|a| a.limits)
            .unwrap_or_default();

        let threads = budget::ThreadBudget::compute(limits.job_threads);
        let memory =
            budget::MemoryBudget::compute(sysmem::total_physical_bytes(), limits.max_memory_mb);

        crate::jobs::configure(threads.job_threads);

        tracing::info!(
            "Thread budget: {} core(s), {} job worker(s){}",
            threads.total_cores,
            threads.job_threads,
            if limits.job_threads > 0 {
                " [Application override]"
            } else {
                ""
            }
        );
        tracing::info!(
            "Memory budget: {} MiB{} (total RAM {})",
            memory.budget_mib(),
            if memory.overridden {
                " [Application override]"
            } else {
                ""
            },
            match memory.total_ram_bytes {
                Some(bytes) => format!("{} MiB", bytes / (1024 * 1024)),
                None => "unknown".to_string(),
            }
        );

        self.world.insert_resource(threads);
        self.world.insert_resource(memory);
    }

    /// Replace the current world and reset to Created so start() can be called again.
    /// Used to load a new scene at runtime.
    pub fn load_world(&mut self, world: World) {
        self.world = world;
        self.status = AppStatus::Created;
    }

    // single world step, for callers that drive their own outer loop
    // (e.g. run_loop_macos in crate::app::run, which interleaves CFRunLoop pumps).
    // The FPS-cap pacer holds the step's start to its target interval first,
    // then the simulation clock publishes the frame's fixed-tick budget. The
    // menu state read is the previous frame's, the same one-frame lag the
    // pacer's clamp accepts.
    pub(crate) fn world_step(&mut self) -> StepResult {
        self.pacer.pace(&self.world);
        let paused = self
            .world
            .resource::<crate::ecs::MenuActive>()
            .is_some_and(|m| m.0);
        let timing = self.clock.advance(std::time::Instant::now(), paused);
        self.world.insert_resource(timing);
        self.world.step()
    }

    /// Run this app on the runtime loop with default options, consuming it.
    pub fn run(self) -> std::io::Result<()> {
        self.run_with(crate::app::run::RunOptions::default())
    }

    // Run this app on the runtime loop, consuming it. Drives frames until the
    // window closes, a system stops the world, or CTRL+C is received.
    pub(crate) fn run_with(self, options: crate::app::run::RunOptions) -> std::io::Result<()> {
        crate::app::run::start_runtime(self, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{AppLimits, Application};

    // Starting the app publishes the thread + memory budgets as world resources,
    // honoring an `Application`'s limits. A world with no GraphicsConfig starts
    // without building a GPU, so this exercises the budget install in isolation.
    #[test]
    fn start_publishes_budgets_honoring_application_limits() {
        let mut app = App::new();
        app.world_mut().add_component(Application {
            limits: AppLimits {
                max_memory_mb: 512,
                job_threads: 2,
            },
        });
        app.start().unwrap();

        let threads = app
            .world()
            .thread_budget()
            .expect("thread budget published");
        assert_eq!(threads.job_threads, 2.min(threads.total_cores));

        let memory = app
            .world()
            .memory_budget()
            .expect("memory budget published");
        assert!(memory.overridden, "the Application override is recorded");
        // 512 MiB is well under 85% of any test machine's RAM, so it passes through.
        assert_eq!(memory.budget_bytes, 512 * 1024 * 1024);
    }

    // With no Application declared, the budgets are still published, computed
    // from the host machine (no override).
    #[test]
    fn start_publishes_auto_budgets_without_an_application() {
        let mut app = App::new();
        app.start().unwrap();

        let threads = app
            .world()
            .thread_budget()
            .expect("thread budget published");
        assert_eq!(
            threads.job_threads,
            threads.total_cores.saturating_sub(1).max(1)
        );
        let memory = app
            .world()
            .memory_budget()
            .expect("memory budget published");
        assert!(!memory.overridden);
        assert!(memory.budget_bytes > 0);
    }

    // Only a Created app starts, and a default-constructed one is Created. The
    // second call is refused by the status guard rather than re-initing every
    // system on the running world.
    #[test]
    fn start_twice_is_rejected() {
        let mut app = App::default();
        assert_eq!(app.start(), Ok(()));
        assert_eq!(app.start(), Err(CnResult::InvalidState));
    }

    // load_world swaps in a new world and resets to Created, so a started app
    // can be started again on the new content (the runtime scene-load path).
    #[test]
    fn load_world_replaces_the_world_and_allows_a_restart() {
        let mut app = App::new();
        app.start().unwrap();
        assert!(app.start().is_err(), "the app is Started");

        let mut world = World::new();
        world.add_component(Application {
            limits: AppLimits {
                max_memory_mb: 256,
                job_threads: 1,
            },
        });
        app.load_world(world);

        assert!(
            app.world().query::<Application>().next().is_some(),
            "the loaded world replaced the empty one"
        );
        assert_eq!(app.start(), Ok(()), "the reset status permits a restart");
        // The restart budgeted against the new world's limits, not the old one's.
        let memory = app
            .world()
            .memory_budget()
            .expect("memory budget published");
        assert_eq!(memory.budget_bytes, 256 * 1024 * 1024);
    }

    // from_world hands the app a world that is already populated, in the
    // Created state so it can be started straight away.
    #[test]
    fn from_world_adopts_the_world_ready_to_start() {
        let mut world = World::new();
        world.add_component(Application {
            limits: AppLimits {
                max_memory_mb: 128,
                job_threads: 1,
            },
        });

        let mut app = App::from_world(world);
        assert!(app.world().query::<Application>().next().is_some());
        assert_eq!(app.start(), Ok(()), "an adopted world starts");
    }

    // With no FrameRateCap published the pacer has nothing to hold the frame
    // to, so the step runs straight through; an empty world reports Done as it
    // has no systems left to run.
    #[test]
    fn world_step_without_a_frame_rate_cap_runs_unpaced() {
        let mut app = App::new();
        app.start().unwrap();
        assert!(
            app.world().resource::<crate::ecs::FrameRateCap>().is_none(),
            "no cap is published without a GraphicsConfig"
        );
        assert_eq!(app.world_step(), StepResult::Done);
    }
}
