// src/app/state.rs
use crate::blob;
use crate::ecs::{StepResult, World};
use crate::result::CnResult;
use concinnity_core::shutdown::ShutdownToken;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppStatus {
    Created,
    Started,
    #[allow(dead_code)]
    Stopped,
}

#[derive(Debug)]
pub struct App {
    status: AppStatus,
    world: World,
    shutdown: ShutdownToken,
    // FPS-cap pacer, run before each world step so no system pays the sleep
    // inside its own step time (see `app::pacing`).
    pacer: crate::app::pacing::FramePacer,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    pub fn new() -> Self {
        Self {
            status: AppStatus::Created,
            world: World::new_empty(),
            shutdown: ShutdownToken::new(),
            pacer: Default::default(),
        }
    }

    pub fn new_with_token(shutdown: ShutdownToken) -> Self {
        Self {
            status: AppStatus::Created,
            world: World::new_empty(),
            shutdown,
            pacer: Default::default(),
        }
    }

    // load assets and blob payload data from the primary blob and
    // populate the world. Replaces any previously loaded world
    pub fn load_blob(&mut self) -> Result<(), CnResult> {
        let (assets, resources, blob_data) = blob::load()?;

        let mut world = World::new(blob_data);
        for asset in assets {
            world.add(asset);
        }
        // Load the blob's resource stream into the per-kind tables the systems
        // read by handle. AudioSystem reads the AudioClipTable at init; the
        // renderer reads the TextureTable to build its shared texture pool.
        crate::resource::install_resource_tables(&mut world, &resources);
        self.world = world;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn world(&self) -> &World {
        &self.world
    }

    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    // clone of the root cancellation token. Pass this to systems or the
    // ctrl+c handler so they all share a single cancellation source
    pub fn shutdown_token(&self) -> ShutdownToken {
        self.shutdown.clone()
    }

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
            .query::<crate::assets::Application>()
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

    // Replace the current world and reset to Created so start() can be called again.
    // Used to load a new scene at runtime.
    #[allow(dead_code)]
    pub fn load_world(&mut self, world: World) {
        self.world = world;
        self.status = AppStatus::Created;
    }

    // single world step, for callers that drive their own outer loop
    // (e.g. run_loop_macos in crate::app::run, which interleaves CFRunLoop pumps).
    // The FPS-cap pacer holds the step's start to its target interval first.
    pub fn world_step(&mut self) -> StepResult {
        self.pacer.pace(&self.world);
        self.world.step()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{AppLimits, Application};

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
            ..Default::default()
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

        let mut world = World::new_empty();
        world.add_component(Application {
            limits: AppLimits {
                max_memory_mb: 256,
                job_threads: 1,
            },
            ..Default::default()
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

    // The token handed to new_with_token is the same cancellation source
    // shutdown_token() clones out, so one cancel stops every holder.
    #[test]
    fn new_with_token_shares_one_cancellation_source() {
        let token = ShutdownToken::new();
        let app = App::new_with_token(token.clone());
        let handed_out = app.shutdown_token();
        assert!(!handed_out.is_cancelled());

        token.cancel();
        assert!(handed_out.is_cancelled(), "the clone observes the cancel");
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
