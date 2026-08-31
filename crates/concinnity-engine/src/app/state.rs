//! The `App` value: a world plus the loop state that drives it.

use crate::app::startup_error::StartupError;
use crate::blob;
use crate::ecs::{SYSTEMS, StepResult, World};
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
    ///
    /// Anchors the state tree beside the blob unless a host already installed
    /// one, so the settings and saves the app writes land with the world it
    /// read rather than under the directory it was launched from. The world's
    /// own `AppConfig.home` overrides that at `start`.
    pub fn from_blob(path: &std::path::Path) -> Result<Self, StartupError> {
        if concinnity_host::store::paths::state_dir().is_none()
            && let Some(state) = state_dir_for_blob(path)
        {
            concinnity_host::store::paths::set_state_dir(state);
        }
        let loaded = blob::load_at(path)
            .map_err(|e| StartupError::from_blob_failure(path.to_path_buf(), e))?;
        let mut app = Self::new();
        app.install(loaded);
        Ok(app)
    }

    /// load assets and blob payload data from the primary blob and
    /// populate the world. Replaces any previously loaded world
    pub fn load_blob(&mut self) -> Result<(), CnResult> {
        self.install(blob::load()?);
        Ok(())
    }

    // `load_blob` against a primary blob file named directly, returning the
    // world's highest blob index so the caller can check the layout it resolved
    // can actually hold it.
    pub(crate) fn load_blob_from(&mut self, primary: &std::path::Path) -> Result<u32, CnResult> {
        let loaded = blob::load_at(primary)?;
        let max_blob_index = loaded.manifest.max_blob_index;
        self.install(loaded);
        Ok(max_blob_index)
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

        let mut world = blob::world_from(blob_data);
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
        self.install_home();
        self.install_budgets();
        // The world times each system against this; without it the profile's
        // per-system micros read zero.
        self.world
            .insert_resource(crate::ecs::Clock(crate::app::clock::monotonic_micros));
        self.world.start(SYSTEMS)?;
        self.status = AppStatus::Started;
        Ok(())
    }

    // Point the runtime-writable state (`settings`, `saves/`, `crashes/`, the
    // shader caches) at the world's `AppConfig.home`. Runs before
    // `world.start(SYSTEMS)`, which is where the systems that capture a save directory
    // are built, and before anything reads the settings file. An empty `home`
    // leaves whatever the host installed, which is what keeps the writability
    // redirect a shipped player performs for itself in force.
    fn install_home(&mut self) {
        let Some(home) = self
            .world
            .query::<crate::components::AppConfig>()
            .next()
            .map(|c| c.home.clone())
            .filter(|h| !h.is_empty())
        else {
            return;
        };
        let Some(dir) = resolve_home(&home, concinnity_host::store::paths::state_dir().as_deref())
        else {
            tracing::warn!(
                "AppConfig home '{home}' is relative but no state root is installed; \
                 leaving writable state where it is"
            );
            return;
        };
        tracing::info!("Writable state: {}", dir.display());
        concinnity_host::store::paths::set_writable_state_dir(dir);
    }

    // Compute the process thread + memory budgets from the host machine and the
    // world's `AppConfig` overrides, size the shared job pool, and publish both
    // as world resources (read by the debug server and, later, the streaming
    // budget enforcement). Runs before `world.start(SYSTEMS)` so the pool is sized
    // before the first system uses it. Idempotent: a second start (the editor's
    // live rebuild) recomputes the same values and the pool sizing no-ops.
    fn install_budgets(&mut self) {
        use crate::app::{budget, sysmem};

        let config = self
            .world
            .query::<crate::components::AppConfig>()
            .next()
            .cloned()
            .unwrap_or_default();

        let threads = budget::ThreadBudget::compute(config.job_threads);
        let memory =
            budget::MemoryBudget::compute(sysmem::total_physical_bytes(), config.max_memory_mb);

        crate::jobs::configure(threads.job_threads);

        tracing::info!(
            "Thread budget: {} core(s), {} job worker(s){}",
            threads.total_cores,
            threads.job_threads,
            if config.job_threads > 0 {
                " [AppConfig override]"
            } else {
                ""
            }
        );
        tracing::info!(
            "Memory budget: {} MiB{} (total RAM {})",
            memory.budget_mib(),
            if memory.overridden {
                " [AppConfig override]"
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

    /// Take the app's world back, so a caller can put it on a different loop.
    pub fn into_world(self) -> World {
        self.world
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
    pub fn run(self) -> Result<(), CnResult> {
        self.run_with(crate::app::run::RunOptions::default())
    }

    // Run this app on the runtime loop, consuming it. Drives frames until the
    // window closes, a system stops the world, or CTRL+C is received.
    pub(crate) fn run_with(self, options: crate::app::run::RunOptions) -> Result<(), CnResult> {
        crate::app::run::start_runtime(self, options)
    }
}

// The state tree a named blob file implies: the directory holding it, stepping
// out of a `data` directory so the tree matches what a build produces (`data/`
// under the state dir, with `saves/` and `settings` beside it). `None` for a
// bare file name, which has no directory to anchor to.
fn state_dir_for_blob(primary: &std::path::Path) -> Option<std::path::PathBuf> {
    let dir = primary.parent().filter(|p| !p.as_os_str().is_empty())?;
    if dir.file_name() == Some(std::ffi::OsStr::new("data")) {
        return Some(dir.parent().unwrap_or(dir).to_path_buf());
    }
    Some(dir.to_path_buf())
}

// Resolve an authored `home` against the content root: an absolute path is used
// verbatim, a relative one hangs off the state dir. `None` when a relative path
// has no state dir to hang off, which leaves the host's own anchor in place
// rather than resolving against the working directory.
fn resolve_home(home: &str, state_dir: Option<&std::path::Path>) -> Option<std::path::PathBuf> {
    let home = std::path::Path::new(home);
    if home.is_absolute() {
        return Some(home.to_path_buf());
    }
    state_dir.map(|state| state.join(home))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::AppConfig;

    // Starting the app publishes the thread + memory budgets as world resources,
    // honoring an `AppConfig`'s overrides. A world with no GraphicsConfig starts
    // without building a GPU, so this exercises the budget install in isolation.
    #[test]
    fn start_publishes_budgets_honoring_app_config_limits() {
        let mut app = App::new();
        app.world_mut().add_component(AppConfig {
            home: String::new(),
            max_memory_mb: 512,
            job_threads: 2,
        });
        app.start().unwrap();

        let threads = crate::ecs::thread_budget(app.world()).expect("thread budget published");
        assert_eq!(threads.job_threads, 2.min(threads.total_cores));

        let memory = crate::ecs::memory_budget(app.world()).expect("memory budget published");
        assert!(memory.overridden, "the AppConfig override is recorded");
        // 512 MiB is well under 85% of any test machine's RAM, so it passes through.
        assert_eq!(memory.budget_bytes, 512 * 1024 * 1024);
    }

    // With no AppConfig declared, the budgets are still published, computed
    // from the host machine (no override).
    #[test]
    fn start_publishes_auto_budgets_without_an_app_config() {
        let mut app = App::new();
        app.start().unwrap();

        let threads = crate::ecs::thread_budget(app.world()).expect("thread budget published");
        assert_eq!(
            threads.job_threads,
            threads.total_cores.saturating_sub(1).max(1)
        );
        let memory = crate::ecs::memory_budget(app.world()).expect("memory budget published");
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
        world.add_component(AppConfig {
            home: String::new(),
            max_memory_mb: 256,
            job_threads: 1,
        });
        app.load_world(world);

        assert!(
            app.world().query::<AppConfig>().next().is_some(),
            "the loaded world replaced the empty one"
        );
        assert_eq!(app.start(), Ok(()), "the reset status permits a restart");
        // The restart budgeted against the new world's limits, not the old one's.
        let memory = crate::ecs::memory_budget(app.world()).expect("memory budget published");
        assert_eq!(memory.budget_bytes, 256 * 1024 * 1024);
    }

    // from_world hands the app a world that is already populated, in the
    // Created state so it can be started straight away.
    #[test]
    fn from_world_adopts_the_world_ready_to_start() {
        let mut world = World::new();
        world.add_component(AppConfig {
            home: String::new(),
            max_memory_mb: 128,
            job_threads: 1,
        });

        let mut app = App::from_world(world);
        assert!(app.world().query::<AppConfig>().next().is_some());
        assert_eq!(app.start(), Ok(()), "an adopted world starts");
    }

    // `home` picks where the running app writes. An absolute path is taken
    // verbatim; a relative one hangs off the content root, which is what puts a
    // portable install's state in a subfolder of its own bundle.
    #[test]
    fn home_resolves_absolute_verbatim_and_relative_against_the_content_root() {
        // What counts as absolute is platform-specific: Windows wants a drive
        // prefix, and a rooted `/var/lib` there names the current drive rather
        // than a whole path, so it takes the relative branch.
        let (root, absolute) = if cfg!(windows) {
            (r"C:\apps\MyGame", r"C:\ProgramData\mygame")
        } else {
            ("/apps/MyGame", "/var/lib/mygame")
        };
        let state = std::path::Path::new(root);

        assert_eq!(
            resolve_home("state", Some(state)),
            Some(state.join("state"))
        );
        assert_eq!(
            resolve_home(absolute, Some(state)),
            Some(std::path::PathBuf::from(absolute))
        );
        // An absolute home needs no content root behind it.
        assert_eq!(
            resolve_home(absolute, None),
            Some(std::path::PathBuf::from(absolute))
        );
    }

    // A relative `home` with nothing to resolve against is declined rather than
    // anchored to the working directory, so the host's own choice stands.
    #[test]
    fn a_relative_home_without_a_content_root_resolves_to_nothing() {
        assert_eq!(resolve_home("state", None), None);
    }

    // A blob named directly anchors the state tree beside the world it holds,
    // stepping out of a `data` directory so `saves/` and `settings` end up
    // where a build would have put them.
    #[test]
    fn a_named_blob_anchors_the_state_tree_beside_its_world() {
        use std::path::{Path, PathBuf};

        assert_eq!(
            state_dir_for_blob(Path::new("mygame/data/0")),
            Some(PathBuf::from("mygame"))
        );
        // A blob directory called anything else is the state dir itself.
        assert_eq!(
            state_dir_for_blob(Path::new("out/blobs/0")),
            Some(PathBuf::from("out").join("blobs"))
        );
        // `data/0` relative to the cwd leaves the tree at the cwd.
        assert_eq!(
            state_dir_for_blob(Path::new("data/0")),
            Some(PathBuf::new())
        );
        // A bare file name has no directory to anchor to.
        assert_eq!(state_dir_for_blob(Path::new("0")), None);
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
