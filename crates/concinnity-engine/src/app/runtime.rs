//! The `Runtime` value: a world plus the loop state that drives it.

use concinnity_core::components::AppConfig;
use concinnity_core::ecs::{Clock, MenuActive, StepResult, World};
use concinnity_core::error::WorldError;
use concinnity_host::store::paths::StateTree;
use concinnity_host::thread::jobs::configure;
use concinnity_host::thread::jobs::pool;

use crate::app::run::LaunchRequest;
use crate::app::startup_error::StartupError;
use crate::blob;
use crate::ecs::SYSTEMS;
use crate::shutdown::ShutdownToken;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RuntimeStatus {
    Created,
    Started,
}

#[derive(Debug)]
/// The windowed loop: a world plus the loop state that drives it.
pub struct Runtime {
    status: RuntimeStatus,
    world: World,
    // Where this runtime reads and writes, or `None` for a runtime with no tree: its
    // world runs, and everything that would touch disk does nothing. Published
    // to the world at `start` so the systems are told rather than resolving
    // paths of their own.
    state: Option<StateTree>,
    shutdown: ShutdownToken,
    // FPS-cap pacer, run before each world step so no system pays the sleep
    // inside its own step time (see `app::pacing`).
    pacer: crate::app::pacing::FramePacer,
    // Fixed-timestep accumulator; publishes the frame's `SimTiming` and
    // `FrameTime` resources before each world step (see `app::clock`).
    clock: crate::app::clock::SimClock,
    // What the launch asked the engine to arm; published at every `start`, so a
    // world loaded later inherits it.
    launch: LaunchRequest,
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

impl Runtime {
    /// A runtime holding an empty world.
    pub fn new() -> Self {
        Self {
            status: RuntimeStatus::Created,
            world: World::new(),
            state: None,
            shutdown: ShutdownToken::new(),
            pacer: Default::default(),
            clock: Default::default(),
            launch: LaunchRequest::default(),
        }
    }

    /// A runtime that reads and writes under `tree`: its blobs, its settings, its
    /// saves, and the caches it warms. Without one the runtime runs a world and
    /// touches no disk.
    #[must_use]
    pub fn in_tree(mut self, tree: StateTree) -> Self {
        self.state = Some(tree);
        self
    }

    /// A runtime that arms what `launch` asks for: a development session, forced
    /// render settings, or frame capture. Every world this runtime starts sees it.
    #[must_use]
    pub fn with_launch(mut self, launch: LaunchRequest) -> Self {
        self.launch = launch;
        self
    }

    // The launch request, for the runtime entry that adds to it before start.
    pub(crate) fn launch_mut(&mut self) -> &mut LaunchRequest {
        &mut self.launch
    }

    /// The state tree this runtime runs against, if it has one.
    pub fn state_tree(&self) -> Option<&StateTree> {
        self.state.as_ref()
    }

    /// A runtime holding an already-built world, ready to start or run.
    pub fn from_world(world: World) -> Self {
        let mut runtime = Self::new();
        runtime.load_world(world);
        runtime
    }

    /// A runtime holding the world compiled into the blob file at `path`.
    /// Overflow payload blobs are its siblings named by index, so a world
    /// written to `data/0` reads `data/1`, `data/2`, ... beside it.
    ///
    /// Derives the state tree from the blob's own directory, so the settings
    /// and saves the runtime writes land with the world it read rather than under
    /// the directory it was launched from. A caller with a tree of its own
    /// builds the runtime with [`in_tree`](Self::in_tree) instead. The world's own
    /// `AppConfig.home` overrides either at `start`.
    pub fn from_blob(path: &std::path::Path) -> Result<Self, StartupError> {
        let mut runtime = Self::new();
        runtime.load_blob_from(path)?;
        runtime.state = state_dir_for_blob(path).map(StateTree::at);
        Ok(runtime)
    }

    /// Load assets and blob payload data from the primary blob under this runtime's
    /// state tree, and populate the world. Replaces any previously loaded
    /// world. `NoStateRoot` when the runtime has no tree to read from.
    pub fn load_blob(&mut self) -> Result<(), StartupError> {
        let primary = self.primary_blob().ok_or(StartupError::NoStateRoot)?;
        self.load_blob_from(&primary)?;
        Ok(())
    }

    /// The primary blob this runtime reads: blob 0 under its tree's `data/`.
    /// `None` for a runtime with no tree.
    pub fn primary_blob(&self) -> Option<std::path::PathBuf> {
        self.state
            .as_ref()
            .map(|tree| concinnity_host::store::blob::primary_in(&tree.data_dir()))
    }

    // `load_blob` against a primary blob file named directly, returning the
    // world's highest blob index so the caller can check the layout it resolved
    // can actually hold it.
    pub(crate) fn load_blob_from(
        &mut self,
        primary: &std::path::Path,
    ) -> Result<u32, StartupError> {
        let loaded = blob::load_at(primary)
            .map_err(|e| StartupError::from_blob_failure(primary.to_path_buf(), e))?;
        let max_blob_index = loaded.manifest.max_blob_index;
        self.install(loaded);
        Ok(max_blob_index)
    }

    // Populate the world from an already-decoded blob, replacing whatever the
    // runtime held.
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
        // Every entity is identified as it is minted, so any asset id resolves
        // to its entity through `EntityById`.
        for (id, asset) in assets {
            world.add(asset, id);
        }
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

    /// Borrow the runtime's world.
    pub fn world(&self) -> &World {
        &self.world
    }

    /// Mutably borrow the runtime's world.
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
    pub fn start(&mut self) -> Result<(), WorldError> {
        if self.status != RuntimeStatus::Created {
            return Err(WorldError::AlreadyStarted);
        }
        self.install_home();
        self.publish_state_tree();
        self.install_budgets();
        // The world times each system against this; without it the profile's
        // per-system micros read zero.
        self.world
            .insert_resource(Clock(crate::app::clock::monotonic_micros));
        self.world.insert_resource(self.launch);
        self.world.start(SYSTEMS)?;
        self.status = RuntimeStatus::Started;
        Ok(())
    }

    // Point the runtime-writable state (`settings`, `saves/`, `crashes/`, the
    // shader caches) at the world's `AppConfig.home`. Runs before
    // `world.start(SYSTEMS)`, which is where the systems that capture a save
    // directory are built, and before anything reads the settings file. An
    // empty `home` leaves the tree the host built, which is what keeps the
    // writability redirect a shipped player performs for itself in force.
    fn install_home(&mut self) {
        let Some(home) = self
            .world
            .query::<AppConfig>()
            .next()
            .map(|c| c.home.clone())
            .filter(|h| !h.is_empty())
        else {
            return;
        };
        let Some(tree) = self.state.as_ref() else {
            tracing::warn!(
                "AppConfig home '{home}' has no state tree to resolve against; \
                 the runtime writes nowhere"
            );
            return;
        };
        let Some(dir) = resolve_home(&home, tree.content_root()) else {
            tracing::warn!(
                "AppConfig home '{home}' is relative but the state tree has no root; \
                 leaving writable state where it is"
            );
            return;
        };
        tracing::info!("Writable state: {}", dir.display());
        self.state = Some(tree.clone().with_writable(dir));
    }

    // Hand the state tree to the world, and to the process-wide caches that
    // outlive any one call. Runs after `install_home`, so what the systems read
    // is the tree the world asked for, and before `world.start(SYSTEMS)`, which
    // is where the systems that capture a directory are built. A runtime with no
    // tree drops the anchor a previous one left, so it touches no cache either.
    fn publish_state_tree(&mut self) {
        let Some(tree) = self.state.clone() else {
            concinnity_host::store::cache::clear_anchor();
            return;
        };
        concinnity_host::store::cache::anchor(tree.runtime_cache_path());
        self.world.insert_resource(tree);
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
            .query::<AppConfig>()
            .next()
            .cloned()
            .unwrap_or_default();

        let threads = budget::ThreadBudget::compute(config.job_threads);
        let memory =
            budget::MemoryBudget::compute(sysmem::total_physical_bytes(), config.max_memory_mb);

        let sized = configure(threads.job_threads);
        let job_workers = pool().thread_count();
        if !sized && job_workers != threads.job_threads {
            tracing::warn!(
                "job pool already built with {job_workers} worker(s); the requested {} cannot take effect",
                threads.job_threads
            );
        }
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

    /// Take the runtime's world back, so a caller can put it on a different loop.
    pub fn into_world(self) -> World {
        self.world
    }

    /// Replace the current world and reset to Created so start() can be called again.
    /// Used to load a new scene at runtime.
    pub fn load_world(&mut self, world: World) {
        self.world = world;
        self.status = RuntimeStatus::Created;
    }

    /// Advance the world one frame, for a caller that drives its own outer
    /// loop: the run loop that interleaves platform event pumps, and a host
    /// application whose OS owns the loop and calls this per display refresh.
    ///
    /// The FPS-cap pacer holds the step's start to its target interval first,
    /// then the simulation clock publishes the frame's fixed-tick budget and
    /// real frame time. The menu state read is the previous frame's, the same
    /// one-frame lag the pacer's clamp accepts.
    pub fn world_step(&mut self) -> StepResult {
        self.pacer.pace(&self.world);
        let paused = self.world.resource::<MenuActive>().is_some_and(|m| m.0);
        let (timing, frame) = self.clock.advance(std::time::Instant::now(), paused);
        self.world.insert_resource(timing);
        self.world.insert_resource(frame);
        self.world.step()
    }

    /// Run this on the run loop with default options, consuming it.
    pub fn run(self) -> Result<(), WorldError> {
        self.run_with(crate::app::run::RunOptions::default())
    }

    // Run this on the run loop, consuming it. Drives frames until the
    // window closes, a system stops the world, or CTRL+C is received.
    pub(crate) fn run_with(self, options: crate::app::run::RunOptions) -> Result<(), WorldError> {
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
// verbatim, a relative one hangs off the content root. `None` when a relative
// path has no root to hang off, which leaves the host's own tree in place
// rather than resolving against the working directory.
fn resolve_home(home: &str, content_root: &std::path::Path) -> Option<std::path::PathBuf> {
    let home = std::path::Path::new(home);
    if home.is_absolute() {
        return Some(home.to_path_buf());
    }
    (!content_root.as_os_str().is_empty()).then(|| content_root.join(home))
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::AppConfig;
    use concinnity_core::ecs::FrameRateCap;

    // Starting the runtime publishes the thread + memory budgets as world resources,
    // honoring an `AppConfig`'s overrides. A world with no GraphicsConfig starts
    // without building a GPU, so this exercises the budget install in isolation.
    #[test]
    fn start_publishes_budgets_honoring_app_config_limits() {
        let mut runtime = Runtime::new();
        runtime.world_mut().add_component(AppConfig {
            home: String::new(),
            max_memory_mb: 512,
            job_threads: 2,
        });
        runtime.start().unwrap();

        let threads = crate::ecs::thread_budget(runtime.world()).expect("thread budget published");
        assert_eq!(threads.job_threads, 2.min(threads.total_cores));

        let memory = crate::ecs::memory_budget(runtime.world()).expect("memory budget published");
        assert!(memory.overridden, "the AppConfig override is recorded");
        // 512 MiB is well under 85% of any test machine's RAM, so it passes through.
        assert_eq!(memory.budget_bytes, 512 * 1024 * 1024);
    }

    // With no AppConfig declared, the budgets are still published, computed
    // from the host machine (no override).
    #[test]
    fn start_publishes_auto_budgets_without_an_app_config() {
        let mut runtime = Runtime::new();
        runtime.start().unwrap();

        let threads = crate::ecs::thread_budget(runtime.world()).expect("thread budget published");
        assert_eq!(
            threads.job_threads,
            threads.total_cores.saturating_sub(1).max(1)
        );
        let memory = crate::ecs::memory_budget(runtime.world()).expect("memory budget published");
        assert!(!memory.overridden);
        assert!(memory.budget_bytes > 0);
    }

    // Only a Created runtime starts, and a default-constructed one is Created. The
    // second call is refused by the status guard rather than re-initing every
    // system on the running world.
    #[test]
    fn start_twice_is_rejected() {
        let mut runtime = Runtime::default();
        runtime.start().expect("a Created runtime starts");
        assert!(matches!(runtime.start(), Err(WorldError::AlreadyStarted)));
    }

    // load_world swaps in a new world and resets to Created, so a started runtime
    // can be started again on the new content (the runtime scene-load path).
    #[test]
    fn load_world_replaces_the_world_and_allows_a_restart() {
        let mut runtime = Runtime::new();
        runtime.start().unwrap();
        assert!(runtime.start().is_err(), "the runtime is Started");

        let mut world = World::new();
        world.add_component(AppConfig {
            home: String::new(),
            max_memory_mb: 256,
            job_threads: 1,
        });
        runtime.load_world(world);

        assert!(
            runtime.world().query::<AppConfig>().next().is_some(),
            "the loaded world replaced the empty one"
        );
        runtime.start().expect("the reset status permits a restart");
        // The restart budgeted against the new world's limits, not the old one's.
        let memory = crate::ecs::memory_budget(runtime.world()).expect("memory budget published");
        assert_eq!(memory.budget_bytes, 256 * 1024 * 1024);
    }

    // The launch request outlives a world swap: the editor's rebuild loads a new
    // world into the same runtime and starts it again.
    #[test]
    fn the_launch_request_survives_a_world_load_and_restart() {
        let launch = LaunchRequest {
            dev_loop: true,
            ..Default::default()
        };
        let mut runtime = Runtime::new().with_launch(launch);
        runtime.start().unwrap();
        assert_eq!(runtime.world().resource::<LaunchRequest>(), Some(&launch));

        runtime.load_world(World::new());
        assert_eq!(runtime.world().resource::<LaunchRequest>(), None);
        runtime.start().unwrap();
        assert_eq!(runtime.world().resource::<LaunchRequest>(), Some(&launch));
    }

    // from_world hands the runtime a world that is already populated, in the
    // Created state so it can be started straight away.
    #[test]
    fn from_world_adopts_the_world_ready_to_start() {
        let mut world = World::new();
        world.add_component(AppConfig {
            home: String::new(),
            max_memory_mb: 128,
            job_threads: 1,
        });

        let mut runtime = Runtime::from_world(world);
        assert!(runtime.world().query::<AppConfig>().next().is_some());
        runtime.start().expect("an adopted world starts");
    }

    // `home` picks where the running runtime writes. An absolute path is taken
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

        assert_eq!(resolve_home("state", state), Some(state.join("state")));
        assert_eq!(
            resolve_home(absolute, state),
            Some(std::path::PathBuf::from(absolute))
        );
        // An absolute home needs no content root behind it.
        assert_eq!(
            resolve_home(absolute, std::path::Path::new("")),
            Some(std::path::PathBuf::from(absolute))
        );
    }

    // A relative `home` with nothing to resolve against is declined rather than
    // anchored to the working directory, so the host's own choice stands.
    #[test]
    fn a_relative_home_without_a_content_root_resolves_to_nothing() {
        assert_eq!(resolve_home("state", std::path::Path::new("")), None);
    }

    // A world's `home` splits the writable root off the tree the host built,
    // leaving the content (and the blobs the runtime reads) where it was.
    #[test]
    fn an_app_config_home_moves_only_the_writable_root() {
        // Starting one with a tree anchors the process-wide runtime cache.
        let _guard = concinnity_testing::exclusive();
        let root = if cfg!(windows) {
            r"C:\apps\MyGame"
        } else {
            "/apps/MyGame"
        };
        let mut runtime = Runtime::new().in_tree(StateTree::at(root));
        runtime.world_mut().add_component(AppConfig {
            home: "state".to_string(),
            max_memory_mb: 0,
            job_threads: 0,
        });
        runtime.start().unwrap();

        let tree = runtime.state_tree().expect("the runtime kept its tree");
        assert_eq!(tree.content_root(), std::path::Path::new(root));
        assert_eq!(
            tree.saves_dir(),
            std::path::Path::new(root).join("state").join("saves")
        );
        assert_eq!(
            tree.data_dir(),
            std::path::Path::new(root).join("data"),
            "the world's home never moves what a build wrote"
        );
        assert_eq!(
            runtime.world().resource::<StateTree>(),
            Some(tree),
            "the systems are handed the same tree the runtime resolved"
        );
    }

    // A runtime with no tree touches no disk, and publishes nothing for the
    // systems to read: a world runs, everything it would persist does nothing,
    // including the runtime cache a previous one anchored.
    #[test]
    fn an_app_without_a_tree_publishes_none() {
        use concinnity_host::store::cache::{self, CacheEntryKind};

        let _guard = concinnity_testing::exclusive();
        let tmp = concinnity_testing::TempTree::new();
        cache::anchor(tmp.join("cache"));
        assert!(cache::store(CacheEntryKind::Shader, "k", b"v"));

        let mut runtime = Runtime::new();
        assert_eq!(runtime.primary_blob(), None);
        assert!(matches!(
            runtime.load_blob(),
            Err(StartupError::NoStateRoot)
        ));
        runtime.start().unwrap();
        assert!(runtime.world().resource::<StateTree>().is_none());
        assert!(
            !cache::store(CacheEntryKind::Shader, "k", b"v"),
            "the previous anchor no longer takes entries"
        );
        assert!(!cache::flush(), "nothing is left to write");
    }

    #[test]
    fn a_tree_without_a_build_reports_its_primary_blob_missing() {
        let tmp = concinnity_testing::TempTree::new();
        let mut runtime = Runtime::new().in_tree(StateTree::at(tmp.path()));
        let primary = runtime.primary_blob().expect("a tree names a primary blob");

        let error = runtime.load_blob().expect_err("there is no blob to load");
        assert!(
            matches!(&error, StartupError::MissingData { blob } if *blob == primary),
            "{error:?}"
        );
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
        let mut runtime = Runtime::new();
        runtime.start().unwrap();
        assert!(
            runtime.world().resource::<FrameRateCap>().is_none(),
            "no cap is published without a GraphicsConfig"
        );
        assert_eq!(runtime.world_step(), StepResult::Done);
    }
}
