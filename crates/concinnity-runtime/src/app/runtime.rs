//! The `Runtime` value: a world plus the loop state that drives it.

use crate::app::clock::SimClock;
use crate::app::pacing::FramePacer;
use concinnity_core::fanout::FanOut;
use concinnity_engine::budget::{MemoryBudget, ThreadBudget};
use concinnity_engine::components::AppConfig;
use concinnity_engine::ecs::{StepResult, World};
use concinnity_engine::error::WorldError;
use concinnity_host::store::paths::StateTree;
use concinnity_host::sys::{cpu, memory};
use concinnity_host::thread::jobs::{self, configure, pool};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::app::blob;
use crate::app::render_mode;
use crate::app::startup_error::StartupError;
use crate::lend::SYSTEMS;
use crate::render::system::RenderInitFailure;
use concinnity_engine::ecs::ActiveRenderBackend;
use concinnity_engine::input::mailbox::InputMailbox;
use concinnity_engine::launch::{LaunchArgs, LaunchRequest};
use concinnity_engine::platform::Platform;
use concinnity_engine::render::backend::RenderBackend;
use concinnity_engine::render::backend::render_mode::{RenderMode, Resolution};
use concinnity_engine::shutdown::ShutdownToken;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuntimeStatus {
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
    pacer: FramePacer,
    // Fixed-timestep accumulator; publishes the frame's `SimTiming` and
    // `FrameTime` resources before each world step (see `app::clock`).
    clock: SimClock,
    // The pipelined render half's go-ahead, waited on before each step so the
    // tick starts no earlier than its frame can be recorded. `None` when serial.
    frame_gate: Option<crate::app::frame_gate::FrameGate>,
    // What the launch asked the runtime to arm; published at every `start`, so a
    // world loaded later inherits it.
    launch: LaunchRequest,
    // How this world runs, resolved on first ask. Cached because the resolution
    // probes the GPU, and because the caller asks before `start` (a windowed
    // macOS run must activate NSApplication ahead of the first window) while
    // `start` itself needs the same answer once the columns it reads are gone.
    render_mode: Option<Resolution>,
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
            pacer: FramePacer::default(),
            clock: SimClock::default(),
            frame_gate: None,
            launch: LaunchRequest::default(),
            render_mode: None,
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

    // The launch arguments the run takes: the world's own when it holds any,
    // replacing the request's whole, as `LaunchRequest::publish` takes them.
    pub(crate) fn launch_args(&self) -> &LaunchArgs {
        self.world
            .resource::<LaunchArgs>()
            .unwrap_or(&self.launch.args)
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
    ///
    /// # Errors
    ///
    /// Returns an error if the blob at `path` or one of its overflow siblings
    /// cannot be read or decoded.
    pub fn from_blob(path: &Path) -> Result<Self, StartupError> {
        let mut runtime = Self::new();
        runtime.load_blob_from(path)?;
        runtime.state = StateTree::for_blob(path);
        Ok(runtime)
    }

    /// Load assets and blob payload data from the primary blob under this runtime's
    /// state tree, and populate the world. Replaces any previously loaded
    /// world.
    ///
    /// # Errors
    ///
    /// Returns [`StartupError::NoStateRoot`] when the runtime has no tree to
    /// read from, or an error if the blob cannot be read or decoded.
    pub fn load_blob(&mut self) -> Result<(), StartupError> {
        let primary = self.primary_blob().ok_or(StartupError::NoStateRoot)?;
        self.load_blob_from(&primary)?;
        Ok(())
    }

    /// The primary blob this runtime reads: blob 0 under its tree's `data/`.
    /// `None` for a runtime with no tree.
    pub fn primary_blob(&self) -> Option<PathBuf> {
        self.state
            .as_ref()
            .map(|tree| concinnity_host::store::blob::primary_in(&tree.data_dir()))
    }

    // `load_blob` against a primary blob file named directly, returning the
    // world's highest blob index so the caller can check the layout it resolved
    // can actually hold it.
    pub(crate) fn load_blob_from(&mut self, primary: &Path) -> Result<u32, StartupError> {
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
        world.insert_resource(concinnity_engine::ecs::BlobSceneGroups(scene_groups));
        world.insert_resource(concinnity_engine::ecs::BlobMeshBounds(mesh_bounds));
        // Absent for a world with no physics content, which is also a world
        // with no PhysicsSystem to read it.
        if let Some(budget) = physics_budget {
            world.insert_resource(concinnity_engine::ecs::WorldPhysicsBudget(budget));
        }
        // Load the blob's resource stream into the per-kind tables the systems
        // read by handle. AudioSystem reads the AudioClipTable at init; the
        // renderer reads the TextureTable to build its shared texture pool.
        concinnity_engine::resource::install_tables(&mut world, &mut resources);
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
    ///
    /// # Errors
    ///
    /// Returns an error if the runtime has already started, no renderer can
    /// serve the resolved render mode, or a system's `init` fails.
    pub fn start(&mut self) -> Result<(), WorldError> {
        if self.status != RuntimeStatus::Created {
            return Err(WorldError::AlreadyStarted);
        }
        for (asset, clamped) in self.world.take_clamped_params() {
            tracing::warn!(
                "{asset}.{}: {} is outside its range, clamped to {}",
                clamped.path(),
                clamped.from,
                clamped.to
            );
        }
        self.install_home();
        self.publish_state_tree();
        self.install_budgets();
        self.launch.publish(&mut self.world);
        // Before `world.start`, which both drains the columns the resolution
        // reads and runs the gates that read the result.
        let mode = self
            .render_resolution()
            .startable()
            .map_err(WorldError::RenderUnavailable)?;
        self.world.insert_resource(mode);
        self.world.start(SYSTEMS)?;
        self.status = RuntimeStatus::Started;
        // A world that resolved to a windowed run and then could not build one
        // has a GPU that refused, not a machine without one. Report it rather
        // than leaving the loop to spin over a renderer that draws nothing.
        if let Some(failure) = self.world.remove_resource::<RenderInitFailure>() {
            return Err(WorldError::RenderUnavailable(failure.0));
        }
        Ok(())
    }

    /// How this world runs: with a window and a renderer, or with neither.
    ///
    /// Resolved on the first call and cached, so the GPU is probed once. Ask
    /// before [`start`](Runtime::start): several of the columns the resolution
    /// reads are drained there.
    pub fn render_mode(&mut self) -> RenderMode {
        self.render_resolution().mode()
    }

    /// How this world resolved, including a world with something to draw that
    /// this build or machine cannot draw, which [`start`](Runtime::start)
    /// refuses. Cached with [`render_mode`](Runtime::render_mode).
    pub fn render_resolution(&mut self) -> Resolution {
        if self.render_mode.is_none() {
            // Before the first probe, which runs on the graphics API this
            // process renders with: the launch's, else the player's choice
            // from the settings the world's home holds.
            match self.launch_graphics_api() {
                Some(api) => {
                    crate::platform::select(Some(api));
                }
                None => {
                    crate::platform::select_persisted(self.writable_tree().as_ref());
                }
            }
        }
        *self
            .render_mode
            .get_or_insert_with(|| render_mode::resolve_and_report(&self.world))
    }

    // The graphics API the launch asks for.
    fn launch_graphics_api(&self) -> Option<Platform> {
        self.launch_args().graphics_api
    }

    // Point the runtime-writable state (`settings`, `saves/`, `crashes/`, the
    // shader caches) at the world's `AppConfig.home`. Runs before
    // `world.start(SYSTEMS)`, which is where the systems that capture a save
    // directory are built, and before anything reads the settings file. An
    // empty `home` leaves the tree the host built, which is what keeps the
    // writability redirect a shipped player performs for itself in force.
    fn install_home(&mut self) {
        match self.home_dir() {
            Ok(None) => {}
            Ok(Some(dir)) => {
                tracing::info!("Writable state: {}", dir.display());
                self.state = self.state.take().map(|tree| tree.with_writable(dir));
            }
            Err(why) => tracing::warn!("{why}"),
        }
    }

    // The directory the world's `AppConfig.home` names, resolved against the
    // state tree: `None` for no home, an error saying why one cannot be used.
    fn home_dir(&self) -> Result<Option<PathBuf>, String> {
        let Some(home) = self
            .world
            .query::<AppConfig>()
            .next()
            .map(|c| c.home.clone())
            .filter(|h| !h.is_empty())
        else {
            return Ok(None);
        };
        let Some(tree) = self.state.as_ref() else {
            return Err(format!(
                "AppConfig home '{home}' has no state tree to resolve against; \
                 the runtime writes nowhere"
            ));
        };
        tree.resolve_home(&home).map(Some).ok_or_else(|| {
            format!(
                "AppConfig home '{home}' is relative but the state tree has no root; \
                 leaving writable state where it is"
            )
        })
    }

    // The state tree as `install_home` will leave it, for a read of the
    // writable state before the runtime starts.
    fn writable_tree(&self) -> Option<StateTree> {
        match self.home_dir() {
            Ok(Some(dir)) => self.state.clone().map(|tree| tree.with_writable(dir)),
            _ => self.state.clone(),
        }
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
        let config = self
            .world
            .query::<AppConfig>()
            .next()
            .cloned()
            .unwrap_or_default();

        let threads = ThreadBudget::compute(
            cpu::available_cores(),
            jobs::default_threads(),
            config.job_threads,
            config.frame_priority,
        );
        let memory = MemoryBudget::compute(memory::total_physical_bytes(), config.max_memory_mb);

        let sized = configure(threads.job_threads, threads.frame_priority);
        let job_workers = pool().workers();
        let job_priority = pool().priority();
        if !sized && (job_workers, job_priority) != (threads.job_threads, threads.frame_priority) {
            tracing::warn!(
                "job pool already built with {job_workers} worker(s) at {job_priority:?}; the requested {} at {:?} cannot take effect",
                threads.job_threads,
                threads.frame_priority
            );
        }
        tracing::info!(
            "Thread budget: {} core(s), {} job worker(s), frame priority {:?}{}",
            threads.total_cores,
            threads.job_threads,
            threads.frame_priority,
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

    /// Take the live render backend out of the world's parked slot, leaving
    /// the world backend-less. A host hands it to a rebuilt world as a
    /// [`PendingBackend`](crate::live_edit::PendingBackend), so the rebuild
    /// keeps the OS window and GPU device. `None` when the world never built a
    /// backend (or it was already taken).
    pub fn take_render_backend(&mut self) -> Option<Box<dyn RenderBackend>> {
        self.world
            .resource_mut::<ActiveRenderBackend>()
            .and_then(|slot| slot.0.take())
    }

    /// Replace the current world and reset to Created so `start()` can be called again.
    /// Used to load a new scene at runtime.
    pub fn load_world(&mut self, world: World) {
        self.world = world;
        self.status = RuntimeStatus::Created;
        // The new world decides for itself whether it opens a window.
        self.render_mode = None;
    }

    /// Advance the world one frame, for a caller that drives its own outer
    /// loop: the run loop that interleaves platform event pumps, and a host
    /// application whose OS owns the loop and calls this per display refresh.
    ///
    /// The FPS-cap pacer holds the step's start to its target interval first.
    /// A pipelined runtime then waits for the render half's go-ahead, which
    /// carries the input sampled when it went (once the GPU had a slot for
    /// this step's frame, while frames wait on it). Only then does the
    /// simulation clock publish the frame's fixed-tick budget and real frame
    /// time. The menu state read is the previous frame's, the same one-frame
    /// lag the pacer's clamp accepts.
    pub fn world_step(&mut self) -> StepResult {
        self.pacer.pace(&self.world);
        if let Some(gate) = &self.frame_gate {
            let Some((input, sampled_us)) = gate.wait() else {
                return StepResult::Stop;
            };
            InputMailbox::deposit_in(self.world.context().resources, input, sampled_us);
        }
        self.clock.publish(
            &mut self.world,
            Instant::now(),
            self.launch.args.fixed_frame_rate,
            self.launch.args.start_at(),
        );
        self.world.step()
    }

    // Step only on the pipelined render half's go-ahead from here on.
    pub(crate) fn gate_steps_on(&mut self, gate: crate::app::frame_gate::FrameGate) {
        self.frame_gate = Some(gate);
    }

    /// Run this on the run loop with default options, consuming it.
    ///
    /// # Errors
    ///
    /// Returns an error if the runtime fails to start (see
    /// [`start`](Self::start)).
    pub fn run(self) -> Result<(), WorldError> {
        crate::app::run::start_runtime(self)
    }
}

#[cfg(test)]
mod tests;
