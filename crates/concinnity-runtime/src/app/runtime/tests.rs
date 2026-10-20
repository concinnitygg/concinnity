use super::*;
use concinnity_engine::budget::{MemoryBudget, ThreadBudget};
use concinnity_engine::components::{AppConfig, FramePriority};
use concinnity_engine::ecs::FrameRateCap;

fn thread_budget(runtime: &Runtime) -> ThreadBudget {
    *runtime.world().resource().expect("thread budget published")
}

fn memory_budget(runtime: &Runtime) -> MemoryBudget {
    *runtime.world().resource().expect("memory budget published")
}

// Starting the runtime publishes the thread + memory budgets as world resources,
// honoring an `AppConfig`'s overrides. A world with nothing to draw starts
// without building a GPU, so this exercises the budget install in isolation.
#[test]
fn start_publishes_budgets_honoring_app_config_limits() {
    let mut runtime = Runtime::new();
    runtime.world_mut().add_component(AppConfig {
        home: String::new(),
        max_memory_mb: 512,
        job_threads: 2,
        frame_priority: FramePriority::default(),
        headless: false,
    });
    runtime.start().unwrap();

    let threads = thread_budget(&runtime);
    assert_eq!(threads.job_threads, 2.min(threads.total_cores));

    let memory = memory_budget(&runtime);
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

    let threads = thread_budget(&runtime);
    assert_eq!(
        threads.job_threads,
        threads.total_cores.saturating_sub(1).max(1)
    );
    let memory = memory_budget(&runtime);
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
        frame_priority: FramePriority::default(),
        headless: false,
    });
    runtime.load_world(world);

    assert!(
        runtime.world().query::<AppConfig>().next().is_some(),
        "the loaded world replaced the empty one"
    );
    runtime.start().expect("the reset status permits a restart");
    // The restart budgeted against the new world's limits, not the old one's.
    let memory = memory_budget(&runtime);
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

// A benchmark launch arms the world before its systems are built, so the
// report is scheduled; an ordinary one leaves a declared config inert.
#[test]
fn only_a_benchmark_launch_arms_the_world() {
    use concinnity_engine::benchmark::BenchmarkRun;
    use concinnity_engine::components::BenchmarkConfig;
    use concinnity_engine::launch::LaunchArgs;

    let declared = || {
        let mut world = World::new();
        world.add_component(BenchmarkConfig {
            warmup_seconds: 1.0,
            ..Default::default()
        });
        world
    };
    let mut plain = Runtime::from_world(declared());
    plain.start().unwrap();
    assert!(plain.world().resource::<BenchmarkRun>().is_none());

    let mut benchmarked = Runtime::from_world(declared()).with_launch(LaunchRequest {
        args: LaunchArgs {
            benchmark: true,
            ..LaunchArgs::NONE
        },
        ..Default::default()
    });
    benchmarked.start().unwrap();
    let run = benchmarked.world().resource::<BenchmarkRun>();
    assert_eq!(run.map(|r| r.config.warmup_seconds), Some(1.0));
}

// An app's own args, published on the world before start, replace the
// launch's and are what the run publishes.
#[test]
fn an_apps_args_replace_the_launchs() {
    use concinnity_engine::benchmark::BenchmarkRun;
    use concinnity_engine::launch::LaunchArgs;

    let mut world = World::new();
    let args = LaunchArgs {
        benchmark: true,
        frames: Some(9),
        ..LaunchArgs::NONE
    };
    world.insert_resource(args);
    let mut runtime = Runtime::from_world(world).with_launch(LaunchRequest {
        args: LaunchArgs {
            frames: Some(1),
            ..LaunchArgs::NONE
        },
        ..Default::default()
    });
    runtime.start().unwrap();
    let launch = runtime.world().resource::<LaunchRequest>().copied();
    assert_eq!(launch.map(|l| l.args), Some(args));
    assert!(runtime.world().resource::<LaunchArgs>().is_none());
    assert!(runtime.world().resource::<BenchmarkRun>().is_some());
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
        frame_priority: FramePriority::default(),
        headless: false,
    });

    let mut runtime = Runtime::from_world(world);
    assert!(runtime.world().query::<AppConfig>().next().is_some());
    runtime.start().expect("an adopted world starts");
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
        frame_priority: FramePriority::default(),
        headless: false,
    });
    runtime.start().unwrap();

    let tree = runtime.state_tree().expect("the runtime kept its tree");
    assert_eq!(tree.content_root(), Path::new(root));
    assert_eq!(
        tree.saves_dir(),
        Path::new(root).join("state").join("saves")
    );
    assert_eq!(
        tree.data_dir(),
        Path::new(root).join("data"),
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

// With no FrameRateCap published the pacer has nothing to hold the frame
// to, so the step runs straight through; an empty world reports Done as it
// has no systems left to run.
#[test]
fn world_step_without_a_frame_rate_cap_runs_unpaced() {
    let mut runtime = Runtime::new();
    runtime.start().unwrap();
    assert!(
        runtime.world().resource::<FrameRateCap>().is_none(),
        "no cap is published without a renderer"
    );
    assert_eq!(runtime.world_step(), StepResult::Done);
}

// A world that never built a backend has none to yield.
#[test]
fn a_runtime_without_a_backend_yields_none() {
    assert!(Runtime::new().take_render_backend().is_none());
}
