//! End-to-end tests that drive the built `concinnity` binary through the paths
//! that never touch the renderer: `--help`, a missing subcommand, the `cn debug`
//! client commands pointed at a dead server, and every authoring subcommand
//! (`build` / `add` / `rm` / `list` / `explain` / `test` / `docs` / `export` /
//! `init` / `new`). They exercise `fn main()` and the command dispatch (which the
//! in-crate unit tests cannot, since those never run the binary), plus the
//! world-discovery wrappers in `cli/`, whose fallbacks read process-global path
//! anchors that a unit test in the shared test binary could not redirect without
//! racing its neighbours. Under `cargo llvm-cov` the profile data the spawned
//! binary writes on exit is merged, so this coverage counts.
//!
//! Only the non-engine paths are driven here; `cn run` and a bare `cn debug`
//! stand up a renderer + window and are verified by screenshot probes instead.

use std::io::ErrorKind;
use std::net::{Ipv4Addr, SocketAddr, TcpStream};
use std::path::Path;
use std::process::{Command, Output};
use std::time::Duration;

// Path to the freshly built binary, provided by Cargo to integration tests.
const BIN: &str = env!("CARGO_BIN_EXE_concinnity");

// Candidate localhost ports for the "server is not running" tests. They sit
// below every platform's ephemeral range (Linux from 32768, macOS/Windows from
// 49152), which `bind(:0)` never allocates, so no concurrent test can occupy
// one -- avoiding the race in the older bind-an-ephemeral-port-then-drop-it
// trick, where the kernel could hand the freed port to a parallel test that
// then listens on it.
const DEAD_PORT_CANDIDATES: [u16; 4] = [28470, 28471, 28472, 28473];

// The first candidate that promptly refuses a connection (nothing listening),
// or `None` when the host refuses none. Callers skip when `None`: the exit-code
// paths they assert are platform-independent and still run on hosts where the
// connection is refused.
fn find_dead_port() -> Option<String> {
    DEAD_PORT_CANDIDATES
        .into_iter()
        .find(|&port| connect_refused(port))
        .map(|port| port.to_string())
}

// True when connecting to `127.0.0.1:port` is refused, i.e. nothing is
// listening. A connect that instead times out (a dropped SYN with no RST, as
// some firewalled loopback stacks do) returns false: the port is unused but not
// promptly refused, so it is not a usable dead port and the caller skips rather
// than wait out the client's multi-second connect timeout on every test.
fn connect_refused(port: u16) -> bool {
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    match TcpStream::connect_timeout(&addr, Duration::from_millis(250)) {
        Ok(_) => false,
        Err(e) => e.kind() == ErrorKind::ConnectionRefused,
    }
}

// Stable Rust has no runtime "skipped" test state, so a test whose host
// precondition is unmet prints this and returns (counting as passed). Visible
// under `cargo test -- --nocapture`.
fn skip_no_dead_port(test: &str) {
    eprintln!("[skip] {test}: no localhost port refuses connections on this host");
}

fn run(args: &[&str]) -> Output {
    Command::new(BIN)
        .args(args)
        .output()
        .expect("spawn concinnity binary")
}

// The starter world the authoring tests operate on. A lone TextLabel, which the
// build expands into a runnable world by injecting the renderer stack.
const HELLO_WORLD: &str =
    "{\"name\":\"hello_world\",\"type\":\"TextLabel\",\"args\":{\"content\":\"Hello, world!\"}}\n";

// An isolated project for one test: a temp directory the spawned binary runs
// in, so the `.concinnity/` it anchors to its working directory resolves inside
// it. Isolation is per-process rather than per-thread, which is what makes
// these tests safe to run in parallel: the path anchors they steer are
// process-global, so an in-crate unit test could not do the same without racing
// every other test in its binary.
//
// A temp root also keeps the discovery fallback honest. `find_world_jsonl` walks
// up from the working directory looking for a `world.jsonl`, and the repository
// root has one; running from a temp directory means an "empty project" really is
// empty.
struct Project {
    dir: tempfile::TempDir,
}

impl Project {
    // A project with no world at all, so discovery fails and each subcommand
    // takes its not-found path.
    fn empty() -> Self {
        Project {
            dir: tempfile::tempdir().expect("create temp project dir"),
        }
    }

    // A project whose discovered world holds `content`.
    fn with_world(content: &str) -> Self {
        let project = Project::empty();
        let worlds = project.path().join(".concinnity").join("worlds");
        std::fs::create_dir_all(&worlds).expect("create worlds dir");
        std::fs::write(worlds.join("main.jsonl"), content).expect("write world");
        project
    }

    fn path(&self) -> &Path {
        self.dir.path()
    }

    // The path of the world `with_world` wrote, for the subcommands that take an
    // explicit `--file` instead of discovering one.
    fn world_path(&self) -> String {
        self.path()
            .join(".concinnity")
            .join("worlds")
            .join("main.jsonl")
            .to_string_lossy()
            .into_owned()
    }

    fn cn(&self, args: &[&str]) -> Output {
        Command::new(BIN)
            .args(args)
            .current_dir(self.path())
            .output()
            .expect("spawn concinnity binary")
    }
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

// Assert success, surfacing the child's stderr on failure so a broken run is
// diagnosable from the test output alone.
fn expect_ok(out: &Output, what: &str) {
    assert!(out.status.success(), "{what} failed: {}", stderr(out));
}

#[test]
fn help_exits_zero() {
    let out = run(&["--help"]);
    assert!(out.status.success(), "--help should exit 0");
    assert!(String::from_utf8_lossy(&out.stdout).contains("Usage"));
}

#[test]
fn missing_subcommand_is_a_usage_error() {
    let out = run(&[]);
    assert!(
        !out.status.success(),
        "a bare invocation should be a usage error"
    );
}

#[test]
fn debug_send_invalid_json_exits_transport() {
    // Validation fails before any socket work, so the port is never used (any
    // value works; a dead port is not required, so this never skips).
    let port = DEAD_PORT_CANDIDATES[0].to_string();
    let out = run(&["debug", "send", "{not json", "--port", &port]);
    assert_eq!(out.status.code(), Some(3));
}

#[test]
fn debug_send_dead_port_exits_transport() {
    let Some(port) = find_dead_port() else {
        return skip_no_dead_port("debug_send_dead_port_exits_transport");
    };
    let out = run(&["debug", "send", r#"{"cmd":"state"}"#, "--port", &port]);
    assert_eq!(out.status.code(), Some(3));
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot connect"));
}

#[test]
fn debug_watch_dead_port_exits_transport_on_first_poll() {
    let Some(port) = find_dead_port() else {
        return skip_no_dead_port("debug_watch_dead_port_exits_transport_on_first_poll");
    };
    let out = run(&["debug", "watch", "camera", "--port", &port]);
    assert_eq!(out.status.code(), Some(3));
}

// The capture never happens, so no PNG is written; the point is that the
// screenshot client fails on the transport like its siblings rather than
// creating an empty file at the requested path.
#[test]
fn debug_screenshot_dead_port_exits_transport() {
    let Some(port) = find_dead_port() else {
        return skip_no_dead_port("debug_screenshot_dead_port_exits_transport");
    };
    let project = Project::empty();
    let shot = project.path().join("frame.png");
    let out = run(&[
        "debug",
        "screenshot",
        &shot.to_string_lossy(),
        "--port",
        &port,
    ]);
    assert_eq!(out.status.code(), Some(3));
    assert!(!shot.exists(), "a failed capture should write no file");
}

// `cn list` with no --file discovers the world under `.concinnity/worlds/`,
// the branch an explicit path never reaches.
#[test]
fn list_discovers_the_world_and_prints_the_asset_table() {
    let project = Project::with_world(HELLO_WORLD);
    let out = project.cn(&["list"]);
    expect_ok(&out, "cn list");

    let printed = stdout(&out);
    assert!(printed.contains("hello_world"), "got: {printed}");
    assert!(printed.contains("TextLabel"), "got: {printed}");
    assert!(printed.contains("1 asset(s)"), "got: {printed}");
}

// The expanded listing runs the build's front half, so it reports the injected
// companions the authored file never mentions, each tagged with its pass.
#[test]
fn list_expanded_tags_injected_assets_with_their_pass() {
    let project = Project::with_world(HELLO_WORLD);
    let out = project.cn(&["list", "--expanded"]);
    expect_ok(&out, "cn list --expanded");

    let printed = stdout(&out);
    assert!(printed.contains("hello_world"), "got: {printed}");
    assert!(printed.contains("authored"), "got: {printed}");
    assert!(printed.contains("injected:"), "got: {printed}");
    assert!(printed.contains("after expansion"), "got: {printed}");
}

// The system listing builds the world the way the runtime would and pairs each
// scheduled system with the condition that gates it in.
#[test]
fn list_systems_names_each_system_and_its_gate() {
    let project = Project::with_world(HELLO_WORLD);
    let out = project.cn(&["list", "--systems"]);
    expect_ok(&out, "cn list --systems");

    let printed = stdout(&out);
    assert!(printed.contains("system(s), in order"), "got: {printed}");
    assert!(printed.contains("GraphicsSystem"), "got: {printed}");
    // The gate column names the asset that pulled the system in.
    assert!(printed.contains("GraphicsConfig"), "got: {printed}");
}

// An injected asset has no line in the authored file, so `cn explain` printing a
// pasteable one is the only way to override it.
#[test]
fn explain_prints_an_injected_asset_as_a_pasteable_line() {
    let project = Project::with_world(HELLO_WORLD);
    let out = project.cn(&["explain", "debug_hud"]);
    expect_ok(&out, "cn explain debug_hud");

    let printed = stdout(&out);
    assert!(printed.contains("injected:debug_hud"), "got: {printed}");
    assert!(printed.contains("\"type\":\"DebugHud\""), "got: {printed}");
}

#[test]
fn explain_of_an_unknown_name_fails() {
    let project = Project::with_world(HELLO_WORLD);
    let out = project.cn(&["explain", "no_such_asset"]);
    assert!(!out.status.success(), "an unknown name should fail");
    assert!(stderr(&out).contains("no_such_asset"), "{}", stderr(&out));
}

// `cn test` validates without building; the world comes from discovery.
#[test]
fn test_validates_the_discovered_world() {
    let project = Project::with_world(HELLO_WORLD);
    let out = project.cn(&["test"]);
    expect_ok(&out, "cn test");
    assert!(stdout(&out).contains("passed"), "{}", stdout(&out));
}

#[test]
fn test_reports_an_invalid_world() {
    let project =
        Project::with_world("{\"name\":\"x\",\"type\":\"NotARealAssetType\",\"args\":{}}\n");
    let out = project.cn(&["test"]);
    assert!(!out.status.success(), "an unknown asset type should fail");
}

// A real compile, covering both ways `cn build` resolves its world: discovery
// when no --file is given, and an explicit path when one is. The second build
// reuses the first one's cache, so only one cook is paid for.
#[test]
fn build_compiles_the_discovered_world_and_an_explicit_one() {
    let project = Project::with_world(HELLO_WORLD);

    let out = project.cn(&["build"]);
    expect_ok(&out, "cn build");
    // The blobs and the lock file land inside the project, not the developer's tree.
    assert!(
        project
            .path()
            .join(".concinnity")
            .join("data")
            .join("0")
            .exists(),
        "no blob written"
    );
    assert!(
        project.path().join("world-lock.json").exists(),
        "no lock file written"
    );

    let out = project.cn(&["build", "-f", &project.world_path()]);
    expect_ok(&out, "cn build -f");
}

// Removing a name the world does not declare reports the ones it does, so a typo
// is self-correcting.
#[test]
fn rm_of_an_unknown_name_lists_the_known_names() {
    let project = Project::with_world(HELLO_WORLD);
    let out = project.cn(&["rm", "not_here"]);
    assert!(!out.status.success(), "removing an absent name should fail");

    let failure = stderr(&out);
    assert!(failure.contains("not_here"), "got: {failure}");
    assert!(failure.contains("hello_world"), "got: {failure}");
}

#[test]
fn add_of_an_unresolvable_target_fails() {
    let project = Project::with_world(HELLO_WORLD);
    let out = project.cn(&["add", "NotARealAssetType"]);
    assert!(!out.status.success(), "an unresolvable target should fail");
    assert!(
        stderr(&out).contains("could not resolve"),
        "{}",
        stderr(&out)
    );
}

// With nothing to discover, `cn add` falls back to `world.jsonl` in the working
// directory rather than propagating the not-found. The target still fails to
// resolve, but the failure names the fallback path, which is what distinguishes
// the two branches.
#[test]
fn add_without_a_discoverable_world_falls_back_to_world_jsonl() {
    let project = Project::empty();
    let out = project.cn(&["add", "NotARealAssetType"]);
    assert!(!out.status.success(), "an empty project should fail");
    assert!(
        stderr(&out).contains("no world found at 'world.jsonl'"),
        "{}",
        stderr(&out)
    );
}

// `cn docs` reads the asset prose out of the engine's own sources, so it needs
// no world but does need a checkout. The sources are copied into the temp root
// rather than pointing `--root` at the repository, so the run cannot write to
// the working tree.
#[test]
fn docs_writes_the_asset_reference_pages() {
    let project = Project::empty();
    for tree in [
        "crates/concinnity-asset/src",
        "crates/concinnity-core/src/components",
    ] {
        copy_dir(&repo_root().join(tree), &project.path().join(tree));
    }

    let root = project.path().to_string_lossy().into_owned();
    let out = project.cn(&["docs", "--root", &root]);
    expect_ok(&out, "cn docs");

    assert!(stdout(&out).contains("asset pages in"), "{}", stdout(&out));
    let pages = project.path().join("docs").join("assets");
    assert!(pages.join("TextLabel.md").exists(), "no TextLabel page");
    assert!(pages.join("Window.md").exists(), "no Window page");
}

// Without those sources there is nothing to read, and saying so beats writing an
// empty reference over the pages already on disk.
#[test]
fn docs_outside_an_engine_checkout_says_so() {
    let project = Project::empty();
    let root = project.path().to_string_lossy().into_owned();
    let out = project.cn(&["docs", "--root", &root]);

    assert!(!out.status.success(), "cn docs should fail with no sources");
    assert!(
        stderr(&out).contains("checkout of the engine"),
        "{}",
        stderr(&out)
    );
}

// The repository root, which is this package's own directory.
fn repo_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).expect("create the destination tree");
    for entry in std::fs::read_dir(from).expect("read the source tree") {
        let path = entry.expect("directory entry").path();
        let dest = to.join(path.file_name().expect("named entry"));
        if path.is_dir() {
            copy_dir(&path, &dest);
        } else {
            std::fs::copy(&path, &dest).expect("copy a source file");
        }
    }
}

// Export refuses a target that is not the host before it builds anything, so
// this costs no cook.
#[test]
fn export_rejects_a_foreign_platform() {
    let project = Project::with_world(HELLO_WORLD);
    let foreign = if cfg!(target_os = "windows") {
        "linux"
    } else {
        "windows"
    };
    let out = project.cn(&["export", "--platform", foreign]);
    assert!(!out.status.success(), "a foreign target should fail");
    assert!(
        stderr(&out).contains("cross-platform export is not supported"),
        "{}",
        stderr(&out)
    );
}

#[test]
fn init_skips_a_directory_that_already_has_a_world() {
    let project = Project::empty();
    std::fs::write(project.path().join("world.jsonl"), HELLO_WORLD).expect("write world");

    let out = project.cn(&["init"]);
    expect_ok(&out, "cn init");
    assert!(stdout(&out).contains("skipping init"), "{}", stdout(&out));
}

// The full scaffold: a new directory, the starter world written into it, and an
// initial build over it.
#[test]
fn new_scaffolds_and_builds_a_fresh_project() {
    let project = Project::empty();
    let target = project.path().join("fresh");
    let target_arg = target.to_string_lossy().into_owned();

    let out = project.cn(&["new", &target_arg]);
    expect_ok(&out, "cn new");
    assert!(target.join("world.jsonl").exists(), "no starter world");
    assert!(stdout(&out).contains("Created"), "{}", stdout(&out));
}

// A pre-existing but empty directory is a valid target: only an existing world
// in it is refused.
#[test]
fn new_accepts_a_pre_existing_empty_directory() {
    let project = Project::empty();
    let target = project.path().join("prepared");
    std::fs::create_dir_all(&target).expect("create target dir");

    let out = project.cn(&["new", &target.to_string_lossy()]);
    expect_ok(&out, "cn new into an empty directory");
    assert!(target.join("world.jsonl").exists(), "no starter world");
}

#[test]
fn new_refuses_a_directory_that_already_has_a_world() {
    let project = Project::empty();
    let target = project.path().join("taken");
    std::fs::create_dir_all(&target).expect("create target dir");
    std::fs::write(target.join("world.jsonl"), HELLO_WORLD).expect("write world");

    let out = project.cn(&["new", &target.to_string_lossy()]);
    assert!(!out.status.success(), "an occupied directory should fail");
    assert!(
        stderr(&out).contains("already contains"),
        "{}",
        stderr(&out)
    );
}
