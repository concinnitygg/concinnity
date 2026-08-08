// concinnity-cli/src/entry.rs
//
// The `concinnity` dev CLI: parse argv and dispatch build / add / rm / list /
// test / run / debug / editor / export. `run` is the entry point main() calls;
// `cn run` is delegated to the runtime (concinnity-engine), and every other
// command drives the compile pipeline (concinnity-cook), the authoring API,
// and the dev-session entry points owned by concinnity-editor.

use crate::cli;
use concinnity_editor::{WatchTarget, debug_client};
use concinnity_engine::app::dev_flags;

use clap::{Parser, Subcommand};

const BANNER: &str = r#"
   ______                                
  / ____/___  ____  ___________  ____  __________  __
 / /   / __ \/ __ \/ ___/ / __ \/ __ \/ /_  __/ / / /
/ /___/ /_/ / / / / /__/ / / / / / / / / / / / /_/ /
\____/\____/_/ /_/\___/_/_/ /_/_/ /_/_/ /_/  \__, /
                                            /____/"#;

#[derive(Subcommand, Debug)]
enum Commands {
    /// Create a new app in the current directory
    #[command(name = "init")]
    Init,

    /// Create a new app in a new directory
    #[command(name = "new")]
    New(NewArgs),

    /// Build a world from .concinnity/worlds/ into binary blobs
    #[command(name = "build")]
    Build(BuildArgs),

    /// Run a compiled world
    //
    // Production path: no debug server and no WebSocket command channel.
    // A shipped run is neither remotely inspectable nor remotely driven:
    // use `cn debug` for that.
    #[command(name = "run")]
    Run(RunArgs),

    /// Run interpreted directly from a world jsonl file
    //
    // Compiles the world in memory (no prior `cn build` needed) and stands
    // up the localhost debug server.
    // This is the development run, and the path the agentic loop / a host UI
    // use when they need to read or drive runtime state over a WebSocket.
    #[command(name = "debug")]
    Debug(DebugArgs),

    /// Edit a compiled world in-engine with a save-back HUD
    //
    // Reads the compiled blobs (a prior `cn build` is required), overlays the
    // editor HUD (a SAVE button plus an add-asset button), and persists edits by
    // recompiling world.jsonl. No WebSocket command channel unless --debug-port
    // is given (which stands up the same debug server `cn debug` uses, so
    // `cn debug smoke` / `screenshot` can drive an editor session).
    #[command(name = "editor")]
    Editor(EditorArgs),

    /// Add an asset to the active world
    //
    // TARGET can be:
    //   - A file path  (shaders/pbr.vert, models/scene.obj)
    //     Type is inferred from the file extension or the JSON `type` field.
    //   - A type name  (Logger, LLM, HttpServer, VulkanRenderer, ...)
    //     Asset is created with the type's registered default args.
    #[command(name = "add")]
    Add(AddArgs),

    /// Remove an asset from the active world by its unique name
    //
    // NAME is the value of the `name` field in world.jsonl
    // (e.g. "my_llm", "pbr_vert", "tool_agent").
    #[command(name = "rm")]
    Rm(RmArgs),

    /// List all declared assets
    #[command(name = "list")]
    List(ListArgs),

    /// Print an asset's effective entry from the expanded world
    //
    // Prints the full JSONL line for NAME as the build sees it, including
    // assets that only exist through build-time expansion or injection. The
    // output can be pasted into world.jsonl verbatim to override a default.
    #[command(name = "explain")]
    Explain(ExplainArgs),

    /// Regenerate the asset reference pages under docs/assets
    //
    // Reads the asset schema and its rustdoc out of the engine source tree, so
    // this runs against a checkout of the engine itself, not an app.
    #[command(name = "docs")]
    Docs(DocsArgs),

    /// Validate a world without building
    #[command(name = "test")]
    Test(TestArgs),

    /// Package a built world into a distributable app
    #[command(name = "export")]
    Export(ExportArgs),
}

#[derive(Parser, Debug)]
#[command(name = "concinnity")]
#[command(about = BANNER, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, clap::Args)]
pub struct DebugArgs {
    /// Query or drive a running `cn debug` server instead of starting one
    // When present, `cn debug` acts as a client: it connects to an already
    // running server's localhost WebSocket. When absent, `cn debug` starts the
    // server (the interpreted run below).
    #[command(subcommand)]
    pub client: Option<DebugClientCommand>,

    /// Path to a world JSONL file (default: discover from .concinnity/worlds/)
    #[arg(short = 'f', long)]
    pub file: Option<String>,

    /// Base HTTP URL of the infra server used to fetch missing asset files
    // Defaults to the value in the client config (~/.config/concinnity/config.json).
    #[arg(long)]
    pub server: Option<String>,

    /// Account ID for asset fetching authentication
    // Defaults to the value in the client config.
    #[arg(long)]
    pub user: Option<String>,

    /// Port for the localhost runtime debug server (default 8777)
    #[arg(long)]
    pub debug_port: Option<u16>,

    /// Enable graphics API validation. Omitted defaults to on for debug builds
    // and off for release. See `RunArgs::validation`.
    #[arg(long)]
    pub validation: Option<bool>,
}

// Client-side `cn debug` subcommands. Each connects to a running server's
// localhost WebSocket, sends one request, and prints the reply; the transport
// lives in `debug::client` (re-exported from `debug::wire`).
#[derive(Subcommand, Debug)]
pub enum DebugClientCommand {
    /// Send one raw JSON command and print the reply
    #[command(name = "send")]
    Send(DebugSendArgs),

    /// Headless render-loop liveness check (pass/fail)
    #[command(name = "smoke")]
    Smoke(DebugSmokeArgs),

    /// Capture the last presented frame to a PNG
    #[command(name = "screenshot")]
    Screenshot(DebugScreenshotArgs),

    /// Poll a read-only snapshot and print it until Ctrl-C
    #[command(name = "watch")]
    Watch(DebugWatchArgs),
}

#[derive(Debug, clap::Args)]
pub struct DebugSendArgs {
    /// Raw JSON object including its own "cmd" field
    // e.g. '{"cmd":"state"}' or '{"cmd":"emitter-remove","id":0}'
    pub json: String,

    /// Debug server port
    #[arg(long, default_value_t = 8777)]
    pub port: u16,
}

#[derive(Debug, clap::Args)]
pub struct DebugSmokeArgs {
    /// Seconds to wait for the render loop to start
    // The first build bakes a 4K HDRI and is slow, hence the generous default.
    #[arg(long, default_value_t = 240)]
    pub wait: u64,

    /// Stop the client after probing so no window is left running
    #[arg(long)]
    pub shutdown: bool,

    /// Debug server port
    #[arg(long, default_value_t = 8777)]
    pub port: u16,
}

#[derive(Debug, clap::Args)]
pub struct DebugScreenshotArgs {
    /// Output PNG path (resolved to absolute)
    pub path: String,

    /// Debug server port
    #[arg(long, default_value_t = 8777)]
    pub port: u16,
}

// The argv face of the library's `WatchTarget`: the value-enum derive lives
// here so concinnity-editor carries no clap dependency.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum WatchTargetArg {
    Camera,
    State,
    Streaming,
    Profile,
}

impl From<WatchTargetArg> for WatchTarget {
    fn from(t: WatchTargetArg) -> Self {
        match t {
            WatchTargetArg::Camera => WatchTarget::Camera,
            WatchTargetArg::State => WatchTarget::State,
            WatchTargetArg::Streaming => WatchTarget::Streaming,
            WatchTargetArg::Profile => WatchTarget::Profile,
        }
    }
}

#[derive(Debug, clap::Args)]
pub struct DebugWatchArgs {
    /// Which read-only snapshot to poll
    #[arg(value_enum, default_value = "camera")]
    pub target: WatchTargetArg,

    /// Milliseconds between polls
    #[arg(long, default_value_t = 500)]
    pub interval: u64,

    /// Debug server port
    #[arg(long, default_value_t = 8777)]
    pub port: u16,
}

#[derive(Debug, clap::Args)]
pub struct EditorArgs {
    /// Path to a world JSONL file (default: discover from .concinnity/worlds/)
    #[arg(short = 'f', long)]
    pub file: Option<String>,

    /// Start the localhost debug server on this port alongside the editor
    // Absent leaves the editor without a WebSocket channel; present makes an
    // editor session inspectable/drivable (e.g. `cn debug smoke`, `screenshot`).
    #[arg(long)]
    pub debug_port: Option<u16>,

    /// Enable graphics API validation. Omitted defaults to on for debug builds
    // and off for release. See `RunArgs::validation`.
    #[arg(long)]
    pub validation: Option<bool>,
}

#[derive(Debug, clap::Args)]
pub struct RunArgs {
    /// Enable graphics API validation. Omitted defaults to on for debug builds
    // The DirectX / Vulkan debug layers, or on macOS the Metal API-validation
    // layer (the process re-execs once with `MTL_DEBUG_LAYER` set, since Metal
    // cannot toggle it from inside a running process). Omitted defaults to on for
    // debug builds and off for release; pass `--validation false` to force it off
    // in a debug build. The heavier Metal shader validation is not enabled by
    // this flag; set `MTL_SHADER_VALIDATION=1` in the environment for that.
    #[arg(long)]
    pub validation: Option<bool>,
}

#[derive(Debug, clap::Args)]
pub struct AddArgs {
    /// Path to an asset file or type name
    pub target: String,

    /// Override the asset name written into the world
    // If omitted, the name is derived from the filename (including extension).
    #[arg(short, long)]
    pub name: Option<String>,

    /// Named scaffold preset used when bootstrapping a new world
    // Currently only "minimal-3d-world" (a camera, sun, room, and sky on top of
    // the base scaffold). Ignored when scaffolding doesn't fire.
    #[arg(short = 't', long)]
    pub template: Option<String>,
}

#[derive(Debug, clap::Args)]
pub struct RmArgs {
    /// The `name` field of the asset to remove
    pub name: String,
}

#[derive(Debug, clap::Args)]
pub struct TestArgs {
    /// Path to a world JSONL file (default: discover from .concinnity/worlds/)
    #[arg(short = 'f', long)]
    pub file: Option<String>,
}

#[derive(Debug, clap::Args)]
pub struct ListArgs {
    /// Path to a world JSONL file (default: discover from .concinnity/worlds/)
    #[arg(short = 'f', long)]
    pub file: Option<String>,

    /// List the expanded world the build produces
    // build-time macros are expanded and injected defaults included,
    // each row tagged with its provenance (authored / injected / expanded).
    #[arg(long)]
    pub expanded: bool,

    /// List the systems this world runs, in order, each with the condition
    // that includes it. Builds the world and reports `World::system_manifest()`
    // -- the same gates the runtime runs at start -- so it cannot drift.
    #[arg(long)]
    pub systems: bool,
}

#[derive(Debug, clap::Args)]
pub struct ExplainArgs {
    /// The `name` field of the asset to print
    pub name: String,

    /// Path to a world JSONL file (default: discover from .concinnity/worlds/)
    #[arg(short = 'f', long)]
    pub file: Option<String>,
}

#[derive(Debug, clap::Args)]
pub struct DocsArgs {
    /// Engine repository root to read sources from and write pages into
    #[arg(long, default_value = ".")]
    pub root: Option<String>,
}

#[derive(Debug, clap::Args)]
pub struct NewArgs {
    /// Directory to create the project in
    pub path: String,
}

#[derive(Debug, clap::Args)]
pub struct BuildArgs {
    /// Path to a world JSONL file (default: discover from .concinnity/worlds/)
    #[arg(short = 'f', long)]
    pub file: Option<String>,
}

#[derive(Debug, clap::Args)]
pub struct ExportArgs {
    /// Path to a world JSONL file (default: discover from .concinnity/worlds/)
    #[arg(short = 'f', long)]
    pub file: Option<String>,

    /// Override the application name
    #[arg(short = 'n', long)]
    pub name: Option<String>,

    /// Override the application version
    #[arg(long)]
    pub version: Option<String>,

    /// Target platform
    #[arg(long)]
    pub platform: Option<String>,

    /// Output directory for the exported app
    #[arg(long, default_value = "dist")]
    pub out: String,

    /// Output format: zip (default) or dir
    #[arg(long, default_value = "zip")]
    pub format: String,

    /// Also produce a .dmg wrapping the .app (macOS-only)
    #[arg(long)]
    pub dmg: bool,
}

// When a render command requests graphics validation on macOS, relaunch the
// process with Metal's API-validation layer (`MTL_DEBUG_LAYER`) set in the
// environment, then return into the replacement image. Metal reads that
// variable during early framework initialisation, so it cannot be toggled from
// a process that has already touched Metal -- and `std::env::set_var` is
// unsound once worker threads exist (the frameworks call `getenv` off-thread).
// Re-exec sidesteps both: the child starts with the variable present from PID
// birth. DirectX / Vulkan take the request through `dev_flags` and need no
// relaunch, so this is a macOS-only concern.
//
// The heavier `MTL_SHADER_VALIDATION` is deliberately left off: it is far more
// expensive and its memory footprint climbs over a long run, so it stays an
// explicit manual opt-in rather than riding a flag that defaults on in debug
// builds.
#[cfg(target_os = "macos")]
fn reexec_with_metal_validation(cli: &Cli) {
    use std::os::unix::process::CommandExt;

    // Only the rendering commands create a Metal context. A `cn debug` client
    // subcommand starts no renderer, so it needs no validation re-exec.
    let requested = match &cli.command {
        Commands::Run(args) => args.validation,
        Commands::Debug(args) if args.client.is_none() => args.validation,
        Commands::Editor(args) => args.validation,
        _ => return,
    };
    if !requested.unwrap_or(cfg!(debug_assertions)) {
        return;
    }
    // The relaunched child inherits the variable, so the guard is
    // self-terminating: it stops the second pass from re-execing again.
    if std::env::var_os("MTL_DEBUG_LAYER").is_some() {
        return;
    }
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("validation: cannot locate current executable to re-exec: {e}");
            return;
        }
    };
    // `exec` replaces this image in place (no lingering parent process) and only
    // returns on failure. On failure we fall through and run without Metal
    // validation rather than aborting the user's session.
    let err = std::process::Command::new(exe)
        .args(std::env::args_os().skip(1))
        .env("MTL_DEBUG_LAYER", "1")
        .exec();
    eprintln!("validation: failed to re-exec with Metal validation enabled: {err}");
}

#[cfg(not(target_os = "macos"))]
fn reexec_with_metal_validation(_cli: &Cli) {}

pub fn run() -> std::io::Result<()> {
    let cli = Cli::parse();

    // Must run before any thread spawns or the Metal framework initialises.
    reexec_with_metal_validation(&cli);

    // Give the cook this build's shader compilers. `cn build` reaches the
    // compile pipeline directly rather than through the authoring API, so
    // installing once here is what covers every subcommand that compiles.
    concinnity_shader::install();

    match &cli.command {
        Commands::Init => cli::init(),
        Commands::New(args) => cli::new(&args.path),
        Commands::Build(args) => cli::build(args.file.as_deref()),
        Commands::Run(args) => {
            dev_flags::set_validation(args.validation);
            concinnity_engine::app::run()
        }
        // A client subcommand talks to an already running server; its absence
        // means start the server (the interpreted run below).
        Commands::Debug(args) => match &args.client {
            Some(DebugClientCommand::Send(a)) => debug_client::send(a.port, &a.json),
            Some(DebugClientCommand::Smoke(a)) => debug_client::smoke(a.port, a.wait, a.shutdown),
            Some(DebugClientCommand::Screenshot(a)) => debug_client::screenshot(a.port, &a.path),
            Some(DebugClientCommand::Watch(a)) => {
                debug_client::watch(a.port, a.target.into(), a.interval)
            }
            None => {
                dev_flags::set_enabled(true);
                dev_flags::set_validation(args.validation);
                let port = args.debug_port.unwrap_or(8777);
                concinnity_editor::run_debug(args.file.as_deref(), port)
            }
        },
        Commands::Editor(args) => {
            dev_flags::set_validation(args.validation);
            // The editor is a dev session: arm the same dev flags the
            // `cn debug` host sets (above) so init captures hot-reload
            // sources and the backend takes its disk-first shader path.
            // Asset + shader hot-reload then works in every editor session;
            // a debug port only adds the WS probe surface on top.
            dev_flags::set_enabled(true);
            concinnity_editor::run_editor(args.file.as_deref(), args.debug_port)
        }
        Commands::Add(args) => {
            cli::add(args.name.as_deref(), &args.target, args.template.as_deref())
        }
        Commands::Rm(args) => cli::rm(&args.name),
        Commands::List(args) => cli::list(args.file.as_deref(), args.expanded, args.systems),
        Commands::Explain(args) => cli::explain(&args.name, args.file.as_deref()),
        Commands::Docs(args) => cli::docs(args.root.as_deref()),
        Commands::Test(args) => {
            let path = args.file.as_deref().unwrap_or("");
            cli::check(path)
        }
        Commands::Export(args) => cli::export(
            args.file.as_deref(),
            args.name.as_deref(),
            args.version.as_deref(),
            args.platform.as_deref(),
            &args.out,
            &args.format,
            args.dmg,
        ),
    }
}

// Argument-parsing tests. `main` itself (the engine-launching dispatch and the
// Metal re-exec) is exercised by the `tests/cli.rs` integration tests, which
// run the built binary; these cover the clap surface -- the command tree, the
// per-subcommand defaults, and the value-enum parsing -- without a process.
#[cfg(test)]
mod tests {
    use super::*;
    use clap::{CommandFactory, Parser};

    // Walks the whole command tree and asserts clap's own invariants (no
    // conflicting args, valid defaults, unique names). A cheap guard against
    // an ill-formed derive that would only surface at runtime otherwise.
    #[test]
    fn cli_config_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn debug_without_subcommand_starts_the_server() {
        let cli = Cli::try_parse_from(["concinnity", "debug", "-f", "world.jsonl"]).unwrap();
        let Commands::Debug(a) = cli.command else {
            panic!("expected debug");
        };
        assert!(a.client.is_none());
        assert_eq!(a.file.as_deref(), Some("world.jsonl"));
    }

    #[test]
    fn debug_send_parses_json_with_default_port() {
        let cli =
            Cli::try_parse_from(["concinnity", "debug", "send", r#"{"cmd":"state"}"#]).unwrap();
        let Commands::Debug(a) = cli.command else {
            panic!("expected debug");
        };
        let Some(DebugClientCommand::Send(s)) = a.client else {
            panic!("expected send");
        };
        assert_eq!(s.json, r#"{"cmd":"state"}"#);
        assert_eq!(s.port, 8777);
    }

    #[test]
    fn debug_smoke_defaults() {
        let cli = Cli::try_parse_from(["concinnity", "debug", "smoke"]).unwrap();
        let Commands::Debug(a) = cli.command else {
            panic!("expected debug");
        };
        let Some(DebugClientCommand::Smoke(s)) = a.client else {
            panic!("expected smoke");
        };
        assert_eq!(s.wait, 240);
        assert!(!s.shutdown);
        assert_eq!(s.port, 8777);
    }

    #[test]
    fn debug_smoke_flags_override_defaults() {
        let cli = Cli::try_parse_from([
            "concinnity",
            "debug",
            "smoke",
            "--wait",
            "30",
            "--shutdown",
            "--port",
            "8799",
        ])
        .unwrap();
        let Commands::Debug(a) = cli.command else {
            panic!("expected debug");
        };
        let Some(DebugClientCommand::Smoke(s)) = a.client else {
            panic!("expected smoke");
        };
        assert_eq!(s.wait, 30);
        assert!(s.shutdown);
        assert_eq!(s.port, 8799);
    }

    #[test]
    fn debug_screenshot_parses_path() {
        let cli = Cli::try_parse_from(["concinnity", "debug", "screenshot", "out.png"]).unwrap();
        let Commands::Debug(a) = cli.command else {
            panic!("expected debug");
        };
        let Some(DebugClientCommand::Screenshot(s)) = a.client else {
            panic!("expected screenshot");
        };
        assert_eq!(s.path, "out.png");
        assert_eq!(s.port, 8777);
    }

    #[test]
    fn debug_watch_defaults_to_camera() {
        let cli = Cli::try_parse_from(["concinnity", "debug", "watch"]).unwrap();
        let Commands::Debug(a) = cli.command else {
            panic!("expected debug");
        };
        let Some(DebugClientCommand::Watch(w)) = a.client else {
            panic!("expected watch");
        };
        assert!(matches!(w.target, WatchTargetArg::Camera));
        assert_eq!(w.interval, 500);
        assert_eq!(w.port, 8777);
    }

    #[test]
    fn debug_watch_accepts_named_target() {
        let cli = Cli::try_parse_from(["concinnity", "debug", "watch", "streaming"]).unwrap();
        let Commands::Debug(a) = cli.command else {
            panic!("expected debug");
        };
        let Some(DebugClientCommand::Watch(w)) = a.client else {
            panic!("expected watch");
        };
        assert!(matches!(w.target, WatchTargetArg::Streaming));
    }

    #[test]
    fn debug_watch_rejects_unknown_target() {
        assert!(Cli::try_parse_from(["concinnity", "debug", "watch", "nonsense"]).is_err());
    }

    #[test]
    fn export_defaults() {
        let cli = Cli::try_parse_from(["concinnity", "export"]).unwrap();
        let Commands::Export(e) = cli.command else {
            panic!("expected export");
        };
        assert_eq!(e.out, "dist");
        assert_eq!(e.format, "zip");
        assert!(!e.dmg);
    }

    #[test]
    fn add_requires_a_target() {
        assert!(Cli::try_parse_from(["concinnity", "add"]).is_err());
        let cli = Cli::try_parse_from(["concinnity", "add", "Logger", "--name", "log"]).unwrap();
        let Commands::Add(a) = cli.command else {
            panic!("expected add");
        };
        assert_eq!(a.target, "Logger");
        assert_eq!(a.name.as_deref(), Some("log"));
    }

    #[test]
    fn missing_subcommand_is_an_error() {
        assert!(Cli::try_parse_from(["concinnity"]).is_err());
    }
}
