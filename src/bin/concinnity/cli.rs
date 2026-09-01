// The `concinnity` binary's argv face: the clap command tree, the value enums
// that mirror engine types so no library carries a clap dependency, and the
// macOS validation re-exec that has to happen before any thread starts.
//
// Parsing only. What each command does lives in concinnity-dev; `dispatch`
// is the one file that joins the two.
use concinnity_dev::WatchTarget;
use concinnity_engine::app::dev_flags;
use concinnity_engine::app::dev_flags::{QualityPreset, RtDynamicMode};

use clap::{Parser, Subcommand};

const BANNER: &str = r#"
   ______                                
  / ____/___  ____  ___________  ____  __________  __
 / /   / __ \/ __ \/ ___/ / __ \/ __ \/ /_  __/ / / /
/ /___/ /_/ / / / / /__/ / / / / / / / / / / / /_/ /
\____/\____/_/ /_/\___/_/_/ /_/_/ /_/_/ /_/  \__, /
                                            /____/"#;

#[derive(Subcommand, Debug)]
pub(crate) enum Commands {
    /// Create a new app in the current directory
    #[command(name = "init")]
    Init,

    /// Create a new app in a new directory
    #[command(name = "new")]
    New(NewArgs),

    /// Build a world from worlds/ into binary blobs
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

    /// Edit a world in-engine with a save-back HUD
    //
    // Compiles world.jsonl in memory (no prior `cn build` needed, and the blobs
    // one wrote are neither read nor refreshed here), overlays the editor HUD (a
    // SAVE button plus an add-asset button), and persists edits by writing
    // world.jsonl. No WebSocket command channel unless --debug-port is given
    // (which stands up the same debug server `cn debug` uses, so `cn debug send`
    // / `screenshot` can drive an editor session).
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

    /// Print the version
    #[command(name = "version")]
    Version,
}

#[derive(Parser, Debug)]
#[command(name = "concinnity")]
#[command(about = BANNER, long_about = None)]
// clap renders its own flag as "{name} {version}", which is the pair
// `command::version_line` prints, so `cn version` and `cn --version` render
// the same line off one source. The auto-generated flag is off only so the
// short can carry `-v` alongside clap's conventional `-V`.
#[command(version = concinnity_dev::command::version_details(), disable_version_flag = true)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Commands,

    /// Print the version
    #[arg(short = 'V', short_alias = 'v', long, action = clap::ArgAction::Version)]
    pub(crate) version: (),
}

// The argv face of the launch-time render knobs the engine reads through
// `dev_flags`. Each is a diagnostic: omitting it leaves the shipping behaviour,
// and none is persisted, so a probe run can force one without writing settings.
// Flattened into every command that launches a world (`run` / `debug` /
// `editor`); `arm` is what hands them to the engine before the world is built.
#[derive(Debug, Default, clap::Args)]
pub(crate) struct RenderArgs {
    /// Force the master graphics-quality preset for this launch, unpersisted
    // Outranks the settings-menu choice in .concinnity/settings. Only the
    // `ultra` ceiling permits ray-traced reflections, so an RT probe that omits
    // this measures a frame with RT clamped off and no log line saying so.
    #[arg(long, value_enum)]
    pub(crate) quality_preset: Option<QualityPresetArg>,

    /// How the ray-tracing acceleration structure tracks moving props
    // Omitted = `auto`, the dirty-gated TLAS rebuild a shipped run uses.
    #[arg(long, value_enum)]
    pub(crate) rt_dynamic: Option<RtDynamicArg>,

    /// Whether skinned meshes join the ray-tracing acceleration structure
    // Omitted = in. `--rt-skinned-geometry false` leaves the BVH over static +
    // instanced geometry only, which isolates the skinned trace path.
    #[arg(long)]
    pub(crate) rt_skinned_geometry: Option<bool>,
}

impl RenderArgs {
    // Hand the requests to the engine. Called before the world is built, since
    // graphics init reads them once while resolving the render settings.
    pub(crate) fn arm(&self) {
        dev_flags::set_quality_preset(self.quality_preset.map(Into::into));
        dev_flags::set_rt_dynamic(self.rt_dynamic.map(Into::into));
        dev_flags::set_rt_skinned_geometry(self.rt_skinned_geometry);
    }
}

// The argv face of the engine's `QualityPreset`: the value-enum derive lives
// here so concinnity-engine carries no clap dependency.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub(crate) enum QualityPresetArg {
    Auto,
    Low,
    Medium,
    High,
    Ultra,
    Custom,
}

impl From<QualityPresetArg> for QualityPreset {
    fn from(p: QualityPresetArg) -> Self {
        match p {
            QualityPresetArg::Auto => QualityPreset::Auto,
            QualityPresetArg::Low => QualityPreset::Low,
            QualityPresetArg::Medium => QualityPreset::Medium,
            QualityPresetArg::High => QualityPreset::High,
            QualityPresetArg::Ultra => QualityPreset::Ultra,
            QualityPresetArg::Custom => QualityPreset::Custom,
        }
    }
}

// The argv face of the render layer's `RtDynamicMode`, for the same reason.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub(crate) enum RtDynamicArg {
    Off,
    Auto,
    Rebuild,
    Tlas,
}

impl From<RtDynamicArg> for RtDynamicMode {
    fn from(m: RtDynamicArg) -> Self {
        match m {
            RtDynamicArg::Off => RtDynamicMode::Off,
            RtDynamicArg::Auto => RtDynamicMode::Auto,
            RtDynamicArg::Rebuild => RtDynamicMode::Rebuild,
            RtDynamicArg::Tlas => RtDynamicMode::Tlas,
        }
    }
}

#[derive(Debug, clap::Args)]
pub(crate) struct DebugArgs {
    /// Query or drive a running `cn debug` server instead of starting one
    // When present, `cn debug` acts as a client: it connects to an already
    // running server's localhost WebSocket. When absent, `cn debug` starts the
    // server (the interpreted run below).
    #[command(subcommand)]
    pub client: Option<DebugClientCommand>,

    /// Path to a world JSONL file (default: discover from worlds/)
    #[arg(short = 'f', long)]
    pub file: Option<String>,

    /// Base HTTP URL of the infra server used to fetch missing asset files
    // Defaults to the value in the client config (~/.config/concinnity/config.json).
    #[arg(long)]
    pub server: Option<String>,

    // Account ID for asset fetching authentication
    // Defaults to the value in the client config.
    #[arg(long)]
    pub(crate) user: Option<String>,

    // Port for the localhost runtime debug server (default 8777)
    #[arg(long)]
    pub(crate) debug_port: Option<u16>,

    // Enable graphics API validation, overriding the build profile
    // Omitting the flag defers to the build profile. See `RunArgs::validation`.
    #[arg(long)]
    pub(crate) validation: Option<bool>,

    #[command(flatten)]
    pub(crate) render: RenderArgs,
}

// Client-side `cn debug` subcommands. Each connects to a running server's
// localhost WebSocket, sends one request, and prints the reply; the transport
// lives in `debug::client` (re-exported from `debug::wire`).
#[derive(Subcommand, Debug)]
pub(crate) enum DebugClientCommand {
    /// Send one raw JSON command and print the reply
    #[command(name = "send")]
    Send(DebugSendArgs),

    /// Capture the last presented frame to a PNG
    #[command(name = "screenshot")]
    Screenshot(DebugScreenshotArgs),

    /// Poll a read-only snapshot and print it until Ctrl-C
    #[command(name = "watch")]
    Watch(DebugWatchArgs),
}

#[derive(Debug, clap::Args)]
pub(crate) struct DebugSendArgs {
    /// Raw JSON object including its own "cmd" field
    // e.g. '{"cmd":"state"}' or '{"cmd":"emitter-remove","id":0}'
    pub json: String,

    /// Debug server port
    #[arg(long, default_value_t = 8777)]
    pub port: u16,
}

#[derive(Debug, clap::Args)]
pub(crate) struct DebugScreenshotArgs {
    /// Output PNG path (resolved to absolute)
    pub path: String,

    /// Debug server port
    #[arg(long, default_value_t = 8777)]
    pub port: u16,
}

// The argv face of the library's `WatchTarget`: the value-enum derive lives
// here so concinnity-dev carries no clap dependency.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub(crate) enum WatchTargetArg {
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
pub(crate) struct DebugWatchArgs {
    /// Which read-only snapshot to poll
    #[arg(value_enum, default_value = "camera")]
    pub target: WatchTargetArg,

    // Milliseconds between polls
    #[arg(long, default_value_t = 500)]
    pub(crate) interval: u64,

    /// Debug server port
    #[arg(long, default_value_t = 8777)]
    pub port: u16,
}

#[derive(Debug, clap::Args)]
pub(crate) struct EditorArgs {
    /// Path to a world JSONL file (default: discover from worlds/)
    #[arg(short = 'f', long)]
    pub file: Option<String>,

    // Start the localhost debug server on this port alongside the editor
    // Absent leaves the editor without a WebSocket channel; present makes an
    // editor session inspectable/drivable (e.g. `cn debug send`, `screenshot`).
    #[arg(long)]
    pub(crate) debug_port: Option<u16>,

    // Enable graphics API validation, overriding the build profile
    // Omitting the flag defers to the build profile. See `RunArgs::validation`.
    #[arg(long)]
    pub(crate) validation: Option<bool>,

    #[command(flatten)]
    pub(crate) render: RenderArgs,
}

#[derive(Debug, clap::Args)]
pub(crate) struct RunArgs {
    // Enable graphics API validation, overriding the build profile
    // The DirectX / Vulkan debug layers, or on macOS the Metal API-validation
    // layer (the process re-execs once with `MTL_DEBUG_LAYER` set, since Metal
    // cannot toggle it from inside a running process). Omitting the flag defers
    // to the build profile: on for debug builds, off for release. Pass
    // `--validation false` to force it off in a debug build. The heavier Metal
    // shader validation is not enabled by this flag; set `MTL_SHADER_VALIDATION=1`
    // in the environment for that.
    #[arg(long)]
    pub(crate) validation: Option<bool>,

    // Step simulation and rendering serially on one thread instead of
    // pipelining them (A/B comparison, escape hatch)
    #[arg(long)]
    pub(crate) serial: bool,

    // Keep every system's internal work on the sim thread instead of the
    // job pool (determinism oracle, escape hatch)
    #[arg(long)]
    pub(crate) serial_schedule: bool,

    /// Capture the last presented frame to this PNG when the run stops
    #[arg(long)]
    pub screenshot: Option<String>,

    // Stop after this many frames (overrides GraphicsConfig.max_frames)
    #[arg(long)]
    pub(crate) frames: Option<u64>,

    #[command(flatten)]
    pub(crate) render: RenderArgs,
}

#[derive(Debug, clap::Args)]
pub(crate) struct AddArgs {
    /// Path to an asset file or type name
    pub target: String,

    /// Override the asset name written into the world
    // If omitted, the name is derived from the filename (including extension).
    #[arg(short, long)]
    pub name: Option<String>,

    // Named scaffold preset used when bootstrapping a new world
    // Currently only "minimal-3d-world" (a camera, sun, room, and sky on top of
    // the base scaffold). Ignored when scaffolding doesn't fire.
    #[arg(short = 't', long)]
    pub(crate) template: Option<String>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct RmArgs {
    /// The `name` field of the asset to remove
    pub name: String,
}

#[derive(Debug, clap::Args)]
pub(crate) struct TestArgs {
    /// Path to a world JSONL file (default: discover from worlds/)
    #[arg(short = 'f', long)]
    pub file: Option<String>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct ListArgs {
    /// Path to a world JSONL file (default: discover from worlds/)
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
pub(crate) struct ExplainArgs {
    /// The `name` field of the asset to print
    pub name: String,

    /// Path to a world JSONL file (default: discover from worlds/)
    #[arg(short = 'f', long)]
    pub file: Option<String>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct DocsArgs {
    /// Engine repository root to read sources from and write pages into
    #[arg(long, default_value = ".")]
    pub root: Option<String>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct NewArgs {
    /// Directory to create the project in
    pub path: String,
}

#[derive(Debug, clap::Args)]
pub(crate) struct BuildArgs {
    /// Path to a world JSONL file (default: discover from worlds/)
    #[arg(short = 'f', long)]
    pub file: Option<String>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct ExportArgs {
    /// Path to a world JSONL file (default: discover from worlds/)
    #[arg(short = 'f', long)]
    pub file: Option<String>,

    /// Override the application name
    #[arg(short = 'n', long)]
    pub name: Option<String>,

    // Override the application version
    #[arg(long)]
    pub(crate) version: Option<String>,

    /// EntityTarget platform
    #[arg(long)]
    pub platform: Option<String>,

    /// Output directory for the exported app
    #[arg(long, default_value = "dist")]
    pub out: String,

    // Output format: zip (default) or dir
    #[arg(long, default_value = "zip")]
    pub(crate) format: String,

    // Also produce a .dmg wrapping the .app (macOS-only)
    #[arg(long)]
    pub(crate) dmg: bool,
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
pub(crate) fn reexec_with_metal_validation(cli: &Cli) {
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
pub(crate) fn reexec_with_metal_validation(_cli: &Cli) {}

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

    // Every world-launching command carries the render flags, and each one
    // defaults to absent (which the engine resolves to today's behaviour).
    #[test]
    fn the_render_flags_default_to_absent_on_every_launch_command() {
        for argv in [
            vec!["concinnity", "run"],
            vec!["concinnity", "debug"],
            vec!["concinnity", "editor"],
        ] {
            let cli = Cli::try_parse_from(&argv).unwrap();
            let render = match &cli.command {
                Commands::Run(a) => &a.render,
                Commands::Debug(a) => &a.render,
                Commands::Editor(a) => &a.render,
                _ => panic!("expected a launch command for {argv:?}"),
            };
            assert!(render.quality_preset.is_none(), "{argv:?}");
            assert!(render.rt_dynamic.is_none(), "{argv:?}");
            assert!(render.rt_skinned_geometry.is_none(), "{argv:?}");
        }
    }

    #[test]
    fn the_render_flags_parse_on_every_launch_command() {
        for command in ["run", "debug", "editor"] {
            let cli = Cli::try_parse_from([
                "concinnity",
                command,
                "--quality-preset",
                "ultra",
                "--rt-dynamic",
                "rebuild",
                "--rt-skinned-geometry",
                "false",
            ])
            .unwrap();
            let render = match &cli.command {
                Commands::Run(a) => &a.render,
                Commands::Debug(a) => &a.render,
                Commands::Editor(a) => &a.render,
                _ => panic!("expected a launch command for {command}"),
            };
            assert!(
                matches!(render.quality_preset, Some(QualityPresetArg::Ultra)),
                "{command}"
            );
            assert!(
                matches!(render.rt_dynamic, Some(RtDynamicArg::Rebuild)),
                "{command}"
            );
            assert_eq!(render.rt_skinned_geometry, Some(false), "{command}");
        }
    }

    // The value enums are the argv face of the engine types, so a variant added
    // to one and not the other has to fail here rather than at a launch.
    #[test]
    fn the_render_value_enums_map_onto_the_engine_types() {
        use clap::ValueEnum;

        let presets: Vec<QualityPreset> = QualityPresetArg::value_variants()
            .iter()
            .map(|&a| a.into())
            .collect();
        assert_eq!(presets, QualityPreset::ALL.to_vec());

        let modes: Vec<RtDynamicMode> = RtDynamicArg::value_variants()
            .iter()
            .map(|&a| a.into())
            .collect();
        assert_eq!(
            modes,
            vec![
                RtDynamicMode::Off,
                RtDynamicMode::Auto,
                RtDynamicMode::Rebuild,
                RtDynamicMode::Tlas,
            ]
        );
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

    // The argv enum exists only so concinnity-dev carries no clap dependency,
    // which makes this conversion the seam between the two. It is applied in the
    // dispatch, so cover every variant here rather than through a running
    // `cn debug watch`.
    #[test]
    fn watch_target_arg_converts_every_variant() {
        assert!(matches!(
            WatchTarget::from(WatchTargetArg::Camera),
            WatchTarget::Camera
        ));
        assert!(matches!(
            WatchTarget::from(WatchTargetArg::State),
            WatchTarget::State
        ));
        assert!(matches!(
            WatchTarget::from(WatchTargetArg::Streaming),
            WatchTarget::Streaming
        ));
        assert!(matches!(
            WatchTarget::from(WatchTargetArg::Profile),
            WatchTarget::Profile
        ));
    }

    #[test]
    fn every_watch_target_name_parses() {
        for name in ["camera", "state", "streaming", "profile"] {
            let cli = Cli::try_parse_from(["concinnity", "debug", "watch", name])
                .unwrap_or_else(|e| panic!("{name} should parse: {e}"));
            let Commands::Debug(a) = cli.command else {
                panic!("expected debug");
            };
            assert!(matches!(a.client, Some(DebugClientCommand::Watch(_))));
        }
    }

    #[test]
    fn init_takes_no_arguments() {
        let cli = Cli::try_parse_from(["concinnity", "init"]).unwrap();
        assert!(matches!(cli.command, Commands::Init));
        assert!(Cli::try_parse_from(["concinnity", "init", "extra"]).is_err());
    }

    #[test]
    fn new_requires_a_path() {
        assert!(Cli::try_parse_from(["concinnity", "new"]).is_err());
        let cli = Cli::try_parse_from(["concinnity", "new", "my-app"]).unwrap();
        let Commands::New(a) = cli.command else {
            panic!("expected new");
        };
        assert_eq!(a.path, "my-app");
    }

    // `build` and `test` share the same optional --file, defaulting to discovery.
    #[test]
    fn build_and_test_take_an_optional_world_file() {
        let cli = Cli::try_parse_from(["concinnity", "build"]).unwrap();
        let Commands::Build(a) = cli.command else {
            panic!("expected build");
        };
        assert!(a.file.is_none());

        let cli = Cli::try_parse_from(["concinnity", "build", "-f", "w.jsonl"]).unwrap();
        let Commands::Build(a) = cli.command else {
            panic!("expected build");
        };
        assert_eq!(a.file.as_deref(), Some("w.jsonl"));

        let cli = Cli::try_parse_from(["concinnity", "test", "--file", "w.jsonl"]).unwrap();
        let Commands::Test(a) = cli.command else {
            panic!("expected test");
        };
        assert_eq!(a.file.as_deref(), Some("w.jsonl"));
    }

    #[test]
    fn list_flags_are_independent() {
        let cli = Cli::try_parse_from(["concinnity", "list"]).unwrap();
        let Commands::List(a) = cli.command else {
            panic!("expected list");
        };
        assert!(!a.expanded);
        assert!(!a.systems);

        let cli = Cli::try_parse_from(["concinnity", "list", "--expanded", "--systems"]).unwrap();
        let Commands::List(a) = cli.command else {
            panic!("expected list");
        };
        assert!(a.expanded);
        assert!(a.systems);
    }

    #[test]
    fn explain_requires_a_name() {
        assert!(Cli::try_parse_from(["concinnity", "explain"]).is_err());
        let cli = Cli::try_parse_from(["concinnity", "explain", "gfx", "-f", "w.jsonl"]).unwrap();
        let Commands::Explain(a) = cli.command else {
            panic!("expected explain");
        };
        assert_eq!(a.name, "gfx");
        assert_eq!(a.file.as_deref(), Some("w.jsonl"));
    }

    #[test]
    fn rm_requires_a_name() {
        assert!(Cli::try_parse_from(["concinnity", "rm"]).is_err());
        let cli = Cli::try_parse_from(["concinnity", "rm", "my_llm"]).unwrap();
        let Commands::Rm(a) = cli.command else {
            panic!("expected rm");
        };
        assert_eq!(a.name, "my_llm");
    }

    #[test]
    fn add_takes_a_scaffold_template() {
        let cli = Cli::try_parse_from(["concinnity", "add", "scene.glb", "-t", "minimal-3d-world"])
            .unwrap();
        let Commands::Add(a) = cli.command else {
            panic!("expected add");
        };
        assert_eq!(a.target, "scene.glb");
        assert_eq!(a.template.as_deref(), Some("minimal-3d-world"));
    }

    #[test]
    fn docs_root_defaults_to_the_current_directory() {
        let cli = Cli::try_parse_from(["concinnity", "docs"]).unwrap();
        let Commands::Docs(a) = cli.command else {
            panic!("expected docs");
        };
        assert_eq!(a.root.as_deref(), Some("."));

        let cli = Cli::try_parse_from(["concinnity", "docs", "--root", "/engine"]).unwrap();
        let Commands::Docs(a) = cli.command else {
            panic!("expected docs");
        };
        assert_eq!(a.root.as_deref(), Some("/engine"));
    }

    // Validation is tri-state: absent means "default for this build profile",
    // so an explicit `false` must survive parsing as a value rather than
    // collapsing into the absent case.
    #[test]
    fn run_validation_is_tri_state() {
        let cli = Cli::try_parse_from(["concinnity", "run"]).unwrap();
        let Commands::Run(a) = cli.command else {
            panic!("expected run");
        };
        assert_eq!(a.validation, None);

        for (arg, expected) in [("true", true), ("false", false)] {
            let cli = Cli::try_parse_from(["concinnity", "run", "--validation", arg]).unwrap();
            let Commands::Run(a) = cli.command else {
                panic!("expected run");
            };
            assert_eq!(a.validation, Some(expected));
        }
    }

    #[test]
    fn editor_takes_a_file_debug_port_and_validation() {
        let cli = Cli::try_parse_from(["concinnity", "editor"]).unwrap();
        let Commands::Editor(a) = cli.command else {
            panic!("expected editor");
        };
        assert!(a.file.is_none());
        assert!(a.debug_port.is_none());
        assert!(a.validation.is_none());

        let cli = Cli::try_parse_from([
            "concinnity",
            "editor",
            "-f",
            "w.jsonl",
            "--debug-port",
            "9001",
            "--validation",
            "false",
        ])
        .unwrap();
        let Commands::Editor(a) = cli.command else {
            panic!("expected editor");
        };
        assert_eq!(a.file.as_deref(), Some("w.jsonl"));
        assert_eq!(a.debug_port, Some(9001));
        assert_eq!(a.validation, Some(false));
    }

    #[test]
    fn export_flags_override_every_default() {
        let cli = Cli::try_parse_from([
            "concinnity",
            "export",
            "-f",
            "w.jsonl",
            "-n",
            "My Game",
            "--version",
            "2.0.0",
            "--platform",
            "macos",
            "--out",
            "build",
            "--format",
            "dir",
            "--dmg",
        ])
        .unwrap();
        let Commands::Export(e) = cli.command else {
            panic!("expected export");
        };
        assert_eq!(e.file.as_deref(), Some("w.jsonl"));
        assert_eq!(e.name.as_deref(), Some("My Game"));
        assert_eq!(e.version.as_deref(), Some("2.0.0"));
        assert_eq!(e.platform.as_deref(), Some("macos"));
        assert_eq!(e.out, "build");
        assert_eq!(e.format, "dir");
        assert!(e.dmg);
    }

    #[test]
    fn debug_takes_a_port_and_validation_without_a_client() {
        let cli = Cli::try_parse_from([
            "concinnity",
            "debug",
            "--debug-port",
            "9100",
            "--validation",
            "true",
        ])
        .unwrap();
        let Commands::Debug(a) = cli.command else {
            panic!("expected debug");
        };
        assert!(a.client.is_none());
        assert_eq!(a.debug_port, Some(9100));
        assert_eq!(a.validation, Some(true));
    }

    // Only the paths that decline to re-exec are driven: the guard returns for a
    // command that starts no renderer, and for a render command with validation
    // explicitly off. Requesting validation would `exec` and replace the test
    // process, so no case here leaves `validation` unset or true.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_validation_reexec_declines_when_not_requested() {
        for args in [
            ["concinnity", "run", "--validation", "false"],
            ["concinnity", "editor", "--validation", "false"],
            ["concinnity", "debug", "--validation", "false"],
        ] {
            reexec_with_metal_validation(&Cli::try_parse_from(args).unwrap());
        }
        // A `cn debug` client subcommand stands up no renderer, so it returns at
        // the match rather than reaching the request check.
        reexec_with_metal_validation(
            &Cli::try_parse_from(["concinnity", "debug", "screenshot", "out.png"]).unwrap(),
        );
        reexec_with_metal_validation(&Cli::try_parse_from(["concinnity", "list"]).unwrap());
    }
}
