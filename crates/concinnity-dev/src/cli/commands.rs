// The command tree itself: every subcommand `concinnity` accepts and the
// arguments it takes. Parsing only -- what each command does lives behind
// `dispatch`.
use concinnity_engine::launch::{DEFAULT_DEBUG_PORT, LaunchArgs};

use super::launch_args::{BenchScope, LaunchArgsArg, RunScope};
use super::value_enums::BundleFormatArg;

// Each command's rustdoc is its `--help` text and its section of the command
// line reference: the first line is the summary, and the ```sh block lists
// examples, each a whole command line.
#[derive(clap::Subcommand, Debug)]
pub(crate) enum Commands {
    /// Create a new app in the current directory
    ///
    /// Writes a starter world to `world.jsonl` and cooks it. A directory that
    /// already holds a world is left alone.
    ///
    /// ```sh
    /// concinnity init
    /// ```
    #[command(name = "init", verbatim_doc_comment)]
    Init,

    /// Create a new app in a new directory
    ///
    /// Creates the directory, writes a starter world to `world.jsonl` in it and
    /// cooks it. A directory that already holds a world is refused.
    ///
    /// ```sh
    /// concinnity new my-app
    /// ```
    #[command(name = "new", verbatim_doc_comment)]
    New(NewArgs),

    /// Cook an app into binary blobs
    ///
    /// Compiles the world into the blobs a run loads and writes
    /// `world-lock.json`, showing each step as it runs and the warnings the
    /// build raises.
    ///
    /// ```sh
    /// concinnity cook
    /// concinnity cook -f arena.jsonl --verbose
    /// ```
    #[command(name = "cook", verbatim_doc_comment)]
    Cook(CookArgs),

    /// Validate an app without cooking it
    ///
    /// ```sh
    /// concinnity check
    /// concinnity check -f arena.jsonl
    /// ```
    #[command(name = "check", verbatim_doc_comment)]
    Check(CheckArgs),

    /// Run an app
    ///
    /// Without `-f`, runs the cooked blobs, cooking the discovered world first
    /// when there are none. With `-f`, cooks the named world in memory and
    /// reloads it as its files change. `--debug-port` serves the debug
    /// protocol (MCP) either way.
    ///
    /// ```sh
    /// concinnity run
    /// concinnity run -f arena.jsonl --debug-port
    /// concinnity run --graphics-api vulkan --graphics-quality high --no-vsync
    /// ```
    #[command(name = "run", verbatim_doc_comment)]
    Run(RunArgs),

    /// Benchmark an app and print what its frames cost
    ///
    /// Finds its world the way `run` does, and prints the frame costs when the
    /// run ends.
    ///
    /// ```sh
    /// concinnity bench
    /// concinnity bench --start-at 10 --stop-at 40 --screenshot-at 20:mid.png
    /// ```
    #[command(name = "bench", verbatim_doc_comment)]
    Bench(BenchArgs),

    /// Edit an app in-engine with a save-back HUD
    ///
    /// Cooks the world in memory, so it needs no `concinnity cook` first and
    /// neither reads nor refreshes the cooked blobs. The editor HUD saves edits
    /// back to the world file and adds assets. `--debug-port` serves the same
    /// debug protocol `run` does, so an MCP client can drive the editor.
    ///
    /// ```sh
    /// concinnity edit
    /// concinnity edit -f arena.jsonl --debug-port 8777
    /// ```
    #[command(name = "edit", verbatim_doc_comment)]
    Edit(EditArgs),

    /// Add an asset to the active app
    ///
    /// TARGET is one of:
    ///
    /// - A file path, such as `shaders/pbr.vert` or `models/scene.glb`. The type
    ///   comes from the file extension or the JSON `type` field.
    /// - A type name, such as `PointLight`. The asset takes the type's
    ///   registered default args.
    /// - A whole entry, such as `'["Window", {}]'`.
    ///
    /// ```sh
    /// concinnity add models/scene.glb --template minimal-3d-world
    /// concinnity add PointLight --id lamp
    /// concinnity add '["Window", {}]'
    /// ```
    #[command(name = "add", verbatim_doc_comment)]
    Add(AddArgs),

    /// Remove an asset from the active app by its id
    ///
    /// ID is the `$id` an entry declares in its args, or the `<Type>#<ordinal>`
    /// handle of an anonymous one.
    ///
    /// ```sh
    /// concinnity rm lamp
    /// concinnity rm 'Prop#3'
    /// ```
    #[command(name = "rm", verbatim_doc_comment)]
    Rm(RmArgs),

    /// List all declared assets
    ///
    /// ```sh
    /// concinnity list
    /// concinnity list --expanded --systems
    /// ```
    #[command(name = "list", verbatim_doc_comment)]
    List(ListArgs),

    /// Print an asset's effective entry from the expanded app
    ///
    /// Prints the whole world line for ID as the cook sees it, including assets
    /// that only exist through cook-time expansion or injection. The line can
    /// be pasted into the world file as it stands to override a default.
    ///
    /// ```sh
    /// concinnity explain lamp
    /// concinnity explain 'Window#0' -f arena.jsonl
    /// ```
    #[command(name = "explain", verbatim_doc_comment)]
    Explain(ExplainArgs),

    /// Regenerate the reference pages
    ///
    /// Writes the asset reference, the graphics settings page and the command
    /// line reference from the rustdoc compiled into this build, so it needs no
    /// source tree.
    ///
    /// ```sh
    /// concinnity docs
    /// concinnity docs --root ../engine
    /// ```
    #[command(name = "docs", verbatim_doc_comment)]
    Docs(DocsArgs),

    /// Package a cooked app for distribution
    ///
    /// ```sh
    /// concinnity export
    /// concinnity export --name "My App" --version 1.2.0 --format dir --out build
    /// ```
    #[command(name = "export", verbatim_doc_comment)]
    Export(ExportArgs),

    /// Serve the debug protocol to an MCP client over stdio
    ///
    /// An MCP client spawns it as a child process. It speaks JSON-RPC over
    /// stdin and stdout and forwards each tool call to a running app's debug
    /// port, which serves the same protocol to any client that posts to it.
    ///
    /// ```sh
    /// concinnity mcp
    /// concinnity mcp --debug-port 9100
    /// ```
    #[command(name = "mcp", verbatim_doc_comment)]
    Mcp(McpArgs),

    /// Print the version
    ///
    /// ```sh
    /// concinnity version
    /// ```
    #[command(name = "version", verbatim_doc_comment)]
    Version,
}

impl Commands {
    // The launch a world-running command asks for, `quiet` from the root
    // command. `None` for a command that runs no world.
    pub(crate) fn launch_args(&self, quiet: bool) -> Option<LaunchArgs> {
        let args = match self {
            Self::Run(a) => &a.launch.args,
            Self::Bench(a) => &a.launch.args,
            Self::Edit(a) => &a.launch.args,
            _ => return None,
        };
        Some(LaunchArgs {
            quiet,
            ..args.clone()
        })
    }
}

// The `-f` world a command operates on. Flattened into every command that
// reads a world from disk; an absent flag discovers it, a named file that does
// not exist is an error.
#[derive(Debug, clap::Args)]
pub(crate) struct WorldArg {
    /// Path to a world JSONL file
    ///
    /// Defaults to the nearest `world.jsonl` up from the current directory.
    #[arg(short = 'f', long)]
    file: Option<String>,
}

impl WorldArg {
    pub(crate) fn file(&self) -> Option<&str> {
        self.file.as_deref()
    }
}

#[derive(Debug, clap::Args)]
pub(crate) struct RunArgs {
    #[command(flatten)]
    pub(crate) world: WorldArg,

    #[command(flatten)]
    pub(crate) launch: LaunchArgsArg<RunScope>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct BenchArgs {
    #[command(flatten)]
    pub(crate) world: WorldArg,

    #[command(flatten)]
    pub(crate) launch: LaunchArgsArg<BenchScope>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct EditArgs {
    #[command(flatten)]
    pub(crate) world: WorldArg,

    #[command(flatten)]
    pub(crate) launch: LaunchArgsArg<RunScope>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct McpArgs {
    /// Port the runtime debug server is listening on
    #[arg(long, default_value_t = DEFAULT_DEBUG_PORT)]
    pub(crate) debug_port: u16,
}

#[derive(Debug, clap::Args)]
pub(crate) struct AddArgs {
    /// Path to an asset file, a type name, or an entry like '["Window", {}]'
    pub(crate) target: String,

    /// The `$id` written into the world
    ///
    /// Without one, a file target's id comes from its file name, and a type
    /// name or an entry without one is added anonymous.
    #[arg(short, long)]
    pub(crate) id: Option<String>,

    /// Named scaffold preset used when bootstrapping a new world
    ///
    /// `minimal-3d-world` adds a camera, sun, room and sky to the base
    /// scaffold. Ignored when the world already exists.
    #[arg(short = 't', long)]
    pub(crate) template: Option<String>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct RmArgs {
    /// The asset's `$id`, or the `<Type>#<ordinal>` of an anonymous one
    pub(crate) id: String,
}

#[derive(Debug, clap::Args)]
pub(crate) struct CheckArgs {
    #[command(flatten)]
    pub(crate) world: WorldArg,
}

#[derive(Debug, clap::Args)]
pub(crate) struct ListArgs {
    #[command(flatten)]
    pub(crate) world: WorldArg,

    /// List the expanded world the cook produces
    ///
    /// Cook-time macros are expanded and injected defaults included, each row
    /// tagged with where it came from: authored, injected or expanded.
    #[arg(long)]
    pub(crate) expanded: bool,

    /// List the systems this world runs, in order, each with the condition that
    /// includes it
    // Builds the world and reports `World::system_manifest()`, the same gates
    // the runtime runs at start, so it cannot drift.
    #[arg(long)]
    pub(crate) systems: bool,
}

#[derive(Debug, clap::Args)]
pub(crate) struct ExplainArgs {
    /// The asset's `$id`, or the `<Type>#<ordinal>` of an anonymous one
    pub(crate) id: String,

    #[command(flatten)]
    pub(crate) world: WorldArg,
}

#[derive(Debug, clap::Args)]
pub(crate) struct DocsArgs {
    /// Directory to write the docs/ pages under
    #[arg(long, default_value = ".")]
    pub(crate) root: Option<String>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct NewArgs {
    /// Directory to create the project in
    pub(crate) path: String,
}

#[derive(Debug, clap::Args)]
pub(crate) struct CookArgs {
    #[command(flatten)]
    pub(crate) world: WorldArg,

    /// Also show the cook's informational messages, such as each imported mesh
    #[arg(short, long)]
    pub(crate) verbose: bool,
}

#[derive(Debug, clap::Args)]
pub(crate) struct ExportArgs {
    #[command(flatten)]
    pub(crate) world: WorldArg,

    /// Override the application name
    #[arg(short = 'n', long)]
    pub(crate) name: Option<String>,

    /// Override the application version
    #[arg(long)]
    pub(crate) version: Option<String>,

    /// Target platform, which must be this machine's (`macos`, `windows` or
    /// `linux`)
    #[arg(long)]
    pub(crate) platform: Option<String>,

    /// Output directory for the exported app
    #[arg(long, default_value = "dist")]
    pub(crate) out: String,

    /// Output format
    #[arg(long, value_enum, default_value = "zip")]
    pub(crate) format: BundleFormatArg,

    /// Also produce a .dmg wrapping the .app (macOS-only)
    #[arg(long)]
    pub(crate) dmg: bool,
}

// Argument-parsing tests. The engine-launching dispatch and the Metal re-exec
// are exercised by the `tests/cli.rs` integration tests, which run the built
// binary; these cover the clap surface -- the command tree, the per-subcommand
// defaults, and the value parsing -- without a process.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::Cli;
    use clap::Parser;
    use concinnity_engine::gfx::view_modes::ShowFlags;
    use concinnity_engine::launch::LaunchCommand;
    use concinnity_engine::render::output::dlss::DlssPreset;
    use concinnity_engine::render::rt_accel::geom::RtDynamicMode;
    use concinnity_engine::settings::QualityPreset;
    use core::num::NonZeroU32;

    fn launch(argv: &[&str]) -> LaunchArgs {
        let cli = Cli::try_parse_from(argv).unwrap();
        cli.command
            .launch_args(cli.quiet)
            .expect("a launch command")
    }

    // Every world-launching command defaults every flag to absent, which the
    // engine resolves to today's behavior, and names its own launch command.
    #[test]
    fn the_launch_flags_default_to_absent_on_every_launch_command() {
        for (command, expected) in [
            ("run", LaunchCommand::Run),
            ("bench", LaunchCommand::Bench),
            ("edit", LaunchCommand::Run),
        ] {
            assert_eq!(
                launch(&["concinnity", command]),
                LaunchArgs {
                    command: expected,
                    ..LaunchArgs::NONE
                },
                "{command}"
            );
        }
        let cli = Cli::try_parse_from(["concinnity", "cook"]).unwrap();
        assert_eq!(cli.command.launch_args(false), None);
    }

    #[test]
    fn the_launch_flags_parse_on_every_launch_command() {
        for command in ["run", "bench", "edit"] {
            let args = launch(&[
                "concinnity",
                command,
                "--graphics-api",
                "vulkan",
                "--graphics-quality",
                "ultra",
                "--rt-dynamic",
                "rebuild",
                "--no-rt-skinned-geometry",
                "--dlss-preset",
                "m",
                "--fixed-frame-rate",
                "60",
                "--hide-pass",
                "reactive",
                "--hide-pass",
                "fog",
                "--hide-pass",
                "reactive",
            ]);
            assert_eq!(
                args.graphics_api,
                Some(concinnity_engine::platform::Platform::Vulkan),
                "{command}"
            );
            assert_eq!(args.graphics_quality, Some(QualityPreset::Ultra));
            assert_eq!(args.rt_dynamic, Some(RtDynamicMode::Rebuild));
            assert_eq!(args.rt_skinned_geometry, Some(false));
            assert_eq!(args.dlss_preset, Some(DlssPreset::M));
            assert_eq!(args.fixed_frame_rate, NonZeroU32::new(60));
            assert_eq!(
                args.show,
                ShowFlags::all()
                    .with(ShowFlags::REACTIVE, false)
                    .with(ShowFlags::FOG, false),
            );
        }
    }

    #[test]
    fn only_bench_takes_the_bench_flags() {
        let args = launch(&[
            "concinnity",
            "bench",
            "--start-at",
            "2",
            "--stop-at",
            "30",
            "--screenshot-at",
            "4:start",
        ]);
        assert_eq!(args.start_at(), Some(2.0));
        assert_eq!(args.stop_at(), Some(30.0));
        assert_eq!(args.screenshot_at.len(), 1);
        for flag in ["--start-at", "--stop-at", "--screenshot-at"] {
            for command in ["run", "edit"] {
                assert!(
                    Cli::try_parse_from(["concinnity", command, flag, "1"]).is_err(),
                    "{command} {flag}"
                );
            }
        }
    }

    // The flags the reworked command line dropped or renamed are refused.
    #[test]
    fn the_retired_flags_are_refused() {
        for argv in [
            &["concinnity", "run", "--benchmark"][..],
            &["concinnity", "run", "--serial"],
            &["concinnity", "run", "--serial-schedule"],
            &["concinnity", "run", "--screenshot", "a.png"],
            &["concinnity", "run", "--screenshots"],
            &["concinnity", "run", "--validation", "true"],
            &["concinnity", "run", "--fps-cap", "60"],
            &["concinnity", "run", "--frames", "60"],
            &["concinnity", "run", "--quality-preset", "ultra"],
            &["concinnity", "run", "--hide", "fog"],
            &["concinnity", "run", "--upscaling", "on"],
        ] {
            assert!(Cli::try_parse_from(argv).is_err(), "{argv:?}");
        }
    }

    #[test]
    fn the_retired_commands_are_unknown() {
        for command in ["build", "test", "debug", "editor"] {
            assert!(
                Cli::try_parse_from(["concinnity", command]).is_err(),
                "{command}"
            );
        }
    }

    #[test]
    fn quiet_is_taken_before_or_after_the_command() {
        assert!(launch(&["concinnity", "-q", "run"]).quiet);
        assert!(launch(&["concinnity", "bench", "--quiet"]).quiet);
        assert!(!launch(&["concinnity", "run"]).quiet);
        assert!(
            Cli::try_parse_from(["concinnity", "cook", "-q"])
                .unwrap()
                .quiet
        );
    }

    // A rate of 0 would stop the clock, so it is refused at the flag.
    #[test]
    fn a_zero_fixed_frame_rate_is_refused() {
        let parsed = Cli::try_parse_from(["concinnity", "run", "--fixed-frame-rate", "0"]);
        assert!(parsed.is_err());
    }

    #[test]
    fn run_takes_a_world_and_a_debug_port() {
        let cli = Cli::try_parse_from(["concinnity", "run", "-f", "world.jsonl"]).unwrap();
        let Commands::Run(a) = &cli.command else {
            panic!("expected run");
        };
        assert_eq!(a.world.file(), Some("world.jsonl"));
        assert_eq!(a.launch.args.debug_port, None);
        let port = |argv: &[&str]| launch(argv).debug_port;
        assert_eq!(
            port(&["concinnity", "run", "--debug-port"]),
            Some(DEFAULT_DEBUG_PORT),
            "the same default concinnity mcp dials"
        );
        assert_eq!(
            port(&["concinnity", "run", "--debug-port", "9100"]),
            Some(9100)
        );
        assert_eq!(
            launch(&["concinnity", "run", "--no-graphics-validation"]).graphics_validation,
            Some(false)
        );
    }

    #[test]
    fn run_takes_no_subcommand() {
        for argv in [
            ["concinnity", "run", "send"],
            ["concinnity", "run", "screenshot"],
        ] {
            assert!(Cli::try_parse_from(argv).is_err(), "{argv:?}");
        }
    }

    #[test]
    fn mcp_defaults_to_the_debug_port() {
        let cli = Cli::try_parse_from(["concinnity", "mcp"]).unwrap();
        let Commands::Mcp(a) = &cli.command else {
            panic!("expected mcp");
        };
        assert_eq!(a.debug_port, DEFAULT_DEBUG_PORT);
        let cli = Cli::try_parse_from(["concinnity", "mcp", "--debug-port", "9001"]).unwrap();
        let Commands::Mcp(a) = &cli.command else {
            panic!("expected mcp");
        };
        assert_eq!(a.debug_port, 9001);
    }

    #[test]
    fn export_defaults() {
        let cli = Cli::try_parse_from(["concinnity", "export"]).unwrap();
        let Commands::Export(e) = &cli.command else {
            panic!("expected export");
        };
        assert_eq!(e.out, "dist");
        assert_eq!(e.format, BundleFormatArg::Zip);
        assert!(!e.dmg);
    }

    #[test]
    fn export_rejects_an_unknown_format() {
        let err = Cli::try_parse_from(["concinnity", "export", "--format", "tarball"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::InvalidValue);
    }

    #[test]
    fn add_requires_a_target() {
        assert!(Cli::try_parse_from(["concinnity", "add"]).is_err());
        let cli = Cli::try_parse_from(["concinnity", "add", "Logger", "--id", "log"]).unwrap();
        let Commands::Add(a) = &cli.command else {
            panic!("expected add");
        };
        assert_eq!(a.target, "Logger");
        assert_eq!(a.id.as_deref(), Some("log"));
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
        let Commands::New(a) = &cli.command else {
            panic!("expected new");
        };
        assert_eq!(a.path, "my-app");
    }

    // `run`, `bench`, `cook`, `check` and `edit` share the same optional
    // --file, defaulting to discovery.
    #[test]
    fn the_world_commands_take_an_optional_world_file() {
        let file = |argv: &[&str]| -> Option<String> {
            let cli = Cli::try_parse_from(argv).unwrap();
            let world = match &cli.command {
                Commands::Cook(a) => &a.world,
                Commands::Check(a) => &a.world,
                Commands::Run(a) => &a.world,
                Commands::Bench(a) => &a.world,
                Commands::Edit(a) => &a.world,
                _ => panic!("expected a world command for {argv:?}"),
            };
            world.file().map(str::to_string)
        };
        for command in ["cook", "check", "run", "bench", "edit"] {
            assert_eq!(file(&["concinnity", command]), None, "{command}");
            assert_eq!(
                file(&["concinnity", command, "-f", "w.jsonl"]).as_deref(),
                Some("w.jsonl")
            );
            assert_eq!(
                file(&["concinnity", command, "--file", "w.jsonl"]).as_deref(),
                Some("w.jsonl")
            );
        }
    }

    #[test]
    fn cook_shows_its_messages_when_verbose() {
        let cli = Cli::try_parse_from(["concinnity", "cook", "-v"]).unwrap();
        let Commands::Cook(a) = &cli.command else {
            panic!("expected cook");
        };
        assert!(a.verbose);
    }

    #[test]
    fn list_flags_are_independent() {
        let cli = Cli::try_parse_from(["concinnity", "list"]).unwrap();
        let Commands::List(a) = &cli.command else {
            panic!("expected list");
        };
        assert!(!a.expanded);
        assert!(!a.systems);

        let cli = Cli::try_parse_from(["concinnity", "list", "--expanded", "--systems"]).unwrap();
        let Commands::List(a) = &cli.command else {
            panic!("expected list");
        };
        assert!(a.expanded);
        assert!(a.systems);
    }

    #[test]
    fn explain_requires_an_id() {
        assert!(Cli::try_parse_from(["concinnity", "explain"]).is_err());
        let cli = Cli::try_parse_from(["concinnity", "explain", "gfx", "-f", "w.jsonl"]).unwrap();
        let Commands::Explain(a) = &cli.command else {
            panic!("expected explain");
        };
        assert_eq!(a.id, "gfx");
        assert_eq!(a.world.file(), Some("w.jsonl"));
    }

    #[test]
    fn rm_requires_an_id() {
        assert!(Cli::try_parse_from(["concinnity", "rm"]).is_err());
        let cli = Cli::try_parse_from(["concinnity", "rm", "Prop#3"]).unwrap();
        let Commands::Rm(a) = &cli.command else {
            panic!("expected rm");
        };
        assert_eq!(a.id, "Prop#3");
    }

    #[test]
    fn add_takes_a_scaffold_template() {
        let cli = Cli::try_parse_from(["concinnity", "add", "scene.glb", "-t", "minimal-3d-world"])
            .unwrap();
        let Commands::Add(a) = &cli.command else {
            panic!("expected add");
        };
        assert_eq!(a.target, "scene.glb");
        assert_eq!(a.template.as_deref(), Some("minimal-3d-world"));
    }

    #[test]
    fn docs_root_defaults_to_the_current_directory() {
        let cli = Cli::try_parse_from(["concinnity", "docs"]).unwrap();
        let Commands::Docs(a) = &cli.command else {
            panic!("expected docs");
        };
        assert_eq!(a.root.as_deref(), Some("."));

        let cli = Cli::try_parse_from(["concinnity", "docs", "--root", "/engine"]).unwrap();
        let Commands::Docs(a) = &cli.command else {
            panic!("expected docs");
        };
        assert_eq!(a.root.as_deref(), Some("/engine"));
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
        let Commands::Export(e) = &cli.command else {
            panic!("expected export");
        };
        assert_eq!(e.world.file(), Some("w.jsonl"));
        assert_eq!(e.name.as_deref(), Some("My Game"));
        assert_eq!(e.version.as_deref(), Some("2.0.0"));
        assert_eq!(e.platform.as_deref(), Some("macos"));
        assert_eq!(e.out, "build");
        assert_eq!(e.format, BundleFormatArg::Dir);
        assert!(e.dmg);
    }
}
