// The command tree itself: every subcommand `concinnity` accepts and the
// arguments it takes. Parsing only -- what each command does lives behind
// `dispatch`.
use concinnity_engine::app::run::LaunchRequest;

use super::value_enums::{BundleFormatArg, QualityPresetArg, RtDynamicArg};

#[derive(clap::Subcommand, Debug)]
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
    // Production path: no debug server and no command channel. A shipped run
    // is neither remotely inspectable nor remotely driven: use `cn debug` for
    // that.
    #[command(name = "run")]
    Run(RunArgs),

    /// Run interpreted directly from a world jsonl file
    //
    // Compiles the world in memory (no prior `cn build` needed) and stands
    // up the localhost debug server.
    // This is the development run, and the path the agentic loop / a host UI
    // use when they need to read or drive runtime state: the port it opens
    // serves MCP.
    #[command(name = "debug")]
    Debug(DebugArgs),

    /// Edit a world in-engine with a save-back HUD [default command]
    //
    // Compiles world.jsonl in memory (no prior `cn build` needed, and the blobs
    // one wrote are neither read nor refreshed here), overlays the editor HUD (a
    // SAVE button plus an add-asset button), and persists edits by writing
    // world.jsonl. No command channel unless --debug-port is given (which
    // stands up the same MCP debug server `cn debug` uses, so an agent can
    // drive an editor session).
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

    /// Remove an asset from the active world by its id
    //
    // ID is the `$id` an entry declares in its args (e.g. "pbr_vert"), or the
    // `<Type>#<ordinal>` handle of an anonymous one (e.g. "Prop#3").
    #[command(name = "rm")]
    Rm(RmArgs),

    /// List all declared assets
    #[command(name = "list")]
    List(ListArgs),

    /// Print an asset's effective entry from the expanded world
    //
    // Prints the full JSONL line for ID as the build sees it, including
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

    /// Serve the debug protocol to an MCP client over stdio
    //
    // A child process an MCP client spawns, not a listener: it speaks JSON-RPC
    // over stdin/stdout and forwards each tool call to a running app's debug
    // port, which serves the same protocol to any client that posts to it.
    #[command(name = "mcp")]
    Mcp(McpArgs),

    /// Print the version
    #[command(name = "version")]
    Version,
}

// The argv face of the launch-time render knobs on the engine's `LaunchRequest`.
// Each is a diagnostic: omitting it leaves the shipping behavior, and none is
// persisted, so a probe run can force one without writing settings. Flattened
// into every command that launches a world (`run` / `debug` / `editor`).
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
    // The launch request these flags make, with the command's own validation
    // flag and whether it runs a development session.
    pub(crate) fn launch(&self, validation: Option<bool>, dev_loop: bool) -> LaunchRequest {
        LaunchRequest {
            capture: false,
            dev_loop,
            validation,
            quality_preset: self.quality_preset.map(Into::into),
            rt_dynamic: self.rt_dynamic.map(Into::into),
            rt_skinned_geometry: self.rt_skinned_geometry,
        }
    }
}

#[derive(Debug, clap::Args)]
pub(crate) struct DebugArgs {
    /// Path to a world JSONL file (default: discover from worlds/)
    #[arg(short = 'f', long)]
    pub(crate) file: Option<String>,

    /// Port for the localhost runtime debug server
    #[arg(long, default_value_t = 8777)]
    pub(crate) debug_port: u16,

    /// Enable graphics API validation, overriding the build profile
    // Omitting the flag defers to the build profile. See `RunArgs::validation`.
    #[arg(long)]
    pub(crate) validation: Option<bool>,

    #[command(flatten)]
    pub(crate) render: RenderArgs,
}

#[derive(Debug, clap::Args)]
pub(crate) struct McpArgs {
    /// Port the runtime debug server is listening on
    #[arg(long, default_value_t = 8777)]
    pub(crate) debug_port: u16,
}

#[derive(Debug, clap::Args)]
pub(crate) struct EditorArgs {
    /// Path to a world JSONL file (default: discover from worlds/)
    #[arg(short = 'f', long)]
    pub(crate) file: Option<String>,

    /// Start the localhost debug server on this port alongside the editor
    // Absent leaves the editor without a command channel; present makes an
    // editor session inspectable and drivable by any MCP client.
    #[arg(long)]
    pub(crate) debug_port: Option<u16>,

    /// Enable graphics API validation, overriding the build profile
    // Omitting the flag defers to the build profile. See `RunArgs::validation`.
    #[arg(long)]
    pub(crate) validation: Option<bool>,

    #[command(flatten)]
    pub(crate) render: RenderArgs,
}

#[derive(Debug, clap::Args)]
pub(crate) struct RunArgs {
    /// Enable graphics API validation, overriding the build profile
    // The DirectX / Vulkan debug layers, or on macOS the Metal API-validation
    // layer (the process re-execs once with `MTL_DEBUG_LAYER` set, since Metal
    // cannot toggle it from inside a running process). Omitting the flag defers
    // to the build profile: on for debug builds, off for release. Pass
    // `--validation false` to force it off in a debug build. The heavier Metal
    // shader validation is not enabled by this flag; set `MTL_SHADER_VALIDATION=1`
    // in the environment for that.
    #[arg(long)]
    pub(crate) validation: Option<bool>,

    /// Step simulation and rendering serially on one thread instead of pipelining them
    // A/B comparison, escape hatch.
    #[arg(long)]
    pub(crate) serial: bool,

    /// Keep every system's internal work on the sim thread instead of the job pool
    // Determinism oracle, escape hatch.
    #[arg(long)]
    pub(crate) serial_schedule: bool,

    /// Capture the last presented frame to this PNG when the run stops
    #[arg(long)]
    pub(crate) screenshot: Option<String>,

    /// Stop after this many frames (overrides GraphicsConfig.max_frames)
    #[arg(long)]
    pub(crate) frames: Option<u64>,

    #[command(flatten)]
    pub(crate) render: RenderArgs,
}

#[derive(Debug, clap::Args)]
pub(crate) struct AddArgs {
    /// Path to an asset file, a type name, or an entry like '["Window", {}]'
    pub(crate) target: String,

    /// The `$id` written into the world
    // If omitted, a file target's id is derived from the filename, and a type
    // name or inline entry without one is added anonymous.
    #[arg(short, long)]
    pub(crate) id: Option<String>,

    /// Named scaffold preset used when bootstrapping a new world
    // Currently only "minimal-3d-world" (a camera, sun, room, and sky on top of
    // the base scaffold). Ignored when scaffolding doesn't fire.
    #[arg(short = 't', long)]
    pub(crate) template: Option<String>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct RmArgs {
    /// The asset's `$id`, or the `<Type>#<ordinal>` of an anonymous one
    pub(crate) id: String,
}

#[derive(Debug, clap::Args)]
pub(crate) struct TestArgs {
    /// Path to a world JSONL file (default: discover from worlds/)
    #[arg(short = 'f', long)]
    pub(crate) file: Option<String>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct ListArgs {
    /// Path to a world JSONL file (default: discover from worlds/)
    #[arg(short = 'f', long)]
    pub(crate) file: Option<String>,

    /// List the expanded world the build produces
    // build-time macros are expanded and injected defaults included,
    // each row tagged with its provenance (authored / injected / expanded).
    #[arg(long)]
    pub(crate) expanded: bool,

    /// List the systems this world runs, in order, each with the condition
    // that includes it. Builds the world and reports `World::system_manifest()`
    // -- the same gates the runtime runs at start -- so it cannot drift.
    #[arg(long)]
    pub(crate) systems: bool,
}

#[derive(Debug, clap::Args)]
pub(crate) struct ExplainArgs {
    /// The asset's `$id`, or the `<Type>#<ordinal>` of an anonymous one
    pub(crate) id: String,

    /// Path to a world JSONL file (default: discover from worlds/)
    #[arg(short = 'f', long)]
    pub(crate) file: Option<String>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct DocsArgs {
    /// Engine repository root to read sources from and write pages into
    #[arg(long, default_value = ".")]
    pub(crate) root: Option<String>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct NewArgs {
    /// Directory to create the project in
    pub(crate) path: String,
}

#[derive(Debug, clap::Args)]
pub(crate) struct BuildArgs {
    /// Path to a world JSONL file (default: discover from worlds/)
    #[arg(short = 'f', long)]
    pub(crate) file: Option<String>,
}

#[derive(Debug, clap::Args)]
pub(crate) struct ExportArgs {
    /// Path to a world JSONL file (default: discover from worlds/)
    #[arg(short = 'f', long)]
    pub(crate) file: Option<String>,

    /// Override the application name
    #[arg(short = 'n', long)]
    pub(crate) name: Option<String>,

    /// Override the application version
    #[arg(long)]
    pub(crate) version: Option<String>,

    /// EntityTarget platform
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
// defaults, and the value-enum parsing -- without a process.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::Cli;
    use clap::Parser;
    use concinnity_core::render::rt_geom::RtDynamicMode;
    use concinnity_engine::gfx::quality_preset::QualityPreset;

    // Every world-launching command carries the render flags, and each one
    // defaults to absent (which the engine resolves to today's behavior).
    #[test]
    fn the_render_flags_default_to_absent_on_every_launch_command() {
        for argv in [
            vec!["concinnity", "run"],
            vec!["concinnity", "debug"],
            vec!["concinnity", "editor"],
        ] {
            let cli = Cli::try_parse_from(&argv).unwrap();
            let render = match cli.resolved_command() {
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
    fn the_render_flags_map_onto_the_launch_request() {
        let cli = Cli::try_parse_from([
            "concinnity",
            "run",
            "--quality-preset",
            "ultra",
            "--rt-dynamic",
            "rebuild",
            "--rt-skinned-geometry",
            "false",
        ])
        .unwrap();
        let Commands::Run(args) = cli.resolved_command() else {
            panic!("expected run");
        };
        assert_eq!(
            args.render.launch(Some(true), true),
            LaunchRequest {
                capture: false,
                dev_loop: true,
                validation: Some(true),
                quality_preset: Some(QualityPreset::Ultra),
                rt_dynamic: Some(RtDynamicMode::Rebuild),
                rt_skinned_geometry: Some(false),
            }
        );
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
            let render = match cli.resolved_command() {
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

    #[test]
    fn debug_starts_the_server_on_the_named_world() {
        let cli = Cli::try_parse_from(["concinnity", "debug", "-f", "world.jsonl"]).unwrap();
        let Commands::Debug(a) = cli.resolved_command() else {
            panic!("expected debug");
        };
        assert_eq!(a.file.as_deref(), Some("world.jsonl"));
        assert_eq!(a.debug_port, 8777, "the same default cn mcp dials");
    }

    #[test]
    fn debug_takes_no_subcommand() {
        for argv in [
            ["concinnity", "debug", "send"],
            ["concinnity", "debug", "screenshot"],
            ["concinnity", "debug", "watch"],
        ] {
            assert!(Cli::try_parse_from(argv).is_err(), "{argv:?}");
        }
    }

    #[test]
    fn mcp_defaults_to_the_debug_port() {
        let cli = Cli::try_parse_from(["concinnity", "mcp"]).unwrap();
        let Commands::Mcp(a) = cli.resolved_command() else {
            panic!("expected mcp");
        };
        assert_eq!(a.debug_port, 8777);
        let cli = Cli::try_parse_from(["concinnity", "mcp", "--debug-port", "9001"]).unwrap();
        let Commands::Mcp(a) = cli.resolved_command() else {
            panic!("expected mcp");
        };
        assert_eq!(a.debug_port, 9001);
    }

    #[test]
    fn export_defaults() {
        let cli = Cli::try_parse_from(["concinnity", "export"]).unwrap();
        let Commands::Export(e) = cli.resolved_command() else {
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
        let Commands::Add(a) = cli.resolved_command() else {
            panic!("expected add");
        };
        assert_eq!(a.target, "Logger");
        assert_eq!(a.id.as_deref(), Some("log"));
    }

    #[test]
    fn init_takes_no_arguments() {
        let cli = Cli::try_parse_from(["concinnity", "init"]).unwrap();
        assert!(matches!(cli.resolved_command(), Commands::Init));
        assert!(Cli::try_parse_from(["concinnity", "init", "extra"]).is_err());
    }

    #[test]
    fn new_requires_a_path() {
        assert!(Cli::try_parse_from(["concinnity", "new"]).is_err());
        let cli = Cli::try_parse_from(["concinnity", "new", "my-app"]).unwrap();
        let Commands::New(a) = cli.resolved_command() else {
            panic!("expected new");
        };
        assert_eq!(a.path, "my-app");
    }

    // `build` and `test` share the same optional --file, defaulting to discovery.
    #[test]
    fn build_and_test_take_an_optional_world_file() {
        let cli = Cli::try_parse_from(["concinnity", "build"]).unwrap();
        let Commands::Build(a) = cli.resolved_command() else {
            panic!("expected build");
        };
        assert!(a.file.is_none());

        let cli = Cli::try_parse_from(["concinnity", "build", "-f", "w.jsonl"]).unwrap();
        let Commands::Build(a) = cli.resolved_command() else {
            panic!("expected build");
        };
        assert_eq!(a.file.as_deref(), Some("w.jsonl"));

        let cli = Cli::try_parse_from(["concinnity", "test", "--file", "w.jsonl"]).unwrap();
        let Commands::Test(a) = cli.resolved_command() else {
            panic!("expected test");
        };
        assert_eq!(a.file.as_deref(), Some("w.jsonl"));
    }

    #[test]
    fn list_flags_are_independent() {
        let cli = Cli::try_parse_from(["concinnity", "list"]).unwrap();
        let Commands::List(a) = cli.resolved_command() else {
            panic!("expected list");
        };
        assert!(!a.expanded);
        assert!(!a.systems);

        let cli = Cli::try_parse_from(["concinnity", "list", "--expanded", "--systems"]).unwrap();
        let Commands::List(a) = cli.resolved_command() else {
            panic!("expected list");
        };
        assert!(a.expanded);
        assert!(a.systems);
    }

    #[test]
    fn explain_requires_an_id() {
        assert!(Cli::try_parse_from(["concinnity", "explain"]).is_err());
        let cli = Cli::try_parse_from(["concinnity", "explain", "gfx", "-f", "w.jsonl"]).unwrap();
        let Commands::Explain(a) = cli.resolved_command() else {
            panic!("expected explain");
        };
        assert_eq!(a.id, "gfx");
        assert_eq!(a.file.as_deref(), Some("w.jsonl"));
    }

    #[test]
    fn rm_requires_an_id() {
        assert!(Cli::try_parse_from(["concinnity", "rm"]).is_err());
        let cli = Cli::try_parse_from(["concinnity", "rm", "Prop#3"]).unwrap();
        let Commands::Rm(a) = cli.resolved_command() else {
            panic!("expected rm");
        };
        assert_eq!(a.id, "Prop#3");
    }

    #[test]
    fn add_takes_a_scaffold_template() {
        let cli = Cli::try_parse_from(["concinnity", "add", "scene.glb", "-t", "minimal-3d-world"])
            .unwrap();
        let Commands::Add(a) = cli.resolved_command() else {
            panic!("expected add");
        };
        assert_eq!(a.target, "scene.glb");
        assert_eq!(a.template.as_deref(), Some("minimal-3d-world"));
    }

    #[test]
    fn docs_root_defaults_to_the_current_directory() {
        let cli = Cli::try_parse_from(["concinnity", "docs"]).unwrap();
        let Commands::Docs(a) = cli.resolved_command() else {
            panic!("expected docs");
        };
        assert_eq!(a.root.as_deref(), Some("."));

        let cli = Cli::try_parse_from(["concinnity", "docs", "--root", "/engine"]).unwrap();
        let Commands::Docs(a) = cli.resolved_command() else {
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
        let Commands::Run(a) = cli.resolved_command() else {
            panic!("expected run");
        };
        assert_eq!(a.validation, None);

        for (arg, expected) in [("true", true), ("false", false)] {
            let cli = Cli::try_parse_from(["concinnity", "run", "--validation", arg]).unwrap();
            let Commands::Run(a) = cli.resolved_command() else {
                panic!("expected run");
            };
            assert_eq!(a.validation, Some(expected));
        }
    }

    #[test]
    fn editor_takes_a_file_debug_port_and_validation() {
        let cli = Cli::try_parse_from(["concinnity", "editor"]).unwrap();
        let Commands::Editor(a) = cli.resolved_command() else {
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
        let Commands::Editor(a) = cli.resolved_command() else {
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
        let Commands::Export(e) = cli.resolved_command() else {
            panic!("expected export");
        };
        assert_eq!(e.file.as_deref(), Some("w.jsonl"));
        assert_eq!(e.name.as_deref(), Some("My Game"));
        assert_eq!(e.version.as_deref(), Some("2.0.0"));
        assert_eq!(e.platform.as_deref(), Some("macos"));
        assert_eq!(e.out, "build");
        assert_eq!(e.format, BundleFormatArg::Dir);
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
        let Commands::Debug(a) = cli.resolved_command() else {
            panic!("expected debug");
        };
        assert_eq!(a.debug_port, 9100);
        assert_eq!(a.validation, Some(true));
    }
}
