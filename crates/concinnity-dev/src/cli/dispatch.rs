// The one file that joins argv to behavior: match the parsed command and call
// the matching entry point. Everything here is a call; any logic worth testing
// belongs on the other side of it, where the `concinnity` package's
// `tests/cli.rs` and this crate's own tests can reach it.

use super::{Cli, Commands};
use crate::command;
use crate::command::export::ExportOptions;
use concinnity_engine::launch::LaunchArgs;
use concinnity_host::store::paths::StateTree;
use std::path::Path;

pub(super) fn dispatch(cli: &Cli, tree: &StateTree) -> std::io::Result<()> {
    // A command that launches an application installs its own logging as it starts.
    if cli.quiet && cli.command.launch_args(cli.quiet).is_none() {
        concinnity_runtime::install_logging(true);
    }
    let launch = |args: &LaunchArgs| LaunchArgs {
        quiet: cli.quiet,
        ..args.clone()
    };
    // Shaders compile with the dxc `concinnity vendor` pinned, and a launch
    // loads the runtime SDKs it staged, which the renderer has to know of
    // before anything probes a device.
    crate::vendor::use_vendored_tools();
    if cli.command.launch_args(cli.quiet).is_some() {
        crate::vendor::use_staged_runtime();
    }
    match &cli.command {
        Commands::Init => command::init(),
        Commands::New(args) => command::new(Path::new(&args.path)),
        Commands::Cook(args) => command::cook(
            args.target.target(),
            if args.verbose {
                command::Verbosity::Verbose
            } else {
                command::Verbosity::Normal
            },
        ),
        Commands::Check(args) => command::check(args.file.as_deref()),
        Commands::Run(args) => {
            crate::run_app(args.target.target(), launch(&args.launch.args), tree)
        }
        Commands::Bench(args) => {
            crate::run_app(args.target.target(), launch(&args.launch.args), tree)
        }
        Commands::Edit(args) => {
            let launch = launch(&args.launch.args);
            crate::run::select_graphics_api(&launch, tree);
            let file = args
                .target
                .target()
                .named()?
                .map(|path| path.to_string_lossy().into_owned());
            crate::run_editor(launch, file.as_deref())
        }
        Commands::Add(args) => command::add(
            args.target.target(),
            args.id.as_deref(),
            &args.targets,
            args.template.as_deref(),
        ),
        Commands::Rm(args) => command::rm(args.target.target(), &args.id),
        Commands::List(args) => command::list(args.target.target(), args.expanded, args.systems),
        Commands::Explain(args) => command::explain(&args.id, args.target.target()),
        Commands::Docs(args) => crate::docs::docs(args.root.as_deref()),
        Commands::Export(args) => command::export::export(&ExportOptions {
            target: args.target.target(),
            name: args.name.clone(),
            version: args.version.clone(),
            platform: args.platform.clone(),
            out: args.out.clone(),
            format: args.format.into(),
            dmg: args.dmg,
        }),
        Commands::Mcp(args) => crate::run_mcp(args.debug_port),
        Commands::Vendor(args) => crate::vendor::vendor(&crate::vendor::VendorRequest {
            components: &args.components,
            force: args.force,
            accept_licenses: args.accept_licenses,
            status: args.status,
        }),
        Commands::Version => {
            command::version();
            Ok(())
        }
    }
}
