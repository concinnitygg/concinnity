// The one file that joins argv to behavior: match the parsed command and call
// the matching entry point. Everything here is a call; any logic worth testing
// belongs on the other side of it, where `tests/cli.rs` and this crate's own
// tests can reach it.

use super::{Cli, Commands};
use crate::command;
use crate::export::ExportOptions;
use concinnity_engine::launch::LaunchArgs;
use concinnity_host::store::paths::StateTree;

pub(super) fn dispatch(cli: &Cli, tree: &StateTree) -> std::io::Result<()> {
    // A command that launches a world installs its own logging as it starts.
    if cli.quiet && cli.command.launch_args(cli.quiet).is_none() {
        concinnity_runtime::install_logging(true);
    }
    let launch = |args: &LaunchArgs| LaunchArgs {
        quiet: cli.quiet,
        ..args.clone()
    };
    match &cli.command {
        Commands::Init => command::init(),
        Commands::New(args) => command::new(&args.path),
        Commands::Cook(args) => command::cook(
            args.world.file(),
            if args.verbose {
                command::Verbosity::Verbose
            } else {
                command::Verbosity::Normal
            },
        ),
        Commands::Check(args) => command::check(args.world.file()),
        Commands::Run(args) => crate::run_world(args.world.file(), launch(&args.launch.args), tree),
        Commands::Bench(args) => {
            crate::run_world(args.world.file(), launch(&args.launch.args), tree)
        }
        Commands::Edit(args) => {
            let launch = launch(&args.launch.args);
            crate::run::select_graphics_api(&launch, tree);
            crate::run_editor(launch, args.world.file())
        }
        Commands::Add(args) => {
            command::add(args.id.as_deref(), &args.target, args.template.as_deref())
        }
        Commands::Rm(args) => command::rm(&args.id),
        Commands::List(args) => command::list(args.world.file(), args.expanded, args.systems),
        Commands::Explain(args) => command::explain(&args.id, args.world.file()),
        Commands::Docs(args) => crate::docs::docs(args.root.as_deref()),
        Commands::Export(args) => crate::export::export(&ExportOptions {
            world: args.world.file().map(str::to_string),
            name: args.name.clone(),
            version: args.version.clone(),
            platform: args.platform.clone(),
            out: args.out.clone(),
            format: args.format.into(),
            dmg: args.dmg,
        }),
        Commands::Mcp(args) => crate::run_mcp(args.debug_port),
        Commands::Version => {
            command::version();
            Ok(())
        }
    }
}
