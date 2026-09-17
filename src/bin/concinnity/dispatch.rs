// The one file that joins argv to behavior: match the parsed command and call
// the matching entry point in concinnity-dev. Everything here is a call; any
// logic worth testing belongs on the other side of it, where `tests/cli.rs`
// and the library's own tests can reach it.

use crate::cli::{Cli, Commands};
use concinnity_dev::command;
use concinnity_dev::export::ExportOptions;
use concinnity_engine::StateTree;

pub(crate) fn dispatch(cli: &Cli, tree: &StateTree) -> std::io::Result<()> {
    match cli.resolved_command() {
        Commands::Init => command::init(),
        Commands::New(args) => command::new(&args.path),
        Commands::Build(args) => command::build(args.file.as_deref()),
        Commands::Run(args) => concinnity_engine::app::run(
            tree,
            concinnity_engine::app::run::RunOptions {
                mode: if args.serial {
                    concinnity_engine::app::run::PipelineMode::Serial
                } else {
                    concinnity_engine::app::run::PipelineMode::Pipelined
                },
                schedule: if args.serial_schedule {
                    concinnity_core::ecs::ScheduleMode::Serial
                } else {
                    concinnity_core::ecs::ScheduleMode::Parallel
                },
                screenshot: args.screenshot.clone(),
                max_frames: args.frames,
                launch: args.render.launch(args.validation, false),
            },
        ),
        Commands::Debug(args) => {
            let launch = args.render.launch(args.validation, true);
            concinnity_dev::run_debug(launch, args.file.as_deref(), args.debug_port)
        }
        Commands::Editor(args) => {
            // Every editor session hot-reloads, with or without a debug port.
            let launch = args.render.launch(args.validation, true);
            concinnity_dev::run_editor(launch, args.file.as_deref(), args.debug_port)
        }
        Commands::Add(args) => {
            command::add(args.name.as_deref(), &args.target, args.template.as_deref())
        }
        Commands::Rm(args) => command::rm(&args.name),
        Commands::List(args) => command::list(args.file.as_deref(), args.expanded, args.systems),
        Commands::Explain(args) => command::explain(&args.name, args.file.as_deref()),
        Commands::Docs(args) => concinnity_dev::docs::docs(args.root.as_deref()),
        Commands::Test(args) => {
            let path = args.file.as_deref().unwrap_or("");
            command::check(path)
        }
        Commands::Export(args) => concinnity_dev::export::export(&ExportOptions {
            world: args.file.clone(),
            name: args.name.clone(),
            version: args.version.clone(),
            platform: args.platform.clone(),
            out: args.out.clone(),
            format: args.format.into(),
            dmg: args.dmg,
        }),
        Commands::Mcp(args) => concinnity_dev::run_mcp(args.debug_port),
        Commands::Version => command::version(),
    }
}
