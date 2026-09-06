// The one file that joins argv to behaviour: match the parsed command and call
// the matching entry point in concinnity-dev. Everything here is a call; any
// logic worth testing belongs on the other side of it, where `tests/cli.rs`
// and the library's own tests can reach it.

use crate::cli::{Cli, Commands};
use concinnity_dev::command;
use concinnity_engine::StateTree;
use concinnity_engine::app::dev_flags;

pub(crate) fn dispatch(cli: &Cli, tree: &StateTree) -> std::io::Result<()> {
    match &cli.command {
        Commands::Init => command::init(),
        Commands::New(args) => command::new(&args.path),
        Commands::Build(args) => command::build(args.file.as_deref()),
        Commands::Run(args) => {
            dev_flags::set_validation(args.validation);
            args.render.arm();
            concinnity_engine::app::run(
                tree,
                concinnity_engine::app::run::RunOptions {
                    mode: if args.serial {
                        concinnity_engine::app::run::PipelineMode::Serial
                    } else {
                        concinnity_engine::app::run::PipelineMode::Pipelined
                    },
                    schedule: if args.serial_schedule {
                        concinnity_engine::ecs::ScheduleMode::Serial
                    } else {
                        concinnity_engine::ecs::ScheduleMode::Parallel
                    },
                    screenshot: args.screenshot.clone(),
                    max_frames: args.frames,
                },
            )
        }
        Commands::Debug(args) => {
            dev_flags::set_enabled(true);
            dev_flags::set_validation(args.validation);
            args.render.arm();
            let port = args.debug_port.unwrap_or(8777);
            concinnity_dev::run_debug(args.file.as_deref(), port)
        }
        Commands::Editor(args) => {
            dev_flags::set_validation(args.validation);
            args.render.arm();
            // The editor is a dev session: arm the same dev flags the
            // `cn debug` host sets (above) so init captures hot-reload
            // sources and the backend takes its disk-first shader path.
            // Asset + shader hot-reload then works in every editor session;
            // a debug port only adds the MCP probe surface on top.
            dev_flags::set_enabled(true);
            concinnity_dev::run_editor(args.file.as_deref(), args.debug_port)
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
        Commands::Export(args) => concinnity_dev::export::export(
            args.file.as_deref(),
            args.name.as_deref(),
            args.version.as_deref(),
            args.platform.as_deref(),
            &args.out,
            &args.format,
            args.dmg,
        ),
        Commands::Mcp(args) => concinnity_dev::run_mcp(args.port),
        Commands::Version => command::version(),
    }
}
