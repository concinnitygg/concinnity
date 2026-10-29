//! The `concinnity` binary's argv face: the clap command tree, the value enums
//! that mirror engine types so no other layer carries a clap dependency, and
//! the dispatch that turns a parsed command into a call.

mod commands;
mod dispatch;
mod launch_args;
mod value_enums;

use clap::Parser;

pub(crate) use commands::Commands;

const BANNER: &str = r"
   ______                                
  / ____/___  ____  ___________  ____  __________  __
 / /   / __ \/ __ \/ ___/ / __ \/ __ \/ /_  __/ / / /
/ /___/ /_/ / / / / /__/ / / / / / / / / / / / /_/ /
\____/\____/_/ /_/\___/_/_/ /_/_/ /_/_/ /_/  \__, /
                                            /____/";

/// One run of the `concinnity` command line: the arguments it was given, and
/// the command they name.
pub struct Invocation(Cli);

impl Invocation {
    /// Parse this process's arguments, exiting the process on a parse error or
    /// a `--help` / `--version` request the way a command line is expected to.
    ///
    /// On macOS a command that asks for graphics validation replaces this
    /// process image here: Metal reads its validation switch out of the
    /// environment during early framework initialization, so the only way to
    /// turn it on is to relaunch with the variable already set. Call this
    /// before spawning a thread or touching Metal.
    pub fn from_args() -> Self {
        let cli = Cli::parse();
        if let Some(args) = cli.command.launch_args(cli.quiet) {
            concinnity_runtime::reexec_with_metal_validation(args.graphics_validation);
        }
        Self(cli)
    }

    /// Run the command the arguments named, against `tree`.
    ///
    /// # Errors
    ///
    /// Returns the error the command failed with.
    pub fn dispatch(&self, tree: &concinnity_host::store::paths::StateTree) -> std::io::Result<()> {
        dispatch::dispatch(&self.0, tree)
    }
}

#[derive(Parser, Debug)]
#[command(name = "concinnity")]
#[command(about = BANNER, long_about = None)]
#[command(arg_required_else_help = true)]
// clap renders its own `-V, --version` as "{name} {version}", which is the
// pair `command::version_line` prints, so `cn version` and `cn --version`
// render the same line off one source.
#[command(version = crate::command::version_details())]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Commands,

    /// Do not print log messages
    #[arg(short = 'q', long, global = true)]
    pub(crate) quiet: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    // Walks the whole command tree and asserts clap's own invariants (no
    // conflicting args, valid defaults, unique names). A cheap guard against
    // an ill-formed derive that would only surface at runtime otherwise.
    #[test]
    fn cli_config_is_valid() {
        Cli::command().debug_assert();
    }

    // A bare invocation prints the help rather than running anything.
    #[test]
    fn a_bare_invocation_asks_for_a_command() {
        let err = Cli::try_parse_from(["concinnity"]).unwrap_err();
        assert_eq!(
            err.kind(),
            clap::error::ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
        );
    }

    #[test]
    fn the_version_is_upper_case_v_only() {
        let err = Cli::try_parse_from(["concinnity", "-V"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::DisplayVersion);
        assert!(Cli::try_parse_from(["concinnity", "-v"]).is_err());
        let cli = Cli::try_parse_from(["concinnity", "version"]).unwrap();
        assert!(matches!(cli.command, Commands::Version));
    }

    #[test]
    fn an_unknown_subcommand_is_an_error() {
        assert!(Cli::try_parse_from(["concinnity", "edtior"]).is_err());
    }
}
