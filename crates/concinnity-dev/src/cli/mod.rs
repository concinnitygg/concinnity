//! The `concinnity` binary's argv face: the clap command tree, the value enums
//! that mirror engine types so no other layer carries a clap dependency, the
//! macOS validation re-exec, and the dispatch that turns a parsed command into
//! a call.

mod commands;
mod dispatch;
mod reexec;
mod value_enums;

use clap::Parser;

use commands::{EditorArgs, RenderArgs};

pub(crate) use commands::Commands;

const BANNER: &str = r#"
   ______                                
  / ____/___  ____  ___________  ____  __________  __
 / /   / __ \/ __ \/ ___/ / __ \/ __ \/ /_  __/ / / /
/ /___/ /_/ / / / / /__/ / / / / / / / / / / / /_/ /
\____/\____/_/ /_/\___/_/_/ /_/_/ /_/_/ /_/  \__, /
                                            /____/"#;

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
        reexec::reexec_with_metal_validation(&cli);
        Self(cli)
    }

    /// Run the command the arguments named, against `tree`.
    pub fn dispatch(&self, tree: &concinnity_engine::StateTree) -> std::io::Result<()> {
        dispatch::dispatch(&self.0, tree)
    }
}

#[derive(Parser, Debug)]
#[command(name = "concinnity")]
#[command(about = BANNER, long_about = None)]
// clap renders its own flag as "{name} {version}", which is the pair
// `command::version_line` prints, so `cn version` and `cn --version` render
// the same line off one source. The auto-generated flag is off only so the
// short can carry `-v` alongside clap's conventional `-V`.
#[command(version = crate::command::version_details(), disable_version_flag = true)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Option<Commands>,

    /// Print the version
    #[arg(short = 'V', short_alias = 'v', long, action = clap::ArgAction::Version)]
    pub(crate) version: (),
}

// What a bare `concinnity` runs. Launching the binary with no argv at all --
// a double click, where there is no terminal to type a subcommand into --
// opens the editor on the world discovered from `worlds/`.
const DEFAULT_COMMAND: Commands = Commands::Editor(EditorArgs {
    file: None,
    debug_port: None,
    validation: None,
    render: RenderArgs {
        quality_preset: None,
        rt_dynamic: None,
        rt_skinned_geometry: None,
    },
});

impl Cli {
    /// The command this run performs: the one argv named, or [`DEFAULT_COMMAND`].
    pub(crate) fn resolved_command(&self) -> &Commands {
        self.command.as_ref().unwrap_or(&DEFAULT_COMMAND)
    }
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

    // A bare invocation is the double-click case: no terminal, no subcommand,
    // and the useful thing to do is open a window.
    #[test]
    fn a_bare_invocation_runs_the_editor() {
        let cli = Cli::try_parse_from(["concinnity"]).unwrap();
        assert!(cli.command.is_none());
        let Commands::Editor(a) = cli.resolved_command() else {
            panic!("expected the editor by default");
        };
        assert!(a.file.is_none());
        assert!(a.debug_port.is_none());
        assert!(a.validation.is_none());
        assert!(a.render.quality_preset.is_none());
        assert!(a.render.rt_dynamic.is_none());
        assert!(a.render.rt_skinned_geometry.is_none());
    }

    // The default is `editor` itself, not a look-alike: an unknown subcommand
    // still fails rather than silently falling back to it.
    #[test]
    fn an_unknown_subcommand_is_still_an_error() {
        assert!(Cli::try_parse_from(["concinnity", "edtior"]).is_err());
    }
}
