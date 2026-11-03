// The command line reference page: every command `concinnity` takes, with its
// rustdoc, usage, arguments and examples, read off the clap command tree the
// derives build from that rustdoc. The launch flags come from the engine's
// launch table, which feeds the same tree.

use clap::{Arg, ArgAction, Command, CommandFactory};

use super::page::AUTOGEN_MARKER;
use crate::cli::Cli;

// Where the page lands, relative to the directory given on the command line.
pub(super) const PAGE: &str = "docs/cli.md";

/// The command line reference page.
pub(super) fn render() -> String {
    render_tree(Cli::command())
}

fn render_tree(mut root: Command) -> String {
    root.build();
    let name = root.get_name().to_string();
    let mut out = format!(
        "{AUTOGEN_MARKER}\n\n# Command line\n\n`{name} <COMMAND> --help` prints a \
         command's section in a terminal.\n\n## Options every command takes\n\n"
    );
    for arg in root.get_arguments().filter(|a| a.is_global_set()) {
        out.push_str(&arg_line(arg, &root));
    }
    for command in commands(&root) {
        out.push('\n');
        out.push_str(&command_section(&name, command));
    }
    out
}

// The documented subcommands, without the `help` one clap adds.
fn commands(root: &Command) -> impl Iterator<Item = &Command> {
    root.get_subcommands()
        .filter(|c| !c.is_hide_set() && c.get_name() != "help")
}

fn command_section(root: &str, command: &Command) -> String {
    let mut out = format!("## `{root} {}`\n\n", command.get_name());
    let doc = command
        .get_long_about()
        .or_else(|| command.get_about())
        .map(ToString::to_string)
        .unwrap_or_default();
    out.push_str(doc.trim());
    out.push_str("\n\n");
    let usage = command.clone().render_usage().to_string();
    let usage = usage.strip_prefix("Usage: ").unwrap_or(&usage);
    out.push_str("Usage: `");
    out.push_str(usage);
    out.push_str("`\n");

    let args: Vec<String> = command
        .get_arguments()
        .filter(|a| !a.is_global_set() && !a.is_hide_set() && !is_builtin(a))
        .filter(|a| negates(a, command).is_none())
        .map(|a| arg_line(a, command))
        .collect();
    if !args.is_empty() {
        out.push('\n');
        out.push_str(&args.concat());
    }
    out
}

// clap's own `--help` and `--version`.
fn is_builtin(arg: &Arg) -> bool {
    matches!(arg.get_action(), ArgAction::Help | ArgAction::Version)
}

// The switch `arg` turns off, for a `--no-<name>` beside a `--<name>`.
fn negates<'a>(arg: &Arg, command: &'a Command) -> Option<&'a Arg> {
    let name = arg.get_long()?.strip_prefix("no-")?;
    command.get_arguments().find(|a| a.get_long() == Some(name))
}

// One bullet: the argument's spelling, then its help, the values it takes and
// its default.
fn arg_line(arg: &Arg, command: &Command) -> String {
    let mut line = format!("- `{}`", spelling(arg, command));
    let help = arg
        .get_long_help()
        .or_else(|| arg.get_help())
        .map(ToString::to_string)
        .unwrap_or_default();
    if !help.is_empty() {
        line.push_str(": ");
        line.push_str(&prose(&help));
    }
    let values: Vec<String> = arg
        .get_possible_values()
        .iter()
        .filter(|v| !v.is_hide_set())
        .map(|v| format!("`{}`", v.get_name()))
        .collect();
    let takes_values = arg.get_action().takes_values();
    if !values.is_empty() && takes_values {
        line.push_str(" One of ");
        line.push_str(&values.join(", "));
        line.push('.');
    }
    let defaults: Vec<String> = arg
        .get_default_values()
        .iter()
        .map(|v| format!("`{}`", v.to_string_lossy()))
        .collect();
    if !defaults.is_empty() && takes_values && !arg.is_positional() {
        line.push_str(" Defaults to ");
        line.push_str(&defaults.join(", "));
        line.push('.');
    }
    line.push('\n');
    line
}

// How an argument is written: a positional's value name, or an option's short
// and long forms with its value. A switch with a `--no-` twin is `--[no-]name`.
fn spelling(arg: &Arg, command: &Command) -> String {
    let value = || {
        let name = arg
            .get_value_names()
            .and_then(|n| n.first())
            .map_or_else(|| arg.get_id().to_string(), ToString::to_string);
        if arg.get_num_args().is_some_and(|n| n.min_values() == 0) {
            format!(" [<{name}>]")
        } else {
            format!(" <{name}>")
        }
    };
    if arg.is_positional() {
        return value().trim_start().to_string();
    }
    let negatable = arg.get_long().is_some_and(|long| {
        command
            .get_arguments()
            .any(|a| a.get_long() == Some(&format!("no-{long}")))
    });
    let long = arg.get_long().map(|long| {
        if negatable {
            format!("--[no-]{long}")
        } else {
            format!("--{long}")
        }
    });
    let mut out = [arg.get_short().map(|s| format!("-{s}")), long]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(", ");
    if arg.get_action().takes_values() {
        out.push_str(&value());
    }
    out
}

// Help text as prose on one line: each paragraph a sentence, with a capital and
// a full stop.
fn prose(help: &str) -> String {
    help.split("\n\n")
        .map(super::prose::collapse_doc)
        .filter(|p| !p.is_empty())
        .map(|p| {
            let mut chars = p.chars();
            let mut out: String = chars
                .next()
                .map(|c| c.to_uppercase().chain(chars).collect())
                .unwrap_or_default();
            if !out.ends_with(['.', '!', '?']) {
                out.push('.');
            }
            out
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    // A tree the test wrote, in the shape the derives build: a documented
    // command with an example block, a negatable switch pair, a valued option
    // with a default, and a positional.
    fn tree() -> Command {
        Command::new("tool")
            .arg(
                Arg::new("quiet")
                    .short('q')
                    .long("quiet")
                    .help("Do not print log messages")
                    .action(ArgAction::SetTrue)
                    .global(true),
            )
            .subcommand(
                Command::new("make")
                    .about("Make a thing")
                    .long_about("Make a thing\n\nIt is made.\n\n```sh\ntool make box\n```")
                    .arg(Arg::new("target").help("What to make").required(true))
                    .arg(
                        Arg::new("vsync")
                            .long("vsync")
                            .help("Turn vertical sync on or off")
                            .action(ArgAction::SetTrue),
                    )
                    .arg(
                        Arg::new("no-vsync")
                            .long("no-vsync")
                            .help("Turn off --vsync")
                            .action(ArgAction::SetTrue),
                    )
                    .arg(
                        Arg::new("format")
                            .long("format")
                            .help("Output format")
                            .value_parser(["zip", "dir"])
                            .default_value("zip"),
                    )
                    .arg(
                        Arg::new("port")
                            .long("port")
                            .value_name("PORT")
                            .num_args(0..=1)
                            .default_missing_value("8777")
                            .help("Serve on this port"),
                    ),
            )
    }

    #[test]
    fn a_command_section_carries_its_doc_usage_and_arguments() {
        let page = render_tree(tree());
        assert!(page.starts_with(AUTOGEN_MARKER), "{page}");
        assert!(
            page.contains(
                "## Options every command takes\n\n- `-q, --quiet`: Do not print log messages.\n"
            ),
            "{page}"
        );
        for line in [
            "## `tool make`\n\nMake a thing\n\nIt is made.\n\n```sh\ntool make box\n```\n\n",
            "Usage: `tool make [OPTIONS] <target>`\n",
            "- `<target>`: What to make.\n",
            "- `--[no-]vsync`: Turn vertical sync on or off.\n",
            "- `--format <format>`: Output format. One of `zip`, `dir`. Defaults to `zip`.\n",
            "- `--port [<PORT>]`: Serve on this port.\n",
        ] {
            assert!(page.contains(line), "{line:?} in:\n{page}");
        }
        assert!(!page.contains("--no-vsync`"), "{page}");
        assert!(!page.contains("- `-h, --help`"), "{page}");
        assert!(!page.contains("## `tool help`"), "{page}");
    }

    #[test]
    fn help_reads_as_sentences() {
        assert_eq!(prose("do a thing"), "Do a thing.");
        assert_eq!(prose("Done."), "Done.");
        assert_eq!(
            prose("Two\nlines (default: on)"),
            "Two lines (default: on)."
        );
        assert_eq!(
            prose("The id\n\nWithout one, none."),
            "The id. Without one, none."
        );
    }

    // The examples in the shipped tree: every command has some, and each is a
    // whole `concinnity` command line the parser accepts.
    #[test]
    fn every_command_has_examples_that_parse() {
        let mut root = Cli::command();
        root.build();
        for command in commands(&root) {
            let doc = command
                .get_long_about()
                .map(ToString::to_string)
                .unwrap_or_default();
            let examples = example_lines(&doc);
            assert!(
                !examples.is_empty(),
                "{} has no example",
                command.get_name()
            );
            for line in examples {
                let argv = split_shell_words(&line);
                assert_eq!(
                    argv.first().map(String::as_str),
                    Some("concinnity"),
                    "{line}"
                );
                assert_eq!(
                    argv.get(1).map(String::as_str),
                    Some(command.get_name()),
                    "{line}"
                );
                if let Err(e) = Cli::try_parse_from(&argv) {
                    panic!("{line}: {e}");
                }
            }
        }
    }

    #[test]
    fn the_shipped_page_documents_every_command() {
        let page = render();
        let mut root = Cli::command();
        root.build();
        for command in commands(&root) {
            assert!(
                page.contains(&format!("## `concinnity {}`\n", command.get_name())),
                "{}",
                command.get_name()
            );
        }
        assert!(page.contains("- `--[no-]vsync`: "), "{page}");
        assert!(page.contains("- `--debug-port [<PORT>]`: "), "{page}");
    }

    // The lines of the doc's ```sh blocks.
    fn example_lines(doc: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut in_block = false;
        for line in doc.lines() {
            match line.trim() {
                "```sh" => in_block = true,
                "```" => in_block = false,
                l if in_block && !l.is_empty() => out.push(l.to_string()),
                _ => {}
            }
        }
        out
    }

    // Split a command line the way a POSIX shell does for plain words and
    // single- or double-quoted ones.
    fn split_shell_words(line: &str) -> Vec<String> {
        let mut words = Vec::new();
        let mut word = String::new();
        let mut quote = None;
        let mut started = false;
        for c in line.chars() {
            match (quote, c) {
                (Some(q), c) if c == q => quote = None,
                (Some(_), c) => word.push(c),
                (None, '\'' | '"') => {
                    quote = Some(c);
                    started = true;
                }
                (None, c) if c.is_whitespace() => {
                    if started {
                        words.push(core::mem::take(&mut word));
                        started = false;
                    }
                }
                (None, c) => {
                    word.push(c);
                    started = true;
                }
            }
        }
        if started {
            words.push(word);
        }
        words
    }

    #[test]
    fn shell_words_split_on_unquoted_space() {
        assert_eq!(
            split_shell_words(r#"a 'b c' "d e" '["W", {}]'"#),
            ["a", "b c", "d e", r#"["W", {}]"#]
        );
    }
}
