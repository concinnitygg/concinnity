//! Workspace tasks, run as `cargo xtask <task>` through the alias in
//! `.cargo/config.toml`.
//!
//! The one task is `vendor`, which puts the third-party binaries this checkout
//! builds against under `vendor/`. A component is pinned to one release and
//! unpacks into `vendor/<name>-<version>-<os>-<arch>/`, which the build prefers
//! over whatever the machine happens to have installed, so what a checkout
//! builds with is a property of the revision rather than of the host.
//!
//! Nothing here runs during `cargo build`: a build script that reached the
//! network could not build offline, could not build on docs.rs, and would pull
//! a dependency past Cargo.lock, cargo-vendor and cargo-deny. Vendoring is a
//! setup step, run once, and the build only ever reads what it left behind.
//! That is also why this is a crate of its own rather than a `cn` subcommand:
//! the engine crates need dxc to compile, and vendoring is how a fresh checkout
//! gets one, so nothing here may depend on them.

mod components;
mod fetch;
mod host;
mod source_build;
mod status;
mod tools;
mod unpack;

use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::builder::PossibleValuesParser;
use clap::{Args, Parser, Subcommand};

use crate::components::{COMPONENTS, Component, NAMES};
use crate::host::Host;

#[derive(Parser)]
#[command(bin_name = "cargo xtask", about = "Workspace setup tasks")]
struct Cli {
    #[command(subcommand)]
    task: Task,
}

#[derive(Subcommand)]
enum Task {
    /// Vendor the third-party binaries this checkout builds against: with no
    /// verb, fetch what this host downloads and build what it cannot
    #[command(args_conflicts_with_subcommands = true)]
    Vendor {
        #[command(subcommand)]
        verb: Option<Verb>,
        #[command(flatten)]
        selection: Selection,
        #[command(flatten)]
        options: source_build::Options,
    },
}

#[derive(Subcommand)]
enum Verb {
    /// Download the pinned release
    Fetch {
        #[command(flatten)]
        selection: Selection,
        /// Re-download over an existing copy
        #[arg(long)]
        force: bool,
    },
    /// Build what no upstream publishes
    Build {
        #[command(flatten)]
        selection: Selection,
        #[command(flatten)]
        options: source_build::Options,
    },
    /// Report what the build would pick
    Status {
        #[command(flatten)]
        selection: Selection,
    },
}

#[derive(Args)]
struct Selection {
    /// The components to act on, every one when none is named. A component the
    /// verb does not apply to on this host is then skipped: most are
    /// downloaded, and `dxc` is downloaded on Windows and Linux and built from
    /// the same commit on macOS.
    #[arg(value_parser = PossibleValuesParser::new(NAMES))]
    components: Vec<String>,
}

impl Selection {
    // The named components, or every one, with whether any was named.
    fn resolve(&self) -> (Vec<&'static Component>, bool) {
        if self.components.is_empty() {
            return (COMPONENTS.to_vec(), false);
        }
        let named = self
            .components
            .iter()
            .filter_map(|name| components::find(name))
            .collect();
        (named, true)
    }
}

/// Why a task stopped, as the message it prints.
#[derive(Debug)]
pub(crate) struct Failure(String);

pub(crate) type Result<T = ()> = std::result::Result<T, Failure>;

pub(crate) fn fail<T>(message: impl Into<String>) -> Result<T> {
    Err(Failure(message.into()))
}

/// Names what was being done when an error surfaced.
pub(crate) trait Context<T> {
    fn context(self, what: impl Display) -> Result<T>;
}

impl<T, E: Display> Context<T> for std::result::Result<T, E> {
    fn context(self, what: impl Display) -> Result<T> {
        self.map_err(|e| Failure(format!("{what}: {e}")))
    }
}

/// The engine checkout, two levels above this crate's manifest.
pub(crate) fn workspace_root() -> &'static Path {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().and_then(Path::parent).unwrap_or(manifest)
}

pub(crate) fn vendor_dir() -> PathBuf {
    workspace_root().join("vendor")
}

fn main() -> ExitCode {
    match run(Cli::parse().task) {
        Ok(()) => ExitCode::SUCCESS,
        Err(Failure(message)) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run(task: Task) -> Result {
    let Task::Vendor {
        verb,
        selection,
        options,
    } = task;
    let host = Host::current();
    let Some(verb) = verb else {
        return each(&selection, |component, named| {
            vendor(component, host, named, &options)
        });
    };
    match &verb {
        Verb::Fetch { selection, force } => each(selection, |component, named| {
            fetch::fetch(component, host, named, *force)
        }),
        Verb::Build { selection, options } => each(selection, |component, named| {
            source_build::build(component, host, named, options)
        }),
        Verb::Status { selection } => each(selection, |component, named| {
            status::status(component, host, named)
        }),
    }
}

// Fetch or build `component`, whichever this host gets it by.
fn vendor(
    component: &Component,
    current: Option<Host>,
    named: bool,
    options: &source_build::Options,
) -> Result {
    let Some(host) = component.host_for(current, named)? else {
        println!("{}: no release for this host, skipped", component.name);
        return Ok(());
    };
    if component.is_built(host) {
        source_build::build(component, Some(host), named, options)
    } else {
        fetch::fetch(component, Some(host), named, options.force)
    }
}

// Apply a verb to each selected component, stopping at the first failure.
fn each(selection: &Selection, apply: impl Fn(&Component, bool) -> Result) -> Result {
    let (chosen, named) = selection.resolve();
    chosen
        .into_iter()
        .try_for_each(|component| apply(component, named))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn the_command_tree_is_well_formed() {
        Cli::command().debug_assert();
    }

    #[test]
    fn unknown_components_are_refused() {
        assert!(Cli::try_parse_from(["xtask", "vendor", "fetch", "slang"]).is_err());
        assert!(Cli::try_parse_from(["xtask", "vendor", "fetch", "dxc", "xess"]).is_ok());
    }

    fn parse(args: &[&str]) -> Task {
        Cli::try_parse_from(["xtask"].iter().chain(args))
            .unwrap()
            .task
    }

    #[test]
    fn vendor_takes_components_without_a_verb() {
        let Task::Vendor {
            verb, selection, ..
        } = parse(&["vendor", "dxc", "--force"]);
        assert!(verb.is_none());
        assert_eq!(selection.components, ["dxc"]);

        let Task::Vendor { verb, .. } = parse(&["vendor", "fetch", "dxc"]);
        assert!(matches!(verb, Some(Verb::Fetch { .. })));
    }

    fn options() -> source_build::Options {
        source_build::Options { force: false }
    }

    #[test]
    fn vendor_skips_a_component_this_host_has_no_release_of() {
        let mac = Some(Host::MacosArm64);
        assert!(vendor(&components::XESS, mac, false, &options()).is_ok());
        assert!(vendor(&components::XESS, mac, true, &options()).is_err());
    }

    #[test]
    fn a_bare_verb_selects_every_component() {
        let (chosen, named) = Selection {
            components: Vec::new(),
        }
        .resolve();
        assert_eq!(chosen.len(), COMPONENTS.len());
        assert!(!named);

        let (chosen, named) = Selection {
            components: vec!["xess".into()],
        }
        .resolve();
        assert_eq!(chosen.iter().map(|c| c.name).collect::<Vec<_>>(), ["xess"]);
        assert!(named);
    }

    #[test]
    fn the_workspace_root_holds_the_workspace_manifest() {
        assert!(workspace_root().join("Cargo.toml").is_file());
        assert!(workspace_root().join("crates/xtask").is_dir());
    }
}
