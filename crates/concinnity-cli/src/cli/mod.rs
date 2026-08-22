// Command-line subcommand implementations for the `concinnity` binary: thin
// discovery/printing wrappers over the concinnity-editor authoring API and the
// concinnity-cook compile pipeline. CLI-only std code (argv handling, stdout
// tables, packaging) belongs here; the libraries stay free of it.

mod add;
mod build;
mod check;
mod docs;
mod explain;
mod export;
mod list;
mod new;
mod rm;

// Create and apply an asset to the current app
pub(crate) use add::add;

// Analyze the current app and report errors, but don't build blob files
pub(crate) use check::check;

// Regenerate the asset reference pages under docs/assets/
pub(crate) use docs::docs;

// Print one asset's effective entry from the expanded world
pub(crate) use explain::explain;

// Package a built world into a distributable game
pub(crate) use export::export;

// List all declared assets
pub(crate) use list::list;

// Create a new app (in the current directory, or a new one)
pub(crate) use new::{init, new};

// Delete an asset from the current app
pub(crate) use rm::rm;

pub(crate) use build::build;
