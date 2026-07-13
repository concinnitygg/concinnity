// Command-line subcommand implementations for the `concinnity` binary: thin
// discovery/printing wrappers over the concinnity-editor authoring API and the
// concinnity-cook compile pipeline. CLI-only std code (argv handling, stdout
// tables, packaging) belongs here; the libraries stay free of it.

mod add;
mod build;
mod check;
mod explain;
mod export;
mod list;
mod new;
mod rm;

// Create and apply an asset to the current app
pub use add::add;

// Analyze the current app and report errors, but don't build blob files
pub use check::check;

// Print one asset's effective entry from the expanded world
pub use explain::explain;

// Package a built world into a distributable game
pub use export::export;

// List all declared assets
pub use list::list;

// Create a new app (in the current directory, or a new one)
pub use new::{init, new};

// Delete an asset from the current app
pub use rm::rm;

pub use build::build;
