//! The implementations behind each `cn` subcommand: discovery and printing
//! wrappers over this crate's authoring API and the concinnity-cook compile
//! pipeline.
//!
//! This is where stdout lives. The authoring API returns values and errors; a
//! command resolves which world was meant, calls it, and formats the result for
//! a terminal. Keeping that split means the same authoring call is equally
//! usable from the editor and the debug server.

mod add;
mod check;
mod cook;
mod explain;
mod list;
mod new;
mod rm;
mod version;
mod world_path;

pub(crate) use add::add;
pub(crate) use check::check;
pub(crate) use cook::{Verbosity, cook};
pub(crate) use explain::explain;
pub(crate) use list::{list, provenance};
pub(crate) use new::{init, new};
pub(crate) use rm::rm;
pub(crate) use version::{version, version_details};
pub(crate) use world_path::{discover_world_path, resolve_world_path};
