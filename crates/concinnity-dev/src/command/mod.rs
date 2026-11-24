//! The implementations behind each `cn` subcommand: discovery and printing
//! wrappers over this crate's authoring API and the concinnity-cook compile
//! pipeline.
//!
//! This is where stdout lives. The authoring API returns values and errors; a
//! command resolves which file was meant, calls it, and formats the result for
//! a terminal. Keeping that split means the same authoring call is equally
//! usable from the editor and the debug server.

mod add;
mod app;
mod check;
mod cook;
mod data_files;
mod explain;
pub(crate) mod export;
mod list;
mod new;
mod rm;
mod target;
mod version;

pub(crate) use add::add;
pub(crate) use app::no_application;
pub(crate) use check::check;
pub(crate) use cook::{Verbosity, cook, cook_path};
pub(crate) use explain::explain;
pub(crate) use list::{list, provenance};
pub(crate) use new::{init, new};
pub(crate) use rm::rm;
pub(crate) use target::Target;
pub(crate) use version::{version, version_details};
