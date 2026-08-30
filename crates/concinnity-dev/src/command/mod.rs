//! The implementations behind each `cn` subcommand: discovery and printing
//! wrappers over this crate's authoring API and the concinnity-cook compile
//! pipeline.
//!
//! This is where stdout lives. The authoring API returns values and errors; a
//! command resolves which world was meant, calls it, and formats the result for
//! a terminal. Keeping that split means the same authoring call is equally
//! usable from the editor, the debug server, and an out-of-tree host.

mod add;
mod build;
mod check;
mod explain;
mod list;
mod new;
mod rm;
mod version;

pub use add::add;
pub use build::build;
pub use check::check;
pub use explain::explain;
pub use list::list;
pub use new::{init, new};
pub use rm::rm;
pub use version::{VERSION, version, version_details, version_line};

pub(crate) use list::{provenance, resolve_world_path};
