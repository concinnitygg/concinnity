//! Vendoring the pinned third-party binaries Concinnity builds with and ships.
//!
//! A [`Component`] is pinned to one release. Fetching downloads it, checks the
//! archive against its pinned sha256 where it has one, and unpacks it into
//! `<vendor>/<name>-<version>-<os>-<arch>/`. Nothing here runs during a Cargo
//! build: a build that reached the network could not build offline, and would
//! pull a dependency past Cargo.lock and cargo-deny.
//!
//! Two tools drive it. `cargo xtask vendor` fetches the shader compiler the
//! build itself needs, and so cannot depend on anything that needs it.
//! `concinnity vendor` fetches the runtime SDKs in [`sdks`], which an app
//! loads when it runs rather than links, and stages their libraries into one
//! [`runtime`] directory that development runs load from and exports bundle.

pub mod component;
pub mod fetch;
pub mod files;
pub mod host;
pub mod root;
pub mod runtime;
pub mod sdks;
mod unpack;

use std::fmt::{self, Display};

pub use component::{Component, License, RuntimeFile, Source};
pub use host::Host;

/// Why a vendoring step stopped, as the message it prints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failure(pub String);

impl Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Failure {}

/// A vendoring step's outcome.
pub type Result<T = ()> = std::result::Result<T, Failure>;

/// A [`Failure`] carrying `message`.
///
/// # Errors
///
/// Always.
pub fn fail<T>(message: impl Into<String>) -> Result<T> {
    Err(Failure(message.into()))
}

/// Names what was being done when an error surfaced.
pub trait Context<T> {
    /// The error, prefixed with `what`.
    ///
    /// # Errors
    ///
    /// When `self` is an error.
    fn context(self, what: impl Display) -> Result<T>;
}

impl<T, E: Display> Context<T> for std::result::Result<T, E> {
    fn context(self, what: impl Display) -> Result<T> {
        self.map_err(|e| Failure(format!("{what}: {e}")))
    }
}
