//! `concinnity vendor`: the pinned third-party binaries a development run
//! compiles shaders with and loads, and an export ships.
//!
//! A [`Component`] is pinned to one release. Fetching downloads it, checks the
//! archive against its pinned sha256, and unpacks it into
//! `<vendor>/<name>-<version>-<os>-<arch>/`; what no upstream publishes for a
//! host is built from the pinned source instead. Nothing here runs during a
//! Cargo build: a build that reached the network could not build offline, and
//! would pull a dependency past Cargo.lock and cargo-deny.
//!
//! The shader compiler in [`tools`] compiles the engine's shaders on first use
//! and for an export. The runtime SDKs in [`sdks`] an app loads when it runs
//! rather than links, and their libraries are staged into one [`runtime`]
//! directory that development runs load from and exports bundle.

mod command;
mod component;
mod fetch;
mod files;
mod host;
mod process;
mod root;
mod runtime;
mod sdks;
mod source_build;
mod tools;
mod unpack;

use std::fmt::{self, Display};

pub(crate) use command::{
    VendorRequest, staged_runtime, use_staged_runtime, use_vendored_tools, vendor, vendor_root,
};
pub(crate) use component::Component;
pub(crate) use host::Host;

/// Every component's name, for the command line to offer.
pub(crate) const NAMES: [&str; 5] = [
    tools::DXC.name,
    sdks::AGILITY.name,
    sdks::FIDELITYFX.name,
    sdks::XESS.name,
    sdks::STREAMLINE.name,
];

/// Every component, in the order a bare request visits them.
pub(crate) const COMPONENTS: [&Component; 5] = [
    &tools::DXC,
    &sdks::AGILITY,
    &sdks::FIDELITYFX,
    &sdks::XESS,
    &sdks::STREAMLINE,
];

/// The component called `name`.
pub(crate) fn find(name: &str) -> Option<&'static Component> {
    COMPONENTS
        .into_iter()
        .find(|component| component.name == name)
}

/// Why a vendoring step stopped, as the message it prints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Failure(pub(crate) String);

impl Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Failure {}

/// A vendoring step's outcome.
pub(crate) type Result<T = ()> = std::result::Result<T, Failure>;

/// A [`Failure`] carrying `message`.
pub(crate) fn fail<T>(message: impl Into<String>) -> Result<T> {
    Err(Failure(message.into()))
}

/// Names what was being done when an error surfaced.
pub(crate) trait Context<T> {
    /// The error, prefixed with `what`.
    fn context(self, what: impl Display) -> Result<T>;
}

impl<T, E: Display> Context<T> for std::result::Result<T, E> {
    fn context(self, what: impl Display) -> Result<T> {
        self.map_err(|e| Failure(format!("{what}: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_follow_the_table() {
        let names: Vec<&str> = COMPONENTS.iter().map(|c| c.name).collect();
        assert_eq!(names, NAMES);
        assert!(find("dxc").is_some());
        assert!(find("xess").is_some());
        assert!(find("slang").is_none());
    }
}
