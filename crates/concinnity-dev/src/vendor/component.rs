//! What a vendored component is: its pinned release, where each host gets it
//! from, the file whose presence means it is installed, and, for a runtime SDK,
//! the libraries an app loads from it and the terms they come under.

use std::path::{Path, PathBuf};

use super::host::Host;
use super::{Result, fail};

/// One pinned third-party release.
pub(crate) struct Component {
    /// What a command line names it by.
    pub(crate) name: &'static str,
    /// The pinned release, which names its install directory.
    pub(crate) release: &'static str,
    /// One line on what it is for.
    pub(crate) summary: &'static str,
    /// The hosts it has a release for.
    pub(crate) hosts: &'static [Host],
    /// How a host gets it.
    pub(crate) source: fn(Host) -> Source,
    /// The file whose presence means it is installed, relative to its directory.
    pub(crate) payload: fn(Host) -> &'static str,
    /// The sha256 of each host's download. `None` leaves every download
    /// unpinned; a component that pins any host must pin every one it downloads.
    pub(crate) sha256: Option<&'static [(Host, &'static str)]>,
    /// Whether the payload is a tool that reports its own version.
    pub(crate) tool: bool,
    /// The libraries an app loads from it, staged into the runtime directory.
    pub(crate) runtime: &'static [RuntimeFile],
    /// The license and notice files its libraries ship with, staged under
    /// `licenses/<name>/` in the runtime directory.
    pub(crate) notices: &'static [RuntimeFile],
    /// The terms its binaries come under.
    pub(crate) licenses: &'static [License],
}

/// Where a host gets a component from.
pub(crate) enum Source {
    /// An upstream archive, unpacked whole or cut to the members `keep` names.
    Download {
        /// The archive's address.
        url: String,
        /// The members to unpack, as shell globs; `None` unpacks all of them.
        keep: Option<&'static [&'static str]>,
    },
    /// Built from source by the tool that vendors it, from what no upstream
    /// publishes for the host.
    Built,
}

/// A file an app ships with: where it sits in the component's release, and
/// where it goes in the runtime directory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RuntimeFile {
    /// Its path in the installed release.
    pub(crate) from: &'static str,
    /// Its path in the runtime directory, and beside a shipped app.
    pub(crate) to: &'static str,
}

/// The terms a component's binaries come under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct License {
    /// The terms' name.
    pub(crate) name: &'static str,
    /// Where the terms can be read.
    pub(crate) url: &'static str,
    /// Whether the terms are an agreement to accept before the binaries are
    /// downloaded, rather than a notice to keep with them.
    pub(crate) agreement: bool,
}

impl Component {
    /// The host to act for, `None` when the component has no release for this
    /// machine. A bare request applies to every component and most hosts can
    /// use only some, so that is a skip; naming the component makes it a
    /// failure.
    ///
    /// # Errors
    ///
    /// When `named` and it has no release for `current`.
    pub(crate) fn host_for(&self, current: Option<Host>, named: bool) -> Result<Option<Host>> {
        let host = current.filter(|host| self.hosts.contains(host));
        if host.is_none() && named {
            return fail(format!(
                "{} publishes no release for {}/{}",
                self.name,
                std::env::consts::OS,
                std::env::consts::ARCH
            ));
        }
        Ok(host)
    }

    /// The directory the pinned release for `host` installs into.
    pub(crate) fn install_dir(&self, vendor: &Path, host: Host) -> PathBuf {
        vendor.join(format!("{}-{}-{}", self.name, self.release, host.slug()))
    }

    /// Whether the pinned release for `host` is installed under `vendor`, with
    /// every file it ships.
    pub(crate) fn installed(&self, vendor: &Path, host: Host) -> bool {
        let dir = self.install_dir(vendor, host);
        std::iter::once((self.payload)(host))
            .chain(self.shipped().map(|file| file.from))
            .all(|file| dir.join(file).is_file())
    }

    /// Every file an app ships from it: its libraries, then its notices.
    pub(crate) fn shipped(&self) -> impl Iterator<Item = &'static RuntimeFile> {
        self.runtime.iter().chain(self.notices)
    }

    /// Whether `host` builds it from source rather than downloading it.
    pub(crate) fn is_built(&self, host: Host) -> bool {
        matches!((self.source)(host), Source::Built)
    }

    /// The sha256 `host`'s download is pinned to, if it is.
    pub(crate) fn pinned_digest(&self, host: Host) -> Option<&'static str> {
        self.sha256?
            .iter()
            .find(|(pinned, _)| *pinned == host)
            .map(|(_, digest)| *digest)
    }

    /// Whether any of its terms are an agreement to accept first.
    pub(crate) fn needs_acceptance(&self) -> bool {
        self.licenses.iter().any(|license| license.agreement)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vendor::sdks::XESS;

    #[test]
    fn a_host_without_a_release_is_skipped_unless_named() {
        assert_eq!(
            XESS.host_for(Some(Host::MacosArm64), false).ok(),
            Some(None)
        );
        assert!(XESS.host_for(Some(Host::MacosArm64), true).is_err());
        assert_eq!(
            XESS.host_for(Some(Host::WindowsX64), true).ok(),
            Some(Some(Host::WindowsX64))
        );
        assert!(XESS.host_for(None, true).is_err());
    }

    #[test]
    fn installed_means_every_shipped_file_is_there() {
        let vendor = tempfile::tempdir().unwrap();
        let host = Host::WindowsX64;
        let installed = vendor.path().join("xess-3.0.1-windows-x86_64");
        std::fs::create_dir_all(installed.join("bin")).unwrap();
        std::fs::write(installed.join("bin/libxess.dll"), b"").unwrap();
        // Installed means every file it ships is there too, its notices included.
        assert!(!XESS.installed(vendor.path(), host));
        for notice in XESS.notices {
            std::fs::write(installed.join(notice.from), b"").unwrap();
        }
        assert!(XESS.installed(vendor.path(), host));
        assert!(!XESS.installed(&vendor.path().join("absent"), host));
    }
}
