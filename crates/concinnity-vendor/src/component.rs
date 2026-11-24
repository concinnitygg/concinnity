//! What a vendored component is: its pinned release, where each host gets it
//! from, the file whose presence means it is installed, and, for a runtime SDK,
//! the libraries an app loads from it and the terms they come under.

use std::path::{Path, PathBuf};

use crate::host::Host;
use crate::{Result, fail};

/// One pinned third-party release.
pub struct Component {
    /// What a command line names it by.
    pub name: &'static str,
    /// The pinned release, which names its install directory.
    pub release: &'static str,
    /// One line on what it is for.
    pub summary: &'static str,
    /// The hosts it has a release for.
    pub hosts: &'static [Host],
    /// How a host gets it.
    pub source: fn(Host) -> Source,
    /// The file whose presence means it is installed, relative to its directory.
    pub payload: fn(Host) -> &'static str,
    /// The sha256 of each host's download. `None` leaves every download
    /// unpinned; a component that pins any host must pin every one it downloads.
    pub sha256: Option<&'static [(Host, &'static str)]>,
    /// Whether the payload is a tool that reports its own version.
    pub tool: bool,
    /// The libraries an app loads from it, staged into the runtime directory.
    pub runtime: &'static [RuntimeFile],
    /// The license and notice files its libraries ship with, staged under
    /// `licenses/<name>/` in the runtime directory.
    pub notices: &'static [RuntimeFile],
    /// The terms its binaries come under.
    pub licenses: &'static [License],
}

/// Where a host gets a component from.
pub enum Source {
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
pub struct RuntimeFile {
    /// Its path in the installed release.
    pub from: &'static str,
    /// Its path in the runtime directory, and beside a shipped app.
    pub to: &'static str,
}

/// The terms a component's binaries come under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct License {
    /// The terms' name.
    pub name: &'static str,
    /// Where the terms can be read.
    pub url: &'static str,
    /// Whether the terms are an agreement to accept before the binaries are
    /// downloaded, rather than a notice to keep with them.
    pub agreement: bool,
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
    pub fn host_for(&self, current: Option<Host>, named: bool) -> Result<Option<Host>> {
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
    pub fn install_dir(&self, vendor: &Path, host: Host) -> PathBuf {
        vendor.join(format!("{}-{}-{}", self.name, self.release, host.slug()))
    }

    /// Whether the pinned release for `host` is installed under `vendor`, with
    /// every file it ships.
    pub fn installed(&self, vendor: &Path, host: Host) -> bool {
        let dir = self.install_dir(vendor, host);
        std::iter::once((self.payload)(host))
            .chain(self.shipped().map(|file| file.from))
            .all(|file| dir.join(file).is_file())
    }

    /// Every file an app ships from it: its libraries, then its notices.
    pub fn shipped(&self) -> impl Iterator<Item = &'static RuntimeFile> {
        self.runtime.iter().chain(self.notices)
    }

    /// Whether `host` builds it from source rather than downloading it.
    pub fn is_built(&self, host: Host) -> bool {
        matches!((self.source)(host), Source::Built)
    }

    /// The sha256 `host`'s download is pinned to, if it is.
    pub fn pinned_digest(&self, host: Host) -> Option<&'static str> {
        self.sha256?
            .iter()
            .find(|(pinned, _)| *pinned == host)
            .map(|(_, digest)| *digest)
    }

    /// Whether any of its terms are an agreement to accept first.
    pub fn needs_acceptance(&self) -> bool {
        self.licenses.iter().any(|license| license.agreement)
    }

    /// Every vendored release of this component for `host` under `vendor`,
    /// installed or not, sorted by name.
    pub fn vendored_in(&self, vendor: &Path, host: Host) -> Vec<PathBuf> {
        let Ok(entries) = std::fs::read_dir(vendor) else {
            return Vec::new();
        };
        let mut found: Vec<PathBuf> = entries
            .flatten()
            .filter(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .is_some_and(|name| self.is_release_dir(name, host))
            })
            .map(|entry| entry.path())
            .filter(|path| path.join((self.payload)(host)).is_file())
            .collect();
        found.sort();
        found
    }

    // Whether `dir` is `<name>-<version>-<slug>` with a dotted numeric version,
    // which keeps a name from claiming the releases of a longer one it prefixes.
    fn is_release_dir(&self, dir: &str, host: Host) -> bool {
        dir.strip_prefix(self.name)
            .and_then(|rest| rest.strip_prefix('-'))
            .and_then(|rest| rest.strip_suffix(host.slug()))
            .and_then(|rest| rest.strip_suffix('-'))
            .is_some_and(|version| {
                !version.is_empty()
                    && version
                        .split('.')
                        .all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdks::{FIDELITYFX, XESS};

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
    fn release_dirs_match_by_name_version_and_slug() {
        let host = Host::WindowsX64;
        assert!(FIDELITYFX.is_release_dir("fidelityfx-2.4.0-windows-x86_64", host));
        assert!(!FIDELITYFX.is_release_dir("fidelityfx-vk-2.4.0-windows-x86_64", host));
        assert!(!FIDELITYFX.is_release_dir("fidelityfx-2.4.0-linux-x86_64", host));
        assert!(!FIDELITYFX.is_release_dir("fidelityfx--windows-x86_64", host));
        assert!(!FIDELITYFX.is_release_dir(".fidelityfx-2.4.0-windows-x86_64.incoming", host));
    }

    #[test]
    fn only_installed_releases_count_as_vendored() {
        let vendor = tempfile::tempdir().unwrap();
        let host = Host::WindowsX64;
        let installed = vendor.path().join("xess-3.0.1-windows-x86_64");
        std::fs::create_dir_all(installed.join("bin")).unwrap();
        std::fs::write(installed.join("bin/libxess.dll"), b"").unwrap();
        std::fs::create_dir_all(vendor.path().join("xess-2.0.0-windows-x86_64")).unwrap();
        assert_eq!(
            XESS.vendored_in(vendor.path(), host),
            vec![installed.clone()]
        );
        // Installed means every file it ships is there too, its notices included.
        assert!(!XESS.installed(vendor.path(), host));
        for notice in XESS.notices {
            std::fs::write(installed.join(notice.from), b"").unwrap();
        }
        assert!(XESS.installed(vendor.path(), host));
        assert!(
            XESS.vendored_in(&vendor.path().join("absent"), host)
                .is_empty()
        );
    }
}
