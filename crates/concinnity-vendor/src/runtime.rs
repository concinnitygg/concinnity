//! The runtime directory: every installed SDK's libraries staged flat into
//! `<vendor>/runtime-<os>-<arch>/`, laid out as they sit beside a shipped app,
//! with their license notices under `licenses/<sdk>/`. A development run loads
//! from it, and an export copies it beside the player.

use std::fs;
use std::path::{Path, PathBuf};

use crate::component::Component;
use crate::files::{install, remove_dir_quietly, staging_dir};
use crate::host::Host;
use crate::{Context, Result};

/// The runtime directory for `host` under `vendor`.
pub fn runtime_dir(vendor: &Path, host: Host) -> PathBuf {
    vendor.join(format!("runtime-{}", host.slug()))
}

/// The staged runtime directory for a tool started in `cwd`, in the vendor
/// directory [`vendor_root`](crate::root::vendor_root) resolves, if one is.
pub fn staged_for(cwd: &Path, per_user_base: Option<&Path>) -> Option<PathBuf> {
    staged(&crate::root::vendor_root(cwd, per_user_base)?)
}

/// The staged runtime directory for the machine this runs on, if one is.
pub fn staged(vendor: &Path) -> Option<PathBuf> {
    let dir = runtime_dir(vendor, Host::current()?);
    dir.is_dir().then_some(dir)
}

/// What staging left in the runtime directory.
#[derive(Debug, PartialEq, Eq)]
pub struct Staged {
    /// The SDKs it holds.
    pub sdks: Vec<&'static str>,
    /// Whether it already held exactly these, so nothing was rewritten.
    pub unchanged: bool,
}

/// Bring the runtime directory for `host` in line with the `sdks` installed
/// under `vendor`. A directory that already holds exactly their files is left
/// alone; otherwise a new one is assembled aside and swapped in, and the old
/// one is kept until the swap has succeeded, so an app with the old libraries
/// loaded never leaves it half-deleted.
///
/// # Errors
///
/// When a file cannot be copied, or the directory cannot be swapped, as when
/// an app still has its libraries loaded.
pub fn stage(sdks: &[&Component], vendor: &Path, host: Host) -> Result<Staged> {
    let installed: Vec<&Component> = sdks
        .iter()
        .copied()
        .filter(|sdk| sdk.installed(vendor, host))
        .collect();
    let names = installed.iter().map(|sdk| sdk.name).collect();
    let target = runtime_dir(vendor, host);
    if holds_exactly(&target, &installed, vendor, host) {
        return Ok(Staged {
            sdks: names,
            unchanged: true,
        });
    }

    let staging = staging_dir(&target);
    remove_dir_quietly(&staging);
    fs::create_dir_all(&staging).context(staging.display())?;
    for sdk in &installed {
        let release = sdk.install_dir(vendor, host);
        for file in sdk.shipped() {
            install(&release.join(file.from), &staging.join(file.to))?;
        }
    }
    swap_in(&staging, &target)?;
    Ok(Staged {
        sdks: names,
        unchanged: false,
    })
}

// Replace `target` with `staging`. The old directory moves aside first and
// comes back if the new one cannot take its place; it is deleted only once
// the new one has.
fn swap_in(staging: &Path, target: &Path) -> Result {
    let name = target.file_name().unwrap_or_default().to_string_lossy();
    let aside = target.with_file_name(format!(".{name}.old"));
    remove_dir_quietly(&aside);
    let moved_aside = target.exists();
    if moved_aside {
        fs::rename(target, &aside).context(format!(
            "{} (is an app that loads from it still running?)",
            target.display()
        ))?;
    }
    if let Err(e) = fs::rename(staging, target) {
        if moved_aside {
            let _restored = fs::rename(&aside, target);
        }
        remove_dir_quietly(staging);
        return Err(e).context(target.display());
    }
    remove_dir_quietly(&aside);
    Ok(())
}

// Whether `dir` holds exactly the files `sdks` ship, each the same bytes as in
// its release.
fn holds_exactly(dir: &Path, sdks: &[&Component], vendor: &Path, host: Host) -> bool {
    let mut expected: Vec<PathBuf> = Vec::new();
    for sdk in sdks {
        let release = sdk.install_dir(vendor, host);
        for file in sdk.shipped() {
            let staged = dir.join(file.to);
            let same = fs::read(&staged)
                .ok()
                .zip(fs::read(release.join(file.from)).ok())
                .is_some_and(|(have, want)| have == want);
            if !same {
                return false;
            }
            expected.push(staged);
        }
    }
    let mut present = Vec::new();
    if !files_under(dir, &mut present) {
        return false;
    }
    expected.sort();
    present.sort();
    expected == present
}

// Collect every file under `dir` into `out`; `false` when it cannot be read.
fn files_under(dir: &Path, out: &mut Vec<PathBuf>) -> bool {
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if !files_under(&path, out) {
                return false;
            }
        } else {
            out.push(path);
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdks::{AGILITY, XESS};

    fn install_release(vendor: &Path, sdk: &Component, host: Host) {
        let release = sdk.install_dir(vendor, host);
        let files = std::iter::once((sdk.payload)(host)).chain(sdk.shipped().map(|f| f.from));
        for file in files {
            let path = release.join(file);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, file).unwrap();
        }
    }

    #[test]
    fn installed_sdks_are_staged_as_an_app_loads_them() {
        let vendor = tempfile::tempdir().unwrap();
        let host = Host::WindowsX64;
        install_release(vendor.path(), &AGILITY, host);
        let staged = stage(&[&AGILITY, &XESS], vendor.path(), host).unwrap();
        assert_eq!(staged.sdks, ["agility"]);
        assert!(!staged.unchanged);
        let dir = runtime_dir(vendor.path(), host);
        assert!(dir.join("D3D12/D3D12Core.dll").is_file());
        assert!(dir.join("D3D12/d3d12SDKLayers.dll").is_file());
        assert!(dir.join("licenses/agility/LICENSE.txt").is_file());
        assert!(!dir.join("libxess.dll").exists());
    }

    #[test]
    fn an_unchanged_runtime_directory_is_left_alone() {
        let vendor = tempfile::tempdir().unwrap();
        let host = Host::WindowsX64;
        install_release(vendor.path(), &XESS, host);
        assert!(!stage(&[&XESS], vendor.path(), host).unwrap().unchanged);
        assert!(stage(&[&XESS], vendor.path(), host).unwrap().unchanged);

        // A stray file or a changed library makes it restage.
        let dir = runtime_dir(vendor.path(), host);
        fs::write(dir.join("stray.dll"), b"").unwrap();
        assert!(!stage(&[&XESS], vendor.path(), host).unwrap().unchanged);
        assert!(!dir.join("stray.dll").exists());
        fs::write(dir.join("libxess.dll"), b"old").unwrap();
        assert!(!stage(&[&XESS], vendor.path(), host).unwrap().unchanged);
        assert!(stage(&[&XESS], vendor.path(), host).unwrap().unchanged);
    }

    #[test]
    fn restaging_drops_what_is_no_longer_installed() {
        let vendor = tempfile::tempdir().unwrap();
        let host = Host::WindowsX64;
        install_release(vendor.path(), &XESS, host);
        stage(&[&XESS], vendor.path(), host).unwrap();
        let dir = runtime_dir(vendor.path(), host);
        assert!(dir.join("libxess.dll").is_file());

        remove_dir_quietly(&XESS.install_dir(vendor.path(), host));
        assert!(
            stage(&[&XESS], vendor.path(), host)
                .unwrap()
                .sdks
                .is_empty()
        );
        assert!(!dir.join("libxess.dll").exists());
        assert!(!staging_dir(&dir).exists());
    }

    // A swap that cannot place the new directory puts the old one back.
    #[test]
    fn a_failed_swap_keeps_the_old_directory() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("runtime");
        fs::create_dir_all(&target).unwrap();
        fs::write(target.join("kept.dll"), b"").unwrap();
        let missing = dir.path().join("no-such-staging");
        assert!(swap_in(&missing, &target).is_err());
        assert!(target.join("kept.dll").is_file());
    }

    #[test]
    fn the_runtime_directory_is_named_for_its_host() {
        assert_eq!(
            runtime_dir(Path::new("/v"), Host::WindowsX64),
            Path::new("/v/runtime-windows-x86_64")
        );
    }
}
