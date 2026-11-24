//! Downloading a pinned release and unpacking it.
//!
//! A downloaded archive is checked against its pinned sha256 before it is
//! unpacked, so a release re-tagged in place fails the fetch rather than
//! reaching a build or an app.

use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

use crate::component::{Component, Source};
use crate::files::file_name;
use crate::host::Host;
use crate::{Context, Result, fail, unpack};

/// Download `component`'s release for `host` and install it under `vendor`,
/// unless it is installed already and `force` is off. Returns its directory.
///
/// # Errors
///
/// When the host builds it from source, the download fails, the archive does
/// not match its pinned sha256, or it does not hold the payload.
pub fn fetch(component: &Component, host: Host, vendor: &Path, force: bool) -> Result<PathBuf> {
    let name = component.name;
    let Source::Download { url, keep } = (component.source)(host) else {
        return fail(format!(
            "{name} is built from source for {}, not downloaded",
            host.slug()
        ));
    };
    let target = component.install_dir(vendor, host);
    let payload = (component.payload)(host);
    if component.installed(vendor, host) && !force {
        println!(
            "{name}: already vendored, {} ({})",
            file_name(&target),
            component.release
        );
        return Ok(target);
    }

    println!("{name}: downloading {url}");
    let archive = download(&url)?;
    let digest = sha256_hex(&archive);
    if let Some(reason) = digest_error(component.sha256, host, &digest) {
        return fail(format!("{url}: {reason}"));
    }
    let state = if component.sha256.is_some() {
        "pinned"
    } else {
        "unpinned"
    };
    #[expect(clippy::cast_precision_loss, reason = "a size to one decimal")]
    let megabytes = archive.len() as f64 / 1e6;
    println!("  {megabytes:.1} MB, sha256 {digest} ({state})");

    unpack::unpack(&archive, &url, payload, &target, keep)?;
    println!(
        "{name}: vendored {} ({})",
        target.display(),
        component.release
    );
    Ok(target)
}

// curl ships with every host this vendors for (Windows since 10), and keeps a
// TLS stack and its certificate store out of this crate.
fn download(url: &str) -> Result<Vec<u8>> {
    let done = Command::new("curl")
        .args([
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--retry",
            "3",
            url,
        ])
        .output()
        .context("could not run curl")?;
    if !done.status.success() {
        return fail(format!(
            "{url}: {}",
            String::from_utf8_lossy(&done.stderr).trim()
        ));
    }
    Ok(done.stdout)
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .fold(String::new(), |mut hex, b| {
            let _infallible = write!(hex, "{b:02x}");
            hex
        })
}

/// Why `digest` is not the archive `host` was pinned to, or `None` if it is.
fn digest_error(pins: Option<&[(Host, &str)]>, host: Host, digest: &str) -> Option<String> {
    let pins = pins?;
    match pins.iter().find(|(pinned, _)| *pinned == host) {
        None => Some(format!(
            "no pinned sha256 for {}; add one before fetching it",
            host.slug()
        )),
        Some((_, expected)) if *expected != digest => Some(format!(
            "sha256 {digest} does not match the pinned {expected}"
        )),
        Some(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_digest_gate_refuses_what_the_pin_does_not_name() {
        let a = "a".repeat(64);
        let b = "b".repeat(64);
        let pins: &[(Host, &str)] = &[(Host::LinuxX64, &a)];
        assert_eq!(digest_error(None, Host::LinuxX64, &b), None);
        assert_eq!(digest_error(Some(pins), Host::LinuxX64, &a), None);
        assert!(
            digest_error(Some(pins), Host::LinuxX64, &b)
                .is_some_and(|reason| reason.contains("does not match the pinned"))
        );
        assert!(
            digest_error(Some(pins), Host::MacosArm64, &a)
                .is_some_and(|reason| reason.contains("no pinned sha256"))
        );
    }

    #[test]
    fn digests_are_lowercase_hex() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
