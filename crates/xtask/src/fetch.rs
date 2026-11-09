//! `vendor fetch`: download a pinned release and unpack it.
//!
//! A downloaded archive is checked against its pinned sha256 before it is
//! unpacked, so a release re-tagged in place fails the fetch rather than
//! reaching a build.

use std::fmt::Write;
use std::path::Path;
use std::process::Command;

use sha2::{Digest, Sha256};

use crate::components::{Component, Source};
use crate::host::Host;
use crate::{Context, Result, fail, status, tools, unpack, workspace_root};

pub(crate) fn fetch(
    component: &Component,
    current: Option<Host>,
    named: bool,
    force: bool,
) -> Result {
    let name = component.name;
    let Some(host) = component.host_for(current, named)? else {
        println!("{name}: no release for this host, skipped");
        return Ok(());
    };
    let Source::Download { url, keep } = (component.source)(host) else {
        if named {
            return fail(format!(
                "{name} is built here, not downloaded -- run `cargo xtask vendor build {name}`"
            ));
        }
        println!("{name}: built here, skipped -- `cargo xtask vendor build {name}` builds it");
        return Ok(());
    };
    let target = component.install_dir(host);
    let payload = (component.payload)(host);
    if target.join(payload).is_file() && !force {
        println!(
            "{name}: already vendored, {} ({})",
            file_name(&target),
            component.release
        );
        return Ok(());
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
    tools::touch_watchers(component);
    let shown = target.strip_prefix(workspace_root()).unwrap_or(&target);
    println!(
        "{name}: vendored {} ({})",
        shown.display(),
        component.release
    );
    status::print_report(component, &target, host);
    Ok(())
}

pub(crate) fn file_name(path: &Path) -> String {
    path.file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
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
