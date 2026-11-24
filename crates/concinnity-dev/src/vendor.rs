//! The runtime SDKs a development run loads: `concinnity vendor`, which fetches
//! and stages them, and where `run`, `bench`, `edit` and `export` find what it
//! staged.

use std::io::{self, BufRead, IsTerminal, Write};
use std::path::PathBuf;

use concinnity_vendor::sdks::{SDKS, find};
use concinnity_vendor::{Component, Host, fetch, root, runtime};

/// What `concinnity vendor` was asked to do.
pub(crate) struct VendorRequest<'a> {
    /// The SDKs named, every one when empty.
    pub(crate) sdks: &'a [String],
    pub(crate) force: bool,
    pub(crate) accept_licenses: bool,
    pub(crate) status: bool,
}

/// The vendor directory this command works in: the enclosing workspace's, else
/// the per-user one.
pub(crate) fn vendor_root() -> Option<PathBuf> {
    let cwd = std::env::current_dir().ok()?;
    let base = concinnity_host::store::paths::per_user_base();
    root::vendor_root(&cwd, base.as_deref())
}

/// The staged runtime directory, if `concinnity vendor` staged one.
pub(crate) fn staged_runtime() -> Option<PathBuf> {
    let cwd = std::env::current_dir().ok()?;
    let base = concinnity_host::store::paths::per_user_base();
    runtime::staged_for(&cwd, base.as_deref())
}

/// Have the renderer load the staged runtime libraries. Called before anything
/// probes a device.
pub(crate) fn use_staged_runtime() {
    if let Some(dir) = staged_runtime() {
        concinnity_runtime::use_runtime_libraries(&dir);
    }
}

/// `concinnity vendor`.
pub(crate) fn vendor(request: &VendorRequest<'_>) -> io::Result<()> {
    let vendor = vendor_root().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "no vendor directory: not in a Cargo workspace, and no per-user directory",
        )
    })?;
    let host = Host::current();
    let named = !request.sdks.is_empty();
    let chosen: Vec<&'static Component> = if named {
        request.sdks.iter().filter_map(|name| find(name)).collect()
    } else {
        SDKS.to_vec()
    };
    if request.status {
        status(&chosen, &vendor, host);
        return Ok(());
    }

    println!("vendor: {}", vendor.display());
    let mut unanswered = Vec::new();
    for sdk in chosen {
        let Some(host) = sdk.host_for(host, named).map_err(io::Error::other)? else {
            println!("{}: no release for this host, skipped", sdk.name);
            continue;
        };
        if sdk.installed(&vendor, host) && !request.force {
            println!("{}: already vendored ({})", sdk.name, sdk.release);
            continue;
        }
        if sdk.needs_acceptance() {
            match acceptance(sdk, request.accept_licenses)? {
                Acceptance::Accepted => {}
                Acceptance::Declined => {
                    println!("{}: terms declined, skipped", sdk.name);
                    continue;
                }
                Acceptance::Unanswerable => {
                    println!("{}: terms not accepted, skipped", sdk.name);
                    unanswered.push(sdk.name);
                    continue;
                }
            }
        }
        fetch::fetch(sdk, host, &vendor, request.force).map_err(io::Error::other)?;
    }

    if let Some(host) = host.filter(|host| SDKS.iter().any(|sdk| sdk.hosts.contains(host))) {
        let staged = runtime::stage(&SDKS, &vendor, host).map_err(io::Error::other)?;
        let held = if staged.sdks.is_empty() {
            "nothing".to_string()
        } else {
            staged.sdks.join(", ")
        };
        let state = if staged.unchanged {
            "already staged"
        } else {
            "staged"
        };
        println!(
            "runtime: {held} {state} in {}",
            runtime::runtime_dir(&vendor, host).display()
        );
    }
    // A run with no terminal to ask cannot accept terms, and a skip it only
    // printed would pass for a successful fetch.
    if !unanswered.is_empty() {
        return Err(io::Error::other(format!(
            "{} not fetched: their terms are an agreement and there was no terminal to accept \
             them on; pass --accept-licenses to accept them",
            unanswered.join(", ")
        )));
    }
    Ok(())
}

// How a request for an SDK's terms was answered.
enum Acceptance {
    Accepted,
    Declined,
    // Nothing could be asked: no terminal, and no `--accept-licenses`.
    Unanswerable,
}

// Whether `sdk`'s terms are accepted: on the command line, or by answering a
// terminal that can be asked.
fn acceptance(sdk: &Component, on_the_command_line: bool) -> io::Result<Acceptance> {
    println!("{} comes under these terms:", sdk.name);
    for license in sdk.licenses {
        println!("  {}: {}", license.name, license.url);
    }
    if on_the_command_line {
        println!("  accepted with --accept-licenses");
        return Ok(Acceptance::Accepted);
    }
    if !io::stdin().is_terminal() {
        return Ok(Acceptance::Unanswerable);
    }
    print!("Accept them? [y/N] ");
    io::stdout().flush()?;
    let mut answer = String::new();
    io::stdin().lock().read_line(&mut answer)?;
    Ok(if is_yes(&answer) {
        Acceptance::Accepted
    } else {
        Acceptance::Declined
    })
}

fn is_yes(answer: &str) -> bool {
    matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes")
}

fn status(chosen: &[&Component], vendor: &std::path::Path, host: Option<Host>) {
    println!("vendor: {}", vendor.display());
    for sdk in chosen {
        let state = match host.filter(|host| sdk.hosts.contains(host)) {
            None => "no release for this host".to_string(),
            Some(host) if sdk.installed(vendor, host) => "vendored".to_string(),
            Some(_) => "not vendored".to_string(),
        };
        println!("{} {}: {state} -- {}", sdk.name, sdk.release, sdk.summary);
    }
    match runtime::staged(vendor) {
        Some(dir) => println!("runtime: {}", dir.display()),
        None => println!("runtime: nothing staged in {}", vendor.display()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_yes_accepts() {
        for yes in ["y\n", "Y", " yes \r\n"] {
            assert!(is_yes(yes), "{yes:?}");
        }
        for no in ["", "\n", "n", "no", "yess", "sure"] {
            assert!(!is_yes(no), "{no:?}");
        }
    }
}
