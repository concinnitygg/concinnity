//! `concinnity vendor`, which fetches or builds every component and stages the
//! runtime SDKs' libraries, and where `run`, `bench`, `edit` and `export` find
//! what it left.

use std::io::{self, BufRead, IsTerminal, Write};
use std::path::PathBuf;

use super::sdks::SDKS;
use super::{COMPONENTS, Component, Host, fetch, find, process, root, runtime, source_build};

/// What `concinnity vendor` was asked to do.
pub(crate) struct VendorRequest<'a> {
    /// The components named, every one when empty.
    pub(crate) components: &'a [String],
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

/// Have the shader compiler look for the tools vendored here first. Called
/// before anything compiles a shader.
pub(crate) fn use_vendored_tools() {
    if let Some(vendor) = vendor_root() {
        concinnity_shader::use_vendor_dir(&vendor);
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
    let named = !request.components.is_empty();
    let chosen: Vec<&'static Component> = if named {
        request
            .components
            .iter()
            .filter_map(|name| find(name))
            .collect()
    } else {
        COMPONENTS.to_vec()
    };
    if request.status {
        status(&chosen, &vendor, host);
        return Ok(());
    }

    println!("vendor: {}", vendor.display());
    let mut unanswered = Vec::new();
    for component in chosen {
        let name = component.name;
        let Some(host) = component.host_for(host, named).map_err(io::Error::other)? else {
            println!("{name}: no release for this host, skipped");
            continue;
        };
        if component.installed(&vendor, host) && !request.force {
            println!("{name}: already vendored ({})", component.release);
            continue;
        }
        if component.needs_acceptance() {
            match acceptance(component, request.accept_licenses)? {
                Acceptance::Accepted => {}
                Acceptance::Declined => {
                    println!("{name}: terms declined, skipped");
                    continue;
                }
                Acceptance::Unanswerable => {
                    println!("{name}: terms not accepted, skipped");
                    unanswered.push(name);
                    continue;
                }
            }
        }
        let installed = if component.is_built(host) {
            source_build::build(component, host, &vendor)
        } else {
            fetch::fetch(component, host, &vendor, request.force)
        }
        .map_err(io::Error::other)?;
        if component.tool {
            println!(
                "  reports   {}",
                process::report(&installed.join((component.payload)(host)))
            );
        }
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

// Whether `component`'s terms are accepted: on the command line, or by answering a
// terminal that can be asked.
fn acceptance(component: &Component, on_the_command_line: bool) -> io::Result<Acceptance> {
    println!("{} comes under these terms:", component.name);
    for license in component.licenses {
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
    for component in chosen {
        let host = host.filter(|host| component.hosts.contains(host));
        let state = match host {
            None => "no release for this host",
            Some(host) if component.installed(vendor, host) => "vendored",
            Some(host) if component.is_built(host) => "not vendored (built from source here)",
            Some(_) => "not vendored",
        };
        println!(
            "{} {}: {state} -- {}",
            component.name, component.release, component.summary
        );
        if let Some(digest) = host.and_then(|host| component.pinned_digest(host)) {
            println!("  sha256 {digest}");
        }
        if let Some(host) = host.filter(|host| component.tool && component.installed(vendor, *host))
        {
            let exe = component
                .install_dir(vendor, host)
                .join((component.payload)(host));
            println!("  reports {}", process::report(&exe));
        }
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
