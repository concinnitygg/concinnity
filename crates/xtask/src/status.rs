//! `vendor status`: what is vendored, and what the build would pick.

use std::path::Path;
use std::process::Command;

use crate::components::Component;
use crate::fetch::file_name;
use crate::host::Host;
use crate::{Result, tools, vendor_dir};

pub(crate) fn status(component: &Component, current: Option<Host>, named: bool) -> Result {
    let host = component.host_for(current, named)?;
    println!("{}: {}", component.name, component.summary);
    let Some(host) = host else {
        println!(
            "  host      no release for {}/{}",
            std::env::consts::OS,
            std::env::consts::ARCH
        );
        return Ok(());
    };
    println!("  host      {}", host.slug());
    println!("  pinned    {}", component.release);
    let built = component.is_built(host);
    if built {
        println!("  digest    built here");
    } else {
        println!(
            "  digest    {}",
            component.pinned_digest(host).unwrap_or("unpinned")
        );
    }

    let vendored = component.vendored_in(&vendor_dir(), host);
    if vendored.is_empty() {
        let verb = if built { "build" } else { "fetch" };
        println!(
            "  vendored  none -- run `cargo xtask vendor {verb} {}`",
            component.name
        );
    }
    let pinned = component.install_dir(host);
    for path in &vendored {
        let mark = if *path == pinned { "*" } else { " " };
        println!("  vendored {mark}{}", file_name(path));
        print_report(component, path, host);
    }

    if component.tool {
        let exe = Path::new((component.payload)(host));
        if let Some(found) = exe
            .file_name()
            .and_then(|name| tools::on_path(&name.to_string_lossy()))
        {
            println!("  on PATH   {}", found.display());
            println!("  reports   {}", report(&found));
        }
    }
    Ok(())
}

/// Print what the tool installed under `root` says it is, for a component
/// that is a tool.
pub(crate) fn print_report(component: &Component, root: &Path, host: Host) {
    if component.tool {
        println!(
            "  reports   {}",
            report(&root.join((component.payload)(host)))
        );
    }
}

// What a compiler says it is, which proves it loads its library.
fn report(exe: &Path) -> String {
    match Command::new(exe).arg("--version").output() {
        Err(e) => format!("does not run: {e}"),
        Ok(done) if !done.status.success() => {
            let stderr = String::from_utf8_lossy(&done.stderr);
            let stdout = String::from_utf8_lossy(&done.stdout);
            let said = if stderr.trim().is_empty() {
                stdout
            } else {
                stderr
            };
            format!("does not run: {}", said.trim())
        }
        Ok(done) => String::from_utf8_lossy(&done.stdout)
            .lines()
            .next()
            .unwrap_or_default()
            .trim()
            .to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_tool_reports_that_it_does_not_run() {
        assert!(report(Path::new("/nonexistent/dxc")).starts_with("does not run:"));
    }
}
