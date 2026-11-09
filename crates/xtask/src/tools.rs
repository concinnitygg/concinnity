//! Running the external tools a source build needs, and the small file moves
//! every install shares.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::SystemTime;

use crate::components::Component;
use crate::{Context, Result, fail, workspace_root};

/// Run `command`, failing unless it exits cleanly.
pub(crate) fn run(command: &mut Command) -> Result {
    let what = describe(command);
    let status = command.status().context(format!("could not run {what}"))?;
    if !status.success() {
        return fail(format!("{what} exited {status}"));
    }
    Ok(())
}

/// Run `command` and return what it printed, trimmed.
pub(crate) fn output(command: &mut Command) -> Result<String> {
    let what = describe(command);
    let done = command.output().context(format!("could not run {what}"))?;
    if !done.status.success() {
        return fail(format!(
            "{what}: {}",
            String::from_utf8_lossy(&done.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&done.stdout).trim().to_string())
}

// The program and its first argument, which is enough to say which step failed.
fn describe(command: &Command) -> String {
    let program = command.get_program().to_string_lossy();
    match command.get_args().next() {
        Some(first) => format!("`{program} {}`", first.to_string_lossy()),
        None => format!("`{program}`"),
    }
}

/// Fail unless `tool` runs.
pub(crate) fn require(tool: &str) -> Result {
    let runs = Command::new(tool)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success());
    if runs {
        Ok(())
    } else {
        fail(format!("{tool} is not on PATH"))
    }
}

/// The first `file` on PATH.
#[expect(
    clippy::disallowed_methods,
    reason = "PATH is where a build looks for a tool nothing vendored"
)]
pub(crate) fn on_path(file: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(file))
        .find(|candidate| candidate.is_file())
}

/// Put `built` at `payload`, which only exists once the copy is whole.
pub(crate) fn install(built: &Path, payload: &Path) -> Result {
    if let Some(parent) = payload.parent() {
        fs::create_dir_all(parent).context(parent.display())?;
    }
    let mut incoming = payload.as_os_str().to_owned();
    incoming.push(".incoming");
    fs::copy(built, &incoming).context(format!("copy {}", built.display()))?;
    fs::rename(&incoming, payload).context(payload.display())
}

/// Remove `dir` if it is there, ignoring what was not.
pub(crate) fn remove_dir_quietly(dir: &Path) {
    let _removed = fs::remove_dir_all(dir);
}

/// Where an install is assembled before it replaces `target` whole.
pub(crate) fn staging_dir(target: &Path) -> PathBuf {
    let name = target.file_name().unwrap_or_default().to_string_lossy();
    target.with_file_name(format!(".{name}.incoming"))
}

/// Make the build scripts that resolve `component` re-run.
pub(crate) fn touch_watchers(component: &Component) {
    for name in component.watchers {
        let script = workspace_root().join(name);
        if !script.is_file() {
            continue;
        }
        let touched = fs::File::options()
            .write(true)
            .open(&script)
            .and_then(|file| file.set_modified(SystemTime::now()));
        if let Err(e) = touched {
            println!("  warning: could not touch {name}: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staging_sits_beside_the_target() {
        assert_eq!(
            staging_dir(Path::new("/vendor/dxc-1.9.2607-linux-x86_64")),
            Path::new("/vendor/.dxc-1.9.2607-linux-x86_64.incoming")
        );
    }

    #[test]
    fn install_creates_the_payload_directory() {
        let dir = tempfile::tempdir().unwrap();
        let built = dir.path().join("built.dll");
        fs::write(&built, b"dll").unwrap();
        let payload = dir.path().join("sdk/bin/out.dll");
        install(&built, &payload).unwrap();
        assert_eq!(fs::read(&payload).unwrap(), b"dll");
        assert!(!dir.path().join("sdk/bin/out.dll.incoming").exists());
    }

    #[test]
    fn a_missing_tool_is_reported_by_name() {
        let Err(failure) = require("cn-no-such-tool") else {
            panic!("a missing tool passed");
        };
        assert!(failure.0.contains("cn-no-such-tool"));
    }

    #[test]
    fn a_failing_command_names_its_step() {
        let Err(failure) = run(Command::new("cn-no-such-tool").arg("clone")) else {
            panic!("a missing program ran");
        };
        assert!(failure.0.contains("`cn-no-such-tool clone`"));
    }
}
