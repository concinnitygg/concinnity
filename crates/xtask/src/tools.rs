//! Running the external tools a source build needs, and telling the build
//! scripts about a new install.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::SystemTime;

use concinnity_vendor::{Component, Context, Result, fail};

use crate::components::watchers;
use crate::workspace_root;

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

/// Make the build scripts that resolve `component` re-run.
pub(crate) fn touch_watchers(component: &Component) {
    for name in watchers(component) {
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
