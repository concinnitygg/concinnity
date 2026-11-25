//! Running the external programs vendoring needs.

use std::path::Path;
use std::process::{Command, Stdio};

use super::{Context, Result, fail};

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

/// What a tool says it is, which proves it loads its libraries.
pub(crate) fn report(exe: &Path) -> String {
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

    #[test]
    fn a_missing_tool_reports_that_it_does_not_run() {
        assert!(report(Path::new("/nonexistent/dxc")).starts_with("does not run:"));
    }
}
