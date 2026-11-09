// Reading a launch from this process's command line, and the macOS validation
// re-exec that has to run before any thread starts.

use concinnity_engine::launch::{LaunchArgs, LaunchArgsError, LaunchCommand};

/// The launch this process's arguments ask for, read by [`LaunchArgs::parse`].
///
/// On macOS a launch that renders with graphics validation replaces this
/// process image here (see [`reexec_with_metal_validation`]), so call this
/// before spawning a thread or touching Metal.
///
/// # Errors
///
/// Returns the error [`LaunchArgs::parse`] returns, `--help` included.
pub fn launch_args_from_env() -> Result<LaunchArgs, LaunchArgsError> {
    let args = LaunchArgs::parse(
        std::env::args_os()
            .skip(1)
            .map(|arg| arg.to_string_lossy().into_owned()),
    )?;
    if args.command != LaunchCommand::Cook {
        reexec_with_metal_validation(args.graphics_validation);
    }
    Ok(args)
}

/// When a rendering launch asks for graphics validation on macOS (`requested`,
/// `None` following the build profile), relaunch the process with Metal's
/// API-validation layer (`MTL_DEBUG_LAYER`) set, then return into the
/// replacement image. Metal reads that variable during early framework
/// initialization, so it cannot be toggled from a process that has already
/// touched Metal, and setting it from inside a process is unsound once worker
/// threads exist. DirectX and Vulkan take the request through the launch and
/// need no relaunch, so this does nothing elsewhere.
///
/// The heavier `MTL_SHADER_VALIDATION` is left off: it is far more expensive and
/// its memory footprint climbs over a long run, so it stays a manual opt-in.
#[expect(
    clippy::disallowed_methods,
    reason = "Metal's debug layer is switched on by its own environment variable"
)]
#[cfg(target_os = "macos")]
pub fn reexec_with_metal_validation(requested: Option<bool>) {
    use std::os::unix::process::CommandExt;

    if !requested.unwrap_or(cfg!(debug_assertions)) {
        return;
    }
    // The relaunched child inherits the variable, so it does not re-exec again.
    if std::env::var_os("MTL_DEBUG_LAYER").is_some() {
        return;
    }
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("validation: cannot locate current executable to re-exec: {e}");
            return;
        }
    };
    // `exec` replaces this image in place and only returns on failure, which
    // runs on without Metal validation rather than aborting the session.
    let err = std::process::Command::new(exe)
        .args(std::env::args_os().skip(1))
        .env("MTL_DEBUG_LAYER", "1")
        .exec();
    eprintln!("validation: failed to re-exec with Metal validation enabled: {err}");
}

/// Relaunch the process with Metal validation on macOS; nothing to do here.
#[cfg(not(target_os = "macos"))]
pub fn reexec_with_metal_validation(_requested: Option<bool>) {}

#[cfg(test)]
mod tests {
    use super::*;

    // Requesting validation would `exec` and replace the test process, so only
    // the declining path is driven.
    #[test]
    fn metal_validation_reexec_declines_when_turned_off() {
        reexec_with_metal_validation(Some(false));
    }
}
