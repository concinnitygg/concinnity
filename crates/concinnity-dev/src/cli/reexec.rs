// The macOS validation re-exec, which has to run before any thread starts.
use super::Cli;

// When a render command requests graphics validation on macOS, relaunch the
// process with Metal's API-validation layer (`MTL_DEBUG_LAYER`) set in the
// environment, then return into the replacement image. Metal reads that
// variable during early framework initialization, so it cannot be toggled from
// a process that has already touched Metal -- and `std::env::set_var` is
// unsound once worker threads exist (the frameworks call `getenv` off-thread).
// Re-exec sidesteps both: the child starts with the variable present from PID
// birth. DirectX / Vulkan take the request through `LaunchRequest` and need no
// relaunch, so this is a macOS-only concern.
//
// The heavier `MTL_SHADER_VALIDATION` is deliberately left off: it is far more
// expensive and its memory footprint climbs over a long run, so it stays an
// explicit manual opt-in rather than riding a flag that defaults on in debug
// builds.
#[cfg(target_os = "macos")]
pub(super) fn reexec_with_metal_validation(cli: &Cli) {
    use super::Commands;
    use std::os::unix::process::CommandExt;

    // Only the rendering commands create a Metal context; every other
    // subcommand starts no renderer, so it needs no validation re-exec.
    let requested = match cli.resolved_command() {
        Commands::Run(args) => args.validation,
        Commands::Debug(args) => args.validation,
        Commands::Editor(args) => args.validation,
        _ => return,
    };
    if !requested.unwrap_or(cfg!(debug_assertions)) {
        return;
    }
    // The relaunched child inherits the variable, so the guard is
    // self-terminating: it stops the second pass from re-execing again.
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
    // `exec` replaces this image in place (no lingering parent process) and only
    // returns on failure. On failure we fall through and run without Metal
    // validation rather than aborting the user's session.
    let err = std::process::Command::new(exe)
        .args(std::env::args_os().skip(1))
        .env("MTL_DEBUG_LAYER", "1")
        .exec();
    eprintln!("validation: failed to re-exec with Metal validation enabled: {err}");
}

#[cfg(not(target_os = "macos"))]
pub(super) fn reexec_with_metal_validation(_cli: &Cli) {}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use clap::Parser;

    // Only the paths that decline to re-exec are driven: the guard returns for a
    // command that starts no renderer, and for a render command with validation
    // explicitly off. Requesting validation would `exec` and replace the test
    // process, so no case here leaves `validation` unset or true.
    #[test]
    fn metal_validation_reexec_declines_when_not_requested() {
        for args in [
            ["concinnity", "run", "--validation", "false"],
            ["concinnity", "editor", "--validation", "false"],
            ["concinnity", "debug", "--validation", "false"],
        ] {
            reexec_with_metal_validation(&Cli::try_parse_from(args).unwrap());
        }
        // An authoring subcommand stands up no renderer, so it returns at the
        // match rather than reaching the request check.
        reexec_with_metal_validation(&Cli::try_parse_from(["concinnity", "list"]).unwrap());
        reexec_with_metal_validation(&Cli::try_parse_from(["concinnity", "mcp"]).unwrap());
    }
}
