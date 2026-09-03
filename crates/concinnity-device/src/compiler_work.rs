// Scratch directory for the shader compilers that work on files: slangc, and
// the `xcrun metal` two-step the Metal backend runs for a runtime metallib.
// Source and artifact are written and removed around each compile, so this is
// transient scratch rather than cache -- nothing here survives a compile, and
// nothing here belongs in the project's state tree.

// One compile's working directory, removed when the returned value drops.
//
// A directory of its own per compile rather than one shared by the process:
// slangc names its own outputs, and a run that fails part-way leaves them for
// the guard rather than for the next compile to trip over.
pub(crate) fn dir() -> Result<concinnity_host::scratch::Scratch, String> {
    concinnity_host::scratch::Scratch::dir("slang")
        .map_err(|e| format!("create the shader compiler's work directory: {e}"))
}
