// Scratch directory for the shader compilers that work on files: slangc, and
// the `xcrun metal` two-step the Metal backend runs for a runtime metallib.
// Source and artifact are written and removed around each compile, so this is
// transient scratch rather than cache -- nothing here survives a compile, and
// nothing here belongs in the project's state tree.

use std::path::PathBuf;

// Shared by every compile in the process; the scratch names inside carry the
// pid, so two applications on one checkout do not collide.
pub(crate) fn dir() -> PathBuf {
    std::env::temp_dir().join("concinnity-shader-work")
}
