//! Target wrapper around `crate::authoring::rm_at_path`.

use crate::authoring::rm_at_path;
use crate::command::Target;

/// Delete the asset named `name` from the file `target` names.
pub(crate) fn rm(target: Target<'_>, name: &str) -> std::io::Result<()> {
    rm_at_path(&target.resolve()?, name)
}
