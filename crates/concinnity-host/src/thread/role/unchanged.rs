// No scheduler change: the thread keeps the platform's defaults.

use super::ThreadRole;

pub(super) fn apply(_role: ThreadRole) -> std::io::Result<()> {
    Ok(())
}
