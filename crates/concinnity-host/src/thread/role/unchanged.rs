// No scheduler change: the thread keeps the platform's defaults.

use super::ThreadRole;

#[expect(
    clippy::unnecessary_wraps,
    reason = "matches the Apple and Windows `apply`, which can fail"
)]
pub(super) fn apply(_role: ThreadRole) -> std::io::Result<()> {
    Ok(())
}
