// Which binary wrote a build segment, folded to the u32 the container header
// carries as its validity token.
//
// A cached payload is a function of its inputs and of the code that compiled
// them. The key covers the inputs; this covers the code, once for the whole
// segment rather than per entry: a segment stamped by another binary is dropped
// whole and regenerated. Modification time and length of the running executable
// are what stand for that binary, which is coarse on purpose -- any relink of
// the engine drops the build cache. A user never rebuilds the binary, so their
// cache always holds, and a developer who just changed a compiler wants exactly
// this invalidation.

use sha2::{Digest, Sha256};
use std::sync::OnceLock;
use std::time::UNIX_EPOCH;

/// The running binary's identity, or `None` when the executable cannot be
/// stated. Read once per process: it cannot change while that process runs.
///
/// A caller that gets `None` must not use the cache at all. An unidentifiable
/// binary has no token to disagree with, so its entries would be served to
/// every later build whatever code produced them.
pub(super) fn token() -> Option<u32> {
    static TOKEN: OnceLock<Option<u32>> = OnceLock::new();
    *TOKEN.get_or_init(|| {
        let exe = std::env::current_exe().ok()?;
        let meta = std::fs::metadata(exe).ok()?;
        let modified = meta.modified().ok()?.duration_since(UNIX_EPOCH).ok()?;
        Some(fold(modified.as_nanos(), meta.len()))
    })
}

// Both halves through SHA-256, so a one-bit change anywhere in either moves the
// whole token rather than a byte of it.
fn fold(modified_nanos: u128, len: u64) -> u32 {
    let mut hasher = Sha256::new();
    hasher.update(modified_nanos.to_le_bytes());
    hasher.update(len.to_le_bytes());
    let digest: [u8; 32] = hasher.finalize().into();
    u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_binary_folds_to_the_same_token() {
        assert_eq!(
            fold(1_700_000_000_000_000_000, 4096),
            fold(1_700_000_000_000_000_000, 4096)
        );
    }

    // A relink moves the modification time, and an edit that happens to keep
    // the time moves the length. Either must drop the segment.
    #[test]
    fn a_rebuilt_binary_folds_to_another_token() {
        let base = fold(1_700_000_000_000_000_000, 4096);
        assert_ne!(base, fold(1_700_000_000_000_000_001, 4096));
        assert_ne!(base, fold(1_700_000_000_000_000_000, 4097));
    }

    // The process running the tests is a binary like any other, so its identity
    // resolves rather than disabling the cache on a developer's machine.
    #[test]
    fn the_running_binary_has_a_token() {
        assert!(token().is_some());
    }
}
