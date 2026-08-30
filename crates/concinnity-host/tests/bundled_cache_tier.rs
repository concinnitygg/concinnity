//! The two tiers a cache lookup consults, driven through the process-global
//! state roots.
//!
//! One test, and one test binary, on purpose: the roots and both loaded
//! segments are process-wide, so a second test function here would race this
//! one for them. The unit tests beside the code cover the pieces; this covers
//! what a shipped bundle actually does with them.

use std::fs;
use std::path::Path;

use concinnity_host::store::cache::{self, CacheEntryKind, Segment};
use concinnity_host::store::paths;

const KIND: CacheEntryKind = CacheEntryKind::Shader;
const SHIPPED: &str = "shipped-artifact";
const BUDGET: u64 = 1 << 20;

// Warm a segment at `root`'s `cache/0` the way `cn export` does: entries, and
// no toolchain stamp.
fn warm(root: &Path, key: &str, bytes: &[u8]) {
    let mut segment = Segment::read_from(&paths::runtime_cache_in(root));
    segment.put(KIND, key, bytes);
    assert!(segment.write_to(&paths::runtime_cache_in(root), BUDGET));
}

#[test]
fn a_shipped_segment_serves_both_install_layouts() {
    let content = tempfile::tempdir().unwrap();
    let per_user = tempfile::tempdir().unwrap();
    warm(content.path(), SHIPPED, &[1, 2, 3, 4]);
    let shipped_image = fs::read(paths::runtime_cache_in(content.path())).unwrap();

    // A read-only install: the roots diverge, so the shipped segment is a tier
    // of its own and the application writes a segment beside the user's saves.
    paths::set_state_dir(content.path());
    paths::set_writable_state_dir(per_user.path());
    assert_eq!(cache::load(KIND, SHIPPED), None, "not in the writable tier");
    assert_eq!(cache::load_bundled(KIND, SHIPPED), Some(vec![1, 2, 3, 4]));

    assert!(cache::store(KIND, "compiled-at-launch", &[9, 9]));
    assert!(cache::flush());
    let mut written = Segment::read_from(&paths::runtime_cache_in(per_user.path()));
    assert_eq!(written.get(KIND, "compiled-at-launch"), Some(&[9, 9][..]));
    assert_eq!(
        fs::read(paths::runtime_cache_in(content.path())).unwrap(),
        shipped_image,
        "the shipped segment is read-only"
    );

    // The portable folder: one root, so one file in both roles. The writable
    // tier serves the shipped entries, the second tier reports a miss rather
    // than putting a stale copy of the same file in front of it, and a launch
    // that compiles something new must not drop what the bundle shipped.
    let bundle = tempfile::tempdir().unwrap();
    warm(bundle.path(), SHIPPED, &[1, 2, 3, 4]);
    paths::set_state_dir(bundle.path());
    paths::clear_writable_state_dir();
    assert_eq!(cache::load(KIND, SHIPPED), Some(vec![1, 2, 3, 4]));
    assert_eq!(
        cache::load_bundled(KIND, SHIPPED),
        None,
        "one file, one tier"
    );

    // An unstamped segment is what `cn export` writes, so a player whose own
    // shader toolchain differs from the exporter's keeps the shipped entries.
    assert!(!cache::verify_toolchain("slang 2026.9"));
    assert!(cache::store(KIND, "compiled-at-launch", &[7]));
    assert!(cache::flush());
    let mut relaunched = Segment::read_from(&paths::runtime_cache_in(bundle.path()));
    assert_eq!(relaunched.get(KIND, SHIPPED), Some(&[1, 2, 3, 4][..]));
    assert_eq!(relaunched.get(KIND, "compiled-at-launch"), Some(&[7][..]));

    paths::clear_state_dir();
}
