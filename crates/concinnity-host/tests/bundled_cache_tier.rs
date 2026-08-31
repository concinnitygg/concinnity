//! The two tiers a cache lookup consults, driven through the anchors a host
//! installs.
//!
//! One test, and one test binary, on purpose: both loaded segments are
//! process-wide, so a second test function here would race this one for them.
//! The unit tests beside the code cover the pieces; this covers what a shipped
//! bundle actually does with them.

use std::fs;
use std::path::Path;

use concinnity_host::store::cache::{self, CacheAnchor, CacheEntryKind, Segment};
use concinnity_host::store::paths::StateTree;

const KIND: CacheEntryKind = CacheEntryKind::Shader;
const SHIPPED: &str = "shipped-artifact";
const BUDGET: u64 = 1 << 20;

// The segment `cn export` warms inside a bundle rooted at `root`.
fn shipped_segment(root: &Path) -> std::path::PathBuf {
    StateTree::at(root).bundled_runtime_cache_path()
}

// Warm a bundle's segment the way `cn export` does: entries, and no toolchain
// stamp.
fn warm(root: &Path, key: &str, bytes: &[u8]) {
    let path = shipped_segment(root);
    let mut segment = Segment::read_from(&path);
    segment.put(KIND, key, bytes);
    assert!(segment.write_to(&path, BUDGET));
}

// The anchor a host builds from the tree it resolved.
fn anchor_to(tree: &StateTree) {
    cache::anchor(
        CacheAnchor::new(tree.runtime_cache_path()).with_bundled(tree.bundled_runtime_cache_path()),
    );
}

#[test]
fn a_shipped_segment_serves_every_install_layout() {
    let content = tempfile::tempdir().unwrap();
    let per_user = tempfile::tempdir().unwrap();
    warm(content.path(), SHIPPED, &[1, 2, 3, 4]);
    let shipped_image = fs::read(shipped_segment(content.path())).unwrap();

    // A read-only install: the roots diverge, so the shipped segment is a tier
    // of its own and the application writes a segment beside the user's saves.
    anchor_to(&StateTree::at(content.path()).with_writable(per_user.path()));
    assert_eq!(cache::load(KIND, SHIPPED), None, "not in the writable tier");
    assert_eq!(cache::load_bundled(KIND, SHIPPED), Some(vec![1, 2, 3, 4]));

    assert!(cache::store(KIND, "compiled-at-launch", &[9, 9]));
    assert!(cache::flush());
    let mut written = Segment::read_from(&StateTree::at(per_user.path()).runtime_cache_path());
    assert_eq!(written.get(KIND, "compiled-at-launch"), Some(&[9, 9][..]));
    assert_eq!(
        fs::read(shipped_segment(content.path())).unwrap(),
        shipped_image,
        "the shipped segment is read-only"
    );

    // A read-only content mount with a cache root of its own: the writable
    // segment follows the cache root rather than the saves, which is the layout
    // one anchor tied to the content root could never express.
    let cache_root = tempfile::tempdir().unwrap();
    anchor_to(
        &StateTree::at(content.path())
            .with_writable(per_user.path())
            .with_cache(cache_root.path()),
    );
    assert_eq!(
        cache::load_bundled(KIND, SHIPPED),
        Some(vec![1, 2, 3, 4]),
        "the shipped tier still reads from the content mount"
    );
    assert!(cache::store(KIND, "compiled-on-a-read-only-mount", &[5]));
    assert!(cache::flush());
    let mut on_cache_root =
        Segment::read_from(&StateTree::at(cache_root.path()).runtime_cache_path());
    assert_eq!(
        on_cache_root.get(KIND, "compiled-on-a-read-only-mount"),
        Some(&[5][..])
    );

    // The portable folder: one root, so one file in both roles. The writable
    // tier serves the shipped entries, the second tier reports a miss rather
    // than putting a stale copy of the same file in front of it, and a launch
    // that compiles something new must not drop what the bundle shipped.
    let bundle = tempfile::tempdir().unwrap();
    warm(bundle.path(), SHIPPED, &[1, 2, 3, 4]);
    anchor_to(&StateTree::at(bundle.path()));
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
    let mut relaunched = Segment::read_from(&shipped_segment(bundle.path()));
    assert_eq!(relaunched.get(KIND, SHIPPED), Some(&[1, 2, 3, 4][..]));
    assert_eq!(relaunched.get(KIND, "compiled-at-launch"), Some(&[7][..]));

    cache::clear_anchor();
}
