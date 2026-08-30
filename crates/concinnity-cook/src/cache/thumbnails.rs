//! The thumbnail set inside the build segment: the baked PNG entries and the
//! one entry mapping asset name to key.
//!
//! Cook produces these and the editor reads them, which makes this the one
//! place in the cache a consumer lives outside the writer's process. Both
//! halves are here so the key space stays in one file: a PNG is keyed by a
//! digest of what it depicts, so two assets that look alike share one entry,
//! and the name map is a single entry beside them rather than being folded into
//! those keys.
//!
//! A reader opens the segment's index and seeks to the entries it wants. The
//! set changes exactly when the name map does -- a content change moves a key,
//! a rename moves a name -- so [`Thumbnails::revision`] is a digest of that one
//! entry and nothing else. Stamping on the file itself would be wrong: the
//! payload cache shares this segment, so every build would bump it.

use std::path::Path;

use sha2::{Digest, Sha256};

use super::segment::Index;
use concinnity_core::blob::CacheEntryKind;

const THUMBNAIL: CacheEntryKind = CacheEntryKind::Thumbnail;

// The entry holding the whole set's name -> key map. Every other thumbnail key
// is a 64-character hex digest, so this name cannot collide with one.
const NAMES_KEY: &str = "names";

/// A read-only view of the thumbnail entries the build segment holds: the
/// asset-name-to-key map, and the PNG bytes each key addresses.
///
/// Reading is best effort throughout. An absent, unreadable, or foreign
/// segment opens as `None` and a key that resolves to nothing reads as `None`,
/// which costs a consumer its previews and never more than that.
pub struct Thumbnails {
    index: Index,
    names: Vec<(String, String)>,
    revision: u64,
}

impl Thumbnails {
    /// Open the installed state root's build segment. `None` when there is no
    /// state root, when the running binary cannot be identified, or when the
    /// segment holds no thumbnail set (a deleted `cache/` reads this way).
    pub fn open() -> Option<Self> {
        Self::open_at(
            &crate::paths::build_cache_path()?,
            super::identity::token()?,
        )
    }

    fn open_at(path: &Path, token: u32) -> Option<Self> {
        let index = Index::read(path, token);
        let bytes = index.get(THUMBNAIL, NAMES_KEY)?;
        Some(Self {
            names: decode_names(&bytes)?,
            revision: revision_of(&bytes),
            index,
        })
    }

    /// Asset name paired with the key of its thumbnail, in bake order.
    pub fn names(&self) -> &[(String, String)] {
        &self.names
    }

    /// What the set is at: the same value for two segments holding the same
    /// thumbnails under the same names, a different one as soon as either
    /// moves. A consumer caching decoded images reloads on a change of this
    /// and nothing else.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// The PNG bytes stored under `key`, read from the segment. `None` when
    /// the entry is absent, or when the file was replaced under the reader
    /// (a build publishes by rename, so a stale offset reads short or reads
    /// bytes that fail to decode).
    pub fn png(&self, key: &str) -> Option<Vec<u8>> {
        self.index.get(THUMBNAIL, key)
    }
}

/// Hold a finished bake for the next [`flush`](super::flush): every image the
/// segment does not already have, plus the name map when it moved.
///
/// The map is rewritten only when it differs from what is stored, so a build
/// whose thumbnails are all reused stores nothing and writes no file.
pub(crate) fn hold(images: &[(String, Vec<u8>)], names: &[(String, String)]) {
    for (key, png) in images {
        super::store(THUMBNAIL, key, png);
    }
    let encoded = encode_names(names);
    if super::load(THUMBNAIL, NAMES_KEY).as_deref() != Some(encoded.as_slice()) {
        super::store(THUMBNAIL, NAMES_KEY, &encoded);
    }
}

/// Whether the segment already holds the thumbnail keyed `key`.
pub(crate) fn holds(key: &str) -> bool {
    super::contains(THUMBNAIL, key)
}

fn encode_names(names: &[(String, String)]) -> Vec<u8> {
    postcard::to_allocvec(names).unwrap_or_default()
}

fn decode_names(bytes: &[u8]) -> Option<Vec<(String, String)>> {
    postcard::from_bytes(bytes).ok()
}

// The leading eight bytes of the map's digest: a rename or a content change
// moves the map, and moving the map moves this.
fn revision_of(names: &[u8]) -> u64 {
    let digest: [u8; 32] = Sha256::digest(names).into();
    u64::from_le_bytes(digest[..8].try_into().expect("eight bytes of a digest"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKEN: u32 = 0xB0BA;

    fn names(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(n, k)| ((*n).to_string(), (*k).to_string()))
            .collect()
    }

    // Write a set the way `hold` + a flush would, without the process-global
    // segment those go through (disabled under `cargo test`).
    fn write_set(path: &Path, images: &[(&str, &[u8])], pairs: &[(&str, &str)]) {
        let encoded = encode_names(&names(pairs));
        let mut items: Vec<(CacheEntryKind, &str, &[u8])> = images
            .iter()
            .map(|(key, png)| (THUMBNAIL, *key, *png))
            .collect();
        items.push((THUMBNAIL, NAMES_KEY, &encoded));
        let index = Index::read(path, TOKEN);
        assert!(super::super::segment::write(path, &index, &items, TOKEN));
    }

    #[test]
    fn a_set_round_trips_through_a_segment() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cache").join("1");
        write_set(
            &path,
            &[("aa", &[1, 2, 3]), ("bb", &[4])],
            &[("red_tex", "aa"), ("box_mesh", "bb")],
        );

        let thumbs = Thumbnails::open_at(&path, TOKEN).expect("a set");
        assert_eq!(
            thumbs.names(),
            names(&[("red_tex", "aa"), ("box_mesh", "bb")])
        );
        assert_eq!(thumbs.png("aa"), Some(vec![1, 2, 3]));
        assert_eq!(thumbs.png("nope"), None);
    }

    // Two assets that look alike share one entry, which is what keeps the map
    // out of the key space.
    #[test]
    fn two_names_may_address_one_image() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("1");
        write_set(&path, &[("aa", &[9])], &[("one", "aa"), ("two", "aa")]);

        let thumbs = Thumbnails::open_at(&path, TOKEN).expect("a set");
        assert_eq!(thumbs.names().len(), 2);
        assert_eq!(thumbs.png("aa"), Some(vec![9]));
    }

    // The staleness stamp: a rename moves the set without moving any key, and
    // a content change moves a key without moving any name. Both have to be
    // visible, and an unchanged bake must not be.
    #[test]
    fn the_revision_follows_the_name_map() {
        let dir = tempfile::tempdir().unwrap();
        let one = dir.path().join("one");
        let two = dir.path().join("two");
        let three = dir.path().join("three");
        let four = dir.path().join("four");
        write_set(&one, &[("aa", &[1])], &[("red_tex", "aa")]);
        write_set(&two, &[("aa", &[1])], &[("red_tex", "aa")]);
        write_set(&three, &[("aa", &[1])], &[("blue_tex", "aa")]);
        write_set(&four, &[("bb", &[1])], &[("red_tex", "bb")]);

        let revision = |p: &Path| Thumbnails::open_at(p, TOKEN).expect("a set").revision();
        assert_eq!(revision(&one), revision(&two), "an unchanged bake holds");
        assert_ne!(revision(&one), revision(&three), "a rename shows");
        assert_ne!(revision(&one), revision(&four), "a content change shows");
    }

    // Deleting `cache/` costs previews and nothing else, and so does a segment
    // some other binary wrote.
    #[test]
    fn an_absent_or_foreign_segment_opens_to_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("1");
        assert!(Thumbnails::open_at(&path, TOKEN).is_none());

        write_set(&path, &[("aa", &[1])], &[("red_tex", "aa")]);
        assert!(Thumbnails::open_at(&path, TOKEN).is_some());
        assert!(Thumbnails::open_at(&path, TOKEN + 1).is_none());

        std::fs::write(&path, b"not a segment").unwrap();
        assert!(Thumbnails::open_at(&path, TOKEN).is_none());
    }

    // A segment holding payloads but no thumbnails is not a thumbnail set: the
    // editor shows typed icons rather than an empty grid of broken cells.
    #[test]
    fn a_segment_without_a_name_map_opens_to_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("1");
        let index = Index::read(&path, TOKEN);
        let payload: &[(CacheEntryKind, &str, &[u8])] =
            &[(CacheEntryKind::Payload, "cafe", &[1, 2, 3])];
        assert!(super::super::segment::write(&path, &index, payload, TOKEN));

        assert!(Thumbnails::open_at(&path, TOKEN).is_none());
    }
}
