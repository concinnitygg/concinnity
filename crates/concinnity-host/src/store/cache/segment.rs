// One cache segment, held in memory between its two moments of file I/O: the
// read that loads it and the write that replaces it.
//
// A container is written whole while a cache accumulates, so the segment is
// read once, every lookup and every store lands in the `Segment` below, and the
// file is replaced at a flush. A producer that stores in a loop -- a renderer
// init compiling fifty shaders -- therefore costs one read and one write rather
// than fifty of each, and the read it does costs less than the first compile it
// saves.
//
// Every function here takes the path, so the segment machinery is exercised
// without the process-global state root; `super` resolves which file a caller
// means.

use std::fs;
use std::io::Write;
use std::path::Path;

use concinnity_core::blob::{
    CACHE_SEGMENT_VERSION, CacheEntry, CacheEntryKind, CacheMeta, encode_cnb, parse_cnb,
};

// One entry held in memory. `used` marks an entry this process looked up or
// wrote, which eviction spares.
struct Item {
    kind: CacheEntryKind,
    key: String,
    bytes: Vec<u8>,
    used: bool,
}

/// A segment's entries in memory, plus whether they differ from the file they
/// were read from.
///
/// `super` holds one of these per process for the segment this application
/// writes, and one for the read-only segment a bundle ships. `cn export` owns
/// a third directly: the segment it warms into a bundle is one this process
/// never reads back, so it is built and written as a value rather than through
/// the process-global tiers.
pub struct Segment {
    items: Vec<Item>,
    toolchain: String,
    dirty: bool,
}

impl Segment {
    /// Read `path` into memory. An absent, unreadable, or foreign file reads as
    /// an empty segment: whatever it held is regenerated, and the next write
    /// replaces it.
    pub fn read_from(path: &Path) -> Self {
        let Ok(image) = fs::read(path) else {
            return Self::empty();
        };
        let Ok((meta, payload_start)) = parse_cnb::<CacheMeta>(CACHE_SEGMENT_VERSION, &image)
        else {
            return Self::empty();
        };
        Self {
            items: meta
                .entries
                .iter()
                .filter_map(|entry| {
                    Some(Item {
                        kind: entry.kind,
                        key: entry.key.clone(),
                        bytes: entry_bytes(&image, payload_start, entry)?.to_vec(),
                        used: false,
                    })
                })
                .collect(),
            toolchain: meta.toolchain,
            dirty: false,
        }
    }

    fn empty() -> Self {
        Self {
            items: Vec::new(),
            toolchain: String::new(),
            dirty: false,
        }
    }

    /// The bytes stored for `key`, or `None` when the segment holds no such
    /// entry. Marks the entry used, so a flush does not evict what this run is
    /// running on.
    pub fn get(&mut self, kind: CacheEntryKind, key: &str) -> Option<&[u8]> {
        let item = self
            .items
            .iter_mut()
            .find(|i| i.kind == kind && i.key == key)?;
        item.used = true;
        Some(&item.bytes)
    }

    /// Take `bytes` as `key`'s entry, reporting whether the segment took them.
    ///
    /// An entry already holding at least as many bytes is left alone. Growth is
    /// the only reliable "new content" signal for a driver pipeline blob, which
    /// only accumulates but does not serialize deterministically (MoltenVK
    /// shuffles entry order run to run), so a byte compare would dirty the
    /// segment every launch. A content-addressed artifact re-stored under its
    /// own key is by definition the bytes already there.
    pub fn put(&mut self, kind: CacheEntryKind, key: &str, bytes: &[u8]) -> bool {
        if bytes.is_empty() {
            return false;
        }
        match self
            .items
            .iter_mut()
            .find(|i| i.kind == kind && i.key == key)
        {
            Some(item) if bytes.len() <= item.bytes.len() => {
                item.used = true;
                return false;
            }
            Some(item) => {
                item.bytes.clear();
                item.bytes.extend_from_slice(bytes);
                item.used = true;
            }
            None => self.items.push(Item {
                kind,
                key: key.to_owned(),
                bytes: bytes.to_vec(),
                used: true,
            }),
        }
        self.dirty = true;
        true
    }

    /// Drop `key`'s entry, for a caller whose artifact turned out unusable.
    pub(super) fn remove(&mut self, kind: CacheEntryKind, key: &str) {
        let before = self.items.len();
        self.items.retain(|i| !(i.kind == kind && i.key == key));
        self.dirty |= self.items.len() != before;
    }

    /// Adopt `id` as the toolchain the segment's entries were produced by,
    /// discarding every entry when it names another one. Reports whether it
    /// discarded.
    ///
    /// An unstamped segment predates any entry a toolchain produced -- a
    /// shader compile stamps the segment before it stores -- so adopting the
    /// first stamp keeps what is already there.
    pub(super) fn adopt_toolchain(&mut self, id: &str) -> bool {
        if self.toolchain == id {
            return false;
        }
        let discarded = !self.toolchain.is_empty() && !self.items.is_empty();
        if discarded {
            self.items.clear();
        }
        self.toolchain = id.to_owned();
        self.dirty = true;
        discarded
    }

    /// Replace `path` with what the segment now holds, first evicting entries
    /// until its payload fits `budget`. Reports whether the file was written.
    ///
    /// A segment nothing changed is not rewritten: this is where the per-store
    /// growth check pays off, since a launch that only read the cache leaves
    /// the file untouched.
    pub fn write_to(&mut self, path: &Path, budget: u64) -> bool {
        if !self.dirty {
            return false;
        }
        self.evict_to(budget);
        if self.items.is_empty() {
            // A cache nothing needs leaves nothing behind. The stamp goes with
            // it, which costs the next run a discard of an empty segment.
            let _ = fs::remove_file(path);
            self.dirty = false;
            return false;
        }
        let mut payload = Vec::with_capacity(self.items.iter().map(|i| i.bytes.len()).sum());
        let entries = self
            .items
            .iter()
            .map(|item| {
                let entry = CacheEntry {
                    kind: item.kind,
                    key: item.key.clone(),
                    offset: payload.len() as u64,
                    len: item.bytes.len() as u64,
                };
                payload.extend_from_slice(&item.bytes);
                entry
            })
            .collect();
        let meta = CacheMeta {
            toolchain: self.toolchain.clone(),
            entries,
        };
        let Ok(image) = encode_cnb(CACHE_SEGMENT_VERSION, &meta, &payload) else {
            return false;
        };
        if !crate::store::atomic::replace(path, |out| out.write_all(&image)) {
            return false;
        }
        self.dirty = false;
        true
    }

    // Drop entries oldest-first until the payload fits `budget`, sparing the
    // ones this process used: evicting an artifact the live run is holding
    // guarantees the next launch recompiles it. Content-addressed entries are
    // interchangeable, so index order -- the order they were first stored in --
    // is the same least-recently-written proxy the directory sweep had in mtimes.
    fn evict_to(&mut self, budget: u64) {
        let mut total: u64 = self.items.iter().map(|i| i.bytes.len() as u64).sum();
        if total <= budget {
            return;
        }
        let before = self.items.len();
        self.items.retain(|item| {
            if total <= budget || item.used {
                return true;
            }
            total -= item.bytes.len() as u64;
            false
        });
        self.dirty |= self.items.len() != before;
    }
}

// An entry's slice of the payload section, or `None` when the index points past
// the image (a truncated or hand-edited segment).
fn entry_bytes<'a>(image: &'a [u8], payload_start: usize, entry: &CacheEntry) -> Option<&'a [u8]> {
    let start = payload_start.checked_add(usize::try_from(entry.offset).ok()?)?;
    let end = start.checked_add(usize::try_from(entry.len).ok()?)?;
    image.get(start..end)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PIPELINE: CacheEntryKind = CacheEntryKind::Pipeline;
    const SHADER: CacheEntryKind = CacheEntryKind::Shader;
    const BUDGET: u64 = 1024;

    fn segment_path(dir: &tempfile::TempDir) -> std::path::PathBuf {
        dir.path().join("cache").join("0")
    }

    // Store into a fresh segment and write it, the shape every test below
    // starts from.
    fn written(path: &Path, entries: &[(CacheEntryKind, &str, &[u8])]) -> Segment {
        let mut segment = Segment::read_from(path);
        for (kind, key, bytes) in entries {
            segment.put(*kind, key, bytes);
        }
        segment.write_to(path, BUDGET);
        segment
    }

    #[test]
    fn an_entry_round_trips_through_a_segment() {
        let dir = tempfile::tempdir().unwrap();
        let path = segment_path(&dir);
        written(&path, &[(PIPELINE, "vk-aa", &[1, 2, 3])]);

        let mut reread = Segment::read_from(&path);
        assert_eq!(reread.get(PIPELINE, "vk-aa"), Some(&[1, 2, 3][..]));

        let leftovers = fs::read_dir(path.parent().unwrap())
            .unwrap()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "tmp"))
            .count();
        assert_eq!(leftovers, 0, "temp files must not survive a write");
    }

    // The reason the index exists: two adapters used in one run share the
    // segment, and the second store must not take the first's bytes with it.
    // Two producers share it the same way, which is what admits the shader cache.
    #[test]
    fn one_entry_does_not_clobber_another() {
        let dir = tempfile::tempdir().unwrap();
        let path = segment_path(&dir);
        written(
            &path,
            &[
                (PIPELINE, "vk-aa", &[1, 2, 3]),
                (PIPELINE, "vk-bb", &[9]),
                (SHADER, "vk-aa", &[4, 4]),
            ],
        );

        let mut reread = Segment::read_from(&path);
        assert_eq!(reread.get(PIPELINE, "vk-aa"), Some(&[1, 2, 3][..]));
        assert_eq!(reread.get(PIPELINE, "vk-bb"), Some(&[9][..]));
        assert_eq!(reread.get(SHADER, "vk-aa"), Some(&[4, 4][..]));

        // A rewrite of one entry leaves the others intact and readdressed.
        reread.put(PIPELINE, "vk-aa", &[4, 5, 6, 7]);
        reread.write_to(&path, BUDGET);
        let mut last = Segment::read_from(&path);
        assert_eq!(last.get(PIPELINE, "vk-aa"), Some(&[4, 5, 6, 7][..]));
        assert_eq!(last.get(PIPELINE, "vk-bb"), Some(&[9][..]));
        assert_eq!(last.get(SHADER, "vk-aa"), Some(&[4, 4][..]));
    }

    // The whole point of holding the segment in memory: a run that only reads
    // it touches the file once, and never writes it.
    #[test]
    fn a_segment_nothing_changed_is_not_rewritten() {
        let dir = tempfile::tempdir().unwrap();
        let path = segment_path(&dir);
        written(&path, &[(SHADER, "cafe", &[1, 2, 3])]);
        let before = fs::metadata(&path).unwrap().len();

        let mut warm = Segment::read_from(&path);
        assert_eq!(warm.get(SHADER, "cafe"), Some(&[1, 2, 3][..]));
        assert!(
            !warm.write_to(&path, BUDGET),
            "a read-only run writes nothing"
        );
        assert_eq!(fs::metadata(&path).unwrap().len(), before);
    }

    // A driver blob's serialization is nondeterministic, so equal-length bytes
    // must leave the segment alone; only growth is new content.
    #[test]
    fn only_growth_replaces_an_entry() {
        let dir = tempfile::tempdir().unwrap();
        let path = segment_path(&dir);
        let mut segment = written(&path, &[(PIPELINE, "vk-aa", &[5, 5])]);

        assert!(!segment.put(PIPELINE, "vk-aa", &[5, 5]), "unchanged");
        assert!(!segment.put(PIPELINE, "vk-aa", &[6, 5]), "reshuffled");
        assert!(!segment.put(PIPELINE, "vk-aa", &[5]), "shrunk");
        assert!(!segment.put(PIPELINE, "vk-bb", &[]), "empty");
        assert!(!segment.write_to(&path, BUDGET), "none of those is content");

        assert!(segment.put(PIPELINE, "vk-aa", &[5, 5, 6]), "grew");
        assert!(segment.write_to(&path, BUDGET));
        let mut reread = Segment::read_from(&path);
        assert_eq!(reread.get(PIPELINE, "vk-aa"), Some(&[5, 5, 6][..]));
    }

    // Deleting `cache/` at any time has to leave the app working, so an absent
    // segment reads as a miss rather than an error, and the next write recreates
    // the file and the directory under it.
    #[test]
    fn an_absent_segment_reads_as_a_miss_and_is_recreated() {
        let dir = tempfile::tempdir().unwrap();
        let path = segment_path(&dir);
        assert!(!path.exists());
        assert_eq!(Segment::read_from(&path).get(PIPELINE, "vk-aa"), None);

        written(&path, &[(PIPELINE, "vk-aa", &[1])]);
        fs::remove_dir_all(path.parent().unwrap()).unwrap();
        assert_eq!(Segment::read_from(&path).get(PIPELINE, "vk-aa"), None);
        written(&path, &[(PIPELINE, "vk-aa", &[1])]);
        assert_eq!(
            Segment::read_from(&path).get(PIPELINE, "vk-aa"),
            Some(&[1][..])
        );
    }

    #[test]
    fn a_missing_entry_reads_as_a_miss() {
        let dir = tempfile::tempdir().unwrap();
        let path = segment_path(&dir);
        let mut segment = written(&path, &[(PIPELINE, "vk-aa", &[1])]);
        assert_eq!(segment.get(PIPELINE, "vk-bb"), None);
    }

    #[test]
    fn a_removed_entry_is_gone_and_the_last_one_takes_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = segment_path(&dir);
        let mut segment = written(
            &path,
            &[(PIPELINE, "vk-aa", &[1]), (PIPELINE, "vk-bb", &[2])],
        );

        segment.remove(PIPELINE, "vk-aa");
        assert!(segment.write_to(&path, BUDGET));
        let mut reread = Segment::read_from(&path);
        assert_eq!(reread.get(PIPELINE, "vk-aa"), None);
        assert_eq!(reread.get(PIPELINE, "vk-bb"), Some(&[2][..]));

        reread.remove(PIPELINE, "vk-bb");
        assert!(!reread.write_to(&path, BUDGET));
        assert!(!path.exists(), "an empty segment leaves no file behind");
        // Removing what is not there is a no-op, not a rewrite.
        reread.remove(PIPELINE, "vk-bb");
        assert!(!reread.write_to(&path, BUDGET));
        assert!(!path.exists());
    }

    // Garbage in the segment's place (a truncated write from an older layout,
    // a file that was never a segment) costs a regeneration, never a failure.
    #[test]
    fn a_corrupt_segment_reads_empty_and_is_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let path = segment_path(&dir);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"not a segment at all").unwrap();
        assert_eq!(Segment::read_from(&path).get(PIPELINE, "vk-aa"), None);

        written(&path, &[(PIPELINE, "vk-aa", &[7])]);
        assert_eq!(
            Segment::read_from(&path).get(PIPELINE, "vk-aa"),
            Some(&[7][..])
        );
    }

    // The index can outlive the payload it addresses if a write was truncated
    // by something outside the rename. Such an entry reads as absent.
    #[test]
    fn an_entry_pointing_past_the_image_reads_as_a_miss() {
        let dir = tempfile::tempdir().unwrap();
        let path = segment_path(&dir);
        written(&path, &[(PIPELINE, "vk-aa", &[1, 2, 3])]);
        let image = fs::read(&path).unwrap();
        fs::write(&path, &image[..image.len() - 1]).unwrap();
        assert_eq!(Segment::read_from(&path).get(PIPELINE, "vk-aa"), None);
    }

    // The runtime writes its own segment and nothing else: a build segment
    // beside it is neither read nor disturbed, whether or not it exists.
    #[test]
    fn a_sibling_segment_is_left_alone() {
        let dir = tempfile::tempdir().unwrap();
        let path = segment_path(&dir);
        let sibling = path.parent().unwrap().join("1");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&sibling, b"build segment").unwrap();

        let mut segment = written(&path, &[(PIPELINE, "vk-aa", &[1])]);
        segment.remove(PIPELINE, "vk-aa");
        segment.write_to(&path, BUDGET);
        assert!(!path.exists());
        assert_eq!(fs::read(&sibling).unwrap(), b"build segment");
    }

    // The stamp rides the segment rather than a sidecar file, and a host
    // compiler upgrade is what it exists to catch.
    #[test]
    fn a_toolchain_change_discards_the_entries() {
        let dir = tempfile::tempdir().unwrap();
        let path = segment_path(&dir);
        let mut segment = written(&path, &[(SHADER, "cafe", &[1, 2])]);
        // The first stamp claims what is there rather than discarding it: a
        // compile stamps the segment before it stores, so an unstamped entry
        // came from no toolchain this could disagree with.
        assert!(!segment.adopt_toolchain("slang 2026.1"));
        assert!(segment.write_to(&path, BUDGET), "the stamp is a change");

        // The same toolchain keeps every entry and dirties nothing.
        let mut warm = Segment::read_from(&path);
        assert!(!warm.adopt_toolchain("slang 2026.1"));
        assert_eq!(warm.get(SHADER, "cafe"), Some(&[1, 2][..]));
        assert!(!warm.write_to(&path, BUDGET));

        // Another one drops what it did not produce, and the drop reaches disk.
        let mut upgraded = Segment::read_from(&path);
        assert!(upgraded.adopt_toolchain("slang 2026.2"), "discarded");
        assert_eq!(upgraded.get(SHADER, "cafe"), None);
        upgraded.put(SHADER, "f00d", &[3]);
        upgraded.write_to(&path, BUDGET);
        let mut reread = Segment::read_from(&path);
        assert_eq!(reread.get(SHADER, "cafe"), None);
        assert_eq!(reread.get(SHADER, "f00d"), Some(&[3][..]));
        assert!(!reread.adopt_toolchain("slang 2026.2"), "stamp persisted");
    }

    #[test]
    fn nothing_is_evicted_under_budget() {
        let dir = tempfile::tempdir().unwrap();
        let path = segment_path(&dir);
        let mut segment = written(&path, &[(SHADER, "a", &[0; 8]), (SHADER, "b", &[0; 8])]);
        segment.evict_to(64);
        assert!(segment.get(SHADER, "a").is_some());
        assert!(segment.get(SHADER, "b").is_some());
    }

    // Eviction takes the oldest entries first, and spares whatever this run
    // touched: dropping an artifact the live process is running on would only
    // buy a recompile on the next launch.
    #[test]
    fn eviction_drops_oldest_first_and_spares_what_this_run_used() {
        let dir = tempfile::tempdir().unwrap();
        let path = segment_path(&dir);
        written(
            &path,
            &[
                (SHADER, "oldest", &[0; 40]),
                (SHADER, "middle", &[0; 40]),
                (SHADER, "newest", &[0; 40]),
            ],
        );

        let mut warm = Segment::read_from(&path);
        assert!(warm.get(SHADER, "oldest").is_some(), "this run needs it");
        warm.evict_to(80);
        assert!(warm.get(SHADER, "oldest").is_some());
        assert!(warm.get(SHADER, "middle").is_none());
        assert!(warm.get(SHADER, "newest").is_some());

        // An eviction is itself a change, so it reaches disk on the next flush.
        assert!(warm.write_to(&path, 1024));
        assert!(Segment::read_from(&path).get(SHADER, "middle").is_none());
    }

    // A budget nothing untouched can satisfy evicts what it can and keeps the
    // rest, rather than throwing away the run's own artifacts.
    #[test]
    fn eviction_stops_at_the_entries_this_run_used() {
        let dir = tempfile::tempdir().unwrap();
        let path = segment_path(&dir);
        let mut segment = written(&path, &[(SHADER, "a", &[0; 40]), (SHADER, "b", &[0; 40])]);
        segment.evict_to(0);
        assert!(segment.get(SHADER, "a").is_some());
        assert!(segment.get(SHADER, "b").is_some());
    }
}
