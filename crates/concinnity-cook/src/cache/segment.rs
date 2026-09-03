// The build segment on disk, read once at the start of a build and replaced
// once at the end of it.
//
// The runtime segment holds its entries in memory whole, which its 64 MB budget
// makes reasonable. A build cache carries no such ceiling -- one world's meshes
// and textures run to hundreds of megabytes -- so this one keeps only the index
// resident and seeks to the single entry a lookup asks for, the way `data/`
// locates a payload through its `PayloadLocator`. The write side follows: the
// entries a build produced are held in memory, everything the previous segment
// held is copied through from it, and neither is ever materialized whole.
//
// An index is a set of byte offsets, so it addresses the file it was read from
// and no other. A second process replacing that file between the read and the
// use is routine -- an editor and a build share a cache root -- and the offsets
// then land somewhere else in a file of the same name. So the file is opened
// once, when the index is read, and every span the index serves comes from that
// handle: a lookup, and the carry-through a write does. A replaced segment
// costs the entries the other process added, never the wrong bytes under the
// right key.
//
// Which file the segment is arrives as an argument and is never resolved here,
// so the machinery is exercised without the process-global state root and no
// name a state tree chose appears in this module.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::{Mutex, MutexGuard, PoisonError};

use concinnity_core::blob::{
    CacheEntry, CacheEntryKind, CacheMeta, HEADER_SIZE, encode_cnb_prefix, parse_cnb,
    parse_payload_section_start,
};

// Where one entry's bytes sit, relative to the payload section.
#[derive(Clone, Copy)]
struct Span {
    offset: u64,
    len: u64,
}

/// A build segment's index, and the open file those offsets address.
///
/// Immutable once read, so a parallel compile shares one of these: a lookup is
/// a hash lookup plus a read of the one entry it wants. The handle carries a
/// cursor the readers have to take turns on, which is the price of every span
/// coming from the file the offsets were read from.
pub(super) struct Index {
    file: Option<Mutex<File>>,
    payload_start: u64,
    entries: HashMap<(CacheEntryKind, String), Span>,
}

impl Index {
    /// Read `path`'s header and index, holding the file open. An absent,
    /// unreadable, foreign, or differently stamped file reads as an empty
    /// index: whatever it held is recompiled, and the next write replaces it.
    pub(super) fn read(path: &Path, token: u32) -> Self {
        read_index(path, token).unwrap_or_else(Self::empty)
    }

    fn empty() -> Self {
        Self {
            file: None,
            payload_start: 0,
            entries: HashMap::new(),
        }
    }

    /// The bytes stored for `key`, read from the file. `None` when the segment
    /// holds no such entry, or when the read fails.
    pub(super) fn get(&self, kind: CacheEntryKind, key: &str) -> Option<Vec<u8>> {
        let span = *self.entries.get(&(kind, key.to_owned()))?;
        let mut file = self.handle()?;
        read_span(&mut file, self.payload_start, span).ok()
    }

    // The indexed file, for the length of one read. A poisoned lock is taken
    // anyway: a panic mid-read leaves a cursor, not a corrupt file, and every
    // read seeks before it starts.
    fn handle(&self) -> Option<MutexGuard<'_, File>> {
        Some(
            self.file
                .as_ref()?
                .lock()
                .unwrap_or_else(PoisonError::into_inner),
        )
    }

    /// Whether the segment holds an entry under `key`, without reading it.
    pub(super) fn contains(&self, kind: CacheEntryKind, key: &str) -> bool {
        self.entries.contains_key(&(kind, key.to_owned()))
    }

    /// How many entries the file holds.
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }
}

// A segment's header and index, or `None` for anything this build cannot read.
fn read_index(path: &Path, token: u32) -> Option<Index> {
    let mut file = File::open(path).ok()?;
    let size = file.metadata().ok()?.len();

    let mut header = [0u8; HEADER_SIZE];
    file.read_exact(&mut header).ok()?;
    let payload_start = parse_payload_section_start::<CacheMeta>(&header).ok()?;
    // A header claiming more index than the file holds is truncated or
    // corrupt; refusing it here is what keeps the read below from allocating
    // whatever length those bytes happened to spell.
    if payload_start > size {
        return None;
    }

    let mut prefix = vec![0u8; usize::try_from(payload_start).ok()?];
    prefix[..HEADER_SIZE].copy_from_slice(&header);
    file.read_exact(&mut prefix[HEADER_SIZE..]).ok()?;
    let (meta, _) = parse_cnb::<CacheMeta>(token, &prefix).ok()?;

    Some(Index {
        file: Some(Mutex::new(file)),
        payload_start,
        entries: meta
            .entries
            .into_iter()
            .filter(|entry| entry.offset.saturating_add(entry.len) <= size - payload_start)
            .map(|entry| {
                (
                    (entry.kind, entry.key),
                    Span {
                        offset: entry.offset,
                        len: entry.len,
                    },
                )
            })
            .collect(),
    })
}

// One entry's bytes, seeking straight to it rather than reading what surrounds
// it.
fn read_span(file: &mut File, payload_start: u64, span: Span) -> io::Result<Vec<u8>> {
    file.seek(SeekFrom::Start(payload_start + span.offset))?;
    let mut bytes = vec![0u8; usize::try_from(span.len).map_err(io::Error::other)?];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

/// Replace `path` with `index`'s entries plus `stored`, the entries this build
/// produced. Reports whether the file was written.
///
/// An entry `stored` names replaces the one the index holds under that key;
/// everything else is carried through, since a cache the next build wants is
/// not only what this one touched. Carried bytes come from the file `index` was
/// read from, which need not be the one at `path` any more.
pub(super) fn write(
    path: &Path,
    index: &Index,
    stored: &[(CacheEntryKind, &str, &[u8])],
    token: u32,
) -> bool {
    let plan = plan(index, stored);
    let meta = CacheMeta {
        toolchain: String::new(),
        entries: plan.iter().map(|p| p.entry.clone()).collect(),
    };
    let Ok(prefix) = encode_cnb_prefix(token, &meta) else {
        return false;
    };
    let mut carried_from = index.handle();
    concinnity_host::store::atomic::replace(path, |out| {
        out.write_all(&prefix)?;
        for planned in &plan {
            match &planned.source {
                Source::Stored(bytes) => out.write_all(bytes)?,
                Source::Carried(span) => {
                    let file = carried_from
                        .as_mut()
                        .ok_or_else(|| io::Error::other("the indexed segment is not open"))?;
                    file.seek(SeekFrom::Start(index.payload_start + span.offset))?;
                    io::copy(&mut Read::by_ref(&mut **file).take(span.len), out)?;
                }
            }
        }
        Ok(())
    })
}

// Where one payload byte range in the new file comes from.
enum Source<'a> {
    // The segment being replaced, at a span in its payload section.
    Carried(Span),
    // This build, which holds the bytes already.
    Stored(&'a [u8]),
}

// One entry of the file about to be written: its index record and where its
// bytes come from.
struct Planned<'a> {
    entry: CacheEntry,
    source: Source<'a>,
}

fn plan<'a>(index: &Index, stored: &[(CacheEntryKind, &'a str, &'a [u8])]) -> Vec<Planned<'a>> {
    let replaced: HashSet<(CacheEntryKind, &str)> =
        stored.iter().map(|(kind, key, _)| (*kind, *key)).collect();
    let mut carried: Vec<(CacheEntryKind, &String, Span)> = index
        .entries
        .iter()
        .filter(|((kind, key), _)| !replaced.contains(&(*kind, key.as_str())))
        .map(|((kind, key), span)| (*kind, key, *span))
        .collect();
    // Sorted, so the file a build writes is a function of the entries it holds
    // rather than of hash iteration order.
    carried.sort_by(|a, b| (a.1, a.0 as u8).cmp(&(b.1, b.0 as u8)));

    let sources =
        carried
            .iter()
            .map(|(kind, key, span)| (*kind, key.as_str(), span.len, Source::Carried(*span)))
            .chain(stored.iter().map(|(kind, key, bytes)| {
                (*kind, *key, bytes.len() as u64, Source::Stored(bytes))
            }));

    let mut offset = 0u64;
    sources
        .map(|(kind, key, len, source)| {
            let entry = CacheEntry {
                kind,
                key: key.to_owned(),
                offset,
                len,
            };
            offset += len;
            Planned { entry, source }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    const PAYLOAD: CacheEntryKind = CacheEntryKind::Payload;
    const EXPANSION: CacheEntryKind = CacheEntryKind::Expansion;
    const TOKEN: u32 = 0xC0FFEE;

    // Any path will do: which file a build's segment is belongs to the state
    // tree that resolved it, not to this module.
    fn segment_path(dir: &tempfile::TempDir) -> PathBuf {
        dir.path().join("segment")
    }

    // Write `items` into a fresh segment, the shape every test below starts from.
    fn written(path: &Path, items: &[(CacheEntryKind, &str, &[u8])]) -> bool {
        let index = Index::read(path, TOKEN);
        write(path, &index, items, TOKEN)
    }

    #[test]
    fn an_entry_round_trips_through_a_segment() {
        let dir = tempfile::tempdir().unwrap();
        let path = segment_path(&dir);
        assert!(written(&path, &[(PAYLOAD, "cafe", &[1, 2, 3])]));

        let index = Index::read(&path, TOKEN);
        assert_eq!(index.get(PAYLOAD, "cafe"), Some(vec![1, 2, 3]));
        assert_eq!(index.get(PAYLOAD, "f00d"), None);
    }

    // The reason the index carries a kind: the two key spaces the build segment
    // holds are no longer separated by anything in the key itself.
    #[test]
    fn a_payload_and_an_expansion_may_share_one_key() {
        let dir = tempfile::tempdir().unwrap();
        let path = segment_path(&dir);
        written(
            &path,
            &[(PAYLOAD, "cafe", &[1, 2, 3]), (EXPANSION, "cafe", &[9])],
        );

        let index = Index::read(&path, TOKEN);
        assert_eq!(index.get(PAYLOAD, "cafe"), Some(vec![1, 2, 3]));
        assert_eq!(index.get(EXPANSION, "cafe"), Some(vec![9]));
    }

    // A build stores what it compiled and nothing else, so everything the
    // previous segment held has to survive the write that adds to it.
    #[test]
    fn a_later_build_carries_the_earlier_entries_through() {
        let dir = tempfile::tempdir().unwrap();
        let path = segment_path(&dir);
        written(&path, &[(PAYLOAD, "aa", &[1]), (PAYLOAD, "bb", &[2, 2])]);

        let first = Index::read(&path, TOKEN);
        assert!(write(&path, &first, &[(PAYLOAD, "cc", &[3; 300])], TOKEN));

        let second = Index::read(&path, TOKEN);
        assert_eq!(second.len(), 3);
        assert_eq!(second.get(PAYLOAD, "aa"), Some(vec![1]));
        assert_eq!(second.get(PAYLOAD, "bb"), Some(vec![2, 2]));
        assert_eq!(second.get(PAYLOAD, "cc"), Some(vec![3; 300]));
    }

    // A key stored again takes the new bytes rather than being written twice:
    // the index would otherwise carry two entries for it, and a lookup would
    // answer with whichever the map happened to keep.
    #[test]
    fn a_restored_key_replaces_what_the_segment_held() {
        let dir = tempfile::tempdir().unwrap();
        let path = segment_path(&dir);
        written(&path, &[(PAYLOAD, "aa", &[1]), (PAYLOAD, "bb", &[2])]);

        let first = Index::read(&path, TOKEN);
        write(&path, &first, &[(PAYLOAD, "aa", &[7, 7, 7])], TOKEN);

        let second = Index::read(&path, TOKEN);
        assert_eq!(second.len(), 2);
        assert_eq!(second.get(PAYLOAD, "aa"), Some(vec![7, 7, 7]));
        assert_eq!(second.get(PAYLOAD, "bb"), Some(vec![2]));
    }

    // Two builds holding the same entries must produce the same file, or a
    // rewrite churns bytes that did not change.
    #[test]
    fn one_set_of_entries_writes_one_image() {
        let dir = tempfile::tempdir().unwrap();
        let items: &[(CacheEntryKind, &str, &[u8])] = &[
            (PAYLOAD, "bb", &[2, 2]),
            (EXPANSION, "aa", &[1]),
            (PAYLOAD, "aa", &[3, 3, 3]),
        ];
        let one = segment_path(&dir);
        written(&one, items);
        // The carried entries reach the second write through a map, whose
        // iteration order is not the order they went in.
        let two = dir.path().join("second").join("1");
        write(&two, &Index::read(&one, TOKEN), &[], TOKEN);
        let three = dir.path().join("third").join("1");
        write(&three, &Index::read(&two, TOKEN), &[], TOKEN);

        assert_eq!(std::fs::read(&two).unwrap(), std::fs::read(&three).unwrap());
    }

    // An index is a set of offsets into one file. Another process replacing
    // that file moves every offset in it, so an index that then read by name
    // would answer with whichever entry now sits where its own used to -- the
    // right key over the wrong bytes, published as a cache hit.
    #[test]
    fn a_segment_replaced_under_a_held_index_still_reads_its_own_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = segment_path(&dir);
        written(&path, &[(PAYLOAD, "aa", &[1])]);

        // What one build holds while it works.
        let held = Index::read(&path, TOKEN);

        // What another writes in the meantime, putting a longer entry first.
        written(&path, &[(PAYLOAD, "zz", &[9; 500]), (PAYLOAD, "aa", &[1])]);

        assert_eq!(
            held.get(PAYLOAD, "aa"),
            Some(vec![1]),
            "a lookup reads what it indexed"
        );

        assert!(write(&path, &held, &[(PAYLOAD, "bb", &[2])], TOKEN));
        let after = Index::read(&path, TOKEN);
        assert_eq!(
            after.get(PAYLOAD, "aa"),
            Some(vec![1]),
            "a carried entry is its own bytes"
        );
        assert_eq!(after.get(PAYLOAD, "bb"), Some(vec![2]));
    }

    // Deleting the segment at any point costs recomputation and nothing else, so
    // an absent segment reads as an empty index and the next write recreates
    // the file and the directory under it.
    #[test]
    fn an_absent_segment_reads_empty_and_is_recreated() {
        let dir = tempfile::tempdir().unwrap();
        let path = segment_path(&dir);
        assert_eq!(Index::read(&path, TOKEN).get(PAYLOAD, "cafe"), None);

        written(&path, &[(PAYLOAD, "cafe", &[1])]);
        std::fs::remove_file(&path).unwrap();
        assert_eq!(Index::read(&path, TOKEN).get(PAYLOAD, "cafe"), None);
        written(&path, &[(PAYLOAD, "cafe", &[1])]);
        assert_eq!(
            Index::read(&path, TOKEN).get(PAYLOAD, "cafe"),
            Some(vec![1])
        );
    }

    // The invalidation the source hashes used to do: a segment another binary
    // wrote is dropped whole rather than replayed against code that moved.
    #[test]
    fn a_segment_of_another_binary_reads_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = segment_path(&dir);
        written(&path, &[(PAYLOAD, "cafe", &[1, 2, 3])]);

        assert_eq!(Index::read(&path, TOKEN + 1).get(PAYLOAD, "cafe"), None);
        assert_eq!(
            Index::read(&path, TOKEN).get(PAYLOAD, "cafe"),
            Some(vec![1, 2, 3])
        );

        // What the newer binary writes is its own segment, and the older one's
        // entries do not survive into it.
        write(
            &path,
            &Index::read(&path, TOKEN + 1),
            &[(PAYLOAD, "f00d", &[4])],
            TOKEN + 1,
        );
        let reread = Index::read(&path, TOKEN + 1);
        assert_eq!(reread.len(), 1);
        assert_eq!(reread.get(PAYLOAD, "f00d"), Some(vec![4]));
    }

    // Garbage in the segment's place (a world blob, a half-written file, a
    // directory of the old layout) costs a recompile, never a failure.
    #[test]
    fn a_corrupt_segment_reads_empty_and_is_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let path = segment_path(&dir);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"not a segment at all").unwrap();
        assert_eq!(Index::read(&path, TOKEN).get(PAYLOAD, "cafe"), None);

        written(&path, &[(PAYLOAD, "cafe", &[7])]);
        assert_eq!(
            Index::read(&path, TOKEN).get(PAYLOAD, "cafe"),
            Some(vec![7])
        );
    }

    // An index can outlive the payload it addresses if something truncated the
    // file outside the rename. Such an entry is dropped rather than read, which
    // is what keeps a short payload from being carried into the next segment as
    // if it were whole.
    #[test]
    fn an_entry_pointing_past_the_file_is_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let path = segment_path(&dir);
        written(&path, &[(PAYLOAD, "aa", &[1]), (PAYLOAD, "bb", &[2; 64])]);
        let image = std::fs::read(&path).unwrap();
        std::fs::write(&path, &image[..image.len() - 8]).unwrap();

        let index = Index::read(&path, TOKEN);
        assert_eq!(index.get(PAYLOAD, "aa"), Some(vec![1]));
        assert_eq!(index.get(PAYLOAD, "bb"), None);
        assert_eq!(index.len(), 1);
    }

    // The build writes its own segment and nothing else: the runtime's file
    // beside it is neither read nor disturbed.
    #[test]
    fn the_runtime_segment_is_left_alone() {
        let dir = tempfile::tempdir().unwrap();
        let path = segment_path(&dir);
        let sibling = path.parent().unwrap().join("0");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&sibling, b"runtime segment").unwrap();

        written(&path, &[(PAYLOAD, "cafe", &[1])]);
        assert_eq!(std::fs::read(&sibling).unwrap(), b"runtime segment");
    }

    // A payload big enough to exercise the streamed copy, so a carried entry
    // that spans more than one buffer still lands whole.
    #[test]
    fn a_large_entry_survives_being_carried_through() {
        let dir = tempfile::tempdir().unwrap();
        let path = segment_path(&dir);
        let big: Vec<u8> = (0..512 * 1024).map(|i| (i % 251) as u8).collect();
        written(&path, &[(PAYLOAD, "big", &big), (PAYLOAD, "small", &[9])]);

        let first = Index::read(&path, TOKEN);
        write(&path, &first, &[(PAYLOAD, "later", &[8])], TOKEN);

        let second = Index::read(&path, TOKEN);
        assert_eq!(second.get(PAYLOAD, "big"), Some(big));
        assert_eq!(second.get(PAYLOAD, "small"), Some(vec![9]));
        assert_eq!(second.get(PAYLOAD, "later"), Some(vec![8]));
    }
}
