//! How a pack's records divide into chunks.

use core::ops::Range;

/// Most chunks one pack is split into.
pub const MAX_PACK_CHUNKS: usize = 32;

/// Fewest records a chunk is given: below this a job's dispatch costs more than
/// the records it packs.
pub const MIN_CHUNK_RECORDS: usize = 1_024;

/// How many chunks to split `records` into across `workers`: one per worker,
/// fewer when there are too few records to give each one a full chunk, and
/// never fewer than one.
pub fn chunk_count(records: usize, workers: usize) -> usize {
    let most = workers.clamp(1, MAX_PACK_CHUNKS);
    (records / MIN_CHUNK_RECORDS).clamp(1, most)
}

/// Chunk `index`'s records when `len` records split into `count` chunks. The
/// chunks are contiguous, in order, cover every record once, and differ in
/// length by at most one.
pub fn chunk_range(len: usize, count: usize, index: usize) -> Range<usize> {
    let count = count.max(1);
    let (base, extra) = (len / count, len % count);
    let start = index * base + index.min(extra);
    let end = start + base + usize::from(index < extra);
    start.min(len)..end.min(len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_cover_every_record_once_in_order() {
        for len in [0, 1, 5, 31, 32, 33, 1_000, 4_097] {
            for count in 1..=MAX_PACK_CHUNKS {
                let mut next = 0;
                for index in 0..count {
                    let range = chunk_range(len, count, index);
                    assert_eq!(range.start, next, "len {len} count {count}");
                    assert!(range.len() <= len / count + 1);
                    assert!(range.len() >= len / count);
                    next = range.end;
                }
                assert_eq!(next, len, "len {len} count {count}");
            }
        }
    }

    #[test]
    fn the_chunk_count_follows_the_workers_and_the_record_floor() {
        assert_eq!(chunk_count(0, 8), 1);
        assert_eq!(chunk_count(MIN_CHUNK_RECORDS - 1, 8), 1);
        assert_eq!(chunk_count(3 * MIN_CHUNK_RECORDS, 8), 3);
        assert_eq!(chunk_count(100 * MIN_CHUNK_RECORDS, 8), 8);
        assert_eq!(chunk_count(100 * MIN_CHUNK_RECORDS, 0), 1);
        assert_eq!(chunk_count(1_000 * MIN_CHUNK_RECORDS, 500), MAX_PACK_CHUNKS);
    }
}
