// A planet's edits as one file: a header, an index of every page, and the
// pages themselves, each compressed on its own.
//
// Little-endian throughout. The header is the magic, the version and the page
// count; each index entry is the page's tile, whether it stores a detail
// weight, where its bytes lie in the file, and its figures. A page's bytes
// decompress to its offset and scale, its height steps, and its weights if
// stored.

use alloc::vec;
use alloc::vec::Vec;

use super::super::shape::CubeFace;
use super::super::tile::TileId;
use super::EditPages;
use super::page::{EditPage, PAGE_SAMPLES};

/// The bytes every edits file starts with.
pub const EDITS_MAGIC: [u8; 4] = *b"CNPE";

/// The edits file layout this engine reads.
pub const EDITS_VERSION: u32 = 1;

const HEADER_LEN: usize = 12;
const ENTRY_LEN: usize = 40;
const FLAG_W: u8 = 1;
// A page's offset and scale, then its height steps.
const E_LEN: usize = 8 + PAGE_SAMPLES * 2;

/// Why an edits file does not read.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EditFileError {
    /// The file is shorter than its header or index says it is.
    #[error("the edits file ends early")]
    Truncated,
    /// The file does not start with the edits magic.
    #[error("not an edits file: it does not start with \"CNPE\"")]
    Magic,
    /// The file is laid out in a version this engine does not read.
    #[error("edits file version {0} is not {EDITS_VERSION}")]
    Version(u32),
    /// An index entry names a tile no quadtree holds.
    #[error("edit page {0} names no tile: face {1}, level {2} (at most 20), ({3}, {4})")]
    Tile(usize, u8, u8, u32, u32),
    /// Two index entries name the same tile.
    #[error("edit page {0} repeats the tile of an earlier page")]
    Repeated(usize),
    /// A page's bytes lie outside the file.
    #[error("edit page {0} lies outside the file")]
    Range(usize),
    /// A page's bytes do not decompress, or decompress to the wrong size.
    #[error("edit page {0} does not decompress to a page")]
    Page(usize),
    /// A page's figures in the index are not the ones its samples give.
    #[error("edit page {0}'s figures in the index do not match its samples")]
    Figures(usize),
}

/// The file holding `pages`, each compressed on its own.
pub fn encode_edits(pages: &EditPages) -> Vec<u8> {
    let mut tiles: Vec<TileId> = pages.iter().map(|(t, _)| t).collect();
    tiles.sort_unstable();
    let bodies: Vec<Vec<u8>> = tiles
        .iter()
        .filter_map(|t| pages.get(*t))
        .map(|page| {
            ruzstd::encoding::compress_to_vec(
                page_bytes(page).as_slice(),
                ruzstd::encoding::CompressionLevel::Fastest,
            )
        })
        .collect();
    let mut out = Vec::with_capacity(
        HEADER_LEN + ENTRY_LEN * tiles.len() + bodies.iter().map(Vec::len).sum::<usize>(),
    );
    out.extend_from_slice(&EDITS_MAGIC);
    out.extend_from_slice(&EDITS_VERSION.to_le_bytes());
    out.extend_from_slice(&u32::try_from(tiles.len()).unwrap_or(u32::MAX).to_le_bytes());
    let mut offset = (HEADER_LEN + ENTRY_LEN * tiles.len()) as u64;
    for (tile, body) in tiles.iter().zip(&bodies) {
        let has_w = pages.get(*tile).is_some_and(|p| p.has_w());
        let figures = pages.get(*tile).map(|p| p.figures());
        out.extend_from_slice(&[tile.face.0, tile.level, if has_w { FLAG_W } else { 0 }, 0]);
        out.extend_from_slice(&tile.x.to_le_bytes());
        out.extend_from_slice(&tile.y.to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&u32::try_from(body.len()).unwrap_or(u32::MAX).to_le_bytes());
        for v in figures.map_or([0.0; 4], |f| [f.e_min, f.e_max, f.e_step, f.w_step]) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        offset += body.len() as u64;
    }
    for body in &bodies {
        out.extend_from_slice(body);
    }
    out
}

/// The pages of the edits file `bytes`, every one checked: its tile, its
/// place in the file, its size once decompressed, and its figures.
///
/// # Errors
///
/// Returns why the file does not read when any part of it fails a check.
pub fn decode_edits(bytes: &[u8]) -> Result<EditPages, EditFileError> {
    let header = bytes.get(..HEADER_LEN).ok_or(EditFileError::Truncated)?;
    if header[..4] != EDITS_MAGIC {
        return Err(EditFileError::Magic);
    }
    let version = u32_at(header, 4);
    if version != EDITS_VERSION {
        return Err(EditFileError::Version(version));
    }
    let count = usize::try_from(u32_at(header, 8)).map_err(|_| EditFileError::Truncated)?;
    let index_len = count
        .checked_mul(ENTRY_LEN)
        .ok_or(EditFileError::Truncated)?;
    let index = bytes
        .get(HEADER_LEN..HEADER_LEN + index_len)
        .ok_or(EditFileError::Truncated)?;
    let mut pages = EditPages::default();
    let mut buffer = vec![0u8; E_LEN + PAGE_SAMPLES + 1];
    let mut decoder = ruzstd::decoding::FrameDecoder::new();
    for (i, entry) in index.chunks_exact(ENTRY_LEN).enumerate() {
        let tile = TileId {
            face: CubeFace(entry[0]),
            level: entry[1],
            x: u32_at(entry, 4),
            y: u32_at(entry, 8),
        };
        if !tile.is_valid() {
            return Err(EditFileError::Tile(i, entry[0], entry[1], tile.x, tile.y));
        }
        if pages.get(tile).is_some() {
            return Err(EditFileError::Repeated(i));
        }
        let start = usize::try_from(u64_at(entry, 12)).map_err(|_| EditFileError::Range(i))?;
        let len = usize::try_from(u32_at(entry, 20)).map_err(|_| EditFileError::Range(i))?;
        let body = start
            .checked_add(len)
            .and_then(|end| bytes.get(start..end))
            .ok_or(EditFileError::Range(i))?;
        let has_w = entry[2] & FLAG_W != 0;
        let want = if has_w { E_LEN + PAGE_SAMPLES } else { E_LEN };
        let got = decoder
            .decode_all(body, &mut buffer)
            .map_err(|_| EditFileError::Page(i))?;
        if got != want {
            return Err(EditFileError::Page(i));
        }
        let page = page_from_bytes(&buffer[..want], has_w).ok_or(EditFileError::Page(i))?;
        let f = page.figures();
        let listed = [24, 28, 32, 36].map(|at| f32::from_le_bytes(array_at(entry, at)));
        if [f.e_min, f.e_max, f.e_step, f.w_step].map(f32::to_bits) != listed.map(f32::to_bits) {
            return Err(EditFileError::Figures(i));
        }
        pages.insert(tile, alloc::sync::Arc::new(page));
    }
    Ok(pages)
}

// A page's bytes before compression.
fn page_bytes(page: &EditPage) -> Vec<u8> {
    let (offset, scale, e, w) = page.parts();
    let mut out = Vec::with_capacity(E_LEN + PAGE_SAMPLES);
    out.extend_from_slice(&offset.to_le_bytes());
    out.extend_from_slice(&scale.to_le_bytes());
    for q in e {
        out.extend_from_slice(&q.to_le_bytes());
    }
    if let Some(w) = w {
        out.extend_from_slice(w);
    }
    out
}

// The page `bytes` hold, with weights when `has_w`.
fn page_from_bytes(bytes: &[u8], has_w: bool) -> Option<EditPage> {
    let offset = f32::from_le_bytes(bytes.get(0..4)?.try_into().ok()?);
    let scale = f32::from_le_bytes(bytes.get(4..8)?.try_into().ok()?);
    let e = bytes
        .get(8..E_LEN)?
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    let w = if has_w {
        Some(bytes.get(E_LEN..E_LEN + PAGE_SAMPLES)?.to_vec())
    } else {
        None
    };
    EditPage::from_parts(offset, scale, e, w)
}

fn array_at<const N: usize>(bytes: &[u8], at: usize) -> [u8; N] {
    core::array::from_fn(|k| bytes.get(at + k).copied().unwrap_or(0))
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(array_at(bytes, at))
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(array_at(bytes, at))
}

#[cfg(test)]
mod tests {
    use super::super::page::{PAGE_CELLS, PAGE_SIDE};
    use super::*;

    fn page(lift: f64, with_w: bool) -> EditPage {
        let n = PAGE_CELLS as f64;
        let e: Vec<f64> = (0..PAGE_SAMPLES)
            .map(|k| lift * ((k % PAGE_SIDE) as f64 / n) - 0.5 * (k / PAGE_SIDE) as f64)
            .collect();
        let w: Vec<f64> = (0..PAGE_SAMPLES)
            .map(|k| {
                if with_w {
                    (k % PAGE_SIDE) as f64 / n
                } else {
                    1.0
                }
            })
            .collect();
        EditPage::from_samples(&e, &w).unwrap()
    }

    fn tile(level: u8, x: u32, y: u32) -> TileId {
        TileId {
            face: CubeFace(4),
            level,
            x,
            y,
        }
    }

    fn pages() -> EditPages {
        EditPages::new([
            (tile(3, 2, 5), page(4.0, true)),
            (tile(9, 300, 100), page(-12.0, false)),
            (TileId::root(CubeFace(1)), page(0.25, true)),
        ])
    }

    #[test]
    fn a_file_round_trips_every_page() {
        let set = pages();
        let bytes = encode_edits(&set);
        assert_eq!(&bytes[..4], b"CNPE");
        let back = decode_edits(&bytes).unwrap();
        assert_eq!(back.len(), 3);
        for (t, p) in set.iter() {
            assert_eq!(back.get(t).map(|b| &**b), Some(&**p), "{t:?}");
            assert_eq!(back.figures(t), set.figures(t));
        }
        assert_eq!(back.lowest_change(), set.lowest_change());
        assert_eq!(back.highest_change(), set.highest_change());
        let empty = decode_edits(&encode_edits(&EditPages::default())).unwrap();
        assert!(empty.is_empty());
    }

    // Compressed, a smooth page takes much less room than its samples.
    #[test]
    fn pages_are_compressed() {
        let set = EditPages::new([(tile(3, 2, 5), page(4.0, false))]);
        let bytes = encode_edits(&set);
        assert!(
            bytes.len() < HEADER_LEN + ENTRY_LEN + E_LEN,
            "{}",
            bytes.len()
        );
    }

    #[test]
    fn a_bad_header_is_refused() {
        let bytes = encode_edits(&pages());
        assert_eq!(decode_edits(&bytes[..6]), Err(EditFileError::Truncated));
        let mut magic = bytes.clone();
        magic[0] = b'X';
        assert_eq!(decode_edits(&magic), Err(EditFileError::Magic));
        let mut version = bytes.clone();
        version[4] = 9;
        assert_eq!(decode_edits(&version), Err(EditFileError::Version(9)));
        let mut count = bytes;
        count[8..12].copy_from_slice(&0x0100_0000u32.to_le_bytes());
        assert_eq!(decode_edits(&count), Err(EditFileError::Truncated));
    }

    // Each index entry is checked: its tile, its range, its page and its
    // figures.
    #[test]
    fn a_bad_index_entry_is_refused() {
        let bytes = encode_edits(&pages());
        let entry = |k: usize| HEADER_LEN + ENTRY_LEN * k;
        let mut level = bytes.clone();
        level[entry(1) + 1] = 21;
        assert!(matches!(
            decode_edits(&level),
            Err(EditFileError::Tile(1, _, 21, _, _))
        ));
        let mut column = bytes.clone();
        column[entry(0) + 4..entry(0) + 8].copy_from_slice(&8u32.to_le_bytes());
        assert!(matches!(
            decode_edits(&column),
            Err(EditFileError::Tile(0, ..))
        ));
        let mut face = bytes.clone();
        face[entry(2)] = 6;
        assert!(matches!(
            decode_edits(&face),
            Err(EditFileError::Tile(2, 6, ..))
        ));
        let mut repeat = bytes.clone();
        let first: Vec<u8> = repeat[entry(0)..entry(0) + 12].to_vec();
        repeat[entry(1)..entry(1) + 12].copy_from_slice(&first);
        assert_eq!(decode_edits(&repeat), Err(EditFileError::Repeated(1)));
        let mut range = bytes.clone();
        range[entry(2) + 12..entry(2) + 20].copy_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(decode_edits(&range), Err(EditFileError::Range(2)));
        assert_eq!(
            decode_edits(&bytes[..bytes.len() - 1]),
            Err(EditFileError::Range(2))
        );
        let mut flag = bytes.clone();
        flag[entry(0) + 2] ^= FLAG_W;
        assert_eq!(decode_edits(&flag), Err(EditFileError::Page(0)));
        let mut figures = bytes.clone();
        figures[entry(1) + 28] ^= 1;
        assert_eq!(decode_edits(&figures), Err(EditFileError::Figures(1)));
        let mut body = bytes;
        let at = usize::try_from(u64_at(&body[entry(0)..], 12)).unwrap();
        body[at..at + 4].copy_from_slice(&[0; 4]);
        assert_eq!(decode_edits(&body), Err(EditFileError::Page(0)));
    }
}
