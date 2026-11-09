// One edit page as the blob carries it: its storage, compressed.
//
// Before compression a page is a flag byte (bit 0: it stores detail
// weights), its offset and scale as little-endian `f32`s, its height steps as
// little-endian `i16`s, and its weights if stored.

use alloc::vec;
use alloc::vec::Vec;

use super::page::{EditPage, PAGE_SAMPLES};

const FLAG_W: u8 = 1;
// The flag, the offset and scale, then the height steps.
const E_LEN: usize = 1 + 8 + PAGE_SAMPLES * 2;

/// Why a compiled page does not read.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PagePayloadError {
    /// The bytes do not decompress.
    #[error("an edit page's payload does not decompress")]
    Compressed,
    /// The bytes decompress to something other than a page.
    #[error("an edit page's payload holds {0} bytes, not a page")]
    Size(usize),
}

/// `page` compressed for the blob.
pub fn encode_page(page: &EditPage) -> Vec<u8> {
    let (offset, scale, e, w) = page.parts();
    let mut raw = Vec::with_capacity(E_LEN + PAGE_SAMPLES);
    raw.push(if w.is_some() { FLAG_W } else { 0 });
    raw.extend_from_slice(&offset.to_le_bytes());
    raw.extend_from_slice(&scale.to_le_bytes());
    for q in e {
        raw.extend_from_slice(&q.to_le_bytes());
    }
    if let Some(w) = w {
        raw.extend_from_slice(w);
    }
    ruzstd::encoding::compress_to_vec(raw.as_slice(), ruzstd::encoding::CompressionLevel::Fastest)
}

/// The page `bytes` hold.
///
/// # Errors
///
/// Returns why the bytes do not hold a page.
pub fn decode_page(bytes: &[u8]) -> Result<EditPage, PagePayloadError> {
    let mut raw = vec![0u8; E_LEN + PAGE_SAMPLES + 1];
    let len = ruzstd::decoding::FrameDecoder::new()
        .decode_all(bytes, &mut raw)
        .map_err(|_| PagePayloadError::Compressed)?;
    let raw = &raw[..len];
    let has_w = raw.first().is_some_and(|f| f & FLAG_W != 0);
    if len != if has_w { E_LEN + PAGE_SAMPLES } else { E_LEN } {
        return Err(PagePayloadError::Size(len));
    }
    let float = |at: usize| f32::from_le_bytes([raw[at], raw[at + 1], raw[at + 2], raw[at + 3]]);
    let e = raw[9..E_LEN]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    let w = has_w.then(|| raw[E_LEN..].to_vec());
    EditPage::from_parts(float(1), float(5), e, w).ok_or(PagePayloadError::Size(len))
}

#[cfg(test)]
mod tests {
    use super::super::page::{PAGE_CELLS, PAGE_SIDE};
    use super::*;

    fn page(with_w: bool) -> EditPage {
        let n = PAGE_CELLS as f64;
        let e: Vec<f64> = (0..PAGE_SAMPLES)
            .map(|k| 4.0 * ((k % PAGE_SIDE) as f64 / n) - 0.5 * (k / PAGE_SIDE) as f64)
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

    #[test]
    fn a_page_round_trips() {
        for with_w in [false, true] {
            let p = page(with_w);
            let bytes = encode_page(&p);
            assert!(bytes.len() < E_LEN, "compressed: {}", bytes.len());
            assert_eq!(decode_page(&bytes).unwrap(), p);
        }
        let flat = EditPage::neutral();
        assert_eq!(decode_page(&encode_page(&flat)).unwrap(), flat);
    }

    #[test]
    fn bytes_that_hold_no_page_are_refused() {
        assert_eq!(
            decode_page(b"not a page"),
            Err(PagePayloadError::Compressed)
        );
        let short = ruzstd::encoding::compress_to_vec(
            [0u8; 20].as_slice(),
            ruzstd::encoding::CompressionLevel::Fastest,
        );
        assert_eq!(decode_page(&short), Err(PagePayloadError::Size(20)));
        let mut flagged = vec![FLAG_W];
        flagged.extend_from_slice(&[0u8; E_LEN - 1]);
        let missing_w = ruzstd::encoding::compress_to_vec(
            flagged.as_slice(),
            ruzstd::encoding::CompressionLevel::Fastest,
        );
        assert_eq!(decode_page(&missing_w), Err(PagePayloadError::Size(E_LEN)));
    }
}
