//! Binary glTF containers, and the buffer bytes that go inside one.
//!
//! The JSON chunk is taken as text rather than a parsed document, so the
//! harness stays out of the business of which JSON library a caller uses.

/// Assemble a GLB: the 12-byte header, the JSON chunk, and an optional BIN
/// chunk. Both chunks are padded to the 4-byte alignment the format requires.
///
/// ```
/// # use concinnity_testing::fixtures::glb;
/// let bytes = glb::container(r#"{"asset":{"version":"2.0"}}"#, None);
/// assert_eq!(&bytes[..4], b"glTF");
/// ```
pub fn container(json: &str, bin: Option<&[u8]>) -> Vec<u8> {
    let mut json_bytes = json.as_bytes().to_vec();
    while !json_bytes.len().is_multiple_of(4) {
        json_bytes.push(b' ');
    }

    let bin_bytes = bin.map(|b| {
        let mut b = b.to_vec();
        while !b.len().is_multiple_of(4) {
            b.push(0);
        }
        b
    });

    let total = 12 + 8 + json_bytes.len() + bin_bytes.as_ref().map_or(0, |b| 8 + b.len());
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&json_bytes);
    if let Some(b) = &bin_bytes {
        out.extend_from_slice(&(b.len() as u32).to_le_bytes());
        out.extend_from_slice(b"BIN\0");
        out.extend_from_slice(b);
    }
    out
}

/// `values` as little-endian `f32` bytes, for a buffer chunk.
pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// `values` as little-endian `u16` bytes, for an index chunk.
pub fn u16_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u32_at(bytes: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(
            bytes[offset..offset + 4]
                .try_into()
                .expect("four bytes at the offset"),
        )
    }

    #[test]
    fn a_json_only_container_declares_its_own_length() {
        let bytes = container(r#"{"a":1}"#, None);

        assert_eq!(&bytes[..4], b"glTF");
        assert_eq!(u32_at(&bytes, 4), 2, "glTF 2.0");
        assert_eq!(u32_at(&bytes, 8) as usize, bytes.len());
        assert_eq!(&bytes[16..20], b"JSON");
        // 7 bytes of JSON pad up to 8.
        assert_eq!(u32_at(&bytes, 12), 8);
    }

    #[test]
    fn a_bin_chunk_follows_the_padded_json() {
        let bytes = container(r#"{"a":1}"#, Some(&[1, 2, 3]));
        let bin_start = 12 + 8 + 8;

        assert_eq!(u32_at(&bytes, 8) as usize, bytes.len());
        assert_eq!(u32_at(&bytes, bin_start), 4, "3 bytes pad up to 4");
        assert_eq!(&bytes[bin_start + 4..bin_start + 8], b"BIN\0");
        assert_eq!(&bytes[bin_start + 8..], &[1, 2, 3, 0]);
    }

    #[test]
    fn the_numeric_helpers_write_little_endian() {
        assert_eq!(f32_bytes(&[1.0]), 1.0f32.to_le_bytes());
        assert_eq!(u16_bytes(&[1, 2]), [1, 0, 2, 0]);
    }
}
