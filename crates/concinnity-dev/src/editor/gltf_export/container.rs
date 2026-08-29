// src/editor/gltf_export/container.rs
//
// The GLB container: 12-byte header, then a JSON chunk and a BIN chunk, each
// padded to a 4-byte boundary as the glTF 2.0 spec requires (JSON with spaces,
// BIN with zeros).

// Wrap serialised glTF JSON and its binary buffer into a GLB byte stream.
pub(crate) fn wrap_glb(mut json: Vec<u8>, mut bin: Vec<u8>) -> Vec<u8> {
    while !json.len().is_multiple_of(4) {
        json.push(b' ');
    }
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }
    let total = 12 + 8 + json.len() + if bin.is_empty() { 0 } else { 8 + bin.len() };
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&json);
    if !bin.is_empty() {
        out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        out.extend_from_slice(b"BIN\0");
        out.extend_from_slice(&bin);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u32_at(bytes: &[u8], at: usize) -> u32 {
        u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
    }

    #[test]
    fn chunks_are_padded_to_four_bytes_and_the_header_totals_match() {
        let glb = wrap_glb(b"{}".to_vec(), vec![1, 2, 3]);
        assert_eq!(&glb[0..4], b"glTF");
        assert_eq!(u32_at(&glb, 4), 2);
        assert_eq!(u32_at(&glb, 8) as usize, glb.len());
        // JSON chunk: length 4 (padded with spaces), type "JSON".
        assert_eq!(u32_at(&glb, 12), 4);
        assert_eq!(&glb[16..20], b"JSON");
        assert_eq!(&glb[20..24], b"{}  ");
        // BIN chunk: length 4 (padded with a zero), type "BIN\0".
        assert_eq!(u32_at(&glb, 24), 4);
        assert_eq!(&glb[28..32], b"BIN\0");
        assert_eq!(&glb[32..36], &[1, 2, 3, 0]);
        assert!(glb.len().is_multiple_of(4));
    }

    #[test]
    fn already_aligned_chunks_gain_no_padding() {
        let glb = wrap_glb(b"{  }".to_vec(), vec![0; 8]);
        assert_eq!(u32_at(&glb, 12), 4);
        assert_eq!(u32_at(&glb, 24), 8);
        assert_eq!(u32_at(&glb, 8) as usize, 12 + 8 + 4 + 8 + 8);
    }

    #[test]
    fn an_empty_binary_buffer_emits_no_bin_chunk() {
        let glb = wrap_glb(b"{}".to_vec(), Vec::new());
        assert_eq!(u32_at(&glb, 8) as usize, glb.len());
        assert_eq!(glb.len(), 12 + 8 + 4);
    }
}
