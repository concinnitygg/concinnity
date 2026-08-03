// src/build/payload.rs
//
// The header preamble every compiled payload shares: a fixed-size block of
// little-endian u32 fields opening with a magic tag. Each payload kind used to
// repeat the length guard, the magic compare, and a run of hand-counted byte
// offsets; they read the header through `HeaderReader` instead, so adding a
// field never means recounting slice indices.

// Sequential reader over a payload's fixed-size u32 header.
#[derive(Debug)]
pub struct HeaderReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> HeaderReader<'a> {
    // Check the payload is long enough for its header and opens with `magic`,
    // then position the reader just past the magic. `label` names the payload
    // kind in the error message.
    pub fn open(
        bytes: &'a [u8],
        magic: u32,
        header_bytes: usize,
        label: &str,
    ) -> Result<Self, String> {
        if bytes.len() < header_bytes {
            return Err(format!(
                "{} payload too short: {} bytes (need at least {} for header)",
                label,
                bytes.len(),
                header_bytes
            ));
        }
        let found = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        if found != magic {
            return Err(format!(
                "{label} payload magic 0x{found:08x} does not match expected 0x{magic:08x}"
            ));
        }
        Ok(Self { bytes, pos: 4 })
    }

    // The next u32 field. The caller has already proved the header is long
    // enough, so this cannot run off the end when read `header_bytes / 4 - 1`
    // times.
    pub fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.bytes[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        v
    }

    // Byte offset just past the fields read so far, where the payload body
    // begins.
    pub fn offset(&self) -> usize {
        self.pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAGIC: u32 = 0x1234_5678;

    fn payload(fields: &[u32]) -> Vec<u8> {
        let mut buf = MAGIC.to_le_bytes().to_vec();
        for f in fields {
            buf.extend_from_slice(&f.to_le_bytes());
        }
        buf
    }

    #[test]
    fn reads_fields_in_order_and_tracks_the_body_offset() {
        let bytes = payload(&[7, 9]);
        let mut h = HeaderReader::open(&bytes, MAGIC, 12, "test").unwrap();
        assert_eq!(h.u32(), 7);
        assert_eq!(h.u32(), 9);
        assert_eq!(h.offset(), 12);
    }

    #[test]
    fn rejects_a_short_payload() {
        let err = HeaderReader::open(&[0u8; 6], MAGIC, 12, "test").unwrap_err();
        assert!(err.contains("too short"), "{err}");
    }

    #[test]
    fn rejects_a_wrong_magic() {
        let mut bytes = payload(&[0, 0]);
        bytes[0] ^= 0xff;
        let err = HeaderReader::open(&bytes, MAGIC, 12, "test").unwrap_err();
        assert!(err.contains("magic"), "{err}");
    }
}
