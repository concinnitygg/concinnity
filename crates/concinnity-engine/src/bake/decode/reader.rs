//! Sequential bounds-checked reader over a byte buffer. Every accessor returns
//! `Result`, so a decoder written against it cannot index past the end of its
//! input no matter what lengths the input declares. Reading a fixed-width
//! integer goes through `array`, which yields an owned `[u8; N]` and removes
//! the `try_into().unwrap()` that hand-rolled cursors need.

use alloc::format;
use alloc::string::String;

/// Reader over `bytes`, tracking a cursor and the payload name used in errors.
#[derive(Debug, Clone)]
pub struct ByteReader<'a> {
    bytes: &'a [u8],
    pos: usize,
    label: &'static str,
}

impl<'a> ByteReader<'a> {
    /// A reader positioned at the start of `bytes`. `label` names the payload
    /// kind in every error this reader produces.
    pub fn new(bytes: &'a [u8], label: &'static str) -> Self {
        Self {
            bytes,
            pos: 0,
            label,
        }
    }

    /// Open a tagged payload: prove it is long enough to hold a `header_bytes`
    /// header and that it opens with `magic`, then position the reader just
    /// past the magic. Lets a decoder report a short buffer once rather than
    /// once per header field.
    pub fn open_payload(
        bytes: &'a [u8],
        magic: u32,
        header_bytes: usize,
        label: &'static str,
    ) -> Result<Self, String> {
        if bytes.len() < header_bytes {
            return Err(format!(
                "{} payload too short: {} bytes (need at least {} for header)",
                label,
                bytes.len(),
                header_bytes
            ));
        }
        let mut r = Self::new(bytes, label);
        let found = r.u32()?;
        if found != magic {
            return Err(format!(
                "{label} payload magic 0x{found:08x} does not match expected 0x{magic:08x}"
            ));
        }
        Ok(r)
    }

    /// Current byte offset.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Bytes left between the cursor and the end of the buffer.
    pub fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    /// Whether no bytes remain.
    pub fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    /// Total buffer length, including bytes already consumed.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Consume `n` bytes and return them, or report where the buffer ran out.
    pub fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.pos.checked_add(n).ok_or_else(|| {
            format!(
                "{} length overflow reading {} bytes at offset {}",
                self.label, n, self.pos
            )
        })?;
        let out = self.bytes.get(self.pos..end).ok_or_else(|| {
            format!(
                "unexpected end of {}: need {} bytes at offset {}, have {}",
                self.label,
                n,
                self.pos,
                self.bytes.len()
            )
        })?;
        self.pos = end;
        Ok(out)
    }

    /// Consume exactly `N` bytes as an owned array, ready for `from_le_bytes`.
    pub fn array<const N: usize>(&mut self) -> Result<[u8; N], String> {
        let mut out = [0u8; N];
        out.copy_from_slice(self.take(N)?);
        Ok(out)
    }

    /// Read one byte.
    pub fn u8(&mut self) -> Result<u8, String> {
        Ok(u8::from_le_bytes(self.array::<1>()?))
    }

    /// Read a little-endian `u16`.
    pub fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.array::<2>()?))
    }

    /// Read a little-endian `u32`.
    pub fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.array::<4>()?))
    }

    /// Read a little-endian `u64`.
    pub fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.array::<8>()?))
    }

    /// Read a little-endian `i32`.
    pub fn i32(&mut self) -> Result<i32, String> {
        Ok(i32::from_le_bytes(self.array::<4>()?))
    }

    /// Read a little-endian `f32`.
    pub fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.array::<4>()?))
    }

    /// Advance past `n` bytes without returning them.
    pub fn skip(&mut self, n: usize) -> Result<(), String> {
        self.take(n).map(|_| ())
    }

    /// Move the cursor to an absolute offset, which must lie within the buffer.
    pub fn seek(&mut self, pos: usize) -> Result<(), String> {
        if pos > self.bytes.len() {
            return Err(format!(
                "{} seek to offset {} past end of {} bytes",
                self.label,
                pos,
                self.bytes.len()
            ));
        }
        self.pos = pos;
        Ok(())
    }

    /// Whether the bytes at the cursor equal `magic`, without consuming them.
    pub fn peek(&self, magic: &[u8]) -> bool {
        self.pos
            .checked_add(magic.len())
            .and_then(|end| self.bytes.get(self.pos..end))
            .is_some_and(|b| b == magic)
    }

    // Consume `magic`, or report that the buffer does not start with it.
    #[cfg(test)]
    pub(crate) fn expect_magic(&mut self, magic: &[u8]) -> Result<(), String> {
        let found = self.take(magic.len())?;
        if found != magic {
            return Err(format!(
                "{} magic {:02x?} does not match expected {:02x?}",
                self.label, found, magic
            ));
        }
        Ok(())
    }

    /// Everything from the cursor to the end, leaving the cursor in place.
    pub fn remainder(&self) -> &'a [u8] {
        self.bytes.get(self.pos..).unwrap_or(&[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn reader(bytes: &[u8]) -> ByteReader<'_> {
        ByteReader::new(bytes, "test")
    }

    #[test]
    fn reads_fixed_width_integers_in_order() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&7u32.to_le_bytes());
        buf.extend_from_slice(&9u16.to_le_bytes());
        buf.extend_from_slice(&1.5f32.to_le_bytes());
        buf.extend_from_slice(&(-3i32).to_le_bytes());
        buf.extend_from_slice(&11u64.to_le_bytes());
        buf.push(200);

        let mut r = reader(&buf);
        assert_eq!(r.u32().unwrap(), 7);
        assert_eq!(r.u16().unwrap(), 9);
        assert_eq!(r.f32().unwrap(), 1.5);
        assert_eq!(r.i32().unwrap(), -3);
        assert_eq!(r.u64().unwrap(), 11);
        assert_eq!(r.u8().unwrap(), 200);
        assert!(r.is_empty());
    }

    #[test]
    fn take_advances_and_tracks_position() {
        let buf = [1u8, 2, 3, 4, 5];
        let mut r = reader(&buf);
        assert_eq!(r.take(2).unwrap(), &[1, 2]);
        assert_eq!(r.position(), 2);
        assert_eq!(r.remaining(), 3);
        assert_eq!(r.len(), 5);
    }

    #[test]
    fn take_past_end_errors_instead_of_panicking() {
        let buf = [1u8, 2, 3];
        let mut r = reader(&buf);
        let err = r.take(4).unwrap_err();
        assert!(err.contains("unexpected end of test"), "{}", err);
        assert!(err.contains("have 3"), "{}", err);
    }

    // A declared length near usize::MAX must not wrap the cursor arithmetic
    // into a range that looks in-bounds.
    #[test]
    fn take_length_overflow_errors() {
        let buf = [1u8, 2, 3, 4];
        let mut r = reader(&buf);
        r.skip(2).unwrap();
        let err = r.take(usize::MAX).unwrap_err();
        assert!(err.contains("length overflow"), "{}", err);
    }

    #[test]
    fn failed_take_leaves_cursor_untouched() {
        let buf = [1u8, 2, 3];
        let mut r = reader(&buf);
        r.skip(1).unwrap();
        assert!(r.take(99).is_err());
        assert_eq!(r.position(), 1);
        assert_eq!(r.u8().unwrap(), 2);
    }

    #[test]
    fn truncated_integer_read_errors() {
        let buf = [1u8, 2];
        let mut r = reader(&buf);
        assert!(r.u32().is_err());
    }

    #[test]
    fn seek_moves_cursor_and_rejects_past_end() {
        let buf = [1u8, 2, 3, 4];
        let mut r = reader(&buf);
        r.seek(3).unwrap();
        assert_eq!(r.u8().unwrap(), 4);
        r.seek(4).unwrap();
        assert!(r.is_empty());
        assert!(r.seek(5).is_err());
    }

    #[test]
    fn peek_does_not_consume() {
        let buf = *b"CNB\0rest";
        let mut r = reader(&buf);
        assert!(r.peek(b"CNB\0"));
        assert!(!r.peek(b"XXXX"));
        assert_eq!(r.position(), 0);
        r.expect_magic(b"CNB\0").unwrap();
        assert_eq!(r.position(), 4);
    }

    #[test]
    fn peek_past_end_is_false_not_a_panic() {
        let buf = [1u8, 2];
        let r = reader(&buf);
        assert!(!r.peek(b"CNB\0"));
    }

    #[test]
    fn expect_magic_reports_mismatch() {
        let buf = *b"XXXXrest";
        let mut r = reader(&buf);
        let err = r.expect_magic(b"CNB\0").unwrap_err();
        assert!(err.contains("does not match"), "{}", err);
    }

    #[test]
    fn expect_magic_on_short_buffer_errors() {
        let buf = *b"CN";
        let mut r = reader(&buf);
        assert!(r.expect_magic(b"CNB\0").is_err());
    }

    #[test]
    fn remainder_returns_unconsumed_tail() {
        let buf = [1u8, 2, 3, 4];
        let mut r = reader(&buf);
        r.skip(2).unwrap();
        assert_eq!(r.remainder(), &[3, 4]);
        assert_eq!(r.position(), 2);
    }

    const MAGIC: u32 = u32::from_le_bytes(*b"TEST");

    fn tagged(fields: &[u32]) -> Vec<u8> {
        let mut buf = MAGIC.to_le_bytes().to_vec();
        for f in fields {
            buf.extend_from_slice(&f.to_le_bytes());
        }
        buf
    }

    #[test]
    fn open_payload_positions_past_the_magic() {
        let bytes = tagged(&[7, 8]);
        let mut r = ByteReader::open_payload(&bytes, MAGIC, 12, "test").unwrap();
        assert_eq!(r.position(), 4);
        assert_eq!(r.u32().unwrap(), 7);
        assert_eq!(r.u32().unwrap(), 8);
    }

    #[test]
    fn open_payload_rejects_a_short_header() {
        let bytes = tagged(&[7]);
        let err = ByteReader::open_payload(&bytes, MAGIC, 12, "test").unwrap_err();
        assert!(err.contains("too short"), "{}", err);
    }

    #[test]
    fn open_payload_rejects_a_wrong_magic() {
        let bytes = tagged(&[7, 8]);
        let err = ByteReader::open_payload(&bytes, 0xDEAD_BEEF, 12, "test").unwrap_err();
        assert!(err.contains("magic"), "{}", err);
    }

    // The header length guard has to run before the magic read, so an empty
    // buffer reports the short header rather than an end-of-buffer error.
    #[test]
    fn open_payload_on_an_empty_buffer_reports_a_short_header() {
        let err = ByteReader::open_payload(&[], MAGIC, 12, "test").unwrap_err();
        assert!(err.contains("too short"), "{}", err);
    }

    #[test]
    fn empty_buffer_reads_error() {
        let mut r = reader(&[]);
        assert!(r.is_empty());
        assert_eq!(r.remainder(), &[] as &[u8]);
        assert!(r.u8().is_err());
    }
}
