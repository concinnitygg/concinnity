//! Sequential bounds-checked reader over a byte buffer. Every accessor returns
//! `Result`, so a decoder written against it cannot index past the end of its
//! input no matter what lengths the input declares. Reading a fixed-width
//! integer goes through `array`, which yields an owned `[u8; N]` and removes
//! the `try_into().unwrap()` that hand-rolled cursors need.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Why a payload did not decode, naming the payload kind (or, for an
/// overflow, what was being sized) it was reading.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DecodeError {
    /// The buffer is shorter than the payload's fixed header.
    #[error("{payload} payload too short: {len} bytes (need at least {need} for header)")]
    ShortHeader {
        /// The payload kind.
        payload: &'static str,
        /// The buffer's length.
        len: usize,
        /// The header's length.
        need: usize,
    },
    /// The payload does not open with its kind's magic number.
    #[error("{payload} payload magic 0x{found:08x} does not match expected 0x{expected:08x}")]
    BadMagic {
        /// The payload kind.
        payload: &'static str,
        /// The number found.
        found: u32,
        /// The number expected.
        expected: u32,
    },
    /// The bytes at the cursor are not the tag the format places there.
    #[error("{payload} tag {found:02x?} does not match expected {expected:02x?}")]
    BadTag {
        /// The payload kind.
        payload: &'static str,
        /// The bytes found.
        found: Vec<u8>,
        /// The tag expected.
        expected: Vec<u8>,
    },
    /// A read ran past the end of the buffer.
    #[error("unexpected end of {payload}: need {need} bytes at offset {offset}, have {len}")]
    Truncated {
        /// The payload kind.
        payload: &'static str,
        /// Bytes the read wanted.
        need: usize,
        /// Where the read started.
        offset: usize,
        /// The buffer's length.
        len: usize,
    },
    /// A declared length overflowed before it could be checked.
    #[error("{payload} length overflow reading {need} bytes at offset {offset}")]
    LengthOverflow {
        /// The payload kind.
        payload: &'static str,
        /// Bytes the read wanted.
        need: usize,
        /// Where the read started.
        offset: usize,
    },
    /// A seek named an offset past the end of the buffer.
    #[error("{payload} seek to offset {offset} past end of {len} bytes")]
    SeekPastEnd {
        /// The payload kind.
        payload: &'static str,
        /// The offset sought.
        offset: usize,
        /// The buffer's length.
        len: usize,
    },
    /// A size computed from declared dimensions overflowed.
    #[error("{what} size overflow in {factors:?}")]
    SizeOverflow {
        /// What was being sized.
        what: &'static str,
        /// The factors whose product overflowed.
        factors: Vec<usize>,
    },
    /// An offset computed from declared lengths overflowed.
    #[error("{what} offset overflow in {terms:?}")]
    OffsetOverflow {
        /// What was being placed.
        what: &'static str,
        /// The terms whose sum overflowed.
        terms: Vec<usize>,
    },
    /// The bytes read but do not describe a valid payload.
    #[error("{payload}: {message}")]
    Invalid {
        /// The payload kind.
        payload: &'static str,
        /// What is wrong with it.
        message: String,
    },
}

impl DecodeError {
    /// A payload whose bytes read but whose contents are not valid.
    pub fn invalid(payload: &'static str, message: impl Into<String>) -> Self {
        Self::Invalid {
            payload,
            message: message.into(),
        }
    }
}

/// A bridge for the decoders that still report failures as text (cook's
/// image codecs and color LUT reader, mesh reimport, the runtime's streaming
/// sources), so they can read through a typed reader with `?`.
impl From<DecodeError> for String {
    fn from(error: DecodeError) -> Self {
        error.to_string()
    }
}

/// Reader over `bytes`, tracking a cursor and the payload name used in errors.
#[derive(Debug, Clone)]
pub struct ByteReader<'a> {
    bytes: &'a [u8],
    pos: usize,
    label: &'static str,
}

impl<'a> ByteReader<'a> {
    /// The payload kind this reader names in its errors.
    pub fn label(&self) -> &'static str {
        self.label
    }

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
    ///
    /// # Errors
    ///
    /// Returns an error if `bytes` is shorter than `header_bytes` or does not
    /// open with `magic`.
    pub fn open_payload(
        bytes: &'a [u8],
        magic: u32,
        header_bytes: usize,
        label: &'static str,
    ) -> Result<Self, DecodeError> {
        if bytes.len() < header_bytes {
            return Err(DecodeError::ShortHeader {
                payload: label,
                len: bytes.len(),
                need: header_bytes,
            });
        }
        let mut r = Self::new(bytes, label);
        let found = r.u32()?;
        if found != magic {
            return Err(DecodeError::BadMagic {
                payload: label,
                found,
                expected: magic,
            });
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
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than `n` bytes remain.
    pub fn take(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        let end = self.pos.checked_add(n).ok_or(DecodeError::LengthOverflow {
            payload: self.label,
            need: n,
            offset: self.pos,
        })?;
        let out = self
            .bytes
            .get(self.pos..end)
            .ok_or(DecodeError::Truncated {
                payload: self.label,
                need: n,
                offset: self.pos,
                len: self.bytes.len(),
            })?;
        self.pos = end;
        Ok(out)
    }

    /// Consume exactly `N` bytes as an owned array, ready for `from_le_bytes`.
    pub(crate) fn array<const N: usize>(&mut self) -> Result<[u8; N], DecodeError> {
        let mut out = [0u8; N];
        out.copy_from_slice(self.take(N)?);
        Ok(out)
    }

    /// Read one byte.
    #[cfg(test)]
    pub(crate) fn u8(&mut self) -> Result<u8, DecodeError> {
        Ok(u8::from_le_bytes(self.array::<1>()?))
    }

    /// Read a little-endian `u16`.
    pub(crate) fn u16(&mut self) -> Result<u16, DecodeError> {
        Ok(u16::from_le_bytes(self.array::<2>()?))
    }

    /// Read a little-endian `u32`.
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than four bytes remain.
    pub fn u32(&mut self) -> Result<u32, DecodeError> {
        Ok(u32::from_le_bytes(self.array::<4>()?))
    }

    /// Read a little-endian `u64`.
    #[cfg(test)]
    pub(crate) fn u64(&mut self) -> Result<u64, DecodeError> {
        Ok(u64::from_le_bytes(self.array::<8>()?))
    }

    /// Read a little-endian `i32`.
    pub(crate) fn i32(&mut self) -> Result<i32, DecodeError> {
        Ok(i32::from_le_bytes(self.array::<4>()?))
    }

    /// Read a little-endian `f32`.
    pub(crate) fn f32(&mut self) -> Result<f32, DecodeError> {
        Ok(f32::from_le_bytes(self.array::<4>()?))
    }

    /// Advance past `n` bytes without returning them.
    pub(crate) fn skip(&mut self, n: usize) -> Result<(), DecodeError> {
        self.take(n).map(|_| ())
    }

    /// Move the cursor to an absolute offset, which must lie within the buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if `pos` is past the end of the buffer.
    pub fn seek(&mut self, pos: usize) -> Result<(), DecodeError> {
        if pos > self.bytes.len() {
            return Err(DecodeError::SeekPastEnd {
                payload: self.label,
                offset: pos,
                len: self.bytes.len(),
            });
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

    /// Consume `tag`, or report that the bytes at the cursor are not it.
    ///
    /// # Errors
    ///
    /// Returns an error if the bytes at the cursor are not `tag`.
    pub fn expect_tag(&mut self, magic: &[u8]) -> Result<(), DecodeError> {
        let found = self.take(magic.len())?;
        if found != magic {
            return Err(DecodeError::BadTag {
                payload: self.label,
                found: found.to_vec(),
                expected: magic.to_vec(),
            });
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
        assert!(
            err.to_string().contains("unexpected end of test"),
            "{}",
            err
        );
        assert!(err.to_string().contains("have 3"), "{}", err);
    }

    // A declared length near usize::MAX must not wrap the cursor arithmetic
    // into a range that looks in-bounds.
    #[test]
    fn take_length_overflow_errors() {
        let buf = [1u8, 2, 3, 4];
        let mut r = reader(&buf);
        r.skip(2).unwrap();
        let err = r.take(usize::MAX).unwrap_err();
        assert!(err.to_string().contains("length overflow"), "{}", err);
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
        r.expect_tag(b"CNB\0").unwrap();
        assert_eq!(r.position(), 4);
    }

    #[test]
    fn peek_past_end_is_false_not_a_panic() {
        let buf = [1u8, 2];
        let r = reader(&buf);
        assert!(!r.peek(b"CNB\0"));
    }

    #[test]
    fn expect_tag_reports_mismatch() {
        let buf = *b"XXXXrest";
        let mut r = reader(&buf);
        let err = r.expect_tag(b"CNB\0").unwrap_err();
        assert!(err.to_string().contains("does not match"), "{}", err);
    }

    #[test]
    fn expect_tag_on_short_buffer_errors() {
        let buf = *b"CN";
        let mut r = reader(&buf);
        assert!(r.expect_tag(b"CNB\0").is_err());
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
        assert!(err.to_string().contains("too short"), "{}", err);
    }

    #[test]
    fn open_payload_rejects_a_wrong_magic() {
        let bytes = tagged(&[7, 8]);
        let err = ByteReader::open_payload(&bytes, 0xDEAD_BEEF, 12, "test").unwrap_err();
        assert!(err.to_string().contains("magic"), "{}", err);
    }

    // The header length guard has to run before the magic read, so an empty
    // buffer reports the short header rather than an end-of-buffer error.
    #[test]
    fn open_payload_on_an_empty_buffer_reports_a_short_header() {
        let err = ByteReader::open_payload(&[], MAGIC, 12, "test").unwrap_err();
        assert!(err.to_string().contains("too short"), "{}", err);
    }

    #[test]
    fn empty_buffer_reads_error() {
        let mut r = reader(&[]);
        assert!(r.is_empty());
        assert_eq!(r.remainder(), &[] as &[u8]);
        assert!(r.u8().is_err());
    }

    // Each failure keeps its shape, so a caller can tell a truncated payload
    // from one that names the wrong kind.
    #[test]
    fn failures_carry_their_kind() {
        let mut r = reader(&[1, 2, 3]);
        assert_eq!(
            r.take(4),
            Err(DecodeError::Truncated {
                payload: "test",
                need: 4,
                offset: 0,
                len: 3,
            })
        );
        let err = ByteReader::open_payload(&[0; 12], 0xDEAD_BEEF, 12, "test").unwrap_err();
        assert!(
            matches!(
                err,
                DecodeError::BadMagic {
                    payload: "test",
                    ..
                }
            ),
            "{err}"
        );
    }
}
