// Length-delimited postcard decoding.
//
// Every decode the runtime performs on blob bytes reads a frame whose length
// the container already recorded: the header's `meta_len` for the metadata
// block, the length prefix serde_bytes writes on a def's `args_bytes`, the same
// on a resource record's `data_bytes`. postcard is positional and stops as soon
// as the target type is satisfied, so a frame holding more bytes than the type
// reads decodes without complaint. Requiring the frame to be consumed exactly
// turns a type that lost a field since the blob was written into a load error
// instead of a silent partial decode.

use serde::Deserialize;
use thiserror::Error;

/// Why a length-delimited postcard frame did not decode.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FrameError {
    /// postcard rejected the bytes.
    #[error("postcard decode failed: {0}")]
    Decode(postcard::Error),
    /// The value decoded without reaching the end of its frame, leaving this
    /// many bytes the type never read.
    #[error("frame has {0} trailing bytes")]
    Trailing(usize),
}

/// Decode `bytes` as a postcard frame of `T`, requiring `T` to consume every
/// byte of it.
pub fn decode_exact<'a, T: Deserialize<'a>>(bytes: &'a [u8]) -> Result<T, FrameError> {
    let (value, rest) = postcard::take_from_bytes(bytes).map_err(FrameError::Decode)?;
    match rest.len() {
        0 => Ok(value),
        n => Err(FrameError::Trailing(n)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use serde::Serialize;

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Two {
        a: f32,
        b: f32,
    }

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Three {
        a: f32,
        b: f32,
        c: f32,
    }

    fn encode<T: Serialize>(value: &T) -> Vec<u8> {
        postcard::to_allocvec(value).expect("serialize")
    }

    #[test]
    fn an_exact_frame_decodes() {
        let bytes = encode(&Two { a: 1.0, b: 2.0 });
        assert_eq!(decode_exact::<Two>(&bytes), Ok(Two { a: 1.0, b: 2.0 }));
    }

    // The case plain `from_bytes` accepts: the frame was written by a schema
    // with a field this build no longer has, so the tail goes unread.
    #[test]
    fn a_frame_with_a_dropped_field_is_rejected() {
        let bytes = encode(&Three {
            a: 1.0,
            b: 2.0,
            c: 3.0,
        });
        assert!(postcard::from_bytes::<Two>(&bytes).is_ok());
        assert_eq!(decode_exact::<Two>(&bytes), Err(FrameError::Trailing(4)));
    }

    // The other direction already fails inside postcard, since the frame ends
    // before the added field.
    #[test]
    fn a_frame_missing_an_added_field_is_rejected() {
        let bytes = encode(&Two { a: 1.0, b: 2.0 });
        assert_eq!(
            decode_exact::<Three>(&bytes),
            Err(FrameError::Decode(
                postcard::Error::DeserializeUnexpectedEnd
            ))
        );
    }

    #[test]
    fn an_empty_frame_is_rejected() {
        assert!(decode_exact::<Two>(&[]).is_err());
    }
}
