//! Bounds-checked decoding of bytes the process did not produce: compiled
//! payloads and blob records loaded off disk, and the artist-supplied files
//! the cook pipeline imports. Both are external input, so a truncated,
//! corrupt, or hostile buffer has to surface as an error rather than a panic.
//!
//! Three failure modes matter and none is caught by ordinary slicing: running
//! off the end of the buffer, size arithmetic that overflows before it is ever
//! compared against the buffer length, and a postcard frame that decodes
//! without consuming every byte its container recorded. `ByteReader` covers
//! the first, the `size` helpers the second, and `frame` the third.

mod frame;
mod reader;
mod size;

pub use frame::FrameError;
pub(crate) use frame::decode_exact;
pub use reader::{ByteReader, DecodeError};
pub use size::checked_product;
