// src/decode/mod.rs
//
// Bounds-checked primitives for reading bytes the process did not produce:
// compiled payloads loaded off disk, and the artist-supplied image files the
// cook pipeline imports. Both are external input, so a truncated, corrupt, or
// hostile buffer has to surface as an error rather than a panic.
//
// Two failure modes matter here and neither is caught by ordinary slicing:
// running off the end of the buffer, and size arithmetic that overflows before
// it is ever compared against the buffer length. A `width * height * 4` that
// wraps produces a small product, passes the length check that follows it, and
// decodes from the wrong offsets. `ByteReader` covers the first, the `size`
// helpers cover the second.

pub mod reader;
pub mod size;

pub use reader::ByteReader;
pub use size::{checked_product, checked_sum};
