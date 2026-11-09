//! The 64-bit FNV-1a hash.

/// A 64-bit FNV-1a hash: cheap, stable across builds and platforms, and not
/// for anything adversarial. Behavior saves persist it, so its output never
/// changes.
///
/// # Examples
///
/// ```
/// use concinnity_core::fnv::Fnv1a;
///
/// let mut a = Fnv1a::new();
/// a.write(b"world");
/// let mut b = Fnv1a::new();
/// b.write(b"world");
/// assert_eq!(a.finish(), b.finish());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Fnv1a(u64);

impl Fnv1a {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    /// A hash of nothing yet.
    pub const fn new() -> Self {
        Self(Self::OFFSET_BASIS)
    }

    /// Fold `bytes` into the hash.
    pub fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(Self::PRIME);
        }
    }

    /// The hash of everything written so far.
    pub const fn finish(self) -> u64 {
        self.0
    }

    /// The hash of `bytes` alone.
    pub fn of(bytes: &[u8]) -> u64 {
        let mut hash = Self::new();
        hash.write(bytes);
        hash.finish()
    }
}

impl Default for Fnv1a {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The published FNV-1a 64 test vectors.
    #[test]
    fn matches_the_reference_vectors() {
        assert_eq!(Fnv1a::of(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(Fnv1a::of(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(Fnv1a::of(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn writes_in_pieces_hash_like_one_write() {
        let mut pieces = Fnv1a::new();
        pieces.write(b"foo");
        pieces.write(b"bar");
        assert_eq!(pieces.finish(), Fnv1a::of(b"foobar"));
    }
}
