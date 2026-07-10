// Dense u32 asset identity.
//
// Asset names declared in a world are resolved to an `AssetId` at cook time
// (see `AssetRef`); the compiled blob and the runtime carry only the integer,
// so every cross-reference lookup is an integer compare, not a string compare.
//
// Unlike the reference type, `AssetId` is a plain dense id: it (de)serializes
// as a bare `u32` in every format and does no name interning. Name -> id
// resolution is an explicit cook pass over `AssetRef` values, not a side effect
// of deserialization.

use core::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A dense integer handle for one asset, assigned at cook time in world
/// declaration order. Equality and hashing are integer ops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct AssetId(pub u32);

impl fmt::Display for AssetId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

impl Serialize for AssetId {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u32(self.0)
    }
}

impl<'de> Deserialize<'de> for AssetId {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(AssetId(u32::deserialize(d)?))
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;

    use super::*;

    #[test]
    fn round_trips_through_json_as_a_bare_integer() {
        let bytes = serde_json::to_vec(&AssetId(7)).unwrap();
        assert_eq!(bytes, b"7");
        let back: AssetId = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back, AssetId(7));
    }

    #[test]
    fn round_trips_through_postcard() {
        // postcard is the blob defs-table format (BlobAssetDef.name is an AssetId).
        let bytes = postcard::to_allocvec(&AssetId(1234)).unwrap();
        let back: AssetId = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back, AssetId(1234));
    }

    #[test]
    fn display_formats_with_a_hash_prefix() {
        assert_eq!(AssetId(42).to_string(), "#42");
    }

    #[test]
    fn default_is_zero() {
        assert_eq!(AssetId::default(), AssetId(0));
    }
}
