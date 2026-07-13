// Dense u32 asset identity.
//
// Asset names declared in a world are resolved to an `AssetId` at build time
// (via the installed name resolver, see `resolver`); the compiled blob and the
// runtime carry only the integer, so every cross-reference lookup is an integer
// compare, not a string compare.
//
// `AssetId` serializes as a bare `u32` in every format. It deserializes from a
// `u32` in non-self-describing formats (postcard, the blob defs table) and from
// either an integer or a name string in human-readable formats (JSON, the
// compiled `args_bytes`): an integer is an already-resolved id, a name string is
// run through the installed resolver.

use core::fmt;

use alloc::format;
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::resolver::resolve_name;

/// A dense integer handle for one asset, assigned at build time in world
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

struct AssetIdVisitor;

impl Visitor<'_> for AssetIdVisitor {
    type Value = AssetId;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("an asset id integer or a name string")
    }

    fn visit_u64<E: de::Error>(self, v: u64) -> Result<AssetId, E> {
        Ok(AssetId(v as u32))
    }
    fn visit_i64<E: de::Error>(self, v: i64) -> Result<AssetId, E> {
        Ok(AssetId(v as u32))
    }
    fn visit_str<E: de::Error>(self, v: &str) -> Result<AssetId, E> {
        resolve_name(v).map(AssetId).ok_or_else(|| {
            E::custom(format!(
                "no asset-name resolver installed to resolve reference {v:?}"
            ))
        })
    }
    fn visit_string<E: de::Error>(self, v: alloc::string::String) -> Result<AssetId, E> {
        self.visit_str(&v)
    }
}

impl<'de> Deserialize<'de> for AssetId {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        if d.is_human_readable() {
            d.deserialize_any(AssetIdVisitor)
        } else {
            Ok(AssetId(u32::deserialize(d)?))
        }
    }
}

/// `serde` `deserialize_with` helper for an optional cross-reference field.
///
/// Accepts a name string (resolved), an integer id, an empty string, or null;
/// the latter two resolve to `None`. Apply with `#[serde(default,
/// deserialize_with = "concinnity_asset::de_opt_asset_ref")]` so a missing field
/// is also `None`.
pub fn de_opt_asset_ref<'de, D>(d: D) -> Result<Option<AssetId>, D::Error>
where
    D: Deserializer<'de>,
{
    // A non-self-describing format (postcard, the baked blob form) carries the
    // already-resolved id; names only appear in human-readable input.
    if !d.is_human_readable() {
        return Option::<AssetId>::deserialize(d);
    }

    struct OptVisitor;

    impl Visitor<'_> for OptVisitor {
        type Value = Option<AssetId>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("an asset reference name string, id integer, or null")
        }

        fn visit_unit<E: de::Error>(self) -> Result<Option<AssetId>, E> {
            Ok(None)
        }
        fn visit_none<E: de::Error>(self) -> Result<Option<AssetId>, E> {
            Ok(None)
        }
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<Option<AssetId>, E> {
            Ok(Some(AssetId(v as u32)))
        }
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<Option<AssetId>, E> {
            Ok(Some(AssetId(v as u32)))
        }
        fn visit_str<E: de::Error>(self, v: &str) -> Result<Option<AssetId>, E> {
            if v.is_empty() {
                Ok(None)
            } else {
                AssetIdVisitor.visit_str(v).map(Some)
            }
        }
        fn visit_string<E: de::Error>(
            self,
            v: alloc::string::String,
        ) -> Result<Option<AssetId>, E> {
            self.visit_str(&v)
        }
    }

    d.deserialize_any(OptVisitor)
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
    fn deserializes_from_an_already_resolved_integer() {
        // The compiled-args / runtime path: refs are ints, no resolver needed.
        let id: AssetId = serde_json::from_str("5").unwrap();
        assert_eq!(id, AssetId(5));
    }

    #[test]
    fn round_trips_through_postcard() {
        // postcard is the blob defs-table format (BlobAssetDef.name is an AssetId).
        let bytes = postcard::to_allocvec(&AssetId(1234)).unwrap();
        let back: AssetId = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back, AssetId(1234));
    }

    #[test]
    fn opt_ref_treats_empty_null_and_missing_as_none() {
        #[derive(serde::Deserialize)]
        struct Holder {
            #[serde(default, deserialize_with = "de_opt_asset_ref")]
            r: Option<AssetId>,
        }
        assert!(
            serde_json::from_str::<Holder>("{\"r\":\"\"}")
                .unwrap()
                .r
                .is_none()
        );
        assert!(
            serde_json::from_str::<Holder>("{\"r\":null}")
                .unwrap()
                .r
                .is_none()
        );
        assert!(serde_json::from_str::<Holder>("{}").unwrap().r.is_none());
        assert_eq!(
            serde_json::from_str::<Holder>("{\"r\":5}").unwrap().r,
            Some(AssetId(5))
        );
    }

    #[test]
    fn opt_ref_round_trips_through_postcard() {
        // The baked blob form: not self-describing, carries the resolved id.
        #[derive(serde::Serialize, serde::Deserialize)]
        struct Holder {
            #[serde(default, deserialize_with = "de_opt_asset_ref")]
            r: Option<AssetId>,
            #[serde(default, deserialize_with = "de_opt_asset_ref")]
            none: Option<AssetId>,
        }
        let h = Holder {
            r: Some(AssetId(7)),
            none: None,
        };
        let bytes = postcard::to_allocvec(&h).unwrap();
        let back: Holder = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.r, Some(AssetId(7)));
        assert_eq!(back.none, None);
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
