//! Asset identity: names declared in world.jsonl are interned to an `AssetId` in
//! declaration order, and the blob and the runtime carry only the integer, so
//! every cross-reference lookup is an integer compare.
//!
//! The interner a name resolves through keeps a per-thread table and so belongs
//! to the std-linked crate above; concinnity-host owns it and installs it into
//! [`super::resolver`]. At runtime references are already integers, so the seam
//! is never consulted.

use core::fmt;

use alloc::format;
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::ecs::resolver::resolve_name;

/// A world has minted every asset id its reserved range holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "this world has minted all {} asset ids it reserves",
    AssetId::MINTED_CAPACITY
)]
pub struct AssetIdsExhausted;

/// A dense integer handle for one asset, assigned at build time in world
/// declaration order. Equality and hashing are integer ops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct AssetId(pub u32);

impl AssetId {
    /// How many ids the minted range holds, and so the most one world can mint.
    pub const MINTED_CAPACITY: u32 = 1 << 24;

    /// First id in the range a running world mints from.
    ///
    /// A build interns names from zero in declaration order, so reserving the
    /// top of the space lets a load-time pass name what it injects without
    /// consulting an interner the shipped runtime does not carry. A world would
    /// have to declare four billion assets to reach it.
    pub const MINTED_BASE: u32 = u32::MAX - Self::MINTED_CAPACITY + 1;

    /// Whether this id was minted by a running world rather than interned from
    /// a declared name.
    pub fn is_minted(self) -> bool {
        self.0 >= Self::MINTED_BASE
    }
}

/// Hands out ids in the range a running world mints from, in call order.
///
/// One per world, carried as a world resource: everything that mints -- a
/// data-entry method before start, the completion pass at start -- draws from
/// the same counter, so no two minted assets collide.
#[derive(Debug, Clone, Default)]
pub struct MintedIds {
    next: u32,
}

impl MintedIds {
    /// The next unused minted id.
    ///
    /// Errors once the world has minted [`AssetId::MINTED_CAPACITY`] of them:
    /// the range ends at [`u32::MAX`], so a counter past it would name a
    /// declared asset instead.
    pub fn next_id(&mut self) -> Result<AssetId, AssetIdsExhausted> {
        if self.next >= AssetId::MINTED_CAPACITY {
            return Err(AssetIdsExhausted);
        }
        let id = AssetId(AssetId::MINTED_BASE + self.next);
        self.next += 1;
        Ok(id)
    }
}

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

pub(super) struct AssetIdVisitor;

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
    fn a_truncated_baked_id_is_an_error_not_a_panic() {
        // A blob whose defs table is cut short must surface as a decode error:
        // the baked path reads the id straight through with no visitor to
        // fall back on.
        assert!(postcard::from_bytes::<AssetId>(&[]).is_err());
    }

    #[test]
    fn display_formats_with_a_hash_prefix() {
        assert_eq!(AssetId(42).to_string(), "#42");
    }

    #[test]
    fn default_is_zero() {
        assert_eq!(AssetId::default(), AssetId(0));
    }

    #[test]
    fn minting_counts_up_from_the_reserved_base() {
        let mut ids = MintedIds::default();
        assert_eq!(ids.next_id(), Ok(AssetId(AssetId::MINTED_BASE)));
        assert_eq!(ids.next_id(), Ok(AssetId(AssetId::MINTED_BASE + 1)));
        assert!(AssetId(AssetId::MINTED_BASE).is_minted());
        assert!(!AssetId(AssetId::MINTED_BASE - 1).is_minted());
    }

    #[test]
    fn the_range_ends_at_the_top_of_the_id_space() {
        // The last reserved slot mints the highest id there is, so a counter
        // past it would have to wrap into the interned range.
        let mut ids = MintedIds {
            next: AssetId::MINTED_CAPACITY - 1,
        };
        let last = ids.next_id().expect("the last reserved id");
        assert_eq!(last, AssetId(u32::MAX));
        assert!(last.is_minted());
    }

    #[test]
    fn minting_past_the_range_is_refused_every_time() {
        let mut ids = MintedIds {
            next: AssetId::MINTED_CAPACITY,
        };
        assert_eq!(ids.next_id(), Err(AssetIdsExhausted));
        assert_eq!(ids.next_id(), Err(AssetIdsExhausted));
    }

    #[test]
    fn deserializes_a_name_through_the_seam() {
        crate::test_support::install_resolvers();
        assert_eq!(
            serde_json::from_str::<AssetId>("\"floor\"").unwrap(),
            AssetId(5)
        );
        // An owned string, the form the serde_json::Value bridge hands over.
        assert_eq!(
            serde_json::from_value::<AssetId>(serde_json::json!("wall")).unwrap(),
            AssetId(4)
        );
    }

    #[test]
    fn deserializes_a_signed_integer_narrowed_to_id_width() {
        assert_eq!(
            serde_json::from_str::<AssetId>("-1").unwrap(),
            AssetId(u32::MAX)
        );
    }

    #[test]
    fn a_wrong_typed_id_names_what_it_accepts() {
        let err = serde_json::from_str::<AssetId>("true")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("an asset id integer or a name string"),
            "{err}"
        );
    }
}
