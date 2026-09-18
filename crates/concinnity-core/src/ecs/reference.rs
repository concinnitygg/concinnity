//! A typed reference from one asset to another.
//!
//! A field of type [`Ref<T>`] names a separately declared asset. `T` is the
//! reference target -- an asset type or a category marker naming several -- so
//! what the field may point at lives in the type system, where the authoring
//! registry reads it, instead of in a hand-kept side table. The value is the
//! dense [`AssetId`] alone: four bytes, `Copy`, and never a name.
//!
//! Deserialization matches a bare [`AssetId`]: an integer is an already
//! resolved id (the compiled-args and baked forms), and a `$id` string goes
//! through the installed name resolver. [`de_opt_ref`] adds the optional form,
//! where an empty string or null is `None`.

use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::ecs::asset_id::{AssetId, AssetIdVisitor};

/// What a [`Ref<T>`] may point at: the registry names of the asset types that
/// satisfy it.
///
/// Every registered asset type is its own target. A category marker lists
/// several types; an empty list accepts any declared asset.
pub trait RefTarget {
    /// The registry names of the types a reference to `Self` may resolve to.
    const TYPES: &'static [&'static str];
}

/// A reference to any declared asset, whatever its type.
#[derive(Debug, Clone, Copy)]
pub struct AnyAsset;

impl RefTarget for AnyAsset {
    const TYPES: &'static [&'static str] = &[];
}

/// A reference to another asset whose target is `T`.
#[repr(transparent)]
pub struct Ref<T: RefTarget> {
    id: AssetId,
    // Variance-neutral tag: keeps `Ref<T>: Send + Sync` whatever `T` is.
    _target: PhantomData<fn() -> T>,
}

impl<T: RefTarget> Ref<T> {
    /// A reference to the asset with this id.
    pub const fn new(id: AssetId) -> Self {
        Self {
            id,
            _target: PhantomData,
        }
    }

    /// The referenced asset's id.
    pub const fn id(self) -> AssetId {
        self.id
    }
}

impl<T: RefTarget> From<AssetId> for Ref<T> {
    fn from(id: AssetId) -> Self {
        Self::new(id)
    }
}

impl<T: RefTarget> From<Ref<T>> for AssetId {
    fn from(r: Ref<T>) -> Self {
        r.id
    }
}

// Hand-written so no impl asks anything of `T` beyond being a target: the tag
// is phantom and never a value.
impl<T: RefTarget> Clone for Ref<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: RefTarget> Copy for Ref<T> {}

impl<T: RefTarget> PartialEq for Ref<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T: RefTarget> Eq for Ref<T> {}

impl<T: RefTarget> PartialEq<AssetId> for Ref<T> {
    fn eq(&self, other: &AssetId) -> bool {
        self.id == *other
    }
}

impl<T: RefTarget> PartialOrd for Ref<T> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: RefTarget> Ord for Ref<T> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

impl<T: RefTarget> Hash for Ref<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl<T: RefTarget> fmt::Debug for Ref<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ref({})", self.id)
    }
}

impl<T: RefTarget> fmt::Display for Ref<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.id.fmt(f)
    }
}

impl<T: RefTarget> Serialize for Ref<T> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.id.serialize(s)
    }
}

impl<'de, T: RefTarget> Deserialize<'de> for Ref<T> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        AssetId::deserialize(d).map(Self::new)
    }
}

/// `serde` `deserialize_with` helper for an optional [`Ref<T>`] field.
///
/// Accepts a `$id` string (resolved), an integer id, an empty string, or null;
/// the latter two are `None`. Apply with `#[serde(default, deserialize_with =
/// "de_opt_ref")]` so a missing field is also `None`.
pub fn de_opt_ref<'de, D, T>(d: D) -> Result<Option<Ref<T>>, D::Error>
where
    D: Deserializer<'de>,
    T: RefTarget,
{
    // A non-self-describing format (postcard, the baked blob form) carries the
    // already-resolved id; names only appear in human-readable input.
    if !d.is_human_readable() {
        return Option::<Ref<T>>::deserialize(d);
    }
    d.deserialize_any(OptVisitor).map(|id| id.map(Ref::new))
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
        AssetIdVisitor.visit_u64(v).map(Some)
    }
    fn visit_i64<E: de::Error>(self, v: i64) -> Result<Option<AssetId>, E> {
        AssetIdVisitor.visit_i64(v).map(Some)
    }
    fn visit_str<E: de::Error>(self, v: &str) -> Result<Option<AssetId>, E> {
        if v.is_empty() {
            Ok(None)
        } else {
            AssetIdVisitor.visit_str(v).map(Some)
        }
    }
    fn visit_string<E: de::Error>(self, v: alloc::string::String) -> Result<Option<AssetId>, E> {
        self.visit_str(&v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    struct Texture;

    impl RefTarget for Texture {
        const TYPES: &'static [&'static str] = &["Texture"];
    }

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    struct Holder {
        #[serde(default, deserialize_with = "de_opt_ref")]
        r: Option<Ref<Texture>>,
    }

    #[test]
    fn is_the_size_of_an_id_and_thread_safe_whatever_the_target() {
        fn assert_send_sync<U: Send + Sync>() {}
        assert_send_sync::<Ref<Texture>>();
        assert_eq!(size_of::<Ref<Texture>>(), size_of::<AssetId>());
        assert_eq!(
            size_of::<Option<Ref<Texture>>>(),
            size_of::<Option<AssetId>>()
        );
    }

    #[test]
    fn reads_a_resolved_id_and_writes_it_back_as_an_integer() {
        let r: Ref<Texture> = serde_json::from_str("5").unwrap();
        assert_eq!(r.id(), AssetId(5));
        assert_eq!(serde_json::to_string(&r).unwrap(), "5");
        let r: Ref<Texture> = serde_json::from_str("-1").unwrap();
        assert_eq!(r, AssetId(u32::MAX));
    }

    #[test]
    fn resolves_a_name_through_the_seam() {
        crate::test_support::install_resolvers();
        let r: Ref<Texture> = serde_json::from_str("\"floor\"").unwrap();
        assert_eq!(r.id(), AssetId(5));
        // An owned string, the form the serde_json::Value bridge hands over.
        let r: Ref<Texture> = serde_json::from_value(serde_json::json!("wall")).unwrap();
        assert_eq!(r.id(), AssetId(4));
    }

    #[test]
    fn an_optional_ref_treats_empty_null_and_missing_as_none() {
        crate::test_support::install_resolvers();
        let parse = |json: &str| serde_json::from_str::<Holder>(json).unwrap().r;
        assert_eq!(parse("{\"r\":\"\"}"), None);
        assert_eq!(parse("{\"r\":null}"), None);
        assert_eq!(parse("{}"), None);
        assert_eq!(parse("{\"r\":5}"), Some(Ref::new(AssetId(5))));
        assert_eq!(parse("{\"r\":-1}"), Some(Ref::new(AssetId(u32::MAX))));
        assert_eq!(parse("{\"r\":\"floor\"}"), Some(Ref::new(AssetId(5))));
        let owned = serde_json::from_value::<Holder>(serde_json::json!({"r": ""}));
        assert_eq!(owned.unwrap().r, None);
        let owned = serde_json::from_value::<Holder>(serde_json::json!({"r": "wall"}));
        assert_eq!(owned.unwrap().r, Some(Ref::new(AssetId(4))));
        // A `None` reported by an option-aware format, rather than a null unit.
        let none = de_opt_ref::<_, Texture>(crate::test_support::NoneDeserializer);
        assert_eq!(none.unwrap(), None);
    }

    #[test]
    fn round_trips_through_postcard_as_a_resolved_id() {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct Baked {
            plain: Ref<Texture>,
            #[serde(default, deserialize_with = "de_opt_ref")]
            opt: Option<Ref<Texture>>,
            #[serde(default, deserialize_with = "de_opt_ref")]
            none: Option<Ref<Texture>>,
            list: alloc::vec::Vec<Ref<Texture>>,
        }
        let baked = Baked {
            plain: Ref::new(AssetId(7)),
            opt: Some(Ref::new(AssetId(9))),
            none: None,
            list: alloc::vec![Ref::new(AssetId(1))],
        };
        let bytes = postcard::to_allocvec(&baked).unwrap();
        // The same bytes a bare id writes: the tag costs nothing on the wire.
        assert_eq!(
            bytes,
            postcard::to_allocvec(&(7u32, Some(9u32), None::<u32>, [1u32].as_slice())).unwrap()
        );
        let back: Baked = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.plain.id(), AssetId(7));
        assert_eq!(back.opt, Some(Ref::new(AssetId(9))));
        assert_eq!(back.none, None);
        assert_eq!(back.list, [Ref::new(AssetId(1))]);
        assert!(postcard::from_bytes::<Ref<Texture>>(&[]).is_err());
    }

    #[test]
    fn a_wrong_typed_reference_names_what_it_accepts() {
        let err = serde_json::from_str::<Ref<Texture>>("true")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("an asset id integer or a name string"),
            "{err}"
        );
        let err = serde_json::from_str::<Holder>("{\"r\":true}")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("an asset reference name string, id integer, or null"),
            "{err}"
        );
    }

    #[test]
    fn converts_to_and_from_an_id_and_orders_by_it() {
        let r = Ref::<Texture>::from(AssetId(3));
        assert_eq!(AssetId::from(r), AssetId(3));
        assert!(Ref::<Texture>::new(AssetId(2)) < r);
        assert_eq!(alloc::format!("{r:?}"), "Ref(#3)");
        assert_eq!(r.to_string(), "#3");
        assert!(AnyAsset::TYPES.is_empty());
    }
}
