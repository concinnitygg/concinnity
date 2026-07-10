// A typed reference from one asset to another, declared by name.
//
// A field of type `AssetRef<T>` points at a separately-declared asset. `T` is
// the reference target -- either a concrete asset type or a category marker
// (e.g. any mesh source) -- so the target kind lives in the type system instead
// of a hand-maintained side table. On the wire and at runtime the reference is
// a name (authoring) or a dense id (resolved); the referenced asset's data is
// never embedded.
//
// Lifecycle:
//   * Authoring: a world declares `{"clip":"footsteps"}`; deserialization keeps
//     the NAME and leaves the id unresolved.
//   * Cook: a resolution pass assigns each asset a dense `AssetId` in
//     declaration order and fills every `AssetRef`'s id from its name.
//   * Runtime: the compiled args carry the resolved integer id; deserialization
//     reads it directly, with no name lookup.
//
// Serialization matches the old bare-`AssetId` field exactly, so the blob format
// is unchanged: a resolved reference serializes as its integer id, an
// unresolved one as its name string.

use core::fmt;
use core::marker::PhantomData;

use alloc::boxed::Box;
use alloc::string::String;
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::AssetId;

/// A by-name reference to another asset of target `T`.
pub struct AssetRef<T> {
    // The referenced asset's declared name (authoring / unresolved). Empty once
    // a reference has only its resolved id (e.g. read back from compiled args).
    name: Box<str>,
    // The dense id, filled by the cook resolution pass. `None` until resolved.
    id: Option<AssetId>,
    // Variance-neutral tag: keeps `AssetRef<T>: Send + Sync` and imposes no
    // trait bounds on `T`.
    _target: PhantomData<fn() -> T>,
}

impl<T> AssetRef<T> {
    /// An unresolved reference to the asset declared as `name`.
    pub fn by_name(name: impl Into<Box<str>>) -> Self {
        Self {
            name: name.into(),
            id: None,
            _target: PhantomData,
        }
    }

    /// A reference already resolved to a dense id (no name retained).
    pub fn resolved(id: AssetId) -> Self {
        Self {
            name: Box::from(""),
            id: Some(id),
            _target: PhantomData,
        }
    }

    /// The declared name, or `""` if this reference carries only a resolved id.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The resolved dense id, or `None` before the cook resolution pass runs.
    pub fn id(&self) -> Option<AssetId> {
        self.id
    }

    /// Whether the cook resolution pass has filled in the id.
    pub fn is_resolved(&self) -> bool {
        self.id.is_some()
    }

    /// Fill in the resolved id (called by the cook resolution pass).
    pub fn resolve(&mut self, id: AssetId) {
        self.id = Some(id);
    }
}

// Manual auto-trait-friendly impls: no `T: Trait` bounds, since `T` is a phantom
// tag and never a value.
impl<T> Clone for AssetRef<T> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            id: self.id,
            _target: PhantomData,
        }
    }
}

impl<T> fmt::Debug for AssetRef<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AssetRef")
            .field("name", &self.name)
            .field("id", &self.id)
            .finish()
    }
}

impl<T> PartialEq for AssetRef<T> {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.id == other.id
    }
}

impl<T> Eq for AssetRef<T> {}

// An empty, unresolved reference. Lets a required `AssetRef<T>` field carry
// `#[serde(default)]` and a "not yet set" placeholder before authoring.
impl<T> Default for AssetRef<T> {
    fn default() -> Self {
        Self {
            name: Box::from(""),
            id: None,
            _target: PhantomData,
        }
    }
}

impl<T> Serialize for AssetRef<T> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        // A resolved reference serializes as its integer id (the compiled-args /
        // blob form); an unresolved one serializes as its name. This matches the
        // legacy bare-`AssetId` field, so compiled output is byte-identical.
        match self.id {
            Some(id) => s.serialize_u32(id.0),
            None => s.serialize_str(&self.name),
        }
    }
}

struct AssetRefVisitor<T>(PhantomData<fn() -> T>);

impl<'de, T> Visitor<'de> for AssetRefVisitor<T> {
    type Value = AssetRef<T>;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("an asset reference name string or a resolved id integer")
    }

    fn visit_str<E: de::Error>(self, v: &str) -> Result<AssetRef<T>, E> {
        Ok(AssetRef::by_name(v))
    }
    fn visit_string<E: de::Error>(self, v: String) -> Result<AssetRef<T>, E> {
        Ok(AssetRef::by_name(v.into_boxed_str()))
    }
    fn visit_u64<E: de::Error>(self, v: u64) -> Result<AssetRef<T>, E> {
        Ok(AssetRef::resolved(AssetId(v as u32)))
    }
    fn visit_i64<E: de::Error>(self, v: i64) -> Result<AssetRef<T>, E> {
        Ok(AssetRef::resolved(AssetId(v as u32)))
    }
}

impl<'de, T> Deserialize<'de> for AssetRef<T> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        // Asset args are always a human-readable format (JSON), both at cook
        // time (a name) and at runtime (a resolved id). A non-self-describing
        // format still deserializes from the integer id.
        if d.is_human_readable() {
            d.deserialize_any(AssetRefVisitor::<T>(PhantomData))
        } else {
            Ok(AssetRef::resolved(AssetId(u32::deserialize(d)?)))
        }
    }
}

/// `serde` `deserialize_with` helper for an optional reference field.
///
/// Accepts a name string, an integer id, an empty string, or null; the latter
/// two resolve to `None`. Apply with `#[serde(default, deserialize_with =
/// "concinnity_asset::de_opt_asset_ref")]` so a missing field is also `None`.
pub fn de_opt_asset_ref<'de, D, T>(d: D) -> Result<Option<AssetRef<T>>, D::Error>
where
    D: Deserializer<'de>,
{
    struct OptVisitor<T>(PhantomData<fn() -> T>);

    impl<'de, T> Visitor<'de> for OptVisitor<T> {
        type Value = Option<AssetRef<T>>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("an asset reference name string, id integer, or null")
        }

        fn visit_unit<E: de::Error>(self) -> Result<Option<AssetRef<T>>, E> {
            Ok(None)
        }
        fn visit_none<E: de::Error>(self) -> Result<Option<AssetRef<T>>, E> {
            Ok(None)
        }
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<Option<AssetRef<T>>, E> {
            Ok(Some(AssetRef::resolved(AssetId(v as u32))))
        }
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<Option<AssetRef<T>>, E> {
            Ok(Some(AssetRef::resolved(AssetId(v as u32))))
        }
        fn visit_str<E: de::Error>(self, v: &str) -> Result<Option<AssetRef<T>>, E> {
            if v.is_empty() {
                Ok(None)
            } else {
                Ok(Some(AssetRef::by_name(v)))
            }
        }
        fn visit_string<E: de::Error>(self, v: String) -> Result<Option<AssetRef<T>>, E> {
            self.visit_str(&v)
        }
    }

    d.deserialize_any(OptVisitor(PhantomData))
}

#[cfg(test)]
mod tests {
    use super::*;

    // A phantom target for exercising the reference machinery.
    struct Texture;

    #[test]
    fn deserializes_a_name_and_defers_resolution() {
        let r: AssetRef<Texture> = serde_json::from_str("\"floor\"").unwrap();
        assert_eq!(r.name(), "floor");
        assert_eq!(r.id(), None);
        assert!(!r.is_resolved());
    }

    #[test]
    fn deserializes_a_resolved_id_from_compiled_args() {
        let r: AssetRef<Texture> = serde_json::from_str("5").unwrap();
        assert_eq!(r.id(), Some(AssetId(5)));
        assert!(r.is_resolved());
    }

    #[test]
    fn resolve_fills_the_id() {
        let mut r: AssetRef<Texture> = AssetRef::by_name("wall");
        r.resolve(AssetId(3));
        assert_eq!(r.id(), Some(AssetId(3)));
    }

    #[test]
    fn serializes_a_resolved_reference_as_an_integer() {
        // The compiled-args form: a resolved reference is a bare int, matching
        // the legacy AssetId field so the blob is byte-identical.
        let r = AssetRef::<Texture>::resolved(AssetId(7));
        assert_eq!(serde_json::to_string(&r).unwrap(), "7");
    }

    #[test]
    fn serializes_an_unresolved_reference_as_its_name() {
        let r = AssetRef::<Texture>::by_name("floor");
        assert_eq!(serde_json::to_string(&r).unwrap(), "\"floor\"");
    }

    #[test]
    fn name_then_resolve_then_serialize_round_trips_to_the_blob_int() {
        // The full cook path: author a name, resolve it, re-serialize the args.
        let mut r: AssetRef<Texture> = serde_json::from_str("\"floor\"").unwrap();
        r.resolve(AssetId(2));
        assert_eq!(serde_json::to_string(&r).unwrap(), "2");
        // And the runtime reads that back as a resolved reference.
        let back: AssetRef<Texture> = serde_json::from_str("2").unwrap();
        assert_eq!(back.id(), Some(AssetId(2)));
    }

    #[derive(serde::Deserialize)]
    struct Holder {
        #[serde(default, deserialize_with = "de_opt_asset_ref")]
        r: Option<AssetRef<Texture>>,
    }

    #[test]
    fn opt_ref_treats_empty_null_and_missing_as_none() {
        let empty: Holder = serde_json::from_str("{\"r\":\"\"}").unwrap();
        assert!(empty.r.is_none());
        let null: Holder = serde_json::from_str("{\"r\":null}").unwrap();
        assert!(null.r.is_none());
        let missing: Holder = serde_json::from_str("{}").unwrap();
        assert!(missing.r.is_none());
    }

    #[test]
    fn opt_ref_reads_a_name_and_an_id() {
        let named: Holder = serde_json::from_str("{\"r\":\"mesh_a\"}").unwrap();
        assert_eq!(named.r.as_ref().unwrap().name(), "mesh_a");
        let by_id: Holder = serde_json::from_str("{\"r\":5}").unwrap();
        assert_eq!(by_id.r.unwrap().id(), Some(AssetId(5)));
    }

    #[test]
    fn is_send_and_sync_regardless_of_target() {
        // The phantom tag must not leak auto-trait requirements onto T.
        fn assert_send_sync<U: Send + Sync>() {}
        assert_send_sync::<AssetRef<Texture>>();
    }
}
