// A typed reference from one asset to another, declared by name.
//
// A field of type `AssetRef<T>` points at a separately-declared asset. `T` is
// the reference target -- a concrete asset type or a category marker (e.g. any
// mesh source) -- so the target kind lives in the type system instead of a
// hand-maintained side table. On the wire and at runtime the reference is a name
// (authoring) or a dense id (resolved); the referenced asset's data is never
// embedded.
//
// Deserialization matches the bare-`AssetId` field: an integer is an
// already-resolved id; a name string is run through the installed resolver, or
// kept as a name if none is installed (an out-of-engine tool reading authoring
// JSON). Serialization is likewise blob-identical: a resolved reference is its
// integer id, an unresolved one its name string.

use core::fmt;
use core::marker::PhantomData;

use alloc::boxed::Box;
use alloc::string::String;
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::AssetId;
use crate::resolver::resolve_name;

/// A by-name reference to another asset of target `T`.
pub struct AssetRef<T> {
    // The referenced asset's declared name (authoring / unresolved). Empty once
    // a reference carries only its resolved id.
    name: Box<str>,
    // The dense id, filled once resolved. `None` until then.
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

    /// The resolved dense id, or `None` if still unresolved.
    pub fn id(&self) -> Option<AssetId> {
        self.id
    }

    /// Whether the reference has been resolved to a dense id.
    pub fn is_resolved(&self) -> bool {
        self.id.is_some()
    }

    /// Fill in the resolved id.
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

// An empty, unresolved reference: lets a required `AssetRef<T>` field carry
// `#[serde(default)]` and a "not yet set" placeholder.
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

impl<T> AssetRefVisitor<T> {
    fn from_name(name: &str) -> AssetRef<T> {
        match resolve_name(name) {
            Some(id) => AssetRef::resolved(AssetId(id)),
            None => AssetRef::by_name(name),
        }
    }
}

impl<'de, T> Visitor<'de> for AssetRefVisitor<T> {
    type Value = AssetRef<T>;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("an asset reference name string or a resolved id integer")
    }

    fn visit_str<E: de::Error>(self, v: &str) -> Result<AssetRef<T>, E> {
        Ok(Self::from_name(v))
    }
    fn visit_string<E: de::Error>(self, v: String) -> Result<AssetRef<T>, E> {
        Ok(Self::from_name(&v))
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
        if d.is_human_readable() {
            d.deserialize_any(AssetRefVisitor::<T>(PhantomData))
        } else {
            Ok(AssetRef::resolved(AssetId(u32::deserialize(d)?)))
        }
    }
}

/// `serde` `deserialize_with` helper for an optional typed reference field.
///
/// Accepts a name string, an integer id, an empty string, or null; the latter
/// two resolve to `None`. Apply with `#[serde(default, deserialize_with =
/// "concinnity_asset::de_opt_asset_ref_typed")]`.
pub fn de_opt_asset_ref_typed<'de, D, T>(d: D) -> Result<Option<AssetRef<T>>, D::Error>
where
    D: Deserializer<'de>,
{
    // A non-self-describing format (postcard, the baked blob form) carries the
    // already-resolved id; names only appear in human-readable input.
    if !d.is_human_readable() {
        return Option::<AssetRef<T>>::deserialize(d);
    }

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
                Ok(Some(AssetRefVisitor::<T>::from_name(v)))
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
    // These tests stay resolver-independent (integer paths + constructors), so
    // they don't depend on the process-global resolver another test may install.
    // Name -> id resolution is exercised in the `resolver` tests.
    use super::*;

    struct Texture;

    #[test]
    fn deserializes_a_resolved_id_from_compiled_args() {
        let r: AssetRef<Texture> = serde_json::from_str("5").unwrap();
        assert_eq!(r.id(), Some(AssetId(5)));
        assert!(r.is_resolved());
    }

    #[test]
    fn resolve_fills_the_id() {
        let mut r: AssetRef<Texture> = AssetRef::by_name("wall");
        assert_eq!(r.name(), "wall");
        r.resolve(AssetId(3));
        assert_eq!(r.id(), Some(AssetId(3)));
    }

    #[test]
    fn serializes_a_resolved_reference_as_an_integer() {
        // The compiled-args form matches the legacy AssetId field byte-for-byte.
        let r = AssetRef::<Texture>::resolved(AssetId(7));
        assert_eq!(serde_json::to_string(&r).unwrap(), "7");
    }

    #[test]
    fn serializes_an_unresolved_reference_as_its_name() {
        let r = AssetRef::<Texture>::by_name("floor");
        assert_eq!(serde_json::to_string(&r).unwrap(), "\"floor\"");
    }

    #[derive(serde::Deserialize)]
    struct Holder {
        #[serde(default, deserialize_with = "de_opt_asset_ref_typed")]
        r: Option<AssetRef<Texture>>,
    }

    #[test]
    fn opt_ref_treats_empty_null_and_missing_as_none() {
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
            serde_json::from_str::<Holder>("{\"r\":5}")
                .unwrap()
                .r
                .unwrap()
                .id(),
            Some(AssetId(5))
        );
    }

    #[test]
    fn is_send_and_sync_regardless_of_target() {
        fn assert_send_sync<U: Send + Sync>() {}
        assert_send_sync::<AssetRef<Texture>>();
    }
}
