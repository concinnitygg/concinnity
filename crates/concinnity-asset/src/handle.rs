// Per-kind resource handles.
//
// A resource (a mesh, texture, material, ...) is shared, compiled data the
// runtime addresses by a dense integer index into a per-kind resource table.
// Each kind has its own `0..N` index space, assigned by cook in declaration
// order. The handle is a newtype per kind so a `TextureHandle` cannot be passed
// where a `MeshHandle` is expected. Like `AssetId`, a handle serializes as a
// bare `u32`.
//
// These are the runtime replacement for the per-reference `AssetId` a component
// carries today: cook resolves the name to the resource's handle at build time,
// so the runtime never scans to resolve a reference.

use core::fmt;

use alloc::format;
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

use crate::resolver::{resolve_name, resolve_texture_handle};

macro_rules! resource_handles {
    ( $( $(#[$m:meta])* $name:ident ),+ $(,)? ) => {
        $(
            $(#[$m])*
            #[derive(
                Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default,
                Serialize, Deserialize,
            )]
            #[serde(transparent)]
            pub struct $name(pub u32);

            impl $name {
                // The handle's index into its per-kind resource table.
                pub fn index(self) -> usize {
                    self.0 as usize
                }
            }
        )+
    };
}

resource_handles! {
    MeshHandle,
    TextureHandle,
    MaterialHandle,
    FontHandle,
    AudioClipHandle,
    CubemapTextureHandle,
    EnvironmentMapHandle,
    ColorLutHandle,
    SkinnedMeshHandle,
}

// Resolve a texture reference name to its handle value. A real build has the
// declaration-ordered texture map installed, so this is the resource's handle.
// Outside a build (single-asset validation, the editor's add form) the map is
// absent; fall back to the name interner so the reference still parses to *a*
// handle value -- one that is never used to index a texture pool in those
// contexts. `None` only when neither resolver is installed at all.
fn resolve_texture_ref(name: &str) -> Option<u32> {
    resolve_texture_handle(name).or_else(|| resolve_name(name))
}

/// `serde` `deserialize_with` helper for an optional texture reference field.
///
/// Mirrors [`crate::de_opt_asset_ref`] but resolves to a [`TextureHandle`]: an
/// integer is an already-resolved handle (the compiled-args / runtime form); a
/// name string is resolved through the installed texture-handle resolver; an
/// empty string or null is `None`. Apply with `#[serde(default,
/// deserialize_with = "concinnity_asset::de_opt_texture_handle")]`.
pub fn de_opt_texture_handle<'de, D>(d: D) -> Result<Option<TextureHandle>, D::Error>
where
    D: Deserializer<'de>,
{
    struct OptVisitor;

    impl Visitor<'_> for OptVisitor {
        type Value = Option<TextureHandle>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a texture handle integer, reference name string, or null")
        }

        fn visit_unit<E: de::Error>(self) -> Result<Option<TextureHandle>, E> {
            Ok(None)
        }
        fn visit_none<E: de::Error>(self) -> Result<Option<TextureHandle>, E> {
            Ok(None)
        }
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<Option<TextureHandle>, E> {
            Ok(Some(TextureHandle(v as u32)))
        }
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<Option<TextureHandle>, E> {
            Ok(Some(TextureHandle(v as u32)))
        }
        fn visit_str<E: de::Error>(self, v: &str) -> Result<Option<TextureHandle>, E> {
            if v.is_empty() {
                return Ok(None);
            }
            resolve_texture_ref(v)
                .map(|h| Some(TextureHandle(h)))
                .ok_or_else(|| {
                    E::custom(format!(
                        "no texture-handle resolver installed to resolve reference {v:?}"
                    ))
                })
        }
        fn visit_string<E: de::Error>(
            self,
            v: alloc::string::String,
        ) -> Result<Option<TextureHandle>, E> {
            self.visit_str(&v)
        }
    }

    d.deserialize_any(OptVisitor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_serializes_as_a_bare_u32() {
        // Same wire form as AssetId: a bare integer, not a one-tuple.
        let json = serde_json::to_string(&TextureHandle(7)).unwrap();
        assert_eq!(json, "7");
        let back: TextureHandle = serde_json::from_str("7").unwrap();
        assert_eq!(back, TextureHandle(7));
    }

    #[test]
    fn index_is_the_inner_value() {
        assert_eq!(MeshHandle(0).index(), 0);
        assert_eq!(MeshHandle(42).index(), 42);
    }

    #[test]
    fn per_kind_handles_are_distinct_types_with_independent_values() {
        // A round-trip through a small table keyed by the raw index works the
        // same for each kind; the types just keep the spaces from mixing.
        let table = ["a", "b", "c"];
        assert_eq!(table[TextureHandle(1).index()], "b");
        assert_eq!(table[MeshHandle(2).index()], "c");
    }

    // A name resolves to its byte length: a deterministic, order-independent
    // stand-in for the build's real declaration-ordered handle map.
    fn len_texture_resolver(name: &str) -> Option<u32> {
        Some(name.len() as u32)
    }

    #[derive(serde::Deserialize)]
    struct Holder {
        #[serde(default, deserialize_with = "de_opt_texture_handle")]
        tex: Option<TextureHandle>,
    }

    #[test]
    fn de_opt_texture_handle_reads_an_already_resolved_integer() {
        // The compiled-args / runtime path: refs are integers, no resolver
        // needed (mirrors an already-resolved AssetId reference).
        let h: Holder = serde_json::from_str("{\"tex\":7}").unwrap();
        assert_eq!(h.tex, Some(TextureHandle(7)));
    }

    #[test]
    fn de_opt_texture_handle_resolves_a_name_through_the_seam() {
        crate::set_texture_handle_resolver(len_texture_resolver);
        let h: Holder = serde_json::from_str("{\"tex\":\"floor\"}").unwrap();
        assert_eq!(h.tex, Some(TextureHandle(5)));
    }

    #[test]
    fn de_opt_texture_handle_treats_empty_null_and_missing_as_none() {
        assert!(
            serde_json::from_str::<Holder>("{\"tex\":\"\"}")
                .unwrap()
                .tex
                .is_none()
        );
        assert!(
            serde_json::from_str::<Holder>("{\"tex\":null}")
                .unwrap()
                .tex
                .is_none()
        );
        assert!(serde_json::from_str::<Holder>("{}").unwrap().tex.is_none());
    }
}
