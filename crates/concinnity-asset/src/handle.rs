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

use alloc::vec::Vec;

use crate::resolver::{
    resolve_audio_clip_handle, resolve_font_handle, resolve_mesh_handle, resolve_name,
    resolve_texture_handle,
};

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

/// `serde` `deserialize_with` helper for a required texture reference field.
///
/// Like [`de_opt_texture_handle`] but for a non-optional [`TextureHandle`]: an
/// integer is an already-resolved handle (the compiled-args / runtime form); a
/// name string is resolved through the installed texture-handle resolver. Used
/// by the compiled `StoryImage.texture`, which always names a texture. Apply
/// with `#[serde(deserialize_with = "concinnity_asset::de_texture_handle")]`.
pub fn de_texture_handle<'de, D>(d: D) -> Result<TextureHandle, D::Error>
where
    D: Deserializer<'de>,
{
    struct HandleVisitor;

    impl Visitor<'_> for HandleVisitor {
        type Value = TextureHandle;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a texture handle integer or reference name string")
        }

        fn visit_u64<E: de::Error>(self, v: u64) -> Result<TextureHandle, E> {
            Ok(TextureHandle(v as u32))
        }
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<TextureHandle, E> {
            Ok(TextureHandle(v as u32))
        }
        fn visit_str<E: de::Error>(self, v: &str) -> Result<TextureHandle, E> {
            resolve_texture_ref(v).map(TextureHandle).ok_or_else(|| {
                E::custom(format!(
                    "no texture-handle resolver installed to resolve reference {v:?}"
                ))
            })
        }
        fn visit_string<E: de::Error>(self, v: alloc::string::String) -> Result<TextureHandle, E> {
            self.visit_str(&v)
        }
    }

    d.deserialize_any(HandleVisitor)
}

// Resolve an audio-clip reference name to its handle. A real build has the
// declaration-ordered audio-clip map installed; outside a build (single-asset
// validation, the editor's add form) it falls back to the name interner so the
// reference still parses to *a* handle value that is never used to index a clip
// table there. Mirrors [`resolve_texture_ref`].
fn resolve_audio_clip_ref(name: &str) -> Option<u32> {
    resolve_audio_clip_handle(name).or_else(|| resolve_name(name))
}

// Resolve a font reference name to its handle, with the same build / fallback
// behaviour as [`resolve_texture_ref`].
fn resolve_font_ref(name: &str) -> Option<u32> {
    resolve_font_handle(name).or_else(|| resolve_name(name))
}

// Resolve a mesh-source reference name to its handle, with the same build /
// fallback behaviour as [`resolve_texture_ref`]. The mesh-source handle space is
// shared across Mesh, ProceduralMesh, VoxelChunk, and mesh-kind File, so a `.mesh`
// name resolves the same way whichever kind declared it.
fn resolve_mesh_ref(name: &str) -> Option<u32> {
    resolve_mesh_handle(name).or_else(|| resolve_name(name))
}

/// `serde` `deserialize_with` helper for an optional mesh reference field.
///
/// The mesh analogue of [`de_opt_texture_handle`]: an integer is an
/// already-resolved [`MeshHandle`]; a name string is resolved through the
/// installed mesh-handle resolver; an empty string or null is `None`. The handle
/// addresses the shared mesh-source space (Mesh / ProceduralMesh / VoxelChunk /
/// mesh-kind File). Apply with `#[serde(default, deserialize_with =
/// "concinnity_asset::de_opt_mesh_handle")]`.
pub fn de_opt_mesh_handle<'de, D>(d: D) -> Result<Option<MeshHandle>, D::Error>
where
    D: Deserializer<'de>,
{
    struct OptVisitor;

    impl Visitor<'_> for OptVisitor {
        type Value = Option<MeshHandle>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a mesh handle integer, reference name string, or null")
        }

        fn visit_unit<E: de::Error>(self) -> Result<Option<MeshHandle>, E> {
            Ok(None)
        }
        fn visit_none<E: de::Error>(self) -> Result<Option<MeshHandle>, E> {
            Ok(None)
        }
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<Option<MeshHandle>, E> {
            Ok(Some(MeshHandle(v as u32)))
        }
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<Option<MeshHandle>, E> {
            Ok(Some(MeshHandle(v as u32)))
        }
        fn visit_str<E: de::Error>(self, v: &str) -> Result<Option<MeshHandle>, E> {
            if v.is_empty() {
                return Ok(None);
            }
            resolve_mesh_ref(v)
                .map(|h| Some(MeshHandle(h)))
                .ok_or_else(|| {
                    E::custom(format!(
                        "no mesh-handle resolver installed to resolve reference {v:?}"
                    ))
                })
        }
        fn visit_string<E: de::Error>(
            self,
            v: alloc::string::String,
        ) -> Result<Option<MeshHandle>, E> {
            self.visit_str(&v)
        }
    }

    d.deserialize_any(OptVisitor)
}

/// `serde` `deserialize_with` helper for an optional font reference field.
///
/// The font analogue of [`de_opt_texture_handle`]: an integer is an
/// already-resolved [`FontHandle`]; a name string is resolved through the
/// installed font-handle resolver; an empty string or null is `None`. Apply with
/// `#[serde(default, deserialize_with = "concinnity_asset::de_opt_font_handle")]`.
pub fn de_opt_font_handle<'de, D>(d: D) -> Result<Option<FontHandle>, D::Error>
where
    D: Deserializer<'de>,
{
    struct OptVisitor;

    impl Visitor<'_> for OptVisitor {
        type Value = Option<FontHandle>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a font handle integer, reference name string, or null")
        }

        fn visit_unit<E: de::Error>(self) -> Result<Option<FontHandle>, E> {
            Ok(None)
        }
        fn visit_none<E: de::Error>(self) -> Result<Option<FontHandle>, E> {
            Ok(None)
        }
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<Option<FontHandle>, E> {
            Ok(Some(FontHandle(v as u32)))
        }
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<Option<FontHandle>, E> {
            Ok(Some(FontHandle(v as u32)))
        }
        fn visit_str<E: de::Error>(self, v: &str) -> Result<Option<FontHandle>, E> {
            if v.is_empty() {
                return Ok(None);
            }
            resolve_font_ref(v)
                .map(|h| Some(FontHandle(h)))
                .ok_or_else(|| {
                    E::custom(format!(
                        "no font-handle resolver installed to resolve reference {v:?}"
                    ))
                })
        }
        fn visit_string<E: de::Error>(
            self,
            v: alloc::string::String,
        ) -> Result<Option<FontHandle>, E> {
            self.visit_str(&v)
        }
    }

    d.deserialize_any(OptVisitor)
}

/// `serde` `deserialize_with` helper for an optional audio-clip reference field.
///
/// The audio-clip analogue of [`de_opt_texture_handle`]: an integer is an
/// already-resolved [`AudioClipHandle`]; a name string is resolved through the
/// installed audio-clip-handle resolver; an empty string or null is `None`.
/// Apply with `#[serde(default, deserialize_with =
/// "concinnity_asset::de_opt_audio_clip_handle")]`.
pub fn de_opt_audio_clip_handle<'de, D>(d: D) -> Result<Option<AudioClipHandle>, D::Error>
where
    D: Deserializer<'de>,
{
    struct OptVisitor;

    impl Visitor<'_> for OptVisitor {
        type Value = Option<AudioClipHandle>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("an audio-clip handle integer, reference name string, or null")
        }

        fn visit_unit<E: de::Error>(self) -> Result<Option<AudioClipHandle>, E> {
            Ok(None)
        }
        fn visit_none<E: de::Error>(self) -> Result<Option<AudioClipHandle>, E> {
            Ok(None)
        }
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<Option<AudioClipHandle>, E> {
            Ok(Some(AudioClipHandle(v as u32)))
        }
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<Option<AudioClipHandle>, E> {
            Ok(Some(AudioClipHandle(v as u32)))
        }
        fn visit_str<E: de::Error>(self, v: &str) -> Result<Option<AudioClipHandle>, E> {
            if v.is_empty() {
                return Ok(None);
            }
            resolve_audio_clip_ref(v)
                .map(|h| Some(AudioClipHandle(h)))
                .ok_or_else(|| {
                    E::custom(format!(
                        "no audio-clip-handle resolver installed to resolve reference {v:?}"
                    ))
                })
        }
        fn visit_string<E: de::Error>(
            self,
            v: alloc::string::String,
        ) -> Result<Option<AudioClipHandle>, E> {
            self.visit_str(&v)
        }
    }

    d.deserialize_any(OptVisitor)
}

/// `serde` `deserialize_with` helper for a list of audio-clip reference fields.
///
/// Each element is either an already-resolved handle integer or a name string
/// resolved through the installed audio-clip-handle resolver, so the compiled /
/// runtime form (integers) and the authoring form (names) both parse. Apply with
/// `#[serde(default, deserialize_with =
/// "concinnity_asset::de_audio_clip_handle_vec")]`.
pub fn de_audio_clip_handle_vec<'de, D>(d: D) -> Result<Vec<AudioClipHandle>, D::Error>
where
    D: Deserializer<'de>,
{
    struct VecVisitor;

    impl<'de> Visitor<'de> for VecVisitor {
        type Value = Vec<AudioClipHandle>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a list of audio-clip handle integers or reference name strings")
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Vec<AudioClipHandle>, A::Error>
        where
            A: de::SeqAccess<'de>,
        {
            let mut out = Vec::new();
            // Each element is one optional audio-clip reference; drop the `None`
            // (empty / null) entries so a list never carries a dangling handle.
            while let Some(handle) = seq.next_element_seed(OneRef)? {
                if let Some(handle) = handle {
                    out.push(handle);
                }
            }
            Ok(out)
        }
    }

    // Deserialize one list element via the same integer-or-name path as
    // `de_opt_audio_clip_handle`.
    struct OneRef;
    impl<'de> de::DeserializeSeed<'de> for OneRef {
        type Value = Option<AudioClipHandle>;
        fn deserialize<D2>(self, d: D2) -> Result<Option<AudioClipHandle>, D2::Error>
        where
            D2: Deserializer<'de>,
        {
            de_opt_audio_clip_handle(d)
        }
    }

    d.deserialize_seq(VecVisitor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

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

    // A name resolves to its byte length, standing in for the build's real
    // declaration-ordered audio-clip handle map.
    fn len_audio_resolver(name: &str) -> Option<u32> {
        Some(name.len() as u32)
    }

    #[derive(serde::Deserialize)]
    struct AudioHolder {
        #[serde(default, deserialize_with = "de_opt_audio_clip_handle")]
        clip: Option<AudioClipHandle>,
        #[serde(default, deserialize_with = "de_audio_clip_handle_vec")]
        sounds: Vec<AudioClipHandle>,
    }

    #[test]
    fn de_opt_audio_clip_handle_reads_integers_and_resolves_names() {
        crate::set_audio_clip_handle_resolver(len_audio_resolver);
        // Already-resolved integer passes through; a name resolves via the seam.
        let h: AudioHolder = serde_json::from_str("{\"clip\":7}").unwrap();
        assert_eq!(h.clip, Some(AudioClipHandle(7)));
        let h: AudioHolder = serde_json::from_str("{\"clip\":\"theme\"}").unwrap();
        assert_eq!(h.clip, Some(AudioClipHandle(5)));
        // Empty, null, and missing are None.
        assert!(
            serde_json::from_str::<AudioHolder>("{\"clip\":\"\"}")
                .unwrap()
                .clip
                .is_none()
        );
        assert!(
            serde_json::from_str::<AudioHolder>("{}")
                .unwrap()
                .clip
                .is_none()
        );
    }

    #[test]
    fn de_audio_clip_handle_vec_resolves_mixed_and_drops_empties() {
        crate::set_audio_clip_handle_resolver(len_audio_resolver);
        // A mix of integers and names; empty entries drop out.
        let h: AudioHolder = serde_json::from_str("{\"sounds\":[3,\"door\",\"\"]}").unwrap();
        assert_eq!(h.sounds, vec![AudioClipHandle(3), AudioClipHandle(4)]);
        // Missing defaults to an empty list.
        assert!(
            serde_json::from_str::<AudioHolder>("{}")
                .unwrap()
                .sounds
                .is_empty()
        );
    }
}
