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
    resolve_audio_clip_handle, resolve_font_handle, resolve_material_handle, resolve_mesh_handle,
    resolve_name, resolve_shader_handle, resolve_skinned_mesh_handle, resolve_texture_handle,
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
    ShaderHandle,
}

// One reference-resolution seam and `deserialize_with` helper per resource
// kind. A real build has the declaration-ordered handle map installed, so a
// name resolves to the resource's handle. Outside a build (single-asset
// validation, the editor's add form) the map is absent; fall back to the name
// interner so the reference still parses to *a* handle value -- one that is
// never used to index a resource table in those contexts. `None` only when
// neither resolver is installed at all.
macro_rules! handle_ref_de {
    (
        $(#[$extra:meta])*
        $handle:ident, $article:literal, $noun:literal,
        $resolve:ident, $ref_fn:ident, $opt_fn:ident $(,)?
    ) => {
        fn $ref_fn(name: &str) -> Option<u32> {
            $resolve(name).or_else(|| resolve_name(name))
        }

        #[doc = concat!(
            "`serde` `deserialize_with` helper for an optional ", $noun,
            " reference field."
        )]
        ///
        #[doc = concat!(
            "Mirrors [`crate::de_opt_asset_ref`] but resolves to a [`",
            stringify!($handle),
            "`]: an integer is an already-resolved handle (the compiled-args / ",
            "runtime form); a name string is resolved through the installed ",
            $noun,
            "-handle resolver; an empty string or null is `None`. Apply with ",
            "`#[serde(default, deserialize_with = \"concinnity_asset::",
            stringify!($opt_fn),
            "\")]`."
        )]
        $(#[$extra])*
        pub fn $opt_fn<'de, D>(d: D) -> Result<Option<$handle>, D::Error>
        where
            D: Deserializer<'de>,
        {
            // A non-self-describing format (postcard, the baked blob form)
            // carries the already-resolved handle; names only appear in
            // human-readable input.
            if !d.is_human_readable() {
                return Option::<$handle>::deserialize(d);
            }

            struct OptVisitor;

            impl Visitor<'_> for OptVisitor {
                type Value = Option<$handle>;

                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str(concat!(
                        $article, " ", $noun,
                        " handle integer, reference name string, or null"
                    ))
                }

                fn visit_unit<E: de::Error>(self) -> Result<Option<$handle>, E> {
                    Ok(None)
                }
                fn visit_none<E: de::Error>(self) -> Result<Option<$handle>, E> {
                    Ok(None)
                }
                fn visit_u64<E: de::Error>(self, v: u64) -> Result<Option<$handle>, E> {
                    Ok(Some($handle(v as u32)))
                }
                fn visit_i64<E: de::Error>(self, v: i64) -> Result<Option<$handle>, E> {
                    Ok(Some($handle(v as u32)))
                }
                fn visit_str<E: de::Error>(self, v: &str) -> Result<Option<$handle>, E> {
                    if v.is_empty() {
                        return Ok(None);
                    }
                    $ref_fn(v).map(|h| Some($handle(h))).ok_or_else(|| {
                        E::custom(format!(
                            concat!(
                                "no ", $noun,
                                "-handle resolver installed to resolve reference {:?}"
                            ),
                            v
                        ))
                    })
                }
                fn visit_string<E: de::Error>(
                    self,
                    v: alloc::string::String,
                ) -> Result<Option<$handle>, E> {
                    self.visit_str(&v)
                }
            }

            d.deserialize_any(OptVisitor)
        }
    };
}

handle_ref_de! {
    TextureHandle, "a", "texture",
    resolve_texture_handle, resolve_texture_ref, de_opt_texture_handle,
}
handle_ref_de! {
    ShaderHandle, "a", "shader",
    resolve_shader_handle, resolve_shader_ref, de_opt_shader_handle,
}
handle_ref_de! {
    MaterialHandle, "a", "material",
    resolve_material_handle, resolve_material_ref, de_opt_material_handle,
}
handle_ref_de! {
    /// The handle addresses the shared mesh-source space (Mesh / ProceduralMesh /
    /// VoxelChunk / mesh-kind File).
    MeshHandle, "a", "mesh",
    resolve_mesh_handle, resolve_mesh_ref, de_opt_mesh_handle,
}
handle_ref_de! {
    /// Used by the SkinnedMesh correlation references (`Animation.target`,
    /// `AnimGraph.target`, `FollowController.target`): a SkinnedMesh stays an ECS
    /// component, but its authored references bake to its dense handle instead of
    /// an interned id.
    SkinnedMeshHandle, "a", "skinned-mesh",
    resolve_skinned_mesh_handle, resolve_skinned_mesh_ref, de_opt_skinned_mesh_handle,
}
handle_ref_de! {
    FontHandle, "a", "font",
    resolve_font_handle, resolve_font_ref, de_opt_font_handle,
}
handle_ref_de! {
    AudioClipHandle, "an", "audio-clip",
    resolve_audio_clip_handle, resolve_audio_clip_ref, de_opt_audio_clip_handle,
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
    if !d.is_human_readable() {
        return TextureHandle::deserialize(d);
    }

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
    if !d.is_human_readable() {
        return Vec::<AudioClipHandle>::deserialize(d);
    }

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
    use alloc::string::ToString;
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

    use crate::test_support::{NoneDeserializer, install_resolvers, len_handle_resolver};

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
        install_resolvers();
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

    #[derive(Debug, serde::Deserialize)]
    struct AudioHolder {
        #[serde(default, deserialize_with = "de_opt_audio_clip_handle")]
        clip: Option<AudioClipHandle>,
        #[serde(default, deserialize_with = "de_audio_clip_handle_vec")]
        sounds: Vec<AudioClipHandle>,
    }

    #[test]
    fn de_opt_audio_clip_handle_reads_integers_and_resolves_names() {
        install_resolvers();
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
        install_resolvers();
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

    #[derive(serde::Deserialize)]
    struct TargetHolder {
        #[serde(default, deserialize_with = "de_opt_skinned_mesh_handle")]
        target: Option<SkinnedMeshHandle>,
    }

    // The baked blob form: postcard is not self-describing, so every helper
    // must read the plain resolved value instead of probing with a visitor.
    #[test]
    fn handle_helpers_round_trip_through_postcard() {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct BakedHolder {
            #[serde(default, deserialize_with = "de_opt_texture_handle")]
            tex: Option<TextureHandle>,
            #[serde(deserialize_with = "de_texture_handle")]
            stage: TextureHandle,
            #[serde(default, deserialize_with = "de_opt_mesh_handle")]
            mesh: Option<MeshHandle>,
            #[serde(default, deserialize_with = "de_opt_material_handle")]
            material: Option<MaterialHandle>,
            #[serde(default, deserialize_with = "de_opt_skinned_mesh_handle")]
            target: Option<SkinnedMeshHandle>,
            #[serde(default, deserialize_with = "de_opt_font_handle")]
            font: Option<FontHandle>,
            #[serde(default, deserialize_with = "de_opt_audio_clip_handle")]
            clip: Option<AudioClipHandle>,
            #[serde(default, deserialize_with = "de_audio_clip_handle_vec")]
            sounds: Vec<AudioClipHandle>,
            #[serde(default, deserialize_with = "de_opt_shader_handle")]
            shader: Option<ShaderHandle>,
        }
        let holder = BakedHolder {
            tex: Some(TextureHandle(3)),
            stage: TextureHandle(9),
            mesh: None,
            material: Some(MaterialHandle(1)),
            target: Some(SkinnedMeshHandle(2)),
            font: None,
            clip: Some(AudioClipHandle(4)),
            sounds: alloc::vec![AudioClipHandle(5), AudioClipHandle(6)],
            shader: Some(ShaderHandle(8)),
        };
        let bytes = postcard::to_allocvec(&holder).unwrap();
        let back: BakedHolder = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.tex, holder.tex);
        assert_eq!(back.stage, holder.stage);
        assert_eq!(back.mesh, holder.mesh);
        assert_eq!(back.material, holder.material);
        assert_eq!(back.target, holder.target);
        assert_eq!(back.font, holder.font);
        assert_eq!(back.clip, holder.clip);
        assert_eq!(back.sounds, holder.sounds);
        assert_eq!(back.shader, holder.shader);
    }

    #[test]
    fn de_opt_skinned_mesh_handle_reads_integers_and_resolves_names() {
        // The correlation-reference seam (Animation/AnimGraph/FollowController
        // `target`): an already-resolved integer passes through, a name resolves
        // through the installed skinned-mesh resolver, empty/null/missing are None.
        install_resolvers();
        let h: TargetHolder = serde_json::from_str("{\"target\":3}").unwrap();
        assert_eq!(h.target, Some(SkinnedMeshHandle(3)));
        let h: TargetHolder = serde_json::from_str("{\"target\":\"hero\"}").unwrap();
        assert_eq!(h.target, Some(SkinnedMeshHandle(4)));
        assert!(
            serde_json::from_str::<TargetHolder>("{\"target\":\"\"}")
                .unwrap()
                .target
                .is_none()
        );
        assert!(
            serde_json::from_str::<TargetHolder>("{}")
                .unwrap()
                .target
                .is_none()
        );
    }

    // Every optional-handle helper accepts the same set of input forms. One
    // generated case set per kind keeps them from drifting apart as kinds are
    // added, and pins the diagnostic each one produces for a wrong-typed field.
    macro_rules! opt_handle_cases {
        ($name:ident, $helper:literal, $helper_fn:path, $handle:ident, $expected:literal) => {
            #[test]
            fn $name() {
                install_resolvers();

                #[derive(Debug, serde::Deserialize)]
                struct Holder {
                    #[serde(default, deserialize_with = $helper)]
                    r: Option<$handle>,
                }
                let parse = |s: &str| serde_json::from_str::<Holder>(s).unwrap().r;

                // The compiled-args / runtime form: an already-resolved integer.
                assert_eq!(parse(r#"{"r":6}"#), Some($handle(6)));
                // A signed integer takes the same path, narrowed to handle width.
                assert_eq!(parse(r#"{"r":-1}"#), Some($handle(u32::MAX)));
                // The authoring form: a name resolved through the installed seam.
                assert_eq!(parse(r#"{"r":"floor"}"#), Some($handle(5)));
                // A name the build declares no resource of this kind for falls
                // back to the interner, so single-asset validation still parses.
                assert_eq!(parse(r#"{"r":"unknown_x"}"#), Some($handle(9)));
                // An owned string, the form the serde_json::Value bridge hands over.
                assert_eq!(
                    serde_json::from_value::<Holder>(serde_json::json!({"r": "wall"}))
                        .unwrap()
                        .r,
                    Some($handle(4))
                );
                // Empty, null, and missing are all absent.
                assert_eq!(parse(r#"{"r":""}"#), None);
                assert_eq!(parse(r#"{"r":null}"#), None);
                assert_eq!(parse("{}"), None);
                // As is a `None` reported by an option-aware format.
                assert_eq!($helper_fn(NoneDeserializer).unwrap(), None);
                // A wrong-typed field names what the field accepts.
                let err = serde_json::from_str::<Holder>(r#"{"r":true}"#)
                    .unwrap_err()
                    .to_string();
                assert!(err.contains($expected), "{err}");
            }
        };
    }

    opt_handle_cases!(
        opt_texture_handle_accepts_every_form,
        "de_opt_texture_handle",
        de_opt_texture_handle,
        TextureHandle,
        "a texture handle integer, reference name string, or null"
    );
    opt_handle_cases!(
        opt_shader_handle_accepts_every_form,
        "de_opt_shader_handle",
        de_opt_shader_handle,
        ShaderHandle,
        "a shader handle integer, reference name string, or null"
    );
    opt_handle_cases!(
        opt_material_handle_accepts_every_form,
        "de_opt_material_handle",
        de_opt_material_handle,
        MaterialHandle,
        "a material handle integer, reference name string, or null"
    );
    opt_handle_cases!(
        opt_mesh_handle_accepts_every_form,
        "de_opt_mesh_handle",
        de_opt_mesh_handle,
        MeshHandle,
        "a mesh handle integer, reference name string, or null"
    );
    opt_handle_cases!(
        opt_skinned_mesh_handle_accepts_every_form,
        "de_opt_skinned_mesh_handle",
        de_opt_skinned_mesh_handle,
        SkinnedMeshHandle,
        "a skinned-mesh handle integer, reference name string, or null"
    );
    opt_handle_cases!(
        opt_font_handle_accepts_every_form,
        "de_opt_font_handle",
        de_opt_font_handle,
        FontHandle,
        "a font handle integer, reference name string, or null"
    );
    opt_handle_cases!(
        opt_audio_clip_handle_accepts_every_form,
        "de_opt_audio_clip_handle",
        de_opt_audio_clip_handle,
        AudioClipHandle,
        "an audio-clip handle integer, reference name string, or null"
    );

    #[derive(Debug, serde::Deserialize)]
    struct StageHolder {
        #[serde(deserialize_with = "de_texture_handle")]
        stage: TextureHandle,
    }

    #[test]
    fn required_texture_handle_accepts_integers_and_names() {
        install_resolvers();
        let parse = |s: &str| serde_json::from_str::<StageHolder>(s).unwrap().stage;

        assert_eq!(parse(r#"{"stage":6}"#), TextureHandle(6));
        assert_eq!(parse(r#"{"stage":-1}"#), TextureHandle(u32::MAX));
        assert_eq!(parse(r#"{"stage":"floor"}"#), TextureHandle(5));
        // A name with no declared texture falls back to the interner.
        assert_eq!(parse(r#"{"stage":"unknown_x"}"#), TextureHandle(9));
        assert_eq!(
            serde_json::from_value::<StageHolder>(serde_json::json!({"stage": "wall"}))
                .unwrap()
                .stage,
            TextureHandle(4)
        );
    }

    #[test]
    fn required_texture_handle_rejects_a_missing_or_wrong_typed_field() {
        // Unlike the optional helper there is no `None` to fall back to, so an
        // absent or wrong-typed reference is an error naming what it accepts.
        let err = serde_json::from_str::<StageHolder>("{}")
            .unwrap_err()
            .to_string();
        assert!(err.contains("missing field `stage`"), "{err}");
        let err = serde_json::from_str::<StageHolder>(r#"{"stage":true}"#)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("a texture handle integer or reference name string"),
            "{err}"
        );
    }

    #[test]
    fn audio_clip_handle_vec_rejects_a_non_list() {
        let err = serde_json::from_str::<AudioHolder>(r#"{"sounds":5}"#)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("a list of audio-clip handle integers or reference name strings"),
            "{err}"
        );
    }

    #[test]
    fn audio_clip_handle_vec_resolves_owned_strings() {
        install_resolvers();
        // The serde_json::Value bridge hands each element over as an owned string.
        let h: AudioHolder =
            serde_json::from_value(serde_json::json!({"sounds": ["door", "unknown_x"]})).unwrap();
        assert_eq!(h.sounds, vec![AudioClipHandle(4), AudioClipHandle(9)]);
    }

    #[test]
    fn handles_default_to_zero_and_order_by_index() {
        // The derived ordering is the index ordering the resource tables use.
        assert_eq!(TextureHandle::default(), TextureHandle(0));
        assert!(MeshHandle(1) < MeshHandle(2));
        assert_eq!(len_handle_resolver("unknown_floor"), None);
    }
}
