//! Per-kind resource handles.
//!
//! A resource (a mesh, texture, material, ...) is shared, compiled data the
//! runtime addresses by a dense integer index into a per-kind resource table.
//! Each kind has its own `0..N` index space, assigned by cook in declaration
//! order. [`Handle<K>`] is tagged with its space `K`, so a [`TextureHandle`]
//! cannot be passed where a [`MeshHandle`] is expected. Like `AssetId`, a handle
//! serializes as a bare `u32`.
//!
//! Cook resolves a reference name to the resource's handle at build time, so
//! the runtime never scans to resolve a reference.

use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;

use alloc::format;
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::ecs::asset_fields::ReferenceField;
use crate::ecs::resolver::{resolve_handle, resolve_name};

/// A dense index space resources are assigned handles in.
pub trait HandleSpace: 'static {
    /// Which space this is.
    const KIND: HandleKind;
}

/// A handle space holding a single asset type, so a reference into it names
/// an asset of that type.
pub trait SingleTypeSpace: HandleSpace {
    /// The registry name of the asset type the space holds.
    const TYPE: &'static str;
}

macro_rules! handle_spaces {
    ( $( $(#[$m:meta])* $space:ident => $alias:ident, $noun:literal $(, $ty:literal)? ; )+ ) => {
        /// Which dense index space a [`Handle`] addresses.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum HandleKind {
            $( $(#[$m])* $space, )+
        }

        impl HandleKind {
            /// Every handle space.
            pub const ALL: &'static [HandleKind] = &[$(HandleKind::$space),+];

            /// How many handle spaces there are.
            pub const COUNT: usize = Self::ALL.len();

            /// The space's name, the asset type it is named after.
            pub const fn name(self) -> &'static str {
                match self {
                    $( HandleKind::$space => stringify!($space), )+
                }
            }

            /// The lowercase noun diagnostics name the space by.
            pub const fn noun(self) -> &'static str {
                match self {
                    $( HandleKind::$space => $noun, )+
                }
            }
        }

        /// The marker types naming each handle space.
        pub mod space {
            $(
                $(#[$m])*
                #[derive(Debug)]
                pub enum $space {}

                impl super::HandleSpace for $space {
                    const KIND: super::HandleKind = super::HandleKind::$space;
                }

                $(
                    impl super::SingleTypeSpace for $space {
                        const TYPE: &'static str = $ty;
                    }
                )?
            )+
        }

        $(
            #[doc = concat!("An index into the runtime ", $noun, " table.")]
            pub type $alias = Handle<space::$space>;
        )+
    };
}

handle_spaces! {
    /// The shared mesh-source space: Mesh, ProceduralMesh, VoxelChunk, and
    /// mesh-kind File. Which types a reference may name depends on a File's
    /// kind, so mesh references keep a structured check.
    Mesh => MeshHandle, "mesh";
    /// Textures.
    Texture => TextureHandle, "texture", "Texture";
    /// Materials.
    Material => MaterialHandle, "material", "Material";
    /// Fonts.
    Font => FontHandle, "font", "Font";
    /// Audio clips.
    AudioClip => AudioClipHandle, "audio-clip", "AudioClip";
    /// Cubemap textures.
    CubemapTexture => CubemapTextureHandle, "cubemap-texture", "CubemapTexture";
    /// Environment maps.
    EnvironmentMap => EnvironmentMapHandle, "environment-map", "EnvironmentMap";
    /// Color lookup tables.
    ColorLut => ColorLutHandle, "color-lut", "ColorLut";
    /// Skinned meshes. A SkinnedMesh's authored references bake to its dense
    /// handle rather than an interned id.
    SkinnedMesh => SkinnedMeshHandle, "skinned-mesh", "SkinnedMesh";
    /// Shaders. A Shader is a component, but a Material's `shader` reference
    /// bakes to a dense declaration-order index.
    Shader => ShaderHandle, "shader", "Shader";
}

impl HandleKind {
    pub(crate) const fn index(self) -> usize {
        self as usize
    }

    /// The resource kind whose table this space indexes, or `None` for a space
    /// a component owns (shaders).
    pub fn resource_kind(self) -> Option<crate::blob::ResourceKind> {
        crate::blob::ResourceKind::parse(self.name())
    }
}

/// A dense index into the resource table of handle space `K`.
///
/// Deserializes from an integer (an already resolved handle, the compiled and
/// baked forms) or from a reference name, which resolves through the handle
/// resolver installed for `K`'s space and falls back to the name interner
/// outside a build. Serializes as the bare index.
#[repr(transparent)]
pub struct Handle<K: HandleSpace>(
    /// The handle's index into its space's resource table.
    pub u32,
    PhantomData<fn() -> K>,
);

impl<K: HandleSpace> Handle<K> {
    /// The handle at `index`.
    pub const fn new(index: u32) -> Self {
        Self(index, PhantomData)
    }

    /// The handle's index into its space's resource table.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

// Hand-written so no impl asks anything of the space marker, which is never a
// value.
impl<K: HandleSpace> Clone for Handle<K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K: HandleSpace> Copy for Handle<K> {}

impl<K: HandleSpace> Default for Handle<K> {
    fn default() -> Self {
        Self::new(0)
    }
}

impl<K: HandleSpace> PartialEq for Handle<K> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<K: HandleSpace> Eq for Handle<K> {}

impl<K: HandleSpace> PartialOrd for Handle<K> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<K: HandleSpace> Ord for Handle<K> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl<K: HandleSpace> Hash for Handle<K> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<K: HandleSpace> fmt::Debug for Handle<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}Handle({})", K::KIND.name(), self.0)
    }
}

impl<K: HandleSpace> Serialize for Handle<K> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u32(self.0)
    }
}

impl<'de, K: HandleSpace> Deserialize<'de> for Handle<K> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        // A non-self-describing format (postcard, the baked blob form) carries
        // the already-resolved handle; names only appear in human-readable input.
        if !d.is_human_readable() {
            return u32::deserialize(d).map(Self::new);
        }
        d.deserialize_any(HandleVisitor(K::KIND)).map(Self::new)
    }
}

impl<K: SingleTypeSpace> ReferenceField for Handle<K> {
    const TARGETS: &'static [&'static str] = &[K::TYPE];
}

struct HandleVisitor(HandleKind);

impl Visitor<'_> for HandleVisitor {
    type Value = u32;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let noun = self.0.noun();
        let article = if noun.starts_with(['a', 'e', 'i', 'o', 'u']) {
            "an"
        } else {
            "a"
        };
        write!(
            f,
            "{article} {noun} handle integer or reference name string"
        )
    }

    fn visit_u64<E: de::Error>(self, v: u64) -> Result<u32, E> {
        Ok(v as u32)
    }

    fn visit_i64<E: de::Error>(self, v: i64) -> Result<u32, E> {
        Ok(v as u32)
    }

    // A build has the declaration-ordered handle map installed, so a name
    // resolves to the resource's handle. Outside a build (single-asset
    // validation, the editor's add form) the map is absent; the name interner
    // still gives the reference *a* value, one never used to index a table.
    fn visit_str<E: de::Error>(self, v: &str) -> Result<u32, E> {
        if v.is_empty() {
            return Err(E::custom(crate::ecs::asset_id::EMPTY_REFERENCE));
        }
        resolve_handle(self.0, v)
            .or_else(|| resolve_name(v))
            .ok_or_else(|| {
                E::custom(format!(
                    "no {}-handle resolver installed to resolve reference {v:?}",
                    self.0.noun()
                ))
            })
    }

    fn visit_string<E: de::Error>(self, v: alloc::string::String) -> Result<u32, E> {
        self.visit_str(&v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;
    use alloc::vec::Vec;

    use crate::test_support::{from_json, install_resolvers};

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    struct Holder {
        #[serde(default)]
        tex: Option<TextureHandle>,
        #[serde(default)]
        clips: Vec<AudioClipHandle>,
    }

    #[test]
    fn a_handle_serializes_as_a_bare_u32() {
        assert_eq!(serde_json::to_string(&TextureHandle::new(7)).unwrap(), "7");
        let back: TextureHandle = serde_json::from_str("7").unwrap();
        assert_eq!(back, TextureHandle::new(7));
        assert_eq!(size_of::<MeshHandle>(), size_of::<u32>());
        assert_eq!(size_of::<Option<MeshHandle>>(), size_of::<Option<u32>>());
    }

    #[test]
    fn integers_are_already_resolved_handles() {
        assert_eq!(from_json::<TextureHandle>("6"), TextureHandle::new(6));
        assert_eq!(
            from_json::<TextureHandle>("-1"),
            TextureHandle::new(u32::MAX)
        );
    }

    #[test]
    fn a_name_resolves_through_its_own_space() {
        install_resolvers();
        fn only_shaders(kind: HandleKind, name: &str) -> Option<u32> {
            (kind == HandleKind::Shader).then_some(name.len() as u32 + 100)
        }
        for kind in HandleKind::ALL {
            crate::ecs::resolver::set_handle_resolver(*kind, only_shaders);
        }
        let shader: ShaderHandle = serde_json::from_str("\"water\"").unwrap();
        assert_eq!(shader, ShaderHandle::new(105));
        // The texture space declares no "water", so the interner answers.
        let texture: TextureHandle = serde_json::from_str("\"water\"").unwrap();
        assert_eq!(texture, TextureHandle::new(5));
    }

    #[test]
    fn a_name_no_resource_declares_falls_back_to_the_interner() {
        assert_eq!(from_json::<MeshHandle>("\"floor\""), MeshHandle::new(5));
        assert_eq!(from_json::<MeshHandle>("\"unknown_x\""), MeshHandle::new(9));
        // An owned string, the form the serde_json::Value bridge hands over.
        let owned: MeshHandle = serde_json::from_value(serde_json::json!("wall")).unwrap();
        assert_eq!(owned, MeshHandle::new(4));
    }

    #[test]
    fn null_and_missing_are_none_and_an_empty_name_is_an_error() {
        let h: Holder = from_json("{\"tex\":null}");
        assert_eq!(h.tex, None);
        let h: Holder = from_json("{}");
        assert_eq!(h.tex, None);
        assert!(h.clips.is_empty());

        install_resolvers();
        let err = serde_json::from_str::<Holder>("{\"tex\":\"\"}")
            .unwrap_err()
            .to_string();
        assert!(err.contains("empty reference name"), "{err}");
        // An empty name inside a list is refused the same way, not dropped.
        let err = serde_json::from_str::<Holder>("{\"clips\":[\"door\",\"\"]}")
            .unwrap_err()
            .to_string();
        assert!(err.contains("empty reference name"), "{err}");
    }

    #[test]
    fn a_list_mixes_integers_and_names() {
        let h: Holder = from_json("{\"clips\":[3,\"door\",\"unknown_x\"]}");
        assert_eq!(
            h.clips,
            vec![
                AudioClipHandle::new(3),
                AudioClipHandle::new(4),
                AudioClipHandle::new(9)
            ]
        );
    }

    #[test]
    fn a_wrong_typed_field_names_what_the_space_accepts() {
        let err = serde_json::from_str::<TextureHandle>("true")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("a texture handle integer or reference name string"),
            "{err}"
        );
        let err = serde_json::from_str::<AudioClipHandle>("1.5")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("an audio-clip handle integer or reference name string"),
            "{err}"
        );
    }

    #[test]
    fn round_trips_through_postcard_as_the_bare_index() {
        let h = Holder {
            tex: Some(TextureHandle::new(3)),
            clips: vec![AudioClipHandle::new(5), AudioClipHandle::new(6)],
        };
        let bytes = postcard::to_allocvec(&h).unwrap();
        assert_eq!(
            bytes,
            postcard::to_allocvec(&(Some(3u32), [5u32, 6].as_slice())).unwrap()
        );
        let back: Holder = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.tex, h.tex);
        assert_eq!(back.clips, h.clips);
    }

    #[test]
    fn handles_order_by_index_and_print_their_space() {
        assert_eq!(TextureHandle::default(), TextureHandle::new(0));
        assert!(MeshHandle::new(1) < MeshHandle::new(2));
        assert_eq!(MeshHandle::new(42).index(), 42);
        assert_eq!(
            alloc::format!("{:?}", SkinnedMeshHandle::new(3)),
            "SkinnedMeshHandle(3)"
        );
    }

    #[test]
    fn every_space_but_shaders_indexes_a_resource_table() {
        for kind in HandleKind::ALL {
            assert_eq!(
                kind.resource_kind().is_none(),
                *kind == HandleKind::Shader,
                "{kind:?}"
            );
        }
        assert_eq!(HandleKind::ALL.len(), HandleKind::COUNT);
        for (i, kind) in HandleKind::ALL.iter().enumerate() {
            assert_eq!(kind.index(), i);
        }
    }

    #[test]
    fn a_single_type_space_names_its_type_as_the_reference_target() {
        assert_eq!(<TextureHandle as ReferenceField>::TARGETS, ["Texture"]);
        assert_eq!(
            <Option<SkinnedMeshHandle> as ReferenceField>::TARGETS,
            ["SkinnedMesh"]
        );
    }
}
