// The resource-asset vocabulary: `ResourceAssetType`, the enum of asset types
// that live in the blob's resource stream (not the component registry), plus
// the classifiers that decide which resource kind an authored asset is. The
// per-type compile dispatch lives in concinnity-cook (each type compiles
// differently and the compilers live there); this module owns only naming,
// parsing, kind mapping, and authoring metadata.

use crate::registry::ComponentType;

// The resource kinds (and their `resource_kind` blob tag) are defined with the
// blob format; re-exported here for the classifiers and the cook-side handle
// assigner so both sides agree on the kind and its tag.
pub use concinnity_core::ecs::ResourceKind;

// The resource kind a *component-registry* asset type is, or `None` if it is
// not a resource. Every resource kind has now left the component registry (all
// classify through `ResourceAssetType`); kept so `asset_resource_kind` retains
// its two-registry shape until the endgame retires it.
pub fn resource_kind(ct: ComponentType) -> Option<ResourceKind> {
    let _ = ct;
    None
}

// The mesh-source handle space is shared across every geometry-producing kind
// (Mesh, ProceduralMesh, VoxelChunk, and mesh-kind File), so it is not assigned
// through the type-name classifier above: File is polymorphic (only a mesh-kind
// File is geometry, which the type name alone cannot tell), and the four kinds
// must draw from one dense space in a fixed order. cook's
// `assign_mesh_source_handles` owns that assignment, driven by
// `mesh_source_block` below.

// Normalize an asset type name the same way the cross-reference checker does, so
// both agree on what counts as a mesh source.
fn norm_type(t: &str) -> String {
    t.to_lowercase().replace('_', "")
}

// True when a `File`'s args name a mesh-kind file (the only File that produces
// geometry).
fn file_is_mesh(args: &serde_json::Value) -> bool {
    args.get("kind")
        .and_then(|k| k.as_str())
        .and_then(crate::assets::FileKind::from_ext)
        .map(|fk| fk.is_mesh())
        .unwrap_or(false)
}

// The mesh-source block an asset belongs to, or `None` if it is not a geometry
// producer. The blocks are the fixed order the runtime enumerates mesh sources
// in (Mesh, then ProceduralMesh, then VoxelChunk, then mesh-kind File), so a
// handle assigned in block order equals the runtime's mesh-source index.
pub fn mesh_source_block(asset_type: &str, args: &serde_json::Value) -> Option<u8> {
    match norm_type(asset_type).as_str() {
        "mesh" => Some(0),
        "proceduralmesh" => Some(1),
        "voxelchunk" | "chunk" => Some(2),
        "file" => file_is_mesh(args).then_some(3),
        _ => None,
    }
}

// Whether an asset produces geometry addressable by a mesh handle. The single
// classifier the cross-reference checker and the handle assigner share.
pub fn is_mesh_source(asset_type: &str, args: &serde_json::Value) -> bool {
    mesh_source_block(asset_type, args).is_some()
}

// The resource kind a declarable asset type name maps to, across both registries:
// a component-registry resource (SkinnedMesh) or a resource-only asset (Texture,
// AudioClip, ...). `None` for non-resource types. This is the single classifier
// the build uses to assign handles over the world's assets.
pub fn asset_resource_kind(asset_type: &str) -> Option<ResourceKind> {
    if let Some(ct) = ComponentType::parse(asset_type) {
        return resource_kind(ct);
    }
    let rt = ResourceAssetType::parse(asset_type)?;
    // Mesh draws from the shared mesh-source handle space (assigned by cook's
    // `assign_mesh_source_handles` in block order across all four geometry
    // producers), so the per-kind declaration-order classifier must not also
    // assign it.
    if rt == ResourceAssetType::Mesh {
        return None;
    }
    Some(rt.resource_kind())
}

// Generate `ResourceAssetType` from the shared resource-asset list: the enum of
// asset types that live in the blob's resource stream (not the component
// registry), plus the uniform authoring operations over it. The per-type compile
// dispatch is hand-written in concinnity-cook (each type compiles differently).
macro_rules! define_resource_asset_type {
    ( $( $variant:ident => $ty:path { resource: $kind:ident $($meta:tt)* } ),+ $(,)? ) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum ResourceAssetType {
            $( $variant ),+
        }

        impl ResourceAssetType {
            pub fn as_str(self) -> &'static str {
                match self { $( Self::$variant => stringify!($variant) ),+ }
            }
            pub fn parse(s: &str) -> Option<Self> {
                $( if s == stringify!($variant) { return Some(Self::$variant); } )+
                None
            }
            // The dense per-kind handle space this asset is assigned into.
            pub fn resource_kind(self) -> ResourceKind {
                match self { $( Self::$variant => ResourceKind::$kind ),+ }
            }
            // Authoring metadata: a resource asset is External with a compiled
            // payload; its default args come from the schema struct's `Default`.
            pub fn registration(self) -> crate::registry::Registration {
                match self {
                    $( Self::$variant => crate::registry::Registration {
                        type_name: stringify!($variant),
                        origin: crate::ecs::AssetOrigin::External,
                        payload: crate::ecs::AssetPayload::Compiled,
                        default_args: serde_json::to_value(<$ty as Default>::default()).ok(),
                    } ),+
                }
            }
            pub fn all() -> &'static [ResourceAssetType] {
                &[ $( Self::$variant ),+ ]
            }
            // The structural flags shared with the component registry: see
            // `ComponentType::{useful_blank, renders}`. Resource assets are
            // never singletons.
            pub fn useful_blank(self) -> bool {
                match self {
                    $( Self::$variant => crate::registry::__meta_useful_blank!($($meta)*) ),+
                }
            }
            pub fn renders(self) -> bool {
                match self {
                    $( Self::$variant => crate::registry::__meta_renders!($($meta)*) ),+
                }
            }
        }
    };
}

concinnity_core::for_each_resource_asset!(define_resource_asset_type);

impl ResourceAssetType {
    // Whether this resource is a DATA resource: its compiled bytes are the
    // record's inline `data_bytes`, not a blob payload the record points at.
    // Material (small, per-object surface params) is the only one today; every
    // other kind is a `compiled` payload resource.
    pub fn is_data(self) -> bool {
        matches!(self, Self::Material)
    }

    // The asset-reference fields this resource declares, as `(field, target type)`.
    // Mirrors `ComponentType::ref_fields`: the editor add-form turns each into a
    // name picker over the world's assets of that type. Material carries its
    // texture references here now that it has left the component registry (which
    // used to supply this via the `refs:` metadata).
    pub fn ref_fields(self) -> &'static [(&'static str, &'static str)] {
        match self {
            Self::Material => &[
                ("albedo", "Texture"),
                ("normal_map", "Texture"),
                ("emissive_map", "Texture"),
                ("orm_map", "Texture"),
                ("albedo_secondary", "Texture"),
                ("normal_secondary", "Texture"),
            ],
            _ => &[],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_types_classify_others_do_not() {
        // Texture has left the component registry: it classifies through
        // `ResourceAssetType`, folded in by `asset_resource_kind`.
        assert_eq!(ComponentType::parse("Texture"), None);
        assert_eq!(asset_resource_kind("Texture"), Some(ResourceKind::Texture));
        // CubemapTexture has also left: a resource-only asset, no longer a
        // `ComponentType`, classified through `ResourceAssetType`.
        assert_eq!(ComponentType::parse("CubemapTexture"), None);
        assert_eq!(
            ResourceAssetType::parse("CubemapTexture"),
            Some(ResourceAssetType::CubemapTexture)
        );
        assert_eq!(
            asset_resource_kind("CubemapTexture"),
            Some(ResourceKind::CubemapTexture)
        );
        // EnvironmentMap, ColorLut, and Font have left too.
        assert_eq!(ComponentType::parse("EnvironmentMap"), None);
        assert_eq!(
            asset_resource_kind("EnvironmentMap"),
            Some(ResourceKind::EnvironmentMap)
        );
        assert_eq!(ComponentType::parse("ColorLut"), None);
        assert_eq!(
            asset_resource_kind("ColorLut"),
            Some(ResourceKind::ColorLut)
        );
        assert_eq!(ComponentType::parse("Font"), None);
        assert_eq!(asset_resource_kind("Font"), Some(ResourceKind::Font));
        // Mesh has left the component registry, but its handle is still not
        // assigned through the type-name classifier: it shares the mesh-source
        // handle space with ProceduralMesh / VoxelChunk / File, assigned by
        // cook's `assign_mesh_source_handles`. So the generic classifier does
        // not treat it as a resource, while `ResourceAssetType` still parses it
        // (compile + record emission go through the resource path).
        assert_eq!(ComponentType::parse("Mesh"), None);
        assert_eq!(
            ResourceAssetType::parse("Mesh"),
            Some(ResourceAssetType::Mesh)
        );
        assert_eq!(asset_resource_kind("Mesh"), None);
        assert!(is_mesh_source("Mesh", &serde_json::json!({})));
        // Material has left the component registry: a DATA resource, no longer a
        // `ComponentType`, classified through `ResourceAssetType`.
        assert_eq!(ComponentType::parse("Material"), None);
        assert_eq!(
            asset_resource_kind("Material"),
            Some(ResourceKind::Material)
        );
        assert!(ResourceAssetType::Material.is_data());
        assert!(!ResourceAssetType::Texture.is_data());
        // SkinnedMesh has left the component registry too: it classifies
        // through `ResourceAssetType` (a compiled payload + inline baked data).
        assert_eq!(ComponentType::parse("SkinnedMesh"), None);
        assert_eq!(
            ResourceAssetType::parse("SkinnedMesh"),
            Some(ResourceAssetType::SkinnedMesh)
        );
        assert_eq!(
            asset_resource_kind("SkinnedMesh"),
            Some(ResourceKind::SkinnedMesh)
        );
        // Pure-data components and containers are not resources.
        assert_eq!(resource_kind(ComponentType::PointLight), None);
        assert_eq!(resource_kind(ComponentType::Prop), None);
        assert_eq!(resource_kind(ComponentType::Transform), None);

        // AudioClip has left the component registry: it is a resource-only asset,
        // classified through `ResourceAssetType`, and no longer a `ComponentType`.
        assert_eq!(ComponentType::parse("AudioClip"), None);
        assert_eq!(
            ResourceAssetType::parse("AudioClip"),
            Some(ResourceAssetType::AudioClip)
        );
        // The combined classifier covers both registries by type name.
        assert_eq!(
            asset_resource_kind("AudioClip"),
            Some(ResourceKind::AudioClip)
        );
        assert_eq!(asset_resource_kind("PointLight"), None);
    }

    // The structural flags on the resource registry: Material and Font are the
    // blank-useful addables; EnvironmentMap and SkinnedMesh are the
    // render-implying kinds (a placed skinned mesh is scene content in its own
    // right, so its presence renders like any static Prop).
    #[test]
    fn resource_structural_flags_mark_the_curated_sets() {
        let flagged = |f: fn(ResourceAssetType) -> bool| -> Vec<&'static str> {
            ResourceAssetType::all()
                .iter()
                .copied()
                .filter(|&t| f(t))
                .map(ResourceAssetType::as_str)
                .collect()
        };
        assert_eq!(
            flagged(ResourceAssetType::useful_blank),
            ["Font", "Material"]
        );
        assert_eq!(
            flagged(ResourceAssetType::renders),
            ["EnvironmentMap", "SkinnedMesh"]
        );
    }

    #[test]
    fn mesh_source_blocks_follow_the_fixed_order() {
        let none = serde_json::json!({});
        assert_eq!(mesh_source_block("Mesh", &none), Some(0));
        assert_eq!(mesh_source_block("ProceduralMesh", &none), Some(1));
        assert_eq!(mesh_source_block("VoxelChunk", &none), Some(2));
        // Only a mesh-kind File is a geometry producer.
        assert_eq!(
            mesh_source_block("File", &serde_json::json!({"kind": "obj"})),
            Some(3)
        );
        assert_eq!(
            mesh_source_block("File", &serde_json::json!({"kind": "png"})),
            None
        );
        assert_eq!(mesh_source_block("PointLight", &none), None);
    }
}
