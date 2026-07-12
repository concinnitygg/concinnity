// Resource handle assignment.
//
// A resource (mesh, texture, material, ...) is addressed at runtime by a dense
// per-kind handle. Today the renderer assigns those indices itself at init (it
// drains the component column and enumerates it) and resolves every `AssetId`
// reference to one by scanning. This module moves the assignment to build time:
// cook walks the world's assets, gives each resource the next handle in its
// kind's `0..N` space (declaration order), and records `AssetId -> handle` so a
// later pass can resolve a reference to a handle without a runtime scan.
//
// This is the additive first step of the resource-table migration; nothing
// consumes the map yet.

use crate::ecs::asset_id::AssetId;
use crate::registry::ComponentType;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Once;

// The resource kinds (and their `resource_kind` blob tag) are defined in core,
// the owner of the blob format; re-exported here for the cook-side classifier and
// handle assigner so both sides agree on the kind and its tag.
pub use crate::ecs::ResourceKind;

// The resource kind a *component-registry* asset type is, or `None` if it is not
// (yet) a resource. These kinds are still stored as ECS components today (they
// migrate off the registry on Windows); AudioClip has already left, so it is
// classified through `ResourceAssetType`, not here. `asset_resource_kind` folds
// the two together for callers that only have the type name.
pub fn resource_kind(ct: ComponentType) -> Option<ResourceKind> {
    Some(match ct {
        ComponentType::Mesh => ResourceKind::Mesh,
        ComponentType::Material => ResourceKind::Material,
        ComponentType::Font => ResourceKind::Font,
        ComponentType::CubemapTexture => ResourceKind::CubemapTexture,
        ComponentType::EnvironmentMap => ResourceKind::EnvironmentMap,
        ComponentType::ColorLut => ResourceKind::ColorLut,
        ComponentType::SkinnedMesh => ResourceKind::SkinnedMesh,
        _ => return None,
    })
}

// The resource kind a declarable asset type name maps to, across both registries:
// a component-registry resource (Texture, Mesh, ...) or a resource-only asset
// (AudioClip). `None` for non-resource types. This is the single classifier the
// build uses to assign handles over the world's assets.
pub fn asset_resource_kind(asset_type: &str) -> Option<ResourceKind> {
    if let Some(ct) = ComponentType::parse(asset_type) {
        return resource_kind(ct);
    }
    ResourceAssetType::parse(asset_type).map(|rt| rt.resource_kind())
}

// Generate `ResourceAssetType` from the shared resource-asset list: the enum of
// asset types that live in the blob's resource stream (not the component
// registry), plus the uniform authoring operations over it. The per-type compile
// dispatch is hand-written below (each type compiles differently).
macro_rules! define_resource_asset_type {
    ( $( $variant:ident => $ty:path { resource: $kind:ident $(, $flag:ident)* $(,)? } ),+ $(,)? ) => {
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
            pub fn registration(self) -> crate::ecs::Registration {
                match self {
                    $( Self::$variant => crate::ecs::Registration {
                        type_name: stringify!($variant),
                        origin: crate::ecs::AssetOrigin::External,
                        payload: crate::ecs::AssetPayload::Compiled,
                        default_args: serde_json::to_value(<$ty as Default>::default()).ok(),
                    } ),+
                }
            }
            #[allow(dead_code)]
            pub fn all() -> &'static [ResourceAssetType] {
                &[ $( Self::$variant ),+ ]
            }
        }
    };
}

concinnity_core::for_each_resource_asset!(define_resource_asset_type);

impl ResourceAssetType {
    // Compile this resource's payload from its authored args. Bypasses the
    // `BuildAsset` trait (which requires `Component`, a thing a resource no longer
    // is) and calls the concrete compiler directly.
    pub fn compile_payload(self, args: &serde_json::Value) -> std::io::Result<Vec<u8>> {
        match self {
            Self::AudioClip => crate::audio_clip::compile_audio_clip_payload(args)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
            Self::Texture => crate::texture::compile_texture_payload(args)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
        }
    }

    // The source files this resource reads, folded into its payload cache key so
    // an unchanged source is a cache hit.
    pub fn source_files(self, args: &serde_json::Value) -> Vec<String> {
        match self {
            Self::AudioClip | Self::Texture => args
                .get("source")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .into_iter()
                .collect(),
        }
    }
}

// Per-kind handles assigned to each resource asset, keyed by its identity.
#[derive(Debug, Default, Clone)]
pub struct ResourceHandles {
    // Next unused handle per kind (the count assigned so far).
    next: HashMap<ResourceKind, u32>,
    // The handle each resource asset received.
    map: HashMap<(ResourceKind, AssetId), u32>,
}

impl ResourceHandles {
    // Give one resource the next handle in its kind's space and record it.
    // Declaration order in, dense `0..N` out.
    pub fn assign(&mut self, kind: ResourceKind, id: AssetId) -> u32 {
        let next = self.next.entry(kind).or_insert(0);
        let handle = *next;
        *next += 1;
        self.map.insert((kind, id), handle);
        handle
    }

    // The handle a resource received, if it was assigned one.
    pub fn get(&self, kind: ResourceKind, id: AssetId) -> Option<u32> {
        self.map.get(&(kind, id)).copied()
    }

    // How many handles a kind has assigned (its table length).
    pub fn count(&self, kind: ResourceKind) -> u32 {
        self.next.get(&kind).copied().unwrap_or(0)
    }

    // Assign handles across a world's resources, in the order given. The caller
    // has already classified each asset (via `asset_resource_kind`) and passes
    // only the resources; each kind counts independently from zero.
    pub fn from_assets(assets: impl IntoIterator<Item = (AssetId, ResourceKind)>) -> Self {
        let mut handles = Self::default();
        for (id, kind) in assets {
            handles.assign(kind, id);
        }
        handles
    }
}

// The current build's handle map, consulted by the per-kind resource-handle
// resolver seams. One map holds every kind's handles; each seam closure reads it
// with its own `ResourceKind`. Thread-local so parallel builds (and the interner
// they share) stay isolated, exactly like `concinnity_core::ecs::asset_id`'s
// interner.
thread_local! {
    static RESOURCE_HANDLES: RefCell<ResourceHandles> = RefCell::new(ResourceHandles::default());
}

// Install the schema crate's per-kind resource-handle resolver seams so a
// reference name deserializes to its dense handle value. Each closure is
// non-capturing (the map is a thread-local static), so it coerces to the plain
// `fn` pointer the seam holds; it resolves a name to its `AssetId` through the
// shared build interner, then looks up that resource's handle for its kind. An
// unknown name yields `None`, letting the deserializer fall back or fail as its
// context requires. Idempotent and cheap after the first call. Windows kinds
// (Texture aside) plug in here as they migrate.
pub fn ensure_resource_handle_resolvers() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        crate::ecs::set_texture_handle_resolver(|name| {
            let id = crate::ecs::asset_id::intern(name);
            RESOURCE_HANDLES.with(|h| h.borrow().get(ResourceKind::Texture, id))
        });
        crate::ecs::set_audio_clip_handle_resolver(|name| {
            let id = crate::ecs::asset_id::intern(name);
            RESOURCE_HANDLES.with(|h| h.borrow().get(ResourceKind::AudioClip, id))
        });
    });
}

// Install this build's handle map so resource references resolve to handles
// during the component reserialize pass. Call after `from_assets` over the
// expanded + injected world, before reserializing any component.
pub fn install_resource_handles(handles: ResourceHandles) {
    ensure_resource_handle_resolvers();
    RESOURCE_HANDLES.with(|h| *h.borrow_mut() = handles);
}

// Clear the thread-local handle map. Call at the start of a build so a prior
// build's map cannot leak into this one (mirrors `reset_interner`).
pub fn reset_resource_handles() {
    ensure_resource_handle_resolvers();
    RESOURCE_HANDLES.with(|h| *h.borrow_mut() = ResourceHandles::default());
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
        // Component-registry resources still classify through `resource_kind`.
        assert_eq!(resource_kind(ComponentType::Mesh), Some(ResourceKind::Mesh));
        assert_eq!(
            resource_kind(ComponentType::Material),
            Some(ResourceKind::Material)
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
        assert_eq!(asset_resource_kind("Texture"), Some(ResourceKind::Texture));
        assert_eq!(
            asset_resource_kind("AudioClip"),
            Some(ResourceKind::AudioClip)
        );
        assert_eq!(asset_resource_kind("PointLight"), None);
    }

    #[test]
    fn handles_are_dense_per_kind_in_order() {
        // The caller passes only the resources (pre-classified via
        // `asset_resource_kind`), each with its kind; non-resources never reach
        // here. An AudioClip lands in its own handle space, independent of Mesh.
        let assets = [
            (AssetId(10), ResourceKind::Texture),
            (AssetId(11), ResourceKind::Mesh),
            (AssetId(12), ResourceKind::Texture),
            (AssetId(20), ResourceKind::AudioClip),
            (AssetId(14), ResourceKind::Texture),
        ];
        let handles = ResourceHandles::from_assets(assets);

        // Each kind counts independently from zero, in declaration order.
        assert_eq!(handles.get(ResourceKind::Texture, AssetId(10)), Some(0));
        assert_eq!(handles.get(ResourceKind::Texture, AssetId(12)), Some(1));
        assert_eq!(handles.get(ResourceKind::Texture, AssetId(14)), Some(2));
        assert_eq!(handles.get(ResourceKind::Mesh, AssetId(11)), Some(0));

        assert_eq!(handles.count(ResourceKind::Texture), 3);
        assert_eq!(handles.count(ResourceKind::Mesh), 1);
        assert_eq!(handles.count(ResourceKind::AudioClip), 1);
        assert_eq!(handles.count(ResourceKind::Material), 0);
        // The AudioClip counts from zero in its own space, not after the textures.
        assert_eq!(handles.get(ResourceKind::AudioClip, AssetId(20)), Some(0));

        // An unassigned id has no handle.
        assert_eq!(handles.get(ResourceKind::Texture, AssetId(99)), None);
    }

    #[test]
    fn the_same_id_in_two_kinds_gets_independent_handles() {
        // Handle spaces are per kind, so the same AssetId can hold a Texture 0
        // and a Mesh 0 without collision (they are distinct resources).
        let mut handles = ResourceHandles::default();
        assert_eq!(handles.assign(ResourceKind::Texture, AssetId(1)), 0);
        assert_eq!(handles.assign(ResourceKind::Mesh, AssetId(1)), 0);
        assert_eq!(handles.get(ResourceKind::Texture, AssetId(1)), Some(0));
        assert_eq!(handles.get(ResourceKind::Mesh, AssetId(1)), Some(0));
    }

    // The load-bearing invariant of the migration: a texture reference name
    // resolves, through the installed seam, to the texture's declaration-order
    // handle -- the same order the blob emits and the runtime drains its Texture
    // column, so the handle equals that texture's albedo pool slot. Exercised on
    // the real cook path (`reserialize_args`), the same call the build uses.
    #[test]
    fn a_texture_reference_name_reserializes_to_its_declaration_order_handle() {
        use crate::ecs::asset_id;

        // Declaration order: tex_a -> Texture handle 0, tex_b -> Texture 1.
        asset_id::reset_interner();
        let a = asset_id::intern("tex_a");
        let b = asset_id::intern("tex_b");

        reset_resource_handles();
        install_resource_handles(ResourceHandles::from_assets([
            (a, ResourceKind::Texture),
            (b, ResourceKind::Texture),
        ]));

        // A Material referencing the second texture bakes albedo == 1, the first
        // bakes albedo == 0. A Decal (single texture ref) resolves the same way.
        let bake = |field_json: serde_json::Value| -> serde_json::Value {
            let bytes = ComponentType::Material
                .reserialize_args(&field_json)
                .unwrap();
            serde_json::from_slice(&bytes).unwrap()
        };
        assert_eq!(
            bake(serde_json::json!({"albedo": "tex_b"}))["albedo"],
            serde_json::json!(1)
        );
        assert_eq!(
            bake(serde_json::json!({"albedo": "tex_a"}))["albedo"],
            serde_json::json!(0)
        );

        let decal_bytes = ComponentType::Decal
            .reserialize_args(&serde_json::json!({"texture": "tex_b"}))
            .unwrap();
        let decal: serde_json::Value = serde_json::from_slice(&decal_bytes).unwrap();
        assert_eq!(decal["texture"], serde_json::json!(1));

        // The legacy texture-on-mesh path (Prop / InstancedProp / SkinnedMesh
        // `texture`) resolves the same declaration-order handle. SkinnedMesh
        // uses the identical `de_opt_texture_handle` deserializer.
        let prop_bytes = ComponentType::Prop
            .reserialize_args(&serde_json::json!({"texture": "tex_b"}))
            .unwrap();
        let prop: serde_json::Value = serde_json::from_slice(&prop_bytes).unwrap();
        assert_eq!(prop["texture"], serde_json::json!(1));

        let inst_bytes = ComponentType::InstancedProp
            .reserialize_args(&serde_json::json!({"texture": "tex_a"}))
            .unwrap();
        let inst: serde_json::Value = serde_json::from_slice(&inst_bytes).unwrap();
        assert_eq!(inst["texture"], serde_json::json!(0));
    }

    // The same invariant for audio clips: an audio-clip reference name resolves
    // to the clip's declaration-order handle, which the runtime uses to index
    // its AudioClip drain (later its resource table). Independent handle space
    // from textures, so a clip declared after two textures is still clip 0.
    #[test]
    fn an_audio_clip_reference_name_reserializes_to_its_declaration_order_handle() {
        use crate::ecs::asset_id;

        // Declaration order interleaves a texture so the per-kind counting is
        // exercised: clip_a -> AudioClip 0, clip_b -> AudioClip 1 (the texture
        // does not consume an AudioClip handle).
        asset_id::reset_interner();
        let clip_a = asset_id::intern("clip_a");
        let tex = asset_id::intern("tex");
        let clip_b = asset_id::intern("clip_b");

        reset_resource_handles();
        install_resource_handles(ResourceHandles::from_assets([
            (clip_a, ResourceKind::AudioClip),
            (tex, ResourceKind::Texture),
            (clip_b, ResourceKind::AudioClip),
        ]));

        // An AudioEmitter / AudioCue referencing a clip bakes that clip's handle.
        let clip_field = |ct: ComponentType, name: &str| -> serde_json::Value {
            let bytes = ct
                .reserialize_args(&serde_json::json!({ "clip": name }))
                .unwrap();
            let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            v["clip"].clone()
        };
        assert_eq!(
            clip_field(ComponentType::AudioEmitter, "clip_a"),
            serde_json::json!(0)
        );
        assert_eq!(
            clip_field(ComponentType::AudioEmitter, "clip_b"),
            serde_json::json!(1)
        );
        assert_eq!(
            clip_field(ComponentType::AudioCue, "clip_b"),
            serde_json::json!(1)
        );
    }
}
