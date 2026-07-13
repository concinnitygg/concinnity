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
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Once;

// The vocabulary half -- `ResourceAssetType`, `ResourceKind`, and the
// classifiers (`asset_resource_kind`, `resource_kind`, `is_mesh_source`) --
// lives in concinnity-world; re-exported here so cook code and downstream
// consumers keep resolving `resource_handles::...` paths. This module keeps
// the build mechanics: handle assignment, the resolver seams, and the compile
// dispatch below.
pub use concinnity_world::resource_type::{
    ResourceAssetType, ResourceKind, asset_resource_kind, is_mesh_source, resource_kind,
};

// Compile dispatch for resource assets, as an extension trait: the vocabulary
// enum lives in concinnity-world (which links no compilers), so the per-type
// compile arms attach here in cook, where the compilers live.
pub trait ResourceAssetCompile {
    // Compile this resource's payload from its authored args. Bypasses the
    // `BuildAsset` trait (which requires `Component`, a thing a resource no
    // longer is) and calls the concrete compiler directly.
    fn compile_payload(self, args: &serde_json::Value) -> std::io::Result<Vec<u8>>;
    // The source files this resource reads, folded into its payload cache key
    // so an unchanged source is a cache hit.
    fn source_files(self, args: &serde_json::Value) -> Vec<String>;
}

impl ResourceAssetCompile for ResourceAssetType {
    fn compile_payload(self, args: &serde_json::Value) -> std::io::Result<Vec<u8>> {
        match self {
            Self::AudioClip => crate::audio_clip::compile_audio_clip_payload(args)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
            Self::Texture => crate::texture::compile_texture_payload(args)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
            Self::CubemapTexture => crate::cubemap::compile_cubemap_payload(args)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
            Self::EnvironmentMap => crate::environment_map::compile_environment_map_payload(args)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
            Self::ColorLut => crate::color_lut::compile_color_lut_payload(args)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
            Self::Font => crate::font::compile_font_payload(args)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
            Self::Material => compile_material_data(args)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
        }
    }

    fn source_files(self, args: &serde_json::Value) -> Vec<String> {
        match self {
            Self::AudioClip
            | Self::Texture
            | Self::CubemapTexture
            | Self::EnvironmentMap
            | Self::ColorLut => args
                .get("source")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .into_iter()
                .collect(),
            // A Font keys its TTF source under `path`, not `source`.
            Self::Font => args
                .get("path")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .into_iter()
                .collect(),
            // A Material bakes only its authored args (texture refs are already
            // resolved handles); it reads no external source file.
            Self::Material => Vec::new(),
        }
    }
}

// Bake a Material into its resource `data_bytes`: resolve its texture references
// to handles (the texture-handle resolver is installed before this runs), apply
// the runtime validation clamps -- which no longer run via a generated `from_args`
// now that Material is a resource -- and serialize. The runtime deserializes these
// bytes straight back into a `Material` to build its material map.
fn compile_material_data(args: &serde_json::Value) -> Result<Vec<u8>, String> {
    let mat: crate::assets::Material =
        serde_json::from_value(args.clone()).map_err(|e| format!("Material args: {e}"))?;
    let mat = crate::assets::validate::material(mat);
    serde_json::to_vec(&mat).map_err(|e| format!("Material serialize: {e}"))
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

// Assign the shared mesh-source handle space over the expanded world. Every
// geometry-producing kind draws from one dense `ResourceKind::Mesh` space; the
// handles are assigned in the fixed block order the runtime enumerates them
// (Mesh, then ProceduralMesh, then VoxelChunk, then mesh-kind File), each in
// declaration order, so a `.mesh` reference resolves to the same index the
// runtime assigns while decoding geometry. Call over the same expanded +
// injected `assets` list the blob is emitted from, after the per-kind generic
// pass and before installing the map.
pub fn assign_mesh_source_handles(
    handles: &mut ResourceHandles,
    assets: &[crate::world::WorldJsonlAsset],
) {
    for block in 0..=3u8 {
        for asset in assets.iter().filter(|a| {
            concinnity_world::resource_type::mesh_source_block(&a.asset_type, &a.args)
                == Some(block)
        }) {
            handles.assign(
                ResourceKind::Mesh,
                crate::ecs::asset_id::intern(&asset.name),
            );
        }
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
        crate::ecs::set_font_handle_resolver(|name| {
            let id = crate::ecs::asset_id::intern(name);
            RESOURCE_HANDLES.with(|h| h.borrow().get(ResourceKind::Font, id))
        });
        crate::ecs::set_mesh_handle_resolver(|name| {
            let id = crate::ecs::asset_id::intern(name);
            RESOURCE_HANDLES.with(|h| h.borrow().get(ResourceKind::Mesh, id))
        });
        crate::ecs::set_material_handle_resolver(|name| {
            let id = crate::ecs::asset_id::intern(name);
            RESOURCE_HANDLES.with(|h| h.borrow().get(ResourceKind::Material, id))
        });
        crate::ecs::set_skinned_mesh_handle_resolver(|name| {
            let id = crate::ecs::asset_id::intern(name);
            RESOURCE_HANDLES.with(|h| h.borrow().get(ResourceKind::SkinnedMesh, id))
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
    use crate::registry::ComponentType;

    // The classification tests (which types are resources, mesh-source blocks)
    // live in concinnity-world with the classifiers.

    // The load-bearing invariant of the shared mesh-source handle space: handles
    // are assigned across all four producer kinds in the fixed block order the
    // runtime enumerates them (Mesh, ProceduralMesh, VoxelChunk, mesh-kind File),
    // each in declaration order. Non-mesh Files and non-geometry assets get no
    // mesh handle. The runtime derives the same order while decoding geometry, so
    // a `.mesh` reference resolves to the mesh source the runtime loaded there.
    #[test]
    fn mesh_source_handles_are_block_ordered_across_kinds() {
        use crate::ecs::asset_id;
        use crate::world::WorldJsonlAsset;

        let a = |name: &str, ty: &str, args: serde_json::Value| WorldJsonlAsset {
            name: name.to_string(),
            asset_type: ty.to_string(),
            args,
        };

        // Interleaved declaration order across every producer kind, plus a
        // non-mesh File and a non-geometry asset that must be skipped.
        let assets = vec![
            a("m0", "Mesh", serde_json::json!({})),
            a(
                "p0",
                "ProceduralMesh",
                serde_json::json!({"generator": "box"}),
            ),
            a("f_tex", "File", serde_json::json!({"kind": "png"})),
            a("v0", "VoxelChunk", serde_json::json!({})),
            a("m1", "Mesh", serde_json::json!({})),
            a("f_mesh", "File", serde_json::json!({"kind": "obj"})),
            a("light", "PointLight", serde_json::json!({})),
            a(
                "p1",
                "ProceduralMesh",
                serde_json::json!({"generator": "sphere"}),
            ),
        ];

        // Intern in declaration order, mirroring the pipeline.
        asset_id::reset_interner();
        let names: Vec<&str> = assets.iter().map(|x| x.name.as_str()).collect();
        asset_id::intern_all(&names);

        let mut handles = ResourceHandles::default();
        assign_mesh_source_handles(&mut handles, &assets);

        let h = |name: &str| handles.get(ResourceKind::Mesh, asset_id::intern(name));
        // Block order: all Meshes, then ProceduralMeshes, then VoxelChunks, then
        // mesh-kind Files; declaration order within each block.
        assert_eq!(h("m0"), Some(0));
        assert_eq!(h("m1"), Some(1));
        assert_eq!(h("p0"), Some(2));
        assert_eq!(h("p1"), Some(3));
        assert_eq!(h("v0"), Some(4));
        assert_eq!(h("f_mesh"), Some(5));
        // A non-mesh File and a non-geometry component get no mesh handle.
        assert_eq!(h("f_tex"), None);
        assert_eq!(h("light"), None);
        assert_eq!(handles.count(ResourceKind::Mesh), 6);
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
        // Material is a resource now, so it bakes through `compile_payload` (its
        // `data_bytes` is the serialized Material) rather than `reserialize_args`.
        let bake = |field_json: serde_json::Value| -> serde_json::Value {
            let bytes = ResourceAssetType::Material
                .compile_payload(&field_json)
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
