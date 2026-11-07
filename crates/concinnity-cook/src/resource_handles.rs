//! Resource handle assignment over an authored world.
//!
//! A resource (mesh, texture, material, ...) is addressed at runtime by a dense
//! per-kind handle, and the table index *is* the handle. The rules that decide
//! it -- per-kind declaration order, and block order across the four geometry
//! producers -- are `concinnity_engine::resource::ResourceHandles`, shared with
//! the typed bake builder so the two producers of a world can never drift. What
//! lives here is the authored-world half: classifying JSON assets into those
//! spaces, and the resolver seams a reference name deserializes through.

use concinnity_engine::ecs::{HandleKind, HandleResolver};
use concinnity_engine::resource::ResourceHandles;
use concinnity_host::thread::asset_id;
use std::cell::RefCell;
use std::sync::Arc;

use crate::authoring::registry::RegisteredType;

// Assign the shader handle space over the expanded world, in declaration
// order. Walks the same expanded + injected `assets` list the blob defs are
// emitted from, so a shader's handle equals the position the runtime
// encounters it when draining the Shader column. Call alongside
// `assign_mesh_source_handles`, before installing the map.
pub(crate) fn assign_shader_handles(
    handles: &mut ResourceHandles,
    assets: &[crate::authoring::world::WorldJsonlAsset],
) {
    for asset in assets
        .iter()
        .filter(|a| a.asset_type == RegisteredType::Shader)
    {
        handles.assign_shader(asset_id::intern(&asset.id));
    }
}

// Classify the expanded world's geometry producers into the shared mesh-source
// handle space and assign it. Call over the same expanded + injected `assets`
// list the blob is emitted from, after the per-kind generic pass and before
// installing the map; the block ordering itself is the engine's.
pub(crate) fn assign_mesh_source_handles(
    handles: &mut ResourceHandles,
    assets: &[crate::authoring::world::WorldJsonlAsset],
) {
    handles.assign_mesh_sources(assets.iter().filter_map(|a| {
        crate::authoring::resource_type::mesh_source_block(a.asset_type, &a.args)
            .map(|block| (asset_id::intern(&a.id), block))
    }));
}

// The current build's handle map, consulted by the resource-handle resolver.
// One map holds every space's handles. Each thread resolves through the map it
// was given, which the threads of one build share, so parallel builds stay
// isolated, as `concinnity_host::thread::asset_id`'s interner does.
thread_local! {
    static RESOURCE_HANDLES: RefCell<Arc<ResourceHandles>> = RefCell::default();
}

// The seam's view of the calling thread's handle map: a name resolves to its
// `AssetId` through the shared build interner, then to that resource's handle
// in the requested space. An unknown name yields `None`, letting the
// deserializer fall back or fail as its context requires.
struct ThreadHandles;

impl HandleResolver for ThreadHandles {
    fn resolve(&self, kind: HandleKind, name: &str) -> Option<u32> {
        let id = asset_id::resolve_reference(name);
        RESOURCE_HANDLES.with(|h| h.borrow().handle(kind, id))
    }
}

// Install the resource-handle resolver, so a reference name deserializes to
// its dense handle value. Idempotent and cheap after the first call.
pub(crate) fn ensure_resource_handle_resolvers() {
    static RESOLVER: ThreadHandles = ThreadHandles;
    concinnity_engine::ecs::install_handle_resolver(&RESOLVER);
}

// Install this build's handle map so resource references resolve to handles
// during the component reserialize pass. Call after `from_assets` over the
// expanded + injected world, before reserializing any component.
pub(crate) fn install_resource_handles(handles: ResourceHandles) {
    adopt_resource_handles(Arc::new(handles));
}

// This thread's handle map, for handing to the threads a build gives work.
pub(crate) fn current_resource_handles() -> Arc<ResourceHandles> {
    RESOURCE_HANDLES.with(|h| Arc::clone(&h.borrow()))
}

// Resolve through `handles` on the calling thread.
pub(crate) fn adopt_resource_handles(handles: Arc<ResourceHandles>) {
    ensure_resource_handle_resolvers();
    RESOURCE_HANDLES.with(|h| *h.borrow_mut() = handles);
}

// Start this thread on an empty handle map. Call at the start of a build so a
// prior build's map cannot leak into this one (mirrors `reset_interner`).
pub(crate) fn reset_resource_handles() {
    adopt_resource_handles(Arc::default());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authoring::world::WorldJsonlAsset;
    use concinnity_core::math::SaturatingFrom;
    use concinnity_engine::blob::ResourceKind;
    use concinnity_engine::components::Material;
    use concinnity_engine::components::{AudioCue, AudioEmitter, Decal, Prop};
    use concinnity_engine::ecs::{AudioClipHandle, MaterialHandle, ShaderHandle, TextureHandle};

    // The classification tests (which types are resources, mesh-source blocks)
    // live in `crate::authoring::resource_type` with the classifiers.

    // The load-bearing invariant of the shared mesh-source handle space: handles
    // are assigned across all four producer kinds in the fixed block order the
    // runtime enumerates them (Mesh, ProceduralMesh, VoxelChunk, mesh-kind File),
    // each in declaration order. Non-mesh Files and non-geometry assets get no
    // mesh handle. The runtime derives the same order while decoding geometry, so
    // a `.mesh` reference resolves to the mesh source the runtime loaded there.
    #[test]
    fn mesh_source_handles_are_block_ordered_across_kinds() {
        let a = |name: &str, asset_type: RegisteredType, args: serde_json::Value| WorldJsonlAsset {
            id: name.to_string(),
            asset_type,
            args,
        };

        // Interleaved declaration order across every producer kind, plus a
        // non-mesh File and a non-geometry asset that must be skipped.
        let assets = vec![
            a("m0", RegisteredType::Mesh, serde_json::json!({})),
            a(
                "p0",
                RegisteredType::ProceduralMesh,
                serde_json::json!({"generator": "box"}),
            ),
            a(
                "f_tex",
                RegisteredType::File,
                serde_json::json!({"kind": "png"}),
            ),
            a("v0", RegisteredType::VoxelChunk, serde_json::json!({})),
            a("m1", RegisteredType::Mesh, serde_json::json!({})),
            a(
                "f_mesh",
                RegisteredType::File,
                serde_json::json!({"kind": "obj"}),
            ),
            a("light", RegisteredType::PointLight, serde_json::json!({})),
            a(
                "p1",
                RegisteredType::ProceduralMesh,
                serde_json::json!({"generator": "sphere"}),
            ),
        ];

        // Intern in declaration order, mirroring the pipeline.
        asset_id::reset_interner();
        let names: Vec<&str> = assets.iter().map(|x| x.id.as_str()).collect();
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

    // A Material bakes from its authored args alone -- its texture references
    // are already resolved handles -- so it declares no source file, and a
    // component naming one resolves it to its declaration-order handle through
    // the installed seam.
    #[test]
    fn a_material_reads_no_source_file_and_resolves_by_declaration_order() {
        assert!(
            RegisteredType::Material
                .source_files(
                    &serde_json::json!({"albedo": "wall_tex", "source": "m.json"}),
                    None
                )
                .is_empty()
        );

        asset_id::reset_interner();
        let stone = asset_id::intern("stone");
        let wood = asset_id::intern("wood");
        reset_resource_handles();
        install_resource_handles(ResourceHandles::from_assets([
            (stone, ResourceKind::Material),
            (wood, ResourceKind::Material),
        ]));

        let bytes = RegisteredType::Prop
            .reserialize_args(&serde_json::json!({"material": "wood"}))
            .unwrap();
        let prop: Prop = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(prop.material, Some(MaterialHandle::new(1)));

        // A name no material declared has no slot to point at, so it falls back
        // to the interned id rather than failing the parse.
        let bytes = RegisteredType::Prop
            .reserialize_args(&serde_json::json!({"material": "granite"}))
            .unwrap();
        let prop: Prop = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(
            prop.material,
            Some(MaterialHandle::new(u32::saturating_from(
                asset_id::intern("granite").0 as usize
            )))
        );
    }

    // A Material's `shader` reference bakes to the Shader's declaration-order
    // handle -- the same order the blob emits Shader defs and the runtime
    // drains the Shader column, so the handle indexes the pipeline table
    // directly. Shaders are components, so their space is assigned by
    // `assign_shader_handles` over the asset list, not by kind classification.
    #[test]
    fn a_shader_reference_name_bakes_to_its_declaration_order_handle() {
        asset_id::reset_interner();
        let world_asset = |name: &str, asset_type: RegisteredType| WorldJsonlAsset {
            id: name.to_string(),
            asset_type,
            args: serde_json::json!({}),
        };
        let assets = vec![
            world_asset("mat", RegisteredType::Material),
            world_asset("shader_a", RegisteredType::Shader),
            world_asset("tex", RegisteredType::Texture),
            world_asset("shader_b", RegisteredType::Shader),
        ];

        reset_resource_handles();
        let mut handles = ResourceHandles::default();
        assign_shader_handles(&mut handles, &assets);
        assert_eq!(handles.shader_count(), 2);
        assert_eq!(handles.shader(asset_id::intern("shader_a")), Some(0));
        assert_eq!(handles.shader(asset_id::intern("shader_b")), Some(1));
        install_resource_handles(handles);

        let bytes = RegisteredType::Material
            .compile_payload(&serde_json::json!({"shader": "shader_b"}), None)
            .unwrap();
        let mat: Material = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(mat.shader, Some(ShaderHandle::new(1)));

        // An unreferenced shader field stays None.
        let bytes = RegisteredType::Material
            .compile_payload(&serde_json::json!({}), None)
            .unwrap();
        let mat: Material = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(mat.shader, None);
    }

    // The load-bearing invariant of the migration: a texture reference name
    // resolves, through the installed seam, to the texture's declaration-order
    // handle -- the same order the blob emits and the runtime drains its Texture
    // column, so the handle equals that texture's albedo pool slot. Exercised on
    // the real cook path (`reserialize_args`), the same call the build uses.
    #[test]
    fn a_texture_reference_name_reserializes_to_its_declaration_order_handle() {
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
        let bake = |field_json: serde_json::Value| -> Material {
            let bytes = RegisteredType::Material
                .compile_payload(&field_json, None)
                .unwrap();
            postcard::from_bytes(&bytes).unwrap()
        };
        assert_eq!(
            bake(serde_json::json!({"albedo": "tex_b"})).albedo,
            Some(TextureHandle::new(1))
        );
        assert_eq!(
            bake(serde_json::json!({"albedo": "tex_a"})).albedo,
            Some(TextureHandle::new(0))
        );

        let decal_bytes = RegisteredType::Decal
            .reserialize_args(&serde_json::json!({"texture": "tex_b"}))
            .unwrap();
        let decal: Decal = postcard::from_bytes(&decal_bytes).unwrap();
        assert_eq!(decal.texture, Some(TextureHandle::new(1)));
    }

    // The same invariant for audio clips: an audio-clip reference name resolves
    // to the clip's declaration-order handle, which the runtime uses to index
    // its AudioClip drain (later its resource table). Independent handle space
    // from textures, so a clip declared after two textures is still clip 0.
    #[test]
    fn an_audio_clip_reference_name_reserializes_to_its_declaration_order_handle() {
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
        let clip_field = |ct: RegisteredType, name: &str| -> Option<AudioClipHandle> {
            let bytes = ct
                .reserialize_args(&serde_json::json!({ "clip": name }))
                .unwrap();
            match ct {
                RegisteredType::AudioEmitter => {
                    postcard::from_bytes::<AudioEmitter>(&bytes).unwrap().clip
                }
                RegisteredType::AudioCue => postcard::from_bytes::<AudioCue>(&bytes).unwrap().clip,
                other => panic!("unexpected type {other:?}"),
            }
        };
        assert_eq!(
            clip_field(RegisteredType::AudioEmitter, "clip_a"),
            Some(AudioClipHandle::new(0))
        );
        assert_eq!(
            clip_field(RegisteredType::AudioEmitter, "clip_b"),
            Some(AudioClipHandle::new(1))
        );
        assert_eq!(
            clip_field(RegisteredType::AudioCue, "clip_b"),
            Some(AudioClipHandle::new(1))
        );
    }

    // The record's baked data has to be blind to the geometry the desugar pass
    // inlines: those args are present only when the payload was compiled this
    // build, so anything of theirs that reached the record would make the blob
    // depend on whether the payload came from the cache.
    #[test]
    fn skinned_mesh_data_is_the_same_before_and_after_the_geometry_is_inlined() {
        asset_id::reset_interner();
        let authored = serde_json::json!({
            "source": "hero.glb",
            "position": [1.0, 2.0, 3.0],
            "scale": [1.0, 1.0, 1.0],
            "lod_levels": 2,
            "capsule": {"half_height": 0.9, "radius": 0.35},
        });
        let mut desugared = authored.clone();
        let obj = desugared.as_object_mut().unwrap();
        obj.insert(
            "vertices".into(),
            serde_json::json!([{"pos": [0.0, 0.0, 0.0]}, {"pos": [1.0, 0.0, 0.0]}]),
        );
        obj.insert("indices".into(), serde_json::json!([0, 1, 0]));
        obj.insert("skeleton".into(), serde_json::json!([{"name": "root"}]));
        obj.insert(
            "morph_target_names".into(),
            serde_json::json!(["weight", "jaw+"]),
        );
        obj.insert(
            "morph_deltas".into(),
            serde_json::json!([
                {"position": [0.1, 0.0, 0.0]}, {"position": [0.2, 0.0, 0.0]},
                {"position": [0.0, 0.3, 0.0]}, {"position": [0.0, 0.4, 0.0]}
            ]),
        );

        let data = |args| {
            RegisteredType::SkinnedMesh
                .compile_data("body", args)
                .unwrap()
        };
        assert_eq!(data(&desugared), data(&authored));
    }
}
