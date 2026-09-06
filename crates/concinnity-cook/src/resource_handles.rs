//! Resource handle assignment over an authored world.
//!
//! A resource (mesh, texture, material, ...) is addressed at runtime by a dense
//! per-kind handle, and the table index *is* the handle. The rules that decide
//! it -- per-kind declaration order, and block order across the four geometry
//! producers -- are `concinnity_core::resource::ResourceHandles`, shared with
//! the typed bake builder so the two producers of a world can never drift. What
//! lives here is the authored-world half: classifying JSON assets into those
//! spaces, the resolver seams a reference name deserializes through, and the
//! per-type compile dispatch.

use serde::Deserialize;

use std::cell::RefCell;
use std::path::Path;
use std::sync::Once;

// The vocabulary half -- `RegisteredType` and the classifiers
// (`asset_resource_kind`, `is_mesh_source`) -- lives in `crate::authoring`;
// re-exported here so cook code keeps resolving `resource_handles::...`
// paths. The assignment rules come from core.
pub(crate) use crate::authoring::registry::RegisteredType;
pub use crate::authoring::resource_type::ResourceKind;
pub(crate) use crate::authoring::resource_type::{asset_resource_kind, is_mesh_source};
pub(crate) use concinnity_core::resource::ResourceHandles;

// Compile dispatch for resource assets, as an extension trait: the vocabulary
// enum is authoring-side, so the per-type compile arms attach here, where the
// compilers live.
pub(crate) trait ResourceAssetCompile {
    // Compile this resource's payload from its authored args, resolving bare
    // source filenames under `assets_dir`. Bypasses the `BuildAsset` trait
    // (which requires `Component`, a thing a resource no longer is) and calls
    // the concrete compiler directly.
    fn compile_payload(
        self,
        args: &serde_json::Value,
        assets_dir: Option<&Path>,
    ) -> std::io::Result<Vec<u8>>;
    // The source files this resource reads, folded into its payload cache key
    // so an unchanged source is a cache hit.
    fn source_files(self, args: &serde_json::Value, assets_dir: Option<&Path>) -> Vec<String>;
    // The baked runtime data that rides the resource record's `data_bytes`
    // ALONGSIDE a compiled payload, or `None`. SkinnedMesh is the only such
    // hybrid today: its geometry compiles into a blob payload while its
    // authored placement/material/capsule fields bake here. (Material, a pure
    // data resource, routes its bytes through `compile_payload` + `is_data`
    // instead.) `name` is the asset's declared name, interned into the baked
    // form where the runtime needs the identity.
    fn compile_data(self, name: &str, args: &serde_json::Value)
    -> std::io::Result<Option<Vec<u8>>>;
}

impl ResourceAssetCompile for RegisteredType {
    fn compile_payload(
        self,
        args: &serde_json::Value,
        assets_dir: Option<&Path>,
    ) -> std::io::Result<Vec<u8>> {
        match self {
            Self::AudioClip => crate::compile::audio_clip::compile_audio_clip_payload(args)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
            Self::Texture => crate::compile::texture::compile_texture_payload(args, assets_dir)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
            Self::CubemapTexture => crate::compile::cubemap::compile_cubemap_payload(args)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
            Self::EnvironmentMap => {
                crate::compile::environment_map::compile_environment_map_payload(args, assets_dir)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
            }
            Self::ColorLut => {
                crate::compile::color_lut::compile_color_lut_payload(args, assets_dir)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
            }
            Self::Font => crate::compile::font::compile_font_payload(args)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
            Self::Material => compile_material_data(args)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
            Self::Mesh => crate::compile::mesh_compile::compile_mesh_payload(args, assets_dir)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
            Self::SkinnedMesh => compile_skinned_mesh_payload(args),
            // The registry spans every declarable type; only the resource group
            // reaches here, and callers select on `is_resource` first.
            other => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{} is not a resource asset", other.as_str()),
            )),
        }
    }

    fn source_files(self, args: &serde_json::Value, assets_dir: Option<&Path>) -> Vec<String> {
        match self {
            Self::AudioClip
            | Self::Texture
            | Self::CubemapTexture
            | Self::EnvironmentMap
            | Self::ColorLut
            | Self::Mesh
            | Self::SkinnedMesh => {
                let mut files: Vec<String> = args
                    .get("source")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .into_iter()
                    .collect();
                // A text `.gltf` reads sibling files (external buffers /
                // images) the args never name; fold them into the cache key
                // so editing one recompiles the payload.
                if let Some(src) = files.first().cloned()
                    && src.to_lowercase().ends_with(".gltf")
                {
                    files.extend(crate::import::gltf_source::referenced_files(
                        &src, assets_dir,
                    ));
                }
                // A character model reads its source file.
                if let Some(arg) = args.get("character_model").and_then(|v| {
                    serde_json::from_value::<crate::compile::character::import::CharacterModelArg>(
                        v.clone(),
                    )
                    .ok()
                }) {
                    files.extend(arg.source_files());
                }
                files
            }
            // A Font keys its TTF source under `path`, not `source`.
            Self::Font => args
                .get("path")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .into_iter()
                .collect(),
            // A Material bakes only its authored args (texture refs are already
            // resolved handles); it reads no external source file, and neither
            // does anything outside the resource group.
            _ => Vec::new(),
        }
    }

    fn compile_data(
        self,
        name: &str,
        args: &serde_json::Value,
    ) -> std::io::Result<Option<Vec<u8>>> {
        match self {
            Self::SkinnedMesh => compile_skinned_mesh_data(name, args)
                .map(Some)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
            _ => Ok(None),
        }
    }
}

// Bake a Material into its resource `data_bytes`: resolve its texture references
// to handles (the texture-handle resolver is installed before this runs), apply
// the runtime validation clamps -- which no longer run via a generated `from_args`
// now that Material is a resource -- and serialize. The runtime deserializes these
// bytes straight back into a `Material` to build its material map.
fn compile_material_data(args: &serde_json::Value) -> Result<Vec<u8>, String> {
    let mat: crate::components::Material =
        Deserialize::deserialize(args).map_err(|e| format!("Material args: {e}"))?;
    let mat = crate::authoring::validate::material(mat);
    postcard::to_allocvec(&mat).map_err(|e| format!("Material serialize: {e}"))
}

// Compile a SkinnedMesh's geometry payload: vertices + indices + the skeleton
// the desugar pass wrote into the args, with the LOD trailer. The placement /
// material fields ride separately in the record's `data_bytes` (see
// `compile_skinned_mesh_data`).
fn compile_skinned_mesh_payload(args: &serde_json::Value) -> std::io::Result<Vec<u8>> {
    let mesh: crate::components::SkinnedMesh = Deserialize::deserialize(args)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    // `skeleton` is not a field on the schema struct; the desugar pass writes it
    // into the args JSON for glTF-sourced meshes, and inline-authored worlds may
    // carry it directly. Read it straight from the JSON so it can be baked into
    // the compiled payload alongside vertices and indices.
    let skeleton: Vec<crate::components::SkeletonJoint> = match args.get("skeleton") {
        Some(v) => Deserialize::deserialize(v).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("SkinnedMesh: invalid skeleton args: {e}"),
            )
        })?,
        None => Vec::new(),
    };
    crate::compile::geometry::compile_skinned_mesh_payload_with_lods(
        &mesh.vertices,
        &mesh.indices,
        &skeleton,
        &mesh.morph_target_names,
        &mesh.morph_deltas,
        &crate::compile::geometry::SkinnedLods {
            levels: mesh.lod_levels,
            distances: &mesh.lod_distances,
        },
    )
    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

// Bake a SkinnedMesh's runtime data (its placement, material/texture handles,
// capsule, and spawn reserve) into the resource record's `data_bytes`, applying
// the clamps that used to run in the retired `from_args`. The geometry is
// dropped from the baked form -- it rides the compiled payload. Serialized as a
// `(name_id, mesh)` JSON tuple: `asset_id` is `#[serde(skip)]` on the schema
// struct, so the interned name (which the runtime's spawn-by-name registration
// still needs) travels beside it.
fn compile_skinned_mesh_data(name: &str, args: &serde_json::Value) -> Result<Vec<u8>, String> {
    let mut sm: crate::components::SkinnedMesh =
        Deserialize::deserialize(args).map_err(|e| format!("SkinnedMesh args: {e}"))?;
    // A zero scale would collapse the world matrix; clamp to a sane unit.
    if sm.scale == [0.0, 0.0, 0.0] {
        sm.scale = [1.0, 1.0, 1.0];
    }
    // 0 and 1 are equivalent (no alternates); cap at the 8-level LOD ceiling.
    if sm.lod_levels == 0 {
        sm.lod_levels = 1;
    }
    sm.lod_levels = sm.lod_levels.min(8);
    // A runaway reserve would pre-expand the skinned vertex buffer without
    // bound; cap it. The skinned index buffer is u16, so init applies a further
    // per-mesh limit when the copies would overflow that.
    sm.max_instances = sm.max_instances.min(4096);
    // Geometry rides the compiled payload, not the baked data. The morph
    // fields are geometry too: the desugar pass writes them into the args only
    // on a payload-cache miss, so leaving them here would make the record
    // depend on whether the payload was compiled or replayed.
    sm.vertices = Vec::new();
    sm.indices = Vec::new();
    sm.morph_target_names = Vec::new();
    sm.morph_deltas = Vec::new();
    let name_id = crate::ecs::asset_id::intern(name);
    postcard::to_allocvec(&(name_id.0, sm)).map_err(|e| format!("SkinnedMesh serialize: {e}"))
}

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
        .filter(|a| a.asset_type.to_lowercase().replace('_', "") == "shader")
    {
        handles.assign_shader(crate::ecs::asset_id::intern(&asset.name));
    }
}

// Classify the expanded world's geometry producers into the shared mesh-source
// handle space and assign it. Call over the same expanded + injected `assets`
// list the blob is emitted from, after the per-kind generic pass and before
// installing the map; the block ordering itself is core's.
pub(crate) fn assign_mesh_source_handles(
    handles: &mut ResourceHandles,
    assets: &[crate::authoring::world::WorldJsonlAsset],
) {
    handles.assign_mesh_sources(assets.iter().filter_map(|a| {
        crate::authoring::resource_type::mesh_source_block(&a.asset_type, &a.args)
            .map(|block| (crate::ecs::asset_id::intern(&a.name), block))
    }));
}

// The current build's handle map, consulted by the per-kind resource-handle
// resolver seams. One map holds every kind's handles; each seam closure reads it
// with its own `ResourceKind`. Thread-local so parallel builds (and the interner
// they share) stay isolated, exactly like `concinnity_host::thread::asset_id`'s
// interner.
thread_local! {
    static RESOURCE_HANDLES: RefCell<ResourceHandles> = RefCell::new(ResourceHandles::default());
}

// Install the per-kind resource-handle resolver seams so a
// reference name deserializes to its dense handle value. Each closure is
// non-capturing (the map is a thread-local static), so it coerces to the plain
// `fn` pointer the seam holds; it resolves a name to its `AssetId` through the
// shared build interner, then looks up that resource's handle for its kind. An
// unknown name yields `None`, letting the deserializer fall back or fail as its
// context requires. Idempotent and cheap after the first call.
pub(crate) fn ensure_resource_handle_resolvers() {
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
        crate::ecs::set_shader_handle_resolver(|name| {
            let id = crate::ecs::asset_id::intern(name);
            RESOURCE_HANDLES.with(|h| h.borrow().shader(id))
        });
    });
}

// Install this build's handle map so resource references resolve to handles
// during the component reserialize pass. Call after `from_assets` over the
// expanded + injected world, before reserializing any component.
pub(crate) fn install_resource_handles(handles: ResourceHandles) {
    ensure_resource_handle_resolvers();
    RESOURCE_HANDLES.with(|h| *h.borrow_mut() = handles);
}

// Clear the thread-local handle map. Call at the start of a build so a prior
// build's map cannot leak into this one (mirrors `reset_interner`).
pub(crate) fn reset_resource_handles() {
    ensure_resource_handle_resolvers();
    RESOURCE_HANDLES.with(|h| *h.borrow_mut() = ResourceHandles::default());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::RegisteredType;

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
        use crate::authoring::world::WorldJsonlAsset;
        use crate::ecs::asset_id;

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

    // A Material bakes from its authored args alone -- its texture references
    // are already resolved handles -- so it declares no source file, and a
    // component naming one resolves it to its declaration-order handle through
    // the installed seam.
    #[test]
    fn a_material_reads_no_source_file_and_resolves_by_declaration_order() {
        use crate::ecs::asset_id;

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
        let prop: crate::components::Prop = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(prop.material, Some(crate::ecs::MaterialHandle(1)));

        // A name no material declared has no slot to point at, so it falls back
        // to the interned id rather than failing the parse.
        let bytes = RegisteredType::Prop
            .reserialize_args(&serde_json::json!({"material": "granite"}))
            .unwrap();
        let prop: crate::components::Prop = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(
            prop.material,
            Some(crate::ecs::MaterialHandle(
                asset_id::intern("granite").0 as usize as u32
            ))
        );
    }

    // A Material's `shader` reference bakes to the Shader's declaration-order
    // handle -- the same order the blob emits Shader defs and the runtime
    // drains the Shader column, so the handle indexes the pipeline table
    // directly. Shaders are components, so their space is assigned by
    // `assign_shader_handles` over the asset list, not by kind classification.
    #[test]
    fn a_shader_reference_name_bakes_to_its_declaration_order_handle() {
        use crate::authoring::world::WorldJsonlAsset;
        use crate::ecs::ShaderHandle;
        use crate::ecs::asset_id;

        asset_id::reset_interner();
        let world_asset = |name: &str, ty: &str| WorldJsonlAsset {
            name: name.to_string(),
            asset_type: ty.to_string(),
            args: serde_json::json!({}),
        };
        let assets = vec![
            world_asset("mat", "Material"),
            world_asset("shader_a", "Shader"),
            world_asset("tex", "Texture"),
            world_asset("shader_b", "Shader"),
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
        let mat: crate::components::Material = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(mat.shader, Some(ShaderHandle(1)));

        // An unreferenced shader field stays None.
        let bytes = RegisteredType::Material
            .compile_payload(&serde_json::json!({}), None)
            .unwrap();
        let mat: crate::components::Material = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(mat.shader, None);
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
        let bake = |field_json: serde_json::Value| -> crate::components::Material {
            let bytes = RegisteredType::Material
                .compile_payload(&field_json, None)
                .unwrap();
            postcard::from_bytes(&bytes).unwrap()
        };
        use crate::ecs::TextureHandle;
        assert_eq!(
            bake(serde_json::json!({"albedo": "tex_b"})).albedo,
            Some(TextureHandle(1))
        );
        assert_eq!(
            bake(serde_json::json!({"albedo": "tex_a"})).albedo,
            Some(TextureHandle(0))
        );

        let decal_bytes = RegisteredType::Decal
            .reserialize_args(&serde_json::json!({"texture": "tex_b"}))
            .unwrap();
        let decal: crate::components::Decal = postcard::from_bytes(&decal_bytes).unwrap();
        assert_eq!(decal.texture, Some(TextureHandle(1)));

        // The legacy texture-on-mesh path (Prop / InstancedProp / SkinnedMesh
        // `texture`) resolves the same declaration-order handle. SkinnedMesh
        // uses the identical `de_opt_texture_handle` deserializer.
        let prop_bytes = RegisteredType::Prop
            .reserialize_args(&serde_json::json!({"texture": "tex_b"}))
            .unwrap();
        let prop: crate::components::Prop = postcard::from_bytes(&prop_bytes).unwrap();
        assert_eq!(prop.texture, Some(TextureHandle(1)));

        let inst_bytes = RegisteredType::InstancedProp
            .reserialize_args(&serde_json::json!({"texture": "tex_a"}))
            .unwrap();
        let inst: crate::components::InstancedProp = postcard::from_bytes(&inst_bytes).unwrap();
        assert_eq!(inst.texture, Some(TextureHandle(0)));
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
        use crate::ecs::AudioClipHandle;
        let clip_field = |ct: RegisteredType, name: &str| -> Option<AudioClipHandle> {
            let bytes = ct
                .reserialize_args(&serde_json::json!({ "clip": name }))
                .unwrap();
            match ct {
                RegisteredType::AudioEmitter => {
                    postcard::from_bytes::<crate::components::AudioEmitter>(&bytes)
                        .unwrap()
                        .clip
                }
                RegisteredType::AudioCue => {
                    postcard::from_bytes::<crate::components::AudioCue>(&bytes)
                        .unwrap()
                        .clip
                }
                other => panic!("unexpected type {other:?}"),
            }
        };
        assert_eq!(
            clip_field(RegisteredType::AudioEmitter, "clip_a"),
            Some(AudioClipHandle(0))
        );
        assert_eq!(
            clip_field(RegisteredType::AudioEmitter, "clip_b"),
            Some(AudioClipHandle(1))
        );
        assert_eq!(
            clip_field(RegisteredType::AudioCue, "clip_b"),
            Some(AudioClipHandle(1))
        );
    }

    // The record's baked data has to be blind to the geometry the desugar pass
    // inlines: those args are present only when the payload was compiled this
    // build, so anything of theirs that reached the record would make the blob
    // depend on whether the payload came from the cache.
    #[test]
    fn skinned_mesh_data_is_the_same_before_and_after_the_geometry_is_inlined() {
        crate::ecs::asset_id::reset_interner();
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

        assert_eq!(
            compile_skinned_mesh_data("body", &desugared).unwrap(),
            compile_skinned_mesh_data("body", &authored).unwrap(),
        );
    }
}
