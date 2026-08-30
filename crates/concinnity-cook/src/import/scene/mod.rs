//! Build-time expansion of a scene file into Concinnity asset entries
//! (Texture / Material / Mesh / Model / Prop / Camera3D, plus SkinnedMesh /
//! Animation for a file that carries rigs, see `rig`). Each entry references
//! the source file by path: geometry is filled in later by the desugar passes
//! in `pipeline::desugar` and texture pixels by `compile_texture_payload`, so the
//! generated entries carry no inline vertex or pixel data. The expansion is
//! driven from a `SceneImport` asset by
//! `crate::build_only::scene_import::expand_scene_imports`.
//!
//! Two container formats are supported, dispatched by `source` extension:
//!   - `.fbx` via `crate::import::fbx`
//!   - `.glb` via `crate::import::gltf` / `crate::import::glb`
//!
//! PBR mapping, FBX texture slot -> Concinnity Material field:
//!   DiffuseColor  -> albedo
//!   NormalMap     -> normal_map
//!   SpecularColor -> orm_map      (packed occlusion / roughness / metalness)
//!   EmissiveColor -> emissive_map
//!
//! PBR mapping, glTF -> Concinnity Material:
//!   baseColorTexture         -> albedo
//!   baseColorFactor.rgb      -> tint
//!   normalTexture            -> normal_map
//!   metallicRoughnessTexture -> orm_map  (glTF packs G = roughness, B = metalness)
//!   emissiveTexture          -> emissive_map
//!   metallicFactor           -> metallic
//!   roughnessFactor          -> roughness
//!   emissiveFactor           -> emissive_factor
//!   alphaMode MASK           -> alpha_cutoff (from alphaCutoff, default 0.5)
//! occlusionTexture is dropped on purpose: the screen-space pass is the engine's
//! ambient-occlusion source, and `Material::orm_map` reserves its red channel for
//! that reason. alphaMode BLEND is dropped too, because `Material::transparent`
//! means refracting glass, not a blended card.

mod gltf_material;
mod rig;

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::gfx::transform::{IDENTITY, Mat4, decompose, euler_yxz_from_quat, mat4_mul};
use rig::{SkinnedPart, rig_entries};

// u16 index ceiling: a primitive with more vertices than this fans into chunks.
const U16_CAPACITY: usize = u16::MAX as usize + 1;

// Knobs threaded in from the `SceneImport` asset's args. `name_prefix` is the
// import's (unique) asset name, sanitized; every generated asset name carries
// it so the expansion never collides with hand-authored assets.
#[derive(Debug, Clone)]
pub(crate) struct ImportOptions {
    pub(crate) name_prefix: String,
    pub texture_max_size: u32,
    pub(crate) emissive_map_strength: f32,
    pub(crate) emit_camera: bool,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self {
            name_prefix: "scene".to_string(),
            texture_max_size: 512,
            emissive_map_strength: 3.0,
            emit_camera: true,
        }
    }
}

// Expand a scene file into asset entries, dispatching on the source extension.
pub(crate) fn entries_from_scene(
    source: &str,
    opts: &ImportOptions,
    assets_dir: Option<&Path>,
) -> std::io::Result<Vec<serde_json::Value>> {
    let ext = Path::new(source)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "fbx" => entries_from_fbx(source, opts),
        "glb" | "gltf" => entries_from_glb(source, opts, assets_dir),
        other => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "SceneImport source '{}': unsupported format '.{}' (supported: .fbx, .glb, .gltf)",
                source, other
            ),
        )),
    }
}

// Lowercase ASCII-alphanumeric/underscore sanitizer for an asset-name prefix
// and for node-derived prop names. Everything else collapses to underscore so
// the result reads like an identifier.
pub(crate) fn sanitize_name(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for c in input.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "scene".to_string()
    } else {
        out
    }
}

// FBX -> asset entries.
fn entries_from_fbx(path: &str, opts: &ImportOptions) -> std::io::Result<Vec<serde_json::Value>> {
    let scene = crate::import::fbx::parse_fbx(path).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("'{}': {}", path, e),
        )
    })?;

    let prefix = &opts.name_prefix;
    let mut entries: Vec<serde_json::Value> = Vec::new();

    // Materials -> entries, interning each referenced texture path once.
    let mut tex_names: HashMap<String, String> = HashMap::new();
    let default_mat = format!("{prefix}_mat_default");
    entries.push(serde_json::json!({
        "name": default_mat,
        "type": "Material",
        "args": { "roughness": 0.8, "metallic": 0.0 }
    }));

    let mut material_names: Vec<String> = Vec::with_capacity(scene.materials.len());
    for (i, m) in scene.materials.iter().enumerate() {
        let name = format!("{prefix}_mat_{i}");
        let mut args = serde_json::Map::new();
        // Smooth, see-through glass: flagged transparent by the FBX, or named
        // like glass/window. Glass renders smooth + translucent (transparent
        // pass when ray tracing is available, else a smooth reflective opaque
        // surface). Frosted glass stays rough/diffuse and emissive "glass"
        // (lamp lenses) stays an opaque glow, so both are excluded.
        let lname = m.name.to_lowercase();
        let is_glass = (m.opacity < 0.95 || lname.contains("glass") || lname.contains("window"))
            && !lname.contains("frosted")
            && !lname.contains("emissive");
        if let Some(p) = &m.albedo {
            let t = intern_texture(
                p,
                prefix,
                opts.texture_max_size,
                &mut tex_names,
                &mut entries,
            );
            args.insert("albedo".into(), serde_json::Value::String(t));
        }
        if let Some(p) = &m.normal {
            let t = intern_texture(
                p,
                prefix,
                opts.texture_max_size,
                &mut tex_names,
                &mut entries,
            );
            args.insert("normal_map".into(), serde_json::Value::String(t));
        }
        // Glass drops the packed ORM map: its per-texel roughness would override
        // the low scalar roughness below and leave the surface non-reflective.
        if let (false, Some(p)) = (is_glass, &m.orm) {
            let t = intern_texture(
                p,
                prefix,
                opts.texture_max_size,
                &mut tex_names,
                &mut entries,
            );
            args.insert("orm_map".into(), serde_json::Value::String(t));
        }
        if let Some(p) = &m.emissive {
            let t = intern_texture(
                p,
                prefix,
                opts.texture_max_size,
                &mut tex_names,
                &mut entries,
            );
            args.insert("emissive_map".into(), serde_json::Value::String(t));
        }
        args.insert(
            "tint".into(),
            serde_json::json!([m.diffuse[0], m.diffuse[1], m.diffuse[2]]),
        );
        // A textured emissive drives the glow through a punchy factor; without a
        // map, fall back to the FBX emissive factor (usually zero).
        let emissive_factor = if m.emissive.is_some() {
            [opts.emissive_map_strength; 3]
        } else {
            m.emissive_factor
        };
        args.insert(
            "emissive_factor".into(),
            serde_json::json!([emissive_factor[0], emissive_factor[1], emissive_factor[2]]),
        );
        // Scalar fallbacks; the orm_map overrides roughness/metalness per-texel
        // when present (never for glass, which dropped the orm map above).
        if is_glass {
            // Smooth dielectric so the reflection passes (SSR / RT) pick it up,
            // plus the transparency that routes it through the transparent pass.
            args.insert("roughness".into(), serde_json::json!(0.08));
            args.insert("metallic".into(), serde_json::json!(0.0));
            let opacity = if m.opacity < 0.95 { m.opacity } else { 0.25 };
            args.insert("opacity".into(), serde_json::json!(opacity));
            args.insert("transparent".into(), serde_json::json!(true));
        } else {
            args.insert("roughness".into(), serde_json::json!(0.7));
            args.insert("metallic".into(), serde_json::json!(0.0));
        }

        entries.push(serde_json::json!({
            "name": name,
            "type": "Material",
            "args": serde_json::Value::Object(args),
        }));
        material_names.push(name);
    }

    // Primitives drawn by a skin-deformed node expand as a SkinnedMesh below,
    // so they emit no static entries; the primitive index space is the
    // parser's and stays untouched either way.
    let skinned_primitives: HashSet<usize> = scene
        .props
        .iter()
        .filter(|p| p.skin_index.is_some())
        .flat_map(|p| p.primitives.iter().copied())
        .collect();

    // Meshes: one per primitive, fanning oversized primitives into u16 chunks.
    // Record the mesh asset name(s) produced for each primitive so the prop's
    // model can list them as submeshes.
    let mut primitive_meshes: Vec<Vec<String>> = vec![Vec::new(); scene.primitives.len()];
    for (i, prim) in scene.primitives.iter().enumerate() {
        if skinned_primitives.contains(&i) {
            continue;
        }
        if prim.vertices.len() <= U16_CAPACITY {
            let mesh_name = format!("{prefix}_prim_{i}");
            entries.push(serde_json::json!({
                "name": mesh_name,
                "type": "Mesh",
                "args": { "source": path, "primitive_index": i }
            }));
            primitive_meshes[i].push(mesh_name);
            continue;
        }
        // Oversized: count the chunks the build will produce and emit one Mesh
        // per chunk, each carrying its `chunk_index`.
        let (_, indices32) = crate::import::fbx::read_primitive_geometry(&scene, i as u32)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let chunk_count = crate::import::glb::count_u16_chunks(&indices32);
        for chunk_idx in 0..chunk_count {
            let mesh_name = format!("{prefix}_prim_{i}_chunk_{chunk_idx}");
            entries.push(serde_json::json!({
                "name": mesh_name,
                "type": "Mesh",
                "args": { "source": path, "primitive_index": i, "chunk_index": chunk_idx }
            }));
            primitive_meshes[i].push(mesh_name);
        }
    }

    // Models + Props: one of each per scene node that carries static geometry.
    for (pi, prop) in scene.props.iter().enumerate() {
        if prop.skin_index.is_some() {
            continue;
        }
        let mut submeshes: Vec<serde_json::Value> = Vec::new();
        for &prim_idx in &prop.primitives {
            let material = scene.primitives[prim_idx]
                .material
                .and_then(|mi| material_names.get(mi).cloned())
                .unwrap_or_else(|| default_mat.clone());
            for mesh_name in &primitive_meshes[prim_idx] {
                submeshes.push(serde_json::json!({ "mesh": mesh_name, "material": material }));
            }
        }
        if submeshes.is_empty() {
            continue;
        }

        let model_name = format!("{prefix}_model_{pi}");
        entries.push(serde_json::json!({
            "name": model_name,
            "type": "Model",
            "args": { "meshes": submeshes }
        }));

        // Prop name: descriptive when the node is named, always suffixed with
        // the index so 1000+ nodes stay unique even with duplicate names.
        let prop_name = if prop.name.is_empty() {
            format!("{prefix}_node_{pi}")
        } else {
            format!("{prefix}_{}_{pi}", sanitize_name(&prop.name))
        };
        entries.push(serde_json::json!({
            "name": prop_name,
            "type": "Prop",
            "args": {
                "model": model_name,
                "position": prop.position,
                "rotation_deg": prop.rotation_deg,
                "scale": prop.scale,
            }
        }));
    }

    // SkinnedMesh + Animation, one set per skin-deformed node.
    let mut parts: Vec<SkinnedPart> = scene
        .props
        .iter()
        .filter_map(|p| {
            let skin_index = p.skin_index?;
            let material = p
                .primitives
                .first()
                .and_then(|&i| scene.primitives[i].material)
                .and_then(|mi| material_names.get(mi).cloned())
                .unwrap_or_else(|| default_mat.clone());
            Some(SkinnedPart {
                skin_index,
                material,
            })
        })
        .collect();
    if !parts.is_empty() {
        parts.sort_by_key(|p| p.skin_index);
        let clips = crate::import::fbx::fbx_animation_names(path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        entries.extend(rig_entries(prefix, path, &parts, &clips));
    }

    // Camera framed to the scene's world AABB, mirroring the glTF importer.
    if opts.emit_camera
        && let Some(camera) = framed_camera_entry(prefix, scene.aabb)
    {
        entries.push(camera);
    }

    Ok(entries)
}

// glTF (.glb / .gltf) -> asset entries. The generated entries name the source
// exactly as the world declared it, so each compiles through the same
// resolution the import ran here.
fn entries_from_glb(
    path: &str,
    opts: &ImportOptions,
    assets_dir: Option<&Path>,
) -> std::io::Result<Vec<serde_json::Value>> {
    let doc = crate::import::gltf_source::GltfDoc::parse_file(&crate::import::glb::resolve_source(
        path, assets_dir,
    ))
    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    let prefix = &opts.name_prefix;
    let mut entries: Vec<serde_json::Value> = Vec::new();

    // Textures: one entry per glTF image, referencing the GLB binary chunk.
    for (i, _img) in doc.doc.document.images().enumerate() {
        let name = format!("{prefix}_tex_{i}");
        let mut args = serde_json::Map::new();
        args.insert("source".into(), serde_json::Value::String(path.to_string()));
        args.insert("image_index".into(), serde_json::json!(i));
        if opts.texture_max_size > 0 {
            args.insert("max_size".into(), serde_json::json!(opts.texture_max_size));
        }
        entries.push(serde_json::json!({
            "name": name,
            "type": "Texture",
            "args": serde_json::Value::Object(args),
        }));
    }

    // Materials: one entry per glTF material, mapped onto Concinnity's PBR
    // subset. An asset that references a default (unnamed) glTF material would
    // need a fallback, so emit one extra "default" material the meshes fall
    // back to.
    let default_mat_name = format!("{prefix}_mat_default");
    entries.push(serde_json::json!({
        "name": default_mat_name,
        "type": "Material",
        "args": {
            "roughness": 0.8,
            "metallic": 0.0,
        }
    }));
    let material_names: Vec<String> = doc
        .doc
        .document
        .materials()
        .enumerate()
        .map(|(i, mat)| {
            let name = format!("{prefix}_mat_{i}");
            let mapped = gltf_material::map_material(prefix, &mat);
            if !mapped.extra_uv_sets.is_empty() {
                tracing::warn!(
                    "glTF material '{}' samples UV set(s) {:?}; only set 0 is imported, so those textures sample the wrong coordinates",
                    mat.name().unwrap_or(&name),
                    mapped.extra_uv_sets
                );
            }
            entries.push(serde_json::json!({
                "name": name,
                "type": "Material",
                "args": serde_json::Value::Object(mapped.args),
            }));
            name
        })
        .collect();

    // Meshes: one entry per primitive, flattened across glTF meshes in
    // declaration order. Every Mesh entry references the `.glb` by path with
    // no inline vertex/index data; the build's desugar pass parses the file
    // and fills the geometry in. A primitive that exceeds Concinnity's u16
    // index limit needs one Mesh per u16-safe chunk; we count the chunks
    // here (one geometry read per oversized primitive) and emit named
    // entries carrying the `chunk_index` the desugar pass will slice on.
    //
    // A mesh drawn only by skinned nodes expands as a SkinnedMesh instead, so
    // its primitives emit no static entries; the counter still advances over
    // them because `primitive_index` addresses the file's flattened primitive
    // list, which the desugar pass re-derives from the source.
    let skinned_nodes: Vec<gltf::Node<'_>> = doc
        .doc
        .document
        .nodes()
        .filter(|n| n.mesh().is_some() && n.skin().is_some())
        .collect();
    let drawn_static: HashSet<usize> = doc
        .doc
        .document
        .nodes()
        .filter(|n| n.skin().is_none())
        .filter_map(|n| n.mesh().map(|m| m.index()))
        .collect();
    let skinned_only: HashSet<usize> = skinned_nodes
        .iter()
        .filter_map(|n| n.mesh().map(|m| m.index()))
        .filter(|i| !drawn_static.contains(i))
        .collect();

    let mut primitive_counter: usize = 0;
    let mut mesh_to_submesh_refs: Vec<Vec<serde_json::Value>> = Vec::new();
    // First primitive's material per glTF mesh, for the rigs below.
    let mut mesh_first_material: Vec<String> = Vec::new();
    for gltf_mesh in doc.doc.document.meshes() {
        let mut submesh_refs: Vec<serde_json::Value> = Vec::new();
        let skip_static = skinned_only.contains(&gltf_mesh.index());
        for primitive in gltf_mesh.primitives() {
            let prim_idx = primitive_counter;
            primitive_counter += 1;

            let material_name = primitive
                .material()
                .index()
                .and_then(|i| material_names.get(i).cloned())
                .unwrap_or_else(|| default_mat_name.clone());
            if mesh_first_material.len() == gltf_mesh.index() {
                mesh_first_material.push(material_name.clone());
            }
            if skip_static {
                continue;
            }

            let vert_count = primitive
                .get(&gltf::Semantic::Positions)
                .map(|a| a.count())
                .unwrap_or(0);

            if vert_count <= U16_CAPACITY {
                let mesh_name = format!("{prefix}_prim_{prim_idx}");
                entries.push(serde_json::json!({
                    "name": mesh_name,
                    "type": "Mesh",
                    "args": {
                        "source": path,
                        "primitive_index": prim_idx,
                    }
                }));
                submesh_refs.push(serde_json::json!({
                    "mesh": mesh_name,
                    "material": material_name,
                }));
                continue;
            }

            // Oversized: parse geometry now solely to learn how many u16-safe
            // chunks the build will produce. Emit one Mesh per chunk by name;
            // each carries `chunk_index` so desugar can re-split the primitive
            // and pick the right slice: no inline data baked into world.jsonl.
            let (_, indices32) =
                crate::import::glb::read_primitive_geometry(&doc, path, prim_idx as u32)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let chunk_count = crate::import::glb::count_u16_chunks(&indices32);
            for chunk_idx in 0..chunk_count {
                let mesh_name = format!("{prefix}_prim_{prim_idx}_chunk_{chunk_idx}");
                entries.push(serde_json::json!({
                    "name": mesh_name,
                    "type": "Mesh",
                    "args": {
                        "source": path,
                        "primitive_index": prim_idx,
                        "chunk_index": chunk_idx,
                    }
                }));
                submesh_refs.push(serde_json::json!({
                    "mesh": mesh_name,
                    "material": material_name,
                }));
            }
        }
        if mesh_first_material.len() == gltf_mesh.index() {
            mesh_first_material.push(default_mat_name.clone());
        }
        mesh_to_submesh_refs.push(submesh_refs);
    }

    // Models: one entry per glTF mesh, grouping its primitives + materials.
    // A skinned-only mesh has no static entries to group.
    let model_names: Vec<Option<String>> = mesh_to_submesh_refs
        .iter()
        .enumerate()
        .map(|(i, submeshes)| {
            if skinned_only.contains(&i) {
                return None;
            }
            let name = format!("{prefix}_model_{i}");
            entries.push(serde_json::json!({
                "name": name,
                "type": "Model",
                "args": { "meshes": submeshes }
            }));
            Some(name)
        })
        .collect();

    // Props: walk the default scene graph. Mesh-bearing nodes become Props
    // with world-space transforms; transform-only nodes are flattened into
    // their descendants. This keeps the hierarchy simple (every emitted Prop
    // is independent) at the cost of losing the original parent links.
    // The walk also accumulates a world-space AABB so the camera can be
    // framed to the scene's actual scale.
    let scene = doc
        .doc
        .document
        .default_scene()
        .or_else(|| doc.doc.document.scenes().next());
    let mut scene_aabb: Option<([f32; 3], [f32; 3])> = None;
    if let Some(scene) = scene {
        let mut walk = SceneWalk {
            prefix,
            model_names: &model_names,
            prop_counter: 0,
            entries: &mut entries,
            aabb: &mut scene_aabb,
        };
        for root in scene.nodes() {
            walk.walk_node(&root, IDENTITY);
        }
    }

    // SkinnedMesh + Animation, one set per skinned node. A skinned node's own
    // transform is not applied to its geometry (the skin's joints place it),
    // so each part renders at the origin.
    if !skinned_nodes.is_empty() {
        let parts: Vec<SkinnedPart> = skinned_nodes
            .iter()
            .enumerate()
            .map(|(skin_index, node)| SkinnedPart {
                skin_index,
                material: node
                    .mesh()
                    .and_then(|m| mesh_first_material.get(m.index()).cloned())
                    .unwrap_or_else(|| default_mat_name.clone()),
            })
            .collect();
        let clips: Vec<String> = doc
            .doc
            .document
            .animations()
            .map(|a| a.name().unwrap_or("").to_string())
            .collect();
        entries.extend(rig_entries(prefix, path, &parts, &clips));
    }

    // Camera3D framed to the scene's world AABB.
    if opts.emit_camera
        && let Some(camera_entry) = framed_camera_entry(prefix, scene_aabb)
    {
        entries.push(camera_entry);
    }

    Ok(entries)
}

// Intern a texture path: emit one Texture asset per unique on-disk path and
// return its asset name. A non-zero `max_size` caps the longest edge so very
// large source maps don't bloat the compiled (uncompressed) blob.
fn intern_texture(
    path: &str,
    prefix: &str,
    max_size: u32,
    tex_names: &mut HashMap<String, String>,
    entries: &mut Vec<serde_json::Value>,
) -> String {
    if let Some(name) = tex_names.get(path) {
        return name.clone();
    }
    let name = format!("{prefix}_tex_{}", tex_names.len());
    let mut args = serde_json::Map::new();
    args.insert("source".into(), serde_json::Value::String(path.to_string()));
    if max_size > 0 {
        args.insert("max_size".into(), serde_json::json!(max_size));
    }
    entries.push(serde_json::json!({
        "name": name,
        "type": "Texture",
        "args": serde_json::Value::Object(args),
    }));
    tex_names.insert(path.to_string(), name.clone());
    name
}

// Read-only context plus the mutable accumulators for the scene-graph walk.
// `prop_counter` names emitted Props sequentially; `entries` collects the Prop
// JSON; `aabb` grows to the world-space bounds of every primitive visited.
struct SceneWalk<'a> {
    prefix: &'a str,
    model_names: &'a [Option<String>],
    prop_counter: usize,
    entries: &'a mut Vec<serde_json::Value>,
    aabb: &'a mut Option<([f32; 3], [f32; 3])>,
}

impl SceneWalk<'_> {
    // Recursively visit `node` with `parent_world` already composed in. For
    // mesh-bearing nodes emit a Prop entry; recurse with the updated world
    // matrix either way so children pick up the inherited transform. Also
    // accumulates `aabb` from each primitive's POSITION min/max so callers can
    // frame a camera to the scene without a second pass.
    fn walk_node(&mut self, node: &gltf::Node<'_>, parent_world: Mat4) {
        let local = node.transform().matrix();
        let world = mat4_mul(parent_world, local);

        // A skinned node expands to a SkinnedMesh, never a Prop: emitting both
        // would draw the character twice, once frozen in bind pose. Its bounds
        // still frame the camera, taken at the origin where the SkinnedMesh
        // renders rather than at the (unapplied) node transform.
        if let (Some(mesh), true) = (node.mesh(), node.skin().is_some()) {
            for prim in mesh.primitives() {
                let local_bbox = prim.bounding_box();
                for c in aabb_corners(local_bbox.min, local_bbox.max) {
                    expand_aabb(self.aabb, c);
                }
            }
        } else if let Some(mesh) = node.mesh() {
            let mesh_idx = mesh.index();
            if let Some(Some(model_name)) = self.model_names.get(mesh_idx) {
                let (t, q, s) = decompose(world);
                let rotation_deg = euler_yxz_from_quat(q);
                let idx = self.prop_counter;
                self.prop_counter += 1;
                let prefix = self.prefix;
                let prop_name = node
                    .name()
                    .map(|n| format!("{prefix}_{}", sanitize_name(n)))
                    .unwrap_or_else(|| format!("{prefix}_node_{idx}"));
                self.entries.push(serde_json::json!({
                    "name": prop_name,
                    "type": "Prop",
                    "args": {
                        "model": model_name,
                        "position": [t[0], t[1], t[2]],
                        "rotation_deg": [rotation_deg[0], rotation_deg[1], rotation_deg[2]],
                        "scale": [s[0], s[1], s[2]],
                    }
                }));

                // Expand each primitive's local POSITION AABB into world space
                // by transforming all eight corners: works for any rotation /
                // non-uniform scale combination.
                for prim in mesh.primitives() {
                    let local_bbox = prim.bounding_box();
                    let corners = aabb_corners(local_bbox.min, local_bbox.max);
                    for c in corners {
                        let w = transform_point(world, c);
                        expand_aabb(self.aabb, w);
                    }
                }
            }
        }

        for child in node.children() {
            self.walk_node(&child, world);
        }
    }
}

fn aabb_corners(min: [f32; 3], max: [f32; 3]) -> [[f32; 3]; 8] {
    [
        [min[0], min[1], min[2]],
        [max[0], min[1], min[2]],
        [min[0], max[1], min[2]],
        [max[0], max[1], min[2]],
        [min[0], min[1], max[2]],
        [max[0], min[1], max[2]],
        [min[0], max[1], max[2]],
        [max[0], max[1], max[2]],
    ]
}

fn transform_point(m: Mat4, p: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
        m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
        m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2],
    ]
}

fn expand_aabb(aabb: &mut Option<([f32; 3], [f32; 3])>, p: [f32; 3]) {
    match aabb {
        None => *aabb = Some((p, p)),
        Some((min, max)) => {
            for i in 0..3 {
                if p[i] < min[i] {
                    min[i] = p[i];
                }
                if p[i] > max[i] {
                    max[i] = p[i];
                }
            }
        }
    }
}

// Build a Camera3D framed to look at the centre of `aabb` from slightly above
// and in front. Returns `None` for a degenerate (empty) scene: there is
// nothing to frame, so the runtime falls back to whatever Camera3D the world
// authored (or none at all).
//
// FOV is fixed at 60 degrees vertical and the orbit distance fits the bounding
// sphere of the AABB at that FOV, with a 1.4x margin so the scene doesn't touch
// the frame edges. Works for any scale: a 0.5 m chess board and a 50 m building
// both land in view from a sensible viewpoint.
fn framed_camera_entry(
    prefix: &str,
    aabb: Option<([f32; 3], [f32; 3])>,
) -> Option<serde_json::Value> {
    let (min, max) = aabb?;
    let center = [
        0.5 * (min[0] + max[0]),
        0.5 * (min[1] + max[1]),
        0.5 * (min[2] + max[2]),
    ];
    let size = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    let radius = 0.5 * (size[0] * size[0] + size[1] * size[1] + size[2] * size[2]).sqrt();
    if !radius.is_finite() || radius <= 0.0 {
        return None;
    }
    let fov_y_degrees = 60.0_f32;
    let half_fov = fov_y_degrees.to_radians() * 0.5;
    let distance = (radius * 1.4) / half_fov.sin();
    let height_above = radius * 0.6;

    // Camera looks down -Z (yaw=0); place it on the +Z side of the centre.
    let pos = [center[0], center[1] + height_above, center[2] + distance];
    let pitch = -(height_above / distance).atan();
    // Near/far framed around the orbit distance so we don't clip the scene.
    let near = (radius * 0.05).max(0.01);
    let far = (distance + radius) * 4.0;

    Some(serde_json::json!({
        "name": format!("{prefix}_cam"),
        "type": "Camera3D",
        "args": {
            "fov_y_degrees": fov_y_degrees,
            "near": near,
            "far": far,
            "yaw": 0.0,
            "pitch": pitch,
            "position": [pos[0], pos[1], pos[2]],
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_name_lowercases_and_replaces_punctuation() {
        assert_eq!(sanitize_name("BistroExterior"), "bistroexterior");
        assert_eq!(sanitize_name("My-Cool.Asset"), "my_cool_asset");
        assert_eq!(sanitize_name(""), "scene");
    }

    #[test]
    fn unsupported_format_errors() {
        let err = entries_from_scene("model.obj", &ImportOptions::default(), None)
            .expect_err("'.obj' is not a SceneImport container format");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains(".obj"));
    }

    #[test]
    fn intern_texture_reuses_name_and_honors_max_size() {
        let mut names: HashMap<String, String> = HashMap::new();
        let mut entries: Vec<serde_json::Value> = Vec::new();

        let a = intern_texture("wall.dds", "scn", 512, &mut names, &mut entries);
        let b = intern_texture("wall.dds", "scn", 512, &mut names, &mut entries);
        // Same path interns to the same name and emits only one Texture entry.
        assert_eq!(a, b);
        assert_eq!(a, "scn_tex_0");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["args"]["max_size"], serde_json::json!(512));

        // A distinct path gets the next index; max_size 0 omits the cap.
        let c = intern_texture("floor.dds", "scn", 0, &mut names, &mut entries);
        assert_eq!(c, "scn_tex_1");
        assert_eq!(entries.len(), 2);
        assert!(entries[1]["args"].get("max_size").is_none());
    }

    #[test]
    fn framed_camera_none_for_degenerate_aabb() {
        assert!(framed_camera_entry("x", None).is_none());
        // Zero-size AABB has no radius.
        assert!(framed_camera_entry("x", Some(([0.0; 3], [0.0; 3]))).is_none());
    }

    #[test]
    fn framed_camera_frames_a_box() {
        let cam = framed_camera_entry("scene", Some(([-1.0; 3], [1.0; 3]))).unwrap();
        assert_eq!(cam["type"], "Camera3D");
        assert_eq!(cam["name"], "scene_cam");
    }

    #[test]
    fn framed_camera_sits_above_and_in_front_looking_down() {
        let cam = framed_camera_entry("s", Some(([-1.0; 3], [1.0; 3]))).unwrap();
        let args = &cam["args"];
        assert_eq!(args["fov_y_degrees"], serde_json::json!(60.0));

        let pos: Vec<f64> = args["position"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        // Centre is the origin: the camera sits above (y > 0) and on the +Z
        // side, pitched down toward the scene.
        assert!(pos[1] > 0.0);
        assert!(pos[2] > 1.0);
        assert!(args["pitch"].as_f64().unwrap() < 0.0);

        let near = args["near"].as_f64().unwrap();
        let far = args["far"].as_f64().unwrap();
        assert!(near > 0.0);
        assert!(far > near);
    }

    #[test]
    fn transform_point_applies_rotation_scale_and_translation() {
        // Column-major affine: columns scale x by 2 and translate by (10, 20, 30).
        let m: Mat4 = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [10.0, 20.0, 30.0, 1.0],
        ];
        assert_eq!(transform_point(m, [1.0, 2.0, 3.0]), [12.0, 22.0, 33.0]);
        assert_eq!(transform_point(IDENTITY, [4.0, 5.0, 6.0]), [4.0, 5.0, 6.0]);
    }

    #[test]
    fn expand_aabb_grows_to_cover_every_point() {
        let mut aabb = None;
        expand_aabb(&mut aabb, [1.0, 2.0, 3.0]);
        assert_eq!(aabb, Some(([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])));

        expand_aabb(&mut aabb, [-1.0, 5.0, 3.0]);
        assert_eq!(aabb, Some(([-1.0, 2.0, 3.0], [1.0, 5.0, 3.0])));

        // A point inside the current bounds changes nothing.
        expand_aabb(&mut aabb, [0.0, 3.0, 3.0]);
        assert_eq!(aabb, Some(([-1.0, 2.0, 3.0], [1.0, 5.0, 3.0])));
    }

    #[test]
    fn aabb_corners_enumerates_all_eight() {
        let corners = aabb_corners([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]);
        let mut unique: Vec<String> = corners.iter().map(|c| format!("{c:?}")).collect();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), 8);
        for c in corners {
            assert!(c[0] == 0.0 || c[0] == 1.0);
            assert!(c[1] == 0.0 || c[1] == 2.0);
            assert!(c[2] == 0.0 || c[2] == 3.0);
        }
    }

    // Assemble a valid binary glTF container: 12-byte header, a JSON chunk
    // padded with spaces, and a BIN chunk padded with zeros.
    fn glb_bytes(json: &str, bin: &[u8]) -> Vec<u8> {
        let mut json_bytes = json.as_bytes().to_vec();
        while !json_bytes.len().is_multiple_of(4) {
            json_bytes.push(b' ');
        }
        let mut bin_bytes = bin.to_vec();
        while !bin_bytes.len().is_multiple_of(4) {
            bin_bytes.push(0);
        }
        let total = 12 + 8 + json_bytes.len() + 8 + bin_bytes.len();
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(b"glTF");
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&(total as u32).to_le_bytes());
        out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(b"JSON");
        out.extend_from_slice(&json_bytes);
        out.extend_from_slice(&(bin_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(b"BIN\0");
        out.extend_from_slice(&bin_bytes);
        out
    }

    // One scene holding a transform-only pivot at (0,0,1) whose child node
    // "Crate Box" sits at (1,2,3) and carries a single triangle primitive of
    // `vertex_count` positions plus one PBR material. Growing `vertex_count`
    // past the u16 limit exercises the chunking path.
    fn triangle_glb(vertex_count: usize) -> Vec<u8> {
        triangle_glb_with_mode(vertex_count, 4)
    }

    // As above, with an explicit glTF primitive mode (4 = TRIANGLES).
    fn triangle_glb_with_mode(vertex_count: usize, mode: u32) -> Vec<u8> {
        let mut bin = Vec::with_capacity(vertex_count * 12 + 12);
        for i in 0..vertex_count {
            let p: [f32; 3] = match i {
                0 => [0.0, 0.0, 0.0],
                1 => [1.0, 0.0, 0.0],
                2 => [0.0, 1.0, 0.0],
                _ => [0.0; 3],
            };
            for c in p {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        let pos_len = bin.len();
        for idx in [0u32, 1, 2] {
            bin.extend_from_slice(&idx.to_le_bytes());
        }
        let json = format!(
            r#"{{
  "asset": {{"version": "2.0"}},
  "scene": 0,
  "scenes": [{{"nodes": [0]}}],
  "nodes": [
    {{"name": "Pivot", "children": [1], "translation": [0, 0, 1]}},
    {{"mesh": 0, "name": "Crate Box", "translation": [1, 2, 3]}}
  ],
  "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0}}, "indices": 1, "material": 0, "mode": {mode}}}]}}],
  "materials": [{{"pbrMetallicRoughness": {{"baseColorFactor": [0.5, 0.25, 0.125, 1.0], "metallicFactor": 0.5, "roughnessFactor": 0.25}}, "emissiveFactor": [0.5, 0.0, 0.0]}}],
  "accessors": [
    {{"bufferView": 0, "componentType": 5126, "count": {vc}, "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 0]}},
    {{"bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR"}}
  ],
  "bufferViews": [
    {{"buffer": 0, "byteOffset": 0, "byteLength": {pos_len}}},
    {{"buffer": 0, "byteOffset": {pos_len}, "byteLength": 12}}
  ],
  "buffers": [{{"byteLength": {total}}}]
}}"#,
            vc = vertex_count,
            mode = mode,
            pos_len = pos_len,
            total = bin.len(),
        );
        glb_bytes(&json, &bin)
    }

    fn find<'a>(entries: &'a [serde_json::Value], name: &str, ty: &str) -> &'a serde_json::Value {
        entries
            .iter()
            .find(|e| e["name"] == name && e["type"] == ty)
            .unwrap_or_else(|| panic!("expected entry {name} of type {ty} in {entries:#?}"))
    }

    #[test]
    fn glb_scene_expands_to_materials_meshes_models_props_and_camera() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tri.glb");
        std::fs::write(&path, triangle_glb(3)).unwrap();
        let path = path.to_str().unwrap();

        let opts = ImportOptions {
            name_prefix: "scn".to_string(),
            ..ImportOptions::default()
        };
        let entries = entries_from_scene(path, &opts, None).expect("expand glb");

        // Material 0 maps the glTF PBR factors; the powers of two used in the
        // fixture are exact in f32/f64, so equality holds.
        let mat = find(&entries, "scn_mat_0", "Material");
        assert_eq!(mat["args"]["tint"], serde_json::json!([0.5, 0.25, 0.125]));
        assert_eq!(mat["args"]["metallic"], serde_json::json!(0.5));
        assert_eq!(mat["args"]["roughness"], serde_json::json!(0.25));
        assert_eq!(
            mat["args"]["emissive_factor"],
            serde_json::json!([0.5, 0.0, 0.0])
        );

        // A fallback material is always emitted for unassigned primitives.
        find(&entries, "scn_mat_default", "Material");

        // The mesh entry references the source by path with no inline data.
        let mesh = find(&entries, "scn_prim_0", "Mesh");
        assert_eq!(mesh["args"]["source"], serde_json::json!(path));
        assert_eq!(mesh["args"]["primitive_index"], serde_json::json!(0));
        assert!(mesh["args"].get("chunk_index").is_none());
        assert!(mesh["args"].get("vertices").is_none());

        // The model groups the primitive with its material.
        let model = find(&entries, "scn_model_0", "Model");
        assert_eq!(model["args"]["meshes"][0]["mesh"], "scn_prim_0");
        assert_eq!(model["args"]["meshes"][0]["material"], "scn_mat_0");

        // The mesh-bearing node becomes a Prop named from its sanitized node
        // name, flattened to a world transform: its own (1,2,3) composed with
        // the transform-only parent pivot's (0,0,1). The pivot itself emits
        // nothing.
        let prop = find(&entries, "scn_crate_box", "Prop");
        assert_eq!(prop["args"]["model"], "scn_model_0");
        assert_eq!(prop["args"]["position"], serde_json::json!([1.0, 2.0, 4.0]));
        assert!(entries.iter().all(|e| e["name"] != "scn_pivot"));

        // A camera is framed to the world-space AABB.
        let cam = find(&entries, "scn_cam", "Camera3D");
        assert!(cam["args"]["position"][2].as_f64().unwrap() > 4.0);
    }

    #[test]
    fn glb_skinned_nodes_expand_to_skinned_meshes_and_clips_not_props() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hero.glb");
        std::fs::write(&path, crate::import::glb::test_fixtures::two_skin_glb()).unwrap();
        let path = path.to_str().unwrap();

        let opts = ImportOptions {
            name_prefix: "hero".to_string(),
            ..ImportOptions::default()
        };
        let entries = entries_from_scene(path, &opts, None).expect("expand glb");

        // One SkinnedMesh per skinned node, each selecting its own part and
        // taking its mesh's material.
        let body = find(&entries, "hero_skin_0", "SkinnedMesh");
        assert_eq!(body["args"]["source"], serde_json::json!(path));
        assert_eq!(body["args"]["skin_index"], serde_json::json!(0));
        assert_eq!(body["args"]["material"], "hero_mat_0");
        let hair = find(&entries, "hero_skin_1", "SkinnedMesh");
        assert_eq!(hair["args"]["skin_index"], serde_json::json!(1));
        assert_eq!(hair["args"]["material"], "hero_mat_1");

        // Every part gets the clip, targeting its own mesh.
        let body_clip = find(&entries, "hero_anim_0_wave_0", "Animation");
        assert_eq!(body_clip["args"]["target"], "hero_skin_0");
        assert_eq!(body_clip["args"]["source"], serde_json::json!(path));
        let hair_clip = find(&entries, "hero_anim_1_wave_0", "Animation");
        assert_eq!(hair_clip["args"]["target"], "hero_skin_1");

        // The skinned geometry must not also expand statically: a Prop beside
        // the SkinnedMesh would draw the character twice, once in bind pose.
        assert!(
            entries.iter().all(|e| e["type"] != "Prop"),
            "skinned nodes emit no Prop: {entries:#?}"
        );
        assert!(entries.iter().all(|e| e["type"] != "Mesh"));
        assert!(entries.iter().all(|e| e["type"] != "Model"));

        // The character still bounds a camera, framed where it renders.
        find(&entries, "hero_cam", "Camera3D");
    }

    #[test]
    fn glb_mesh_drawn_by_both_a_skinned_and_a_static_node_keeps_its_static_entries() {
        // Only geometry drawn *exclusively* by skinned nodes drops its static
        // entries; a mesh a plain node also draws still needs them.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mixed.glb");
        let mut json = crate::import::glb::test_fixtures::two_skin_json();
        json["nodes"] = serde_json::json!([
            {"mesh": 0, "skin": 0, "name": "body"},
            {"name": "root", "children": [2], "translation": [0.0, 1.0, 0.0]},
            {"name": "tip", "translation": [0.0, 0.5, 0.0]},
            {"mesh": 1, "skin": 0, "name": "hair"},
            {"mesh": 1, "name": "hair_prop", "translation": [9.0, 0.0, 0.0]}
        ]);
        json["scenes"] = serde_json::json!([{"nodes": [0, 1, 3, 4]}]);
        let bytes = crate::import::glb::test_fixtures::make_glb(
            &json,
            Some(&crate::import::glb::test_fixtures::two_skin_bin()),
        );
        std::fs::write(&path, bytes).unwrap();

        let opts = ImportOptions {
            name_prefix: "mix".to_string(),
            ..ImportOptions::default()
        };
        let entries = entries_from_scene(path.to_str().unwrap(), &opts, None).expect("expand glb");

        // Mesh 1 keeps its static entries for the plain node...
        find(&entries, "mix_prim_1", "Mesh");
        find(&entries, "mix_model_1", "Model");
        let prop = find(&entries, "mix_hair_prop", "Prop");
        assert_eq!(prop["args"]["model"], "mix_model_1");
        // ...while mesh 0, drawn only by the skinned body, drops them. Its
        // primitive index is still 0, so `mix_prim_1` addresses the file's
        // second primitive as it always did.
        assert!(entries.iter().all(|e| e["name"] != "mix_prim_0"));
        assert!(entries.iter().all(|e| e["name"] != "mix_model_0"));
        // Both skinned parts still expand.
        find(&entries, "mix_skin_0", "SkinnedMesh");
        find(&entries, "mix_skin_1", "SkinnedMesh");
    }

    #[test]
    fn glb_without_skins_generates_no_rig_entries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tri.glb");
        std::fs::write(&path, triangle_glb(3)).unwrap();

        let entries = entries_from_scene(path.to_str().unwrap(), &ImportOptions::default(), None)
            .expect("expand");
        assert!(entries.iter().all(|e| e["type"] != "SkinnedMesh"));
        assert!(entries.iter().all(|e| e["type"] != "Animation"));
    }

    #[test]
    fn glb_skin_without_clips_generates_the_mesh_only() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bind.glb");
        let bytes = crate::import::glb::test_fixtures::make_glb(
            &crate::import::glb::test_fixtures::skinned_json(true, true, false),
            Some(&crate::import::glb::test_fixtures::skinned_bin()),
        );
        std::fs::write(&path, bytes).unwrap();

        let opts = ImportOptions {
            name_prefix: "rig".to_string(),
            ..ImportOptions::default()
        };
        let entries = entries_from_scene(path.to_str().unwrap(), &opts, None).expect("expand");
        find(&entries, "rig_skin_0", "SkinnedMesh");
        assert!(entries.iter().all(|e| e["type"] != "Animation"));
    }

    #[test]
    fn glb_scene_with_emit_camera_false_omits_the_camera() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tri.glb");
        std::fs::write(&path, triangle_glb(3)).unwrap();

        let opts = ImportOptions {
            name_prefix: "scn".to_string(),
            emit_camera: false,
            ..ImportOptions::default()
        };
        let entries = entries_from_scene(path.to_str().unwrap(), &opts, None).expect("expand glb");
        assert!(entries.iter().all(|e| e["type"] != "Camera3D"));
    }

    #[test]
    fn glb_oversized_primitive_fans_into_chunked_meshes() {
        // One more vertex than the u16 index space forces the chunk path.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big.glb");
        std::fs::write(&path, triangle_glb(U16_CAPACITY + 1)).unwrap();
        let path = path.to_str().unwrap();

        let opts = ImportOptions {
            name_prefix: "big".to_string(),
            ..ImportOptions::default()
        };
        let entries = entries_from_scene(path, &opts, None).expect("expand glb");

        // The single triangle re-chunks into one u16-safe slice; the entry
        // carries the chunk index and there is no unchunked mesh entry.
        let mesh = find(&entries, "big_prim_0_chunk_0", "Mesh");
        assert_eq!(mesh["args"]["chunk_index"], serde_json::json!(0));
        assert!(entries.iter().all(|e| e["name"] != "big_prim_0"));

        let model = find(&entries, "big_model_0", "Model");
        assert_eq!(model["args"]["meshes"][0]["mesh"], "big_prim_0_chunk_0");
    }

    #[test]
    fn glb_without_a_scene_emits_geometry_but_no_props_or_camera() {
        // A glTF may omit the scene graph entirely. There is then no node to
        // place, so no Prop is emitted and nothing bounds a camera, but the
        // mesh and material entries still expand.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("no_scene.glb");
        let json = r#"{
  "asset": {"version": "2.0"},
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 0]},
    {"bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR"}
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36},
    {"buffer": 0, "byteOffset": 36, "byteLength": 12}
  ],
  "buffers": [{"byteLength": 48}]
}"#;
        let mut bin: Vec<u8> = Vec::new();
        for c in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bin.extend_from_slice(&c.to_le_bytes());
        }
        for idx in [0u32, 1, 2] {
            bin.extend_from_slice(&idx.to_le_bytes());
        }
        std::fs::write(&path, glb_bytes(json, &bin)).unwrap();

        let opts = ImportOptions {
            name_prefix: "scn".to_string(),
            ..ImportOptions::default()
        };
        let entries = entries_from_scene(path.to_str().unwrap(), &opts, None).expect("expand glb");
        find(&entries, "scn_prim_0", "Mesh");
        find(&entries, "scn_model_0", "Model");
        assert!(entries.iter().all(|e| e["type"] != "Prop"));
        assert!(entries.iter().all(|e| e["type"] != "Camera3D"));
    }

    #[test]
    fn glb_oversized_primitive_with_unsupported_topology_errors() {
        // Counting the chunks of an oversized primitive is the one geometry
        // read the expansion performs, so its failure must surface as an error
        // rather than a silently empty mesh list.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("points.glb");
        std::fs::write(&path, triangle_glb_with_mode(U16_CAPACITY + 1, 0)).unwrap();

        let err = entries_from_scene(path.to_str().unwrap(), &ImportOptions::default(), None)
            .expect_err("POINTS topology is unsupported");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("only TRIANGLES is supported"),
            "got: {err}"
        );
    }

    #[test]
    fn glb_extension_dispatch_is_case_insensitive() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("upper.GLB");
        std::fs::write(&path, triangle_glb(3)).unwrap();

        let entries = entries_from_scene(path.to_str().unwrap(), &ImportOptions::default(), None)
            .expect("expand");
        assert!(entries.iter().any(|e| e["type"] == "Mesh"));
    }

    #[test]
    fn missing_glb_file_errors_with_the_path() {
        let err = entries_from_scene("/no/such/scene.glb", &ImportOptions::default(), None)
            .expect_err("missing file");
        assert!(err.to_string().contains("/no/such/scene.glb"));
    }

    #[test]
    fn corrupt_glb_errors_as_invalid_data() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("junk.glb");
        std::fs::write(&path, b"definitely not a glb").unwrap();

        let err = entries_from_scene(path.to_str().unwrap(), &ImportOptions::default(), None)
            .expect_err("corrupt file");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("not a valid glTF"));
    }

    #[test]
    fn missing_fbx_file_errors() {
        let err = entries_from_scene("/no/such/scene.fbx", &ImportOptions::default(), None)
            .expect_err("missing file");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("/no/such/scene.fbx"));
    }

    // Like `triangle_glb`, but with one embedded image and a material that
    // binds it as both base-color and normal texture, exercising the images()
    // loop and the texture-reference branches in `entries_from_glb`.
    fn textured_triangle_glb() -> Vec<u8> {
        let mut bin = Vec::new();
        for p in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            for c in p {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        let pos_len = bin.len();
        for idx in [0u32, 1, 2] {
            bin.extend_from_slice(&idx.to_le_bytes());
        }
        let idx_end = bin.len();
        // The image bytes are never decoded during expansion (only a Texture
        // entry pointing at the source is emitted), so arbitrary padding stands
        // in for the encoded pixels. Its odd length leaves the binary chunk
        // needing tail padding.
        bin.extend_from_slice(&[0u8; 14]);
        let json = format!(
            r#"{{
  "asset": {{"version": "2.0"}},
  "scene": 0,
  "scenes": [{{"nodes": [0, 1]}}],
  "nodes": [
    {{"mesh": 0, "name": "Crate", "translation": [0, 0, 0]}},
    {{"mesh": 0, "translation": [5, 0, 0]}}
  ],
  "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0}}, "indices": 1, "material": 0, "mode": 4}}]}}],
  "materials": [{{
    "pbrMetallicRoughness": {{
      "baseColorTexture": {{"index": 0}},
      "metallicRoughnessTexture": {{"index": 1}},
      "metallicFactor": 0.0,
      "roughnessFactor": 1.0
    }},
    "normalTexture": {{"index": 0}},
    "emissiveTexture": {{"index": 2}},
    "emissiveFactor": [1.0, 1.0, 1.0],
    "occlusionTexture": {{"index": 3}}
  }}, {{
    "pbrMetallicRoughness": {{"baseColorTexture": {{"index": 0}}}},
    "alphaMode": "MASK",
    "alphaCutoff": 0.25
  }}, {{
    "pbrMetallicRoughness": {{"baseColorTexture": {{"index": 0}}}},
    "alphaMode": "BLEND"
  }}, {{
    "pbrMetallicRoughness": {{"baseColorTexture": {{"index": 0, "texCoord": 1}}}}
  }}],
  "textures": [{{"source": 0}}, {{"source": 1}}, {{"source": 2}}, {{"source": 3}}],
  "images": [
    {{"bufferView": 2, "mimeType": "image/png"}},
    {{"bufferView": 2, "mimeType": "image/png"}},
    {{"bufferView": 2, "mimeType": "image/png"}},
    {{"bufferView": 2, "mimeType": "image/png"}}
  ],
  "accessors": [
    {{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 0]}},
    {{"bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR"}}
  ],
  "bufferViews": [
    {{"buffer": 0, "byteOffset": 0, "byteLength": {pos_len}}},
    {{"buffer": 0, "byteOffset": {pos_len}, "byteLength": 12}},
    {{"buffer": 0, "byteOffset": {idx_end}, "byteLength": 14}}
  ],
  "buffers": [{{"byteLength": {total}}}]
}}"#,
            pos_len = pos_len,
            idx_end = idx_end,
            total = bin.len(),
        );
        glb_bytes(&json, &bin)
    }

    // Minimal binary FBX writer: emits only the nodes `crate::import::fbx::parse_fbx`
    // reads, so a fixture scene can be assembled without a checked-in file.
    struct FbxWriter {
        writer: fbxcel::writer::v7400::binary::Writer<std::io::Cursor<Vec<u8>>>,
    }

    impl FbxWriter {
        fn new() -> Self {
            Self {
                writer: fbxcel::writer::v7400::binary::Writer::new(
                    std::io::Cursor::new(Vec::new()),
                    fbxcel::low::FbxVersion::V7_4,
                )
                .expect("fbx writer"),
            }
        }

        fn open(&mut self, name: &str) {
            self.writer.new_node(name).expect("open node");
        }

        fn close(&mut self) {
            self.writer.close_node().expect("close node");
        }

        // Object header: id, then the `Name\0\u{1}Class` pair exporters emit.
        fn open_object(&mut self, class: &str, id: i64, name: &str) {
            let mut attrs = self.writer.new_node(class).expect("open object");
            attrs.append_i64(id).expect("object id");
            attrs
                .append_string_direct(&format!("{name}\u{0}\u{1}{class}"))
                .expect("object name");
            attrs.append_string_direct("").expect("object subclass");
        }

        fn text(&mut self, name: &str, value: &str) {
            let mut attrs = self.writer.new_node(name).expect("open node");
            attrs.append_string_direct(value).expect("text value");
            self.close();
        }

        fn f64_array(&mut self, name: &str, values: &[f64]) {
            let mut attrs = self.writer.new_node(name).expect("open node");
            attrs
                .append_arr_f64_from_iter(None, values.iter().copied())
                .expect("f64 array");
            self.close();
        }

        fn i32_array(&mut self, name: &str, values: &[i32]) {
            let mut attrs = self.writer.new_node(name).expect("open node");
            attrs
                .append_arr_i32_from_iter(None, values.iter().copied())
                .expect("i32 array");
            self.close();
        }

        // A `Properties70` entry: name, type, subtype, flags, then values.
        fn prop_header(
            &mut self,
            name: &str,
            ty: &str,
        ) -> fbxcel::writer::v7400::binary::AttributesWriter<'_, std::io::Cursor<Vec<u8>>> {
            let mut attrs = self.writer.new_node("P").expect("open P");
            for field in [name, ty, "", "A"] {
                attrs.append_string_direct(field).expect("P field");
            }
            attrs
        }

        fn prop_vec3(&mut self, name: &str, value: [f64; 3]) {
            let mut attrs = self.prop_header(name, "Vector3D");
            for v in value {
                attrs.append_f64(v).expect("P value");
            }
            self.close();
        }

        fn prop_scalar(&mut self, name: &str, value: f64) {
            let mut attrs = self.prop_header(name, "double");
            attrs.append_f64(value).expect("P value");
            self.close();
        }

        fn connect(&mut self, kind: &str, child: i64, parent: i64, property: Option<&str>) {
            let mut attrs = self.writer.new_node("C").expect("open C");
            attrs.append_string_direct(kind).expect("connection kind");
            attrs.append_i64(child).expect("child id");
            attrs.append_i64(parent).expect("parent id");
            if let Some(property) = property {
                attrs.append_string_direct(property).expect("property");
            }
            self.close();
        }

        fn write(self, path: &Path) {
            let sink = self
                .writer
                .finalize_and_flush(&Default::default())
                .expect("finalize fbx");
            std::fs::write(path, sink.into_inner()).expect("write fbx");
        }
    }

    // A two-model scene: "Crate Box" carries a quad split across an opaque and
    // a glass material slot; the second model is unnamed and has no material.
    // Two further materials cover the frosted and named-glass paths.
    fn write_scene_fbx(path: &Path) {
        let mut f = FbxWriter::new();
        f.open("Objects");

        f.open_object("Geometry", 100, "CubeMesh");
        f.f64_array(
            "Vertices",
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        );
        // Two triangles; the last corner of each polygon is stored complemented.
        f.i32_array("PolygonVertexIndex", &[0, 1, !2, 1, 3, !2]);
        f.open("LayerElementMaterial");
        f.text("MappingInformationType", "ByPolygon");
        f.i32_array("Materials", &[0, 1]);
        f.close();
        f.close();

        f.open_object("Model", 200, "Crate Box");
        f.open("Properties70");
        f.prop_vec3("Lcl Translation", [1.0, 2.0, 3.0]);
        f.close();
        f.close();

        f.open_object("Geometry", 101, "PlaneMesh");
        f.f64_array("Vertices", &[0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0]);
        f.i32_array("PolygonVertexIndex", &[0, 1, !2]);
        f.close();

        f.open_object("Model", 201, "");
        f.close();

        f.open_object("Material", 300, "Stone");
        f.open("Properties70");
        f.prop_vec3("DiffuseColor", [0.5, 0.25, 0.125]);
        f.close();
        f.close();

        f.open_object("Material", 301, "GlassPane");
        f.open("Properties70");
        f.prop_scalar("Opacity", 0.25);
        f.prop_vec3("EmissiveColor", [0.25, 0.5, 0.75]);
        f.close();
        f.close();

        f.open_object("Material", 302, "FrostedGlass");
        f.close();

        f.open_object("Material", 303, "WindowPane");
        f.open("Properties70");
        f.prop_scalar("TransparencyFactor", 0.0);
        f.close();
        f.close();

        for (id, file) in [
            (400, "tex/stone_albedo.png"),
            (401, "tex/stone_normal.png"),
            (402, "tex/stone_orm.png"),
            (403, "tex/stone_emissive.png"),
            (404, "tex/glass_orm.png"),
        ] {
            f.open_object("Texture", id, file);
            f.text("RelativeFilename", file);
            f.close();
        }
        f.close();

        f.open("Connections");
        f.connect("OO", 100, 200, None);
        f.connect("OO", 200, 0, None);
        f.connect("OO", 300, 200, None);
        f.connect("OO", 301, 200, None);
        f.connect("OO", 101, 201, None);
        f.connect("OO", 201, 0, None);
        f.connect("OP", 400, 300, Some("DiffuseColor"));
        f.connect("OP", 401, 300, Some("NormalMap"));
        f.connect("OP", 402, 300, Some("SpecularColor"));
        f.connect("OP", 403, 300, Some("EmissiveColor"));
        f.connect("OP", 404, 301, Some("SpecularColor"));
        f.close();

        f.write(path);
    }

    fn scene_opts() -> ImportOptions {
        ImportOptions {
            name_prefix: "scn".to_string(),
            texture_max_size: 256,
            emissive_map_strength: 2.5,
            emit_camera: true,
        }
    }

    #[test]
    fn fbx_materials_map_textures_factors_and_glass_onto_the_pbr_subset() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("scene.fbx");
        write_scene_fbx(&path);
        let entries =
            entries_from_scene(path.to_str().unwrap(), &scene_opts(), None).expect("expand");

        // Each referenced texture is interned once, in material/slot order, and
        // resolves relative to the .fbx directory.
        let tex = find(&entries, "scn_tex_0", "Texture");
        assert_eq!(
            tex["args"]["source"],
            serde_json::json!(
                dir.path()
                    .join("tex/stone_albedo.png")
                    .to_string_lossy()
                    .replace('\\', "/")
            )
        );
        assert_eq!(tex["args"]["max_size"], serde_json::json!(256));

        let stone = find(&entries, "scn_mat_0", "Material");
        assert_eq!(stone["args"]["albedo"], "scn_tex_0");
        assert_eq!(stone["args"]["normal_map"], "scn_tex_1");
        assert_eq!(stone["args"]["orm_map"], "scn_tex_2");
        assert_eq!(stone["args"]["emissive_map"], "scn_tex_3");
        assert_eq!(stone["args"]["tint"], serde_json::json!([0.5, 0.25, 0.125]));
        // A textured emissive drives the glow from the import option, not the
        // FBX factor.
        assert_eq!(
            stone["args"]["emissive_factor"],
            serde_json::json!([2.5, 2.5, 2.5])
        );
        assert_eq!(stone["args"]["roughness"], serde_json::json!(0.7));
        assert_eq!(stone["args"]["metallic"], serde_json::json!(0.0));
        assert!(stone["args"].get("transparent").is_none());

        // Glass keeps its FBX emissive factor, drops the packed ORM map, and
        // turns smooth + transparent.
        let glass = find(&entries, "scn_mat_1", "Material");
        assert!(glass["args"].get("orm_map").is_none());
        assert!(
            !entries.iter().any(|e| e["args"]["source"]
                .as_str()
                .is_some_and(|s| s.contains("glass_orm"))),
            "the dropped glass ORM map must not be interned"
        );
        assert_eq!(
            glass["args"]["emissive_factor"],
            serde_json::json!([0.25, 0.5, 0.75])
        );
        assert_eq!(glass["args"]["roughness"], serde_json::json!(0.08));
        assert_eq!(glass["args"]["opacity"], serde_json::json!(0.25));
        assert_eq!(glass["args"]["transparent"], serde_json::json!(true));

        // Frosted glass stays a rough opaque surface.
        let frosted = find(&entries, "scn_mat_2", "Material");
        assert_eq!(frosted["args"]["roughness"], serde_json::json!(0.7));
        assert!(frosted["args"].get("transparent").is_none());

        // An opaque material named like a window is glass with a default
        // see-through amount.
        let window = find(&entries, "scn_mat_3", "Material");
        assert_eq!(window["args"]["roughness"], serde_json::json!(0.08));
        assert_eq!(window["args"]["opacity"], serde_json::json!(0.25));

        // The fallback material unassigned primitives use.
        let default_mat = find(&entries, "scn_mat_default", "Material");
        assert_eq!(default_mat["args"]["roughness"], serde_json::json!(0.8));
    }

    #[test]
    fn fbx_scene_expands_to_meshes_models_props_and_a_framed_camera() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("scene.fbx");
        write_scene_fbx(&path);
        let source = path.to_str().unwrap();
        let entries = entries_from_scene(source, &scene_opts(), None).expect("expand");

        // One Mesh per material group, referencing the source by path only.
        for i in 0..3 {
            let mesh = find(&entries, &format!("scn_prim_{i}"), "Mesh");
            assert_eq!(mesh["args"]["source"], serde_json::json!(source));
            assert_eq!(mesh["args"]["primitive_index"], serde_json::json!(i));
            assert!(mesh["args"].get("chunk_index").is_none());
        }
        assert!(entries.iter().all(|e| e["name"] != "scn_prim_3"));

        // The first model groups both of its material slots.
        let model = find(&entries, "scn_model_0", "Model");
        assert_eq!(
            model["args"]["meshes"],
            serde_json::json!([
                {"mesh": "scn_prim_0", "material": "scn_mat_0"},
                {"mesh": "scn_prim_1", "material": "scn_mat_1"},
            ])
        );
        // The second has no material connection and falls back to the default.
        let model = find(&entries, "scn_model_1", "Model");
        assert_eq!(
            model["args"]["meshes"],
            serde_json::json!([{"mesh": "scn_prim_2", "material": "scn_mat_default"}])
        );

        // A named node becomes a sanitized, index-suffixed Prop at its world
        // transform; an unnamed one falls back to its index.
        let prop = find(&entries, "scn_crate_box_0", "Prop");
        assert_eq!(prop["args"]["model"], "scn_model_0");
        assert_eq!(prop["args"]["position"], serde_json::json!([1.0, 2.0, 3.0]));
        assert_eq!(prop["args"]["scale"], serde_json::json!([1.0, 1.0, 1.0]));
        let unnamed = find(&entries, "scn_node_1", "Prop");
        assert_eq!(unnamed["args"]["model"], "scn_model_1");

        // The camera is framed to the scene's world-space bounds.
        let cam = find(&entries, "scn_cam", "Camera3D");
        assert!(cam["args"]["position"][2].as_f64().unwrap() > 3.0);
    }

    #[test]
    fn fbx_scene_with_emit_camera_false_omits_the_camera() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("scene.fbx");
        write_scene_fbx(&path);
        let opts = ImportOptions {
            emit_camera: false,
            ..scene_opts()
        };
        let entries = entries_from_scene(path.to_str().unwrap(), &opts, None).expect("expand");
        assert!(entries.iter().all(|e| e["type"] != "Camera3D"));
    }

    // One primitive whose polygon-vertex corners outnumber the u16 index
    // space: corners never share a vertex, so 21846 triangles over three
    // control points overflow it.
    fn write_oversized_fbx(path: &Path) {
        let mut f = FbxWriter::new();
        f.open("Objects");
        f.open_object("Geometry", 100, "BigMesh");
        f.f64_array("Vertices", &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let triangles = U16_CAPACITY / 3 + 1;
        let mut pvi: Vec<i32> = Vec::with_capacity(triangles * 3);
        for _ in 0..triangles {
            pvi.extend_from_slice(&[0, 1, !2]);
        }
        f.i32_array("PolygonVertexIndex", &pvi);
        f.close();
        f.open_object("Model", 200, "Big");
        f.close();
        f.close();

        f.open("Connections");
        f.connect("OO", 100, 200, None);
        f.connect("OO", 200, 0, None);
        f.close();
        f.write(path);
    }

    #[test]
    fn fbx_oversized_primitive_fans_into_chunked_meshes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big.fbx");
        write_oversized_fbx(&path);
        let opts = ImportOptions {
            name_prefix: "big".to_string(),
            ..ImportOptions::default()
        };
        let entries = entries_from_scene(path.to_str().unwrap(), &opts, None).expect("expand");

        // 65538 vertices split into a full u16 chunk plus the remainder, each
        // emitted as its own Mesh carrying the chunk index to re-slice on.
        for chunk in 0..2 {
            let mesh = find(&entries, &format!("big_prim_0_chunk_{chunk}"), "Mesh");
            assert_eq!(mesh["args"]["primitive_index"], serde_json::json!(0));
            assert_eq!(mesh["args"]["chunk_index"], serde_json::json!(chunk));
        }
        assert!(entries.iter().all(|e| e["name"] != "big_prim_0"));

        let model = find(&entries, "big_model_0", "Model");
        assert_eq!(model["args"]["meshes"][0]["mesh"], "big_prim_0_chunk_0");
        assert_eq!(model["args"]["meshes"][1]["mesh"], "big_prim_0_chunk_1");
    }

    #[test]
    fn fbx_skinned_nodes_expand_to_skinned_meshes_and_clips_not_props() {
        let file = crate::import::fbx::fixtures::two_part_rig_with_clip();
        let opts = ImportOptions {
            name_prefix: "hero".to_string(),
            ..ImportOptions::default()
        };
        let entries = entries_from_scene(file.path(), &opts, None).expect("expand fbx");

        // Both skin-deformed geometries expand, ranked in Geometry declaration
        // order so the selector matches the importer's own scan.
        let body = find(&entries, "hero_skin_0", "SkinnedMesh");
        assert_eq!(body["args"]["source"], serde_json::json!(file.path()));
        assert_eq!(body["args"]["skin_index"], serde_json::json!(0));
        let hair = find(&entries, "hero_skin_1", "SkinnedMesh");
        assert_eq!(hair["args"]["skin_index"], serde_json::json!(1));

        // The clip is generated for each part against its own mesh.
        assert_eq!(
            find(&entries, "hero_anim_0_wave_0", "Animation")["args"]["target"],
            "hero_skin_0"
        );
        assert_eq!(
            find(&entries, "hero_anim_1_wave_0", "Animation")["args"]["target"],
            "hero_skin_1"
        );

        // No static twin of the skinned geometry.
        assert!(
            entries.iter().all(|e| e["type"] != "Prop"),
            "skinned nodes emit no Prop: {entries:#?}"
        );
        assert!(entries.iter().all(|e| e["type"] != "Mesh"));
    }

    #[test]
    fn fbx_static_scene_generates_no_rig_entries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("scene.fbx");
        write_scene_fbx(&path);
        let entries =
            entries_from_scene(path.to_str().unwrap(), &scene_opts(), None).expect("expand");
        assert!(entries.iter().all(|e| e["type"] != "SkinnedMesh"));
        assert!(entries.iter().all(|e| e["type"] != "Animation"));
        // The static expansion is untouched: every prop and mesh still lands.
        find(&entries, "scn_prim_0", "Mesh");
        find(&entries, "scn_crate_box_0", "Prop");
    }

    #[test]
    fn corrupt_fbx_errors_as_invalid_data() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("junk.fbx");
        std::fs::write(&path, b"definitely not an fbx").unwrap();

        let err = entries_from_scene(path.to_str().unwrap(), &ImportOptions::default(), None)
            .expect_err("corrupt file");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("not a valid FBX file"), "{err}");
    }

    #[test]
    fn glb_embedded_image_becomes_a_texture_bound_to_the_material() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tex.glb");
        std::fs::write(&path, textured_triangle_glb()).unwrap();
        let path = path.to_str().unwrap();

        let opts = ImportOptions {
            name_prefix: "scn".to_string(),
            ..ImportOptions::default()
        };
        let entries = entries_from_scene(path, &opts, None).expect("expand glb");

        // The image yields a Texture entry referencing the GLB by path + index.
        let tex = find(&entries, "scn_tex_0", "Texture");
        assert_eq!(tex["args"]["source"], serde_json::json!(path));
        assert_eq!(tex["args"]["image_index"], serde_json::json!(0));

        // The material binds that texture as both albedo and normal map, and
        // carries the packed metallic-roughness + emissive images from the file.
        let mat = find(&entries, "scn_mat_0", "Material");
        assert_eq!(mat["args"]["albedo"], "scn_tex_0");
        assert_eq!(mat["args"]["normal_map"], "scn_tex_0");
        assert_eq!(mat["args"]["orm_map"], "scn_tex_1");
        assert_eq!(mat["args"]["emissive_map"], "scn_tex_2");
        assert_eq!(
            mat["args"]["emissive_factor"],
            serde_json::json!([1.0, 1.0, 1.0])
        );
        // The source's occlusion texture (image 3) is deliberately unwired:
        // ambient occlusion comes from the screen-space pass, and `orm_map`
        // reserves its red channel for exactly that reason.
        assert!(
            mat["args"]
                .as_object()
                .expect("material args")
                .values()
                .all(|v| v.as_str() != Some("scn_tex_3")),
            "the occlusion image must not be bound: {}",
            mat["args"]
        );

        // alphaMode MASK becomes a cutout threshold; BLEND stays unmapped
        // (`transparent` means refracting glass, not a blended card).
        let cutout = find(&entries, "scn_mat_1", "Material");
        assert_eq!(cutout["args"]["alpha_cutoff"], serde_json::json!(0.25));
        let blended = find(&entries, "scn_mat_2", "Material");
        assert!(blended["args"].get("alpha_cutoff").is_none());
        assert!(blended["args"].get("transparent").is_none());
        assert!(mat["args"].get("alpha_cutoff").is_none());

        // The second node draws the same mesh but carries no name, so it is
        // named after its walk order instead.
        let unnamed = find(&entries, "scn_node_1", "Prop");
        assert_eq!(unnamed["args"]["model"], "scn_model_0");
        assert_eq!(
            unnamed["args"]["position"],
            serde_json::json!([5.0, 0.0, 0.0])
        );
    }

    #[test]
    #[ignore = "needs the local Blender pbr_maps fixture under private/assets"]
    fn a_blender_authored_glb_keeps_its_packed_maps_and_cutout() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../private/assets/models/pbr_maps/pbr_maps.glb"
        );
        let opts = ImportOptions {
            name_prefix: "pbr".to_string(),
            ..ImportOptions::default()
        };
        let entries = entries_from_scene(path, &opts, None).expect("expand glb");

        // Metal sphere: a packed metallic-roughness image, no cutout.
        let plate = find(&entries, "pbr_mat_0", "Material");
        assert_eq!(plate["args"]["albedo"], "pbr_tex_0");
        assert_eq!(plate["args"]["orm_map"], "pbr_tex_1");
        assert!(plate["args"].get("alpha_cutoff").is_none());

        // Banded emissive cube: the map plus the factor that scales it.
        let glow = find(&entries, "pbr_mat_1", "Material");
        assert_eq!(glow["args"]["emissive_map"], "pbr_tex_2");
        assert_eq!(glow["args"]["albedo"], "pbr_tex_3");
        assert_eq!(
            glow["args"]["emissive_factor"],
            serde_json::json!([1.0, 1.0, 1.0])
        );

        // Leaf card: MASK with no explicit alphaCutoff takes the glTF default,
        // and stays out of the transparent (glass) pass.
        let leaf = find(&entries, "pbr_mat_2", "Material");
        assert_eq!(leaf["args"]["alpha_cutoff"], serde_json::json!(0.5));
        assert!(leaf["args"].get("transparent").is_none());
        assert!(leaf["args"].get("see_through").is_none());
    }

    #[test]
    fn glb_texture_max_size_of_zero_leaves_the_texture_uncapped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tex.glb");
        std::fs::write(&path, textured_triangle_glb()).unwrap();

        let opts = ImportOptions {
            name_prefix: "scn".to_string(),
            texture_max_size: 0,
            ..ImportOptions::default()
        };
        let entries = entries_from_scene(path.to_str().unwrap(), &opts, None).expect("expand glb");
        let tex = find(&entries, "scn_tex_0", "Texture");
        assert!(tex["args"].get("max_size").is_none());
    }
}
