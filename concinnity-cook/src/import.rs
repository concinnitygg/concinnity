// src/build/import.rs
//
// Build-time expansion of a scene file into Concinnity asset entries
// (Texture / Material / Mesh / Model / Prop / Camera3D). Each entry references
// the source file by path: geometry is filled in later by the desugar passes
// in `pipeline.rs` and texture pixels by `compile_texture_payload`, so the
// generated entries carry no inline vertex or pixel data. The expansion is
// driven from a `SceneImport` asset by
// `crate::world::scene_import::expand_scene_imports`.
//
// Two container formats are supported, dispatched by `source` extension:
//   - `.fbx` via `crate::fbx`
//   - `.glb` via `crate::gltf` / `crate::glb`
//
// PBR mapping, FBX texture slot -> Concinnity Material field:
//   DiffuseColor  -> albedo
//   NormalMap     -> normal_map
//   SpecularColor -> orm_map      (packed occlusion / roughness / metalness)
//   EmissiveColor -> emissive_map
//
// PBR mapping, glTF -> Concinnity Material:
//   baseColorTexture    -> albedo
//   baseColorFactor.rgb -> tint
//   normalTexture       -> normal_map
//   metallicFactor      -> metallic
//   roughnessFactor     -> roughness
//   emissiveFactor      -> emissive_factor
// glTF metallic-roughness packed textures, occlusion textures, and alpha modes
// are dropped for now; the demo scenes still render correctly without them.

use std::collections::HashMap;
use std::path::Path;

use crate::gfx::skinning::{IDENTITY, Mat4, decompose, euler_yxz_from_quat, mat4_mul};

// u16 index ceiling: a primitive with more vertices than this fans into chunks.
const U16_CAPACITY: usize = u16::MAX as usize + 1;

// Knobs threaded in from the `SceneImport` asset's args. `name_prefix` is the
// import's (unique) asset name, sanitized; every generated asset name carries
// it so the expansion never collides with hand-authored assets.
#[derive(Debug, Clone)]
pub struct ImportOptions {
    pub name_prefix: String,
    pub texture_max_size: u32,
    pub emissive_map_strength: f32,
    pub emit_camera: bool,
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
pub fn entries_from_scene(
    source: &str,
    opts: &ImportOptions,
) -> std::io::Result<Vec<serde_json::Value>> {
    let ext = Path::new(source)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "fbx" => entries_from_fbx(source, opts),
        "glb" => entries_from_glb(source, opts),
        other => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "SceneImport source '{}': unsupported format '.{}' (supported: .fbx, .glb)",
                source, other
            ),
        )),
    }
}

// Lowercase ASCII-alphanumeric/underscore sanitizer for an asset-name prefix
// and for node-derived prop names. Everything else collapses to underscore so
// the result reads like an identifier.
pub fn sanitize_name(input: &str) -> String {
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
    let scene = crate::fbx::parse_fbx(path).map_err(|e| {
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
        // Glass detection: the FBX flags it transparent, or the material is named
        // like glass. Glass renders smooth + translucent (transparent pass when
        // ray tracing is available, else a smooth reflective opaque surface).
        let lname = m.name.to_lowercase();
        // Smooth, see-through glass: flagged transparent by the FBX, or named
        // like glass/window. Frosted glass stays rough/diffuse and emissive
        // "glass" (lamp lenses) stays an opaque glow, so both are excluded.
        let is_glass = (m.opacity < 0.95 || lname.contains("glass") || lname.contains("window"))
            && !lname.contains("frosted")
            && !lname.contains("emissive");
        if std::env::var_os("CN_IMPORT_DEBUG").is_some() {
            eprintln!(
                "[import] mat '{}' opacity={} glass={}",
                m.name, m.opacity, is_glass
            );
        }
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

    // Meshes: one per primitive, fanning oversized primitives into u16 chunks.
    // Record the mesh asset name(s) produced for each primitive so the prop's
    // model can list them as submeshes.
    let mut primitive_meshes: Vec<Vec<String>> = vec![Vec::new(); scene.primitives.len()];
    for (i, prim) in scene.primitives.iter().enumerate() {
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
        let (verts, indices32) = crate::fbx::read_primitive_geometry(&scene, i as u32)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let chunk_count = crate::glb::split_into_u16_chunks(&verts, &indices32).len();
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

    // Models + Props: one of each per scene node that carries geometry.
    for (pi, prop) in scene.props.iter().enumerate() {
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

    // Camera framed to the scene's world AABB, mirroring the glTF importer.
    if opts.emit_camera
        && let Some(camera) = framed_camera_entry(prefix, scene.aabb)
    {
        entries.push(camera);
    }

    Ok(entries)
}

// glTF (.glb) -> asset entries.
fn entries_from_glb(path: &str, opts: &ImportOptions) -> std::io::Result<Vec<serde_json::Value>> {
    let bytes = std::fs::read(path)
        .map_err(|e| std::io::Error::new(e.kind(), format!("could not read '{}': {}", path, e)))?;
    let doc = gltf::Gltf::from_slice(&bytes).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("'{}': not a valid glTF/GLB file: {}", path, e),
        )
    })?;

    let prefix = &opts.name_prefix;
    let mut entries: Vec<serde_json::Value> = Vec::new();

    // Textures: one entry per glTF image, referencing the GLB binary chunk.
    for (i, _img) in doc.document.images().enumerate() {
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
        .document
        .materials()
        .enumerate()
        .map(|(i, mat)| {
            let name = format!("{prefix}_mat_{i}");
            let pbr = mat.pbr_metallic_roughness();
            let base_color_factor = pbr.base_color_factor();
            let mut args = serde_json::Map::new();

            if let Some(info) = pbr.base_color_texture() {
                let image_idx = info.texture().source().index();
                args.insert(
                    "albedo".to_string(),
                    serde_json::Value::String(format!("{prefix}_tex_{image_idx}")),
                );
            }
            if let Some(info) = mat.normal_texture() {
                let image_idx = info.texture().source().index();
                args.insert(
                    "normal_map".to_string(),
                    serde_json::Value::String(format!("{prefix}_tex_{image_idx}")),
                );
            }
            args.insert(
                "tint".to_string(),
                serde_json::json!([
                    base_color_factor[0],
                    base_color_factor[1],
                    base_color_factor[2],
                ]),
            );
            args.insert(
                "metallic".to_string(),
                serde_json::json!(pbr.metallic_factor()),
            );
            args.insert(
                "roughness".to_string(),
                serde_json::json!(pbr.roughness_factor()),
            );
            let e = mat.emissive_factor();
            args.insert(
                "emissive_factor".to_string(),
                serde_json::json!([e[0], e[1], e[2]]),
            );

            entries.push(serde_json::json!({
                "name": name,
                "type": "Material",
                "args": serde_json::Value::Object(args),
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
    let mut primitive_counter: usize = 0;
    let mut mesh_to_submesh_refs: Vec<Vec<serde_json::Value>> = Vec::new();
    for gltf_mesh in doc.document.meshes() {
        let mut submesh_refs: Vec<serde_json::Value> = Vec::new();
        for primitive in gltf_mesh.primitives() {
            let prim_idx = primitive_counter;
            primitive_counter += 1;

            let material_name = primitive
                .material()
                .index()
                .and_then(|i| material_names.get(i).cloned())
                .unwrap_or_else(|| default_mat_name.clone());

            let vert_count =
                crate::gltf::primitive_vertex_count(&doc, prim_idx as u32).unwrap_or(0);

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
            let (verts, indices32) =
                crate::glb::read_primitive_geometry(&doc, path, prim_idx as u32)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let chunk_count = crate::glb::split_into_u16_chunks(&verts, &indices32).len();
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
        mesh_to_submesh_refs.push(submesh_refs);
    }

    // Models: one entry per glTF mesh, grouping its primitives + materials.
    let model_names: Vec<String> = mesh_to_submesh_refs
        .iter()
        .enumerate()
        .map(|(i, submeshes)| {
            let name = format!("{prefix}_model_{i}");
            entries.push(serde_json::json!({
                "name": name,
                "type": "Model",
                "args": { "meshes": submeshes }
            }));
            name
        })
        .collect();

    // Props: walk the default scene graph. Mesh-bearing nodes become Props
    // with world-space transforms; transform-only nodes are flattened into
    // their descendants. This keeps the hierarchy simple (every emitted Prop
    // is independent) at the cost of losing the original parent links.
    // The walk also accumulates a world-space AABB so the camera can be
    // framed to the scene's actual scale.
    let scene = doc
        .document
        .default_scene()
        .or_else(|| doc.document.scenes().next());
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
    model_names: &'a [String],
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

        if let Some(mesh) = node.mesh() {
            let mesh_idx = mesh.index();
            if let Some(model_name) = self.model_names.get(mesh_idx) {
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
        let err = entries_from_scene("model.obj", &ImportOptions::default())
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

    // One scene, one node named "Crate Box" at (1,2,3), one mesh with a single
    // triangle primitive of `vertex_count` positions, and one PBR material.
    // Growing `vertex_count` past the u16 limit exercises the chunking path.
    fn triangle_glb(vertex_count: usize) -> Vec<u8> {
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
  "nodes": [{{"mesh": 0, "name": "Crate Box", "translation": [1, 2, 3]}}],
  "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0}}, "indices": 1, "material": 0, "mode": 4}}]}}],
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
        let entries = entries_from_scene(path, &opts).expect("expand glb");

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

        // The node becomes a Prop named from the sanitized node name with the
        // node's translation applied.
        let prop = find(&entries, "scn_crate_box", "Prop");
        assert_eq!(prop["args"]["model"], "scn_model_0");
        assert_eq!(prop["args"]["position"], serde_json::json!([1.0, 2.0, 3.0]));

        // A camera is framed to the world-space AABB.
        let cam = find(&entries, "scn_cam", "Camera3D");
        assert!(cam["args"]["position"][2].as_f64().unwrap() > 3.0);
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
        let entries = entries_from_scene(path.to_str().unwrap(), &opts).expect("expand glb");
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
        let entries = entries_from_scene(path, &opts).expect("expand glb");

        // The single triangle re-chunks into one u16-safe slice; the entry
        // carries the chunk index and there is no unchunked mesh entry.
        let mesh = find(&entries, "big_prim_0_chunk_0", "Mesh");
        assert_eq!(mesh["args"]["chunk_index"], serde_json::json!(0));
        assert!(entries.iter().all(|e| e["name"] != "big_prim_0"));

        let model = find(&entries, "big_model_0", "Model");
        assert_eq!(model["args"]["meshes"][0]["mesh"], "big_prim_0_chunk_0");
    }

    #[test]
    fn glb_extension_dispatch_is_case_insensitive() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("upper.GLB");
        std::fs::write(&path, triangle_glb(3)).unwrap();

        let entries =
            entries_from_scene(path.to_str().unwrap(), &ImportOptions::default()).expect("expand");
        assert!(entries.iter().any(|e| e["type"] == "Mesh"));
    }

    #[test]
    fn missing_glb_file_errors_with_the_path() {
        let err = entries_from_scene("/no/such/scene.glb", &ImportOptions::default())
            .expect_err("missing file");
        assert!(err.to_string().contains("/no/such/scene.glb"));
    }

    #[test]
    fn corrupt_glb_errors_as_invalid_data() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("junk.glb");
        std::fs::write(&path, b"definitely not a glb").unwrap();

        let err = entries_from_scene(path.to_str().unwrap(), &ImportOptions::default())
            .expect_err("corrupt file");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("not a valid glTF"));
    }

    #[test]
    fn missing_fbx_file_errors() {
        let err = entries_from_scene("/no/such/scene.fbx", &ImportOptions::default())
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
        // in for the encoded pixels.
        bin.extend_from_slice(&[0u8; 16]);
        let json = format!(
            r#"{{
  "asset": {{"version": "2.0"}},
  "scene": 0,
  "scenes": [{{"nodes": [0]}}],
  "nodes": [{{"mesh": 0, "name": "Crate", "translation": [0, 0, 0]}}],
  "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0}}, "indices": 1, "material": 0, "mode": 4}}]}}],
  "materials": [{{
    "pbrMetallicRoughness": {{"baseColorTexture": {{"index": 0}}, "metallicFactor": 0.0, "roughnessFactor": 1.0}},
    "normalTexture": {{"index": 0}}
  }}],
  "textures": [{{"source": 0}}],
  "images": [{{"bufferView": 2, "mimeType": "image/png"}}],
  "accessors": [
    {{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 0]}},
    {{"bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR"}}
  ],
  "bufferViews": [
    {{"buffer": 0, "byteOffset": 0, "byteLength": {pos_len}}},
    {{"buffer": 0, "byteOffset": {pos_len}, "byteLength": 12}},
    {{"buffer": 0, "byteOffset": {idx_end}, "byteLength": 16}}
  ],
  "buffers": [{{"byteLength": {total}}}]
}}"#,
            pos_len = pos_len,
            idx_end = idx_end,
            total = bin.len(),
        );
        glb_bytes(&json, &bin)
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
        let entries = entries_from_scene(path, &opts).expect("expand glb");

        // The image yields a Texture entry referencing the GLB by path + index.
        let tex = find(&entries, "scn_tex_0", "Texture");
        assert_eq!(tex["args"]["source"], serde_json::json!(path));
        assert_eq!(tex["args"]["image_index"], serde_json::json!(0));

        // The material binds that texture as both albedo and normal map.
        let mat = find(&entries, "scn_mat_0", "Material");
        assert_eq!(mat["args"]["albedo"], "scn_tex_0");
        assert_eq!(mat["args"]["normal_map"], "scn_tex_0");
    }
}
