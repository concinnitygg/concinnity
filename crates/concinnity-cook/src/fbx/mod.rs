//! Imports a binary FBX scene (v7.4 / v7.5) into engine-native assets. The
//! scene is reduced to three flat lists: materials (with their resolved texture
//! paths), geometry primitives (one per material group within a mesh, in
//! geometry-local space), and props (one per scene node that carries geometry,
//! holding the node's world transform decomposed into translation / Euler
//! rotation / scale and the indices of its primitives).
//!
//! Only what the renderer needs is extracted: positions, the first UV set, and
//! per-polygon material assignment. Normals and tangents are recomputed by the
//! mesh build, so FBX normal layers are ignored. Geometry is kept in local space
//! and the node transform travels on the prop, mirroring the glTF import.
//!
//! FBX texture slots are mapped from the connection property they bind to:
//! DiffuseColor -> albedo, NormalMap -> normal, SpecularColor -> packed ORM
//! (occlusion/roughness/metalness), EmissiveColor -> emissive.

mod anim;
#[cfg(test)]
pub(super) mod fixtures;
mod skin;

pub(crate) use anim::fbx_animation_names;
pub use anim::import_fbx_animation;
pub(crate) use skin::import_skinned_fbx;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use fbxcel::low::v7400::AttributeValue;
use fbxcel::tree::any::AnyTree;
use fbxcel::tree::v7400::NodeHandle;

use crate::components::VertexData;
use crate::gfx::transform::{IDENTITY, Mat4, decompose, euler_yxz_from_quat, mat4_mul};

use crate::import::NEUTRAL_COLOR;

/// A material with its scalar factors and resolved on-disk texture paths.
pub(crate) struct FbxMaterial {
    /// The material's name in the FBX document.
    pub name: String,
    /// Albedo texture path, when the material names one.
    pub albedo: Option<String>,
    /// Normal-map path, when the material names one.
    pub normal: Option<String>,
    /// Occlusion / roughness / metalness texture path, when named.
    pub orm: Option<String>,
    /// Emissive texture path, when the material names one.
    pub emissive: Option<String>,
    /// Diffuse colour, linear RGB.
    pub diffuse: [f32; 3],
    /// Emissive colour, linear RGB.
    pub emissive_factor: [f32; 3],
    /// Surface opacity in [0, 1]; 1 = fully opaque. Read from the FBX Opacity /
    /// TransparencyFactor property (glass/windows export < 1). 1.0 when absent.
    pub opacity: f32,
}

/// One material group of a mesh: geometry-local vertices and triangle indices.
pub(crate) struct FbxPrimitive {
    /// The primitive's vertices.
    pub vertices: Vec<VertexData>,
    /// Triangle indices into `vertices`.
    pub indices: Vec<u32>,
    /// Index into the scene's material list, when the primitive names one.
    pub material: Option<usize>,
}

/// A scene node carrying geometry: a world transform plus the primitives drawn
/// at that transform.
pub(crate) struct FbxProp {
    /// The prop's name in the FBX document.
    pub name: String,
    /// World-space position.
    pub position: [f32; 3],
    /// YXZ Euler rotation in degrees.
    pub rotation_deg: [f32; 3],
    /// Per-axis scale.
    pub scale: [f32; 3],
    /// Indices into the scene's primitive list.
    pub primitives: Vec<usize>,
    /// Rank of this prop's geometry among the file's skin-deformed geometries,
    /// in Geometry declaration order, or `None` when it is static. Matches the
    /// `skin_index` the skinned importer selects on.
    pub skin_index: Option<usize>,
}

/// The fully extracted scene.
pub(crate) struct FbxScene {
    /// Materials declared by the document.
    pub materials: Vec<FbxMaterial>,
    /// Primitives declared by the document.
    pub primitives: Vec<FbxPrimitive>,
    /// Props declared by the document.
    pub props: Vec<FbxProp>,
    /// World-space bounding box (min, max) of all geometry, for framing a camera.
    pub aabb: Option<([f32; 3], [f32; 3])>,
}

// Raw geometry of a primitive (clones out of the parsed scene), in the same
// `(Vec<VertexData>, Vec<u32>)` shape the glTF importer produces so the shared
// `split_into_u16_chunks` chunker applies unchanged.
pub(crate) fn read_primitive_geometry(
    scene: &FbxScene,
    index: u32,
) -> Result<(Vec<VertexData>, Vec<u32>), String> {
    scene
        .primitives
        .get(index as usize)
        .map(|p| (p.vertices.clone(), p.indices.clone()))
        .ok_or_else(|| format!("fbx primitive_index {index} out of range"))
}

fn attr_i64(a: &AttributeValue) -> Option<i64> {
    match a {
        AttributeValue::I64(v) => Some(*v),
        AttributeValue::I32(v) => Some(*v as i64),
        _ => None,
    }
}

fn attr_str(a: &AttributeValue) -> Option<&str> {
    match a {
        AttributeValue::String(s) => Some(s.as_str()),
        _ => None,
    }
}

fn attr_f64(a: &AttributeValue) -> Option<f64> {
    match a {
        AttributeValue::F64(v) => Some(*v),
        AttributeValue::F32(v) => Some(*v as f64),
        AttributeValue::I32(v) => Some(*v as f64),
        AttributeValue::I64(v) => Some(*v as f64),
        _ => None,
    }
}

// First attribute of a node as a typed array, one accessor per FBX array
// attribute kind.
macro_rules! arr_accessors {
    ($($name:ident => $ty:ty, $variant:ident;)*) => {$(
        fn $name<'a>(n: &NodeHandle<'a>) -> Option<&'a [$ty]> {
            match n.attributes().first()? {
                AttributeValue::$variant(v) => Some(v),
                _ => None,
            }
        }
    )*};
}

arr_accessors! {
    arr_f64 => f64, ArrF64;
    arr_i32 => i32, ArrI32;
    arr_i64 => i64, ArrI64;
    arr_f32 => f32, ArrF32;
}

// First string attribute of a named child node (e.g. RelativeFilename).
fn child_str<'a>(n: &NodeHandle<'a>, name: &str) -> Option<&'a str> {
    n.first_child_by_name(name)
        .and_then(|c| c.attributes().first().and_then(attr_str))
}

// Object id (first attribute) and clean name (second attribute, trimmed at the
// FBX `name\0\u{1}Class` separator).
fn object_id(n: &NodeHandle) -> Option<i64> {
    n.attributes().first().and_then(attr_i64)
}

fn object_name(n: &NodeHandle) -> String {
    n.attributes()
        .get(1)
        .and_then(attr_str)
        .map(|s| s.split('\u{0}').next().unwrap_or(s).to_string())
        .unwrap_or_default()
}

// Read a `Properties70` P entry's three numeric values (x, y, z at attribute
// indices 4..7).
fn prop_vec3(p70: &NodeHandle, name: &str) -> Option<[f64; 3]> {
    for p in p70.children_by_name("P") {
        let a = p.attributes();
        if a.first().and_then(attr_str) == Some(name) {
            return Some([
                a.get(4).and_then(attr_f64)?,
                a.get(5).and_then(attr_f64)?,
                a.get(6).and_then(attr_f64)?,
            ]);
        }
    }
    None
}

// Read a `Properties70` P entry's single numeric value (at attribute index 4).
fn prop_scalar(p70: &NodeHandle, name: &str) -> Option<f64> {
    for p in p70.children_by_name("P") {
        let a = p.attributes();
        if a.first().and_then(attr_str) == Some(name) {
            return a.get(4).and_then(attr_f64);
        }
    }
    None
}

fn translate(t: [f64; 3]) -> Mat4 {
    let mut m = IDENTITY;
    m[3][0] = t[0] as f32;
    m[3][1] = t[1] as f32;
    m[3][2] = t[2] as f32;
    m
}

fn scale_mat(s: [f64; 3]) -> Mat4 {
    let mut m = IDENTITY;
    m[0][0] = s[0] as f32;
    m[1][1] = s[1] as f32;
    m[2][2] = s[2] as f32;
    m
}

fn rot_x(deg: f64) -> Mat4 {
    let (s, c) = (deg as f32).to_radians().sin_cos();
    let mut m = IDENTITY;
    m[1][1] = c;
    m[1][2] = s;
    m[2][1] = -s;
    m[2][2] = c;
    m
}

fn rot_y(deg: f64) -> Mat4 {
    let (s, c) = (deg as f32).to_radians().sin_cos();
    let mut m = IDENTITY;
    m[0][0] = c;
    m[0][2] = -s;
    m[2][0] = s;
    m[2][2] = c;
    m
}

fn rot_z(deg: f64) -> Mat4 {
    let (s, c) = (deg as f32).to_radians().sin_cos();
    let mut m = IDENTITY;
    m[0][0] = c;
    m[0][1] = s;
    m[1][0] = -s;
    m[1][1] = c;
    m
}

// XYZ euler composition, the order FBX PreRotation always uses.
fn rot_ordered_xyz(r_deg: [f64; 3]) -> Mat4 {
    rot_ordered(r_deg, 0)
}

// Euler composition honouring an FBX RotationOrder enum: 0 XYZ, 1 XZY,
// 2 YZX, 3 YXZ, 4 ZXY, 5 ZYX (application order; matrices multiply
// right-to-left for column vectors). SphericXYZ (6) falls back to XYZ.
fn rot_ordered(r_deg: [f64; 3], order: i32) -> Mat4 {
    let x = rot_x(r_deg[0]);
    let y = rot_y(r_deg[1]);
    let z = rot_z(r_deg[2]);
    let [a, b, c] = match order {
        1 => [x, z, y],
        2 => [y, z, x],
        3 => [y, x, z],
        4 => [z, x, y],
        5 => [z, y, x],
        _ => [x, y, z],
    };
    mat4_mul(c, mat4_mul(b, a))
}

// Compose an FBX node transform T * R * S with XYZ rotation order.
fn trs_matrix(t: [f64; 3], r_deg: [f64; 3], s: [f64; 3]) -> Mat4 {
    mat4_mul(translate(t), mat4_mul(rot_ordered_xyz(r_deg), scale_mat(s)))
}

// Scene-pose local transform of a node including its PreRotation and
// RotationOrder: T * Rpre * R * S. Used for skeleton joints that carry no
// cluster bind matrix.
fn node_scene_local(model: &NodeHandle) -> Mat4 {
    let Some(p70) = model.first_child_by_name("Properties70") else {
        return IDENTITY;
    };
    let t = prop_vec3(&p70, "Lcl Translation").unwrap_or([0.0, 0.0, 0.0]);
    let r = prop_vec3(&p70, "Lcl Rotation").unwrap_or([0.0, 0.0, 0.0]);
    let s = prop_vec3(&p70, "Lcl Scaling").unwrap_or([1.0, 1.0, 1.0]);
    let pre = prop_vec3(&p70, "PreRotation").unwrap_or([0.0, 0.0, 0.0]);
    let order = prop_scalar(&p70, "RotationOrder").unwrap_or(0.0) as i32;
    let rot = mat4_mul(rot_ordered_xyz(pre), rot_ordered(r, order));
    mat4_mul(translate(t), mat4_mul(rot, scale_mat(s)))
}

// (node-local matrix, geometric-offset matrix) for a Model node.
fn local_matrices(model: &NodeHandle) -> (Mat4, Mat4) {
    let Some(p70) = model.first_child_by_name("Properties70") else {
        return (IDENTITY, IDENTITY);
    };
    let t = prop_vec3(&p70, "Lcl Translation").unwrap_or([0.0, 0.0, 0.0]);
    let r = prop_vec3(&p70, "Lcl Rotation").unwrap_or([0.0, 0.0, 0.0]);
    let s = prop_vec3(&p70, "Lcl Scaling").unwrap_or([1.0, 1.0, 1.0]);
    let gt = prop_vec3(&p70, "GeometricTranslation").unwrap_or([0.0, 0.0, 0.0]);
    let gr = prop_vec3(&p70, "GeometricRotation").unwrap_or([0.0, 0.0, 0.0]);
    let gs = prop_vec3(&p70, "GeometricScaling").unwrap_or([1.0, 1.0, 1.0]);
    (trs_matrix(t, r, s), trs_matrix(gt, gr, gs))
}

// World matrix of a model, walking the parent chain (depth is shallow in
// practice; most nodes parent directly to the scene root id 0).
fn world_matrix(id: i64, locals: &HashMap<i64, (Mat4, Mat4)>, parents: &HashMap<i64, i64>) -> Mat4 {
    let local = locals.get(&id).map(|x| x.0).unwrap_or(IDENTITY);
    match parents.get(&id) {
        Some(&p) if p != 0 && locals.contains_key(&p) => {
            mat4_mul(world_matrix(p, locals, parents), local)
        }
        _ => local,
    }
}

// Whether an FBX-authored path is absolute on the machine that wrote it.
// `Path::is_absolute` cannot answer this: an FBX may come from any OS, and a
// POSIX "/tex/x.png" is not absolute on Windows while a "C:/tex/x.png" is not
// absolute anywhere else. Separators are normalized to "/" before this runs.
fn is_author_absolute(path: &str) -> bool {
    let bytes = path.as_bytes();
    path.starts_with('/')
        || (bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':')
}

// Resolve an FBX texture's RelativeFilename to a path under the FBX directory.
fn resolve_texture_path(tex: &NodeHandle, fbx_dir: &Path) -> Option<String> {
    let rel = child_str(tex, "RelativeFilename").or_else(|| child_str(tex, "FileName"))?;
    let rel = rel.replace('\\', "/");
    let joined = if is_author_absolute(&rel) {
        // Absolute author-machine path: keep only the file name under the local dir.
        match Path::new(&rel).file_name() {
            Some(name) => fbx_dir.join(name),
            // Nothing to rebase (e.g. a bare root); leave it as authored.
            None => return Some(rel),
        }
    } else {
        fbx_dir.join(&rel)
    };
    // `join` uses the host separator; the result is an asset source string that
    // keys the cook cache, so keep it forward-slashed on every platform.
    Some(joined.to_string_lossy().replace('\\', "/"))
}

// Meters per FBX file unit, from GlobalSettings. FBX's native unit is the
// centimeter: UnitScaleFactor 1.0 means cm, 100.0 means meters. The skinned
// and animation importers normalize to meters so characters, capsules, and
// root motion agree with the glTF path; the static-scene path keeps file
// units (Bistro-era worlds are authored against them).
fn unit_scale_to_meters(root: &NodeHandle) -> f32 {
    let factor = root
        .first_child_by_name("GlobalSettings")
        .and_then(|gs| gs.first_child_by_name("Properties70"))
        .and_then(|p| prop_scalar(&p, "UnitScaleFactor"))
        .unwrap_or(1.0);
    (factor / 100.0) as f32
}

// Read a binary FBX file into its v7.4/7.5 node tree.
fn load_tree(path: &str) -> Result<fbxcel::tree::v7400::Tree, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("could not open '{path}': {e}"))?;
    let reader = std::io::BufReader::new(file);
    match AnyTree::from_seekable_reader(reader)
        .map_err(|e| format!("'{path}': not a valid FBX file: {e}"))?
    {
        AnyTree::V7400(_ver, tree, _footer) => Ok(tree),
        _ => Err(format!("'{path}': unsupported FBX version")),
    }
}

/// Parse a binary FBX file into an [`FbxScene`].
pub(crate) fn parse_fbx(path: &str) -> Result<FbxScene, String> {
    let tree = load_tree(path)?;
    let root = tree.root();
    let objects = root
        .first_child_by_name("Objects")
        .ok_or_else(|| format!("'{path}': FBX has no Objects section"))?;

    // Index objects by id and type.
    let mut geom_by_id: HashMap<i64, NodeHandle> = HashMap::new();
    let mut geom_order: Vec<i64> = Vec::new();
    let mut model_by_id: HashMap<i64, NodeHandle> = HashMap::new();
    let mut material_by_id: HashMap<i64, NodeHandle> = HashMap::new();
    let mut texture_by_id: HashMap<i64, NodeHandle> = HashMap::new();
    let mut skin_ids: HashSet<i64> = HashSet::new();
    for c in objects.children() {
        let Some(id) = object_id(&c) else { continue };
        match c.name() {
            "Deformer" if c.attributes().get(2).and_then(attr_str) == Some("Skin") => {
                skin_ids.insert(id);
            }
            "Geometry" => {
                geom_by_id.insert(id, c);
                geom_order.push(id);
            }
            "Model" => {
                model_by_id.insert(id, c);
            }
            "Material" => {
                material_by_id.insert(id, c);
            }
            "Texture" => {
                texture_by_id.insert(id, c);
            }
            _ => {}
        }
    }

    // Connections: object-object (hierarchy / geometry / material) and
    // object-property (texture -> material slot).
    let mut model_parent: HashMap<i64, i64> = HashMap::new();
    let mut model_geometry: HashMap<i64, i64> = HashMap::new();
    let mut model_materials: HashMap<i64, Vec<i64>> = HashMap::new();
    let mut mat_textures: HashMap<i64, Vec<(i64, String)>> = HashMap::new();
    let mut skinned_geoms: HashSet<i64> = HashSet::new();
    if let Some(conns) = root.first_child_by_name("Connections") {
        for c in conns.children_by_name("C") {
            let a = c.attributes();
            let ty = a.first().and_then(attr_str).unwrap_or("");
            let (Some(child), Some(parent)) =
                (a.get(1).and_then(attr_i64), a.get(2).and_then(attr_i64))
            else {
                continue;
            };
            match ty {
                "OO" => {
                    if model_by_id.contains_key(&child) {
                        model_parent.insert(child, parent);
                    }
                    if geom_by_id.contains_key(&child) && model_by_id.contains_key(&parent) {
                        model_geometry.insert(parent, child);
                    }
                    if material_by_id.contains_key(&child) && model_by_id.contains_key(&parent) {
                        model_materials.entry(parent).or_default().push(child);
                    }
                    if skin_ids.contains(&child) && geom_by_id.contains_key(&parent) {
                        skinned_geoms.insert(parent);
                    }
                }
                "OP" if texture_by_id.contains_key(&child)
                    && material_by_id.contains_key(&parent) =>
                {
                    let prop = a.get(3).and_then(attr_str).unwrap_or("").to_string();
                    mat_textures.entry(parent).or_default().push((child, prop));
                }
                _ => {}
            }
        }
    }

    let fbx_dir = Path::new(path)
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();

    // Materials, in object-declaration order (deterministic).
    let mut material_index: HashMap<i64, usize> = HashMap::new();
    let mut materials: Vec<FbxMaterial> = Vec::new();
    for c in objects.children().filter(|c| c.name() == "Material") {
        let Some(id) = object_id(&c) else { continue };
        let p70 = c.first_child_by_name("Properties70");
        let diffuse = p70
            .and_then(|p| prop_vec3(&p, "DiffuseColor"))
            .map(|v| [v[0] as f32, v[1] as f32, v[2] as f32])
            .unwrap_or([1.0, 1.0, 1.0]);
        let emissive_factor = p70
            .and_then(|p| prop_vec3(&p, "EmissiveColor").or_else(|| prop_vec3(&p, "Emissive")))
            .map(|v| [v[0] as f32, v[1] as f32, v[2] as f32])
            .unwrap_or([0.0, 0.0, 0.0]);
        // Opacity: `Opacity` (1 = opaque) preferred; else `TransparencyFactor`
        // (0 = opaque, so invert). Absent -> fully opaque.
        let opacity = p70
            .and_then(|p| {
                prop_scalar(&p, "Opacity")
                    .or_else(|| prop_scalar(&p, "TransparencyFactor").map(|t| 1.0 - t))
            })
            .map(|v| v.clamp(0.0, 1.0) as f32)
            .unwrap_or(1.0);

        let mut mat = FbxMaterial {
            name: object_name(&c),
            albedo: None,
            normal: None,
            orm: None,
            emissive: None,
            diffuse,
            emissive_factor,
            opacity,
        };
        if let Some(texs) = mat_textures.get(&id) {
            for (tex_id, prop) in texs {
                let Some(tex) = texture_by_id.get(tex_id) else {
                    continue;
                };
                let path = resolve_texture_path(tex, &fbx_dir);
                match prop.as_str() {
                    "DiffuseColor" => mat.albedo = path,
                    "NormalMap" => mat.normal = path,
                    "SpecularColor" => mat.orm = path,
                    "EmissiveColor" => mat.emissive = path,
                    _ => {}
                }
            }
        }
        material_index.insert(id, materials.len());
        materials.push(mat);
    }

    // Per-model local matrices (computed once for the world walk).
    let locals: HashMap<i64, (Mat4, Mat4)> = model_by_id
        .iter()
        .map(|(id, h)| (*id, local_matrices(h)))
        .collect();

    // Skin selector ranks, in Geometry declaration order so they agree with
    // the skinned importer's own scan.
    let skin_rank: HashMap<i64, usize> = geom_order
        .iter()
        .filter(|g| skinned_geoms.contains(g))
        .enumerate()
        .map(|(rank, id)| (*id, rank))
        .collect();

    // Props + primitives, iterating models in object order for a stable
    // primitive index space (desugar and `cn add` must agree).
    let mut primitives: Vec<FbxPrimitive> = Vec::new();
    let mut props: Vec<FbxProp> = Vec::new();
    let mut aabb: Option<([f32; 3], [f32; 3])> = None;
    for model in objects.children().filter(|c| c.name() == "Model") {
        let Some(model_id) = object_id(&model) else {
            continue;
        };
        let Some(geom_id) = model_geometry.get(&model_id) else {
            continue;
        };
        let Some(geom) = geom_by_id.get(geom_id) else {
            continue;
        };

        let node_world = world_matrix(model_id, &locals, &model_parent);
        let geometric = locals.get(&model_id).map(|x| x.1).unwrap_or(IDENTITY);
        let mesh_world = mat4_mul(node_world, geometric);
        let (position, quat, scale) = decompose(mesh_world);
        let rotation_deg = euler_yxz_from_quat(quat);

        let local_mats: Vec<Option<usize>> = model_materials
            .get(&model_id)
            .map(|ids| ids.iter().map(|m| material_index.get(m).copied()).collect())
            .unwrap_or_default();

        let groups = extract_geometry_groups(geom, &local_mats);
        let mut prim_indices = Vec::new();
        for (material, vertices, indices) in groups {
            if vertices.is_empty() || indices.is_empty() {
                continue;
            }
            expand_aabb(&mut aabb, &vertices, mesh_world);
            prim_indices.push(primitives.len());
            primitives.push(FbxPrimitive {
                vertices,
                indices,
                material,
            });
        }
        if prim_indices.is_empty() {
            continue;
        }
        props.push(FbxProp {
            name: object_name(&model),
            position,
            rotation_deg,
            scale,
            primitives: prim_indices,
            skin_index: skin_rank.get(geom_id).copied(),
        });
    }

    Ok(FbxScene {
        materials,
        primitives,
        props,
        aabb,
    })
}

// Transform a point by a column-major matrix.
fn transform_point(m: Mat4, p: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
        m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
        m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2],
    ]
}

// Expand `aabb` to include a primitive's vertices after `world` transform. The
// primitive's local min/max is transformed at all eight corners so any rotation
// or non-uniform scale is captured.
fn expand_aabb(aabb: &mut Option<([f32; 3], [f32; 3])>, vertices: &[VertexData], world: Mat4) {
    if vertices.is_empty() {
        return;
    }
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for v in vertices {
        for i in 0..3 {
            lo[i] = lo[i].min(v.pos[i]);
            hi[i] = hi[i].max(v.pos[i]);
        }
    }
    let corners = [
        [lo[0], lo[1], lo[2]],
        [hi[0], lo[1], lo[2]],
        [lo[0], hi[1], lo[2]],
        [hi[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]],
        [hi[0], lo[1], hi[2]],
        [lo[0], hi[1], hi[2]],
        [hi[0], hi[1], hi[2]],
    ];
    for c in corners {
        let w = transform_point(world, c);
        match aabb {
            None => *aabb = Some((w, w)),
            Some((min, max)) => {
                for i in 0..3 {
                    min[i] = min[i].min(w[i]);
                    max[i] = max[i].max(w[i]);
                }
            }
        }
    }
}

// Decode the control-point index of a polygon-vertex entry. The last vertex of
// each polygon is stored as the bitwise complement of its index.
fn decode_pvi(raw: i32) -> (i32, bool) {
    if raw < 0 { (!raw, true) } else { (raw, false) }
}

// Look up a polygon-vertex UV, flipping V from FBX bottom-up to top-down.
fn lookup_uv(uv: Option<&[f64]>, indexed: bool, uv_index: Option<&[i32]>, pv: usize) -> [f32; 2] {
    let Some(uv) = uv else { return [0.0, 0.0] };
    let k = if indexed {
        uv_index
            .and_then(|ui| ui.get(pv))
            .copied()
            .unwrap_or(0)
            .max(0) as usize
    } else {
        pv
    };
    let u = uv.get(k * 2).copied().unwrap_or(0.0) as f32;
    let v = uv.get(k * 2 + 1).copied().unwrap_or(0.0) as f32;
    [u, 1.0 - v]
}

// Split a geometry into per-material primitives: triangulate polygons (fan) and
// deduplicate vertices by (control point, UV index) so UV seams stay sharp.
fn extract_geometry_groups(
    geom: &NodeHandle,
    local_mats: &[Option<usize>],
) -> Vec<(Option<usize>, Vec<VertexData>, Vec<u32>)> {
    let positions = geom
        .first_child_by_name("Vertices")
        .as_ref()
        .and_then(arr_f64);
    let pvi = geom
        .first_child_by_name("PolygonVertexIndex")
        .as_ref()
        .and_then(arr_i32);
    let (Some(positions), Some(pvi)) = (positions, pvi) else {
        return Vec::new();
    };

    // First texture UV layer (prefer the one named "TextureUV").
    let uv_layer = geom
        .children_by_name("LayerElementUV")
        .find(|l| child_str(l, "Name") == Some("TextureUV"))
        .or_else(|| geom.children_by_name("LayerElementUV").next());
    let (uv, uv_index, uv_indexed) = match uv_layer {
        Some(l) => (
            l.first_child_by_name("UV").as_ref().and_then(arr_f64),
            l.first_child_by_name("UVIndex").as_ref().and_then(arr_i32),
            child_str(&l, "ReferenceInformationType") == Some("IndexToDirect"),
        ),
        None => (None, None, false),
    };

    // Per-polygon material assignment.
    let mat_layer = geom.first_child_by_name("LayerElementMaterial");
    let mat_all_same = mat_layer
        .map(|l| child_str(&l, "MappingInformationType") == Some("AllSame"))
        .unwrap_or(true);
    let mat_per_poly = mat_layer.and_then(|l| {
        l.first_child_by_name("Materials")
            .as_ref()
            .and_then(arr_i32)
    });

    type Group = (HashMap<(i32, i32), u32>, Vec<VertexData>, Vec<u32>);
    let mut groups: HashMap<Option<usize>, Group> = HashMap::new();
    let mut corners: Vec<(i32, usize)> = Vec::new();
    // Reused across polygons alongside `corners`, so a millions-of-polygons
    // scene does not allocate a fresh index buffer per face.
    let mut corner_idx: Vec<u32> = Vec::new();
    let mut poly = 0usize;

    for (pv, &raw) in pvi.iter().enumerate() {
        let (cp, end) = decode_pvi(raw);
        corners.push((cp, pv));
        if !end {
            continue;
        }

        let local_mat = if mat_all_same {
            mat_per_poly.and_then(|m| m.first()).copied()
        } else {
            mat_per_poly.and_then(|m| m.get(poly)).copied()
        };
        let material = match local_mat {
            Some(li) if li >= 0 => local_mats.get(li as usize).copied().flatten(),
            _ => None,
        };

        let group = groups
            .entry(material)
            .or_insert_with(|| (HashMap::new(), Vec::new(), Vec::new()));
        for &(cp, pv) in &corners {
            let uv_key = if uv_indexed {
                uv_index.and_then(|ui| ui.get(pv)).copied().unwrap_or(0)
            } else {
                pv as i32
            };
            let key = (cp, uv_key);
            let out = if let Some(&i) = group.0.get(&key) {
                i
            } else {
                let c = (cp as usize) * 3;
                let pos = [
                    positions.get(c).copied().unwrap_or(0.0) as f32,
                    positions.get(c + 1).copied().unwrap_or(0.0) as f32,
                    positions.get(c + 2).copied().unwrap_or(0.0) as f32,
                ];
                let vd = VertexData {
                    pos,
                    color: NEUTRAL_COLOR,
                    uv: lookup_uv(uv, uv_indexed, uv_index, pv),
                };
                let i = group.1.len() as u32;
                group.1.push(vd);
                group.0.insert(key, i);
                i
            };
            corner_idx.push(out);
        }
        // Fan triangulation.
        for k in 1..corner_idx.len().saturating_sub(1) {
            group.2.push(corner_idx[0]);
            group.2.push(corner_idx[k]);
            group.2.push(corner_idx[k + 1]);
        }

        corners.clear();
        corner_idx.clear();
        poly += 1;
    }

    // Deterministic primitive order: by material index, untextured group last.
    let mut out: Vec<(Option<usize>, Vec<VertexData>, Vec<u32>)> =
        groups.into_iter().map(|(m, (_, v, i))| (m, v, i)).collect();
    out.sort_by_key(|(m, _, _)| m.map(|x| x as i64).unwrap_or(i64::MAX));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fbx::fixtures as fx;
    use crate::fbx::fixtures::assert_vec3_eq;

    #[test]
    fn decode_pvi_marks_polygon_end() {
        assert_eq!(decode_pvi(5), (5, false));
        // ~5 == -6 marks the last vertex of a polygon and decodes back to 5.
        assert_eq!(decode_pvi(!5), (5, true));
        assert_eq!(decode_pvi(0), (0, false));
        assert_eq!(decode_pvi(!0), (0, true));
    }

    #[test]
    fn lookup_uv_flips_v() {
        let uv = [0.25f64, 0.75];
        // Direct mapping, pv 0 -> (0.25, 1 - 0.75).
        let out = lookup_uv(Some(&uv), false, None, 0);
        assert!((out[0] - 0.25).abs() < 1e-6);
        assert!((out[1] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn lookup_uv_indexed() {
        let uv = [0.0f64, 0.0, 0.5, 0.5];
        let idx = [1i32];
        // IndexToDirect: pv 0 -> uv[idx[0]=1] = (0.5, 0.5) -> v flipped to 0.5.
        let out = lookup_uv(Some(&uv), true, Some(&idx), 0);
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-6);
    }

    // rot_x(90) sends +Y to +Z (column-major, right-handed).
    #[test]
    fn rot_x_90_maps_y_to_z() {
        let m = rot_x(90.0);
        // Column 1 is the image of the Y basis vector.
        assert!((m[1][1]).abs() < 1e-6);
        assert!((m[1][2] - 1.0).abs() < 1e-6);
    }

    // A pure translation decomposes back to the same position.
    #[test]
    fn trs_translation_round_trips() {
        let m = trs_matrix([3.0, -2.0, 5.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let (t, _q, s) = decompose(m);
        assert!((t[0] - 3.0).abs() < 1e-4);
        assert!((t[1] + 2.0).abs() < 1e-4);
        assert!((t[2] - 5.0).abs() < 1e-4);
        assert!((s[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn rot_y_90_maps_z_to_x() {
        assert_vec3_eq(
            transform_point(rot_y(90.0), [0.0, 0.0, 1.0]),
            [1.0, 0.0, 0.0],
        );
    }

    #[test]
    fn rot_z_90_maps_x_to_y() {
        assert_vec3_eq(
            transform_point(rot_z(90.0), [1.0, 0.0, 0.0]),
            [0.0, 1.0, 0.0],
        );
    }

    #[test]
    fn transform_point_applies_translation_last() {
        let m = translate([1.0, 2.0, 3.0]);
        assert_vec3_eq(transform_point(m, [1.0, 1.0, 1.0]), [2.0, 3.0, 4.0]);
    }

    // TRS order: scale first, then rotate, then translate.
    #[test]
    fn trs_matrix_applies_scale_then_rotation_then_translation() {
        let m = trs_matrix([1.0, 0.0, 0.0], [0.0, 0.0, 90.0], [2.0, 2.0, 2.0]);
        // (1,0,0) scales to (2,0,0), rotates about Z to (0,2,0), then
        // translates to (1,2,0).
        assert_vec3_eq(transform_point(m, [1.0, 0.0, 0.0]), [1.0, 2.0, 0.0]);
    }

    #[test]
    fn world_matrix_chains_parent_transforms() {
        let mut locals: HashMap<i64, (Mat4, Mat4)> = HashMap::new();
        locals.insert(1, (translate([1.0, 0.0, 0.0]), IDENTITY));
        locals.insert(2, (translate([2.0, 0.0, 0.0]), IDENTITY));
        let mut parents: HashMap<i64, i64> = HashMap::new();
        parents.insert(1, 0); // scene root
        parents.insert(2, 1);
        let w = world_matrix(2, &locals, &parents);
        assert_vec3_eq(transform_point(w, [0.0, 0.0, 0.0]), [3.0, 0.0, 0.0]);
        // A root-parented node keeps its local transform.
        let w = world_matrix(1, &locals, &parents);
        assert_vec3_eq(transform_point(w, [0.0, 0.0, 0.0]), [1.0, 0.0, 0.0]);
    }

    #[test]
    fn world_matrix_of_an_unknown_id_is_identity() {
        let locals = HashMap::new();
        let parents = HashMap::new();
        let w = world_matrix(42, &locals, &parents);
        assert_vec3_eq(transform_point(w, [5.0, 6.0, 7.0]), [5.0, 6.0, 7.0]);
    }

    #[test]
    fn expand_aabb_ignores_empty_vertex_lists() {
        let mut aabb = None;
        expand_aabb(&mut aabb, &[], IDENTITY);
        assert!(aabb.is_none());
    }

    fn vert(pos: [f32; 3]) -> VertexData {
        VertexData {
            pos,
            color: NEUTRAL_COLOR,
            uv: [0.0, 0.0],
        }
    }

    #[test]
    fn expand_aabb_merges_transformed_bounds() {
        let mut aabb = None;
        expand_aabb(
            &mut aabb,
            &[vert([-1.0, 0.0, 0.0]), vert([1.0, 1.0, 1.0])],
            IDENTITY,
        );
        let (min, max) = aabb.expect("first primitive seeds the box");
        assert_vec3_eq(min, [-1.0, 0.0, 0.0]);
        assert_vec3_eq(max, [1.0, 1.0, 1.0]);

        // A second primitive translated +10 on X widens the box.
        expand_aabb(
            &mut aabb,
            &[vert([0.0, 0.0, 0.0]), vert([1.0, 0.0, 0.0])],
            translate([10.0, 0.0, 0.0]),
        );
        let (min, max) = aabb.expect("merged box");
        assert_vec3_eq(min, [-1.0, 0.0, 0.0]);
        assert_vec3_eq(max, [11.0, 1.0, 1.0]);
    }

    #[test]
    fn lookup_uv_defaults_to_zero_without_a_layer() {
        assert_eq!(lookup_uv(None, false, None, 3), [0.0, 0.0]);
    }

    #[test]
    fn lookup_uv_indexed_falls_back_to_the_first_entry_when_out_of_range() {
        let uv = [0.25f64, 0.75, 0.5, 0.5];
        let idx = [1i32];
        // pv 5 is past the index array: k falls back to 0 -> uv[0].
        let out = lookup_uv(Some(&uv), true, Some(&idx), 5);
        assert!((out[0] - 0.25).abs() < 1e-6);
        assert!((out[1] - 0.25).abs() < 1e-6);
    }

    // Attribute coercions

    #[test]
    fn attr_i64_accepts_both_integer_widths() {
        assert_eq!(attr_i64(&AttributeValue::I64(7)), Some(7));
        assert_eq!(attr_i64(&AttributeValue::I32(-3)), Some(-3));
        assert_eq!(attr_i64(&AttributeValue::F64(1.0)), None);
    }

    #[test]
    fn attr_f64_coerces_numeric_variants() {
        assert_eq!(attr_f64(&AttributeValue::F64(1.5)), Some(1.5));
        assert_eq!(attr_f64(&AttributeValue::F32(0.5)), Some(0.5));
        assert_eq!(attr_f64(&AttributeValue::I32(2)), Some(2.0));
        assert_eq!(attr_f64(&AttributeValue::I64(3)), Some(3.0));
        assert_eq!(attr_f64(&AttributeValue::Bool(true)), None);
    }

    #[test]
    fn attr_str_only_accepts_strings() {
        assert_eq!(attr_str(&AttributeValue::String("hi".into())), Some("hi"));
        assert_eq!(attr_str(&AttributeValue::I32(1)), None);
    }

    // Scene accessors

    fn one_primitive_scene() -> FbxScene {
        FbxScene {
            materials: Vec::new(),
            primitives: vec![FbxPrimitive {
                vertices: vec![vert([0.0, 0.0, 0.0]), vert([1.0, 0.0, 0.0])],
                indices: vec![0, 1, 0],
                material: None,
            }],
            props: Vec::new(),
            aabb: None,
        }
    }

    #[test]
    fn read_primitive_geometry_clones_out_of_the_scene() {
        let scene = one_primitive_scene();
        let (vertices, indices) = read_primitive_geometry(&scene, 0).expect("geometry");
        assert_eq!(vertices.len(), 2);
        assert_eq!(indices, vec![0, 1, 0]);
        let err = read_primitive_geometry(&scene, 9).unwrap_err();
        assert!(err.contains("primitive_index 9 out of range"), "got: {err}");
    }

    // parse_fbx error paths (a valid binary FBX fixture requires a writer we
    // do not link; the tree-walking import itself stays covered by the real
    // asset imports exercised in `cn add`).

    // Multi-part oracle: a body + hair rig on one armature, exported to both
    // containers, must expose both parts through `skin_index` in both. Ignored
    // by default; the fixture is a local Blender export under private/assets
    // (git-ignored), regenerated by private/scripts/gen_multi_skin_rig.py.
    // Run with: cargo test -p concinnity-cook multi_skin -- --ignored
    #[test]
    #[ignore = "needs the local Blender rig_oracle fixture under private/assets"]
    fn multi_skin_rig_imports_every_part_from_both_containers() {
        let base = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../private/assets/models/rig_oracle"
        );
        let fbx_path = format!("{base}/rig_multi.fbx");
        let glb_path = format!("{base}/rig_multi.glb");

        for (label, parts) in [
            (
                "fbx",
                vec![
                    crate::fbx::import_skinned_fbx(&fbx_path, 0).expect("fbx part 0"),
                    crate::fbx::import_skinned_fbx(&fbx_path, 1).expect("fbx part 1"),
                ],
            ),
            (
                "glb",
                vec![
                    crate::gltf::import_skinned_glb(&glb_path, 0, None).expect("glb part 0"),
                    crate::gltf::import_skinned_glb(&glb_path, 1, None).expect("glb part 1"),
                ],
            ),
        ] {
            // Both parts import real geometry over the shared armature. The
            // FBX exporter adds the armature object itself to the chain, so
            // the joint counts differ per container; the bones do not.
            for (i, part) in parts.iter().enumerate() {
                assert!(
                    !part.vertices.is_empty(),
                    "{label} part {i} has no vertices"
                );
                let names: Vec<&str> = part.skeleton.iter().map(|j| j.name.as_str()).collect();
                for bone in ["root", "upper"] {
                    assert!(names.contains(&bone), "{label} part {i} joints: {names:?}");
                }
                for v in &part.vertices {
                    let sum: f32 = v.weights.iter().sum();
                    assert!((sum - 1.0).abs() < 1e-3, "{label} part {i} weights: {v:?}");
                    assert!(v.joints.iter().all(|&j| (j as usize) < part.skeleton.len()));
                }
            }
            // Both parts carry their own copy of the one shared skeleton.
            let joints = |p: &crate::glb::ImportedSkinnedMesh| -> Vec<String> {
                p.skeleton.iter().map(|j| j.name.clone()).collect()
            };
            assert_eq!(joints(&parts[0]), joints(&parts[1]), "{label} skeletons");

            // The selectors return distinct meshes, not the same one twice.
            let bounds = |p: &crate::glb::ImportedSkinnedMesh| {
                let mut lo = [f32::MAX; 3];
                let mut hi = [f32::MIN; 3];
                for v in &p.vertices {
                    for i in 0..3 {
                        lo[i] = lo[i].min(v.pos[i]);
                        hi[i] = hi[i].max(v.pos[i]);
                    }
                }
                (lo, hi)
            };
            assert_ne!(
                bounds(&parts[0]),
                bounds(&parts[1]),
                "{label}: both selectors returned the same mesh"
            );
        }

        // Every part is reachable from the scene expansion with no
        // hand-authored asset lines, and none of them also expands statically.
        for source in [&fbx_path, &glb_path] {
            let opts = crate::import::ImportOptions {
                name_prefix: "rig".to_string(),
                ..Default::default()
            };
            let entries = crate::import::entries_from_scene(source, &opts, None).expect("expand");
            let named =
                |name: &str, ty: &str| entries.iter().any(|e| e["name"] == name && e["type"] == ty);
            assert!(named("rig_skin_0", "SkinnedMesh"), "{source}: part 0");
            assert!(named("rig_skin_1", "SkinnedMesh"), "{source}: part 1");
            assert!(
                entries.iter().any(|e| e["type"] == "Animation"),
                "{source}: the Wave clip must expand"
            );
            assert!(
                entries.iter().all(|e| e["type"] != "Prop"),
                "{source}: skinned geometry must not also expand as a Prop"
            );
        }
    }

    // FBX-vs-glTF oracle: the same Blender rig exported to both containers
    // must import the same skeleton and animation. Ignored by default; the
    // fixture is a local Blender export under private/assets (git-ignored).
    // Run with: cargo test -p concinnity-cook fbx_and_glb -- --ignored
    #[test]
    #[ignore = "needs the local Blender rig_oracle fixture under private/assets"]
    fn fbx_and_glb_rig_imports_agree() {
        use crate::gfx::skeleton::JointPose;

        let base = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../private/assets/models/rig_oracle"
        );
        let fbx_path = format!("{base}/rig.fbx");
        let glb_path = format!("{base}/rig.glb");

        let fbx = crate::fbx::import_skinned_fbx(&fbx_path, 0).expect("fbx skinned import");
        let glb = crate::gltf::import_skinned_glb(&glb_path, 0, None).expect("glb skinned import");

        // World matrix per joint name via the same JointPose math the runtime
        // uses to rebuild bind poses.
        fn worlds(skeleton: &[crate::components::SkeletonJoint]) -> HashMap<String, Mat4> {
            let mut w: Vec<Mat4> = Vec::with_capacity(skeleton.len());
            let mut by_name = HashMap::new();
            for j in skeleton {
                let local = JointPose {
                    translation: j.translation,
                    rotation_deg: j.rotation_deg,
                    scale: j.scale,
                }
                .to_matrix();
                let world = if j.parent < 0 {
                    local
                } else {
                    mat4_mul(w[j.parent as usize], local)
                };
                w.push(world);
                by_name.insert(j.name.clone(), world);
            }
            by_name
        }
        let fbx_worlds = worlds(&fbx.skeleton);
        let glb_worlds = worlds(&glb.skeleton);

        // Bind: shared bones land at the same world positions.
        for name in ["Root", "Tip"] {
            let f = fbx_worlds.get(name).unwrap_or_else(|| {
                panic!(
                    "fbx skeleton missing '{name}' (has {:?})",
                    fbx.skeleton.iter().map(|j| &j.name).collect::<Vec<_>>()
                )
            });
            let g = glb_worlds.get(name).unwrap_or_else(|| {
                panic!(
                    "glb skeleton missing '{name}' (has {:?})",
                    glb.skeleton.iter().map(|j| &j.name).collect::<Vec<_>>()
                )
            });
            for i in 0..3 {
                assert!(
                    (f[3][i] - g[3][i]).abs() < 1e-2,
                    "bind world position of '{name}' differs: fbx {:?} vs glb {:?}",
                    f[3],
                    g[3]
                );
            }
        }

        // Mesh: both containers carry the same cylinder, weighted to 2 bones,
        // and the vertex bounds must agree in the shared bind frame.
        assert!(!fbx.vertices.is_empty());
        for v in &fbx.vertices {
            let sum: f32 = v.weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-3, "weights must normalize: {v:?}");
            assert!(v.joints.iter().all(|&j| (j as usize) < fbx.skeleton.len()));
        }
        fn bounds(verts: &[crate::components::SkinnedVertexData]) -> ([f32; 3], [f32; 3]) {
            let mut lo = [f32::MAX; 3];
            let mut hi = [f32::MIN; 3];
            for v in verts {
                for i in 0..3 {
                    lo[i] = lo[i].min(v.pos[i]);
                    hi[i] = hi[i].max(v.pos[i]);
                }
            }
            (lo, hi)
        }
        let (flo, fhi) = bounds(&fbx.vertices);
        let (glo, ghi) = bounds(&glb.vertices);
        for i in 0..3 {
            assert!(
                (flo[i] - glo[i]).abs() < 0.05 && (fhi[i] - ghi[i]).abs() < 0.05,
                "vertex bounds differ: fbx {flo:?}..{fhi:?} vs glb {glo:?}..{ghi:?}"
            );
        }

        // Animation: sample both clips at the same normalized times and
        // compare the Tip joint's world position (its parent chain folds in
        // every evaluated channel plus PreRotation composition).
        let fbx_anim =
            crate::fbx::import_fbx_animation(&fbx_path, 0, "", 30.0, 0).expect("fbx animation");
        let glb_anim =
            crate::gltf::import_glb_animation(&glb_path, 0, 0, None).expect("glb animation");

        fn sample(
            tracks: &[crate::glb::ImportedAnimationTrack],
            joint: usize,
            t: f32,
            bind: &crate::components::SkeletonJoint,
        ) -> Mat4 {
            let bind_pose = JointPose {
                translation: bind.translation,
                rotation_deg: bind.rotation_deg,
                scale: bind.scale,
            };
            let Some(track) = tracks.iter().find(|tr| tr.joint == joint) else {
                return bind_pose.to_matrix();
            };
            let keys = &track.keys;
            if t <= keys[0].time {
                return keys[0].pose.to_matrix();
            }
            for pair in keys.windows(2) {
                if t <= pair[1].time {
                    let span = (pair[1].time - pair[0].time).max(1e-6);
                    let f = (t - pair[0].time) / span;
                    return pair[0].pose.blend_matrix(&pair[1].pose, f);
                }
            }
            keys[keys.len() - 1].pose.to_matrix()
        }

        fn joint_world_at(
            skeleton: &[crate::components::SkeletonJoint],
            tracks: &[crate::glb::ImportedAnimationTrack],
            name: &str,
            t: f32,
        ) -> [f32; 3] {
            let idx = skeleton
                .iter()
                .position(|j| j.name == name)
                .expect("joint by name");
            let mut chain: Vec<usize> = Vec::new();
            let mut cur = idx as i32;
            while cur >= 0 {
                chain.push(cur as usize);
                cur = skeleton[cur as usize].parent;
            }
            chain.reverse();
            let mut world = IDENTITY;
            for j in chain {
                world = mat4_mul(world, sample(tracks, j, t, &skeleton[j]));
            }
            // The joint's world origin. FBX rigs may carry unit-compensation
            // scale in ancestor frames, so a fixed local offset is not
            // comparable across containers; the origin is. Root rotation
            // sweeps the Tip origin, so rotation errors still surface.
            transform_point(world, [0.0, 0.0, 0.0])
        }

        for f in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let ft = f * fbx_anim.duration;
            let gt = f * glb_anim.duration;
            let fp = joint_world_at(&fbx.skeleton, &fbx_anim.tracks, "Tip", ft);
            let gp = joint_world_at(&glb.skeleton, &glb_anim.tracks, "Tip", gt);
            for i in 0..3 {
                assert!(
                    (fp[i] - gp[i]).abs() < 0.05,
                    "animated Tip end at t={f}: fbx {fp:?} vs glb {gp:?}"
                );
            }
        }
    }

    #[test]
    fn parse_fbx_reports_a_missing_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("missing.fbx");
        let err = parse_fbx(path.to_str().unwrap())
            .err()
            .expect("expected error");
        assert!(err.contains("could not open"), "got: {err}");
    }

    #[test]
    fn parse_fbx_rejects_non_fbx_content() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("junk.fbx");
        std::fs::write(&path, b"this is not an fbx file at all").expect("write junk");
        let err = parse_fbx(path.to_str().unwrap())
            .err()
            .expect("expected error");
        assert!(err.contains("not a valid FBX file"), "got: {err}");
    }

    // Node-tree accessors, read back off synthetic FBX documents.

    #[test]
    fn array_accessors_only_accept_their_own_element_type() {
        let tree = fx::tree(vec![
            fx::node("Arrays")
                .child(fx::node("F64").arr_f64(vec![1.0, 2.0]))
                .child(fx::node("I32").arr_i32(vec![3, 4]))
                .child(fx::node("I64").arr_i64(vec![5, 6]))
                .child(fx::node("F32").arr_f32(vec![7.0, 8.0]))
                .child(fx::node("NoAttributes")),
        ]);
        let arrays = tree.root().first_child_by_name("Arrays").expect("Arrays");
        let by = |name: &str| arrays.first_child_by_name(name).expect("child node");

        assert_eq!(arr_f64(&by("F64")), Some(&[1.0f64, 2.0][..]));
        assert_eq!(arr_i32(&by("I32")), Some(&[3i32, 4][..]));
        assert_eq!(arr_i64(&by("I64")), Some(&[5i64, 6][..]));
        assert_eq!(arr_f32(&by("F32")), Some(&[7.0f32, 8.0][..]));

        assert_eq!(arr_f64(&by("I32")), None);
        assert_eq!(arr_i32(&by("F64")), None);
        assert_eq!(arr_i64(&by("F32")), None);
        assert_eq!(arr_f32(&by("I64")), None);

        assert_eq!(arr_f64(&by("NoAttributes")), None);
        assert_eq!(arr_i32(&by("NoAttributes")), None);
        assert_eq!(arr_i64(&by("NoAttributes")), None);
        assert_eq!(arr_f32(&by("NoAttributes")), None);
    }

    #[test]
    fn child_str_reads_the_first_string_of_a_named_child() {
        let tree = fx::tree(vec![fx::texture(1, "Tex", "textures/wall.png")]);
        let tex = tree.root().first_child_by_name("Texture").expect("Texture");
        assert_eq!(
            child_str(&tex, "RelativeFilename"),
            Some("textures/wall.png")
        );
        assert_eq!(child_str(&tex, "FileName"), None);
    }

    #[test]
    fn object_name_strips_the_class_suffix() {
        let tree = fx::tree(vec![fx::model(77, "Cube", "Mesh"), fx::node("Model")]);
        let mut models = tree.root().children_by_name("Model");
        let named = models.next().expect("first Model");
        assert_eq!(object_id(&named), Some(77));
        assert_eq!(object_name(&named), "Cube");
        // An attribute-less node has neither an id nor a name.
        let bare = models.next().expect("second Model");
        assert_eq!(object_id(&bare), None);
        assert_eq!(object_name(&bare), "");
    }

    #[test]
    fn properties70_lookups_read_vectors_and_scalars() {
        let tree = fx::tree(vec![fx::properties70(vec![
            fx::p_vec3("Lcl Translation", [1.0, 2.0, 3.0]),
            fx::p_scalar("RotationOrder", 4.0),
            fx::node("P").text("Truncated").text("Vector3D"),
            fx::p_scalar("OneValue", 1.0),
            fx::node("P")
                .text("TwoValues")
                .text("Vector3D")
                .text("Vector")
                .text("A")
                .float(1.0)
                .float(2.0),
        ])]);
        let p70 = tree
            .root()
            .first_child_by_name("Properties70")
            .expect("Properties70");
        assert_eq!(prop_vec3(&p70, "Lcl Translation"), Some([1.0, 2.0, 3.0]));
        assert_eq!(prop_scalar(&p70, "RotationOrder"), Some(4.0));
        assert_eq!(prop_vec3(&p70, "Missing"), None);
        assert_eq!(prop_scalar(&p70, "Missing"), None);
        // P entries short of their value attributes read as absent.
        assert_eq!(prop_vec3(&p70, "Truncated"), None);
        assert_eq!(prop_scalar(&p70, "Truncated"), None);
        assert_eq!(prop_vec3(&p70, "OneValue"), None);
        assert_eq!(prop_vec3(&p70, "TwoValues"), None);
    }

    #[test]
    fn rot_ordered_applies_the_axes_in_the_declared_order() {
        // Each order permutes 90 degrees about all three axes differently, so
        // the image of (1, 2, 3) identifies the composition uniquely.
        let r = [90.0, 90.0, 90.0];
        let cases: [(i32, [f32; 3]); 7] = [
            (0, [3.0, 2.0, -1.0]),
            (1, [2.0, 1.0, -3.0]),
            (2, [-2.0, 1.0, 3.0]),
            (3, [-1.0, 3.0, 2.0]),
            (4, [1.0, -3.0, 2.0]),
            (5, [3.0, -2.0, 1.0]),
            // SphericXYZ is unsupported and falls back to XYZ.
            (6, [3.0, 2.0, -1.0]),
        ];
        for (order, expected) in cases {
            let got = transform_point(rot_ordered(r, order), [1.0, 2.0, 3.0]);
            assert_vec3_eq(got, expected);
        }
    }

    // Model transforms

    #[test]
    fn node_scene_local_scales_then_rotates_then_translates() {
        let tree = fx::tree(vec![fx::model(1, "Joint", "LimbNode").child(
            fx::properties70(vec![
                fx::p_vec3("Lcl Translation", [0.0, 1.0, 0.0]),
                fx::p_vec3("Lcl Rotation", [0.0, 0.0, 90.0]),
                fx::p_vec3("Lcl Scaling", [2.0, 2.0, 2.0]),
            ]),
        )]);
        let model = tree.root().first_child_by_name("Model").expect("Model");
        // (1,0,0) scales to (2,0,0), rotates about Z to (0,2,0), translates up.
        assert_vec3_eq(
            transform_point(node_scene_local(&model), [1.0, 0.0, 0.0]),
            [0.0, 3.0, 0.0],
        );
    }

    #[test]
    fn node_scene_local_applies_prerotation_before_the_local_rotation() {
        let tree = fx::tree(vec![fx::model(1, "Joint", "LimbNode").child(
            fx::properties70(vec![
                fx::p_vec3("Lcl Rotation", [0.0, 0.0, 90.0]),
                fx::p_vec3("PreRotation", [90.0, 0.0, 0.0]),
            ]),
        )]);
        let model = tree.root().first_child_by_name("Model").expect("Model");
        // Rz90 takes +X to +Y, then the PreRotation Rx90 takes +Y to +Z.
        assert_vec3_eq(
            transform_point(node_scene_local(&model), [1.0, 0.0, 0.0]),
            [0.0, 0.0, 1.0],
        );
    }

    #[test]
    fn node_scene_local_honours_the_rotation_order_property() {
        let tree = fx::tree(vec![fx::model(1, "Joint", "LimbNode").child(
            fx::properties70(vec![
                fx::p_vec3("Lcl Rotation", [90.0, 0.0, 90.0]),
                fx::p_scalar("RotationOrder", 5.0),
            ]),
        )]);
        let model = tree.root().first_child_by_name("Model").expect("Model");
        // ZYX applies Z first: +X to +Y, then Rx90 to +Z. XYZ would end at +Y.
        assert_vec3_eq(
            transform_point(node_scene_local(&model), [1.0, 0.0, 0.0]),
            [0.0, 0.0, 1.0],
        );
    }

    #[test]
    fn node_scene_local_without_properties_is_identity() {
        let tree = fx::tree(vec![fx::model(1, "Bare", "Null")]);
        let model = tree.root().first_child_by_name("Model").expect("Model");
        assert_eq!(node_scene_local(&model), IDENTITY);
    }

    #[test]
    fn local_matrices_separate_the_node_and_geometric_transforms() {
        let tree = fx::tree(vec![
            fx::model(1, "Mesh", "Mesh").child(fx::properties70(vec![
                fx::p_vec3("Lcl Translation", [5.0, 0.0, 0.0]),
                fx::p_vec3("GeometricTranslation", [0.0, 0.0, 2.0]),
                fx::p_vec3("GeometricScaling", [2.0, 2.0, 2.0]),
            ])),
            fx::model(2, "Bare", "Null"),
        ]);
        let mut models = tree.root().children_by_name("Model");
        let (local, geometric) = local_matrices(&models.next().expect("first Model"));
        assert_vec3_eq(transform_point(local, [0.0, 0.0, 0.0]), [5.0, 0.0, 0.0]);
        // The geometric offset is a transform of its own, not folded into the node.
        assert_vec3_eq(transform_point(geometric, [1.0, 0.0, 0.0]), [2.0, 0.0, 2.0]);

        let (local, geometric) = local_matrices(&models.next().expect("second Model"));
        assert_eq!(local, IDENTITY);
        assert_eq!(geometric, IDENTITY);
    }

    #[test]
    fn unit_scale_to_meters_reads_the_global_unit_factor() {
        let meters = fx::tree(vec![fx::global_settings(100.0)]);
        assert!((unit_scale_to_meters(&meters.root()) - 1.0).abs() < 1e-6);
        let centimeters = fx::tree(vec![fx::global_settings(1.0)]);
        assert!((unit_scale_to_meters(&centimeters.root()) - 0.01).abs() < 1e-6);
        // No GlobalSettings: FBX's native centimeter unit.
        let bare = fx::tree(vec![fx::node("Objects")]);
        assert!((unit_scale_to_meters(&bare.root()) - 0.01).abs() < 1e-6);
    }

    #[test]
    fn resolve_texture_path_rebases_onto_the_fbx_directory() {
        let tree = fx::tree(vec![
            fx::texture(1, "Backslashes", "textures\\wall.png"),
            fx::object("Texture", 2, "Absolute", "")
                .child(fx::node("FileName").text("/authors/machine/wall.png")),
            fx::object("Texture", 3, "RootOnly", "").child(fx::node("FileName").text("/")),
            fx::object("Texture", 4, "Unresolvable", ""),
        ]);
        let dir = Path::new("/scene");
        let mut texs = tree.root().children_by_name("Texture");
        let mut next = || resolve_texture_path(&texs.next().expect("Texture"), dir);

        assert_eq!(next().as_deref(), Some("/scene/textures/wall.png"));
        // An absolute author-machine path keeps only its file name.
        assert_eq!(next().as_deref(), Some("/scene/wall.png"));
        assert_eq!(next().as_deref(), Some("/"));
        assert_eq!(next(), None);
    }

    // parse_fbx

    #[test]
    fn parse_fbx_reports_a_file_without_an_objects_section() {
        let file = fx::write(vec![fx::node("Definitions")]);
        let err = parse_fbx(file.path()).err().expect("no Objects section");
        assert!(err.contains("FBX has no Objects section"), "got: {err}");
    }

    #[test]
    fn parse_fbx_extracts_materials_props_and_per_material_primitives() {
        let mut doc = fx::doc();
        doc.objects = vec![
            fx::geometry(
                100,
                "quad",
                vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                vec![0, 1, -3, 0, 2, -4],
            )
            .child(fx::uv_layer(
                "TextureUV",
                vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                None,
            ))
            .child(fx::material_layer("ByPolygon", vec![0, 1])),
            fx::model(200, "Cube", "Mesh").child(fx::properties70(vec![fx::p_vec3(
                "Lcl Translation",
                [5.0, 0.0, 0.0],
            )])),
            fx::material(300, "Red").child(fx::properties70(vec![
                fx::p_vec3("DiffuseColor", [1.0, 0.0, 0.0]),
                fx::p_vec3("EmissiveColor", [0.0, 0.5, 0.0]),
                fx::p_scalar("Opacity", 0.25),
            ])),
            fx::material(301, "Blue"),
            fx::texture(400, "Albedo", "textures/albedo.png"),
            fx::texture(401, "Normal", "textures/normal.png"),
            fx::texture(402, "Orm", "textures/orm.png"),
            fx::texture(403, "Emissive", "textures/emissive.png"),
        ];
        doc.connections = vec![
            fx::oo(100, 200),
            fx::oo(200, 0),
            fx::oo(300, 200),
            fx::oo(301, 200),
            fx::op(400, 300, "DiffuseColor"),
            fx::op(401, 300, "NormalMap"),
            fx::op(402, 300, "SpecularColor"),
            fx::op(403, 300, "EmissiveColor"),
            // An unrecognised texture slot is ignored.
            fx::op(401, 300, "AmbientColor"),
        ];
        let file = doc.write();
        let scene = parse_fbx(file.path()).expect("parse");

        assert_eq!(scene.materials.len(), 2);
        let red = &scene.materials[0];
        assert_eq!(red.name, "Red");
        assert_vec3_eq(red.diffuse, [1.0, 0.0, 0.0]);
        assert_vec3_eq(red.emissive_factor, [0.0, 0.5, 0.0]);
        assert!((red.opacity - 0.25).abs() < 1e-6);
        let dir = file.dir();
        assert_eq!(red.albedo, Some(fx::under(dir, "textures/albedo.png")));
        assert_eq!(red.normal, Some(fx::under(dir, "textures/normal.png")));
        assert_eq!(red.orm, Some(fx::under(dir, "textures/orm.png")));
        assert_eq!(red.emissive, Some(fx::under(dir, "textures/emissive.png")));

        let blue = &scene.materials[1];
        assert_eq!(blue.name, "Blue");
        assert_vec3_eq(blue.diffuse, [1.0, 1.0, 1.0]);
        assert_vec3_eq(blue.emissive_factor, [0.0, 0.0, 0.0]);
        assert!((blue.opacity - 1.0).abs() < 1e-6);
        assert_eq!(blue.albedo, None);

        assert_eq!(scene.props.len(), 1);
        let prop = &scene.props[0];
        assert_eq!(prop.name, "Cube");
        assert_vec3_eq(prop.position, [5.0, 0.0, 0.0]);
        assert_vec3_eq(prop.scale, [1.0, 1.0, 1.0]);
        assert_eq!(prop.primitives, vec![0, 1]);

        // One primitive per material, ordered by material index.
        assert_eq!(scene.primitives.len(), 2);
        assert_eq!(scene.primitives[0].material, Some(0));
        assert_eq!(scene.primitives[1].material, Some(1));
        let first = &scene.primitives[0];
        assert_eq!(first.indices, vec![0, 1, 2]);
        assert_vec3_eq(first.vertices[0].pos, [0.0, 0.0, 0.0]);
        assert_vec3_eq(first.vertices[1].pos, [1.0, 0.0, 0.0]);
        assert_vec3_eq(first.vertices[2].pos, [1.0, 1.0, 0.0]);
        // V is flipped from FBX bottom-up to top-down.
        assert_eq!(first.vertices[0].uv, [0.0, 1.0]);
        assert_eq!(first.vertices[1].uv, [1.0, 1.0]);
        assert_eq!(first.vertices[2].uv, [1.0, 0.0]);
        assert_vec3_eq(first.vertices[0].color, NEUTRAL_COLOR);

        // The geometry is local; the AABB is the node-transformed box.
        let (min, max) = scene.aabb.expect("aabb");
        assert_vec3_eq(min, [5.0, 0.0, 0.0]);
        assert_vec3_eq(max, [6.0, 1.0, 0.0]);
    }

    // A single-triangle mesh whose polygons carry the given material layer.
    fn material_layer_scene(
        layer: Option<fx::Node>,
        pvi: Vec<i32>,
        materials: Vec<i64>,
    ) -> FbxScene {
        let mut geom = fx::geometry(
            100,
            "mesh",
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            pvi,
        );
        if let Some(layer) = layer {
            geom = geom.child(layer);
        }
        let mut doc = fx::doc();
        doc.objects = vec![geom, fx::model(200, "Mesh", "Mesh")];
        doc.connections = vec![fx::oo(100, 200), fx::oo(200, 0)];
        for id in &materials {
            doc.objects.push(fx::material(*id, "M"));
            doc.connections.push(fx::oo(*id, 200));
        }
        let file = doc.write();
        parse_fbx(file.path()).expect("parse")
    }

    #[test]
    fn parse_fbx_applies_an_all_same_material_to_every_polygon() {
        let scene = material_layer_scene(
            Some(fx::material_layer("AllSame", vec![1])),
            vec![0, 1, -3, 0, 2, -4],
            vec![300, 301],
        );
        assert_eq!(scene.primitives.len(), 1);
        assert_eq!(scene.primitives[0].material, Some(1));
        assert_eq!(scene.primitives[0].indices.len(), 6);
    }

    #[test]
    fn parse_fbx_leaves_geometry_without_a_material_layer_unassigned() {
        let scene = material_layer_scene(None, vec![0, 1, -3], vec![300]);
        assert_eq!(scene.primitives.len(), 1);
        assert_eq!(scene.primitives[0].material, None);
    }

    #[test]
    fn parse_fbx_sorts_unresolved_material_groups_last() {
        // Polygon 0 is unassigned, 1 uses material 0, 2 indexes past the model's
        // material list; both unresolved polygons share the trailing group.
        let scene = material_layer_scene(
            Some(fx::material_layer("ByPolygon", vec![-1, 0, 5])),
            vec![0, 1, -3, 0, 2, -4, 1, 2, -4],
            vec![300],
        );
        assert_eq!(scene.primitives.len(), 2);
        assert_eq!(scene.primitives[0].material, Some(0));
        assert_eq!(scene.primitives[0].vertices.len(), 3);
        assert_eq!(scene.primitives[1].material, None);
        assert_eq!(scene.primitives[1].vertices.len(), 6);
    }

    #[test]
    fn parse_fbx_triangulates_a_quad_as_a_fan() {
        let scene = material_layer_scene(None, vec![0, 1, 2, -4], Vec::new());
        assert_eq!(scene.primitives.len(), 1);
        assert_eq!(scene.primitives[0].vertices.len(), 4);
        assert_eq!(scene.primitives[0].indices, vec![0, 1, 2, 0, 2, 3]);
    }

    #[test]
    fn parse_fbx_drops_polygons_with_fewer_than_three_corners() {
        // Two corners produce no triangles, so the model contributes no prop.
        let scene = material_layer_scene(None, vec![0, -2], Vec::new());
        assert!(scene.primitives.is_empty());
        assert!(scene.props.is_empty());
        assert!(scene.aabb.is_none());
    }

    #[test]
    fn parse_fbx_skips_geometry_without_polygon_data() {
        let mut doc = fx::doc();
        doc.objects = vec![
            fx::object("Geometry", 100, "mesh", "Mesh")
                .child(fx::node("Vertices").arr_f64(vec![0.0, 0.0, 0.0])),
            fx::model(200, "Mesh", "Mesh"),
        ];
        doc.connections = vec![fx::oo(100, 200), fx::oo(200, 0)];
        let file = doc.write();
        let scene = parse_fbx(file.path()).expect("parse");
        assert!(scene.primitives.is_empty());
        assert!(scene.props.is_empty());
    }

    #[test]
    fn parse_fbx_prefers_the_texture_uv_layer_over_other_uv_sets() {
        let mut doc = fx::doc();
        doc.objects = vec![
            fx::geometry(
                100,
                "mesh",
                vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                vec![0, 1, -3],
            )
            .child(fx::uv_layer(
                "Lightmap",
                vec![0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
                None,
            ))
            .child(fx::uv_layer(
                "TextureUV",
                vec![0.25, 0.0, 0.25, 0.0, 0.25, 0.0],
                None,
            )),
            fx::model(200, "Mesh", "Mesh"),
        ];
        doc.connections = vec![fx::oo(100, 200), fx::oo(200, 0)];
        let file = doc.write();
        let scene = parse_fbx(file.path()).expect("parse");
        assert_eq!(scene.primitives[0].vertices[0].uv, [0.25, 1.0]);
    }

    #[test]
    fn parse_fbx_deduplicates_vertices_sharing_an_indexed_uv() {
        let mut doc = fx::doc();
        doc.objects = vec![
            fx::geometry(
                100,
                "quad",
                vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                vec![0, 1, -3, 0, 2, -4],
            )
            .child(fx::uv_layer(
                "TextureUV",
                vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                Some(vec![0, 1, 2, 0, 2, 3]),
            )),
            fx::model(200, "Mesh", "Mesh"),
        ];
        doc.connections = vec![fx::oo(100, 200), fx::oo(200, 0)];
        let file = doc.write();
        let scene = parse_fbx(file.path()).expect("parse");
        // Six polygon-vertices collapse onto the four distinct (point, uv) pairs.
        assert_eq!(scene.primitives[0].vertices.len(), 4);
        assert_eq!(scene.primitives[0].indices, vec![0, 1, 2, 0, 2, 3]);
        assert_eq!(scene.primitives[0].vertices[3].uv, [0.0, 0.0]);
    }

    #[test]
    fn parse_fbx_derives_opacity_from_transparency_and_clamps_it() {
        let mut doc = fx::doc();
        doc.objects = vec![
            fx::material(300, "Glass").child(fx::properties70(vec![fx::p_scalar(
                "TransparencyFactor",
                0.25,
            )])),
            fx::material(301, "Overbright")
                .child(fx::properties70(vec![fx::p_scalar("Opacity", 2.0)])),
            fx::material(302, "Negative")
                .child(fx::properties70(vec![fx::p_scalar("Opacity", -1.0)])),
            // `Emissive` is the fallback spelling of `EmissiveColor`.
            fx::material(303, "Lamp").child(fx::properties70(vec![fx::p_vec3(
                "Emissive",
                [0.0, 0.0, 4.0],
            )])),
        ];
        let file = doc.write();
        let scene = parse_fbx(file.path()).expect("parse");
        assert!((scene.materials[0].opacity - 0.75).abs() < 1e-6);
        assert!((scene.materials[1].opacity - 1.0).abs() < 1e-6);
        assert!((scene.materials[2].opacity - 0.0).abs() < 1e-6);
        assert_vec3_eq(scene.materials[3].emissive_factor, [0.0, 0.0, 4.0]);
    }

    #[test]
    fn parse_fbx_skips_objects_without_an_id() {
        let mut doc = fx::doc();
        doc.objects = vec![
            fx::node("Material"),
            fx::node("Model"),
            fx::material(300, "Only"),
            // An object kind the static-scene import does not index.
            fx::object("Deformer", 400, "skin", "Skin"),
        ];
        let file = doc.write();
        let scene = parse_fbx(file.path()).expect("parse");
        assert_eq!(scene.materials.len(), 1);
        assert_eq!(scene.materials[0].name, "Only");
        assert!(scene.props.is_empty());
    }

    #[test]
    fn parse_fbx_ignores_malformed_and_unrelated_connections() {
        let mut doc = fx::doc();
        doc.objects = vec![
            fx::geometry(
                100,
                "mesh",
                vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                vec![0, 1, -3],
            )
            .child(fx::material_layer("AllSame", vec![0])),
            fx::model(200, "Mesh", "Mesh"),
            fx::material(300, "Red"),
        ];
        doc.connections = vec![
            fx::oo(100, 200),
            fx::oo(200, 0),
            fx::oo(300, 200),
            // A record without object ids.
            fx::node("C").text("OO"),
            // A property link whose child is a material, not a texture.
            fx::op(300, 200, "DiffuseColor"),
            // An unknown connection kind.
            fx::node("C").text("PP").int64(300).int64(200),
        ];
        let file = doc.write();
        let scene = parse_fbx(file.path()).expect("parse");
        assert_eq!(scene.props.len(), 1);
        assert_eq!(scene.primitives[0].material, Some(0));
        assert_eq!(scene.materials[0].albedo, None);
    }

    #[test]
    fn parse_fbx_without_connections_finds_no_geometry_for_its_models() {
        let mut doc = fx::doc();
        doc.objects = vec![
            fx::geometry(100, "mesh", vec![0.0; 9], vec![0, 1, -3]),
            fx::model(200, "Mesh", "Mesh"),
            fx::material(300, "Red"),
        ];
        let file = fx::write(vec![fx::objects(doc.objects)]);
        let scene = parse_fbx(file.path()).expect("parse");
        assert_eq!(scene.materials.len(), 1);
        assert!(scene.props.is_empty());
        assert!(scene.primitives.is_empty());
    }

    #[test]
    fn parse_fbx_folds_the_parent_chain_into_the_prop_transform() {
        let mut doc = fx::doc();
        doc.objects = vec![
            fx::geometry(
                100,
                "mesh",
                vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                vec![0, 1, -3],
            ),
            fx::model(200, "Child", "Mesh").child(fx::properties70(vec![fx::p_vec3(
                "Lcl Translation",
                [1.0, 0.0, 0.0],
            )])),
            fx::model(210, "Parent", "Null").child(fx::properties70(vec![fx::p_vec3(
                "Lcl Translation",
                [0.0, 10.0, 0.0],
            )])),
        ];
        doc.connections = vec![
            fx::oo(100, 200),
            fx::oo(200, 210),
            fx::oo(210, 0),
            // A geometry connected to a non-model parent is not a mesh binding.
            fx::oo(100, 999),
        ];
        let file = doc.write();
        let scene = parse_fbx(file.path()).expect("parse");
        assert_eq!(scene.props.len(), 1);
        assert_vec3_eq(scene.props[0].position, [1.0, 10.0, 0.0]);
    }
}
