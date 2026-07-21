// src/glb.rs
//
// glTF geometry / skeleton / animation import: turns a parsed glTF document
// (binary `.glb` or text `.gltf`, see `crate::gltf_source`) into the engine's
// inline mesh / skeleton / animation forms.
//
// glTF stores a skin's joints in an arbitrary order; this engine's `JointDef`
// list requires parents before children. Joints are therefore topologically
// reordered and a remap table rewrites both each joint's parent index and
// every vertex's `JOINTS_0` binding into the new index space.
//
// This is the decode half of the glTF pipeline. The asset-level desugar
// wrappers (`import_skinned_glb`, `import_glb_animation`, ...) live in
// `crate::gltf` and call into here.

use std::collections::HashMap;

use crate::assets::{JointDef, SkinnedVertexData, VertexData};
use crate::gfx::skinning::{JointPose, euler_yxz_from_quat};
use crate::gltf_source::GltfDoc;

// Neutral grey for vertex color, matches the wavefront/OBJ importer so
// imported geometry takes the material albedo without tinting.
const NEUTRAL_COLOR: [f32; 3] = [0.75, 0.74, 0.72];

// The inline `SkinnedMesh` fields produced from a glTF file.
pub struct ImportedSkinnedMesh {
    pub vertices: Vec<SkinnedVertexData>,
    pub indices: Vec<u16>,
    pub skeleton: Vec<JointDef>,
    // Morph-target names, one per target; empty when the mesh has none.
    pub morph_target_names: Vec<String>,
    // Dense target-major deltas: entry `t * vertices.len() + v`.
    pub morph_deltas: Vec<crate::assets::MorphDelta>,
}

// Same as [`import_skinned_glb`] but takes a pre-parsed glTF document. The
// asset hot-reload pass uses this directly so it can amortise the `.glb`
// parse across every Mesh / SkinnedMesh entry that references the same file.
pub fn import_skinned_from_doc(
    doc: &GltfDoc,
    source: &str,
    skin_index: u32,
) -> Result<ImportedSkinnedMesh, String> {
    let node = skinned_node(doc, source, skin_index)?;
    let mesh = node.mesh().unwrap();
    let skin = node.skin().unwrap();

    let skeleton = import_skeleton(&skin)?;
    let (vertices, indices, morph_deltas) = import_geometry(&mesh, doc, &skeleton.remap)?;
    let target_count = if vertices.is_empty() {
        0
    } else {
        morph_deltas.len() / vertices.len()
    };
    let morph_target_names = morph_target_names(&mesh, target_count);

    Ok(ImportedSkinnedMesh {
        vertices,
        indices,
        skeleton: skeleton.joints,
        morph_target_names,
        morph_deltas,
    })
}

// Target names from the mesh extras `targetNames` convention, padded or
// truncated to `target_count`; a missing name falls back to `target_{i}`.
fn morph_target_names(mesh: &gltf::Mesh<'_>, target_count: usize) -> Vec<String> {
    let from_extras: Vec<String> = mesh
        .extras()
        .as_deref()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw.get()).ok())
        .and_then(|v| {
            v.get("targetNames").map(|names| {
                names
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .map(|n| n.as_str().unwrap_or("").to_string())
                            .collect()
                    })
                    .unwrap_or_default()
            })
        })
        .unwrap_or_default();
    (0..target_count)
        .map(|i| {
            from_extras
                .get(i)
                .filter(|n| !n.is_empty())
                .cloned()
                .unwrap_or_else(|| format!("target_{i}"))
        })
        .collect()
}

// Parse a `.glb` or `.gltf` file from disk with every buffer resolved. Shared
// by the skinned and static importers; the desugar pass uses this directly so
// it can memoize one document across many primitive/material/image lookups.
pub fn parse_glb(source: &str) -> Result<GltfDoc, String> {
    GltfDoc::parse_file(source)
}

// Import the indexed primitive (flattened across glTF meshes in declaration
// order) from a parsed `.glb` into a static `Mesh`'s `(vertices, indices)`.
// UVs are taken from `TEXCOORD_0` when present; vertex colors fall back to
// neutral grey so the material albedo controls surface color.
//
// Only TRIANGLES topology is supported. POINTS/LINES/strip variants
// error out so a regression is obvious rather than silently mis-rendered.
//
// Errors if the primitive's vertex count or any index exceeds Concinnity's
// u16 index limit; `cn add` pre-splits oversized primitives at add time so
// the desugar pass never encounters one.
pub fn import_static_glb_primitive_from_doc(
    doc: &GltfDoc,
    source: &str,
    primitive_index: u32,
) -> Result<(Vec<VertexData>, Vec<u16>), String> {
    let (vertices, indices_u32) = read_primitive_geometry(doc, source, primitive_index)?;
    let mut indices: Vec<u16> = Vec::with_capacity(indices_u32.len());
    for v in indices_u32 {
        if v > u16::MAX as u32 {
            return Err(format!(
                "'{}': primitive {} exceeds the {}-vertex u16 index limit; \
                 import via `cn add` to auto-split it",
                source,
                primitive_index,
                u16::MAX
            ));
        }
        indices.push(v as u16);
    }
    Ok((vertices, indices))
}

// Read a primitive's vertices and u32-indexed triangle list from a parsed
// `.glb`. The shared backbone of [`import_static_glb_primitive_from_doc`]
// and the `cn add` splitting path; neither caller needs to repeat the
// triangle-topology check or attribute reads.
pub fn read_primitive_geometry(
    doc: &GltfDoc,
    source: &str,
    primitive_index: u32,
) -> Result<(Vec<VertexData>, Vec<u32>), String> {
    let primitive = doc
        .doc
        .document
        .meshes()
        .flat_map(|m| m.primitives())
        .nth(primitive_index as usize)
        .ok_or_else(|| {
            format!(
                "'{}': primitive_index {} is out of range",
                source, primitive_index
            )
        })?;

    if primitive.mode() != gltf::mesh::Mode::Triangles {
        return Err(format!(
            "'{}': primitive {} uses topology {:?}; only TRIANGLES is supported",
            source,
            primitive_index,
            primitive.mode()
        ));
    }

    let reader = primitive.reader(|b| doc.buffer_bytes(b));

    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .ok_or_else(|| {
            format!(
                "'{}': primitive {} has no POSITION data (missing buffer data)",
                source, primitive_index
            )
        })?
        .collect();
    let uvs: Vec<[f32; 2]> = reader
        .read_tex_coords(0)
        .map(|t| t.into_f32().collect())
        .unwrap_or_default();
    let colors: Vec<[f32; 3]> = reader
        .read_colors(0)
        .map(|c| c.into_rgb_f32().collect())
        .unwrap_or_default();

    let mut vertices: Vec<VertexData> = Vec::with_capacity(positions.len());
    for (i, &pos) in positions.iter().enumerate() {
        vertices.push(VertexData {
            pos,
            color: colors.get(i).copied().unwrap_or(NEUTRAL_COLOR),
            uv: uvs.get(i).copied().unwrap_or([0.0, 0.0]),
        });
    }

    let indices: Vec<u32> = match reader.read_indices() {
        Some(idx) => idx.into_u32().collect(),
        None => (0..positions.len() as u32).collect(),
    };

    if vertices.is_empty() {
        return Err(format!(
            "'{}': primitive {} has no vertices",
            source, primitive_index
        ));
    }
    // Indices come straight from the (untrusted) glTF index buffer. Reject any
    // that reference past the vertex array here so downstream consumers like
    // split_into_u16_chunks can index `vertices` without bounds checks.
    if let Some(&bad) = indices.iter().find(|&&i| i as usize >= vertices.len()) {
        return Err(format!(
            "'{}': primitive {} index {} out of range ({} vertices)",
            source,
            primitive_index,
            bad,
            vertices.len()
        ));
    }
    Ok((vertices, indices))
}

// Split an oversized triangle list (any vertex count, u32 indices) into
// chunks that each fit in u16. Each chunk has its own vertex buffer containing
// only the vertices its triangles use; indices are remapped to that local
// buffer. The split is greedy by triangle order: fast and stable, with no
// attempt at locality optimisation. Vertices are duplicated across chunks
// when a triangle straddles a flush boundary; for chess-piece geometry the
// duplication is well under one percent.
pub fn split_into_u16_chunks(
    vertices: &[VertexData],
    indices: &[u32],
) -> Vec<(Vec<VertexData>, Vec<u16>)> {
    let limit: usize = u16::MAX as usize + 1;
    let mut chunks: Vec<(Vec<VertexData>, Vec<u16>)> = Vec::new();
    let mut cur_verts: Vec<VertexData> = Vec::new();
    let mut cur_indices: Vec<u16> = Vec::new();
    let mut remap: std::collections::HashMap<u32, u16> = std::collections::HashMap::new();

    for tri in indices.chunks_exact(3) {
        let new_in_tri = tri.iter().filter(|&&v| !remap.contains_key(&v)).count();
        if !cur_verts.is_empty() && cur_verts.len() + new_in_tri > limit {
            chunks.push((
                std::mem::take(&mut cur_verts),
                std::mem::take(&mut cur_indices),
            ));
            remap.clear();
        }
        for &v in tri {
            let local = *remap.entry(v).or_insert_with(|| {
                let idx = cur_verts.len() as u16;
                cur_verts.push(vertices[v as usize].clone());
                idx
            });
            cur_indices.push(local);
        }
    }
    if !cur_verts.is_empty() {
        chunks.push((cur_verts, cur_indices));
    }
    chunks
}

// Resolve a `SkinnedMesh.source` to an on-disk path. A bare filename (no
// directory component) is searched for under the fetched-assets directory
// (the same resolution `ColorLut` and `EnvironmentMap` sources use) while a
// path with a directory component is taken as-is, so a relative or absolute
// path still works for local test worlds.
pub fn resolve_source(source: &str) -> String {
    let bare = std::path::Path::new(source)
        .parent()
        .map(|d| d.as_os_str().is_empty())
        .unwrap_or(true);
    if !bare {
        return source.to_string();
    }
    if let Some(found) = crate::paths::find_in_assets(source) {
        return found;
    }
    concinnity_core::paths::assets_dir()
        .join(source)
        .to_string_lossy()
        .into_owned()
}

// A skeleton reordered into parents-before-children order, plus the lookup
// tables an animation importer needs to resolve a glTF channel target back
// to a joint in this skeleton.
pub struct ImportedSkeleton {
    pub joints: Vec<JointDef>,
    // `remap[skin_joint_index] = topologically-sorted index`.
    pub remap: Vec<usize>,
    // `node_to_joint[glTF_node_index] = skin_joint_index`. An animation
    // channel whose target node is missing from this map is targeting a
    // non-joint node and should be dropped.
    pub node_to_joint: HashMap<usize, usize>,
}

// Build the engine's skeleton (joints in parents-before-children order) from
// a glTF skin. Public so the animation importer can reuse the remap +
// node-to-joint table without re-deriving them.
pub fn import_skeleton(skin: &gltf::Skin<'_>) -> Result<ImportedSkeleton, String> {
    let joint_nodes: Vec<gltf::Node<'_>> = skin.joints().collect();
    let n = joint_nodes.len();
    if n == 0 {
        return Err("glTF skin has no joints".to_string());
    }

    // glTF node index -> skin-joint index, so a node's children can be
    // resolved back to joints.
    let node_to_joint: HashMap<usize, usize> = joint_nodes
        .iter()
        .enumerate()
        .map(|(sj, node)| (node.index(), sj))
        .collect();

    // A joint's parent is whichever joint lists it as a child. Joints not
    // claimed by any other joint are roots. A self-parent is dropped.
    let mut parents: Vec<Option<usize>> = vec![None; n];
    for (sj, node) in joint_nodes.iter().enumerate() {
        for child in node.children() {
            if let Some(&cj) = node_to_joint.get(&child.index())
                && cj != sj
            {
                parents[cj] = Some(sj);
            }
        }
    }

    let (order, remap) = topological_order(&parents);

    let joints = order
        .iter()
        .map(|&sj| {
            let node = &joint_nodes[sj];
            let (translation, rotation, scale) = node.transform().decomposed();
            JointDef {
                name: node.name().unwrap_or("").to_string(),
                parent: parents[sj].map_or(-1, |p| remap[p] as i32),
                translation,
                rotation_deg: euler_yxz_from_quat(rotation),
                scale,
            }
        })
        .collect();

    Ok(ImportedSkeleton {
        joints,
        remap,
        node_to_joint,
    })
}

// Order joint indices so every parent precedes its children, and return the
// inverse remap (skin-joint index -> sorted index). A joint whose parent
// chain cannot be resolved (a cycle, which a valid glTF skin never has) is
// emitted in its original order as a safety fallback so the function always
// returns a total order over every joint.
fn topological_order(parents: &[Option<usize>]) -> (Vec<usize>, Vec<usize>) {
    let n = parents.len();
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut emitted = vec![false; n];

    loop {
        let progress = order.len();
        for (s, parent) in parents.iter().enumerate() {
            if emitted[s] {
                continue;
            }
            let ready = match *parent {
                None => true,
                Some(p) => p >= n || emitted[p],
            };
            if ready {
                emitted[s] = true;
                order.push(s);
            }
        }
        if order.len() == progress {
            break;
        }
    }
    // Stragglers (a cycle): emit in original order rather than dropping them.
    for (s, done) in emitted.iter().enumerate() {
        if !done {
            order.push(s);
        }
    }

    let mut remap = vec![0usize; n];
    for (new_idx, &s) in order.iter().enumerate() {
        remap[s] = new_idx;
    }
    (order, remap)
}

// Concatenated skinned geometry of one glTF mesh: vertices, u16 indices, and
// dense target-major morph deltas.
type SkinnedGeometry = (
    Vec<SkinnedVertexData>,
    Vec<u16>,
    Vec<crate::assets::MorphDelta>,
);

// One primitive's morph targets: per-vertex position and normal deltas.
type PrimTargetDeltas = (Vec<[f32; 3]>, Vec<[f32; 3]>);

fn import_geometry(
    mesh: &gltf::Mesh<'_>,
    doc: &GltfDoc,
    remap: &[usize],
) -> Result<SkinnedGeometry, String> {
    use crate::assets::MorphDelta;

    let mut vertices: Vec<SkinnedVertexData> = Vec::new();
    let mut indices: Vec<u16> = Vec::new();
    // Per-target deltas kept parallel to `vertices`; a primitive that lacks a
    // target other primitives declare contributes zero deltas for its range.
    let mut targets: Vec<Vec<MorphDelta>> = Vec::new();
    let mut tangent_deltas_seen = false;

    for primitive in mesh.primitives() {
        let reader = primitive.reader(|b| doc.buffer_bytes(b));

        // A primitive with no JOINTS_0 is static geometry, skip it; the
        // SkinnedMesh asset only carries skinned vertices.
        let joints: Vec<[u16; 4]> = match reader.read_joints(0) {
            Some(j) => j.into_u16().collect(),
            None => continue,
        };
        let positions: Vec<[f32; 3]> = reader
            .read_positions()
            .ok_or_else(|| {
                "skinned primitive has no POSITION data (missing buffer data)".to_string()
            })?
            .collect();
        let weights: Vec<[f32; 4]> = reader
            .read_weights(0)
            .ok_or_else(|| "skinned primitive missing WEIGHTS_0".to_string())?
            .into_f32()
            .collect();
        let uvs: Vec<[f32; 2]> = reader
            .read_tex_coords(0)
            .map(|t| t.into_f32().collect())
            .unwrap_or_default();
        let colors: Vec<[f32; 3]> = reader
            .read_colors(0)
            .map(|c| c.into_rgb_f32().collect())
            .unwrap_or_default();

        let base = vertices.len() as u32;

        let prim_targets: Vec<PrimTargetDeltas> = reader
            .read_morph_targets()
            .map(|(dp, dn, dt)| {
                tangent_deltas_seen |= dt.is_some();
                (
                    dp.map(|it| it.collect()).unwrap_or_default(),
                    dn.map(|it| it.collect()).unwrap_or_default(),
                )
            })
            .collect();
        for t in 0..targets.len().max(prim_targets.len()) {
            if targets.len() <= t {
                targets.push(vec![MorphDelta::default(); base as usize]);
            }
            let dst = &mut targets[t];
            match prim_targets.get(t) {
                Some((dp, dn)) => {
                    for i in 0..positions.len() {
                        dst.push(MorphDelta {
                            position: dp.get(i).copied().unwrap_or_default(),
                            normal: dn.get(i).copied().unwrap_or_default(),
                        });
                    }
                }
                None => {
                    dst.extend(std::iter::repeat_with(MorphDelta::default).take(positions.len()));
                }
            }
        }

        for (i, &pos) in positions.iter().enumerate() {
            let raw = joints.get(i).copied().unwrap_or([0; 4]);
            let bound = |j: u16| -> u32 {
                // A JOINTS_0 index always indexes the skin's joint array;
                // remap it into the topologically-sorted index space.
                remap.get(j as usize).map_or(0, |&r| r as u32)
            };
            vertices.push(SkinnedVertexData {
                pos,
                color: colors.get(i).copied().unwrap_or([1.0, 1.0, 1.0]),
                uv: uvs.get(i).copied().unwrap_or([0.0, 0.0]),
                joints: [bound(raw[0]), bound(raw[1]), bound(raw[2]), bound(raw[3])],
                weights: weights.get(i).copied().unwrap_or([1.0, 0.0, 0.0, 0.0]),
            });
        }

        let push_index = |indices: &mut Vec<u16>, v: u32| -> Result<(), String> {
            let abs = base + v;
            if abs > u16::MAX as u32 {
                return Err(format!(
                    "imported skinned mesh exceeds the {}-vertex u16 index limit",
                    u16::MAX
                ));
            }
            indices.push(abs as u16);
            Ok(())
        };
        match reader.read_indices() {
            Some(idx) => {
                for v in idx.into_u32() {
                    push_index(&mut indices, v)?;
                }
            }
            None => {
                // A non-indexed primitive draws vertices sequentially.
                for v in 0..positions.len() as u32 {
                    push_index(&mut indices, v)?;
                }
            }
        }
    }

    if vertices.is_empty() {
        return Err("glTF mesh has no skinned primitives (no JOINTS_0)".to_string());
    }
    if tangent_deltas_seen {
        tracing::info!("glTF morph targets carry tangent deltas; they are not imported");
    }
    let total = vertices.len();
    let mut morph_deltas = Vec::with_capacity(targets.len() * total);
    for mut t in targets {
        t.resize(total, crate::assets::MorphDelta::default());
        morph_deltas.extend(t);
    }
    Ok((vertices, indices, morph_deltas))
}

// glTF animation import

// A single keyframe extracted from a glTF animation channel.
#[derive(Debug, Clone, Copy)]
pub struct ImportedKeyframe {
    pub time: f32,
    pub pose: JointPose,
}

// Per-joint channel of an imported animation.
#[derive(Debug, Clone)]
pub struct ImportedAnimationTrack {
    // Index in the engine's parents-before-children joint array.
    pub joint: usize,
    pub keys: Vec<ImportedKeyframe>,
}

// One morph-weight keyframe: per-target weights at one sample time.
#[derive(Debug, Clone)]
pub struct ImportedMorphKey {
    pub time: f32,
    pub weights: Vec<f32>,
}

// One animation extracted from a glTF file.
#[derive(Debug, Clone)]
pub struct ImportedAnimation {
    // glTF-side name; empty if the source did not name the clip.
    pub name: String,
    // Clip length in seconds: the largest sample time across all channels.
    pub duration: f32,
    // Joint-targeted channels, deduplicated and merged across translation /
    // rotation / scale targets so each joint has at most one entry.
    pub tracks: Vec<ImportedAnimationTrack>,
    // Morph-target weight keys for the skinned mesh node; empty when the
    // clip animates no morph targets.
    pub morph_track: Vec<ImportedMorphKey>,
}

// Same as [`import_glb_animations`] but takes a pre-parsed glTF document.
// The asset hot-reload pass uses this directly so a single reload pass can
// amortise the `.glb` parse across every Animation entry that references
// the same file.
pub fn import_glb_animations_from_doc(
    doc: &GltfDoc,
    source: &str,
    skin_index: u32,
) -> Result<Vec<ImportedAnimation>, String> {
    // Morph-weight channels target the mesh node itself, so the node index
    // travels with the skeleton.
    let node = skinned_node(doc, source, skin_index)?;
    let (mesh_node, skin) = (node.index(), node.skin().unwrap());
    let skeleton = import_skeleton(&skin)?;
    Ok(doc
        .doc
        .document
        .animations()
        .map(|anim| import_animation(&anim, &skeleton, doc, mesh_node))
        .collect())
}

// Resolve a `(animation_name, animation_index)` pair on a pre-parsed `.glb`,
// returning the selected clip. `animation_name` takes precedence: when
// non-empty, looks up the matching clip by name; otherwise falls back to
// `animation_index`. Used by the asset hot-reload pass to mirror the
// desugar pass's selection logic exactly so a reload picks the same clip
// the build chose at compile time.
pub fn import_glb_animation_from_doc(
    doc: &GltfDoc,
    source: &str,
    skin_index: u32,
    animation_index: u32,
    animation_name: &str,
) -> Result<ImportedAnimation, String> {
    let mut anims = import_glb_animations_from_doc(doc, source, skin_index)?;
    let idx = if !animation_name.is_empty() {
        anims
            .iter()
            .position(|a| a.name == animation_name)
            .ok_or_else(|| {
                format!(
                    "'{}': no animation named '{}' (file has {} clip{})",
                    source,
                    animation_name,
                    anims.len(),
                    if anims.len() == 1 { "" } else { "s" }
                )
            })?
    } else {
        let i = animation_index as usize;
        if i >= anims.len() {
            return Err(format!(
                "'{}': animation_index {} out of range (file has {} animation{})",
                source,
                animation_index,
                anims.len(),
                if anims.len() == 1 { "" } else { "s" }
            ));
        }
        i
    };
    Ok(anims.swap_remove(idx))
}

// The `skin_index`-th node in `doc` carrying both a mesh and a skin, counted
// in node declaration order. Shared by the skinned-mesh and animation
// importers so both resolve against the same skeleton.
//
// The selector counts skinned NODES, not entries in the glTF `skins` array:
// exporters share one skin across every mesh bound to an armature, so a
// character's body and hair typically differ only by node.
pub fn skinned_node<'a>(
    doc: &'a GltfDoc,
    source: &str,
    skin_index: u32,
) -> Result<gltf::Node<'a>, String> {
    let skinned: Vec<gltf::Node<'a>> = doc
        .doc
        .document
        .nodes()
        .filter(|n| n.mesh().is_some() && n.skin().is_some())
        .collect();
    if skinned.is_empty() {
        return Err(format!("'{}': no node with both a mesh and a skin", source));
    }
    let count = skinned.len();
    skinned.into_iter().nth(skin_index as usize).ok_or_else(|| {
        format!(
            "'{}': skin_index {} out of range (file has {} skinned mesh{})",
            source,
            skin_index,
            count,
            if count == 1 { "" } else { "es" }
        )
    })
}

// Build one `ImportedAnimation` from a glTF animation, dropping channels that
// target non-joint nodes. The skeleton's `remap` rewrites skin-joint indices
// into the engine's parents-before-children order.
fn import_animation(
    anim: &gltf::Animation<'_>,
    skeleton: &ImportedSkeleton,
    doc: &GltfDoc,
    mesh_node: usize,
) -> ImportedAnimation {
    // joint index -> JointPose per sample time. Each channel writes only its
    // own property (T/R/S) and leaves the others at the bind pose, so we seed
    // every joint pose from the bind transform before walking channels.
    let bind_pose = |j: usize| -> JointPose {
        let def = &skeleton.joints[j];
        JointPose {
            translation: def.translation,
            rotation_deg: def.rotation_deg,
            scale: def.scale,
        }
    };

    // joint index -> (time -> pose)
    let mut tracks: HashMap<usize, Vec<(f32, JointPose)>> = HashMap::new();
    let mut morph_track: Vec<ImportedMorphKey> = Vec::new();
    let mut max_time: f32 = 0.0;

    for channel in anim.channels() {
        let target_node = channel.target().node().index();

        // Morph-weight channels target the mesh node, not a joint.
        if target_node == mesh_node
            && channel.target().property() == gltf::animation::Property::MorphTargetWeights
        {
            let reader = channel.reader(|b| doc.buffer_bytes(b));
            let times: Vec<f32> = match reader.read_inputs() {
                Some(t) => t.collect(),
                None => continue,
            };
            let Some(gltf::animation::util::ReadOutputs::MorphTargetWeights(w)) =
                reader.read_outputs()
            else {
                continue;
            };
            let flat: Vec<f32> = w.into_f32().collect();
            if times.is_empty() || !flat.len().is_multiple_of(times.len()) {
                continue;
            }
            for &t in &times {
                max_time = max_time.max(t);
            }
            // CUBICSPLINE stores in-tangent / value / out-tangent per key;
            // take the value, like `sampled` does for T/R/S channels.
            let stride = flat.len() / times.len();
            let (stride, offset) = match channel.sampler().interpolation() {
                gltf::animation::Interpolation::CubicSpline if stride.is_multiple_of(3) => {
                    (stride / 3, stride / 3)
                }
                _ => (stride, 0),
            };
            morph_track = times
                .iter()
                .enumerate()
                .map(|(i, &time)| {
                    let start = i * (stride + 2 * offset) + offset;
                    ImportedMorphKey {
                        time,
                        weights: flat[start..start + stride].to_vec(),
                    }
                })
                .collect();
            continue;
        }

        let Some(&skin_joint) = skeleton.node_to_joint.get(&target_node) else {
            // Channel targets a non-joint (camera, prop, mesh node), drop.
            continue;
        };
        let joint_idx = skeleton
            .remap
            .get(skin_joint)
            .copied()
            .unwrap_or(skin_joint);
        let reader = channel.reader(|b| doc.buffer_bytes(b));
        let times: Vec<f32> = match reader.read_inputs() {
            Some(t) => t.collect(),
            None => continue,
        };
        for &t in &times {
            if t > max_time {
                max_time = t;
            }
        }

        let interpolation = channel.sampler().interpolation();
        let entry = tracks.entry(joint_idx).or_default();
        let bind = bind_pose(joint_idx);
        let upsert = |entry: &mut Vec<(f32, JointPose)>, time: f32| -> usize {
            // Same-time pose merges across T/R/S channels.
            if let Some(pos) = entry.iter().position(|(t, _)| (*t - time).abs() < 1e-6) {
                pos
            } else {
                entry.push((time, bind));
                entry.len() - 1
            }
        };

        match reader.read_outputs() {
            Some(gltf::animation::util::ReadOutputs::Translations(it)) => {
                let values: Vec<[f32; 3]> = it.collect();
                let samples = sampled(&times, &values, interpolation);
                for (time, t) in samples {
                    let i = upsert(entry, time);
                    entry[i].1.translation = t;
                }
            }
            Some(gltf::animation::util::ReadOutputs::Rotations(rot)) => {
                let values: Vec<[f32; 4]> = rot.into_f32().collect();
                let samples = sampled(&times, &values, interpolation);
                for (time, q) in samples {
                    let i = upsert(entry, time);
                    entry[i].1.rotation_deg = euler_yxz_from_quat(q);
                }
            }
            Some(gltf::animation::util::ReadOutputs::Scales(it)) => {
                let values: Vec<[f32; 3]> = it.collect();
                let samples = sampled(&times, &values, interpolation);
                for (time, s) in samples {
                    let i = upsert(entry, time);
                    entry[i].1.scale = s;
                }
            }
            // Weight channels on other nodes target meshes this import does
            // not carry; drop them like any other non-joint channel.
            Some(gltf::animation::util::ReadOutputs::MorphTargetWeights(_)) | None => continue,
        }
    }

    // Sort each joint's keys by time so the runtime's linear-scan sampling
    // walks them in order.
    let mut sorted_tracks: Vec<ImportedAnimationTrack> = tracks
        .into_iter()
        .map(|(joint, mut keys)| {
            keys.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            ImportedAnimationTrack {
                joint,
                keys: keys
                    .into_iter()
                    .map(|(time, pose)| ImportedKeyframe { time, pose })
                    .collect(),
            }
        })
        .collect();
    sorted_tracks.sort_by_key(|t| t.joint);

    morph_track.sort_by(|a, b| {
        a.time
            .partial_cmp(&b.time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ImportedAnimation {
        name: anim.name().unwrap_or("").to_string(),
        duration: max_time.max(1e-3),
        tracks: sorted_tracks,
        morph_track,
    }
}

// Pair each input time with its output value, accounting for glTF's three
// interpolation modes:
//
// - `LINEAR`: 1 value per time, pass-through.
// - `STEP`: 1 value per time, also pass-through: the runtime's linear
//   interp between two equal values is a no-op, and successive different
//   values blend over their gap. This is mildly lossy for step animations
//   (e.g. an on/off blink) but never misaligned at the keyframes.
// - `CUBICSPLINE`: 3 values per time (`in_tangent`, `value`, `out_tangent`).
//   We take only `value` and treat it as `LINEAR`; tangent-driven shapes
//   degrade but joint positions at each keyframe are correct.
//
// Mismatched lengths (a malformed file) return an empty vec rather than
// panicking.
fn sampled<T: Copy>(
    times: &[f32],
    values: &[T],
    interp: gltf::animation::Interpolation,
) -> Vec<(f32, T)> {
    use gltf::animation::Interpolation;
    match interp {
        Interpolation::Linear | Interpolation::Step => {
            if values.len() != times.len() {
                return Vec::new();
            }
            times.iter().copied().zip(values.iter().copied()).collect()
        }
        Interpolation::CubicSpline => {
            if values.len() != times.len() * 3 {
                return Vec::new();
            }
            times
                .iter()
                .copied()
                .enumerate()
                .map(|(i, t)| (t, values[i * 3 + 1]))
                .collect()
        }
    }
}

// Builders for minimal in-memory GLB containers. Shared by the glb, gltf,
// and texture test modules so each can assemble a fixture without a binary
// file checked into the repo.
#[cfg(test)]
pub(crate) mod test_fixtures {
    // Little-endian byte helpers for building GLB binary chunks.
    pub(crate) fn f32s(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    pub(crate) fn u16s(vals: &[u16]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    // Assemble a GLB container: header, JSON chunk, optional BIN chunk.
    pub(crate) fn make_glb(json: &serde_json::Value, bin: Option<&[u8]>) -> Vec<u8> {
        let mut json_bytes = serde_json::to_vec(json).expect("serialise glTF json");
        while !json_bytes.len().is_multiple_of(4) {
            json_bytes.push(b' ');
        }
        let mut bin_bytes = bin.map(|b| b.to_vec());
        if let Some(b) = bin_bytes.as_mut() {
            while !b.len().is_multiple_of(4) {
                b.push(0);
            }
        }
        let total = 12 + 8 + json_bytes.len() + bin_bytes.as_ref().map_or(0, |b| 8 + b.len());
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(b"glTF");
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&(total as u32).to_le_bytes());
        out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(b"JSON");
        out.extend_from_slice(&json_bytes);
        if let Some(b) = &bin_bytes {
            out.extend_from_slice(&(b.len() as u32).to_le_bytes());
            out.extend_from_slice(b"BIN\0");
            out.extend_from_slice(b);
        }
        out
    }

    // A single indexed triangle: positions at bin offset 0, u16 indices at 36.
    pub(crate) fn static_triangle_bin() -> Vec<u8> {
        let mut bin = f32s(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        bin.extend(u16s(&[0, 1, 2]));
        bin
    }

    pub(crate) fn static_triangle_json() -> serde_json::Value {
        serde_json::json!({
            "asset": {"version": "2.0"},
            "buffers": [{"byteLength": 42}],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": 36},
                {"buffer": 0, "byteOffset": 36, "byteLength": 6}
            ],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                 "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0]},
                {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}
            ],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
            "nodes": [{"mesh": 0}],
            "scenes": [{"nodes": [0]}],
            "scene": 0
        })
    }

    pub(crate) fn static_triangle_glb() -> Vec<u8> {
        make_glb(&static_triangle_json(), Some(&static_triangle_bin()))
    }

    // A one-triangle skinned mesh with a two-joint skeleton authored
    // child-first (skin joint 0 = node 2 "tip", skin joint 1 = node 1 "root")
    // so the importer's topological remap is exercised. Every vertex binds to
    // skin joint 0 with full weight. The bin also carries one animation
    // sampler (times 0 and 1, translations [0,0,0] and [0,2,0]).
    pub(crate) fn skinned_bin() -> Vec<u8> {
        let mut bin = f32s(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]); // 36
        bin.extend(u16s(&[0, 1, 2])); // 42
        bin.extend([0u8; 2]); // pad to 44
        bin.extend([0u8; 12]); // JOINTS_0, all skin joint 0 -> 56
        bin.extend(f32s(&[
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ])); // WEIGHTS_0 -> 104
        bin.extend(f32s(&[0.0, 1.0])); // animation input times -> 112
        bin.extend(f32s(&[0.0, 0.0, 0.0, 0.0, 2.0, 0.0])); // translations -> 136
        bin
    }

    pub(crate) fn skinned_json(
        with_joints: bool,
        with_weights: bool,
        with_anim: bool,
    ) -> serde_json::Value {
        let mut attributes = serde_json::json!({"POSITION": 0});
        if with_joints {
            attributes["JOINTS_0"] = 2.into();
        }
        if with_weights {
            attributes["WEIGHTS_0"] = 3.into();
        }
        let mut root = serde_json::json!({
            "asset": {"version": "2.0"},
            "buffers": [{"byteLength": 136}],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": 36},
                {"buffer": 0, "byteOffset": 36, "byteLength": 6},
                {"buffer": 0, "byteOffset": 44, "byteLength": 12},
                {"buffer": 0, "byteOffset": 56, "byteLength": 48},
                {"buffer": 0, "byteOffset": 104, "byteLength": 8},
                {"buffer": 0, "byteOffset": 112, "byteLength": 24}
            ],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                 "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0]},
                {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
                {"bufferView": 2, "componentType": 5121, "count": 3, "type": "VEC4"},
                {"bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC4"},
                {"bufferView": 4, "componentType": 5126, "count": 2, "type": "SCALAR",
                 "min": [0.0], "max": [1.0]},
                {"bufferView": 5, "componentType": 5126, "count": 2, "type": "VEC3"}
            ],
            "meshes": [{"primitives": [{"attributes": attributes, "indices": 1}]}],
            "skins": [{"joints": [2, 1]}],
            "nodes": [
                {"mesh": 0, "skin": 0},
                {"name": "root", "children": [2], "translation": [0.0, 1.0, 0.0]},
                {"name": "tip", "translation": [0.0, 0.5, 0.0]}
            ],
            "scenes": [{"nodes": [0, 1]}],
            "scene": 0
        });
        if with_anim {
            // Channel 0 targets the "tip" joint; channel 1 targets the mesh
            // node (not a joint) and must be dropped by the importer.
            root["animations"] = serde_json::json!([{
                "name": "wave",
                "channels": [
                    {"sampler": 0, "target": {"node": 2, "path": "translation"}},
                    {"sampler": 1, "target": {"node": 0, "path": "translation"}}
                ],
                "samplers": [
                    {"input": 4, "output": 5, "interpolation": "LINEAR"},
                    {"input": 4, "output": 5, "interpolation": "LINEAR"}
                ]
            }]);
        }
        root
    }

    pub(crate) fn skinned_glb() -> Vec<u8> {
        make_glb(&skinned_json(true, true, true), Some(&skinned_bin()))
    }

    // [`skinned_bin`] plus a second triangle's positions, offset in +X so the
    // two skinned meshes are distinguishable.
    pub(crate) fn two_skin_bin() -> Vec<u8> {
        let mut bin = skinned_bin();
        bin.extend(f32s(&[5.0, 0.0, 0.0, 6.0, 0.0, 0.0, 5.0, 1.0, 0.0])); // -> 172
        bin
    }

    // Two mesh+skin nodes sharing one skin, the shape an exporter produces for
    // a character split into several meshes bound to one armature. Each mesh
    // carries its own material.
    pub(crate) fn two_skin_json() -> serde_json::Value {
        let mut root = skinned_json(true, true, true);
        root["buffers"] = serde_json::json!([{"byteLength": 172}]);
        root["bufferViews"]
            .as_array_mut()
            .expect("bufferViews")
            .push(serde_json::json!({"buffer": 0, "byteOffset": 136, "byteLength": 36}));
        root["accessors"]
            .as_array_mut()
            .expect("accessors")
            .push(serde_json::json!({
                "bufferView": 6, "componentType": 5126, "count": 3, "type": "VEC3",
                "min": [5.0, 0.0, 0.0], "max": [6.0, 1.0, 0.0]
            }));
        root["meshes"] = serde_json::json!([
            {"primitives": [{
                "attributes": {"POSITION": 0, "JOINTS_0": 2, "WEIGHTS_0": 3},
                "indices": 1, "material": 0
            }]},
            {"primitives": [{
                "attributes": {"POSITION": 6, "JOINTS_0": 2, "WEIGHTS_0": 3},
                "indices": 1, "material": 1
            }]}
        ]);
        root["materials"] = serde_json::json!([
            {"pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 1.0}},
            {"pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 0.5}}
        ]);
        root["nodes"] = serde_json::json!([
            {"mesh": 0, "skin": 0, "name": "body"},
            {"name": "root", "children": [2], "translation": [0.0, 1.0, 0.0]},
            {"name": "tip", "translation": [0.0, 0.5, 0.0]},
            {"mesh": 1, "skin": 0, "name": "hair"}
        ]);
        root["scenes"] = serde_json::json!([{"nodes": [0, 1, 3]}]);
        root
    }

    pub(crate) fn two_skin_glb() -> Vec<u8> {
        make_glb(&two_skin_json(), Some(&two_skin_bin()))
    }

    pub(crate) fn parse(bytes: &[u8]) -> crate::gltf_source::GltfDoc {
        crate::gltf_source::GltfDoc::from_slice(bytes, None, "fixture").expect("fixture must parse")
    }
}

#[cfg(test)]
mod tests {
    use super::test_fixtures::*;
    use super::*;

    #[test]
    fn topological_order_keeps_an_already_sorted_chain() {
        let parents = [None, Some(0), Some(1)];
        let (order, remap) = topological_order(&parents);
        assert_eq!(order, vec![0, 1, 2]);
        assert_eq!(remap, vec![0, 1, 2]);
    }

    #[test]
    fn topological_order_sorts_children_after_parents() {
        // A 3-chain authored child-first: joint 0's parent is 1, 1's is 2,
        // 2 is the root. The sorted order must emit 2, then 1, then 0.
        let parents = [Some(1), Some(2), None];
        let (order, remap) = topological_order(&parents);
        assert_eq!(order, vec![2, 1, 0]);
        assert_eq!(remap, vec![2, 1, 0]);
        // Every parent now precedes its child under the remap.
        for (sj, p) in parents.iter().enumerate() {
            if let Some(&p) = p.as_ref() {
                assert!(remap[p] < remap[sj], "parent {p} not before child {sj}");
            }
        }
    }

    #[test]
    fn topological_order_handles_a_forest_with_multiple_roots() {
        // Two independent roots, each with one child.
        let parents = [None, Some(0), None, Some(2)];
        let (order, remap) = topological_order(&parents);
        assert_eq!(order.len(), 4);
        for (sj, p) in parents.iter().enumerate() {
            if let Some(&p) = p.as_ref() {
                assert!(remap[p] < remap[sj]);
            }
        }
    }

    #[test]
    fn topological_order_does_not_drop_joints_in_a_cycle() {
        // A cycle (never valid glTF) must still yield a total order so no
        // joint or its vertex bindings are silently lost.
        let parents = [Some(1), Some(0)];
        let (order, _) = topological_order(&parents);
        assert_eq!(order.len(), 2);
        let mut seen = order.clone();
        seen.sort();
        assert_eq!(seen, vec![0, 1]);
    }

    #[test]
    fn topological_order_treats_an_out_of_range_parent_as_a_root() {
        let parents = [Some(9), Some(0)];
        let (order, remap) = topological_order(&parents);
        assert_eq!(order.len(), 2);
        assert!(remap[0] < remap[1]);
    }

    use gltf::animation::Interpolation;

    #[test]
    fn sampled_linear_pairs_times_with_values() {
        let times = [0.0_f32, 0.5, 1.0];
        let values = [[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let out = sampled(&times, &values, Interpolation::Linear);
        assert_eq!(out.len(), 3);
        assert_eq!(out[1], (0.5, [0.0, 1.0, 0.0]));
    }

    #[test]
    fn sampled_step_is_pass_through() {
        let times = [0.0_f32, 1.0];
        let values = [[1.0_f32; 3], [2.0_f32; 3]];
        let out = sampled(&times, &values, Interpolation::Step);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn sampled_cubicspline_takes_middle_value_of_each_triplet() {
        // CubicSpline emits 3 values per time: in_tangent, value, out_tangent.
        let times = [0.0_f32, 0.5];
        // Times[0]'s triplet: in=11, val=12, out=13. Times[1]'s: 21, 22, 23.
        let values = [
            [11.0_f32; 3],
            [12.0; 3],
            [13.0; 3],
            [21.0; 3],
            [22.0; 3],
            [23.0; 3],
        ];
        let out = sampled(&times, &values, Interpolation::CubicSpline);
        assert_eq!(out, vec![(0.0, [12.0; 3]), (0.5, [22.0; 3])]);
    }

    #[test]
    fn sampled_with_mismatched_lengths_returns_empty() {
        let times = [0.0_f32, 1.0];
        let values: Vec<[f32; 3]> = vec![[0.0; 3]];
        assert!(sampled(&times, &values, Interpolation::Linear).is_empty());
        assert!(sampled(&times, &values, Interpolation::CubicSpline).is_empty());
    }

    // Container / static geometry

    #[test]
    fn read_primitive_geometry_reads_an_indexed_triangle() {
        let doc = parse(&static_triangle_glb());
        let (vertices, indices) = read_primitive_geometry(&doc, "t.glb", 0).expect("geometry");
        assert_eq!(vertices.len(), 3);
        assert_eq!(indices, vec![0, 1, 2]);
        assert_eq!(vertices[1].pos, [1.0, 0.0, 0.0]);
        // No TEXCOORD_0 / COLOR_0 in the fixture: fall back to defaults.
        assert_eq!(vertices[0].uv, [0.0, 0.0]);
        assert_eq!(vertices[0].color, NEUTRAL_COLOR);
    }

    #[test]
    fn read_primitive_geometry_without_indices_draws_sequentially() {
        let mut json = static_triangle_json();
        json["meshes"][0]["primitives"][0]
            .as_object_mut()
            .unwrap()
            .remove("indices");
        let doc = parse(&make_glb(&json, Some(&static_triangle_bin())));
        let (vertices, indices) = read_primitive_geometry(&doc, "t.glb", 0).expect("geometry");
        assert_eq!(vertices.len(), 3);
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn read_primitive_geometry_rejects_out_of_range_primitive_index() {
        let doc = parse(&static_triangle_glb());
        let err = read_primitive_geometry(&doc, "t.glb", 3).unwrap_err();
        assert!(err.contains("out of range"), "got: {err}");
    }

    #[test]
    fn read_primitive_geometry_rejects_non_triangle_topology() {
        let mut json = static_triangle_json();
        // Mode 0 is POINTS.
        json["meshes"][0]["primitives"][0]["mode"] = 0.into();
        let doc = parse(&make_glb(&json, Some(&static_triangle_bin())));
        let err = read_primitive_geometry(&doc, "t.glb", 0).unwrap_err();
        assert!(err.contains("only TRIANGLES is supported"), "got: {err}");
    }

    #[test]
    fn read_primitive_geometry_rejects_missing_binary_chunk() {
        // A GLB with no BIN chunk cannot resolve the POSITION accessor.
        let doc = parse(&make_glb(&static_triangle_json(), None));
        let err = read_primitive_geometry(&doc, "t.glb", 0).unwrap_err();
        assert!(err.contains("no POSITION data"), "got: {err}");
    }

    #[test]
    fn read_primitive_geometry_rejects_indices_past_the_vertex_array() {
        let mut bin = f32s(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        bin.extend(u16s(&[0, 1, 9]));
        let doc = parse(&make_glb(&static_triangle_json(), Some(&bin)));
        let err = read_primitive_geometry(&doc, "t.glb", 0).unwrap_err();
        assert!(err.contains("index 9 out of range"), "got: {err}");
    }

    #[test]
    fn import_static_glb_primitive_narrows_indices_to_u16() {
        let doc = parse(&static_triangle_glb());
        let (vertices, indices) =
            import_static_glb_primitive_from_doc(&doc, "t.glb", 0).expect("import");
        assert_eq!(vertices.len(), 3);
        assert_eq!(indices, vec![0u16, 1, 2]);
    }

    // A non-indexed primitive larger than u16 can address: JSON references one
    // POSITION accessor of 65538 vertices (all zero), forcing sequential u32
    // indices past u16::MAX.
    fn oversized_nonindexed_glb() -> Vec<u8> {
        let count = u16::MAX as usize + 3; // 65538, a multiple of 3
        let bin = vec![0u8; count * 12];
        let json = serde_json::json!({
            "asset": {"version": "2.0"},
            "buffers": [{"byteLength": bin.len()}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": bin.len()}],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": count, "type": "VEC3",
                 "min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}
            ],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
            "nodes": [{"mesh": 0}],
            "scenes": [{"nodes": [0]}],
            "scene": 0
        });
        make_glb(&json, Some(&bin))
    }

    #[test]
    fn import_static_glb_primitive_rejects_oversized_primitives() {
        let doc = parse(&oversized_nonindexed_glb());
        let err = import_static_glb_primitive_from_doc(&doc, "t.glb", 0).unwrap_err();
        assert!(err.contains("u16 index limit"), "got: {err}");
    }

    // Skinned import

    #[test]
    fn import_skinned_reorders_joints_and_remaps_vertex_bindings() {
        let doc = parse(&skinned_glb());
        let imported = import_skinned_from_doc(&doc, "s.glb", 0).expect("skinned import");

        // Skin joints are authored child-first ([tip, root]); the importer
        // must emit the root before its child.
        assert_eq!(imported.skeleton.len(), 2);
        assert_eq!(imported.skeleton[0].name, "root");
        assert_eq!(imported.skeleton[0].parent, -1);
        assert_eq!(imported.skeleton[0].translation, [0.0, 1.0, 0.0]);
        assert_eq!(imported.skeleton[1].name, "tip");
        assert_eq!(imported.skeleton[1].parent, 0);
        assert_eq!(imported.skeleton[1].translation, [0.0, 0.5, 0.0]);

        // Vertices bind to skin joint 0 (the tip), remapped to sorted index 1.
        assert_eq!(imported.vertices.len(), 3);
        assert_eq!(imported.indices, vec![0, 1, 2]);
        assert_eq!(imported.vertices[0].joints, [1, 1, 1, 1]);
        assert_eq!(imported.vertices[0].weights, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn skin_index_selects_each_skinned_node_in_declaration_order() {
        let doc = parse(&two_skin_glb());
        let body = import_skinned_from_doc(&doc, "s.glb", 0).expect("body import");
        let hair = import_skinned_from_doc(&doc, "s.glb", 1).expect("hair import");

        // Both nodes reference the same skin, so the selector counts nodes,
        // not entries in the glTF `skins` array.
        assert_eq!(body.skeleton.len(), 2);
        assert_eq!(hair.skeleton.len(), body.skeleton.len());
        assert_eq!(body.vertices[0].pos, [0.0, 0.0, 0.0]);
        assert_eq!(hair.vertices[0].pos, [5.0, 0.0, 0.0]);
    }

    #[test]
    fn skin_index_past_the_last_skinned_node_errors() {
        let doc = parse(&two_skin_glb());
        let err = import_skinned_from_doc(&doc, "s.glb", 2)
            .err()
            .expect("only two skinned nodes");
        assert!(
            err.contains("skin_index 2 out of range") && err.contains("2 skinned meshes"),
            "got: {err}"
        );
    }

    #[test]
    fn animations_resolve_against_the_selected_skin() {
        // Both skinned nodes share a skeleton, so the joint tracks agree; the
        // morph-weight channel target is what the selector really moves, and
        // resolving either index must succeed rather than silently fall back.
        let doc = parse(&two_skin_glb());
        for skin_index in 0..2 {
            let anims = import_glb_animations_from_doc(&doc, "s.glb", skin_index)
                .expect("animations for every skin");
            assert_eq!(anims.len(), 1);
            assert_eq!(anims[0].name, "wave");
        }
        let err = import_glb_animations_from_doc(&doc, "s.glb", 5).expect_err("out of range");
        assert!(err.contains("skin_index 5 out of range"), "got: {err}");
    }

    #[test]
    fn import_skinned_rejects_a_file_with_no_skinned_node() {
        let doc = parse(&static_triangle_glb());
        let err = import_skinned_from_doc(&doc, "t.glb", 0)
            .err()
            .expect("expected error");
        assert!(
            err.contains("no node with both a mesh and a skin"),
            "got: {err}"
        );
    }

    #[test]
    fn import_skinned_rejects_missing_weights() {
        let doc = parse(&make_glb(
            &skinned_json(true, false, false),
            Some(&skinned_bin()),
        ));
        let err = import_skinned_from_doc(&doc, "s.glb", 0)
            .err()
            .expect("expected error");
        assert!(err.contains("missing WEIGHTS_0"), "got: {err}");
    }

    #[test]
    fn import_skinned_rejects_a_mesh_with_only_static_primitives() {
        // The node carries a skin but no primitive has JOINTS_0.
        let doc = parse(&make_glb(
            &skinned_json(false, false, false),
            Some(&skinned_bin()),
        ));
        let err = import_skinned_from_doc(&doc, "s.glb", 0)
            .err()
            .expect("expected error");
        assert!(err.contains("no skinned primitives"), "got: {err}");
    }

    // Animation import

    #[test]
    fn import_animations_extracts_joint_tracks_and_drops_non_joint_channels() {
        let doc = parse(&skinned_glb());
        let anims = import_glb_animations_from_doc(&doc, "s.glb", 0).expect("animations");
        assert_eq!(anims.len(), 1);
        let anim = &anims[0];
        assert_eq!(anim.name, "wave");
        assert!((anim.duration - 1.0).abs() < 1e-6);

        // Only the joint-targeted channel survives; the mesh-node channel is
        // dropped. Joint 1 is the remapped "tip".
        assert_eq!(anim.tracks.len(), 1);
        let track = &anim.tracks[0];
        assert_eq!(track.joint, 1);
        assert_eq!(track.keys.len(), 2);
        assert_eq!(track.keys[0].time, 0.0);
        assert_eq!(track.keys[0].pose.translation, [0.0, 0.0, 0.0]);
        assert_eq!(track.keys[1].time, 1.0);
        assert_eq!(track.keys[1].pose.translation, [0.0, 2.0, 0.0]);
        // Untouched properties keep the bind pose.
        assert_eq!(track.keys[1].pose.scale, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn import_animation_from_doc_selects_by_name() {
        let doc = parse(&skinned_glb());
        let anim = import_glb_animation_from_doc(&doc, "s.glb", 0, 7, "wave").expect("clip");
        assert_eq!(anim.name, "wave");
    }

    #[test]
    fn import_animation_from_doc_rejects_an_unknown_name() {
        let doc = parse(&skinned_glb());
        let err = import_glb_animation_from_doc(&doc, "s.glb", 0, 0, "sprint").unwrap_err();
        assert!(err.contains("no animation named 'sprint'"), "got: {err}");
        assert!(err.contains("1 clip"), "got: {err}");
    }

    #[test]
    fn import_animation_from_doc_falls_back_to_index_when_name_is_empty() {
        let doc = parse(&skinned_glb());
        let anim = import_glb_animation_from_doc(&doc, "s.glb", 0, 0, "").expect("clip");
        assert_eq!(anim.name, "wave");
    }

    #[test]
    fn import_animation_from_doc_rejects_an_out_of_range_index() {
        let doc = parse(&skinned_glb());
        let err = import_glb_animation_from_doc(&doc, "s.glb", 0, 1, "").unwrap_err();
        assert!(err.contains("animation_index 1 out of range"), "got: {err}");
    }

    #[test]
    fn import_animation_from_doc_pluralizes_a_multi_clip_count() {
        let doc = parse(&animated_glb());
        let err = import_glb_animation_from_doc(&doc, "a.glb", 0, 99, "").unwrap_err();
        assert!(err.contains("file has 7 animations"), "got: {err}");
        let err = import_glb_animation_from_doc(&doc, "a.glb", 0, 0, "sprint").unwrap_err();
        assert!(err.contains("file has 7 clips"), "got: {err}");
    }

    #[test]
    fn importing_animations_requires_a_skinned_node_with_joints() {
        // No node carries both a mesh and a skin.
        let doc = parse(&static_triangle_glb());
        let err = import_glb_animations_from_doc(&doc, "t.glb", 0).unwrap_err();
        assert!(
            err.contains("no node with both a mesh and a skin"),
            "got: {err}"
        );
        // The same failure surfaces through the single-clip selector.
        let err = import_glb_animation_from_doc(&doc, "t.glb", 0, 0, "").unwrap_err();
        assert!(
            err.contains("no node with both a mesh and a skin"),
            "got: {err}"
        );
    }

    // Where a fixture's unbacked bufferViews start: far past any real data, so
    // the JSON validates but the reader can resolve no bytes for them.
    const UNBACKED_BASE: usize = 1 << 20;

    // Fixture assembler: appends each accessor's bytes to the binary chunk and
    // records the matching bufferView, so a fixture declares data instead of
    // hand-computed byte offsets.
    struct Fixture {
        bin: Vec<u8>,
        views: Vec<serde_json::Value>,
        accessors: Vec<serde_json::Value>,
        unbacked_len: usize,
    }

    impl Fixture {
        fn new() -> Self {
            Self {
                bin: Vec::new(),
                views: Vec::new(),
                accessors: Vec::new(),
                unbacked_len: 0,
            }
        }

        fn accessor(&mut self, bytes: &[u8], component_type: u32, count: usize, ty: &str) -> usize {
            while !self.bin.len().is_multiple_of(4) {
                self.bin.push(0);
            }
            let byte_offset = self.bin.len();
            self.bin.extend_from_slice(bytes);
            self.views.push(serde_json::json!({
                "buffer": 0,
                "byteOffset": byte_offset,
                "byteLength": bytes.len(),
            }));
            self.accessors.push(serde_json::json!({
                "bufferView": self.views.len() - 1,
                "componentType": component_type,
                "count": count,
                "type": ty,
            }));
            self.accessors.len() - 1
        }

        // A float accessor whose bufferView lies past the end of the binary
        // chunk the container actually carries: the file still validates, but
        // no bytes resolve, which is how a consumer sees an unreadable sampler.
        fn unbacked(&mut self, count: usize, ty: &str) -> usize {
            let components = if ty == "SCALAR" { 1 } else { 3 };
            let byte_length = count * components * 4;
            let byte_offset = UNBACKED_BASE + self.unbacked_len;
            self.unbacked_len += byte_length;
            self.views.push(serde_json::json!({
                "buffer": 0,
                "byteOffset": byte_offset,
                "byteLength": byte_length,
            }));
            self.accessors.push(serde_json::json!({
                "bufferView": self.views.len() - 1,
                "componentType": 5126,
                "count": count,
                "type": ty,
            }));
            self.accessors.len() - 1
        }

        fn vec2(&mut self, values: &[[f32; 2]]) -> usize {
            let flat: Vec<f32> = values.iter().flatten().copied().collect();
            self.accessor(&f32s(&flat), 5126, values.len(), "VEC2")
        }

        fn vec3(&mut self, values: &[[f32; 3]]) -> usize {
            let flat: Vec<f32> = values.iter().flatten().copied().collect();
            self.accessor(&f32s(&flat), 5126, values.len(), "VEC3")
        }

        // POSITION accessors must declare their bounds to pass glTF validation.
        fn set_bounds(&mut self, index: usize, min: [f32; 3], max: [f32; 3]) {
            self.accessors[index]["min"] = serde_json::json!(min);
            self.accessors[index]["max"] = serde_json::json!(max);
        }

        fn positions(&mut self, values: &[[f32; 3]]) -> usize {
            let index = self.vec3(values);
            let mut min = [f32::MAX; 3];
            let mut max = [f32::MIN; 3];
            for v in values {
                for i in 0..3 {
                    min[i] = min[i].min(v[i]);
                    max[i] = max[i].max(v[i]);
                }
            }
            self.set_bounds(index, min, max);
            index
        }

        fn vec4(&mut self, values: &[[f32; 4]]) -> usize {
            let flat: Vec<f32> = values.iter().flatten().copied().collect();
            self.accessor(&f32s(&flat), 5126, values.len(), "VEC4")
        }

        fn scalars(&mut self, values: &[f32]) -> usize {
            self.accessor(&f32s(values), 5126, values.len(), "SCALAR")
        }

        fn build(self, mut root: serde_json::Value) -> Vec<u8> {
            let declared = if self.unbacked_len > 0 {
                UNBACKED_BASE + self.unbacked_len
            } else {
                self.bin.len()
            };
            root["buffers"] = serde_json::json!([{"byteLength": declared}]);
            root["bufferViews"] = serde_json::Value::Array(self.views);
            root["accessors"] = serde_json::Value::Array(self.accessors);
            make_glb(&root, Some(&self.bin))
        }
    }

    // Two skinned nodes: node 1 "root" parents node 2 "tip", authored
    // child-first in the skin so the topological remap sends tip to joint 1.
    fn skinned_nodes() -> serde_json::Value {
        serde_json::json!([
            {"mesh": 0, "skin": 0},
            {"name": "root", "children": [2], "translation": [0.0, 1.0, 0.0]},
            {"name": "tip", "translation": [0.0, 0.5, 0.0]}
        ])
    }

    // A two-primitive skinned mesh with morph targets. The first primitive is
    // indexed and declares two targets (one carrying tangent deltas, which the
    // importer drops); the second is non-indexed and declares only the first
    // target, so the importer must pad the second with zero deltas.
    fn morph_glb(extras: serde_json::Value) -> Vec<u8> {
        let mut f = Fixture::new();
        let pos = f.positions(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        let idx = f.accessor(&u16s(&[0, 1, 2]), 5123, 3, "SCALAR");
        let joints = f.accessor(&[0u8; 12], 5121, 3, "VEC4");
        let weights = f.vec4(&[[1.0, 0.0, 0.0, 0.0]; 3]);
        let dp0 = f.vec3(&[[1.0, 0.0, 0.0]; 3]);
        let dn0 = f.vec3(&[[0.0, 1.0, 0.0]; 3]);
        let dt0 = f.vec3(&[[0.0, 0.0, 1.0]; 3]);
        let dp1 = f.vec3(&[[0.0, 2.0, 0.0]; 3]);
        let dp2 = f.vec3(&[[3.0, 0.0, 0.0]; 3]);
        let uv = f.vec2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        let color = f.vec3(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let attributes =
            serde_json::json!({"POSITION": pos, "JOINTS_0": joints, "WEIGHTS_0": weights});
        // The second primitive additionally carries UVs and vertex colors.
        let textured = serde_json::json!({
            "POSITION": pos,
            "JOINTS_0": joints,
            "WEIGHTS_0": weights,
            "TEXCOORD_0": uv,
            "COLOR_0": color,
        });
        f.build(serde_json::json!({
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0, 1]}],
            "nodes": skinned_nodes(),
            "skins": [{"joints": [2, 1]}],
            "meshes": [{
                "extras": extras,
                "primitives": [
                    {
                        "attributes": attributes,
                        "indices": idx,
                        "targets": [
                            {"POSITION": dp0, "NORMAL": dn0, "TANGENT": dt0},
                            {"POSITION": dp1}
                        ]
                    },
                    {
                        "attributes": textured,
                        "targets": [{"POSITION": dp2}]
                    }
                ]
            }]
        }))
    }

    #[test]
    fn import_skinned_concatenates_primitives_and_pads_absent_morph_targets() {
        let doc = parse(&morph_glb(serde_json::json!({"targetNames": ["bulge", 7]})));
        let mesh = import_skinned_from_doc(&doc, "m.glb", 0).expect("skinned import");

        // Both primitives contribute; the non-indexed one draws sequentially
        // from its own base offset.
        assert_eq!(mesh.vertices.len(), 6);
        assert_eq!(mesh.indices, vec![0, 1, 2, 3, 4, 5]);

        // A non-string target name falls back to the positional default.
        assert_eq!(mesh.morph_target_names, vec!["bulge", "target_1"]);

        // The first primitive declares neither UVs nor colors and takes the
        // defaults; the second carries its own.
        assert_eq!(mesh.vertices[0].uv, [0.0, 0.0]);
        assert_eq!(mesh.vertices[0].color, [1.0, 1.0, 1.0]);
        assert_eq!(mesh.vertices[4].uv, [1.0, 0.0]);
        assert_eq!(mesh.vertices[4].color, [0.0, 1.0, 0.0]);

        // Deltas are dense and target-major: 2 targets x 6 vertices.
        assert_eq!(mesh.morph_deltas.len(), 12);
        assert_eq!(mesh.morph_deltas[0].position, [1.0, 0.0, 0.0]);
        assert_eq!(mesh.morph_deltas[0].normal, [0.0, 1.0, 0.0]);
        // Target 0 continues into the second primitive's own deltas.
        assert_eq!(mesh.morph_deltas[3].position, [3.0, 0.0, 0.0]);
        // Target 1 exists only on the first primitive; the rest is zero-padded.
        assert_eq!(mesh.morph_deltas[6].position, [0.0, 2.0, 0.0]);
        assert_eq!(mesh.morph_deltas[9].position, [0.0, 0.0, 0.0]);
        assert_eq!(mesh.morph_deltas[9].normal, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn morph_target_names_fall_back_when_extras_are_unusable() {
        for extras in [
            serde_json::json!({"targetNames": "not an array"}),
            serde_json::json!({"unrelated": 1}),
        ] {
            let doc = parse(&morph_glb(extras.clone()));
            let mesh = import_skinned_from_doc(&doc, "m.glb", 0).expect("skinned import");
            assert_eq!(
                mesh.morph_target_names,
                vec!["target_0", "target_1"],
                "extras {extras}"
            );
        }
    }

    #[test]
    fn read_primitive_geometry_reads_texcoords_and_vertex_colors() {
        let mut f = Fixture::new();
        let pos = f.positions(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        let uv = f.vec2(&[[0.0, 0.0], [1.0, 0.0], [0.25, 0.5]]);
        let color = f.vec3(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let glb = f.build(serde_json::json!({
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{
                "attributes": {"POSITION": pos, "TEXCOORD_0": uv, "COLOR_0": color}
            }]}]
        }));
        let doc = parse(&glb);
        let (vertices, indices) = read_primitive_geometry(&doc, "t.glb", 0).expect("geometry");

        assert_eq!(indices, vec![0, 1, 2]);
        assert_eq!(vertices[2].uv, [0.25, 0.5]);
        // Authored colors replace the neutral fallback.
        assert_eq!(vertices[0].color, [1.0, 0.0, 0.0]);
        assert_eq!(vertices[2].color, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn import_skinned_rejects_a_primitive_whose_positions_have_no_data() {
        let mut f = Fixture::new();
        let joints = f.accessor(&[0u8; 12], 5121, 3, "VEC4");
        let weights = f.vec4(&[[1.0, 0.0, 0.0, 0.0]; 3]);
        // Bindings resolve but the POSITION accessor points past the binary
        // chunk, so the file parses and only the position read fails.
        let pos = f.unbacked(3, "VEC3");
        f.set_bounds(pos, [0.0; 3], [1.0, 1.0, 0.0]);
        let glb = f.build(serde_json::json!({
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0, 1]}],
            "nodes": skinned_nodes(),
            "skins": [{"joints": [2, 1]}],
            "meshes": [{"primitives": [
                {"attributes": {"POSITION": pos, "JOINTS_0": joints, "WEIGHTS_0": weights}}
            ]}]
        }));
        let doc = parse(&glb);
        let err = import_skinned_from_doc(&doc, "m.glb", 0)
            .err()
            .expect("expected error");
        assert!(err.contains("no POSITION data"), "got: {err}");
    }

    #[test]
    fn import_skinned_rejects_indices_past_the_u16_limit() {
        let mut f = Fixture::new();
        let pos = f.positions(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        let idx = f.accessor(
            &[0u32, 1, 70_000]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
            5125,
            3,
            "SCALAR",
        );
        let joints = f.accessor(&[0u8; 12], 5121, 3, "VEC4");
        let weights = f.vec4(&[[1.0, 0.0, 0.0, 0.0]; 3]);
        let glb = f.build(serde_json::json!({
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0, 1]}],
            "nodes": skinned_nodes(),
            "skins": [{"joints": [2, 1]}],
            "meshes": [{"primitives": [{
                "attributes": {"POSITION": pos, "JOINTS_0": joints, "WEIGHTS_0": weights},
                "indices": idx
            }]}]
        }));
        let doc = parse(&glb);
        let err = import_skinned_from_doc(&doc, "m.glb", 0)
            .err()
            .expect("expected error");
        assert_eq!(
            err,
            "imported skinned mesh exceeds the 65535-vertex u16 index limit"
        );
    }

    // A skinned node whose skin declares no joints at all.
    fn jointless_glb() -> Vec<u8> {
        let mut f = Fixture::new();
        let pos = f.positions(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        let joints = f.accessor(&[0u8; 12], 5121, 3, "VEC4");
        let weights = f.vec4(&[[1.0, 0.0, 0.0, 0.0]; 3]);
        f.build(serde_json::json!({
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "skin": 0}],
            "skins": [{"joints": []}],
            "meshes": [{"primitives": [{
                "attributes": {"POSITION": pos, "JOINTS_0": joints, "WEIGHTS_0": weights}
            }]}]
        }))
    }

    #[test]
    fn import_skeleton_rejects_a_skin_with_no_joints() {
        let doc = parse(&jointless_glb());
        let err = import_skinned_from_doc(&doc, "m.glb", 0)
            .err()
            .expect("expected error");
        assert_eq!(err, "glTF skin has no joints");
        // The animation importer builds the same skeleton and fails alike.
        let err = import_glb_animations_from_doc(&doc, "m.glb", 0).unwrap_err();
        assert_eq!(err, "glTF skin has no joints");
    }

    // A skinned node whose clips cover every animation channel path: T/R/S on
    // one joint sampled at the same times, morph weights on the mesh node in
    // both interpolation modes, and the channels the importer must drop.
    fn animated_glb() -> Vec<u8> {
        let mut f = Fixture::new();
        let pos = f.positions(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        let joints = f.accessor(&[0u8; 12], 5121, 3, "VEC4");
        let weights = f.vec4(&[[1.0, 0.0, 0.0, 0.0]; 3]);
        let times = f.scalars(&[0.0, 1.0]);
        let translations = f.vec3(&[[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]);
        let scales = f.vec3(&[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]);
        // Identity, then a quarter turn about Z.
        let half = std::f32::consts::FRAC_1_SQRT_2;
        let rotations = f.vec4(&[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, half, half]]);
        // The same two rotations wrapped in in/out tangents, plus a channel
        // whose output count does not match its sample times.
        let tangent = [1.0, 0.0, 0.0, 0.0];
        let rotations_cubic = f.vec4(&[
            tangent,
            [0.0, 0.0, 0.0, 1.0],
            tangent,
            tangent,
            [0.0, 0.0, half, half],
            tangent,
        ]);
        let rotations_ragged = f.vec4(&[[0.0, 0.0, 0.0, 1.0]; 3]);
        let w_linear = f.scalars(&[0.0, 0.25, 1.0, 0.5]);
        let w_cubic = f.scalars(&[9.0, 9.0, 0.1, 0.2, 9.0, 9.0, 9.0, 9.0, 0.3, 0.4, 9.0, 9.0]);
        let w_ragged = f.scalars(&[0.0, 0.5, 1.0]);
        let unbacked_times = f.unbacked(2, "SCALAR");
        let unbacked_weights = f.unbacked(4, "SCALAR");

        f.build(serde_json::json!({
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0, 1]}],
            "nodes": skinned_nodes(),
            "skins": [{"joints": [2, 1]}],
            "meshes": [{"primitives": [{
                "attributes": {"POSITION": pos, "JOINTS_0": joints, "WEIGHTS_0": weights}
            }]}],
            "animations": [
                {
                    "name": "pose",
                    "samplers": [
                        {"input": times, "output": translations, "interpolation": "LINEAR"},
                        {"input": times, "output": rotations, "interpolation": "LINEAR"},
                        {"input": times, "output": scales, "interpolation": "LINEAR"},
                        {"input": times, "output": times, "interpolation": "LINEAR"},
                        {"input": unbacked_times, "output": translations}
                    ],
                    "channels": [
                        {"sampler": 0, "target": {"node": 2, "path": "translation"}},
                        {"sampler": 1, "target": {"node": 2, "path": "rotation"}},
                        {"sampler": 2, "target": {"node": 2, "path": "scale"}},
                        {"sampler": 3, "target": {"node": 2, "path": "weights"}},
                        {"sampler": 4, "target": {"node": 2, "path": "translation"}}
                    ]
                },
                {
                    "name": "pose_cubic",
                    "samplers": [
                        {"input": times, "output": rotations_cubic, "interpolation": "CUBICSPLINE"}
                    ],
                    "channels": [{"sampler": 0, "target": {"node": 2, "path": "rotation"}}]
                },
                {
                    "name": "pose_ragged",
                    "samplers": [
                        {"input": times, "output": rotations_ragged, "interpolation": "LINEAR"}
                    ],
                    "channels": [{"sampler": 0, "target": {"node": 2, "path": "rotation"}}]
                },
                {
                    "name": "morph_linear",
                    "samplers": [{"input": times, "output": w_linear, "interpolation": "LINEAR"}],
                    "channels": [{"sampler": 0, "target": {"node": 0, "path": "weights"}}]
                },
                {
                    "name": "morph_cubic",
                    "samplers": [
                        {"input": times, "output": w_cubic, "interpolation": "CUBICSPLINE"}
                    ],
                    "channels": [{"sampler": 0, "target": {"node": 0, "path": "weights"}}]
                },
                {
                    "name": "morph_ragged",
                    "samplers": [{"input": times, "output": w_ragged, "interpolation": "LINEAR"}],
                    "channels": [{"sampler": 0, "target": {"node": 0, "path": "weights"}}]
                },
                {
                    "name": "morph_unreadable",
                    "samplers": [
                        {"input": unbacked_times, "output": w_linear},
                        {"input": times, "output": unbacked_weights}
                    ],
                    "channels": [
                        {"sampler": 0, "target": {"node": 0, "path": "weights"}},
                        {"sampler": 1, "target": {"node": 0, "path": "weights"}}
                    ]
                }
            ]
        }))
    }

    fn clip<'a>(anims: &'a [ImportedAnimation], name: &str) -> &'a ImportedAnimation {
        anims
            .iter()
            .find(|a| a.name == name)
            .unwrap_or_else(|| panic!("no clip named '{name}'"))
    }

    #[test]
    fn import_animation_merges_translation_rotation_and_scale_at_shared_times() {
        let doc = parse(&animated_glb());
        let anims = import_glb_animations_from_doc(&doc, "a.glb", 0).expect("animations");
        let pose = clip(&anims, "pose");

        // The three T/R/S channels target one joint at identical sample times,
        // so they merge into a single track of two keys.
        assert_eq!(pose.tracks.len(), 1);
        let track = &pose.tracks[0];
        assert_eq!(track.joint, 1);
        assert_eq!(track.keys.len(), 2);

        assert_eq!(track.keys[0].time, 0.0);
        assert_eq!(track.keys[0].pose.translation, [0.0, 0.0, 0.0]);
        assert_eq!(track.keys[0].pose.scale, [1.0, 1.0, 1.0]);
        assert_eq!(track.keys[0].pose.rotation_deg, [0.0, 0.0, 0.0]);

        assert_eq!(track.keys[1].time, 1.0);
        assert_eq!(track.keys[1].pose.translation, [0.0, 2.0, 0.0]);
        assert_eq!(track.keys[1].pose.scale, [2.0, 2.0, 2.0]);
        // A quarter turn about Z comes back as roll in the YXZ euler triple.
        let roll = track.keys[1].pose.rotation_deg[2];
        assert!((roll - 90.0).abs() < 1e-3, "roll was {roll}");

        // A weights channel aimed at a joint, and a channel whose sampler input
        // has no data, contribute nothing.
        assert!(pose.morph_track.is_empty());
        assert!((pose.duration - 1.0).abs() < 1e-6);
    }

    #[test]
    fn import_animation_takes_the_value_of_each_cubicspline_rotation_triplet() {
        let doc = parse(&animated_glb());
        let anims = import_glb_animations_from_doc(&doc, "a.glb", 0).expect("animations");
        let track = &clip(&anims, "pose_cubic").tracks[0];

        // The in / out tangent quaternions flanking each value are discarded.
        assert_eq!(track.keys.len(), 2);
        assert_eq!(track.keys[0].pose.rotation_deg, [0.0, 0.0, 0.0]);
        let roll = track.keys[1].pose.rotation_deg[2];
        assert!((roll - 90.0).abs() < 1e-3, "roll was {roll}");
    }

    #[test]
    fn import_animation_drops_samples_when_the_output_count_disagrees() {
        let doc = parse(&animated_glb());
        let anims = import_glb_animations_from_doc(&doc, "a.glb", 0).expect("animations");
        let track = &clip(&anims, "pose_ragged").tracks[0];
        assert_eq!(track.joint, 1);
        assert!(track.keys.is_empty());
    }

    #[test]
    fn import_animation_reads_linear_morph_weight_keys() {
        let doc = parse(&animated_glb());
        let anims = import_glb_animations_from_doc(&doc, "a.glb", 0).expect("animations");
        let morph = clip(&anims, "morph_linear");

        assert!(morph.tracks.is_empty());
        assert_eq!(morph.morph_track.len(), 2);
        assert_eq!(morph.morph_track[0].time, 0.0);
        assert_eq!(morph.morph_track[0].weights, vec![0.0, 0.25]);
        assert_eq!(morph.morph_track[1].time, 1.0);
        assert_eq!(morph.morph_track[1].weights, vec![1.0, 0.5]);
    }

    #[test]
    fn import_animation_takes_the_value_of_each_cubicspline_morph_triplet() {
        let doc = parse(&animated_glb());
        let anims = import_glb_animations_from_doc(&doc, "a.glb", 0).expect("animations");
        let morph = clip(&anims, "morph_cubic");

        // Each key stores in-tangent / value / out-tangent per target; only the
        // middle triple survives.
        assert_eq!(morph.morph_track.len(), 2);
        assert_eq!(morph.morph_track[0].weights, vec![0.1, 0.2]);
        assert_eq!(morph.morph_track[1].weights, vec![0.3, 0.4]);
    }

    #[test]
    fn import_animation_drops_unusable_morph_channels() {
        let doc = parse(&animated_glb());
        let anims = import_glb_animations_from_doc(&doc, "a.glb", 0).expect("animations");
        // A weight count that is not a whole multiple of the sample times, and
        // samplers whose input or output carries no data, are all skipped
        // rather than producing partial keys.
        assert!(clip(&anims, "morph_ragged").morph_track.is_empty());
        assert!(clip(&anims, "morph_unreadable").morph_track.is_empty());
    }

    // Morph fixtures: the local Blender export carries two shape keys with a
    // keyed weights animation. Ignored by default (private/assets is local).
    // Run with: cargo test -p concinnity-cook morph_fixture -- --ignored
    #[test]
    #[ignore = "needs the local Blender rig_oracle fixture under private/assets"]
    fn morph_fixture_imports_deltas_names_and_weight_keys() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../private/assets/models/rig_oracle/rig_morph.glb"
        );
        let doc = parse_glb(path).expect("parse");
        let mesh = import_skinned_from_doc(&doc, path, 0).expect("skinned import");
        assert_eq!(mesh.morph_target_names, vec!["bulge", "shift"]);
        assert_eq!(mesh.morph_deltas.len(), 2 * mesh.vertices.len());
        assert!(
            mesh.morph_deltas
                .iter()
                .any(|d| d.position.iter().any(|c| c.abs() > 0.1)),
            "bulge target must carry real position deltas"
        );

        let anims = import_glb_animations_from_doc(&doc, path, 0).expect("animations");
        let weighted = anims
            .iter()
            .find(|a| !a.morph_track.is_empty())
            .expect("one clip carries the weights channel");
        assert!(weighted.morph_track.len() >= 2);
        assert!(weighted.morph_track.iter().all(|k| k.weights.len() == 2));
        let max_bulge = weighted
            .morph_track
            .iter()
            .map(|k| k.weights[0])
            .fold(0.0f32, f32::max);
        assert!((max_bulge - 1.0).abs() < 1e-3, "bulge peaks at 1.0");
    }

    // parse_glb / resolve_source

    #[test]
    fn parse_glb_reads_a_file_from_disk() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("tri.glb");
        std::fs::write(&path, static_triangle_glb()).expect("write glb");
        let doc = parse_glb(path.to_str().unwrap()).expect("parse");
        assert_eq!(doc.doc.document.meshes().count(), 1);
    }

    #[test]
    fn parse_glb_reports_a_missing_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("missing.glb");
        let err = parse_glb(path.to_str().unwrap()).unwrap_err();
        assert!(err.contains("failed to read"), "got: {err}");
    }

    #[test]
    fn parse_glb_reports_invalid_content() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("junk.glb");
        std::fs::write(&path, b"not a glb at all").expect("write junk");
        let err = parse_glb(path.to_str().unwrap()).unwrap_err();
        assert!(err.contains("not a valid glTF/GLB file"), "got: {err}");
    }

    #[test]
    fn resolve_source_keeps_paths_with_a_directory_component() {
        assert_eq!(resolve_source("sub/f.glb"), "sub/f.glb");
        assert_eq!(resolve_source("./f.glb"), "./f.glb");
        assert_eq!(resolve_source("/abs/f.glb"), "/abs/f.glb");
    }

    #[test]
    fn resolve_source_anchors_a_bare_filename_under_the_assets_dir() {
        // Nothing by this name exists to be found, so the fallback is the
        // assets directory joined with the filename.
        let resolved = resolve_source("cn_test_no_such_model.glb");
        let expected = concinnity_core::paths::assets_dir().join("cn_test_no_such_model.glb");
        assert_eq!(resolved, expected.to_string_lossy());
    }

    // split_into_u16_chunks

    fn vert(i: u32) -> VertexData {
        VertexData {
            pos: [i as f32, 0.0, 0.0],
            color: [0.0; 3],
            uv: [0.0; 2],
        }
    }

    #[test]
    fn split_small_mesh_yields_one_chunk_with_first_use_order() {
        let vertices: Vec<VertexData> = (0..4).map(vert).collect();
        // Two triangles sharing vertices 0 and 2, authored out of order.
        let indices = [2u32, 1, 0, 0, 2, 3];
        let chunks = split_into_u16_chunks(&vertices, &indices);
        assert_eq!(chunks.len(), 1);
        let (cv, ci) = &chunks[0];
        assert_eq!(cv.len(), 4);
        // Local indices follow first use: 2 -> 0, 1 -> 1, 0 -> 2, 3 -> 3.
        assert_eq!(ci, &vec![0u16, 1, 2, 2, 0, 3]);
        // Remapped geometry references the original positions.
        assert_eq!(cv[ci[0] as usize].pos, vertices[2].pos);
        assert_eq!(cv[ci[5] as usize].pos, vertices[3].pos);
    }

    #[test]
    fn split_with_no_triangles_yields_no_chunks() {
        let vertices: Vec<VertexData> = (0..3).map(vert).collect();
        assert!(split_into_u16_chunks(&vertices, &[]).is_empty());
        // A trailing partial triangle is dropped by chunks_exact.
        assert!(split_into_u16_chunks(&vertices, &[0, 1]).is_empty());
    }

    #[test]
    fn split_flushes_a_chunk_when_the_u16_limit_would_overflow() {
        // 21846 disjoint triangles = 65538 unique vertices, just past u16.
        let count = u16::MAX as u32 + 3;
        let vertices: Vec<VertexData> = (0..count).map(vert).collect();
        let indices: Vec<u32> = (0..count).collect();
        let chunks = split_into_u16_chunks(&vertices, &indices);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].0.len(), 65535);
        assert_eq!(chunks[1].0.len(), 3);
        // Every chunk index stays within its own vertex buffer and the
        // triangle count is preserved across the split.
        let mut triangles = 0;
        for (cv, ci) in &chunks {
            assert!(ci.iter().all(|&i| (i as usize) < cv.len()));
            triangles += ci.len() / 3;
        }
        assert_eq!(triangles, count as usize / 3);
        // The second chunk's remapped triangle is the original last triangle.
        let (cv, ci) = &chunks[1];
        assert_eq!(cv[ci[0] as usize].pos, vert(count - 3).pos);
        assert_eq!(cv[ci[2] as usize].pos, vert(count - 1).pos);
    }
}
