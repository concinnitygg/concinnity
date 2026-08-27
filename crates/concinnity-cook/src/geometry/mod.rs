// src/geometry/mod.rs
//
// Build-time mesh compilation: the procedural generators (room, extrude,
// primitives, terrain, skybox, heightfield) and the `compile_*_payload`
// functions that pack their output into the binary blob. These run only at cook
// time; the runtime plays the compiled payload. The shared low-level mesh math
// (per-vertex tangents, face normals, the `Vert` tuple, the voxel mesher) lives
// in `concinnity_core::geometry` because the runtime chunk-streaming path uses
// it too; this module calls back into it.
//
// Adding a new generator
// 1. Add a branch to the match in compile_mesh_payload().
// 2. Add a pub(super) build_* function in the appropriate submodule.
// 3. No other files need to change.

mod extrude;
mod heightfield;
mod primitives;
mod room;
mod skybox;
mod terrain;

use concinnity_core::geometry::{PaletteSlot, Vert, build_voxel_mesh, compute_tangents};
use concinnity_core::math::vec3::{vec3_add, vec3_face_normal, vec3_normalise};

// Re-exported so cook code that also needs the runtime-side joint conversion
// (mesh_reimport) can reach it through `crate::geometry`.
pub use concinnity_core::geometry::payload_joints_to_defs;

// Tangent-bearing and raw inline vertex forms the build helpers pack into the
// payload:
//   VertT   = position, normal, tangent, color, uv
//   RawVert = position, color, uv (inline input, before normals are derived)
type VertT = ([f32; 3], [f32; 3], [f32; 3], [f32; 3], [f32; 2]);
type RawVert = ([f32; 3], [f32; 3], [f32; 2]);

// LOD alternates: (switch_distance, index buffer) pairs.
type LodAlternates = Vec<(f32, Vec<u16>)>;

// Convert an args-form `SkeletonJoint` into the payload joint form.
fn joint_def_to_payload(
    j: &crate::components::SkeletonJoint,
) -> crate::gfx::mesh_payload::PayloadJoint {
    crate::gfx::mesh_payload::PayloadJoint {
        name: j.name.clone(),
        parent: j.parent,
        translation: j.translation,
        rotation_deg: j.rotation_deg,
        scale: j.scale,
    }
}

/// Compile a Mesh component's JSON args into a packed binary payload.
pub fn compile_mesh_payload(args: &serde_json::Value) -> Result<Vec<u8>, String> {
    let generator = args.get("generator").and_then(|v| v.as_str()).unwrap_or("");

    let (vertices, indices): (Vec<Vert>, Vec<u16>) = match generator {
        "room" => room::build_room(args)?,
        "box" => primitives::build_box(args)?,
        "cylinder" => primitives::build_cylinder(args)?,
        "plane" => primitives::build_plane(args)?,
        "sphere" => primitives::build_sphere(args)?,
        "terrain" => terrain::build_terrain(args)?,
        "heightfield" => {
            // The heightfield generator needs the source image decoded, which
            // the build crate does before calling compile_heightfield_payload.
            return Err(
                "the `heightfield` generator decodes a source image; compile it \
                 through the build crate's heightfield path"
                    .to_string(),
            );
        }
        // The `water_grid` generator stays in the runtime crate (the GPU
        // backends call it directly), so reach it there for the compile branch.
        "water_grid" => {
            let half_width = args
                .get("half_width")
                .and_then(|v| v.as_f64())
                .unwrap_or(10.0) as f32;
            let half_depth = args
                .get("half_depth")
                .and_then(|v| v.as_f64())
                .unwrap_or(10.0) as f32;
            let subdivisions = args
                .get("subdivisions")
                .and_then(|v| v.as_u64())
                .unwrap_or(64) as u32;
            concinnity_core::geometry::water_grid::build_water_grid(
                half_width,
                half_depth,
                subdivisions,
            )?
        }
        "skybox" => skybox::build_skybox(args)?,
        "extrude" => extrude::build_extrude(args)?,
        // empty generator string = inline vertex data supplied directly in the blob
        "" => build_inline(args)?,
        other => return Err(format!("unknown mesh generator '{other}'")),
    };

    finish_mesh_payload(vertices, indices, args)
}

// Compile a heightfield mesh from a pre-decoded source image into a packed
// payload with a baked collider height grid. The build crate decodes the source
// image (the runtime links no image decoders) and passes the RGBA pixels in. The
// collider grid is the mesh's own per-vertex Y in row-major order, so the
// physics terrain collider tracks the rendered surface vertex-for-vertex with
// no runtime image decode.
pub(crate) fn compile_heightfield_payload(
    args: &serde_json::Value,
    img_w: u32,
    img_h: u32,
    rgba: Vec<u8>,
) -> Result<Vec<u8>, String> {
    let (vertices, indices) =
        heightfield::build_heightfield_from_pixels(args, img_w, img_h, &rgba)?;
    // Read the LOD0 mesh's per-vertex Y for the collider grid before the payload
    // tail consumes the vertices.
    let (n, heights) = heightfield_collider_grid(&vertices)?;
    let mut payload = finish_mesh_payload(vertices, indices, args)?;
    payload.extend_from_slice(&crate::gfx::mesh_payload::serialise_heightfield_trailer(
        n, n, &heights,
    ));
    Ok(payload)
}

// Shared payload tail for every generator: derive tangents from the UV
// gradient, build the optional LOD alternate index lists, and serialise the
// packed mesh payload (with the LOD trailer when alternates are present).
//
// `lod_levels` (read from `args`) is the total count including LOD0;
// `build_lod_alternates` generates `levels - 1` decimated index lists via
// vertex clustering on the LOD0 vertex set, each paired with a switch distance.
// `lod_distances` either gives explicit thresholds or, when empty, the build
// derives a default doubling sequence from the bounding-sphere radius.
// `lod_levels = 1` (the default) produces no trailer and emits the legacy
// single-LOD payload byte-for-byte.
fn finish_mesh_payload(
    vertices: Vec<Vert>,
    indices: Vec<u16>,
    args: &serde_json::Value,
) -> Result<Vec<u8>, String> {
    let tangents = compute_tangents(&vertices, &indices);
    let verts5: Vec<VertT> = vertices
        .into_iter()
        .zip(tangents)
        .map(|((pos, normal, color, uv), tangent)| (pos, normal, tangent, color, uv))
        .collect();
    let alternates = build_lod_alternates(args, &verts5, &indices)?;
    Ok(crate::gfx::mesh_payload::serialise_with_lods(
        &verts5,
        &indices,
        &alternates,
    ))
}

// Extract the square collider height grid from a heightfield mesh's vertices.
// The generator emits exactly `(subdivisions + 1)²` vertices in row-major
// order with each vertex's Y already mapped through the elevation range, so the
// grid is just those Y values; `n` is the side length.
fn heightfield_collider_grid(verts: &[Vert]) -> Result<(usize, Vec<f32>), String> {
    let n = verts.len().isqrt();
    if n * n != verts.len() {
        return Err(format!(
            "heightfield mesh has {} vertices, not a square grid",
            verts.len()
        ));
    }
    let heights = verts.iter().map(|v| v.0[1]).collect();
    Ok((n, heights))
}

// Build the per-LOD `(switch_distance, indices)` list for a mesh, honouring
// the `lod_levels` and `lod_distances` args. Returns an empty `Vec` when
// `lod_levels <= 1` so the payload writer emits a legacy single-LOD blob.
fn build_lod_alternates(
    args: &serde_json::Value,
    verts: &[VertT],
    indices: &[u16],
) -> Result<LodAlternates, String> {
    let lod_levels = args
        .get("lod_levels")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
        .unwrap_or(1)
        .clamp(1, 8);
    if lod_levels <= 1 {
        return Ok(Vec::new());
    }
    let alt_count = (lod_levels - 1) as usize;

    // Read explicit thresholds or derive them from the bounding-sphere
    // radius. The default cascade doubles per level, which keeps successive
    // LODs visibly apart when seen in the showcase's free-fly camera.
    let explicit_distances: Vec<f32> = args
        .get("lod_distances")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_f64().map(|n| n as f32))
                .collect()
        })
        .unwrap_or_default();
    if !explicit_distances.is_empty() && explicit_distances.len() != alt_count {
        return Err(format!(
            "lod_distances has {} entries but lod_levels = {} expects {}",
            explicit_distances.len(),
            lod_levels,
            alt_count,
        ));
    }
    let positions: Vec<[f32; 3]> = verts.iter().map(|(p, _, _, _, _)| *p).collect();
    let radius = bounding_sphere_radius(&positions);
    let lod0_tri_count = indices.len() / 3;
    let mut out = Vec::with_capacity(alt_count);
    for level in 1..lod_levels {
        let target = crate::gfx::lod::target_tri_count_for_level(lod0_tri_count, level);
        let idx = crate::gfx::lod::decimate_by_qem(&positions, indices, target);
        // Drop LOD if the decimator collapsed everything (degenerate input);
        // remaining levels would also be empty so we stop early.
        if idx.is_empty() {
            break;
        }
        let distance = if explicit_distances.is_empty() {
            crate::gfx::lod::default_distance_for_level(radius, level)
        } else {
            explicit_distances[(level - 1) as usize]
        };
        out.push((distance, idx));
    }
    Ok(out)
}

// Bounding-sphere radius around the mesh AABB centre. Cheap upper bound on
// the per-vertex distance to centre, used to seed default LOD thresholds.
fn bounding_sphere_radius(positions: &[[f32; 3]]) -> f32 {
    if positions.is_empty() {
        return 1.0;
    }
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for p in positions {
        for k in 0..3 {
            mn[k] = mn[k].min(p[k]);
            mx[k] = mx[k].max(p[k]);
        }
    }
    let dx = mx[0] - mn[0];
    let dy = mx[1] - mn[1];
    let dz = mx[2] - mn[2];
    (0.5 * (dx * dx + dy * dy + dz * dz).sqrt()).max(0.25)
}

// Compile typed vertex and index data into a packed binary mesh payload.
//
// Normals are computed from triangle geometry; tangents from UV gradients.
// Shared by the inline Mesh path and file-backed mesh formats (e.g. OBJ).
pub(crate) fn compile_mesh_from_vertex_data(
    vertex_data: &[crate::components::VertexData],
    indices: &[u16],
) -> Vec<u8> {
    let mut normals: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]; vertex_data.len()];
    let tris = indices.len() / 3;
    for t in 0..tris {
        let ia = indices[t * 3] as usize;
        let ib = indices[t * 3 + 1] as usize;
        let ic = indices[t * 3 + 2] as usize;
        if ia >= vertex_data.len() || ib >= vertex_data.len() || ic >= vertex_data.len() {
            continue;
        }
        let n = vec3_face_normal(
            vertex_data[ia].pos,
            vertex_data[ib].pos,
            vertex_data[ic].pos,
        );
        vec3_add(&mut normals[ia], n);
        vec3_add(&mut normals[ib], n);
        vec3_add(&mut normals[ic], n);
    }
    let vertices: Vec<Vert> = vertex_data
        .iter()
        .enumerate()
        .map(|(i, v)| (v.pos, vec3_normalise(normals[i]), v.color, v.uv))
        .collect();
    let tangents = compute_tangents(&vertices, indices);
    let verts5: Vec<_> = vertices
        .into_iter()
        .zip(tangents)
        .map(|((pos, normal, color, uv), tangent)| (pos, normal, tangent, color, uv))
        .collect();
    crate::gfx::mesh_payload::serialise(&verts5, indices)
}

// The level-of-detail request of a skinned mesh: `levels` includes LOD0 (so
// `1` emits no alternates) and `distances` may be empty (a doubling cascade
// derived from the bounding sphere) or must hold one threshold per
// alternate.
pub(crate) struct SkinnedLods<'a> {
    pub levels: u32,
    pub distances: &'a [f32],
}

// Compile a skinned mesh payload with optional LOD alternates. Decimation is
// QEM half-edge collapse against the LOD0 vertex set, mirroring the static
// [`build_lod_alternates`] path.
pub(crate) fn compile_skinned_mesh_payload_with_lods(
    vertex_data: &[crate::components::SkinnedVertexData],
    indices: &[u16],
    skeleton: &[crate::components::SkeletonJoint],
    morph_target_names: &[String],
    morph_deltas: &[crate::components::MorphDelta],
    lods: &SkinnedLods,
) -> Result<Vec<u8>, String> {
    let lod_levels = lods.levels.clamp(1, 8);
    let lod_distances = lods.distances;
    if vertex_data.is_empty() {
        return Err("SkinnedMesh requires at least one vertex".to_string());
    }
    let tris = indices.len() / 3;
    for t in 0..tris {
        for k in 0..3 {
            if indices[t * 3 + k] as usize >= vertex_data.len() {
                return Err(format!("SkinnedMesh index out of range in triangle {t}"));
            }
        }
    }

    let mut normals: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]; vertex_data.len()];
    for t in 0..tris {
        let ia = indices[t * 3] as usize;
        let ib = indices[t * 3 + 1] as usize;
        let ic = indices[t * 3 + 2] as usize;
        let n = vec3_face_normal(
            vertex_data[ia].pos,
            vertex_data[ib].pos,
            vertex_data[ic].pos,
        );
        vec3_add(&mut normals[ia], n);
        vec3_add(&mut normals[ib], n);
        vec3_add(&mut normals[ic], n);
    }
    let pnt: Vec<Vert> = vertex_data
        .iter()
        .enumerate()
        .map(|(i, v)| (v.pos, vec3_normalise(normals[i]), v.color, v.uv))
        .collect();
    let tangents = compute_tangents(&pnt, indices);

    let skinned: Vec<crate::gfx::mesh_payload::SkinnedVertex> = vertex_data
        .iter()
        .zip(pnt.iter())
        .zip(tangents)
        .map(|((v, (pos, normal, color, uv)), tangent)| {
            let sum: f32 = v.weights.iter().sum();
            let weights = if sum > 1e-6 {
                [
                    v.weights[0] / sum,
                    v.weights[1] / sum,
                    v.weights[2] / sum,
                    v.weights[3] / sum,
                ]
            } else {
                // No weights authored: bind fully to the first joint.
                [1.0, 0.0, 0.0, 0.0]
            };
            crate::gfx::mesh_payload::SkinnedVertex {
                pos: *pos,
                normal: *normal,
                tangent,
                color: *color,
                uv: *uv,
                joints: [
                    v.joints[0] as u16,
                    v.joints[1] as u16,
                    v.joints[2] as u16,
                    v.joints[3] as u16,
                ],
                weights,
            }
        })
        .collect();

    let payload_joints: Vec<crate::gfx::mesh_payload::PayloadJoint> =
        skeleton.iter().map(joint_def_to_payload).collect();

    // Bake LOD alternates against the skinned vertex set. Half-edge QEM
    // preserves the vertex range so the runtime can share one SkinnedVertex
    // buffer across LOD0 and every alternate; only the per-LOD index list
    // varies.
    let alternates = if lod_levels > 1 {
        let alt_count = (lod_levels - 1) as usize;
        if !lod_distances.is_empty() && lod_distances.len() != alt_count {
            return Err(format!(
                "lod_distances has {} entries but lod_levels = {} expects {}",
                lod_distances.len(),
                lod_levels,
                alt_count,
            ));
        }
        let positions: Vec<[f32; 3]> = skinned.iter().map(|v| v.pos).collect();
        let radius = bounding_sphere_radius(&positions);
        let lod0_tris = indices.len() / 3;
        let mut out: Vec<(f32, Vec<u16>)> = Vec::with_capacity(alt_count);
        for level in 1..lod_levels {
            let target = crate::gfx::lod::target_tri_count_for_level(lod0_tris, level);
            let idx = crate::gfx::lod::decimate_by_qem(&positions, indices, target);
            if idx.is_empty() {
                break;
            }
            let distance = if lod_distances.is_empty() {
                crate::gfx::lod::default_distance_for_level(radius, level)
            } else {
                lod_distances[(level - 1) as usize]
            };
            out.push((distance, idx));
        }
        out
    } else {
        Vec::new()
    };

    let dense: Vec<crate::gfx::mesh_payload::MorphDelta> = morph_deltas
        .iter()
        .map(|d| crate::gfx::mesh_payload::MorphDelta {
            position: d.position,
            normal: d.normal,
        })
        .collect();
    let morphs = crate::gfx::mesh_payload::PayloadMorphs::from_dense(
        morph_target_names.to_vec(),
        vertex_data.len(),
        &dense,
    )
    .map_err(|e| format!("SkinnedMesh {e}"))?;

    Ok(crate::gfx::mesh_payload::serialise_skinned_with_lods(
        &skinned,
        indices,
        &payload_joints,
        &morphs,
        &alternates,
    ))
}

// Compile a VoxelChunk component into a packed binary mesh payload.
//
// `palette_lookup` resolves each palette entry name to its BlockType args
// (typically by scanning the world.jsonl asset list). Non-solid entries
// (`solid: false`) become `None` in the resolved palette and emit no faces.
pub(crate) fn compile_voxel_chunk_payload<F>(
    args: &serde_json::Value,
    mut palette_lookup: F,
) -> Result<Vec<u8>, String>
where
    F: FnMut(&str) -> Option<serde_json::Value>,
{
    let dim = parse_u32x3(args.get("dim"), "dim")?;
    let block_size = args
        .get("block_size")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;
    let palette_names: Vec<String> = args
        .get("palette")
        .and_then(|v| v.as_array())
        .ok_or("VoxelChunk: `palette` must be an array of BlockType names")?
        .iter()
        .map(|v| {
            v.as_str()
                .ok_or_else(|| "VoxelChunk: palette entries must be strings".to_string())
                .map(str::to_string)
        })
        .collect::<Result<_, _>>()?;
    let blocks: Vec<u32> = args
        .get("blocks")
        .and_then(|v| v.as_array())
        .ok_or("VoxelChunk: `blocks` must be an array of palette indices")?
        .iter()
        .enumerate()
        .map(|(i, v)| {
            v.as_u64()
                .map(|x| x as u32)
                .ok_or_else(|| format!("VoxelChunk: blocks[{i}] must be a non-negative integer"))
        })
        .collect::<Result<_, _>>()?;

    let mut palette: Vec<Option<PaletteSlot>> = Vec::with_capacity(palette_names.len());
    for name in &palette_names {
        let bt_args = palette_lookup(name).ok_or_else(|| {
            format!("VoxelChunk: palette entry '{name}' has no matching BlockType asset")
        })?;
        palette.push(resolve_block_type(&bt_args));
    }

    let (vertices, indices) = build_voxel_mesh(dim, block_size, &blocks, &palette)?;
    finish_mesh_payload(vertices, indices, args)
}

fn resolve_block_type(bt_args: &serde_json::Value) -> Option<PaletteSlot> {
    let solid = bt_args
        .get("solid")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    if !solid {
        return None;
    }
    let uv_min = parse_f32x2(bt_args.get("uv_min"), "uv_min").unwrap_or([0.0, 0.0]);
    let uv_max = parse_f32x2(bt_args.get("uv_max"), "uv_max").unwrap_or([1.0, 1.0]);
    let default_rect = [uv_min[0], uv_min[1], uv_max[0], uv_max[1]];
    let parse_rect = |v: Option<&serde_json::Value>| -> [f32; 4] {
        v.and_then(|x| x.as_array())
            .and_then(|a| {
                if a.len() < 4 {
                    return None;
                }
                let mut out = [0.0f32; 4];
                for (i, e) in out.iter_mut().enumerate() {
                    *e = a[i].as_f64()? as f32;
                }
                Some(out)
            })
            .unwrap_or(default_rect)
    };
    Some(PaletteSlot {
        uv_top: parse_rect(bt_args.get("uv_top")),
        uv_bottom: parse_rect(bt_args.get("uv_bottom")),
        uv_side: parse_rect(bt_args.get("uv_side")),
    })
}

fn parse_u32x3(v: Option<&serde_json::Value>, label: &str) -> Result<[u32; 3], String> {
    let arr = v
        .and_then(|x| x.as_array())
        .ok_or_else(|| format!("{label} must be an array of 3 non-negative integers"))?;
    if arr.len() < 3 {
        return Err(format!("{label} must have 3 elements, got {}", arr.len()));
    }
    let f = |i: usize| -> Result<u32, String> {
        arr[i]
            .as_u64()
            .map(|x| x as u32)
            .ok_or_else(|| format!("{label}[{i}] must be a non-negative integer"))
    };
    Ok([f(0)?, f(1)?, f(2)?])
}

// Compile a Room component's JSON args into a packed binary payload.
//
// Handles the `size: [width, depth, height]` shorthand in addition to the
// `half_width` / `half_depth` / `ceiling_height` fields. Texture references
// in the args are ignored here; they are resolved by the build pipeline and
// GraphicsSystem at load time.
pub(crate) fn compile_room_payload(args: &serde_json::Value) -> Result<Vec<u8>, String> {
    let (half_width, half_depth, ceiling_height) = if let Some(size) = args
        .get("size")
        .and_then(|v| v.as_array())
        .filter(|a| a.len() >= 3)
    {
        let w = size[0].as_f64().unwrap_or(16.0) as f32;
        let d = size[1].as_f64().unwrap_or(20.0) as f32;
        let h = size[2].as_f64().unwrap_or(3.5) as f32;
        (w / 2.0, d / 2.0, h)
    } else {
        let hw = args
            .get("half_width")
            .and_then(|v| v.as_f64())
            .unwrap_or(8.0) as f32;
        let hd = args
            .get("half_depth")
            .and_then(|v| v.as_f64())
            .unwrap_or(10.0) as f32;
        let ch = args
            .get("ceiling_height")
            .and_then(|v| v.as_f64())
            .unwrap_or(3.5) as f32;
        (hw, hd, ch)
    };

    let (vertices, indices) =
        room::build_room_geometry(half_width, half_depth, 0.0, ceiling_height);
    finish_mesh_payload(vertices, indices, args)
}

// Builds geometry from inline vertex/index data in the blob JSON.
//
// Vertices may be specified in one of two forms:
//
//   Named fields (preferred):
//     {"pos": [x, y, z], "color": [r, g, b], "uv": [u, v]}
//
//   Flat array (legacy, still accepted):
//     [x, y, z, r, g, b, u, v]   (8 values) or [x, y, z, r, g, b] (6 values, uv defaults to 0)
//
// Normals are computed automatically from the triangle data.
fn build_inline(args: &serde_json::Value) -> Result<(Vec<Vert>, Vec<u16>), String> {
    let verts = args
        .get("vertices")
        .and_then(|v| v.as_array())
        .ok_or("inline Mesh requires a `vertices` array")?;

    let idxs = args
        .get("indices")
        .and_then(|v| v.as_array())
        .ok_or("inline Mesh requires an `indices` array")?;

    let parsed: Vec<RawVert> = verts
        .iter()
        .enumerate()
        .map(|(i, v)| parse_vertex(v, i))
        .collect::<Result<Vec<_>, _>>()?;

    let indices: Vec<u16> = idxs
        .iter()
        .enumerate()
        .map(|(i, v)| {
            v.as_u64()
                .map(|x| x as u16)
                .ok_or_else(|| format!("index[{i}] must be an integer"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut normals: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]; parsed.len()];
    let tris = indices.len() / 3;
    for t in 0..tris {
        let ia = indices[t * 3] as usize;
        let ib = indices[t * 3 + 1] as usize;
        let ic = indices[t * 3 + 2] as usize;
        if ia >= parsed.len() || ib >= parsed.len() || ic >= parsed.len() {
            return Err(format!("index out of range in triangle {t}"));
        }
        let n = vec3_face_normal(parsed[ia].0, parsed[ib].0, parsed[ic].0);
        vec3_add(&mut normals[ia], n);
        vec3_add(&mut normals[ib], n);
        vec3_add(&mut normals[ic], n);
    }

    let vertices = parsed
        .into_iter()
        .enumerate()
        .map(|(i, (pos, color, uv))| (pos, vec3_normalise(normals[i]), color, uv))
        .collect();

    Ok((vertices, indices))
}

fn parse_vertex(v: &serde_json::Value, idx: usize) -> Result<RawVert, String> {
    if let Some(obj) = v.as_object() {
        let pos = parse_f32x3(obj.get("pos"), &format!("vertex[{idx}].pos"))?;
        let color = parse_f32x3(obj.get("color"), &format!("vertex[{idx}].color"))?;
        let uv = if let Some(u) = obj.get("uv") {
            parse_f32x2(Some(u), &format!("vertex[{idx}].uv"))?
        } else {
            [0.0, 0.0]
        };
        Ok((pos, color, uv))
    } else if let Some(arr) = v.as_array() {
        if arr.len() < 6 {
            return Err(format!(
                "vertex[{idx}] flat array must have 6 or 8 elements, got {}",
                arr.len()
            ));
        }
        let f = |i: usize| -> Result<f32, String> {
            arr[i]
                .as_f64()
                .map(|x| x as f32)
                .ok_or_else(|| format!("vertex[{idx}][{i}] must be a number"))
        };
        let uv = if arr.len() >= 8 {
            [f(6)?, f(7)?]
        } else {
            [0.0, 0.0]
        };
        Ok(([f(0)?, f(1)?, f(2)?], [f(3)?, f(4)?, f(5)?], uv))
    } else {
        Err(format!(
            "vertex[{idx}] must be an object or a 6- or 8-element array"
        ))
    }
}

pub(super) fn parse_f32x3(v: Option<&serde_json::Value>, label: &str) -> Result<[f32; 3], String> {
    let arr = v
        .and_then(|x| x.as_array())
        .ok_or_else(|| format!("{label} must be an array of 3 numbers"))?;
    if arr.len() < 3 {
        return Err(format!("{label} must have 3 elements, got {}", arr.len()));
    }
    let f = |i: usize| -> Result<f32, String> {
        arr[i]
            .as_f64()
            .map(|x| x as f32)
            .ok_or_else(|| format!("{label}[{i}] must be a number"))
    };
    Ok([f(0)?, f(1)?, f(2)?])
}

fn parse_f32x2(v: Option<&serde_json::Value>, label: &str) -> Result<[f32; 2], String> {
    let arr = v
        .and_then(|x| x.as_array())
        .ok_or_else(|| format!("{label} must be an array of 2 numbers"))?;
    if arr.len() < 2 {
        return Err(format!("{label} must have 2 elements, got {}", arr.len()));
    }
    let f = |i: usize| -> Result<f32, String> {
        arr[i]
            .as_f64()
            .map(|x| x as f32)
            .ok_or_else(|| format!("{label}[{i}] must be a number"))
    };
    Ok([f(0)?, f(1)?])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{MorphDelta, SkeletonJoint, SkinnedVertexData, VertexData};
    use crate::gfx::mesh_payload::{
        Vertex, deserialise_heightfield, deserialise_skinned_with_lods, deserialise_with_lods,
    };

    // core's plain `deserialise` is `#[cfg(test)]`-scoped to that crate, so it is
    // not linkable here; recover the (vertices, indices) pair from the public
    // with-LODs reader (it ignores any collider trailer, so heightfield payloads
    // read back fine too).
    fn deserialise(bytes: &[u8]) -> Result<(Vec<Vertex>, Vec<u16>), String> {
        let (verts, indices, _) = deserialise_with_lods(bytes)?;
        Ok((verts, indices))
    }

    #[test]
    fn compile_mesh_payload_room_generator_succeeds() {
        let args = serde_json::json!({"generator": "room"});
        assert!(compile_mesh_payload(&args).is_ok());
    }

    #[test]
    fn compile_mesh_payload_sphere_generator_succeeds() {
        let args =
            serde_json::json!({"generator": "sphere", "radius": 1.0, "rings": 6, "segments": 8});
        assert!(compile_mesh_payload(&args).is_ok());
    }

    #[test]
    fn compile_mesh_payload_unknown_generator_errors() {
        let args = serde_json::json!({"generator": "nonexistent"});
        assert!(compile_mesh_payload(&args).is_err());
    }

    #[test]
    fn compile_mesh_payload_known_generators_ok() {
        for name in &[
            "room", "box", "cylinder", "plane", "sphere", "terrain", "skybox",
        ] {
            let args = serde_json::json!({"generator": name});
            assert!(
                compile_mesh_payload(&args).is_ok(),
                "expected ok for generator '{name}'"
            );
        }
    }

    #[test]
    fn compile_mesh_payload_water_grid_generator_builds_a_flat_tessellated_quad() {
        let args = serde_json::json!({
            "generator": "water_grid",
            "half_width": 2.0, "half_depth": 3.0, "subdivisions": 8,
        });
        let (verts, indices) = deserialise(&compile_mesh_payload(&args).unwrap()).unwrap();
        assert_eq!(verts.len(), 9 * 9);
        assert_eq!(indices.len(), 8 * 8 * 6);
        assert!(verts.iter().all(|v| v.pos[1] == 0.0));
        assert!(verts.iter().all(|v| v.normal == [0.0, 1.0, 0.0]));
        let max_x = verts.iter().fold(f32::NEG_INFINITY, |m, v| m.max(v.pos[0]));
        let max_z = verts.iter().fold(f32::NEG_INFINITY, |m, v| m.max(v.pos[2]));
        assert_eq!((max_x, max_z), (2.0, 3.0));
    }

    #[test]
    fn compile_mesh_payload_water_grid_defaults_to_a_sixty_four_subdivision_grid() {
        let args = serde_json::json!({"generator": "water_grid"});
        let (verts, _) = deserialise(&compile_mesh_payload(&args).unwrap()).unwrap();
        assert_eq!(verts.len(), 65 * 65);
    }

    #[test]
    fn compile_mesh_payload_surfaces_generator_errors() {
        // The sphere generator refuses a tessellation past the u16 index limit.
        let sphere = serde_json::json!({"generator": "sphere", "rings": 255, "segments": 255});
        assert!(compile_mesh_payload(&sphere).unwrap_err().contains("u16"));
        // The extrude generator refuses a missing profile.
        let extrude = serde_json::json!({"generator": "extrude"});
        assert!(
            compile_mesh_payload(&extrude)
                .unwrap_err()
                .contains("`profile` array")
        );
    }

    #[test]
    fn compile_mesh_payload_extrude_generator_succeeds() {
        let args = serde_json::json!({
            "generator": "extrude",
            "profile": [[-1, -1], [1, -1], [1, 1], [-1, 1]],
            "height": 1.0
        });
        assert!(compile_mesh_payload(&args).is_ok());
    }

    #[test]
    fn compile_mesh_payload_unknown_generator_name_errors() {
        let args = serde_json::json!({"generator": "teapot"});
        let err = compile_mesh_payload(&args).unwrap_err();
        assert!(
            err.contains("teapot"),
            "expected error mentioning 'teapot', got: {err}"
        );
    }

    #[test]
    fn compile_room_payload_size_shorthand() {
        let args = serde_json::json!({"size": [16.0, 20.0, 3.5]});
        let result = compile_room_payload(&args);
        assert!(result.is_ok());
        assert!(!result.unwrap().is_empty());
    }

    #[test]
    fn compile_room_payload_half_extents() {
        let args =
            serde_json::json!({"half_width": 8.0, "half_depth": 10.0, "ceiling_height": 3.5});
        let args_size = serde_json::json!({"size": [16.0, 20.0, 3.5]});
        assert_eq!(
            compile_room_payload(&args).unwrap(),
            compile_room_payload(&args_size).unwrap()
        );
    }

    #[test]
    fn compile_room_payload_texture_fields_are_ignored() {
        let args = serde_json::json!({
            "size": [16.0, 20.0, 3.5],
            "wall_texture": "brick",
            "floor_texture": "concrete",
            "ceiling_texture": "checker"
        });
        assert!(compile_room_payload(&args).is_ok());
    }

    #[test]
    fn compile_room_payload_surfaces_lod_configuration_errors() {
        let args = serde_json::json!({
            "size": [16.0, 20.0, 3.5], "lod_levels": 3, "lod_distances": [10.0],
        });
        let err = compile_room_payload(&args).unwrap_err();
        assert!(
            err.contains("lod_distances has 1 entries"),
            "error was: {err}"
        );
    }

    // A unit quad in the XY plane with CCW winding facing +Z.
    fn quad_args() -> serde_json::Value {
        serde_json::json!({
            "vertices": [
                {"pos": [0.0, 0.0, 0.0], "color": [1.0, 1.0, 1.0], "uv": [0.0, 0.0]},
                {"pos": [1.0, 0.0, 0.0], "color": [1.0, 1.0, 1.0], "uv": [1.0, 0.0]},
                {"pos": [1.0, 1.0, 0.0], "color": [1.0, 1.0, 1.0], "uv": [1.0, 1.0]},
                {"pos": [0.0, 1.0, 0.0], "color": [1.0, 1.0, 1.0], "uv": [0.0, 1.0]},
            ],
            "indices": [0, 1, 2, 2, 3, 0],
        })
    }

    #[test]
    fn inline_mesh_round_trips_with_derived_normals_and_tangents() {
        let payload = compile_mesh_payload(&quad_args()).unwrap();
        let (verts, indices) = deserialise(&payload).unwrap();
        assert_eq!(verts.len(), 4);
        assert_eq!(indices, vec![0, 1, 2, 2, 3, 0]);
        for v in &verts {
            // CCW in the XY plane faces +Z; the UV gradient runs along +X.
            assert!((v.normal[2] - 1.0).abs() < 1e-5);
            assert!((v.tangent[0] - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn inline_mesh_accepts_flat_vertex_arrays() {
        let args = serde_json::json!({
            "vertices": [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.25, 0.5],
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            ],
            "indices": [0, 1, 2],
        });
        let (verts, indices) = deserialise(&compile_mesh_payload(&args).unwrap()).unwrap();
        assert_eq!(indices, vec![0, 1, 2]);
        assert_eq!(verts[0].uv, [0.25, 0.5]);
        // 6-element vertices default their uv to zero.
        assert_eq!(verts[1].uv, [0.0, 0.0]);
        assert_eq!(verts[2].color, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn inline_mesh_rejects_malformed_args() {
        // Missing vertices / indices.
        assert!(compile_mesh_payload(&serde_json::json!({"indices": [0]})).is_err());
        assert!(compile_mesh_payload(&serde_json::json!({"vertices": []})).is_err());
        // A vertex that is neither an object nor an array.
        let bad_vertex = serde_json::json!({"vertices": ["nope"], "indices": []});
        assert!(
            compile_mesh_payload(&bad_vertex)
                .unwrap_err()
                .contains("vertex[0]")
        );
        // Flat arrays must carry at least 6 numbers.
        let short = serde_json::json!({"vertices": [[0.0, 0.0, 0.0]], "indices": []});
        assert!(compile_mesh_payload(&short).unwrap_err().contains("6 or 8"));
        // Every slot of a flat vertex array must be a number.
        for slot in 0..8 {
            let mut flat = vec![serde_json::json!(0.0); 8];
            flat[slot] = serde_json::json!("x");
            let args = serde_json::json!({"vertices": [flat], "indices": []});
            let err = compile_mesh_payload(&args).unwrap_err();
            assert!(
                err.contains(&format!("vertex[0][{slot}] must be a number")),
                "slot {slot} gave: {err}"
            );
        }
        // Indices must be integers and in range.
        let bad_index = serde_json::json!({"vertices": [[0, 0, 0, 1, 1, 1]], "indices": ["x"]});
        assert!(
            compile_mesh_payload(&bad_index)
                .unwrap_err()
                .contains("index[0]")
        );
        let oob = serde_json::json!({
            "vertices": [[0, 0, 0, 1, 1, 1], [1, 0, 0, 1, 1, 1], [0, 1, 0, 1, 1, 1]],
            "indices": [0, 1, 9],
        });
        assert!(
            compile_mesh_payload(&oob)
                .unwrap_err()
                .contains("triangle 0")
        );
    }

    #[test]
    fn inline_mesh_validates_named_vertex_fields() {
        let vertex = |v: serde_json::Value| serde_json::json!({"vertices": [v], "indices": []});
        let cases: [(serde_json::Value, &str); 4] = [
            (serde_json::json!({"color": [1, 1, 1]}), "vertex[0].pos"),
            (serde_json::json!({"pos": [0, 0, 0]}), "vertex[0].color"),
            (
                serde_json::json!({"pos": [0, 0, 0], "color": [1, 1, 1], "uv": [0]}),
                "vertex[0].uv must have 2 elements",
            ),
            (
                serde_json::json!({"pos": [0, 0], "color": [1, 1, 1]}),
                "vertex[0].pos must have 3 elements",
            ),
        ];
        for (v, expected) in cases {
            let err = compile_mesh_payload(&vertex(v)).unwrap_err();
            assert!(err.contains(expected), "expected '{expected}', got: {err}");
        }
    }

    #[test]
    fn inline_mesh_named_vertices_default_their_uv_to_zero() {
        let args = serde_json::json!({
            "vertices": [
                {"pos": [0.0, 0.0, 0.0], "color": [1.0, 1.0, 1.0]},
                {"pos": [1.0, 0.0, 0.0], "color": [1.0, 1.0, 1.0]},
                {"pos": [0.0, 1.0, 0.0], "color": [1.0, 1.0, 1.0]},
            ],
            "indices": [0, 1, 2],
        });
        let (verts, _) = deserialise(&compile_mesh_payload(&args).unwrap()).unwrap();
        assert!(verts.iter().all(|v| v.uv == [0.0, 0.0]));
    }

    #[test]
    fn lod_alternates_stop_when_decimation_collapses() {
        // Vertices with no triangles decimate to nothing, so no alternate is
        // emitted even though three levels were requested.
        let args = serde_json::json!({
            "vertices": [[0, 0, 0, 1, 1, 1], [1, 0, 0, 1, 1, 1], [0, 1, 0, 1, 1, 1]],
            "indices": [],
            "lod_levels": 3,
        });
        let (verts, lod0, alternates) =
            deserialise_with_lods(&compile_mesh_payload(&args).unwrap()).unwrap();
        assert_eq!(verts.len(), 3);
        assert!(lod0.is_empty());
        assert!(alternates.is_empty());
    }

    #[test]
    fn heightfield_generator_is_routed_through_the_build_crate() {
        let args = serde_json::json!({"generator": "heightfield"});
        let err = compile_mesh_payload(&args).unwrap_err();
        assert!(err.contains("heightfield"), "error was: {err}");
    }

    #[test]
    fn heightfield_payload_bakes_a_square_collider_grid() {
        let args = serde_json::json!({
            "subdivisions": 4,
            "elevation_min": 0.0,
            "elevation_max": 10.0,
        });
        // A uniform white 2x2 source image maps every sample to elevation_max.
        let rgba = vec![255u8; 2 * 2 * 4];
        let payload = compile_heightfield_payload(&args, 2, 2, rgba).unwrap();
        let (verts, _) = deserialise(&payload).unwrap();
        assert_eq!(verts.len(), 25);
        let grid = deserialise_heightfield(&payload)
            .unwrap()
            .expect("collider trailer");
        assert_eq!((grid.rows, grid.cols), (5, 5));
        assert_eq!(grid.heights.len(), 25);
        assert!(grid.heights.iter().all(|h| (h - 10.0).abs() < 1e-4));
    }

    #[test]
    fn heightfield_payload_requires_elevation_max() {
        let args = serde_json::json!({"subdivisions": 4});
        let rgba = vec![255u8; 2 * 2 * 4];
        assert!(compile_heightfield_payload(&args, 2, 2, rgba).is_err());
    }

    #[test]
    fn heightfield_payload_surfaces_lod_configuration_errors() {
        let args = serde_json::json!({
            "subdivisions": 4,
            "elevation_max": 10.0,
            "lod_levels": 3,
            "lod_distances": [10.0],
        });
        let rgba = vec![255u8; 2 * 2 * 4];
        let err = compile_heightfield_payload(&args, 2, 2, rgba).unwrap_err();
        assert!(
            err.contains("lod_distances has 1 entries"),
            "error was: {err}"
        );
    }

    #[test]
    fn collider_grid_rejects_non_square_vertex_counts() {
        let v = |y: f32| -> Vert { ([0.0, y, 0.0], [0.0, 1.0, 0.0], [1.0; 3], [0.0, 0.0]) };
        assert!(heightfield_collider_grid(&[v(0.0), v(1.0), v(2.0)]).is_err());
        let (n, heights) = heightfield_collider_grid(&[v(0.0), v(1.0), v(2.0), v(3.0)]).unwrap();
        assert_eq!(n, 2);
        assert_eq!(heights, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn lod_levels_emit_decimated_alternates_with_rising_distances() {
        let args = serde_json::json!({
            "generator": "sphere", "radius": 1.0, "rings": 16, "segments": 24,
            "lod_levels": 3,
        });
        let payload = compile_mesh_payload(&args).unwrap();
        let (_, lod0, alternates) = deserialise_with_lods(&payload).unwrap();
        assert_eq!(alternates.len(), 2);
        assert!(alternates[0].1.len() < lod0.len());
        assert!(alternates[1].1.len() <= alternates[0].1.len());
        assert!(alternates[0].0 > 0.0);
        assert!(alternates[1].0 > alternates[0].0);
    }

    #[test]
    fn single_lod_payload_matches_the_legacy_format() {
        let args = serde_json::json!({"generator": "box"});
        let one = compile_mesh_payload(&args).unwrap();
        let (_, _, alternates) = deserialise_with_lods(&one).unwrap();
        assert!(alternates.is_empty());
        let explicit = serde_json::json!({"generator": "box", "lod_levels": 1});
        assert_eq!(compile_mesh_payload(&explicit).unwrap(), one);
    }

    #[test]
    fn lod_distance_count_must_match_lod_levels() {
        let args = serde_json::json!({
            "generator": "box", "lod_levels": 3, "lod_distances": [10.0],
        });
        let err = compile_mesh_payload(&args).unwrap_err();
        assert!(err.contains("lod_distances"), "error was: {err}");
    }

    #[test]
    fn explicit_lod_distances_are_stored_in_the_trailer() {
        let args = serde_json::json!({
            "generator": "sphere", "radius": 1.0, "rings": 16, "segments": 24,
            "lod_levels": 2, "lod_distances": [42.0],
        });
        let (_, _, alternates) =
            deserialise_with_lods(&compile_mesh_payload(&args).unwrap()).unwrap();
        assert_eq!(alternates.len(), 1);
        assert_eq!(alternates[0].0, 42.0);
    }

    #[test]
    fn bounding_sphere_radius_covers_empty_small_and_boxed_inputs() {
        assert_eq!(bounding_sphere_radius(&[]), 1.0);
        // Degenerate extents clamp to the 0.25 floor.
        assert_eq!(bounding_sphere_radius(&[[0.0; 3]]), 0.25);
        assert_eq!(bounding_sphere_radius(&[[3.0; 3]]), 0.25);
        // A unit cube has a half-diagonal of sqrt(3)/2.
        let cube = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
        assert!((bounding_sphere_radius(&cube) - (3.0f32).sqrt() / 2.0).abs() < 1e-6);
    }

    #[test]
    fn scalar_array_parsers_validate_shape_and_types() {
        assert_eq!(
            parse_f32x3(Some(&serde_json::json!([1, 2, 3])), "v").unwrap(),
            [1.0, 2.0, 3.0]
        );
        assert!(parse_f32x3(None, "v").is_err());
        assert!(parse_f32x3(Some(&serde_json::json!("x")), "v").is_err());
        assert!(parse_f32x3(Some(&serde_json::json!([1, 2])), "v").is_err());

        assert_eq!(
            parse_f32x2(Some(&serde_json::json!([0.5, 1.5])), "uv").unwrap(),
            [0.5, 1.5]
        );
        assert!(parse_f32x2(Some(&serde_json::json!([0.5])), "uv").is_err());

        assert_eq!(
            parse_u32x3(Some(&serde_json::json!([1, 2, 3])), "dim").unwrap(),
            [1, 2, 3]
        );
        assert!(parse_u32x3(None, "dim").is_err());
        assert!(parse_u32x3(Some(&serde_json::json!([1])), "dim").is_err());
    }

    #[test]
    fn scalar_array_parsers_name_the_offending_element() {
        // Each slot is validated in turn, and the message points at the one
        // that failed rather than at the array as a whole.
        for slot in 0..3 {
            let mut v = vec![serde_json::json!(1); 3];
            v[slot] = serde_json::json!("x");
            let err = parse_f32x3(Some(&serde_json::Value::Array(v.clone())), "v").unwrap_err();
            assert_eq!(err, format!("v[{slot}] must be a number"));

            v[slot] = serde_json::json!(-1);
            let err = parse_u32x3(Some(&serde_json::Value::Array(v)), "dim").unwrap_err();
            assert_eq!(err, format!("dim[{slot}] must be a non-negative integer"));
        }
        for slot in 0..2 {
            let mut v = vec![serde_json::json!(1); 2];
            v[slot] = serde_json::json!(true);
            let err = parse_f32x2(Some(&serde_json::Value::Array(v)), "uv").unwrap_err();
            assert_eq!(err, format!("uv[{slot}] must be a number"));
        }
    }

    #[test]
    fn vertex_data_compilation_derives_normals() {
        let vd = |pos: [f32; 3], uv: [f32; 2]| VertexData {
            pos,
            color: [1.0; 3],
            uv,
        };
        let verts = [
            vd([0.0, 0.0, 0.0], [0.0, 0.0]),
            vd([1.0, 0.0, 0.0], [1.0, 0.0]),
            vd([0.0, 1.0, 0.0], [0.0, 1.0]),
        ];
        let payload = compile_mesh_from_vertex_data(&verts, &[0, 1, 2]);
        let (out, indices) = deserialise(&payload).unwrap();
        assert_eq!(indices, vec![0, 1, 2]);
        assert!(out.iter().all(|v| (v.normal[2] - 1.0).abs() < 1e-5));

        // Out-of-range indices are skipped rather than failing the compile.
        let payload = compile_mesh_from_vertex_data(&verts, &[0, 1, 9]);
        let (out, _) = deserialise(&payload).unwrap();
        assert_eq!(out.len(), 3);
        // No triangle touched the vertices, so normals fall back to +Y.
        assert!(out.iter().all(|v| v.normal == [0.0, 1.0, 0.0]));
    }

    fn joint(name: &str) -> SkeletonJoint {
        SkeletonJoint {
            name: name.to_string(),
            parent: -1,
            translation: [0.0; 3],
            rotation_deg: [0.0; 3],
            scale: [1.0; 3],
        }
    }

    fn skinned_vertex(pos: [f32; 3], weights: [f32; 4]) -> SkinnedVertexData {
        SkinnedVertexData {
            pos,
            color: [1.0; 3],
            uv: [0.0, 0.0],
            joints: [0; 4],
            weights,
        }
    }

    fn morph_delta(position: [f32; 3]) -> MorphDelta {
        MorphDelta {
            position,
            normal: [0.0, 1.0, 0.0],
        }
    }

    #[test]
    fn skinned_mesh_normalises_weights_and_keeps_the_skeleton() {
        let verts = [
            skinned_vertex([0.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0]),
            skinned_vertex([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            skinned_vertex([0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]),
        ];
        let payload = compile_skinned_mesh_payload_with_lods(
            &verts,
            &[0, 1, 2],
            &[joint("root")],
            &[],
            &[],
            &SkinnedLods {
                levels: 1,
                distances: &[],
            },
        )
        .unwrap();
        let p = deserialise_skinned_with_lods(&payload).unwrap();
        let (out, indices, joints, alternates) = (p.vertices, p.indices, p.joints, p.lods);
        assert_eq!(indices, vec![0, 1, 2]);
        assert!(alternates.is_empty());
        assert_eq!(joints.len(), 1);
        assert_eq!(joints[0].name, "root");
        // Authored weights are normalised to sum 1.
        assert_eq!(out[0].weights, [0.5, 0.5, 0.0, 0.0]);
        // Unweighted vertices bind fully to joint 0.
        assert_eq!(out[2].weights, [1.0, 0.0, 0.0, 0.0]);
        assert!((out[0].normal[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn skinned_mesh_rejects_empty_and_out_of_range_input() {
        assert!(
            compile_skinned_mesh_payload_with_lods(
                &[],
                &[],
                &[joint("root")],
                &[],
                &[],
                &SkinnedLods {
                    levels: 1,
                    distances: &[],
                },
            )
            .is_err()
        );
        let tri = [
            skinned_vertex([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            skinned_vertex([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            skinned_vertex([0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
        ];
        let err = compile_skinned_mesh_payload_with_lods(
            &tri,
            &[0, 1, 5],
            &[joint("root")],
            &[],
            &[],
            &SkinnedLods {
                levels: 1,
                distances: &[],
            },
        )
        .unwrap_err();
        assert!(err.contains("out of range"), "error was: {err}");
        // LOD distance count must match lod_levels - 1.
        let err = compile_skinned_mesh_payload_with_lods(
            &tri,
            &[0, 1, 2],
            &[joint("root")],
            &[],
            &[],
            &SkinnedLods {
                levels: 3,
                distances: &[5.0],
            },
        )
        .unwrap_err();
        assert!(err.contains("lod_distances"), "error was: {err}");
    }

    #[test]
    fn skinned_mesh_emits_lod_alternates_for_dense_input() {
        // A 5x5 vertex grid (32 triangles) is dense enough to decimate.
        let n = 5usize;
        let mut verts = Vec::new();
        for z in 0..n {
            for x in 0..n {
                verts.push(skinned_vertex(
                    [x as f32, ((x * z) % 3) as f32 * 0.1, z as f32],
                    [1.0, 0.0, 0.0, 0.0],
                ));
            }
        }
        let mut indices: Vec<u16> = Vec::new();
        for z in 0..n - 1 {
            for x in 0..n - 1 {
                let a = (z * n + x) as u16;
                let b = a + 1;
                let c = a + n as u16;
                let d = c + 1;
                indices.extend_from_slice(&[a, b, d, d, c, a]);
            }
        }
        let payload = compile_skinned_mesh_payload_with_lods(
            &verts,
            &indices,
            &[joint("root")],
            &[],
            &[],
            &SkinnedLods {
                levels: 2,
                distances: &[12.0],
            },
        )
        .unwrap();
        let p = deserialise_skinned_with_lods(&payload).unwrap();
        let (lod0, alternates) = (p.indices, p.lods);
        assert_eq!(lod0.len(), indices.len());
        assert_eq!(alternates.len(), 1);
        assert_eq!(alternates[0].0, 12.0);
        assert!(alternates[0].1.len() < indices.len());
        assert_eq!(alternates[0].1.len() % 3, 0);
    }

    #[test]
    fn skinned_mesh_rejects_a_mismatched_morph_delta_count() {
        let tri = [
            skinned_vertex([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            skinned_vertex([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            skinned_vertex([0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
        ];
        let err = compile_skinned_mesh_payload_with_lods(
            &tri,
            &[0, 1, 2],
            &[joint("root")],
            &["smile".to_string()],
            &[morph_delta([0.1, 0.0, 0.0])],
            &SkinnedLods {
                levels: 1,
                distances: &[],
            },
        )
        .unwrap_err();
        assert!(
            err.contains("morph_deltas has 1 entries; 1 target(s) x 3 vertices requires 3"),
            "error was: {err}"
        );
    }

    #[test]
    fn skinned_mesh_round_trips_morph_targets() {
        let tri = [
            skinned_vertex([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            skinned_vertex([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            skinned_vertex([0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
        ];
        let deltas = [
            morph_delta([0.5, 0.0, 0.0]),
            morph_delta([0.0, 0.5, 0.0]),
            morph_delta([0.0, 0.0, 0.5]),
        ];
        let payload = compile_skinned_mesh_payload_with_lods(
            &tri,
            &[0, 1, 2],
            &[joint("root")],
            &["smile".to_string()],
            &deltas,
            &SkinnedLods {
                levels: 1,
                distances: &[],
            },
        )
        .unwrap();
        let p = deserialise_skinned_with_lods(&payload).unwrap();
        assert_eq!(p.morphs.names, vec!["smile".to_string()]);
        assert_eq!(p.morphs.entries.len(), 3);
        let dense = p.morphs.to_dense();
        assert_eq!(dense.len(), 3);
        assert_eq!(dense[0].position, [0.5, 0.0, 0.0]);
        assert_eq!(dense[2].position, [0.0, 0.0, 0.5]);
        assert_eq!(dense[1].normal, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn skinned_mesh_derives_lod_distances_when_none_are_given() {
        let n = 5usize;
        let mut verts = Vec::new();
        for z in 0..n {
            for x in 0..n {
                verts.push(skinned_vertex(
                    [x as f32, 0.0, z as f32],
                    [1.0, 0.0, 0.0, 0.0],
                ));
            }
        }
        let mut indices: Vec<u16> = Vec::new();
        for z in 0..n - 1 {
            for x in 0..n - 1 {
                let a = (z * n + x) as u16;
                indices.extend_from_slice(&[
                    a,
                    a + 1,
                    a + 1 + n as u16,
                    a + 1 + n as u16,
                    a + n as u16,
                    a,
                ]);
            }
        }
        let payload = compile_skinned_mesh_payload_with_lods(
            &verts,
            &indices,
            &[joint("root")],
            &[],
            &[],
            &SkinnedLods {
                levels: 3,
                distances: &[],
            },
        )
        .unwrap();
        let lods = deserialise_skinned_with_lods(&payload).unwrap().lods;
        assert_eq!(lods.len(), 2);
        // The default cascade doubles per level off the bounding-sphere radius.
        assert!(lods[0].0 > 0.0);
        assert!((lods[1].0 - lods[0].0 * 2.0).abs() < 1e-4);
    }

    #[test]
    fn skinned_mesh_lod_stops_when_decimation_collapses() {
        let tri = [
            skinned_vertex([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            skinned_vertex([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            skinned_vertex([0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
        ];
        // No triangles to decimate, so the alternate list stays empty.
        let payload = compile_skinned_mesh_payload_with_lods(
            &tri,
            &[],
            &[joint("root")],
            &[],
            &[],
            &SkinnedLods {
                levels: 3,
                distances: &[],
            },
        )
        .unwrap();
        assert!(
            deserialise_skinned_with_lods(&payload)
                .unwrap()
                .lods
                .is_empty()
        );
    }

    fn voxel_args(dim: [u32; 3], blocks: Vec<u32>) -> serde_json::Value {
        serde_json::json!({
            "dim": dim,
            "palette": ["stone"],
            "blocks": blocks,
        })
    }

    #[test]
    fn voxel_chunk_single_block_emits_six_faces() {
        let args = voxel_args([1, 1, 1], vec![0]);
        let lookup = |_: &str| Some(serde_json::json!({"solid": true}));
        let payload = compile_voxel_chunk_payload(&args, lookup).unwrap();
        let (verts, indices) = deserialise(&payload).unwrap();
        assert_eq!(verts.len(), 6 * 4);
        assert_eq!(indices.len(), 6 * 6);
    }

    #[test]
    fn voxel_chunk_air_palette_emits_no_geometry() {
        let args = voxel_args([1, 1, 1], vec![0]);
        let lookup = |_: &str| Some(serde_json::json!({"solid": false}));
        let payload = compile_voxel_chunk_payload(&args, lookup).unwrap();
        let (verts, indices) = deserialise(&payload).unwrap();
        assert!(verts.is_empty());
        assert!(indices.is_empty());
    }

    #[test]
    fn voxel_chunk_validates_its_args() {
        let solid = |_: &str| Some(serde_json::json!({"solid": true}));
        // Missing dim / palette / blocks.
        assert!(compile_voxel_chunk_payload(&serde_json::json!({}), solid).is_err());
        assert!(
            compile_voxel_chunk_payload(
                &serde_json::json!({"dim": [1, 1, 1], "palette": ["stone"]}),
                solid
            )
            .is_err()
        );
        // Palette entries must be strings.
        assert!(
            compile_voxel_chunk_payload(
                &serde_json::json!({"dim": [1, 1, 1], "palette": [7], "blocks": [0]}),
                solid
            )
            .is_err()
        );
        // The blocks length must match the dim.
        assert!(compile_voxel_chunk_payload(&voxel_args([2, 1, 1], vec![0]), solid).is_err());
        // Block ids must index the palette.
        assert!(compile_voxel_chunk_payload(&voxel_args([1, 1, 1], vec![3]), solid).is_err());
        // Block ids must be non-negative integers.
        let bad = serde_json::json!({"dim": [1, 1, 1], "palette": ["stone"], "blocks": [1.5]});
        assert!(compile_voxel_chunk_payload(&bad, solid).is_err());
        // Unresolvable palette entries name the missing BlockType.
        let err =
            compile_voxel_chunk_payload(&voxel_args([1, 1, 1], vec![0]), |_| None).unwrap_err();
        assert!(err.contains("stone"), "error was: {err}");
        // A valid dim with no palette at all is reported against `palette`.
        let no_palette = serde_json::json!({"dim": [1, 1, 1], "blocks": [0]});
        let err = compile_voxel_chunk_payload(&no_palette, solid).unwrap_err();
        assert!(
            err.contains("`palette` must be an array"),
            "error was: {err}"
        );
    }

    #[test]
    fn voxel_chunk_surfaces_lod_configuration_errors() {
        let mut args = voxel_args([1, 1, 1], vec![0]);
        args["lod_levels"] = serde_json::json!(3);
        args["lod_distances"] = serde_json::json!([10.0]);
        let err =
            compile_voxel_chunk_payload(&args, |_: &str| Some(serde_json::json!({"solid": true})))
                .unwrap_err();
        assert!(
            err.contains("lod_distances has 1 entries"),
            "error was: {err}"
        );
    }

    #[test]
    fn block_type_resolution_handles_uv_fields_and_air() {
        assert!(resolve_block_type(&serde_json::json!({"solid": false})).is_none());
        // Defaults: full-atlas rect on every face.
        let slot = resolve_block_type(&serde_json::json!({})).unwrap();
        assert_eq!(slot.uv_top, [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(slot.uv_side, [0.0, 0.0, 1.0, 1.0]);
        // uv_min / uv_max seed the default rect; explicit per-face rects win;
        // malformed rects fall back to the default.
        let slot = resolve_block_type(&serde_json::json!({
            "uv_min": [0.25, 0.25],
            "uv_max": [0.5, 0.5],
            "uv_top": [0.0, 0.0, 0.125, 0.125],
            "uv_side": [0.1],
        }))
        .unwrap();
        assert_eq!(slot.uv_top, [0.0, 0.0, 0.125, 0.125]);
        assert_eq!(slot.uv_bottom, [0.25, 0.25, 0.5, 0.5]);
        assert_eq!(slot.uv_side, [0.25, 0.25, 0.5, 0.5]);
        // A long-enough rect holding a non-number also falls back.
        let slot = resolve_block_type(&serde_json::json!({
            "uv_min": [0.5, 0.5],
            "uv_max": [0.75, 0.75],
            "uv_top": [0.0, 0.0, "x", 1.0],
        }))
        .unwrap();
        assert_eq!(slot.uv_top, [0.5, 0.5, 0.75, 0.75]);
    }
}
