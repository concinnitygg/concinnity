//! Mesh payload baking: derive tangents from the UV gradient, build the
//! optional LOD alternate index lists, and serialise the packed mesh payload.
//! The shared tail every mesh generator's output runs through before it can be
//! played, whether the caller is the cook pipeline or a runtime bake.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::geometry::{Vert, compute_tangents};
use crate::math::sqrt;

// Tangent-bearing vertex form the payload serialiser packs:
// position, normal, tangent, color, uv.
type VertT = ([f32; 3], [f32; 3], [f32; 3], [f32; 3], [f32; 2]);

/// Bake generated geometry into a packed binary mesh payload.
///
/// `lod_levels` is the total count including LOD0 (clamped to `[1, 8]`); the
/// alternates are `levels - 1` decimated index lists via QEM vertex clustering
/// on the LOD0 vertex set, each paired with a switch distance. `lod_distances`
/// either gives explicit thresholds or, when empty, a default doubling
/// sequence derived from the bounding-sphere radius. `lod_levels = 1` (the
/// default) produces no trailer and emits the single-LOD payload.
pub fn finish_mesh_payload(
    vertices: Vec<Vert>,
    indices: Vec<u16>,
    lod_levels: u32,
    lod_distances: &[f32],
) -> Result<Vec<u8>, String> {
    let tangents = compute_tangents(&vertices, &indices);
    let verts5: Vec<VertT> = vertices
        .into_iter()
        .zip(tangents)
        .map(|((pos, normal, color, uv), tangent)| (pos, normal, tangent, color, uv))
        .collect();
    let alternates = build_lod_alternates(lod_levels, lod_distances, &verts5, &indices)?;
    Ok(crate::gfx::mesh_payload::serialise_with_lods(
        &verts5,
        &indices,
        &alternates,
    ))
}

// Build the per-LOD `(switch_distance, indices)` list for a mesh. Returns an
// empty `Vec` when `lod_levels <= 1` so the payload writer emits a
// single-LOD blob.
fn build_lod_alternates(
    lod_levels: u32,
    lod_distances: &[f32],
    verts: &[VertT],
    indices: &[u16],
) -> Result<Vec<(f32, Vec<u16>)>, String> {
    let lod_levels = lod_levels.clamp(1, 8);
    if lod_levels <= 1 {
        return Ok(Vec::new());
    }
    let alt_count = (lod_levels - 1) as usize;

    // Explicit thresholds, or a default cascade derived from the
    // bounding-sphere radius that doubles per level, which keeps successive
    // LODs visibly apart when seen in the showcase's free-fly camera.
    if !lod_distances.is_empty() && lod_distances.len() != alt_count {
        return Err(format!(
            "lod_distances has {} entries but lod_levels = {} expects {}",
            lod_distances.len(),
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
        let distance = if lod_distances.is_empty() {
            crate::gfx::lod::default_distance_for_level(radius, level)
        } else {
            lod_distances[(level - 1) as usize]
        };
        out.push((distance, idx));
    }
    Ok(out)
}

/// Bounding-sphere radius around the mesh AABB centre. Cheap upper bound on
/// the per-vertex distance to centre, used to seed default LOD thresholds.
pub fn bounding_sphere_radius(positions: &[[f32; 3]]) -> f32 {
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
    (0.5 * sqrt(dx * dx + dy * dy + dz * dz)).max(0.25)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{build_box, build_sphere};
    use crate::gfx::mesh_payload::deserialise_with_lods;

    #[test]
    fn single_lod_payload_carries_no_alternates() {
        let (verts, indices) = build_box([0.5, 0.5, 0.5]);
        let payload = finish_mesh_payload(verts, indices, 1, &[]).unwrap();
        let (out, lod0, alternates) = deserialise_with_lods(&payload).unwrap();
        assert_eq!(out.len(), 24);
        assert_eq!(lod0.len(), 36);
        assert!(alternates.is_empty());
    }

    #[test]
    fn lod_levels_emit_decimated_alternates_with_rising_distances() {
        let (verts, indices) = build_sphere(1.0, 16, 24).unwrap();
        let payload = finish_mesh_payload(verts, indices, 3, &[]).unwrap();
        let (_, lod0, alternates) = deserialise_with_lods(&payload).unwrap();
        assert_eq!(alternates.len(), 2);
        assert!(alternates[0].1.len() < lod0.len());
        assert!(alternates[1].1.len() <= alternates[0].1.len());
        assert!(alternates[0].0 > 0.0);
        assert!(alternates[1].0 > alternates[0].0);
    }

    #[test]
    fn lod_distance_count_must_match_lod_levels() {
        let (verts, indices) = build_box([0.5, 0.5, 0.5]);
        let err = finish_mesh_payload(verts, indices, 3, &[10.0]).unwrap_err();
        assert!(err.contains("lod_distances"), "error was: {err}");
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
}
