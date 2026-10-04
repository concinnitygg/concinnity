//! The CPU half of a skinned-geometry edit: the bounds of a size-matched
//! in-place write, and the repack of the shared skinned buffers when a
//! reloaded mesh no longer fits its slot.
//!
//! The shared skinned index buffer holds absolute indices: each slot's
//! mesh-relative indices are rebased onto its `vertex_base`.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::ToString;
use alloc::vec::Vec;
use core::mem::size_of;

use crate::gfx::mesh_payload::SkinnedVertex;
use crate::gfx::render_types::{SkinnedDrawObject, SkinnedIndex};
use crate::render::backend::{SkinnedDrawGeometryUpdate, SkinnedSlotLayout};
use crate::render::error::{RenderError, RenderResult};

/// Where a size-matched skinned update lands in the shared buffers.
#[derive(Debug, PartialEq, Eq)]
pub struct SkinnedSlotWrite {
    /// Byte offset of the slot's vertices in the shared vertex buffer.
    pub vertex_offset: u64,
    /// Byte offset of the slot's indices in the shared index buffer.
    pub index_offset: u64,
    /// The update's indices, rebased onto the slot's vertices.
    pub indices: Vec<u32>,
}

/// Place a size-matched update of slot `skinned_index`: `vertex_count`
/// vertices written at `vertex_base` and `indices` over the slot's own index
/// region. Errors when the slot does not exist, when the index count differs
/// from the slot's (a size change needs [`repack_skinned_geometry`]), or when
/// the vertices would overrun the `vertex_buffer_bytes` the shared vertex
/// buffer holds and corrupt a neighboring slot.
pub fn place_skinned_update(
    objects: &[SkinnedDrawObject],
    skinned_index: SkinnedIndex,
    vertex_base: u32,
    vertex_count: usize,
    indices: &[u16],
    vertex_buffer_bytes: usize,
) -> RenderResult<SkinnedSlotWrite> {
    let obj = objects.get(skinned_index.index()).ok_or_else(|| {
        RenderError::Other(format!(
            "update_skinned_mesh_geometry: skinned object {skinned_index} out of range"
        ))
    })?;
    if indices.len() != obj.index_count {
        return Err(RenderError::Other(format!(
            "update_skinned_mesh_geometry: skinned {skinned_index} expects {} indices, got {} \
             (in-place path is size-matched only; size changes route through \
             rebuild_skinned_geometry)",
            obj.index_count,
            indices.len()
        )));
    }
    let stride = size_of::<SkinnedVertex>();
    let start = (vertex_base as usize).saturating_mul(stride);
    let end = start.saturating_add(vertex_count.saturating_mul(stride));
    if end > vertex_buffer_bytes {
        return Err(RenderError::Other(format!(
            "update_skinned_mesh_geometry: vertex region [{start}, {end}) overruns skinned \
             vertex buffer length {vertex_buffer_bytes}"
        )));
    }
    Ok(SkinnedSlotWrite {
        vertex_offset: start as u64,
        index_offset: (obj.index_offset * size_of::<u32>()) as u64,
        indices: indices
            .iter()
            .map(|&i| u32::from(i) + vertex_base)
            .collect(),
    })
}

/// One skinned slot's region of the repacked buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SkinnedRegion {
    /// First vertex of the slot's region.
    pub vertex_base: u32,
    /// Vertices in the region.
    pub vertex_count: usize,
    /// Element offset of the slot's indices.
    pub index_offset: usize,
    /// Indices in the region.
    pub index_count: usize,
}

/// The repacked skinned buffer contents and each slot's region in them.
#[derive(Debug)]
pub struct RepackedSkinnedGeometry {
    /// The new shared skinned vertex buffer contents.
    pub vertices: Vec<SkinnedVertex>,
    /// The new shared skinned index buffer contents, absolute.
    pub indices: Vec<u32>,
    /// One region per slot, in slot order.
    pub regions: Vec<SkinnedRegion>,
    /// Changes naming a slot past the slot list, which were skipped.
    pub ignored_changes: usize,
}

impl RepackedSkinnedGeometry {
    /// Point each slot at its repacked region.
    pub fn apply_to(&self, objects: &mut [SkinnedDrawObject]) {
        for (obj, region) in objects.iter_mut().zip(&self.regions) {
            obj.vertex_base = region.vertex_base;
            obj.vertex_count = region.vertex_count;
            obj.index_offset = region.index_offset;
            obj.index_count = region.index_count;
        }
    }

    /// Each slot's new layout, in slot order.
    pub fn layouts(&self) -> Vec<SkinnedSlotLayout> {
        self.regions
            .iter()
            .enumerate()
            .map(|(i, region)| SkinnedSlotLayout {
                skinned_index: SkinnedIndex::from_usize(i),
                vertex_base: region.vertex_base,
                vertex_count: region.vertex_count,
                index_count: region.index_count,
            })
            .collect()
    }
}

/// Repack the shared skinned vertex and index buffers, replacing the geometry
/// of each slot named in `changes` and copying every other slot from the old
/// buffers, rebased onto its new vertex region. Errors when a slot's recorded
/// region lies outside the old buffers, when an old index lies below its
/// slot's `vertex_base`, or when the result would leave either buffer empty.
pub fn repack_skinned_geometry(
    objects: &[SkinnedDrawObject],
    old_vertices: &[SkinnedVertex],
    old_indices: &[u32],
    changes: Vec<SkinnedDrawGeometryUpdate>,
) -> RenderResult<RepackedSkinnedGeometry> {
    let mut change_map: BTreeMap<SkinnedIndex, SkinnedDrawGeometryUpdate> =
        changes.into_iter().map(|c| (c.skinned_index, c)).collect();
    let mut vertices: Vec<SkinnedVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut regions = Vec::with_capacity(objects.len());
    for (i, obj) in objects.iter().enumerate() {
        let skinned_index = SkinnedIndex::from_usize(i);
        let vertex_base = vertices.len() as u32;
        let index_offset = indices.len();
        let (vertex_count, index_count) = match change_map.remove(&skinned_index) {
            Some(change) => {
                vertices.extend_from_slice(&change.vertices);
                indices.extend(change.indices.iter().map(|&i| u32::from(i) + vertex_base));
                (change.vertices.len(), change.indices.len())
            }
            None => {
                let v_start = obj.vertex_base as usize;
                let v_end = v_start + obj.vertex_count;
                let src = old_vertices.get(v_start..v_end).ok_or_else(|| {
                    RenderError::Other(format!(
                        "rebuild_skinned_geometry: slot {skinned_index} vertex region \
                         [{v_start}, {v_end}) out of bounds (buffer has {} vertices)",
                        old_vertices.len()
                    ))
                })?;
                vertices.extend_from_slice(src);
                let i_end = obj.index_offset + obj.index_count;
                let src = old_indices.get(obj.index_offset..i_end).ok_or_else(|| {
                    RenderError::Other(format!(
                        "rebuild_skinned_geometry: slot {skinned_index} index region \
                         [{}, {i_end}) out of bounds (buffer has {} indices)",
                        obj.index_offset,
                        old_indices.len()
                    ))
                })?;
                let old_base = obj.vertex_base;
                for &abs in src {
                    let local = abs.checked_sub(old_base).ok_or_else(|| {
                        RenderError::Other(format!(
                            "rebuild_skinned_geometry: stale index {abs} below \
                             vertex_base {old_base} on slot {skinned_index}"
                        ))
                    })?;
                    indices.push(local + vertex_base);
                }
                (obj.vertex_count, obj.index_count)
            }
        };
        regions.push(SkinnedRegion {
            vertex_base,
            vertex_count,
            index_offset,
            index_count,
        });
    }
    if vertices.is_empty() || indices.is_empty() {
        return Err(RenderError::Other(
            "rebuild_skinned_geometry: post-rebuild buffers would be empty (no \
             skinned draws to ship)"
                .to_string(),
        ));
    }
    Ok(RepackedSkinnedGeometry {
        vertices,
        indices,
        regions,
        ignored_changes: change_map.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::skinned_draw_object;
    use alloc::vec;

    const STRIDE: usize = size_of::<SkinnedVertex>();

    fn vertex(x: f32) -> SkinnedVertex {
        SkinnedVertex {
            pos: [x, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            color: [1.0; 3],
            uv: [0.0; 2],
            joints: [0; 4],
            weights: [1.0, 0.0, 0.0, 0.0],
        }
    }

    fn vertices(n: usize, first: f32) -> Vec<SkinnedVertex> {
        (0..n).map(|i| vertex(first + i as f32)).collect()
    }

    fn slot(
        vertex_base: u32,
        vertex_count: usize,
        index_offset: usize,
        index_count: usize,
    ) -> SkinnedDrawObject {
        SkinnedDrawObject {
            vertex_base,
            vertex_count,
            index_offset,
            index_count,
            ..skinned_draw_object()
        }
    }

    fn change(slot: u32, vertex_count: usize, indices: Vec<u16>) -> SkinnedDrawGeometryUpdate {
        SkinnedDrawGeometryUpdate {
            skinned_index: SkinnedIndex(slot),
            vertices: vertices(vertex_count, 100.0),
            indices,
        }
    }

    // Two triangles side by side: slot 0 over vertices 0..3, slot 1 over 3..6.
    fn two_slots() -> ([SkinnedDrawObject; 2], Vec<SkinnedVertex>, [u32; 6]) {
        (
            [slot(0, 3, 0, 3), slot(3, 3, 3, 3)],
            vertices(6, 0.0),
            [0, 1, 2, 3, 4, 5],
        )
    }

    fn message(err: RenderError) -> alloc::string::String {
        match err {
            RenderError::Other(m) => m,
            other => format!("{other:?}"),
        }
    }

    #[test]
    fn an_update_lands_at_its_slot_with_rebased_indices() {
        let (objects, _, _) = two_slots();
        let write =
            place_skinned_update(&objects, SkinnedIndex(1), 3, 3, &[2, 1, 0], 6 * STRIDE).unwrap();
        assert_eq!(write.vertex_offset, (3 * STRIDE) as u64);
        assert_eq!(write.index_offset, (3 * size_of::<u32>()) as u64);
        assert_eq!(write.indices, [5, 4, 3]);
    }

    #[test]
    fn an_update_of_a_missing_slot_errors() {
        let (objects, _, _) = two_slots();
        let err = place_skinned_update(&objects, SkinnedIndex(2), 0, 3, &[0, 1, 2], 6 * STRIDE)
            .unwrap_err();
        assert!(message(err).contains("out of range"));
    }

    #[test]
    fn an_update_that_changes_the_index_count_errors() {
        let (objects, _, _) = two_slots();
        let err =
            place_skinned_update(&objects, SkinnedIndex(0), 0, 3, &[0, 1], 6 * STRIDE).unwrap_err();
        assert!(message(err).contains("expects 3 indices, got 2"));
    }

    #[test]
    fn an_update_past_the_vertex_buffer_errors() {
        let (objects, _, _) = two_slots();
        let fits = place_skinned_update(&objects, SkinnedIndex(1), 3, 3, &[0, 1, 2], 6 * STRIDE);
        assert!(fits.is_ok());
        let err = place_skinned_update(&objects, SkinnedIndex(1), 3, 4, &[0, 1, 2], 6 * STRIDE)
            .unwrap_err();
        assert!(message(err).contains("overruns"));
        let err = place_skinned_update(
            &objects,
            SkinnedIndex(1),
            u32::MAX,
            3,
            &[0, 1, 2],
            6 * STRIDE,
        )
        .unwrap_err();
        assert!(message(err).contains("overruns"));
    }

    #[test]
    fn a_grown_slot_shifts_its_successor_and_rebases_it() {
        let (objects, old_v, old_i) = two_slots();
        let grown = change(0, 5, vec![0, 1, 2, 2, 3, 4]);

        let out = repack_skinned_geometry(&objects, &old_v, &old_i, vec![grown]).unwrap();

        assert_eq!(out.vertices.len(), 8);
        assert_eq!(out.indices, [0, 1, 2, 2, 3, 4, 5, 6, 7]);
        assert_eq!(
            out.regions,
            [
                SkinnedRegion {
                    vertex_base: 0,
                    vertex_count: 5,
                    index_offset: 0,
                    index_count: 6
                },
                SkinnedRegion {
                    vertex_base: 5,
                    vertex_count: 3,
                    index_offset: 6,
                    index_count: 3
                },
            ]
        );
        assert_eq!(out.vertices[5].pos[0], 3.0, "the old region is copied");
    }

    #[test]
    fn a_shrunk_slot_pulls_its_successor_back() {
        let (objects, old_v, old_i) = two_slots();
        let shrunk = change(0, 2, vec![0, 1, 1]);
        let out = repack_skinned_geometry(&objects, &old_v, &old_i, vec![shrunk]).unwrap();
        assert_eq!(out.indices, [0, 1, 1, 2, 3, 4]);
        assert_eq!(out.regions[1].vertex_base, 2);
    }

    #[test]
    fn applying_a_repack_points_every_slot_at_its_region() {
        let (mut objects, old_v, old_i) = two_slots();
        let grown = change(1, 4, vec![0, 1, 2, 3]);
        let out = repack_skinned_geometry(&objects, &old_v, &old_i, vec![grown]).unwrap();
        out.apply_to(&mut objects);
        assert_eq!((objects[1].vertex_base, objects[1].vertex_count), (3, 4));
        assert_eq!((objects[1].index_offset, objects[1].index_count), (3, 4));
        let layouts = out.layouts();
        assert_eq!(layouts[1].skinned_index, SkinnedIndex(1));
        assert_eq!(layouts[1].vertex_count, 4);
        assert_eq!(layouts[1].index_count, 4);
    }

    #[test]
    fn a_region_outside_the_old_buffers_errors() {
        let objects = [slot(0, 4, 0, 3)];
        let err = repack_skinned_geometry(&objects, &vertices(3, 0.0), &[0, 1, 2], Vec::new())
            .unwrap_err();
        assert!(message(err).contains("vertex region"));

        let objects = [slot(0, 3, 1, 3)];
        let err = repack_skinned_geometry(&objects, &vertices(3, 0.0), &[0, 1, 2], Vec::new())
            .unwrap_err();
        assert!(message(err).contains("index region"));
    }

    #[test]
    fn an_index_below_its_slot_base_errors() {
        let objects = [slot(0, 3, 0, 3), slot(3, 3, 3, 3)];
        let err =
            repack_skinned_geometry(&objects, &vertices(6, 0.0), &[0, 1, 2, 3, 1, 5], Vec::new())
                .unwrap_err();
        assert!(message(err).contains("stale index 1"));
    }

    #[test]
    fn an_empty_repack_errors() {
        let err = repack_skinned_geometry(&[], &[], &[], Vec::new()).unwrap_err();
        assert!(message(err).contains("empty"));
    }

    #[test]
    fn a_change_past_the_slot_list_is_counted_as_ignored() {
        let (objects, old_v, old_i) = two_slots();
        let stray = change(7, 3, vec![0, 1, 2]);
        let out = repack_skinned_geometry(&objects, &old_v, &old_i, vec![stray]).unwrap();
        assert_eq!(out.ignored_changes, 1);
        assert_eq!(out.indices, old_i);
    }
}
