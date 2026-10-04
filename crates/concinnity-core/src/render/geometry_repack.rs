//! The CPU half of a geometry rebuild: repacking the shared vertex and index
//! buffers when a reloaded mesh no longer fits its draw slot. The skinned
//! buffers have their own repack, [`repack_skinned_geometry`], and the bounds of
//! a size-matched in-place write, [`place_skinned_update`].
//!
//! Every draw keeps its index convention. A draw with `base_vertex == 0` holds
//! absolute indices, which are rebased onto its new vertex region; any other
//! draw holds mesh-relative indices, which are kept and given a new
//! `base_vertex` instead.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::ToString;
use alloc::vec::Vec;
use core::mem::size_of;
use core::ops::Range;

use crate::gfx::mesh_payload::Vertex;
use crate::gfx::render_types::{DrawIndex, DrawObject, LodSlice};
use crate::render::backend::DrawGeometryUpdate;
use crate::render::error::{RenderError, RenderResult};

mod skinned;

pub use skinned::{
    RepackedSkinnedGeometry, SkinnedRegion, SkinnedSlotWrite, place_skinned_update,
    repack_skinned_geometry,
};

/// Where one draw's geometry lands in the repacked buffers.
#[derive(Clone, Debug)]
pub struct DrawLayout {
    /// Byte offset into the repacked vertex buffer.
    pub vertex_offset: usize,
    /// Vertex count of the draw's region.
    pub vertex_count: usize,
    /// Element offset into the repacked index buffer.
    pub index_offset: usize,
    /// Index count of the draw's LOD0 slice.
    pub index_count: usize,
    /// Value added to every index at vertex fetch; 0 for absolute indices.
    pub base_vertex: i32,
    /// The draw's LOD slices, rebased into the repacked index buffer.
    pub lod_alternates: Vec<LodSlice>,
}

impl DrawLayout {
    /// Point `obj` at its repacked geometry.
    pub fn apply_to(self, obj: &mut DrawObject) {
        obj.vertex_offset = self.vertex_offset;
        obj.vertex_count = self.vertex_count;
        obj.index_offset = self.index_offset;
        obj.index_count = self.index_count;
        obj.base_vertex = self.base_vertex;
        obj.lod_alternates = self.lod_alternates;
    }
}

/// The repacked buffer contents and each draw's layout in them.
#[derive(Debug)]
pub struct RepackedGeometry {
    /// The new shared vertex buffer contents.
    pub vertices: Vec<Vertex>,
    /// The new shared index buffer contents.
    pub indices: Vec<u32>,
    /// One layout per draw, in draw order.
    pub layouts: Vec<DrawLayout>,
    /// Changes naming a draw index past the draw list, which were skipped.
    pub ignored_changes: usize,
}

/// Repack the shared static vertex and index buffers, replacing the geometry
/// of each draw named in `changes` and copying every other draw from the old
/// buffers. Errors when a draw's recorded region lies outside the old buffers,
/// or when the result would leave either buffer empty.
pub fn repack_static_geometry(
    objects: &[DrawObject],
    old_vertices: &[Vertex],
    old_indices: &[u32],
    changes: Vec<DrawGeometryUpdate>,
) -> RenderResult<RepackedGeometry> {
    let mut change_map: BTreeMap<DrawIndex, DrawGeometryUpdate> =
        changes.into_iter().map(|c| (c.draw_idx, c)).collect();
    let mut packer = Packer::default();
    let mut layouts = Vec::with_capacity(objects.len());
    for (i, obj) in objects.iter().enumerate() {
        let draw_idx = DrawIndex::from_usize(i);
        let layout = match change_map.remove(&draw_idx) {
            Some(change) => packer.push_change(obj, &change),
            None => packer.copy_draw(draw_idx, obj, old_vertices, old_indices)?,
        };
        layouts.push(layout);
    }
    if packer.vertices.is_empty() || packer.indices.is_empty() {
        return Err(RenderError::Other(
            "rebuild_static_geometry: post-rebuild buffers would be empty (no \
             static draws to ship)"
                .to_string(),
        ));
    }
    Ok(RepackedGeometry {
        vertices: packer.vertices,
        indices: packer.indices,
        layouts,
        ignored_changes: change_map.len(),
    })
}

// Where the draw being appended starts, and how its indices are addressed.
struct Placement {
    vertex_offset: usize,
    index_offset: usize,
    base: u32,
    absolute: bool,
}

impl Placement {
    fn layout(&self, vertex_count: usize, index_count: usize, lods: Vec<LodSlice>) -> DrawLayout {
        DrawLayout {
            vertex_offset: self.vertex_offset,
            vertex_count,
            index_offset: self.index_offset,
            index_count,
            base_vertex: if self.absolute { 0 } else { self.base as i32 },
            lod_alternates: lods,
        }
    }
}

#[derive(Default)]
struct Packer {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

impl Packer {
    fn place(&self, obj: &DrawObject) -> Placement {
        Placement {
            vertex_offset: self.vertices.len() * size_of::<Vertex>(),
            index_offset: self.indices.len(),
            base: self.vertices.len() as u32,
            absolute: obj.base_vertex == 0,
        }
    }

    fn push_change(&mut self, obj: &DrawObject, change: &DrawGeometryUpdate) -> DrawLayout {
        let place = self.place(obj);
        self.vertices.extend_from_slice(&change.vertices);
        self.push_mesh_relative(&change.indices, &place);
        let lods = change
            .lod_alternates
            .iter()
            .map(|(switch_distance, alt)| {
                let index_offset = self.indices.len();
                self.push_mesh_relative(alt, &place);
                LodSlice {
                    index_offset,
                    index_count: alt.len(),
                    switch_distance: *switch_distance,
                }
            })
            .collect();
        place.layout(change.vertices.len(), change.indices.len(), lods)
    }

    fn copy_draw(
        &mut self,
        draw_idx: DrawIndex,
        obj: &DrawObject,
        old_vertices: &[Vertex],
        old_indices: &[u32],
    ) -> RenderResult<DrawLayout> {
        let place = self.place(obj);
        let v_start = obj.vertex_offset / size_of::<Vertex>();
        let src = region(
            old_vertices,
            v_start..v_start + obj.vertex_count,
            draw_idx,
            "vertex region",
            "vertices",
        )?;
        self.vertices.extend_from_slice(src);
        let old_base = if place.absolute {
            v_start as u32
        } else {
            obj.base_vertex as u32
        };
        let range = obj.index_offset..obj.index_offset + obj.index_count;
        let src = region(old_indices, range, draw_idx, "index region", "indices")?;
        self.push_rebased(src, old_base, &place);
        let mut lods = Vec::with_capacity(obj.lod_alternates.len());
        for slice in &obj.lod_alternates {
            let range = slice.index_offset..slice.index_offset + slice.index_count;
            let src = region(old_indices, range, draw_idx, "LOD slice", "indices")?;
            let index_offset = self.indices.len();
            self.push_rebased(src, old_base, &place);
            lods.push(LodSlice {
                index_offset,
                index_count: slice.index_count,
                switch_distance: slice.switch_distance,
            });
        }
        Ok(place.layout(obj.vertex_count, obj.index_count, lods))
    }

    fn push_mesh_relative(&mut self, src: &[u16], place: &Placement) {
        let offset = if place.absolute { place.base } else { 0 };
        self.indices
            .extend(src.iter().map(|&i| u32::from(i) + offset));
    }

    fn push_rebased(&mut self, src: &[u32], old_base: u32, place: &Placement) {
        if place.absolute {
            self.indices
                .extend(src.iter().map(|&i| i.wrapping_sub(old_base) + place.base));
        } else {
            self.indices.extend_from_slice(src);
        }
    }
}

fn region<'a, T>(
    buf: &'a [T],
    range: Range<usize>,
    draw_idx: DrawIndex,
    what: &str,
    unit: &str,
) -> RenderResult<&'a [T]> {
    buf.get(range.clone()).ok_or_else(|| {
        RenderError::Other(format!(
            "rebuild_static_geometry: draw {draw_idx} {what} [{}, {}) out of bounds \
             (buffer has {} {unit})",
            range.start,
            range.end,
            buf.len()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::draw_object;
    use alloc::vec;

    const STRIDE: usize = size_of::<Vertex>();

    fn vertex(x: f32) -> Vertex {
        Vertex {
            pos: [x, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            color: [1.0; 3],
            uv: [0.0; 2],
        }
    }

    fn vertices(n: usize, first: f32) -> Vec<Vertex> {
        (0..n).map(|i| vertex(first + i as f32)).collect()
    }

    // A draw over `vertex_count` vertices at `first_vertex` and `index_count`
    // indices at `index_offset`, absolute when `base_vertex` is 0.
    fn draw(
        first_vertex: usize,
        vertex_count: usize,
        index_offset: usize,
        index_count: usize,
        base_vertex: i32,
    ) -> DrawObject {
        DrawObject {
            vertex_offset: first_vertex * STRIDE,
            vertex_count,
            index_offset,
            index_count,
            base_vertex,
            ..draw_object()
        }
    }

    fn change(draw_idx: u32, vertex_count: usize, indices: Vec<u16>) -> DrawGeometryUpdate {
        DrawGeometryUpdate {
            draw_idx: DrawIndex(draw_idx),
            vertices: vertices(vertex_count, 100.0),
            indices,
            lod_alternates: Vec::new(),
        }
    }

    #[test]
    fn unchanged_absolute_draw_is_rebased_after_a_grown_predecessor() {
        let objects = [draw(0, 3, 0, 3, 0), draw(3, 3, 3, 3, 0)];
        let old_v = vertices(6, 0.0);
        let old_i = [0, 1, 2, 3, 4, 5];
        let grown = change(0, 5, vec![0, 1, 2, 2, 3, 4]);

        let out = repack_static_geometry(&objects, &old_v, &old_i, vec![grown]).unwrap();

        assert_eq!(out.vertices.len(), 8);
        assert_eq!(out.indices, [0, 1, 2, 2, 3, 4, 5, 6, 7]);
        let second = &out.layouts[1];
        assert_eq!(second.vertex_offset, 5 * STRIDE);
        assert_eq!(second.index_offset, 6);
        assert_eq!(second.base_vertex, 0);
        assert_eq!(out.vertices[5].pos[0], 3.0, "the old region is copied");
    }

    #[test]
    fn mesh_relative_draw_keeps_its_indices_and_gets_the_new_base() {
        let objects = [draw(0, 2, 0, 3, 0), draw(2, 3, 3, 3, 2)];
        let old_v = vertices(5, 0.0);
        let old_i = [0, 1, 0, 0, 1, 2];
        let grown = change(0, 4, vec![0, 1, 2, 3]);

        let out = repack_static_geometry(&objects, &old_v, &old_i, vec![grown]).unwrap();

        let chunk = &out.layouts[1];
        assert_eq!(chunk.base_vertex, 4);
        assert_eq!(chunk.vertex_offset, 4 * STRIDE);
        assert_eq!(&out.indices[chunk.index_offset..], [0, 1, 2]);
    }

    #[test]
    fn lod_alternates_are_rebased() {
        let mut far = draw(2, 3, 3, 3, 0);
        far.lod_alternates = vec![LodSlice {
            index_offset: 6,
            index_count: 3,
            switch_distance: 40.0,
        }];
        let objects = [draw(0, 2, 0, 3, 0), far];
        let old_v = vertices(5, 0.0);
        let old_i = [0, 1, 0, 2, 3, 4, 4, 3, 2];
        let mut grown = change(0, 3, vec![0, 1, 2]);
        grown.lod_alternates = vec![(20.0, vec![2, 1])];

        let out = repack_static_geometry(&objects, &old_v, &old_i, vec![grown]).unwrap();

        let changed = &out.layouts[0].lod_alternates[0];
        assert_eq!((changed.index_offset, changed.index_count), (3, 2));
        assert_eq!(&out.indices[3..5], [2, 1]);
        let copied = &out.layouts[1].lod_alternates[0];
        assert_eq!((copied.index_offset, copied.index_count), (8, 3));
        assert_eq!(copied.switch_distance, 40.0);
        assert_eq!(&out.indices[8..11], [5, 4, 3]);
    }

    #[test]
    fn out_of_range_region_errors() {
        let objects = [draw(0, 4, 0, 3, 0)];
        let err = repack_static_geometry(&objects, &vertices(3, 0.0), &[0, 1, 2], Vec::new())
            .unwrap_err();
        assert!(matches!(err, RenderError::Other(m) if m.contains("vertex region")));

        let objects = [draw(0, 3, 1, 3, 0)];
        let err = repack_static_geometry(&objects, &vertices(3, 0.0), &[0, 1, 2], Vec::new())
            .unwrap_err();
        assert!(matches!(err, RenderError::Other(m) if m.contains("index region")));
    }

    #[test]
    fn empty_result_errors() {
        let err = repack_static_geometry(&[], &[], &[], Vec::new()).unwrap_err();
        assert!(matches!(err, RenderError::Other(m) if m.contains("empty")));
    }

    #[test]
    fn change_past_the_draw_list_is_counted_as_ignored() {
        let objects = [draw(0, 3, 0, 3, 0)];
        let stray = change(5, 3, vec![0, 1, 2]);
        let out =
            repack_static_geometry(&objects, &vertices(3, 0.0), &[0, 1, 2], vec![stray]).unwrap();
        assert_eq!(out.ignored_changes, 1);
        assert_eq!(out.indices, [0, 1, 2]);
    }
}
