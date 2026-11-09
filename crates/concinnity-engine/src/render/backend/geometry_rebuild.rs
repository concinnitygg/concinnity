//! Rewriting a built world's geometry underneath the running frame.
//!
//! The asset hot-reload rebuilders: the in-place write a same-size reload
//! makes, and the repack a size-changing one needs. The size queries
//! (`draw_geometry_size`, `draw_lod_index_counts`) exist so the caller can tell
//! whether the cheap in-place write fits before it asks for the rebuild.

use crate::gfx::mesh_payload::{SkinnedVertex, Vertex};
use crate::gfx::render_types::{DrawIndex, SkinnedIndex};
use crate::render::backend::SceneHost;
use crate::render::backend::error::{RenderError, RenderResult};
use crate::render::slots::geometry_repack::{repack_skinned_geometry, repack_static_geometry};
use alloc::vec::Vec;

/// One draw slot's fresh geometry, supplied to
/// [`GeometryRebuild::rebuild_static_geometry`] when an asset hot-reload
/// changed its vertex / index count and the slot can no longer hold the new
/// data in place. The backend rebuilds the entire shared vertex / index
/// buffer; draws not named here keep their current geometry, copied byte-for-
/// byte from the live buffers. `indices` are mesh-relative (0-based); the
/// backend rebases them onto whatever new vertex region the draw lands in.
pub struct DrawGeometryUpdate {
    /// The draw slot whose geometry is replaced.
    pub draw_idx: DrawIndex,
    /// Replacement vertices.
    pub vertices: Vec<Vertex>,
    /// Replacement indices, mesh-relative.
    pub indices: Vec<u16>,
    /// One slice per additional LOD, ordered mip 0 → mip N-1. Each is
    /// `(switch_distance, mesh-relative indices)`. Empty for meshes
    /// declared `lod_levels <= 1`.
    pub lod_alternates: Vec<(f32, Vec<u16>)>,
}

/// One skinned draw slot's fresh geometry, supplied to
/// [`GeometryRebuild::rebuild_skinned_geometry`] when an asset hot-reload
/// changed its vertex / index count and the slot can no longer hold the new
/// data in its existing region of the shared skinned vertex / index buffers.
/// The backend rebuilds both shared buffers; slots not named here keep their
/// current geometry, copied byte-for-byte from the live buffers and re-based
/// onto whatever new vertex region they land in. `indices` are mesh-relative
/// (0-based); the backend rebases them onto the new vertex region.
pub struct SkinnedDrawGeometryUpdate {
    /// The skinned slot whose geometry is replaced.
    pub skinned_index: SkinnedIndex,
    /// Replacement vertices.
    pub vertices: Vec<SkinnedVertex>,
    /// Replacement indices, mesh-relative.
    pub indices: Vec<u16>,
}

/// The post-rebuild layout for one skinned slot, returned by
/// [`GeometryRebuild::rebuild_skinned_geometry`] so the asset hot-reload
/// helper can refresh its `SkinnedMeshSourceEntry`s'
/// `vertex_base` / `vertex_count` / `index_count` to point at the new
/// regions. Returned for every slot (both the ones whose geometry was
/// replaced and the ones whose geometry was carried over) because the
/// rebuild may have shifted every slot's `vertex_base`.
pub struct SkinnedSlotLayout {
    /// The skinned slot this layout describes.
    pub skinned_index: SkinnedIndex,
    /// First vertex of the slot's region in the shared skinned buffer.
    pub vertex_base: u32,
    /// Vertices in the slot's region.
    pub vertex_count: usize,
    /// Indices in the slot's region.
    pub index_count: usize,
}

/// The shared vertex and index buffers a backend draws the static and the
/// skinned meshes from, read back and replaced whole when a size-changing
/// reload repacks them.
///
/// The six methods come in matched read, replace and moved sets, called in
/// that order only by the [`GeometryRebuild`] defaults, so the GPU the read
/// drains stays drained through the replace. Defaulted for a backend without a
/// rebuild path: the reads and replaces report [`RenderError::Unsupported`], so
/// nothing is replaced and no layout is rewritten.
pub trait GeometryBuffers {
    /// Drain the GPU, then read the shared static vertex and index buffers
    /// back.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] on a backend without a rebuild
    /// path, or an error if the readback fails.
    fn read_static_geometry(&mut self) -> RenderResult<(Vec<Vertex>, Vec<u32>)> {
        Err(RenderError::Unsupported {
            op: "rebuild_static_geometry",
        })
    }

    /// Replace the shared static vertex and index buffers with ones holding
    /// `vertices` and `indices`. The GPU is still drained from the read, and
    /// nothing is swapped when the new buffers cannot be made.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] on a backend without a rebuild
    /// path, or an error if the new buffers cannot be allocated or written.
    fn replace_static_geometry(
        &mut self,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> RenderResult<()> {
        let _ = (vertices, indices);
        Err(RenderError::Unsupported {
            op: "rebuild_static_geometry",
        })
    }

    /// Bring what was built over the old static buffers and the old draw
    /// layouts up to date with the new ones, such as a ray-tracing
    /// acceleration structure. A failure is the backend's to log: the
    /// geometry itself is already replaced.
    fn static_geometry_moved(&mut self) {}

    /// Drain the GPU, then read the shared skinned vertex and index buffers
    /// back.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] on a backend without a rebuild
    /// path, or an error if no skinned geometry was uploaded or the readback
    /// fails.
    fn read_skinned_geometry(&mut self) -> RenderResult<(Vec<SkinnedVertex>, Vec<u32>)> {
        Err(RenderError::Unsupported {
            op: "rebuild_skinned_geometry",
        })
    }

    /// Replace the shared skinned vertex and index buffers with ones holding
    /// `vertices` and `indices`, with the usage the skinned passes need. The
    /// GPU is still drained from the read, and nothing is swapped when the
    /// new buffers cannot be made.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] on a backend without a rebuild
    /// path, or an error if the new buffers cannot be allocated or written.
    fn replace_skinned_geometry(
        &mut self,
        vertices: &[SkinnedVertex],
        indices: &[u32],
    ) -> RenderResult<()> {
        let _ = (vertices, indices);
        Err(RenderError::Unsupported {
            op: "rebuild_skinned_geometry",
        })
    }

    /// Bring what binds the skinned buffers up to date with the new ones of
    /// `vertex_count` vertices.
    ///
    /// # Errors
    ///
    /// Returns an error if what binds them cannot be rebuilt.
    fn skinned_geometry_moved(&mut self, vertex_count: usize) -> RenderResult<()> {
        let _ = vertex_count;
        Ok(())
    }
}

/// The geometry an asset hot-reload rewrites in a built world.
///
/// All defaulted, so a backend implements it empty: the size-changing
/// rebuilds are written once here over the backend's [`GeometryBuffers`].
/// The in-place write and the size queries only touch the scene's draw
/// slots, so they are defaulted over [`SceneHost`]: without a
/// scene the queries report `None`, which tells the caller there is nothing to
/// rewrite. The rewrites report [`RenderError::Unsupported`], so a reload is
/// never counted as applied when nothing changed.
pub trait GeometryRebuild: SceneHost + GeometryBuffers {
    /// `(vertex_count, index_count)` for the static draw at `draw_idx`, or
    /// `None` when the index is out of range / the backend does not expose
    /// the field. Used by asset hot-reload to detect size-changing
    /// reloads before attempting [`Self::update_mesh_geometry`], which
    /// rejects size mismatches.
    fn draw_geometry_size(&self, draw_idx: DrawIndex) -> Option<(usize, usize)> {
        self.scene()?.draw_geometry_size(draw_idx)
    }

    /// Per-LOD-alternate index counts for the static draw at `draw_idx`,
    /// ordered from LOD1 upward (LOD0 is reported by
    /// [`Self::draw_geometry_size`]). Returns `None` when the index is out of
    /// range or the backend does not expose its LOD layout. Used by asset
    /// hot-reload alongside [`Self::draw_geometry_size`] to detect
    /// size-changing reloads: a `.glb` that re-exports with a different LOD
    /// breakdown queues the entry for [`Self::rebuild_static_geometry`]
    /// instead of [`Self::update_mesh_geometry`]'s in-place write.
    fn draw_lod_index_counts(&self, draw_idx: DrawIndex) -> Option<Vec<usize>> {
        self.scene()?.draw_lod_index_counts(draw_idx)
    }

    /// Rebuild the shared static-mesh vertex + index buffers, replacing the
    /// geometry of each `DrawGeometryUpdate.draw_idx` with the new
    /// vertices / indices / LOD alternates. Draws not named in `changes`
    /// keep their current geometry, copied byte-for-byte from the live
    /// buffers. The slot's `vertex_count`, `index_count`, and
    /// `lod_alternates` index offsets are rewritten as the new buffers are
    /// laid out. Driven by asset hot-reload (`cn run -f` only) when a
    /// size-changing `.glb` re-export means the existing
    /// [`Self::update_mesh_geometry`] in-place write no longer fits. The
    /// backend drains the GPU and reads its live buffers back through
    /// [`GeometryBuffers`], the repack is shared, and the rebuilt buffers are
    /// swapped in wholesale.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] without a scene or on a backend
    /// without a rebuild path, or an error if the buffers cannot be read back,
    /// a change names no draw, or the new buffers cannot be allocated.
    fn rebuild_static_geometry(&mut self, changes: Vec<DrawGeometryUpdate>) -> RenderResult<()> {
        let unsupported = RenderError::Unsupported {
            op: "rebuild_static_geometry",
        };
        if self.scene().is_none() {
            return Err(unsupported);
        }
        let (old_vertices, old_indices) = self.read_static_geometry()?;
        let Some(scene) = self.scene() else {
            return Err(unsupported);
        };
        let repacked =
            repack_static_geometry(&scene.draw.objects, &old_vertices, &old_indices, changes)?;
        if repacked.ignored_changes > 0 {
            tracing::warn!(
                "rebuild_static_geometry: {} change(s) targeted draw indices not in \
                 draw_objects (ignored)",
                repacked.ignored_changes
            );
        }
        self.replace_static_geometry(&repacked.vertices, &repacked.indices)?;
        if let Some(scene) = self.scene_mut() {
            for (layout, obj) in repacked.layouts.into_iter().zip(&mut scene.draw.objects) {
                layout.apply_to(obj);
            }
            scene.placement.forget_layout();
        }
        self.static_geometry_moved();
        Ok(())
    }

    /// Rebuild the shared skinned-mesh vertex + index buffers, replacing the
    /// geometry of each `SkinnedDrawGeometryUpdate.skinned_index` with the
    /// new vertices / indices. Slots not named in `changes` keep their
    /// current geometry, copied byte-for-byte from the live buffers and
    /// re-based onto the new vertex region they land in. Returns the
    /// post-rebuild layout (one [`SkinnedSlotLayout`] per slot, in
    /// `skinned_index` order) so the caller can refresh its source-map
    /// `vertex_base` / `vertex_count` / `index_count` to point at the new
    /// regions. Driven by asset hot-reload (`cn run -f` only) when a
    /// size-changing `.glb` re-export means the existing
    /// [`SkinnedDraws::update_skinned_mesh_geometry`](super::SkinnedDraws::update_skinned_mesh_geometry) in-place write no longer fits.
    /// The backend drains the GPU and reads its live buffers back through
    /// [`GeometryBuffers`], and the rebuilt buffers are swapped in wholesale.
    /// The skinned pipelines, shadow + velocity + SSAO + SSR variants, and
    /// `skinned_draw_objects` slot metadata (`texture_slot` /
    /// `normal_map_slot` / `material` / `joint_count`) all stay untouched;
    /// only the `index_offset` / `index_count` on each `SkinnedDrawObject`
    /// (and the buffers themselves) move.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] without a scene or on a backend
    /// without a rebuild path, or an error if the buffers cannot be read back,
    /// a change names no slot, or the new buffers cannot be allocated.
    fn rebuild_skinned_geometry(
        &mut self,
        changes: Vec<SkinnedDrawGeometryUpdate>,
    ) -> RenderResult<Vec<SkinnedSlotLayout>> {
        let unsupported = RenderError::Unsupported {
            op: "rebuild_skinned_geometry",
        };
        if self.scene().is_none() {
            return Err(unsupported);
        }
        let (old_vertices, old_indices) = self.read_skinned_geometry()?;
        let Some(scene) = self.scene() else {
            return Err(unsupported);
        };
        let repacked = repack_skinned_geometry(
            &scene.skinned.draw_objects,
            &old_vertices,
            &old_indices,
            changes,
        )?;
        if repacked.ignored_changes > 0 {
            tracing::warn!(
                "rebuild_skinned_geometry: {} change(s) targeted skinned indices not \
                 in skinned_draw_objects (ignored)",
                repacked.ignored_changes
            );
        }
        self.replace_skinned_geometry(&repacked.vertices, &repacked.indices)?;
        if let Some(scene) = self.scene_mut() {
            repacked.apply_to(&mut scene.skinned.draw_objects);
        }
        self.skinned_geometry_moved(repacked.vertices.len())?;
        Ok(repacked.layouts())
    }

    /// Replace a `Mesh` draw slot's vertex + index data in place. Driven by
    /// asset hot-reload (`cn run -f` only). Reuses the slot's existing offset
    /// in the shared vertex / index buffers, so the new geometry must match
    /// the slot's init-time vertex count + index count; a size-changing
    /// reload returns an error so the caller can queue
    /// [`Self::rebuild_static_geometry`] instead, which repacks the shared
    /// buffers. Each entry in
    /// `lod_alternates` (`(switch_distance, mesh-relative indices)`) is
    /// written to the matching slot's pre-allocated LOD index region; the
    /// number of LODs and each LOD's index count must match the slot's
    /// init-time layout, otherwise the call returns an error so the caller
    /// can queue [`Self::rebuild_static_geometry`]. `switch_distance` is
    /// re-stored per LOD so a JSON-side tweak to `lod_distances` propagates
    /// without a process restart.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::Unsupported`] without a scene, or an error if the
    /// slot is unknown or the new geometry or LOD layout does not match it.
    fn update_mesh_geometry(
        &mut self,
        draw_idx: DrawIndex,
        verts: &[Vertex],
        idxs: &[u16],
        lod_alternates: &[(f32, Vec<u16>)],
    ) -> RenderResult<()> {
        self.edit_geometry(&mut |scene, writer| {
            scene.update_mesh_geometry(draw_idx, verts, idxs, lod_alternates, writer)
        })
        .unwrap_or(Err(RenderError::Unsupported {
            op: "update_mesh_geometry",
        }))
    }
}
