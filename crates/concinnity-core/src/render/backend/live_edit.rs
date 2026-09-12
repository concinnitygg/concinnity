//! Rewriting a world's GPU content underneath the running frame.
//!
//! The asset hot-reload rebuilders, the per-draw material and cull-distance
//! writes the editor previews with, and the world swap a live reload performs
//! on the retained device. The queries here (`draw_geometry_size`,
//! `draw_lod_index_counts`, `hot_swap_config`) exist so the caller can tell
//! whether the cheap in-place write fits before it asks for the rebuild.

use crate::components::ShaderPrograms;
use crate::gfx::mesh_payload::{SkinnedVertex, Vertex};
use crate::gfx::render_types::MaterialUniforms;
use crate::render::backend_init::{BackendInit, SwapchainConfig};
use crate::render::error::{RenderError, RenderResult};
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;

/// One draw slot's fresh geometry, supplied to
/// [`LiveEdit::rebuild_static_geometry`] when an asset hot-reload
/// changed its vertex / index count and the slot can no longer hold the new
/// data in place. The backend rebuilds the entire shared vertex / index
/// buffer; draws not named here keep their current geometry, copied byte-for-
/// byte from the live buffers. `indices` are mesh-relative (0-based); the
/// backend rebases them onto whatever new vertex region the draw lands in.
pub struct DrawGeometryUpdate {
    /// The draw slot whose geometry is replaced.
    pub draw_idx: usize,
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
/// [`LiveEdit::rebuild_skinned_geometry`] when an asset hot-reload
/// changed its vertex / index count and the slot can no longer hold the new
/// data in its existing region of the shared skinned vertex / index buffers.
/// The backend rebuilds both shared buffers; slots not named here keep their
/// current geometry, copied byte-for-byte from the live buffers and re-based
/// onto whatever new vertex region they land in. `indices` are mesh-relative
/// (0-based); the backend rebases them onto the new vertex region.
pub struct SkinnedDrawGeometryUpdate {
    /// The skinned slot whose geometry is replaced.
    pub skinned_index: usize,
    /// Replacement vertices.
    pub vertices: Vec<SkinnedVertex>,
    /// Replacement indices, mesh-relative.
    pub indices: Vec<u16>,
}

/// The post-rebuild layout for one skinned slot, returned by
/// [`LiveEdit::rebuild_skinned_geometry`] so the asset hot-reload
/// helper can refresh its `SkinnedMeshSourceEntry`s'
/// `vertex_base` / `vertex_count` / `index_count` to point at the new
/// regions. Returned for every slot (both the ones whose geometry was
/// replaced and the ones whose geometry was carried over) because the
/// rebuild may have shifted every slot's `vertex_base`.
/// Constructed only by the `cn debug` binary's skinned-rebuild reload pass;
/// reads as dead under `cargo check --lib`.
pub struct SkinnedSlotLayout {
    /// The skinned slot this layout describes.
    pub skinned_index: usize,
    /// First vertex of the slot's region in the shared skinned buffer.
    pub vertex_base: u32,
    /// Vertices in the slot's region.
    pub vertex_count: usize,
    /// Indices in the slot's region.
    pub index_count: usize,
}

/// The paths an asset hot-reload and the editor's live previews mutate a
/// built world through.
///
/// All defaulted, so a shipped backend that never reloads an asset implements
/// none of it. The defaults are deliberately inert rather than failing: the
/// size queries report `None`, which is what makes the callers above decide
/// there is nothing to rewrite.
pub trait LiveEdit {
    /// Shared atomic flag the backend polls at frame start to trigger a
    /// shader rebuild. `Some` only under `cn debug` on backends that ship
    /// hot-reload; `None` on production runs and on backends that do not. The debug server reads this
    /// to forward `reload-shaders` commands; the filesystem watcher writes
    /// it directly. Default: `None`.
    fn shader_reload_flag(&self) -> Option<alloc::sync::Arc<core::sync::atomic::AtomicBool>> {
        None
    }

    /// Replace the live color-grading LUT with a fresh `size³` RGBA8 payload.
    /// Driven by asset hot-reload (`cn debug` only). Default no-op: backends
    /// that have not implemented the swap leave the LUT bound at whatever
    /// payload was uploaded at init.
    fn update_color_lut(&mut self, size: u32, data: &[u8]) -> Result<(), String> {
        let _ = (size, data);
        Ok(())
    }

    /// `(vertex_count, index_count)` for the static draw at `draw_idx`, or
    /// `None` when the index is out of range / the backend does not expose
    /// the field. Used by asset hot-reload to detect size-changing
    /// reloads before attempting [`Self::update_mesh_geometry`], which
    /// rejects size mismatches. Default returns `None`; backends that
    /// implement the rebuild path also override this.
    fn draw_geometry_size(&self, draw_idx: usize) -> Option<(usize, usize)> {
        let _ = draw_idx;
        None
    }

    /// Per-LOD-alternate index counts for the static draw at `draw_idx`,
    /// ordered from LOD1 upward (LOD0 is reported by
    /// [`Self::draw_geometry_size`]). Returns `None` when the index is out of
    /// range or the backend does not expose its LOD layout. Used by asset
    /// hot-reload alongside [`Self::draw_geometry_size`] to detect
    /// size-changing reloads: a `.glb` that re-exports with a different LOD
    /// breakdown queues the entry for [`Self::rebuild_static_geometry`]
    /// instead of [`Self::update_mesh_geometry`]'s in-place write.
    fn draw_lod_index_counts(&self, draw_idx: usize) -> Option<Vec<usize>> {
        let _ = draw_idx;
        None
    }

    /// Rebuild the shared static-mesh vertex + index buffers, replacing the
    /// geometry of each `DrawGeometryUpdate.draw_idx` with the new
    /// vertices / indices / LOD alternates. Draws not named in `changes`
    /// keep their current geometry, copied byte-for-byte from the live
    /// buffers. The slot's `vertex_count`, `index_count`, and
    /// `lod_alternates` index offsets are rewritten as the new buffers are
    /// laid out. Driven by asset hot-reload (`cn debug` only) when a
    /// size-changing `.glb` re-export means the existing
    /// [`Self::update_mesh_geometry`] in-place write no longer fits.
    /// `wait_idle` first; the rebuild swaps the GPU buffers wholesale.
    /// Default no-op: a backend that has not implemented the rebuild reports
    /// success without rebuilding. Nothing reaches this path there, because the
    /// default [`Self::draw_geometry_size`] returns `None`, so no size change is
    /// ever detected.
    fn rebuild_static_geometry(&mut self, changes: Vec<DrawGeometryUpdate>) -> RenderResult<()> {
        let _ = changes;
        Ok(())
    }

    /// Replace a `SkinnedMesh` draw slot's vertex + index data in place.
    /// Driven by asset hot-reload (`cn debug` only). Reuses the slot's
    /// existing vertex region + index region in the shared skinned vertex /
    /// index buffers (created once by [`SkinnedDraws::upload_skinned`]), so the new
    /// geometry must match the slot's init-time vertex count + index count
    /// and the new skeleton must keep the same joint count; pipelines stay
    /// untouched, only the bytes change. `vertex_base` is the init-time
    /// vertex offset (in vertex units) into the shared buffer; indices are
    /// rebased onto it before writing. Default no-op.
    fn update_skinned_mesh_geometry(
        &mut self,
        skinned_index: usize,
        vertex_base: u32,
        verts: &[SkinnedVertex],
        idxs: &[u16],
    ) -> Result<(), String> {
        let _ = (skinned_index, vertex_base, verts, idxs);
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
    /// regions. Driven by asset hot-reload (`cn debug` only) when a
    /// size-changing `.glb` re-export means the existing
    /// [`Self::update_skinned_mesh_geometry`] in-place write no longer fits.
    /// The backend `wait_idle`s first; the rebuild swaps the GPU buffers
    /// wholesale. The skinned pipelines, shadow + velocity + SSAO + SSR
    /// variants, and `skinned_draw_objects` slot metadata
    /// (`texture_slot` / `normal_map_slot` / `material` / `joint_count`)
    /// all stay untouched; only the `index_offset` / `index_count` on each
    /// `SkinnedDrawObject` (and the buffers themselves) move. Default no-op
    /// (returns an empty layout vec): a backend that has not implemented the
    /// rebuild reports success without rebuilding, and nothing reaches this
    /// path there for the same reason as the static case above.
    fn rebuild_skinned_geometry(
        &mut self,
        changes: Vec<SkinnedDrawGeometryUpdate>,
    ) -> Result<Vec<SkinnedSlotLayout>, String> {
        let _ = changes;
        Ok(Vec::new())
    }

    /// Update a skinned slot's joint count and resize the backend's per-slot
    /// joint-matrix buffers to match. Driven by asset hot-reload (`cn debug`
    /// only) when a re-imported `.glb`'s skeleton has a different joint
    /// count than the slot was initialized with. Shrinking truncates the
    /// per-slot Vec; growing seeds the new entries to identity so the slot
    /// renders undeformed on the next `update_skinned_pose`. The skinned
    /// shaders consume the joints buffer through a pointer (not a fixed-
    /// size array) and use vertex-attribute-encoded joint indices, so no
    /// pipeline or shader rebuild is required for a joint-count change;
    /// only the CPU-side per-slot buffer and `SkinnedDrawObject.joint_count`
    /// change. Default no-op: backends that have not implemented the resize
    /// leave the skeleton-shape change logged + skipped at the caller.
    fn update_skinned_skeleton(
        &mut self,
        skinned_index: usize,
        new_joint_count: usize,
    ) -> Result<(), String> {
        let _ = (skinned_index, new_joint_count);
        Ok(())
    }

    /// Replace a `Mesh` draw slot's vertex + index data in place. Driven by
    /// asset hot-reload (`cn debug` only). Reuses the slot's existing offset
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
    /// without a process restart. Default no-op.
    fn update_mesh_geometry(
        &mut self,
        draw_idx: usize,
        verts: &[Vertex],
        idxs: &[u16],
        lod_alternates: &[(f32, Vec<u16>)],
    ) -> Result<(), String> {
        let _ = (draw_idx, verts, idxs, lod_alternates);
        Ok(())
    }

    /// Replace the live IBL environment map with a freshly precomputed payload.
    /// `payload` is the serialized byte format emitted by
    /// `crate::bake::environment_map::compile_environment_map_payload`
    /// (header + irradiance cube + prefilter mip chain), so init and hot-reload
    /// share a single byte format. Driven by asset hot-reload (`cn debug`
    /// only). Default no-op: backends that have not implemented the swap leave
    /// the IBL cubes bound at whatever payload was uploaded at init.
    fn update_environment_map(&mut self, payload: &[u8]) -> RenderResult<()> {
        let _ = payload;
        Ok(())
    }

    /// Rewrite a draw slot's material parameters + texture/normal-map pool
    /// indices in place. Driven by the editor's live draw seam when a Prop edits
    /// its `material` arg. Default no-op; a backend that implements it reports
    /// [`DeviceCapabilities::rewrites_draws`], which is what the caller gates on
    /// rather than pushing an edit that would not land.
    fn set_draw_material(
        &mut self,
        draw_idx: usize,
        material: MaterialUniforms,
        texture_slot: usize,
        normal_map_slot: usize,
    ) {
        let _ = (draw_idx, material, texture_slot, normal_map_slot);
    }

    /// Rewrite a draw slot's `cull_distance` in place. Driven by the editor's
    /// live draw seam when a Prop edits its `cull_distance` arg. Default no-op,
    /// gated by the same [`DeviceCapabilities::rewrites_draws`] flag.
    fn set_draw_cull_distance(&mut self, draw_idx: usize, cull_distance: f32) {
        let _ = (draw_idx, cull_distance);
    }

    /// Rebuild the live world-default pipelines (main, instanced, skinned)
    /// from a freshly compiled Shader payload. Driven by asset hot-reload
    /// (`cn debug` only) when one of the Shader's files is saved or a debug-WS
    /// `reload-assets` command fires. The backend builds every replacement
    /// into a temporary first and only swaps when every build succeeds, so a
    /// compile error never overwrites a live pipeline with a half-built
    /// replacement. Default `Err`: a backend without an implementation leaves
    /// the reload logged and skipped at the caller.
    fn update_world_shader_pipelines(&mut self, programs: &ShaderPrograms) -> Result<(), String> {
        let _ = programs;
        Err("update_world_shader_pipelines: not implemented on this backend".to_string())
    }

    /// The swapchain-level configuration this live backend can hot-swap a world
    /// onto, or `None` when the backend cannot reload a world in place (it must
    /// be fully rebuilt instead). Read by GraphicsSystem when a transplanted
    /// backend is handed a new world (the `cn editor` live SAVE): the swap reuses
    /// the backend via [`Self::reload_world`] only when this equals the new
    /// world's `BackendInit::swapchain_config`; a `None` or a mismatch routes to a
    /// full rebuild (recreating the window). Default `None`: DirectX / Vulkan
    /// (and any backend without a real `reload_world`) always rebuild.
    fn hot_swap_config(&self) -> Option<SwapchainConfig> {
        None
    }

    /// Re-upload a new world's GPU content onto this already-constructed backend,
    /// reusing the live device + window + swapchain instead of building a new one.
    /// Driven by the `cn editor` live SAVE: after a structural edit recompiles the
    /// blobs, GraphicsSystem transplants the running backend into the rebuilt
    /// world and calls this so the edit applies without recreating the OS window
    /// or re-initializing the GPU device. The backend waits for the GPU to idle,
    /// drops the old world's content resources, and rebuilds them from `init` on
    /// the retained hardware. Only ever called when [`Self::hot_swap_config`]
    /// reported a config matching `init.swapchain_config()`, so the swapchain
    /// (pixel format / frames-in-flight / EDR) is guaranteed unchanged. Default
    /// `Err`/unsupported: DirectX / Vulkan fall back to a full rebuild (no
    /// regression; a real implementation is Windows-pending like the rest).
    fn reload_world(&mut self, init: BackendInit<'_>) -> RenderResult<()> {
        let _ = init;
        Err(RenderError::Other(
            "reload_world: not supported on this backend".to_string(),
        ))
    }
}
