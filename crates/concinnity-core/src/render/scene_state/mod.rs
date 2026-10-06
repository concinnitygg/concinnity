//! The CPU-side scene every graphics backend draws from: the draw list, the
//! camera view, the skinned slots, the model-history tracker, the placement of
//! streamed geometry in the shared vertex and index buffers, and the flags that
//! tell the GPU side what to rebuild.
//!
//! None of it is a device object. A backend embeds one [`SceneState`], reads it
//! while recording a frame, and lends it to the
//! [`SceneHost`](crate::render::backend::SceneHost) defaults that apply the
//! engine's per-frame pushes. The only bytes that reach the GPU from here go
//! through a [`GeometryWriter`], which is where the backends differ.

use alloc::vec::Vec;
use core::cell::RefCell;

use crate::gfx::render_types::{DrawObject, clone_reserve, runtime_reserve_full};
use crate::gfx::view_modes::{ShowFlags, ViewMode};
use crate::render::model_history::ModelHistory;
use crate::render::skinned_slots::SkinnedSlots;
use crate::sky::SkyOrientation;
use crate::transform::IDENTITY;

// Per-slot edits of the draw list: transforms, visibility, clones, materials.
mod draws;
// Placement and upload of streamed meshes and chunks in the shared buffers.
mod geometry;

pub use geometry::{GeometryBuffer, GeometryPlacement, GeometryWriter};

/// The scene a backend draws, as the CPU keeps it.
pub struct SceneState {
    /// The draw objects and the record counts that partition the GPU-driven
    /// cull buffers.
    pub draw: DrawList,
    /// The camera and presentation state for the next frame.
    pub view: SceneView,
    /// The skinned draw objects and their per-frame poses.
    pub skinned: SkinnedSlots,
    /// Which cull records hold a usable previous-frame transform. Interior
    /// mutability because a backend records its draw-args build from `&self`.
    pub model_history: RefCell<ModelHistory>,
    /// Where streamed meshes and chunks sit in the shared geometry buffers.
    pub placement: GeometryPlacement,
    /// What the GPU side must rebuild before the next frame reads the scene.
    pub gpu_dirty: GpuDirty,
}

impl SceneState {
    /// A scene over `draw` with an identity view cleared to `clear_color`.
    pub fn new(draw: DrawList, clear_color: [f32; 4]) -> Self {
        Self {
            draw,
            view: SceneView::new(clear_color),
            skinned: SkinnedSlots::new(),
            model_history: RefCell::new(ModelHistory::new()),
            placement: GeometryPlacement::default(),
            gpu_dirty: GpuDirty::default(),
        }
    }
}

/// The draw objects plus the record counts that partition the GPU-driven cull
/// buffers: the build-time objects, then the instance records, then the
/// runtime reserve, then the skinned tail.
pub struct DrawList {
    /// One entry per renderable object, indexed by
    /// [`DrawIndex`](crate::gfx::render_types::DrawIndex).
    pub objects: Vec<DrawObject>,
    /// The build-time object count. Streamed chunks and runtime clones land
    /// past it.
    pub n_objects: usize,
    /// Instanced-cluster instances folded into the cull buffers after the
    /// build-time objects.
    pub n_instances: usize,
    /// Cull records reserved at init for objects that appear after it:
    /// streamed chunks and spawned clones. Zero when the backend folds them
    /// into the live list instead of a fixed reserve.
    pub n_runtime: usize,
    /// Skinned draw objects folded into the cull buffers after the runtime
    /// reserve. Zero until skinned geometry uploads.
    pub n_skinned: usize,
    // Whether a clone is refused once the runtime reserve is full.
    runtime_capped: bool,
}

impl DrawList {
    /// A draw list whose runtime objects ride a fixed reserve of cull records:
    /// the worst-case resident chunk window plus the runtime-clone budget. A
    /// clone past it is refused.
    pub fn with_runtime_reserve(
        objects: Vec<DrawObject>,
        n_instances: usize,
        n_chunk_max: usize,
    ) -> Self {
        let n_objects = objects.len();
        Self {
            n_objects,
            objects,
            n_instances,
            n_runtime: n_chunk_max + clone_reserve(n_objects),
            n_skinned: 0,
            runtime_capped: true,
        }
    }

    /// A draw list whose cull buffers grow with it, so a runtime clone is never
    /// refused for want of a record.
    pub fn unreserved(objects: Vec<DrawObject>, n_instances: usize) -> Self {
        Self {
            n_objects: objects.len(),
            objects,
            n_instances,
            n_runtime: 0,
            n_skinned: 0,
            runtime_capped: false,
        }
    }

    /// Whether another runtime object would overflow the reserve.
    pub fn runtime_reserve_full(&self) -> bool {
        self.runtime_capped && runtime_reserve_full(&self.objects, self.n_objects, self.n_runtime)
    }
}

/// The camera and presentation state a frame is drawn with.
pub struct SceneView {
    /// The color the frame clears to.
    pub clear_color: [f32; 4],
    /// Scene-transition fade to black in `[0, 1]`, applied in the composite
    /// pass so it covers the whole image rather than only the cleared pixels.
    pub scene_fade: f32,
    /// The viewport view mode.
    pub mode: ViewMode,
    /// The feature passes the frame runs.
    pub show: ShowFlags,
    /// The camera near plane, where the composite's depth-channel view starts.
    pub near: f32,
    /// The camera view distance, where the depth-channel view ends when set.
    pub view_distance: Option<f32>,
    /// The camera's view matrix, column-major.
    pub matrix: [[f32; 4]; 4],
    /// Rows of the sky's inverse rotation, uploaded into every uniform block
    /// whose pass samples the environment cubemaps.
    pub sky_rot: [[f32; 4]; 3],
}

impl SceneView {
    /// An identity view with no fade, cleared to `clear_color`.
    pub fn new(clear_color: [f32; 4]) -> Self {
        Self {
            clear_color,
            scene_fade: 0.0,
            mode: ViewMode::default(),
            show: ShowFlags::default(),
            near: 0.05,
            view_distance: None,
            matrix: IDENTITY,
            sky_rot: SkyOrientation::IDENTITY_ROWS,
        }
    }
}

/// GPU-side work the scene's edits have made necessary.
#[derive(Debug, Default)]
pub struct GpuDirty {
    /// The ray-tracing-relevant draw set changed: an object joined or left it,
    /// moved its geometry, or changed what its geometry-table entry reads. The
    /// backend's next acceleration-structure update takes it.
    pub rt_topology: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::draw_object;
    use alloc::vec;

    #[test]
    fn a_reserved_list_partitions_its_records() {
        let draw = DrawList::with_runtime_reserve(vec![draw_object(), draw_object()], 3, 5);
        assert_eq!(draw.n_objects, 2);
        assert_eq!(draw.n_instances, 3);
        assert_eq!(draw.n_runtime, 5 + clone_reserve(2));
        assert_eq!(draw.n_skinned, 0);
    }

    #[test]
    fn an_unreserved_list_is_never_full() {
        let mut draw = DrawList::unreserved(vec![draw_object()], 0);
        draw.objects.push(draw_object());
        draw.objects[1].resident = true;
        assert_eq!(draw.n_runtime, 0);
        assert!(!draw.runtime_reserve_full());
    }

    #[test]
    fn a_reserved_list_fills_with_resident_runtime_objects() {
        let mut draw = DrawList::with_runtime_reserve(vec![], 0, 1);
        assert!(!draw.runtime_reserve_full());
        let mut chunk = draw_object();
        chunk.resident = true;
        draw.objects.push(chunk);
        assert!(draw.runtime_reserve_full());
        draw.objects[0].resident = false;
        assert!(!draw.runtime_reserve_full());
    }

    #[test]
    fn a_new_scene_starts_at_identity_with_no_fade() {
        let scene = SceneState::new(DrawList::unreserved(vec![], 0), [0.1, 0.2, 0.3, 1.0]);
        assert_eq!(scene.view.clear_color, [0.1, 0.2, 0.3, 1.0]);
        assert_eq!(scene.view.matrix, IDENTITY);
        assert_eq!(scene.view.scene_fade, 0.0);
        assert_eq!(scene.view.sky_rot, SkyOrientation::IDENTITY_ROWS);
        assert!(!scene.gpu_dirty.rt_topology);
    }
}
