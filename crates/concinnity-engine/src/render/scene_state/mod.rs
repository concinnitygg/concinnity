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

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::cell::RefCell;

use crate::gfx::render_types::{DrawIndex, DrawObject, clone_reserve, runtime_reserve_full};
use crate::gfx::view_modes::{ShowFlags, ViewMode};
use crate::planet::{HorizonOccluder, TileMorphFrame};
use crate::render::backend::PlanetChunk;
use crate::render::environment::grass::{CanopyField, DrawnPlanetTile, GrassLodView};
use crate::render::frame::plan::FramePlanner;
use crate::render::slots::skinned_slots::SkinnedSlots;
use crate::render::temporal::model_history::ModelHistory;
use crate::render::uniforms::canopy::CanopyParams;
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
    /// The streamed chunks that draw a planet's tiles.
    pub planet_chunks: BTreeMap<DrawIndex, PlanetChunk>,
    /// A planet world's lowest ground, which hides the draws wholly behind
    /// its horizon.
    pub horizon_occluder: Option<HorizonOccluder>,
    /// The camera a planet's tiles morph by, which their vertices
    /// morph by; `None` for a world without one.
    pub tile_morph: Option<TileMorphFrame>,
    /// Plans each frame the backend executes.
    pub planner: FramePlanner,
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
            planet_chunks: BTreeMap::new(),
            horizon_occluder: None,
            tile_morph: None,
            planner: FramePlanner::default(),
        }
    }

    /// Set where planet tile `draw` meets tiles of other levels. A no-op for
    /// a draw that is not a planet tile.
    pub fn set_planet_tile_seams(&mut self, draw: DrawIndex, seams: crate::planet::TileSeams) {
        if let Some(tile) = self
            .draw
            .objects
            .get_mut(draw.index())
            .and_then(|o| o.planet_tile.as_mut())
        {
            tile.seams = seams;
        }
    }

    /// The planet tiles drawn this frame into `out`, replacing what it held:
    /// every planet chunk shown and resident, with the model and first vertex
    /// its draw renders with.
    pub fn drawn_planet_tiles_into(&self, out: &mut Vec<DrawnPlanetTile>) {
        out.clear();
        out.extend(self.drawn_planet_tiles_iter());
    }

    /// This frame's canopy block for `canopy`, laid out in the frame the
    /// planet's tiles are drawn in this frame.
    pub fn canopy_params(
        &self,
        canopy: &CanopyField,
        hand_off: (&GrassLodView, f32),
    ) -> CanopyParams {
        canopy.params(hand_off, self.drawn_planet_tiles_iter().next().as_ref())
    }

    fn drawn_planet_tiles_iter(&self) -> impl Iterator<Item = DrawnPlanetTile> + '_ {
        self.planet_chunks.iter().filter_map(|(&draw, chunk)| {
            let obj = self.draw.objects.get(draw.index())?;
            (obj.visible && obj.resident).then_some(DrawnPlanetTile {
                id: chunk.tile,
                model: obj.model,
                vertex_base: obj.base_vertex.cast_unsigned(),
                radial: chunk.radial,
                origin_altitude: chunk.origin_altitude,
                morph_extent: chunk.morph_extent,
                morph: self
                    .tile_morph
                    .map(|m| m.row(obj.planet_tile, obj.base_vertex.cast_unsigned()))
                    .unwrap_or_default(),
            })
        })
    }
}

/// The draw objects plus the record counts that partition the GPU-driven cull
/// buffers: the build-time objects, then the instance records, then the
/// runtime reserve, then the skinned tail.
pub struct DrawList {
    /// One entry per renderable object, indexed by
    /// [`DrawIndex`].
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

    /// Cull records across the build-time objects, the instances, the runtime
    /// reserve and the skinned tail.
    pub fn record_count(&self) -> usize {
        self.n_objects + self.n_instances + self.n_runtime + self.n_skinned
    }

    /// Record index of the runtime reserve's first slot, past the build-time
    /// objects and the instances.
    pub fn runtime_record_base(&self) -> usize {
        self.n_objects + self.n_instances
    }

    /// Record index of the first skinned record, past the runtime reserve. The
    /// records before it share one indirect draw, the skinned tail another.
    pub fn skinned_record_base(&self) -> usize {
        self.runtime_record_base() + self.n_runtime
    }

    /// Call `emit` with each resident runtime object's reserve slot, its draw
    /// index and the object, packing them into the front of the reserve.
    /// Returns how many were emitted, so the caller can disable the rest.
    ///
    /// A retired chunk or clone leaves a gap in the objects past `n_objects`;
    /// skipping gaps keeps a live object from being pushed past the reserve.
    pub fn for_each_runtime_record<F>(&self, mut emit: F) -> usize
    where
        F: FnMut(usize, usize, &DrawObject),
    {
        let mut slot = 0;
        for (index, object) in self.objects.iter().enumerate().skip(self.n_objects) {
            if slot >= self.n_runtime {
                break;
            }
            if object.resident {
                emit(slot, index, object);
                slot += 1;
            }
        }
        slot
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
    /// The world's sun intensity, which stands for full sunlight when the
    /// scene's luminance is read in absolute units.
    pub sunlight: f32,
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
            sunlight: 1.0,
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
    use alloc::vec::Vec;

    #[test]
    fn a_reserved_list_partitions_its_records() {
        let draw = DrawList::with_runtime_reserve(vec![draw_object(), draw_object()], 3, 5);
        assert_eq!(draw.n_objects, 2);
        assert_eq!(draw.n_instances, 3);
        assert_eq!(draw.n_runtime, 5 + clone_reserve(2));
        assert_eq!(draw.n_skinned, 0);
    }

    #[test]
    fn records_lay_out_objects_instances_reserve_then_skinned() {
        let mut draw = DrawList::with_runtime_reserve(vec![draw_object(), draw_object()], 3, 5);
        draw.n_skinned = 4;
        let reserve = draw.n_runtime;
        assert_eq!(draw.runtime_record_base(), 5);
        assert_eq!(draw.skinned_record_base(), 5 + reserve);
        assert_eq!(draw.record_count(), 5 + reserve + 4);
    }

    #[test]
    fn runtime_records_pack_residents_past_gaps() {
        let mut draw = DrawList::with_runtime_reserve(vec![draw_object()], 0, 2);
        for resident in [true, false, true] {
            let mut object = draw_object();
            object.resident = resident;
            draw.objects.push(object);
        }
        let mut seen = Vec::new();
        let emitted = draw.for_each_runtime_record(|slot, index, _| seen.push((slot, index)));
        assert_eq!(emitted, 2);
        assert_eq!(seen, vec![(0, 1), (1, 3)]);
    }

    #[test]
    fn an_unreserved_list_has_no_runtime_records() {
        let mut draw = DrawList::unreserved(vec![draw_object()], 0);
        let mut object = draw_object();
        object.resident = true;
        draw.objects.push(object);
        assert_eq!(draw.for_each_runtime_record(|_, _, _| {}), 0);
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
