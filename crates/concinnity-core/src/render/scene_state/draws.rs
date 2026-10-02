//! Per-slot edits of the draw list and the view: what the engine pushes each
//! frame, and the runtime spawns and live edits that rewrite a slot in place.

use alloc::format;
use alloc::vec::Vec;

use super::SceneState;
use crate::gfx::render_types::{DrawIndex, DrawObject, MaterialUniforms, SkinnedIndex};
use crate::render::draw_slot::{self, SlotAlloc};
use crate::render::error::{RenderError, RenderResult};

impl SceneState {
    /// Replace the camera's view matrix for the next frame.
    pub fn update_view(&mut self, matrix: [[f32; 4]; 4]) {
        self.view.matrix = matrix;
    }

    /// Set the scene-transition fade for the next frame, clamped to `[0, 1]`.
    pub fn set_fade(&mut self, fade: f32) {
        self.view.scene_fade = fade.clamp(0.0, 1.0);
    }

    /// Write the model matrices of the given draw slots, in order. An
    /// out-of-range slot is ignored.
    pub fn update_models(&mut self, updates: &[(DrawIndex, [[f32; 4]; 4])]) {
        for &(index, model) in updates {
            if let Some(obj) = self.draw.objects.get_mut(index.index()) {
                obj.model = model;
            }
        }
    }

    /// Show or hide one draw slot in every raster pass. An out-of-range slot is
    /// ignored.
    pub fn update_visibility(&mut self, index: DrawIndex, visible: bool) {
        if let Some(obj) = self.draw.objects.get_mut(index.index()) {
            obj.visible = visible;
        }
    }

    /// Hide a slot from every raster pass and drop it from the ray-tracing
    /// draw set, leaving its geometry where it is for the slot allocator to
    /// recycle. An out-of-range slot is ignored.
    pub fn retire_draw_object(&mut self, index: DrawIndex) {
        if let Some(obj) = self.draw.objects.get_mut(index.index()) {
            obj.visible = false;
            obj.resident = false;
            self.gpu_dirty.rt_topology = true;
        }
    }

    /// Write `obj` at the slot the engine's allocator chose, and mark the slot's
    /// model history as belonging to a new occupant.
    pub fn place_draw_object(&mut self, obj: DrawObject, dst: SlotAlloc) -> DrawIndex {
        let slot = draw_slot::place_draw_object(&mut self.draw.objects, obj, dst);
        self.model_history.get_mut().reoccupy_draw(slot);
        slot
    }

    /// Copy the draw at `src` to `dst` at a new transform. The copy shares the
    /// source's geometry, LOD slices, textures, material and cull distance, and
    /// carries a sentinel AABB so the cull draws it every frame. `Err` when the
    /// source is out of range or the runtime reserve is full.
    pub fn clone_static_draw_object(
        &mut self,
        src: DrawIndex,
        model: [[f32; 4]; 4],
        dst: SlotAlloc,
    ) -> RenderResult<()> {
        if self.draw.runtime_reserve_full() {
            return Err(RenderError::Other(format!(
                "clone_static_draw_object: the runtime draw reserve ({}) is full",
                self.draw.n_runtime
            )));
        }
        let source = self.draw.objects.get(src.index()).ok_or_else(|| {
            RenderError::Other(format!(
                "clone_static_draw_object: src draw {src} out of range"
            ))
        })?;
        let obj = DrawObject {
            vertex_offset: source.vertex_offset,
            vertex_count: source.vertex_count,
            index_offset: source.index_offset,
            index_count: source.index_count,
            base_vertex: source.base_vertex,
            geometry_generation: source.geometry_generation,
            model,
            texture_slot: source.texture_slot,
            normal_map_slot: source.normal_map_slot,
            material: source.material,
            shader_bucket: source.shader_bucket,
            visible: true,
            resident: true,
            bb_min: [f32::NAN; 3],
            bb_max: [f32::NAN; 3],
            cull_distance: source.cull_distance,
            lod_alternates: source.lod_alternates.clone(),
        };
        self.place_draw_object(obj, dst);
        self.gpu_dirty.rt_topology = true;
        Ok(())
    }

    /// Rewrite a slot's material and texture-pool slots in place. A material
    /// can change the slot's ray-tracing participation, so the geometry table
    /// is flagged for a refresh. An out-of-range slot is ignored.
    pub fn set_draw_material(
        &mut self,
        index: DrawIndex,
        material: MaterialUniforms,
        texture_slot: usize,
        normal_map_slot: usize,
    ) {
        if let Some(obj) = self.draw.objects.get_mut(index.index()) {
            obj.material = material;
            obj.texture_slot = texture_slot;
            obj.normal_map_slot = normal_map_slot;
            self.gpu_dirty.rt_topology = true;
        }
    }

    /// Rewrite a slot's cull distance in place, clamped to non-negative. An
    /// out-of-range slot is ignored.
    pub fn set_draw_cull_distance(&mut self, index: DrawIndex, cull_distance: f32) {
        if let Some(obj) = self.draw.objects.get_mut(index.index()) {
            obj.cull_distance = cull_distance.max(0.0);
        }
    }

    /// Replace a streamed chunk's placement matrix.
    pub fn set_chunk_model(&mut self, index: DrawIndex, model: [[f32; 4]; 4]) -> RenderResult<()> {
        draw_slot::set_chunk_model(&mut self.draw.objects, index, model).map_err(RenderError::Other)
    }

    /// A slot's `(vertex_count, index_count)`, or `None` out of range.
    pub fn draw_geometry_size(&self, index: DrawIndex) -> Option<(usize, usize)> {
        self.draw
            .objects
            .get(index.index())
            .map(|o| (o.vertex_count, o.index_count))
    }

    /// A slot's per-LOD index counts past LOD0, or `None` out of range.
    pub fn draw_lod_index_counts(&self, index: DrawIndex) -> Option<Vec<usize>> {
        self.draw
            .objects
            .get(index.index())
            .map(|o| o.lod_alternates.iter().map(|s| s.index_count).collect())
    }

    /// Reveal a pre-reserved skinned instance at `model`; see
    /// [`SkinnedSlots::reveal`](crate::render::skinned_slots::SkinnedSlots::reveal).
    pub fn reveal_skinned_instance(&mut self, index: SkinnedIndex, model: [[f32; 4]; 4]) {
        self.skinned
            .reveal(index, model, self.model_history.get_mut());
    }

    /// Resize a skinned slot's palette to a reloaded skeleton; see
    /// [`SkinnedSlots::update_skeleton`](crate::render::skinned_slots::SkinnedSlots::update_skeleton).
    pub fn update_skinned_skeleton(
        &mut self,
        index: SkinnedIndex,
        new_joint_count: usize,
    ) -> RenderResult<()> {
        self.skinned
            .update_skeleton(index, new_joint_count)
            .map_err(RenderError::Other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::render_types::LodSlice;
    use crate::render::model_history::HistoryMode;
    use crate::render::scene_state::DrawList;
    use crate::test_support::draw_object;
    use alloc::string::ToString;
    use alloc::vec;

    const MOVED: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [3.0, 4.0, 5.0, 1.0],
    ];

    fn scene(n: usize) -> SceneState {
        let objects = (0..n).map(|_| draw_object()).collect();
        SceneState::new(DrawList::unreserved(objects, 0), [0.0; 4])
    }

    // One tracked draw-args build that observes every draw slot's occupant.
    fn observe(scene: &mut SceneState) {
        let n = scene.draw.objects.len();
        let history = scene.model_history.get_mut();
        history.begin(HistoryMode::Track, n);
        (0..n).for_each(|i| {
            history.draw_flags(i, i);
        });
    }

    // Whether the next tracked build reports `slot` as holding a new occupant.
    fn reoccupied(scene: &mut SceneState, slot: usize) -> bool {
        let n = scene.draw.objects.len();
        let history = scene.model_history.get_mut();
        history.begin(HistoryMode::Track, n);
        history.draw_flags(slot, slot) != 0
    }

    #[test]
    fn update_models_writes_only_the_listed_slots_and_skips_out_of_range() {
        let mut s = scene(2);
        s.update_models(&[(DrawIndex(1), MOVED), (DrawIndex(9), MOVED)]);
        assert_ne!(s.draw.objects[0].model, MOVED);
        assert_eq!(s.draw.objects[1].model, MOVED);
    }

    #[test]
    fn update_view_and_fade_land_in_the_view() {
        let mut s = scene(0);
        s.update_view(MOVED);
        assert_eq!(s.view.matrix, MOVED);
        s.set_fade(1.5);
        assert_eq!(s.view.scene_fade, 1.0);
        s.set_fade(-0.5);
        assert_eq!(s.view.scene_fade, 0.0);
    }

    #[test]
    fn visibility_toggles_one_slot() {
        let mut s = scene(2);
        s.update_visibility(DrawIndex(0), false);
        assert!(!s.draw.objects[0].visible);
        assert!(s.draw.objects[1].visible);
        s.update_visibility(DrawIndex(5), false);
    }

    #[test]
    fn retiring_hides_the_slot_and_drops_residency() {
        let mut s = scene(1);
        s.retire_draw_object(DrawIndex(0));
        assert!(!s.draw.objects[0].visible);
        assert!(!s.draw.objects[0].resident);
        assert!(s.gpu_dirty.rt_topology);
    }

    #[test]
    fn retiring_an_out_of_range_slot_changes_nothing() {
        let mut s = scene(1);
        s.retire_draw_object(DrawIndex(3));
        assert!(s.draw.objects[0].visible && s.draw.objects[0].resident);
        assert!(!s.gpu_dirty.rt_topology);
    }

    #[test]
    fn a_clone_copies_the_source_at_a_new_transform() {
        let mut s = scene(1);
        {
            let src = &mut s.draw.objects[0];
            src.vertex_offset = 64;
            src.index_count = 36;
            src.texture_slot = 4;
            src.cull_distance = 12.0;
            src.shader_bucket = 2;
            src.lod_alternates = vec![LodSlice {
                index_offset: 7,
                index_count: 9,
                switch_distance: 20.0,
            }];
        }
        s.clone_static_draw_object(DrawIndex(0), MOVED, SlotAlloc::Append(DrawIndex(1)))
            .expect("source exists");
        let clone = &s.draw.objects[1];
        assert_eq!(clone.vertex_offset, 64);
        assert_eq!(clone.index_count, 36);
        assert_eq!(clone.texture_slot, 4);
        assert_eq!(clone.cull_distance, 12.0);
        assert_eq!(clone.shader_bucket, 2);
        assert_eq!(clone.lod_alternates.len(), 1);
        assert_eq!(clone.model, MOVED);
        assert!(clone.visible && clone.resident);
        assert!(!clone.cullable());
        assert!(s.gpu_dirty.rt_topology);
    }

    #[test]
    fn a_clone_into_a_reused_slot_reads_as_a_new_occupant() {
        let mut s = scene(2);
        observe(&mut s);
        assert!(!reoccupied(&mut s, 1));
        s.clone_static_draw_object(DrawIndex(0), MOVED, SlotAlloc::Reuse(DrawIndex(1)))
            .expect("source exists");
        assert!(reoccupied(&mut s, 1));
    }

    #[test]
    fn a_clone_of_an_out_of_range_source_is_refused() {
        let mut s = scene(1);
        let err = s
            .clone_static_draw_object(DrawIndex(4), MOVED, SlotAlloc::Append(DrawIndex(1)))
            .unwrap_err();
        assert!(err.to_string().contains("src draw 4 out of range"), "{err}");
        assert_eq!(s.draw.objects.len(), 1);
    }

    #[test]
    fn a_clone_past_a_full_reserve_is_refused() {
        let mut s = SceneState::new(DrawList::with_runtime_reserve(vec![], 0, 1), [0.0; 4]);
        s.draw.objects.push(draw_object());
        s.draw.objects[0].resident = true;
        let err = s
            .clone_static_draw_object(DrawIndex(0), MOVED, SlotAlloc::Append(DrawIndex(1)))
            .unwrap_err();
        assert!(err.to_string().contains("reserve (1) is full"), "{err}");
        assert!(!s.gpu_dirty.rt_topology);
    }

    #[test]
    fn a_material_edit_rewrites_the_slot_and_flags_the_geometry_table() {
        let mut s = scene(1);
        let mut material = MaterialUniforms::DEFAULT;
        material.opacity = 0.5;
        s.set_draw_material(DrawIndex(0), material, 3, 4);
        let obj = &s.draw.objects[0];
        assert_eq!(obj.material.opacity, 0.5);
        assert_eq!((obj.texture_slot, obj.normal_map_slot), (3, 4));
        assert!(s.gpu_dirty.rt_topology);
    }

    #[test]
    fn a_material_edit_out_of_range_changes_nothing() {
        let mut s = scene(1);
        s.set_draw_material(DrawIndex(2), MaterialUniforms::DEFAULT, 3, 4);
        assert!(!s.gpu_dirty.rt_topology);
    }

    #[test]
    fn a_cull_distance_is_clamped_to_non_negative() {
        let mut s = scene(1);
        s.set_draw_cull_distance(DrawIndex(0), -3.0);
        assert_eq!(s.draw.objects[0].cull_distance, 0.0);
        s.set_draw_cull_distance(DrawIndex(0), 40.0);
        assert_eq!(s.draw.objects[0].cull_distance, 40.0);
    }

    #[test]
    fn set_chunk_model_reports_an_out_of_range_slot() {
        let mut s = scene(1);
        assert!(s.set_chunk_model(DrawIndex(0), MOVED).is_ok());
        assert_eq!(s.draw.objects[0].model, MOVED);
        assert!(s.set_chunk_model(DrawIndex(6), MOVED).is_err());
    }

    #[test]
    fn geometry_queries_read_the_slot() {
        let mut s = scene(1);
        s.draw.objects[0].vertex_count = 8;
        s.draw.objects[0].index_count = 12;
        s.draw.objects[0].lod_alternates = vec![LodSlice {
            index_offset: 0,
            index_count: 6,
            switch_distance: 10.0,
        }];
        assert_eq!(s.draw_geometry_size(DrawIndex(0)), Some((8, 12)));
        assert_eq!(s.draw_lod_index_counts(DrawIndex(0)), Some(vec![6]));
        assert_eq!(s.draw_geometry_size(DrawIndex(1)), None);
        assert_eq!(s.draw_lod_index_counts(DrawIndex(1)), None);
    }

    #[test]
    fn revealing_a_skinned_instance_shows_it_and_resets_its_history() {
        let mut s = scene(0);
        s.skinned
            .draw_objects
            .push(crate::test_support::skinned_draw_object());
        s.skinned.joint_matrices.push(vec![MOVED]);
        s.skinned.morph_weights.push(vec![]);
        s.skinned.draw_objects[0].visible = false;
        let history = s.model_history.get_mut();
        history.begin(HistoryMode::Track, 1);
        history.skinned_flags(0, 0);
        s.reveal_skinned_instance(SkinnedIndex(0), MOVED);
        assert!(s.skinned.draw_objects[0].visible);
        assert_eq!(s.skinned.draw_objects[0].model, MOVED);
        let history = s.model_history.get_mut();
        history.begin(HistoryMode::Track, 1);
        assert_ne!(history.skinned_flags(0, 0), 0);
    }

    #[test]
    fn a_skeleton_update_out_of_range_is_an_error() {
        let mut s = scene(0);
        let err = s.update_skinned_skeleton(SkinnedIndex(2), 4).unwrap_err();
        assert!(err.to_string().contains("out of range"), "{err}");
    }
}
