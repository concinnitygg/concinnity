//! Per-slot edits of the draw list and the view: what the engine pushes each
//! frame, and the runtime spawns and live edits that rewrite a slot in place.

use alloc::format;
use alloc::vec::Vec;

use super::{SceneState, caster_bounds};
use crate::environment::planet::Rebase;
use crate::gfx::frustum::Aabb;
use crate::gfx::render_types::{DrawIndex, DrawObject, MaterialUniforms, SkinnedIndex};
use crate::render::backend::SkinnedFrame;
use crate::render::backend::error::{RenderError, RenderResult};
use crate::render::slots::draw_slot::{self, SlotAlloc};

impl SceneState {
    /// Replace the camera's view matrix for the next frame.
    pub fn update_view(&mut self, matrix: [[f32; 4]; 4]) {
        self.view.matrix = matrix;
    }

    /// Set the scene-transition fade for the next frame, clamped to `[0, 1]`.
    pub fn set_fade(&mut self, fade: f32) {
        self.view.scene_fade = fade.clamp(0.0, 1.0);
    }

    /// Write the model matrices of the given draw slots, in order, each
    /// change including `rebase` when the frame moves the world. A shown draw
    /// that moved more than the rebase carried it tells the shadow plan where
    /// it was and where it is, in the frame the plan is in until the rebase
    /// reaches it. An out-of-range slot is ignored.
    pub fn update_models(
        &mut self,
        updates: &[(DrawIndex, [[f32; 4]; 4])],
        rebase: Option<&Rebase>,
    ) {
        for &(index, model) in updates {
            let Some(obj) = self.draw.objects.get_mut(index.index()) else {
                continue;
            };
            let moved = obj.visible && obj.resident && moved_past(obj.model, model, rebase);
            let before = caster_bounds(obj);
            obj.set_model(model);
            if moved {
                let after = caster_bounds(obj).map(|b| before_rebase(b, rebase));
                self.planner.shadow.casters_moved(before);
                self.planner.shadow.casters_moved(after);
            }
        }
    }

    /// Apply a frame's skinned pushes: each changed instance's joint matrices
    /// and morph weights, then each moved instance's model. A shown instance
    /// that took a new pose or moved more than the frame's rebase carried it
    /// tells the shadow plan where it is, and where it was when it moved.
    pub fn update_skinned(&mut self, frame: &SkinnedFrame<'_>) {
        let slots = &mut self.skinned;
        slots.push_marks.begin(slots.draw_objects.len());
        for (index, matrices) in frame.poses.iter() {
            slots.update_pose(index, matrices);
            slots.push_marks.reshape(index.index());
        }
        for (index, weights) in frame.morphs.iter() {
            slots.update_morph_weights(index, weights);
            slots.push_marks.reshape(index.index());
        }
        for &(index, model) in frame.models {
            let i = index.index();
            let reshaped = slots.push_marks.reshaped(i);
            let Some(obj) = slots.draw_objects.get_mut(i) else {
                continue;
            };
            let moved = obj.visible && moved_past(obj.model, model, frame.rebase);
            let before = obj.world_bounds();
            obj.model = model;
            if moved {
                self.planner.shadow.casters_moved(Some(before));
            }
            if moved || (obj.visible && reshaped) {
                let after = before_rebase(obj.world_bounds(), frame.rebase);
                self.planner.shadow.casters_moved(Some(after));
            }
            slots.push_marks.place(i);
        }
        // A reshaped instance not placed above reports where it stands, once.
        let reshaped = frame.poses.iter().map(|(i, _)| i);
        for index in reshaped.chain(frame.morphs.iter().map(|(i, _)| i)) {
            let i = index.index();
            if slots.push_marks.placed(i) {
                continue;
            }
            slots.push_marks.place(i);
            if let Some(obj) = slots.draw_objects.get(i)
                && obj.visible
            {
                let bounds = before_rebase(obj.world_bounds(), frame.rebase);
                self.planner.shadow.casters_moved(Some(bounds));
            }
        }
    }

    /// Show or hide one draw slot in every raster pass. An out-of-range slot is
    /// ignored. A resident draw that flips changes the casters the shadow
    /// views draw, except a planet tile's: its show and hide is the terrain's
    /// detail changing hands, which the far shadow's bias covers.
    pub fn update_visibility(&mut self, index: DrawIndex, visible: bool) {
        let Some(obj) = self.draw.objects.get_mut(index.index()) else {
            return;
        };
        let flipped = obj.visible != visible;
        obj.visible = visible;
        // The models were written first this frame, so on a rebase frame the
        // box may already be in the frame the rebase moves to.
        if flipped && obj.resident && obj.planet_tile.is_none() {
            let bounds = caster_bounds(obj);
            self.planner.shadow.casters_placed(bounds);
        }
    }

    /// Hide a slot from every raster pass and drop it from the ray-tracing
    /// draw set, leaving its geometry where it is for the slot allocator to
    /// recycle. A draw that was showing changes the casters the shadow views
    /// draw. An out-of-range slot is ignored.
    pub fn retire_draw_object(&mut self, index: DrawIndex) {
        if let Some(obj) = self.draw.objects.get_mut(index.index()) {
            if obj.visible && obj.resident {
                let bounds = caster_bounds(obj);
                self.planner.shadow.casters_changed(bounds);
            }
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
    /// source's geometry, LOD slices, textures, material, cull distance and
    /// model-space box, which its culling box follows; as a new caster it
    /// changes what the shadow views draw. `Err` when the source is out of
    /// range or the runtime reserve is full.
    pub(crate) fn clone_static_draw_object(
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
            planet_tile: source.planet_tile,
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
            local_bounds: source.local_bounds,
            cull_distance: source.cull_distance,
            lod_alternates: source.lod_alternates.clone(),
        };
        let slot = self.place_draw_object(obj, dst);
        if let Some(obj) = self.draw.objects.get_mut(slot.index()) {
            obj.set_model(model);
            let bounds = caster_bounds(obj);
            self.planner.shadow.casters_placed(bounds);
        }
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
    pub(crate) fn set_chunk_model(
        &mut self,
        index: DrawIndex,
        model: [[f32; 4]; 4],
    ) -> RenderResult<()> {
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

    /// Reveal a pre-reserved skinned instance at `model`, a new caster for
    /// the shadow views; see `SkinnedSlots::reveal`.
    pub fn reveal_skinned_instance(&mut self, index: SkinnedIndex, model: [[f32; 4]; 4]) {
        self.skinned
            .reveal(index, model, self.model_history.get_mut());
        if let Some(obj) = self.skinned.draw_objects.get(index.index()) {
            self.planner.shadow.casters_placed(Some(obj.world_bounds()));
        }
    }

    /// Hide a live skinned instance, which the shadow views stop drawing when
    /// it was showing; see `SkinnedSlots::retire`.
    pub fn retire_skinned_instance(&mut self, index: SkinnedIndex) {
        if let Some(obj) = self.skinned.draw_objects.get(index.index())
            && obj.visible
        {
            self.planner
                .shadow
                .casters_changed(Some(obj.world_bounds()));
        }
        self.skinned.retire(index);
    }

    /// Resize a skinned slot's palette to a reloaded skeleton; see
    /// `SkinnedSlots::update_skeleton`.
    pub(crate) fn update_skinned_skeleton(
        &mut self,
        index: SkinnedIndex,
        new_joint_count: usize,
    ) -> RenderResult<()> {
        self.skinned
            .update_skeleton(index, new_joint_count)
            .map_err(RenderError::Other)
    }
}

// The world box `bounds`, given in the frame `rebase` moves the world into,
// back in the frame before it.
fn before_rebase(bounds: Aabb, rebase: Option<&Rebase>) -> Aabb {
    rebase.map_or(bounds, |r| {
        crate::gfx::frustum::transform_aabb(bounds.0, bounds.1, r.inverse_matrix())
    })
}

// Whether a draw placed at `before` that the frame's `rebase` carries, if any,
// moved by more than float noise to reach `after`. The noise grows with the
// distances the matrices and the rebase carry: carrying a position rounds it
// to a few units in its last place.
fn moved_past(before: [[f32; 4]; 4], after: [[f32; 4]; 4], rebase: Option<&Rebase>) -> bool {
    let carried = rebase.map_or(before, |r| r.apply_matrix(before));
    let reach = [before[3], after[3]]
        .iter()
        .flat_map(|t| t[..3].iter())
        .chain(rebase.map_or(&[0.0; 3], |r| &r.translation))
        .fold(0.0_f32, |m, v| m.max(v.abs()));
    let tolerance = MOVE_TOLERANCE + MOVE_ULPS * f32::EPSILON * reach;
    carried
        .iter()
        .flatten()
        .zip(after.iter().flatten())
        .any(|(a, b)| (a - b).abs() > tolerance)
}

// How far a model matrix entry may drift before the draw counts as moved,
// near the origin: a tenth of a millimeter in translation.
const MOVE_TOLERANCE: f32 = 1e-4;

// Units in the last place of the largest distance involved that carrying a
// draw through a rebase may round it by.
const MOVE_ULPS: f32 = 16.0;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::render_types::LodSlice;
    use crate::render::scene_state::DrawList;
    use crate::render::temporal::model_history::HistoryMode;
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
        SceneState::new(
            DrawList::unreserved(objects, 0),
            [0.0; 4],
            crate::render::light::state::LightState::new(
                crate::gfx::render_types::LightUniforms::DEFAULT,
                &[],
                1,
            ),
            crate::render::material_params::MaterialParamTable::new(Vec::new(), 1),
            crate::render::environment::volumetric_fog::SceneFog::default(),
            crate::render::decal::DecalSet::new(0, 1),
        )
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

    // A prop the simulation carries through a rebase, as the planet carry
    // does (its position and rotation each carried, its matrix rebuilt), did
    // not move, out to kilometers from the origin; one nudged a centimeter
    // past it did.
    #[test]
    fn a_draw_carried_by_a_rebase_did_not_move() {
        use crate::environment::planet::Rebase;
        use crate::transform::Transform;
        let rebase = Rebase {
            rotation: concinnity_core::math::quat_from_axis_angle([0.6, 0.0, 0.8], 0.02),
            translation: [-913.7, 4.3, 1_208.9],
        };
        for i in 0..400 {
            let f = i as f32;
            let t = Transform {
                position: [
                    (f * 37.1).sin() * 2_500.0,
                    (f * 11.3).cos() * 300.0,
                    (f * 23.7).cos() * 2_500.0,
                ],
                rotation_deg: [f * 7.0 % 90.0, f * 13.0 % 360.0, f * 3.0 % 45.0],
                scale: [1.0 + (f * 0.1) % 2.0; 3],
            };
            let carried = Transform {
                position: rebase.apply_point(t.position),
                rotation_deg: rebase.apply_euler_deg(t.rotation_deg),
                ..t
            };
            let (before, after) = (t.model_matrix(), carried.model_matrix());
            assert!(
                !moved_past(before, after, Some(&rebase)),
                "{:?}",
                t.position
            );
            let mut nudged = after;
            nudged[3][0] += 0.01;
            assert!(
                moved_past(before, nudged, Some(&rebase)),
                "{:?}",
                t.position
            );
        }
        assert!(!moved_past(MOVED, MOVED, None));
        assert!(moved_past(MOVED, rebase.apply_matrix(MOVED), None));
    }

    #[test]
    fn update_models_writes_only_the_listed_slots_and_skips_out_of_range() {
        let mut s = scene(2);
        s.update_models(&[(DrawIndex(1), MOVED), (DrawIndex(9), MOVED)], None);
        assert_ne!(s.draw.objects[0].model, MOVED);
        assert_eq!(s.draw.objects[1].model, MOVED);
    }

    // A static draw carried through a planet frame move: its culling box
    // follows the model into the new frame, so the camera carried with it
    // still sees it, where the box it was built with would now be culled.
    #[test]
    fn a_static_draw_moved_with_the_frame_stays_culled_right() {
        use crate::environment::planet::LocalFrame;
        use crate::gfx::camera::view_matrix;
        use crate::gfx::frustum::{Frustum, transform_aabb};
        use crate::render::depth::camera_projection;
        use crate::transform::mat4_mul;

        let mut s = scene(2);
        // A tall post a kilometer out, and a draw whose box was built fixed.
        let local = ([-0.6, -60.0, -0.6], [0.6, 60.0, 0.6]);
        let post = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [3.0, 20.0, -1_060.0, 1.0],
        ];
        let built = transform_aabb(local.0, local.1, post);
        for (i, follows) in [(0, true), (1, false)] {
            let obj = &mut s.draw.objects[i];
            (obj.bb_min, obj.bb_max) = built;
            obj.local_bounds = follows.then_some(local);
            obj.set_model(post);
            assert_eq!((obj.bb_min, obj.bb_max), built);
        }

        // The camera walks to 1 km out and the frame moves under it, turned
        // onto the planet's up there.
        let camera = [0.0, 4.0, -1_000.0];
        let center = [0.0, -50_000.0, 0.0];
        let world = LocalFrame::AUTHORED.to_world(camera);
        let up = concinnity_core::math::dvec3::normalize_or(
            concinnity_core::math::dvec3::sub(world, center),
            [0.0, 1.0, 0.0],
        );
        let (_, rebase) = LocalFrame::AUTHORED.recentered(camera, up);
        let moved = rebase.apply_matrix(post);
        s.update_models(
            &[(DrawIndex(0), moved), (DrawIndex(1), moved)],
            Some(&rebase),
        );

        let proj = camera_projection(70f32.to_radians(), 16.0 / 9.0, 0.1);
        let view = rebase.reproject(view_matrix(camera, 0.0, -0.1));
        let frustum = Frustum::from_camera(mat4_mul(proj, view), Some(500.0));
        let carried = &s.draw.objects[0];
        assert_eq!(
            (carried.bb_min, carried.bb_max),
            transform_aabb(local.0, local.1, moved),
            "not stale"
        );
        assert!(frustum.intersects_aabb(carried.bb_min, carried.bb_max));
        // The fixed box stayed where the post was in the old frame, a
        // kilometer behind the camera's new origin.
        assert_eq!((s.draw.objects[1].bb_min, s.draw.objects[1].bb_max), built);
        assert!(!frustum.intersects_aabb(built.0, built.1));
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
            src.local_bounds = Some(([-1.0; 3], [1.0; 3]));
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
        assert_eq!(clone.local_bounds, Some(([-1.0; 3], [1.0; 3])));
        assert_eq!(
            (clone.bb_min, clone.bb_max),
            crate::gfx::frustum::transform_aabb([-1.0; 3], [1.0; 3], MOVED),
            "the clone's box follows its model"
        );
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
        let mut s = SceneState::new(
            DrawList::with_runtime_reserve(vec![], 0, 1),
            [0.0; 4],
            crate::render::light::state::LightState::new(
                crate::gfx::render_types::LightUniforms::DEFAULT,
                &[],
                1,
            ),
            crate::render::material_params::MaterialParamTable::new(Vec::new(), 1),
            crate::render::environment::volumetric_fog::SceneFog::default(),
            crate::render::decal::DecalSet::new(0, 1),
        );
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
