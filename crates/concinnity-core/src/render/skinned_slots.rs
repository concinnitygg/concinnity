//! The per-slot skinned records every graphics backend keeps on the CPU: the
//! skinned draw objects, their joint palettes, and their morph weights, held in
//! three arrays that run parallel by slot index.
//!
//! A backend's skinned state is otherwise GPU handles (pipelines, the shared
//! vertex / index buffers, the per-frame joint and weight uploads), and those
//! differ per API. These three arrays do not: they are the source the per-frame
//! upload reads, and the entry points that mutate them (a new pose, a revealed
//! instance-pool slot, a retired slot, a moved instance, a hot-reloaded
//! skeleton) touch no device object at all. Metal, DirectX, and Vulkan embed
//! this and share those entry points.
//!
//! Slot indices are stable skinned-draw-object indices (see
//! [`crate::render::skinned_pool`]); nothing is compacted, so the parallel
//! arrays stay valid across a spawn and a despawn.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::gfx::render_types::{MAX_JOINTS, SkinnedDrawObject};
use crate::gfx::transform::IDENTITY;
use crate::render::model_history::ModelHistory;

/// The skinned draw objects and their parallel pose arrays.
#[derive(Default)]
pub struct SkinnedSlots {
    /// One entry per skinned slot: an authored mesh or one of its pre-reserved
    /// instance copies.
    pub draw_objects: Vec<SkinnedDrawObject>,
    /// Current skinning matrices per slot, parallel to [`Self::draw_objects`].
    /// Rewritten each frame by [`Self::update_pose`] and uploaded per frame.
    pub joint_matrices: Vec<Vec<[[f32; 4]; 4]>>,
    /// Current morph weights per slot, parallel to [`Self::draw_objects`];
    /// empty for a slot whose mesh has no morph targets.
    pub morph_weights: Vec<Vec<f32>>,
}

impl SkinnedSlots {
    /// Empty; a backend's `upload_skinned` sizes all three arrays to the world.
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the skinning matrices for one slot. Called each frame from
    /// `GraphicsSystem` with the pose `AnimationSystem` computed. An empty pose
    /// is stored as a single identity joint so the shader always has one
    /// matrix to blend. Out-of-range indices are ignored.
    pub fn update_pose(&mut self, skinned_index: usize, matrices: &[[[f32; 4]; 4]]) {
        if let Some(slot) = self.joint_matrices.get_mut(skinned_index) {
            slot.clear();
            slot.extend_from_slice(matrices);
            if slot.is_empty() {
                slot.push(IDENTITY);
            }
        }
    }

    /// Replace one slot's morph weights. Out-of-range indices and slots without
    /// morph targets are ignored; extra weights are dropped.
    pub fn update_morph_weights(&mut self, skinned_index: usize, weights: &[f32]) {
        if let Some(slot) = self.morph_weights.get_mut(skinned_index) {
            for (i, w) in slot.iter_mut().enumerate() {
                *w = weights.get(i).copied().unwrap_or(0.0);
            }
        }
    }

    /// Update a slot's joint count and resize its joint palette. Driven by
    /// asset hot-reload when a re-imported `.glb`'s skeleton has a different
    /// joint count than the slot was initialized with. No GPU resource has to
    /// grow: a backend's per-frame joint upload is sized for [`MAX_JOINTS`] and
    /// the shaders address the palette by vertex-encoded joint index, so a new
    /// count only resizes the palette here and updates
    /// `SkinnedDrawObject::joint_count`. Shrinking truncates; growing seeds the
    /// new entries to identity, so the slot renders undeformed until the next
    /// [`Self::update_pose`]. Counts above [`MAX_JOINTS`] are clamped.
    pub fn update_skeleton(
        &mut self,
        skinned_index: usize,
        new_joint_count: usize,
    ) -> Result<(), String> {
        let obj = self.draw_objects.get_mut(skinned_index).ok_or_else(|| {
            format!(
                "update_skinned_skeleton: skinned object {} out of range",
                skinned_index
            )
        })?;
        let capped = new_joint_count.min(MAX_JOINTS);
        obj.joint_count = capped;
        let size = capped.max(1);
        if let Some(slot) = self.joint_matrices.get_mut(skinned_index) {
            slot.resize(size, IDENTITY);
        }
        Ok(())
    }

    /// Reveal the pre-reserved instance slot at `instance_index` (the engine's
    /// instance pool decided which): show it at `model` and reset its palette
    /// to the bind pose so it does not flash its previous occupant's last frame
    /// (the owning `SkeletonPose`'s first pose push replaces it next frame).
    /// A no-op if the index is out of range.
    pub fn reveal(
        &mut self,
        instance_index: usize,
        model: [[f32; 4]; 4],
        history: &mut ModelHistory,
    ) {
        let Some(obj) = self.draw_objects.get_mut(instance_index) else {
            return;
        };
        obj.model = model;
        obj.visible = true;
        // The slot's model-history entry belongs to the previous occupant, so
        // the next pre-pass must reproject through the revealed model instead.
        history.reoccupy_skinned(instance_index);
        if let Some(palette) = self.joint_matrices.get_mut(instance_index) {
            palette.iter_mut().for_each(|m| *m = IDENTITY);
        }
    }

    /// Hide a slot; the engine's instance pool recycles it. A no-op if the
    /// index is out of range.
    pub fn retire(&mut self, skinned_index: usize) {
        if let Some(obj) = self.draw_objects.get_mut(skinned_index) {
            obj.visible = false;
        }
    }

    /// Push the model-to-world matrices of the given slots, one
    /// `(skinned index, matrix)` entry per moved instance. The per-frame cull
    /// records read `obj.model` directly (they are rebuilt every frame), so
    /// this only writes the fields. Out-of-range indices have no effect.
    pub fn update_models(&mut self, updates: &[(u32, [[f32; 4]; 4])]) {
        for &(skinned_index, model) in updates {
            if let Some(obj) = self.draw_objects.get_mut(skinned_index as usize) {
                obj.model = model;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::render_types::draw_args_no_history;
    use crate::render::model_history::{HistoryMode, ModelHistory};

    const ONE: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [7.0, 8.0, 9.0, 1.0],
    ];

    // Two slots, the first with a 3-joint palette and two morph weights, the
    // second morphless with a single joint. Both start hidden.
    fn two_slots() -> SkinnedSlots {
        let mut slots = SkinnedSlots::new();
        for _ in 0..2 {
            let mut obj = crate::test_support::skinned_draw_object();
            obj.visible = false;
            obj.joint_count = 1;
            slots.draw_objects.push(obj);
        }
        slots.joint_matrices = alloc::vec![alloc::vec![IDENTITY; 3], alloc::vec![IDENTITY; 1]];
        slots.morph_weights = alloc::vec![alloc::vec![0.0; 2], Vec::new()];
        slots
    }

    #[test]
    fn update_pose_replaces_the_palette() {
        let mut slots = two_slots();
        slots.update_pose(0, &[ONE, ONE]);
        assert_eq!(slots.joint_matrices[0], alloc::vec![ONE, ONE]);
    }

    #[test]
    fn an_empty_pose_leaves_one_identity_joint() {
        let mut slots = two_slots();
        slots.update_pose(0, &[]);
        assert_eq!(slots.joint_matrices[0], alloc::vec![IDENTITY]);
    }

    #[test]
    fn update_pose_ignores_an_out_of_range_slot() {
        let mut slots = two_slots();
        slots.update_pose(9, &[ONE]);
        assert_eq!(slots.joint_matrices.len(), 2);
    }

    #[test]
    fn morph_weights_are_padded_and_truncated_to_the_slot() {
        let mut slots = two_slots();
        slots.update_morph_weights(0, &[0.5]);
        assert_eq!(slots.morph_weights[0], alloc::vec![0.5, 0.0]);
        slots.update_morph_weights(0, &[0.25, 0.75, 1.0]);
        assert_eq!(slots.morph_weights[0], alloc::vec![0.25, 0.75]);
    }

    #[test]
    fn morph_weights_on_a_morphless_slot_are_ignored() {
        let mut slots = two_slots();
        slots.update_morph_weights(1, &[0.5]);
        assert!(slots.morph_weights[1].is_empty());
    }

    #[test]
    fn a_grown_skeleton_seeds_new_joints_to_identity() {
        let mut slots = two_slots();
        slots.update_pose(0, &[ONE, ONE, ONE]);
        assert!(slots.update_skeleton(0, 5).is_ok());
        assert_eq!(slots.draw_objects[0].joint_count, 5);
        assert_eq!(slots.joint_matrices[0].len(), 5);
        assert_eq!(slots.joint_matrices[0][4], IDENTITY);
    }

    #[test]
    fn a_shrunk_skeleton_truncates_the_palette() {
        let mut slots = two_slots();
        assert!(slots.update_skeleton(0, 1).is_ok());
        assert_eq!(slots.draw_objects[0].joint_count, 1);
        assert_eq!(slots.joint_matrices[0].len(), 1);
    }

    #[test]
    fn a_zero_joint_skeleton_keeps_one_palette_entry() {
        let mut slots = two_slots();
        assert!(slots.update_skeleton(0, 0).is_ok());
        assert_eq!(slots.draw_objects[0].joint_count, 0);
        assert_eq!(slots.joint_matrices[0].len(), 1);
    }

    #[test]
    fn a_skeleton_above_the_cap_is_clamped() {
        let mut slots = two_slots();
        assert!(slots.update_skeleton(0, MAX_JOINTS + 16).is_ok());
        assert_eq!(slots.draw_objects[0].joint_count, MAX_JOINTS);
        assert_eq!(slots.joint_matrices[0].len(), MAX_JOINTS);
    }

    #[test]
    fn an_out_of_range_skeleton_update_reports_the_slot() {
        let mut slots = two_slots();
        let err = slots.update_skeleton(9, 4).unwrap_err();
        assert!(err.contains('9'), "{err}");
    }

    #[test]
    fn reveal_shows_the_slot_and_resets_its_palette() {
        let mut slots = two_slots();
        let mut history = ModelHistory::new();
        slots.update_pose(0, &[ONE, ONE, ONE]);
        slots.reveal(0, ONE, &mut history);
        assert!(slots.draw_objects[0].visible);
        assert_eq!(slots.draw_objects[0].model, ONE);
        assert_eq!(slots.joint_matrices[0], alloc::vec![IDENTITY; 3]);
    }

    #[test]
    fn reveal_invalidates_the_slots_model_history() {
        let mut slots = two_slots();
        let mut history = ModelHistory::new();
        // Prime record 0 so the tracker holds a usable previous transform.
        history.begin(HistoryMode::Track, 2);
        history.skinned_flags(0, 0);
        history.begin(HistoryMode::Track, 2);
        assert_eq!(history.skinned_flags(0, 0), 0);
        slots.reveal(0, ONE, &mut history);
        history.begin(HistoryMode::Track, 2);
        assert_eq!(history.skinned_flags(0, 0), draw_args_no_history());
    }

    #[test]
    fn reveal_ignores_an_out_of_range_slot() {
        let mut slots = two_slots();
        let mut history = ModelHistory::new();
        slots.reveal(9, ONE, &mut history);
        assert!(!slots.draw_objects[0].visible);
        assert!(!slots.draw_objects[1].visible);
    }

    #[test]
    fn retire_hides_the_slot() {
        let mut slots = two_slots();
        let mut history = ModelHistory::new();
        slots.reveal(1, ONE, &mut history);
        slots.retire(1);
        assert!(!slots.draw_objects[1].visible);
        slots.retire(9);
    }

    #[test]
    fn update_models_writes_only_the_listed_slots() {
        let mut slots = two_slots();
        slots.update_models(&[(1, ONE), (9, ONE)]);
        assert_eq!(slots.draw_objects[0].model, IDENTITY);
        assert_eq!(slots.draw_objects[1].model, ONE);
    }
}
