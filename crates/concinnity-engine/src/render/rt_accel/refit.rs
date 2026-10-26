//! Backend-agnostic refit cadence for the acceleration structures updated
//! every frame something moves: the skinned bottom-level structures, whose
//! vertices a compute pass re-poses, and the static top-level structure, whose
//! instance transforms follow the moving props. While the primitive or instance
//! set is unchanged that update can be a REFIT (Vulkan's
//! `VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR`, DXR's `PERFORM_UPDATE`,
//! Metal's `refitAccelerationStructure`), which re-fits the existing tree's
//! bounding volumes instead of rebuilding it from scratch.
//!
//! A refit keeps the tree the last full build produced, so traversal quality
//! decays as the contents drift away from where that tree was built; this
//! module bounds that with a periodic full rebuild. It owns only the pure
//! decision -- the descriptors, the allocation and the recorded build are
//! per-backend. Split out so the cadence is unit-testable without a GPU.

use alloc::vec::Vec;

/// Full rebuilds per ring slot: after this many consecutive refits the next
/// skinned update rebuilds that slot's BLAS from scratch. Each slot counts
/// independently and they are touched on different frames, so the rebuilds
/// stagger rather than landing on one frame.
pub const REFIT_LIMIT: u32 = 32;

/// Consecutive refits of one static TLAS ring slot before its next update
/// rebuilds it. Lower than [`REFIT_LIMIT`]: a prop that rolls across the scene
/// stretches the boxes above it on every refit, while a skinned pose stays near
/// its origin. Each slot takes every Nth update of a ring N deep, so a slot's
/// tree is refit across up to N times this many frames of motion, and the
/// slots' first builds land on consecutive frames, so their rebuilds come in
/// runs of N.
pub const TLAS_REFIT_LIMIT: u32 = 16;

/// The geometry one skinned BLAS is built over: its slice of the shared skinned
/// index buffer plus the vertex range the deformed buffer spans. Equal
/// signatures mean the same triangles addressing the same vertex range with only
/// the positions moved, which is exactly when a refit is legal; anything else (a
/// mesh hot-reload, a different mesh becoming visible, a grown deformed buffer)
/// changes the geometry description and needs a full rebuild. `vertex_extent` is
/// carried because DXR and Vulkan require the vertex count to match the structure
/// being updated, even though the vertex buffer's address may move. Metal's
/// descriptors pin no vertex count, so it leaves `vertex_extent` 0 and a grown
/// deformed range alone does not force a rebuild there.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct SkinnedShape {
    /// First index of this slot's range in the shared index buffer.
    pub index_offset: usize,
    /// Indices in this slot's range.
    pub index_count: usize,
    /// Vertices the slot's range spans.
    pub vertex_extent: u32,
}

/// Whether an acceleration structure can be refit from this frame's contents
/// or must be rebuilt from scratch.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AccelUpdate {
    /// Rebuild the acceleration structure from scratch.
    Build,
    /// Refit the existing acceleration structure in place.
    Refit,
}

// Rebuild when the geometry changed (a refit is then illegal), when the slot has
// no built tree to refit, or once every `limit` refits to bound the quality
// drift. Pure so the cadence is unit-testable without a GPU.
fn accel_update(shape_changed: bool, built: bool, refits: u32, limit: u32) -> AccelUpdate {
    if shape_changed || !built || refits >= limit {
        AccelUpdate::Build
    } else {
        AccelUpdate::Refit
    }
}

/// One ring slot's refit bookkeeping: the geometry its BLAS were last built over,
/// whether they hold a tree a refit can update, and how many consecutive refits
/// have run since the last full build.
#[derive(Default)]
pub struct SkinnedRefit {
    shapes: Vec<SkinnedShape>,
    built: bool,
    refits: u32,
}

impl SkinnedRefit {
    /// How this frame's skinned BLAS should be updated, recording the choice so
    /// the refit run stays bounded and the shapes so the next frame can compare.
    /// `storage_changed` must be set when the structures or the buffer they trace
    /// were (re)allocated this frame, which leaves no tree to refit. Call once the
    /// frame's fallible work has passed: recording a build the backend never
    /// encodes would leave the slot claiming a tree a later refit cannot update.
    pub fn plan(&mut self, shapes: &[SkinnedShape], storage_changed: bool) -> AccelUpdate {
        let changed = storage_changed || self.shapes != shapes;
        let update = accel_update(changed, self.built, self.refits, REFIT_LIMIT);
        if changed {
            self.shapes.clear();
            self.shapes.extend_from_slice(shapes);
        }
        match update {
            AccelUpdate::Build => {
                self.built = true;
                self.refits = 0;
            }
            AccelUpdate::Refit => self.refits += 1,
        }
        update
    }

    /// Whether the last planned update was over exactly `shapes`, so structures
    /// allocated for them can still be refit.
    pub fn matches(&self, shapes: &[SkinnedShape]) -> bool {
        self.shapes == shapes
    }

    /// Forget the tree this slot's BLAS hold, so the next update rebuilds rather
    /// than refitting. Called when the slot stops publishing (no skinned object is
    /// visible) or its structures are otherwise invalidated.
    pub fn reset(&mut self) {
        self.shapes.clear();
        self.built = false;
        self.refits = 0;
    }
}

/// What a static TLAS is laid out over: the BLAS head generation its instances
/// name and how many instances it holds. A refit must keep both.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TlasTree {
    /// The head generation the instances reference.
    pub head: u64,
    /// The instance count.
    pub instances: usize,
}

/// Refit bookkeeping for one static TLAS ring slot. A slot refits its own
/// tree in place, so the live TLAS other frames trace is never written, from
/// the tree it was last fully built as: a new head or instance count rebuilds
/// it, and so does storage reallocated under it ([`Self::reset`]).
#[derive(Debug, Default)]
pub struct TlasRefit {
    built: Option<TlasTree>,
    refits: u32,
}

impl TlasRefit {
    /// How this slot's TLAS should be updated to hold `tree`.
    pub fn plan(&self, tree: TlasTree) -> AccelUpdate {
        accel_update(
            self.built != Some(tree),
            self.built.is_some(),
            self.refits,
            TLAS_REFIT_LIMIT,
        )
    }

    /// The slot's TLAS was updated by `update` to hold `tree`.
    pub fn published(&mut self, update: AccelUpdate, tree: TlasTree) {
        match update {
            AccelUpdate::Build => {
                self.built = Some(tree);
                self.refits = 0;
            }
            AccelUpdate::Refit => self.refits += 1,
        }
    }

    /// Forget the slot's tree, so its next update builds. Call wherever the
    /// slot's TLAS storage is (re)allocated, before anything that can fail.
    pub fn reset(&mut self) {
        self.built = None;
        self.refits = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(tag: usize) -> SkinnedShape {
        SkinnedShape {
            index_offset: tag,
            index_count: 300,
            vertex_extent: 512,
        }
    }

    #[test]
    fn a_changed_triangle_set_forces_a_full_build() {
        // A refit cannot add or remove geometry, so a changed shape always
        // rebuilds even when the slot has a tree and refits to spare.
        assert_eq!(accel_update(true, true, 0, 32), AccelUpdate::Build);
    }

    #[test]
    fn an_unbuilt_slot_cannot_be_refit() {
        assert_eq!(accel_update(false, false, 0, 32), AccelUpdate::Build);
    }

    #[test]
    fn a_stable_shape_refits_until_the_limit() {
        assert_eq!(accel_update(false, true, 0, 32), AccelUpdate::Refit);
        assert_eq!(accel_update(false, true, 31, 32), AccelUpdate::Refit);
        // The 32nd refit is instead a rebuild, bounding the quality drift.
        assert_eq!(accel_update(false, true, 32, 32), AccelUpdate::Build);
        assert_eq!(accel_update(false, true, 99, 32), AccelUpdate::Build);
    }

    #[test]
    fn a_zero_limit_never_refits() {
        assert_eq!(accel_update(false, true, 0, 0), AccelUpdate::Build);
    }

    #[test]
    fn shape_equality_covers_slice_and_vertex_extent() {
        let a = shape(12);
        assert_eq!(a, a);
        assert_ne!(a, shape(13));
        assert_ne!(
            a,
            SkinnedShape {
                index_count: 303,
                ..a
            }
        );
        assert_ne!(
            a,
            SkinnedShape {
                vertex_extent: 513,
                ..a
            }
        );
    }

    #[test]
    fn a_fresh_slot_builds_then_refits() {
        let mut slot = SkinnedRefit::default();
        let shapes = [shape(0), shape(1)];
        assert_eq!(slot.plan(&shapes, false), AccelUpdate::Build);
        assert_eq!(slot.plan(&shapes, false), AccelUpdate::Refit);
        assert_eq!(slot.plan(&shapes, false), AccelUpdate::Refit);
    }

    #[test]
    fn reallocated_storage_rebuilds_even_on_an_unchanged_shape() {
        let mut slot = SkinnedRefit::default();
        let shapes = [shape(0)];
        assert_eq!(slot.plan(&shapes, false), AccelUpdate::Build);
        assert_eq!(slot.plan(&shapes, true), AccelUpdate::Build);
        assert_eq!(slot.plan(&shapes, false), AccelUpdate::Refit);
    }

    #[test]
    fn a_changed_object_set_rebuilds_and_restarts_the_run() {
        let mut slot = SkinnedRefit::default();
        assert_eq!(slot.plan(&[shape(0)], false), AccelUpdate::Build);
        assert_eq!(slot.plan(&[shape(0)], false), AccelUpdate::Refit);
        // A second skinned object became visible: one more BLAS, so a full build.
        assert_eq!(slot.plan(&[shape(0), shape(1)], false), AccelUpdate::Build);
        assert_eq!(slot.plan(&[shape(0), shape(1)], false), AccelUpdate::Refit);
    }

    #[test]
    fn the_refit_run_is_bounded_by_the_limit() {
        let mut slot = SkinnedRefit::default();
        let shapes = [shape(0)];
        assert_eq!(slot.plan(&shapes, false), AccelUpdate::Build);
        for _ in 0..REFIT_LIMIT {
            assert_eq!(slot.plan(&shapes, false), AccelUpdate::Refit);
        }
        // The run has reached the limit: the next update is a full rebuild, and
        // the run then restarts.
        assert_eq!(slot.plan(&shapes, false), AccelUpdate::Build);
        assert_eq!(slot.plan(&shapes, false), AccelUpdate::Refit);
    }

    #[test]
    fn matches_tracks_the_last_planned_shapes() {
        let mut slot = SkinnedRefit::default();
        assert!(slot.matches(&[]));
        slot.plan(&[shape(0)], false);
        assert!(slot.matches(&[shape(0)]));
        assert!(!slot.matches(&[shape(1)]));
        slot.reset();
        assert!(!slot.matches(&[shape(0)]));
    }

    fn tree(head: u64, instances: usize) -> TlasTree {
        TlasTree { head, instances }
    }

    #[test]
    fn a_tlas_slot_builds_first_then_refits_up_to_its_limit() {
        let mut slot = TlasRefit::default();
        assert_eq!(slot.plan(tree(0, 8)), AccelUpdate::Build);
        slot.published(AccelUpdate::Build, tree(0, 8));
        for _ in 0..TLAS_REFIT_LIMIT {
            assert_eq!(slot.plan(tree(0, 8)), AccelUpdate::Refit);
            slot.published(AccelUpdate::Refit, tree(0, 8));
        }
        assert_eq!(slot.plan(tree(0, 8)), AccelUpdate::Build);
        slot.published(AccelUpdate::Build, tree(0, 8));
        assert_eq!(slot.plan(tree(0, 8)), AccelUpdate::Refit);
    }

    #[test]
    fn a_new_head_rebuilds_the_tlas_slot() {
        let mut slot = TlasRefit::default();
        slot.published(AccelUpdate::Build, tree(3, 8));
        assert_eq!(slot.plan(tree(4, 8)), AccelUpdate::Build);
        slot.published(AccelUpdate::Build, tree(4, 8));
        assert_eq!(slot.plan(tree(4, 8)), AccelUpdate::Refit);
    }

    // A refit cannot change how many instances the tree holds, even over the
    // same head.
    #[test]
    fn a_changed_instance_count_rebuilds_the_tlas_slot() {
        let mut slot = TlasRefit::default();
        slot.published(AccelUpdate::Build, tree(0, 8));
        assert_eq!(slot.plan(tree(0, 7)), AccelUpdate::Build);
    }

    #[test]
    fn reallocated_tlas_storage_rebuilds() {
        let mut slot = TlasRefit::default();
        slot.published(AccelUpdate::Build, tree(0, 8));
        slot.reset();
        assert_eq!(slot.plan(tree(0, 8)), AccelUpdate::Build);
    }

    #[test]
    fn reset_makes_the_next_update_a_build() {
        let mut slot = SkinnedRefit::default();
        let shapes = [shape(0)];
        assert_eq!(slot.plan(&shapes, false), AccelUpdate::Build);
        assert_eq!(slot.plan(&shapes, false), AccelUpdate::Refit);
        slot.reset();
        assert_eq!(slot.plan(&shapes, false), AccelUpdate::Build);
    }
}
