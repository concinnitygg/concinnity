//! Backend-agnostic refit cadence for the per-frame skinned bottom-level
//! acceleration structures. A skinned object's BLAS traces vertices a compute
//! pass re-poses every frame, so it has to be updated every frame -- but while
//! the triangle set is unchanged that update can be a REFIT (Vulkan's
//! `VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR`, DXR's `PERFORM_UPDATE`),
//! which re-fits the existing tree's bounding volumes in place instead of
//! rebuilding it from scratch.
//!
//! A refit keeps the tree the last full build produced, so traversal quality
//! decays as the pose drifts away from the one that tree was built for; this
//! module bounds that with a periodic full rebuild. It owns only the pure
//! decision -- the descriptors, the allocation and the recorded build are
//! per-backend (directx/raytrace.rs, vulkan/raytrace.rs). Split out so the
//! cadence is unit-testable without a GPU.
//!
//! Consumed by the DirectX + Vulkan backends. The Metal backend keeps its own
//! equivalent copy (metal/rt_ring.rs), the same split `rt_topology` already has.

use alloc::vec::Vec;

/// Full rebuilds per ring slot: after this many consecutive refits the next
/// skinned update rebuilds that slot's BLAS from scratch. Each slot counts
/// independently and they are touched on different frames, so the rebuilds
/// stagger rather than landing on one frame.
pub const REFIT_LIMIT: u32 = 32;

/// The geometry one skinned BLAS is built over: its slice of the shared skinned
/// index buffer plus the vertex range the deformed buffer spans. Equal
/// signatures mean the same triangles addressing the same vertex range with only
/// the positions moved, which is exactly when a refit is legal; anything else (a
/// mesh hot-reload, a different mesh becoming visible, a grown deformed buffer)
/// changes the geometry description and needs a full rebuild. `vertex_extent` is
/// carried because both APIs require the vertex count to match the structure
/// being updated, even though the vertex buffer's address may move.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct SkinnedShape {
    /// First index of this slot's range in the shared index buffer.
    pub index_offset: usize,
    /// Indices in this slot's range.
    pub index_count: usize,
    /// Vertices the slot's range spans.
    pub vertex_extent: u32,
}

/// Whether a slot's skinned BLAS can be refit from this frame's pose or must be
/// rebuilt from scratch.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlasUpdate {
    /// Rebuild the acceleration structure from scratch.
    Build,
    /// Refit the existing acceleration structure in place.
    Refit,
}

// Rebuild when the geometry changed (a refit is then illegal), when the slot has
// no built tree to refit, or once every `limit` refits to bound the quality
// drift. Pure so the cadence is unit-testable without a GPU.
fn blas_update(shape_changed: bool, built: bool, refits: u32, limit: u32) -> BlasUpdate {
    if shape_changed || !built || refits >= limit {
        BlasUpdate::Build
    } else {
        BlasUpdate::Refit
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
    pub fn plan(&mut self, shapes: &[SkinnedShape], storage_changed: bool) -> BlasUpdate {
        let changed = storage_changed || self.shapes != shapes;
        let update = blas_update(changed, self.built, self.refits, REFIT_LIMIT);
        if changed {
            self.shapes.clear();
            self.shapes.extend_from_slice(shapes);
        }
        match update {
            BlasUpdate::Build => {
                self.built = true;
                self.refits = 0;
            }
            BlasUpdate::Refit => self.refits += 1,
        }
        update
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
        assert_eq!(blas_update(true, true, 0, 32), BlasUpdate::Build);
    }

    #[test]
    fn an_unbuilt_slot_cannot_be_refit() {
        assert_eq!(blas_update(false, false, 0, 32), BlasUpdate::Build);
    }

    #[test]
    fn a_stable_shape_refits_until_the_limit() {
        assert_eq!(blas_update(false, true, 0, 32), BlasUpdate::Refit);
        assert_eq!(blas_update(false, true, 31, 32), BlasUpdate::Refit);
        // The 32nd refit is instead a rebuild, bounding the quality drift.
        assert_eq!(blas_update(false, true, 32, 32), BlasUpdate::Build);
        assert_eq!(blas_update(false, true, 99, 32), BlasUpdate::Build);
    }

    #[test]
    fn a_zero_limit_never_refits() {
        assert_eq!(blas_update(false, true, 0, 0), BlasUpdate::Build);
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
        assert_eq!(slot.plan(&shapes, false), BlasUpdate::Build);
        assert_eq!(slot.plan(&shapes, false), BlasUpdate::Refit);
        assert_eq!(slot.plan(&shapes, false), BlasUpdate::Refit);
    }

    #[test]
    fn reallocated_storage_rebuilds_even_on_an_unchanged_shape() {
        let mut slot = SkinnedRefit::default();
        let shapes = [shape(0)];
        assert_eq!(slot.plan(&shapes, false), BlasUpdate::Build);
        assert_eq!(slot.plan(&shapes, true), BlasUpdate::Build);
        assert_eq!(slot.plan(&shapes, false), BlasUpdate::Refit);
    }

    #[test]
    fn a_changed_object_set_rebuilds_and_restarts_the_run() {
        let mut slot = SkinnedRefit::default();
        assert_eq!(slot.plan(&[shape(0)], false), BlasUpdate::Build);
        assert_eq!(slot.plan(&[shape(0)], false), BlasUpdate::Refit);
        // A second skinned object became visible: one more BLAS, so a full build.
        assert_eq!(slot.plan(&[shape(0), shape(1)], false), BlasUpdate::Build);
        assert_eq!(slot.plan(&[shape(0), shape(1)], false), BlasUpdate::Refit);
    }

    #[test]
    fn the_refit_run_is_bounded_by_the_limit() {
        let mut slot = SkinnedRefit::default();
        let shapes = [shape(0)];
        assert_eq!(slot.plan(&shapes, false), BlasUpdate::Build);
        for _ in 0..REFIT_LIMIT {
            assert_eq!(slot.plan(&shapes, false), BlasUpdate::Refit);
        }
        // The run has reached the limit: the next update is a full rebuild, and
        // the run then restarts.
        assert_eq!(slot.plan(&shapes, false), BlasUpdate::Build);
        assert_eq!(slot.plan(&shapes, false), BlasUpdate::Refit);
    }

    #[test]
    fn reset_makes_the_next_update_a_build() {
        let mut slot = SkinnedRefit::default();
        let shapes = [shape(0)];
        assert_eq!(slot.plan(&shapes, false), BlasUpdate::Build);
        assert_eq!(slot.plan(&shapes, false), BlasUpdate::Refit);
        slot.reset();
        assert_eq!(slot.plan(&shapes, false), BlasUpdate::Build);
    }
}
