//! The decal slot table the backends share. Holds one slot per decal id with a
//! tombstone free-list, caches the world AABB the frustum cull tests and the
//! uniform block the pass uploads, and tracks which frame-in-flight copy of
//! that block is still stale.

use core::cell::Cell;

use crate::gfx::frustum::Frustum;
use crate::render::decal::DecalRecord;
use crate::render::frame_dirty::FrameDirty;
use crate::render::slot_table::{AtCapacity, RemoveError, SlotTable};
use crate::render::uniforms::DecalParams;

/// Exponent of the decal's edge-fade curve, applied by the fragment shader to
/// the distance from the projection box's faces.
const EDGE_FADE_POW: f32 = 2.0;

// One live decal. `aabb_*` and `params` are derived from `record` at insert and
// never recomputed: the pass reads them every frame, and a decal's transform is
// fixed for the life of its slot.
struct Slot {
    record: DecalRecord,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    params: DecalParams,
    dirty: Cell<FrameDirty>,
}

impl Slot {
    fn new(record: DecalRecord, frames: usize) -> Self {
        let (aabb_min, aabb_max) = record.aabb();
        Self {
            aabb_min,
            aabb_max,
            params: DecalParams {
                model: record.model,
                inv_model: record.inv_model,
                tint: record.tint,
                fade_pow: EDGE_FADE_POW,
                _pad0: 0.0,
                _pad1: 0.0,
                _pad2: 0.0,
            },
            record,
            dirty: Cell::new(FrameDirty::new(frames)),
        }
    }
}

/// The decals a backend's projected-decal pass draws, addressed by the slot id
/// [`DecalSet::insert`] returns. That id is stable until [`DecalSet::remove`]
/// tombstones it, and indexes the backend's parallel per-decal resources (an
/// albedo descriptor, a uniform ring slot).
///
/// `capacity` caps the slot table at the backend's reserved descriptor count;
/// `frames` is its frames in flight, which sizes the per-slot upload tracking.
///
/// ```rust
/// # use concinnity_core::transform::IDENTITY;
/// # use concinnity_core::render::decal::{DecalRecord, DecalSet};
/// # let record = DecalRecord {
/// #     model: IDENTITY,
/// #     inv_model: IDENTITY,
/// #     texture_slot: 0,
/// #     tint: [1.0; 4],
/// # };
/// let mut decals = DecalSet::new(2, 2);
/// let id = decals.insert(record).expect("room for two");
/// decals.remove(id).expect("live slot");
/// assert!(decals.is_empty());
/// ```
pub struct DecalSet {
    slots: SlotTable<Slot>,
    frames: usize,
}

impl DecalSet {
    /// An empty set holding at most `capacity` slots, tracking uploads for
    /// `frames` frames in flight.
    pub fn new(capacity: usize, frames: usize) -> Self {
        Self {
            slots: SlotTable::new(capacity),
            frames,
        }
    }

    /// Place `record` in a slot and return its id, reusing a tombstoned slot
    /// before growing the table so a spawn / despawn cycle stays bounded. The
    /// new slot is marked stale for every frame in flight.
    pub fn insert(&mut self, record: DecalRecord) -> Result<usize, AtCapacity> {
        self.slots.insert(Slot::new(record, self.frames))
    }

    /// Tombstone the slot at `id`, freeing it for the next [`Self::insert`].
    pub fn remove(&mut self, id: usize) -> Result<(), RemoveError> {
        self.slots.remove(id).map(drop)
    }

    /// Carry every live decal into the frame `rebase` moves the world to,
    /// keeping its id; each slot is re-uploaded for every frame in flight.
    pub fn rebase(&mut self, rebase: &crate::planet::Rebase) {
        let frames = self.frames;
        for (_, slot) in self.slots.iter_mut() {
            let mut record = slot.record;
            record.model = rebase.apply_matrix(record.model);
            record.inv_model =
                crate::transform::mat4_mul(record.inv_model, rebase.inverse_matrix());
            *slot = Slot::new(record, frames);
        }
    }

    /// Whether no slot is live, so the pass can be skipped outright.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// The live decals whose cached world AABB meets `frustum`, in slot order.
    /// Each decal is tested once: a caller that needs to know whether the pass
    /// draws anything peeks this iterator rather than counting it first.
    pub fn visible<'a>(
        &'a self,
        frustum: &'a Frustum,
    ) -> impl Iterator<Item = VisibleDecal<'a>> + 'a {
        self.slots.iter().filter_map(move |(id, slot)| {
            if !frustum.intersects_aabb(slot.aabb_min, slot.aabb_max) {
                return None;
            }
            Some(VisibleDecal {
                id,
                record: &slot.record,
                params: &slot.params,
                dirty: &slot.dirty,
            })
        })
    }
}

/// A decal that survived the frustum cull, with the per-decal inputs its draw
/// needs.
pub struct VisibleDecal<'a> {
    /// The decal's slot id, indexing the backend's parallel per-decal
    /// resources.
    pub id: usize,
    /// The decal's record.
    pub record: &'a DecalRecord,
    /// The uniform block the pass binds for this decal.
    pub params: &'a DecalParams,
    dirty: &'a Cell<FrameDirty>,
}

impl VisibleDecal<'_> {
    /// Whether frame `frame`'s copy of [`Self::params`] is stale, clearing the
    /// flag if so. A backend whose ring slot survives the frame writes only
    /// when this reports true; one that re-supplies the block inline with every
    /// draw ignores it.
    pub fn take_upload(&self, frame: usize) -> bool {
        let mut dirty = self.dirty.get();
        let pending = dirty.take(frame);
        self.dirty.set(dirty);
        pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    // A frame move carries every live decal's box and its inverse to where
    // the move says, keeps ids and tombstones, and re-arms the uploads.
    #[test]
    fn a_frame_move_carries_every_live_decal() {
        use crate::planet::Rebase;
        let rebase = Rebase {
            rotation: crate::math::quat_from_axis_angle([1.0, 0.0, 0.0], 0.02),
            translation: [-1_000.0, 5.0, 3.0],
        };
        let mut set = DecalSet::new(4, 2);
        let a = set.insert(record_at([1_002.0, 0.0, -1.0])).unwrap();
        let gone = set.insert(record_at([0.0; 3])).unwrap();
        set.remove(gone).unwrap();
        let uploaded = &set.slots.get(a).unwrap().dirty;
        let mut d = uploaded.get();
        d.take(0);
        uploaded.set(d);
        set.rebase(&rebase);

        let slot = set.slots.get(a).unwrap();
        let corner = |m: crate::transform::Mat4, p: [f32; 3]| -> [f32; 3] {
            core::array::from_fn(|i| m[0][i] * p[0] + m[1][i] * p[1] + m[2][i] * p[2] + m[3][i])
        };
        let moved = corner(slot.record.model, [0.5, 0.5, 0.5]);
        let expected = rebase.apply_point([1_002.5, 0.5, -0.5]);
        assert!(
            (0..3).all(|i| (moved[i] - expected[i]).abs() < 1e-3),
            "{moved:?}"
        );
        let back = corner(slot.record.inv_model, moved);
        assert!((0..3).all(|i| (back[i] - 0.5).abs() < 1e-3), "{back:?}");
        assert_eq!(slot.params.model, slot.record.model);
        assert!(
            set.slots.get(gone).is_none(),
            "a tombstone stays a tombstone"
        );
        let mut dirty = slot.dirty.get();
        assert!(dirty.take(0), "re-uploaded");
    }
    use crate::render::decal::decal_model_matrix;
    use crate::render::decal::invert_decal_model;

    fn record_at(position: [f32; 3]) -> DecalRecord {
        let model = decal_model_matrix(position, [0.0; 3], [1.0; 3]);
        DecalRecord {
            model,
            inv_model: invert_decal_model(model).expect("unit decal inverts"),
            texture_slot: 0,
            tint: [1.0; 4],
        }
    }

    // A frustum that admits everything: six planes whose normals all point
    // inward from far outside the sampled range.
    fn frustum_containing_everything() -> Frustum {
        use crate::gfx::frustum::Plane;
        let plane = |normal: [f32; 3]| Plane { normal, d: 1.0e6 };
        Frustum {
            planes: [
                plane([1.0, 0.0, 0.0]),
                plane([-1.0, 0.0, 0.0]),
                plane([0.0, 1.0, 0.0]),
                plane([0.0, -1.0, 0.0]),
                plane([0.0, 0.0, 1.0]),
                plane([0.0, 0.0, -1.0]),
            ],
        }
    }

    // A frustum admitting only points with x <= 10: the +X plane's half-space.
    fn frustum_left_of_ten() -> Frustum {
        use crate::gfx::frustum::Plane;
        let wide = |normal: [f32; 3]| Plane { normal, d: 1.0e6 };
        let mut planes = frustum_containing_everything().planes;
        planes[0] = Plane {
            normal: [-1.0, 0.0, 0.0],
            d: 10.0,
        };
        planes[1] = wide([1.0, 0.0, 0.0]);
        Frustum { planes }
    }

    #[test]
    fn an_empty_set_is_empty_and_draws_nothing() {
        let decals = DecalSet::new(4, 2);
        assert!(decals.is_empty());
        assert_eq!(decals.visible(&frustum_containing_everything()).count(), 0);
    }

    #[test]
    fn a_removed_slot_is_reused_before_the_table_grows() {
        let mut decals = DecalSet::new(2, 2);
        let first = decals.insert(record_at([0.0; 3])).expect("first slot");
        decals.insert(record_at([0.0; 3])).expect("second slot");
        decals.remove(first).expect("live slot");
        assert_eq!(decals.insert(record_at([0.0; 3])), Ok(first));
        // The table never grew past its capacity, so a third add still fails.
        assert_eq!(decals.insert(record_at([0.0; 3])), Err(AtCapacity));
    }

    #[test]
    fn remove_rejects_unknown_and_repeated_ids() {
        let mut decals = DecalSet::new(2, 2);
        assert_eq!(decals.remove(0), Err(RemoveError::OutOfRange));
        let id = decals.insert(record_at([0.0; 3])).expect("free slot");
        assert_eq!(decals.remove(id), Ok(()));
        assert_eq!(decals.remove(id), Err(RemoveError::AlreadyRemoved));
        assert!(decals.is_empty());
    }

    #[test]
    fn only_slots_meeting_the_frustum_are_visible() {
        let mut decals = DecalSet::new(4, 2);
        let near = decals.insert(record_at([0.0; 3])).expect("free slot");
        let far = decals
            .insert(record_at([100.0, 0.0, 0.0]))
            .expect("free slot");
        let frustum = frustum_left_of_ten();
        let ids: Vec<usize> = decals.visible(&frustum).map(|d| d.id).collect();
        assert_eq!(ids, alloc::vec![near]);
        // The culled slot is still live: a wider frustum draws both.
        let all: Vec<usize> = decals
            .visible(&frustum_containing_everything())
            .map(|d| d.id)
            .collect();
        assert_eq!(all, alloc::vec![near, far]);
    }

    #[test]
    fn a_tombstoned_slot_never_becomes_visible() {
        let mut decals = DecalSet::new(4, 2);
        let id = decals.insert(record_at([0.0; 3])).expect("free slot");
        decals.remove(id).expect("live slot");
        assert_eq!(decals.visible(&frustum_containing_everything()).count(), 0);
    }

    #[test]
    fn visible_carries_the_cached_params_of_its_record() {
        let mut decals = DecalSet::new(4, 2);
        let record = record_at([1.0, 2.0, 3.0]);
        decals.insert(record).expect("free slot");
        let frustum = frustum_containing_everything();
        let decal = decals.visible(&frustum).next().expect("one visible decal");
        assert_eq!(decal.record.model, record.model);
        assert_eq!(decal.params.model, record.model);
        assert_eq!(decal.params.inv_model, record.inv_model);
        assert_eq!(decal.params.tint, record.tint);
        assert_eq!(decal.params.fade_pow, EDGE_FADE_POW);
    }

    #[test]
    fn a_new_slot_uploads_once_per_frame_in_flight() {
        let mut decals = DecalSet::new(4, 3);
        decals.insert(record_at([0.0; 3])).expect("free slot");
        let frustum = frustum_containing_everything();
        for frame in 0..3 {
            let decal = decals.visible(&frustum).next().expect("visible");
            assert!(decal.take_upload(frame), "frame {frame} seeds stale");
        }
        // Steady state: the ring has caught up and every frame writes nothing.
        for frame in (0..3).cycle().take(9) {
            let decal = decals.visible(&frustum).next().expect("visible");
            assert!(!decal.take_upload(frame), "frame {frame} stays clean");
        }
    }

    #[test]
    fn a_reused_slot_is_stale_again_for_every_frame() {
        let mut decals = DecalSet::new(4, 2);
        let id = decals.insert(record_at([0.0; 3])).expect("free slot");
        let frustum = frustum_containing_everything();
        for frame in 0..2 {
            assert!(
                decals
                    .visible(&frustum)
                    .next()
                    .expect("visible")
                    .take_upload(frame)
            );
        }
        decals.remove(id).expect("live slot");
        assert_eq!(decals.insert(record_at([5.0, 0.0, 0.0])), Ok(id));
        for frame in 0..2 {
            let decal = decals.visible(&frustum).next().expect("visible");
            assert!(
                decal.take_upload(frame),
                "frame {frame} re-armed by the add"
            );
        }
    }

    #[test]
    fn a_decal_culled_this_frame_keeps_its_pending_upload() {
        let mut decals = DecalSet::new(4, 2);
        decals
            .insert(record_at([100.0, 0.0, 0.0]))
            .expect("free slot");
        // Culled: the pass never reaches it, so nothing is taken.
        assert_eq!(decals.visible(&frustum_left_of_ten()).count(), 0);
        let frustum = frustum_containing_everything();
        for frame in 0..2 {
            let decal = decals.visible(&frustum).next().expect("visible");
            assert!(decal.take_upload(frame), "frame {frame} still owes a write");
        }
    }
}
