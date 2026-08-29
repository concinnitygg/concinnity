// src/gfx/render_slots.rs
//
// Engine-side allocation authority for backend draw slots and pre-reserved
// skinned instances. Systems allocate here and record the destination into
// their backend ops, so slot decisions never require a synchronous backend
// round trip; the backend only writes at the slot an op names. Correct
// because ops replay exactly once, in record order: an `Append` index always
// matches the backend's draw-object count when its op applies.

use concinnity_core::render::draw_slot::{DrawSlotAllocator, SlotAlloc};
use concinnity_core::render::skinned_pool::SkinnedInstancePool;

// World resource: the draw-slot free list plus the skinned instance pool.
// Published by graphics init once the backend's build-time draw count and
// capabilities are known.
pub(crate) struct RenderSlots {
    draws: DrawSlotAllocator,
    skinned: SkinnedInstancePool,
    // Slots below this index are never recycled: a backend whose cull BVH and
    // RT tables key fixed build-time indices cannot refit a reused one (see
    // `DeviceCapabilities::reuses_build_slots`). Zero when the backend
    // re-admits recycled build-time slots.
    reuse_floor: usize,
}

impl RenderSlots {
    pub(crate) fn new(
        build_draw_count: usize,
        reuses_build_slots: bool,
        skinned_reservations: &[(usize, usize)],
    ) -> Self {
        let mut skinned = SkinnedInstancePool::new();
        for &(template, instance) in skinned_reservations {
            skinned.reserve(template, instance);
        }
        Self {
            draws: DrawSlotAllocator::with_len(build_draw_count),
            skinned,
            reuse_floor: if reuses_build_slots {
                0
            } else {
                build_draw_count
            },
        }
    }

    // Hand out a draw slot for a runtime clone or streamed chunk.
    pub(crate) fn allocate_draw(&mut self) -> SlotAlloc {
        self.draws.allocate()
    }

    // Return a retired draw slot for reuse. Build-time slots stay allocated
    // on backends that cannot refit them (`reuse_floor`).
    pub(crate) fn free_draw(&mut self, slot: usize) {
        if slot >= self.reuse_floor {
            self.draws.free(slot);
        }
    }

    // Claim a free pre-reserved skinned instance of `template`, or `None`
    // when the reserve is exhausted.
    pub(crate) fn claim_skinned(&mut self, template: usize) -> Option<usize> {
        self.skinned.acquire(template)
    }

    // Return a live skinned instance to its template's pool. False if the
    // slot was never a pre-reserved instance (an authored template slot).
    pub(crate) fn release_skinned(&mut self, instance: usize) -> bool {
        self.skinned.release(instance)
    }

    // Total free skinned instances, for the profiler's pool chip.
    pub(crate) fn skinned_free(&self) -> usize {
        self.skinned.total_free()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_slots_recycle_only_above_the_floor() {
        // A backend that cannot refit build-time slots: floor = build count.
        let mut slots = RenderSlots::new(4, false, &[]);
        slots.free_draw(2);
        assert_eq!(
            slots.allocate_draw(),
            SlotAlloc::Append(4),
            "a freed build-time slot is not recycled"
        );
        slots.free_draw(4);
        assert_eq!(
            slots.allocate_draw(),
            SlotAlloc::Reuse(4),
            "a runtime-append slot is recycled"
        );
    }

    #[test]
    fn build_slots_recycle_everywhere_without_the_floor() {
        let mut slots = RenderSlots::new(4, true, &[]);
        slots.free_draw(2);
        assert_eq!(slots.allocate_draw(), SlotAlloc::Reuse(2));
    }

    #[test]
    fn skinned_pool_claims_and_releases_per_template() {
        let mut slots = RenderSlots::new(0, true, &[(0, 5), (0, 6), (3, 9)]);
        assert_eq!(slots.skinned_free(), 3);
        let a = slots.claim_skinned(0).expect("first copy");
        let b = slots.claim_skinned(0).expect("second copy");
        assert!(slots.claim_skinned(0).is_none(), "reserve exhausted");
        assert!(slots.release_skinned(a));
        assert_eq!(slots.claim_skinned(0), Some(a), "released slot recycles");
        assert!(
            !slots.release_skinned(42),
            "unknown slot is not a pool slot"
        );
        let _ = b;
        assert_eq!(slots.claim_skinned(3), Some(9));
    }
}
