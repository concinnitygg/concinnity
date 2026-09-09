// src/vulkan/post/set_arena.rs
//
// Descriptor sets for the shared fullscreen post passes, allocated per frame
// rather than per pass.
//
// The pattern this replaces is a pool and a pre-wired set per frame in flight
// owned by every effect, plus a `rewire_*` for each input that can change which
// image it points at. That is where a Vulkan post pass grew its bulk and its
// coupling: an effect had to be told about every other effect that might own its
// scene input. A set allocated at encode time is written from what the pass
// actually holds this frame, so there is nothing to rewire and no effect has to
// know about another.
//
// One pool per frame in flight, reset when the frame comes round again. The
// reset is safe without a fence of its own: `begin_frame` is called from the top
// of the frame, after that slot's fence wait, so every set the pool handed out
// last time round has retired.

use std::sync::Mutex;

use ash::vk;

use crate::vulkan::owned::{OwnedDescriptorPool, VkDevice};

// Sets one frame's post passes may allocate. Six fullscreen post passes exist,
// none allocating more than one set per frame, so this is roughly double the
// ceiling and leaves room for a pass to gain a second.
const SETS_PER_FRAME: u32 = 16;

// Combined image samplers those sets may hold in total. The widest post pass
// binds a handful of sources, so eight per set covers every one of them.
const SAMPLERS_PER_FRAME: u32 = SETS_PER_FRAME * 8;

// A per-frame descriptor pool ring for the shared post passes.
pub(in crate::vulkan) struct PostSetArena {
    // One pool per frame in flight, plus the frame tick it was last reset on.
    // `Mutex` because the graph executor allocates from worker threads holding
    // `&self`, and Vulkan requires external synchronisation over a pool.
    slots: Vec<Mutex<PoolSlot>>,
}

struct PoolSlot {
    pool: OwnedDescriptorPool,
}

impl PostSetArena {
    // A pool per frame in flight.
    pub(in crate::vulkan) fn new(device: &VkDevice, frames: usize) -> Result<Self, String> {
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(SAMPLERS_PER_FRAME)];
        let mut slots = Vec::with_capacity(frames.max(1));
        for _ in 0..frames.max(1) {
            let pool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .pool_sizes(&sizes)
                        .max_sets(SETS_PER_FRAME),
                )
                .map_err(|e| format!("post descriptor pool: {e}"))?;
            slots.push(Mutex::new(PoolSlot { pool }));
        }
        Ok(Self { slots })
    }

    // Reclaim frame slot `frame`'s sets. Called from the top of the frame, after
    // that slot's fence wait, which is what makes the reclaim legal.
    pub(in crate::vulkan) fn begin_frame(&self, device: &VkDevice, frame: usize) {
        let Ok(slot) = self.slots[frame % self.slots.len()].lock() else {
            tracing::error!("post descriptor arena poisoned");
            return;
        };
        // SAFETY: the caller waited on this frame slot's fence, so every set this pool handed out
        // on its previous pass has retired; the pool belongs to this device and the lock makes this
        // the only thread touching it.
        let reset = unsafe {
            device.reset_descriptor_pool(slot.pool.handle(), vk::DescriptorPoolResetFlags::empty())
        };
        if let Err(e) = reset {
            tracing::error!("post descriptor pool reset: {e}");
        }
    }

    // One set of `layout` from `frame`'s pool.
    pub(in crate::vulkan) fn alloc(
        &self,
        device: &VkDevice,
        frame: usize,
        layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet, String> {
        let slot = self.slots[frame % self.slots.len()]
            .lock()
            .map_err(|_| "post descriptor arena poisoned".to_string())?;
        let layouts = [layout];
        let sets =
            crate::vulkan::resources::alloc_descriptor_sets(device, slot.pool.handle(), &layouts)?;
        sets.into_iter()
            .next()
            .ok_or_else(|| "post descriptor arena returned no set".to_string())
    }
}
