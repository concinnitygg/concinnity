// src/vulkan/owned.rs
//
// Owning handles for the Vulkan objects the backend creates once and keeps:
// pipelines, pipeline layouts, descriptor set layouts, descriptor pools, render
// passes, framebuffers and samplers.
//
// ash spells every one of those as a `Copy` u64 newtype. That is why the
// `destroy_*` family and the command-recording family could not be given safe
// wrappers: with a `Copy` handle a safe `destroy(handle)` permits a second
// destroy, and a safe `bind(handle)` permits binding one that was already
// destroyed. Both are reachable from safe code, so both APIs would be lying.
// Wrapping each handle in a single-owner, non-`Copy` value is what removes the
// obligation from the call site, and it removes it for both families at once:
// destruction happens in `Drop`, and a recorded bind borrows the owner.
//
// Destruction is deferred, not immediate. Dropping an owned handle queues it on
// the device's retire list, and the queue destroys it once `frames_in_flight +
// 1` frame ticks have passed, by which point no submission that could still
// name it is in flight. That is the discipline `allocator.rs` already applies to
// pooled buffers and images, and it is what makes replacing a live pipeline
// (shader hot-reload, lazily built wireframe twins, a quality-toggle rebuild)
// correct without the caller reasoning about GPU progress. The alternative --
// destroying in `Drop` and leaning on the existing "wait_idle, then tear down"
// ordering -- would have left the safe API still lying: `drop(pipeline)`
// followed by `submit(cmd)` compiles either way, and only the deferred queue
// makes it harmless.
//
// [`VkDevice`] owns the logical device, so the device outlives every handle by
// construction rather than by field ordering: the last owner to drop drains the
// retire queue and only then calls `vkDestroyDevice`. A live editor
// `reload_world` inherits the device by cloning the handle, so the outgoing
// context's retiring objects are drained by its successor.

use std::sync::{Arc, Mutex};

use ash::vk;

// One retired object, tagged with the call that destroys it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Retired {
    Pipeline(vk::Pipeline),
    PipelineLayout(vk::PipelineLayout),
    SetLayout(vk::DescriptorSetLayout),
    DescriptorPool(vk::DescriptorPool),
    RenderPass(vk::RenderPass),
    Framebuffer(vk::Framebuffer),
    Sampler(vk::Sampler),
}

impl Retired {
    // Destroy the object.
    //
    // # Safety
    // The handle must have come from `device`, must not have been destroyed
    // already, and no submission that names it may still be in flight. The
    // retire queue below is what establishes all three: it holds each handle
    // exactly once, only owned wrappers built from this device push into it,
    // and it withholds every entry for `retire_depth` frame ticks.
    unsafe fn destroy(self, device: &ash::Device) {
        // SAFETY: the caller upholds provenance, single-destruction and quiescence; see above.
        unsafe {
            match self {
                Retired::Pipeline(h) => device.destroy_pipeline(h, None),
                Retired::PipelineLayout(h) => device.destroy_pipeline_layout(h, None),
                Retired::SetLayout(h) => device.destroy_descriptor_set_layout(h, None),
                Retired::DescriptorPool(h) => device.destroy_descriptor_pool(h, None),
                Retired::RenderPass(h) => device.destroy_render_pass(h, None),
                Retired::Framebuffer(h) => device.destroy_framebuffer(h, None),
                Retired::Sampler(h) => device.destroy_sampler(h, None),
            }
        }
    }
}

// A handle waiting out its retire window.
#[derive(Clone, Copy)]
struct Pending {
    handle: Retired,
    retire_at: u64,
}

// The frame-tick bookkeeping behind the deferred destruction, with no Vulkan in
// it so the window arithmetic is testable on its own.
struct RetireQueue {
    pending: Vec<Pending>,
    // Monotonic frame tick, not the wrapping frame-in-flight index.
    frame: u64,
    // How many ticks a handle is withheld for: `frames_in_flight + 1`, matching
    // the device allocator, so a handle replaced between frames outlives the
    // submission that was already in flight when it was replaced.
    depth: u64,
}

impl RetireQueue {
    fn new(frames_in_flight: usize) -> Self {
        Self {
            pending: Vec::new(),
            frame: 0,
            depth: frames_in_flight as u64 + 1,
        }
    }

    fn push(&mut self, handle: Retired) {
        self.pending.push(Pending {
            handle,
            retire_at: self.frame + self.depth,
        });
    }

    // Advance one frame and return everything whose window has closed.
    fn tick(&mut self) -> Vec<Retired> {
        self.frame += 1;
        let frame = self.frame;
        let mut due = Vec::new();
        self.pending.retain(|p| {
            if p.retire_at <= frame {
                due.push(p.handle);
                false
            } else {
                true
            }
        });
        due
    }

    // Return everything queued regardless of its window. Only for a caller that
    // has already idled the device.
    fn drain(&mut self) -> Vec<Retired> {
        std::mem::take(&mut self.pending)
            .into_iter()
            .map(|p| p.handle)
            .collect()
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.pending.len()
    }
}

// The logical device plus the retire queue every owned handle drains into.
// Owns `vkDestroyDevice` and `vkDestroyInstance`, so it keeps the entry and the
// instance alive for exactly as long as the device it loaded them for.
struct DeviceInner {
    raw: ash::Device,
    instance: ash::Instance,
    // The loader must outlive the instance it produced; nothing reads it.
    _entry: ash::Entry,
    queue: Mutex<RetireQueue>,
    debug: DebugMessenger,
}

// The validation messenger, and the budget its callback reads.
//
// Owned by the device rather than by `VkContext` so that it outlives
// `vkDestroyDevice`. The object-tracking leak report the layer emits from inside
// that call is exactly the diagnostic a teardown bug produces, and a messenger
// destroyed any earlier never sees it: the message falls back to the layer's own
// output, with no `[Vulkan]` prefix for the log scan to find.
pub(in crate::vulkan) struct DebugMessenger {
    pub(in crate::vulkan) utils: Option<ash::ext::debug_utils::Instance>,
    pub(in crate::vulkan) messenger: Option<vk::DebugUtilsMessengerEXT>,
    // Boxed so the address baked into the messenger's `user_data` stays put.
    // Held here for the same reason as the messenger: the callback reads it for
    // as long as the messenger can fire, which is now past the device teardown.
    pub(in crate::vulkan) filter: Option<Box<std::sync::atomic::AtomicU32>>,
}

impl DeviceInner {
    // The retire queue. Poison is ignored: the thread that panicked while
    // holding it has already failed, and refusing the queue here would strand
    // every handle waiting in it.
    fn queue(&self) -> std::sync::MutexGuard<'_, RetireQueue> {
        self.queue.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Drop for DeviceInner {
    fn drop(&mut self) {
        // Every owned handle has dropped by now: they hold `VkDevice` clones,
        // and this is the last one. Destroy what they queued, then the hardware.
        // Take the entries out before destroying any, so the queue lock is not
        // held across the `destroy_*` calls.
        let due = self.queue().drain();
        for handle in due {
            // SAFETY: the queue holds each handle exactly once, every entry came from this device,
            // and the last owner dropping means nothing can submit against them any more.
            unsafe { handle.destroy(&self.raw) };
        }
        // Persist and destroy the device's pipeline cache while the device
        // still exists. A lost device fails the serialize inside and keeps the
        // blob already on disk.
        super::pipeline_cache::shutdown(&self.raw);
        // SAFETY: destroyed exactly once, when the last handle to it drops; every object created
        // from it has been destroyed above or by the context's own teardown, which runs first.
        unsafe { self.raw.destroy_device(None) };
        // After the device, so the messenger is still installed for anything the
        // layer reports from inside `vkDestroyDevice` -- object-tracking leaks
        // land exactly there.
        if let (Some(du), Some(dm)) = (&self.debug.utils, self.debug.messenger) {
            // SAFETY: the handle was created from this instance and is destroyed exactly once,
            // here, before the instance that produced it.
            unsafe { du.destroy_debug_utils_messenger(dm, None) };
        }
        // SAFETY: destroyed exactly once, after the device it produced and after the messenger;
        // the surface is an instance-level child destroyed in `VkContext::drop`, which runs before
        // any field of the context drops.
        unsafe { self.instance.destroy_instance(None) };
    }
}

// A shared handle to the logical device. Clone is handle semantics: every clone
// names one device and one retire queue, and the device lives until the last of
// them drops.
//
// The refcount and the retire queue are both thread-safe (`Arc` + `Mutex`), and
// that is load-bearing rather than defensive: `VkContext` is `unsafe impl Send`
// and its render-graph pass encoders run on the parallel encoder's worker
// threads, where several of them clone this handle. A non-atomic `Rc` there
// races the refcount -- a lost decrement destroys the device while owned handles
// are still live, which shows up as objects reported alive at `vkDestroyDevice`
// and an access violation on the way out.
//
// Derefs to `ash::Device`, so anything that only reads or records through the
// device keeps its existing spelling; the inherent methods below shadow the ash
// ones that create an object this module owns, which is deliberate. A call that
// still wants the raw creation has to say so.
#[derive(Clone)]
pub(in crate::vulkan) struct VkDevice {
    inner: Arc<DeviceInner>,
}

// Worker threads clone this handle, so the property has to be enforced rather
// than assumed: `VkContext`'s `unsafe impl Send` would otherwise hide a
// thread-unsafe field, which is exactly how the `Rc` above survived review.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<VkDevice>();
};

impl std::ops::Deref for VkDevice {
    type Target = ash::Device;

    fn deref(&self) -> &ash::Device {
        &self.inner.raw
    }
}

impl VkDevice {
    // Take ownership of a freshly created device. The entry and instance are
    // held for the device's whole life and destroyed after it.
    pub(in crate::vulkan) fn new(
        entry: ash::Entry,
        instance: ash::Instance,
        raw: ash::Device,
        frames_in_flight: usize,
        debug: DebugMessenger,
    ) -> Self {
        Self {
            inner: Arc::new(DeviceInner {
                raw,
                instance,
                _entry: entry,
                queue: Mutex::new(RetireQueue::new(frames_in_flight)),
                debug,
            }),
        }
    }

    // The messenger callback's benign-error budget, for the two callers that
    // arm or read it.
    pub(in crate::vulkan) fn debug_filter(&self) -> Option<&std::sync::atomic::AtomicU32> {
        self.inner.debug.filter.as_deref()
    }

    // Advance the retire clock and destroy everything whose window has closed.
    // Called once per frame alongside `DeviceAllocator::begin_frame`.
    pub(in crate::vulkan) fn begin_frame(&self) {
        let due = self.inner.queue().tick();
        self.destroy_all(due);
    }

    // Destroy everything queued without waiting out the window. The caller must
    // have idled the device; used by the live editor reload, which idles before
    // it frees the old world so the successor rebuilds into freed memory.
    pub(in crate::vulkan) fn reclaim_idle(&self) {
        let due = self.inner.queue().drain();
        self.destroy_all(due);
    }

    fn destroy_all(&self, due: Vec<Retired>) {
        for handle in due {
            // SAFETY: the queue holds each handle exactly once and every entry came from this
            // device; `retire_depth` frame ticks have passed since it was queued (or the caller
            // idled the device), so no submission still names it.
            unsafe { handle.destroy(&self.inner.raw) };
        }
    }

    fn retire(&self, handle: Retired) {
        self.inner.queue().push(handle);
    }
}

// Generate the safe creation call for one owned family. Each shadows the ash
// method of the same name, which is the point: a `&VkDevice` in scope can only
// reach the owning version, and reaching the raw one takes an explicit deref.
//
// Safe because the create-info is a borrow that outlives the call and every
// handle it names is a live borrow of another owned object, and because the
// result is owned rather than handed back as a bare handle.
macro_rules! create_owned {
    ($name:ident, $info:ty, $owned:ident) => {
        impl VkDevice {
            pub(in crate::vulkan) fn $name(&self, info: &$info) -> Result<$owned, vk::Result> {
                // SAFETY: the create-info and every slice it borrows are live for the call, and
                // each handle it names belongs to this device.
                let handle = unsafe { self.inner.raw.$name(info, None) }?;
                Ok($owned::new(self, handle))
            }
        }
    };
}

create_owned!(
    create_pipeline_layout,
    vk::PipelineLayoutCreateInfo<'_>,
    OwnedPipelineLayout
);
create_owned!(
    create_descriptor_set_layout,
    vk::DescriptorSetLayoutCreateInfo<'_>,
    OwnedSetLayout
);
create_owned!(
    create_descriptor_pool,
    vk::DescriptorPoolCreateInfo<'_>,
    OwnedDescriptorPool
);
create_owned!(
    create_render_pass,
    vk::RenderPassCreateInfo<'_>,
    OwnedRenderPass
);
create_owned!(
    create_framebuffer,
    vk::FramebufferCreateInfo<'_>,
    OwnedFramebuffer
);
create_owned!(create_sampler, vk::SamplerCreateInfo<'_>, OwnedSampler);

// Generate one owning wrapper. Each is a non-`Copy`, non-`Clone` value holding
// the handle and a share of the device that made it; `Drop` queues the handle
// for destruction.
//
// `null()` is the inert placeholder the pass structs use for a resource that is
// not built yet, mirroring `PooledBuffer::null` / `GpuImage::null`. A null
// wrapper holds no device share and drops to nothing, so `std::mem::take` on an
// owned field is how a caller retires one early.
macro_rules! owned_handle {
    ($name:ident, $vk:ty, $kind:ident) => {
        pub(in crate::vulkan) struct $name {
            handle: $vk,
            // `None` for a null placeholder, which owns nothing.
            device: Option<VkDevice>,
        }

        impl $name {
            // Take ownership of a handle created from `device`. The handle must
            // not be owned anywhere else.
            pub(in crate::vulkan) fn new(device: &VkDevice, handle: $vk) -> Self {
                Self {
                    handle,
                    device: Some(device.clone()),
                }
            }

            // The raw handle, for the calls that still take one. Borrowing
            // `self` is what ties the handle's use to its owner being alive.
            pub(in crate::vulkan) fn handle(&self) -> $vk {
                self.handle
            }

            // An inert placeholder for a resource that has not been built.
            pub(in crate::vulkan) fn null() -> Self {
                Self {
                    handle: <$vk>::null(),
                    device: None,
                }
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::null()
            }
        }

        impl Drop for $name {
            fn drop(&mut self) {
                if let Some(device) = &self.device {
                    device.retire(Retired::$kind(self.handle));
                }
            }
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_tuple(stringify!($name))
                    .field(&self.handle)
                    .finish()
            }
        }
    };
}

// A graphics or compute pipeline. Built through
// `pipeline_cache::create_graphics_pipelines` / `create_compute_pipelines`,
// never directly, so every pipeline goes through the persisted cache.
owned_handle!(OwnedPipeline, vk::Pipeline, Pipeline);
owned_handle!(OwnedPipelineLayout, vk::PipelineLayout, PipelineLayout);
owned_handle!(OwnedSetLayout, vk::DescriptorSetLayout, SetLayout);
owned_handle!(OwnedDescriptorPool, vk::DescriptorPool, DescriptorPool);
owned_handle!(OwnedRenderPass, vk::RenderPass, RenderPass);
owned_handle!(OwnedFramebuffer, vk::Framebuffer, Framebuffer);
owned_handle!(OwnedSampler, vk::Sampler, Sampler);

impl OwnedFramebuffer {
    // Whether the slot is still the inert placeholder. Only the families a pass
    // rebuilds in place ask this, so it is not part of the shared surface.
    pub(in crate::vulkan) fn is_null(&self) -> bool {
        self.device.is_none()
    }
}

#[cfg(test)]
mod tests {
    use ash::vk::Handle as _;

    use super::*;

    fn pipeline(raw: u64) -> Retired {
        Retired::Pipeline(vk::Pipeline::from_raw(raw))
    }

    #[test]
    fn a_handle_is_withheld_for_frames_in_flight_plus_one_ticks() {
        // frames_in_flight = 2, so the window is 3 ticks.
        let mut queue = RetireQueue::new(2);
        queue.push(pipeline(1));
        assert!(queue.tick().is_empty());
        assert!(queue.tick().is_empty());
        assert_eq!(queue.tick(), vec![pipeline(1)]);
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn handles_queued_on_different_frames_retire_on_different_ticks() {
        let mut queue = RetireQueue::new(1);
        queue.push(pipeline(1));
        assert!(queue.tick().is_empty());
        queue.push(pipeline(2));
        // Tick 2 closes the first handle's window but not the second's.
        assert_eq!(queue.tick(), vec![pipeline(1)]);
        assert_eq!(queue.tick(), vec![pipeline(2)]);
    }

    #[test]
    fn a_single_tick_returns_every_handle_whose_window_closed() {
        let mut queue = RetireQueue::new(0);
        queue.push(pipeline(1));
        queue.push(pipeline(2));
        assert_eq!(queue.tick(), vec![pipeline(1), pipeline(2)]);
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn draining_ignores_the_window() {
        let mut queue = RetireQueue::new(8);
        queue.push(pipeline(1));
        queue.push(pipeline(2));
        assert_eq!(queue.drain(), vec![pipeline(1), pipeline(2)]);
        assert_eq!(queue.len(), 0);
        assert!(queue.drain().is_empty());
    }

    #[test]
    fn each_kind_carries_its_own_handle() {
        // The tag is what picks the destroy call, so a wrapper must not be able
        // to retire under another kind's tag.
        let entries = [
            Retired::Pipeline(vk::Pipeline::from_raw(1)),
            Retired::PipelineLayout(vk::PipelineLayout::from_raw(1)),
            Retired::SetLayout(vk::DescriptorSetLayout::from_raw(1)),
            Retired::DescriptorPool(vk::DescriptorPool::from_raw(1)),
            Retired::RenderPass(vk::RenderPass::from_raw(1)),
            Retired::Framebuffer(vk::Framebuffer::from_raw(1)),
            Retired::Sampler(vk::Sampler::from_raw(1)),
        ];
        for (i, a) in entries.iter().enumerate() {
            for (j, b) in entries.iter().enumerate() {
                assert_eq!(i == j, a == b, "{a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn a_null_wrapper_owns_nothing() {
        let p = OwnedPipeline::null();
        assert_eq!(p.handle(), vk::Pipeline::null());
        // Dropping it must not need a device, which is the whole point of the
        // placeholder: a pass struct can hold one before its device exists and
        // before its pipeline is built.
        drop(p);
        assert!(OwnedFramebuffer::default().is_null());
    }
}
