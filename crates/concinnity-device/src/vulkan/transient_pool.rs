// src/vulkan/transient_pool.rs
//
// Backing store for the render graph's transient images. Stage 1's
// `gfx::render_graph::alias` planner decides which transient resources may
// share physical memory; this pool is where the Vulkan backend realises that
// plan. Features stop owning these images and read them back by label, so the
// pool can repoint several labels at one shared allocation without touching the
// features. This mirrors how the graph plans barriers while each backend emits
// them.
//
// Structure: the pool is organised into alias *slots*. A slot owns one
// `VkDeviceMemory` per frame in flight; every member image of a slot binds into
// that one allocation at offset 0. Members of a slot must have pairwise-disjoint
// lifetimes (they are never live at the same time), so reusing the bytes is
// safe within a frame; the per-frame copies keep the reuse safe across frames in
// flight (the single-frame planner does not model frames-in-flight, so the
// backend supplies the per-frame buffering). A single-member slot is just a
// per-frame target with its own memory (no sharing); a multi-member slot is a
// realised alias.
//
// A resource is "managed" iff its owning feature is enabled at build time (e.g.
// `ao_output` only when SSAO is on); the `*_for` lookups return `None`
// otherwise and the consumer falls back exactly as it did before.

use ash::{Device, vk};

use super::texture::{
    create_image_view, find_memory_type, one_shot_submit, transition_image_layout,
};
use crate::gfx::render_graph::{
    FrameGraphInputs, PixelFormat, TextureUsage, TransientSlot, TransientTexture,
    plan_transient_slots,
};

// The raw device handles the pool allocates with. The pool deliberately stays
// off the device allocator: its slots alias images on purpose, which the
// general pool must never do.
#[derive(Clone, Copy)]
pub(super) struct TransientPoolGpu<'a> {
    pub instance: &'a ash::Instance,
    pub device: &'a Device,
    pub physical_device: vk::PhysicalDevice,
    pub command_pool: vk::CommandPool,
    pub queue: vk::Queue,
}

// One managed image, resolved for one frame in flight.
struct TransientImage {
    label: &'static str,
    frame: usize,
    image: vk::Image,
    view: vk::ImageView,
    aspect: vk::ImageAspectFlags,
}

// The transient image pool owned by `VkContext`. Resolution-dependent, so it is
// rebuilt on swapchain resize.
pub(super) struct TransientImagePool {
    // One backing allocation per (slot, frame). Owned here, freed on destroy /
    // rebuild after the member images + views are gone.
    slot_memories: Vec<vk::DeviceMemory>,
    // Every member image across all slots + frames.
    images: Vec<TransientImage>,
    // The member labels of each slot, in lifetime order (the order they reuse the
    // slot's memory). Drives the executor's aliasing barriers: a member's
    // predecessor in this list is the resource it reuses memory from.
    slot_labels: Vec<Vec<&'static str>>,
    // The pool's aliased footprint: the sum of its slot allocations across every
    // frame in flight. Reported to the memory ledger, which would otherwise not
    // see this pool at all -- it deliberately sits off the device allocator.
    allocated_bytes: u64,
}

impl TransientImagePool {
    // Allocate every slot's per-frame backing memory and bind its member images
    // into it. Each member is pre-transitioned to `SHADER_READ_ONLY_OPTIMAL`, the
    // resting layout a producing pass leaves it in, so a consumer that samples it
    // before any producer has run binds a valid layout (the Composite samples
    // `ao_output` and `bloom_top` unconditionally, but a world hidden behind an
    // opaque menu masks off every pass that writes them). Producers still open
    // from `UNDEFINED` and discard, so this costs nothing after the first frame.
    // Matches the committed bloom mips and the TAA history images.
    pub(super) fn build(
        ctx: &TransientPoolGpu,
        frames: usize,
        slots: &[TransientSlot],
    ) -> Result<Self, String> {
        let &TransientPoolGpu {
            instance,
            device,
            physical_device,
            command_pool,
            queue,
        } = ctx;
        let mut slot_memories = Vec::new();
        let mut images = Vec::new();
        let slot_labels: Vec<Vec<&'static str>> = slots
            .iter()
            .map(|s| s.members.iter().map(|m| m.label).collect())
            .collect();
        // Footprint accounting: `aliased_bytes` is what the pool actually
        // allocates (one slot allocation per (slot, frame)); `unaliased_bytes` is
        // what the same images would cost with no sharing. Their difference is the
        // VRAM aliasing reclaims, reported below.
        let mut aliased_bytes: u64 = 0;
        let mut unaliased_bytes: u64 = 0;
        for slot in slots {
            for f in 0..frames {
                // Create every member image (unbound), gathering the combined
                // memory requirements: the slot's allocation must be large
                // enough for the biggest member and of a type all members accept.
                let mut member_images: Vec<(&TransientTexture, vk::Image)> =
                    Vec::with_capacity(slot.members.len());
                let mut type_bits = u32::MAX;
                let mut slot_size: vk::DeviceSize = 0;
                for m in &slot.members {
                    let image = create_image_unbound(device, m)?;
                    let reqs = unsafe { device.get_image_memory_requirements(image) };
                    type_bits &= reqs.memory_type_bits;
                    slot_size = slot_size.max(reqs.size);
                    unaliased_bytes += reqs.size;
                    member_images.push((m, image));
                }
                aliased_bytes += slot_size;

                // One device-local allocation backs every member of this slot
                // for this frame; bind each member at offset 0 (their disjoint
                // lifetimes make the overlap safe).
                let memory = unsafe {
                    device.allocate_memory(
                        &vk::MemoryAllocateInfo::default()
                            .allocation_size(slot_size)
                            .memory_type_index(find_memory_type(
                                instance,
                                physical_device,
                                type_bits,
                                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                            )?),
                        None,
                    )
                }
                .map_err(|e| format!("transient pool slot memory: {e}"))?;
                slot_memories.push(memory);

                for (spec, image) in member_images {
                    unsafe { device.bind_image_memory(image, memory, 0) }
                        .map_err(|e| format!("transient pool bind {}: {e}", spec.label))?;
                    let aspect = image_aspect(spec.format);
                    let view = create_image_view(device, image, image_format(spec.format), aspect)?;
                    images.push(TransientImage {
                        label: spec.label,
                        frame: f,
                        image,
                        view,
                        aspect,
                    });
                }
            }
        }
        if !images.is_empty() {
            one_shot_submit(device, command_pool, queue, |cmd| {
                for p in &images {
                    transition_image_layout(
                        device,
                        cmd,
                        p.image,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        p.aspect,
                    );
                }
            })?;
        }
        tracing::info!(
            "transient image pool: {} slot allocation(s), {} KiB ({} KiB saved by aliasing)",
            slot_memories.len(),
            aliased_bytes / 1024,
            unaliased_bytes.saturating_sub(aliased_bytes) / 1024,
        );
        Ok(Self {
            slot_memories,
            images,
            slot_labels,
            allocated_bytes: aliased_bytes,
        })
    }

    // The label `label` reuses slot memory from, i.e. the member immediately
    // before it in its slot's lifetime order, or `None` when `label` is the
    // first member of its slot (or unmanaged, or alone). The executor emits an
    // aliasing barrier on `label` against this predecessor before `label`'s
    // first write, since they share one allocation.
    pub(super) fn alias_predecessor(&self, label: &str) -> Option<&'static str> {
        for members in &self.slot_labels {
            if let Some(pos) = members.iter().position(|&l| l == label) {
                return if pos == 0 {
                    None
                } else {
                    Some(members[pos - 1])
                };
            }
        }
        None
    }

    // The pool's aliased footprint in bytes, for the memory ledger.
    pub(super) fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes
    }

    // The pooled G-buffer colour channels for every frame in flight. Empty
    // `Vec`s when the pool was built without the G-buffer gate (no screen-space
    // consumer, so the pre-pass node is absent and nothing was allocated); the
    // caller treats that as "the feature is not built" rather than an error,
    // matching every other `*_for` lookup here.
    pub(super) fn gbuffer_pooled(&self, frames: usize) -> super::post::gbuffer::GbufferPooled {
        let channel = |label: &str| {
            self.pairs_for_frames(label, frames)
                .into_iter()
                .map(|(image, view)| super::post::gbuffer::PooledTarget { image, view })
                .collect()
        };
        super::post::gbuffer::GbufferPooled {
            normal_depth: channel("gbuffer_normal_depth"),
            roughness: channel("gbuffer_roughness"),
            velocity: channel("gbuffer_velocity"),
        }
    }

    // The member labels of each slot, for the executor's per-frame check that
    // no slot has two resources live at once in the graph it is about to run.
    pub(super) fn slot_labels(&self) -> &[Vec<&'static str>] {
        &self.slot_labels
    }

    // The managed image for `label` at frame-in-flight `frame`, or `None` when
    // the owning feature was disabled at build time (so no image was allocated).
    pub(super) fn image_for(&self, label: &str, frame: usize) -> Option<vk::Image> {
        self.lookup(label, frame).map(|p| p.image)
    }

    // The sampled / attachment view for `label` at frame-in-flight `frame`.
    pub(super) fn view_for(&self, label: &str, frame: usize) -> Option<vk::ImageView> {
        self.lookup(label, frame).map(|p| p.view)
    }

    // Every frame-in-flight view for `label`, frames `0..frames` in order.
    // Empty when the label is unmanaged. When the label is managed the pool
    // holds one image per frame, so the result has exactly `frames` entries.
    pub(super) fn views_for_frames(&self, label: &str, frames: usize) -> Vec<vk::ImageView> {
        (0..frames)
            .filter_map(|f| self.view_for(label, f))
            .collect()
    }

    // Every (image, view) pair for `label`, frames `0..frames` in order. Empty
    // when unmanaged; one entry per frame when managed. Used to hand a per-frame
    // pooled image to a feature that wraps it (bloom mip 0).
    pub(super) fn pairs_for_frames(
        &self,
        label: &str,
        frames: usize,
    ) -> Vec<(vk::Image, vk::ImageView)> {
        (0..frames)
            .filter_map(|f| self.lookup(label, f).map(|p| (p.image, p.view)))
            .collect()
    }

    fn lookup(&self, label: &str, frame: usize) -> Option<&TransientImage> {
        self.images
            .iter()
            .find(|p| p.label == label && p.frame == frame)
    }

    // Rebuild every managed image at a new extent / frame count. The caller has
    // already idled the device. The old images are freed first, so any feature
    // framebuffer / descriptor that referenced their views must be rebuilt by
    // the caller afterward.
    pub(super) fn rebuild(
        &mut self,
        ctx: &TransientPoolGpu,
        frames: usize,
        slots: &[TransientSlot],
    ) -> Result<(), String> {
        self.destroy(ctx.device);
        *self = Self::build(ctx, frames, slots)?;
        Ok(())
    }

    // Free every managed image, view, and slot allocation. The caller has
    // already idled the device and destroyed any framebuffer that referenced
    // these views.
    pub(super) fn destroy(&mut self, device: &Device) {
        unsafe {
            for p in &self.images {
                device.destroy_image_view(p.view, None);
                device.destroy_image(p.image, None);
            }
            for &mem in &self.slot_memories {
                device.free_memory(mem, None);
            }
        }
        self.images.clear();
        self.slot_memories.clear();
        self.slot_labels.clear();
        self.allocated_bytes = 0;
    }
}

// Create a `VkImage` without backing memory: the pool binds it into a slot
// allocation afterward (so several aliased images can share one allocation).
// Mirrors `texture::create_image` minus the allocate + bind, and translates the
// graph's declared shape rather than restating it.
fn create_image_unbound(device: &Device, spec: &TransientTexture) -> Result<vk::Image, String> {
    let info = vk::ImageCreateInfo::default()
        .image_type(if spec.depth.max(1) > 1 {
            vk::ImageType::TYPE_3D
        } else {
            vk::ImageType::TYPE_2D
        })
        .extent(vk::Extent3D {
            width: spec.width.max(1),
            height: spec.height.max(1),
            depth: spec.depth.max(1),
        })
        .mip_levels(spec.mip_levels.max(1))
        .array_layers(spec.array_layers.max(1))
        .format(image_format(spec.format))
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(image_usage(spec.usage))
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(sample_count(spec.sample_count));
    unsafe { device.create_image(&info, None) }.map_err(|e| format!("transient pool image: {e}"))
}

fn image_format(format: PixelFormat) -> vk::Format {
    match format {
        PixelFormat::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
        PixelFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        PixelFormat::Rg16Float => vk::Format::R16G16_SFLOAT,
        PixelFormat::R8Unorm => vk::Format::R8_UNORM,
        PixelFormat::R32Float => vk::Format::R32_SFLOAT,
        PixelFormat::Depth32Float => vk::Format::D32_SFLOAT,
        PixelFormat::BgraSwapchain => vk::Format::B8G8R8A8_UNORM,
    }
}

fn image_aspect(format: PixelFormat) -> vk::ImageAspectFlags {
    if format.is_depth() {
        vk::ImageAspectFlags::DEPTH
    } else {
        vk::ImageAspectFlags::COLOR
    }
}

fn image_usage(usage: TextureUsage) -> vk::ImageUsageFlags {
    let mut flags = vk::ImageUsageFlags::empty();
    if usage.contains(TextureUsage::SHADER_READ) {
        flags |= vk::ImageUsageFlags::SAMPLED;
    }
    if usage.contains(TextureUsage::RENDER_TARGET) {
        flags |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
    }
    if usage.contains(TextureUsage::DEPTH_STENCIL) {
        flags |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
    }
    if usage.contains(TextureUsage::STORAGE) {
        flags |= vk::ImageUsageFlags::STORAGE;
    }
    if usage.contains(TextureUsage::TRANSFER_SRC) {
        flags |= vk::ImageUsageFlags::TRANSFER_SRC;
    }
    if usage.contains(TextureUsage::TRANSFER_DST) {
        flags |= vk::ImageUsageFlags::TRANSFER_DST;
    }
    flags
}

fn sample_count(samples: u32) -> vk::SampleCountFlags {
    match samples.max(1) {
        2 => vk::SampleCountFlags::TYPE_2,
        4 => vk::SampleCountFlags::TYPE_4,
        8 => vk::SampleCountFlags::TYPE_8,
        16 => vk::SampleCountFlags::TYPE_16,
        _ => vk::SampleCountFlags::TYPE_1,
    }
}

// The transients this pool owns. Everything else the graph declares transient
// stays backend-owned; adding a label here is what hands it to the pool.
//
// `gbuffer_depth` is deliberately absent while its three colour siblings are
// here, matching DirectX: the pool creates one image per (slot, frame) with a
// single format, and a shader-readable depth target needs different resource
// and view formats on D3D12. Keeping the pooled set identical across the two
// explicit backends is what makes their footprints comparable.
fn pooled(label: &str) -> bool {
    matches!(
        label,
        "ao_output"
            | "bloom_top"
            | "gbuffer_normal_depth"
            | "gbuffer_roughness"
            | "gbuffer_velocity"
    )
}

// The alias-slot list for the transients the pool manages this build, taken
// straight from the graph: `plan_transient_slots` decides the grouping and
// hands back each member's extent, format and usage, so init and resize cannot
// drift apart and neither can the graph and the image it describes.
//
// `render_extent` sizes the render-resolution transients and `output_extent` is
// the drawable the half-resolution ones scale off; under temporal upscaling
// they differ, which is exactly why both are passed rather than derived. Unlike
// Metal and DirectX (where `bloom_top` is managed unconditionally because bloom
// toggles per frame), Vulkan manages it only while bloom is on, so the build
// configuration carries the real flag.
//
// A planning graph that does not compile is a hard error rather than an empty
// pool: every consumer reads its image back out by label, so silently pooling
// nothing would fail later and further from the cause.
pub(super) fn transient_slots(
    ssao_enabled: bool,
    bloom_enabled: bool,
    gbuffer_enabled: bool,
    render_extent: vk::Extent2D,
    output_extent: vk::Extent2D,
) -> Result<Vec<TransientSlot>, String> {
    let mut build = FrameGraphInputs::all_off();
    build.hdr_width = render_extent.width;
    build.hdr_height = render_extent.height;
    build.ssao_enabled = ssao_enabled;
    build.bloom_enabled = bloom_enabled;
    // `unified_gbuffer_prepass` substitutes for the separate SsrPrepass /
    // Velocity nodes rather than adding to them, so `planning_inputs` cannot
    // force it on: it is a build gate the pool follows. `velocity_enabled` is
    // what makes the node appear once the flag is set.
    build.unified_gbuffer_prepass = gbuffer_enabled;
    build.velocity_enabled = gbuffer_enabled;
    plan_transient_slots(&build, &pooled, output_extent.width, output_extent.height)
        .ok_or_else(|| "transient pool: the planning frame graph failed to compile".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    // `transient_slots` is pure CPU (it builds slot descriptions; no device), so
    // the planner-routed grouping is testable headlessly.

    fn extent(width: u32, height: u32) -> vk::Extent2D {
        vk::Extent2D { width, height }
    }

    #[test]
    fn the_late_bloom_target_aliases_an_early_one() {
        // The pool's whole saving, in the configuration a real session runs:
        // SSAO on implies the G-buffer pre-pass is on, so all three gates are
        // true here. `bloom_top` is the only genuinely late member (Bloom ->
        // Composite), so it is the one that can reuse an earlier member's
        // allocation; everything else is live across most of the frame and needs
        // its own. Mirrors the DirectX test.
        let slots =
            transient_slots(true, true, true, extent(1024, 768), extent(1024, 768)).expect("plans");
        let shared: Vec<Vec<&str>> = slots
            .iter()
            .map(|s| s.labels())
            .filter(|l| l.len() > 1)
            .collect();
        assert!(
            shared.iter().any(|l| l.contains(&"bloom_top")),
            "bloom_top should reuse an earlier member's allocation: {:?}",
            slots.iter().map(|s| s.labels()).collect::<Vec<_>>()
        );
        // Whatever it pairs with must start first: the pool's predecessor wiring
        // depends on lifetime-start order.
        let pair = shared
            .iter()
            .find(|l| l.contains(&"bloom_top"))
            .expect("checked above");
        assert_ne!(pair[0], "bloom_top", "{pair:?}");
    }

    #[test]
    fn ao_output_alone_is_unshared() {
        // SSAO on, bloom off: `ao_output` is the only managed transient, so it
        // sits in its own single-member slot (no aliasing barriers).
        let slots = transient_slots(true, false, false, extent(1024, 768), extent(1024, 768))
            .expect("plans");
        assert_eq!(slots.len(), 1);
        assert_eq!(slots[0].members.len(), 1);
        assert_eq!(slots[0].members[0].label, "ao_output");
    }

    #[test]
    fn bloom_top_alone_is_unshared() {
        // Bloom on, SSAO off: `bloom_top` is the only managed transient.
        let slots = transient_slots(false, true, false, extent(1024, 768), extent(1024, 768))
            .expect("plans");
        assert_eq!(slots.len(), 1);
        assert_eq!(slots[0].members.len(), 1);
        assert_eq!(slots[0].members[0].label, "bloom_top");
    }

    #[test]
    fn nothing_managed_yields_no_slots() {
        // Neither feature on: the pool manages nothing, so there are no slots.
        let slots = transient_slots(false, false, false, extent(1024, 768), extent(1024, 768))
            .expect("plans");
        assert!(slots.is_empty());
    }

    #[test]
    fn translated_images_match_the_feature_formats() {
        // The graph is the single source of the shape now, so what this pins is
        // the *translation*: a divergence from each feature's own constant
        // would silently mis-back the image that feature binds.
        let slots = transient_slots(true, true, true, extent(1024, 768), extent(1920, 1080))
            .expect("plans");
        let member = |label: &str| {
            slots
                .iter()
                .flat_map(|s| &s.members)
                .find(|m| m.label == label)
                .unwrap_or_else(|| panic!("{label} pooled"))
                .clone()
        };

        // `ao_output` follows the render extent; `bloom_top` is half the
        // output extent, which is what `create_bloom_chain` sizes mip 0 to.
        let ao = member("ao_output");
        assert_eq!((ao.width, ao.height), (1024, 768));
        assert_eq!(
            image_format(ao.format),
            super::super::post::ssao::SSAO_OCCLUSION_FORMAT
        );
        assert_eq!(image_aspect(ao.format), vk::ImageAspectFlags::COLOR);
        assert_eq!(
            image_usage(ao.usage),
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED
        );

        let bloom = member("bloom_top");
        assert_eq!((bloom.width, bloom.height), (960, 540));
        assert_eq!(
            image_format(bloom.format),
            super::super::context::HDR_FORMAT
        );

        // The G-buffer colour channels. A format divergence here would silently
        // mis-back an MRT attachment the pre-pass render pass declares, which is
        // a framebuffer-incompatibility error rather than a wrong picture.
        use super::super::post::gbuffer::{
            GBUFFER_NORMAL_DEPTH_FORMAT, GBUFFER_ROUGHNESS_FORMAT, GBUFFER_VELOCITY_FORMAT,
        };
        for (label, format) in [
            ("gbuffer_normal_depth", GBUFFER_NORMAL_DEPTH_FORMAT),
            ("gbuffer_roughness", GBUFFER_ROUGHNESS_FORMAT),
            ("gbuffer_velocity", GBUFFER_VELOCITY_FORMAT),
        ] {
            let m = member(label);
            assert_eq!(image_format(m.format), format, "{label}");
            // Render extent, not the drawable: the pre-pass rasterises at the
            // scene resolution, which differs under temporal upscaling.
            assert_eq!((m.width, m.height), (1024, 768), "{label}");
            assert_eq!(
                image_usage(m.usage),
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                "{label}"
            );
        }
    }

    #[test]
    fn the_gbuffer_gate_places_its_colour_channels() {
        // `unified_gbuffer_prepass` substitutes passes rather than adding them,
        // so `planning_inputs` cannot force it on and the pool follows the build
        // gate. Off, none of the channels are placed -- and the pre-pass
        // framebuffers would have nothing to attach, which is why init derives
        // the gate once and uses it for both.
        let off = transient_slots(true, true, false, extent(1024, 768), extent(1024, 768))
            .expect("plans");
        let off_labels: Vec<&str> = off.iter().flat_map(|s| s.labels()).collect();
        assert!(
            !off_labels.contains(&"gbuffer_normal_depth"),
            "{off_labels:?}"
        );

        let on =
            transient_slots(true, true, true, extent(1024, 768), extent(1024, 768)).expect("plans");
        let on_labels: Vec<&str> = on.iter().flat_map(|s| s.labels()).collect();
        for want in [
            "gbuffer_normal_depth",
            "gbuffer_roughness",
            "gbuffer_velocity",
        ] {
            assert!(on_labels.contains(&want), "{want}: {on_labels:?}");
        }
        // `gbuffer_depth` stays feature-owned on both explicit backends.
        assert!(!on_labels.contains(&"gbuffer_depth"), "{on_labels:?}");
    }

    #[test]
    fn depth_and_storage_usages_translate() {
        // Nothing pooled needs these yet, but the descs the graph now carries
        // do, so the translator has to be right before they can be pooled.
        assert_eq!(
            image_format(PixelFormat::Depth32Float),
            vk::Format::D32_SFLOAT
        );
        assert_eq!(
            image_aspect(PixelFormat::Depth32Float),
            vk::ImageAspectFlags::DEPTH
        );
        assert_eq!(
            image_usage(TextureUsage::DEPTH_STENCIL.union(TextureUsage::SHADER_READ)),
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED
        );
        assert_eq!(
            image_usage(TextureUsage::STORAGE.union(TextureUsage::SHADER_READ)),
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED
        );
        assert_eq!(sample_count(4), vk::SampleCountFlags::TYPE_4);
        assert_eq!(sample_count(1), vk::SampleCountFlags::TYPE_1);
        assert_eq!(sample_count(0), vk::SampleCountFlags::TYPE_1);
    }
}
