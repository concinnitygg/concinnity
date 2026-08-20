// src/metal/transient_pool.rs
//
// Backing store for the render graph's transient textures on Metal. The shared
// `gfx::render_graph::alias` planner decides which transient resources may share
// physical memory; this pool is where the Metal backend realises that plan.
// Features stop owning these textures and read them back by label, so the pool
// repoints several labels at one aliased allocation without touching the
// features. This mirrors how the graph plans barriers while each backend emits
// them, and the Vulkan / DirectX `transient_pool.rs`.
//
// Structure: the pool is organised into alias slots. Each slot owns one
// `MTLHeap` sized to its largest member, and every member is placed at offset 0.
// Members of a slot have pairwise-disjoint lifetimes (they are never live at the
// same time), so reusing the bytes is safe. A single-member slot is a plain
// placed target; a multi-member slot is a realised alias.
//
// The heaps are `Placement` so the pool picks the offset (0 for every member),
// which maps onto the planner's slot abstraction directly. They are explicitly
// `Tracked`: on a tracked heap Metal delays reads and writes of every resource
// suballocated on it until in-flight modifications of any of them complete,
// which is exactly the ordering aliased members need at both the within-frame
// and the cross-frame reuse boundary. `MTLHazardTrackingMode::Default` is
// treated as `Untracked` for heaps and would silently drop the automatic
// tracking the rest of the backend assumes. One heap per slot keeps that
// heap-granularity tracking scoped to members that genuinely alias.
//
// Slots are single-buffered, like DirectX and unlike Vulkan: the tracked heap
// orders a frame's writes against the previous frame's reads of the same bytes,
// so one heap per slot (rather than one per slot per frame) is safe. Per-frame
// slots would multiply an already single-buffered footprint by the
// frames-in-flight depth and cost more than aliasing saves.
//
// A texture is "managed" iff its owning feature is enabled at build time;
// `texture_for` returns `None` otherwise and the consumer falls back exactly as
// before (the main pass samples a 1x1 white texture when SSAO is off).
#![deny(unsafe_op_in_unsafe_fn)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLDevice as _, MTLHazardTrackingMode, MTLHeap, MTLHeapDescriptor, MTLHeapType, MTLPixelFormat,
    MTLStorageMode, MTLTexture, MTLTextureDescriptor, MTLTextureType, MTLTextureUsage,
};

// `ClearValue` is deliberately absent: Metal takes a clear value on the render
// pass descriptor rather than baking it into the texture, so the graph's clear
// is consumed where the pass is encoded, not here. DirectX translates it at
// creation, which is why its pool reads the field and this one does not.
use crate::gfx::render_graph::{
    PixelFormat, PoolGates, TextureUsage, TransientSlot, TransientTexture, plan_pool_slots,
};
use crate::metal::context::MtlContext;

struct PooledTexture {
    label: &'static str,
    texture: Retained<ProtocolObject<dyn MTLTexture>>,
}

// The transient texture pool owned by `MtlContext`. Resolution-dependent, so it
// is rebuilt on resize alongside the other render-resolution targets.
pub(super) struct TransientTexturePool {
    // One heap per slot, backing every member of that slot at offset 0. Held so
    // a rebuild releases a slot's memory as a unit, and so the pool's own
    // lifetime covers the heaps its textures were placed on.
    heaps: Vec<Retained<ProtocolObject<dyn MTLHeap>>>,
    textures: Vec<PooledTexture>,
    // The member labels of each slot, in the order they reuse its memory. Read
    // back by the executor's per-frame soundness assertion: members of one slot
    // share bytes, so two live at once is silent corruption.
    slot_labels: Vec<Vec<&'static str>>,
}

impl TransientTexturePool {
    // Allocate one heap per slot and place every member at offset 0. Each
    // texture starts undefined; its first-use contents come from its graph
    // producer pass exactly as when the feature owned it. (Unlike D3D12's placed
    // resources, a Metal placed texture needs no Discard-style initialisation
    // before its first use.)
    pub(super) fn build(
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        slots: &[TransientSlot],
    ) -> Result<Self, String> {
        let mut heaps = Vec::with_capacity(slots.len());
        let mut textures = Vec::new();
        let slot_labels: Vec<Vec<&'static str>> = slots.iter().map(|s| s.labels()).collect();
        // What the members would cost with one allocation each; the heaps' real
        // size is the aliased footprint. The difference is the VRAM aliasing
        // reclaims, reported below.
        let mut unaliased_bytes: u64 = 0;
        for slot in slots {
            // Size the heap to the largest member. Offset 0 satisfies every
            // member's alignment, so aliased members all place there.
            let mut slot_size: usize = 0;
            let mut descs = Vec::with_capacity(slot.members.len());
            for m in &slot.members {
                let desc = texture_descriptor(m);
                let size = device.heapTextureSizeAndAlignWithDescriptor(&desc).size;
                unaliased_bytes += size as u64;
                slot_size = slot_size.max(size);
                descs.push((m.label, desc));
            }
            let heap = new_slot_heap(device, slot_size)?;
            for (label, desc) in descs {
                // Placement heap at offset 0: in bounds because the heap is
                // sized to the largest member, and aligned because every
                // alignment divides 0.
                // SAFETY: the heap is sized to its largest member, so offset 0 is in bounds, and
                // every alignment divides 0.
                let texture = unsafe { heap.newTextureWithDescriptor_offset(&desc, 0) }
                    .ok_or_else(|| format!("failed to place transient texture {label}"))?;
                textures.push(PooledTexture { label, texture });
            }
            heaps.push(heap);
        }
        let pool = Self {
            heaps,
            textures,
            slot_labels,
        };
        let aliased_bytes = pool.heap_bytes();
        tracing::info!(
            "transient texture pool: {} slot heap(s), {} KiB ({} KiB saved by aliasing)",
            pool.heaps.len(),
            aliased_bytes / 1024,
            unaliased_bytes.saturating_sub(aliased_bytes) / 1024,
        );
        Ok(pool)
    }

    // The managed texture for `label`, or `None` when its owning feature was
    // disabled at build time (so the pool holds no entry for it).
    pub(super) fn texture_for(&self, label: &str) -> Option<&ProtocolObject<dyn MTLTexture>> {
        self.lookup(label).map(|t| t.texture.as_ref())
    }

    // The pooled `bloom_top` (bloom mip 0) as an owned handle for the bloom
    // chain to hold. Always managed, so a missing entry is a build bug rather
    // than a disabled feature.
    pub(super) fn bloom_top(&self) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, String> {
        self.lookup("bloom_top")
            .map(|t| t.texture.clone())
            .ok_or_else(|| "bloom_top missing from transient pool".to_string())
    }

    // Total bytes the slot heaps occupy: the pool's aliased footprint, as Metal
    // rounded it up at creation.
    pub(super) fn heap_bytes(&self) -> u64 {
        self.heaps.iter().map(|h| h.size() as u64).sum()
    }

    // The member labels of each slot, for the executor's per-frame check that
    // no slot has two resources live at once in the graph it is about to run.
    pub(super) fn slot_labels(&self) -> &[Vec<&'static str>] {
        &self.slot_labels
    }

    fn lookup(&self, label: &str) -> Option<&PooledTexture> {
        self.textures.iter().find(|t| t.label == label)
    }

    // Rebuild every managed texture at a new extent. Metal reference-counts, and
    // a heap-placed texture holds a reference to its heap, so an in-flight
    // command buffer keeps both the old texture and the memory under it alive
    // until the GPU retires that frame; the heaps released here are only those
    // no frame still references. The caller rebinds the new textures into the
    // affected consumers (the per-frame bindless argument buffer re-encodes
    // `ao_output` itself; the bloom chain re-reads `bloom_top`).
    pub(super) fn rebuild(
        &mut self,
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        slots: &[TransientSlot],
    ) -> Result<(), String> {
        *self = Self::build(device, slots)?;
        Ok(())
    }
}

// One `Private` placement heap for a slot, sized to its largest member. Hazard
// tracking is set explicitly: heaps treat `Default` as `Untracked`, which would
// drop the automatic tracking the aliased members and the rest of the backend
// rely on.
fn new_slot_heap(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    size: usize,
) -> Result<Retained<ProtocolObject<dyn MTLHeap>>, String> {
    let desc = MTLHeapDescriptor::new();
    desc.setType(MTLHeapType::Placement);
    desc.setStorageMode(MTLStorageMode::Private);
    desc.setHazardTrackingMode(MTLHazardTrackingMode::Tracked);
    desc.setSize(size.max(1));
    device
        .newHeapWithDescriptor(&desc)
        .ok_or_else(|| format!("failed to create {size}-byte transient slot heap"))
}

// Translate one graph-declared transient into its Metal descriptor. This is the
// backend's whole share of describing a pooled resource: the extent, format,
// mip count and usage all come from the graph, so there is no second table here
// that could disagree with it. GPU-private to match its heap's storage mode.
fn texture_descriptor(spec: &TransientTexture) -> Retained<MTLTextureDescriptor> {
    let desc = MTLTextureDescriptor::new();
    // SAFETY: plain descriptor property setters, all values in range.
    unsafe {
        desc.setTextureType(texture_type(spec));
        desc.setPixelFormat(pixel_format(spec.format));
        desc.setWidth(spec.width.max(1) as usize);
        desc.setHeight(spec.height.max(1) as usize);
        desc.setDepth(spec.depth.max(1) as usize);
        desc.setArrayLength(spec.array_layers.max(1) as usize);
        desc.setMipmapLevelCount(spec.mip_levels.max(1) as usize);
        desc.setSampleCount(spec.sample_count.max(1) as usize);
        desc.setUsage(texture_usage(spec.usage));
        desc.setStorageMode(MTLStorageMode::Private);
    }
    desc
}

fn texture_type(spec: &TransientTexture) -> MTLTextureType {
    match (
        spec.depth.max(1) > 1,
        spec.array_layers.max(1) > 1,
        spec.sample_count.max(1) > 1,
    ) {
        (true, _, _) => MTLTextureType::Type3D,
        (_, true, _) => MTLTextureType::Type2DArray,
        (_, _, true) => MTLTextureType::Type2DMultisample,
        _ => MTLTextureType::Type2D,
    }
}

fn pixel_format(format: PixelFormat) -> MTLPixelFormat {
    match format {
        PixelFormat::Rgba16Float => MTLPixelFormat::RGBA16Float,
        PixelFormat::Rgba8Unorm => MTLPixelFormat::RGBA8Unorm,
        PixelFormat::Rg16Float => MTLPixelFormat::RG16Float,
        PixelFormat::R8Unorm => MTLPixelFormat::R8Unorm,
        PixelFormat::R32Float => MTLPixelFormat::R32Float,
        PixelFormat::Depth32Float => MTLPixelFormat::Depth32Float,
        PixelFormat::BgraSwapchain => MTLPixelFormat::BGRA8Unorm,
    }
}

fn texture_usage(usage: TextureUsage) -> MTLTextureUsage {
    let mut bits = 0;
    if usage.contains(TextureUsage::SHADER_READ) {
        bits |= MTLTextureUsage::ShaderRead.0;
    }
    if usage.contains(TextureUsage::RENDER_TARGET) || usage.contains(TextureUsage::DEPTH_STENCIL) {
        bits |= MTLTextureUsage::RenderTarget.0;
    }
    if usage.contains(TextureUsage::STORAGE) {
        bits |= MTLTextureUsage::ShaderRead.0 | MTLTextureUsage::ShaderWrite.0;
    }
    MTLTextureUsage(bits)
}

// The alias-slot list for the transients the pool manages this build. The
// grouping, the pooled label set and each member's shape all come from the
// shared planner, so nothing here can disagree with the graph or with another
// backend.
//
// Bloom is always managed: Metal toggles it per frame off `bloom_intensity`
// while the composite binds mip 0 unconditionally, so a pool built at init /
// resize cannot gate on it and the heap must cover the frames where it is live.
// DirectX does the same; Vulkan rebuilds on the flag and passes it through.
pub(super) fn transient_slots(
    ssao_enabled: bool,
    gbuffer_enabled: bool,
    render_extent: (u32, u32),
    output_extent: (u32, u32),
) -> Result<Vec<TransientSlot>, String> {
    plan_pool_slots(
        PoolGates {
            ssao: ssao_enabled,
            bloom: true,
            gbuffer: gbuffer_enabled,
        },
        render_extent,
        output_extent,
    )
}

impl MtlContext {
    // The texture the main pass and the bindless argument buffer sample for
    // ambient occlusion: the pooled `ao_output` (SSAO's blurred occlusion) when
    // SSAO is on, else the SSAO state's 1x1 white fallback so `shade_surface`
    // reads a constant 1.0 (fully unoccluded).
    pub(in crate::metal) fn ao_output_texture(&self) -> &ProtocolObject<dyn MTLTexture> {
        self.transient_pool
            .texture_for("ao_output")
            .unwrap_or_else(|| self.ssao.white.as_ref())
    }

    // The unified pre-pass's three colour channels, read straight out of the
    // pool at the point of use. Nothing caches these, and that is deliberate:
    // **a pool rebuild repacks every slot**, so a handle cached when SSAO was
    // toggled would point into a heap region that now belongs to a different
    // resource. The explicit backends have to re-point their descriptors and
    // framebuffers at every rebuild for exactly this reason; fetching by label
    // costs a short scan per encode and makes the whole class impossible here.
    //
    // `None` when the pre-pass is not built, which is the same gate the pool
    // was created under, so a consumer that finds `None` should skip rather
    // than substitute a fallback.
    pub(in crate::metal) fn gbuffer_normal_depth(&self) -> Option<&ProtocolObject<dyn MTLTexture>> {
        self.transient_pool.texture_for("gbuffer_normal_depth")
    }

    pub(in crate::metal) fn gbuffer_roughness(&self) -> Option<&ProtocolObject<dyn MTLTexture>> {
        self.transient_pool.texture_for("gbuffer_roughness")
    }

    pub(in crate::metal) fn gbuffer_velocity(&self) -> Option<&ProtocolObject<dyn MTLTexture>> {
        self.transient_pool.texture_for("gbuffer_velocity")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `transient_slots` is pure CPU (it builds slot descriptions; no device), so
    // the planner-routed grouping is testable headlessly.

    #[test]
    fn the_late_bloom_target_aliases_an_early_one() {
        // The pool's whole saving, in the configuration a real session runs:
        // SSAO on implies the G-buffer pre-pass is on, so both gates are true
        // here. `bloom_top` is the only genuinely late member (Bloom ->
        // Composite), so it is the one that can reuse an earlier member's
        // memory; everything else is live across most of the frame and needs
        // its own heap. A plan where `bloom_top` sits alone means the aliasing
        // stopped working -- which is exactly what happened on Metal between
        // pooling the G-buffer on the explicit backends and pooling it here.
        let slots = transient_slots(true, true, (1024, 768), (1024, 768)).expect("plans");
        let shared: Vec<Vec<&str>> = slots
            .iter()
            .map(|s| s.labels())
            .filter(|l| l.len() > 1)
            .collect();
        assert!(
            shared.iter().any(|l| l.contains(&"bloom_top")),
            "bloom_top should reuse an earlier member's heap: {:?}",
            slots.iter().map(|s| s.labels()).collect::<Vec<_>>()
        );
        // Whatever it pairs with must start first: members are kept in the
        // planner's lifetime-start order, which is the order they reuse the heap.
        let pair = shared
            .iter()
            .find(|l| l.contains(&"bloom_top"))
            .expect("checked above");
        assert_ne!(pair[0], "bloom_top", "{pair:?}");
    }

    #[test]
    fn the_gbuffer_colour_targets_are_pooled_and_depth_is_not() {
        let slots = transient_slots(true, true, (1024, 768), (1024, 768)).expect("plans");
        let labels: Vec<&str> = slots.iter().flat_map(|s| s.labels()).collect();
        for want in [
            "gbuffer_normal_depth",
            "gbuffer_roughness",
            "gbuffer_velocity",
        ] {
            assert!(labels.contains(&want), "{want} pooled: {labels:?}");
        }
        assert!(
            !labels.contains(&"gbuffer_depth"),
            "gbuffer_depth must stay feature-owned: {labels:?}"
        );
    }

    #[test]
    fn the_gbuffer_gate_is_what_places_them() {
        // `unified_gbuffer_prepass` substitutes passes rather than adding them,
        // so `planning_inputs` cannot force it on and the pool must follow the
        // build. Without the gate the pre-pass node is absent and none of its
        // targets are placed -- which would leave every consumer reading a
        // label the pool never created, and the pre-pass encoder erroring out.
        let slots = transient_slots(true, false, (1024, 768), (1024, 768)).expect("plans");
        let labels: Vec<&str> = slots.iter().flat_map(|s| s.labels()).collect();
        assert!(!labels.contains(&"gbuffer_normal_depth"), "{labels:?}");
    }

    #[test]
    fn bloom_top_alone_is_unshared() {
        // Nothing else managed: `bloom_top` sits in its own single-member slot.
        let slots = transient_slots(false, false, (1024, 768), (1024, 768)).expect("plans");
        assert_eq!(slots.len(), 1);
        assert_eq!(slots[0].members.len(), 1);
        assert_eq!(slots[0].members[0].label, "bloom_top");
    }

    #[test]
    fn translated_descriptors_match_the_feature_formats() {
        // The graph is the single source of the shape now, so what this pins is
        // the *translation*: the descriptor the pool creates must still be the
        // one each feature's own constant describes, or the pool silently
        // mis-backs the texture that feature binds.
        let slots = transient_slots(true, true, (1024, 768), (1024, 768)).expect("plans");
        let member = |label: &str| {
            slots
                .iter()
                .flat_map(|s| &s.members)
                .find(|m| m.label == label)
                .expect("member present")
        };

        let ao = texture_descriptor(member("ao_output"));
        assert_eq!(
            ao.pixelFormat(),
            super::super::post::ssao::SSAO_OCCLUSION_FORMAT
        );
        assert_eq!((ao.width(), ao.height()), (1024, 768));
        assert_eq!(ao.textureType(), MTLTextureType::Type2D);
        assert_eq!(ao.mipmapLevelCount(), 1);
        assert_eq!(ao.sampleCount(), 1);
        assert_eq!(
            ao.usage().0,
            MTLTextureUsage::ShaderRead.0 | MTLTextureUsage::RenderTarget.0
        );

        // Half the *output* extent, which is what `create_bloom_targets` sizes
        // its mip 0 to.
        let bloom = texture_descriptor(member("bloom_top"));
        assert_eq!(bloom.pixelFormat(), super::super::post::bloom::BLOOM_FORMAT);
        assert_eq!((bloom.width(), bloom.height()), (512, 384));
    }

    #[test]
    fn a_volume_translates_to_a_3d_descriptor() {
        // Nothing pooled is a volume yet, but the froxel volume's desc is now
        // real, so the translator has to handle it before it can be pooled.
        let desc = texture_descriptor(&TransientTexture {
            label: "probe_volume",
            width: 80,
            height: 45,
            depth: 64,
            format: PixelFormat::Rgba16Float,
            sample_count: 1,
            array_layers: 1,
            mip_levels: 1,
            usage: TextureUsage::STORAGE.union(TextureUsage::SHADER_READ),
            // Carried by the graph for DirectX's sake; the Metal descriptor has
            // nowhere to put it.
            clear: crate::gfx::render_graph::ClearValue::Color([0.0; 4]),
        });
        assert_eq!(desc.textureType(), MTLTextureType::Type3D);
        assert_eq!(desc.depth(), 64);
        assert_eq!(
            desc.usage().0,
            MTLTextureUsage::ShaderRead.0 | MTLTextureUsage::ShaderWrite.0
        );
    }
}
