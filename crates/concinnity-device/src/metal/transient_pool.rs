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

use crate::metal::context::MtlContext;

// One managed transient texture: the graph label plus the parameters the pool
// needs to place it. The label is the same string the shared `build_frame_graph`
// declares, so every feature consumer agrees on one identifier.
pub(super) struct TextureSpec {
    pub label: &'static str,
    pub width: u32,
    pub height: u32,
    pub format: MTLPixelFormat,
}

// One alias slot: the members that share a heap. A single-member slot is a plain
// placed target; a multi-member slot realises an alias (the members have
// disjoint lifetimes, so they reuse one heap).
pub(super) struct SlotSpec {
    pub members: Vec<TextureSpec>,
}

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
}

impl TransientTexturePool {
    // Allocate one heap per slot and place every member at offset 0. Each
    // texture starts undefined; its first-use contents come from its graph
    // producer pass exactly as when the feature owned it. (Unlike D3D12's placed
    // resources, a Metal placed texture needs no Discard-style initialisation
    // before its first use.)
    pub(super) fn build(
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        slots: &[SlotSpec],
    ) -> Result<Self, String> {
        let mut heaps = Vec::with_capacity(slots.len());
        let mut textures = Vec::new();
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
                let texture = unsafe { heap.newTextureWithDescriptor_offset(&desc, 0) }
                    .ok_or_else(|| format!("failed to place transient texture {label}"))?;
                textures.push(PooledTexture { label, texture });
            }
            heaps.push(heap);
        }
        let pool = Self { heaps, textures };
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
        slots: &[SlotSpec],
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

// The descriptor for a managed transient: single-sample 2D, one mip, sampled
// render target, GPU-private to match its heap's storage mode.
fn texture_descriptor(spec: &TextureSpec) -> Retained<MTLTextureDescriptor> {
    let desc = MTLTextureDescriptor::new();
    unsafe {
        desc.setTextureType(MTLTextureType::Type2D);
        desc.setPixelFormat(spec.format);
        desc.setWidth(spec.width.max(1) as usize);
        desc.setHeight(spec.height.max(1) as usize);
        desc.setUsage(MTLTextureUsage(
            MTLTextureUsage::ShaderRead.0 | MTLTextureUsage::RenderTarget.0,
        ));
        desc.setStorageMode(MTLStorageMode::Private);
    }
    desc
}

// Build the alias-slot list for the transients the pool manages this build.
// Centralises the label -> (format, extent) mapping and the slot grouping so
// init and resize stay in lockstep. The shared `gfx::render_graph::alias`
// planner decides which of the managed transients may share a heap;
// `group_by_plan` packs them into one `SlotSpec` per planner slot, so
// disjoint-lifetime transients alias -- `ao_output` (used early, [SsaoBlur,
// Main]) and `bloom_top` (used late, [Bloom, Composite]) share ONE slot, with
// `ao_output` first since it uses the memory first.
pub(super) fn transient_slots(
    ssao_enabled: bool,
    ao_extent: (u32, u32),
    bloom_top_extent: (u32, u32),
) -> Vec<SlotSpec> {
    // `bloom_top` is always managed. Unlike Vulkan, where bloom is a build-time
    // flag, Metal toggles bloom per frame off `bloom_intensity` while the
    // composite binds mip 0 unconditionally, so a pool built at init / resize
    // cannot gate on it. DirectX manages it always for the same reason.
    let mut specs = vec![bloom_top_spec(bloom_top_extent)];
    if ssao_enabled {
        specs.push(ao_output_spec(ao_extent));
    }
    group_by_plan(specs, ssao_enabled)
}

// Group the managed specs into shared slots per the aliasing planner. The
// planner runs on a minimal worst-case graph (only the managed features on, and
// bloom forced on since it toggles per frame yet the heap must cover the frames
// where it is live) so the generic greedy pairs exactly the pooled candidates --
// on the full frame graph it could instead pair `bloom_top` with the unpooled
// `gbuffer`. The grouping is lifetime-based, so the extent passed for the
// planner's sizing is irrelevant and a fixed one is used. Members of a planner
// slot keep its order (lifetime-start). Falls back to one slot per spec if the
// worst-case graph fails to compile, leaving the build render-neutral.
fn group_by_plan(specs: Vec<TextureSpec>, ssao_enabled: bool) -> Vec<SlotSpec> {
    use crate::gfx::render_graph::{FrameGraphInputs, build_frame_graph, plan_aliasing};

    let mut inputs = FrameGraphInputs::all_off();
    inputs.bloom_enabled = true;
    inputs.ssao_enabled = ssao_enabled;

    let groups: Vec<Vec<usize>> = match build_frame_graph(&inputs) {
        Ok(graph) => {
            let plan = plan_aliasing(&graph, 1920, 1080);
            let by_label: std::collections::HashMap<&str, usize> = specs
                .iter()
                .enumerate()
                .map(|(i, s)| (s.label, i))
                .collect();
            let mut groups: Vec<Vec<usize>> = Vec::new();
            let mut grouped = vec![false; specs.len()];
            for slot in &plan.slots {
                let group: Vec<usize> = slot
                    .members
                    .iter()
                    .filter_map(|&res_idx| by_label.get(graph.resources[res_idx].label).copied())
                    .collect();
                for &si in &group {
                    grouped[si] = true;
                }
                if !group.is_empty() {
                    groups.push(group);
                }
            }
            // Any managed spec the planner did not place (no graph resource for
            // it) gets its own un-aliased slot.
            for (si, placed) in grouped.iter().enumerate() {
                if !placed {
                    groups.push(vec![si]);
                }
            }
            groups
        }
        Err(_) => (0..specs.len()).map(|i| vec![i]).collect(),
    };

    // Materialize: move each spec into its group's slot.
    let mut specs_opt: Vec<Option<TextureSpec>> = specs.into_iter().map(Some).collect();
    groups
        .into_iter()
        .map(|g| SlotSpec {
            members: g
                .into_iter()
                .map(|si| specs_opt[si].take().expect("each spec joins one group"))
                .collect(),
        })
        .collect()
}

// `bloom_top`: bloom mip 0, half the extent the bloom chain was built for,
// sampled by the composite. The prefilter writes it, the downsample chain reads
// it, the final upsample accumulates into it.
fn bloom_top_spec((width, height): (u32, u32)) -> TextureSpec {
    TextureSpec {
        label: "bloom_top",
        width,
        height,
        format: super::post::BLOOM_FORMAT,
    }
}

// `ao_output`: SSAO's blurred occlusion at full render resolution, single-channel
// R8, sampled by the main pass's ambient term.
fn ao_output_spec((width, height): (u32, u32)) -> TextureSpec {
    TextureSpec {
        label: "ao_output",
        width,
        height,
        format: super::post::SSAO_OCCLUSION_FORMAT,
    }
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
}

#[cfg(test)]
mod tests {
    use super::*;

    // `transient_slots` is pure CPU (it builds slot descriptions; no device), so
    // the planner-routed grouping is testable headlessly.

    #[test]
    fn ssao_and_bloom_alias_into_one_slot() {
        // Both managed: the planner sees `ao_output` (early: SsaoBlur -> Main)
        // and `bloom_top` (late: Bloom -> Composite) with disjoint lifetimes, so
        // the pool packs them into one shared slot -- one heap instead of two.
        let slots = transient_slots(true, (1024, 768), (512, 384));
        assert_eq!(
            slots.len(),
            1,
            "ao_output + bloom_top should share one slot"
        );
        let labels: Vec<&str> = slots[0].members.iter().map(|m| m.label).collect();
        assert!(labels.contains(&"ao_output"), "{labels:?}");
        assert!(labels.contains(&"bloom_top"), "{labels:?}");
        // `ao_output` uses the memory first, so it must be the first member:
        // members are kept in the planner's lifetime-start order.
        assert_eq!(slots[0].members[0].label, "ao_output", "{labels:?}");
    }

    #[test]
    fn bloom_top_alone_is_unshared() {
        // SSAO off: `bloom_top` is the only managed transient, so it sits in its
        // own single-member slot (no aliasing).
        let slots = transient_slots(false, (1024, 768), (512, 384));
        assert_eq!(slots.len(), 1);
        assert_eq!(slots[0].members.len(), 1);
        assert_eq!(slots[0].members[0].label, "bloom_top");
    }

    #[test]
    fn slots_carry_the_feature_formats_and_extents() {
        // The pool sizes and formats each member off its owning feature, so a
        // divergence here would silently mis-back the texture the feature binds.
        let slots = transient_slots(true, (1024, 768), (512, 384));
        let member = |label: &str| {
            slots[0]
                .members
                .iter()
                .find(|m| m.label == label)
                .expect("member present")
        };
        let ao = member("ao_output");
        assert_eq!((ao.width, ao.height), (1024, 768));
        assert_eq!(ao.format, super::super::post::SSAO_OCCLUSION_FORMAT);
        let bloom = member("bloom_top");
        assert_eq!((bloom.width, bloom.height), (512, 384));
        assert_eq!(bloom.format, super::super::post::BLOOM_FORMAT);
    }
}
