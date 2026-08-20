// src/directx/transient_pool.rs
//
// Backing store for the render graph's transient render targets on D3D12. The
// shared `gfx::render_graph::alias` planner decides which transients can share
// physical memory; this pool realises that on D3D12 with placed resources on an
// `ID3D12Heap` (the analogue of Vulkan's aliased `VkImage`s on a shared
// `VkDeviceMemory`). Features stop owning these resources and read them back by
// label, so the pool can repoint several labels at one heap region without
// touching the features.
//
// Buffering: D3D12 is single-buffered for these targets. The command queue runs
// frames in submission order and the per-resource state-transition barriers
// serialise a frame's writes against a prior frame's reads of the same resource,
// so a single resource is safe across frames in flight (unlike Vulkan, whose
// explicit-layout model led that backend to per-frame buffer its bloom chain).
// So a DX alias slot is ONE shared heap region (not per-frame): making the
// members per-frame would multiply already-single-buffered resources and cost
// more memory than aliasing saves. The cross-frame reuse ordering is carried by
// aliasing barriers at both reuse boundaries (added when sharing lands).
//
// A resource is "managed" iff its owning feature is enabled at build time (e.g.
// `ao_output` only when SSAO is on); `resource_for` returns `None` otherwise and
// the consumer keeps its disabled-feature fallback.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::texture::{one_shot_submit, transition_barrier};
use crate::gfx::render_graph::{
    ClearValue, PixelFormat, PoolGates, TextureUsage, TransientSlot, TransientTexture,
    plan_pool_slots,
};

struct PlacedResource {
    label: &'static str,
    resource: ID3D12Resource,
}

// The transient render-target pool owned by `DxContext`. Resolution-dependent,
// so it is rebuilt on swapchain resize. COM resources release on drop, so the
// old heaps + resources free when the pool is reassigned (the caller has already
// idled the device).
pub(super) struct TransientResourcePool {
    // One heap per slot. Held only to keep the heaps alive: a placed resource
    // does not keep its heap alive (D3D12 requires the heap to outlive the
    // resource), so the pool must retain them. Never read after construction.
    #[allow(dead_code)]
    heaps: Vec<ID3D12Heap>,
    resources: Vec<PlacedResource>,
    // For each member of a shared (multi-member) slot, its cyclic predecessor:
    // the label of the resource whose heap memory it reclaims. Cyclic because
    // D3D12 is single-buffered, so the first member reclaims from the last across
    // the frame boundary (the wrap), giving every shared member a predecessor.
    // Empty when no slot is shared. Drives the executor's aliasing barriers.
    alias_pred: Vec<(&'static str, &'static str)>,
    // The member labels of each slot, in the order they reuse its heap region.
    // Read back by the executor's per-frame soundness assertion.
    slot_labels: Vec<Vec<&'static str>>,
    // The pool's aliased footprint: the sum of its slot heap sizes. Reported to
    // the memory ledger, which would otherwise not see this pool at all -- it
    // deliberately sits off the device allocator.
    allocated_bytes: u64,
}

impl TransientResourcePool {
    // Allocate one heap per slot, sized to the largest member, and place every
    // member resource at offset 0. Each is created in its `initial_state` with
    // its optimized clear value, exactly as the committed version was, so its
    // first-use barrier is unchanged.
    pub(super) fn build(
        device: &ID3D12Device,
        queue: &ID3D12CommandQueue,
        slots: &[TransientSlot],
    ) -> Result<Self, String> {
        let mut heaps = Vec::new();
        let mut resources = Vec::new();
        let mut alias_pred: Vec<(&'static str, &'static str)> = Vec::new();
        let slot_labels: Vec<Vec<&'static str>> = slots.iter().map(|s| s.labels()).collect();
        // `allocated_bytes` is what the pool really reserves (one heap per slot,
        // sized to its largest member); `unaliased_bytes` is what the same
        // members would cost one heap each. Their difference is the aliasing
        // saving, which is otherwise invisible: the ledger sees only the total,
        // so a plan that quietly stopped sharing would look like a bigger scene.
        let mut allocated_bytes: u64 = 0;
        let mut unaliased_bytes: u64 = 0;
        // (resource, resting state) for the one-shot init below. A placed
        // render-target resource is NOT auto-zeroed like a committed one, so
        // D3D12 rejects its first draw/sample until a Clear/Discard/Copy
        // initializes it. Discard suffices (no need to define the contents):
        // every managed transient is fully written each frame before it is read
        // (the SSAO blur writes `ao_output`, the bloom prefilter writes
        // `bloom_top`), and a consumer that may run while a target is unwritten
        // guards its read (the composite skips `bloom_top` when bloom is off),
        // so the undefined initial contents are never observed. Only single-
        // member slots are initialized here; a shared slot's members are
        // re-initialized per frame by the executor's aliasing barrier + Discard
        // before each first write (Discarding them here, on shared memory with no
        // aliasing barrier between, would itself be an aliasing hazard).
        let mut to_init: Vec<(ID3D12Resource, D3D12_RESOURCE_STATES)> = Vec::new();
        for slot in slots {
            let shared = slot.members.len() > 1;
            // Size the heap to the largest member; offset 0 satisfies every
            // member's alignment, so aliased members all place there.
            let mut slot_size: u64 = 0;
            let mut slot_align: u64 = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64;
            let descs: Vec<(&TransientTexture, D3D12_RESOURCE_DESC)> = slot
                .members
                .iter()
                .map(|m| {
                    let desc = rt_desc(m);
                    // SAFETY: a query on a live COM object; the descriptor it reads and the out-
                    // parameters it fills are live locals that outlive the call.
                    let info = unsafe { device.GetResourceAllocationInfo(0, &[desc]) };
                    slot_size = slot_size.max(info.SizeInBytes);
                    slot_align = slot_align.max(info.Alignment);
                    unaliased_bytes += info.SizeInBytes;
                    (m, desc)
                })
                .collect();

            let heap_desc = D3D12_HEAP_DESC {
                SizeInBytes: slot_size,
                Properties: D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                },
                Alignment: slot_align,
                // These targets are all render targets, so a heap restricted to
                // RT/DS textures is valid on every resource-heap tier.
                Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES,
            };
            allocated_bytes += slot_size;
            let mut heap: Option<ID3D12Heap> = None;
            // SAFETY: the create descriptor and every pointer it borrows are live for the call, and
            // the new COM object lands in a binding that owns it.
            unsafe { device.CreateHeap(&heap_desc, &mut heap) }
                .map_err(|e| format!("transient pool heap: {e}"))?;
            let heap = heap.ok_or("transient pool heap returned None")?;

            for (m, desc) in &descs {
                let clear = clear_value(m);
                let mut res: Option<ID3D12Resource> = None;
                // SAFETY: the create descriptor and every pointer it borrows are live for the call,
                // and the new COM object lands in a binding that owns it.
                unsafe {
                    device.CreatePlacedResource(
                        &heap,
                        0,
                        desc,
                        resting_state(m),
                        Some(&clear),
                        &mut res,
                    )
                }
                .map_err(|e| format!("transient pool place {}: {e}", m.label))?;
                let resource = res.ok_or("transient pool placed resource None")?;
                if !shared {
                    to_init.push((resource.clone(), resting_state(m)));
                }
                resources.push(PlacedResource {
                    label: m.label,
                    resource,
                });
            }
            // Wire each shared-slot member to its cyclic predecessor (the prior
            // member, the first to the last) so the executor can claim the memory
            // before each first write.
            if shared {
                let n = slot.members.len();
                for i in 0..n {
                    alias_pred.push((slot.members[i].label, slot.members[(i + n - 1) % n].label));
                }
            }
            heaps.push(heap);
        }

        // Initialize every placed resource (Discard in its RENDER_TARGET state,
        // then back to its resting state) so its first real use is legal.
        if !to_init.is_empty() {
            one_shot_submit(device, queue, |cmd| {
                for (res, resting) in &to_init {
                    // SAFETY: the command list is in the recording state, and every resource,
                    // descriptor and slice these commands name is live for the call.
                    unsafe {
                        cmd.ResourceBarrier(&[transition_barrier(
                            res,
                            *resting,
                            D3D12_RESOURCE_STATE_RENDER_TARGET,
                        )]);
                        cmd.DiscardResource(res, None);
                        cmd.ResourceBarrier(&[transition_barrier(
                            res,
                            D3D12_RESOURCE_STATE_RENDER_TARGET,
                            *resting,
                        )]);
                    }
                }
            })?;
        }

        tracing::info!(
            "transient heap pool: {} heap allocation(s), {} KiB ({} KiB saved by aliasing)",
            heaps.len(),
            allocated_bytes / 1024,
            unaliased_bytes.saturating_sub(allocated_bytes) / 1024,
        );
        Ok(Self {
            heaps,
            resources,
            alias_pred,
            slot_labels,
            allocated_bytes,
        })
    }

    // The managed resource for `label`, or `None` when the owning feature was
    // disabled at build time (so nothing was placed).
    pub(super) fn resource_for(&self, label: &str) -> Option<&ID3D12Resource> {
        self.resources
            .iter()
            .find(|r| r.label == label)
            .map(|r| &r.resource)
    }

    // The label of the resource whose heap memory `label` reclaims (its cyclic
    // slot predecessor), or `None` when `label` is not a shared-slot member (so
    // it is not aliased and needs no aliasing barrier). The executor emits an
    // aliasing barrier before the pass that first-writes any resource for which
    // this returns `Some`.
    pub(super) fn alias_predecessor(&self, label: &str) -> Option<&'static str> {
        self.alias_pred
            .iter()
            .find(|(l, _)| *l == label)
            .map(|(_, p)| *p)
    }

    // The pool's aliased footprint in bytes, for the memory ledger.
    pub(super) fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes
    }

    // The three pooled G-buffer colour targets, or `None` when the pool was
    // built without the G-buffer gate (no screen-space consumer, so the
    // pre-pass node is absent and nothing was placed). All three are placed
    // together or not at all, so a partial result is a planner bug rather than
    // a state a caller should handle.
    pub(super) fn gbuffer_pooled(&self) -> Option<super::post::gbuffer::GbufferPooled> {
        Some(super::post::gbuffer::GbufferPooled {
            normal_depth: self.resource_for("gbuffer_normal_depth")?.clone(),
            roughness: self.resource_for("gbuffer_roughness")?.clone(),
            velocity: self.resource_for("gbuffer_velocity")?.clone(),
        })
    }

    // The member labels of each slot, for the executor's per-frame check that
    // no slot has two resources live at once in the graph it is about to run.
    pub(super) fn slot_labels(&self) -> &[Vec<&'static str>] {
        &self.slot_labels
    }

    // Rebuild every managed resource at a new extent. The caller has already
    // idled the device; reassigning drops the old heaps + placed resources
    // (COM release), so any feature descriptor that referenced them must be
    // rewritten by the caller afterward.
    pub(super) fn rebuild(
        &mut self,
        device: &ID3D12Device,
        queue: &ID3D12CommandQueue,
        slots: &[TransientSlot],
    ) -> Result<(), String> {
        *self = Self::build(device, queue, slots)?;
        Ok(())
    }
}

// The optimized clear value a pooled target is created with, from the graph's
// desc. D3D12 matches this against the value a real `Clear*View` passes: a
// mismatch is a debug-layer warning and costs the fast-clear path, so it is the
// graph's business rather than a constant here. A depth target carries a depth
// value; every colour target carries four floats.
fn clear_value(m: &TransientTexture) -> D3D12_CLEAR_VALUE {
    let format = dxgi_format(m.format);
    match m.clear {
        ClearValue::Color(color) => D3D12_CLEAR_VALUE {
            Format: format,
            Anonymous: D3D12_CLEAR_VALUE_0 { Color: color },
        },
        ClearValue::Depth(depth) => D3D12_CLEAR_VALUE {
            Format: format,
            Anonymous: D3D12_CLEAR_VALUE_0 {
                DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
                    Depth: depth,
                    Stencil: 0,
                },
            },
        },
    }
}

// Translate one graph-declared transient into its D3D12 resource desc. This is
// the backend's whole share of describing a pooled resource: the extent,
// format, mip count and flags all come from the graph, so there is no second
// table here that could disagree with it. `Alignment` 0 lets the runtime pick
// the default (64 KiB) placement alignment.
fn rt_desc(m: &TransientTexture) -> D3D12_RESOURCE_DESC {
    let (dimension, depth_or_array) = if m.depth.max(1) > 1 {
        (D3D12_RESOURCE_DIMENSION_TEXTURE3D, m.depth.max(1))
    } else {
        (D3D12_RESOURCE_DIMENSION_TEXTURE2D, m.array_layers.max(1))
    };
    D3D12_RESOURCE_DESC {
        Dimension: dimension,
        Alignment: 0,
        Width: m.width.max(1) as u64,
        Height: m.height.max(1),
        DepthOrArraySize: depth_or_array as u16,
        MipLevels: m.mip_levels.max(1) as u16,
        Format: dxgi_format(m.format),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: m.sample_count.max(1),
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: resource_flags(m.usage),
    }
}

fn dxgi_format(format: PixelFormat) -> DXGI_FORMAT {
    match format {
        PixelFormat::Rgba16Float => DXGI_FORMAT_R16G16B16A16_FLOAT,
        PixelFormat::Rgba8Unorm => DXGI_FORMAT_R8G8B8A8_UNORM,
        PixelFormat::Rg16Float => DXGI_FORMAT_R16G16_FLOAT,
        PixelFormat::R8Unorm => DXGI_FORMAT_R8_UNORM,
        PixelFormat::R32Float => DXGI_FORMAT_R32_FLOAT,
        PixelFormat::Depth32Float => DXGI_FORMAT_D32_FLOAT,
        PixelFormat::BgraSwapchain => DXGI_FORMAT_B8G8R8A8_UNORM,
    }
}

fn resource_flags(usage: TextureUsage) -> D3D12_RESOURCE_FLAGS {
    let mut flags = D3D12_RESOURCE_FLAG_NONE;
    if usage.contains(TextureUsage::RENDER_TARGET) {
        flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    }
    if usage.contains(TextureUsage::DEPTH_STENCIL) {
        flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    }
    if usage.contains(TextureUsage::STORAGE) {
        flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
    flags
}

// Where a pooled target sits between frames, which is also the state it is
// created in so its first derived transition names a state it is really in.
// Colour targets rest sampled, matching the barrier registry; a depth target
// would rest as its attachment, which is why this follows the declared usage
// rather than being one constant.
fn resting_state(m: &TransientTexture) -> D3D12_RESOURCE_STATES {
    if m.usage.contains(TextureUsage::DEPTH_STENCIL) {
        D3D12_RESOURCE_STATE_DEPTH_WRITE
    } else {
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
    }
}

// The alias-slot list for the transients the pool manages this build. The
// grouping, the pooled label set and each member's shape all come from the
// shared planner, so nothing here can disagree with the graph or with another
// backend.
//
// `bloom_top` is always managed: the bloom chain always exists and the
// composite samples mip 0 even when bloom is disabled, so a pool built at init
// / resize cannot gate on it. Metal does the same; Vulkan rebuilds on the flag
// and passes it through.
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

#[cfg(test)]
mod tests {
    use super::super::post::gbuffer::GBUFFER_ROUGHNESS_CLEAR;
    use super::*;

    // `transient_slots` is pure CPU (it builds slot descriptions; no device), so
    // the planner-routed grouping is testable headlessly.

    #[test]
    fn the_late_bloom_target_aliases_an_early_one() {
        // The pool's whole saving, in the configuration a real session runs:
        // SSAO on implies the G-buffer pre-pass is on, so both gates are true
        // here. `bloom_top` is the only genuinely late member (Bloom ->
        // Composite), so it is the one that can reuse an earlier member's
        // memory; everything else is live across most of the frame and needs its
        // own slot. A plan where `bloom_top` sits alone means the aliasing
        // stopped working.
        let slots = transient_slots(true, true, (1024, 768), (1024, 768)).expect("plans");
        let shared: Vec<Vec<&str>> = slots
            .iter()
            .map(|s| s.labels())
            .filter(|l| l.len() > 1)
            .collect();
        assert!(
            shared.iter().any(|l| l.contains(&"bloom_top")),
            "bloom_top should reuse an earlier member's heap region: {:?}",
            slots.iter().map(|s| s.labels()).collect::<Vec<_>>()
        );
        // Whatever it pairs with must start first: the pool's cyclic predecessor
        // wiring depends on lifetime-start order.
        let pair = shared
            .iter()
            .find(|l| l.contains(&"bloom_top"))
            .expect("checked above");
        assert_ne!(pair[0], "bloom_top", "{pair:?}");
    }

    #[test]
    fn bloom_top_alone_is_unshared() {
        // SSAO off: `bloom_top` is the only managed transient, so it sits in its
        // own single-member slot (no aliasing, no aliasing barriers).
        let slots = transient_slots(false, false, (1024, 768), (1024, 768)).expect("plans");
        assert_eq!(slots.len(), 1);
        assert_eq!(slots[0].members.len(), 1);
        assert_eq!(slots[0].members[0].label, "bloom_top");
    }

    #[test]
    fn the_gbuffer_colour_targets_are_pooled_and_depth_is_not() {
        // The G-buffer group migration: the three colour targets join the pool,
        // and `gbuffer_depth` stays feature-owned because D3D12 needs a typeless
        // resource format for a shader-readable depth target (see `pooled`).
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
        // resource the pool never created.
        let slots = transient_slots(true, false, (1024, 768), (1024, 768)).expect("plans");
        let labels: Vec<&str> = slots.iter().flat_map(|s| s.labels()).collect();
        assert!(!labels.contains(&"gbuffer_normal_depth"), "{labels:?}");
    }

    #[test]
    fn the_roughness_clear_matches_the_feature_constant() {
        // The reason `TextureDesc` models a clear value at all. D3D12 bakes an
        // optimized clear into a placed resource, and roughness clears to 1.0
        // (fully rough) rather than 0: a mismatch here costs the fast-clear path
        // and, if the pool won, would make untouched pixels mirror-smooth.
        let slots = transient_slots(true, true, (1024, 768), (1024, 768)).expect("plans");
        let roughness = slots
            .iter()
            .flat_map(|s| &s.members)
            .find(|m| m.label == "gbuffer_roughness")
            .expect("roughness pooled");
        assert_eq!(
            roughness.clear,
            crate::gfx::render_graph::ClearValue::Color(GBUFFER_ROUGHNESS_CLEAR)
        );
    }

    #[test]
    fn translated_descs_match_the_feature_formats() {
        // The graph is the single source of the shape now, so what this pins is
        // the *translation*: a divergence from each feature's own constant
        // would silently mis-back the resource that feature binds.
        let slots = transient_slots(true, true, (1024, 768), (1920, 1080)).expect("plans");
        let member = |label: &str| {
            slots
                .iter()
                .flat_map(|s| &s.members)
                .find(|m| m.label == label)
                .unwrap_or_else(|| panic!("{label} pooled"))
                .clone()
        };

        // `ao_output` follows the render extent; `bloom_top` is half the output
        // extent, which is what `create_bloom_mips` sizes mip 0 to.
        let ao = member("ao_output");
        let ao_desc = rt_desc(&ao);
        assert_eq!((ao_desc.Width, ao_desc.Height), (1024, 768));
        assert_eq!(
            ao_desc.Format,
            super::super::post::ssao::SSAO_OCCLUSION_FORMAT
        );
        assert_eq!(ao_desc.MipLevels, 1);
        assert_eq!(ao_desc.SampleDesc.Count, 1);
        assert_eq!(ao_desc.Dimension, D3D12_RESOURCE_DIMENSION_TEXTURE2D);
        assert_eq!(ao_desc.Flags, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        // Rests sampled, matching the barrier registry's resting state.
        assert_eq!(
            resting_state(&ao),
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
        );

        let bloom = member("bloom_top");
        let bloom_desc = rt_desc(&bloom);
        assert_eq!((bloom_desc.Width, bloom_desc.Height), (960, 540));
        assert_eq!(bloom_desc.Format, super::super::texture::HDR_FORMAT);

        // The G-buffer colour targets. A format divergence here would silently
        // mis-back the resource the pre-pass MRT binds, and the render-target
        // flag is what makes it bindable at all.
        use super::super::post::gbuffer::{
            GBUFFER_NORMAL_DEPTH_FORMAT, GBUFFER_ROUGHNESS_FORMAT, GBUFFER_VELOCITY_FORMAT,
        };
        for (label, format) in [
            ("gbuffer_normal_depth", GBUFFER_NORMAL_DEPTH_FORMAT),
            ("gbuffer_roughness", GBUFFER_ROUGHNESS_FORMAT),
            ("gbuffer_velocity", GBUFFER_VELOCITY_FORMAT),
        ] {
            let desc = rt_desc(&member(label));
            assert_eq!(desc.Format, format, "{label}");
            // Render resolution, not the drawable: the pre-pass rasterises at
            // the scene resolution, which differs under temporal upscaling.
            assert_eq!((desc.Width, desc.Height), (1024, 768), "{label}");
            assert_eq!(desc.SampleDesc.Count, 1, "{label} rasterises once");
            assert_eq!(
                desc.Flags, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                "{label}"
            );
        }
    }

    #[test]
    fn depth_and_volume_shapes_translate() {
        // Nothing pooled needs these yet, but the descs the graph now carries
        // do, so the translator has to be right before they can be pooled. A
        // depth transient in particular rests in its attachment state, not
        // sampled, which is why resting is derived rather than constant.
        let depth = TransientTexture {
            label: "probe_depth",
            width: 8,
            height: 8,
            depth: 1,
            format: PixelFormat::Depth32Float,
            sample_count: 4,
            array_layers: 1,
            mip_levels: 1,
            usage: TextureUsage::DEPTH_STENCIL.union(TextureUsage::SHADER_READ),
            clear: ClearValue::Depth(1.0),
        };
        let desc = rt_desc(&depth);
        assert_eq!(desc.Format, DXGI_FORMAT_D32_FLOAT);
        assert_eq!(desc.SampleDesc.Count, 4);
        assert_eq!(desc.Flags, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
        assert_eq!(resting_state(&depth), D3D12_RESOURCE_STATE_DEPTH_WRITE);
        // A depth target's optimized clear must be the depth arm: handing
        // D3D12 a colour for a D32 resource is a creation failure.
        assert_eq!(
            // SAFETY: the union arm is the one `clear_value` just wrote for a `ClearValue::Depth`,
            // which the assertion above pins.
            unsafe { clear_value(&depth).Anonymous.DepthStencil.Depth },
            1.0
        );

        let volume = TransientTexture {
            label: "probe_volume",
            depth: 64,
            format: PixelFormat::Rgba16Float,
            sample_count: 1,
            usage: TextureUsage::STORAGE.union(TextureUsage::SHADER_READ),
            clear: ClearValue::Color([0.0; 4]),
            ..depth
        };
        let desc = rt_desc(&volume);
        assert_eq!(desc.Dimension, D3D12_RESOURCE_DIMENSION_TEXTURE3D);
        assert_eq!(desc.DepthOrArraySize, 64);
        assert_eq!(desc.Flags, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    }
}
