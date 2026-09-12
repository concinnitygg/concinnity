//! Temporal anti-aliasing, written once for every backend.
//!
//! The resolve blends the current scene with its own reprojected, neighborhood
//! clipped output from the previous frame, reading per-pixel motion from the
//! G-buffer's velocity channel. Everything that shape needs -- the accumulation
//! targets, the ring that decides which one this frame writes, the gate that
//! suppresses history until there is some, and the one draw -- is portable, so
//! it lives here and each backend contributes only its [`PostPassDevice`].
//!
//! The projection jitter that gives the accumulator fresh sample positions is
//! deliberately *not* here. It is applied to the camera matrix long before any
//! post pass encodes, and it drives the upscalers as well as this pass, so it
//! belongs to each backend's frame setup rather than to the resolve.

use alloc::string::String;
use alloc::vec::Vec;

use crate::render::render_graph::{
    ClearValue, PassId, PixelFormat, TextureDesc, TextureSize, TextureUsage,
};
use crate::render::uniforms::TaaParams;

use super::device::{
    PostBind, PostBlend, PostDraw, PostExtent, PostLoadOp, PostPassDevice, PostSampler,
    PostTargetState, PostTiming,
};
use super::history::HistoryRing;
use super::program::PostProgram;

/// How deep the accumulation ring is and how its write index walks.
///
/// The two differ on a backend whose downstream consumers are pre-bound per
/// frame in flight: there the write target is dictated by the frame slot, so
/// the ring is as deep as the frame count and the backend reports the slot.
/// Where nothing downstream is pinned, [`TaaRing::ping_pong`] is the whole
/// story.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TaaRing {
    /// Accumulation targets to create. Floored at two by the ring itself.
    pub slots: usize,
    /// The modulus the write index advances under.
    pub stride: usize,
}

impl TaaRing {
    /// Two targets, alternating every frame.
    pub fn ping_pong() -> Self {
        Self {
            slots: 2,
            stride: 2,
        }
    }

    /// One target per frame in flight, indexed by the frame slot. Floored at two
    /// targets so a single frame in flight still has a history target distinct
    /// from the one being written.
    pub fn per_frame(frames: usize) -> Self {
        Self {
            slots: frames.max(2),
            stride: frames.max(1),
        }
    }
}

/// The per-frame inputs one resolve draw reads, beyond the pass's own history.
pub struct TaaInputs<'t, D: PostPassDevice + ?Sized + 't> {
    /// The scene this frame produced: the reflection composite's output where a
    /// reflection path owns the scene, else the resolved HDR target.
    pub scene: D::TextureRef<'t>,
    /// The G-buffer's per-pixel motion vectors.
    pub velocity: D::TextureRef<'t>,
}

/// The temporal resolve: its pipeline, its accumulation targets, and the ring
/// that walks them.
///
/// Parameterized by the two resource types rather than by the device, so a
/// backend whose device value borrows (a Metal device plus a sampler, a Vulkan
/// device plus its per-frame arena) can still store the pass: the pipeline and
/// the target are owned handles with no lifetime of their own, and the device is
/// supplied per call.
pub struct TaaPass<Pipeline, Target> {
    pipeline: Pipeline,
    targets: Vec<Target>,
    ring: HistoryRing,
    desc: TextureDesc,
}

// The graph label the accumulation targets carry into a backend's own
// debug naming.
const TARGET_LABEL: &str = "taa_history";

/// The accumulation target's shape: a full-resolution single-sample HDR color
/// target that is both rendered to and sampled. Declared as a render-graph
/// description so a backend creates it through the same translation its
/// transient pool already applies to a pooled resource, rather than through a
/// second hand-written path per host.
fn target_desc() -> TextureDesc {
    TextureDesc {
        width: TextureSize::Drawable,
        height: TextureSize::Drawable,
        depth: 1,
        format: PixelFormat::Rgba16Float,
        sample_count: 1,
        array_layers: 1,
        mip_levels: 1,
        usage: TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
        clear: ClearValue::Color([0.0, 0.0, 0.0, 0.0]),
    }
}

/// Build the resolve pipeline on its own, without touching the targets. What
/// shader hot reload rebuilds and hands to [`TaaPass::swap_pipeline`].
pub fn build_pipeline<D: PostPassDevice>(device: &D) -> Result<D::Pipeline, String> {
    device.create_pipeline(
        PostProgram::TaaResolve,
        target_desc().format,
        PostBlend::Replace,
    )
}

impl<Pipeline, Target> TaaPass<Pipeline, Target> {
    /// Build the resolve pipeline and the accumulation targets at `extent`.
    pub fn new<D>(device: &D, ring: TaaRing, extent: PostExtent) -> Result<Self, String>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target>,
    {
        let desc = target_desc();
        let pipeline = build_pipeline(device)?;
        let ring = HistoryRing::new(ring.slots, ring.stride);
        let targets = create_targets(device, &desc, extent, ring.slots())?;
        Ok(Self {
            pipeline,
            targets,
            ring,
            desc,
        })
    }

    /// The target at `slot`.
    pub fn target(&self, slot: usize) -> &Target {
        &self.targets[slot % self.targets.len()]
    }

    /// The ring's own state: which slot it would write, and whether history is
    /// valid.
    pub fn ring(&self) -> &HistoryRing {
        &self.ring
    }

    /// Step the ring to the next frame. Called once per frame from the
    /// backend's own end-of-frame temporal bookkeeping, beside the jitter
    /// counter it advances in lockstep.
    pub fn advance(&mut self) {
        self.ring.advance();
    }

    // Which target `write` samples as history.
    fn history_slot(&self, write: usize) -> usize {
        let slots = self.targets.len();
        (write % slots + slots - 1) % slots
    }

    /// Recreate the accumulation targets at a new extent and forget the
    /// accumulated history, which was rendered at a resolution this one cannot
    /// reproject from. The caller has already idled the device.
    pub fn resize<D>(&mut self, device: &D, extent: PostExtent) -> Result<(), String>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target>,
    {
        let slots = self.ring.slots();
        // Dropped before the new set is created so a resize does not hold two
        // full-resolution HDR rings resident at once.
        self.targets.clear();
        self.targets = create_targets(device, &self.desc, extent, slots)?;
        self.ring.invalidate();
        Ok(())
    }

    /// Swap in a freshly built pipeline. Driven by shader hot reload; the caller
    /// has already idled the device, so the outgoing pipeline is not in flight.
    pub fn swap_pipeline(&mut self, pipeline: Pipeline) {
        self.pipeline = pipeline;
    }

    /// Encode the resolve: one fullscreen draw into ring slot `write`, blending
    /// `inputs.scene` with the slot before it under `inputs.velocity`.
    ///
    /// `write` is a parameter rather than the ring's own index because a backend
    /// whose consumers are pre-bound per frame in flight has no choice about
    /// which target a frame writes; it reports the frame slot and the ring
    /// depth matches.
    pub fn encode<'t, D>(
        &'t self,
        device: &D,
        rec: &D::Recorder,
        write: usize,
        inputs: TaaInputs<'t, D>,
    ) -> Result<(), String>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target>,
    {
        let slots = self.targets.len();
        let write = write % slots;
        let history = self.history_slot(write);
        let params = TaaParams {
            history_valid: if self.ring.valid() { 1.0 } else { 0.0 },
        };
        let binds = [
            PostBind {
                texture: inputs.scene,
                sampler: PostSampler::LinearClamp,
            },
            PostBind {
                texture: inputs.velocity,
                sampler: PostSampler::LinearClamp,
            },
            PostBind {
                texture: device.target_ref(&self.targets[history]),
                sampler: PostSampler::LinearClamp,
            },
        ];
        device.encode(
            rec,
            &PostDraw {
                target: device.target_attachment(&self.targets[write]),
                // The graph declares the history slot this frame writes as its
                // post-TAA scene.
                state: PostTargetState::Graph,
                // The fullscreen triangle covers every pixel, so nothing the
                // target already holds survives the draw.
                load: PostLoadOp::DontCare,
                timing: PostTiming::Whole(PassId::TaaResolve),
                pipeline: &self.pipeline,
                binds: &binds,
                constants: bytemuck::bytes_of(&params),
                label: "TAA resolve",
            },
        )
    }
}

// The ring's targets, all at the same resolved extent.
fn create_targets<D: PostPassDevice>(
    device: &D,
    desc: &TextureDesc,
    extent: PostExtent,
    slots: usize,
) -> Result<Vec<D::Target>, String> {
    let mut targets = Vec::with_capacity(slots);
    for _ in 0..slots {
        targets.push(device.create_target(TARGET_LABEL, desc, extent)?);
    }
    Ok(targets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::post::device::resolve_extent;

    #[test]
    fn the_accumulation_target_is_a_sampled_full_resolution_hdr_color_target() {
        let d = target_desc();
        assert_eq!(d.format, PixelFormat::Rgba16Float);
        assert_eq!(d.sample_count, 1);
        assert_eq!(d.mip_levels, 1);
        assert!(d.usage.contains(TextureUsage::RENDER_TARGET));
        assert!(d.usage.contains(TextureUsage::SHADER_READ));
        let e = PostExtent {
            width: 1600,
            height: 900,
        };
        assert_eq!(resolve_extent(&d, e), e);
    }

    #[test]
    fn a_ping_pong_ring_is_two_deep() {
        let r = TaaRing::ping_pong();
        assert_eq!((r.slots, r.stride), (2, 2));
    }

    #[test]
    fn a_per_frame_ring_is_as_deep_as_the_frame_count() {
        assert_eq!(
            TaaRing::per_frame(3),
            TaaRing {
                slots: 3,
                stride: 3
            }
        );
        // A single frame in flight still needs a second target: a pass cannot
        // sample the target it writes.
        assert_eq!(
            TaaRing::per_frame(1),
            TaaRing {
                slots: 2,
                stride: 1
            }
        );
    }
}
