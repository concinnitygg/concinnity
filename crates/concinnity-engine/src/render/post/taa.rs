//! Temporal anti-aliasing, written once for every backend.
//!
//! The resolve blends the current scene with its own reprojected, neighborhood
//! clipped output from the previous frame, reading per-pixel motion from the
//! G-buffer's velocity channel. It keeps a pixel's history only as far as last
//! frame's motion snapshot shows the same surface where the pixel was, so what
//! a moving object uncovers drops the object's history even where the object
//! mirrored the ground. Everything that shape needs -- the accumulation
//! targets, the ring that decides which one this frame writes, the gate that
//! suppresses history until there is some, the snapshot, and the two draws --
//! is portable, so it lives here and each backend contributes only its
//! [`PostPassDevice`].
//!
//! The projection jitter that gives the accumulator fresh sample positions is
//! deliberately *not* here. It is applied to the camera matrix long before any
//! post pass encodes, and it drives the upscalers as well as this pass, so it
//! belongs to each backend's frame setup rather than to the resolve.

use alloc::vec::Vec;

use crate::gfx::jitter::TAA_CYCLE;
use crate::render::backend::error::RenderResult;
use crate::render::render_graph::{
    ClearValue, PassId, PixelFormat, TextureDesc, TextureSize, TextureUsage,
};
use crate::render::temporal::reactive_mask::ReactiveReader;
use crate::render::uniforms::TaaParams;

use super::device::{
    PostBind, PostBlend, PostDraw, PostExtent, PostLoadOp, PostPassDevice, PostSampler,
    PostTargetState, PostTiming,
};
use super::history::HistoryRing;
use super::motion_snapshot::{self, MotionSnapshot, SnapshotSources, point};
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
    /// The G-buffer's view normal (`.rgb`) and linear depth (`.a`).
    pub normal_depth: D::TextureRef<'t>,
    /// The reactive mask, when a particle or transparent pass wrote it this
    /// frame.
    pub reactive: Option<D::TextureRef<'t>>,
    /// What the history is multiplied by to read at this frame's pre-exposure
    /// ([`crate::gfx::auto_exposure::PreExposure::history_scale`]).
    pub history_scale: f32,
}

/// The temporal resolve: its pipeline, its accumulation targets, the ring that
/// walks them, and the motion snapshot it compares surfaces against.
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
    snapshot: MotionSnapshot<Pipeline, Target>,
}

/// The resolve's two pipelines, built together. What shader hot reload
/// rebuilds and hands to [`TaaPass::swap_pipelines`].
pub struct TaaPipelines<Pipeline> {
    /// The temporal resolve.
    pub resolve: Pipeline,
    /// The G-buffer copy the next frame's resolve reads.
    pub snapshot: Pipeline,
}

// Thin-feature lock lost per frame the feature is not seen. A feature narrower
// than a pixel covers no pixel center on some jitter frames; its lock has to
// outlast the longest run of those for the resolve to hold its history, and a
// feature the cycle hits at all is hit at least once per cycle.
const LOCK_DECAY: f32 = 1.0 / TAA_CYCLE as f32;

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

/// Build both pipelines on their own, without touching the targets.
///
/// # Errors
///
/// Returns an error if a pipeline fails to build.
pub fn build_pipelines<D: PostPassDevice>(device: &D) -> RenderResult<TaaPipelines<D::Pipeline>> {
    Ok(TaaPipelines {
        resolve: build_resolve(device)?,
        snapshot: motion_snapshot::build_pipeline(device)?,
    })
}

fn build_resolve<D: PostPassDevice>(device: &D) -> RenderResult<D::Pipeline> {
    device.create_pipeline(
        PostProgram::TaaResolve,
        target_desc().format,
        PostBlend::Replace,
    )
}

impl<Pipeline, Target> TaaPass<Pipeline, Target> {
    /// Build the resolve pipeline and the accumulation targets at `extent`.
    ///
    /// # Errors
    ///
    /// Returns an error if the pipeline or a target cannot be created.
    pub fn new<D>(device: &D, ring: TaaRing, extent: PostExtent) -> RenderResult<Self>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target>,
    {
        let desc = target_desc();
        let pipeline = build_resolve(device)?;
        let ring = HistoryRing::new(ring.slots, ring.stride);
        let targets = create_targets(device, &desc, extent, ring.slots())?;
        Ok(Self {
            pipeline,
            targets,
            ring,
            desc,
            snapshot: MotionSnapshot::new(device, extent)?,
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
        self.snapshot.advance();
    }

    /// Forget the accumulated history, so the next resolve seeds it from the
    /// current frame instead of blending against a view it no longer matches.
    pub fn reset_history(&mut self) {
        self.ring.invalidate();
        self.snapshot.reset_history();
    }

    // Which target `write` samples as history.
    fn history_slot(&self, write: usize) -> usize {
        let slots = self.targets.len();
        (write % slots + slots - 1) % slots
    }

    /// Recreate the accumulation targets at a new extent and forget the
    /// accumulated history, which was rendered at a resolution this one cannot
    /// reproject from. The caller has already idled the device.
    ///
    /// # Errors
    ///
    /// Returns an error if a target cannot be created.
    pub fn resize<D>(&mut self, device: &D, extent: PostExtent) -> RenderResult<()>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target>,
    {
        let slots = self.ring.slots();
        // Dropped before the new set is created so a resize does not hold two
        // full-resolution HDR rings resident at once.
        self.targets.clear();
        self.targets = create_targets(device, &self.desc, extent, slots)?;
        self.ring.invalidate();
        self.snapshot.resize(device, extent)
    }

    /// Swap in freshly built pipelines. Driven by shader hot reload; the caller
    /// has already idled the device, so the outgoing pipelines are not in
    /// flight.
    pub fn swap_pipelines(&mut self, pipelines: TaaPipelines<Pipeline>) {
        self.pipeline = pipelines.resolve;
        self.snapshot.swap_pipeline(pipelines.snapshot);
    }

    /// Encode the resolve: one fullscreen draw into ring slot `write`, blending
    /// `inputs.scene` with the slot before it under `inputs.velocity`, then the
    /// snapshot of this frame's G-buffer the next resolve compares against.
    ///
    /// `write` is a parameter rather than the ring's own index because a backend
    /// whose consumers are pre-bound per frame in flight has no choice about
    /// which target a frame writes; it reports the frame slot and the ring
    /// depth matches.
    ///
    /// # Errors
    ///
    /// Returns an error if a snapshot slot cannot be addressed or the device
    /// fails to encode a draw.
    pub fn encode<'t, D>(
        &'t self,
        device: &D,
        rec: &D::Recorder,
        write: usize,
        inputs: &TaaInputs<'t, D>,
    ) -> RenderResult<()>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target> + 't,
    {
        let slots = self.targets.len();
        let write = write % slots;
        let history = self.history_slot(write);
        let surface = self.snapshot.previous()?;
        let params = TaaParams {
            history_valid: if self.ring.valid() { 1.0 } else { 0.0 },
            reactive_max: ReactiveReader::Taa
                .cap()
                .filter(|_| inputs.reactive.is_some())
                .unwrap_or(0.0),
            lock_decay: LOCK_DECAY,
            history_scale: inputs.history_scale,
            surface_valid: if surface.is_some() { 1.0 } else { 0.0 },
        };
        // Without a mask or a snapshot the slot still needs a texture; the
        // motion vectors stand in, and the resolve does not read them there.
        let reactive = inputs.reactive.unwrap_or(inputs.velocity);
        let surface = surface.map_or(inputs.velocity, |t| device.target_ref(t));
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
            PostBind {
                texture: reactive,
                sampler: PostSampler::LinearClamp,
            },
            point(inputs.normal_depth),
            point(surface),
        ];
        let timing = PassId::TaaResolve.timing();
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
                timing: PostTiming::First(timing),
                pipeline: &self.pipeline,
                binds: &binds,
                constants: bytemuck::bytes_of(&params),
                label: "TAA resolve",
            },
        )?;
        self.snapshot.encode(
            device,
            rec,
            &SnapshotSources {
                normal_depth: inputs.normal_depth,
                velocity: inputs.velocity,
            },
            PostTiming::Last(timing),
        )
    }
}

// The ring's targets, all at the same resolved extent.
fn create_targets<D: PostPassDevice>(
    device: &D,
    desc: &TextureDesc,
    extent: PostExtent,
    slots: usize,
) -> RenderResult<Vec<D::Target>> {
    let mut targets = Vec::with_capacity(slots);
    for _ in 0..slots {
        targets.push(device.create_target(TARGET_LABEL, desc, extent)?);
    }
    Ok(targets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::jitter::offset_in_cycle;
    use crate::render::post::device::resolve_extent;
    use crate::render::post::mock::{MockDevice, MockDraw, MockPipeline, MockTexture};

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
    fn a_history_reset_drops_the_history_until_the_next_advance() {
        let device = MockDevice::new();
        let extent = PostExtent {
            width: 64,
            height: 64,
        };
        let mut pass = TaaPass::new(&device, TaaRing::ping_pong(), extent).expect("pass");
        pass.advance();
        assert!(pass.ring().valid());
        pass.reset_history();
        assert!(!pass.ring().valid());
        assert_eq!(
            pass.ring().write(),
            0,
            "accumulation restarts at the first slot"
        );
        pass.advance();
        assert!(pass.ring().valid());
    }

    // Whether a vertical feature spanning `[left, left + width)` pixels covers
    // the center of pixel `pixel` on `frame`. Every backend's projection shifts
    // the scene right by the jitter offset (it subtracts the offset from the
    // view-z coefficient, and clip w is minus view z).
    fn covers(left: f32, width: f32, pixel: f32, frame: u32) -> bool {
        let [jx, _] = offset_in_cycle(frame, TAA_CYCLE);
        let center = pixel + 0.5;
        left + jx <= center && center < left + width + jx
    }

    #[test]
    fn a_feature_narrower_than_a_pixel_misses_every_pixel_on_some_frames() {
        // The night planet lamp post: 1.2 m wide at 1060 m is 0.73 px at
        // 1600x900, its face spanning x = 801.46..802.18. On two frames of the
        // cycle it covers no pixel center at all, so the frame holds no trace
        // of it.
        let missed = (0..TAA_CYCLE)
            .filter(|&f| !covers(801.46, 0.73, 801.0, f) && !covers(801.46, 0.73, 802.0, f))
            .count();
        assert_eq!(missed, 2);
    }

    #[test]
    fn a_thin_feature_stays_locked_between_the_frames_the_jitter_hits_it() {
        // Mirrors the resolve's lock: set on a frame the pixel sees the feature,
        // less `LOCK_DECAY` on every other. Once a pixel has seen a feature of
        // any width at any phase, the lock must never run out while the
        // feature stays put.
        for w in 1..=64 {
            let width = w as f32 / 64.0;
            for p in 0..64 {
                let left = p as f32 / 64.0;
                let mut lock: Option<f32> = None;
                for frame in 0..3 * TAA_CYCLE {
                    lock = if covers(left, width, 0.0, frame) {
                        Some(1.0)
                    } else {
                        lock.map(|l| (l - LOCK_DECAY).max(0.0))
                    };
                    if let Some(l) = lock {
                        assert!(
                            l > 0.0,
                            "width {width} at {left}: lock ran out on frame {frame}"
                        );
                    }
                }
            }
        }
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

    // The constants a recorded draw was given, read back as the lanes of
    // `TaaParams`.
    fn lanes(constants: &[u8]) -> Vec<f32> {
        constants
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    const SCENE: MockTexture = MockTexture::External(1);
    const VELOCITY: MockTexture = MockTexture::External(2);
    const NORMAL_DEPTH: MockTexture = MockTexture::External(4);
    // The mock numbers targets in creation order: the two accumulation slots,
    // then the snapshot's two.
    const SNAPSHOTS: [MockTexture; 2] = [MockTexture::Target(2), MockTexture::Target(3)];

    fn pass(device: &MockDevice) -> TaaPass<MockPipeline, usize> {
        let extent = PostExtent {
            width: 64,
            height: 64,
        };
        TaaPass::new(device, TaaRing::ping_pong(), extent).expect("pass")
    }

    fn inputs<'t>(reactive: Option<MockTexture>, history_scale: f32) -> TaaInputs<'t, MockDevice> {
        TaaInputs {
            scene: SCENE,
            velocity: VELOCITY,
            normal_depth: NORMAL_DEPTH,
            reactive,
            history_scale,
        }
    }

    fn encode<'t>(
        device: &MockDevice,
        pass: &'t TaaPass<MockPipeline, usize>,
        write: usize,
        inputs: &TaaInputs<'t, MockDevice>,
    ) -> Vec<MockDraw> {
        device.draws.borrow_mut().clear();
        pass.encode(device, &(), write, inputs).expect("encode");
        device.draws.borrow().clone()
    }

    fn binds(draw: &MockDraw) -> Vec<MockTexture> {
        draw.binds.iter().map(|b| b.0).collect()
    }

    // The resolve writes this frame's slot from the scene, the motion vectors,
    // the slot before it and the G-buffer, standing the motion vectors in for a
    // reactive mask no pass wrote and a snapshot no frame kept, which it then
    // ignores. The frame's snapshot follows it.
    #[test]
    fn the_resolve_reads_the_previous_slot_into_this_one_then_snapshots() {
        let device = MockDevice::new();
        let pass = pass(&device);
        let draws = encode(&device, &pass, 1, &inputs(None, 2.0));
        let [resolve, snapshot] = draws.as_slice() else {
            panic!("two draws, got {}", draws.len());
        };
        let timing = PassId::TaaResolve.timing();
        assert_eq!(resolve.program, PostProgram::TaaResolve);
        assert_eq!(resolve.target, MockTexture::Target(1));
        assert_eq!(resolve.timing, PostTiming::First(timing));
        assert_eq!(
            binds(resolve),
            [
                SCENE,
                VELOCITY,
                MockTexture::Target(0),
                VELOCITY,
                NORMAL_DEPTH,
                VELOCITY
            ]
        );
        assert_eq!(lanes(&resolve.constants), [0.0, 0.0, LOCK_DECAY, 2.0, 0.0]);
        assert_eq!(snapshot.program, PostProgram::MotionSnapshot);
        assert_eq!(snapshot.target, SNAPSHOTS[0]);
        assert_eq!(snapshot.timing, PostTiming::Last(timing));
        assert_eq!(binds(snapshot), [NORMAL_DEPTH, VELOCITY]);
    }

    // Once the ring holds history it is read, and a mask written this frame is
    // bound and capped at what TAA's reader allows.
    #[test]
    fn valid_history_and_a_written_mask_reach_the_resolve() {
        let device = MockDevice::new();
        let mut pass = pass(&device);
        pass.advance();
        let mask = MockTexture::External(3);
        let draws = encode(&device, &pass, 0, &inputs(Some(mask), 1.0));
        let draw = &draws[0];
        assert_eq!(draw.target, MockTexture::Target(0));
        assert_eq!(
            draw.binds[2].0,
            MockTexture::Target(1),
            "the slot before slot 0"
        );
        assert_eq!(draw.binds[3].0, mask);
        let cap = ReactiveReader::Taa.cap().expect("TAA reads the mask");
        assert_eq!(lanes(&draw.constants), [1.0, cap, LOCK_DECAY, 1.0, 0.0]);
    }

    // The snapshot a frame keeps is what the next frame's resolve compares
    // surfaces against, while that frame keeps its own in the other slot; a
    // reset forgets it with the history.
    #[test]
    fn the_next_resolve_compares_against_the_kept_snapshot() {
        let device = MockDevice::new();
        let mut pass = pass(&device);
        encode(&device, &pass, 0, &inputs(None, 1.0));
        pass.advance();
        let draws = encode(&device, &pass, 1, &inputs(None, 1.0));
        assert_eq!(draws[0].binds[5].0, SNAPSHOTS[0]);
        assert_eq!(lanes(&draws[0].constants)[4], 1.0);
        assert_eq!(draws[1].target, SNAPSHOTS[1]);

        pass.advance();
        pass.reset_history();
        let draws = encode(&device, &pass, 0, &inputs(None, 1.0));
        assert_eq!(draws[0].binds[5].0, VELOCITY);
        assert_eq!(lanes(&draws[0].constants)[..1], [0.0]);
        assert_eq!(lanes(&draws[0].constants)[4], 0.0);
    }
}
