//! Screen-space global illumination, written once for every backend.
//!
//! Four stages at two resolutions. At the trace resolution (the render
//! resolution divided by `gi_scale`):
//!
//! 1. the closest and farthest linear depth of each trace pixel's footprint in
//!    the G-buffer, the top of a depth pyramid;
//! 2. the pyramid's remaining levels, each the depth range of the level below;
//! 3. the trace: cosine-weighted hemisphere rays walked through the pyramid,
//!    the lit scene color each one hits blended into last frame's
//!    accumulation, reprojected by the G-buffer's motion vectors and rejected
//!    where last frame's pyramid saw another surface.
//!
//! Then at full resolution, 4. the composite: a depth-aware upsample and
//! denoise of the accumulation, added into the scene.
//!
//! The pyramids and the accumulation are each a ping-pong pair, so the trace
//! reads what the previous frame wrote. It also reads the scene the composite
//! then writes. A backend whose resource states follow the graph has to move
//! the scene between those two draws, which is why the stages are also
//! encodable in three pieces.

use alloc::format;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};

use crate::gfx::render_types::SsgiParams;
use crate::render::error::{RenderError, RenderResult};
use crate::render::render_graph::{PassId, PixelFormat};

use super::device::{
    PostBind, PostBlend, PostDraw, PostExtent, PostLoadOp, PostPassDevice, PostSampler,
    PostTargetState, PostTiming, resolve_extent,
};
use super::history::HistoryRing;
use super::program::PostProgram;

/// Clamped SSGI tunables and the per-frame uniform they build.
pub mod settings;

/// The shapes of the targets the pass owns.
pub mod targets;

use settings::SsgiFrame;
use targets::{depth_desc, gi_desc, pyramid_levels};

/// The per-frame inputs the pass reads and writes beyond its own targets.
pub struct SsgiInputs<'t, D: PostPassDevice + ?Sized + 't> {
    /// The lit scene the trace samples its bounce radiance from.
    pub scene: D::TextureRef<'t>,
    /// The same scene, as the composite's target.
    pub scene_target: D::Attachment<'t>,
    /// The G-buffer's view normal (`.rgb`) and linear depth (`.a`).
    pub normal_depth: D::TextureRef<'t>,
    /// The G-buffer's per-pixel motion vectors.
    pub velocity: D::TextureRef<'t>,
}

/// The four pipelines, built together. What shader hot reload rebuilds and
/// hands to [`SsgiPass::swap_pipelines`].
pub struct SsgiPipelines<Pipeline> {
    /// The pyramid's top level from the G-buffer.
    pub depth: Pipeline,
    /// One more pyramid level from the one below.
    pub reduce: Pipeline,
    /// The hemisphere trace and its accumulation.
    pub trace: Pipeline,
    /// The upsample, adding into the scene.
    pub composite: Pipeline,
}

/// Build every pipeline on its own, without touching the targets.
pub fn build_pipelines<D: PostPassDevice>(device: &D) -> RenderResult<SsgiPipelines<D::Pipeline>> {
    let depth = PixelFormat::Rg32Float;
    let color = PixelFormat::Rgba16Float;
    Ok(SsgiPipelines {
        depth: device.create_pipeline(PostProgram::SsgiDepth, depth, PostBlend::Replace)?,
        reduce: device.create_pipeline(PostProgram::SsgiReduce, depth, PostBlend::Replace)?,
        trace: device.create_pipeline(PostProgram::SsgiTrace, color, PostBlend::Replace)?,
        composite: device.create_pipeline(
            PostProgram::SsgiComposite,
            color,
            PostBlend::Additive,
        )?,
    })
}

// The graph labels the targets carry into a backend's own debug naming.
const DEPTH_LABEL: &str = "ssgi_depth";
const HISTORY_LABEL: &str = "ssgi_history";

// The targets sized to one render resolution: a pyramid and an accumulation
// per ring slot.
struct SsgiTargets<Target> {
    depth: Vec<Target>,
    history: Vec<Target>,
    levels: u32,
}

impl<Target> SsgiTargets<Target> {
    fn new<D>(device: &D, gi_scale: u32, extent: PostExtent, slots: usize) -> RenderResult<Self>
    where
        D: PostPassDevice<Target = Target>,
    {
        let depth = depth_desc(gi_scale, extent);
        let gi = gi_desc(gi_scale);
        let mut targets = Self {
            depth: Vec::with_capacity(slots),
            history: Vec::with_capacity(slots),
            levels: pyramid_levels(resolve_extent(&gi, extent)),
        };
        for _ in 0..slots {
            targets
                .depth
                .push(device.create_target(DEPTH_LABEL, &depth, extent)?);
            targets
                .history
                .push(device.create_target(HISTORY_LABEL, &gi, extent)?);
        }
        Ok(targets)
    }

    // The pyramid in ring slot `slot`.
    fn depth(&self, slot: usize) -> RenderResult<&Target> {
        ring_slot(&self.depth, slot, DEPTH_LABEL)
    }

    // The accumulation in ring slot `slot`.
    fn history(&self, slot: usize) -> RenderResult<&Target> {
        ring_slot(&self.history, slot, HISTORY_LABEL)
    }
}

// Ring slot `slot` of `targets`. A resize that failed part way leaves the ring
// empty, which a frame reports rather than indexes.
fn ring_slot<'a, Target>(
    targets: &'a [Target],
    slot: usize,
    label: &str,
) -> RenderResult<&'a Target> {
    targets
        .get(slot)
        .ok_or_else(|| RenderError::Other(format!("{label}: no target in ring slot {slot}")))
}

/// The SSGI pipelines, the targets between them, and the ring that decides
/// which pyramid and accumulation slot a frame writes.
///
/// Parameterized by the two resource types rather than by the device, for the
/// same reason as the temporal resolve: a backend's device value borrows, and
/// the pass is stored on its context.
pub struct SsgiPass<Pipeline, Target> {
    pipelines: SsgiPipelines<Pipeline>,
    targets: SsgiTargets<Target>,
    ring: HistoryRing,
    // Set when this frame's trace is encoded, so `advance` only keeps a history
    // slot that was actually written.
    traced: AtomicBool,
    frame: u32,
    gi_scale: u32,
}

impl<Pipeline, Target> SsgiPass<Pipeline, Target> {
    /// Build every pipeline and target for a render resolution of `extent`.
    pub fn new<D>(device: &D, gi_scale: u32, extent: PostExtent) -> RenderResult<Self>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target>,
    {
        let ring = HistoryRing::ping_pong();
        Ok(Self {
            pipelines: build_pipelines(device)?,
            targets: SsgiTargets::new(device, gi_scale, extent, ring.slots())?,
            ring,
            traced: AtomicBool::new(false),
            frame: 0,
            gi_scale,
        })
    }

    /// Where the accumulation stands this frame, for the frame's uniform.
    pub fn frame(&self) -> SsgiFrame {
        SsgiFrame {
            frame: self.frame,
            history_valid: self.ring.valid(),
            levels: self.targets.levels,
        }
    }

    /// Step to the next frame: what this frame's trace wrote becomes next
    /// frame's history, and the trace draws fresh directions. Called once per
    /// frame from the backend's own end-of-frame temporal bookkeeping. A frame
    /// that never encoded the trace wrote no slot, so it forgets the
    /// accumulation instead.
    pub fn advance(&mut self) {
        if self.traced.swap(false, Ordering::Relaxed) {
            self.ring.advance();
        } else {
            self.ring.invalidate();
        }
        self.frame = self.frame.wrapping_add(1);
    }

    /// Forget the accumulation, so the next trace starts over from the current
    /// frame instead of reprojecting a view it no longer matches.
    pub fn reset_history(&mut self) {
        self.ring.invalidate();
    }

    /// Recreate every target for a new render resolution and forget the
    /// accumulation, which was rendered at a resolution this one cannot
    /// reproject from. The caller has already idled the device.
    pub fn resize<D>(&mut self, device: &D, extent: PostExtent) -> RenderResult<()>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target>,
    {
        let slots = self.ring.slots();
        // Released before the new set is created, so a resize does not hold two
        // sets of rings (and their descriptors) at once.
        self.targets.depth.clear();
        self.targets.history.clear();
        self.ring.invalidate();
        self.targets = SsgiTargets::new(device, self.gi_scale, extent, slots)?;
        Ok(())
    }

    /// Swap in freshly built pipelines. Driven by shader hot reload; the caller
    /// has already idled the device.
    pub fn swap_pipelines(&mut self, pipelines: SsgiPipelines<Pipeline>) {
        self.pipelines = pipelines;
    }

    /// Encode every stage in order.
    pub fn encode<'t, D>(
        &'t self,
        device: &D,
        rec: &D::Recorder,
        inputs: SsgiInputs<'t, D>,
        params: &SsgiParams,
    ) -> RenderResult<()>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target> + 't,
    {
        self.encode_pyramid(device, rec, inputs.normal_depth, params)?;
        self.encode_trace(
            device,
            rec,
            SsgiTraceInputs {
                scene: inputs.scene,
                normal_depth: inputs.normal_depth,
                velocity: inputs.velocity,
            },
            params,
        )?;
        self.encode_composite(
            device,
            rec,
            inputs.scene_target,
            inputs.normal_depth,
            params,
        )
    }

    /// Encode this frame's depth pyramid: its top level from `normal_depth`,
    /// then each level from the one below.
    pub fn encode_pyramid<'t, D>(
        &'t self,
        device: &D,
        rec: &D::Recorder,
        normal_depth: D::TextureRef<'t>,
        params: &SsgiParams,
    ) -> RenderResult<()>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target> + 't,
    {
        let pyramid = self.targets.depth(self.ring.write())?;
        device.encode(
            rec,
            &private_draw(
                device.target_level_attachment(pyramid, 0)?,
                PostTiming::First(PassId::Ssgi),
                &self.pipelines.depth,
                &[screen::<D>(normal_depth)],
                bytemuck::bytes_of(params),
                "SSGI depth",
            ),
        )?;
        for level in 1..self.targets.levels {
            device.encode(
                rec,
                &private_draw(
                    device.target_level_attachment(pyramid, level)?,
                    PostTiming::None,
                    &self.pipelines.reduce,
                    &[screen::<D>(device.target_level_ref(pyramid, level - 1)?)],
                    &[],
                    "SSGI reduce",
                ),
            )?;
        }
        Ok(())
    }

    /// Encode the trace: hemisphere rays walked through this frame's pyramid,
    /// sampling the scene where they hit, accumulated into this frame's slot.
    pub fn encode_trace<'t, D>(
        &'t self,
        device: &D,
        rec: &D::Recorder,
        inputs: SsgiTraceInputs<'t, D>,
        params: &SsgiParams,
    ) -> RenderResult<()>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target> + 't,
    {
        let (write, read) = (self.ring.write(), self.ring.history());
        let targets = &self.targets;
        let encoded = device.encode(
            rec,
            &private_draw(
                device.target_attachment(targets.history(write)?),
                PostTiming::None,
                &self.pipelines.trace,
                &[
                    screen::<D>(inputs.scene),
                    screen::<D>(inputs.normal_depth),
                    screen::<D>(inputs.velocity),
                    screen::<D>(device.target_ref(targets.depth(write)?)),
                    screen::<D>(device.target_ref(targets.depth(read)?)),
                    screen::<D>(device.target_ref(targets.history(read)?)),
                ],
                bytemuck::bytes_of(params),
                "SSGI trace",
            ),
        );
        if encoded.is_ok() {
            self.traced.store(true, Ordering::Relaxed);
        }
        encoded
    }

    /// Encode the composite: this frame's accumulation upsampled over
    /// `normal_depth` and added into `scene_target`.
    pub fn encode_composite<'t, D>(
        &'t self,
        device: &D,
        rec: &D::Recorder,
        scene_target: D::Attachment<'t>,
        normal_depth: D::TextureRef<'t>,
        params: &SsgiParams,
    ) -> RenderResult<()>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target> + 't,
    {
        let write = self.ring.write();
        let binds = [
            screen::<D>(device.target_ref(self.targets.history(write)?)),
            screen::<D>(device.target_ref(self.targets.depth(write)?)),
            screen::<D>(normal_depth),
        ];
        device.encode(
            rec,
            &PostDraw {
                target: scene_target,
                // The scene is the graph's spine, which this node writes.
                state: PostTargetState::Graph,
                // The blend adds onto what the scene already holds.
                load: PostLoadOp::Load,
                timing: PostTiming::Last(PassId::Ssgi),
                pipeline: &self.pipelines.composite,
                binds: &binds,
                constants: bytemuck::bytes_of(params),
                label: "SSGI composite",
            },
        )
    }
}

/// What the trace reads beyond the pass's own targets.
pub struct SsgiTraceInputs<'t, D: PostPassDevice + ?Sized + 't> {
    /// The lit scene the rays sample their bounce radiance from.
    pub scene: D::TextureRef<'t>,
    /// The G-buffer's view normal (`.rgb`) and linear depth (`.a`).
    pub normal_depth: D::TextureRef<'t>,
    /// The G-buffer's per-pixel motion vectors.
    pub velocity: D::TextureRef<'t>,
}

// A draw into one of the pass's own targets, which only this pass reads, so the
// graph does not declare it.
fn private_draw<'a, 't, D: PostPassDevice + ?Sized + 't>(
    target: D::Attachment<'t>,
    timing: PostTiming,
    pipeline: &'a D::Pipeline,
    binds: &'a [PostBind<'t, D>],
    constants: &'a [u8],
    label: &'a str,
) -> PostDraw<'a, 't, D> {
    PostDraw {
        target,
        state: PostTargetState::Pass,
        load: PostLoadOp::DontCare,
        timing,
        pipeline,
        binds,
        constants,
        label,
    }
}

fn screen<'t, D: PostPassDevice + ?Sized + 't>(texture: D::TextureRef<'t>) -> PostBind<'t, D> {
    PostBind {
        texture,
        sampler: PostSampler::LinearClamp,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::post::mock::{MockDevice, MockDraw, MockPipeline, MockTexture};
    use crate::render::post::ssgi::settings::SsgiSettings;
    use crate::render::post::ssgi::targets::DEPTH_LEVELS;
    use alloc::vec;

    const EXTENT: PostExtent = PostExtent {
        width: 1280,
        height: 720,
    };

    // Target indices in the order `SsgiTargets::new` creates them.
    const fn depth(slot: usize) -> usize {
        2 * slot
    }
    const fn history(slot: usize) -> usize {
        2 * slot + 1
    }

    const SCENE: MockTexture = MockTexture::External(1);
    const NORMAL_DEPTH: MockTexture = MockTexture::External(2);
    const VELOCITY: MockTexture = MockTexture::External(3);

    // Draws per frame: the pyramid's levels, the trace and the composite.
    const DRAWS: usize = DEPTH_LEVELS as usize + 2;

    fn params(pass: &SsgiPass<MockPipeline, usize>) -> SsgiParams {
        SsgiSettings::resolve(1.0, 8.0, 1, 2).params(1.0, 1.5, pass.frame())
    }

    fn encode_frame(device: &MockDevice, pass: &SsgiPass<MockPipeline, usize>) -> Vec<MockDraw> {
        device.draws.borrow_mut().clear();
        pass.encode(
            device,
            &(),
            SsgiInputs {
                scene: SCENE,
                scene_target: SCENE,
                normal_depth: NORMAL_DEPTH,
                velocity: VELOCITY,
            },
            &params(pass),
        )
        .expect("encode");
        device.draws.borrow().clone()
    }

    fn binds(draw: &MockDraw) -> Vec<MockTexture> {
        draw.binds.iter().map(|b| b.0).collect()
    }

    #[test]
    fn the_pass_creates_a_pyramid_and_an_accumulation_per_slot() {
        let device = MockDevice::new();
        SsgiPass::new(&device, 2, EXTENT).expect("pass");
        let targets = device.targets.borrow();
        assert_eq!(targets.len(), 4);
        let half = PostExtent {
            width: 640,
            height: 360,
        };
        for slot in 0..2 {
            assert_eq!(targets[depth(slot)].label, DEPTH_LABEL);
            assert_eq!(targets[depth(slot)].levels, DEPTH_LEVELS);
            assert_eq!(targets[history(slot)].label, HISTORY_LABEL);
            assert_eq!(targets[history(slot)].levels, 1);
        }
        assert!(targets.iter().all(|t| t.extent == half));
    }

    #[test]
    fn the_trace_resolution_is_the_size_the_settings_divide_to() {
        // The graph's fractional resolve has to land where the integer division
        // the settings expose does, for every scale an author can pick.
        for gi_scale in [1, 2, 4] {
            let s = SsgiSettings::resolve(1.0, 8.0, 1, gi_scale);
            for (w, h) in [(1280, 720), (1921, 1081), (3, 1), (2560, 1439), (1, 1)] {
                let e = resolve_extent(
                    &gi_desc(s.gi_scale),
                    PostExtent {
                        width: w,
                        height: h,
                    },
                );
                assert_eq!(
                    (e.width, e.height),
                    s.gi_dimensions(w, h),
                    "{w}x{h} at 1/{gi_scale}"
                );
            }
        }
    }

    #[test]
    fn a_frame_builds_the_pyramid_then_traces_and_composites() {
        let device = MockDevice::new();
        let pass = SsgiPass::new(&device, 2, EXTENT).expect("pass");
        let draws = encode_frame(&device, &pass);
        let programs: Vec<PostProgram> = draws.iter().map(|d| d.program).collect();
        let mut expected = vec![PostProgram::SsgiDepth];
        expected.extend((1..DEPTH_LEVELS).map(|_| PostProgram::SsgiReduce));
        expected.extend([PostProgram::SsgiTrace, PostProgram::SsgiComposite]);
        assert_eq!(programs, expected);

        // The pyramid: level 0 from the G-buffer, each level after it from the
        // one below, all in this frame's slot.
        let d = depth(0);
        assert_eq!(draws[0].target, MockTexture::Level(d, 0));
        assert_eq!(binds(&draws[0]), [NORMAL_DEPTH]);
        for level in 1..DEPTH_LEVELS {
            let reduce = &draws[level as usize];
            assert_eq!(reduce.target, MockTexture::Level(d, level));
            assert_eq!(binds(reduce), [MockTexture::Level(d, level - 1)]);
            assert!(reduce.constants.is_empty());
        }

        let n = DEPTH_LEVELS as usize;
        let (trace, composite) = (&draws[n], &draws[n + 1]);
        assert_eq!(trace.target, MockTexture::Target(history(0)));
        assert_eq!(
            binds(trace),
            [
                SCENE,
                NORMAL_DEPTH,
                VELOCITY,
                MockTexture::Target(d),
                MockTexture::Target(depth(1)),
                MockTexture::Target(history(1)),
            ]
        );
        assert_eq!(composite.target, SCENE);
        assert_eq!(
            binds(composite),
            [
                MockTexture::Target(history(0)),
                MockTexture::Target(d),
                NORMAL_DEPTH
            ]
        );
    }

    #[test]
    fn only_the_composite_touches_the_graph_or_blends() {
        let device = MockDevice::new();
        let pass = SsgiPass::new(&device, 2, EXTENT).expect("pass");
        let draws = encode_frame(&device, &pass);
        let (composite, private) = draws.split_last().expect("draws");
        assert_eq!(composite.state, PostTargetState::Graph);
        assert_eq!(composite.load, PostLoadOp::Load);
        for d in private {
            assert_eq!(d.state, PostTargetState::Pass, "{}", d.label);
            assert_eq!(d.load, PostLoadOp::DontCare, "{}", d.label);
        }
        let p = build_pipelines(&device).expect("pipelines");
        assert_eq!(p.composite.blend, PostBlend::Additive);
        for other in [&p.depth, &p.reduce, &p.trace] {
            assert_eq!(other.blend, PostBlend::Replace);
        }
        assert_eq!(p.depth.format, PixelFormat::Rg32Float);
        assert_eq!(p.reduce.format, PixelFormat::Rg32Float);
        assert_eq!(p.trace.format, PixelFormat::Rgba16Float);
    }

    #[test]
    fn the_timing_span_opens_on_the_first_draw_and_closes_on_the_last() {
        let device = MockDevice::new();
        let pass = SsgiPass::new(&device, 2, EXTENT).expect("pass");
        let draws = encode_frame(&device, &pass);
        let timings: Vec<PostTiming> = draws.iter().map(|d| d.timing).collect();
        assert_eq!(timings[0], PostTiming::First(PassId::Ssgi));
        assert_eq!(timings[timings.len() - 1], PostTiming::Last(PassId::Ssgi));
        assert!(
            timings[1..timings.len() - 1]
                .iter()
                .all(|t| *t == PostTiming::None)
        );
    }

    #[test]
    fn consecutive_frames_swap_the_slots_they_write_and_read() {
        let device = MockDevice::new();
        let mut pass = SsgiPass::new(&device, 2, EXTENT).expect("pass");
        let n = DEPTH_LEVELS as usize;
        let first = encode_frame(&device, &pass);
        pass.advance();
        let second = encode_frame(&device, &pass);
        assert_eq!(first[0].target, MockTexture::Level(depth(0), 0));
        assert_eq!(second[0].target, MockTexture::Level(depth(1), 0));
        assert_eq!(second[n].target, MockTexture::Target(history(1)));
        // Last frame's pyramid and accumulation are this frame's history.
        let trace = binds(&second[n]);
        assert_eq!(trace[4], MockTexture::Target(depth(0)));
        assert_eq!(trace[5], MockTexture::Target(history(0)));
    }

    #[test]
    fn history_is_valid_only_after_a_frame_and_a_resize_forgets_it() {
        let device = MockDevice::new();
        let mut pass = SsgiPass::new(&device, 2, EXTENT).expect("pass");
        assert!(!pass.frame().history_valid);
        assert_eq!(params(&pass).history_valid, 0.0);
        encode_frame(&device, &pass);
        pass.advance();
        assert!(pass.frame().history_valid);
        assert_eq!(params(&pass).history_valid, 1.0);
        pass.resize(
            &device,
            PostExtent {
                width: 640,
                height: 360,
            },
        )
        .expect("resize");
        assert!(!pass.frame().history_valid);
    }

    #[test]
    fn a_history_reset_forgets_the_accumulation_until_the_next_trace() {
        let device = MockDevice::new();
        let mut pass = SsgiPass::new(&device, 2, EXTENT).expect("pass");
        encode_frame(&device, &pass);
        pass.advance();
        assert!(pass.frame().history_valid);
        pass.reset_history();
        assert!(!pass.frame().history_valid);
        encode_frame(&device, &pass);
        pass.advance();
        assert!(pass.frame().history_valid);
    }

    #[test]
    fn a_frame_that_skips_the_trace_forgets_the_history() {
        let device = MockDevice::new();
        let mut pass = SsgiPass::new(&device, 2, EXTENT).expect("pass");
        encode_frame(&device, &pass);
        pass.advance();
        assert!(pass.frame().history_valid);
        // Nothing wrote this frame's slot, so neither slot is next frame's
        // history.
        pass.advance();
        assert!(!pass.frame().history_valid);
        assert_eq!(
            encode_frame(&device, &pass)[0].target,
            MockTexture::Level(depth(0), 0)
        );
        pass.advance();
        assert!(pass.frame().history_valid);
    }

    #[test]
    fn a_resized_pass_encodes_a_whole_frame_at_the_new_size() {
        let device = MockDevice::new();
        let mut pass = SsgiPass::new(&device, 2, EXTENT).expect("pass");
        pass.resize(
            &device,
            PostExtent {
                width: 2560,
                height: 1440,
            },
        )
        .expect("resize");
        let targets = device.targets.borrow().clone();
        assert_eq!(targets.len(), 8);
        let resized = PostExtent {
            width: 1280,
            height: 720,
        };
        assert!(targets[4..].iter().all(|t| t.extent == resized));
        assert_eq!(encode_frame(&device, &pass).len(), DRAWS);
    }

    #[test]
    fn the_frame_counter_advances_the_ray_directions() {
        let device = MockDevice::new();
        let mut pass = SsgiPass::new(&device, 2, EXTENT).expect("pass");
        let before = pass.frame().frame;
        pass.advance();
        assert_eq!(pass.frame().frame, before.wrapping_add(1));
        assert_eq!(pass.frame().levels, DEPTH_LEVELS);
    }

    #[test]
    fn a_tiny_render_target_builds_only_the_levels_it_has() {
        let device = MockDevice::new();
        let pass = SsgiPass::new(
            &device,
            4,
            PostExtent {
                width: 8,
                height: 8,
            },
        )
        .expect("pass");
        assert_eq!(pass.frame().levels, 2);
        let draws = encode_frame(&device, &pass);
        let reduces = draws
            .iter()
            .filter(|d| d.program == PostProgram::SsgiReduce)
            .count();
        assert_eq!(reduces, 1);
    }
}
