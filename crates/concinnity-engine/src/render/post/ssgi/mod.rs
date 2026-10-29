//! Screen-space global illumination, written once for every backend.
//!
//! After the main pass, five stages at the trace resolution (the render
//! resolution divided by `gi_scale`):
//!
//! 1. the closest and farthest linear depth of each trace pixel's footprint in
//!    the G-buffer, the top of a depth pyramid;
//! 2. the pyramid's remaining levels, each the depth range of the level below;
//! 3. the age: how many frames of history each pixel's accumulations hold,
//!    last frame's reprojected by the G-buffer's motion vectors and rejected
//!    where last frame's pyramid and normals saw another surface, beside the
//!    normal this frame sees;
//! 4. the trace: cosine-weighted hemisphere rays walked through the pyramid,
//!    the lit scene color each one hits and the share of rays that hit,
//!    blended into last frame's fast accumulation (a few frames long);
//! 5. the accumulation: last frame's long accumulation, reprojected the same
//!    way and clamped to the neighborhood of this frame's fast one, blended
//!    with it. The fast one follows a moving light source within a few frames,
//!    so the long one cannot keep a bounce that has moved away, while a still
//!    scene keeps the long one's smoothness.
//!
//! Then, ahead of the next frame's main pass, the resolve: that long
//! accumulation reprojected onto the new G-buffer through a depth-aware
//! upsample and denoise, at full resolution. It holds the environment's
//! visibility (the ambient occlusion, lowered where the rays hit) and the
//! bounce, which the main pass takes as indirect diffuse light, so the
//! receiver's albedo and Fresnel weight it like the ambient term. The scene the
//! trace reads then already holds last frame's bounce, which is how light
//! bounces more than once.
//!
//! The pyramids, ages and both accumulations are each a ping-pong pair, so
//! every stage reads what the previous frame wrote.

use alloc::format;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};

use crate::gfx::render_types::SsgiParams;
use crate::render::backend::error::{RenderError, RenderResult};
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
use targets::{age_desc, depth_desc, gi_desc, indirect_desc, pyramid_levels};

/// What the trace reads beyond the pass's own targets.
pub struct SsgiTraceInputs<'t, D: PostPassDevice + ?Sized + 't> {
    /// The lit scene the rays sample their bounce radiance from.
    pub scene: D::TextureRef<'t>,
    /// The G-buffer's view normal (`.rgb`) and linear depth (`.a`).
    pub normal_depth: D::TextureRef<'t>,
    /// The G-buffer's per-pixel motion vectors.
    pub velocity: D::TextureRef<'t>,
    /// The G-buffer's roughness (`.r`) and head-on specular reflectance
    /// (`.gba`), which tell how much of a hit's color is mirrored light.
    pub reflectance: D::TextureRef<'t>,
}

/// What the resolve reads beyond the pass's own targets.
pub struct SsgiResolveInputs<'t, D: PostPassDevice + ?Sized + 't> {
    /// This frame's G-buffer view normal (`.rgb`) and linear depth (`.a`).
    pub normal_depth: D::TextureRef<'t>,
    /// This frame's G-buffer motion vectors.
    pub velocity: D::TextureRef<'t>,
    /// This frame's ambient occlusion in `.r`, or a 1x1 white texture
    /// without one.
    pub occlusion: D::TextureRef<'t>,
}

/// The six pipelines, built together. What shader hot reload rebuilds and
/// hands to [`SsgiPass::swap_pipelines`].
pub struct SsgiPipelines<Pipeline> {
    /// The pyramid's top level from the G-buffer.
    pub depth: Pipeline,
    /// One more pyramid level from the one below.
    pub reduce: Pipeline,
    /// The accumulations' length in frames.
    pub age: Pipeline,
    /// The hemisphere trace and its fast accumulation.
    pub trace: Pipeline,
    /// The long accumulation, bounded by the fast one.
    pub accumulate: Pipeline,
    /// The full-resolution visibility and bounce the main pass samples.
    pub resolve: Pipeline,
}

/// Build every pipeline on its own, without touching the targets.
///
/// # Errors
///
/// Returns an error if a pipeline fails to build.
pub fn build_pipelines<D: PostPassDevice>(device: &D) -> RenderResult<SsgiPipelines<D::Pipeline>> {
    let depth = PixelFormat::Rg32Float;
    let color = PixelFormat::Rgba16Float;
    let replace = PostBlend::Replace;
    Ok(SsgiPipelines {
        depth: device.create_pipeline(PostProgram::SsgiDepth, depth, replace)?,
        reduce: device.create_pipeline(PostProgram::SsgiReduce, depth, replace)?,
        age: device.create_pipeline(PostProgram::SsgiAge, color, replace)?,
        trace: device.create_pipeline(PostProgram::SsgiTrace, color, replace)?,
        accumulate: device.create_pipeline(PostProgram::SsgiAccumulate, color, replace)?,
        resolve: device.create_pipeline(PostProgram::SsgiResolve, color, replace)?,
    })
}

// The graph labels the targets carry into a backend's own debug naming.
const DEPTH_LABEL: &str = "ssgi_depth";
const HISTORY_LABEL: &str = "ssgi_history";
const FAST_LABEL: &str = "ssgi_fast";
const AGE_LABEL: &str = "ssgi_age";

/// The graph label of the resolve's output, which the resolve writes and the
/// main pass reads.
pub const INDIRECT_LABEL: &str = "ssgi_indirect";

// The targets sized to one render resolution: a pyramid, an age and both
// accumulations per ring slot, and the one resolve output.
struct SsgiTargets<Target> {
    depth: Vec<Target>,
    history: Vec<Target>,
    fast: Vec<Target>,
    age: Vec<Target>,
    indirect: Target,
    levels: u32,
}

impl<Target> SsgiTargets<Target> {
    fn new<D>(device: &D, gi_scale: u32, extent: PostExtent, slots: usize) -> RenderResult<Self>
    where
        D: PostPassDevice<Target = Target>,
    {
        let depth = depth_desc(gi_scale, extent);
        let gi = gi_desc(gi_scale);
        let age = age_desc(gi_scale);
        let mut targets = Self {
            depth: Vec::with_capacity(slots),
            history: Vec::with_capacity(slots),
            fast: Vec::with_capacity(slots),
            age: Vec::with_capacity(slots),
            indirect: device.create_target(INDIRECT_LABEL, &indirect_desc(), extent)?,
            levels: pyramid_levels(resolve_extent(&gi, extent)),
        };
        for _ in 0..slots {
            targets
                .depth
                .push(device.create_target(DEPTH_LABEL, &depth, extent)?);
            targets
                .history
                .push(device.create_target(HISTORY_LABEL, &gi, extent)?);
            targets
                .fast
                .push(device.create_target(FAST_LABEL, &gi, extent)?);
            targets
                .age
                .push(device.create_target(AGE_LABEL, &age, extent)?);
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

    // The fast accumulation in ring slot `slot`.
    fn fast(&self, slot: usize) -> RenderResult<&Target> {
        ring_slot(&self.fast, slot, FAST_LABEL)
    }

    // The age in ring slot `slot`.
    fn age(&self, slot: usize) -> RenderResult<&Target> {
        ring_slot(&self.age, slot, AGE_LABEL)
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
/// which pyramid, age and accumulation slot a frame writes.
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
    ///
    /// # Errors
    ///
    /// Returns an error if a pipeline or target cannot be created.
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

    /// The resolve's output, which the main pass samples for the environment's
    /// visibility and the bounce.
    pub fn indirect(&self) -> &Target {
        &self.targets.indirect
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

    /// Forget the accumulation, so the next resolve and trace start over from
    /// the current frame instead of reprojecting a view it no longer matches.
    pub fn reset_history(&mut self) {
        self.ring.invalidate();
    }

    /// Recreate every target for a new render resolution and forget the
    /// accumulation, which was rendered at a resolution this one cannot
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
        // Released before the new set is created, so a resize does not hold two
        // sets of rings (and their descriptors) at once.
        self.targets.depth.clear();
        self.targets.history.clear();
        self.targets.fast.clear();
        self.targets.age.clear();
        self.ring.invalidate();
        self.targets = SsgiTargets::new(device, self.gi_scale, extent, slots)?;
        Ok(())
    }

    /// Swap in freshly built pipelines. Driven by shader hot reload; the caller
    /// has already idled the device.
    pub fn swap_pipelines(&mut self, pipelines: SsgiPipelines<Pipeline>) {
        self.pipelines = pipelines;
    }

    /// Encode the resolve ahead of the main pass: last frame's long
    /// accumulation reprojected onto this frame's G-buffer, with `occlusion`,
    /// into the target [`Self::indirect`] names. Without a history to reproject
    /// it holds the occlusion and no bounce.
    ///
    /// # Errors
    ///
    /// Returns an error if the device fails to encode a draw.
    pub fn encode_resolve<'t, D>(
        &'t self,
        device: &D,
        rec: &D::Recorder,
        inputs: &SsgiResolveInputs<'t, D>,
        params: &SsgiParams,
    ) -> RenderResult<()>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target> + 't,
    {
        let read = self.ring.history();
        let targets = &self.targets;
        device.encode(
            rec,
            &PostDraw {
                target: device.target_attachment(&targets.indirect),
                // The graph declares the output, so the main pass's read is
                // ordered after this write.
                state: PostTargetState::Graph,
                load: PostLoadOp::DontCare,
                timing: PostTiming::Whole(PassId::SsgiResolve),
                pipeline: &self.pipelines.resolve,
                binds: &[
                    screen::<D>(device.target_ref(targets.history(read)?)),
                    screen::<D>(device.target_ref(targets.depth(read)?)),
                    screen::<D>(device.target_ref(targets.age(read)?)),
                    screen::<D>(inputs.normal_depth),
                    screen::<D>(inputs.velocity),
                    screen::<D>(inputs.occlusion),
                ],
                constants: bytemuck::bytes_of(params),
                label: "SSGI resolve",
            },
        )
    }

    /// Encode every stage after the main pass, in order: this frame's depth
    /// pyramid from `normal_depth`, the age, then the hemisphere rays walked
    /// through the pyramid, sampling the scene where they hit, accumulated into
    /// this frame's fast slot, which then bounds and feeds the long
    /// accumulation in this frame's slot.
    ///
    /// # Errors
    ///
    /// Returns an error if a ring slot cannot be addressed or the device fails
    /// to encode a draw.
    pub fn encode<'t, D>(
        &'t self,
        device: &D,
        rec: &D::Recorder,
        inputs: &SsgiTraceInputs<'t, D>,
        params: &SsgiParams,
    ) -> RenderResult<()>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target> + 't,
    {
        let (write, read) = (self.ring.write(), self.ring.history());
        let targets = &self.targets;
        let pyramid = targets.depth(write)?;
        device.encode(
            rec,
            &private_draw(
                device.target_level_attachment(pyramid, 0)?,
                PostTiming::First(PassId::Ssgi),
                &self.pipelines.depth,
                &[screen::<D>(inputs.normal_depth)],
                bytemuck::bytes_of(params),
                "SSGI depth",
            ),
        )?;
        for level in 1..targets.levels {
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

        let depth = device.target_ref(pyramid);
        let prev_depth = device.target_ref(targets.depth(read)?);
        let age = device.target_ref(targets.age(write)?);
        let prev_age = device.target_ref(targets.age(read)?);
        device.encode(
            rec,
            &private_draw(
                device.target_attachment(targets.age(write)?),
                PostTiming::None,
                &self.pipelines.age,
                &[
                    screen::<D>(inputs.normal_depth),
                    screen::<D>(inputs.velocity),
                    screen::<D>(depth),
                    screen::<D>(prev_depth),
                    screen::<D>(prev_age),
                ],
                bytemuck::bytes_of(params),
                "SSGI age",
            ),
        )?;
        device.encode(
            rec,
            &private_draw(
                device.target_attachment(targets.fast(write)?),
                PostTiming::None,
                &self.pipelines.trace,
                &[
                    screen::<D>(inputs.scene),
                    screen::<D>(inputs.normal_depth),
                    screen::<D>(inputs.velocity),
                    screen::<D>(depth),
                    screen::<D>(prev_depth),
                    screen::<D>(device.target_ref(targets.fast(read)?)),
                    screen::<D>(age),
                    screen::<D>(prev_age),
                    screen::<D>(inputs.reflectance),
                ],
                bytemuck::bytes_of(params),
                "SSGI trace",
            ),
        )?;
        device.encode(
            rec,
            &private_draw(
                device.target_attachment(targets.history(write)?),
                PostTiming::Last(PassId::Ssgi),
                &self.pipelines.accumulate,
                &[
                    screen::<D>(device.target_ref(targets.fast(write)?)),
                    screen::<D>(inputs.normal_depth),
                    screen::<D>(inputs.velocity),
                    screen::<D>(depth),
                    screen::<D>(prev_depth),
                    screen::<D>(device.target_ref(targets.history(read)?)),
                    screen::<D>(age),
                    screen::<D>(prev_age),
                ],
                bytemuck::bytes_of(params),
                "SSGI accumulate",
            ),
        )?;
        self.traced.store(true, Ordering::Relaxed);
        Ok(())
    }
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

    // Target indices in the order `SsgiTargets::new` creates them: the resolve
    // output, then each slot's pyramid, accumulations and age.
    const INDIRECT: usize = 0;
    const fn depth(slot: usize) -> usize {
        1 + 4 * slot
    }
    const fn history(slot: usize) -> usize {
        2 + 4 * slot
    }
    const fn fast(slot: usize) -> usize {
        3 + 4 * slot
    }
    const fn age(slot: usize) -> usize {
        4 + 4 * slot
    }

    const SCENE: MockTexture = MockTexture::External(1);
    const NORMAL_DEPTH: MockTexture = MockTexture::External(2);
    const VELOCITY: MockTexture = MockTexture::External(3);
    const OCCLUSION: MockTexture = MockTexture::External(4);
    const REFLECTANCE: MockTexture = MockTexture::External(5);

    // Draws after the main pass: the pyramid's levels, the age, the trace and
    // the accumulation.
    const DRAWS: usize = DEPTH_LEVELS as usize + 3;

    fn params(pass: &SsgiPass<MockPipeline, usize>) -> SsgiParams {
        SsgiSettings::resolve(1.0, 8.0, 1, 2).params(1.0, 1.5, pass.frame(), 1.0)
    }

    fn encode_frame(device: &MockDevice, pass: &SsgiPass<MockPipeline, usize>) -> Vec<MockDraw> {
        device.draws.borrow_mut().clear();
        pass.encode_resolve(
            device,
            &(),
            &SsgiResolveInputs {
                normal_depth: NORMAL_DEPTH,
                velocity: VELOCITY,
                occlusion: OCCLUSION,
            },
            &params(pass),
        )
        .expect("resolve");
        pass.encode(
            device,
            &(),
            &SsgiTraceInputs {
                scene: SCENE,
                normal_depth: NORMAL_DEPTH,
                velocity: VELOCITY,
                reflectance: REFLECTANCE,
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
    fn the_pass_creates_the_resolve_output_and_a_set_of_targets_per_slot() {
        let device = MockDevice::new();
        let pass = SsgiPass::new(&device, 2, EXTENT).expect("pass");
        let targets = device.targets.borrow();
        assert_eq!(targets.len(), 9);
        assert_eq!(targets[INDIRECT].label, INDIRECT_LABEL);
        assert_eq!(targets[INDIRECT].extent, EXTENT);
        assert_eq!(*pass.indirect(), INDIRECT);
        let half = PostExtent {
            width: 640,
            height: 360,
        };
        for slot in 0..2 {
            assert_eq!(targets[depth(slot)].label, DEPTH_LABEL);
            assert_eq!(targets[depth(slot)].levels, DEPTH_LEVELS);
            assert_eq!(targets[history(slot)].label, HISTORY_LABEL);
            assert_eq!(targets[fast(slot)].label, FAST_LABEL);
            assert_eq!(targets[age(slot)].label, AGE_LABEL);
            for t in [depth(slot), history(slot), fast(slot), age(slot)] {
                assert_eq!(targets[t].extent, half, "{}", targets[t].label);
            }
        }
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
    fn the_resolve_reads_last_frames_slot_into_the_graph_declared_output() {
        let device = MockDevice::new();
        let pass = SsgiPass::new(&device, 2, EXTENT).expect("pass");
        let draws = encode_frame(&device, &pass);
        let resolve = &draws[0];
        assert_eq!(resolve.program, PostProgram::SsgiResolve);
        assert_eq!(resolve.target, MockTexture::Target(INDIRECT));
        assert_eq!(resolve.state, PostTargetState::Graph);
        assert_eq!(resolve.timing, PostTiming::Whole(PassId::SsgiResolve));
        assert_eq!(
            binds(resolve),
            [
                MockTexture::Target(history(1)),
                MockTexture::Target(depth(1)),
                MockTexture::Target(age(1)),
                NORMAL_DEPTH,
                VELOCITY,
                OCCLUSION,
            ]
        );
    }

    #[test]
    fn a_frame_builds_the_pyramid_then_ages_traces_and_accumulates() {
        let device = MockDevice::new();
        let pass = SsgiPass::new(&device, 2, EXTENT).expect("pass");
        let draws = encode_frame(&device, &pass);
        let after_main = &draws[1..];
        assert_eq!(after_main.len(), DRAWS);
        let programs: Vec<PostProgram> = after_main.iter().map(|d| d.program).collect();
        let mut expected = vec![PostProgram::SsgiDepth];
        expected.extend((1..DEPTH_LEVELS).map(|_| PostProgram::SsgiReduce));
        expected.extend([
            PostProgram::SsgiAge,
            PostProgram::SsgiTrace,
            PostProgram::SsgiAccumulate,
        ]);
        assert_eq!(programs, expected);

        // The pyramid: level 0 from the G-buffer, each level after it from the
        // one below, all in this frame's slot.
        let d = depth(0);
        assert_eq!(after_main[0].target, MockTexture::Level(d, 0));
        assert_eq!(binds(&after_main[0]), [NORMAL_DEPTH]);
        for level in 1..DEPTH_LEVELS {
            let reduce = &after_main[level as usize];
            assert_eq!(reduce.target, MockTexture::Level(d, level));
            assert_eq!(binds(reduce), [MockTexture::Level(d, level - 1)]);
            assert!(reduce.constants.is_empty());
        }

        let n = DEPTH_LEVELS as usize;
        let (aging, trace, accumulate) = (&after_main[n], &after_main[n + 1], &after_main[n + 2]);
        assert_eq!(aging.target, MockTexture::Target(age(0)));
        assert_eq!(
            binds(aging),
            [
                NORMAL_DEPTH,
                VELOCITY,
                MockTexture::Target(d),
                MockTexture::Target(depth(1)),
                MockTexture::Target(age(1)),
            ]
        );
        assert_eq!(trace.target, MockTexture::Target(fast(0)));
        assert_eq!(
            binds(trace),
            [
                SCENE,
                NORMAL_DEPTH,
                VELOCITY,
                MockTexture::Target(d),
                MockTexture::Target(depth(1)),
                MockTexture::Target(fast(1)),
                MockTexture::Target(age(0)),
                MockTexture::Target(age(1)),
                REFLECTANCE,
            ]
        );
        // The long accumulation reads the fast one this frame's trace wrote.
        assert_eq!(accumulate.target, MockTexture::Target(history(0)));
        assert_eq!(
            binds(accumulate),
            [
                MockTexture::Target(fast(0)),
                NORMAL_DEPTH,
                VELOCITY,
                MockTexture::Target(d),
                MockTexture::Target(depth(1)),
                MockTexture::Target(history(1)),
                MockTexture::Target(age(0)),
                MockTexture::Target(age(1)),
            ]
        );
    }

    #[test]
    fn only_the_resolve_touches_the_graph_and_nothing_blends() {
        let device = MockDevice::new();
        let pass = SsgiPass::new(&device, 2, EXTENT).expect("pass");
        let draws = encode_frame(&device, &pass);
        let (resolve, private) = draws.split_first().expect("draws");
        assert_eq!(resolve.state, PostTargetState::Graph);
        for d in private {
            assert_eq!(d.state, PostTargetState::Pass, "{}", d.label);
        }
        for d in &draws {
            assert_eq!(d.load, PostLoadOp::DontCare, "{}", d.label);
        }
        let p = build_pipelines(&device).expect("pipelines");
        for pipeline in [
            &p.depth,
            &p.reduce,
            &p.age,
            &p.trace,
            &p.accumulate,
            &p.resolve,
        ] {
            assert_eq!(pipeline.blend, PostBlend::Replace);
        }
        assert_eq!(p.depth.format, PixelFormat::Rg32Float);
        assert_eq!(p.reduce.format, PixelFormat::Rg32Float);
        assert_eq!(p.age.format, PixelFormat::Rgba16Float);
        assert_eq!(p.trace.format, PixelFormat::Rgba16Float);
        assert_eq!(p.accumulate.format, PixelFormat::Rgba16Float);
        assert_eq!(p.resolve.format, PixelFormat::Rgba16Float);
    }

    #[test]
    fn the_timing_span_covers_the_stages_after_the_main_pass() {
        let device = MockDevice::new();
        let pass = SsgiPass::new(&device, 2, EXTENT).expect("pass");
        let draws = encode_frame(&device, &pass);
        let timings: Vec<PostTiming> = draws[1..].iter().map(|d| d.timing).collect();
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
        let n = DEPTH_LEVELS as usize + 1;
        let first = encode_frame(&device, &pass);
        pass.advance();
        let second = encode_frame(&device, &pass);
        assert_eq!(first[1].target, MockTexture::Level(depth(0), 0));
        assert_eq!(second[1].target, MockTexture::Level(depth(1), 0));
        assert_eq!(second[n].target, MockTexture::Target(age(1)));
        assert_eq!(second[n + 1].target, MockTexture::Target(fast(1)));
        assert_eq!(second[n + 2].target, MockTexture::Target(history(1)));
        // Last frame's pyramid, age and accumulations are this frame's history,
        // and what the resolve reprojects.
        assert_eq!(
            &binds(&second[0])[..3],
            [
                MockTexture::Target(history(0)),
                MockTexture::Target(depth(0)),
                MockTexture::Target(age(0)),
            ]
        );
        assert_eq!(binds(&second[n])[4], MockTexture::Target(age(0)));
        let trace = binds(&second[n + 1]);
        assert_eq!(trace[4], MockTexture::Target(depth(0)));
        assert_eq!(trace[5], MockTexture::Target(fast(0)));
        let accumulate = binds(&second[n + 2]);
        assert_eq!(accumulate[4], MockTexture::Target(depth(0)));
        assert_eq!(accumulate[5], MockTexture::Target(history(0)));
        assert_eq!(accumulate[7], MockTexture::Target(age(0)));
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
            encode_frame(&device, &pass)[1].target,
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
        assert_eq!(targets.len(), 18);
        let full = PostExtent {
            width: 2560,
            height: 1440,
        };
        let resized = PostExtent {
            width: 1280,
            height: 720,
        };
        assert_eq!(targets[9].extent, full);
        assert!(targets[10..].iter().all(|t| t.extent == resized));
        assert_eq!(encode_frame(&device, &pass).len(), DRAWS + 1);
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
