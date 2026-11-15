//! The reactive mask's disocclusion term for a temporal upscaler, written once
//! for every backend.
//!
//! An upscaler judges whether a pixel's history still holds by depth, and
//! where a moving object meets the ground the surface it uncovers sits at the
//! object's own depth, so the object's history smears over it. Ahead of the
//! main pass two draws close that gap:
//!
//! 1. the mark: each pixel followed back along its motion vector into last
//!    frame's motion snapshot, and given the full reactive value when none of
//!    the snapshot's texels around it faces or moves like it. It is the mask's
//!    first writer, so it also clears it;
//! 2. the snapshot: this frame's G-buffer view normal and motion vectors, kept
//!    for the next frame's mark.
//!
//! Without last frame's snapshot (the first frame, after a resize or a reset)
//! the snapshot is written first and the mark reads it, which marks nothing.
//! The engine's TAA resolve makes the same comparison itself and runs neither.

use super::device::{
    PostBlend, PostDraw, PostExtent, PostLoadOp, PostPassDevice, PostTargetState, PostTiming,
};
use super::motion_snapshot::{self, MotionSnapshot, SnapshotSources, point};
use super::program::PostProgram;
use crate::render::backend::error::RenderResult;
use crate::render::render_graph::{PassId, PixelFormat};

/// The reactive mask's format, which the mark draws into.
pub const MASK_FORMAT: PixelFormat = PixelFormat::R8Unorm;

/// What the pass reads and writes beyond its own snapshots.
pub struct DisocclusionInputs<'t, D: PostPassDevice + ?Sized + 't> {
    /// This frame's G-buffer view normal (`.rgb`) and linear depth (`.a`).
    pub normal_depth: D::TextureRef<'t>,
    /// This frame's G-buffer motion vectors.
    pub velocity: D::TextureRef<'t>,
    /// The reactive mask, which the render graph has put in its render state.
    pub mask: D::Attachment<'t>,
}

/// The two pipelines, built together. What shader hot reload rebuilds and
/// hands to [`DisocclusionPass::swap_pipelines`].
pub struct DisocclusionPipelines<Pipeline> {
    /// The reactive mask's disocclusion value.
    pub mark: Pipeline,
    /// The G-buffer copy the next frame reads.
    pub snapshot: Pipeline,
}

/// Whether a backend keeps the pass: the G-buffer the mark reads is built,
/// and an upscaler that takes the mask resolves the frame over time
/// (`upscaler_reads_mask` holds whether the built one, if any, takes it).
pub fn wanted(gbuffer: bool, upscaler_reads_mask: Option<bool>) -> bool {
    gbuffer && upscaler_reads_mask == Some(true)
}

/// Build both pipelines on their own, without touching the snapshots.
///
/// # Errors
///
/// Returns an error if a pipeline fails to build.
pub fn build_pipelines<D: PostPassDevice>(
    device: &D,
) -> RenderResult<DisocclusionPipelines<D::Pipeline>> {
    Ok(DisocclusionPipelines {
        mark: device.create_pipeline(
            PostProgram::DisocclusionMark,
            MASK_FORMAT,
            PostBlend::Replace,
        )?,
        snapshot: motion_snapshot::build_pipeline(device)?,
    })
}

/// The mark's pipeline and the motion snapshot it compares against.
///
/// Parameterized by the two resource types rather than by the device, for the
/// same reason as the other temporal post passes: a backend's device value
/// borrows, and the pass is stored on its context.
pub struct DisocclusionPass<Pipeline, Target> {
    mark: Pipeline,
    snapshot: MotionSnapshot<Pipeline, Target>,
}

impl<Pipeline, Target> DisocclusionPass<Pipeline, Target> {
    /// Build both pipelines and the snapshot ring for a render resolution of
    /// `extent`.
    ///
    /// # Errors
    ///
    /// Returns an error if a pipeline or target cannot be created.
    pub fn new<D>(device: &D, extent: PostExtent) -> RenderResult<Self>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target>,
    {
        Ok(Self {
            mark: device.create_pipeline(
                PostProgram::DisocclusionMark,
                MASK_FORMAT,
                PostBlend::Replace,
            )?,
            snapshot: MotionSnapshot::new(device, extent)?,
        })
    }

    /// Step to the next frame: the snapshot this frame wrote becomes the next
    /// frame's. A frame that never encoded the pass wrote none, so it forgets
    /// the snapshot instead.
    pub fn advance(&mut self) {
        self.snapshot.advance();
    }

    /// Forget the snapshot, so the next frame marks nothing against a view it
    /// no longer matches.
    pub fn reset_history(&mut self) {
        self.snapshot.reset_history();
    }

    /// Recreate the snapshots for a new render resolution and forget the old
    /// one. The caller has already idled the device.
    ///
    /// # Errors
    ///
    /// Returns an error if a target cannot be created.
    pub fn resize<D>(&mut self, device: &D, extent: PostExtent) -> RenderResult<()>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target>,
    {
        self.snapshot.resize(device, extent)
    }

    /// Swap in freshly built pipelines. Driven by shader hot reload; the caller
    /// has already idled the device.
    pub fn swap_pipelines(&mut self, pipelines: DisocclusionPipelines<Pipeline>) {
        self.mark = pipelines.mark;
        self.snapshot.swap_pipeline(pipelines.snapshot);
    }

    /// Encode the mark into `inputs.mask` against last frame's snapshot, then
    /// this frame's; without last frame's, this frame's first.
    ///
    /// # Errors
    ///
    /// Returns an error if a snapshot slot cannot be addressed or the device
    /// fails to encode a draw.
    pub fn encode<'t, D>(
        &'t self,
        device: &D,
        rec: &D::Recorder,
        inputs: &DisocclusionInputs<'t, D>,
    ) -> RenderResult<()>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target> + 't,
    {
        let timing = PassId::Disocclusion.timing();
        let sources = SnapshotSources {
            normal_depth: inputs.normal_depth,
            velocity: inputs.velocity,
        };
        let last = self.snapshot.previous()?;
        let mark_first = last.is_some();
        let previous = if let Some(previous) = last {
            previous
        } else {
            self.snapshot
                .encode(device, rec, &sources, PostTiming::First(timing))?;
            self.snapshot.current()?
        };
        let mark_binds = [
            point(inputs.normal_depth),
            point(inputs.velocity),
            point(device.target_ref(previous)),
        ];
        device.encode(
            rec,
            &PostDraw {
                target: inputs.mask,
                state: PostTargetState::Graph,
                load: PostLoadOp::DontCare,
                timing: if mark_first {
                    PostTiming::First(timing)
                } else {
                    PostTiming::Last(timing)
                },
                pipeline: &self.mark,
                binds: &mark_binds,
                constants: &[],
                label: "Disocclusion mark",
            },
        )?;
        if mark_first {
            self.snapshot
                .encode(device, rec, &sources, PostTiming::Last(timing))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::post::mock::{MockDevice, MockDraw, MockPipeline, MockTexture};
    use alloc::vec::Vec;

    const EXTENT: PostExtent = PostExtent {
        width: 1280,
        height: 720,
    };
    const NORMAL_DEPTH: MockTexture = MockTexture::External(1);
    const VELOCITY: MockTexture = MockTexture::External(2);
    const MASK: MockTexture = MockTexture::External(3);

    fn encode_frame(
        device: &MockDevice,
        pass: &DisocclusionPass<MockPipeline, usize>,
    ) -> Vec<MockDraw> {
        device.draws.borrow_mut().clear();
        pass.encode(
            device,
            &(),
            &DisocclusionInputs {
                normal_depth: NORMAL_DEPTH,
                velocity: VELOCITY,
                mask: MASK,
            },
        )
        .expect("encode");
        device.draws.borrow().clone()
    }

    fn binds(draw: &MockDraw) -> Vec<MockTexture> {
        draw.binds.iter().map(|(t, _)| *t).collect()
    }

    fn programs(draws: &[MockDraw]) -> Vec<PostProgram> {
        draws.iter().map(|d| d.program).collect()
    }

    // Only an upscaler that takes the mask reads the mark, and nothing runs
    // without the G-buffer: TAA compares against the snapshot itself.
    #[test]
    fn the_pass_follows_an_upscaler_that_reads_the_mask() {
        assert!(wanted(true, Some(true)));
        assert!(!wanted(true, None));
        assert!(!wanted(true, Some(false)));
        assert!(!wanted(false, Some(true)));
    }

    // With last frame's snapshot the mark reads it, then this frame's is kept
    // in the other slot.
    #[test]
    fn the_mark_reads_last_frame_s_snapshot_then_the_frame_keeps_its_own() {
        let device = MockDevice::new();
        let mut pass = DisocclusionPass::new(&device, EXTENT).expect("pass");
        encode_frame(&device, &pass);
        pass.advance();
        let draws = encode_frame(&device, &pass);
        assert_eq!(
            programs(&draws),
            [PostProgram::DisocclusionMark, PostProgram::MotionSnapshot]
        );
        let (mark, snapshot) = (&draws[0], &draws[1]);
        assert_eq!(mark.target, MASK);
        assert_eq!(mark.state, PostTargetState::Graph);
        assert_eq!(
            mark.timing,
            PostTiming::First(PassId::Disocclusion.timing())
        );
        assert_eq!(
            binds(mark),
            [NORMAL_DEPTH, VELOCITY, MockTexture::Target(0)]
        );
        assert_eq!(snapshot.target, MockTexture::Target(1));
        assert_eq!(snapshot.state, PostTargetState::Pass);
        assert_eq!(
            snapshot.timing,
            PostTiming::Last(PassId::Disocclusion.timing())
        );
        assert_eq!(binds(snapshot), [NORMAL_DEPTH, VELOCITY]);
        for d in &draws {
            assert_eq!(d.load, PostLoadOp::DontCare, "{}", d.label);
        }
    }

    // Without one, the frame's own snapshot comes first and the mark compares
    // the frame against itself.
    #[test]
    fn without_last_frame_s_snapshot_the_mark_reads_this_frame_s() {
        let device = MockDevice::new();
        let pass = DisocclusionPass::new(&device, EXTENT).expect("pass");
        let draws = encode_frame(&device, &pass);
        assert_eq!(
            programs(&draws),
            [PostProgram::MotionSnapshot, PostProgram::DisocclusionMark]
        );
        assert_eq!(draws[0].target, MockTexture::Target(0));
        assert_eq!(
            draws[0].timing,
            PostTiming::First(PassId::Disocclusion.timing())
        );
        assert_eq!(binds(&draws[1])[2], MockTexture::Target(0));
        assert_eq!(
            draws[1].timing,
            PostTiming::Last(PassId::Disocclusion.timing())
        );
    }

    #[test]
    fn a_frame_without_the_pass_a_reset_or_a_resize_forgets_the_snapshot() {
        let device = MockDevice::new();
        let mut pass = DisocclusionPass::new(&device, EXTENT).expect("pass");
        let first_draw =
            |pass: &DisocclusionPass<MockPipeline, usize>| encode_frame(&device, pass)[0].program;
        encode_frame(&device, &pass);
        pass.advance();
        assert_eq!(first_draw(&pass), PostProgram::DisocclusionMark);

        pass.advance();
        pass.advance();
        assert_eq!(first_draw(&pass), PostProgram::MotionSnapshot);

        pass.advance();
        pass.reset_history();
        assert_eq!(first_draw(&pass), PostProgram::MotionSnapshot);

        pass.advance();
        pass.resize(&device, EXTENT).expect("resize");
        assert_eq!(first_draw(&pass), PostProgram::MotionSnapshot);
    }

    #[test]
    fn the_pipelines_write_the_mask_and_snapshot_formats() {
        let device = MockDevice::new();
        let p = build_pipelines(&device).expect("pipelines");
        assert_eq!(p.mark.format, MASK_FORMAT);
        assert_eq!(p.snapshot.format, PixelFormat::Rgba16Float);
        assert_eq!(p.mark.blend, PostBlend::Replace);
        assert_eq!(p.snapshot.blend, PostBlend::Replace);
    }

    #[test]
    fn a_failed_resize_reports_rather_than_indexes() {
        let device = MockDevice::new();
        let mut pass = DisocclusionPass::new(&device, EXTENT).expect("pass");
        device.fail_creates_after(0);
        assert!(pass.resize(&device, EXTENT).is_err());
        assert!(
            pass.encode(
                &device,
                &(),
                &DisocclusionInputs {
                    normal_depth: NORMAL_DEPTH,
                    velocity: VELOCITY,
                    mask: MASK,
                },
            )
            .is_err()
        );
        assert_eq!(device.targets.borrow().len(), 2);
    }
}
