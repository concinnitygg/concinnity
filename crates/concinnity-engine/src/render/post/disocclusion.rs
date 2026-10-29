//! The reactive mask's disocclusion term, written once for every backend.
//!
//! A temporal upscaler judges whether a pixel's history still holds by depth,
//! and where a moving object meets the ground the surface it uncovers sits at
//! the object's own depth, so the object's history smears over it. Ahead of the
//! main pass two draws close that gap:
//!
//! 1. the mark: each pixel followed back along its motion vector into last
//!    frame's snapshot, and given the full reactive value when none of the
//!    snapshot's texels around it faces or moves like it. It is the mask's
//!    first writer, so it also clears it;
//! 2. the snapshot: this frame's G-buffer view normal and motion vectors, kept
//!    for the next frame's mark.
//!
//! Without last frame's snapshot (the first frame, after a resize or a reset)
//! the snapshot is written first and the mark reads it, which marks nothing.

use alloc::format;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};

use super::device::{
    PostBind, PostBlend, PostDraw, PostExtent, PostLoadOp, PostPassDevice, PostSampler,
    PostTargetState, PostTiming,
};
use super::history::HistoryRing;
use super::program::PostProgram;
use crate::render::backend::error::{RenderError, RenderResult};
use crate::render::render_graph::{
    ClearValue, PassId, PixelFormat, TextureDesc, TextureSize, TextureUsage,
};

/// The reactive mask's format, which the mark draws into.
pub const MASK_FORMAT: PixelFormat = PixelFormat::R8Unorm;

const SNAPSHOT_LABEL: &str = "disocclusion_snapshot";

/// The shape of each snapshot slot: the G-buffer's octahedral view normal
/// (`rg`) and motion vector (`ba`) at the render resolution.
pub fn snapshot_desc() -> TextureDesc {
    TextureDesc {
        width: TextureSize::DrawableScaled(1.0),
        height: TextureSize::DrawableScaled(1.0),
        depth: 1,
        format: PixelFormat::Rgba16Float,
        sample_count: 1,
        array_layers: 1,
        mip_levels: 1,
        usage: TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
        clear: ClearValue::Color([0.0, 0.0, 0.0, 0.0]),
    }
}

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
        snapshot: device.create_pipeline(
            PostProgram::DisocclusionSnapshot,
            snapshot_desc().format,
            PostBlend::Replace,
        )?,
    })
}

/// The disocclusion pipelines, the snapshot ring, and which slot a frame
/// writes.
///
/// Parameterized by the two resource types rather than by the device, for the
/// same reason as the other temporal post passes: a backend's device value
/// borrows, and the pass is stored on its context.
pub struct DisocclusionPass<Pipeline, Target> {
    pipelines: DisocclusionPipelines<Pipeline>,
    snapshots: Vec<Target>,
    ring: HistoryRing,
    // Set when this frame's snapshot is encoded, so `advance` only keeps a slot
    // that was actually written.
    snapshotted: AtomicBool,
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
        let ring = HistoryRing::ping_pong();
        Ok(Self {
            pipelines: build_pipelines(device)?,
            snapshots: create_snapshots(device, extent, ring.slots())?,
            ring,
            snapshotted: AtomicBool::new(false),
        })
    }

    /// Step to the next frame: the snapshot this frame wrote becomes the next
    /// frame's. A frame that never encoded the pass wrote none, so it forgets
    /// the snapshot instead.
    pub fn advance(&mut self) {
        if self.snapshotted.swap(false, Ordering::Relaxed) {
            self.ring.advance();
        } else {
            self.ring.invalidate();
        }
    }

    /// Forget the snapshot, so the next frame marks nothing against a view it
    /// no longer matches.
    pub fn reset_history(&mut self) {
        self.ring.invalidate();
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
        // Released first, so a resize never holds two rings at once.
        self.snapshots.clear();
        self.ring.invalidate();
        self.snapshots = create_snapshots(device, extent, self.ring.slots())?;
        Ok(())
    }

    /// Swap in freshly built pipelines. Driven by shader hot reload; the caller
    /// has already idled the device.
    pub fn swap_pipelines(&mut self, pipelines: DisocclusionPipelines<Pipeline>) {
        self.pipelines = pipelines;
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
        let write = self.snapshot(self.ring.write())?;
        let snapshot_binds = [point(inputs.normal_depth), point(inputs.velocity)];
        let snapshot = |timing| PostDraw {
            target: device.target_attachment(write),
            state: PostTargetState::Pass,
            load: PostLoadOp::DontCare,
            timing,
            pipeline: &self.pipelines.snapshot,
            binds: &snapshot_binds,
            constants: &[],
            label: "Disocclusion snapshot",
        };
        let previous = if self.ring.valid() {
            self.snapshot(self.ring.history())?
        } else {
            device.encode(rec, &snapshot(PostTiming::First(PassId::Disocclusion)))?;
            write
        };
        let mark_binds = [
            point(inputs.normal_depth),
            point(inputs.velocity),
            point(device.target_ref(previous)),
        ];
        let mark_first = self.ring.valid();
        device.encode(
            rec,
            &PostDraw {
                target: inputs.mask,
                state: PostTargetState::Graph,
                load: PostLoadOp::DontCare,
                timing: if mark_first {
                    PostTiming::First(PassId::Disocclusion)
                } else {
                    PostTiming::Last(PassId::Disocclusion)
                },
                pipeline: &self.pipelines.mark,
                binds: &mark_binds,
                constants: &[],
                label: "Disocclusion mark",
            },
        )?;
        if mark_first {
            device.encode(rec, &snapshot(PostTiming::Last(PassId::Disocclusion)))?;
        }
        self.snapshotted.store(true, Ordering::Relaxed);
        Ok(())
    }

    // The snapshot in ring slot `slot`. A resize that failed part way leaves
    // the ring empty, which a frame reports rather than indexes.
    fn snapshot(&self, slot: usize) -> RenderResult<&Target> {
        self.snapshots.get(slot).ok_or_else(|| {
            RenderError::Other(format!("{SNAPSHOT_LABEL}: no target in ring slot {slot}"))
        })
    }
}

fn create_snapshots<D: PostPassDevice>(
    device: &D,
    extent: PostExtent,
    slots: usize,
) -> RenderResult<Vec<D::Target>> {
    let desc = snapshot_desc();
    (0..slots)
        .map(|_| device.create_target(SNAPSHOT_LABEL, &desc, extent))
        .collect()
}

// Every source is read by point load; the sampler only fills its slot.
fn point<'t, D: PostPassDevice + ?Sized + 't>(texture: D::TextureRef<'t>) -> PostBind<'t, D> {
    PostBind {
        texture,
        sampler: PostSampler::LinearClamp,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::post::mock::{MockDevice, MockDraw, MockPipeline, MockTexture};

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
            [
                PostProgram::DisocclusionMark,
                PostProgram::DisocclusionSnapshot
            ]
        );
        let (mark, snapshot) = (&draws[0], &draws[1]);
        assert_eq!(mark.target, MASK);
        assert_eq!(mark.state, PostTargetState::Graph);
        assert_eq!(mark.timing, PostTiming::First(PassId::Disocclusion));
        assert_eq!(
            binds(mark),
            [NORMAL_DEPTH, VELOCITY, MockTexture::Target(0)]
        );
        assert_eq!(snapshot.target, MockTexture::Target(1));
        assert_eq!(snapshot.state, PostTargetState::Pass);
        assert_eq!(snapshot.timing, PostTiming::Last(PassId::Disocclusion));
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
            [
                PostProgram::DisocclusionSnapshot,
                PostProgram::DisocclusionMark
            ]
        );
        assert_eq!(draws[0].target, MockTexture::Target(0));
        assert_eq!(draws[0].timing, PostTiming::First(PassId::Disocclusion));
        assert_eq!(binds(&draws[1])[2], MockTexture::Target(0));
        assert_eq!(draws[1].timing, PostTiming::Last(PassId::Disocclusion));
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
        assert_eq!(first_draw(&pass), PostProgram::DisocclusionSnapshot);

        pass.advance();
        pass.reset_history();
        assert_eq!(first_draw(&pass), PostProgram::DisocclusionSnapshot);

        pass.advance();
        pass.resize(&device, EXTENT).expect("resize");
        assert_eq!(first_draw(&pass), PostProgram::DisocclusionSnapshot);
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
