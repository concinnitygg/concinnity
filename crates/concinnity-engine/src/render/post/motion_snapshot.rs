//! Each frame's G-buffer view normal and motion vectors, kept for the next
//! frame, written once for every backend.
//!
//! A temporal pass that reprojects a pixel into last frame's output cannot
//! tell from depth alone whether that output showed the same surface: where a
//! moving object meets the ground the two depths agree. The snapshot keeps
//! what told them apart last frame, each pixel's view normal and motion, so
//! this frame's pass can compare against it. The TAA resolve reads it
//! directly; for an upscaler the disocclusion mark turns it into the reactive
//! mask.

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
    ClearValue, PixelFormat, TextureDesc, TextureSize, TextureUsage,
};

const LABEL: &str = "motion_snapshot";

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

/// Build the snapshot pipeline on its own, without touching the ring.
///
/// # Errors
///
/// Returns an error if the pipeline fails to build.
pub fn build_pipeline<D: PostPassDevice>(device: &D) -> RenderResult<D::Pipeline> {
    device.create_pipeline(
        PostProgram::MotionSnapshot,
        snapshot_desc().format,
        PostBlend::Replace,
    )
}

/// What the snapshot copies: this frame's G-buffer.
pub struct SnapshotSources<'t, D: PostPassDevice + ?Sized + 't> {
    /// The G-buffer view normal (`.rgb`) and linear depth (`.a`).
    pub normal_depth: D::TextureRef<'t>,
    /// The G-buffer motion vectors.
    pub velocity: D::TextureRef<'t>,
}

/// The snapshot pipeline, its ring of targets, and which slot a frame writes.
///
/// Parameterized by the two resource types rather than by the device, for the
/// same reason as the temporal post passes that own one: a backend's device
/// value borrows, and the pass is stored on its context.
pub struct MotionSnapshot<Pipeline, Target> {
    pipeline: Pipeline,
    targets: Vec<Target>,
    ring: HistoryRing,
    // Set when this frame's snapshot is encoded, so `advance` only keeps a slot
    // that was actually written.
    written: AtomicBool,
}

impl<Pipeline, Target> MotionSnapshot<Pipeline, Target> {
    /// Build the pipeline and the ring for a render resolution of `extent`.
    ///
    /// # Errors
    ///
    /// Returns an error if the pipeline or a target cannot be created.
    pub fn new<D>(device: &D, extent: PostExtent) -> RenderResult<Self>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target>,
    {
        let ring = HistoryRing::ping_pong();
        Ok(Self {
            pipeline: build_pipeline(device)?,
            targets: create_targets(device, extent, ring.slots())?,
            ring,
            written: AtomicBool::new(false),
        })
    }

    /// Step to the next frame: the snapshot this frame wrote becomes the next
    /// frame's. A frame that never encoded one forgets the last instead.
    pub fn advance(&mut self) {
        if self.written.swap(false, Ordering::Relaxed) {
            self.ring.advance();
        } else {
            self.ring.invalidate();
        }
    }

    /// Forget the last snapshot, so the next frame compares against nothing.
    pub fn reset_history(&mut self) {
        self.ring.invalidate();
    }

    /// Recreate the ring for a new render resolution and forget the last
    /// snapshot. The caller has already idled the device.
    ///
    /// # Errors
    ///
    /// Returns an error if a target cannot be created.
    pub fn resize<D>(&mut self, device: &D, extent: PostExtent) -> RenderResult<()>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target>,
    {
        // Released first, so a resize never holds two rings at once.
        self.targets.clear();
        self.ring.invalidate();
        self.targets = create_targets(device, extent, self.ring.slots())?;
        Ok(())
    }

    /// Swap in a freshly built pipeline. Driven by shader hot reload; the
    /// caller has already idled the device.
    pub fn swap_pipeline(&mut self, pipeline: Pipeline) {
        self.pipeline = pipeline;
    }

    /// Last frame's snapshot, or `None` when there is none (the first frame,
    /// after a reset, a resize or a frame that wrote none).
    ///
    /// # Errors
    ///
    /// Returns an error if a failed resize left the ring empty.
    pub fn previous(&self) -> RenderResult<Option<&Target>> {
        if self.ring.valid() {
            self.slot(self.ring.history()).map(Some)
        } else {
            Ok(None)
        }
    }

    /// The slot this frame's snapshot is written to.
    ///
    /// # Errors
    ///
    /// Returns an error if a failed resize left the ring empty.
    pub fn current(&self) -> RenderResult<&Target> {
        self.slot(self.ring.write())
    }

    /// Encode this frame's snapshot of `sources`, timed as `timing`.
    ///
    /// # Errors
    ///
    /// Returns an error if the slot cannot be addressed or the device fails to
    /// encode the draw.
    pub fn encode<'t, D>(
        &'t self,
        device: &D,
        rec: &D::Recorder,
        sources: &SnapshotSources<'t, D>,
        timing: PostTiming,
    ) -> RenderResult<()>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target> + 't,
    {
        let binds = [point(sources.normal_depth), point(sources.velocity)];
        device.encode(
            rec,
            &PostDraw {
                target: device.target_attachment(self.current()?),
                state: PostTargetState::Pass,
                load: PostLoadOp::DontCare,
                timing,
                pipeline: &self.pipeline,
                binds: &binds,
                constants: &[],
                label: "Motion snapshot",
            },
        )?;
        self.written.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn slot(&self, slot: usize) -> RenderResult<&Target> {
        self.targets
            .get(slot)
            .ok_or_else(|| RenderError::Other(format!("{LABEL}: no target in ring slot {slot}")))
    }
}

fn create_targets<D: PostPassDevice>(
    device: &D,
    extent: PostExtent,
    slots: usize,
) -> RenderResult<Vec<D::Target>> {
    let desc = snapshot_desc();
    (0..slots)
        .map(|_| device.create_target(LABEL, &desc, extent))
        .collect()
}

/// A source read by point load; the sampler only fills its slot.
pub fn point<'t, D: PostPassDevice + ?Sized + 't>(texture: D::TextureRef<'t>) -> PostBind<'t, D> {
    PostBind {
        texture,
        sampler: PostSampler::LinearClamp,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::post::mock::{MockDevice, MockPipeline, MockTexture};
    use crate::render::render_graph::PassId;

    const EXTENT: PostExtent = PostExtent {
        width: 1280,
        height: 720,
    };
    const NORMAL_DEPTH: MockTexture = MockTexture::External(1);
    const VELOCITY: MockTexture = MockTexture::External(2);
    const TIMING: PostTiming = PostTiming::Whole(PassId::TaaResolve.timing());

    fn encode(device: &MockDevice, snapshot: &MotionSnapshot<MockPipeline, usize>) {
        snapshot
            .encode(
                device,
                &(),
                &SnapshotSources {
                    normal_depth: NORMAL_DEPTH,
                    velocity: VELOCITY,
                },
                TIMING,
            )
            .expect("encode");
    }

    #[test]
    fn a_frame_copies_the_g_buffer_into_its_own_slot() {
        let device = MockDevice::new();
        let snapshot = MotionSnapshot::new(&device, EXTENT).expect("snapshot");
        encode(&device, &snapshot);
        let draws = device.draws.borrow();
        let [draw] = draws.as_slice() else {
            panic!("one draw, got {}", draws.len());
        };
        assert_eq!(draw.program, PostProgram::MotionSnapshot);
        assert_eq!(draw.target, MockTexture::Target(0));
        assert_eq!(draw.state, PostTargetState::Pass);
        assert_eq!(draw.load, PostLoadOp::DontCare);
        assert_eq!(draw.timing, TIMING);
        let binds: Vec<_> = draw.binds.iter().map(|b| b.0).collect();
        assert_eq!(binds, [NORMAL_DEPTH, VELOCITY]);
    }

    // What a frame wrote is the next frame's previous snapshot, in the other
    // slot from the one that frame writes.
    #[test]
    fn the_written_snapshot_is_the_next_frame_s_previous() {
        let device = MockDevice::new();
        let mut snapshot = MotionSnapshot::new(&device, EXTENT).expect("snapshot");
        assert!(snapshot.previous().expect("ring").is_none());
        encode(&device, &snapshot);
        snapshot.advance();
        assert_eq!(snapshot.previous().expect("ring"), Some(&0));
        assert_eq!(snapshot.current().expect("ring"), &1);
    }

    #[test]
    fn a_frame_without_a_snapshot_a_reset_or_a_resize_forgets_it() {
        let device = MockDevice::new();
        let mut snapshot = MotionSnapshot::new(&device, EXTENT).expect("snapshot");
        let has_previous =
            |s: &MotionSnapshot<MockPipeline, usize>| s.previous().expect("ring").is_some();

        encode(&device, &snapshot);
        snapshot.advance();
        assert!(has_previous(&snapshot));
        snapshot.advance();
        assert!(!has_previous(&snapshot));

        encode(&device, &snapshot);
        snapshot.advance();
        snapshot.reset_history();
        assert!(!has_previous(&snapshot));

        encode(&device, &snapshot);
        snapshot.advance();
        snapshot.resize(&device, EXTENT).expect("resize");
        assert!(!has_previous(&snapshot));
    }

    #[test]
    fn the_pipeline_writes_the_snapshot_format() {
        let device = MockDevice::new();
        let pipeline = build_pipeline(&device).expect("pipeline");
        assert_eq!(pipeline.format, PixelFormat::Rgba16Float);
        assert_eq!(pipeline.blend, PostBlend::Replace);
    }

    #[test]
    fn a_failed_resize_reports_rather_than_indexes() {
        let device = MockDevice::new();
        let mut snapshot = MotionSnapshot::new(&device, EXTENT).expect("snapshot");
        device.fail_creates_after(0);
        assert!(snapshot.resize(&device, EXTENT).is_err());
        assert!(snapshot.current().is_err());
    }
}
