//! Screen-space ambient occlusion (GTAO), written once for every backend.
//!
//! Three draws at the render resolution over the G-buffer's view normal and
//! linear depth: a copy of that depth alone in one half-precision channel, so
//! the many depth taps that follow fetch a quarter of the bytes; the kernel,
//! which integrates each pixel's visible horizon arc into a raw, noisy
//! occlusion; and a depth-aware blur that smooths it into the occlusion the lit
//! forward pass multiplies its ambient term by.
//!
//! The blurred occlusion is the render graph's `ao_output` transient, which the
//! pool owns and the forward pass reads, so the caller supplies it. The depth
//! copy and the raw occlusion are owned here.

use crate::gfx::render_types::SsaoParams;
use crate::render::error::RenderResult;
use crate::render::render_graph::PassId;

use super::device::{
    PostBind, PostBlend, PostDraw, PostExtent, PostLoadOp, PostPassDevice, PostSampler,
    PostTargetState, PostTiming,
};
use super::program::PostProgram;

/// Clamped SSAO tunables and the per-frame uniform they build.
pub mod settings;

/// The shapes of the depth copy and the occlusion targets.
pub mod targets;

pub use targets::{DEPTH_FORMAT, OCCLUSION_FORMAT, depth_desc, raw_desc};

/// The per-frame inputs the pass reads and writes beyond its own targets.
pub struct SsaoInputs<'t, D: PostPassDevice + ?Sized + 't> {
    /// The G-buffer's view normal (`.rgb`) and linear depth (`.a`).
    pub normal_depth: D::TextureRef<'t>,
    /// The blurred occlusion, the graph's `ao_output`.
    pub output: D::Attachment<'t>,
}

/// The pipelines, built together. What shader hot reload rebuilds and hands to
/// [`SsaoPass::swap_pipelines`].
pub struct SsaoPipelines<Pipeline> {
    /// The depth copy.
    pub depth: Pipeline,
    /// The horizon search.
    pub kernel: Pipeline,
    /// The depth-aware blur.
    pub blur: Pipeline,
}

/// Build every pipeline on its own, without touching the targets.
pub fn build_pipelines<D: PostPassDevice>(device: &D) -> RenderResult<SsaoPipelines<D::Pipeline>> {
    let build = |program, format| device.create_pipeline(program, format, PostBlend::Replace);
    Ok(SsaoPipelines {
        depth: build(PostProgram::SsaoDepth, DEPTH_FORMAT)?,
        kernel: build(PostProgram::SsaoKernel, OCCLUSION_FORMAT)?,
        blur: build(PostProgram::SsaoBlur, OCCLUSION_FORMAT)?,
    })
}

// The graph labels the pass's targets carry into a backend's own debug naming.
const DEPTH_LABEL: &str = "ao_depth";
const RAW_LABEL: &str = "ao_raw";

// The targets sized to one render resolution.
struct SsaoTargets<Target> {
    depth: Target,
    raw: Target,
}

impl<Target> SsaoTargets<Target> {
    fn new<D>(device: &D, extent: PostExtent) -> RenderResult<Self>
    where
        D: PostPassDevice<Target = Target>,
    {
        Ok(Self {
            depth: device.create_target(DEPTH_LABEL, &depth_desc(), extent)?,
            raw: device.create_target(RAW_LABEL, &raw_desc(), extent)?,
        })
    }
}

/// The pipelines, the depth copy, and the raw occlusion between the kernel and
/// the blur.
///
/// Parameterized by the two resource types rather than by the device, for the
/// same reason as the temporal resolve: a backend's device value borrows, and
/// the pass is stored on its context.
pub struct SsaoPass<Pipeline, Target> {
    pipelines: SsaoPipelines<Pipeline>,
    targets: SsaoTargets<Target>,
}

impl<Pipeline, Target> SsaoPass<Pipeline, Target> {
    /// Build every pipeline and target for a render resolution of `extent`.
    pub fn new<D>(device: &D, extent: PostExtent) -> RenderResult<Self>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target>,
    {
        Ok(Self {
            pipelines: build_pipelines(device)?,
            targets: SsaoTargets::new(device, extent)?,
        })
    }

    /// Recreate the targets for a new render resolution. The caller has already
    /// idled the device.
    pub fn resize<D>(&mut self, device: &D, extent: PostExtent) -> RenderResult<()>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target>,
    {
        self.targets = SsaoTargets::new(device, extent)?;
        Ok(())
    }

    /// Swap in freshly built pipelines. Driven by shader hot reload; the caller
    /// has already idled the device.
    pub fn swap_pipelines(&mut self, pipelines: SsaoPipelines<Pipeline>) {
        self.pipelines = pipelines;
    }

    /// Encode the depth copy, the kernel into the raw occlusion, then the blur
    /// into `inputs.output`.
    pub fn encode<'t, D>(
        &'t self,
        device: &D,
        rec: &D::Recorder,
        inputs: SsaoInputs<'t, D>,
        params: &SsaoParams,
    ) -> RenderResult<()>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target> + 't,
    {
        let depth = device.target_ref(&self.targets.depth);
        // Only the kernel and the blur read the depth copy and the raw
        // occlusion, so the graph declares neither.
        device.encode(
            rec,
            &PostDraw {
                target: device.target_attachment(&self.targets.depth),
                state: PostTargetState::Pass,
                load: PostLoadOp::DontCare,
                timing: PostTiming::Whole(PassId::SsaoDepth),
                pipeline: &self.pipelines.depth,
                binds: &[linear::<D>(inputs.normal_depth)],
                constants: &[],
                label: "SSAO depth",
            },
        )?;
        device.encode(
            rec,
            &PostDraw {
                target: device.target_attachment(&self.targets.raw),
                state: PostTargetState::Pass,
                load: PostLoadOp::DontCare,
                timing: PostTiming::Whole(PassId::SsaoKernel),
                pipeline: &self.pipelines.kernel,
                binds: &[linear::<D>(inputs.normal_depth), linear::<D>(depth)],
                constants: bytemuck::bytes_of(params),
                label: "SSAO kernel",
            },
        )?;
        device.encode(
            rec,
            &PostDraw {
                target: inputs.output,
                // `ao_output` is the graph's, which this node writes.
                state: PostTargetState::Graph,
                load: PostLoadOp::DontCare,
                timing: PostTiming::Whole(PassId::SsaoBlur),
                pipeline: &self.pipelines.blur,
                binds: &[
                    linear::<D>(device.target_ref(&self.targets.raw)),
                    linear::<D>(depth),
                ],
                constants: &[],
                label: "SSAO blur",
            },
        )
    }
}

fn linear<'t, D: PostPassDevice + ?Sized + 't>(texture: D::TextureRef<'t>) -> PostBind<'t, D> {
    PostBind {
        texture,
        sampler: PostSampler::LinearClamp,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::post::mock::{MockDevice, MockDraw, MockPipeline, MockTexture};
    use crate::render::render_graph::PixelFormat;
    use alloc::vec::Vec;
    use settings::SsaoSettings;

    const EXTENT: PostExtent = PostExtent {
        width: 1280,
        height: 720,
    };

    // The targets in creation order: the depth copy, then the raw occlusion.
    const DEPTH: usize = 0;
    const RAW: usize = 1;

    const NORMAL_DEPTH: MockTexture = MockTexture::External(1);
    const OUTPUT: MockTexture = MockTexture::External(2);

    fn params() -> SsaoParams {
        SsaoSettings::resolve(0.6, 1.1).params(1.0, 1.6)
    }

    fn encode_frame(device: &MockDevice, pass: &SsaoPass<MockPipeline, usize>) -> Vec<MockDraw> {
        device.draws.borrow_mut().clear();
        pass.encode(
            device,
            &(),
            SsaoInputs {
                normal_depth: NORMAL_DEPTH,
                output: OUTPUT,
            },
            &params(),
        )
        .expect("encode");
        device.draws.borrow().clone()
    }

    fn sources(draw: &MockDraw) -> Vec<MockTexture> {
        draw.binds.iter().map(|b| b.0).collect()
    }

    #[test]
    fn the_pass_creates_a_depth_copy_and_a_raw_occlusion_at_the_render_resolution() {
        let device = MockDevice::new();
        SsaoPass::new(&device, EXTENT).expect("pass");
        let targets = device.targets.borrow();
        assert_eq!(targets.len(), 2);
        for t in targets.iter() {
            assert_eq!(t.extent, EXTENT);
            assert_eq!(t.levels, 1);
        }
    }

    #[test]
    fn a_frame_copies_the_depth_then_runs_the_kernel_and_the_blur() {
        let device = MockDevice::new();
        let pass = SsaoPass::new(&device, EXTENT).expect("pass");
        let draws = encode_frame(&device, &pass);
        assert_eq!(draws.len(), 3);
        let (depth, kernel, blur) = (&draws[0], &draws[1], &draws[2]);

        assert_eq!(depth.program, PostProgram::SsaoDepth);
        assert_eq!(depth.target, MockTexture::Target(DEPTH));
        assert_eq!(sources(depth), [NORMAL_DEPTH]);
        assert!(depth.constants.is_empty());

        // The kernel reads the normal from the G-buffer and every step's depth
        // from the copy.
        assert_eq!(kernel.program, PostProgram::SsaoKernel);
        assert_eq!(kernel.target, MockTexture::Target(RAW));
        assert_eq!(sources(kernel), [NORMAL_DEPTH, MockTexture::Target(DEPTH)]);
        assert_eq!(kernel.constants, bytemuck::bytes_of(&params()));
        assert_eq!(kernel.constants.len(), 16);

        assert_eq!(blur.program, PostProgram::SsaoBlur);
        assert_eq!(blur.target, OUTPUT);
        assert_eq!(
            sources(blur),
            [MockTexture::Target(RAW), MockTexture::Target(DEPTH)]
        );
        assert!(blur.constants.is_empty());
    }

    #[test]
    fn only_the_output_is_the_graphs_and_each_draw_is_timed_alone() {
        let device = MockDevice::new();
        let pass = SsaoPass::new(&device, EXTENT).expect("pass");
        let draws = encode_frame(&device, &pass);
        assert_eq!(draws[0].state, PostTargetState::Pass);
        assert_eq!(draws[1].state, PostTargetState::Pass);
        assert_eq!(draws[2].state, PostTargetState::Graph);
        assert_eq!(draws[0].timing, PostTiming::Whole(PassId::SsaoDepth));
        assert_eq!(draws[1].timing, PostTiming::Whole(PassId::SsaoKernel));
        assert_eq!(draws[2].timing, PostTiming::Whole(PassId::SsaoBlur));
        for d in &draws {
            assert_eq!(d.load, PostLoadOp::DontCare);
            assert!(d.binds.iter().all(|b| b.1 == PostSampler::LinearClamp));
        }
    }

    #[test]
    fn the_copy_writes_depth_and_the_kernel_and_blur_occlusion() {
        let device = MockDevice::new();
        let p = build_pipelines(&device).expect("pipelines");
        assert_eq!(p.depth.format, PixelFormat::R16Float);
        assert_eq!(p.kernel.format, PixelFormat::R8Unorm);
        assert_eq!(p.blur.format, PixelFormat::R8Unorm);
        for pipeline in [p.depth, p.kernel, p.blur] {
            assert_eq!(pipeline.blend, PostBlend::Replace);
        }
    }

    #[test]
    fn a_resize_recreates_both_targets() {
        let device = MockDevice::new();
        let mut pass = SsaoPass::new(&device, EXTENT).expect("pass");
        let resized = PostExtent {
            width: 640,
            height: 360,
        };
        pass.resize(&device, resized).expect("resize");
        {
            let targets = device.targets.borrow();
            assert_eq!(targets.len(), 4);
            assert!(targets[2..].iter().all(|t| t.extent == resized));
        }
        let draws = encode_frame(&device, &pass);
        assert_eq!(draws[0].target, MockTexture::Target(2));
        assert_eq!(
            sources(&draws[2]),
            [MockTexture::Target(3), MockTexture::Target(2)]
        );
    }

    #[test]
    fn a_failed_target_creation_fails_the_pass() {
        let device = MockDevice::new();
        device.fail_creates_after(1);
        assert!(SsaoPass::<MockPipeline, usize>::new(&device, EXTENT).is_err());
    }
}
