//! Screen-space ambient occlusion (GTAO), written once for every backend.
//!
//! Two draws at the render resolution over the G-buffer's view normal and
//! linear depth: the kernel integrates each pixel's visible horizon arc into a
//! raw, noisy occlusion, and a depth-aware blur smooths it into the occlusion
//! the lit forward pass multiplies its ambient term by.
//!
//! The blurred occlusion is the render graph's `ao_output` transient, which the
//! pool owns and the forward pass reads, so the caller supplies it. The raw
//! occlusion between the two draws is owned here.

use crate::gfx::render_types::SsaoParams;
use crate::render::error::RenderResult;
use crate::render::render_graph::{
    ClearValue, PassId, PixelFormat, TextureDesc, TextureSize, TextureUsage,
};

use super::device::{
    PostBind, PostBlend, PostDraw, PostExtent, PostLoadOp, PostPassDevice, PostSampler,
    PostTargetState, PostTiming,
};
use super::program::PostProgram;

/// Clamped SSAO tunables and the per-frame uniform they build.
pub mod settings;

/// The per-frame inputs the pass reads and writes beyond its own target.
pub struct SsaoInputs<'t, D: PostPassDevice + ?Sized + 't> {
    /// The G-buffer's view normal (`.rgb`) and linear depth (`.a`).
    pub normal_depth: D::TextureRef<'t>,
    /// The blurred occlusion, the graph's `ao_output`.
    pub output: D::Attachment<'t>,
}

/// The format of both occlusion targets: single-channel visibility, 1.0
/// unoccluded.
pub const OCCLUSION_FORMAT: PixelFormat = PixelFormat::R8Unorm;

/// The raw occlusion's shape: single-channel at the render resolution,
/// rendered to and sampled.
pub fn raw_desc() -> TextureDesc {
    TextureDesc {
        width: TextureSize::Drawable,
        height: TextureSize::Drawable,
        depth: 1,
        format: OCCLUSION_FORMAT,
        sample_count: 1,
        array_layers: 1,
        mip_levels: 1,
        usage: TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
        clear: ClearValue::Color([0.0, 0.0, 0.0, 0.0]),
    }
}

/// The two pipelines, built together. What shader hot reload rebuilds and
/// hands to [`SsaoPass::swap_pipelines`].
pub struct SsaoPipelines<Pipeline> {
    /// The horizon search.
    pub kernel: Pipeline,
    /// The depth-aware blur.
    pub blur: Pipeline,
}

/// Build both pipelines on their own, without touching the target.
pub fn build_pipelines<D: PostPassDevice>(device: &D) -> RenderResult<SsaoPipelines<D::Pipeline>> {
    Ok(SsaoPipelines {
        kernel: device.create_pipeline(
            PostProgram::SsaoKernel,
            OCCLUSION_FORMAT,
            PostBlend::Replace,
        )?,
        blur: device.create_pipeline(
            PostProgram::SsaoBlur,
            OCCLUSION_FORMAT,
            PostBlend::Replace,
        )?,
    })
}

// The graph label the raw occlusion carries into a backend's own debug naming.
const RAW_LABEL: &str = "ao_raw";

/// The kernel and blur pipelines and the raw occlusion between them.
///
/// Parameterized by the two resource types rather than by the device, for the
/// same reason as the temporal resolve: a backend's device value borrows, and
/// the pass is stored on its context.
pub struct SsaoPass<Pipeline, Target> {
    pipelines: SsaoPipelines<Pipeline>,
    raw: Target,
}

impl<Pipeline, Target> SsaoPass<Pipeline, Target> {
    /// Build both pipelines and the raw occlusion for a render resolution of
    /// `extent`.
    pub fn new<D>(device: &D, extent: PostExtent) -> RenderResult<Self>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target>,
    {
        Ok(Self {
            pipelines: build_pipelines(device)?,
            raw: device.create_target(RAW_LABEL, &raw_desc(), extent)?,
        })
    }

    /// Recreate the raw occlusion for a new render resolution. The caller has
    /// already idled the device.
    pub fn resize<D>(&mut self, device: &D, extent: PostExtent) -> RenderResult<()>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target>,
    {
        self.raw = device.create_target(RAW_LABEL, &raw_desc(), extent)?;
        Ok(())
    }

    /// Swap in freshly built pipelines. Driven by shader hot reload; the caller
    /// has already idled the device.
    pub fn swap_pipelines(&mut self, pipelines: SsaoPipelines<Pipeline>) {
        self.pipelines = pipelines;
    }

    /// Encode the kernel into the raw occlusion, then the blur into
    /// `inputs.output`.
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
        device.encode(
            rec,
            &PostDraw {
                target: device.target_attachment(&self.raw),
                // Only the blur reads the raw occlusion, so the graph does not
                // declare it.
                state: PostTargetState::Pass,
                load: PostLoadOp::DontCare,
                timing: PostTiming::Whole(PassId::SsaoKernel),
                pipeline: &self.pipelines.kernel,
                binds: &[linear::<D>(inputs.normal_depth)],
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
                    linear::<D>(device.target_ref(&self.raw)),
                    linear::<D>(inputs.normal_depth),
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
    use alloc::vec::Vec;
    use settings::SsaoSettings;

    const EXTENT: PostExtent = PostExtent {
        width: 1280,
        height: 720,
    };

    // The raw occlusion is the only target the pass creates.
    const RAW: usize = 0;

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
    fn the_raw_occlusion_is_single_channel_at_the_render_resolution() {
        let device = MockDevice::new();
        SsaoPass::new(&device, EXTENT).expect("pass");
        let targets = device.targets.borrow();
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[RAW].extent, EXTENT);
        assert_eq!(raw_desc().format, PixelFormat::R8Unorm);
        assert!(raw_desc().usage.contains(TextureUsage::SHADER_READ));
    }

    #[test]
    fn a_frame_runs_the_kernel_then_blurs_into_the_output() {
        let device = MockDevice::new();
        let pass = SsaoPass::new(&device, EXTENT).expect("pass");
        let draws = encode_frame(&device, &pass);
        assert_eq!(draws.len(), 2);
        let (kernel, blur) = (&draws[0], &draws[1]);
        assert_eq!(kernel.program, PostProgram::SsaoKernel);
        assert_eq!(kernel.target, MockTexture::Target(RAW));
        assert_eq!(sources(kernel), [NORMAL_DEPTH]);
        assert_eq!(kernel.constants, bytemuck::bytes_of(&params()));
        assert_eq!(kernel.constants.len(), 16);
        assert_eq!(blur.program, PostProgram::SsaoBlur);
        assert_eq!(blur.target, OUTPUT);
        assert_eq!(sources(blur), [MockTexture::Target(RAW), NORMAL_DEPTH]);
        assert!(blur.constants.is_empty());
    }

    #[test]
    fn only_the_output_is_the_graphs_and_each_draw_is_timed_alone() {
        let device = MockDevice::new();
        let pass = SsaoPass::new(&device, EXTENT).expect("pass");
        let draws = encode_frame(&device, &pass);
        assert_eq!(draws[0].state, PostTargetState::Pass);
        assert_eq!(draws[1].state, PostTargetState::Graph);
        assert_eq!(draws[0].timing, PostTiming::Whole(PassId::SsaoKernel));
        assert_eq!(draws[1].timing, PostTiming::Whole(PassId::SsaoBlur));
        for d in &draws {
            assert_eq!(d.load, PostLoadOp::DontCare);
            assert!(d.binds.iter().all(|b| b.1 == PostSampler::LinearClamp));
        }
        let p = build_pipelines(&device).expect("pipelines");
        for pipeline in [p.kernel, p.blur] {
            assert_eq!(pipeline.blend, PostBlend::Replace);
            assert_eq!(pipeline.format, PixelFormat::R8Unorm);
        }
    }

    #[test]
    fn a_resize_recreates_the_raw_occlusion() {
        let device = MockDevice::new();
        let mut pass = SsaoPass::new(&device, EXTENT).expect("pass");
        let resized = PostExtent {
            width: 640,
            height: 360,
        };
        pass.resize(&device, resized).expect("resize");
        assert_eq!(device.targets.borrow()[1].extent, resized);
        let draws = encode_frame(&device, &pass);
        assert_eq!(draws[0].target, MockTexture::Target(1));
        assert_eq!(sources(&draws[1])[0], MockTexture::Target(1));
    }
}
