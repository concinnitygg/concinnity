//! The reflection composite, written once for every backend.
//!
//! A reflection resolve (screen-space or ray-traced) writes reflected radiance
//! and a composite weight; this pass blends it over the scene in two draws:
//!
//! 1. the blur, at a reduced resolution: the reflection weight-averaged over a
//!    cone that widens with surface roughness, the expensive part, run at a
//!    fraction of the pixels;
//! 2. the composite, at full resolution: the sharp reflection lerped against
//!    the upsampled blur by roughness, then blended over the scene into the
//!    pass's output, which the rest of the post stack reads as the scene.
//!
//! The pass does not own the reflection it reads. Either resolve writes one,
//! and which one feeds the composite can change from frame to frame.

use crate::render::error::RenderResult;
use crate::render::render_graph::{
    ClearValue, PassId, PixelFormat, TextureDesc, TextureSize, TextureUsage,
};

use super::device::{
    PostBind, PostBlend, PostDraw, PostExtent, PostLoadOp, PostPassDevice, PostSampler,
    PostTargetState, PostTiming,
};
use super::program::PostProgram;

/// The per-frame inputs both draws read.
pub struct ReflectionCompositeInputs<'t, D: PostPassDevice + ?Sized + 't> {
    /// The resolve's output: reflected radiance in `.rgb`, composite weight in
    /// `.a`.
    pub reflection: D::TextureRef<'t>,
    /// The scene the reflection is blended over.
    pub scene: D::TextureRef<'t>,
    /// The G-buffer's view normal (`.rgb`) and linear depth (`.a`).
    pub normal_depth: D::TextureRef<'t>,
    /// The G-buffer's surface roughness.
    pub roughness: D::TextureRef<'t>,
}

/// An HDR color target at the render resolution divided by `scale` on each
/// axis, rendered to and sampled.
///
/// `scale` is a small power of two, so its reciprocal is exact and the graph's
/// fractional resolve floors to the same size integer division does.
fn shape(scale: u32, usage: TextureUsage) -> TextureDesc {
    let size = TextureSize::DrawableScaled(1.0 / scale.max(1) as f32);
    TextureDesc {
        width: size,
        height: size,
        depth: 1,
        format: PixelFormat::Rgba16Float,
        sample_count: 1,
        array_layers: 1,
        mip_levels: 1,
        usage: TextureUsage::RENDER_TARGET
            .union(TextureUsage::SHADER_READ)
            .union(usage),
        clear: ClearValue::Color([0.0, 0.0, 0.0, 0.0]),
    }
}

/// The output's shape: the scene with reflections, at the render resolution.
/// A copy source as well, for the transparent pass's refraction snapshot of
/// the scene it draws over.
pub fn output_desc() -> TextureDesc {
    shape(1, TextureUsage::TRANSFER_SRC)
}

/// The blur's shape: the render resolution divided by `blur_scale`.
pub fn blur_desc(blur_scale: u32) -> TextureDesc {
    shape(blur_scale, TextureUsage(0))
}

/// The two pipelines, built together. What shader hot reload rebuilds and
/// hands to [`ReflectionCompositePass::swap_pipelines`].
pub struct ReflectionCompositePipelines<Pipeline> {
    /// The roughness blur.
    pub blur: Pipeline,
    /// The composite over the scene.
    pub composite: Pipeline,
}

/// Build both pipelines on their own, without touching the targets.
pub fn build_pipelines<D: PostPassDevice>(
    device: &D,
) -> RenderResult<ReflectionCompositePipelines<D::Pipeline>> {
    let format = PixelFormat::Rgba16Float;
    Ok(ReflectionCompositePipelines {
        blur: device.create_pipeline(PostProgram::ReflectionBlur, format, PostBlend::Replace)?,
        composite: device.create_pipeline(
            PostProgram::ReflectionComposite,
            format,
            PostBlend::Replace,
        )?,
    })
}

// The graph labels the targets carry into a backend's own debug naming.
const OUTPUT_LABEL: &str = "reflection_output";
const BLUR_LABEL: &str = "reflection_blur";

/// The blur and composite pipelines, and the two targets they write.
///
/// Parameterized by the two resource types rather than by the device, for the
/// same reason as the temporal resolve: a backend's device value borrows, and
/// the pass is stored on its context.
pub struct ReflectionCompositePass<Pipeline, Target> {
    pipelines: ReflectionCompositePipelines<Pipeline>,
    output: Target,
    blur: Target,
    blur_scale: u32,
}

impl<Pipeline, Target> ReflectionCompositePass<Pipeline, Target> {
    /// Build both pipelines and both targets for a render resolution of
    /// `extent`, with the blur divided by `blur_scale`.
    pub fn new<D>(device: &D, blur_scale: u32, extent: PostExtent) -> RenderResult<Self>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target>,
    {
        let blur_scale = blur_scale.max(1);
        Ok(Self {
            pipelines: build_pipelines(device)?,
            output: device.create_target(OUTPUT_LABEL, &output_desc(), extent)?,
            blur: device.create_target(BLUR_LABEL, &blur_desc(blur_scale), extent)?,
            blur_scale,
        })
    }

    /// The scene with reflections composited in, which the rest of the post
    /// stack reads in place of the resolved scene.
    pub fn output(&self) -> &Target {
        &self.output
    }

    /// The per-axis divisor the blur is sized by.
    pub fn blur_scale(&self) -> u32 {
        self.blur_scale
    }

    /// Recreate both targets for a new render resolution, keeping the old pair
    /// if either fails. The caller has already idled the device.
    pub fn resize<D>(&mut self, device: &D, extent: PostExtent) -> RenderResult<()>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target>,
    {
        let output = device.create_target(OUTPUT_LABEL, &output_desc(), extent)?;
        let blur = device.create_target(BLUR_LABEL, &blur_desc(self.blur_scale), extent)?;
        self.output = output;
        self.blur = blur;
        Ok(())
    }

    /// Swap in freshly built pipelines. Driven by shader hot reload; the caller
    /// has already idled the device.
    pub fn swap_pipelines(&mut self, pipelines: ReflectionCompositePipelines<Pipeline>) {
        self.pipelines = pipelines;
    }

    /// Encode the blur, then the composite into [`Self::output`].
    pub fn encode<'t, D>(
        &'t self,
        device: &D,
        rec: &D::Recorder,
        inputs: ReflectionCompositeInputs<'t, D>,
    ) -> RenderResult<()>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target> + 't,
    {
        device.encode(
            rec,
            &PostDraw {
                target: device.target_attachment(&self.blur),
                // Only the composite reads the blur, so the graph does not
                // declare it.
                state: PostTargetState::Pass,
                load: PostLoadOp::DontCare,
                timing: PostTiming::First(PassId::ReflectionComposite),
                pipeline: &self.pipelines.blur,
                binds: &[
                    linear::<D>(inputs.reflection),
                    linear::<D>(inputs.roughness),
                ],
                constants: &[],
                label: "reflection blur",
            },
        )?;
        let binds = [
            linear::<D>(inputs.reflection),
            linear::<D>(inputs.scene),
            linear::<D>(inputs.normal_depth),
            linear::<D>(inputs.roughness),
            linear::<D>(device.target_ref(&self.blur)),
        ];
        device.encode(
            rec,
            &PostDraw {
                target: device.target_attachment(&self.output),
                // The output is the graph's scene for the passes after it,
                // which this node writes.
                state: PostTargetState::Graph,
                load: PostLoadOp::DontCare,
                timing: PostTiming::Last(PassId::ReflectionComposite),
                pipeline: &self.pipelines.composite,
                binds: &binds,
                constants: &[],
                label: "reflection composite",
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
    use crate::render::post::device::resolve_extent;
    use crate::render::post::mock::{MockDevice, MockDraw, MockPipeline, MockTexture};
    use alloc::vec::Vec;

    const EXTENT: PostExtent = PostExtent {
        width: 1280,
        height: 720,
    };

    // Target indices in the order `new` creates them.
    const OUTPUT: usize = 0;
    const BLUR: usize = 1;

    const REFLECTION: MockTexture = MockTexture::External(1);
    const SCENE: MockTexture = MockTexture::External(2);
    const NORMAL_DEPTH: MockTexture = MockTexture::External(3);
    const ROUGHNESS: MockTexture = MockTexture::External(4);

    fn encode_frame(
        device: &MockDevice,
        pass: &ReflectionCompositePass<MockPipeline, usize>,
    ) -> Vec<MockDraw> {
        device.draws.borrow_mut().clear();
        pass.encode(
            device,
            &(),
            ReflectionCompositeInputs {
                reflection: REFLECTION,
                scene: SCENE,
                normal_depth: NORMAL_DEPTH,
                roughness: ROUGHNESS,
            },
        )
        .expect("encode");
        device.draws.borrow().clone()
    }

    fn sources(draw: &MockDraw) -> Vec<MockTexture> {
        draw.binds.iter().map(|b| b.0).collect()
    }

    #[test]
    fn the_output_is_full_resolution_and_the_blur_is_divided() {
        let device = MockDevice::new();
        ReflectionCompositePass::new(&device, 4, EXTENT).expect("pass");
        let targets = device.targets.borrow();
        assert_eq!(targets.len(), 2);
        assert_eq!(targets[OUTPUT].extent, EXTENT);
        assert_eq!(
            targets[BLUR].extent,
            PostExtent {
                width: 320,
                height: 180
            }
        );
        assert!(output_desc().usage.contains(TextureUsage::TRANSFER_SRC));
        assert!(output_desc().usage.contains(TextureUsage::SHADER_READ));
        assert!(!blur_desc(2).usage.contains(TextureUsage::TRANSFER_SRC));
    }

    #[test]
    fn the_blur_resolution_is_what_integer_division_gives() {
        // The scales an author can pick, against sizes that do not divide.
        for scale in [1, 2, 4] {
            for (w, h) in [(1280, 720), (1921, 1081), (3, 1), (2560, 1439), (1, 1)] {
                let e = resolve_extent(
                    &blur_desc(scale),
                    PostExtent {
                        width: w,
                        height: h,
                    },
                );
                assert_eq!(
                    (e.width, e.height),
                    ((w / scale).max(1), (h / scale).max(1)),
                    "{w}x{h} at 1/{scale}"
                );
            }
        }
    }

    #[test]
    fn a_frame_blurs_then_composites_in_the_shaders_slot_order() {
        let device = MockDevice::new();
        let pass = ReflectionCompositePass::new(&device, 2, EXTENT).expect("pass");
        let draws = encode_frame(&device, &pass);
        assert_eq!(draws.len(), 2);
        let (blur, composite) = (&draws[0], &draws[1]);
        assert_eq!(blur.program, PostProgram::ReflectionBlur);
        assert_eq!(blur.target, MockTexture::Target(BLUR));
        assert_eq!(sources(blur), [REFLECTION, ROUGHNESS]);
        assert_eq!(composite.program, PostProgram::ReflectionComposite);
        assert_eq!(composite.target, MockTexture::Target(OUTPUT));
        assert_eq!(
            sources(composite),
            [
                REFLECTION,
                SCENE,
                NORMAL_DEPTH,
                ROUGHNESS,
                MockTexture::Target(BLUR)
            ]
        );
        for d in &draws {
            assert!(d.constants.is_empty());
            assert_eq!(d.load, PostLoadOp::DontCare);
            assert!(d.binds.iter().all(|b| b.1 == PostSampler::LinearClamp));
        }
    }

    #[test]
    fn only_the_output_is_the_graphs() {
        let device = MockDevice::new();
        let pass = ReflectionCompositePass::new(&device, 2, EXTENT).expect("pass");
        let draws = encode_frame(&device, &pass);
        assert_eq!(draws[0].state, PostTargetState::Pass);
        assert_eq!(draws[1].state, PostTargetState::Graph);
        let p = build_pipelines(&device).expect("pipelines");
        for pipeline in [p.blur, p.composite] {
            assert_eq!(pipeline.blend, PostBlend::Replace);
            assert_eq!(pipeline.format, PixelFormat::Rgba16Float);
        }
    }

    #[test]
    fn the_timing_span_covers_both_draws() {
        let device = MockDevice::new();
        let pass = ReflectionCompositePass::new(&device, 2, EXTENT).expect("pass");
        let draws = encode_frame(&device, &pass);
        assert_eq!(
            draws[0].timing,
            PostTiming::First(PassId::ReflectionComposite)
        );
        assert_eq!(
            draws[1].timing,
            PostTiming::Last(PassId::ReflectionComposite)
        );
    }

    #[test]
    fn a_resize_keeps_the_blur_scale() {
        let device = MockDevice::new();
        let mut pass = ReflectionCompositePass::new(&device, 4, EXTENT).expect("pass");
        pass.resize(
            &device,
            PostExtent {
                width: 2560,
                height: 1440,
            },
        )
        .expect("resize");
        assert_eq!(pass.blur_scale(), 4);
        let targets = device.targets.borrow();
        assert_eq!(targets.len(), 4);
        assert_eq!(
            targets[3].extent,
            PostExtent {
                width: 640,
                height: 360
            }
        );
        // The pass reads the targets it created last.
        drop(targets);
        let draws = encode_frame(&device, &pass);
        assert_eq!(draws[1].target, MockTexture::Target(2));
        assert_eq!(sources(&draws[1])[4], MockTexture::Target(3));
    }

    #[test]
    fn a_failed_resize_keeps_both_old_targets() {
        let device = MockDevice::new();
        let mut pass = ReflectionCompositePass::new(&device, 4, EXTENT).expect("pass");
        device.fail_creates_after(1);
        let resized = PostExtent {
            width: 2560,
            height: 1440,
        };
        assert!(pass.resize(&device, resized).is_err());
        let draws = encode_frame(&device, &pass);
        assert_eq!(draws[0].target, MockTexture::Target(BLUR));
        assert_eq!(draws[1].target, MockTexture::Target(OUTPUT));
        assert_eq!(sources(&draws[1])[4], MockTexture::Target(BLUR));
        let targets = device.targets.borrow();
        assert_eq!(targets[OUTPUT].extent, EXTENT);
        assert_eq!(targets[BLUR].extent.width, EXTENT.width / 4);
    }

    #[test]
    fn a_zero_blur_scale_is_full_resolution() {
        let device = MockDevice::new();
        let pass = ReflectionCompositePass::new(&device, 0, EXTENT).expect("pass");
        assert_eq!(pass.blur_scale(), 1);
        assert_eq!(device.targets.borrow()[BLUR].extent, EXTENT);
    }
}
