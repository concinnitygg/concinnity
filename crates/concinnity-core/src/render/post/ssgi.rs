//! Screen-space global illumination, written once for every backend.
//!
//! Two draws at two resolutions. The gather casts cosine-weighted hemisphere
//! rays against the G-buffer and accumulates the lit scene color each one hits
//! into a `gi_scale`-reduced target. The composite blurs that noisy term by
//! depth similarity, upsampling it as it goes, and adds it into the scene at
//! full resolution.
//!
//! The gather reads the scene the composite then writes. A backend whose
//! resource states follow the graph has to move the scene between those two
//! draws, which is why each draw is also encodable on its own.

use alloc::string::String;

use crate::gfx::render_types::SsgiParams;
use crate::render::render_graph::{
    ClearValue, PassId, PixelFormat, TextureDesc, TextureSize, TextureUsage,
};

use super::device::{
    PostBind, PostBlend, PostDraw, PostExtent, PostLoadOp, PostPassDevice, PostSampler,
    PostTargetState, PostTiming,
};
use super::program::PostProgram;

/// The per-frame inputs both draws read and write.
pub struct SsgiInputs<'t, D: PostPassDevice + ?Sized + 't> {
    /// The lit scene the gather samples its bounce radiance from.
    pub scene: D::TextureRef<'t>,
    /// The same scene, as the composite's target.
    pub scene_target: D::Attachment<'t>,
    /// The G-buffer's view normal (`.rgb`) and linear depth (`.a`).
    pub normal_depth: D::TextureRef<'t>,
}

/// The two pipelines, built together. What shader hot reload rebuilds and
/// hands to [`SsgiPass::swap_pipelines`].
pub struct SsgiPipelines<Pipeline> {
    /// The hemisphere gather, writing the reduced target.
    pub gather: Pipeline,
    /// The depth-aware blur, adding into the scene.
    pub composite: Pipeline,
}

/// Build both pipelines on their own, without touching the target.
pub fn build_pipelines<D: PostPassDevice>(
    device: &D,
) -> Result<SsgiPipelines<D::Pipeline>, String> {
    let format = PixelFormat::Rgba16Float;
    Ok(SsgiPipelines {
        gather: device.create_pipeline(PostProgram::SsgiGather, format, PostBlend::Replace)?,
        composite: device.create_pipeline(
            PostProgram::SsgiComposite,
            format,
            PostBlend::Additive,
        )?,
    })
}

// The graph label the gathered term carries into a backend's own debug naming.
const TARGET_LABEL: &str = "ssgi_gi";

/// The gather target's shape: HDR color at the render resolution divided by
/// `gi_scale` on each axis, rendered to and sampled.
///
/// `gi_scale` is a small power of two, so its reciprocal is exact and the
/// graph's fractional resolve floors to the same size integer division does.
pub fn gi_desc(gi_scale: u32) -> TextureDesc {
    let scale = TextureSize::DrawableScaled(1.0 / gi_scale.max(1) as f32);
    TextureDesc {
        width: scale,
        height: scale,
        depth: 1,
        format: PixelFormat::Rgba16Float,
        sample_count: 1,
        array_layers: 1,
        mip_levels: 1,
        usage: TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
        clear: ClearValue::Color([0.0, 0.0, 0.0, 0.0]),
    }
}

/// The gather and composite pipelines, and the reduced target between them.
///
/// Parameterized by the two resource types rather than by the device, for the
/// same reason as the temporal resolve: a backend's device value borrows, and
/// the pass is stored on its context.
pub struct SsgiPass<Pipeline, Target> {
    pipelines: SsgiPipelines<Pipeline>,
    gi: Target,
    desc: TextureDesc,
}

impl<Pipeline, Target> SsgiPass<Pipeline, Target> {
    /// Build both pipelines and the gather target for a render resolution of
    /// `extent`.
    pub fn new<D>(device: &D, gi_scale: u32, extent: PostExtent) -> Result<Self, String>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target>,
    {
        let desc = gi_desc(gi_scale);
        Ok(Self {
            pipelines: build_pipelines(device)?,
            gi: device.create_target(TARGET_LABEL, &desc, extent)?,
            desc,
        })
    }

    /// The gathered, not yet blurred, indirect term.
    pub fn gi(&self) -> &Target {
        &self.gi
    }

    /// Recreate the gather target for a new render resolution. The caller has
    /// already idled the device.
    pub fn resize<D>(&mut self, device: &D, extent: PostExtent) -> Result<(), String>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target>,
    {
        self.gi = device.create_target(TARGET_LABEL, &self.desc, extent)?;
        Ok(())
    }

    /// Swap in freshly built pipelines. Driven by shader hot reload; the caller
    /// has already idled the device.
    pub fn swap_pipelines(&mut self, pipelines: SsgiPipelines<Pipeline>) {
        self.pipelines = pipelines;
    }

    /// Encode the gather, then the composite.
    pub fn encode<'t, D>(
        &'t self,
        device: &D,
        rec: &D::Recorder,
        inputs: SsgiInputs<'t, D>,
        params: &SsgiParams,
    ) -> Result<(), String>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target> + 't,
    {
        self.encode_gather(device, rec, inputs.scene, inputs.normal_depth, params)?;
        self.encode_composite(
            device,
            rec,
            inputs.scene_target,
            inputs.normal_depth,
            params,
        )
    }

    /// Encode the gather alone: `scene` hemisphere-marched over `normal_depth`
    /// into the reduced target.
    pub fn encode_gather<'t, D>(
        &'t self,
        device: &D,
        rec: &D::Recorder,
        scene: D::TextureRef<'t>,
        normal_depth: D::TextureRef<'t>,
        params: &SsgiParams,
    ) -> Result<(), String>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target> + 't,
    {
        let binds = [screen::<D>(scene), screen::<D>(normal_depth)];
        device.encode(
            rec,
            &PostDraw {
                target: device.target_attachment(&self.gi),
                // Only this pass reads the gathered term, so the graph does not
                // declare it.
                state: PostTargetState::Pass,
                load: PostLoadOp::DontCare,
                timing: PostTiming::First(PassId::Ssgi),
                pipeline: &self.pipelines.gather,
                binds: &binds,
                constants: bytemuck::bytes_of(params),
                label: "SSGI gather",
            },
        )
    }

    /// Encode the composite alone: the gathered term blurred over
    /// `normal_depth` and added into `scene_target`.
    pub fn encode_composite<'t, D>(
        &'t self,
        device: &D,
        rec: &D::Recorder,
        scene_target: D::Attachment<'t>,
        normal_depth: D::TextureRef<'t>,
        params: &SsgiParams,
    ) -> Result<(), String>
    where
        D: PostPassDevice<Pipeline = Pipeline, Target = Target> + 't,
    {
        let binds = [
            screen::<D>(device.target_ref(&self.gi)),
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

fn screen<'t, D: PostPassDevice + ?Sized + 't>(texture: D::TextureRef<'t>) -> PostBind<'t, D> {
    PostBind {
        texture,
        sampler: PostSampler::LinearClamp,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::ssgi::SsgiSettings;
    use crate::render::post::device::resolve_extent;
    use crate::render::post::mock::{MockDevice, MockTexture};

    fn params() -> SsgiParams {
        SsgiParams {
            intensity: 1.0,
            max_distance: 8.0,
            tan_half_fov_y: 0.6,
            aspect: 1.5,
            stride: 0.5,
            thickness: 0.2,
            rays: 8.0,
            steps: 16.0,
        }
    }

    #[test]
    fn the_gather_target_is_the_size_the_settings_divide_to() {
        // The graph's fractional resolve has to land where the integer division
        // the settings expose does, for every scale an author can pick.
        for gi_scale in [1, 2, 4] {
            let s = SsgiSettings::resolve(1.0, 8.0, 8, 16, gi_scale);
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
    fn the_pass_creates_one_reduced_target_and_resizes_it() {
        let device = MockDevice::new();
        let mut pass = SsgiPass::new(
            &device,
            2,
            PostExtent {
                width: 1280,
                height: 720,
            },
        )
        .expect("pass");
        pass.resize(
            &device,
            PostExtent {
                width: 640,
                height: 360,
            },
        )
        .expect("resize");
        let targets = device.targets.borrow();
        assert_eq!(targets.len(), 2);
        assert_eq!(
            targets[0].extent,
            PostExtent {
                width: 640,
                height: 360
            }
        );
        assert_eq!(
            targets[1].extent,
            PostExtent {
                width: 320,
                height: 180
            }
        );
        assert_eq!(*pass.gi(), 1);
    }

    #[test]
    fn the_gather_writes_the_reduced_target_and_the_composite_adds_it_into_the_scene() {
        let device = MockDevice::new();
        let pass = SsgiPass::new(
            &device,
            2,
            PostExtent {
                width: 1280,
                height: 720,
            },
        )
        .expect("pass");
        let p = params();
        pass.encode(
            &device,
            &(),
            SsgiInputs {
                scene: MockTexture::External(1),
                scene_target: MockTexture::External(1),
                normal_depth: MockTexture::External(2),
            },
            &p,
        )
        .expect("encode");
        let draws = device.draws.borrow();
        assert_eq!(draws.len(), 2);
        let (gather, composite) = (&draws[0], &draws[1]);

        assert_eq!(gather.program, PostProgram::SsgiGather);
        assert_eq!(gather.target, MockTexture::Target(0));
        assert_eq!(gather.state, PostTargetState::Pass);
        assert_eq!(gather.load, PostLoadOp::DontCare);
        assert_eq!(gather.timing, PostTiming::First(PassId::Ssgi));
        assert_eq!(gather.binds[0].0, MockTexture::External(1));
        assert_eq!(gather.binds[1].0, MockTexture::External(2));

        assert_eq!(composite.program, PostProgram::SsgiComposite);
        assert_eq!(composite.target, MockTexture::External(1));
        assert_eq!(composite.state, PostTargetState::Graph);
        assert_eq!(composite.load, PostLoadOp::Load);
        assert_eq!(composite.timing, PostTiming::Last(PassId::Ssgi));
        assert_eq!(composite.binds[0].0, MockTexture::Target(0));
        assert_eq!(composite.binds[1].0, MockTexture::External(2));

        for d in [gather, composite] {
            assert_eq!(d.constants, bytemuck::bytes_of(&p));
            assert!(d.binds.iter().all(|b| b.1 == PostSampler::LinearClamp));
        }
    }

    #[test]
    fn the_composite_blends_additively_and_the_gather_replaces() {
        let device = MockDevice::new();
        let p = build_pipelines(&device).expect("pipelines");
        assert_eq!(p.gather.blend, PostBlend::Replace);
        assert_eq!(p.composite.blend, PostBlend::Additive);
        assert_eq!(p.gather.format, PixelFormat::Rgba16Float);
        assert_eq!(p.composite.format, PixelFormat::Rgba16Float);
    }
}
