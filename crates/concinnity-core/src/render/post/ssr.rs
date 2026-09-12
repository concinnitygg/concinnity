//! The screen-space reflection resolve, written once for every backend.
//!
//! The resolve ray-marches each glossy pixel's reflection through the scene
//! against the G-buffer's view normal and linear depth, and writes reflected
//! radiance plus a composite weight. A ray that leaves the frame falls back to
//! the world's reflection probes, or to the prefiltered environment cube when
//! none covers the surface.
//!
//! The pass does not own the target it writes. That target belongs to the
//! reflection composite, which blurs it by roughness and blends it over the
//! scene, and the ray-traced resolve writes the same one where it runs in this
//! pass's place. The caller supplies it, and [`target_desc`] is its shape for a
//! backend that creates it through the seam.

use alloc::string::String;

use crate::gfx::render_types::SsrParams;
use crate::render::render_graph::{
    ClearValue, PassId, PixelFormat, TextureDesc, TextureSize, TextureUsage,
};

use super::device::{
    PostBind, PostBlend, PostDraw, PostLoadOp, PostPassDevice, PostSampler, PostTargetState,
    PostTiming,
};
use super::program::PostProgram;

/// The per-frame inputs one resolve draw reads and writes.
pub struct SsrInputs<'t, D: PostPassDevice + ?Sized + 't> {
    /// The reflection target: radiance in `.rgb`, composite weight in `.a`.
    pub target: D::Attachment<'t>,
    /// The lit scene the rays march through.
    pub scene: D::TextureRef<'t>,
    /// The G-buffer's view normal (`.rgb`) and linear depth (`.a`).
    pub normal_depth: D::TextureRef<'t>,
    /// The G-buffer's surface roughness.
    pub roughness: D::TextureRef<'t>,
    /// The prefiltered environment cube a missed ray falls back to.
    pub prefilter: D::TextureRef<'t>,
}

/// The reflection target's shape: a full-resolution single-sample HDR color
/// target that is both rendered to and sampled.
pub fn target_desc() -> TextureDesc {
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

/// Build the resolve pipeline on its own. What shader hot reload rebuilds and
/// hands to [`SsrPass::swap_pipeline`].
pub fn build_pipeline<D: PostPassDevice>(device: &D) -> Result<D::Pipeline, String> {
    device.create_pipeline(
        PostProgram::SsrResolve,
        target_desc().format,
        PostBlend::Replace,
    )
}

/// The resolve: its pipeline, and the one draw.
pub struct SsrPass<Pipeline> {
    pipeline: Pipeline,
}

impl<Pipeline> SsrPass<Pipeline> {
    /// Build the resolve pipeline.
    pub fn new<D>(device: &D) -> Result<Self, String>
    where
        D: PostPassDevice<Pipeline = Pipeline>,
    {
        Ok(Self {
            pipeline: build_pipeline(device)?,
        })
    }

    /// Swap in a freshly built pipeline. Driven by shader hot reload; the caller
    /// has already idled the device, so the outgoing pipeline is not in flight.
    pub fn swap_pipeline(&mut self, pipeline: Pipeline) {
        self.pipeline = pipeline;
    }

    /// Encode the resolve into `inputs.target` with this frame's `params`.
    pub fn encode<'t, D>(
        &self,
        device: &D,
        rec: &D::Recorder,
        inputs: SsrInputs<'t, D>,
        params: &SsrParams,
    ) -> Result<(), String>
    where
        D: PostPassDevice<Pipeline = Pipeline>,
    {
        let screen = |texture| PostBind {
            texture,
            sampler: PostSampler::LinearClamp,
        };
        let binds = [
            screen(inputs.scene),
            screen(inputs.normal_depth),
            screen(inputs.roughness),
            PostBind {
                texture: inputs.prefilter,
                sampler: PostSampler::LinearCube,
            },
        ];
        device.encode(
            rec,
            &PostDraw {
                target: inputs.target,
                // No other pass reads the reflection target before the
                // composite, so the graph does not declare it.
                state: PostTargetState::Pass,
                load: PostLoadOp::DontCare,
                timing: PostTiming::Whole(PassId::SsrResolve),
                pipeline: &self.pipeline,
                binds: &binds,
                constants: bytemuck::bytes_of(params),
                label: "SSR resolve",
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::post::device::{PostExtent, resolve_extent};
    use crate::render::post::mock::{MockDevice, MockTexture};

    fn params() -> SsrParams {
        SsrParams {
            intensity: 0.5,
            max_distance: 30.0,
            tan_half_fov_y: 0.6,
            aspect: 1.5,
            stride: 0.6,
            thickness: 0.3,
            prefilter_mip_count: 6.0,
            _pad: 0.0,
            inv_view: [[0.0; 4]; 4],
            sky_rot: [[0.0; 4]; 3],
        }
    }

    #[test]
    fn the_target_is_a_sampled_full_resolution_hdr_color_target() {
        let d = target_desc();
        assert_eq!(d.format, PixelFormat::Rgba16Float);
        assert!(d.usage.contains(TextureUsage::RENDER_TARGET));
        assert!(d.usage.contains(TextureUsage::SHADER_READ));
        let e = PostExtent {
            width: 1280,
            height: 720,
        };
        assert_eq!(resolve_extent(&d, e), e);
    }

    #[test]
    fn the_resolve_is_one_draw_reading_the_screen_then_the_cube() {
        let device = MockDevice::new();
        let pass = SsrPass::new(&device).expect("pass");
        let p = params();
        pass.encode(
            &device,
            &(),
            SsrInputs {
                target: MockTexture::External(9),
                scene: MockTexture::External(1),
                normal_depth: MockTexture::External(2),
                roughness: MockTexture::External(3),
                prefilter: MockTexture::External(4),
            },
            &p,
        )
        .expect("encode");
        let draws = device.draws.borrow();
        assert_eq!(draws.len(), 1);
        let d = &draws[0];
        assert_eq!(d.program, PostProgram::SsrResolve);
        assert_eq!(d.target, MockTexture::External(9));
        assert_eq!(d.state, PostTargetState::Pass);
        assert_eq!(d.load, PostLoadOp::DontCare);
        assert_eq!(d.timing, PostTiming::Whole(PassId::SsrResolve));
        assert_eq!(
            d.binds,
            [
                (MockTexture::External(1), PostSampler::LinearClamp),
                (MockTexture::External(2), PostSampler::LinearClamp),
                (MockTexture::External(3), PostSampler::LinearClamp),
                (MockTexture::External(4), PostSampler::LinearCube),
            ]
        );
        assert_eq!(d.constants, bytemuck::bytes_of(&p));
        assert_eq!(d.label, "SSR resolve");
    }

    #[test]
    fn a_swapped_pipeline_is_the_one_drawn_with() {
        let device = MockDevice::new();
        let mut pass = SsrPass::new(&device).expect("pass");
        let mut rebuilt = build_pipeline(&device).expect("rebuild");
        rebuilt.blend = PostBlend::Additive;
        pass.swap_pipeline(rebuilt);
        assert_eq!(pass.pipeline.blend, PostBlend::Additive);
    }
}
