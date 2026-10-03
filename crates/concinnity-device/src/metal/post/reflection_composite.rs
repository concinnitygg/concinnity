//! Metal's share of the reflection composite, which is where its inputs come
//! from this frame. The blur and the composite -- their pipelines, their
//! targets and both draws -- are written once in
//! `concinnity_core::render::post::reflection_composite` and reach Metal
//! through `MtlPostDevice`.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::render::error::RenderResult;
use concinnity_core::render::post::device::PostExtent;
use concinnity_core::render::post::reflection_composite::{
    ReflectionCompositeInputs, ReflectionCompositePass,
};
use objc2::runtime::ProtocolObject;

use crate::metal::context::MtlContext;
use crate::metal::post::post_device::{MtlPostDevice, MtlPostPipeline, MtlPostTarget};

// The shared composite, holding Metal's own pipeline and target handles.
pub(crate) type MtlReflectionCompositePass =
    ReflectionCompositePass<MtlPostPipeline, MtlPostTarget>;

// Build the composite at `render` resolution with its blur divided by
// `blur_scale`.
pub(crate) fn build_reflection_composite(
    device: &MtlPostDevice,
    blur_scale: u32,
    render: (u32, u32),
) -> RenderResult<MtlReflectionCompositePass> {
    ReflectionCompositePass::new(
        device,
        blur_scale,
        PostExtent {
            width: render.0,
            height: render.1,
        },
    )
}

impl MtlContext {
    // Blur the reflection target by surface roughness and composite it over
    // `hdr_resolve` into the composite's output. Shared by the SSR and
    // RT-reflection resolves: both write the reflection target first, then call
    // this. A no-op (leaves the output untouched) when the composite or the
    // G-buffer is absent, which only happens when no reflection path is active.
    pub(in crate::metal) fn encode_reflection_composite(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
    ) -> RenderResult<()> {
        let (Some(pass), Some(reflection), Some(normal_depth), Some(roughness)) = (
            &self.ssr.composite,
            &self.ssr.reflection,
            self.gbuffer_normal_depth(),
            self.gbuffer_roughness(),
        ) else {
            return Ok(());
        };
        pass.encode(
            &self.post_device(),
            cmd_buf,
            ReflectionCompositeInputs {
                reflection: reflection.as_ref(),
                scene: self.targets.hdr.hdr_resolve.as_ref(),
                normal_depth,
                roughness,
            },
        )
    }
}
