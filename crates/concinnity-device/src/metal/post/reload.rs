//! Where the engine-owned post passes live on the Metal context, for the
//! shared post-pass rebuild.

use concinnity_engine::render::post::reload::{PostPass, PostPipelines};

use super::post_device::{MtlPostDevice, PostPipeline};
use crate::metal::context::MtlContext;

impl MtlContext {
    // Whether `pass` is live.
    pub(in crate::metal) fn has_post_pass(&self, pass: PostPass) -> bool {
        match pass {
            PostPass::Taa => self.taa.pass.is_some(),
            PostPass::Bloom => self.bloom.is_some(),
            PostPass::Ssao => self.ssao.pass.is_some(),
            PostPass::Ssr => self.ssr.resolve.is_some(),
            PostPass::Ssgi => self.ssgi.pass.is_some(),
            PostPass::Disocclusion => self.disocclusion.is_some(),
            PostPass::ReflectionComposite => self.ssr.composite.is_some(),
        }
    }

    // The post-pass device pipelines are built through, reading the sources
    // `hot_reload` selects: nothing encodes, so it carries no probe set or
    // timing.
    pub(in crate::metal) fn post_pipeline_device(&self, hot_reload: bool) -> MtlPostDevice<'_> {
        MtlPostDevice {
            device: &self.hw.device,
            sampler: &self.composite.sampler,
            cube_sampler: &self.scene.cube_sampler,
            probes: None,
            timing: None,
            hot_reload,
        }
    }

    // Put one live pass's rebuilt pipelines in place.
    pub(in crate::metal) fn install_post_pipelines(
        &mut self,
        pipelines: PostPipelines<PostPipeline>,
    ) {
        match pipelines {
            PostPipelines::Taa(p) => {
                if let Some(taa) = self.taa.pass.as_mut() {
                    taa.swap_pipelines(p);
                }
            }
            PostPipelines::Bloom(p) => {
                if let Some(bloom) = self.bloom.as_mut() {
                    bloom.swap_pipelines(p);
                }
            }
            PostPipelines::Ssao(p) => {
                if let Some(ssao) = self.ssao.pass.as_mut() {
                    ssao.swap_pipelines(p);
                }
            }
            PostPipelines::Ssr(p) => {
                if let Some(resolve) = self.ssr.resolve.as_mut() {
                    resolve.swap_pipeline(p);
                }
            }
            PostPipelines::Ssgi(p) => {
                if let Some(ssgi) = self.ssgi.pass.as_mut() {
                    ssgi.swap_pipelines(p);
                }
            }
            PostPipelines::Disocclusion(p) => {
                if let Some(pass) = self.disocclusion.as_mut() {
                    pass.swap_pipelines(p);
                }
            }
            PostPipelines::ReflectionComposite(p) => {
                if let Some(pass) = self.ssr.composite.as_mut() {
                    pass.swap_pipelines(p);
                }
            }
        }
    }
}
