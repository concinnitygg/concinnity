//! Where the engine-owned post passes live on the Vulkan context, for the
//! shared post-pass rebuild.

use concinnity_engine::render::post::reload::{PostPass, PostPipelines};

use super::post_device::PostPipeline;
use crate::vulkan::context::VkContext;

impl VkContext {
    // Whether `pass` is live.
    pub(in crate::vulkan) fn has_post_pass(&self, pass: PostPass) -> bool {
        match pass {
            PostPass::Taa => self.taa.is_some(),
            PostPass::Bloom => self.bloom.is_some(),
            PostPass::Ssao => self.ssao.is_some(),
            PostPass::Ssr => self.ssr.is_some(),
            PostPass::Ssgi => self.ssgi.is_some(),
            PostPass::Disocclusion => self.disocclusion.is_some(),
            PostPass::ReflectionComposite => self.reflection_composite.is_some(),
        }
    }

    // Put one live pass's rebuilt pipelines in place.
    pub(in crate::vulkan) fn install_post_pipelines(
        &mut self,
        pipelines: PostPipelines<PostPipeline>,
    ) {
        match pipelines {
            PostPipelines::Taa(p) => {
                if let Some(taa) = self.taa.as_mut() {
                    taa.swap_pipelines(p);
                }
            }
            PostPipelines::Bloom(p) => {
                if let Some(bloom) = self.bloom.as_mut() {
                    bloom.swap_pipelines(p);
                }
            }
            PostPipelines::Ssao(p) => {
                if let Some(ssao) = self.ssao.as_mut() {
                    ssao.swap_pipelines(p);
                }
            }
            PostPipelines::Ssr(p) => {
                if let Some(ssr) = self.ssr.as_mut() {
                    ssr.swap_pipeline(p);
                }
            }
            PostPipelines::Ssgi(p) => {
                if let Some(ssgi) = self.ssgi.as_mut() {
                    ssgi.swap_pipelines(p);
                }
            }
            PostPipelines::Disocclusion(p) => {
                if let Some(pass) = self.disocclusion.as_mut() {
                    pass.swap_pipelines(p);
                }
            }
            PostPipelines::ReflectionComposite(p) => {
                if let Some(pass) = self.reflection_composite.as_mut() {
                    pass.swap_pipelines(p);
                }
            }
        }
    }
}
