//! Bloom: the shared chain's pipelines and the octaves below the pool's top
//! one, at the output (swapchain) resolution.

use ash::vk;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::post::bloom::BloomPass;

use super::InitGpu;
use super::effects::shared_post_device;
use crate::vulkan::context::{VkDescriptors, VkSceneAssets};
use crate::vulkan::post::PostSupport;
use crate::vulkan::post::bloom::VkBloomPass;
use crate::vulkan::post::post_device::post_extent;

pub(super) struct BloomInputs<'a> {
    pub(super) extent: vk::Extent2D,
    pub(super) post: &'a PostSupport,
    pub(super) scene: &'a VkSceneAssets,
    pub(super) descriptors: &'a VkDescriptors,
}

pub(super) fn build_bloom(gpu: &InitGpu<'_>, inputs: BloomInputs<'_>) -> RenderResult<VkBloomPass> {
    let device = shared_post_device(gpu, inputs.post, inputs.scene, inputs.descriptors);
    BloomPass::new(&device, post_extent(inputs.extent))
}
