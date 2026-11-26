//! Framebuffer creation: one single-layer framebuffer over a pass's
//! attachments, or one per view with that view as its only attachment.

use ash::vk;
use concinnity_engine::render::backend::error::RenderResult;

use super::error::map_vk_result;
use super::owned::{OwnedFramebuffer, VkDevice};

// A single-layer framebuffer description over `attachments`, at least 1x1 so a
// minimized window's zero extent stays valid.
fn framebuffer_info(
    render_pass: vk::RenderPass,
    attachments: &[vk::ImageView],
    extent: vk::Extent2D,
) -> vk::FramebufferCreateInfo<'_> {
    vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .attachments(attachments)
        .width(extent.width.max(1))
        .height(extent.height.max(1))
        .layers(1)
}

impl VkDevice {
    // A framebuffer for `render_pass` over `attachments`; `label` names it in
    // the error.
    pub(in crate::vulkan) fn framebuffer(
        &self,
        render_pass: vk::RenderPass,
        attachments: &[vk::ImageView],
        extent: vk::Extent2D,
        label: &str,
    ) -> RenderResult<OwnedFramebuffer> {
        self.create_framebuffer(&framebuffer_info(render_pass, attachments, extent))
            .map_err(|e| map_vk_result(e, label))
    }

    // One framebuffer per view in `views`, each binding that view (color or
    // depth) as the pass's only attachment.
    pub(in crate::vulkan) fn single_view_framebuffers(
        &self,
        render_pass: vk::RenderPass,
        views: impl IntoIterator<Item = vk::ImageView>,
        extent: vk::Extent2D,
        label: &str,
    ) -> RenderResult<Vec<OwnedFramebuffer>> {
        views
            .into_iter()
            .map(|view| self.framebuffer(render_pass, &[view], extent, label))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_framebuffer_spans_the_extent_in_one_layer() {
        let views = [vk::ImageView::null(); 2];
        let info = framebuffer_info(
            vk::RenderPass::null(),
            &views,
            vk::Extent2D {
                width: 640,
                height: 360,
            },
        );
        assert_eq!((info.width, info.height, info.layers), (640, 360, 1));
        assert_eq!(info.attachment_count, 2);
    }

    #[test]
    fn a_zero_extent_clamps_to_one_texel() {
        let info = framebuffer_info(
            vk::RenderPass::null(),
            &[],
            vk::Extent2D {
                width: 0,
                height: 0,
            },
        );
        assert_eq!((info.width, info.height), (1, 1));
    }
}
