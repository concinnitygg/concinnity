//! The render passes and framebuffers a shared fullscreen post pass needs, keyed
//! so a pass never builds its own. Vulkan is the only backend where a draw's
//! target has to be named by an object built ahead of it (this device has no
//! dynamic rendering), which is most of why its post directory was the largest of
//! the three; caching that object here is what lets the passes above the seam
//! stop owning one each.
//!
//! A render pass is compatible with any target of the same format, so it keys on
//! that, the load action and where the attachment's layout rests; a framebuffer
//! binds one image view, so it keys on the view. Both caches are append-only for
//! the life of a swapchain, which bounds them at one entry per key and one per
//! live target.

use ash::vk;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::post::device::PostLoadOp;
use concinnity_engine::render::render_graph::PixelFormat;
use std::sync::{Arc, Mutex};

use crate::vulkan::owned::{OwnedFramebuffer, OwnedRenderPass, VkDevice};
use crate::vulkan::transient_pool::image_format;

// Where an attachment's layout rests between passes, which decides the layouts
// its render pass opens and closes it in.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(in crate::vulkan) enum AttachmentRest {
    // Shader-readable, with the render pass moving it in and back out. Every
    // post target and the HDR scene rest here.
    Sampled,
    // In the color-attachment layout, with the graph executor moving it in and
    // out around the pass, so the render pass leaves it where it found it.
    Graph,
}

// A cached render pass's key: its attachment's format, load and layouts.
#[derive(Copy, Clone, PartialEq, Eq)]
struct PassKey {
    format: PixelFormat,
    load: PostLoadOp,
    rest: AttachmentRest,
}

// How a pass under `load` opens an attachment resting at `rest`. A discarding
// load of a sampled attachment begins `UNDEFINED`; a preserving one begins
// shader-readable. A graph-driven one arrives as an attachment either way.
fn attachment_open(
    load: PostLoadOp,
    rest: AttachmentRest,
) -> (vk::AttachmentLoadOp, vk::ImageLayout) {
    let op = match load {
        PostLoadOp::DontCare => vk::AttachmentLoadOp::DONT_CARE,
        PostLoadOp::Load => vk::AttachmentLoadOp::LOAD,
    };
    let layout = match (rest, load) {
        (AttachmentRest::Graph, _) => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        (AttachmentRest::Sampled, PostLoadOp::DontCare) => vk::ImageLayout::UNDEFINED,
        (AttachmentRest::Sampled, PostLoadOp::Load) => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    };
    (op, layout)
}

// The layout a pass leaves an attachment resting at `rest` in.
fn attachment_close(rest: AttachmentRest) -> vk::ImageLayout {
    match rest {
        AttachmentRest::Sampled => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        AttachmentRest::Graph => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }
}

// The subpass's attachment accesses, the same for every load op so pipelines
// stay compatible across them. The read is the load itself.
fn attachment_access() -> vk::AccessFlags {
    vk::AccessFlags::COLOR_ATTACHMENT_WRITE
        | vk::AccessFlags::COLOR_ATTACHMENT_READ
        | vk::AccessFlags::SHADER_READ
}

// The color attachment a fullscreen post pass writes, ending shader-readable.
// The `EXTERNAL` dependency also orders after last frame's read of this slot,
// so temporal passes ping-pong without a cross-frame barrier.
fn create_render_pass(device: &VkDevice, key: PassKey) -> RenderResult<OwnedRenderPass> {
    let PassKey { format, load, rest } = key;
    let (load_op, initial) = attachment_open(load, rest);
    let attachment = vk::AttachmentDescription::default()
        .format(image_format(format))
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(load_op)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(initial)
        .final_layout(attachment_close(rest));
    let color_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_ref));
    let dependency = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::FRAGMENT_SHADER,
        )
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::FRAGMENT_SHADER,
        )
        .dst_access_mask(attachment_access());
    let info = vk::RenderPassCreateInfo::default()
        .attachments(std::slice::from_ref(&attachment))
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(std::slice::from_ref(&dependency));
    device
        .create_render_pass(&info)
        .map_err(|e| crate::vulkan::error::map_vk_result(e, "post render pass"))
}

// The render passes and framebuffers every post pass draws through, behind a
// `Mutex` since worker-recorded passes share `&self`.
#[derive(Default)]
pub(in crate::vulkan) struct PostPassCache {
    passes: Mutex<Vec<(PassKey, Arc<OwnedRenderPass>)>>,
    framebuffers: Mutex<Vec<(vk::ImageView, vk::RenderPass, Arc<OwnedFramebuffer>)>>,
}

impl PostPassCache {
    /// An empty cache.
    pub(in crate::vulkan) fn new() -> Self {
        Self::default()
    }

    // The render pass for a target of `format` under `load` resting at `rest`,
    // built on first ask.
    pub(in crate::vulkan) fn render_pass(
        &self,
        device: &VkDevice,
        format: PixelFormat,
        load: PostLoadOp,
        rest: AttachmentRest,
    ) -> RenderResult<Arc<OwnedRenderPass>> {
        let key = PassKey { format, load, rest };
        let mut passes = self
            .passes
            .lock()
            .map_err(|_| RenderError::Other("post render-pass cache poisoned".to_string()))?;
        if let Some((_, rp)) = passes.iter().find(|(k, _)| *k == key) {
            return Ok(Arc::clone(rp));
        }
        let rp = Arc::new(create_render_pass(device, key)?);
        passes.push((key, Arc::clone(&rp)));
        Ok(rp)
    }

    // The framebuffer binding `view` to `render_pass` at `extent`, built on first
    // ask. A resize destroys the views it keys on, so `forget_views` clears the
    // entries before the new targets are created.
    pub(in crate::vulkan) fn framebuffer(
        &self,
        device: &VkDevice,
        render_pass: &OwnedRenderPass,
        view: vk::ImageView,
        extent: vk::Extent2D,
    ) -> RenderResult<Arc<OwnedFramebuffer>> {
        let render_pass = render_pass.handle();
        let mut fbs = self
            .framebuffers
            .lock()
            .map_err(|_| RenderError::Other("post framebuffer cache poisoned".to_string()))?;
        if let Some((_, _, fb)) = fbs
            .iter()
            .find(|(v, rp, _)| *v == view && *rp == render_pass)
        {
            return Ok(Arc::clone(fb));
        }
        let fb = device
            .create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(std::slice::from_ref(&view))
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1),
            )
            .map_err(|e| crate::vulkan::error::map_vk_result(e, "post framebuffer"))?;
        let fb = Arc::new(fb);
        fbs.push((view, render_pass, Arc::clone(&fb)));
        Ok(fb)
    }

    // Drop every cached framebuffer. Called before a post pass recreates its
    // targets: a framebuffer outlives neither the view it binds nor the extent
    // it was sized at. The caller has already idled the device.
    pub(in crate::vulkan) fn forget_views(&self) {
        if let Ok(mut fbs) = self.framebuffers.lock() {
            fbs.clear();
        }
    }

    // Drop everything. The caller has already idled the device.
    pub(in crate::vulkan) fn destroy(&self) {
        self.forget_views();
        if let Ok(mut passes) = self.passes.lock() {
            passes.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_preserving_load_begins_readable_and_a_discarding_one_undefined() {
        assert_eq!(
            attachment_open(PostLoadOp::Load, AttachmentRest::Sampled),
            (
                vk::AttachmentLoadOp::LOAD,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            )
        );
        assert_eq!(
            attachment_open(PostLoadOp::DontCare, AttachmentRest::Sampled),
            (vk::AttachmentLoadOp::DONT_CARE, vk::ImageLayout::UNDEFINED)
        );
        assert_eq!(
            attachment_close(AttachmentRest::Sampled),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        );
    }

    #[test]
    fn a_graph_driven_attachment_stays_in_the_attachment_layout() {
        // The executor moves `ao_output` into the attachment layout ahead of the
        // pass and out of it before the forward pass samples it; a render pass
        // that also moved it would leave the executor's next barrier naming a
        // layout the image is no longer in.
        for load in [PostLoadOp::DontCare, PostLoadOp::Load] {
            assert_eq!(
                attachment_open(load, AttachmentRest::Graph).1,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
            );
        }
        assert_eq!(
            attachment_close(AttachmentRest::Graph),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        );
    }

    #[test]
    fn the_dependency_declares_the_read_a_load_performs() {
        // The read must be declared so the layout transition before the SSGI
        // composite's load is ordered; it does not vary with the load op.
        let access = attachment_access();
        assert!(access.contains(vk::AccessFlags::COLOR_ATTACHMENT_READ));
        assert!(access.contains(vk::AccessFlags::COLOR_ATTACHMENT_WRITE));
        assert!(access.contains(vk::AccessFlags::SHADER_READ));
    }
}
