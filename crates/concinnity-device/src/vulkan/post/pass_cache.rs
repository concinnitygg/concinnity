// src/vulkan/post/pass_cache.rs
//
// The render passes and framebuffers a shared fullscreen post pass needs, keyed
// so a pass never builds its own. Vulkan is the only backend where a draw's
// target has to be named by an object built ahead of it (this device has no
// dynamic rendering), which is most of why its post directory was the largest of
// the three; caching that object here is what lets the passes above the seam
// stop owning one each.
//
// A render pass is compatible with any target of the same format and load
// action, so it keys on those two alone; a framebuffer binds one image view, so
// it keys on the view. Both caches are append-only for the life of a swapchain,
// which bounds them at one entry per (format, load) and one per live target.

use std::sync::Mutex;

use ash::vk;

use concinnity_core::render::post::device::PostLoadOp;
use concinnity_core::render::render_graph::PixelFormat;

use crate::vulkan::owned::{OwnedFramebuffer, OwnedRenderPass, VkDevice};
use crate::vulkan::transient_pool::image_format;

// A cached render pass's key: everything a pass's compatibility depends on.
#[derive(Copy, Clone, PartialEq, Eq)]
struct PassKey {
    format: PixelFormat,
    load: PostLoadOp,
}

// The single color attachment a fullscreen post pass writes. It ends
// shader-readable because every consumer of a post target samples it, and it
// begins `UNDEFINED` under a discarding load because nothing the target already
// holds survives a full-coverage draw.
//
// The `EXTERNAL` dependency orders the subpass after the writes it samples *and*
// after the previous frame's read of the slot it is about to overwrite, which is
// what lets a temporal pass ping-pong without a hand-written cross-frame
// barrier.
fn create_render_pass(
    device: &VkDevice,
    format: PixelFormat,
    load: PostLoadOp,
) -> Result<OwnedRenderPass, String> {
    let (load_op, initial) = match load {
        PostLoadOp::DontCare => (vk::AttachmentLoadOp::DONT_CARE, vk::ImageLayout::UNDEFINED),
        PostLoadOp::Load => (
            vk::AttachmentLoadOp::LOAD,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ),
    };
    let attachment = vk::AttachmentDescription::default()
        .format(image_format(format))
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(load_op)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(initial)
        .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
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
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::SHADER_READ);
    let info = vk::RenderPassCreateInfo::default()
        .attachments(std::slice::from_ref(&attachment))
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(std::slice::from_ref(&dependency));
    device
        .create_render_pass(&info)
        .map_err(|e| format!("post render pass: {e}"))
}

// The render passes and framebuffers every shared post pass draws through.
//
// `Mutex` rather than `RefCell`: the graph executor records passes on worker
// threads that all hold `&self`, so two passes can reach a cache at once. The
// locks are taken once per pass per frame, behind a hit on all but the first
// frame.
#[derive(Default)]
pub(in crate::vulkan) struct PostPassCache {
    passes: Mutex<Vec<(PassKey, OwnedRenderPass)>>,
    framebuffers: Mutex<Vec<(vk::ImageView, vk::RenderPass, OwnedFramebuffer)>>,
}

impl PostPassCache {
    /// An empty cache.
    pub(in crate::vulkan) fn new() -> Self {
        Self::default()
    }

    // The render pass for a target of `format` under `load`, built on first ask.
    pub(in crate::vulkan) fn render_pass(
        &self,
        device: &VkDevice,
        format: PixelFormat,
        load: PostLoadOp,
    ) -> Result<vk::RenderPass, String> {
        let key = PassKey { format, load };
        let mut passes = self
            .passes
            .lock()
            .map_err(|_| "post render-pass cache poisoned".to_string())?;
        if let Some((_, rp)) = passes.iter().find(|(k, _)| *k == key) {
            return Ok(rp.handle());
        }
        let rp = create_render_pass(device, format, load)?;
        let handle = rp.handle();
        passes.push((key, rp));
        Ok(handle)
    }

    // The framebuffer binding `view` to `render_pass` at `extent`, built on first
    // ask. A resize destroys the views it keys on, so `forget_views` clears the
    // entries before the new targets are created.
    pub(in crate::vulkan) fn framebuffer(
        &self,
        device: &VkDevice,
        render_pass: vk::RenderPass,
        view: vk::ImageView,
        extent: vk::Extent2D,
    ) -> Result<vk::Framebuffer, String> {
        let mut fbs = self
            .framebuffers
            .lock()
            .map_err(|_| "post framebuffer cache poisoned".to_string())?;
        if let Some((_, _, fb)) = fbs
            .iter()
            .find(|(v, rp, _)| *v == view && *rp == render_pass)
        {
            return Ok(fb.handle());
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
            .map_err(|e| format!("post framebuffer: {e}"))?;
        let handle = fb.handle();
        fbs.push((view, render_pass, fb));
        Ok(handle)
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
