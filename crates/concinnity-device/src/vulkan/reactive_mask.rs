// The reactive mask on Vulkan: one R8 render-resolution image per frame in
// flight, written beside the scene by the particle and transparent passes and
// read by TAA, the upscalers and the reactive view. Like the scene spine it
// rests in SHADER_READ_ONLY_OPTIMAL and its writers' render passes move it in
// and out of the attachment layout, which is why the frame graph's barriers for
// it are left to them.
//
// A writer has three compatible render passes, one per `ReactiveWrite`; they
// differ only in the mask's load and store ops and initial layout, so its
// pipeline and framebuffers serve all three. A device without
// `independentBlend` cannot max-blend the mask beside the alpha-blended scene,
// so there the writers draw the scene alone and the mask is off for the run.

use ash::vk;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::reactive_mask::ReactiveWrite;

use super::allocator::DeviceAllocator;
use super::owned::{OwnedRenderPass, VkDevice};
use super::pipeline_desc::Blend;
use super::texture::{
    GpuImage, ImageSpec, LayoutTransition, SubresourceRange, create_image, create_image_view,
    one_shot_submit, transition_image_layout_range,
};

pub(in crate::vulkan) const REACTIVE_MASK_FORMAT: vk::Format = vk::Format::R8_UNORM;

// The clear a writer's render pass begins with, for the mask's attachment.
pub(in crate::vulkan) const REACTIVE_MASK_CLEAR: vk::ClearValue = vk::ClearValue {
    color: vk::ClearColorValue { float32: [0.0; 4] },
};

// The whole of one mask image.
const RANGE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};

// One mask per frame in flight, rested in SHADER_READ_ONLY_OPTIMAL.
pub(in crate::vulkan) fn create_reactive_masks(
    alloc: &DeviceAllocator,
    device: &VkDevice,
    (command_pool, queue): (vk::CommandPool, vk::Queue),
    (width, height): (u32, u32),
    count: usize,
) -> RenderResult<Vec<GpuImage>> {
    let mut masks = Vec::with_capacity(count);
    for _ in 0..count {
        let pooled = create_image(
            alloc,
            &ImageSpec {
                width: width.max(1),
                height: height.max(1),
                format: REACTIVE_MASK_FORMAT,
                tiling: vk::ImageTiling::OPTIMAL,
                // TRANSFER_DST for the clear a frame with no writer gives a
                // reader that must have one.
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST,
                mem_props: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                samples: vk::SampleCountFlags::TYPE_1,
            },
        )?;
        let image = pooled.image();
        one_shot_submit(device, command_pool, queue, |cmd| {
            transition_image_layout_range(
                device,
                cmd,
                image,
                LayoutTransition {
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    aspect: vk::ImageAspectFlags::COLOR,
                },
                SubresourceRange {
                    base_layer: 0,
                    layer_count: 1,
                    base_mip: 0,
                    mip_count: 1,
                },
            );
        })?;
        let view = create_image_view(
            device,
            image,
            REACTIVE_MASK_FORMAT,
            vk::ImageAspectFlags::COLOR,
        )?;
        masks.push(GpuImage::from_pooled(pooled, view));
    }
    Ok(masks)
}

// The mask's attachment in a writer render pass that treats it as `write`
// says. Every variant ends sampled, where the mask rests.
pub(in crate::vulkan) fn attachment(write: ReactiveWrite) -> vk::AttachmentDescription {
    let (load, store, initial) = match write {
        ReactiveWrite::Unstored => (
            vk::AttachmentLoadOp::DONT_CARE,
            vk::AttachmentStoreOp::DONT_CARE,
            vk::ImageLayout::UNDEFINED,
        ),
        ReactiveWrite::Clear => (
            vk::AttachmentLoadOp::CLEAR,
            vk::AttachmentStoreOp::STORE,
            vk::ImageLayout::UNDEFINED,
        ),
        ReactiveWrite::Load => (
            vk::AttachmentLoadOp::LOAD,
            vk::AttachmentStoreOp::STORE,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ),
    };
    vk::AttachmentDescription::default()
        .format(REACTIVE_MASK_FORMAT)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(load)
        .store_op(store)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(initial)
        .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
}

// The color targets a writer pass draws.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(in crate::vulkan) enum WriterTargets {
    // The scene, alpha-blended, and the mask max-blended beside it.
    SceneAndMask,
    // The scene alone: the device cannot blend two targets differently.
    SceneOnly,
}

impl WriterTargets {
    pub(in crate::vulkan) fn for_device(independent_blend: bool) -> Self {
        if independent_blend {
            Self::SceneAndMask
        } else {
            Self::SceneOnly
        }
    }

    // Whether the writers carry the mask, and so whether it can be live.
    pub(in crate::vulkan) fn carries_mask(self) -> bool {
        self == Self::SceneAndMask
    }

    // The writer pipeline's per-target blend, scene first.
    pub(in crate::vulkan) fn blends(self) -> &'static [Blend] {
        match self {
            Self::SceneAndMask => &[Blend::AlphaOver, Blend::Max],
            Self::SceneOnly => &[Blend::AlphaOver],
        }
    }

    // The writer's color attachments, scene first: the mask's attachment (as
    // `write` treats it) follows the scene's when carried.
    pub(in crate::vulkan) fn attachments(
        self,
        scene: vk::AttachmentDescription,
        write: Option<ReactiveWrite>,
    ) -> Vec<vk::AttachmentDescription> {
        let mut attachments = vec![scene];
        if let Some(write) = write.filter(|_| self.carries_mask()) {
            attachments.push(attachment(write));
        }
        attachments
    }

    // A writer framebuffer's views, scene first.
    pub(in crate::vulkan) fn views(
        self,
        scene: vk::ImageView,
        mask: vk::ImageView,
    ) -> Vec<vk::ImageView> {
        if self.carries_mask() {
            vec![scene, mask]
        } else {
            vec![scene]
        }
    }
}

// A writer's render passes: one per way it can treat the mask, or a single
// scene-only pass when it carries none.
pub(in crate::vulkan) enum WriterRenderPasses {
    SceneAndMask {
        unstored: OwnedRenderPass,
        clear: OwnedRenderPass,
        load: OwnedRenderPass,
    },
    SceneOnly(OwnedRenderPass),
}

impl WriterRenderPasses {
    // Build them from `create`, which makes the writer's render pass with the
    // mask attachment it is handed, or with none.
    pub(in crate::vulkan) fn new(
        targets: WriterTargets,
        mut create: impl FnMut(Option<ReactiveWrite>) -> RenderResult<OwnedRenderPass>,
    ) -> RenderResult<Self> {
        Ok(match targets {
            WriterTargets::SceneAndMask => Self::SceneAndMask {
                unstored: create(Some(ReactiveWrite::Unstored))?,
                clear: create(Some(ReactiveWrite::Clear))?,
                load: create(Some(ReactiveWrite::Load))?,
            },
            WriterTargets::SceneOnly => Self::SceneOnly(create(None)?),
        })
    }

    pub(in crate::vulkan) fn get(&self, write: ReactiveWrite) -> vk::RenderPass {
        match (self, write) {
            (Self::SceneAndMask { unstored, .. }, ReactiveWrite::Unstored) => unstored.handle(),
            (Self::SceneAndMask { clear, .. }, ReactiveWrite::Clear) => clear.handle(),
            (Self::SceneAndMask { load, .. }, ReactiveWrite::Load) => load.handle(),
            (Self::SceneOnly(pass), _) => pass.handle(),
        }
    }

    // The pass pipelines and framebuffers are built against; any of them would
    // do, since they are compatible.
    pub(in crate::vulkan) fn compatible(&self) -> vk::RenderPass {
        self.get(ReactiveWrite::Load)
    }
}

// Clear `mask` outside any render pass: for a writer that draws nothing on a
// frame it was to clear, and for a reader that must have a mask on a frame with
// no writer. From the resting layout to a transfer and back, ordered after any
// earlier read and before any later read or writer pass, whose incoming
// dependency waits on the attachment stage.
pub(in crate::vulkan) fn clear_outside_pass(
    device: &VkDevice,
    cmd: vk::CommandBuffer,
    mask: vk::Image,
) {
    let reads = vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER;
    let later = reads | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
    let to_transfer = vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_READ)
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(mask)
        .subresource_range(RANGE);
    let back = vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(
            vk::AccessFlags::SHADER_READ
                | vk::AccessFlags::COLOR_ATTACHMENT_READ
                | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        )
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(mask)
        .subresource_range(RANGE);
    // SAFETY: `cmd` is recording outside a render pass, `mask` is a live mask image resting in
    // SHADER_READ_ONLY_OPTIMAL, and every slice these commands name is live for the call.
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            reads,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            std::slice::from_ref(&to_transfer),
        );
        device.cmd_clear_color_image(
            cmd,
            mask,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &vk::ClearColorValue { float32: [0.0; 4] },
            std::slice::from_ref(&RANGE),
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            later,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            std::slice::from_ref(&back),
        );
    }
}

impl super::context::VkContext {
    // Clear frame slot `frame_idx`'s mask outside any render pass.
    pub(in crate::vulkan) fn clear_reactive_mask(&self, cmd: vk::CommandBuffer, frame_idx: usize) {
        if let Some(mask) = self.targets.reactive_mask_images.get(frame_idx) {
            clear_outside_pass(&self.hw.device, cmd, mask.image);
        }
    }

    // Frame slot `frame_idx`'s mask.
    pub(in crate::vulkan) fn reactive_mask(&self, frame_idx: usize) -> Option<&GpuImage> {
        self.targets.reactive_mask_images.get(frame_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A device without per-target blending gets scene-only writers with the
    // mask off; one with it gets the mask max-blended beside the scene.
    #[test]
    fn the_device_decides_whether_the_writers_carry_the_mask() {
        let carried = WriterTargets::for_device(true);
        assert!(carried.carries_mask());
        assert_eq!(carried.blends(), [Blend::AlphaOver, Blend::Max]);
        let scene = vk::AttachmentDescription::default().format(vk::Format::R16G16B16A16_SFLOAT);
        assert_eq!(
            carried.attachments(scene, Some(ReactiveWrite::Load)).len(),
            2
        );
        assert_eq!(
            carried
                .views(vk::ImageView::null(), vk::ImageView::null())
                .len(),
            2
        );

        let alone = WriterTargets::for_device(false);
        assert!(!alone.carries_mask());
        assert_eq!(alone.blends(), [Blend::AlphaOver]);
        assert_eq!(alone.attachments(scene, None).len(), 1);
        assert_eq!(alone.attachments(scene, Some(ReactiveWrite::Load)).len(), 1);
        assert_eq!(
            alone
                .views(vk::ImageView::null(), vk::ImageView::null())
                .len(),
            1
        );
    }

    // The three variants differ only in what render-pass compatibility ignores,
    // so one pipeline and one framebuffer serve them all; each ends sampled.
    #[test]
    fn the_writer_attachments_are_compatible_and_end_sampled() {
        let all = [
            attachment(ReactiveWrite::Unstored),
            attachment(ReactiveWrite::Clear),
            attachment(ReactiveWrite::Load),
        ];
        for a in all {
            assert_eq!(a.format, REACTIVE_MASK_FORMAT);
            assert_eq!(a.samples, vk::SampleCountFlags::TYPE_1);
            assert_eq!(a.final_layout, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        }
        assert_eq!(all[0].store_op, vk::AttachmentStoreOp::DONT_CARE);
        assert_eq!(all[1].load_op, vk::AttachmentLoadOp::CLEAR);
        assert_eq!(all[2].load_op, vk::AttachmentLoadOp::LOAD);
        assert_eq!(
            all[2].initial_layout,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        );
    }
}
