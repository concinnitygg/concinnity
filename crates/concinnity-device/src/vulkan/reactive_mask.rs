// The reactive mask: one R8 render-resolution image per frame in flight,
// written beside the scene by the particle and transparent passes and read by
// TAA, the upscalers and the reactive view. It rests SHADER_READ_ONLY and its
// writers move it around their draws, so the graph leaves its barriers to them.
// Without `independentBlend` the mask is off for the run.

use ash::vk;
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::temporal::reactive_mask::ReactiveWrite;

use super::pipeline_desc::Blend;
use super::rendering::{
    Attachment, ColorAttachments, ImageUse, Load, PassFormats, RenderTargets, Transition,
};
use super::texture::{
    GpuImage, GpuUploadContext, ImageSpec, LayoutTransition, SubresourceRange, create_image,
    create_image_view, one_shot_submit, transition_image_layout_range,
};
use crate::vulkan::record::Recorder;

pub(in crate::vulkan) const REACTIVE_MASK_FORMAT: vk::Format = vk::Format::R8_UNORM;

// The clear a writer's pass begins with, for the mask's attachment.
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
    gpu: GpuUploadContext<'_>,
    extent: vk::Extent2D,
    count: usize,
) -> RenderResult<Vec<GpuImage>> {
    let GpuUploadContext {
        alloc,
        device,
        command_pool,
        queue,
    } = gpu;
    let mut masks = Vec::with_capacity(count);
    for _ in 0..count {
        let pooled = create_image(
            alloc,
            &ImageSpec {
                width: extent.width.max(1),
                height: extent.height.max(1),
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
        one_shot_submit(device, command_pool, queue, |rec| {
            transition_image_layout_range(
                rec,
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

    // The attachment formats a writer pipeline renders to, scene first.
    pub(in crate::vulkan) fn formats(self, scene: vk::Format) -> PassFormats {
        if self.carries_mask() {
            PassFormats::colors([scene, REACTIVE_MASK_FORMAT])
        } else {
            PassFormats::color(scene)
        }
    }

    // Render a writer pass over the sampled `scene` and, when carried, `mask`,
    // treating the mask as `write` says. Both end sampled, where they rest.
    pub(in crate::vulkan) fn render(
        self,
        rec: &Recorder<'_>,
        pass: WriterPass<'_>,
        draw: impl FnOnce(&Recorder<'_>),
    ) {
        let WriterPass {
            scene,
            scene_from,
            mask,
            write,
            extent,
        } = pass;
        let mask = mask.filter(|_| self.carries_mask());
        let (mask_load, mask_store, mask_discard) = mask_write(write);
        let color = |image: &GpuImage, from, to| Transition::color(image.image, from, to);
        let scene_in = color(scene, scene_from, ImageUse::ColorAttachment);
        let scene_out = color(scene, ImageUse::ColorAttachment, ImageUse::Sampled);
        let scene_attachment = Attachment::loaded(scene.view);
        let colors = if let Some(mask) = mask {
            let mask_in = color(mask, ImageUse::Sampled, ImageUse::ColorAttachment)
                .discarding_if(mask_discard);
            rec.transition(&[scene_in, mask_in]);
            ColorAttachments::new([
                scene_attachment,
                Attachment {
                    view: mask.view,
                    load: mask_load,
                    store: mask_store,
                    resolve: None,
                },
            ])
        } else {
            rec.transition(&[scene_in]);
            ColorAttachments::new([scene_attachment])
        };
        rec.begin_rendering(&RenderTargets {
            area: vk::Rect2D::default().extent(extent),
            colors,
            depth: None,
        });
        draw(rec);
        rec.end_rendering();
        match mask {
            Some(mask) => rec.transition(&[
                scene_out,
                color(mask, ImageUse::ColorAttachment, ImageUse::Sampled),
            ]),
            None => rec.transition(&[scene_out]),
        }
    }
}

// How a writer pass treats the mask as `write` says: what its attachment
// loads, whether it stores, and whether it discards what the mask held.
fn mask_write(write: ReactiveWrite) -> (Load, bool, bool) {
    match write {
        ReactiveWrite::Unstored => (Load::Discard, false, true),
        ReactiveWrite::Clear => (Load::Clear(REACTIVE_MASK_CLEAR), true, true),
        ReactiveWrite::Load => (Load::Keep, true, false),
    }
}

// What one writer pass renders over: the scene, the frame's reactive mask,
// and how the frame treats it.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct WriterPass<'a> {
    pub scene: &'a GpuImage,
    // The scene's last use before the pass.
    pub scene_from: ImageUse,
    pub mask: Option<&'a GpuImage>,
    pub write: ReactiveWrite,
    pub extent: vk::Extent2D,
}

// Clear `mask` outside any rendering scope, for a writer that draws nothing on a
// clearing frame or a reader with no writer this frame.
pub(in crate::vulkan) fn clear_outside_pass(rec: &Recorder<'_>, mask: vk::Image) {
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
    rec.pipeline_barrier(
        reads,
        vk::PipelineStageFlags::TRANSFER,
        &[],
        &[],
        std::slice::from_ref(&to_transfer),
    );
    rec.clear_color_image(
        mask,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &vk::ClearColorValue { float32: [0.0; 4] },
        RANGE,
    );
    rec.pipeline_barrier(
        vk::PipelineStageFlags::TRANSFER,
        later,
        &[],
        &[],
        std::slice::from_ref(&back),
    );
}

impl super::context::VkContext {
    // Clear frame slot `frame_idx`'s mask outside any rendering scope.
    pub(in crate::vulkan) fn clear_reactive_mask(&self, rec: &Recorder<'_>, frame_idx: usize) {
        if let Some(mask) = self.targets.reactive_mask_images.get(frame_idx) {
            clear_outside_pass(rec, mask.image);
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
        let scene = vk::Format::R16G16B16A16_SFLOAT;
        let carried = WriterTargets::for_device(true);
        assert!(carried.carries_mask());
        assert_eq!(carried.blends(), [Blend::AlphaOver, Blend::Max]);
        assert_eq!(
            carried.formats(scene).color_formats(),
            [scene, REACTIVE_MASK_FORMAT]
        );

        let alone = WriterTargets::for_device(false);
        assert!(!alone.carries_mask());
        assert_eq!(alone.blends(), [Blend::AlphaOver]);
        assert_eq!(alone.formats(scene).color_formats(), [scene]);
    }

    // Only a loading write keeps the mask's contents; the unstored write also
    // discards what it drew.
    #[test]
    fn each_write_loads_stores_and_opens_as_it_treats_the_mask() {
        let (load, store, discard) = mask_write(ReactiveWrite::Unstored);
        assert!(matches!(load, Load::Discard) && !store && discard);
        let (load, store, discard) = mask_write(ReactiveWrite::Clear);
        assert!(matches!(load, Load::Clear(_)) && store && discard);
        let (load, store, discard) = mask_write(ReactiveWrite::Load);
        assert!(matches!(load, Load::Keep) && store && !discard);
    }
}
