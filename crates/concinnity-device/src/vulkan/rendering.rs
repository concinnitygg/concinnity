//! Dynamic rendering: the attachment formats a graphics pipeline renders to,
//! the attachments a pass begins rendering over, and the image layout
//! transitions a pass makes around them.

use ash::vk;

/// Most color attachments one pass renders to.
pub(in crate::vulkan) const MAX_COLORS: usize = 4;

// The attachment formats a pipeline is built against, which every pass that
// draws with it must render to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::vulkan) struct PassFormats {
    colors: [vk::Format; MAX_COLORS],
    color_count: usize,
    depth: vk::Format,
}

impl PassFormats {
    // One color attachment of `format`, no depth.
    pub(in crate::vulkan) const fn color(format: vk::Format) -> Self {
        Self::colors([format])
    }

    // These color attachments in order, no depth. More than `MAX_COLORS` is a
    // compile error.
    pub(in crate::vulkan) const fn colors<const N: usize>(formats: [vk::Format; N]) -> Self {
        const { assert!(N <= MAX_COLORS, "more color attachments than MAX_COLORS") };
        let mut colors = [vk::Format::UNDEFINED; MAX_COLORS];
        let mut i = 0;
        while i < N {
            colors[i] = formats[i];
            i += 1;
        }
        Self {
            colors,
            color_count: N,
            depth: vk::Format::UNDEFINED,
        }
    }

    // A depth attachment of `format` alone.
    pub(in crate::vulkan) const fn depth_only(format: vk::Format) -> Self {
        Self::colors([]).with_depth(format)
    }

    // These formats with a depth attachment of `format`.
    pub(in crate::vulkan) const fn with_depth(self, format: vk::Format) -> Self {
        Self {
            depth: format,
            ..self
        }
    }

    pub(in crate::vulkan) fn color_formats(&self) -> &[vk::Format] {
        &self.colors[..self.color_count]
    }

    // The `pNext` a graphics pipeline built against these formats chains.
    pub(in crate::vulkan) fn pipeline_info(&self) -> vk::PipelineRenderingCreateInfo<'_> {
        vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(self.color_formats())
            .depth_attachment_format(self.depth)
    }
}

// What a pass does with an attachment's contents as it begins.
#[derive(Clone, Copy)]
pub(in crate::vulkan) enum Load {
    Clear(vk::ClearValue),
    Keep,
    Discard,
}

impl Load {
    fn op(self) -> (vk::AttachmentLoadOp, vk::ClearValue) {
        match self {
            Self::Clear(value) => (vk::AttachmentLoadOp::CLEAR, value),
            Self::Keep => (vk::AttachmentLoadOp::LOAD, vk::ClearValue::default()),
            Self::Discard => (vk::AttachmentLoadOp::DONT_CARE, vk::ClearValue::default()),
        }
    }
}

// One attachment a pass renders to, with the single-sample view a
// multisampled color attachment resolves into.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct Attachment {
    pub view: vk::ImageView,
    pub load: Load,
    pub store: bool,
    pub resolve: Option<vk::ImageView>,
}

impl Attachment {
    // `view` cleared to `value` and kept.
    pub(in crate::vulkan) fn cleared(view: vk::ImageView, value: vk::ClearValue) -> Self {
        Self {
            view,
            load: Load::Clear(value),
            store: true,
            resolve: None,
        }
    }

    // `view` drawn over and kept.
    pub(in crate::vulkan) const fn loaded(view: vk::ImageView) -> Self {
        Self {
            view,
            load: Load::Keep,
            store: true,
            resolve: None,
        }
    }

    // The attachment info for an attachment resting in `layout`.
    pub(in crate::vulkan) fn info(
        &self,
        layout: vk::ImageLayout,
    ) -> vk::RenderingAttachmentInfo<'static> {
        let (load_op, clear) = self.load.op();
        let mut info = vk::RenderingAttachmentInfo::default()
            .image_view(self.view)
            .image_layout(layout)
            .load_op(load_op)
            .store_op(if self.store {
                vk::AttachmentStoreOp::STORE
            } else {
                vk::AttachmentStoreOp::DONT_CARE
            })
            .clear_value(clear);
        if let Some(resolve) = self.resolve {
            info = info
                .resolve_mode(vk::ResolveModeFlags::AVERAGE)
                .resolve_image_view(resolve)
                .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        }
        info
    }
}

// A pass's color attachments, in order: at most `MAX_COLORS`, which only
// the constructor's compile-time check admits.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct ColorAttachments {
    attachments: [Attachment; MAX_COLORS],
    count: usize,
}

impl ColorAttachments {
    pub(in crate::vulkan) const fn new<const N: usize>(colors: [Attachment; N]) -> Self {
        const { assert!(N <= MAX_COLORS, "more color attachments than MAX_COLORS") };
        let mut attachments = [Attachment::loaded(vk::ImageView::null()); MAX_COLORS];
        let mut i = 0;
        while i < N {
            attachments[i] = colors[i];
            i += 1;
        }
        Self {
            attachments,
            count: N,
        }
    }

    pub(in crate::vulkan) fn as_slice(&self) -> &[Attachment] {
        &self.attachments[..self.count]
    }
}

// The attachments one pass renders over, within `area`.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct RenderTargets {
    pub area: vk::Rect2D,
    pub colors: ColorAttachments,
    pub depth: Option<Attachment>,
}

// How an image is used on either side of a layout transition: its layout and
// the stages and accesses that touch it there.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::vulkan) enum ImageUse {
    // Rendered to as a color attachment.
    ColorAttachment,
    // Rendered to as a depth attachment.
    DepthAttachment,
    // Sampled by fragment and compute shaders.
    Sampled,
    // Copied from.
    TransferSrc,
    // Copied into.
    TransferDst,
    // Handed to the presentation engine.
    Present,
}

impl ImageUse {
    pub(in crate::vulkan) fn layout(self) -> vk::ImageLayout {
        self.state().0
    }

    fn state(self) -> (vk::ImageLayout, vk::PipelineStageFlags2, vk::AccessFlags2) {
        match self {
            Self::ColorAttachment => (
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            ),
            Self::DepthAttachment => (
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            ),
            Self::Sampled => (
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
            ),
            Self::TransferSrc => (
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::TRANSFER_READ,
            ),
            Self::TransferDst => (
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
            Self::Present => (
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::PipelineStageFlags2::NONE,
                vk::AccessFlags2::NONE,
            ),
        }
    }
}

// One image's move from one use to another, across every mip or one, and
// every array layer or one.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct Transition {
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
    // The one mip level moving, or every level.
    level: Option<u32>,
    // The one array layer moving, or every layer.
    layer: Option<u32>,
    // The move discards the contents, leaving from `UNDEFINED`: legal from any
    // layout, so also for an image never used before, while still waiting on
    // `from`'s stages.
    discard: bool,
    from: ImageUse,
    to: ImageUse,
}

impl Transition {
    // A color image's move.
    pub(in crate::vulkan) fn color(image: vk::Image, from: ImageUse, to: ImageUse) -> Self {
        Self {
            image,
            aspect: vk::ImageAspectFlags::COLOR,
            level: None,
            layer: None,
            discard: false,
            from,
            to,
        }
    }

    // A depth image's move.
    pub(in crate::vulkan) fn depth(image: vk::Image, from: ImageUse, to: ImageUse) -> Self {
        Self {
            aspect: vk::ImageAspectFlags::DEPTH,
            ..Self::color(image, from, to)
        }
    }

    // The same move of mip `level` alone.
    pub(in crate::vulkan) fn at_level(self, level: u32) -> Self {
        Self {
            level: Some(level),
            ..self
        }
    }

    // The same move, discarding the contents `from` left.
    pub(in crate::vulkan) fn discarding(self) -> Self {
        Self {
            discard: true,
            ..self
        }
    }

    // The same move, discarding the contents when `discard` is set.
    pub(in crate::vulkan) fn discarding_if(self, discard: bool) -> Self {
        Self { discard, ..self }
    }

    // The same move of array layer `layer` alone.
    pub(in crate::vulkan) fn at_layer(self, layer: u32) -> Self {
        Self {
            layer: Some(layer),
            ..self
        }
    }

    pub(in crate::vulkan) fn barrier(&self) -> vk::ImageMemoryBarrier2<'static> {
        let (layout, src_stage, src_access) = self.from.state();
        let old = if self.discard {
            vk::ImageLayout::UNDEFINED
        } else {
            layout
        };
        let (new, dst_stage, dst_access) = self.to.state();
        let (base, count) = self.level.map_or((0, vk::REMAINING_MIP_LEVELS), |l| (l, 1));
        let (base_layer, layers) = self
            .layer
            .map_or((0, vk::REMAINING_ARRAY_LAYERS), |l| (l, 1));
        vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access)
            .old_layout(old)
            .new_layout(new)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(self.aspect)
                    .base_mip_level(base)
                    .level_count(count)
                    .base_array_layer(base_layer)
                    .layer_count(layers),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_list_their_colors_in_order_and_an_optional_depth() {
        let formats = PassFormats::colors([vk::Format::R16G16B16A16_SFLOAT, vk::Format::R8_UNORM]);
        assert_eq!(
            formats.color_formats(),
            [vk::Format::R16G16B16A16_SFLOAT, vk::Format::R8_UNORM]
        );
        assert_eq!(formats.depth, vk::Format::UNDEFINED);
        let with_depth = formats.with_depth(vk::Format::D32_SFLOAT);
        assert_eq!(with_depth.depth, vk::Format::D32_SFLOAT);
        let shadow = PassFormats::depth_only(vk::Format::D16_UNORM);
        assert!(shadow.color_formats().is_empty());
        assert_eq!(shadow.depth, vk::Format::D16_UNORM);
    }

    #[test]
    fn color_attachments_keep_their_order_and_count() {
        use ash::vk::Handle;
        let view = |n| Attachment::loaded(vk::ImageView::from_raw(n));
        let colors = ColorAttachments::new([view(1), view(2)]);
        let views: Vec<u64> = colors.as_slice().iter().map(|a| a.view.as_raw()).collect();
        assert_eq!(views, [1, 2]);
        assert!(ColorAttachments::new([]).as_slice().is_empty());
    }

    #[test]
    fn the_pipeline_info_names_every_format() {
        let formats =
            PassFormats::color(vk::Format::B8G8R8A8_UNORM).with_depth(vk::Format::D32_SFLOAT);
        let info = formats.pipeline_info();
        assert_eq!(info.color_attachment_count, 1);
        assert_eq!(info.depth_attachment_format, vk::Format::D32_SFLOAT);
        assert_eq!(info.stencil_attachment_format, vk::Format::UNDEFINED);
    }

    #[test]
    fn an_attachment_maps_its_load_store_and_resolve() {
        let view = vk::ImageView::null();
        let cleared = Attachment::cleared(view, vk::ClearValue::default());
        let info = cleared.info(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        assert_eq!(info.load_op, vk::AttachmentLoadOp::CLEAR);
        assert_eq!(info.store_op, vk::AttachmentStoreOp::STORE);
        assert_eq!(info.resolve_mode, vk::ResolveModeFlags::NONE);

        let resolving = Attachment {
            load: Load::Discard,
            store: false,
            resolve: Some(view),
            ..Attachment::loaded(view)
        };
        let info = resolving.info(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        assert_eq!(info.load_op, vk::AttachmentLoadOp::DONT_CARE);
        assert_eq!(info.store_op, vk::AttachmentStoreOp::DONT_CARE);
        assert_eq!(info.resolve_mode, vk::ResolveModeFlags::AVERAGE);
        assert_eq!(
            info.resolve_image_layout,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        );
    }

    #[test]
    fn a_transition_carries_both_uses_layouts_and_the_aspect() {
        let to_sampled = Transition::depth(
            vk::Image::null(),
            ImageUse::DepthAttachment,
            ImageUse::Sampled,
        )
        .barrier();
        assert_eq!(
            to_sampled.old_layout,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        );
        assert_eq!(
            to_sampled.new_layout,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        );
        assert_eq!(
            to_sampled.subresource_range.aspect_mask,
            vk::ImageAspectFlags::DEPTH
        );
        assert_eq!(
            to_sampled.dst_access_mask,
            vk::AccessFlags2::SHADER_SAMPLED_READ
        );

        let to_color = Transition::color(
            vk::Image::null(),
            ImageUse::Sampled,
            ImageUse::ColorAttachment,
        )
        .barrier();
        assert_eq!(
            to_color.subresource_range.aspect_mask,
            vk::ImageAspectFlags::COLOR
        );
    }

    #[test]
    fn a_discard_leaves_undefined_but_waits_on_the_prior_use() {
        let move_in = |discard| {
            Transition::color(
                vk::Image::null(),
                ImageUse::Sampled,
                ImageUse::ColorAttachment,
            )
            .discarding_if(discard)
            .barrier()
        };
        let discarded = move_in(true);
        assert_eq!(discarded.old_layout, vk::ImageLayout::UNDEFINED);
        assert_eq!(
            discarded.src_stage_mask,
            vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER
        );
        assert_eq!(
            move_in(false).old_layout,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        );
    }

    #[test]
    fn a_copy_waits_on_and_releases_to_the_copy_stage() {
        let into_copy =
            Transition::color(vk::Image::null(), ImageUse::Sampled, ImageUse::TransferSrc)
                .barrier();
        assert_eq!(into_copy.new_layout, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        assert_eq!(into_copy.dst_stage_mask, vk::PipelineStageFlags2::COPY);
        assert_eq!(into_copy.dst_access_mask, vk::AccessFlags2::TRANSFER_READ);
        let out_of_copy =
            Transition::color(vk::Image::null(), ImageUse::TransferDst, ImageUse::Sampled)
                .barrier();
        assert_eq!(
            out_of_copy.old_layout,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL
        );
        assert_eq!(out_of_copy.src_stage_mask, vk::PipelineStageFlags2::COPY);
        assert_eq!(
            out_of_copy.src_access_mask,
            vk::AccessFlags2::TRANSFER_WRITE
        );
    }

    #[test]
    fn a_transition_narrows_to_one_level_and_layer() {
        let whole = Transition::color(
            vk::Image::null(),
            ImageUse::Sampled,
            ImageUse::ColorAttachment,
        )
        .barrier()
        .subresource_range;
        assert_eq!(
            (whole.base_mip_level, whole.level_count),
            (0, vk::REMAINING_MIP_LEVELS)
        );
        assert_eq!(
            (whole.base_array_layer, whole.layer_count),
            (0, vk::REMAINING_ARRAY_LAYERS)
        );
        let one = Transition::color(
            vk::Image::null(),
            ImageUse::Sampled,
            ImageUse::ColorAttachment,
        )
        .at_level(2)
        .at_layer(4)
        .barrier()
        .subresource_range;
        assert_eq!((one.base_mip_level, one.level_count), (2, 1));
        assert_eq!((one.base_array_layer, one.layer_count), (4, 1));
    }
}
