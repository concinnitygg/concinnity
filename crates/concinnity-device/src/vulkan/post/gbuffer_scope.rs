//! The G-buffer passes' rendering scope: the three channels over the private
//! depth, and the layout moves the scope makes around them. The pre-pass opens
//! it cleared; the late pass opens it over what the pre-pass left.

use ash::vk;

use super::gbuffer::GbufferResources;
use crate::vulkan::depth;
use crate::vulkan::record::Recorder;
use crate::vulkan::rendering::{
    Attachment, ColorAttachments, ImageUse, Load, RenderTargets, Transition,
};

// Alpha-0 normal+depth is "no geometry", roughness 1.0 is no reflection, and
// zero velocity is no motion.
const CHANNEL_CLEARS: [vk::ClearValue; 3] = [
    vk::ClearValue {
        color: vk::ClearColorValue { float32: [0.0; 4] },
    },
    vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [1.0, 0.0, 0.0, 0.0],
        },
    },
    vk::ClearValue {
        color: vk::ClearColorValue { float32: [0.0; 4] },
    },
];

// How a G-buffer scope opens its targets.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::vulkan) enum GbufferOpen {
    // Every target cleared.
    Clear,
    // Drawn over what an earlier scope this frame left.
    Resume,
}

impl GbufferOpen {
    fn load(self, clear: vk::ClearValue) -> Load {
        match self {
            Self::Clear => Load::Clear(clear),
            Self::Resume => Load::Keep,
        }
    }
}

// One frame slot's targets, image and view each.
#[derive(Clone, Copy)]
struct ScopeImages {
    channels: [(vk::Image, vk::ImageView); 3],
    depth: (vk::Image, vk::ImageView),
}

// The moves into a scope. The channels rest sampled between scopes and the
// private depth stays an attachment; a cleared scope discards both, waiting
// only on that last use.
fn entry_moves(images: &ScopeImages, open: GbufferOpen) -> [Transition; 4] {
    let discard = open == GbufferOpen::Clear;
    let [a, b, c] = images.channels.map(|(image, _)| {
        Transition::color(image, ImageUse::Sampled, ImageUse::ColorAttachment)
            .discarding_if(discard)
    });
    let to_depth = ImageUse::DepthAttachment;
    let depth = Transition::depth(images.depth.0, to_depth, to_depth).discarding_if(discard);
    [a, b, c, depth]
}

fn exit_moves(images: &ScopeImages) -> [Transition; 3] {
    images
        .channels
        .map(|(image, _)| Transition::color(image, ImageUse::ColorAttachment, ImageUse::Sampled))
}

fn attachments(images: &ScopeImages, open: GbufferOpen) -> ([Attachment; 3], Attachment) {
    let mut i = 0;
    let channels = images.channels.map(|(_, view)| {
        let load = open.load(CHANNEL_CLEARS[i]);
        i += 1;
        Attachment {
            load,
            ..Attachment::loaded(view)
        }
    });
    let depth = Attachment {
        load: open.load(depth::CLEAR_VALUE),
        ..Attachment::loaded(images.depth.1)
    };
    (channels, depth)
}

impl GbufferResources {
    fn scope_images(&self, frame: usize) -> Option<ScopeImages> {
        let channel = |targets: &[super::gbuffer::PooledTarget]| {
            targets.get(frame).map(|t| (t.image, t.view))
        };
        let depth = self.depth_images.get(frame)?;
        Some(ScopeImages {
            channels: [
                channel(&self.normal_depth_images)?,
                channel(&self.roughness_images)?,
                channel(&self.velocity_images)?,
            ],
            depth: (depth.image, depth.view),
        })
    }

    // Render frame slot `frame`'s G-buffer over `extent` under `draw`, opened
    // as `open`, leaving the channels sampled. Nothing is recorded while the
    // slot's targets are missing.
    pub(in crate::vulkan) fn render(
        &self,
        rec: &Recorder<'_>,
        frame: usize,
        extent: vk::Extent2D,
        open: GbufferOpen,
        draw: impl FnOnce(&Recorder<'_>),
    ) {
        let Some(images) = self.scope_images(frame) else {
            return;
        };
        rec.transition(&entry_moves(&images, open));
        let (colors, depth) = attachments(&images, open);
        rec.begin_rendering(&RenderTargets {
            area: vk::Rect2D::default().extent(extent),
            colors: ColorAttachments::new(colors),
            depth: Some(depth),
        });
        draw(rec);
        rec.end_rendering();
        rec.transition(&exit_moves(&images));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    fn images() -> ScopeImages {
        let pair = |n: u64| (vk::Image::from_raw(n), vk::ImageView::from_raw(n + 10));
        ScopeImages {
            channels: [pair(1), pair(2), pair(3)],
            depth: pair(4),
        }
    }

    fn layouts(moves: &[Transition]) -> Vec<(vk::ImageLayout, vk::ImageLayout)> {
        moves
            .iter()
            .map(|t| {
                let b = t.barrier();
                (b.old_layout, b.new_layout)
            })
            .collect()
    }

    #[test]
    fn a_cleared_scope_discards_every_target_and_clears_it() {
        let images = images();
        let color = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
        let depth = vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        let undefined = vk::ImageLayout::UNDEFINED;
        assert_eq!(
            layouts(&entry_moves(&images, GbufferOpen::Clear)),
            [
                (undefined, color),
                (undefined, color),
                (undefined, color),
                (undefined, depth)
            ]
        );
        let (channels, depth_attachment) = attachments(&images, GbufferOpen::Clear);
        for c in channels.iter().chain([&depth_attachment]) {
            assert_eq!(c.info(color).load_op, vk::AttachmentLoadOp::CLEAR);
            assert!(c.store);
        }
    }

    #[test]
    fn a_resumed_scope_loads_the_sampled_channels_and_held_depth() {
        let images = images();
        let color = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
        let sampled = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        let depth = vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        assert_eq!(
            layouts(&entry_moves(&images, GbufferOpen::Resume)),
            [
                (sampled, color),
                (sampled, color),
                (sampled, color),
                (depth, depth)
            ]
        );
        let (channels, depth_attachment) = attachments(&images, GbufferOpen::Resume);
        for c in channels.iter().chain([&depth_attachment]) {
            assert_eq!(c.info(color).load_op, vk::AttachmentLoadOp::LOAD);
        }
    }

    #[test]
    fn every_scope_leaves_the_channels_sampled() {
        let color = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
        let sampled = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        assert_eq!(layouts(&exit_moves(&images())), [(color, sampled); 3]);
    }
}
