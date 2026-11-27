//! The main scene pass's rendering scope: the HDR color (multisampled, or the
//! single-sample scene itself), the single-sample image a multisampled color
//! resolves into, and the depth, with the layout moves the scope makes around
//! them. The frame's main passes, the raymarch pass over them, and the
//! off-camera renders (probe faces, mirror planes) all open it the same way.

use ash::vk;
use concinnity_engine::render::frame::plan::FramePlan;
use concinnity_engine::render::render_graph::PassId;

use super::context::HDR_FORMAT;
use super::record::Recorder;
use super::rendering::{
    Attachment, ColorAttachments, ImageUse, Load, PassFormats, RenderTargets, Transition,
};
use super::texture::GpuImage;

/// What every main-pass pipeline renders to.
pub(in crate::vulkan) const MAIN_FORMATS: PassFormats =
    PassFormats::color(HDR_FORMAT).with_depth(vk::Format::D32_SFLOAT);

// One image a scope renders to: its view, and the image the scope's moves act
// on, narrowed to mip 0 of one array layer when the view names only that.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct Target {
    image: vk::Image,
    view: vk::ImageView,
    layer: Option<u32>,
}

impl Target {
    // The whole of `image`.
    pub(in crate::vulkan) fn of(image: &GpuImage) -> Self {
        Self {
            image: image.image,
            view: image.view,
            layer: None,
        }
    }

    // Mip 0 of `image`'s array layer `layer`, which `view` names.
    pub(in crate::vulkan) fn layer(image: vk::Image, view: vk::ImageView, layer: u32) -> Self {
        Self {
            image,
            view,
            layer: Some(layer),
        }
    }

    fn color(self, from: ImageUse, to: ImageUse) -> Transition {
        let t = Transition::color(self.image, from, to);
        match self.layer {
            Some(layer) => t.at_level(0).at_layer(layer),
            None => t,
        }
    }
}

// The images one main-pass scope renders over.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct MainAttachments {
    // The color drawn: the multisampled color, or the single-sample scene.
    pub color: Target,
    // The single-sample image a multisampled color resolves into.
    pub resolve: Option<Target>,
    pub depth: Target,
    // The last use of the image the scene ends in (the resolve, or the
    // single-sample color), which a scope's move waits on.
    pub scene_from: ImageUse,
    // The depth is the scope's own, its contents discarded as it opens, rather
    // than one the render graph has moved to an attachment.
    pub discard_depth: bool,
    pub area: vk::Rect2D,
}

impl MainAttachments {
    // An off-camera render's own images over `area`: the color drawn and the
    // image it resolves into when multisampled, the last use of the image the
    // render ends in, and a depth nothing reads before or after the render.
    pub(in crate::vulkan) fn off_camera(
        (color, resolve): (Target, Option<Target>),
        scene_from: ImageUse,
        depth: &GpuImage,
        area: vk::Rect2D,
    ) -> Self {
        Self {
            color,
            resolve,
            depth: Target::of(depth),
            scene_from,
            discard_depth: true,
            area,
        }
    }
}

// How a scope opens its color and depth.
#[derive(Clone, Copy)]
pub(in crate::vulkan) enum MainOpen {
    // Both cleared; the color to `clear`.
    Clear(vk::ClearValue),
    // Both drawn over: a multisampled color held as an attachment, a
    // single-sample one resting in `scene_from`.
    Resume,
}

// Where a scope leaves a single-sample color: sampled for the passes after
// it, or held as an attachment for a later scope that resumes it. A
// multisampled color always stays an attachment, stored only when held.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::vulkan) enum MainClose {
    Sampled,
    Held,
}

// The scopes this frame runs after the main pass that resume its scene: the
// late pass of two-pass occlusion, and a raymarch pass with a volume to draw.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub(in crate::vulkan) struct SceneResumers {
    pub late: bool,
    pub raymarch: bool,
}

impl SceneResumers {
    // Where the main pass leaves the scene. A multisampled scene is resumed
    // by the raymarch pass; a single-sample one it reopens from sampled.
    pub(in crate::vulkan) fn main_close(self, msaa: bool) -> MainClose {
        if self.late {
            MainClose::Held
        } else {
            self.late_close(msaa)
        }
    }

    // Where the late pass leaves the scene.
    pub(in crate::vulkan) fn late_close(self, msaa: bool) -> MainClose {
        if msaa && self.raymarch {
            MainClose::Held
        } else {
            MainClose::Sampled
        }
    }
}

// The moves into the scope, the first `n` of the array. A discarded image
// waits only on its last use; one already an attachment still takes a move,
// which orders this scope's writes after the last scope's.
fn entry_moves(at: &MainAttachments, open: MainOpen) -> ([Transition; 3], usize) {
    let clear = matches!(open, MainOpen::Clear(_));
    let to_color = ImageUse::ColorAttachment;
    let to_depth = ImageUse::DepthAttachment;
    let depth =
        Transition::depth(at.depth.image, to_depth, to_depth).discarding_if(at.discard_depth);
    if let Some(resolve) = at.resolve {
        let color = at.color.color(to_color, to_color).discarding_if(clear);
        let resolve = resolve.color(at.scene_from, to_color).discarding();
        ([color, resolve, depth], 3)
    } else {
        let color = at.color.color(at.scene_from, to_color).discarding_if(clear);
        ([color, depth, depth], 2)
    }
}

// The move out of the scope, when the scene leaves it sampled.
fn exit_move(at: &MainAttachments, close: MainClose) -> Option<Transition> {
    let sampled = |t: Target| t.color(ImageUse::ColorAttachment, ImageUse::Sampled);
    match at.resolve {
        Some(resolve) => Some(sampled(resolve)),
        None if close == MainClose::Sampled => Some(sampled(at.color)),
        None => None,
    }
}

// The move that leaves a held single-sample color sampled; a multisampled
// scope already left its resolve sampled.
fn release_move(at: &MainAttachments) -> Option<Transition> {
    at.resolve
        .is_none()
        .then(|| at.color.color(ImageUse::ColorAttachment, ImageUse::Sampled))
}

// The color and depth attachments a scope renders with.
fn attachments(at: &MainAttachments, open: MainOpen, close: MainClose) -> (Attachment, Attachment) {
    let (color_load, depth_load) = match open {
        MainOpen::Clear(value) => (Load::Clear(value), Load::Clear(super::depth::CLEAR_VALUE)),
        MainOpen::Resume => (Load::Keep, Load::Keep),
    };
    let color = Attachment {
        view: at.color.view,
        load: color_load,
        store: at.resolve.is_none() || close == MainClose::Held,
        resolve: at.resolve.map(|r| r.view),
    };
    let depth = Attachment {
        load: depth_load,
        ..Attachment::loaded(at.depth.view)
    };
    (color, depth)
}

// Move the attachments in and begin rendering over them.
pub(in crate::vulkan) fn begin(
    rec: &Recorder<'_>,
    at: &MainAttachments,
    open: MainOpen,
    close: MainClose,
) {
    let (moves, n) = entry_moves(at, open);
    rec.transition(&moves[..n]);
    let (color, depth) = attachments(at, open, close);
    rec.begin_rendering(&RenderTargets {
        area: at.area,
        colors: ColorAttachments::new([color]),
        depth: Some(depth),
    });
}

// End rendering and leave the scene where `close` says.
pub(in crate::vulkan) fn end(rec: &Recorder<'_>, at: &MainAttachments, close: MainClose) {
    rec.end_rendering();
    if let Some(exit) = exit_move(at, close) {
        rec.transition(&[exit]);
    }
}

// Leave the scene an earlier scope held sampled, in place of a resuming scope
// that draws nothing.
pub(in crate::vulkan) fn release(rec: &Recorder<'_>, at: &MainAttachments) {
    if let Some(release) = release_move(at) {
        rec.transition(&[release]);
    }
}

impl super::context::VkContext {
    // Frame slot `frame_idx`'s main attachments, over the depth the graph has
    // moved to an attachment, the scene last used as `scene_from`.
    pub(in crate::vulkan) fn main_attachments(
        &self,
        frame_idx: usize,
        scene_from: ImageUse,
    ) -> MainAttachments {
        let targets = &self.targets;
        let scene = Target::of(&targets.hdr_resolve_images[frame_idx]);
        let (color, resolve) = match targets.color_images.get(frame_idx) {
            Some(color) => (Target::of(color), Some(scene)),
            None => (scene, None),
        };
        MainAttachments {
            color,
            resolve,
            depth: Target::of(&targets.depth_images[frame_idx]),
            scene_from,
            discard_depth: false,
            area: vk::Rect2D::default().extent(targets.render_extent),
        }
    }

    // The scopes after this frame's main pass that resume its scene, from
    // what the frame's graph runs.
    pub(in crate::vulkan) fn scene_resumers(&self, plan: &FramePlan) -> SceneResumers {
        let runs = |pass| plan.graph.pass(pass).is_some();
        SceneResumers {
            late: runs(PassId::MainLate),
            raymarch: runs(PassId::Raymarch)
                && self
                    .raymarch
                    .as_ref()
                    .is_some_and(|rm| rm.draws_any(&plan.camera.frustum)),
        }
    }

    // Whether the main scene is multisampled.
    pub(in crate::vulkan) fn main_msaa(&self) -> bool {
        self.targets.msaa_samples != vk::SampleCountFlags::TYPE_1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    fn target(n: u64) -> Target {
        Target {
            image: vk::Image::from_raw(n),
            view: vk::ImageView::from_raw(n + 100),
            layer: None,
        }
    }

    fn scope(resolve: bool) -> MainAttachments {
        MainAttachments {
            color: target(1),
            resolve: resolve.then(|| target(2)),
            depth: target(3),
            scene_from: ImageUse::Sampled,
            discard_depth: false,
            area: vk::Rect2D::default(),
        }
    }

    fn layouts(moves: &[Transition]) -> Vec<(u64, vk::ImageLayout, vk::ImageLayout)> {
        moves
            .iter()
            .map(|t| {
                let b = t.barrier();
                (b.image.as_raw(), b.old_layout, b.new_layout)
            })
            .collect()
    }

    const COLOR: vk::ImageLayout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
    const DEPTH: vk::ImageLayout = vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    const SAMPLED: vk::ImageLayout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    const UNDEFINED: vk::ImageLayout = vk::ImageLayout::UNDEFINED;

    #[test]
    fn the_main_pass_renders_hdr_color_over_float_depth() {
        assert_eq!(MAIN_FORMATS.color_formats(), [HDR_FORMAT]);
        assert_eq!(
            MAIN_FORMATS.pipeline_info().depth_attachment_format,
            vk::Format::D32_SFLOAT
        );
    }

    const CLEAR: MainOpen = MainOpen::Clear(vk::ClearValue {
        color: vk::ClearColorValue { float32: [0.0; 4] },
    });

    #[test]
    fn a_multisampled_scope_discards_its_resolve_and_orders_its_depth() {
        let (moves, n) = entry_moves(&scope(true), CLEAR);
        // The resolve waits on its last sampling, not on every earlier stage.
        assert_eq!(
            moves[1].barrier().src_stage_mask,
            vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER
        );
        assert_eq!(
            layouts(&moves[..n]),
            [
                (1, UNDEFINED, COLOR),
                (2, UNDEFINED, COLOR),
                (3, DEPTH, DEPTH)
            ]
        );
        assert_eq!(
            layouts(&[exit_move(&scope(true), MainClose::Held).expect("resolve")]),
            [(2, COLOR, SAMPLED)]
        );
    }

    #[test]
    fn a_resumed_color_still_takes_a_move() {
        let held = MainAttachments {
            scene_from: ImageUse::ColorAttachment,
            ..scope(false)
        };
        let (moves, n) = entry_moves(&held, MainOpen::Resume);
        assert_eq!(layouts(&moves[..n]), [(1, COLOR, COLOR), (3, DEPTH, DEPTH)]);
    }

    #[test]
    fn a_held_single_sample_color_stays_an_attachment() {
        assert!(exit_move(&scope(false), MainClose::Held).is_none());
        assert_eq!(
            layouts(&[exit_move(&scope(false), MainClose::Sampled).expect("sampled")]),
            [(1, COLOR, SAMPLED)]
        );
    }

    #[test]
    fn the_scene_is_held_only_for_a_scope_that_resumes_it() {
        let none = SceneResumers::default();
        let late = SceneResumers {
            late: true,
            raymarch: false,
        };
        let raymarch = SceneResumers {
            late: false,
            raymarch: true,
        };
        for msaa in [false, true] {
            assert_eq!(none.main_close(msaa), MainClose::Sampled);
            assert_eq!(late.main_close(msaa), MainClose::Held);
            assert_eq!(late.late_close(msaa), MainClose::Sampled);
        }
        // A single-sample scene reopens from sampled for the raymarch pass.
        assert_eq!(raymarch.main_close(false), MainClose::Sampled);
        assert_eq!(raymarch.main_close(true), MainClose::Held);
        assert_eq!(raymarch.late_close(true), MainClose::Held);
    }

    #[test]
    fn a_release_samples_only_a_held_single_sample_color() {
        assert!(release_move(&scope(true)).is_none());
        assert_eq!(
            layouts(&[release_move(&scope(false)).expect("held color")]),
            [(1, COLOR, SAMPLED)]
        );
    }

    #[test]
    fn a_multisampled_color_is_stored_only_when_held() {
        let (held, _) = attachments(&scope(true), CLEAR, MainClose::Held);
        let (sampled, _) = attachments(&scope(true), CLEAR, MainClose::Sampled);
        assert!(held.store && !sampled.store);
        assert_eq!(sampled.resolve, Some(target(2).view));
        let (single, _) = attachments(&scope(false), CLEAR, MainClose::Sampled);
        assert!(single.store && single.resolve.is_none());
    }

    #[test]
    fn a_resumed_scope_loads_what_a_cleared_scope_clears() {
        for (open, op) in [
            (MainOpen::Resume, vk::AttachmentLoadOp::LOAD),
            (CLEAR, vk::AttachmentLoadOp::CLEAR),
        ] {
            let (color, depth) = attachments(&scope(false), open, MainClose::Sampled);
            assert_eq!(color.info(COLOR).load_op, op);
            assert_eq!(depth.info(DEPTH).load_op, op);
        }
    }

    #[test]
    fn an_off_camera_render_discards_its_depth_and_narrows_a_layer_target() {
        let at = MainAttachments {
            color: Target::layer(vk::Image::from_raw(7), vk::ImageView::from_raw(8), 4),
            discard_depth: true,
            ..scope(false)
        };
        let (moves, n) = entry_moves(&at, CLEAR);
        assert_eq!(layouts(&moves[..n])[1], (3, UNDEFINED, DEPTH));
        let range = moves[0].barrier().subresource_range;
        assert_eq!((range.base_array_layer, range.layer_count), (4, 1));
        assert_eq!((range.base_mip_level, range.level_count), (0, 1));
    }
}
