//! Pipeline builders. [`GraphicsPipelineDesc`] describes a graphics pipeline as
//! data -- stages, color attachments with their blend, depth, raster, topology,
//! samples and vertex input -- over the state every engine pass shares (one
//! dynamic viewport and scissor, all samples enabled, no stencil), and creates
//! it through the pipeline cache. [`compute_pipeline`] is the compute
//! counterpart.

use ash::vk;
use concinnity_core::render::depth::{DEPTH_INCLUSIVE_COMPARE, DEPTH_WRITE_COMPARE};
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::post::device::PostBlend;

use crate::vulkan::depth::compare_op;
use crate::vulkan::error::map_vk_result;
use crate::vulkan::owned::{OwnedPipeline, VkDevice};
use crate::vulkan::pipeline::{SHADER_ENTRY, spv_module};

// How a color attachment combines the fragment with what it holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::vulkan) enum Blend {
    // Overwrite.
    Opaque,
    // `dst + src`.
    Additive,
    // `src + (1 - src.a) * dst`.
    PremultipliedOver,
    // `src.a * src + (1 - src.a) * dst`, alpha included.
    AlphaOver,
    // `max(src, dst)`.
    Max,
}

impl From<PostBlend> for Blend {
    fn from(blend: PostBlend) -> Self {
        match blend {
            PostBlend::Replace => Blend::Opaque,
            PostBlend::Additive => Blend::Additive,
            PostBlend::PremultipliedOver => Blend::PremultipliedOver,
        }
    }
}

impl Blend {
    fn raw(self) -> vk::PipelineColorBlendAttachmentState {
        let base = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA);
        let (src, dst) = match self {
            Blend::Opaque => return base.blend_enable(false),
            Blend::Additive => (vk::BlendFactor::ONE, vk::BlendFactor::ONE),
            Blend::PremultipliedOver => {
                (vk::BlendFactor::ONE, vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            }
            Blend::AlphaOver => (
                vk::BlendFactor::SRC_ALPHA,
                vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            ),
            Blend::Max => (vk::BlendFactor::ONE, vk::BlendFactor::ONE),
        };
        let op = match self {
            Blend::Max => vk::BlendOp::MAX,
            _ => vk::BlendOp::ADD,
        };
        base.blend_enable(true)
            .src_color_blend_factor(src)
            .dst_color_blend_factor(dst)
            .color_blend_op(op)
            .src_alpha_blend_factor(src)
            .dst_alpha_blend_factor(dst)
            .alpha_blend_op(op)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::vulkan) enum Depth {
    Off,
    Test { compare: vk::CompareOp, write: bool },
}

impl Depth {
    // The opaque-geometry test: nearer fragments pass and write. The camera's
    // opaque passes and every shadow caster draw with it.
    pub(in crate::vulkan) const fn write() -> Depth {
        Depth::Test {
            compare: compare_op(DEPTH_WRITE_COMPARE),
            write: true,
        }
    }

    // A pass whose shader writes a depth no farther than the rasterized one:
    // equal depth passes too.
    pub(in crate::vulkan) const fn write_inclusive() -> Depth {
        Depth::Test {
            compare: compare_op(DEPTH_INCLUSIVE_COMPARE),
            write: true,
        }
    }

    // Tested against depth another pass wrote, without writing it.
    pub(in crate::vulkan) const fn read_only() -> Depth {
        Depth::Test {
            compare: compare_op(DEPTH_INCLUSIVE_COMPARE),
            write: false,
        }
    }

    pub(in crate::vulkan) fn raw(self) -> vk::PipelineDepthStencilStateCreateInfo<'static> {
        let (test, compare, write) = match self {
            Depth::Off => (false, vk::CompareOp::ALWAYS, false),
            Depth::Test { compare, write } => (true, compare, write),
        };
        vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(test)
            .depth_write_enable(write)
            .depth_compare_op(compare)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false)
    }
}

// Constant and slope-scaled depth bias, and the clamp on their sum.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(in crate::vulkan) struct DepthBias {
    pub constant: f32,
    pub clamp: f32,
    pub slope: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::vulkan) struct Raster {
    pub cull: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
    pub polygon_mode: vk::PolygonMode,
    pub bias: Option<DepthBias>,
}

impl Default for Raster {
    fn default() -> Self {
        Self {
            cull: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            polygon_mode: vk::PolygonMode::FILL,
            bias: None,
        }
    }
}

impl Raster {
    fn raw(self) -> vk::PipelineRasterizationStateCreateInfo<'static> {
        let bias = self.bias.unwrap_or_default();
        vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(self.polygon_mode)
            .line_width(1.0)
            .cull_mode(self.cull)
            .front_face(self.front_face)
            .depth_bias_enable(self.bias.is_some())
            .depth_bias_constant_factor(bias.constant)
            .depth_bias_clamp(bias.clamp)
            .depth_bias_slope_factor(bias.slope)
    }
}

#[derive(Clone, Copy)]
pub(in crate::vulkan) struct GraphicsPipelineDesc<'a> {
    pub vert: &'a [u8],
    // `None` for a depth-only pipeline.
    pub frag: Option<&'a [u8]>,
    pub layout: vk::PipelineLayout,
    pub render_pass: vk::RenderPass,
    pub subpass: u32,
    // One blend per color attachment of the subpass, in order.
    pub color_targets: &'a [Blend],
    pub depth: Depth,
    pub raster: Raster,
    pub topology: vk::PrimitiveTopology,
    pub samples: vk::SampleCountFlags,
    pub vertex_bindings: &'a [vk::VertexInputBindingDescription],
    pub vertex_attributes: &'a [vk::VertexInputAttributeDescription],
}

impl<'a> GraphicsPipelineDesc<'a> {
    // A vertex-buffer-less fullscreen triangle into the subpass's color
    // attachments, single-sample, no depth.
    pub(in crate::vulkan) fn fullscreen(
        vert: &'a [u8],
        frag: &'a [u8],
        layout: vk::PipelineLayout,
        render_pass: vk::RenderPass,
        color_targets: &'a [Blend],
    ) -> Self {
        Self {
            vert,
            frag: Some(frag),
            layout,
            render_pass,
            subpass: 0,
            color_targets,
            depth: Depth::Off,
            raster: Raster::default(),
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            samples: vk::SampleCountFlags::TYPE_1,
            vertex_bindings: &[],
            vertex_attributes: &[],
        }
    }

    // Create the pipeline through the pipeline cache; `label` names it in a
    // failure.
    pub(in crate::vulkan) fn build(
        &self,
        device: &VkDevice,
        label: &str,
    ) -> RenderResult<OwnedPipeline> {
        let vert = spv_module(device, self.vert)?;
        let frag = self.frag.map(|spv| spv_module(device, spv)).transpose()?;
        let stage = |stage, module| {
            vk::PipelineShaderStageCreateInfo::default()
                .stage(stage)
                .module(module)
                .name(SHADER_ENTRY)
        };
        let mut stages = [stage(vk::ShaderStageFlags::VERTEX, vert.handle()); 2];
        let stage_count = match &frag {
            Some(frag) => {
                stages[1] = stage(vk::ShaderStageFlags::FRAGMENT, frag.handle());
                2
            }
            None => 1,
        };
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(self.vertex_bindings)
            .vertex_attribute_descriptions(self.vertex_attributes);
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(self.topology)
            .primitive_restart_enable(primitive_restart(self.topology));
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let raster = self.raster.raw();
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(self.samples);
        let depth = self.depth.raw();
        let attachments = color_attachments(self.color_targets);
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(&attachments[..self.color_targets.len().min(MAX_TARGETS)]);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let mut info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages[..stage_count])
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth)
            .dynamic_state(&dynamic)
            .layout(self.layout)
            .render_pass(self.render_pass)
            .subpass(self.subpass);
        if !self.color_targets.is_empty() {
            info = info.color_blend_state(&color_blend);
        }
        crate::vulkan::pipeline_cache::create_graphics_pipeline(device, &info)
            .map_err(|e| map_vk_result(e, &format!("create {label} pipeline")))
    }
}

// Most color attachments one pipeline writes.
const MAX_TARGETS: usize = 8;

fn color_attachments(targets: &[Blend]) -> [vk::PipelineColorBlendAttachmentState; MAX_TARGETS] {
    let mut out = [vk::PipelineColorBlendAttachmentState::default(); MAX_TARGETS];
    for (slot, blend) in out.iter_mut().zip(targets) {
        *slot = blend.raw();
    }
    out
}

// A compute pipeline running `spirv` under `layout`; `label` names it in a
// failure.
pub(in crate::vulkan) fn compute_pipeline(
    device: &VkDevice,
    layout: vk::PipelineLayout,
    spirv: &[u8],
    label: &str,
) -> RenderResult<OwnedPipeline> {
    let module = spv_module(device, spirv)?;
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module.handle())
        .name(SHADER_ENTRY);
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layout);
    crate::vulkan::pipeline_cache::create_compute_pipeline(device, &info)
        .map_err(|e| map_vk_result(e, &format!("create {label} pipeline")))
}

// Whether `topology` keeps primitive restart on. Metal restarts every strip at
// the restart index and cannot turn that off, so strips enable it everywhere:
// it only ever applies to indexed draws, and no draw here indexes a strip with
// the restart value. Lists keep it off, since enabling it for them needs a
// device feature.
fn primitive_restart(topology: vk::PrimitiveTopology) -> bool {
    use vk::PrimitiveTopology as T;
    matches!(
        topology,
        T::LINE_STRIP | T::TRIANGLE_STRIP | T::TRIANGLE_FAN
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_keep_primitive_restart_on_and_lists_off() {
        assert!(primitive_restart(vk::PrimitiveTopology::TRIANGLE_STRIP));
        assert!(primitive_restart(vk::PrimitiveTopology::LINE_STRIP));
        assert!(!primitive_restart(vk::PrimitiveTopology::TRIANGLE_LIST));
        assert!(!primitive_restart(vk::PrimitiveTopology::LINE_LIST));
        assert!(!primitive_restart(vk::PrimitiveTopology::POINT_LIST));
    }

    #[test]
    fn blends_set_their_factors_and_opaque_disables_blending() {
        let opaque = Blend::Opaque.raw();
        assert_eq!(opaque.blend_enable, vk::FALSE);
        assert_eq!(opaque.color_write_mask, vk::ColorComponentFlags::RGBA);
        let over = Blend::AlphaOver.raw();
        assert_eq!(over.blend_enable, vk::TRUE);
        assert_eq!(
            (over.src_color_blend_factor, over.dst_color_blend_factor),
            (
                vk::BlendFactor::SRC_ALPHA,
                vk::BlendFactor::ONE_MINUS_SRC_ALPHA
            )
        );
        assert_eq!(
            (over.src_alpha_blend_factor, over.dst_alpha_blend_factor),
            (
                vk::BlendFactor::SRC_ALPHA,
                vk::BlendFactor::ONE_MINUS_SRC_ALPHA
            )
        );
        let pre = Blend::from(PostBlend::PremultipliedOver).raw();
        assert_eq!(
            (pre.src_color_blend_factor, pre.dst_color_blend_factor),
            (vk::BlendFactor::ONE, vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        );
        let add = Blend::from(PostBlend::Additive).raw();
        assert_eq!(
            (add.src_alpha_blend_factor, add.dst_alpha_blend_factor),
            (vk::BlendFactor::ONE, vk::BlendFactor::ONE)
        );
        assert_eq!(Blend::from(PostBlend::Replace), Blend::Opaque);
    }

    #[test]
    fn max_blends_both_channels_by_max() {
        let max = Blend::Max.raw();
        assert_eq!(max.blend_enable, vk::TRUE);
        assert_eq!(max.color_blend_op, vk::BlendOp::MAX);
        assert_eq!(max.alpha_blend_op, vk::BlendOp::MAX);
        assert_eq!(Blend::AlphaOver.raw().color_blend_op, vk::BlendOp::ADD);
    }

    #[test]
    fn depth_modes_map_to_test_write_and_compare() {
        let off = Depth::Off.raw();
        assert_eq!(
            (off.depth_test_enable, off.depth_write_enable),
            (vk::FALSE, vk::FALSE)
        );
        // The camera's opaque passes and the shadow casters share it.
        let write = Depth::write().raw();
        assert_eq!(
            (write.depth_test_enable, write.depth_write_enable),
            (vk::TRUE, vk::TRUE)
        );
        assert_eq!(write.depth_compare_op, vk::CompareOp::GREATER);
        let inclusive = Depth::write_inclusive().raw();
        assert_eq!(inclusive.depth_write_enable, vk::TRUE);
        assert_eq!(inclusive.depth_compare_op, vk::CompareOp::GREATER_OR_EQUAL);
        let read_only = Depth::read_only().raw();
        assert_eq!(
            (read_only.depth_test_enable, read_only.depth_write_enable),
            (vk::TRUE, vk::FALSE)
        );
        assert_eq!(read_only.depth_compare_op, vk::CompareOp::GREATER_OR_EQUAL);
        let read = Depth::Test {
            compare: vk::CompareOp::GREATER_OR_EQUAL,
            write: false,
        }
        .raw();
        assert_eq!(read.depth_write_enable, vk::FALSE);
        assert_eq!(read.depth_compare_op, vk::CompareOp::GREATER_OR_EQUAL);
        assert_eq!(read.stencil_test_enable, vk::FALSE);
    }

    #[test]
    fn raster_enables_bias_only_when_given() {
        let plain = Raster::default().raw();
        assert_eq!(plain.depth_bias_enable, vk::FALSE);
        assert_eq!(plain.cull_mode, vk::CullModeFlags::NONE);
        assert_eq!(plain.front_face, vk::FrontFace::COUNTER_CLOCKWISE);
        assert_eq!(plain.line_width, 1.0);
        let biased = Raster {
            bias: Some(DepthBias {
                constant: 2.0,
                clamp: 0.1,
                slope: 1.5,
            }),
            ..Raster::default()
        }
        .raw();
        assert_eq!(biased.depth_bias_enable, vk::TRUE);
        assert_eq!(
            (
                biased.depth_bias_constant_factor,
                biased.depth_bias_clamp,
                biased.depth_bias_slope_factor
            ),
            (2.0, 0.1, 1.5)
        );
    }

    #[test]
    fn color_attachments_follow_the_targets_in_order() {
        let a = color_attachments(&[Blend::Opaque, Blend::Additive, Blend::Opaque]);
        assert_eq!(a[0].blend_enable, vk::FALSE);
        assert_eq!(a[1].blend_enable, vk::TRUE);
        assert_eq!(a[2].blend_enable, vk::FALSE);
        assert_eq!(a[3].color_write_mask, vk::ColorComponentFlags::empty());
    }

    #[test]
    fn the_fullscreen_preset_is_single_sample_depthless_and_vertex_free() {
        let targets = [Blend::Opaque];
        let d = GraphicsPipelineDesc::fullscreen(
            &[],
            &[],
            vk::PipelineLayout::null(),
            vk::RenderPass::null(),
            &targets,
        );
        assert_eq!(d.samples, vk::SampleCountFlags::TYPE_1);
        assert_eq!(d.depth, Depth::Off);
        assert_eq!(d.topology, vk::PrimitiveTopology::TRIANGLE_LIST);
        assert!(d.vertex_bindings.is_empty() && d.vertex_attributes.is_empty());
        assert_eq!(d.subpass, 0);
        assert_eq!(d.color_targets, &targets);
    }
}
