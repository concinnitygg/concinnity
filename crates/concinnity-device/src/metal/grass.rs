//! The grass field on Metal (see `concinnity_core::render::grass`): the kernel
//! that places this frame's visible blades, and the two indirect draws that
//! render them at the tail of the G-buffer pre-pass and the main pass. Both
//! draws ride the encoder their pass already has open, after the surfaces
//! bound the pass's view blocks, lights and argument buffers, so grass adds
//! only its own block and blade buffer.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::render::error::{RenderError, RenderResult};
pub(in crate::metal) use concinnity_core::render::grass::GrassFrame;
use concinnity_core::render::grass::{GrassCamera, GrassField};
use concinnity_core::render::shader_programs::metal::prepass_buffers;
use concinnity_core::render::uniforms::grass::{
    GRASS_ARGS_BYTES, GpuGrassBlade, grass_args_offset,
};
use concinnity_core::render::uniforms::{GBufferView, ViewUniforms};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer as _, MTLCommandEncoder as _, MTLComputeCommandEncoder as _,
    MTLComputePassDescriptor, MTLComputePipelineState, MTLCullMode, MTLDevice, MTLPixelFormat,
    MTLPrimitiveType, MTLRenderCommandEncoder, MTLRenderPipelineDescriptor, MTLRenderPipelineState,
    MTLResourceOptions, MTLSize,
};

use super::builtin_shaders::{
    self, GRASS_FRAG, GRASS_GENERATE, GRASS_PREPASS_FRAG, GRASS_PREPASS_VERT, GRASS_VERT,
};
use super::context::MtlContext;
use super::encode::{ComputeEncode, RenderEncode};
use super::error::allocation_failed;
use super::scoped_encoder::ScopedEncoder;

// Where the grass draws read their block and blades (`grass.hlsl`), clear of
// every main-pass and pre-pass slot.
const PARAMS_INDEX: usize = 19;
const BLADES_INDEX: usize = 20;

// The kernel's slots, by register.
const KERNEL_PARAMS_INDEX: usize = 0;
const KERNEL_BLADES_INDEX: usize = 1;
const KERNEL_ARGS_INDEX: usize = 2;
const KERNEL_HEIGHTS_INDEX: usize = 3;
const KERNEL_MASKS_INDEX: usize = 4;

pub(super) struct GrassPipelines {
    generate: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    prepass: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    main: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
}

impl GrassPipelines {
    pub(super) fn build(
        device: &ProtocolObject<dyn MTLDevice>,
        sample_count: u32,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        Ok(Self {
            generate: builtin_shaders::compute_pipeline(device, &GRASS_GENERATE, hot_reload)?,
            prepass: build_prepass_pipeline(device, hot_reload)?,
            main: build_main_pipeline(device, sample_count, hot_reload)?,
        })
    }
}

// The grass field and what draws it: built once at init when the world grows
// one.
pub(super) struct GrassState {
    pub(super) field: GrassField,
    pub(super) pipelines: GrassPipelines,
    // The visible blades the kernel appends, `field.capacity` of them. Written
    // and read only on the GPU.
    blades: Retained<ProtocolObject<dyn MTLBuffer>>,
    // Two slots of non-indexed draw arguments; see `GRASS_ARGS_SLOTS`.
    args: Retained<ProtocolObject<dyn MTLBuffer>>,
    // Every terrain's heights and every layer's mask texels, which the kernel
    // reads to root and thin the blades.
    heights: Retained<ProtocolObject<dyn MTLBuffer>>,
    masks: Retained<ProtocolObject<dyn MTLBuffer>>,
    // Frames the kernel has run, which picks the slot it fills.
    runs: u32,
}

// A shared buffer holding a copy of `data`.
fn buffer_with<T: bytemuck::NoUninit>(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[T],
    label: &str,
) -> RenderResult<Retained<ProtocolObject<dyn MTLBuffer>>> {
    let bytes: &[u8] = bytemuck::cast_slice(data);
    let ptr = std::ptr::NonNull::new(bytes.as_ptr() as *mut _)
        .ok_or_else(|| RenderError::Other(format!("{label}: source pointer is null")))?;
    // SAFETY: the pointer and length describe the live `data` slice, and Metal
    // copies those bytes into the new buffer before the call returns.
    unsafe {
        device.newBufferWithBytes_length_options(
            ptr,
            bytes.len(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or_else(|| allocation_failed(label))
}

impl GrassState {
    pub(super) fn build(
        device: &ProtocolObject<dyn MTLDevice>,
        field: GrassField,
        sample_count: u32,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let blade_bytes = field.capacity as usize * size_of::<GpuGrassBlade>();
        let blades = device
            .newBufferWithLength_options(blade_bytes, MTLResourceOptions::StorageModePrivate)
            .ok_or_else(|| allocation_failed("grass blade buffer"))?;
        // Shared, so the CPU can zero it: the kernel then owns every word but
        // the instance count of the slot it fills first.
        let args = device
            .newBufferWithLength_options(GRASS_ARGS_BYTES, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| allocation_failed("grass args buffer"))?;
        // SAFETY: the buffer is shared storage of exactly GRASS_ARGS_BYTES and
        // no GPU work has been encoded against it yet.
        unsafe {
            std::ptr::write_bytes(args.contents().as_ptr().cast::<u8>(), 0, GRASS_ARGS_BYTES);
        }
        let heights = buffer_with(device, &field.buffers.heights, "grass terrain heights")?;
        let masks = buffer_with(device, field.buffers.bound_mask_words(), "grass masks")?;
        Ok(Self {
            field,
            pipelines: GrassPipelines::build(device, sample_count, hot_reload)?,
            blades,
            args,
            heights,
            masks,
            runs: 0,
        })
    }
}

fn build_prepass_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
) -> RenderResult<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    let vert = builtin_shaders::entry_function(device, &GRASS_PREPASS_VERT, hot_reload)?;
    let frag = builtin_shaders::entry_function(device, &GRASS_PREPASS_FRAG, hot_reload)?;
    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexFunction(Some(&vert));
    desc.setFragmentFunction(Some(&frag));
    desc.setRasterSampleCount(1);
    // SAFETY: plain descriptor property setters; the subscripted slots are ones this descriptor
    // declares.
    unsafe {
        let targets = desc.colorAttachments();
        for (i, format) in super::post::gbuffer::GBUFFER_FORMATS.iter().enumerate() {
            targets.objectAtIndexedSubscript(i).setPixelFormat(*format);
        }
    }
    desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
    device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .map_err(|e| RenderError::ShaderCompile(format!("grass pre-pass pipeline state: {e:?}")))
}

fn build_main_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    sample_count: u32,
    hot_reload: bool,
) -> RenderResult<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    let vert = builtin_shaders::entry_function(device, &GRASS_VERT, hot_reload)?;
    let frag = builtin_shaders::entry_function(device, &GRASS_FRAG, hot_reload)?;
    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexFunction(Some(&vert));
    desc.setFragmentFunction(Some(&frag));
    desc.setRasterSampleCount(sample_count as usize);
    // SAFETY: plain descriptor property setters; the subscripted slot is one this descriptor
    // declares.
    unsafe {
        desc.colorAttachments()
            .objectAtIndexedSubscript(0)
            .setPixelFormat(MTLPixelFormat::RGBA16Float);
    }
    desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
    device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .map_err(|e| RenderError::ShaderCompile(format!("grass pipeline state: {e:?}")))
}

impl MtlContext {
    // This frame's grass inputs for a camera at `cam_pos` seeing through the
    // unjittered `vp`, advancing the draw-argument slot. `None` when the world
    // grows no grass.
    pub(in crate::metal) fn prepare_grass_frame(
        &mut self,
        cam_pos: [f32; 3],
        vp: [[f32; 4]; 4],
    ) -> Option<GrassFrame> {
        let grass = self.grass.as_mut()?;
        let camera = GrassCamera {
            position: cam_pos,
            vp,
        };
        let frame = grass.field.frame(&camera, grass.runs);
        grass.runs = grass.runs.wrapping_add(1);
        Some(frame)
    }

    // Encode the `Grass` node: place, cull and append this frame's blades.
    pub(in crate::metal) fn encode_grass(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        frame: &GrassFrame,
    ) -> RenderResult<()> {
        let Some(grass) = &self.grass else {
            return Ok(());
        };
        let desc = MTLComputePassDescriptor::new();
        if let Some(t) = &self.diagnostics.pass_timing {
            t.attach_compute(&desc, super::pass_timing::PassId::Grass);
        }
        let enc = ScopedEncoder::new(
            cmd_buf
                .computeCommandEncoderWithDescriptor(&desc)
                .ok_or_else(|| RenderError::Other("failed to get grass compute encoder".into()))?,
            ns_string!("grass: generate"),
        );
        enc.set_pipeline(&grass.pipelines.generate);
        enc.set_value(&frame.params, KERNEL_PARAMS_INDEX);
        enc.set_buffer(&grass.blades, 0, KERNEL_BLADES_INDEX);
        enc.set_buffer(&grass.args, 0, KERNEL_ARGS_INDEX);
        enc.set_buffer(&grass.heights, 0, KERNEL_HEIGHTS_INDEX);
        enc.set_buffer(&grass.masks, 0, KERNEL_MASKS_INDEX);
        let [x, y, z] = frame.dispatch;
        let groups = MTLSize {
            width: x as usize,
            height: y as usize,
            depth: z as usize,
        };
        let threads = MTLSize {
            width: concinnity_core::render::grass::tiles::GRASS_GROUP_SIZE as usize,
            height: 1,
            depth: 1,
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
        Ok(())
    }

    // Draw the blades into the G-buffer pre-pass `enc` has open: the main
    // view rasterizes them, the pre-pass view gives their motion.
    pub(in crate::metal) fn encode_grass_prepass(
        &self,
        enc: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        frame: &GrassFrame,
        main: &ViewUniforms,
        gbuffer: &GBufferView,
    ) -> u32 {
        let Some(grass) = &self.grass else {
            return 0;
        };
        enc.pushDebugGroup(ns_string!("grass"));
        enc.set_pipeline(&grass.pipelines.prepass);
        enc.set_depth_stencil(&self.targets.depth_state);
        enc.set_vertex_value(main, 0);
        enc.set_vertex_value(gbuffer, prepass_buffers::VIEW);
        enc.set_fragment_value(gbuffer, prepass_buffers::VIEW);
        self.draw_grass(enc, grass, frame);
        enc.popDebugGroup();
        1
    }

    // Draw the lit blades into the main pass `enc` has open. The fragment reads
    // the lights, shadows, clusters and argument buffers the surfaces bound,
    // so this runs after them and only when they drew.
    pub(in crate::metal) fn encode_grass_main(
        &self,
        enc: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        frame: &GrassFrame,
        view: &ViewUniforms,
    ) -> u32 {
        let Some(grass) = &self.grass else {
            return 0;
        };
        enc.pushDebugGroup(ns_string!("grass"));
        enc.set_pipeline(&grass.pipelines.main);
        enc.set_depth_stencil(&self.targets.depth_state);
        enc.set_vertex_value(view, 0);
        self.draw_grass(enc, grass, frame);
        enc.popDebugGroup();
        1
    }

    // The one indirect draw both passes issue: a strip per visible blade,
    // two-sided.
    fn draw_grass(
        &self,
        enc: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        grass: &GrassState,
        frame: &GrassFrame,
    ) {
        enc.setCullMode(MTLCullMode::None);
        enc.set_vertex_value(&frame.params, PARAMS_INDEX);
        enc.set_vertex_buffer(&grass.blades, 0, BLADES_INDEX);
        // SAFETY: the args buffer holds GRASS_ARGS_SLOTS whole draw-argument
        // records and the offset names one of them; the kernel that filled it
        // was committed ahead of this pass, and hazard tracking orders the read.
        unsafe {
            enc.drawPrimitives_indirectBuffer_indirectBufferOffset(
                MTLPrimitiveType::TriangleStrip,
                &grass.args,
                grass_args_offset(frame.params.args_slot),
            );
        }
    }
}
