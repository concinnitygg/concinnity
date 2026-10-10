//! The grass field on Metal (see `concinnity_core::render::grass`): the bend
//! pass that relaxes and stamps the trampling field, the kernel that places
//! this frame's visible blades and, a second time, the nearest shadow
//! cascade's, and the indirect draws, one per detail level, that render them at
//! the tail of the G-buffer pre-pass and the main pass, plus the cascade's
//! depth-only draw. The view's draws ride the encoder their pass already has
//! open, after the surfaces bound the pass's view blocks, lights and argument
//! buffers, so grass adds only its own block and blade buffer.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::render::error::{RenderError, RenderResult};
pub(in crate::metal) use concinnity_core::render::grass::GrassFrame;
use concinnity_core::render::grass::bend::GRASS_BEND_WORDS;
use concinnity_core::render::grass::lod::GRASS_LOD_COUNT;
use concinnity_core::render::grass::{
    GrassBender, GrassCamera, GrassField, GrassFrameInputs, GrassHistory, GrassHiz, GrassPass,
    GrassShadowView,
};
use concinnity_core::render::shader_programs::metal::prepass_buffers;
use concinnity_core::render::shadow_bias;
use concinnity_core::render::uniforms::grass::{
    GRASS_ARGS_BYTES, GRASS_ARGS_STRIDE, GpuGrassBlade, grass_args_offset,
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
    self, GRASS_BEND, GRASS_FRAG, GRASS_GENERATE, GRASS_PREPASS_FRAG, GRASS_PREPASS_VERT,
    GRASS_SHADOW_VERT, GRASS_VERT,
};
use super::context::MtlContext;
use super::encode::{ComputeEncode, RenderEncode};
use super::error::allocation_failed;
use super::pass_timing::PassId;
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
const KERNEL_HIZ_TEXTURE_INDEX: usize = 5;
const KERNEL_BEND_INDEX: usize = 6;

// The bend pass's slots, and the cascade draw's.
const BEND_PARAMS_INDEX: usize = 0;
const BEND_FIELD_INDEX: usize = 1;
const SHADOW_PARAMS_INDEX: usize = 0;
const SHADOW_BLADES_INDEX: usize = 1;

pub(super) struct GrassPipelines {
    generate: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    bend: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    prepass: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    main: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    shadow: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
}

impl GrassPipelines {
    pub(super) fn build(
        device: &ProtocolObject<dyn MTLDevice>,
        sample_count: u32,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        Ok(Self {
            generate: builtin_shaders::compute_pipeline(device, &GRASS_GENERATE, hot_reload)?,
            bend: builtin_shaders::compute_pipeline(device, &GRASS_BEND, hot_reload)?,
            prepass: build_prepass_pipeline(device, hot_reload)?,
            main: build_main_pipeline(device, sample_count, hot_reload)?,
            shadow: build_shadow_pipeline(device, hot_reload)?,
        })
    }
}

// The grass field and what draws it: built once at init when the world grows
// one.
pub(super) struct GrassState {
    pub(super) field: GrassField,
    pub(super) pipelines: GrassPipelines,
    // The visible blades the kernel appends, `field.capacity` of them, each
    // detail level's region after the last. Written and read only on the GPU.
    blades: Retained<ProtocolObject<dyn MTLBuffer>>,
    // Two slots of non-indexed draw arguments, one per detail level; see
    // `GRASS_ARGS_SLOTS`.
    args: Retained<ProtocolObject<dyn MTLBuffer>>,
    // The nearest shadow cascade's blades, `field.shadow_capacity` of them, and
    // their draw arguments, laid out like the view's.
    shadow_blades: Retained<ProtocolObject<dyn MTLBuffer>>,
    shadow_args: Retained<ProtocolObject<dyn MTLBuffer>>,
    // The bend field's two halves, written and read only on the GPU.
    bend_field: Retained<ProtocolObject<dyn MTLBuffer>>,
    // Every terrain's heights and every layer's mask texels, which the kernel
    // reads to root and thin the blades.
    heights: Retained<ProtocolObject<dyn MTLBuffer>>,
    masks: Retained<ProtocolObject<dyn MTLBuffer>>,
    // What the field's frames carry from one to the next.
    history: GrassHistory,
}

// A shared args buffer, zeroed: the kernel then owns every word but the
// instance counts of the slot it fills first.
fn zeroed_args(
    device: &ProtocolObject<dyn MTLDevice>,
    label: &str,
) -> RenderResult<Retained<ProtocolObject<dyn MTLBuffer>>> {
    let args = device
        .newBufferWithLength_options(GRASS_ARGS_BYTES, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| allocation_failed(label))?;
    // SAFETY: the buffer is shared storage of exactly GRASS_ARGS_BYTES and no GPU
    // work has been encoded against it yet.
    unsafe {
        std::ptr::write_bytes(args.contents().as_ptr().cast::<u8>(), 0, GRASS_ARGS_BYTES);
    }
    Ok(args)
}

// A private buffer of `count` blades.
fn blade_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    count: u32,
    label: &str,
) -> RenderResult<Retained<ProtocolObject<dyn MTLBuffer>>> {
    let bytes = count as usize * size_of::<GpuGrassBlade>();
    device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModePrivate)
        .ok_or_else(|| allocation_failed(label))
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
        let blades = blade_buffer(device, field.capacity.total(), "grass blade buffer")?;
        let args = zeroed_args(device, "grass args buffer")?;
        let shadow_blades =
            blade_buffer(device, field.shadow_capacity, "grass shadow blade buffer")?;
        let shadow_args = zeroed_args(device, "grass shadow args buffer")?;
        // Each half is written whole before it is read, so it starts empty.
        let bend_field = device
            .newBufferWithLength_options(
                GRASS_BEND_WORDS * 4,
                MTLResourceOptions::StorageModePrivate,
            )
            .ok_or_else(|| allocation_failed("grass bend field"))?;
        let heights = buffer_with(device, &field.buffers.heights, "grass terrain heights")?;
        let masks = buffer_with(device, field.buffers.bound_mask_words(), "grass masks")?;
        Ok(Self {
            field,
            pipelines: GrassPipelines::build(device, sample_count, hot_reload)?,
            blades,
            args,
            shadow_blades,
            shadow_args,
            bend_field,
            heights,
            masks,
            history: GrassHistory::default(),
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

// The cascade's depth-only draw: no fragment stage, like every shadow caster.
fn build_shadow_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
) -> RenderResult<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    let vert = builtin_shaders::entry_function(device, &GRASS_SHADOW_VERT, hot_reload)?;
    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexFunction(Some(&vert));
    desc.setRasterSampleCount(1);
    desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
    device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .map_err(|e| RenderError::ShaderCompile(format!("grass shadow pipeline state: {e:?}")))
}

// What a frame's grass is prepared from.
pub(in crate::metal) struct GrassRequest<'a> {
    pub cam_pos: [f32; 3],
    // The unjittered view-projection.
    pub vp: [[f32; 4]; 4],
    pub elapsed: f32,
    // The grass casts into the nearest cascade this frame.
    pub cast: bool,
    pub benders: &'a [GrassBender],
}

impl MtlContext {
    // Whether this frame's grass casts into the nearest cascade: shadows are on
    // and that cascade re-renders.
    pub(in crate::metal) fn grass_casts(&self) -> bool {
        self.grass.is_some() && self.shadow.enabled && self.shadow.render_mask & 1 != 0
    }

    // This frame's grass work for `request`, advancing the field's history.
    // `None` when the world grows no grass.
    pub(in crate::metal) fn prepare_grass_frame(
        &mut self,
        request: GrassRequest<'_>,
    ) -> Option<GrassFrame> {
        let GrassRequest {
            cam_pos,
            vp,
            elapsed,
            cast,
            benders,
        } = request;
        let shadow = cast.then(|| GrassShadowView {
            vp: self.shadow.uniforms.light_vps[0],
            to_light: self.shadow.light_dir,
        });
        // The pyramid holds last frame's depth once the cull has seen a frame
        // through, and is tested through last frame's view-projection.
        let hiz = self
            .cull
            .hiz
            .as_ref()
            .filter(|_| self.cull.hiz_valid)
            .map(|h| GrassHiz {
                prev_vp: self.cull.prev_view_proj,
                size: [h.width as f32, h.height as f32],
                mip_count: h.mip_count,
            });
        let grass = self.grass.as_mut()?;
        let camera = GrassCamera {
            position: cam_pos,
            vp,
            hiz,
        };
        let inputs = GrassFrameInputs {
            camera,
            elapsed,
            shadow,
            benders,
        };
        Some(grass.field.frame(&inputs, &mut grass.history))
    }

    // A compute encoder timed as `pass`.
    fn grass_compute_encoder(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        pass: PassId,
    ) -> RenderResult<ScopedEncoder<dyn objc2_metal::MTLComputeCommandEncoder>> {
        let desc = MTLComputePassDescriptor::new();
        if let Some(t) = &self.diagnostics.pass_timing {
            t.attach_compute(&desc, pass);
        }
        let label = match pass {
            PassId::GrassBend => ns_string!("grass: bend"),
            PassId::GrassShadow => ns_string!("grass: shadow"),
            _ => ns_string!("grass: generate"),
        };
        Ok(ScopedEncoder::new(
            cmd_buf
                .computeCommandEncoderWithDescriptor(&desc)
                .ok_or_else(|| RenderError::Other("failed to get grass compute encoder".into()))?,
            label,
        ))
    }

    // Encode the `GrassBend` node: relax the bend field and stamp this frame's
    // footprints into it.
    pub(in crate::metal) fn encode_grass_bend(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        frame: &GrassFrame,
    ) -> RenderResult<()> {
        let Some(grass) = &self.grass else {
            return Ok(());
        };
        let enc = self.grass_compute_encoder(cmd_buf, PassId::GrassBend)?;
        enc.set_pipeline(&grass.pipelines.bend);
        enc.set_value(&frame.bend.params, BEND_PARAMS_INDEX);
        enc.set_buffer(&grass.bend_field, 0, BEND_FIELD_INDEX);
        dispatch_groups(&enc, frame.bend.dispatch);
        Ok(())
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
        let enc = self.grass_compute_encoder(cmd_buf, PassId::Grass)?;
        self.place_blades(&enc, grass, &frame.view, [&grass.blades, &grass.args]);
        Ok(())
    }

    // Encode the `GrassShadow` node: place the blades the nearest cascade
    // draws.
    pub(in crate::metal) fn encode_grass_shadow(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        frame: &GrassFrame,
    ) -> RenderResult<()> {
        let (Some(grass), Some(pass)) = (&self.grass, frame.shadow.as_ref()) else {
            return Ok(());
        };
        let enc = self.grass_compute_encoder(cmd_buf, PassId::GrassShadow)?;
        self.place_blades(
            &enc,
            grass,
            pass,
            [&grass.shadow_blades, &grass.shadow_args],
        );
        Ok(())
    }

    // One run of the blade kernel for `pass`, appending to `blades` and
    // filling `args`.
    fn place_blades(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        grass: &GrassState,
        pass: &GrassPass,
        [blades, args]: [&ProtocolObject<dyn MTLBuffer>; 2],
    ) {
        enc.set_pipeline(&grass.pipelines.generate);
        enc.set_value(&pass.params, KERNEL_PARAMS_INDEX);
        enc.set_buffer(blades, 0, KERNEL_BLADES_INDEX);
        enc.set_buffer(args, 0, KERNEL_ARGS_INDEX);
        enc.set_buffer(&grass.heights, 0, KERNEL_HEIGHTS_INDEX);
        enc.set_buffer(&grass.masks, 0, KERNEL_MASKS_INDEX);
        enc.set_buffer(&grass.bend_field, 0, KERNEL_BEND_INDEX);
        // Bound whenever it exists so the kernel's texture always resolves; the
        // block's `hiz_enabled` gates the reads.
        if let Some(hiz) = &self.cull.hiz {
            enc.set_texture(&hiz.texture, KERNEL_HIZ_TEXTURE_INDEX);
        }
        dispatch_groups(enc, pass.dispatch);
    }

    // Draw the cascade's blades into the shadow slice `enc` has open, with the
    // coarsest strip, after the rasterized casters.
    pub(in crate::metal) fn encode_grass_shadow_draw(
        &self,
        enc: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        frame: &GrassFrame,
    ) -> u32 {
        let (Some(grass), Some(pass)) = (&self.grass, frame.shadow.as_ref()) else {
            return 0;
        };
        enc.pushDebugGroup(ns_string!("grass shadow"));
        enc.set_pipeline(&grass.pipelines.shadow);
        enc.set_depth_stencil(&self.targets.depth_state);
        enc.setDepthBias_slopeScale_clamp(
            shadow_bias::RASTER_CONSTANT,
            shadow_bias::RASTER_SLOPE,
            shadow_bias::RASTER_CLAMP,
        );
        enc.setCullMode(MTLCullMode::None);
        enc.set_vertex_value(&pass.params, SHADOW_PARAMS_INDEX);
        enc.set_vertex_buffer(&grass.shadow_blades, 0, SHADOW_BLADES_INDEX);
        let coarsest =
            grass_args_offset(pass.params.args_slot) + (GRASS_LOD_COUNT - 1) * GRASS_ARGS_STRIDE;
        // SAFETY: the args buffer holds GRASS_ARGS_SLOTS whole slots of
        // GRASS_LOD_COUNT draw-argument records and the offset names the
        // coarsest record of one; the kernel that filled it was committed ahead
        // of this pass, and hazard tracking orders the read.
        unsafe {
            enc.drawPrimitives_indirectBuffer_indirectBufferOffset(
                MTLPrimitiveType::TriangleStrip,
                &grass.shadow_args,
                coarsest,
            );
        }
        enc.popDebugGroup();
        1
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
        GRASS_LOD_COUNT as u32
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
        GRASS_LOD_COUNT as u32
    }

    // The draws both passes issue, one per detail level: a strip per visible
    // blade, two-sided.
    fn draw_grass(
        &self,
        enc: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        grass: &GrassState,
        frame: &GrassFrame,
    ) {
        enc.setCullMode(MTLCullMode::None);
        enc.set_vertex_value(&frame.view.params, PARAMS_INDEX);
        enc.set_vertex_buffer(&grass.blades, 0, BLADES_INDEX);
        let slot = grass_args_offset(frame.view.params.args_slot);
        for lod in 0..GRASS_LOD_COUNT {
            // SAFETY: the args buffer holds GRASS_ARGS_SLOTS whole slots of
            // GRASS_LOD_COUNT draw-argument records and the offset names one of
            // them; the kernel that filled it was committed ahead of this pass,
            // and hazard tracking orders the read.
            unsafe {
                enc.drawPrimitives_indirectBuffer_indirectBufferOffset(
                    MTLPrimitiveType::TriangleStrip,
                    &grass.args,
                    slot + lod * GRASS_ARGS_STRIDE,
                );
            }
        }
    }
}

// Dispatch `groups` groups of the grass kernels' width.
fn dispatch_groups(
    enc: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    groups: [u32; 3],
) {
    let [x, y, z] = groups;
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
}
