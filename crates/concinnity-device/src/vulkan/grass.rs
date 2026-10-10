//! The grass field on Vulkan (see `concinnity_core::render::grass`): the bend
//! pass that relaxes and stamps the trampling field, the kernel that places
//! this frame's visible blades and, a second time, the nearest shadow
//! cascade's, the indirect draws, one per detail level, that render them at the
//! tail of the G-buffer pre-pass and the main pass, and the cascade's
//! depth-only draw inside the shadow pass.
//!
//! Each draw's pipeline layout is its pass's own with one more set, holding the
//! grass block and the blade buffer: set 2 under the main pass's global and
//! bindless sets, set 3 under the pre-pass's three. Sets 0 and up stay
//! compatible with what the surfaces bound, so the lit blades shade on the
//! lights, shadows and environment already in place. The kernel's set 1 is the
//! draw cull's Hi-Z read set, which holds the pyramid it tests tiles against.
//! The visible-blade and draw-argument buffers carry across frames; the graph
//! orders the kernel's writes against the draws that read them, this frame's
//! and the last.

use ash::vk;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::grass::bend::GRASS_BEND_WORDS;
use concinnity_core::render::grass::lod::GRASS_LOD_COUNT;
use concinnity_core::render::grass::{
    GrassBender, GrassCamera, GrassField, GrassFrameInputs, GrassHistory, GrassHiz, GrassPass,
    GrassShadowView,
};
use concinnity_core::render::pass_timing;
use concinnity_core::render::render_graph::PassId;
use concinnity_core::render::shadow_bias;
use concinnity_core::render::uniforms::grass::{
    GRASS_ARGS_BYTES, GRASS_ARGS_STRIDE, GpuGrassBlade, GrassBendParams, GrassParams,
    grass_args_offset,
};

use super::allocator::PooledBuffer;
use super::builtin_shaders::{self, CompileProgram};
use super::context::VkContext;
use super::descriptor_layout::{Binding, PoolSizes};
use super::owned::{
    OwnedDescriptorPool, OwnedPipeline, OwnedPipelineLayout, OwnedSetLayout, VkDevice,
};
use super::pipeline_desc::{
    Blend, Depth, DepthBias, GraphicsPipelineDesc, Raster, compute_pipeline,
};
use super::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use super::set_writes::SetWrites;
use super::texture::GpuUploadContext;

// The grass block and the blades, which only the draws' vertex stage reads.
fn draw_set_bindings() -> [Binding; 2] {
    use vk::DescriptorType as T;
    let vertex = vk::ShaderStageFlags::VERTEX;
    [
        (0, T::UNIFORM_BUFFER, vertex),
        (1, T::STORAGE_BUFFER, vertex),
    ]
}

// The kernel's block, the blades it appends, the draw arguments, the terrain
// heights and mask texels it reads, and the bend field.
fn kernel_set_bindings() -> [Binding; 6] {
    use vk::DescriptorType as T;
    let compute = vk::ShaderStageFlags::COMPUTE;
    [
        (0, T::UNIFORM_BUFFER, compute),
        (1, T::STORAGE_BUFFER, compute),
        (2, T::STORAGE_BUFFER, compute),
        (3, T::STORAGE_BUFFER, compute),
        (4, T::STORAGE_BUFFER, compute),
        (5, T::STORAGE_BUFFER, compute),
    ]
}

// The bend pass's block and the field it reads and writes.
fn bend_set_bindings() -> [Binding; 2] {
    use vk::DescriptorType as T;
    let compute = vk::ShaderStageFlags::COMPUTE;
    [
        (0, T::UNIFORM_BUFFER, compute),
        (1, T::STORAGE_BUFFER, compute),
    ]
}

fn pipeline_layout(
    device: &VkDevice,
    sets: &[vk::DescriptorSetLayout],
    label: &str,
) -> RenderResult<OwnedPipelineLayout> {
    device
        .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(sets))
        .map_err(|e| super::error::map_vk_result(e, label))
}

// The pass layouts the grass draws extend: the main pass's two sets, and the
// pre-pass's three; and the Hi-Z read set the kernel takes as its set 1.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct GrassPassLayouts<'a> {
    pub main: &'a [vk::DescriptorSetLayout],
    pub prepass: &'a [vk::DescriptorSetLayout],
    pub hiz_read: vk::DescriptorSetLayout,
}

// What a grass draw pipeline renders into.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct GrassTargets {
    pub main_render_pass: vk::RenderPass,
    pub msaa_samples: vk::SampleCountFlags,
    pub shadow_render_pass: vk::RenderPass,
    // The device's depth-bias clamp, as the shadow casters use it.
    pub shadow_bias_clamp: f32,
}

// The kernels, the lit draw and the cascade draw, rebuilt as a set on a shader
// reload. The pre-pass draw is built apart, once the G-buffer it renders into
// exists.
pub(in crate::vulkan) struct GrassPipelines {
    generate: OwnedPipeline,
    bend: OwnedPipeline,
    main: OwnedPipeline,
    shadow: OwnedPipeline,
}

impl GrassPipelines {
    fn build(
        device: &VkDevice,
        layouts: &GrassLayouts,
        targets: GrassTargets,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let cs = builtin_shaders::GRASS_GENERATE.compile(hot_reload)?;
        let generate = compute_pipeline(device, layouts.kernel.handle(), &cs, "grass generate")?;
        let cs = builtin_shaders::GRASS_BEND.compile(hot_reload)?;
        let bend = compute_pipeline(device, layouts.bend.handle(), &cs, "grass bend")?;
        let vs = builtin_shaders::GRASS_SHADOW_VERT.compile(hot_reload)?;
        let shadow = GraphicsPipelineDesc {
            frag: None,
            color_targets: &[],
            depth: Depth::write(),
            topology: vk::PrimitiveTopology::TRIANGLE_STRIP,
            raster: Raster {
                bias: Some(DepthBias {
                    constant: shadow_bias::RASTER_CONSTANT,
                    clamp: targets.shadow_bias_clamp,
                    slope: shadow_bias::RASTER_SLOPE,
                }),
                ..Raster::default()
            },
            ..GraphicsPipelineDesc::fullscreen(
                &vs,
                &[],
                layouts.shadow.handle(),
                targets.shadow_render_pass,
                &[],
            )
        }
        .build(device, "grass shadow")?;
        let vs = builtin_shaders::GRASS_VERT.compile(hot_reload)?;
        let fs = builtin_shaders::GRASS_FRAG.compile(hot_reload)?;
        let main = GraphicsPipelineDesc {
            depth: Depth::write(),
            topology: vk::PrimitiveTopology::TRIANGLE_STRIP,
            samples: targets.msaa_samples,
            ..GraphicsPipelineDesc::fullscreen(
                &vs,
                &fs,
                layouts.main.handle(),
                targets.main_render_pass,
                &[Blend::Opaque],
            )
        }
        .build(device, "grass")?;
        Ok(Self {
            generate,
            bend,
            main,
            shadow,
        })
    }
}

fn build_prepass_pipeline(
    device: &VkDevice,
    layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    hot_reload: bool,
) -> RenderResult<OwnedPipeline> {
    let vs = builtin_shaders::GRASS_PREPASS_VERT.compile(hot_reload)?;
    let fs = builtin_shaders::GRASS_PREPASS_FRAG.compile(hot_reload)?;
    GraphicsPipelineDesc {
        depth: Depth::write(),
        topology: vk::PrimitiveTopology::TRIANGLE_STRIP,
        ..GraphicsPipelineDesc::fullscreen(
            &vs,
            &fs,
            layout,
            render_pass,
            &super::post::gbuffer::PREPASS_TARGETS,
        )
    }
    .build(device, "grass prepass")
}

// The set and pipeline layouts the grass pipelines are built against. The
// cascade draw's set has the view draws' shape.
struct GrassLayouts {
    draw_set: OwnedSetLayout,
    kernel_set: OwnedSetLayout,
    bend_set: OwnedSetLayout,
    kernel: OwnedPipelineLayout,
    bend: OwnedPipelineLayout,
    main: OwnedPipelineLayout,
    prepass: OwnedPipelineLayout,
    shadow: OwnedPipelineLayout,
}

// The grass field and what draws it: built once at init when the world grows
// one and the GPU-driven main pass exists to draw it in.
pub(in crate::vulkan) struct GrassResources {
    pub(in crate::vulkan) field: GrassField,
    layouts: GrassLayouts,
    pipelines: GrassPipelines,
    // `None` until a G-buffer exists to render into.
    prepass: Option<OwnedPipeline>,
    // The visible blades the kernel appends, `field.capacity` of them, each
    // detail level's region after the last.
    pub(in crate::vulkan) blades: PooledBuffer,
    // Two slots of one draw per detail level; see `GRASS_ARGS_SLOTS`.
    pub(in crate::vulkan) args: PooledBuffer,
    // The nearest shadow cascade's blades and draw arguments, laid out like
    // the view's.
    pub(in crate::vulkan) shadow_blades: PooledBuffer,
    pub(in crate::vulkan) shadow_args: PooledBuffer,
    // The bend field's two halves.
    pub(in crate::vulkan) bend_field: PooledBuffer,
    // Every terrain's heights and every layer's mask texels, which the kernel
    // reads to root and thin the blades.
    _heights: PooledBuffer,
    _masks: PooledBuffer,
    // Per frame in flight, persistently mapped: the view's `GrassParams`, the
    // cascade's, and the bend pass's block.
    params: Vec<PooledBuffer>,
    shadow_params: Vec<PooledBuffer>,
    bend_params: Vec<PooledBuffer>,
    // Per frame: the view's draw and kernel sets, the cascade's, and the bend
    // pass's set, each over that frame's block.
    draw_sets: Vec<vk::DescriptorSet>,
    kernel_sets: Vec<vk::DescriptorSet>,
    shadow_draw_sets: Vec<vk::DescriptorSet>,
    shadow_kernel_sets: Vec<vk::DescriptorSet>,
    bend_sets: Vec<vk::DescriptorSet>,
    _pool: OwnedDescriptorPool,
    // What the field's frames carry from one to the next. Advanced from
    // `&self` on the render thread before the fan-out.
    history: std::cell::Cell<GrassHistory>,
}

// One host-visible uniform block of `bytes` per frame in flight.
fn uniform_blocks(
    alloc: &super::allocator::DeviceAllocator,
    bytes: u64,
    frames: usize,
) -> RenderResult<Vec<PooledBuffer>> {
    (0..frames)
        .map(|_| {
            alloc.create_buffer(
                bytes,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
        })
        .collect()
}

// A device-local buffer of `bytes` the kernels write, zeroed.
fn zeroed_storage(
    gpu: &GpuUploadContext,
    bytes: u64,
    usage: vk::BufferUsageFlags,
) -> RenderResult<PooledBuffer> {
    let buffer = gpu.alloc.create_buffer(
        bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | usage,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let handle = buffer.buffer();
    let device = gpu.device;
    super::texture::one_shot_submit(device, gpu.command_pool, gpu.queue, |cmd| {
        // SAFETY: `cmd` is a command buffer in the recording state, and the buffer is live and
        // created with TRANSFER_DST.
        unsafe { device.cmd_fill_buffer(cmd, handle, 0, bytes, 0) };
    })?;
    Ok(buffer)
}

impl GrassResources {
    pub(in crate::vulkan) fn build(
        gpu: GpuUploadContext,
        field: GrassField,
        passes: GrassPassLayouts<'_>,
        targets: GrassTargets,
        frames: usize,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let GpuUploadContext { alloc, device, .. } = gpu;
        let draw_set = create_descriptor_set_layout(device, &draw_set_bindings())?;
        let kernel_set = create_descriptor_set_layout(device, &kernel_set_bindings())?;
        let bend_set = create_descriptor_set_layout(device, &bend_set_bindings())?;
        let kernel = pipeline_layout(
            device,
            &[kernel_set.handle(), passes.hiz_read],
            "grass kernel layout",
        )?;
        let bend = pipeline_layout(device, &[bend_set.handle()], "grass bend layout")?;
        let shadow = pipeline_layout(device, &[draw_set.handle()], "grass shadow layout")?;
        let main_sets: Vec<_> = passes
            .main
            .iter()
            .copied()
            .chain([draw_set.handle()])
            .collect();
        let main = pipeline_layout(device, &main_sets, "grass layout")?;
        let prepass_sets: Vec<_> = passes
            .prepass
            .iter()
            .copied()
            .chain([draw_set.handle()])
            .collect();
        let prepass = pipeline_layout(device, &prepass_sets, "grass prepass layout")?;
        let layouts = GrassLayouts {
            draw_set,
            kernel_set,
            bend_set,
            kernel,
            bend,
            main,
            prepass,
            shadow,
        };
        let pipelines = GrassPipelines::build(device, &layouts, targets, hot_reload)?;

        let blade_bytes =
            u64::from(field.capacity.total()) * std::mem::size_of::<GpuGrassBlade>() as u64;
        let blades = alloc.create_buffer(
            blade_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let args_bytes = GRASS_ARGS_BYTES as u64;
        // The kernel owns every word but the instance counts of the slot it
        // fills first.
        let args = zeroed_storage(&gpu, args_bytes, vk::BufferUsageFlags::INDIRECT_BUFFER)?;
        let shadow_args = zeroed_storage(&gpu, args_bytes, vk::BufferUsageFlags::INDIRECT_BUFFER)?;
        let shadow_bytes =
            u64::from(field.shadow_capacity) * std::mem::size_of::<GpuGrassBlade>() as u64;
        let shadow_blades = alloc.create_buffer(
            shadow_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let bend_bytes = (GRASS_BEND_WORDS * 4) as u64;
        let bend_field = zeroed_storage(&gpu, bend_bytes, vk::BufferUsageFlags::empty())?;

        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let heights_data: &[u8] = bytemuck::cast_slice(&field.buffers.heights);
        let masks_data: &[u8] = bytemuck::cast_slice(field.buffers.bound_mask_words());
        let heights = super::decal::upload_static_buffer(
            alloc,
            device,
            gpu.command_pool,
            gpu.queue,
            heights_data,
            storage,
        )?;
        let masks = super::decal::upload_static_buffer(
            alloc,
            device,
            gpu.command_pool,
            gpu.queue,
            masks_data,
            storage,
        )?;

        let params_bytes = std::mem::size_of::<GrassParams>() as u64;
        let bend_params_bytes = std::mem::size_of::<GrassBendParams>() as u64;
        let params = uniform_blocks(alloc, params_bytes, frames)?;
        let shadow_params = uniform_blocks(alloc, params_bytes, frames)?;
        let bend_params = uniform_blocks(alloc, bend_params_bytes, frames)?;

        let n = frames as u32;
        let sizes = PoolSizes::default()
            .sets(&draw_set_bindings(), 2 * n)
            .sets(&kernel_set_bindings(), 2 * n)
            .sets(&bend_set_bindings(), n)
            .build();
        let pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(5 * n)
                    .pool_sizes(&sizes),
            )
            .map_err(|e| super::error::map_vk_result(e, "grass descriptor pool"))?;
        let sets = |layout: &OwnedSetLayout| {
            alloc_descriptor_sets(device, pool.handle(), &vec![layout.handle(); frames])
        };
        let draw_sets = sets(&layouts.draw_set)?;
        let kernel_sets = sets(&layouts.kernel_set)?;
        let shadow_draw_sets = sets(&layouts.draw_set)?;
        let shadow_kernel_sets = sets(&layouts.kernel_set)?;
        let bend_sets = sets(&layouts.bend_set)?;
        // One placement's draw and kernel sets: its block, its blades and its
        // draw arguments, over the shared terrain and bend field.
        let write_placement = |block: &PooledBuffer,
                               (blades, blades_bytes): (&PooledBuffer, u64),
                               args: &PooledBuffer,
                               draw: vk::DescriptorSet,
                               kernel: vk::DescriptorSet| {
            SetWrites::new(draw)
                .uniform_buffer(0, block.buffer(), params_bytes)
                .storage_buffer(1, blades.buffer(), blades_bytes)
                .apply(device);
            SetWrites::new(kernel)
                .uniform_buffer(0, block.buffer(), params_bytes)
                .storage_buffer(1, blades.buffer(), blades_bytes)
                .storage_buffer(2, args.buffer(), args_bytes)
                .storage_buffer(3, heights.buffer(), heights_data.len() as u64)
                .storage_buffer(4, masks.buffer(), masks_data.len() as u64)
                .storage_buffer(5, bend_field.buffer(), bend_bytes)
                .apply(device);
        };
        for f in 0..frames {
            write_placement(
                &params[f],
                (&blades, blade_bytes),
                &args,
                draw_sets[f],
                kernel_sets[f],
            );
            write_placement(
                &shadow_params[f],
                (&shadow_blades, shadow_bytes),
                &shadow_args,
                shadow_draw_sets[f],
                shadow_kernel_sets[f],
            );
            SetWrites::new(bend_sets[f])
                .uniform_buffer(0, bend_params[f].buffer(), bend_params_bytes)
                .storage_buffer(1, bend_field.buffer(), bend_bytes)
                .apply(device);
        }
        Ok(Self {
            field,
            layouts,
            pipelines,
            prepass: None,
            blades,
            args,
            shadow_blades,
            shadow_args,
            bend_field,
            _heights: heights,
            _masks: masks,
            params,
            shadow_params,
            bend_params,
            draw_sets,
            kernel_sets,
            shadow_draw_sets,
            shadow_kernel_sets,
            bend_sets,
            _pool: pool,
            history: std::cell::Cell::new(GrassHistory::default()),
        })
    }

    // Rebuild the kernel and the lit draw from the current shader sources.
    pub(in crate::vulkan) fn rebuild_pipelines(
        &self,
        device: &VkDevice,
        targets: GrassTargets,
        hot_reload: bool,
    ) -> RenderResult<GrassPipelines> {
        GrassPipelines::build(device, &self.layouts, targets, hot_reload)
    }

    pub(in crate::vulkan) fn swap_pipelines(&mut self, pipelines: GrassPipelines) {
        self.pipelines = pipelines;
    }

    // Build the pre-pass draw into the G-buffer's `render_pass` when it is
    // missing, or `rebuild` it anyway after a shader edit. A pipeline that
    // fails to build leaves the blades out of the G-buffer, never out of the
    // lit scene.
    pub(in crate::vulkan) fn sync_prepass(
        &mut self,
        device: &VkDevice,
        render_pass: vk::RenderPass,
        rebuild: bool,
        hot_reload: bool,
    ) {
        if self.prepass.is_some() && !rebuild {
            return;
        }
        match build_prepass_pipeline(
            device,
            self.layouts.prepass.handle(),
            render_pass,
            hot_reload,
        ) {
            Ok(pipeline) => self.prepass = Some(pipeline),
            Err(e) => tracing::warn!("grass: the G-buffer pre-pass draw failed to build: {e}"),
        }
    }
}

// One run of the blade kernel, as the frame's passes read it.
#[derive(Clone, Copy)]
struct GrassPlacement {
    args_offset: u64,
    dispatch: [u32; 3],
}

impl GrassPlacement {
    fn of(pass: &GrassPass) -> Self {
        Self {
            args_offset: grass_args_offset(pass.params.args_slot) as u64,
            dispatch: pass.dispatch,
        }
    }
}

// One frame's grass inputs, prepared before the fan-out.
pub(in crate::vulkan) struct GrassFrame {
    view: GrassPlacement,
    // The cascade's run, when the grass casts this frame.
    shadow: Option<GrassPlacement>,
    bend_dispatch: [u32; 3],
}

// What a frame's grass is prepared from.
pub(in crate::vulkan) struct GrassRequest<'a> {
    pub frame_idx: usize,
    pub cam_pos: [f32; 3],
    // The unjittered view-projection.
    pub vp: [[f32; 4]; 4],
    pub elapsed: f32,
    // The grass casts into the nearest cascade this frame.
    pub cast: bool,
    pub benders: &'a [GrassBender],
}

impl VkContext {
    // Build the grass pre-pass draw against the G-buffer once one exists, or
    // `rebuild` it after a shader edit.
    pub(in crate::vulkan) fn sync_grass_prepass(&mut self, rebuild: bool) {
        let Some(render_pass) = self
            .gbuffer
            .as_ref()
            .map(|gb| gb.prepass_render_pass.handle())
        else {
            return;
        };
        let hot_reload = self.hot_reload.enabled;
        if let Some(grass) = self.grass.as_mut() {
            grass.sync_prepass(&self.hw.device, render_pass, rebuild, hot_reload);
        }
    }

    // The targets the lit grass draw and the cascade draw render into.
    pub(in crate::vulkan) fn grass_targets(&self) -> GrassTargets {
        GrassTargets {
            main_render_pass: self.targets.main_render_pass.handle(),
            msaa_samples: self.targets.msaa_samples,
            shadow_render_pass: self.shadow.render_pass.handle(),
            shadow_bias_clamp: self.hw.device.depth_bias_clamp(),
        }
    }

    // Whether this frame's grass casts into the nearest cascade: shadows are on
    // and that cascade re-renders.
    pub(in crate::vulkan) fn grass_casts(&self) -> bool {
        self.grass.is_some() && self.shadow.enabled() && self.shadow.render_mask & 1 != 0
    }

    // Write this frame's grass blocks for `request`, advancing the field's
    // history. `None` when the world grows no grass.
    pub(in crate::vulkan) fn prepare_grass_frame(
        &self,
        request: GrassRequest<'_>,
    ) -> Option<GrassFrame> {
        let grass = self.grass.as_ref()?;
        let GrassRequest {
            frame_idx,
            cam_pos,
            vp,
            elapsed,
            cast,
            benders,
        } = request;
        // The pyramid holds last frame's depth once a pyramid at this
        // resolution has been built, tested through the view-projection the
        // draw cull tests through.
        let hiz = self
            .cull
            .hiz
            .as_ref()
            .filter(|_| self.cull.hiz_valid)
            .map(|h| GrassHiz {
                prev_vp: self.cull.hiz_prev_view_proj,
                size: [h.width as f32, h.height as f32],
                mip_count: h.mip_count,
            });
        let inputs = GrassFrameInputs {
            camera: GrassCamera {
                position: cam_pos,
                vp,
                hiz,
            },
            elapsed,
            shadow: cast.then(|| GrassShadowView {
                vp: self.shadow.uniforms.light_vps[0],
                to_light: self.shadow.light_dir,
            }),
            benders,
        };
        let mut history = grass.history.get();
        let frame = grass.field.frame(&inputs, &mut history);
        grass.history.set(history);
        grass.params[frame_idx].write_val(0, &frame.view.params);
        grass.bend_params[frame_idx].write_val(0, &frame.bend.params);
        let shadow = frame.shadow.map(|pass| {
            grass.shadow_params[frame_idx].write_val(0, &pass.params);
            GrassPlacement::of(&pass)
        });
        Some(GrassFrame {
            view: GrassPlacement::of(&frame.view),
            shadow,
            bend_dispatch: frame.bend.dispatch,
        })
    }

    // Encode the `GrassBend` node: relax the bend field and stamp this frame's
    // footprints into it.
    pub(in crate::vulkan) fn encode_grass_bend(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        frame: &GrassFrame,
    ) {
        let Some(grass) = &self.grass else {
            return;
        };
        let device = &self.hw.device;
        let [x, y, z] = frame.bend_dispatch;
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle these commands
        // name is live for the call.
        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                grass.pipelines.bend.handle(),
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                grass.layouts.bend.handle(),
                0,
                &[grass.bend_sets[frame_idx]],
                &[],
            );
            device.cmd_dispatch(cmd, x, y, z);
        }
    }

    // Encode the `GrassShadow` node: place the blades the nearest cascade
    // draws.
    pub(in crate::vulkan) fn encode_grass_shadow(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        frame: &GrassFrame,
    ) {
        let (Some(grass), Some(shadow)) = (&self.grass, frame.shadow.as_ref()) else {
            return;
        };
        self.place_blades(
            cmd,
            frame_idx,
            grass,
            shadow,
            grass.shadow_kernel_sets[frame_idx],
        );
    }

    // Draw the cascade's blades into the shadow slice `cmd` has begun, with
    // the coarsest strip.
    pub(in crate::vulkan) fn encode_grass_shadow_draw(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        frame: &GrassFrame,
    ) {
        let (Some(grass), Some(shadow)) = (&self.grass, frame.shadow.as_ref()) else {
            return;
        };
        let device = &self.hw.device;
        let coarsest = shadow.args_offset + ((GRASS_LOD_COUNT - 1) * GRASS_ARGS_STRIDE) as u64;
        self.timed(cmd, frame_idx, PassId::GrassShadowDraw, || {
            // SAFETY: `cmd` is a command buffer in the recording state inside the shadow render
            // pass the pipeline was built against; every handle these commands name is live for
            // the call, and the offset names one whole record of the args buffer.
            unsafe {
                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    grass.pipelines.shadow.handle(),
                );
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    grass.layouts.shadow.handle(),
                    0,
                    &[grass.shadow_draw_sets[frame_idx]],
                    &[],
                );
                device.cmd_draw_indirect(cmd, grass.shadow_args.buffer(), coarsest, 1, 0);
            }
        });
        self.inc_draw_calls(1);
    }

    // Encode the `Grass` node: place, cull and append this frame's blades.
    pub(in crate::vulkan) fn encode_grass(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        frame: &GrassFrame,
    ) {
        let Some(grass) = &self.grass else {
            return;
        };
        self.place_blades(
            cmd,
            frame_idx,
            grass,
            &frame.view,
            grass.kernel_sets[frame_idx],
        );
    }

    // One run of the blade kernel for `placement` over `kernel_set`.
    fn place_blades(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        grass: &GrassResources,
        placement: &GrassPlacement,
        kernel_set: vk::DescriptorSet,
    ) {
        // The kernel's layout takes the Hi-Z read set, which exists whenever
        // the grass does.
        let Some(hiz) = &self.cull.hiz else {
            return;
        };
        let device = &self.hw.device;
        let [x, y, z] = placement.dispatch;
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle these commands
        // name is live for the call.
        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                grass.pipelines.generate.handle(),
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                grass.layouts.kernel.handle(),
                0,
                &[kernel_set, hiz.read_sets[frame_idx]],
                &[],
            );
            device.cmd_dispatch(cmd, x, y, z);
        }
    }

    // Draw the blades into the G-buffer pre-pass `cmd` has begun, over the
    // pre-pass's sets: the main view rasterizes them, the pre-pass view gives
    // their motion.
    pub(in crate::vulkan) fn encode_grass_prepass(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        frame: &GrassFrame,
    ) {
        let (Some(grass), Some(&gbuffer_set)) =
            (&self.grass, self.cull.gbuffer_sets.get(frame_idx))
        else {
            return;
        };
        let Some(pipeline) = &grass.prepass else {
            return;
        };
        self.timed(cmd, frame_idx, PassId::GrassPrepass, || {
            self.draw_grass(
                cmd,
                grass,
                frame,
                (pipeline, grass.layouts.prepass.handle()),
                &[
                    self.descriptors.global_sets[frame_idx],
                    self.cull.bindless_sets[frame_idx],
                    gbuffer_set,
                    grass.draw_sets[frame_idx],
                ],
            );
        });
    }

    // Draw the lit blades into the main pass `cmd` has begun, over the main
    // pass's global and bindless sets.
    pub(in crate::vulkan) fn encode_grass_main(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        frame: &GrassFrame,
    ) {
        let Some(grass) = &self.grass else {
            return;
        };
        self.timed(cmd, frame_idx, PassId::GrassDraw, || {
            self.draw_grass(
                cmd,
                grass,
                frame,
                (&grass.pipelines.main, grass.layouts.main.handle()),
                &[
                    self.descriptors.global_sets[frame_idx],
                    self.cull.bindless_sets[frame_idx],
                    grass.draw_sets[frame_idx],
                ],
            );
        });
    }

    // The draws both passes issue, one per detail level: a strip per visible
    // blade.
    fn draw_grass(
        &self,
        cmd: vk::CommandBuffer,
        grass: &GrassResources,
        frame: &GrassFrame,
        (pipeline, layout): (&OwnedPipeline, vk::PipelineLayout),
        sets: &[vk::DescriptorSet],
    ) {
        let device = &self.hw.device;
        // SAFETY: `cmd` is a command buffer in the recording state inside a render pass the
        // pipeline is compatible with; every handle these commands name is live for the call, and
        // each offset names one whole record of the args buffer.
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.handle());
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                0,
                sets,
                &[],
            );
            for lod in 0..GRASS_LOD_COUNT {
                let offset = frame.view.args_offset + (lod * GRASS_ARGS_STRIDE) as u64;
                device.cmd_draw_indirect(cmd, grass.args.buffer(), offset, 1, 0);
            }
        }
        self.inc_draw_calls(GRASS_LOD_COUNT as u32);
    }

    // `record` bracketed by `pass`'s timestamps inside the enclosing pass.
    fn timed(&self, cmd: vk::CommandBuffer, frame_idx: usize, pass: PassId, record: impl FnOnce()) {
        let device = &self.hw.device;
        let stamps = self
            .hw
            .timestamp_query_pool
            .map(|pool| (pool, pass_timing::pass_pair(frame_idx, pass)));
        if let Some((pool, (start, _))) = stamps {
            // SAFETY: `cmd` is a command buffer in the recording state and the query pool is live;
            // the frame's query block was reset at the frame's start.
            unsafe {
                device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, pool, start)
            };
        }
        record();
        if let Some((pool, (_, end))) = stamps {
            // SAFETY: as above.
            unsafe {
                device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, pool, end)
            };
        }
    }
}
