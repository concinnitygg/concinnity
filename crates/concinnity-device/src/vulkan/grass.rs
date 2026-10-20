//! The grass field on Vulkan (see `concinnity_engine::render::environment::grass`): the bend
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
//! and the last. On a planet the kernel also reads the frame's drawn tiles,
//! from a host-visible buffer per frame in flight, and their meshes, from the
//! shared vertex buffer.

use ash::vk;
use concinnity_core::math::SaturatingFrom;
use concinnity_engine::planet::Rebase;
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::environment::grass::bend::GRASS_BEND_WORDS;
use concinnity_engine::render::environment::grass::lod::{
    GRASS_BAND_REGION, GRASS_LOD_COUNT, GRASS_REGION_COUNT,
};
use concinnity_engine::render::environment::grass::{
    FRAME_PLANET_TILE_RECORDS, FRAME_TILE_RECORDS, GrassBender, GrassCamera, GrassField,
    GrassFrameState, GrassPass, GrassShadowView, GrassView,
};
use concinnity_engine::render::render_graph::PassId;
use concinnity_engine::render::uniforms::grass::{
    GRASS_ARGS_BYTES, GRASS_ARGS_STRIDE, GpuGrassBlade, GrassBendParams, GrassParams, GrassTileGpu,
    PlanetGrassTileGpu, grass_args_offset,
};
use concinnity_engine::render::{backend::pass_timing, shadow::bias};

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
// heights and mask texels it reads, the bend field, the planet tiles, the
// shared vertex stream and the tile records.
fn kernel_set_bindings() -> [Binding; 9] {
    use vk::DescriptorType as T;
    let compute = vk::ShaderStageFlags::COMPUTE;
    [
        (0, T::UNIFORM_BUFFER, compute),
        (1, T::STORAGE_BUFFER, compute),
        (2, T::STORAGE_BUFFER, compute),
        (3, T::STORAGE_BUFFER, compute),
        (4, T::STORAGE_BUFFER, compute),
        (5, T::STORAGE_BUFFER, compute),
        (6, T::STORAGE_BUFFER, compute),
        (7, T::STORAGE_BUFFER, compute),
        (8, T::STORAGE_BUFFER, compute),
    ]
}

// Where the kernel set holds the planet tiles, the shared vertex stream and
// the tile records.
const KERNEL_PLANET_TILES_BINDING: u32 = 6;
const KERNEL_VERTICES_BINDING: u32 = 7;
const KERNEL_TILES_BINDING: u32 = 8;

// Planet tile and tile records one frame holds: the view's run and the
// cascade's.

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

// The kernels, the lit draws and the cascade draw, rebuilt as a set on a
// shader reload. The pre-pass draws are built apart, once the G-buffer they
// render into exists.
pub(in crate::vulkan) struct GrassPipelines {
    generate: OwnedPipeline,
    bend: OwnedPipeline,
    main: GrassDrawPipelines,
    shadow: OwnedPipeline,
}

// One pass's blade draws: every detail level's, and the band's, whose blades
// hand the field to the canopy pixel by pixel and so discard.
pub(in crate::vulkan) struct GrassDrawPipelines {
    levels: OwnedPipeline,
    band: OwnedPipeline,
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
                    constant: bias::RASTER_CONSTANT,
                    clamp: targets.shadow_bias_clamp,
                    slope: bias::RASTER_SLOPE,
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
        let lit = |fs: &[u8], label: &str| {
            GraphicsPipelineDesc {
                depth: Depth::write(),
                topology: vk::PrimitiveTopology::TRIANGLE_STRIP,
                samples: targets.msaa_samples,
                ..GraphicsPipelineDesc::fullscreen(
                    &vs,
                    fs,
                    layouts.main.handle(),
                    targets.main_render_pass,
                    &[Blend::Opaque],
                )
            }
            .build(device, label)
        };
        let main = GrassDrawPipelines {
            levels: lit(&builtin_shaders::GRASS_FRAG.compile(hot_reload)?, "grass")?,
            band: lit(
                &builtin_shaders::GRASS_CROSSFADE_FRAG.compile(hot_reload)?,
                "grass band",
            )?,
        };
        Ok(Self {
            generate,
            bend,
            main,
            shadow,
        })
    }
}

fn build_prepass_pipelines(
    device: &VkDevice,
    layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    hot_reload: bool,
) -> RenderResult<GrassDrawPipelines> {
    let vs = builtin_shaders::GRASS_PREPASS_VERT.compile(hot_reload)?;
    let build = |fs: &[u8], label: &str| {
        GraphicsPipelineDesc {
            depth: Depth::write(),
            topology: vk::PrimitiveTopology::TRIANGLE_STRIP,
            ..GraphicsPipelineDesc::fullscreen(
                &vs,
                fs,
                layout,
                render_pass,
                &super::post::gbuffer::PREPASS_TARGETS,
            )
        }
        .build(device, label)
    };
    Ok(GrassDrawPipelines {
        levels: build(
            &builtin_shaders::GRASS_PREPASS_FRAG.compile(hot_reload)?,
            "grass prepass",
        )?,
        band: build(
            &builtin_shaders::GRASS_PREPASS_CROSSFADE_FRAG.compile(hot_reload)?,
            "grass band prepass",
        )?,
    })
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
    prepass: Option<GrassDrawPipelines>,
    // The visible blades the kernel appends, `field.capacity()` of them, each
    // span after the last.
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
    // Per frame in flight, persistently mapped: the planet tile records and
    // the tile records both runs read.
    planet_tiles: Vec<PooledBuffer>,
    tiles: Vec<PooledBuffer>,
    // Per frame: the view's draw and kernel sets, the cascade's, and the bend
    // pass's set, each over that frame's block.
    draw_sets: Vec<vk::DescriptorSet>,
    kernel_sets: Vec<vk::DescriptorSet>,
    shadow_draw_sets: Vec<vk::DescriptorSet>,
    shadow_kernel_sets: Vec<vk::DescriptorSet>,
    bend_sets: Vec<vk::DescriptorSet>,
    // Per frame: whether its sets still name blade buffers replaced since it
    // last ran, rewritten once its fence shows it no longer runs.
    stale_blades: Vec<bool>,
    _pool: OwnedDescriptorPool,
    // What the field's frames carry from one to the next. Advanced from
    // `&self` on the render thread before the fan-out.
    state: std::cell::RefCell<GrassFrameState>,
}

// Bytes of a buffer of `count` blades, at least one.
fn blades_bytes(count: u32) -> u64 {
    u64::from(count.max(1)) * size_of::<GpuGrassBlade>() as u64
}

// A device-local buffer of `bytes` of blades.
fn blade_buffer(
    alloc: &super::allocator::DeviceAllocator,
    bytes: u64,
) -> RenderResult<PooledBuffer> {
    alloc.create_buffer(
        bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
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
    gpu: &GpuUploadContext<'_>,
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
    // The phase of the next frame's per-pixel hand-off to the canopy, which
    // the canopy block carries beside the blades' own block.
    pub(in crate::vulkan) fn crossfade_phase(&self) -> f32 {
        self.state.borrow().history.crossfade_phase()
    }

    pub(in crate::vulkan) fn build(
        gpu: GpuUploadContext<'_>,
        field: GrassField,
        passes: GrassPassLayouts<'_>,
        targets: GrassTargets,
        frames: usize,
        (vertices, vertex_bytes): (vk::Buffer, u64),
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

        let blade_bytes = blades_bytes(field.capacity().total());
        let blades = blade_buffer(alloc, blade_bytes)?;
        let args_bytes = GRASS_ARGS_BYTES as u64;
        // The kernel owns every word but the instance counts of the slot it
        // fills first.
        let args = zeroed_storage(&gpu, args_bytes, vk::BufferUsageFlags::INDIRECT_BUFFER)?;
        let shadow_args = zeroed_storage(&gpu, args_bytes, vk::BufferUsageFlags::INDIRECT_BUFFER)?;
        let shadow_bytes = blades_bytes(field.shadow_capacity());
        let shadow_blades = blade_buffer(alloc, shadow_bytes)?;
        let bend_bytes = (GRASS_BEND_WORDS * 4) as u64;
        let bend_field = zeroed_storage(&gpu, bend_bytes, vk::BufferUsageFlags::empty())?;

        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let heights_data: &[u8] = bytemuck::cast_slice(field.buffers.bound_heights());
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

        let params_bytes = size_of::<GrassParams>() as u64;
        let bend_params_bytes = size_of::<GrassBendParams>() as u64;
        let params = uniform_blocks(alloc, params_bytes, frames)?;
        let shadow_params = uniform_blocks(alloc, params_bytes, frames)?;
        let bend_params = uniform_blocks(alloc, bend_params_bytes, frames)?;
        let planet_bytes = (FRAME_PLANET_TILE_RECORDS * size_of::<PlanetGrassTileGpu>()) as u64;
        let planet_tiles = (0..frames)
            .map(|_| {
                alloc.create_buffer(
                    planet_bytes,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
            })
            .collect::<RenderResult<Vec<_>>>()?;
        let tile_bytes = (FRAME_TILE_RECORDS * size_of::<GrassTileGpu>()) as u64;
        let tiles = (0..frames)
            .map(|_| {
                alloc.create_buffer(
                    tile_bytes,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
            })
            .collect::<RenderResult<Vec<_>>>()?;

        let n = u32::saturating_from(frames);
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
        // draw arguments, over the shared terrain, bend field, planet tiles,
        // vertex stream and tile records.
        let write_placement = |block: &PooledBuffer,
                               (blades, blades_bytes): (&PooledBuffer, u64),
                               args: &PooledBuffer,
                               [planet, tiles]: [&PooledBuffer; 2],
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
                .storage_buffer(KERNEL_PLANET_TILES_BINDING, planet.buffer(), planet_bytes)
                .storage_buffer(KERNEL_VERTICES_BINDING, vertices, vertex_bytes)
                .storage_buffer(KERNEL_TILES_BINDING, tiles.buffer(), tile_bytes)
                .apply(device);
        };
        for f in 0..frames {
            write_placement(
                &params[f],
                (&blades, blade_bytes),
                &args,
                [&planet_tiles[f], &tiles[f]],
                draw_sets[f],
                kernel_sets[f],
            );
            write_placement(
                &shadow_params[f],
                (&shadow_blades, shadow_bytes),
                &shadow_args,
                [&planet_tiles[f], &tiles[f]],
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
            planet_tiles,
            tiles,
            draw_sets,
            kernel_sets,
            shadow_draw_sets,
            shadow_kernel_sets,
            bend_sets,
            stale_blades: vec![false; frames],
            _pool: pool,
            state: std::cell::RefCell::new(GrassFrameState::default()),
        })
    }

    // Size the visible-blade buffers for render targets `size` pixels across
    // and down, making them again when the blades they hold changed; the
    // field takes its new sizing only once both are made. A replaced buffer
    // is dropped into the allocator, which destroys it only once no frame in
    // flight can read it, and each frame's sets are pointed at the new ones
    // when that frame next runs.
    fn resize(
        &mut self,
        alloc: &super::allocator::DeviceAllocator,
        size: [u32; 2],
    ) -> RenderResult<()> {
        let Some(resize) = self.field.resize_for(size) else {
            return Ok(());
        };
        if resize.rebuild {
            let blades = blade_buffer(alloc, blades_bytes(resize.sizing.capacity.total()))?;
            let shadow_blades = blade_buffer(alloc, blades_bytes(resize.sizing.shadow_capacity))?;
            self.blades = blades;
            self.shadow_blades = shadow_blades;
            self.stale_blades.fill(true);
        }
        self.field.apply(resize);
        Ok(())
    }

    // Point frame `frame_idx`'s sets at the current blade buffers, if they
    // name replaced ones. The frame's fence has passed, so no submission in
    // flight reads its sets.
    fn refresh_blade_sets(&mut self, device: &VkDevice, frame_idx: usize) {
        let Some(stale) = self.stale_blades.get_mut(frame_idx) else {
            return;
        };
        if !std::mem::replace(stale, false) {
            return;
        }
        let bytes = blades_bytes(self.field.capacity().total());
        let shadow_bytes = blades_bytes(self.field.shadow_capacity());
        for set in [self.draw_sets[frame_idx], self.kernel_sets[frame_idx]] {
            SetWrites::new(set)
                .storage_buffer(1, self.blades.buffer(), bytes)
                .apply(device);
        }
        for set in [
            self.shadow_draw_sets[frame_idx],
            self.shadow_kernel_sets[frame_idx],
        ] {
            SetWrites::new(set)
                .storage_buffer(1, self.shadow_blades.buffer(), shadow_bytes)
                .apply(device);
        }
    }

    // Point every kernel set at the shared vertex stream `vertices`, after it
    // was replaced. The device is idle whenever the stream is.
    pub(in crate::vulkan) fn rewire_vertices(
        &self,
        device: &VkDevice,
        (vertices, vertex_bytes): (vk::Buffer, u64),
    ) {
        for &set in self.kernel_sets.iter().chain(&self.shadow_kernel_sets) {
            SetWrites::new(set)
                .storage_buffer(KERNEL_VERTICES_BINDING, vertices, vertex_bytes)
                .apply(device);
        }
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
        match build_prepass_pipelines(
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
#[derive(Clone, Copy)]
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
    // Size the grass's visible-blade buffers for render targets `size` pixels
    // across and down, and point frame `frame_idx`'s sets at them. Called
    // once the frame's fence has passed.
    pub(in crate::vulkan) fn size_grass(
        &mut self,
        size: [u32; 2],
        frame_idx: usize,
    ) -> RenderResult<()> {
        let Some(grass) = self.grass.as_mut() else {
            return Ok(());
        };
        grass.resize(&self.hw.alloc, size)?;
        grass.refresh_blade_sets(&self.hw.device, frame_idx);
        Ok(())
    }

    // Carry the grass into the frame the world moved to.
    pub(in crate::vulkan) fn rebase_grass(&self, rebase: &Rebase) {
        if let Some(grass) = &self.grass {
            grass.state.borrow_mut().history.rebase(rebase);
        }
    }

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
        let hiz = self.cull.hiz.as_ref().and_then(|h| {
            self.state
                .planner
                .temporal()
                .occlusion()
                .grass_hiz([h.width as f32, h.height as f32], h.mip_count)
        });
        let view = GrassView {
            camera: GrassCamera {
                position: cam_pos,
                vp,
                focal: self.grass_lod.map_or(1.0, |l| l.focal),
                hiz,
            },
            elapsed,
            shadow: cast.then(|| GrassShadowView {
                vp: self.state.planner.shadow.uniforms.light_vps[0],
                to_light: self.uniforms.lights.sun().direction,
            }),
            benders,
        };
        let mut state = grass.state.borrow_mut();
        let frame = state.frame(&grass.field, &self.state, &view);
        grass.params[frame_idx].write_val(0, &frame.view.params);
        grass.bend_params[frame_idx].write_val(0, &frame.bend.params);
        grass.planet_tiles[frame_idx].write_slice(0, frame.planet_tile_records());
        grass.tiles[frame_idx].write_slice(0, frame.tile_records());
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
        // SAFETY: `cmd` is recording outside a render pass, and the bend pipeline is owned by
        // `self.grass`, which outlives the frame.
        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                grass.pipelines.bend.handle(),
            );
        }
        // SAFETY: the frame's bend set is owned by `grass` and was written against its bend layout.
        unsafe {
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                grass.layouts.bend.handle(),
                0,
                &[grass.bend_sets[frame_idx]],
                &[],
            );
        }
        // SAFETY: the bend pipeline and its set are bound on this recording command buffer.
        unsafe { device.cmd_dispatch(cmd, x, y, z) };
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
            // SAFETY: `cmd` is recording inside the shadow render pass the pipeline was built
            // against, and the pipeline is owned by `grass`.
            unsafe {
                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    grass.pipelines.shadow.handle(),
                );
            }
            // SAFETY: the frame's shadow draw set is owned by `grass` and matches its shadow
            // layout.
            unsafe {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    grass.layouts.shadow.handle(),
                    0,
                    &[grass.shadow_draw_sets[frame_idx]],
                    &[],
                );
            }
            // SAFETY: the shadow args buffer is owned by `grass`, and `coarsest` names one whole
            // record of it.
            unsafe { device.cmd_draw_indirect(cmd, grass.shadow_args.buffer(), coarsest, 1, 0) };
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
        // SAFETY: `cmd` is recording outside a render pass, and the blade kernel is owned by
        // `grass`, which outlives the frame.
        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                grass.pipelines.generate.handle(),
            );
        }
        // SAFETY: `kernel_set` is one of `grass`'s live kernel sets, and the Hi-Z read set is
        // owned by `self.cull.hiz`; both match the kernel layout.
        unsafe {
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                grass.layouts.kernel.handle(),
                0,
                &[kernel_set, hiz.read_sets[frame_idx]],
                &[],
            );
        }
        // SAFETY: the blade kernel and both of its sets are bound on this recording buffer.
        unsafe { device.cmd_dispatch(cmd, x, y, z) };
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

    // The draws both passes issue, one per region: a strip per visible blade,
    // the band's under the pipeline that discards the pixels the canopy shows.
    fn draw_grass(
        &self,
        cmd: vk::CommandBuffer,
        grass: &GrassResources,
        frame: &GrassFrame,
        (pipelines, layout): (&GrassDrawPipelines, vk::PipelineLayout),
        sets: &[vk::DescriptorSet],
    ) {
        let device = &self.hw.device;
        // SAFETY: `cmd` is recording inside a render pass the level pipeline is compatible with,
        // and the pipeline is owned by `self.grass`.
        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pipelines.levels.handle(),
            );
        }
        // SAFETY: the caller passes live sets built against `layout`, borrowed for the call.
        unsafe {
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                0,
                sets,
                &[],
            );
        }
        for region in 0..GRASS_REGION_COUNT {
            if region == GRASS_BAND_REGION {
                // SAFETY: the band pipeline shares the level pipeline's layout and render pass, so
                // the sets bound above stay valid under it.
                unsafe {
                    device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipelines.band.handle(),
                    );
                }
            }
            let offset = frame.view.args_offset + (region * GRASS_ARGS_STRIDE) as u64;
            // SAFETY: the args buffer is owned by `grass`, and `offset` names one whole record
            // of it.
            unsafe { device.cmd_draw_indirect(cmd, grass.args.buffer(), offset, 1, 0) };
        }
        self.inc_draw_calls(u32::saturating_from(GRASS_REGION_COUNT));
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
                device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    pool,
                    start,
                );
            };
        }
        record();
        if let Some((pool, (_, end))) = stamps {
            // SAFETY: as above.
            unsafe {
                device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, pool, end);
            };
        }
    }
}
