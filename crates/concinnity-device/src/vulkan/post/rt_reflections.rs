// src/vulkan/post/rt_reflections.rs
//
// Hardware ray-traced reflection pass for the Vulkan backend. A fullscreen
// fragment pass that, per glossy pixel, rebuilds a world-space surface point +
// normal from the SSR pre-pass G-buffer, traces a reflection ray against the
// scene's top-level acceleration structure ([`crate::vulkan::raytrace`]) with
// inline `rayQueryEXT`, shades the hit (sun + IBL split-sum, optionally textured)
// or the IBL prefilter cube on a miss, and composites the result over the scene
// with the same Fresnel/gloss weighting SSR uses.
//
// It occupies the `SsrResolve` slot in the frame graph (reads the HDR scene,
// writes its own `output` target) and is mutually exclusive with the SSR
// resolve. Like SSGI it reuses the SSR depth + normal + roughness pre-pass
// G-buffer, so that pre-pass is forced on whenever RT reflections are enabled.
// Mirrors src/directx/post/rt_reflections.rs (DXR inline `RayQuery`); the GLSL
// is compiled with the Vulkan-1.2 / SPIR-V-1.4 target ray query needs.
//
// Unlike DirectX (which binds the TLAS + geometry table as root SRVs by GPU
// virtual address each frame), Vulkan binds them through a descriptor set, so
// `VkContext::rt_update_descriptors` re-points the current frame's set at the
// live TLAS + geometry-table handles every frame (they change on a dynamic
// rebuild; see `crate::vulkan::raytrace`).

use ash::vk;

use crate::vulkan::owned::{
    OwnedDescriptorPool, OwnedFramebuffer, OwnedPipeline, OwnedPipelineLayout, OwnedRenderPass,
    OwnedSampler, OwnedSetLayout, VkDevice,
};

use crate::gfx::render_types::RtParams;
use crate::gfx::rt_reflections::{RtParamsInputs, RtReflectionSettings};

use super::super::allocator::{DeviceAllocator, PooledBuffer};
use super::super::context::{HDR_FORMAT, VkContext};
use super::super::pipeline::*;
use super::super::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use super::super::texture::*;

// SPIR-V blobs for the RT pipelines. Produced by [`compile_rt_shaders`];
// consumed by `RtReflectionsResources::new` at init and by
// `rebuild_rt_pipelines` during shader hot-reload.
pub(in crate::vulkan) struct RtShaders {
    pub vs: Vec<u8>,
    pub flat_fs: Vec<u8>,
    // Textured fragment SPIR-V; `None` when the bindless texture pool is absent
    // (the flat-tint variant is then the only one built).
    pub textured_fs: Option<Vec<u8>>,
}

// Compile the shared fullscreen vertex stage + the flat fragment, plus the
// textured fragment when `pool_size > 0` (the bindless pool is live). slangc
// emits `SPV_KHR_ray_query` for the traversal, which the device already
// advertises wherever this pass is built.
pub(in crate::vulkan) fn compile_rt_shaders(
    hot_reload: bool,
    pool_size: usize,
    probe_cube_count: u32,
) -> Result<RtShaders, String> {
    use super::super::{builtins, slang_builtins};
    // The probe array length comes from the global set layout's binding-8
    // descriptor count; the pool declaration needs at least one slot even where
    // the textured variant is skipped.
    let ctx = builtins::Ctx {
        hot_reload,
        msaa: false,
        pool_size: pool_size.max(1),
        probe_count: probe_cube_count as usize,
    };
    let vs = slang_builtins::FULLSCREEN_VERT.compile(&ctx)?;
    let flat_fs = slang_builtins::RT_REFLECTIONS_FRAG.compile(&ctx)?;
    let textured_fs = if pool_size > 0 {
        Some(slang_builtins::RT_REFLECTIONS_FRAG_TEXTURED.compile(&ctx)?)
    } else {
        None
    };
    Ok(RtShaders {
        vs,
        flat_fs,
        textured_fs,
    })
}

// RT-reflection resources held by `VkContext` when `ray_traced_reflections` is
// on AND the GPU exposes the ray-query extensions AND the acceleration-structure
// build succeeds; otherwise the context leaves this `None` and the graph falls
// back to `SsrResolve`. All `vk::*` handles are owned here and freed on `destroy`.
pub(in crate::vulkan) struct RtReflectionsResources {
    // Resolved authored tunables; turned into a per-frame `RtParams` push.
    pub(in crate::vulkan) settings: RtReflectionSettings,

    // Reflection output: the HDR scene with reflections composited in. Becomes
    // the scene image the bloom / composite / TAA passes consume (a single
    // shared image, like the SSR resolve output). Owns its own slot because RT
    // can be authored with the SSR resolve off.
    pub(in crate::vulkan) output: GpuImage,
    render_pass: OwnedRenderPass,
    framebuffer: OwnedFramebuffer,

    _set_layout: OwnedSetLayout,
    // Flat (material-tint) layout = [set 0]; textured layout = [set 0, bindless
    // pool]. The textured layout/PSO are `Some` only when the bindless pool is
    // live (same gate as the bindless static pass).
    layout_flat: OwnedPipelineLayout,
    layout_textured: Option<OwnedPipelineLayout>,
    flat_pso: OwnedPipeline,
    textured_pso: Option<OwnedPipeline>,

    // Per-frame `RtParams` UBO (144 B), host-mapped.
    params_buffers: Vec<PooledBuffer>,

    _descriptor_pool: OwnedDescriptorPool,
    // Per-frame resolve sets: scene = that frame's HDR resolve, plus the shared
    // gbuffer / roughness / prefilter / verts / indices. The TLAS + geometry
    // table (bindings 1/2) are re-pointed every frame by `wire_dynamic`.
    resolve_sets: Vec<vk::DescriptorSet>,

    // Linear-clamp sampler the pass reads scene / G-buffer / roughness through.
    sampler: OwnedSampler,

    // A 1-element dummy storage buffer bound to the skinned-index SSBO (binding
    // 10) when the scene carries no skinned geometry (the accel data's skinned
    // index handle is then `vk::Buffer::null()`). Keeps the descriptor always
    // valid; the deformed-verts SSBO (binding 9) needs no dummy because the accel
    // data always holds a valid 1-element deformed buffer.
    dummy_ssbo: PooledBuffer,

    // Bindless texture-pool length, kept for the hot-reload recompile of the
    // textured variant.
    pool_size: usize,

    // Probe cube-array length the fragments were built against, kept so the
    // hot-reload recompile sizes `probe_cubes[]` to the same global set layout
    // the pipeline layouts already reference.
    probe_cube_count: u32,
}

// SAFETY: The params UBOs' mapped pointers are host-mapped, render-thread-only; the
// whole struct lives inside `VkContext`, which is already `unsafe impl Send`.
unsafe impl Send for RtReflectionsResources {}

// RT render pass: one HDR-format colour attachment (`output`), no depth. The
// fullscreen triangle overwrites every pixel so `DONT_CARE` is safe on load.
// Ends shader-readable for the bloom + composite passes. Mirrors the SSR resolve
// render pass.
fn create_rt_render_pass(device: &VkDevice) -> Result<OwnedRenderPass, String> {
    let attachment = vk::AttachmentDescription::default()
        .format(HDR_FORMAT)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let color_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_ref));
    // Broad SUBPASS_EXTERNAL dep: synchronise every prior colour write + shader
    // read (the main pass's hdr_resolve, the SSR pre-pass G-buffer / roughness,
    // and the start-buffer's acceleration-structure build) against this pass's
    // reads + write. Same shape as the SSR resolve dep.
    let dep = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::FRAGMENT_SHADER,
        )
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::SHADER_READ)
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::FRAGMENT_SHADER,
        )
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::SHADER_READ);
    let info = vk::RenderPassCreateInfo::default()
        .attachments(std::slice::from_ref(&attachment))
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(std::slice::from_ref(&dep));
    device
        .create_render_pass(&info)
        .map_err(|e| format!("RT reflections render pass: {e}"))
}

// Build one fullscreen RT pipeline. No vertex input (procedural fullscreen
// triangle); no depth; no blend; writes the HDR output. Mirrors the SSR resolve
// pipeline.
fn create_rt_pipeline(
    device: &VkDevice,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
    frag_spv: &[u8],
) -> Result<OwnedPipeline, String> {
    let vert_mod = spv_module(device, vert_spv)?;
    let frag_mod = spv_module(device, frag_spv)?;
    let entry = std::ffi::CString::new("main").unwrap();

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_mod.handle())
            .name(&entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_mod.handle())
            .name(&entry),
    ];
    let vert_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(false)
        .depth_write_enable(false)
        .depth_compare_op(vk::CompareOp::ALWAYS);
    let blend_attach = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(false);
    let blend = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(std::slice::from_ref(&blend_attach));
    let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyn_states);

    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vert_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);
    let pipeline = crate::vulkan::pipeline_cache::create_graphics_pipeline(device, &info)
        .map_err(|e| format!("create rt reflections pso: {e}"))?;
    Ok(pipeline)
}

// Replacement RT pipelines built by the hot-reload pass.
pub(in crate::vulkan) struct RebuiltRtPipelines {
    flat: OwnedPipeline,
    textured: Option<OwnedPipeline>,
}

// Rebuild the RT pipelines from disk-resident GLSL against the existing layouts +
// render pass. Same shape as `rebuild_ssr_pipelines`.
pub(in crate::vulkan) fn rebuild_rt_pipelines(
    device: &VkDevice,
    rt: &RtReflectionsResources,
    hot_reload: bool,
) -> Result<RebuiltRtPipelines, String> {
    let shaders = compile_rt_shaders(hot_reload, rt.pool_size, rt.probe_cube_count)?;
    let flat = create_rt_pipeline(
        device,
        rt.render_pass.handle(),
        rt.layout_flat.handle(),
        &shaders.vs,
        &shaders.flat_fs,
    )?;
    let textured = match (rt.layout_textured.as_ref(), &shaders.textured_fs) {
        (Some(layout), Some(fs)) => Some(create_rt_pipeline(
            device,
            rt.render_pass.handle(),
            layout.handle(),
            &shaders.vs,
            fs,
        )?),
        _ => None,
    };
    Ok(RebuiltRtPipelines { flat, textured })
}

// Allocation context for building RT resources: the device to allocate on, the
// output-target extent, and the number of frames in flight (per-frame UBOs +
// descriptor sets). Everything `create_buffer` / `create_image` / `build_targets`
// need to size and place the pass's GPU memory.
pub(in crate::vulkan) struct RtBuild<'a> {
    pub alloc: &'a DeviceAllocator,
    pub device: &'a VkDevice,
    pub width: u32,
    pub height: u32,
    pub frames: usize,
}

// The resolution-independent static resolve inputs the pass samples every frame:
// the scene vertex/index SSBOs, the per-frame HDR scene / G-buffer / roughness
// views, and the shared IBL prefilter cube. Wired by `wire_static` (at init and
// on resize) into every frame's set.
pub(in crate::vulkan) struct RtStaticInputs<'a> {
    pub vertex_buffer: vk::Buffer,
    pub index_buffer: vk::Buffer,
    pub hdr_resolve_views: &'a [vk::ImageView],
    pub gbuffer_views: &'a [vk::ImageView],
    pub roughness_views: &'a [vk::ImageView],
    pub prefilter_view: vk::ImageView,
    pub cube_sampler: vk::Sampler,
}

// The live acceleration-structure handles the trace binds per frame: the TLAS,
// the geometry table (buffer + byte size), the deformed skinned vertex buffer,
// and the skinned index buffer. All re-pointed each frame by `wire_dynamic`
// because a dynamic rebuild fresh-allocates them.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct RtAccelHandles {
    pub tlas: vk::AccelerationStructureKHR,
    pub geom_buffer: vk::Buffer,
    pub geom_size: vk::DeviceSize,
    pub deformed_verts: vk::Buffer,
    pub skinned_indices: vk::Buffer,
}

// Pipeline-layout / bindless configuration for building the RT pipelines: the
// optional bindless texture-pool layout + its length (enable the textured
// variant), the forward global set's layout (bound as set 1 for the
// reflection-probe miss fallback), and whether this is a hot-reload recompile.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct RtLayoutConfig {
    pub bindless_set_layout: Option<vk::DescriptorSetLayout>,
    // The forward global set's layout, bound as set 1 so the pass can sample the
    // reflection-probe set + cube array (binding 7/8) on a ray miss, and that
    // layout's binding-8 descriptor count (sizes the fragment's array).
    pub global_set_layout: vk::DescriptorSetLayout,
    pub probe_cube_count: u32,
    pub pool_size: usize,
    pub hot_reload: bool,
}

impl RtReflectionsResources {
    // Build every RT-reflection resource. Returns `Err` when the GLSL fails to
    // compile (the caller then falls back to SSR). `accel` holds the initial
    // acceleration-structure handles (re-pointed each frame thereafter);
    // `layout.bindless_set_layout` + `layout.pool_size` enable the textured variant.
    pub(in crate::vulkan) fn new(
        build: RtBuild,
        settings: RtReflectionSettings,
        static_inputs: RtStaticInputs,
        accel: RtAccelHandles,
        layout: RtLayoutConfig,
    ) -> Result<Self, String> {
        let RtBuild {
            alloc,
            device,
            width,
            height,
            frames,
        } = build;
        let RtStaticInputs {
            vertex_buffer,
            index_buffer,
            hdr_resolve_views,
            gbuffer_views,
            roughness_views,
            prefilter_view,
            cube_sampler,
        } = static_inputs;
        let RtAccelHandles {
            tlas,
            geom_buffer,
            geom_size,
            deformed_verts,
            skinned_indices,
        } = accel;
        let RtLayoutConfig {
            bindless_set_layout,
            global_set_layout,
            probe_cube_count,
            pool_size,
            hot_reload,
        } = layout;
        let render_pass = create_rt_render_pass(device)?;

        // set 0: RtParams UBO, TLAS, geom table, verts, indices, scene, gbuffer,
        // roughness, prefilter cube.
        let set_layout = create_descriptor_set_layout(
            device,
            &[
                (
                    0,
                    vk::DescriptorType::UNIFORM_BUFFER,
                    vk::ShaderStageFlags::FRAGMENT,
                ),
                (
                    1,
                    vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                    vk::ShaderStageFlags::FRAGMENT,
                ),
                (
                    2,
                    vk::DescriptorType::STORAGE_BUFFER,
                    vk::ShaderStageFlags::FRAGMENT,
                ),
                (
                    3,
                    vk::DescriptorType::STORAGE_BUFFER,
                    vk::ShaderStageFlags::FRAGMENT,
                ),
                (
                    4,
                    vk::DescriptorType::STORAGE_BUFFER,
                    vk::ShaderStageFlags::FRAGMENT,
                ),
                (
                    5,
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    vk::ShaderStageFlags::FRAGMENT,
                ),
                (
                    6,
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    vk::ShaderStageFlags::FRAGMENT,
                ),
                (
                    7,
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    vk::ShaderStageFlags::FRAGMENT,
                ),
                (
                    8,
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    vk::ShaderStageFlags::FRAGMENT,
                ),
                // 9/10: the deformed (posed) skinned vertex buffer + the
                // skinned index buffer, for skinned hits. Both are re-pointed per
                // frame by `wire_dynamic` (the deformed buffer is fresh per
                // rebuild); a dummy SSBO binds when there is no skinned geometry.
                (
                    9,
                    vk::DescriptorType::STORAGE_BUFFER,
                    vk::ShaderStageFlags::FRAGMENT,
                ),
                (
                    10,
                    vk::DescriptorType::STORAGE_BUFFER,
                    vk::ShaderStageFlags::FRAGMENT,
                ),
            ],
        )?;

        // set 0 = the RT resolve set; set 1 = the global set (probe set/cubes). The
        // textured variant adds the bindless pool as set 2 (kept past the global set
        // so probe_common's set index stays a fixed 1 across both variants).
        let flat_layouts = [set_layout.handle(), global_set_layout];
        let layout_flat = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&flat_layouts),
            )
            .map_err(|e| format!("rt flat pipeline layout: {e}"))?;
        let layout_textured = if let Some(bsl) = bindless_set_layout {
            let layouts = [set_layout.handle(), global_set_layout, bsl];
            Some(
                device
                    .create_pipeline_layout(
                        &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                    )
                    .map_err(|e| format!("rt textured pipeline layout: {e}"))?,
            )
        } else {
            None
        };

        let shaders = compile_rt_shaders(hot_reload, pool_size, probe_cube_count)?;
        let flat_pso = create_rt_pipeline(
            device,
            render_pass.handle(),
            layout_flat.handle(),
            &shaders.vs,
            &shaders.flat_fs,
        )?;
        let textured_pso = match (layout_textured.as_ref(), &shaders.textured_fs) {
            (Some(layout), Some(fs)) => Some(create_rt_pipeline(
                device,
                render_pass.handle(),
                layout.handle(),
                &shaders.vs,
                fs,
            )?),
            _ => None,
        };

        // Per-frame RtParams UBO.
        let params_size = std::mem::size_of::<RtParams>() as vk::DeviceSize;
        let mut params_buffers = Vec::with_capacity(frames);
        for _ in 0..frames {
            let buf = alloc.create_buffer(
                params_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            params_buffers.push(buf);
        }

        // Pool: per-frame sets, each with 1 UBO + 1 TLAS + 5 SSBO (geom table,
        // verts, indices, deformed skinned verts, skinned indices) + 4 samplers.
        let f = frames as u32;
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(f),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(f),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(f * 5),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(f * 4),
        ];
        let descriptor_pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&pool_sizes)
                    .max_sets(f),
            )
            .map_err(|e| format!("rt descriptor pool: {e}"))?;
        let layouts: Vec<_> = (0..frames).map(|_| set_layout.handle()).collect();
        let resolve_sets = alloc_descriptor_sets(device, descriptor_pool.handle(), &layouts)?;

        let sampler = create_sampler_linear_clamp(device)?;

        // 1-element dummy storage buffer for the skinned-index binding when there
        // is no skinned geometry.
        let dummy_ssbo = alloc.create_buffer(
            16,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let mut me = Self {
            settings,
            output: GpuImage::null(),
            render_pass,
            framebuffer: OwnedFramebuffer::null(),
            _set_layout: set_layout,
            layout_flat,
            layout_textured,
            flat_pso,
            textured_pso,
            params_buffers,
            _descriptor_pool: descriptor_pool,
            resolve_sets,
            sampler,
            dummy_ssbo,
            pool_size,
            probe_cube_count,
        };
        me.build_targets(alloc, device, width, height)?;
        me.wire_static(
            device,
            RtStaticInputs {
                vertex_buffer,
                index_buffer,
                hdr_resolve_views,
                gbuffer_views,
                roughness_views,
                prefilter_view,
                cube_sampler,
            },
        );
        for i in 0..frames {
            me.wire_dynamic(
                device,
                i,
                RtAccelHandles {
                    tlas,
                    geom_buffer,
                    geom_size,
                    deformed_verts,
                    skinned_indices,
                },
            );
        }
        Ok(me)
    }

    // Allocate / re-allocate the resolution-dependent output target + framebuffer.
    fn build_targets(
        &mut self,
        alloc: &DeviceAllocator,
        device: &VkDevice,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        let w = width.max(1);
        let h = height.max(1);
        let pooled = create_image(
            alloc,
            &ImageSpec {
                width: w,
                height: h,
                format: HDR_FORMAT,
                tiling: vk::ImageTiling::OPTIMAL,
                // TRANSFER_SRC so the transparent (glass) pass can snapshot the
                // post-RT scene for its refraction tap, the same usage the SSR output
                // carries when SSR owns the scene image.
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_SRC,
                mem_props: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                samples: vk::SampleCountFlags::TYPE_1,
            },
        )?;
        let image = pooled.image();
        let view = create_image_view(device, image, HDR_FORMAT, vk::ImageAspectFlags::COLOR)?;
        self.output = GpuImage::from_pooled(pooled, view);
        self.framebuffer = device
            .create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(self.render_pass.handle())
                    .attachments(std::slice::from_ref(&self.output.view))
                    .width(w)
                    .height(h)
                    .layers(1),
            )
            .map_err(|e| format!("rt framebuffer: {e}"))?;
        Ok(())
    }

    // Re-point every frame's shared static verts (3) + u32 indices (4) at the
    // given buffers. Called by `wire_static`, and again on its own when an asset
    // hot-reload replaces the shared geometry buffers under the pass (the sets
    // would otherwise keep descriptors on destroyed buffers).
    pub(in crate::vulkan) fn rewire_geometry(
        &self,
        device: &VkDevice,
        vertex_buffer: vk::Buffer,
        index_buffer: vk::Buffer,
    ) {
        let verts_info = vk::DescriptorBufferInfo::default()
            .buffer(vertex_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let indices_info = vk::DescriptorBufferInfo::default()
            .buffer(index_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        for &set in &self.resolve_sets {
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&verts_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(4)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&indices_info)),
            ];
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
    }

    // Wire the static per-frame bindings (UBO, verts, indices, scene, gbuffer,
    // roughness, prefilter). The TLAS + geom table (bindings 1/2) are wired by
    // `wire_dynamic`. Called at init + on swapchain resize.
    //
    // `gbuffer_views` / `roughness_views` carry the unified G-buffer pre-pass's
    // per-frame normal+depth / roughness views; resolve set `i` binds slot `i`.
    // A single-entry slice is shared across frames (the legacy SSR pre-pass
    // G-buffer). RT reuses the same byte-identical G-buffer the separate SSR
    // pre-pass produced, so the trace maths is unchanged.
    pub(in crate::vulkan) fn wire_static(&self, device: &VkDevice, inputs: RtStaticInputs) {
        let RtStaticInputs {
            vertex_buffer,
            index_buffer,
            hdr_resolve_views,
            gbuffer_views,
            roughness_views,
            prefilter_view,
            cube_sampler,
        } = inputs;
        self.rewire_geometry(device, vertex_buffer, index_buffer);
        let cube_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(prefilter_view)
            .sampler(cube_sampler);
        for (i, &set) in self.resolve_sets.iter().enumerate() {
            let gb_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(gbuffer_views[i % gbuffer_views.len().max(1)])
                .sampler(self.sampler.handle());
            let rough_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(roughness_views[i % roughness_views.len().max(1)])
                .sampler(self.sampler.handle());
            let ubo_info = vk::DescriptorBufferInfo::default()
                .buffer(self.params_buffers[i].buffer())
                .offset(0)
                .range(std::mem::size_of::<RtParams>() as vk::DeviceSize);
            let scene_view = hdr_resolve_views[i % hdr_resolve_views.len().max(1)];
            let scene_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(scene_view)
                .sampler(self.sampler.handle());
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::slice::from_ref(&ubo_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(5)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&scene_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(6)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&gb_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(7)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&rough_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(8)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&cube_info)),
            ];
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
    }

    // Re-point one frame's TLAS (binding 1), geometry-table (binding 2), deformed
    // skinned verts (binding 9), and skinned indices (binding 10) descriptors at
    // the live handles. Called every frame because a dynamic rebuild
    // fresh-allocates the TLAS / geom table / deformed buffer; the current frame's
    // set is fence-gated (its previous submission completed at the top of
    // `draw_frame`), so the update is safe. `deformed` is always a valid handle
    // (the accel data holds a 1-element dummy when there is no skinned geometry);
    // `skinned_indices` is `vk::Buffer::null()` until the first skinned rebuild,
    // in which case the 1-element dummy SSBO is bound so the descriptor stays
    // valid.
    pub(in crate::vulkan) fn wire_dynamic(
        &self,
        device: &VkDevice,
        frame_idx: usize,
        accel: RtAccelHandles,
    ) {
        let RtAccelHandles {
            tlas,
            geom_buffer,
            geom_size,
            deformed_verts: deformed,
            skinned_indices,
        } = accel;
        let set = self.resolve_sets[frame_idx];
        let accels = [tlas];
        let mut accel_write = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(&accels);
        let mut tlas_write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .push_next(&mut accel_write);
        // `push_next` does not set the count for an acceleration-structure write.
        tlas_write.descriptor_count = 1;
        let geom_info = vk::DescriptorBufferInfo::default()
            .buffer(geom_buffer)
            .offset(0)
            .range(geom_size);
        let geom_write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&geom_info));
        let deformed_info = vk::DescriptorBufferInfo::default()
            .buffer(deformed)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let deformed_write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(9)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&deformed_info));
        let sidx_buffer = if skinned_indices != vk::Buffer::null() {
            skinned_indices
        } else {
            self.dummy_ssbo.buffer()
        };
        let sidx_info = vk::DescriptorBufferInfo::default()
            .buffer(sidx_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let sidx_write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(10)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&sidx_info));
        // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and every
        // set and resource it names belongs to this device.
        unsafe {
            device
                .update_descriptor_sets(&[tlas_write, geom_write, deformed_write, sidx_write], &[])
        };
    }

    // Re-point every frame's prefilter-cube binding (binding 8) at a new IBL
    // prefilter view. Called by `update_environment_map` after an EnvironmentMap
    // hot-reload recreates the cubes (which destroys the old view the RT sets
    // captured); without this the next trace samples a dangling cube view and
    // loses the device. Mirrors the SSR resolve / raymarch cube re-wires.
    pub(in crate::vulkan) fn rewire_prefilter(
        &self,
        device: &VkDevice,
        prefilter_view: vk::ImageView,
        cube_sampler: vk::Sampler,
    ) {
        let cube_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(prefilter_view)
            .sampler(cube_sampler);
        for &set in &self.resolve_sets {
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(8)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&cube_info));
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
        }
    }

    fn destroy_targets(&mut self, _device: &VkDevice) {
        if !self.framebuffer.is_null() {
            // SAFETY: the handle was created from this device and is destroyed exactly once; the
            // caller has already waited for the device to go idle, so no submission still
            // references it.
            self.framebuffer = OwnedFramebuffer::null();
        }
        if self.output.image != vk::Image::null() {
            self.output = GpuImage::null();
        }
    }

    // Rebuild the resolution-dependent output target at a new extent and re-wire
    // the static descriptors (the gbuffer / roughness / scene views all moved).
    // The TLAS + geom table are resolution-independent; the caller re-points them
    // per frame as usual.
    pub(in crate::vulkan) fn rebuild(
        &mut self,
        alloc: &DeviceAllocator,
        device: &VkDevice,
        width: u32,
        height: u32,
        inputs: RtStaticInputs,
    ) -> Result<(), String> {
        self.destroy_targets(device);
        self.build_targets(alloc, device, width, height)?;
        self.wire_static(device, inputs);
        Ok(())
    }

    // Swap freshly-built pipelines into the live resources after a hot-reload.
    pub(in crate::vulkan) fn swap_pipelines(&mut self, rebuilt: RebuiltRtPipelines) {
        self.flat_pso = rebuilt.flat;
        self.textured_pso = rebuilt.textured;
    }

    // Destroy every RT resource. The caller has already idled the device.
    pub(in crate::vulkan) fn destroy(&mut self, device: &VkDevice) {
        self.destroy_targets(device);
        self.dummy_ssbo = PooledBuffer::null();
        self.params_buffers.clear();
    }
}

impl VkContext {
    // True when hardware ray-traced reflections are live (both the pass + the
    // acceleration structure built). Gates `FrameGraphInputs::rt_reflections_enabled`
    // (so the graph emits `RtReflections` in the `SsrResolve` slot) and the
    // post-stack scene-image routing. Mirrors `DxContext::rt_reflections_active`.
    pub(in crate::vulkan) fn rt_reflections_active(&self) -> bool {
        self.rt_reflections.is_some() && self.rt_accel.is_some()
    }

    // True when the glass pass should trace per-pixel RT reflections this frame:
    // RT is live (the scene TLAS is built) AND the glass RT pipelines compiled at
    // init. Single-sources the glass encoder's RT-vs-base selection and the
    // `graph_exec` planar skip, so the two always agree -- gating the skip on
    // `rt_reflections_active()` alone would drop the planar re-render even when the
    // glass RT pipelines failed to build, leaving the glass fallback sampling a
    // stale resolve. Mirrors `DxContext::rt_glass_active`.
    pub(in crate::vulkan) fn rt_glass_active(&self) -> bool {
        self.rt_reflections_active() && self.glass.as_ref().is_some_and(|g| g.rt_pipelines_ready())
    }

    // Run the per-frame dynamic acceleration-structure update on `cmd` (the
    // frame's "start" command buffer, submitted before every per-pass trace),
    // then re-point this frame's RT descriptor set at the live TLAS + geometry
    // table. A no-op when RT reflections are off. The descriptor rewrite happens
    // every frame (not only on a rebuild) so a frame that did not rebuild still
    // binds the current handles rather than a stale / retired one.
    pub(in crate::vulkan) fn rt_dynamic_update(
        &mut self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
    ) {
        // Consumed by the accel's `dynamic_update`, which folds a runtime draw-set
        // change (cloned prop, streamed chunk added/removed) into the BLAS head.
        // (Vulkan builds `rt_accel` + `rt_reflections` together, so an RT-enabled
        // scene that was empty at build time has both `None` and RT stays off until
        // a quality re-toggle rebuilds the pass; there is no seed-from-empty here.)
        let topology_dirty = std::mem::take(&mut self.rt_topology_dirty);
        if self.rt_accel.is_none() || self.rt_reflections.is_none() {
            return;
        }
        let device = self.device.clone();
        let instance = self.instance.clone();
        let pd = self.physical_device;
        let mode = self.rt_dynamic_mode;

        // Assemble this frame's skinned-geometry inputs while `self` is still
        // fully borrowable: the shared skinned VB/IB handles. `None` when there is
        // no skinned geometry resident (the static path runs). Read up-front so
        // they do not overlap the `rt_accel` mutable borrow below; the per-object
        // joint palettes are borrowed straight out of this frame's slot instead of
        // being collected into a per-frame list.
        let skinned_inputs: Option<(vk::Buffer, vk::Buffer)> =
            if !self.skinned.draw_objects.is_empty()
                && !self.skinned.vertex_buffer.is_null()
                && !self.skinned.index_buffer.is_null()
            {
                Some((
                    self.skinned.vertex_buffer.buffer(),
                    self.skinned.index_buffer.buffer(),
                ))
            } else {
                None
            };

        // Take `rt_accel` out so its `&mut` borrow does not overlap the shared
        // `&self` reads (`skinned_draw_objects` / `draw_objects`) the inputs need;
        // put it back immediately after.
        if let Some(mut accel) = self.rt_accel.take() {
            let joint_buffers = self
                .skinned
                .joint_buffers
                .get(frame_idx)
                .map(|b| b.as_slice())
                .unwrap_or(&[]);
            let skinned = skinned_inputs.map(|(vb, ib)| super::super::raytrace::SkinnedRtInputs {
                objects: &self.skinned.draw_objects,
                vertex_buffer: vb,
                index_buffer: ib,
                joint_buffers,
            });
            accel.dynamic_update(
                super::super::raytrace::RtDeviceCtx {
                    alloc: &self.alloc,
                    instance: &instance,
                    device: &device,
                    pd,
                },
                cmd,
                &self.draw.objects,
                super::super::raytrace::RtDynamicInputs {
                    policy: super::super::raytrace::RtRebuildPolicy {
                        mode,
                        topology_dirty,
                    },
                    frame_idx,
                    skinned,
                },
            );
            self.rt_accel = Some(accel);
        }
        let accel = self
            .rt_accel
            .as_ref()
            .expect("RT acceleration structures are live");
        let (geom_buffer, geom_size) = accel.geom_table();
        let tlas = accel.tlas();
        let deformed = accel.deformed_verts();
        let skinned_indices = accel.skinned_indices();
        let rt = self
            .rt_reflections
            .as_ref()
            .expect("RT reflection resources are live");
        rt.wire_dynamic(
            &device,
            frame_idx,
            RtAccelHandles {
                tlas,
                geom_buffer,
                geom_size,
                deformed_verts: deformed,
                skinned_indices,
            },
        );
        // Re-point the glass pass's RT descriptor ring at the same live handles, so
        // a glass trace this frame samples the current TLAS / geometry table. A
        // no-op when the world has no glass or the glass RT pipelines are absent.
        if let Some(glass) = self.glass.as_ref() {
            glass.wire_rt_dynamic(
                &device,
                frame_idx,
                super::super::glass::GlassRtDynamic {
                    tlas,
                    geom_buffer,
                    geom_size,
                    deformed,
                    skinned_indices,
                },
            );
        }
    }

    // Encode the RT-reflection resolve: a fullscreen triangle that traces each
    // glossy pixel's reflection ray against the scene TLAS and composites the
    // reflected colour into `rt_reflections.output`, which then becomes the scene
    // the bloom / composite / TAA passes consume. No-op when RT is off (the graph
    // only schedules this pass when RT is live, so the guard is defensive).
    pub(in crate::vulkan) fn encode_rt_reflections(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        fov_y_radians: f32,
        aspect: f32,
        cam_pos: [f32; 3],
    ) {
        let rt = match &self.rt_reflections {
            Some(r) => r,
            None => return,
        };
        let device = &self.device;
        let extent = self.render_extent;

        // The view->world rotation is the transpose of the view matrix's
        // orthonormal 3x3; `params` fills in the camera-position translation
        // column to complete the camera-to-world transform.
        let v = self.view.matrix;
        let inv_view_rot = [
            [v[0][0], v[1][0], v[2][0], 0.0],
            [v[0][1], v[1][1], v[2][1], 0.0],
            [v[0][2], v[1][2], v[2][2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let params = rt.settings.params(RtParamsInputs {
            fov_y_radians,
            aspect,
            inv_view_rot,
            cam_pos,
            sun_dir: self.fog.sun_dir,
            sun_color: self.fog.sun_color,
            prefilter_mip_count: self.prefilter_mip_count as f32,
        });
        rt.params_buffers[frame_idx].write_val(0, &params);

        // Textured hit shading needs the bindless albedo/normal pool, which only
        // the bindless static path populates; otherwise fall back to the
        // flat-tint variant. Mirrors DirectX's bindless gate.
        let textured = self.cull.bindless_pipeline.is_some() && rt.textured_pso.is_some();
        let (pso, layout) = match (
            textured,
            rt.textured_pso.as_ref(),
            rt.layout_textured.as_ref(),
        ) {
            (true, Some(pso), Some(layout)) => (pso, layout),
            _ => (&rt.flat_pso, &rt.layout_flat),
        };

        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(rt.render_pass.handle())
            .framebuffer(rt.framebuffer.handle())
            .render_area(vk::Rect2D::default().extent(extent));
        let vp = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D::default().extent(extent);
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
            device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&vp));
            device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pso.handle());
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                layout.handle(),
                0,
                std::slice::from_ref(&rt.resolve_sets[frame_idx]),
                &[],
            );
            // set 1: the global set, for the reflection-probe miss fallback.
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                layout.handle(),
                1,
                std::slice::from_ref(&self.descriptors.global_sets[frame_idx]),
                &[],
            );
            if textured {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    layout.handle(),
                    2,
                    std::slice::from_ref(&self.cull.bindless_sets[frame_idx]),
                    &[],
                );
            }
            device.cmd_draw(cmd, 3, 1, 0, 0);
            device.cmd_end_render_pass(cmd);
        }
        // Blur the trace's radiance+weight by roughness and composite it over the
        // scene into the reflection composite's output (the scene image the post
        // stack consumes). No-op when the composite is absent.
        self.encode_reflection_composite(cmd, rt.output.view, frame_idx);
    }
}

#[cfg(test)]
mod tests {
    // The RT fullscreen vert + both fragment variants compile to SPIR-V (ray
    // query target). Guards the `GL_EXT_ray_query` GLSL + the `RT_TEXTURED`
    // split. The CPU<->GPU `RtParams` / `RtGeomEntry` layouts are guarded by the
    // `rt_params_layout_*` / `rt_geom_entry_*` tests in gfx::render_types.
    #[test]
    fn rt_reflections_shaders_compile() {
        // Both the ceiling and a device-shortened probe cube array must compile.
        for probes in [1, concinnity_render::uniforms::MAX_PROBES as u32] {
            let shaders = super::compile_rt_shaders(false, 4, probes).expect("rt shaders compile");
            assert!(super::is_spirv(&shaders.vs));
            assert!(super::is_spirv(&shaders.flat_fs));
            assert!(shaders.textured_fs.is_some(), "pool_size>0 builds textured");
        }
        // pool_size 0 builds only the flat variant.
        let flat_only = super::compile_rt_shaders(false, 0, 4).expect("rt flat compiles");
        assert!(flat_only.textured_fs.is_none());
    }
}
