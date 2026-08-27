// src/vulkan/fog.rs
//
// Volumetric fog for the Vulkan backend. Frostbite-style froxel volume:
//
//   * The `fog_froxel_kernel` compute pass (`encode_fog_froxel`) populates a
//     screen-aligned `(80 x 45 x 64)` 3D `RGBA16F` volume of
//     `(scattered_rgb, 1 - T)` across the view frustum. One thread per
//     (x, y) tile; each thread walks Z front-to-back, accumulating the
//     per-slab scatter + transmittance with a CSM shadow tap per slice.
//
//   * The fullscreen `Fog` render pass (`encode_fog`) samples the volume by
//     `(screen_uv, view_z)` instead of marching per pixel and composites
//     `(scattered, 1 - T)` over the resolved HDR target with the standard
//     `over` blend (`final = scene * T + scattered`).
//
// Runs between the projected-decal pass and the SSR resolve so the fog wraps
// the decal-stamped scene and SSR reflects through it; TAA history then
// reprojects the integrated fog colour and transmittance.
//
// Mirrors src/directx/fog.rs and src/metal/fog.rs.

use concinnity_core::gfx::transform::mat4_inverse;
use std::ffi::CString;

use ash::vk;

use crate::vulkan::owned::{
    OwnedDescriptorPool, OwnedFramebuffer, OwnedPipeline, OwnedPipelineLayout, OwnedRenderPass,
    OwnedSampler, OwnedSetLayout, VkDevice,
};

use crate::gfx::render_graph::{FOG_FROXEL_X, FOG_FROXEL_Y, FOG_FROXEL_Z};
use crate::gfx::render_types::{FogFroxelParams, FogParams, ShadowUniforms};

use super::allocator::{DeviceAllocator, PooledBuffer, PooledImage};
use super::context::VkContext;
use super::pipeline::spv_module;
use super::texture::{
    LayoutTransition, SubresourceRange, one_shot_submit, transition_image_layout_range,
};

// Threadgroup tile for the froxel kernel (8x8, one thread per (x, y) froxel),
// matching the DirectX `[numthreads(8, 8, 1)]` and the Metal dispatch.
const FROXEL_TILE: u32 = 8;

// 3D froxel volume pixel format. RGBA16F holds `(scattered_rgb, 1 - T)` per
// slice; mirrors the DirectX / Metal `RGBA16Float` volume.
const VOLUME_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

// Owned by `VkContext` exactly when the world declared a `VolumetricFog`:
// the fog render pipeline + the froxel compute pipeline + the per-frame
// uniform rings + the shared 3D froxel volume the kernel writes and the
// fragment shader samples.
pub(in crate::vulkan) struct FogResources {
    pub(in crate::vulkan) render_pass: OwnedRenderPass,
    pub(in crate::vulkan) pipeline: OwnedPipeline,
    pub(in crate::vulkan) pipeline_layout: OwnedPipelineLayout,
    pub(in crate::vulkan) _view_set_layout: OwnedSetLayout,
    pub(in crate::vulkan) _descriptor_pool: OwnedDescriptorPool,

    // Per-frame FogParams view UBO (176 bytes). Persistently mapped.
    pub(in crate::vulkan) params_ubos: Vec<PooledBuffer>,

    // Per-frame FogFroxelParams UBO (96 bytes). Persistently mapped. Bound at
    // the froxel kernel's set binding 1 and the fog fragment's set binding 2.
    pub(in crate::vulkan) froxel_ubos: Vec<PooledBuffer>,

    // Per-frame fog-render view sets (binding 0 FogParams, 1 depth, 2
    // FogFroxelParams, 3 volume sampler3D).
    pub(in crate::vulkan) view_sets: Vec<vk::DescriptorSet>,

    // Froxel compute pipeline + its per-frame sets (binding 0 FogParams, 1
    // FogFroxelParams, 2 ShadowUniforms, 3 shadow_map, 4 volume image3D).
    pub(in crate::vulkan) froxel_pipeline: OwnedPipeline,
    pub(in crate::vulkan) froxel_pipeline_layout: OwnedPipelineLayout,
    pub(in crate::vulkan) _froxel_set_layout: OwnedSetLayout,
    pub(in crate::vulkan) froxel_sets: Vec<vk::DescriptorSet>,

    // Shared 3D RGBA16F volume: written by the compute kernel (GENERAL),
    // sampled by the fog fragment (SHADER_READ_ONLY). The open
    // (SHADER_READ_ONLY -> GENERAL) and close (GENERAL -> SHADER_READ_ONLY)
    // transitions are graph-driven (fog_froxel_volume's FogFroxel producer + Fog
    // consumer barriers, emitted by the executor); the cross-frame hazard chain
    // spans submission order on the one queue (same reasoning as the Hi-Z
    // pyramid).
    pub(in crate::vulkan) volume: PooledImage,

    // One framebuffer per frame-in-flight slot, each binding its frame slot's
    // `hdr_resolve_images[i].view` as the sole colour attachment.
    pub(in crate::vulkan) framebuffers: Vec<OwnedFramebuffer>,

    // Depth sampler (the shared linear sampler; depth is read via texelFetch so
    // the filter mode is irrelevant).
    pub(in crate::vulkan) sampler: vk::Sampler,
    // Linear-clamp sampler for the trilinear volume read.
    pub(in crate::vulkan) _volume_sampler: OwnedSampler,
}

// The Vulkan device handles `FogResources::new` needs to create its GPU
// resources and run the one-shot volume-layout transition. `command_pool` /
// `queue` are used once to move the volume into `SHADER_READ_ONLY_OPTIMAL`.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct FogDeviceContext<'a> {
    pub(in crate::vulkan) alloc: &'a DeviceAllocator,
    pub(in crate::vulkan) device: &'a VkDevice,
    pub(in crate::vulkan) command_pool: vk::CommandPool,
    pub(in crate::vulkan) queue: vk::Queue,
}

// The per-frame render targets + config the fog pipeline binds against: the
// resolved HDR colour views (framebuffer attachments), the scene depth views,
// the shared depth sampler, and the frame count / MSAA / format / extent.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct FogFrameTargets<'a> {
    pub(in crate::vulkan) frames: usize,
    pub(in crate::vulkan) msaa: bool,
    pub(in crate::vulkan) hdr_format: vk::Format,
    pub(in crate::vulkan) hdr_resolve_views: &'a [vk::ImageView],
    pub(in crate::vulkan) depth_views: &'a [vk::ImageView],
    pub(in crate::vulkan) sampler: vk::Sampler,
    pub(in crate::vulkan) extent: vk::Extent2D,
}

// The shared CSM resources the froxel kernel taps per slab. `ubos` is the
// per-frame-in-flight ShadowUniforms ring; slot `i` binds into froxel set `i`.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct FogShadowResources<'a> {
    pub(in crate::vulkan) ubos: &'a [PooledBuffer],
    pub(in crate::vulkan) map_view: vk::ImageView,
    pub(in crate::vulkan) sampler: vk::Sampler,
}

impl FogResources {
    // Build the fog render pipeline + the froxel compute pipeline + their
    // dependent resources. Called from `VkContext::new` only when the world
    // declared a `VolumetricFog` and `FogSettings::resolve` returned a value.
    pub(in crate::vulkan) fn new(
        ctx: FogDeviceContext,
        targets: FogFrameTargets,
        shadow: FogShadowResources<'_>,
        hot_reload: bool,
    ) -> Result<Self, String> {
        let FogDeviceContext {
            alloc,
            device,
            command_pool,
            queue,
        } = ctx;
        let FogFrameTargets {
            frames,
            msaa,
            hdr_format,
            hdr_resolve_views,
            depth_views,
            sampler,
            extent,
        } = targets;
        let FogShadowResources {
            ubos: shadow_ubos,
            map_view: shadow_map_view,
            sampler: shadow_sampler,
        } = shadow;
        let render_pass = create_fog_render_pass(device, hdr_format)?;
        let view_set_layout = create_fog_set_layout(device)?;
        let pipeline_layout = create_fog_pipeline_layout(device, view_set_layout.handle())?;

        let (vert_spv, frag_spv) = compile_fog_shaders(hot_reload, msaa)?;
        let pipeline = create_fog_pipeline(
            device,
            render_pass.handle(),
            pipeline_layout.handle(),
            &vert_spv,
            &frag_spv,
        )?;

        // Froxel compute pipeline.
        let froxel_set_layout = create_froxel_set_layout(device)?;
        let froxel_pipeline_layout =
            create_froxel_pipeline_layout(device, froxel_set_layout.handle())?;
        let froxel_spv = compile_fog_froxel_shader(hot_reload)?;
        let froxel_pipeline =
            create_compute_pipeline(device, froxel_pipeline_layout.handle(), &froxel_spv)?;

        // The shared 3D volume + its storage (compute write) + sampled
        // (fragment read) views. Rest it in SHADER_READ_ONLY so the first
        // froxel build's opening barrier (SHADER_READ_ONLY -> GENERAL) matches.
        let volume = create_volume_image(alloc)?;
        one_shot_submit(device, command_pool, queue, |cmd| {
            transition_image_layout_range(
                device,
                cmd,
                volume.image(),
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
        let volume_storage_view = create_volume_view(device, volume.image())?;
        volume.attach_view(volume_storage_view);
        let volume_sampled_view = create_volume_view(device, volume.image())?;
        volume.attach_view(volume_sampled_view);
        let volume_sampler = create_volume_sampler(device)?;

        // Per-frame FogParams UBOs (HOST_VISIBLE | HOST_COHERENT, mapped).
        let params_ubos = alloc_ubo_ring(alloc, frames, std::mem::size_of::<FogParams>() as u64)?;
        // Per-frame FogFroxelParams UBOs.
        let froxel_ubos =
            alloc_ubo_ring(alloc, frames, std::mem::size_of::<FogFroxelParams>() as u64)?;

        let descriptor_pool = create_fog_descriptor_pool(device, frames)?;
        let view_layouts: Vec<_> = (0..frames).map(|_| view_set_layout.handle()).collect();
        let view_sets = alloc_descriptor_sets(device, descriptor_pool.handle(), &view_layouts)?;
        let froxel_layouts: Vec<_> = (0..frames).map(|_| froxel_set_layout.handle()).collect();
        let froxel_sets = alloc_descriptor_sets(device, descriptor_pool.handle(), &froxel_layouts)?;

        let last_depth = depth_views.len().saturating_sub(1);
        for (i, &set) in view_sets.iter().enumerate() {
            write_view_set(
                device,
                set,
                FogViewBindings {
                    params_ubo: params_ubos[i].buffer(),
                    depth_view: depth_views[i.min(last_depth)],
                    depth_sampler: sampler,
                    froxel_ubo: froxel_ubos[i].buffer(),
                    volume_view: volume_sampled_view,
                    _volume_sampler: volume_sampler.handle(),
                },
            );
        }
        for (i, &set) in froxel_sets.iter().enumerate() {
            write_froxel_set(
                device,
                set,
                FogFroxelBindings {
                    params_ubo: params_ubos[i].buffer(),
                    froxel_ubo: froxel_ubos[i].buffer(),
                    shadow_ubo: shadow_ubos[i].buffer(),
                    shadow_map_view,
                    shadow_sampler,
                    volume_storage_view,
                },
            );
        }

        // Per-frame framebuffers (one per frame slot binding that slot's
        // hdr_resolve view as the colour attachment).
        let mut framebuffers = Vec::with_capacity(frames);
        for &view in hdr_resolve_views.iter().take(frames) {
            let attachments = [view];
            let fb_info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass.handle())
                .attachments(&attachments)
                .width(extent.width.max(1))
                .height(extent.height.max(1))
                .layers(1);
            let fb = device
                .create_framebuffer(&fb_info)
                .map_err(|e| format!("fog framebuffer: {e}"))?;
            framebuffers.push(fb);
        }

        Ok(Self {
            render_pass,
            pipeline,
            pipeline_layout,
            _view_set_layout: view_set_layout,
            _descriptor_pool: descriptor_pool,
            params_ubos,
            froxel_ubos,
            view_sets,
            froxel_pipeline,
            froxel_pipeline_layout,
            _froxel_set_layout: froxel_set_layout,
            froxel_sets,
            volume,
            framebuffers,
            sampler,
            _volume_sampler: volume_sampler,
        })
    }

    // Rebuild the framebuffers + re-point the per-frame view set's depth
    // binding after a swapchain resize. Called from
    // `VkContext::rebuild_swapchain`; same pattern as `DecalResources`. The
    // pipelines, layouts, buffers, the froxel sets, the 3D volume, and the
    // samplers all survive (the volume is screen-aligned via the per-froxel
    // reconstruction, not tied to render resolution).
    pub(in crate::vulkan) fn rebuild(
        &mut self,
        device: &VkDevice,
        hdr_resolve_views: &[vk::ImageView],
        depth_views: &[vk::ImageView],
        extent: vk::Extent2D,
    ) -> Result<(), String> {
        self.framebuffers.clear();
        for &view in hdr_resolve_views.iter().take(self.params_ubos.len()) {
            let attachments = [view];
            let fb_info = vk::FramebufferCreateInfo::default()
                .render_pass(self.render_pass.handle())
                .attachments(&attachments)
                .width(extent.width.max(1))
                .height(extent.height.max(1))
                .layers(1);
            let fb = device
                .create_framebuffer(&fb_info)
                .map_err(|e| format!("fog framebuffer (rebuild): {e}"))?;
            self.framebuffers.push(fb);
        }
        let last_depth = depth_views.len().saturating_sub(1);
        for (i, &set) in self.view_sets.iter().enumerate() {
            let depth_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(depth_views[i.min(last_depth)])
                .sampler(self.sampler);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&depth_info));
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
        }
        Ok(())
    }

    // Destroy every non-pooled GPU resource and drop the pooled ones (the
    // UBO rings and the volume + its views retire through the allocator).
    // Called from `VkContext::drop` after `wait_idle`.
    pub(in crate::vulkan) fn destroy(&mut self, _device: &VkDevice) {
        self.framebuffers.clear();
        self.params_ubos.clear();
        self.froxel_ubos.clear();
        self.volume = PooledImage::null();
    }
}

// Allocate `count` host-visible/coherent uniform buffers of `size` bytes,
// each persistently mapped through its pooled block.
fn alloc_ubo_ring(
    alloc: &DeviceAllocator,
    count: usize,
    size: u64,
) -> Result<Vec<PooledBuffer>, String> {
    (0..count)
        .map(|_| {
            alloc
                .create_buffer(
                    size,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
                .map_err(String::from)
        })
        .collect()
}

// Render pass / pipeline construction

fn create_fog_render_pass(
    device: &VkDevice,
    format: vk::Format,
) -> Result<OwnedRenderPass, String> {
    // One colour attachment: the resolved HDR scene. The main pass (and
    // any preceding decal pass) left it in SHADER_READ_ONLY_OPTIMAL; we
    // want it in COLOR_ATTACHMENT during the subpass and
    // SHADER_READ_ONLY_OPTIMAL again on exit so SSR / TAA / bloom /
    // composite can sample it. Mirrors the decal render pass.
    let attachment = vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let color_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_ref));
    let dep_in = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::FRAGMENT_SHADER,
        )
        .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::COLOR_ATTACHMENT_READ,
        );
    let dep_out = vk::SubpassDependency::default()
        .src_subpass(0)
        .dst_subpass(vk::SUBPASS_EXTERNAL)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
    let deps = [dep_in, dep_out];
    let info = vk::RenderPassCreateInfo::default()
        .attachments(std::slice::from_ref(&attachment))
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(&deps);
    device
        .create_render_pass(&info)
        .map_err(|e| format!("fog render pass: {e}"))
}

fn create_fog_set_layout(device: &VkDevice) -> Result<OwnedSetLayout, String> {
    let bindings = [
        // 0: FogParams UBO.
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        // 1: scene depth sampler.
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        // 2: FogFroxelParams UBO.
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        // 3: froxel volume sampler3D.
        vk::DescriptorSetLayoutBinding::default()
            .binding(3)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    device
        .create_descriptor_set_layout(&info)
        .map_err(|e| format!("fog set layout: {e}"))
}

fn create_froxel_set_layout(device: &VkDevice) -> Result<OwnedSetLayout, String> {
    let bindings = [
        // 0: FogParams UBO.
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // 1: FogFroxelParams UBO.
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // 2: ShadowUniforms UBO.
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // 3: shadow map array (sampler2DArrayShadow).
        vk::DescriptorSetLayoutBinding::default()
            .binding(3)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // 4: froxel volume image3D (storage).
        vk::DescriptorSetLayoutBinding::default()
            .binding(4)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    device
        .create_descriptor_set_layout(&info)
        .map_err(|e| format!("fog froxel set layout: {e}"))
}

fn create_fog_pipeline_layout(
    device: &VkDevice,
    view_set_layout: vk::DescriptorSetLayout,
) -> Result<OwnedPipelineLayout, String> {
    let set_layouts = [view_set_layout];
    let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
    device
        .create_pipeline_layout(&info)
        .map_err(|e| format!("fog pipeline layout: {e}"))
}

fn create_froxel_pipeline_layout(
    device: &VkDevice,
    froxel_set_layout: vk::DescriptorSetLayout,
) -> Result<OwnedPipelineLayout, String> {
    let set_layouts = [froxel_set_layout];
    let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
    device
        .create_pipeline_layout(&info)
        .map_err(|e| format!("fog froxel pipeline layout: {e}"))
}

fn create_fog_descriptor_pool(
    device: &VkDevice,
    frames: usize,
) -> Result<OwnedDescriptorPool, String> {
    let f = frames as u32;
    let sizes = [
        // view: FogParams + FogFroxelParams (2). froxel: FogParams +
        // FogFroxelParams + ShadowUniforms (3).
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 5 * f,
        },
        // view: depth + volume sampled (2). froxel: shadow map (1).
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 3 * f,
        },
        // froxel: volume storage (1).
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: f,
        },
    ];
    let info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(2 * f)
        .pool_sizes(&sizes);
    device
        .create_descriptor_pool(&info)
        .map_err(|e| format!("fog descriptor pool: {e}"))
}

fn alloc_descriptor_sets(
    device: &VkDevice,
    pool: vk::DescriptorPool,
    layouts: &[vk::DescriptorSetLayout],
) -> Result<Vec<vk::DescriptorSet>, String> {
    let info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(layouts);
    // SAFETY: the create-info and every slice it borrows are live for the call, and each handle it
    // names belongs to this device.
    unsafe { device.allocate_descriptor_sets(&info) }
        .map_err(|e| format!("fog descriptor sets: {e}"))
}

// The four bindings of a per-frame fog-render view set: the FogParams +
// FogFroxelParams UBOs, the scene depth sampler, and the froxel volume sampler.
#[derive(Clone, Copy)]
struct FogViewBindings {
    params_ubo: vk::Buffer,
    depth_view: vk::ImageView,
    depth_sampler: vk::Sampler,
    froxel_ubo: vk::Buffer,
    volume_view: vk::ImageView,
    _volume_sampler: vk::Sampler,
}

fn write_view_set(device: &VkDevice, set: vk::DescriptorSet, bindings: FogViewBindings) {
    let FogViewBindings {
        params_ubo,
        depth_view,
        depth_sampler,
        froxel_ubo,
        volume_view,
        _volume_sampler: volume_sampler,
    } = bindings;
    let params_info = vk::DescriptorBufferInfo::default()
        .buffer(params_ubo)
        .offset(0)
        .range(std::mem::size_of::<FogParams>() as u64);
    let depth_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(depth_view)
        .sampler(depth_sampler);
    let froxel_info = vk::DescriptorBufferInfo::default()
        .buffer(froxel_ubo)
        .offset(0)
        .range(std::mem::size_of::<FogFroxelParams>() as u64);
    let volume_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(volume_view)
        .sampler(volume_sampler);
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&params_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&depth_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&froxel_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&volume_info)),
    ];
    // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and every set
    // and resource it names belongs to this device.
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

// The five bindings of a per-frame froxel compute set: the FogParams,
// FogFroxelParams, and ShadowUniforms UBOs, the CSM shadow map, and the froxel
// volume storage image.
#[derive(Clone, Copy)]
struct FogFroxelBindings {
    params_ubo: vk::Buffer,
    froxel_ubo: vk::Buffer,
    shadow_ubo: vk::Buffer,
    shadow_map_view: vk::ImageView,
    shadow_sampler: vk::Sampler,
    volume_storage_view: vk::ImageView,
}

fn write_froxel_set(device: &VkDevice, set: vk::DescriptorSet, bindings: FogFroxelBindings) {
    let FogFroxelBindings {
        params_ubo,
        froxel_ubo,
        shadow_ubo,
        shadow_map_view,
        shadow_sampler,
        volume_storage_view,
    } = bindings;
    let params_info = vk::DescriptorBufferInfo::default()
        .buffer(params_ubo)
        .offset(0)
        .range(std::mem::size_of::<FogParams>() as u64);
    let froxel_info = vk::DescriptorBufferInfo::default()
        .buffer(froxel_ubo)
        .offset(0)
        .range(std::mem::size_of::<FogFroxelParams>() as u64);
    let shadow_info = vk::DescriptorBufferInfo::default()
        .buffer(shadow_ubo)
        .offset(0)
        .range(std::mem::size_of::<ShadowUniforms>() as u64);
    let shadow_map_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(shadow_map_view)
        .sampler(shadow_sampler);
    let volume_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::GENERAL)
        .image_view(volume_storage_view);
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&params_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&froxel_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&shadow_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&shadow_map_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(4)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(&volume_info)),
    ];
    // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and every set
    // and resource it names belongs to this device.
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

// Create the shared 3D RGBA16F froxel volume (STORAGE | SAMPLED, GPU-local).
fn create_volume_image(alloc: &DeviceAllocator) -> Result<PooledImage, String> {
    let img_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_3D)
        .extent(vk::Extent3D {
            width: FOG_FROXEL_X,
            height: FOG_FROXEL_Y,
            depth: FOG_FROXEL_Z,
        })
        .mip_levels(1)
        .array_layers(1)
        .format(VOLUME_FORMAT)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::TYPE_1);
    alloc
        .create_image(&img_info, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .map_err(|e| format!("fog volume image: {e}"))
}

// A whole-image 3D view of the froxel volume (used for both the compute
// storage bind and the fragment sampled bind).
fn create_volume_view(device: &VkDevice, image: vk::Image) -> Result<vk::ImageView, String> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_3D)
        .format(VOLUME_FORMAT)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    // SAFETY: the create-info and every slice it borrows are live for the call, and each handle it
    // names belongs to this device.
    unsafe { device.create_image_view(&info, None) }.map_err(|e| format!("fog volume view: {e}"))
}

// Linear clamp-to-edge sampler for the trilinear volume read.
fn create_volume_sampler(device: &VkDevice) -> Result<OwnedSampler, String> {
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
    device
        .create_sampler(&info)
        .map_err(|e| format!("fog volume sampler: {e}"))
}

fn compile_fog_shaders(hot_reload: bool, msaa: bool) -> Result<(Vec<u8>, Vec<u8>), String> {
    let ctx = super::builtins::Ctx {
        msaa,
        ..super::builtins::Ctx::plain(hot_reload)
    };
    let vert = super::slang_builtins::FULLSCREEN_VERT.compile(&ctx)?;
    let frag = super::slang_builtins::FOG_FRAG.compile(&ctx)?;
    Ok((vert, frag))
}

// Compile the froxel-volume compute kernel. MSAA-independent (the kernel does
// not read the scene depth attachment).
fn compile_fog_froxel_shader(hot_reload: bool) -> Result<Vec<u8>, String> {
    super::slang_builtins::FOG_FROXEL.compile(&super::builtins::Ctx::plain(hot_reload))
}

// Rebuild the fog graphics pipeline against the existing render pass +
// layout. Used by the Vulkan shader hot-reload path. The caller is
// responsible for destroying the previous pipeline only after this call
// succeeds.
pub(in crate::vulkan) fn rebuild_fog_pipeline(
    device: &VkDevice,
    fog: &FogResources,
    msaa: bool,
    hot_reload: bool,
) -> Result<OwnedPipeline, String> {
    let (vert_spv, frag_spv) = compile_fog_shaders(hot_reload, msaa)?;
    create_fog_pipeline(
        device,
        fog.render_pass.handle(),
        fog.pipeline_layout.handle(),
        &vert_spv,
        &frag_spv,
    )
}

// Rebuild the froxel compute pipeline against the existing layout. Hot-reload.
pub(in crate::vulkan) fn rebuild_fog_froxel_pipeline(
    device: &VkDevice,
    fog: &FogResources,
    hot_reload: bool,
) -> Result<OwnedPipeline, String> {
    let spv = compile_fog_froxel_shader(hot_reload)?;
    create_compute_pipeline(device, fog.froxel_pipeline_layout.handle(), &spv)
}

fn create_compute_pipeline(
    device: &VkDevice,
    layout: vk::PipelineLayout,
    spv: &[u8],
) -> Result<OwnedPipeline, String> {
    let module = spv_module(device, spv)?;
    let entry = CString::new("main").unwrap();
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module.handle())
        .name(&entry);
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layout);
    let pipeline = crate::vulkan::pipeline_cache::create_compute_pipeline(device, &info)
        .map_err(|e| format!("create fog froxel pipeline: {e}"))?;
    Ok(pipeline)
}

fn create_fog_pipeline(
    device: &VkDevice,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
    frag_spv: &[u8],
) -> Result<OwnedPipeline, String> {
    let vert = spv_module(device, vert_spv)?;
    let frag = spv_module(device, frag_spv)?;
    let entry = CString::new("main").unwrap();
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert.handle())
            .name(&entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag.handle())
            .name(&entry),
    ];
    // Fullscreen triangle is emitted by gl_VertexIndex; no vertex buffer.
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        // The fog pass writes the SINGLE-SAMPLE resolved HDR target, not
        // the MSAA colour, regardless of whether the main pass uses MSAA.
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(false)
        .depth_write_enable(false);
    let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(true)
        // (scattered, 1 - T) over scene: dst = src + (1 - src.a) * dst,
        // resolving to `final = scattered + transmittance * scene`. Matches
        // the DirectX / Metal blend.
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD)
        .color_write_mask(vk::ColorComponentFlags::RGBA);
    let blend_attachments = [blend_attachment];
    let blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .attachments(&blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&blend_state)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass);
    let pipeline = crate::vulkan::pipeline_cache::create_graphics_pipeline(device, &info)
        .map_err(|e| format!("create fog pipeline: {e}"))?;
    Ok(pipeline)
}

// Encoder

impl VkContext {
    // Hot-reload entry point for the volumetric-fog tunables (driven by
    // `world.jsonl` hot-reload under `cn debug`). Writes the new
    // `Option<FogSettings>` into `self.fog.settings`; the next frame's graph
    // seed re-reads it (so `None` drops the FogFroxel + Fog passes) and
    // `encode_fog_froxel` rebuilds `FogParams` / `FogFroxelParams` from it.
    // Mirrors `MtlContext::update_fog_settings`.
    //
    // If the world started with no `VolumetricFog` (so `fog.resources` is
    // `None`), a `Some` update logs once and is dropped: re-enabling fog
    // mid-run requires a relaunch (the froxel pipeline + volume were never
    // built).
    //
    // Named distinctly from the `RenderBackend::update_fog_settings` trait
    // method so the backend forwarder's `self.apply_fog_settings(...)` is
    // unambiguous. Reached through the `RenderBackend` vtable (the bin's
    // `cn debug` world.jsonl hot-reload).
    pub(in crate::vulkan) fn apply_fog_settings(
        &mut self,
        settings: Option<crate::gfx::volumetric_fog::FogSettings>,
    ) {
        if settings.is_some() && self.fog.resources.is_none() {
            tracing::warn!(
                "VolumetricFog hot-reload: world started without fog, so the fog \
                 pipeline + froxel volume were never built: re-enabling fog mid-run \
                 is not supported (relaunch required). Ignoring update."
            );
            return;
        }
        self.fog.settings = settings;
    }

    // Encode the volumetric-fog froxel-volume compute pass. Populates the 3D
    // `(scattered, 1 - T)` volume the fog fragment shader samples. The shared
    // graph seeds `PassId::FogFroxel` before `Fog` so the RAW edge orders the
    // dispatch ahead of the render-pass read. Uploads both per-frame UBOs
    // (`FogParams` + `FogFroxelParams`) so `encode_fog` only reads them.
    pub(in crate::vulkan) fn encode_fog_froxel(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        near: f32,
        vp: [[f32; 4]; 4],
        cam_pos: [f32; 3],
    ) {
        let fog_settings = match &self.fog.settings {
            Some(s) => *s,
            None => return,
        };
        let fog = match &self.fog.resources {
            Some(f) => f,
            None => return,
        };

        let device = &self.device;

        // Per-frame FogParams (drives the volume integration + the fragment's
        // viewport / reconstruction). Uploaded here so `encode_fog` only reads.
        let inv_vp = mat4_inverse(vp);
        let viewport_pix = [
            self.render_extent.width as f32,
            self.render_extent.height as f32,
        ];
        let params = fog_settings.params(
            inv_vp,
            cam_pos,
            self.fog.sun_dir,
            self.fog.sun_color,
            viewport_pix,
        );
        // Per-frame FogFroxelParams: world->view matrix + the volume's discrete
        // dimensions + the linear-Z `[near, max_distance]` mapping. `near` is
        // clamped to >= 1e-3 so the linear-Z reconstruction stays finite.
        let froxel_params = FogFroxelParams {
            view: self.view.matrix,
            froxel_dims: [FOG_FROXEL_X, FOG_FROXEL_Y, FOG_FROXEL_Z],
            _pad_align: 0,
            z_near: near.max(1e-3),
            z_far: fog_settings.max_distance,
            _pad: [0.0; 2],
        };
        fog.params_ubos[frame_idx].write_val(0, &params);
        fog.froxel_ubos[frame_idx].write_val(0, &froxel_params);

        // Both of this pass's transitions are graph-driven. The froxel volume's
        // SHADER_READ_ONLY -> GENERAL open comes from the FogFroxel producer
        // barrier, whose source scope also orders the previous frame's fog
        // fragment read before this write; the shadow map's cascade tap is a
        // declared read of this pass, so the Shadow consumer barrier's stage union
        // carries the compute stage.
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                fog.froxel_pipeline.handle(),
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                fog.froxel_pipeline_layout.handle(),
                0,
                std::slice::from_ref(&fog.froxel_sets[frame_idx]),
                &[],
            );
            device.cmd_dispatch(
                cmd,
                FOG_FROXEL_X.div_ceil(FROXEL_TILE),
                FOG_FROXEL_Y.div_ceil(FROXEL_TILE),
                1,
            );
        }

        // The froxel volume's GENERAL -> SHADER_READ_ONLY close comes from the
        // Fog consumer barrier, and next frame's Shadow producer opens from a
        // resting scope that names both shader stages, so this frame's compute tap
        // is ordered against it.
    }

    // Encode the volumetric-fog pass. Samples the 3D froxel volume the
    // `FogFroxel` compute pass populated this frame. Caller has already ended
    // the main HDR resolve and the projected-decal pass (if any), so
    // `depth_images[frame_idx]` holds the scene depth and
    // `hdr_resolve_images[frame_idx]` holds the resolved scene + decal colour
    // in SHADER_READ_ONLY_OPTIMAL. Alpha-blends `(scattered, 1 - T)` over the
    // resolved HDR target. `FogParams` / `FogFroxelParams` were uploaded by
    // `encode_fog_froxel` for this frame's slot, so this pass only binds.
    pub(in crate::vulkan) fn encode_fog(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        _vp: [[f32; 4]; 4],
        _cam_pos: [f32; 3],
    ) {
        if self.fog.settings.is_none() {
            return;
        }
        let fog = match &self.fog.resources {
            Some(f) => f,
            None => return,
        };

        let device = &self.device;
        let extent = self.render_extent;

        // Main depth is already in SHADER_READ_ONLY for the fragment's scene-depth
        // sample: the graph declares this pass's depth read and the executor emits
        // the transition ahead of this command buffer.
        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(fog.render_pass.handle())
            .framebuffer(fog.framebuffers[frame_idx].handle())
            .render_area(vk::Rect2D::default().extent(extent));

        // Standard positive-height viewport: the fullscreen triangle is
        // emitted in NDC and the fragment shader's reconstruction handles
        // the Y flip against the main pass's depth (which was written with
        // a negative-height viewport).
        let vp_state = vk::Viewport {
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
            device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&vp_state));
            device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, fog.pipeline.handle());
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                fog.pipeline_layout.handle(),
                0,
                std::slice::from_ref(&fog.view_sets[frame_idx]),
                &[],
            );
            device.cmd_draw(cmd, 3, 1, 0, 0);
            device.cmd_end_render_pass(cmd);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::gfx::render_types::{FogFroxelParams, FogParams};
    use std::mem::size_of;

    #[test]
    fn fog_params_ubo_size_matches_glsl() {
        // Both halves of fog.slang read the same 176 B std140 FogBlock.
        assert_eq!(size_of::<FogParams>(), 176);
    }

    #[test]
    fn fog_froxel_params_ubo_size_matches_glsl() {
        // The FogFroxelBlock std140 layout (mat4 + uvec3 + uint + 2 float + vec2)
        // is 96 B; the offsets are pinned by the core render_types tests.
        assert_eq!(size_of::<FogFroxelParams>(), 96);
    }

    #[test]
    fn fog_shaders_compile() {
        // Compile the rewritten froxel-sampling fragment shader (both MSAA
        // modes) + the froxel compute kernel so a GLSL regression fails the
        // test suite without needing a GPU. Mirrors the cull-shader compile
        // guard the two-pass occlusion landing added.
        super::compile_fog_shaders(false, false).expect("fog shaders (no MSAA) compile");
        super::compile_fog_shaders(false, true).expect("fog shaders (MSAA) compile");
        super::compile_fog_froxel_shader(false).expect("fog froxel kernel compiles");
    }
}
