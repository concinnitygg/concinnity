//! Volumetric fog for the Vulkan backend. Frostbite-style froxel volume:
//!
//!   * The `fog_froxel_kernel` compute pass (`encode_fog_froxel`) populates a
//!     screen-aligned `(80 x 45 x 64)` 3D `RGBA16F` volume of
//!     `(scattered_rgb, 1 - T)` across the view frustum. One thread per
//!     (x, y) tile; each thread walks Z front-to-back, accumulating the
//!     per-slab scatter + transmittance with a CSM shadow tap per slice.
//!
//!   * The fullscreen `Fog` pass (`encode_fog`) samples the volume by
//!     `(screen_uv, view_z)` instead of marching per pixel and composites
//!     `(scattered, 1 - T)` over the resolved HDR target with the standard
//!     `over` blend (`final = scene * T + scattered`).
//!
//! Runs between the projected-decal pass and the SSR resolve so the fog wraps
//! the decal-stamped scene and SSR reflects through it; TAA history then
//! reprojects the integrated fog color and transmittance.
//!
//! Mirrors src/directx/fog.rs and src/metal/fog.rs.

use ash::vk;
use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::render_types::{FogFroxelParams, FogParams, ShadowUniforms};
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::frame::plan::FogBlocks;
use concinnity_engine::render::render_graph::{FOG_FROXEL_X, FOG_FROXEL_Y, FOG_FROXEL_Z};
use concinnity_engine::render::shader::programs::abi::fog::FROXEL_GROUP;

use super::allocator::{DeviceAllocator, PooledBuffer, PooledImage};
use super::context::VkContext;
use super::descriptor_layout::{Binding, PoolSizes};
use super::pipeline_desc::{Blend, GraphicsPipelineDesc, compute_pipeline};
use super::rendering::PassFormats;
use super::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use super::set_writes::SetWrites;
use super::texture::{
    LayoutTransition, SubresourceRange, one_shot_submit, transition_image_layout_range,
};
use crate::vulkan::builtin_shaders::{CompileProgram, FOG_FROXEL};
use crate::vulkan::owned::{
    OwnedDescriptorPool, OwnedPipeline, OwnedPipelineLayout, OwnedSampler, OwnedSetLayout, VkDevice,
};
use crate::vulkan::record::Recorder;

// 3D froxel volume pixel format. RGBA16F holds `(scattered_rgb, 1 - T)` per
// slice; mirrors the DirectX / Metal `RGBA16Float` volume.
const VOLUME_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

// Owned by `VkContext` exactly when the world declared a `VolumetricFog`:
// the fog render pipeline + the froxel compute pipeline + the per-frame
// uniform rings + the shared 3D froxel volume the kernel writes and the
// fragment shader samples.
pub(in crate::vulkan) struct FogResources {
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
    // FogFroxelParams, 3 volume, 4 the volume's sampler).
    pub(in crate::vulkan) view_sets: Vec<vk::DescriptorSet>,

    // Froxel compute pipeline + its per-frame sets (binding 0 FogParams, 1
    // FogFroxelParams, 2 ShadowUniforms, 3 shadow_map, 4 volume image3D, 5 the
    // shadow map's compare sampler).
    pub(in crate::vulkan) froxel_pipeline: OwnedPipeline,
    pub(in crate::vulkan) froxel_pipeline_layout: OwnedPipelineLayout,
    pub(in crate::vulkan) _froxel_set_layout: OwnedSetLayout,
    pub(in crate::vulkan) froxel_sets: Vec<vk::DescriptorSet>,

    // The 3D RGBA16F froxel volume, written by the compute kernel in GENERAL and
    // sampled by the fog fragment; the graph drives both transitions.
    pub(in crate::vulkan) volume: PooledImage,

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
// scene depth views, and the frame count / MSAA / resolved HDR format.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct FogFrameTargets<'a> {
    pub(in crate::vulkan) frames: usize,
    pub(in crate::vulkan) msaa: bool,
    pub(in crate::vulkan) hdr_format: vk::Format,
    pub(in crate::vulkan) depth_views: &'a [vk::ImageView],
}

// The shared CSM and far shadow resources the froxel kernel taps per slab.
// `ubos` is the per-frame-in-flight ShadowUniforms ring; slot `i` binds into
// froxel set `i`.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct FogShadowResources<'a> {
    pub(in crate::vulkan) ubos: &'a [PooledBuffer],
    pub(in crate::vulkan) map_view: vk::ImageView,
    pub(in crate::vulkan) far_view: vk::ImageView,
    pub(in crate::vulkan) sampler: vk::Sampler,
}

impl FogResources {
    // Build the fog render pipeline + the froxel compute pipeline + their
    // dependent resources. Called from `VkContext::new` only when the world
    // declared a `VolumetricFog` and `FogSettings::resolve` returned a value.
    pub(in crate::vulkan) fn new(
        ctx: FogDeviceContext<'_>,
        targets: FogFrameTargets<'_>,
        shadow: FogShadowResources<'_>,
        hot_reload: bool,
    ) -> RenderResult<Self> {
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
            depth_views,
        } = targets;
        let FogShadowResources {
            ubos: shadow_ubos,
            map_view: shadow_map_view,
            far_view: far_shadow_view,
            sampler: shadow_sampler,
        } = shadow;
        let view_set_layout = create_descriptor_set_layout(device, &fog_set_bindings())?;
        let pipeline_layout = create_fog_pipeline_layout(device, view_set_layout.handle())?;

        let (vert_spv, frag_spv) = compile_fog_shaders(hot_reload, msaa)?;
        let pipeline = create_fog_pipeline(
            device,
            hdr_format,
            pipeline_layout.handle(),
            &vert_spv,
            &frag_spv,
        )?;

        // Froxel compute pipeline.
        let froxel_set_layout = create_descriptor_set_layout(device, &froxel_set_bindings())?;
        let froxel_pipeline_layout =
            create_froxel_pipeline_layout(device, froxel_set_layout.handle())?;
        let froxel_spv = FOG_FROXEL.compile(hot_reload)?;
        let froxel_pipeline = compute_pipeline(
            device,
            froxel_pipeline_layout.handle(),
            &froxel_spv,
            "fog froxel",
        )?;

        // The shared 3D volume + its storage (compute write) + sampled
        // (fragment read) views. Rest it in SHADER_READ_ONLY so the first
        // froxel build's opening barrier (SHADER_READ_ONLY -> GENERAL) matches.
        let volume = create_volume_image(alloc)?;
        one_shot_submit(device, command_pool, queue, |rec| {
            transition_image_layout_range(
                rec,
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
        let params_ubos = alloc_ubo_ring(alloc, frames, size_of::<FogParams>() as u64)?;
        // Per-frame FogFroxelParams UBOs.
        let froxel_ubos = alloc_ubo_ring(alloc, frames, size_of::<FogFroxelParams>() as u64)?;

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
                    froxel_ubo: froxel_ubos[i].buffer(),
                    volume_view: volume_sampled_view,
                    volume_sampler: volume_sampler.handle(),
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
                    far_shadow_view,
                },
            );
        }

        Ok(Self {
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
            _volume_sampler: volume_sampler,
        })
    }

    // Re-point the view set's depth after a resize; the volume is not tied to
    // render resolution.
    pub(in crate::vulkan) fn repoint_depth(
        &self,
        device: &VkDevice,
        depth_views: &[vk::ImageView],
    ) {
        let last_depth = depth_views.len().saturating_sub(1);
        for (i, &set) in self.view_sets.iter().enumerate() {
            SetWrites::new(set)
                .sampled_image(1, depth_views[i.min(last_depth)])
                .apply(device);
        }
    }

    // Destroy every non-pooled GPU resource and drop the pooled ones (the
    // UBO rings and the volume + its views retire through the allocator).
    // Called from `VkContext::drop` after `wait_idle`.
    pub(in crate::vulkan) fn destroy(&mut self, _device: &VkDevice) {
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
) -> RenderResult<Vec<PooledBuffer>> {
    (0..count)
        .map(|_| {
            alloc.create_buffer(
                size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
        })
        .collect()
}

// Pipeline construction

// The fog-render view set: FogParams UBO, the scene depth (read by texel),
// FogFroxelParams UBO, the froxel volume and the volume's sampler.
fn fog_set_bindings() -> [Binding; 5] {
    use vk::DescriptorType as T;
    let frag = vk::ShaderStageFlags::FRAGMENT;
    [
        (0, T::UNIFORM_BUFFER, frag),
        (1, T::SAMPLED_IMAGE, frag),
        (2, T::UNIFORM_BUFFER, frag),
        (3, T::SAMPLED_IMAGE, frag),
        (4, T::SAMPLER, frag),
    ]
}

// The froxel compute set: FogParams, FogFroxelParams and ShadowUniforms UBOs,
// the shadow map array, the volume storage image and the shadow compare sampler.
fn froxel_set_bindings() -> [Binding; 7] {
    use vk::DescriptorType as T;
    let compute = vk::ShaderStageFlags::COMPUTE;
    [
        (0, T::UNIFORM_BUFFER, compute),
        (1, T::UNIFORM_BUFFER, compute),
        (2, T::UNIFORM_BUFFER, compute),
        (3, T::SAMPLED_IMAGE, compute),
        (4, T::STORAGE_IMAGE, compute),
        (5, T::SAMPLER, compute),
        (6, T::SAMPLED_IMAGE, compute),
    ]
}

fn create_fog_pipeline_layout(
    device: &VkDevice,
    view_set_layout: vk::DescriptorSetLayout,
) -> RenderResult<OwnedPipelineLayout> {
    let set_layouts = [view_set_layout];
    let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
    device
        .create_pipeline_layout(&info)
        .map_err(|e| super::error::map_vk_result(e, "fog pipeline layout"))
}

fn create_froxel_pipeline_layout(
    device: &VkDevice,
    froxel_set_layout: vk::DescriptorSetLayout,
) -> RenderResult<OwnedPipelineLayout> {
    let set_layouts = [froxel_set_layout];
    let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
    device
        .create_pipeline_layout(&info)
        .map_err(|e| super::error::map_vk_result(e, "fog froxel pipeline layout"))
}

fn create_fog_descriptor_pool(
    device: &VkDevice,
    frames: usize,
) -> RenderResult<OwnedDescriptorPool> {
    let f = u32::saturating_from(frames);
    let sizes = PoolSizes::default()
        .sets(&fog_set_bindings(), f)
        .sets(&froxel_set_bindings(), f)
        .build();
    let info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(2 * f)
        .pool_sizes(&sizes);
    device
        .create_descriptor_pool(&info)
        .map_err(|e| super::error::map_vk_result(e, "fog descriptor pool"))
}

// The five bindings of a per-frame fog-render view set: the FogParams +
// FogFroxelParams UBOs, the scene depth, and the froxel volume with its sampler.
#[derive(Clone, Copy)]
struct FogViewBindings {
    params_ubo: vk::Buffer,
    depth_view: vk::ImageView,
    froxel_ubo: vk::Buffer,
    volume_view: vk::ImageView,
    volume_sampler: vk::Sampler,
}

fn write_view_set(device: &VkDevice, set: vk::DescriptorSet, bindings: FogViewBindings) {
    let FogViewBindings {
        params_ubo,
        depth_view,
        froxel_ubo,
        volume_view,
        volume_sampler,
    } = bindings;
    SetWrites::new(set)
        .uniform_buffer(0, params_ubo, size_of::<FogParams>() as u64)
        .sampled_image(1, depth_view)
        .uniform_buffer(2, froxel_ubo, size_of::<FogFroxelParams>() as u64)
        .sampled_image(3, volume_view)
        .sampler(4, volume_sampler)
        .apply(device);
}

// The six bindings of a per-frame froxel compute set: the FogParams,
// FogFroxelParams, and ShadowUniforms UBOs, the CSM shadow map with its compare
// sampler, and the froxel volume storage image.
#[derive(Clone, Copy)]
struct FogFroxelBindings {
    params_ubo: vk::Buffer,
    froxel_ubo: vk::Buffer,
    shadow_ubo: vk::Buffer,
    shadow_map_view: vk::ImageView,
    shadow_sampler: vk::Sampler,
    volume_storage_view: vk::ImageView,
    far_shadow_view: vk::ImageView,
}

fn write_froxel_set(device: &VkDevice, set: vk::DescriptorSet, bindings: FogFroxelBindings) {
    let FogFroxelBindings {
        params_ubo,
        froxel_ubo,
        shadow_ubo,
        shadow_map_view,
        shadow_sampler,
        volume_storage_view,
        far_shadow_view,
    } = bindings;
    SetWrites::new(set)
        .uniform_buffer(0, params_ubo, size_of::<FogParams>() as u64)
        .uniform_buffer(1, froxel_ubo, size_of::<FogFroxelParams>() as u64)
        .uniform_buffer(2, shadow_ubo, size_of::<ShadowUniforms>() as u64)
        .sampled_image(3, shadow_map_view)
        .storage_image(4, volume_storage_view)
        .sampler(5, shadow_sampler)
        .sampled_image(6, far_shadow_view)
        .apply(device);
}

// Create the shared 3D RGBA16F froxel volume (STORAGE | SAMPLED, GPU-local).
fn create_volume_image(alloc: &DeviceAllocator) -> RenderResult<PooledImage> {
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
    alloc.create_image(&img_info, vk::MemoryPropertyFlags::DEVICE_LOCAL)
}

// A whole-image 3D view of the froxel volume (used for both the compute
// storage bind and the fragment sampled bind).
fn create_volume_view(device: &VkDevice, image: vk::Image) -> RenderResult<vk::ImageView> {
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
    unsafe { device.create_image_view(&info, None) }
        .map_err(|e| super::error::map_vk_result(e, "fog volume view"))
}

// Linear clamp-to-edge sampler for the trilinear volume read.
fn create_volume_sampler(device: &VkDevice) -> RenderResult<OwnedSampler> {
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
    device
        .create_sampler(&info)
        .map_err(|e| super::error::map_vk_result(e, "fog volume sampler"))
}

fn compile_fog_shaders(hot_reload: bool, msaa: bool) -> RenderResult<(Vec<u8>, Vec<u8>)> {
    let vert = super::builtin_shaders::FULLSCREEN_VERT.compile(hot_reload)?;
    let frag = super::builtin_shaders::FOG_FRAG
        .at(msaa)
        .compile(hot_reload)?;
    Ok((vert, frag))
}

// Rebuild the fog graphics pipeline against the existing layout. Used by the
// Vulkan shader hot-reload path. The caller is responsible for destroying the
// previous pipeline only after this call succeeds.
pub(in crate::vulkan) fn rebuild_fog_pipeline(
    device: &VkDevice,
    fog: &FogResources,
    msaa: bool,
    hot_reload: bool,
) -> RenderResult<OwnedPipeline> {
    let (vert_spv, frag_spv) = compile_fog_shaders(hot_reload, msaa)?;
    create_fog_pipeline(
        device,
        super::context::HDR_FORMAT,
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
) -> RenderResult<OwnedPipeline> {
    let spv = FOG_FROXEL.compile(hot_reload)?;
    compute_pipeline(
        device,
        fog.froxel_pipeline_layout.handle(),
        &spv,
        "fog froxel",
    )
}

// The fog pass composites `(scattered, 1 - T)` over the SINGLE-SAMPLE resolved
// HDR target (`final = scattered + transmittance * scene`) regardless of MSAA.
fn create_fog_pipeline(
    device: &VkDevice,
    hdr_format: vk::Format,
    layout: vk::PipelineLayout,
    vert_spv: &[u8],
    frag_spv: &[u8],
) -> RenderResult<OwnedPipeline> {
    GraphicsPipelineDesc {
        ..GraphicsPipelineDesc::fullscreen(
            vert_spv,
            frag_spv,
            layout,
            PassFormats::color(hdr_format),
            &[Blend::PremultipliedOver],
        )
    }
    .build(device, "fog")
}

// Encoder

impl VkContext {
    // Encode the froxel volume compute pass ahead of `Fog`, uploading both
    // per-frame UBOs so `encode_fog` only binds.
    pub(in crate::vulkan) fn encode_fog_froxel(
        &self,
        rec: &Recorder<'_>,
        frame_idx: usize,
        blocks: &FogBlocks,
    ) {
        let Some(fog) = &self.fog.resources else {
            return;
        };
        fog.params_ubos[frame_idx].write_val(0, &blocks.params);
        fog.froxel_ubos[frame_idx].write_val(0, &blocks.froxel);

        // Both transitions are graph-driven, including the order after last frame's
        // fog read; the shadow map's compute tap is a declared read.
        rec.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &fog.froxel_pipeline);
        rec.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            &fog.froxel_pipeline_layout,
            0,
            std::slice::from_ref(&fog.froxel_sets[frame_idx]),
            &[],
        );
        rec.dispatch_threads([FOG_FROXEL_X, FOG_FROXEL_Y, 1], FROXEL_GROUP);

        // The froxel volume's GENERAL -> SHADER_READ_ONLY close comes from the
        // Fog consumer barrier, and next frame's Shadow producer opens from a
        // resting scope that names both shader stages, so this frame's compute tap
        // is ordered against it.
    }

    // Encode the fog pass: alpha-blend the froxel volume's `(scattered, 1 - T)`
    // over the resolved HDR scene, using the UBOs `encode_fog_froxel` uploaded.
    pub(in crate::vulkan) fn encode_fog(&self, rec: &Recorder<'_>, frame_idx: usize) {
        let Some(fog) = &self.fog.resources else {
            return;
        };

        let extent = self.targets.render_extent;

        // Main depth is already in SHADER_READ_ONLY for the fragment's scene-depth
        // sample: the graph declares this pass's depth read and the executor emits
        // the transition ahead of this command buffer.

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

        rec.render_over_sampled(&self.targets.hdr_resolve_images[frame_idx], extent, |rec| {
            rec.set_viewport(&vp_state);
            rec.set_scissor(&scissor);
            rec.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, &fog.pipeline);
            rec.bind_descriptor_sets(
                vk::PipelineBindPoint::GRAPHICS,
                &fog.pipeline_layout,
                0,
                std::slice::from_ref(&fog.view_sets[frame_idx]),
                &[],
            );
            rec.draw(3, 1, 0, 0);
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::vulkan::builtin_shaders::{CompileProgram as _, FOG_FROXEL};
    use concinnity_engine::gfx::render_types::{FogFroxelParams, FogParams};
    use std::mem::size_of;

    #[test]
    fn fog_params_ubo_size_matches_hlsl() {
        // Both halves of fog.hlsl read the same 176 B std140 FogBlock.
        assert_eq!(size_of::<FogParams>(), 176);
    }

    #[test]
    fn fog_froxel_params_ubo_size_matches_hlsl() {
        // The FogFroxelBlock std140 layout (mat4 + uvec3 + uint + 2 float + vec2)
        // is 96 B; the offsets are pinned by the engine render_types tests.
        assert_eq!(size_of::<FogFroxelParams>(), 96);
    }

    #[test]
    fn fog_shaders_compile() {
        concinnity_shader::require_dxc!();
        // Compile the froxel-sampling fragment shader (both MSAA modes) and
        // the froxel compute kernel for Vulkan, so a shader regression fails
        // the test suite without needing a GPU.
        super::compile_fog_shaders(false, false).expect("fog shaders (no MSAA) compile");
        super::compile_fog_shaders(false, true).expect("fog shaders (MSAA) compile");
        FOG_FROXEL
            .compile(false)
            .expect("fog froxel kernel compiles");
    }
}
