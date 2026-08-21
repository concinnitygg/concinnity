// Vulkan image/texture creation helpers.
// All uploads go through a host-visible staging buffer that is blit-copied to
// a device-local image via a one-shot command buffer.

use ash::vk;

use crate::vulkan::owned::{OwnedSampler, VkDevice};

use super::allocator::{DeviceAllocator, PooledBuffer, PooledImage};

// Opaque handle to a GPU image: cached raw handles for the bind sites, backed
// by a pooled allocation that owns the image, its memory, and every view.
// Dropping the last holder retires all of them through the allocator.
pub(super) struct GpuImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    // Auxiliary image views for the same image (e.g. per-cascade DSVs for an
    // array shadow map). Owned by the pooled allocation alongside `view`.
    pub aux_views: Vec<vk::ImageView>,
    pooled: PooledImage,
}

impl GpuImage {
    // A null-handle placeholder, replaced by a real allocation before first use.
    pub(super) fn null() -> Self {
        Self {
            image: vk::Image::null(),
            view: vk::ImageView::null(),
            aux_views: Vec::new(),
            pooled: PooledImage::null(),
        }
    }

    // Wrap a pooled image and its primary view, tying the view's lifetime to
    // the allocation.
    pub(super) fn from_pooled(pooled: PooledImage, view: vk::ImageView) -> Self {
        pooled.attach_view(view);
        Self {
            image: pooled.image(),
            view,
            aux_views: Vec::new(),
            pooled,
        }
    }

    // Add an auxiliary view, destroyed with the image.
    pub(super) fn push_aux_view(&mut self, view: vk::ImageView) {
        self.pooled.attach_view(view);
        self.aux_views.push(view);
    }

    // Wrap handles owned elsewhere (e.g. the transient pool), so a mip chain
    // can index owned and borrowed images uniformly. Dropping it releases
    // nothing.
    pub(super) fn borrowed(image: vk::Image, view: vk::ImageView) -> Self {
        Self {
            image,
            view,
            aux_views: Vec::new(),
            pooled: PooledImage::null(),
        }
    }
}

// Find a memory type index that satisfies both the type filter and required properties.
pub(super) fn find_memory_type(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> Result<u32, String> {
    // SAFETY: a property query on a live handle; it only reads.
    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    for i in 0..mem_props.memory_type_count {
        if (type_filter & (1 << i)) != 0
            && mem_props.memory_types[i as usize]
                .property_flags
                .contains(properties)
        {
            return Ok(i);
        }
    }
    Err("no suitable memory type found".to_string())
}

// Immutable description of a 2-D image to allocate.
#[derive(Clone, Copy)]
pub(super) struct ImageSpec {
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub tiling: vk::ImageTiling,
    pub usage: vk::ImageUsageFlags,
    pub mem_props: vk::MemoryPropertyFlags,
    pub samples: vk::SampleCountFlags,
}

// Allocate a pooled VkImage from `spec`.
pub(super) fn create_image(
    alloc: &DeviceAllocator,
    spec: &ImageSpec,
) -> crate::gfx::error::RenderResult<PooledImage> {
    let &ImageSpec {
        width,
        height,
        format,
        tiling,
        usage,
        mem_props,
        samples,
    } = spec;
    let img_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .format(format)
        .tiling(tiling)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(samples);
    alloc.create_image(&img_info, mem_props)
}

// Create a VkImageView for a 2-D image.
pub(super) fn create_image_view(
    device: &VkDevice,
    image: vk::Image,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
) -> Result<vk::ImageView, String> {
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );
    // SAFETY: the create-info and every slice it borrows are live for the call, and each handle it
    // names belongs to this device.
    unsafe { device.create_image_view(&view_info, None) }
        .map_err(|e| format!("create_image_view: {e}"))
}

// Record and submit a short-lived command buffer without waiting for it.
// Returns the still-executing command buffer; the caller must not free it (or
// destroy anything it references) until the GPU provably retired it -- either
// by a queue/device wait, or because a later fence on the same queue signalled
// (fence signals cover all prior submissions on the queue).
pub(super) fn one_shot_submit_nowait<F>(
    device: &VkDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    f: F,
) -> Result<vk::CommandBuffer, String>
where
    F: FnOnce(vk::CommandBuffer),
{
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    // SAFETY: the create-info and every slice it borrows are live for the call, and each handle it
    // names belongs to this device.
    let cmd = unsafe { device.allocate_command_buffers(&alloc_info) }
        .map_err(|e| format!("one_shot allocate: {e}"))?[0];

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    // SAFETY: `cmd` was allocated from this device's pool and is not in flight (its face fence was
    // waited on), so it is in the initial state that `begin` requires.
    unsafe { device.begin_command_buffer(cmd, &begin_info) }
        .map_err(|e| format!("one_shot begin: {e}"))?;

    f(cmd);

    // SAFETY: `cmd` is in the recording state, which is what `end_command_buffer` requires.
    unsafe { device.end_command_buffer(cmd) }.map_err(|e| format!("one_shot end: {e}"))?;

    let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
    // SAFETY: every command buffer in `submit_bufs` was ended and belongs to this frame slot, the
    // semaphores and fence were created from this device, and `submit_info` borrows all of them for
    // the call.
    unsafe { device.queue_submit(queue, std::slice::from_ref(&submit_info), vk::Fence::null()) }
        .map_err(|e| format!("one_shot submit: {e}"))?;
    Ok(cmd)
}

// Execute a short-lived command buffer and wait for it to complete.
pub(super) fn one_shot_submit<F>(
    device: &VkDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    f: F,
) -> Result<(), String>
where
    F: FnOnce(vk::CommandBuffer),
{
    let cmd = one_shot_submit_nowait(device, command_pool, queue, f)?;
    // SAFETY: `queue` belongs to this device; the wait takes no borrowed state.
    unsafe { device.queue_wait_idle(queue) }.map_err(|e| format!("one_shot wait: {e}"))?;
    // SAFETY: every handle here was created from this device and is destroyed exactly once; the
    // caller has already waited for the device to go idle, so no submission still references them.
    unsafe { device.free_command_buffers(command_pool, std::slice::from_ref(&cmd)) };
    Ok(())
}

// A layout transition described by its endpoints and the aspect it applies to.
#[derive(Clone, Copy)]
pub(super) struct LayoutTransition {
    pub old_layout: vk::ImageLayout,
    pub new_layout: vk::ImageLayout,
    pub aspect: vk::ImageAspectFlags,
}

// The subset of an image's layers and mips a barrier or copy touches.
#[derive(Clone, Copy)]
pub(super) struct SubresourceRange {
    pub base_layer: u32,
    pub layer_count: u32,
    pub base_mip: u32,
    pub mip_count: u32,
}

// Transition `layer_count` layers of an image from one layout to another via
// a pipeline barrier. Used by the array shadow map and cube uploads.
pub(super) fn transition_image_layout_range(
    device: &VkDevice,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    transition: LayoutTransition,
    range: SubresourceRange,
) {
    let LayoutTransition {
        old_layout,
        new_layout,
        aspect,
    } = transition;
    let SubresourceRange {
        base_layer,
        layer_count,
        base_mip,
        mip_count,
    } = range;
    let (src_access, dst_access, src_stage, dst_stage) =
        layout_transition_access(old_layout, new_layout);

    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .base_mip_level(base_mip)
                .level_count(mip_count)
                .base_array_layer(base_layer)
                .layer_count(layer_count),
        )
        .src_access_mask(src_access)
        .dst_access_mask(dst_access);

    // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice these
    // commands name is live for the call.
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            std::slice::from_ref(&barrier),
        );
    }
}

fn layout_transition_access(
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) -> (
    vk::AccessFlags,
    vk::AccessFlags,
    vk::PipelineStageFlags,
    vk::PipelineStageFlags,
) {
    match (old_layout, new_layout) {
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        ),
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        ),
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL) => (
            vk::AccessFlags::empty(),
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        ),
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) => (
            vk::AccessFlags::empty(),
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        ),
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
            vk::AccessFlags::empty(),
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        ),
        (
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ) => (
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        ),
        (
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        ) => (
            vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        ),
        _ => (
            vk::AccessFlags::empty(),
            vk::AccessFlags::empty(),
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
        ),
    }
}

// Transition an image from one layout to another via a pipeline barrier.
pub(super) fn transition_image_layout(
    device: &VkDevice,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    aspect: vk::ImageAspectFlags,
) {
    transition_image_layout_range(
        device,
        cmd,
        image,
        LayoutTransition {
            old_layout,
            new_layout,
            aspect,
        },
        SubresourceRange {
            base_layer: 0,
            layer_count: 1,
            base_mip: 0,
            mip_count: 1,
        },
    );
}

// Variant of `transition_image_layout` that covers every layer of an array
// image (e.g. the 4-layer shadow array).
pub(super) fn transition_image_layout_array(
    device: &VkDevice,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    aspect: vk::ImageAspectFlags,
    layer_count: u32,
) {
    transition_image_layout_range(
        device,
        cmd,
        image,
        LayoutTransition {
            old_layout,
            new_layout,
            aspect,
        },
        SubresourceRange {
            base_layer: 0,
            layer_count,
            base_mip: 0,
            mip_count: 1,
        },
    );
}

// The handles needed to allocate a resource and run a one-shot upload: the
// device allocator plus the transient command pool and the queue the copy is
// submitted to.
#[derive(Clone, Copy)]
pub(super) struct GpuUploadContext<'a> {
    pub alloc: &'a DeviceAllocator,
    pub device: &'a VkDevice,
    pub command_pool: vk::CommandPool,
    pub queue: vk::Queue,
}

// The transient resources a deferred texture upload leaves in flight: the
// staging buffer the copy reads from and the submitted one-shot command
// buffer. Neither may be freed until the GPU retired the upload; the texture
// streaming path parks these on `VkContext::stream_retires`.
pub(super) struct UploadInFlight {
    pub staging: PooledBuffer,
    pub cmd: vk::CommandBuffer,
}

// A streamed texture swap's GPU debris, freed once `VkContext::stream_frame`
// reaches `retire_at`: the replaced pool image (pending frames may still
// sample it, and the per-frame pool copies re-point over the next
// `frames_in_flight` ticks) plus the upload's in-flight transients (still
// executing when parked; covered by the first frame fence signalled after the
// upload's submission). The image and staging buffer retire through the
// allocator when this drops; only the command buffer is freed by hand.
pub(super) struct StreamedUploadRetire {
    pub _image: GpuImage,
    pub _staging: PooledBuffer,
    pub cmd: vk::CommandBuffer,
    pub retire_at: u64,
}

impl StreamedUploadRetire {
    pub(super) fn destroy(&self, device: &VkDevice, command_pool: vk::CommandPool) {
        // SAFETY: the handle was created from this device and is destroyed exactly once; the caller
        // has already waited for the device to go idle, so no submission still references it.
        unsafe {
            device.free_command_buffers(command_pool, std::slice::from_ref(&self.cmd));
        }
    }
}

// Upload RGBA pixel data to a device-local RGBA8_UNORM image with a full mip
// chain, without waiting for the copy: the command buffer is submitted and
// left executing. The final layout transition (TRANSFER_DST -> SHADER_READ_
// ONLY, fragment-stage scope) orders every later submission on the same queue
// after the copy, so the image is safe to sample from any subsequently
// submitted frame; only freeing the returned in-flight resources needs GPU
// retirement. The chain is box-filtered on the CPU (`crate::gfx::mipmap`) and
// every level is uploaded so the texture minifies through hardware trilinear /
// aniso selection instead of aliasing from a single mip-0 sample at a distance.
pub(super) fn upload_texture_deferred(
    ctx: &GpuUploadContext,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> crate::gfx::error::RenderResult<(GpuImage, UploadInFlight)> {
    let base = (width as usize) * (height as usize) * 4;
    if pixels.len() < base {
        return Err(format!(
            "pixel data too short for {}x{} RGBA texture ({} bytes, need {})",
            width,
            height,
            pixels.len(),
            base
        )
        .into());
    }

    let chain = crate::gfx::mipmap::generate_mip_chain(width, height, pixels);
    let levels: Vec<TextureLevel<'_>> = chain
        .iter()
        .map(|m| TextureLevel {
            width: m.width,
            height: m.height,
            data: &m.pixels,
        })
        .collect();
    upload_texture_levels_deferred(ctx, vk::Format::R8G8B8A8_UNORM, &levels)
}

// One mip level handed to `upload_texture_levels_deferred`: its texel
// dimensions plus the tightly packed level bytes (RGBA8 pixels or 4x4 blocks,
// per the upload's Vulkan format).
pub(super) struct TextureLevel<'a> {
    pub width: u32,
    pub height: u32,
    pub data: &'a [u8],
}

// Vulkan equivalent of a compiled texture payload format. The BC formats need
// the `textureCompressionBC` device feature, enabled in `vulkan::device`.
fn vk_texture_format(format: concinnity_cpu::build::texture::TextureFormat) -> vk::Format {
    use concinnity_cpu::build::texture::TextureFormat;
    match format {
        TextureFormat::Rgba8 => vk::Format::R8G8B8A8_UNORM,
        TextureFormat::Bc1 => vk::Format::BC1_RGBA_UNORM_BLOCK,
        TextureFormat::Bc3 => vk::Format::BC3_UNORM_BLOCK,
        TextureFormat::Bc5 => vk::Format::BC5_UNORM_BLOCK,
        TextureFormat::Bc7 => vk::Format::BC7_UNORM_BLOCK,
    }
}

// Upload a decoded texture into a 2-D image. RGBA8 images take the CPU
// mip-generation path above; block-compressed images (BC1/BC3/BC5/BC7) upload
// their container mip chain verbatim.
pub(super) fn upload_texture_image_deferred(
    ctx: &GpuUploadContext,
    image: &concinnity_cpu::build::texture::TextureImage,
) -> crate::gfx::error::RenderResult<(GpuImage, UploadInFlight)> {
    use concinnity_cpu::build::texture::TextureFormat;
    if image.format == TextureFormat::Rgba8 {
        let mip = image
            .mips
            .first()
            .ok_or("RGBA8 texture image has no mip level")?;
        return upload_texture_deferred(ctx, mip.width, mip.height, &mip.data);
    }
    let levels: Vec<TextureLevel<'_>> = image
        .mips
        .iter()
        .map(|m| TextureLevel {
            width: m.width,
            height: m.height,
            data: &m.data,
        })
        .collect();
    upload_texture_levels_deferred(ctx, vk_texture_format(image.format), &levels)
}

// Synchronous `upload_texture_image_deferred`.
pub(super) fn upload_texture_image(
    ctx: &GpuUploadContext,
    image: &concinnity_cpu::build::texture::TextureImage,
) -> crate::gfx::error::RenderResult<GpuImage> {
    let (img, in_flight) = upload_texture_image_deferred(ctx, image)?;
    finish_upload(ctx, in_flight)?;
    Ok(img)
}

// Wait out a deferred upload and free its transient resources: the queue
// idles, the command buffer returns to the pool, and the dropped staging
// retires immediately so an upload loop reuses one staging range instead of
// accumulating every upload's.
pub(super) fn finish_upload(
    ctx: &GpuUploadContext,
    in_flight: UploadInFlight,
) -> Result<(), String> {
    // SAFETY: `ctx.queue` belongs to `ctx.device`; the wait takes no borrowed state.
    unsafe { ctx.device.queue_wait_idle(ctx.queue) }.map_err(|e| format!("upload wait: {e}"))?;
    // SAFETY: every handle here was created from this device and is destroyed exactly once; the
    // caller has already waited for the device to go idle, so no submission still references them.
    unsafe {
        ctx.device
            .free_command_buffers(ctx.command_pool, std::slice::from_ref(&in_flight.cmd));
    }
    drop(in_flight);
    ctx.alloc.reclaim_idle();
    Ok(())
}

// Upload pre-built mip levels of any Vulkan format into a device-local image
// without waiting for the copy. `buffer_row_length(0)` keeps each region
// tightly packed, which for block-compressed formats means one row of 4x4
// blocks per image row.
fn upload_texture_levels_deferred(
    ctx: &GpuUploadContext,
    format: vk::Format,
    levels: &[TextureLevel<'_>],
) -> crate::gfx::error::RenderResult<(GpuImage, UploadInFlight)> {
    let &GpuUploadContext {
        alloc,
        device,
        command_pool,
        queue,
    } = ctx;
    let base = levels.first().ok_or("texture upload has no mip level")?;
    let (width, height) = (base.width, base.height);
    let mip_count = levels.len() as u32;

    // One packed staging buffer holding mip 0..N concatenated.
    let total: usize = levels.iter().map(|m| m.data.len()).sum();
    let staging = alloc.create_buffer(
        total as vk::DeviceSize,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let mut off = 0usize;
    for m in levels {
        staging.write_bytes(off, m.data);
        off += m.data.len();
    }

    // Device-local image with the full mip chain.
    let img_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(mip_count)
        .array_layers(1)
        .format(format)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::TYPE_1);
    let pooled = alloc.create_image(&img_info, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
    let image = pooled.image();

    let cmd = one_shot_submit_nowait(device, command_pool, queue, |cmd| {
        transition_image_layout_range(
            device,
            cmd,
            image,
            LayoutTransition {
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                aspect: vk::ImageAspectFlags::COLOR,
            },
            SubresourceRange {
                base_layer: 0,
                layer_count: 1,
                base_mip: 0,
                mip_count,
            },
        );
        let mut regions: Vec<vk::BufferImageCopy> = Vec::with_capacity(mip_count as usize);
        let mut off = 0u64;
        for (m, level) in levels.iter().enumerate() {
            regions.push(
                vk::BufferImageCopy::default()
                    .buffer_offset(off)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(m as u32)
                            .base_array_layer(0)
                            .layer_count(1),
                    )
                    .image_offset(vk::Offset3D::default())
                    .image_extent(vk::Extent3D {
                        width: level.width,
                        height: level.height,
                        depth: 1,
                    }),
            );
            off += level.data.len() as u64;
        }
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_copy_buffer_to_image(
                cmd,
                staging.buffer(),
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            );
        }
        transition_image_layout_range(
            device,
            cmd,
            image,
            LayoutTransition {
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                aspect: vk::ImageAspectFlags::COLOR,
            },
            SubresourceRange {
                base_layer: 0,
                layer_count: 1,
                base_mip: 0,
                mip_count,
            },
        );
    })?;

    // View spanning every mip.
    let view = {
        let info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(mip_count)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        // SAFETY: the create-info and every slice it borrows are live for the call, and each handle
        // it names belongs to this device.
        unsafe { device.create_image_view(&info, None) }
            .map_err(|e| format!("create_image_view: {e}"))?
    };

    Ok((
        GpuImage::from_pooled(pooled, view),
        UploadInFlight { staging, cmd },
    ))
}

// Synchronous `upload_texture_deferred`: waits for the copy, then frees the
// transient upload resources. The init-time upload paths use this.
pub(super) fn upload_texture(
    ctx: &GpuUploadContext,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> Result<GpuImage, String> {
    let (img, in_flight) = upload_texture_deferred(ctx, width, height, pixels)?;
    finish_upload(ctx, in_flight)?;
    Ok(img)
}

// Create a 1x1 opaque white RGBA texture (fallback when no albedo asset is present).
pub(super) fn create_fallback_white(ctx: &GpuUploadContext) -> Result<GpuImage, String> {
    upload_texture(ctx, 1, 1, &[255u8, 255, 255, 255])
}

// Create a 1x1 flat-normal RGBA texture (tangent-space (0,0,1) = no perturbation).
pub(super) fn create_fallback_flat_normal(ctx: &GpuUploadContext) -> Result<GpuImage, String> {
    upload_texture(ctx, 1, 1, &[128u8, 128, 255, 255])
}

// Upload a 3D colour-grading LUT from a `ColorLut` payload. `data` is the raw
// RGBA8 emitted by `build/color_lut.rs`: `size`³ texels ordered red-fastest,
// then green, then blue, which is the natural row/slice order of a `TYPE_3D`
// image, so the byte slice copies in verbatim. The returned `GpuImage` has a
// `VK_IMAGE_VIEW_TYPE_3D` view left in `SHADER_READ_ONLY_OPTIMAL`, ready for
// the composite pass to sample as a `sampler3D`.
pub(super) fn upload_color_lut(
    ctx: &GpuUploadContext,
    size: u32,
    data: &[u8],
) -> Result<GpuImage, String> {
    let &GpuUploadContext {
        alloc,
        device,
        command_pool,
        queue,
    } = ctx;
    let needed = (size as usize).pow(3) * 4;
    if data.len() < needed {
        return Err(format!(
            "color LUT data too short for size {}: {} bytes, need {}",
            size,
            data.len(),
            needed
        ));
    }

    // Staging buffer (host visible).
    let byte_size = needed as vk::DeviceSize;
    let staging = alloc.create_buffer(
        byte_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    staging.write_bytes(0, &data[..needed]);

    // Device-local 3D image. `create_image` is TYPE_2D only, so the LUT image
    // is built inline.
    let img_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_3D)
        .extent(vk::Extent3D {
            width: size,
            height: size,
            depth: size,
        })
        .mip_levels(1)
        .array_layers(1)
        .format(vk::Format::R8G8B8A8_UNORM)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::TYPE_1);
    let pooled = alloc
        .create_image(&img_info, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .map_err(|e| format!("create_image (LUT): {e}"))?;
    let image = pooled.image();

    one_shot_submit(device, command_pool, queue, |cmd| {
        transition_image_layout(
            device,
            cmd,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
        );
        let copy_region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_offset(vk::Offset3D::default())
            .image_extent(vk::Extent3D {
                width: size,
                height: size,
                depth: size,
            });
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_copy_buffer_to_image(
                cmd,
                staging.buffer(),
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&copy_region),
            );
        }
        transition_image_layout(
            device,
            cmd,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
        );
    })?;

    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_3D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );
    // SAFETY: the create-info and every slice it borrows are live for the call, and each handle it
    // names belongs to this device.
    let view = unsafe { device.create_image_view(&view_info, None) }
        .map_err(|e| format!("create_image_view (LUT): {e}"))?;

    Ok(GpuImage::from_pooled(pooled, view))
}

// Upload a square float lookup table as a 2D image. `components` selects the
// format: 4 -> R32G32B32A32_SFLOAT, 2 -> R32G32_SFLOAT. Used for the two
// area-light LTC tables, which are scene-independent (fitted at build time) and
// uploaded once at init.
pub(super) fn upload_float_lut(
    ctx: &GpuUploadContext<'_>,
    size: u32,
    components: u32,
    texels: &[f32],
) -> Result<GpuImage, String> {
    let GpuUploadContext {
        alloc,
        device,
        command_pool,
        queue,
    } = *ctx;
    let needed = (size as usize) * (size as usize) * components as usize;
    if texels.len() < needed {
        return Err(format!(
            "float LUT data too short for {size}x{size}x{components}: {} floats, need {needed}",
            texels.len()
        ));
    }
    let format = match components {
        4 => vk::Format::R32G32B32A32_SFLOAT,
        2 => vk::Format::R32G32_SFLOAT,
        other => return Err(format!("unsupported float LUT component count {other}")),
    };

    let byte_size = (needed * std::mem::size_of::<f32>()) as vk::DeviceSize;
    let staging = alloc.create_buffer(
        byte_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    staging.write_slice(0, &texels[..needed]);

    let img_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width: size,
            height: size,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .format(format)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::TYPE_1);
    let pooled = alloc
        .create_image(&img_info, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .map_err(|e| format!("create_image (float LUT): {e}"))?;
    let image = pooled.image();

    one_shot_submit(device, command_pool, queue, |cmd| {
        transition_image_layout(
            device,
            cmd,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
        );
        let copy_region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_offset(vk::Offset3D::default())
            .image_extent(vk::Extent3D {
                width: size,
                height: size,
                depth: 1,
            });
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_copy_buffer_to_image(
                cmd,
                staging.buffer(),
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&copy_region),
            );
        }
        transition_image_layout(
            device,
            cmd,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
        );
    })?;

    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );
    // SAFETY: the create-info and every slice it borrows are live for the call, and each handle it
    // names belongs to this device.
    let view = unsafe { device.create_image_view(&view_info, None) }
        .map_err(|e| format!("create_image_view (float LUT): {e}"))?;

    Ok(GpuImage::from_pooled(pooled, view))
}

// Build a 2x2x2 identity colour LUT: the eight corners of the unit RGB cube.
// Mirrors `metal/texture.rs::create_fallback_color_lut`. With the identity LUT
// the composite grade is a no-op at any `lut_strength`, so the `sampler3D`
// binding stays valid even when the world declares no `ColorLut`.
pub(super) fn create_fallback_color_lut(ctx: &GpuUploadContext) -> Result<GpuImage, String> {
    // Red-fastest, then green, then blue, matching the payload texel order.
    let mut data = Vec::with_capacity(2 * 2 * 2 * 4);
    for b in 0..2u8 {
        for g in 0..2u8 {
            for r in 0..2u8 {
                data.extend_from_slice(&[r * 255, g * 255, b * 255, 255]);
            }
        }
    }
    upload_color_lut(ctx, 2, &data)
}

// Create a `layers`-slice D32_SFLOAT array shadow map. The returned `view` is
// a single sampled 2D-array view covering every layer (bound at descriptor
// set=0 binding=3 in the main pass); `aux_views` holds one single-layer 2D
// view per cascade for use as a per-slice depth attachment in the shadow
// pass.
//
// When `size > 0`, creates a full shadow map; otherwise a 1×1 single-layer
// fallback (depth=1.0 = fully lit). The fallback intentionally uses a single
// array layer because the shader's cascade selection falls back to cascade 0
// when `cascade_splits == +inf`, so layer 0 is the only one ever sampled.
pub(super) fn create_shadow_map_array(
    ctx: &GpuUploadContext,
    size: u32,
    layers: u32,
) -> Result<GpuImage, String> {
    let &GpuUploadContext {
        alloc,
        device,
        command_pool,
        queue,
    } = ctx;
    let (w, h, layer_count) = if size > 0 {
        (size, size, layers.max(1))
    } else {
        (1, 1, 1)
    };
    let img_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width: w,
            height: h,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(layer_count)
        .format(vk::Format::D32_SFLOAT)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::TYPE_1);
    let pooled = alloc
        .create_image(&img_info, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .map_err(|e| format!("create_image (shadow array): {e}"))?;
    let image = pooled.image();

    // Rest the cascades sampled. The graph's Shadow producer barrier transitions
    // them to DEPTH_STENCIL_ATTACHMENT before each shadow loop and Main's consumer
    // returns them here, so the cross-frame reset is the graph's producer barrier,
    // not an inline end-of-frame transition. Initialising sampled makes frame 0's
    // producer barrier (SHADER_READ_ONLY -> DEPTH_STENCIL_ATTACHMENT) start from
    // the image's real layout.
    one_shot_submit(device, command_pool, queue, |cmd| {
        transition_image_layout_array(
            device,
            cmd,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageAspectFlags::DEPTH,
            layer_count,
        );
    })?;

    // Sampled view: 2D array over every layer.
    let view = {
        let info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
            .format(vk::Format::D32_SFLOAT)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(layer_count),
            );
        // SAFETY: the create-info and every slice it borrows are live for the call, and each handle
        // it names belongs to this device.
        unsafe { device.create_image_view(&info, None) }
            .map_err(|e| format!("shadow array view: {e}"))?
    };

    // Per-slice attachment views (one per cascade).
    let mut aux_views = Vec::with_capacity(layer_count as usize);
    for i in 0..layer_count {
        let info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::D32_SFLOAT)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(i)
                    .layer_count(1),
            );
        // SAFETY: the create-info and every slice it borrows are live for the call, and each handle
        // it names belongs to this device.
        let v = unsafe { device.create_image_view(&info, None) }
            .map_err(|e| format!("shadow slice view {i}: {e}"))?;
        aux_views.push(v);
    }

    let mut img = GpuImage::from_pooled(pooled, view);
    for v in aux_views {
        img.push_aux_view(v);
    }
    Ok(img)
}

// Create a device-local depth image for the main render pass.
pub(super) fn create_depth_image(
    ctx: &GpuUploadContext,
    width: u32,
    height: u32,
    samples: vk::SampleCountFlags,
) -> Result<GpuImage, String> {
    let &GpuUploadContext {
        alloc,
        device,
        command_pool,
        queue,
    } = ctx;
    let pooled = create_image(
        alloc,
        &ImageSpec {
            width,
            height,
            format: vk::Format::D32_SFLOAT,
            tiling: vk::ImageTiling::OPTIMAL,
            // SAMPLED so the projected-decal pass (and any future depth-
            // sampling effect) can read it from a fragment shader. Without
            // this, the validation layer rejects the SHADER_READ_ONLY layout
            // transition and the SAMPLED-bit-required `vkUpdateDescriptorSets`
            // write that bind the depth view to the decal's set 0 binding 2.
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            mem_props: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            samples,
        },
    )?;
    let image = pooled.image();
    one_shot_submit(device, command_pool, queue, |cmd| {
        transition_image_layout(
            device,
            cmd,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            vk::ImageAspectFlags::DEPTH,
        );
    })?;
    let view = create_image_view(
        device,
        image,
        vk::Format::D32_SFLOAT,
        vk::ImageAspectFlags::DEPTH,
    )?;
    Ok(GpuImage::from_pooled(pooled, view))
}

// Create a multisampled color image for the MSAA resolve target.
pub(super) fn create_msaa_color_image(
    ctx: &GpuUploadContext,
    width: u32,
    height: u32,
    format: vk::Format,
    samples: vk::SampleCountFlags,
) -> Result<GpuImage, String> {
    let &GpuUploadContext {
        alloc,
        device,
        command_pool,
        queue,
    } = ctx;
    let pooled = create_image(
        alloc,
        &ImageSpec {
            width,
            height,
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::TRANSIENT_ATTACHMENT
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            mem_props: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            samples,
        },
    )?;
    let image = pooled.image();
    one_shot_submit(device, command_pool, queue, |cmd| {
        transition_image_layout(
            device,
            cmd,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
        );
    })?;
    let view = create_image_view(device, image, format, vk::ImageAspectFlags::COLOR)?;
    Ok(GpuImage::from_pooled(pooled, view))
}

// Create a single-sample colour image usable as both a render target and a
// sampled texture. This is the HDR resolve target: the main pass resolves
// (or, with MSAA off, draws directly) into it, and the composite pass samples
// it to tonemap. No pre-transition is needed: the main render pass declares
// an `UNDEFINED` initial layout for it.
pub(super) fn create_hdr_resolve_image(
    alloc: &DeviceAllocator,
    device: &VkDevice,
    width: u32,
    height: u32,
    format: vk::Format,
) -> Result<GpuImage, String> {
    let pooled = create_image(
        alloc,
        &ImageSpec {
            width,
            height,
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            // TRANSFER_SRC so the raymarch pass can snapshot the resolved scene
            // into its `scene_color` refraction tap before compositing the SDF
            // volumes.
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC,
            mem_props: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            samples: vk::SampleCountFlags::TYPE_1,
        },
    )?;
    let view = create_image_view(device, pooled.image(), format, vk::ImageAspectFlags::COLOR)?;
    Ok(GpuImage::from_pooled(pooled, view))
}

// Linear repeat sampler for albedo and normal map sampling. Now that scene
// textures carry a full mip chain, `max_lod` is unclamped so minified surfaces
// trilinear-select down the chain. `max_anisotropy > 1.0` enables anisotropic
// filtering (the caller passes the device-supported degree, or <= 1.0 when the
// `samplerAnisotropy` feature is unavailable).
pub(super) fn create_sampler_linear_repeat(
    device: &VkDevice,
    max_anisotropy: f32,
) -> Result<OwnedSampler, String> {
    let aniso = max_anisotropy > 1.0;
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(aniso)
        .max_anisotropy(if aniso { max_anisotropy } else { 1.0 })
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .min_lod(0.0)
        .max_lod(vk::LOD_CLAMP_NONE);
    device
        .create_sampler(&info)
        .map_err(|e| format!("linear repeat sampler: {e}"))
}

// Compare sampler for PCF shadow sampling (LessEqual compare op).
pub(super) fn create_sampler_shadow(device: &VkDevice) -> Result<OwnedSampler, String> {
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .anisotropy_enable(false)
        .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
        .unnormalized_coordinates(false)
        .compare_enable(true)
        .compare_op(vk::CompareOp::LESS_OR_EQUAL)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR);
    device
        .create_sampler(&info)
        .map_err(|e| format!("shadow sampler: {e}"))
}

// Linear clamp sampler for text atlas lookups.
pub(super) fn create_sampler_linear_clamp(device: &VkDevice) -> Result<OwnedSampler, String> {
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .anisotropy_enable(false)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR);
    device
        .create_sampler(&info)
        .map_err(|e| format!("linear clamp sampler: {e}"))
}

// IBL textures produced by a single `EnvironmentMap` asset. Mirrors the Metal
// `EnvironmentMapTextures` shape so the fragment-shader code stays portable.
// `prefilter_mip_count == 0` is the runtime signal for "IBL disabled": the
// fragment shader keys off it and falls back to the legacy ambient path.
pub(super) struct EnvironmentMapTextures {
    pub irradiance: GpuImage,
    pub prefilter: GpuImage,
    pub prefilter_mip_count: u32,
}

// Create a RGBA32_SFLOAT cubemap image with `mip_count` mips, then upload the
// supplied byte slices via a staging buffer. `mip_bytes[m]` must hold
// `6 * (face_size >> m)² * 16` bytes in face-major order
// (+X, -X, +Y, -Y, +Z, -Z). Returns a `GpuImage` whose `view` is a
// `VK_IMAGE_VIEW_TYPE_CUBE` view spanning every mip.
fn create_cube_image(
    ctx: &GpuUploadContext,
    face_size: u32,
    mip_bytes: &[&[u8]],
) -> Result<GpuImage, String> {
    let &GpuUploadContext {
        alloc,
        device,
        command_pool,
        queue,
    } = ctx;
    let mip_count = mip_bytes.len() as u32;
    if mip_count == 0 {
        return Err("cubemap upload: mip_bytes must not be empty".into());
    }

    // Validate each mip and compute the staging buffer footprint.
    let mut mip_sizes: Vec<usize> = Vec::with_capacity(mip_count as usize);
    let mut total: usize = 0;
    for (m, bytes) in mip_bytes.iter().enumerate() {
        let s = (face_size >> m) as usize;
        if s == 0 {
            return Err(format!(
                "cubemap mip {} would have zero face size (face_size {} too small)",
                m, face_size
            ));
        }
        let face_bytes = s * s * 16;
        let needed = 6 * face_bytes;
        if bytes.len() < needed {
            return Err(format!(
                "cubemap mip {} too short: {} bytes, need {}",
                m,
                bytes.len(),
                needed
            ));
        }
        mip_sizes.push(needed);
        total += needed;
    }

    // Create the image: array_layers=6, with the CUBE_COMPATIBLE flag.
    let img_info = vk::ImageCreateInfo::default()
        .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE)
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width: face_size,
            height: face_size,
            depth: 1,
        })
        .mip_levels(mip_count)
        .array_layers(6)
        .format(vk::Format::R32G32B32A32_SFLOAT)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::TYPE_1);
    let pooled = alloc
        .create_image(&img_info, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .map_err(|e| format!("create_image (cube): {e}"))?;
    let image = pooled.image();

    // Build one packed staging buffer with mip 0..N concatenated.
    let staging = alloc.create_buffer(
        total as vk::DeviceSize,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let mut off = 0usize;
    for (m, bytes) in mip_bytes.iter().enumerate() {
        staging.write_bytes(off, &bytes[..mip_sizes[m]]);
        off += mip_sizes[m];
    }

    // Transition all 6 layers / N mips to TRANSFER_DST, copy each face per mip,
    // then transition to SHADER_READ_ONLY_OPTIMAL.
    one_shot_submit(device, command_pool, queue, |cmd| {
        transition_image_layout_range(
            device,
            cmd,
            image,
            LayoutTransition {
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                aspect: vk::ImageAspectFlags::COLOR,
            },
            SubresourceRange {
                base_layer: 0,
                layer_count: 6,
                base_mip: 0,
                mip_count,
            },
        );

        // One BufferImageCopy per (mip, face).
        let mut regions: Vec<vk::BufferImageCopy> = Vec::with_capacity((mip_count * 6) as usize);
        let mut off = 0u64;
        for m in 0..mip_count as usize {
            let s = face_size >> m;
            let face_bytes = (s as u64) * (s as u64) * 16;
            for face in 0..6u32 {
                regions.push(
                    vk::BufferImageCopy::default()
                        .buffer_offset(off)
                        .buffer_row_length(0)
                        .buffer_image_height(0)
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .mip_level(m as u32)
                                .base_array_layer(face)
                                .layer_count(1),
                        )
                        .image_offset(vk::Offset3D::default())
                        .image_extent(vk::Extent3D {
                            width: s,
                            height: s,
                            depth: 1,
                        }),
                );
                off += face_bytes;
            }
        }
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_copy_buffer_to_image(
                cmd,
                staging.buffer(),
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            );
        }

        transition_image_layout_range(
            device,
            cmd,
            image,
            LayoutTransition {
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                aspect: vk::ImageAspectFlags::COLOR,
            },
            SubresourceRange {
                base_layer: 0,
                layer_count: 6,
                base_mip: 0,
                mip_count,
            },
        );
    })?;

    // Single cube view covering all mips.
    let view = {
        let info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::CUBE)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(mip_count)
                    .base_array_layer(0)
                    .layer_count(6),
            );
        // SAFETY: the create-info and every slice it borrows are live for the call, and each handle
        // it names belongs to this device.
        unsafe { device.create_image_view(&info, None) }.map_err(|e| format!("cube view: {e}"))?
    };

    Ok(GpuImage::from_pooled(pooled, view))
}

// Upload a six-face HDR cubemap from a `CubemapTexture` payload. RGBA32F,
// 6 * face_size * face_size * 16 bytes in face-major order
// (+X, -X, +Y, -Y, +Z, -Z). Single-mip.
#[allow(dead_code)]
pub(super) fn upload_cubemap(
    ctx: &GpuUploadContext,
    face_size: u32,
    bytes: &[u8],
) -> Result<GpuImage, String> {
    create_cube_image(ctx, face_size, &[bytes])
}

// Create a 1×1 RGBA32F cube of `value` for every face. Used as the IBL
// fallback when no `EnvironmentMap` is bound: the fragment shader keys off
// `prefilter_mip_count == 0` and skips IBL math, but the cube binding must
// still resolve to a valid texture.
pub(super) fn create_fallback_cubemap(
    ctx: &GpuUploadContext,
    value: [f32; 4],
) -> Result<GpuImage, String> {
    let mut face_bytes = Vec::with_capacity(6 * 16);
    for _ in 0..6 {
        for v in &value {
            face_bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    create_cube_image(ctx, 1, &[&face_bytes])
}

// Upload an `EnvironmentMap` payload into two cube textures: a single-mip
// irradiance cube and a multi-mip prefiltered radiance cube. Mirrors the
// Metal and DirectX upload paths.
pub(super) fn upload_environment_map(
    ctx: &GpuUploadContext,
    irradiance_face: u32,
    irradiance_bytes: &[u8],
    prefilter_face: u32,
    mip_bytes: &[&[u8]],
) -> Result<EnvironmentMapTextures, String> {
    if mip_bytes.is_empty() {
        return Err("envmap upload: prefilter mip_bytes must not be empty".into());
    }
    let irradiance = create_cube_image(ctx, irradiance_face, &[irradiance_bytes])
        .map_err(|e| format!("envmap irradiance: {e}"))?;
    let prefilter = create_cube_image(ctx, prefilter_face, mip_bytes)
        .map_err(|e| format!("envmap prefilter: {e}"))?;
    Ok(EnvironmentMapTextures {
        irradiance,
        prefilter,
        prefilter_mip_count: mip_bytes.len() as u32,
    })
}

// Upload one baked reflection-probe's prefiltered radiance cube. A probe is
// sampled only by the specular reflection term (never as a skybox + no diffuse
// irradiance), so just the multi-mip prefilter cube is uploaded, not the
// irradiance cube `upload_environment_map` also builds. `mip_bytes[m]` holds
// `6 * (face >> m)² * 16` bytes in face-major order (the serialised `ENVM`
// prefilter chain from `reflection_probe::build_probe_payload`). The returned
// `GpuImage` carries a `CUBE` view spanning every mip, sampled through the
// shared `cube_sampler`. Mirrors `directx::texture::upload_probe_prefilter_cube`.
#[allow(dead_code)] // installed by the probe capture pass (next slice).
pub(super) fn upload_probe_prefilter_cube(
    ctx: &GpuUploadContext,
    prefilter_face: u32,
    mip_bytes: &[&[u8]],
) -> Result<GpuImage, String> {
    if mip_bytes.is_empty() {
        return Err("probe prefilter upload: mip_bytes must not be empty".into());
    }
    create_cube_image(ctx, prefilter_face, mip_bytes).map_err(|e| format!("probe prefilter: {e}"))
}

// Linear-clamp sampler with full mipmap support, used by the IBL prefilter
// cube (roughness → mip selection) and the irradiance cube.
pub(super) fn create_sampler_cube_linear(device: &VkDevice) -> Result<OwnedSampler, String> {
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .anisotropy_enable(false)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .min_lod(0.0)
        .max_lod(vk::LOD_CLAMP_NONE);
    device
        .create_sampler(&info)
        .map_err(|e| format!("cube sampler: {e}"))
}
