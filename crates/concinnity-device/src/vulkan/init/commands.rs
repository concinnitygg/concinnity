//! Command recording infrastructure: the context's command pool, the
//! timestamp query reset, the per-frame and per-pass command buffers, and the
//! frame sync objects.

use ash::vk;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::render_graph;

use super::InitGpu;
use crate::vulkan::owned::VkDevice;

pub(super) fn create_command_pool(
    device: &VkDevice,
    graphics_family: u32,
) -> RenderResult<vk::CommandPool> {
    let info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(graphics_family);
    // SAFETY: the create-info and every slice it borrows are live for the call, and each
    // handle it names belongs to this device.
    let command_pool = unsafe { device.create_command_pool(&info, None) }
        .map_err(|e| format!("command pool: {e}"))?;
    Ok(command_pool)
}

// Initial reset of every timestamp query slot. Without this the first
// `vkGetQueryPoolResults` call on each slot (before that slot has
// ever been written) hits an uninitialized query and the validation
// layer emits a "query not reset" error. After the reset, the slot
// is in "unavailable" state, so `get_query_pool_results` returns
// NOT_READY → 0 cleanly until `record_frame` writes the first pair.
pub(super) fn reset_timestamp_queries(
    gpu: &InitGpu<'_>,
    timestamp_query_pool: Option<vk::QueryPool>,
) -> RenderResult<()> {
    let InitGpu {
        device,
        command_pool,
        queue: graphics_queue,
        frames,
        ..
    } = *gpu;
    if let Some(pool) = timestamp_query_pool {
        crate::vulkan::texture::one_shot_submit(
            device,
            command_pool,
            graphics_queue,
            // SAFETY: `cmd` is a command buffer in the recording state, and every handle and
            // slice these commands name is live for the call.
            |cmd| unsafe {
                device.cmd_reset_query_pool(
                    cmd,
                    pool,
                    0,
                    (crate::vulkan::pass_timing::SLOTS_PER_FRAME * frames) as u32,
                );
            },
        )?;
    }
    Ok(())
}

pub(super) struct FrameCommands {
    pub(super) command_buffers: Vec<vk::CommandBuffer>,
    pub(super) start_command_pools: Vec<vk::CommandPool>,
    pub(super) start_command_buffers: Vec<vk::CommandBuffer>,
    pub(super) pass_command_pools: Vec<vk::CommandPool>,
    pub(super) pass_command_buffers: Vec<vk::CommandBuffer>,
    pub(super) image_available: Vec<vk::Semaphore>,
    pub(super) render_finished: Vec<vk::Semaphore>,
    pub(super) in_flight: Vec<vk::Fence>,
}

pub(super) fn build_frame_commands(
    gpu: &InitGpu<'_>,
    graphics_family: u32,
    swapchain_images: &[vk::Image],
) -> RenderResult<FrameCommands> {
    let InitGpu {
        device,
        command_pool,
        frames,
        ..
    } = *gpu;
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(frames as u32);
    // SAFETY: the create-info and every slice it borrows are live for the call, and each handle
    // it names belongs to this device.
    let command_buffers = unsafe { device.allocate_command_buffers(&alloc_info) }
        .map_err(|e| format!("allocate command buffers: {e}"))?;

    // Parallel command-buffer recording: a `start` outer buffer per frame
    // (leading timestamp) plus one command pool + primary buffer per
    // (frame, pass) slot. Vulkan command pools are externally
    // synchronized, so every slot gets its own pool - the rayon workers in
    // `execute_graph` never share a pool. `RESET_COMMAND_BUFFER` so each
    // buffer can be reset + re-recorded per frame (the per-frame
    // `in_flight` fence gates reuse). Indexed `frame * PASS_COUNT + pass`.
    let pass_pool_flags =
        vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER | vk::CommandPoolCreateFlags::TRANSIENT;
    let make_pool_with_buffer =
        |device: &VkDevice| -> Result<(vk::CommandPool, vk::CommandBuffer), String> {
            // SAFETY: the create-info and every slice it borrows are live for the call, and
            // each handle it names belongs to this device.
            let pool = unsafe {
                device.create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .flags(pass_pool_flags)
                        .queue_family_index(graphics_family),
                    None,
                )
            }
            .map_err(|e| format!("per-pass command pool: {e}"))?;
            // SAFETY: the create-info and every slice it borrows are live for the call, and
            // each handle it names belongs to this device.
            let buf = unsafe {
                device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
            }
            .map_err(|e| format!("per-pass command buffer: {e}"))?[0];
            Ok((pool, buf))
        };
    let mut start_command_pools = Vec::with_capacity(frames);
    let mut start_command_buffers = Vec::with_capacity(frames);
    for _ in 0..frames {
        let (pool, buf) = make_pool_with_buffer(device)?;
        start_command_pools.push(pool);
        start_command_buffers.push(buf);
    }
    let pass_pool_count = frames * render_graph::PASS_COUNT;
    let mut pass_command_pools = Vec::with_capacity(pass_pool_count);
    let mut pass_command_buffers = Vec::with_capacity(pass_pool_count);
    for _ in 0..pass_pool_count {
        let (pool, buf) = make_pool_with_buffer(device)?;
        pass_command_pools.push(pool);
        pass_command_buffers.push(buf);
    }

    // `image_available` + `in_flight` are per-frame-in-flight. The
    // render-finished semaphore is signaled by submit and waited on by
    // present, so it must be one-per-swapchain-image (indexed by the
    // acquired image index): a per-frame semaphore can still be queued
    // for presentation when its frame slot comes round again.
    let sem_info = vk::SemaphoreCreateInfo::default();
    let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
    let mut image_available = Vec::with_capacity(frames);
    let mut in_flight = Vec::with_capacity(frames);
    let mut render_finished = Vec::with_capacity(swapchain_images.len());
    for _ in 0..swapchain_images.len() {
        render_finished.push(
            // SAFETY: the create-info and every slice it borrows are live for the call, and
            // each handle it names belongs to this device.
            unsafe { device.create_semaphore(&sem_info, None) }
                .map_err(|e| format!("semaphore: {e}"))?,
        );
    }
    for _ in 0..frames {
        image_available.push(
            // SAFETY: the create-info and every slice it borrows are live for the call, and
            // each handle it names belongs to this device.
            unsafe { device.create_semaphore(&sem_info, None) }
                .map_err(|e| format!("semaphore: {e}"))?,
        );
        in_flight.push(
            // SAFETY: the create-info and every slice it borrows are live for the call, and
            // each handle it names belongs to this device.
            unsafe { device.create_fence(&fence_info, None) }.map_err(|e| format!("fence: {e}"))?,
        );
    }
    Ok(FrameCommands {
        command_buffers,
        start_command_pools,
        start_command_buffers,
        pass_command_pools,
        pass_command_buffers,
        image_available,
        render_finished,
        in_flight,
    })
}
