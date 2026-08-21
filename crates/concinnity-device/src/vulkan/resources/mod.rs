// src/vulkan/resources/mod.rs
//
// Runtime GPU resource management for VkContext, split per-category to mirror
// the Metal reference shape (`metal/resources/`):
//
//   textures.rs   Texture-pool slot updates + descriptor rewires (`update_*`,
//                 `evict_*`, `write_object_image`, `write_pool_image`)
//   geometry.rs   Streamed-mesh upload + eviction (`upload_mesh`,
//                 `evict_mesh`, the shared `write_geometry_region` helper)
//   streaming.rs  VoxelWorld chunk streaming (`setup_chunk_streaming`,
//                 `add_chunk_mesh`, `remove_chunk_mesh`, `set_chunk_model`)
//   skinning.rs   Skinned-mesh upload + per-frame joint upload
//                 (`upload_skinned`, `update_skinned_pose`,
//                 `upload_joint_matrices`, `skinned_geometry`)
//   geometry_rebuild.rs  Size-changing static + skinned VB/IB rebuilds
//                 driven by asset hot-reload (`rebuild_static_geometry`,
//                 `rebuild_skinned_geometry`)
//
// The shared low-level helpers (`create_descriptor_set_layout`,
// `alloc_descriptor_sets`, `upload_geometry_buffer{,_raw}`) live in this file
// because every submodule + `init.rs` needs them.

use ash::{Device, vk};

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::texture;

mod geometry;
mod geometry_rebuild;
mod skinning;
mod streaming;
mod textures;

pub(in crate::vulkan) fn create_descriptor_set_layout(
    device: &Device,
    bindings: &[(u32, vk::DescriptorType, vk::ShaderStageFlags)],
) -> Result<vk::DescriptorSetLayout, String> {
    let vk_bindings: Vec<_> = bindings
        .iter()
        .map(|&(b, ty, stage)| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(ty)
                .descriptor_count(1)
                .stage_flags(stage)
        })
        .collect();
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&vk_bindings);
    // SAFETY: the create-info and every slice it borrows are live for the call, and each handle it
    // names belongs to this device.
    unsafe { device.create_descriptor_set_layout(&info, None) }
        .map_err(|e| format!("descriptor set layout: {e}"))
}

pub(in crate::vulkan) fn alloc_descriptor_sets(
    device: &Device,
    pool: vk::DescriptorPool,
    layouts: &[vk::DescriptorSetLayout],
) -> Result<Vec<vk::DescriptorSet>, String> {
    if layouts.is_empty() {
        return Ok(vec![]);
    }
    let alloc = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(layouts);
    // SAFETY: the create-info and every slice it borrows are live for the call, and each handle it
    // names belongs to this device.
    unsafe { device.allocate_descriptor_sets(&alloc) }
        .map_err(|e| format!("allocate descriptor sets: {e}"))
}

// Usage every (re)creation of the shared vertex / index buffers must carry, on
// top of its VERTEX_BUFFER / INDEX_BUFFER role. On an RT-capable device the
// shared buffers double as acceleration-structure build inputs (device
// addressed) and as storage buffers the RT / glass shaders fetch hit-triangle
// attributes from; the flags ride on capability rather than on RT being live,
// since a later quality toggle cannot add usage to an existing buffer. Every
// path that replaces the buffers (chunk-streaming headroom, hot-reload
// geometry rebuild) routes through here: dropping these leaves the shared
// buffers un-addressable and every later acceleration-structure build invalid.
pub(in crate::vulkan) fn shared_geometry_usage(rt_capable: bool) -> vk::BufferUsageFlags {
    let base = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
    if rt_capable {
        base | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::STORAGE_BUFFER
    } else {
        base
    }
}

pub(in crate::vulkan) fn upload_geometry_buffer<T: bytemuck::NoUninit>(
    alloc: &DeviceAllocator,
    device: &Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    data: &[T],
    usage: vk::BufferUsageFlags,
) -> Result<PooledBuffer, String> {
    upload_geometry_buffer_raw(
        alloc,
        device,
        command_pool,
        queue,
        bytemuck::cast_slice(data),
        usage,
    )
}

pub(in crate::vulkan) fn upload_geometry_buffer_raw(
    alloc: &DeviceAllocator,
    device: &Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    data: &[u8],
    usage: vk::BufferUsageFlags,
) -> Result<PooledBuffer, String> {
    // TRANSFER_SRC lets `setup_chunk_streaming` copy the build-time geometry
    // out of these buffers when it grows them for chunk-streaming headroom;
    // TRANSFER_DST lets the staging copy below and `write_geometry_region`
    // write into them.
    let usage = usage | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
    let size = data.len() as u64;
    if size == 0 {
        // Return a minimal 4-byte buffer to keep Vulkan happy.
        return Ok(alloc.create_buffer(4, usage, vk::MemoryPropertyFlags::DEVICE_LOCAL)?);
    }
    let staging = alloc.create_buffer(
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    staging.write_bytes(0, data);
    let buf = alloc.create_buffer(size, usage, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
    texture::one_shot_submit(device, command_pool, queue, |cmd| {
        let copy = vk::BufferCopy::default().size(size);
        // SAFETY: `cmd` is a command buffer in the recording state, and every handle and slice
        // these commands name is live for the call.
        unsafe {
            device.cmd_copy_buffer(
                cmd,
                staging.buffer(),
                buf.buffer(),
                std::slice::from_ref(&copy),
            )
        };
    })?;
    Ok(buf)
}
