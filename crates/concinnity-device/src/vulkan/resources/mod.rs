//! Runtime GPU resource management for VkContext, split per-category to mirror
//! the Metal reference shape (`metal/resources/`):
//!
//!   textures.rs   Texture-pool slot updates + descriptor rewires (`update_*`,
//!                 `evict_*`, `write_object_image`, `write_pool_image`)
//!   geometry.rs   Streamed-mesh upload + eviction (`upload_mesh`,
//!                 `evict_mesh`, the shared `write_geometry_region` helper)
//!   streaming.rs  VoxelWorld chunk streaming (`setup_chunk_streaming`,
//!                 `add_chunk_mesh`, `remove_chunk_mesh`, `set_chunk_model`)
//!   skinning.rs   Skinned-mesh upload + per-frame joint upload
//!                 (`upload_skinned`, `update_skinned_pose`,
//!                 `upload_joint_matrices`, `skinned_geometry`)
//!   geometry_rebuild.rs  Size-changing static + skinned VB/IB rebuilds
//!                 driven by asset hot-reload (`rebuild_static_geometry`,
//!                 `rebuild_skinned_geometry`)
//!
//! The shared low-level helpers (`create_descriptor_set_layout`,
//! `alloc_descriptor_sets`, `upload_geometry_buffer{,_raw}`) live in this file
//! because every submodule + `init.rs` needs them.

use ash::vk;
use concinnity_core::render::error::RenderResult;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::texture;
use crate::vulkan::owned::{OwnedSetLayout, VkDevice};

mod geometry;
mod geometry_rebuild;
mod skinning;
mod streaming;
mod textures;

pub(in crate::vulkan) fn create_descriptor_set_layout(
    device: &VkDevice,
    bindings: &[(u32, vk::DescriptorType, vk::ShaderStageFlags)],
) -> RenderResult<OwnedSetLayout> {
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
    device
        .create_descriptor_set_layout(&info)
        .map_err(|e| crate::vulkan::error::map_vk_result(e, "descriptor set layout"))
}

// The bindings of a set holding `n` fragment-sampled sources: their images at
// `0..n`, then their samplers in the same order at `n..2n`. The shape every
// fullscreen pass's shader declares its sources in.
pub(in crate::vulkan) fn source_set_bindings(
    n: u32,
) -> Vec<(u32, vk::DescriptorType, vk::ShaderStageFlags)> {
    (0..2 * n)
        .map(|b| {
            let ty = if b < n {
                vk::DescriptorType::SAMPLED_IMAGE
            } else {
                vk::DescriptorType::SAMPLER
            };
            (b, ty, vk::ShaderStageFlags::FRAGMENT)
        })
        .collect()
}

// Most sources one `source_set_bindings` set holds.
pub(in crate::vulkan) const MAX_SOURCES: usize = 8;

// Write `sources` into a set laid out by `source_set_bindings(sources.len())`:
// each view at its image binding and each sampler at its sampler binding, in
// one update.
pub(in crate::vulkan) fn write_source_set(
    device: &VkDevice,
    set: vk::DescriptorSet,
    sources: &[(vk::ImageView, vk::Sampler)],
) {
    let n = sources.len();
    assert!(
        (1..=MAX_SOURCES).contains(&n),
        "a source set holds 1..={MAX_SOURCES} sources, not {n}"
    );
    let mut images = [vk::DescriptorImageInfo::default(); MAX_SOURCES];
    let mut samplers = [vk::DescriptorImageInfo::default(); MAX_SOURCES];
    for (i, &(view, sampler)) in sources.iter().enumerate() {
        images[i] = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(view);
        samplers[i] = vk::DescriptorImageInfo::default().sampler(sampler);
    }
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(&images[..n]),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(n as u32)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .image_info(&samplers[..n]),
    ];
    // SAFETY: the writes and the infos they borrow are live for the call, each
    // runs across `n` consecutive count-1 bindings of the one type the layout
    // gives them, and the set and every view and sampler belong to this device.
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

// Write `samplers` into the SAMPLER bindings of `set` that start at `first`,
// one per binding. The bindings must be consecutive and each hold one sampler,
// so a single write runs from each into the next.
pub(in crate::vulkan) fn write_samplers(
    device: &VkDevice,
    set: vk::DescriptorSet,
    first: u32,
    samplers: &[vk::Sampler],
) {
    let n = samplers.len();
    assert!(
        (1..=MAX_SOURCES).contains(&n),
        "a sampler run holds 1..={MAX_SOURCES} samplers, not {n}"
    );
    let mut infos = [vk::DescriptorImageInfo::default(); MAX_SOURCES];
    for (info, &sampler) in infos.iter_mut().zip(samplers) {
        *info = vk::DescriptorImageInfo::default().sampler(sampler);
    }
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(first)
        .descriptor_type(vk::DescriptorType::SAMPLER)
        .image_info(&infos[..n]);
    // SAFETY: the write and the infos it borrows are live for the call, and the set and
    // samplers belong to this device.
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
}

// Bind `range` bytes of `buffer` at storage-buffer binding `binding` of `set`.
pub(in crate::vulkan) fn write_storage_buffer(
    device: &VkDevice,
    set: vk::DescriptorSet,
    binding: u32,
    buffer: vk::Buffer,
    range: u64,
) {
    let info = vk::DescriptorBufferInfo::default()
        .buffer(buffer)
        .offset(0)
        .range(range);
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(std::slice::from_ref(&info));
    // SAFETY: the write and the info it borrows are live for the call, and the set and buffer
    // belong to this device.
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
}

pub(in crate::vulkan) fn alloc_descriptor_sets(
    device: &VkDevice,
    pool: vk::DescriptorPool,
    layouts: &[vk::DescriptorSetLayout],
) -> RenderResult<Vec<vk::DescriptorSet>> {
    if layouts.is_empty() {
        return Ok(vec![]);
    }
    let alloc = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(layouts);
    // SAFETY: the create-info and every slice it borrows are live for the call, and each handle it
    // names belongs to this device.
    unsafe { device.allocate_descriptor_sets(&alloc) }
        .map_err(|e| crate::vulkan::error::map_vk_result(e, "allocate descriptor sets"))
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
    device: &VkDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    data: &[T],
    usage: vk::BufferUsageFlags,
) -> RenderResult<PooledBuffer> {
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
    device: &VkDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    data: &[u8],
    usage: vk::BufferUsageFlags,
) -> RenderResult<PooledBuffer> {
    // TRANSFER_SRC lets `setup_chunk_streaming` copy the build-time geometry
    // out of these buffers when it grows them for chunk-streaming headroom;
    // TRANSFER_DST lets the staging copy below and `write_geometry_region`
    // write into them.
    let usage = usage | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
    let size = data.len() as u64;
    if size == 0 {
        // Return a minimal 4-byte buffer to keep Vulkan happy.
        return alloc.create_buffer(4, usage, vk::MemoryPropertyFlags::DEVICE_LOCAL);
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

#[cfg(test)]
mod tests {
    use super::*;

    // `taa.hlsl`'s three sources are the shape: every image, then every sampler.
    #[test]
    fn a_source_set_lists_every_image_then_every_sampler() {
        use vk::DescriptorType as T;
        let stage = vk::ShaderStageFlags::FRAGMENT;
        assert_eq!(
            source_set_bindings(3),
            [
                (0, T::SAMPLED_IMAGE, stage),
                (1, T::SAMPLED_IMAGE, stage),
                (2, T::SAMPLED_IMAGE, stage),
                (3, T::SAMPLER, stage),
                (4, T::SAMPLER, stage),
                (5, T::SAMPLER, stage),
            ]
        );
        assert!(source_set_bindings(0).is_empty());
    }
}
