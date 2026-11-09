// The material parameter table on Vulkan: the scene's table ringed over one
// persistently mapped storage buffer per frame in flight, each bound at binding
// `MATERIAL_PARAMS_BINDING` of that frame's bindless set, beside its object
// records. A frame's copy is rewritten only after a row changed.

use ash::vk;
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::material_params::MaterialParamTable;

use super::allocator::{DeviceAllocator, PooledBuffer};

// The bindless set's binding the table sits at (`main_bindless.hlsl`).
pub(in crate::vulkan) const MATERIAL_PARAMS_BINDING: u32 = 2;

/// The bindless set binding the shared vertex stream sits at, as floats,
/// where a planet tile's morph reads its neighbors.
pub(in crate::vulkan) const MESH_VERTICES_BINDING: u32 = 3;

pub(in crate::vulkan) struct MaterialParams {
    buffers: Vec<PooledBuffer>,
}

impl MaterialParams {
    // One copy of `table` per copy it tracks, one per frame in flight.
    pub(in crate::vulkan) fn new(
        alloc: &DeviceAllocator,
        table: &MaterialParamTable,
    ) -> RenderResult<Self> {
        let buffers = (0..table.copies())
            .map(|_| host_buffer(alloc, table.byte_len()))
            .collect::<RenderResult<Vec<_>>>()?;
        Ok(Self { buffers })
    }

    // Bring `frame`'s copy of `table` up to date, when a change has left it
    // behind. The caller has waited on the frame's fence, so the GPU no longer
    // reads it.
    pub(in crate::vulkan) fn upload(&self, table: &mut MaterialParamTable, frame: usize) {
        let Some(buf) = self.buffers.get(frame) else {
            return;
        };
        if let Some(rows) = table.take_upload(frame) {
            buf.write_slice(0, rows);
        }
    }

    // `frame`'s copy of `table`, for its bindless set.
    pub(in crate::vulkan) fn descriptor(
        &self,
        table: &MaterialParamTable,
        frame: usize,
    ) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::default()
            .buffer(self.buffers[frame].buffer())
            .offset(0)
            .range(table.byte_len() as u64)
    }
}

// A copy of `table`'s current rows in a buffer of its own, for a capture that
// outlives the frame it started in.
pub(in crate::vulkan) fn snapshot(
    alloc: &DeviceAllocator,
    table: &MaterialParamTable,
) -> RenderResult<PooledBuffer> {
    let buf = host_buffer(alloc, table.byte_len())?;
    buf.write_slice(0, table.rows());
    Ok(buf)
}

fn host_buffer(alloc: &DeviceAllocator, size: usize) -> RenderResult<PooledBuffer> {
    alloc.create_buffer(
        size as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
}
