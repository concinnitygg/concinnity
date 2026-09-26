// The material parameter table on Vulkan: core's table ringed over one
// persistently mapped storage buffer per frame in flight, each bound at binding
// `MATERIAL_PARAMS_BINDING` of that frame's bindless set, beside its object
// records. A frame's copy is rewritten only after a row changed.

use ash::vk;
use concinnity_core::gfx::render_types::{GpuMaterialParams, MATERIAL_PARAM_COUNT};
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::material_params::MaterialParamTable;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::context::VkContext;

// The bindless set's binding the table sits at (`main_bindless.hlsl`).
pub(in crate::vulkan) const MATERIAL_PARAMS_BINDING: u32 = 2;

pub(in crate::vulkan) struct VkMaterialParams {
    table: MaterialParamTable,
    buffers: Vec<PooledBuffer>,
}

impl VkMaterialParams {
    // One copy of `rows` (see `material_params::rows`) per frame in flight.
    pub(in crate::vulkan) fn new(
        alloc: &DeviceAllocator,
        rows: Vec<GpuMaterialParams>,
        frames: usize,
    ) -> RenderResult<Self> {
        let table = MaterialParamTable::new(rows, frames);
        let buffers = (0..frames)
            .map(|_| host_buffer(alloc, table.byte_len()))
            .collect::<RenderResult<Vec<_>>>()?;
        Ok(Self { table, buffers })
    }

    pub(in crate::vulkan) fn set(&mut self, row: u32, params: [f32; MATERIAL_PARAM_COUNT]) {
        self.table.set(row, params);
    }

    // Bring `frame`'s copy up to date, when a change has left it behind. The
    // caller has waited on the frame's fence, so the GPU no longer reads it.
    pub(in crate::vulkan) fn upload(&mut self, frame: usize) {
        let Some(buf) = self.buffers.get(frame) else {
            return;
        };
        if let Some(rows) = self.table.take_upload(frame) {
            buf.write_slice(0, rows);
        }
    }

    // `frame`'s copy, for its bindless set.
    pub(in crate::vulkan) fn descriptor(&self, frame: usize) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::default()
            .buffer(self.buffers[frame].buffer())
            .offset(0)
            .range(self.table.byte_len() as u64)
    }

    // A copy of the current rows in a buffer of its own, for a capture that
    // outlives the frame it started in.
    pub(in crate::vulkan) fn snapshot(
        &self,
        alloc: &DeviceAllocator,
    ) -> RenderResult<PooledBuffer> {
        let buf = host_buffer(alloc, self.table.byte_len())?;
        buf.write_slice(0, self.table.rows());
        Ok(buf)
    }
}

fn host_buffer(alloc: &DeviceAllocator, size: usize) -> RenderResult<PooledBuffer> {
    alloc.create_buffer(
        size as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
}

impl VkContext {
    // Replace one row of the material parameter table; each frame's copy is
    // rewritten as that frame comes round.
    pub(super) fn set_material_params(&mut self, row: u32, params: [f32; MATERIAL_PARAM_COUNT]) {
        if let Some(table) = self.cull.material_params.as_mut() {
            table.set(row, params);
        }
    }
}

// The write that points binding `MATERIAL_PARAMS_BINDING` of `set` at `info`.
pub(in crate::vulkan) fn write(
    set: vk::DescriptorSet,
    info: &vk::DescriptorBufferInfo,
) -> vk::WriteDescriptorSet<'_> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(MATERIAL_PARAMS_BINDING)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(std::slice::from_ref(info))
}
