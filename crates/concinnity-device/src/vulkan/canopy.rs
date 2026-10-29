// The far field's view tables on Vulkan: one persistently mapped storage
// buffer per frame in flight (global set 0 binding 22), each rewritten only
// when the tables changed since that slot was last written.

use ash::vk;
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::environment::grass::canopy_table::MAX_CANOPY_TABLE_ROWS;

use super::allocator::{DeviceAllocator, PooledBuffer};

pub(in crate::vulkan) struct VkCanopyTables {
    // The current tables, which a probe capture snapshots, and how many times
    // they have changed; each slot records the count it holds.
    rows: Vec<[f32; 4]>,
    version: u64,
    copies: Vec<(PooledBuffer, u64)>,
}

impl VkCanopyTables {
    // `frames` slots with room for every layer's table.
    pub(in crate::vulkan) fn new(alloc: &DeviceAllocator, frames: usize) -> RenderResult<Self> {
        let size = (MAX_CANOPY_TABLE_ROWS * size_of::<[f32; 4]>()) as u64;
        let copies = (0..frames)
            .map(|_| {
                let buffer = alloc.create_buffer(
                    size,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )?;
                Ok((buffer, 0))
            })
            .collect::<RenderResult<Vec<_>>>()?;
        Ok(Self {
            rows: Vec::new(),
            version: 0,
            copies,
        })
    }

    // Take `rows` as the tables, and bring `frame`'s slot up to them. The
    // caller has waited on the slot's fence, so the GPU no longer reads it.
    pub(in crate::vulkan) fn update(&mut self, rows: &[[f32; 4]], frame: usize) {
        let rows = &rows[..rows.len().min(MAX_CANOPY_TABLE_ROWS)];
        if self.rows.as_slice() != rows {
            self.rows.clear();
            self.rows.extend_from_slice(rows);
            self.version += 1;
        }
        if let Some((buffer, held)) = self.copies.get_mut(frame)
            && *held != self.version
        {
            buffer.write_slice(0, &self.rows);
            *held = self.version;
        }
    }

    pub(in crate::vulkan) fn rows(&self) -> &[[f32; 4]] {
        &self.rows
    }

    pub(in crate::vulkan) fn buffer(&self, frame: usize) -> vk::Buffer {
        self.copies[frame].0.buffer()
    }

    pub(in crate::vulkan) fn clear(&mut self) {
        self.copies.clear();
    }
}
