// The far field's canopy block on DirectX: one persistently mapped upload
// buffer per frame in flight, rewritten each frame and bound as the bindless
// main pass's root CBV at b8, and its layers' view tables beside it, rewritten
// only when they change and bound as the root SRV at t24.

use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::environment::grass::canopy_table::MAX_CANOPY_TABLE_ROWS;
use concinnity_engine::render::uniforms::canopy::CanopyParams;
use windows::Win32::Graphics::Direct3D12::*;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::com;
use super::context::DxContext;
use super::error::map_hresult;

// The bindless main root signature's parameters the block and the tables bind
// at.
pub(in crate::directx) const CANOPY_ROOT_PARAM: u32 = 23;
pub(in crate::directx) const CANOPY_TABLES_ROOT_PARAM: u32 = 24;

// One mapped copy of the block or the tables.
struct MappedCopy {
    buffer: PooledBuffer,
    ptr: *mut u8,
}

impl MappedCopy {
    fn new(alloc: &DeviceAllocator, size: u64) -> RenderResult<Self> {
        let buffer = alloc.alloc_buffer(
            size,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
        // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a
        // live local that receives the mapping.
        unsafe { buffer.Map(0, None, Some(&raw mut ptr)) }
            .map_err(|e| map_hresult(e.code(), "map canopy block"))?;
        Ok(Self {
            buffer,
            ptr: ptr.cast(),
        })
    }
}

pub(in crate::directx) struct DxCanopy {
    // This frame's block, which the probe bake snapshots.
    pub(in crate::directx) params: std::cell::Cell<CanopyParams>,
    copies: Vec<MappedCopy>,
    // The current tables, which the probe bake snapshots, and how many times
    // they have changed; each slot's copy records the count it holds.
    pub(in crate::directx) tables: std::cell::RefCell<Vec<[f32; 4]>>,
    tables_version: std::cell::Cell<u64>,
    table_copies: Vec<(MappedCopy, std::cell::Cell<u64>)>,
}

impl DxCanopy {
    // `slots` mapped copies of an empty block and of room for every layer's
    // table.
    pub(in crate::directx) fn new(alloc: &DeviceAllocator, slots: usize) -> RenderResult<Self> {
        let size = (size_of::<CanopyParams>() as u64 + 255) & !255;
        let table_size = (MAX_CANOPY_TABLE_ROWS * size_of::<[f32; 4]>()) as u64;
        let copies = (0..slots)
            .map(|_| MappedCopy::new(alloc, size))
            .collect::<RenderResult<Vec<_>>>()?;
        let table_copies = (0..slots)
            .map(|_| Ok((MappedCopy::new(alloc, table_size)?, std::cell::Cell::new(0))))
            .collect::<RenderResult<Vec<_>>>()?;
        let canopy = Self {
            params: std::cell::Cell::new(CanopyParams::default()),
            copies,
            tables: std::cell::RefCell::new(Vec::new()),
            tables_version: std::cell::Cell::new(0),
            table_copies,
        };
        for slot in 0..slots {
            canopy.write(slot);
        }
        Ok(canopy)
    }

    // Take `tables` as the current tables when they differ.
    pub(in crate::directx) fn set_tables(&self, tables: &[[f32; 4]]) {
        let tables = &tables[..tables.len().min(MAX_CANOPY_TABLE_ROWS)];
        let mut current = self.tables.borrow_mut();
        if current.as_slice() != tables {
            current.clear();
            current.extend_from_slice(tables);
            self.tables_version.set(self.tables_version.get() + 1);
        }
    }

    // Copy the current block into `slot`. The caller has waited on the slot's
    // fence, so the GPU no longer reads it.
    pub(in crate::directx) fn write(&self, slot: usize) {
        let Some(copy) = self.copies.get(slot) else {
            return;
        };
        let params = self.params.get();
        let bytes = bytemuck::bytes_of(&params);
        // SAFETY: the copy was allocated at least as large as the block and
        // stays mapped for the buffer's lifetime; the source is a separate value.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), copy.ptr, bytes.len()) };
        let Some((tables_copy, held)) = self.table_copies.get(slot) else {
            return;
        };
        if held.get() != self.tables_version.get() {
            let tables = self.tables.borrow();
            let bytes: &[u8] = bytemuck::cast_slice(tables.as_slice());
            // SAFETY: the copy was allocated for MAX_CANOPY_TABLE_ROWS rows,
            // which set_tables caps the tables at, and stays mapped; the
            // source is a separate allocation.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), tables_copy.ptr, bytes.len()) };
            held.set(self.tables_version.get());
        }
    }

    // The GPU address of `slot`'s copy, for the root CBV.
    pub(in crate::directx) fn gpu_va(&self, slot: usize) -> u64 {
        com::gpu_va(&self.copies[slot].buffer)
    }

    // The GPU address of `slot`'s tables, for the root SRV.
    pub(in crate::directx) fn tables_gpu_va(&self, slot: usize) -> u64 {
        com::gpu_va(&self.table_copies[slot].0.buffer)
    }
}

impl DxContext {
    // Refresh this frame's canopy block, `time` seconds into the world, from
    // the world's grass and the planet tiles drawn this frame, into ring slot
    // `slot`.
    pub(super) fn upload_canopy(&self, slot: usize, time: f64) {
        self.canopy
            .params
            .set(match (&self.grass, &self.grass_lod.get()) {
                (Some(grass), Some(lod)) => self.state.canopy_params(
                    &grass.field.canopy,
                    (lod, grass.crossfade_phase()),
                    time,
                ),
                _ => CanopyParams::default(),
            });
        self.canopy
            .set_tables(self.grass.as_ref().map_or(&[], |g| g.field.canopy.tables()));
        self.canopy.write(slot);
    }
}
