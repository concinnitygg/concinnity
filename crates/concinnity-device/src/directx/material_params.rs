// The material parameter table on DirectX: core's table ringed over one
// persistently mapped upload buffer per frame in flight plus the probe bake's
// slot, bound as the bindless main pass's root SRV at t20. A slot's copy is
// rewritten only after a row changed.

use concinnity_core::gfx::render_types::{GpuMaterialParams, MATERIAL_PARAM_COUNT};
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::material_params::MaterialParamTable;
use windows::Win32::Graphics::Direct3D12::*;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::com;
use super::context::DxContext;
use super::error::map_hresult;

// The bindless main root signature's parameter the table binds at.
pub(in crate::directx) const MATERIAL_PARAMS_ROOT_PARAM: u32 = 20;

// One mapped copy of the table.
struct MappedCopy {
    buffer: PooledBuffer,
    ptr: *mut u8,
}

pub(in crate::directx) struct DxMaterialParams {
    table: MaterialParamTable,
    copies: Vec<MappedCopy>,
}

impl DxMaterialParams {
    // `slots` mapped copies of `rows` (see `material_params::rows`).
    pub(in crate::directx) fn new(
        alloc: &DeviceAllocator,
        rows: Vec<GpuMaterialParams>,
        slots: usize,
    ) -> RenderResult<Self> {
        let table = MaterialParamTable::new(rows, slots);
        let copies = (0..slots)
            .map(|_| {
                let buffer = alloc.alloc_buffer(
                    table.byte_len() as u64,
                    D3D12_HEAP_TYPE_UPLOAD,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                )?;
                let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
                // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a
                // live local that receives the mapping.
                unsafe { buffer.Map(0, None, Some(&mut ptr)) }
                    .map_err(|e| map_hresult(e.code(), "map material parameter table"))?;
                Ok(MappedCopy {
                    buffer,
                    ptr: ptr.cast(),
                })
            })
            .collect::<RenderResult<Vec<_>>>()?;
        Ok(Self { table, copies })
    }

    pub(in crate::directx) fn set(&mut self, row: u32, params: [f32; MATERIAL_PARAM_COUNT]) {
        self.table.set(row, params);
    }

    // Bring `slot`'s copy up to date, when a change has left it behind. The
    // caller has waited on the slot's fence, so the GPU no longer reads it.
    pub(in crate::directx) fn upload(&mut self, slot: usize) {
        let Some(copy) = self.copies.get(slot) else {
            return;
        };
        if let Some(rows) = self.table.take_upload(slot) {
            let bytes = std::mem::size_of_val(rows);
            // SAFETY: the copy was allocated at the table's byte length, which the
            // table never changes, and stays mapped for the buffer's lifetime.
            unsafe {
                std::ptr::copy_nonoverlapping(rows.as_ptr().cast::<u8>(), copy.ptr, bytes);
            }
        }
    }

    // The GPU address of `slot`'s copy, for the root SRV.
    pub(in crate::directx) fn gpu_va(&self, slot: usize) -> u64 {
        com::gpu_va(&self.copies[slot].buffer)
    }
}

impl DxContext {
    // Replace one row of the material parameter table; each slot's copy is
    // rewritten as that slot comes round.
    pub(super) fn set_material_params(&mut self, row: u32, params: [f32; MATERIAL_PARAM_COUNT]) {
        if let Some(table) = self.cull.material_params.as_mut() {
            table.set(row, params);
        }
    }

    // The table's GPU address for the pass recorded into ring slot `slot`, or
    // 0 in a world without the bindless pass, whose pipelines never bind it.
    pub(super) fn material_params_gva(&self, slot: usize) -> u64 {
        self.cull
            .material_params
            .as_ref()
            .map_or(0, |table| table.gpu_va(slot))
    }
}
