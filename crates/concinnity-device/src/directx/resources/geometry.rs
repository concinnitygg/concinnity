//! The scene DxContext lends to the `SceneHost` defaults, and the writer that
//! stages its streamed geometry for the shared vertex / index buffers: each
//! write is copied by the next geometry submit (see `geometry_upload`), so none
//! waits on the GPU. Also the blocking `write_geometry_region` helper for the
//! hot-reload paths that rewrite a region in place.

use concinnity_core::render::backend::{GeometryEdit, SceneHost};
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::scene_state::{GeometryBuffer, GeometryWriter, SceneState};
use windows::Win32::Graphics::Direct3D12::*;

use super::super::allocator::DeviceAllocator;
use super::super::context::*;
use super::super::geometry_upload::{GeometryTarget, GeometryUploads};
use super::super::texture::{one_shot_submit, transition_barrier};
use crate::directx::error::map_hresult;

// Stages each write for the next geometry copy submit, which lands it after
// every earlier GPU read of the buffer and before any later one.
struct StagingWriter<'a> {
    alloc: &'a DeviceAllocator,
    uploads: &'a mut GeometryUploads,
}

impl GeometryWriter for StagingWriter<'_> {
    fn write(&mut self, buffer: GeometryBuffer, offset: usize, bytes: &[u8]) -> RenderResult<()> {
        let target = match buffer {
            GeometryBuffer::Vertex => GeometryTarget::Vertex,
            GeometryBuffer::Index => GeometryTarget::Index,
        };
        self.uploads.stage(self.alloc, target, offset as u64, bytes)
    }

    fn reserve(&mut self, bytes: u64) {
        if let Err(e) = self.uploads.reserve(self.alloc, bytes) {
            tracing::warn!("mesh streaming: geometry staging reserve failed: {e}");
        }
    }
}

impl SceneHost for DxContext {
    fn scene(&self) -> Option<&SceneState> {
        Some(&self.state)
    }

    fn scene_mut(&mut self) -> Option<&mut SceneState> {
        debug_assert_main_thread("scene_mut");
        Some(&mut self.state)
    }

    fn edit_geometry(&mut self, edit: GeometryEdit<'_>) -> Option<RenderResult<()>> {
        debug_assert_main_thread("edit_geometry");
        let mut writer = StagingWriter {
            alloc: &self.hw.alloc,
            uploads: self.geometry_uploads.get_mut(),
        };
        Some(edit(&mut self.state, &mut writer))
    }
}

impl DxContext {
    // Copy `data` into a sub-region of a DEFAULT-heap geometry buffer.
    //
    // `dest` is a buffer currently in `usage_state` (the vertex or index
    // buffer). The copy goes through a temporary UPLOAD-heap staging buffer
    // and a one-shot command list that transitions the resource
    // `usage_state -> COPY_DEST -> usage_state` around a `CopyBufferRegion`.
    // The caller must `wait_idle` first: the COPY_DEST transition covers the
    // whole resource, so no in-flight command list may still reference it.
    pub(in crate::directx) fn write_geometry_region(
        &self,
        dest: &ID3D12Resource,
        usage_state: D3D12_RESOURCE_STATES,
        offset: u64,
        data: &[u8],
    ) -> RenderResult<()> {
        if data.is_empty() {
            return Ok(());
        }
        let upload = self.hw.alloc.alloc_buffer(
            data.len() as u64,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
        // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local
        // that receives the mapping.
        unsafe { upload.Map(0, None, Some(&mut ptr)) }
            .map_err(|e| map_hresult(e.code(), "mesh region map"))?;
        // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload, and the
        // source is a separate allocation, so the ranges cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
            upload.Unmap(0, None);
        }
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        one_shot_submit(&self.hw.device, &self.hw.command_queue, |cmd| unsafe {
            let to_dst = transition_barrier(dest, usage_state, D3D12_RESOURCE_STATE_COPY_DEST);
            cmd.ResourceBarrier(&[to_dst]);
            cmd.CopyBufferRegion(dest, offset, &*upload, 0, data.len() as u64);
            let back = transition_barrier(dest, D3D12_RESOURCE_STATE_COPY_DEST, usage_state);
            cmd.ResourceBarrier(&[back]);
        })
    }
}
