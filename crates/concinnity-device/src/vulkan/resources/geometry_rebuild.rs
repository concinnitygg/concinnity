//! The shared static and skinned vertex / index buffers a size-changing asset
//! hot-reload reads back and replaces whole; the repack itself is the
//! engine's.
//!
//! Vulkan's `DEVICE_LOCAL` buffers are not host-visible, so the read is a
//! one-shot `cmd_copy_buffer` into `HOST_VISIBLE` staging, and the replacement
//! is fresh `DEVICE_LOCAL` buffers at the post-rebuild size, filled through the
//! existing `write_geometry_region` helper. The rebuilt buffers are sized
//! exactly for the current draws, so the streamed-mesh placement allocators
//! are not preserved and a later `upload_mesh` fails allocation.

use ash::vk;
use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::mesh_payload::{SkinnedVertex, Vertex};
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::backend::{GeometryBuffers, GeometryRebuild, RenderBackend};
use concinnity_engine::render::rt_accel::geom;

use super::super::context::VkContext;
use super::super::texture::one_shot_submit;

// Read a DEVICE_LOCAL buffer's full contents back to CPU memory as a typed
// `Vec<T>`. Allocates a HOST_VISIBLE staging buffer, runs a one-shot
// `cmd_copy_buffer` from `src` into it (`wait_idle` already gated the source
// side; the one-shot's internal fence wait gates the destination), maps,
// and `copy_nonoverlapping`s into the Vec. `T`'s stride must match the
// buffer's stride exactly.
fn readback_typed<T: Copy>(ctx: &VkContext, src: vk::Buffer, bytes: u64) -> RenderResult<Vec<T>> {
    if bytes == 0 {
        return Ok(Vec::new());
    }
    let stride = size_of::<T>() as u64;
    if !bytes.is_multiple_of(stride) {
        return Err(RenderError::Other(format!(
            "readback_typed: buffer size {bytes} not a multiple of T stride {stride}"
        )));
    }
    let count = usize::saturating_from(bytes / stride);
    let staging = ctx.hw.alloc.create_buffer(
        bytes,
        vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    one_shot_submit(
        &ctx.hw.device,
        ctx.commands.command_pool,
        ctx.hw.graphics_queue,
        |rec| {
            let copy = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(bytes);
            rec.copy_buffer(src, staging.buffer(), std::slice::from_ref(&copy));
        },
    )?;

    let mut out: Vec<T> = Vec::with_capacity(count);
    let src = staging.mapped_ptr() as *const T;
    // SAFETY: the staging buffer was created HOST_VISIBLE | HOST_COHERENT and sized to `bytes`,
    // so `src` is a live mapping of `count` values; `out` is a separate allocation that holds them.
    unsafe { std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), count) };
    // SAFETY: the copy above initialized all `count` elements within `out`'s capacity.
    unsafe { out.set_len(count) };
    Ok(out)
}

impl GeometryBuffers for VkContext {
    // DEVICE_LOCAL is not host-mappable, so the live buffers come back through
    // HOST_VISIBLE staging.
    fn read_static_geometry(&mut self) -> RenderResult<(Vec<Vertex>, Vec<u32>)> {
        self.wait_idle();
        let vertices = readback_typed(
            self,
            self.geometry.vertex_buffer.buffer(),
            self.geometry.vertex_buffer_bytes,
        )?;
        let indices = readback_typed(
            self,
            self.geometry.index_buffer.buffer(),
            self.geometry.index_buffer_bytes,
        )?;
        Ok((vertices, indices))
    }

    // Fresh DEVICE_LOCAL buffers, filled through staging; the replaced pair
    // retires through the allocator once swapped.
    fn replace_static_geometry(
        &mut self,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> RenderResult<()> {
        let v_bytes = size_of_val(vertices) as u64;
        let i_bytes = size_of_val(indices) as u64;
        let shared = super::shared_geometry_usage(self.hw.rt_capable);
        let vbuf = self.hw.alloc.create_buffer(
            v_bytes,
            vk::BufferUsageFlags::VERTEX_BUFFER | shared,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let ibuf = self.hw.alloc.create_buffer(
            i_bytes,
            vk::BufferUsageFlags::INDEX_BUFFER | shared,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        self.write_geometry_region(vbuf.buffer(), 0, bytemuck::cast_slice(vertices))?;
        self.write_geometry_region(ibuf.buffer(), 0, bytemuck::cast_slice(indices))?;
        self.geometry.vertex_buffer = vbuf;
        self.geometry.vertex_buffer_bytes = v_bytes;
        self.geometry.index_buffer = ibuf;
        self.geometry.index_buffer_bytes = i_bytes;
        Ok(())
    }

    // The RT acceleration structure was built against the buffers just
    // replaced and the per-draw offsets just rewritten. Rebuild it over the
    // fresh layout and re-point the passes that read the buffers directly.
    fn static_geometry_moved(&mut self) {
        if self.rt_reflections.is_some() {
            self.rebuild_rt_accel();
        }
    }

    fn read_skinned_geometry(&mut self) -> RenderResult<(Vec<SkinnedVertex>, Vec<u32>)> {
        if self.skinned.vertex_buffer.is_null() || self.skinned.index_buffer.is_null() {
            return Err(RenderError::Other(
                "rebuild_skinned_geometry: no skinned vertex/index buffer (was \
                 upload_skinned called?)"
                    .into(),
            ));
        }
        self.wait_idle();
        let vertices = readback_typed(
            self,
            self.skinned.vertex_buffer.buffer(),
            self.skinned.vertex_buffer_bytes,
        )?;
        let indices = readback_typed(
            self,
            self.skinned.index_buffer.buffer(),
            self.skinned.index_buffer_bytes,
        )?;
        Ok((vertices, indices))
    }

    // The new buffers carry the usage `upload_skinned` gives them, or the
    // skinning paths lose their inputs after a size-changing reload; the RT
    // index-buffer flags ride along whenever the device is capable, so a live
    // RT toggle keeps working across reloads.
    fn replace_skinned_geometry(
        &mut self,
        vertices: &[SkinnedVertex],
        indices: &[u32],
    ) -> RenderResult<()> {
        let v_bytes = size_of_val(vertices) as u64;
        let i_bytes = size_of_val(indices) as u64;
        let ib_rt = if self.hw.rt_capable {
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
        } else {
            vk::BufferUsageFlags::empty()
        };
        let vbuf = self.hw.alloc.create_buffer(
            v_bytes,
            vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        // Whole u32 words for the index buffer; see `upload_skinned`.
        let ibuf = self.hw.alloc.create_buffer(
            geom::skinned_index_buffer_bytes(indices.len()) as u64,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | ib_rt,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        self.write_geometry_region(vbuf.buffer(), 0, bytemuck::cast_slice(vertices))?;
        self.write_geometry_region(ibuf.buffer(), 0, bytemuck::cast_slice(indices))?;
        self.skinned.vertex_buffer = vbuf;
        self.skinned.vertex_buffer_bytes = v_bytes;
        self.skinned.index_buffer = ibuf;
        self.skinned.index_buffer_bytes = i_bytes;
        Ok(())
    }

    // The skin fold's descriptor sets still bind the replaced bind-pose VB and
    // a deformed ring sized for the old layout; re-point and re-size both. The
    // read's drain gated every in-flight read.
    fn skinned_geometry_moved(&mut self, vertex_count: usize) -> RenderResult<()> {
        self.refresh_main_skin_geometry(vertex_count)
    }
}

impl GeometryRebuild for VkContext {}
