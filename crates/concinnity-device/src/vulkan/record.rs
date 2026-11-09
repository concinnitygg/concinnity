//! Safe command recording. [`Recorder`] is a command buffer that is known to be
//! in the recording state, and the surface over it that every pass records
//! through.
//!
//! The `unsafe` on `vkCmd*` carries two obligations: the command buffer is in
//! the recording state, and every handle and slice the command names is live for
//! the call. Both were restated verbatim at every recording site in this backend,
//! which is how they stop being checked. `Recorder` discharges the first by
//! construction -- the only safe way to get one is [`Recorder::begin`], which
//! puts the buffer into that state itself -- and the second by taking the owning
//! wrappers from `owned.rs` by reference, so a pipeline or layout cannot be bound
//! unless its owner is alive at the call.
//!
//! Every pass, upload and frame scope records through it. What is left raw is
//! stated rather than hidden:
//!
//! - Descriptor sets, buffers and image views are `vk::*` handles here, because
//!   they are owned by a descriptor pool, the device allocator or a swapchain
//!   rather than by a wrapper of their own. They keep the liveness argument they
//!   always had.
//! - [`Recorder::raw`] hands the buffer to the commands this surface does not
//!   cover: the acceleration-structure builds, recorded through the
//!   `VK_KHR_acceleration_structure` loader, and the DLSS, FSR and XeSS
//!   dispatches, whose SDKs record into a raw command buffer themselves.
//! - The cull and compaction kernel tests drive the kernels with raw handles of
//!   their own, outside any pass.

use ash::vk;
use concinnity_engine::render::shader::programs::abi::group::ThreadGroup;

use super::owned::{OwnedFramebuffer, OwnedPipeline, OwnedPipelineLayout, OwnedRenderPass};

// A buffer and a byte offset into it, where a command reads.
#[derive(Copy, Clone)]
pub(in crate::vulkan) struct BufferAt {
    pub(in crate::vulkan) buffer: vk::Buffer,
    pub(in crate::vulkan) offset: vk::DeviceSize,
}

// A command buffer in the recording state.
//
// Borrows the raw device rather than sharing the owning handle: recording never
// creates or retires an owned object, so a recorder holds no claim on the
// device's lifetime and carries only the dispatch table onto the render-graph's
// worker threads.
pub(in crate::vulkan) struct Recorder<'a> {
    device: &'a ash::Device,
    cmd: vk::CommandBuffer,
}

impl<'a> Recorder<'a> {
    // Begin recording into `cmd`, which must not be in flight. The caller
    // establishes that with the frame fence it already waits on; putting the
    // buffer into the recording state is what this call does, so no caller has
    // to claim it.
    pub(in crate::vulkan) fn begin(
        device: &'a ash::Device,
        cmd: vk::CommandBuffer,
        flags: vk::CommandBufferUsageFlags,
    ) -> Result<Self, vk::Result> {
        let info = vk::CommandBufferBeginInfo::default().flags(flags);
        // SAFETY: the create-info is live for the call, and `cmd` belongs to this device. The
        // caller's fence wait is what guarantees it is not in flight.
        unsafe { device.begin_command_buffer(cmd, &info) }?;
        Ok(Self { device, cmd })
    }

    // Adopt a command buffer someone else began.
    //
    // # Safety
    // `cmd` must be in the recording state and must belong to `device`. Only
    // for the frame's outer "end" buffer, which `begin` begins and the frame
    // carries as a raw handle (a recorder would borrow the context it is stored
    // beside) until the graph executor and `draw_frame` record into it.
    pub(in crate::vulkan) unsafe fn assume_recording(
        device: &'a ash::Device,
        cmd: vk::CommandBuffer,
    ) -> Self {
        Self { device, cmd }
    }

    // Finish recording. Consumes the recorder, so nothing can record into the
    // buffer afterwards.
    pub(in crate::vulkan) fn end(self) -> Result<vk::CommandBuffer, vk::Result> {
        // SAFETY: `self` exists only for a buffer in the recording state, which is what
        // `end_command_buffer` requires.
        unsafe { self.device.end_command_buffer(self.cmd) }?;
        Ok(self.cmd)
    }

    // The command buffer, for the submit call and for the extension loaders
    // this surface does not wrap.
    pub(in crate::vulkan) fn raw(&self) -> vk::CommandBuffer {
        self.cmd
    }

    pub(in crate::vulkan) fn bind_pipeline(
        &self,
        bind_point: vk::PipelineBindPoint,
        pipeline: &OwnedPipeline,
    ) {
        // SAFETY: `self.cmd` is in the recording state by construction, and the pipeline is alive
        // for the call because this borrows its owner.
        unsafe {
            self.device
                .cmd_bind_pipeline(self.cmd, bind_point, pipeline.handle());
        };
    }

    pub(in crate::vulkan) fn bind_descriptor_sets(
        &self,
        bind_point: vk::PipelineBindPoint,
        layout: &OwnedPipelineLayout,
        first_set: u32,
        sets: &[vk::DescriptorSet],
        dynamic_offsets: &[u32],
    ) {
        // SAFETY: `self.cmd` is in the recording state by construction, the layout is alive because
        // this borrows its owner, and the slices are live for the call. The sets outlive the
        // submission with the pool they were allocated from.
        unsafe {
            self.device.cmd_bind_descriptor_sets(
                self.cmd,
                bind_point,
                layout.handle(),
                first_set,
                sets,
                dynamic_offsets,
            );
        };
    }

    // Push one `#[repr(C)]` block. Typed, so the byte length comes from the
    // value rather than from a hand-written `size_of` beside a pointer cast.
    pub(in crate::vulkan) fn push_constants<T: bytemuck::NoUninit>(
        &self,
        layout: &OwnedPipelineLayout,
        stages: vk::ShaderStageFlags,
        offset: u32,
        value: &T,
    ) {
        self.push_constant_bytes(layout, stages, offset, bytemuck::bytes_of(value));
    }

    // Push a byte range, for the blocks assembled at runtime rather than from
    // one struct.
    pub(in crate::vulkan) fn push_constant_bytes(
        &self,
        layout: &OwnedPipelineLayout,
        stages: vk::ShaderStageFlags,
        offset: u32,
        bytes: &[u8],
    ) {
        // SAFETY: `self.cmd` is in the recording state by construction, the layout is alive because
        // this borrows its owner, and `bytes` is live for the call.
        unsafe {
            self.device
                .cmd_push_constants(self.cmd, layout.handle(), stages, offset, bytes);
        };
    }

    // Begin a render pass over `framebuffer`. Borrowing both owners is what
    // keeps them alive across the pass this opens.
    pub(in crate::vulkan) fn begin_render_pass(
        &self,
        render_pass: &OwnedRenderPass,
        framebuffer: &OwnedFramebuffer,
        area: vk::Rect2D,
        clear_values: &[vk::ClearValue],
    ) {
        let info = vk::RenderPassBeginInfo::default()
            .render_pass(render_pass.handle())
            .framebuffer(framebuffer.handle())
            .render_area(area)
            .clear_values(clear_values);
        // SAFETY: `self.cmd` is in the recording state by construction, the pass and framebuffer
        // are alive because this borrows their owners, and the info and its slice are live for the
        // call.
        unsafe {
            self.device
                .cmd_begin_render_pass(self.cmd, &info, vk::SubpassContents::INLINE);
        };
    }

    pub(in crate::vulkan) fn end_render_pass(&self) {
        // SAFETY: `self.cmd` is in the recording state by construction.
        unsafe { self.device.cmd_end_render_pass(self.cmd) };
    }

    pub(in crate::vulkan) fn set_viewport(&self, viewport: &vk::Viewport) {
        // SAFETY: `self.cmd` is in the recording state by construction, and the slice is live for
        // the call.
        unsafe {
            self.device
                .cmd_set_viewport(self.cmd, 0, std::slice::from_ref(viewport));
        };
    }

    pub(in crate::vulkan) fn set_scissor(&self, scissor: &vk::Rect2D) {
        // SAFETY: `self.cmd` is in the recording state by construction, and the slice is live for
        // the call.
        unsafe {
            self.device
                .cmd_set_scissor(self.cmd, 0, std::slice::from_ref(scissor));
        };
    }

    pub(in crate::vulkan) fn draw(
        &self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        // SAFETY: `self.cmd` is in the recording state by construction.
        unsafe {
            self.device.cmd_draw(
                self.cmd,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            );
        };
    }

    // Bind the index buffer `buffer` from `offset`.
    pub(in crate::vulkan) fn bind_index_buffer(
        &self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        index_type: vk::IndexType,
    ) {
        // SAFETY: `self.cmd` is in the recording state by construction, and the buffer is live for
        // the call.
        unsafe {
            self.device
                .cmd_bind_index_buffer(self.cmd, buffer, offset, index_type);
        };
    }

    pub(in crate::vulkan) fn draw_indexed(
        &self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        // SAFETY: `self.cmd` is in the recording state by construction.
        unsafe {
            self.device.cmd_draw_indexed(
                self.cmd,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            );
        };
    }

    // `count` non-indexed draws whose arguments sit `stride` apart from `offset`
    // in `args`.
    pub(in crate::vulkan) fn draw_indirect(
        &self,
        args: vk::Buffer,
        offset: vk::DeviceSize,
        count: u32,
        stride: u32,
    ) {
        // SAFETY: `self.cmd` is in the recording state by construction, and the arguments buffer
        // is live for the call.
        unsafe {
            self.device
                .cmd_draw_indirect(self.cmd, args, offset, count, stride);
        };
    }

    // `count` indexed draws whose arguments sit `stride` apart from `offset` in
    // `args`.
    pub(in crate::vulkan) fn draw_indexed_indirect(
        &self,
        args: vk::Buffer,
        offset: vk::DeviceSize,
        count: u32,
        stride: u32,
    ) {
        // SAFETY: `self.cmd` is in the recording state by construction, and the arguments buffer
        // is live for the call.
        unsafe {
            self.device
                .cmd_draw_indexed_indirect(self.cmd, args, offset, count, stride);
        };
    }

    // Indexed draws from `args`, as many as the count at `count` says, at most
    // `max_count`. Recorded through `loader`, the
    // device's `VK_KHR_draw_indirect_count` entry points.
    pub(in crate::vulkan) fn draw_indexed_indirect_count(
        &self,
        loader: &ash::khr::draw_indirect_count::Device,
        args: BufferAt,
        count: BufferAt,
        max_count: u32,
        stride: u32,
    ) {
        // SAFETY: `self.cmd` is in the recording state by construction, the loader was created
        // for this recorder's device, and both buffers are live for the call.
        unsafe {
            loader.cmd_draw_indexed_indirect_count(
                self.cmd,
                args.buffer,
                args.offset,
                count.buffer,
                count.offset,
                max_count,
                stride,
            );
        };
    }

    // Fill `size` bytes of `buffer` from `offset` with `word`.
    pub(in crate::vulkan) fn fill_buffer(
        &self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
        word: u32,
    ) {
        // SAFETY: `self.cmd` is in the recording state by construction, and the buffer is live for
        // the call.
        unsafe {
            self.device
                .cmd_fill_buffer(self.cmd, buffer, offset, size, word);
        };
    }

    pub(in crate::vulkan) fn dispatch(&self, x: u32, y: u32, z: u32) {
        // SAFETY: `self.cmd` is in the recording state by construction.
        unsafe { self.device.cmd_dispatch(self.cmd, x, y, z) };
    }

    // One thread per element of a `threads` grid, in `group` groups, the last
    // on each axis rounded up.
    pub(in crate::vulkan) fn dispatch_threads(&self, threads: [u32; 3], group: ThreadGroup) {
        let [x, y, z] = group.groups(threads);
        self.dispatch(x, y, z);
    }

    // A dispatch whose three group counts sit at `offset` in `args`.
    pub(in crate::vulkan) fn dispatch_indirect(&self, args: vk::Buffer, offset: vk::DeviceSize) {
        // SAFETY: `self.cmd` is in the recording state by construction, and the arguments buffer
        // is live for the call.
        unsafe { self.device.cmd_dispatch_indirect(self.cmd, args, offset) };
    }

    // Write `words` into `buffer` at `offset`, in the command stream.
    pub(in crate::vulkan) fn update_buffer(
        &self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        words: &[u32],
    ) {
        // SAFETY: `self.cmd` is in the recording state by construction, and the buffer and the
        // words are live for the call.
        unsafe {
            self.device
                .cmd_update_buffer(self.cmd, buffer, offset, bytemuck::cast_slice(words));
        };
    }

    // Clear `range` of `image`, which is in `layout`, to `color`.
    pub(in crate::vulkan) fn clear_color_image(
        &self,
        image: vk::Image,
        layout: vk::ImageLayout,
        color: &vk::ClearColorValue,
        range: vk::ImageSubresourceRange,
    ) {
        // SAFETY: `self.cmd` is in the recording state by construction, and the image is live for
        // the call.
        unsafe {
            self.device
                .cmd_clear_color_image(self.cmd, image, layout, color, &[range]);
        };
    }

    // The full barrier form, for a pass that carries buffer or image barriers.
    pub(in crate::vulkan) fn pipeline_barrier(
        &self,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
        memory: &[vk::MemoryBarrier<'_>],
        buffers: &[vk::BufferMemoryBarrier<'_>],
        images: &[vk::ImageMemoryBarrier<'_>],
    ) {
        // SAFETY: `self.cmd` is in the recording state by construction, and the barriers and the
        // handles they name are live for the call.
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.cmd,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                memory,
                buffers,
                images,
            );
        };
    }

    pub(in crate::vulkan) fn bind_vertex_buffers(
        &self,
        first_binding: u32,
        buffers: &[vk::Buffer],
        offsets: &[vk::DeviceSize],
    ) {
        // SAFETY: `self.cmd` is in the recording state by construction, and the buffers and the
        // slices naming them are live for the call.
        unsafe {
            self.device
                .cmd_bind_vertex_buffers(self.cmd, first_binding, buffers, offsets);
        };
    }

    pub(in crate::vulkan) fn copy_buffer(
        &self,
        src: vk::Buffer,
        dst: vk::Buffer,
        regions: &[vk::BufferCopy],
    ) {
        // SAFETY: `self.cmd` is in the recording state by construction, and the buffers and the
        // region slice are live for the call.
        unsafe { self.device.cmd_copy_buffer(self.cmd, src, dst, regions) };
    }

    // Copy `regions` of `buffer` into `image`, which is in `layout`.
    pub(in crate::vulkan) fn copy_buffer_to_image(
        &self,
        buffer: vk::Buffer,
        image: vk::Image,
        layout: vk::ImageLayout,
        regions: &[vk::BufferImageCopy],
    ) {
        // SAFETY: `self.cmd` is in the recording state by construction, and the buffer, the image
        // and the region slice are live for the call.
        unsafe {
            self.device
                .cmd_copy_buffer_to_image(self.cmd, buffer, image, layout, regions);
        };
    }

    // Copy `regions` of `image`, which is in `layout`, into `buffer`.
    pub(in crate::vulkan) fn copy_image_to_buffer(
        &self,
        image: vk::Image,
        layout: vk::ImageLayout,
        buffer: vk::Buffer,
        regions: &[vk::BufferImageCopy],
    ) {
        // SAFETY: `self.cmd` is in the recording state by construction, and the image, the buffer
        // and the region slice are live for the call.
        unsafe {
            self.device
                .cmd_copy_image_to_buffer(self.cmd, image, layout, buffer, regions);
        };
    }

    // Copy `regions` from `src`, in `src_layout`, to `dst`, in `dst_layout`.
    pub(in crate::vulkan) fn copy_image(
        &self,
        src: vk::Image,
        src_layout: vk::ImageLayout,
        dst: vk::Image,
        dst_layout: vk::ImageLayout,
        regions: &[vk::ImageCopy],
    ) {
        // SAFETY: `self.cmd` is in the recording state by construction, and both images and the
        // region slice are live for the call.
        unsafe {
            self.device
                .cmd_copy_image(self.cmd, src, src_layout, dst, dst_layout, regions);
        };
    }

    // Clear `range` of the depth image `image`, which is in `layout`.
    pub(in crate::vulkan) fn clear_depth_stencil_image(
        &self,
        image: vk::Image,
        layout: vk::ImageLayout,
        value: vk::ClearDepthStencilValue,
        range: vk::ImageSubresourceRange,
    ) {
        // SAFETY: `self.cmd` is in the recording state by construction, and the image is live for
        // the call.
        unsafe {
            self.device
                .cmd_clear_depth_stencil_image(self.cmd, image, layout, &value, &[range]);
        };
    }

    // Reset `count` queries of `pool` from `first`.
    pub(in crate::vulkan) fn reset_query_pool(&self, pool: vk::QueryPool, first: u32, count: u32) {
        // SAFETY: `self.cmd` is in the recording state by construction, and the query pool is live
        // for the call.
        unsafe {
            self.device
                .cmd_reset_query_pool(self.cmd, pool, first, count);
        };
    }

    pub(in crate::vulkan) fn write_timestamp(
        &self,
        stage: vk::PipelineStageFlags,
        pool: vk::QueryPool,
        query: u32,
    ) {
        // SAFETY: `self.cmd` is in the recording state by construction, and the query pool is live
        // for the call.
        unsafe {
            self.device
                .cmd_write_timestamp(self.cmd, stage, pool, query);
        };
    }
}
