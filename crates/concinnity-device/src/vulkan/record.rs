// src/vulkan/record.rs
//
// Safe command recording. [`Recorder`] is a command buffer that is known to be
// in the recording state, and the surface over it that every pass records
// through.
//
// The `unsafe` on `vkCmd*` carries two obligations: the command buffer is in
// the recording state, and every handle and slice the command names is live for
// the call. Both were restated verbatim at every recording site in this backend,
// which is how they stop being checked. `Recorder` discharges the first by
// construction -- the only safe way to get one is [`Recorder::begin`], which
// puts the buffer into that state itself -- and the second by taking the owning
// wrappers from `owned.rs` by reference, so a pipeline or layout cannot be bound
// unless its owner is alive at the call.
//
// Passes convert to this surface one at a time: `encode_pass_into` hands each
// arm the recorder, and an arm that has not converted reads the raw buffer back
// out with [`Recorder::raw`]. Growing the surface is what converting the next
// pass costs; it carries only the commands its callers use.
//
// What is left raw is stated rather than hidden: descriptor sets, buffers and
// image views are still `vk::*` handles here, because they are owned by a
// descriptor pool, the device allocator or a swapchain rather than by a wrapper
// of their own. Those keep the liveness argument they always had, and
// [`Recorder::raw`] exists for the commands this surface does not cover (the
// extension loaders: acceleration-structure builds and the upscaler SDKs), so a
// pass that needs one does not have to abandon the type.

use ash::vk;

use super::owned::{OwnedFramebuffer, OwnedPipeline, OwnedPipelineLayout, OwnedRenderPass};

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
    // for the recording scopes whose `vkBeginCommandBuffer` sits in a caller
    // that cannot hand the recorder down (the one-shot upload helpers).
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
                .cmd_bind_pipeline(self.cmd, bind_point, pipeline.handle())
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
            )
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
                .cmd_push_constants(self.cmd, layout.handle(), stages, offset, bytes)
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
                .cmd_begin_render_pass(self.cmd, &info, vk::SubpassContents::INLINE)
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
                .cmd_set_viewport(self.cmd, 0, std::slice::from_ref(viewport))
        };
    }

    pub(in crate::vulkan) fn set_scissor(&self, scissor: &vk::Rect2D) {
        // SAFETY: `self.cmd` is in the recording state by construction, and the slice is live for
        // the call.
        unsafe {
            self.device
                .cmd_set_scissor(self.cmd, 0, std::slice::from_ref(scissor))
        };
    }

    // Viewport and scissor covering `extent`, which is what every fullscreen
    // pass sets and what a geometry pass resets to.
    pub(in crate::vulkan) fn set_full_viewport(&self, extent: vk::Extent2D) {
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        self.set_viewport(&viewport);
        self.set_scissor(&vk::Rect2D::default().extent(extent));
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
            )
        };
    }

    // The fullscreen triangle every post pass draws.
    pub(in crate::vulkan) fn draw_fullscreen_triangle(&self) {
        self.draw(3, 1, 0, 0);
    }

    pub(in crate::vulkan) fn dispatch(&self, x: u32, y: u32, z: u32) {
        // SAFETY: `self.cmd` is in the recording state by construction.
        unsafe { self.device.cmd_dispatch(self.cmd, x, y, z) };
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
                .cmd_write_timestamp(self.cmd, stage, pool, query)
        };
    }
}
