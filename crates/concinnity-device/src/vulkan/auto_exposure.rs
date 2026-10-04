//! Auto-exposure (EV adaptation) on Vulkan: a per-frame CPU readback of a
//! previous frame's average log-luminance, an EMA step that updates the
//! adapted EV, and the histogram build + average compute dispatches that
//! produce next frame's average. The compute passes are encoded after the
//! main HDR resolve (where `hdr_resolve_images[frame_idx]` carries this
//! frame's scene color in SHADER_READ_ONLY_OPTIMAL) and the result is
//! copied into a per-frame HOST_VISIBLE readback buffer that the CPU reads
//! at the top of a later frame, so there is `frames_in_flight` frames of
//! latency between the scene's actual luminance and the exposure applied,
//! invisible at human-scale eye-adaptation rates. Mirrors
//! `metal/auto_exposure.rs` and `directx/auto_exposure.rs`.

use ash::vk;
use concinnity_core::gfx::auto_exposure::HISTOGRAM_BINS;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::uniforms::AutoExposureParams;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::context::VkContext;
use super::descriptor_layout::{Binding, PoolSizes};
use super::pipeline_desc::compute_pipeline;
use super::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use super::set_writes::SetWrites;
use crate::vulkan::builtin_shaders::CompileProgram;
use crate::vulkan::owned::{
    OwnedDescriptorPool, OwnedPipeline, OwnedPipelineLayout, OwnedSetLayout, VkDevice,
};
use crate::vulkan::record::Recorder;
use concinnity_core::render::uniforms::vulkan::AUTO_EXPOSURE_PUSH_BYTES;

// Compile the auto-exposure build + average compute kernels.
fn compile_auto_exposure_shaders(hot_reload: bool) -> RenderResult<(Vec<u8>, Vec<u8>)> {
    let build_cs = super::builtin_shaders::AUTO_EXPOSURE_BUILD.compile(hot_reload)?;
    let average_cs = super::builtin_shaders::AUTO_EXPOSURE_AVERAGE.compile(hot_reload)?;
    Ok((build_cs, average_cs))
}

// Push-constant payload pushed at the top of every compute dispatch.
// Owns the compute pipelines + GPU buffers + per-frame readback driving
// the auto-exposure histogram path. Built only when the world's
// `PostProcessConfig` opts in; the encoder is a no-op otherwise.
pub(in crate::vulkan) struct AutoExposureResources {
    // Build kernel: one thread per HDR-resolve pixel; merges per-threadgroup
    // local histograms into the global histogram SSBO.
    build_pipeline: OwnedPipeline,
    build_pipeline_layout: OwnedPipelineLayout,
    _build_set_layout: OwnedSetLayout,
    // One build set per frame: binding 0 references that frame slot's
    // `hdr_resolve_images[frame_idx]` view.
    build_sets: Vec<vk::DescriptorSet>,

    // Average kernel: one threadgroup of HISTOGRAM_BINS threads reduces the
    // histogram, clears it, and writes the average log-luminance.
    average_pipeline: OwnedPipeline,
    average_pipeline_layout: OwnedPipelineLayout,
    _average_set_layout: OwnedSetLayout,
    // Single shared average set: both buffers are global, no per-frame
    // variation.
    average_set: vk::DescriptorSet,

    _descriptor_pool: OwnedDescriptorPool,

    // Device-local 256-bin u32 histogram. The build kernel atomically
    // increments bins into it; the average kernel reads and clears them.
    histogram_buffer: PooledBuffer,
    // Device-local single f32 the average kernel writes; copied into the
    // per-frame readback after each dispatch.
    output_buffer: PooledBuffer,

    // Per-frame HOST_VISIBLE readback buffers. Each holds 4 bytes (one
    // f32). Persistently mapped; the CPU reads this frame's slot at the top
    // of a later frame after the fence wait gates this slot's previous copy.
    readback_buffers: Vec<PooledBuffer>,
}

impl AutoExposureResources {
    // Build all auto-exposure resources. Called from `VkContext::new` only
    // when `PostProcessConfig.auto_exposure` is enabled.
    pub(in crate::vulkan) fn new(
        alloc: &DeviceAllocator,
        device: &VkDevice,
        frames: usize,
        hdr_resolve_views: &[vk::ImageView],
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let build_set_layout = create_descriptor_set_layout(device, &build_set_bindings())?;
        let average_set_layout = create_descriptor_set_layout(device, &average_set_bindings())?;

        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(AUTO_EXPOSURE_PUSH_BYTES);

        let build_layouts = [build_set_layout.handle()];
        let build_pipeline_layout = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&build_layouts)
                    .push_constant_ranges(std::slice::from_ref(&push_range)),
            )
            .map_err(|e| super::error::map_vk_result(e, "auto-exposure build pipeline layout"))?;
        let average_layouts = [average_set_layout.handle()];
        let average_pipeline_layout = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&average_layouts)
                    .push_constant_ranges(std::slice::from_ref(&push_range)),
            )
            .map_err(|e| super::error::map_vk_result(e, "auto-exposure average pipeline layout"))?;

        let (build_pipeline, average_pipeline) = create_pipelines(
            device,
            build_pipeline_layout.handle(),
            average_pipeline_layout.handle(),
            hot_reload,
        )?;

        // Histogram + output buffers (device-local).
        let histogram_bytes = (HISTOGRAM_BINS * std::mem::size_of::<u32>()) as vk::DeviceSize;
        let histogram_buffer = alloc.create_buffer(
            histogram_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let output_bytes = std::mem::size_of::<f32>() as vk::DeviceSize;
        let output_buffer = alloc.create_buffer(
            output_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        // Per-frame HOST_VISIBLE readback buffers (persistently mapped).
        let mut readback_buffers = Vec::with_capacity(frames);
        for _ in 0..frames {
            readback_buffers.push(alloc.create_buffer(
                output_bytes,
                vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?);
        }

        // Descriptor pool: `frames` build sets + 1 average set.
        let pool_sizes = PoolSizes::default()
            .sets(&build_set_bindings(), frames as u32)
            .sets(&average_set_bindings(), 1)
            .build();
        let descriptor_pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets((frames + 1) as u32)
                    .pool_sizes(&pool_sizes),
            )
            .map_err(|e| super::error::map_vk_result(e, "auto-exposure descriptor pool"))?;

        let build_set_layouts: Vec<_> = (0..frames).map(|_| build_set_layout.handle()).collect();
        let build_sets =
            alloc_descriptor_sets(device, descriptor_pool.handle(), &build_set_layouts)?;
        let average_set = alloc_descriptor_sets(
            device,
            descriptor_pool.handle(),
            &[average_set_layout.handle()],
        )?[0];

        // Write each build set's HDR sampled image + histogram bindings.
        let last_view_idx = hdr_resolve_views.len().saturating_sub(1);
        for (i, &set) in build_sets.iter().enumerate() {
            let view = hdr_resolve_views[i.min(last_view_idx)];
            write_build_set(device, set, view, histogram_buffer.buffer());
        }
        SetWrites::new(average_set)
            .storage_buffer(0, histogram_buffer.buffer(), vk::WHOLE_SIZE)
            .storage_buffer(1, output_buffer.buffer(), vk::WHOLE_SIZE)
            .apply(device);

        Ok(Self {
            build_pipeline,
            build_pipeline_layout,
            _build_set_layout: build_set_layout,
            build_sets,
            average_pipeline,
            average_pipeline_layout,
            _average_set_layout: average_set_layout,
            average_set,
            _descriptor_pool: descriptor_pool,
            histogram_buffer,
            output_buffer,
            readback_buffers,
        })
    }

    // Recompile both kernels and build them against the existing pipeline
    // layouts. Driven by the shader hot-reload pass.
    pub(in crate::vulkan) fn rebuild_pipelines(
        &self,
        device: &VkDevice,
        hot_reload: bool,
    ) -> RenderResult<(OwnedPipeline, OwnedPipeline)> {
        create_pipelines(
            device,
            self.build_pipeline_layout.handle(),
            self.average_pipeline_layout.handle(),
            hot_reload,
        )
    }

    // Swap the freshly-built build + average pipelines into the live
    // resources. The caller has already `device_wait_idle`'d so the old
    // pipelines are not in flight.
    pub(in crate::vulkan) fn swap_pipelines(
        &mut self,
        build_pipeline: OwnedPipeline,
        average_pipeline: OwnedPipeline,
    ) {
        self.build_pipeline = build_pipeline;
        self.average_pipeline = average_pipeline;
    }

    // Rewrite the per-frame build sets' HDR sampled-image binding after a
    // swapchain rebuild swapped the `hdr_resolve_images`. The histogram /
    // output buffers are resolution-independent and survive the rebuild
    // untouched; only binding 0 of each build set needs to point at the
    // new view.
    pub(in crate::vulkan) fn rebuild(
        &mut self,
        device: &VkDevice,
        hdr_resolve_views: &[vk::ImageView],
    ) {
        let last_view_idx = hdr_resolve_views.len().saturating_sub(1);
        for (i, &set) in self.build_sets.iter().enumerate() {
            let view = hdr_resolve_views[i.min(last_view_idx)];
            write_build_set(device, set, view, self.histogram_buffer.buffer());
        }
    }

    // Free every owned handle. Called from `Drop for VkContext` after
    // `device_wait_idle`.
    pub(in crate::vulkan) fn destroy(&mut self, _device: &VkDevice) {
        self.histogram_buffer = PooledBuffer::null();
        self.output_buffer = PooledBuffer::null();
        self.readback_buffers.clear();
    }
}

// Build set: the HDR image, read by texel, and the histogram SSBO.
fn build_set_bindings() -> [Binding; 2] {
    let compute = vk::ShaderStageFlags::COMPUTE;
    [
        (0, vk::DescriptorType::SAMPLED_IMAGE, compute),
        (1, vk::DescriptorType::STORAGE_BUFFER, compute),
    ]
}

// Average set: the histogram SSBO and the output SSBO.
fn average_set_bindings() -> [Binding; 2] {
    let compute = vk::ShaderStageFlags::COMPUTE;
    [
        (0, vk::DescriptorType::STORAGE_BUFFER, compute),
        (1, vk::DescriptorType::STORAGE_BUFFER, compute),
    ]
}

fn write_build_set(
    device: &VkDevice,
    set: vk::DescriptorSet,
    view: vk::ImageView,
    histogram: vk::Buffer,
) {
    SetWrites::new(set)
        .sampled_image(0, view)
        .storage_buffer(1, histogram, vk::WHOLE_SIZE)
        .apply(device);
}

// The build + average compute pipelines against their layouts.
fn create_pipelines(
    device: &VkDevice,
    build_layout: vk::PipelineLayout,
    average_layout: vk::PipelineLayout,
    hot_reload: bool,
) -> RenderResult<(OwnedPipeline, OwnedPipeline)> {
    let (build_spv, average_spv) = compile_auto_exposure_shaders(hot_reload)?;
    let build = compute_pipeline(device, build_layout, &build_spv, "auto-exposure build")?;
    let average = compute_pipeline(
        device,
        average_layout,
        &average_spv,
        "auto-exposure average",
    )?;
    Ok((build, average))
}

impl VkContext {
    // Step the auto-exposure EMA from a previous frame's GPU measurement,
    // then push the new exposure multiplier into `self.post_process.exposure`.
    // A no-op when auto-exposure is disabled: the static authored EV then
    // drives `exposure` unchanged.
    //
    // Called at the top of `draw_frame` after the fence wait for this slot's
    // previous use completes, so the matching readback buffer holds a fully
    // committed GPU result (one or two frames stale, smoothed by the EMA).
    // `elapsed` is the total elapsed seconds since startup; the per-call
    // diff drives `dt` for the EMA.
    pub(in crate::vulkan) fn update_auto_exposure(&mut self, elapsed: f32, frame_idx: usize) {
        let Some(adaptation) = self.auto_exposure.adaptation.as_mut() else {
            return;
        };
        let Some(resources) = self.auto_exposure.resources.as_ref() else {
            return;
        };
        let Some(readback) = resources.readback_buffers.get(frame_idx) else {
            return;
        };
        let ptr = readback.mapped_ptr() as *const f32;

        // Read the previous frame's average log-luminance for this slot. The
        // fence wait above this call already gated the GPU work that wrote
        // it, so the HOST_COHERENT mapping reflects the committed value.
        // SAFETY: `ptr` is the HOST_COHERENT mapping of this slot's output buffer, which holds one
        // f32; the fence wait above gated the GPU write, so the value is committed and initialized.
        let avg_log_lum = unsafe { ptr.read() };

        let dt = (elapsed - self.auto_exposure.last_elapsed).max(0.0);
        self.auto_exposure.last_elapsed = elapsed;

        // `self.post_process.exposure` is the linear multiplier the bloom
        // prefilter and composite consume.
        self.post_process.exposure = adaptation.step(avg_log_lum, dt);
    }

    // Encode the auto-exposure histogram passes against the current frame's
    // resolved HDR scene. The build kernel runs one thread per HDR pixel;
    // the average kernel runs one threadgroup of `HISTOGRAM_BINS` threads
    // that reduces the histogram, clears it for the next frame, and writes
    // the average log-luminance to the device-local output buffer; a copy
    // then carries the value into this frame's readback buffer for the
    // CPU's EMA step at the top of a later frame. A no-op when
    // auto-exposure is disabled.
    pub(in crate::vulkan) fn encode_auto_exposure(&self, rec: &Recorder<'_>, frame_idx: usize) {
        let Some(resources) = self.auto_exposure.resources.as_ref() else {
            return;
        };
        let params = AutoExposureParams::HISTOGRAM;
        let extent = self.targets.render_extent;
        if extent.width == 0 || extent.height == 0 {
            return;
        }

        // Order Main pass's resolve color writes before our compute
        // shader sample of the HDR resolve image. The render pass's
        // exit-dep targets COLOR_ATTACHMENT_OUTPUT (for the next
        // subpass-attachment consumer); compute-shader reads need a
        // dedicated barrier.
        let pre_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        rec.pipeline_barrier(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            std::slice::from_ref(&pre_barrier),
            &[],
            &[],
        );

        // Build dispatch: 16×16 threadgroups, one thread per HDR pixel.
        let build_set = resources
            .build_sets
            .get(frame_idx)
            .copied()
            .unwrap_or_else(|| resources.build_sets[0]);
        rec.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &resources.build_pipeline);
        rec.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            &resources.build_pipeline_layout,
            0,
            std::slice::from_ref(&build_set),
            &[],
        );
        rec.push_constants(
            &resources.build_pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &params,
        );
        rec.dispatch(extent.width.div_ceil(16), extent.height.div_ceil(16), 1);

        // Order build histogram writes before the average read+clear.
        let hist_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        rec.pipeline_barrier(
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            std::slice::from_ref(&hist_barrier),
            &[],
            &[],
        );

        // Average dispatch: one threadgroup of HISTOGRAM_BINS threads.
        rec.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &resources.average_pipeline);
        rec.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            &resources.average_pipeline_layout,
            0,
            std::slice::from_ref(&resources.average_set),
            &[],
        );
        rec.push_constants(
            &resources.average_pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &params,
        );
        rec.dispatch(1, 1, 1);

        // Order the average kernel's output_buf write before the copy
        // into the readback buffer.
        let out_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        rec.pipeline_barrier(
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            std::slice::from_ref(&out_barrier),
            &[],
            &[],
        );

        // Copy the freshly-written average to this slot's readback buffer.
        let readback = resources
            .readback_buffers
            .get(frame_idx)
            .unwrap_or(&resources.readback_buffers[0]);
        let copy = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: std::mem::size_of::<f32>() as vk::DeviceSize,
        };
        rec.copy_buffer(
            resources.output_buffer.buffer(),
            readback.buffer(),
            std::slice::from_ref(&copy),
        );

        // Order the transfer write to the host-visible buffer before the
        // CPU read at the top of a later frame. The fence wait that
        // gates this slot's next trip provides the host-side ordering;
        // this barrier just makes the transfer write visible to the
        // host.
        let host_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);
        rec.pipeline_barrier(
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            std::slice::from_ref(&host_barrier),
            &[],
            &[],
        );
    }
}
