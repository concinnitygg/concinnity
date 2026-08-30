// src/vulkan/probe_prefilter.rs
//
// The convolution half of a runtime reflection-probe bake on Vulkan: the three
// compute pipelines built from `probe_prefilter.slang`, the two cube images one
// bake works between, and the dispatches that turn six captured faces into the
// prefiltered radiance cube the specular term samples. Mirrors
// `metal::probe_prefilter` and `directx::probe_prefilter`.
//
// The capture cube collects the six rendered faces (one array layer each) and
// carries a mip chain the `probe_downsample` kernel fills; the probe cube is the
// result, mip 0 a firefly-clamped copy of the capture and every mip after it a
// GGX convolution at that mip's roughness. Both are R16G16B16A16_SFLOAT: the
// faces are rendered as halfs, the clamp caps luminance well inside the format's
// range, and it halves what a probe costs against the R32G32B32A32 cube the CPU
// convolution used to upload.
//
// Nothing reads back. The whole convolution stays on the graphics queue, so the
// frames that sample the finished cube are ordered after the dispatches that
// wrote it by submission order alone.
//
// Layouts, which the barriers below are the whole of: the capture arrives in
// TRANSFER_DST (the per-face copies write it), moves to GENERAL for the pyramid
// build, then to SHADER_READ_ONLY_OPTIMAL for the GGX dispatches that sample it.
// The probe cube sits in GENERAL for every dispatch that writes it and moves to
// SHADER_READ_ONLY_OPTIMAL at install.

use ash::vk;

use concinnity_core::render::reflection_probe::PrefilterPlan;
use concinnity_core::render::uniforms::ProbePrefilterParams;

use super::allocator::{DeviceAllocator, PooledImage};
use super::owned::{
    OwnedDescriptorPool, OwnedPipeline, OwnedPipelineLayout, OwnedSampler, OwnedSetLayout, VkDevice,
};
use super::resources::alloc_descriptor_sets;

// Colour format of both cubes.
pub(super) const PROBE_CUBE_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

// Threadgroup tile, matching the kernels' `[numthreads(8, 8, 1)]`. The third
// dispatch dimension is the six cube faces, one invocation deep.
const PREFILTER_TILE: u32 = 8;

// Whole-image subresource range of a cube: every mip, all six layers.
fn cube_range(mips: u32) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: mips,
        base_array_layer: 0,
        layer_count: 6,
    }
}

/// The pipelines a probe bake convolves with, plus the layouts and the sampler
/// they bind. Built once and reused by every bake.
pub(super) struct ProbePrefilterPipelines {
    // The mirror-mip copy and the pyramid reduction bind the same shape (two
    // single-mip storage images), so they share a set layout and a pipeline
    // layout; the GGX kernel binds a sampled cube plus a storage image.
    mip_set_layout: OwnedSetLayout,
    ggx_set_layout: OwnedSetLayout,
    mip_pipeline_layout: OwnedPipelineLayout,
    ggx_pipeline_layout: OwnedPipelineLayout,
    mip0: OwnedPipeline,
    downsample: OwnedPipeline,
    ggx: OwnedPipeline,
    // Linear-clamp mipmapped sampler the GGX kernel taps the pyramid with. The
    // solid-angle lod it computes is fractional, so the trilinear filter is what
    // makes the level selection continuous.
    sampler: OwnedSampler,
}

impl ProbePrefilterPipelines {
    pub(super) fn new(device: &VkDevice, hot_reload: bool) -> Result<Self, String> {
        use super::slang_builtins::SlangCompile;
        let mip_set_layout = create_set_layout(
            device,
            &[
                vk::DescriptorType::STORAGE_IMAGE,
                vk::DescriptorType::STORAGE_IMAGE,
            ],
        )?;
        let ggx_set_layout = create_set_layout(
            device,
            &[
                vk::DescriptorType::SAMPLED_IMAGE,
                vk::DescriptorType::SAMPLER,
                vk::DescriptorType::STORAGE_IMAGE,
            ],
        )?;
        let push = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(size_of::<ProbePrefilterParams>() as u32);
        let mip_pipeline_layout = create_pipeline_layout(device, mip_set_layout.handle(), push)?;
        let ggx_pipeline_layout = create_pipeline_layout(device, ggx_set_layout.handle(), push)?;

        let ctx = super::builtins::Ctx::plain(hot_reload);
        let mip0 = create_compute_pipeline(
            device,
            mip_pipeline_layout.handle(),
            &super::slang_builtins::PROBE_MIP0.compile(&ctx)?,
            "probe_mip0",
        )?;
        let downsample = create_compute_pipeline(
            device,
            mip_pipeline_layout.handle(),
            &super::slang_builtins::PROBE_DOWNSAMPLE.compile(&ctx)?,
            "probe_downsample",
        )?;
        let ggx = create_compute_pipeline(
            device,
            ggx_pipeline_layout.handle(),
            &super::slang_builtins::PROBE_GGX.compile(&ctx)?,
            "probe_ggx",
        )?;
        let sampler = super::texture::create_sampler_cube_linear(device)?;
        Ok(Self {
            mip_set_layout,
            ggx_set_layout,
            mip_pipeline_layout,
            ggx_pipeline_layout,
            mip0,
            downsample,
            ggx,
            sampler,
        })
    }
}

/// The two cube images one bake convolves between, their views, and the
/// descriptor sets its dispatches bind. Owned by the bake, freed when it ends.
pub(super) struct PrefilterGpu {
    // The capture and every view of it the dispatches bind: the views are attached
    // to this image's lease, so holding the image holds them.
    capture: PooledImage,
    probe: PooledImage,
    // All-mips cube view of the finished probe, bound into the frame's cube array.
    probe_cube_view: vk::ImageView,
    // One single-mip 2D-array storage view of the probe cube per mip.
    probe_mip_views: Vec<vk::ImageView>,
    // Sets, all written once at construction: the mirror-mip copy, one
    // downsample per destination mip, one GGX per destination mip.
    mip0_set: vk::DescriptorSet,
    downsample_sets: Vec<vk::DescriptorSet>,
    ggx_sets: Vec<vk::DescriptorSet>,
    // Held, not read: destroying it is what frees the sets above.
    #[expect(dead_code, reason = "owns the sets its handles name")]
    pool: OwnedDescriptorPool,
    mips: u32,
}

impl PrefilterGpu {
    /// Allocate both cubes, their views and every descriptor set the bake's
    /// dispatches bind. The capture starts in TRANSFER_DST so the per-face copies
    /// can write it straight away.
    pub(super) fn new(
        device: &VkDevice,
        alloc: &DeviceAllocator,
        pipelines: &ProbePrefilterPipelines,
        plan: &PrefilterPlan,
    ) -> Result<PrefilterGpu, String> {
        let mips = plan.mips();
        let capture = create_cube_image(
            alloc,
            plan.face_size(),
            mips,
            vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::SAMPLED,
        )?;
        let probe = create_cube_image(
            alloc,
            plan.face_size(),
            mips,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
        )?;
        // Every view is attached to its image's lease, so the whole set retires
        // together whether the bake installs or is abandoned.
        let capture_cube_view = create_cube_view(device, capture.image(), mips)?;
        let probe_cube_view = create_cube_view(device, probe.image(), mips)?;
        let capture_mip_views = mip_storage_views(device, capture.image(), mips)?;
        let probe_mip_views = mip_storage_views(device, probe.image(), mips)?;
        capture.attach_view(capture_cube_view);
        probe.attach_view(probe_cube_view);
        for &view in &capture_mip_views {
            capture.attach_view(view);
        }
        for &view in &probe_mip_views {
            probe.attach_view(view);
        }

        // One mirror-mip set, one downsample set and one GGX set per destination
        // mip past 0. Every set is written now and never rewritten, so a dispatch
        // never touches a set a submitted command buffer still references.
        let steps = mips.saturating_sub(1) as usize;
        let pool = create_pool(device, steps)?;
        let mip_layouts = vec![pipelines.mip_set_layout.handle(); steps + 1];
        let ggx_layouts = vec![pipelines.ggx_set_layout.handle(); steps];
        let mut mip_sets = alloc_descriptor_sets(device, pool.handle(), &mip_layouts)?;
        let ggx_sets = alloc_descriptor_sets(device, pool.handle(), &ggx_layouts)?;
        let mip0_set = mip_sets.remove(0);
        let downsample_sets = mip_sets;

        write_storage_pair(device, mip0_set, capture_mip_views[0], probe_mip_views[0]);
        for (step, &set) in downsample_sets.iter().enumerate() {
            let dst = step + 1;
            write_storage_pair(
                device,
                set,
                capture_mip_views[dst - 1],
                capture_mip_views[dst],
            );
        }
        for (step, &set) in ggx_sets.iter().enumerate() {
            write_ggx_set(
                device,
                set,
                capture_cube_view,
                pipelines.sampler.handle(),
                probe_mip_views[step + 1],
            );
        }

        Ok(PrefilterGpu {
            capture,
            probe,
            probe_cube_view,
            probe_mip_views,
            mip0_set,
            downsample_sets,
            ggx_sets,
            pool,
            mips,
        })
    }

    /// The capture image the six face copies write into.
    pub(super) fn capture_image(&self) -> vk::Image {
        self.capture.image()
    }

    /// The probe cube being written, for the install's layout transition.
    pub(super) fn probe_image(&self) -> vk::Image {
        self.probe.image()
    }

    /// The finished probe cube, handed to the probe pool at install. The capture
    /// image, the descriptor pool, and every view the convolution bound drop with
    /// the rest of `self`; the probe cube's own views are already on its lease.
    pub(super) fn into_probe_cube(self) -> super::texture::GpuImage {
        super::texture::GpuImage::from_pooled_with_aux(
            self.probe,
            self.probe_cube_view,
            self.probe_mip_views,
        )
    }
}

impl super::context::VkContext {
    /// Record the cheap half of the convolution: the capture moves from the
    /// per-face copies' TRANSFER_DST into GENERAL, the mirror mip is copied
    /// through with the firefly clamp, the source pyramid is reduced level by
    /// level, and the capture ends in SHADER_READ_ONLY_OPTIMAL for the GGX
    /// dispatches that follow. The probe cube is put in GENERAL first and stays
    /// there until install.
    ///
    /// All of it goes in one command buffer: the reductions are a few taps per
    /// texel, and each depends on the one before, so spreading them over frames
    /// would only lengthen the bake.
    pub(in crate::vulkan) fn encode_probe_pyramid(
        &self,
        cmd: vk::CommandBuffer,
        gpu: &PrefilterGpu,
        plan: &PrefilterPlan,
    ) -> Result<(), String> {
        let pipelines = self
            .probe
            .prefilter
            .as_ref()
            .ok_or("probe: prefilter pipelines missing")?;
        let device = &self.device;

        transition(
            device,
            cmd,
            gpu.capture.image(),
            gpu.mips,
            LayoutSide {
                layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                access: vk::AccessFlags::TRANSFER_WRITE,
                stage: vk::PipelineStageFlags::TRANSFER,
            },
            LayoutSide {
                layout: vk::ImageLayout::GENERAL,
                // The downsample dispatches WRITE mips 1..n as well as reading:
                // without SHADER_WRITE in the destination scope the transition's
                // own write is only execution-ordered against them (a WAW hazard).
                access: vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                stage: vk::PipelineStageFlags::COMPUTE_SHADER,
            },
        );
        transition(
            device,
            cmd,
            gpu.probe.image(),
            gpu.mips,
            LayoutSide {
                layout: vk::ImageLayout::UNDEFINED,
                access: vk::AccessFlags::empty(),
                stage: vk::PipelineStageFlags::TOP_OF_PIPE,
            },
            LayoutSide {
                layout: vk::ImageLayout::GENERAL,
                access: vk::AccessFlags::SHADER_WRITE,
                stage: vk::PipelineStageFlags::COMPUTE_SHADER,
            },
        );

        self.dispatch_prefilter(
            cmd,
            pipelines.mip_pipeline_layout.handle(),
            pipelines.mip0.handle(),
            gpu.mip0_set,
            &plan.mip0_params(),
            plan.face_size(),
        );
        for (step, &set) in gpu.downsample_sets.iter().enumerate() {
            let dst = step as u32 + 1;
            // Each level reads the one the previous dispatch wrote.
            storage_barrier(device, cmd);
            self.dispatch_prefilter(
                cmd,
                pipelines.mip_pipeline_layout.handle(),
                pipelines.downsample.handle(),
                set,
                &plan.downsample_params(dst),
                plan.mip_face_size(dst),
            );
        }

        transition(
            device,
            cmd,
            gpu.capture.image(),
            gpu.mips,
            LayoutSide {
                layout: vk::ImageLayout::GENERAL,
                access: vk::AccessFlags::SHADER_WRITE,
                stage: vk::PipelineStageFlags::COMPUTE_SHADER,
            },
            LayoutSide {
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                access: vk::AccessFlags::SHADER_READ,
                stage: vk::PipelineStageFlags::COMPUTE_SHADER,
            },
        );
        Ok(())
    }

    /// Record the GGX convolution producing probe-cube mip `dst_mip`, sampling
    /// the finished pyramid. Nothing else writes that mip, and the capture is
    /// read-only from here on, so consecutive mips need no barrier between them.
    pub(in crate::vulkan) fn encode_probe_ggx_mip(
        &self,
        cmd: vk::CommandBuffer,
        gpu: &PrefilterGpu,
        plan: &PrefilterPlan,
        dst_mip: u32,
    ) -> Result<(), String> {
        let pipelines = self
            .probe
            .prefilter
            .as_ref()
            .ok_or("probe: prefilter pipelines missing")?;
        let set = *gpu
            .ggx_sets
            .get(dst_mip as usize - 1)
            .ok_or("probe: convolution mip out of range")?;
        self.dispatch_prefilter(
            cmd,
            pipelines.ggx_pipeline_layout.handle(),
            pipelines.ggx.handle(),
            set,
            &plan.ggx_params(dst_mip),
            plan.mip_face_size(dst_mip),
        );
        Ok(())
    }

    /// Move a finished probe cube into SHADER_READ_ONLY_OPTIMAL so the forward,
    /// SSR and ray-traced resolves can sample it.
    pub(in crate::vulkan) fn encode_probe_cube_readable(
        &self,
        cmd: vk::CommandBuffer,
        image: vk::Image,
        mips: u32,
    ) {
        transition(
            &self.device,
            cmd,
            image,
            mips,
            LayoutSide {
                layout: vk::ImageLayout::GENERAL,
                access: vk::AccessFlags::SHADER_WRITE,
                stage: vk::PipelineStageFlags::COMPUTE_SHADER,
            },
            LayoutSide {
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                access: vk::AccessFlags::SHADER_READ,
                stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
            },
        );
    }

    // Bind, push and dispatch one prefilter kernel over a `size`-square cube face,
    // six faces deep. The kernels bounds-guard against `dst_size`, so the
    // rounded-up remainder returns early.
    fn dispatch_prefilter(
        &self,
        cmd: vk::CommandBuffer,
        layout: vk::PipelineLayout,
        pipeline: vk::Pipeline,
        set: vk::DescriptorSet,
        params: &ProbePrefilterParams,
        size: u32,
    ) {
        let groups = size.div_ceil(PREFILTER_TILE).max(1);
        // SAFETY: `cmd` is in the recording state, and every handle these commands name belongs to
        // this device; the push range matches the layout's, declared from the same type.
        unsafe {
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                std::slice::from_ref(&set),
                &[],
            );
            self.device.cmd_push_constants(
                cmd,
                layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(params),
            );
            self.device.cmd_dispatch(cmd, groups, groups, 6);
        }
    }
}

// A cube image: six array layers with the CUBE_COMPATIBLE flag, `mips` levels.
fn create_cube_image(
    alloc: &DeviceAllocator,
    face_size: u32,
    mips: u32,
    usage: vk::ImageUsageFlags,
) -> Result<PooledImage, String> {
    let info = vk::ImageCreateInfo::default()
        .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE)
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width: face_size,
            height: face_size,
            depth: 1,
        })
        .mip_levels(mips)
        .array_layers(6)
        .format(PROBE_CUBE_FORMAT)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::TYPE_1);
    alloc
        .create_image(&info, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .map_err(|e| format!("probe cube image: {e}"))
}

// All-mips CUBE view, the shape a sampler reads.
fn create_cube_view(
    device: &VkDevice,
    image: vk::Image,
    mips: u32,
) -> Result<vk::ImageView, String> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::CUBE)
        .format(PROBE_CUBE_FORMAT)
        .subresource_range(cube_range(mips));
    // SAFETY: the create-info and every slice it borrows are live for the call, and each handle it
    // names belongs to this device.
    unsafe { device.create_image_view(&info, None) }.map_err(|e| format!("probe cube view: {e}"))
}

// One single-mip 2D_ARRAY storage view per mip. A cube is a six-layer array, so
// this is what lets a kernel address (x, y, face) directly.
fn mip_storage_views(
    device: &VkDevice,
    image: vk::Image,
    mips: u32,
) -> Result<Vec<vk::ImageView>, String> {
    (0..mips)
        .map(|mip| {
            let info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
                .format(PROBE_CUBE_FORMAT)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: mip,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 6,
                });
            // SAFETY: the create-info and every slice it borrows are live for the call, and each
            // handle it names belongs to this device.
            unsafe { device.create_image_view(&info, None) }
                .map_err(|e| format!("probe mip {mip} view: {e}"))
        })
        .collect()
}

fn create_set_layout(
    device: &VkDevice,
    types: &[vk::DescriptorType],
) -> Result<OwnedSetLayout, String> {
    let binds: Vec<_> = types
        .iter()
        .enumerate()
        .map(|(i, &ty)| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i as u32)
                .descriptor_type(ty)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    device
        .create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binds),
        )
        .map_err(|e| format!("probe prefilter set layout: {e}"))
}

fn create_pipeline_layout(
    device: &VkDevice,
    set_layout: vk::DescriptorSetLayout,
    push: vk::PushConstantRange,
) -> Result<OwnedPipelineLayout, String> {
    let layouts = [set_layout];
    device
        .create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&layouts)
                .push_constant_ranges(std::slice::from_ref(&push)),
        )
        .map_err(|e| format!("probe prefilter pipeline layout: {e}"))
}

// One bake's sets: the mirror-mip copy plus one downsample and one GGX set per
// destination mip past 0.
fn create_pool(device: &VkDevice, steps: usize) -> Result<OwnedDescriptorPool, String> {
    let steps = steps as u32;
    let sizes = [
        // mip0 (2) + downsample (2 each) + GGX dst (1 each).
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(2 + 3 * steps),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(steps),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::SAMPLER)
            .descriptor_count(steps),
    ];
    device
        .create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&sizes)
                .max_sets(1 + 2 * steps),
        )
        .map_err(|e| format!("probe prefilter descriptor pool: {e}"))
}

// Bindings 0 and 1 of a mirror-copy or downsample set: the source mip and the
// destination mip, both storage images.
fn write_storage_pair(
    device: &VkDevice,
    set: vk::DescriptorSet,
    src: vk::ImageView,
    dst: vk::ImageView,
) {
    let src_info = storage_info(src);
    let dst_info = storage_info(dst);
    let writes = [
        storage_write(set, 0, std::slice::from_ref(&src_info)),
        storage_write(set, 1, std::slice::from_ref(&dst_info)),
    ];
    // SAFETY: `writes` and the image infos it borrows are live for the call, and every handle they
    // name belongs to this device.
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

// The GGX set: the sampled capture pyramid, its sampler, and the destination mip.
fn write_ggx_set(
    device: &VkDevice,
    set: vk::DescriptorSet,
    cube: vk::ImageView,
    sampler: vk::Sampler,
    dst: vk::ImageView,
) {
    let cube_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(cube);
    let sampler_info = vk::DescriptorImageInfo::default().sampler(sampler);
    let dst_info = storage_info(dst);
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(std::slice::from_ref(&cube_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .image_info(std::slice::from_ref(&sampler_info)),
        storage_write(set, 2, std::slice::from_ref(&dst_info)),
    ];
    // SAFETY: `writes` and the image infos it borrows are live for the call, and every handle they
    // name belongs to this device.
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

fn storage_info(view: vk::ImageView) -> vk::DescriptorImageInfo {
    vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::GENERAL)
        .image_view(view)
}

fn storage_write<'a>(
    set: vk::DescriptorSet,
    binding: u32,
    info: &'a [vk::DescriptorImageInfo],
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .image_info(info)
}

// Order one storage-image write before the next dispatch's read of it. The next
// dispatch also writes the following mip, so its writes join the destination
// scope to keep the chained dependency a memory one for them too.
fn storage_barrier(device: &VkDevice, cmd: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
    // SAFETY: `cmd` is in the recording state and the barrier it borrows is live for the call.
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            std::slice::from_ref(&barrier),
            &[],
            &[],
        );
    }
}

// One side of a layout transition: the layout the image is in, the accesses that
// have to be made available (source) or visible (destination), and the stage they
// happen in. The shared `transition_image_layout_range` cannot serve these: it
// derives the masks from the layout pair alone, and its table is graphics-staged
// and falls back to empty masks for a pair it does not know -- which is exactly
// the compute pairs this convolution needs.
#[derive(Clone, Copy)]
struct LayoutSide {
    layout: vk::ImageLayout,
    access: vk::AccessFlags,
    stage: vk::PipelineStageFlags,
}

// Whole-cube layout transition.
fn transition(
    device: &VkDevice,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    mips: u32,
    from: LayoutSide,
    to: LayoutSide,
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .src_access_mask(from.access)
        .dst_access_mask(to.access)
        .old_layout(from.layout)
        .new_layout(to.layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(cube_range(mips));
    // SAFETY: `cmd` is in the recording state, the barrier it borrows is live for the call, and the
    // image belongs to this device.
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            from.stage,
            to.stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            std::slice::from_ref(&barrier),
        );
    }
}

fn create_compute_pipeline(
    device: &VkDevice,
    layout: vk::PipelineLayout,
    spv: &[u8],
    label: &str,
) -> Result<OwnedPipeline, String> {
    let module = super::pipeline::spv_module(device, spv)?;
    let entry = std::ffi::CString::new("main").expect("static entry name has no interior nul");
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module.handle())
        .name(&entry);
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(layout);
    crate::vulkan::pipeline_cache::create_compute_pipeline(device, &info)
        .map_err(|e| format!("create {label} pipeline: {e}"))
}
