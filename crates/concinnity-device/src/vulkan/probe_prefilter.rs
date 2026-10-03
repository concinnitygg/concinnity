//! The convolution half of a runtime reflection-probe bake on Vulkan: the three
//! compute pipelines built from `probe_prefilter.hlsl`, the capture cube one bake
//! works from, and the dispatches that turn six captured faces into the
//! prefiltered radiance cube the specular term samples. Mirrors
//! `metal::probe_prefilter` and `directx::probe_prefilter`.
//!
//! The capture cube collects the six rendered faces (one array layer each) and
//! carries a mip chain the `probe_downsample` kernel fills; the probe cube is
//! the result, one cube of the probe cube array: mip 0 a firefly-clamped copy
//! of the capture and every mip after it a GGX convolution at that mip's
//! roughness. Both are R16G16B16A16_SFLOAT: the faces are rendered as halfs,
//! the clamp caps luminance well inside the format's range, and it halves what
//! a probe costs against an R32G32B32A32 cube.
//!
//! Nothing reads back. The whole convolution stays on the graphics queue, so the
//! frames that sample the finished cube are ordered after the dispatches that
//! wrote it by submission order alone.
//!
//! Layouts, which the barriers below are the whole of: the capture arrives in
//! TRANSFER_DST (the per-face copies write it), moves to GENERAL for the pyramid
//! build, then to SHADER_READ_ONLY_OPTIMAL for the GGX dispatches that sample it.
//! The probe cube array never leaves GENERAL (see `probe_set`), so its cube only
//! takes memory barriers: one ordering the writes after whatever read the cube
//! before, and one making them visible to the fragment reads after install.

use ash::vk;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::reflection_probe::PrefilterPlan;
use concinnity_core::render::uniforms::ProbePrefilterParams;

use super::allocator::{DeviceAllocator, PooledImage};
use super::descriptor_layout::{Binding, PoolSizes};
use super::owned::{
    OwnedDescriptorPool, OwnedPipeline, OwnedPipelineLayout, OwnedSampler, OwnedSetLayout, VkDevice,
};
use super::pipeline_desc::compute_pipeline;
use super::probe_set::{self, CubeImage};
use super::record::cmd_push_constants;
use super::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use super::set_writes::SetWrites;

// Color format of the capture and the probe cube array.
pub(super) const PROBE_CUBE_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

// Threadgroup tile, matching the kernels' `[numthreads(8, 8, 1)]`. The third
// dispatch dimension is the six cube faces, one invocation deep.
const PREFILTER_TILE: u32 = 8;

// Subresource range of cube `cube` of an image: every mip, its six layers.
fn cube_range(cube: u32, mips: u32) -> vk::ImageSubresourceRange {
    probe_set::range(0, mips, 6 * cube, 6)
}

/// The cube of the probe cube array one bake convolves into.
#[derive(Clone, Copy)]
pub(super) struct ProbeSlice<'a> {
    pub(super) cubes: &'a probe_set::ProbeCubeArray,
    pub(super) index: usize,
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
    pub(super) fn new(device: &VkDevice, hot_reload: bool) -> RenderResult<Self> {
        use super::builtin_shaders::CompileProgram;
        let mip_set_layout = create_descriptor_set_layout(device, &mip_set())?;
        let ggx_set_layout = create_descriptor_set_layout(device, &ggx_set())?;
        let push = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(size_of::<ProbePrefilterParams>() as u32);
        let mip_pipeline_layout = create_pipeline_layout(device, mip_set_layout.handle(), push)?;
        let ggx_pipeline_layout = create_pipeline_layout(device, ggx_set_layout.handle(), push)?;

        let mip0 = compute_pipeline(
            device,
            mip_pipeline_layout.handle(),
            &super::builtin_shaders::PROBE_MIP0.compile(hot_reload)?,
            "probe_mip0",
        )?;
        let downsample = compute_pipeline(
            device,
            mip_pipeline_layout.handle(),
            &super::builtin_shaders::PROBE_DOWNSAMPLE.compile(hot_reload)?,
            "probe_downsample",
        )?;
        let ggx = compute_pipeline(
            device,
            ggx_pipeline_layout.handle(),
            &super::builtin_shaders::PROBE_GGX.compile(hot_reload)?,
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

/// The capture image one bake convolves from, the cube of the probe cube array
/// it writes, and the descriptor sets its dispatches bind. Owned by the bake,
/// freed when it ends.
pub(super) struct PrefilterGpu {
    // The capture and every view of it the dispatches bind: the views are attached
    // to this image's lease, so holding the image holds them.
    capture: PooledImage,
    // The probe cube array and the cube this bake writes. The array's own lease
    // holds the storage views the sets below name; it outlives the bake, since a
    // re-placement idles and drops every bake before replacing it.
    probe_image: vk::Image,
    probe_cube: u32,
    // Sets, all written once at construction: the mirror-mip copy, one
    // downsample per destination mip, one GGX per destination mip.
    mip0_set: vk::DescriptorSet,
    downsample_sets: Vec<vk::DescriptorSet>,
    ggx_sets: Vec<vk::DescriptorSet>,
    // Held, not read: destroying it is what frees the sets above.
    _pool: OwnedDescriptorPool,
    mips: u32,
}

impl PrefilterGpu {
    /// Allocate the capture, its views and every descriptor set the bake's
    /// dispatches bind, writing into `slice`. The capture starts in TRANSFER_DST
    /// so the per-face copies can write it straight away.
    pub(super) fn new(
        device: &VkDevice,
        alloc: &DeviceAllocator,
        pipelines: &ProbePrefilterPipelines,
        plan: &PrefilterPlan,
        slice: ProbeSlice<'_>,
    ) -> RenderResult<PrefilterGpu> {
        let mips = plan.mips();
        let capture = probe_set::create_image(
            alloc,
            CubeImage {
                face_size: plan.face_size(),
                mips,
                layers: 6,
                usage: vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::SAMPLED,
            },
        )
        .map_err(|e| e.context("probe capture cube"))?;
        let probe_mip_views = slice
            .cubes
            .mip_views(slice.index)
            .filter(|views| views.len() == mips as usize)
            .ok_or_else(|| {
                RenderError::Other(format!(
                    "probe: the cube array has no cube {} at {mips} mips",
                    slice.index
                ))
            })?;
        // Every view is attached to its image's lease, so the whole set retires
        // together whether the bake installs or is abandoned.
        let capture_cube_view = probe_set::create_view(
            device,
            capture.image(),
            vk::ImageViewType::CUBE,
            cube_range(0, mips),
        )?;
        capture.attach_view(capture_cube_view);
        let capture_mip_views = probe_set::mip_storage_views(device, &capture, 0, mips)?;

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

        SetWrites::new(mip0_set)
            .storage_image(0, capture_mip_views[0])
            .storage_image(1, probe_mip_views[0])
            .apply(device);
        for (step, &set) in downsample_sets.iter().enumerate() {
            let dst = step + 1;
            SetWrites::new(set)
                .storage_image(0, capture_mip_views[dst - 1])
                .storage_image(1, capture_mip_views[dst])
                .apply(device);
        }
        for (step, &set) in ggx_sets.iter().enumerate() {
            SetWrites::new(set)
                .sampled_image(0, capture_cube_view)
                .sampler(1, pipelines.sampler.handle())
                .storage_image(2, probe_mip_views[step + 1])
                .apply(device);
        }

        Ok(PrefilterGpu {
            capture,
            probe_image: slice.cubes.image(),
            probe_cube: slice.index as u32,
            mip0_set,
            downsample_sets,
            ggx_sets,
            _pool: pool,
            mips,
        })
    }

    /// The capture image the six face copies write into.
    pub(super) fn capture_image(&self) -> vk::Image {
        self.capture.image()
    }
}

impl super::context::VkContext {
    /// Record the cheap half of the convolution: the capture moves from the
    /// per-face copies' TRANSFER_DST into GENERAL, the mirror mip is copied
    /// through with the firefly clamp, the source pyramid is reduced level by
    /// level, and the capture ends in SHADER_READ_ONLY_OPTIMAL for the GGX
    /// dispatches that follow. The probe's cube is ordered after whatever read it
    /// last first; it stays in GENERAL throughout.
    ///
    /// All of it goes in one command buffer: the reductions are a few taps per
    /// texel, and each depends on the one before, so spreading them over frames
    /// would only lengthen the bake.
    pub(in crate::vulkan) fn encode_probe_pyramid(
        &self,
        cmd: vk::CommandBuffer,
        gpu: &PrefilterGpu,
        plan: &PrefilterPlan,
    ) -> RenderResult<()> {
        let pipelines =
            self.probe.prefilter.as_ref().ok_or_else(|| {
                RenderError::Other("probe: prefilter pipelines missing".to_string())
            })?;
        let device = &self.hw.device;

        transition(
            device,
            cmd,
            gpu.capture.image(),
            cube_range(0, gpu.mips),
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
        // The cube may hold a replaced placement's probe that earlier frames
        // sampled; this orders the writes after those reads.
        transition(
            device,
            cmd,
            gpu.probe_image,
            cube_range(gpu.probe_cube, gpu.mips),
            LayoutSide {
                layout: probe_set::PROBE_CUBES_LAYOUT,
                access: vk::AccessFlags::empty(),
                stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
            },
            LayoutSide {
                layout: probe_set::PROBE_CUBES_LAYOUT,
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
            cube_range(0, gpu.mips),
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
    ) -> RenderResult<()> {
        let pipelines =
            self.probe.prefilter.as_ref().ok_or_else(|| {
                RenderError::Other("probe: prefilter pipelines missing".to_string())
            })?;
        let set = *gpu
            .ggx_sets
            .get(dst_mip as usize - 1)
            .ok_or_else(|| RenderError::Other("probe: convolution mip out of range".to_string()))?;
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

    /// Make a finished probe cube's writes visible to the forward, SSR,
    /// ray-traced and transparent passes that sample it.
    pub(in crate::vulkan) fn encode_probe_cube_readable(
        &self,
        cmd: vk::CommandBuffer,
        gpu: &PrefilterGpu,
    ) {
        transition(
            &self.hw.device,
            cmd,
            gpu.probe_image,
            cube_range(gpu.probe_cube, gpu.mips),
            LayoutSide {
                layout: probe_set::PROBE_CUBES_LAYOUT,
                access: vk::AccessFlags::SHADER_WRITE,
                stage: vk::PipelineStageFlags::COMPUTE_SHADER,
            },
            LayoutSide {
                layout: probe_set::PROBE_CUBES_LAYOUT,
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
            self.hw
                .device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.hw.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                std::slice::from_ref(&set),
                &[],
            );
            cmd_push_constants(
                &self.hw.device,
                cmd,
                layout,
                vk::ShaderStageFlags::COMPUTE,
                params,
            );
            self.hw.device.cmd_dispatch(cmd, groups, groups, 6);
        }
    }
}

// The mirror-mip copy and the downsample: the source and destination mips, both
// storage images.
fn mip_set() -> [Binding; 2] {
    let compute = vk::ShaderStageFlags::COMPUTE;
    [
        (0, vk::DescriptorType::STORAGE_IMAGE, compute),
        (1, vk::DescriptorType::STORAGE_IMAGE, compute),
    ]
}

// The GGX kernel: the sampled capture pyramid, its sampler, and the destination
// mip.
fn ggx_set() -> [Binding; 3] {
    use vk::DescriptorType as T;
    let compute = vk::ShaderStageFlags::COMPUTE;
    [
        (0, T::SAMPLED_IMAGE, compute),
        (1, T::SAMPLER, compute),
        (2, T::STORAGE_IMAGE, compute),
    ]
}

fn create_pipeline_layout(
    device: &VkDevice,
    set_layout: vk::DescriptorSetLayout,
    push: vk::PushConstantRange,
) -> RenderResult<OwnedPipelineLayout> {
    let layouts = [set_layout];
    device
        .create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&layouts)
                .push_constant_ranges(std::slice::from_ref(&push)),
        )
        .map_err(|e| super::error::map_vk_result(e, "probe prefilter pipeline layout"))
}

// One bake's sets: the mirror-mip copy plus one downsample and one GGX set per
// destination mip past 0.
fn create_pool(device: &VkDevice, steps: usize) -> RenderResult<OwnedDescriptorPool> {
    let steps = steps as u32;
    let sizes = PoolSizes::default()
        .sets(&mip_set(), 1 + steps)
        .sets(&ggx_set(), steps)
        .build();
    device
        .create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&sizes)
                .max_sets(1 + 2 * steps),
        )
        .map_err(|e| super::error::map_vk_result(e, "probe prefilter descriptor pool"))
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

// Layout transition (or, between two equal layouts, a memory barrier) over
// `range` of `image`.
fn transition(
    device: &VkDevice,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    range: vk::ImageSubresourceRange,
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
        .subresource_range(range);
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
