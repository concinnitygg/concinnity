//! The atmosphere for the Vulkan backend (see `concinnity_core::render::atmosphere`):
//! the tables, the compute pass that builds them, the haze pass that blends
//! the aerial volume over the opaque scene, and the sky pipeline that draws
//! the sky-view table behind it. The irradiance cube the last kernel writes is
//! the environment's irradiance cube, and the sky cube the reflection
//! prefilter convolves is its prefiltered cube, so every set that binds the
//! environment's light reads the atmosphere's.
//!
//! The four tables rest in `GENERAL` for good: the kernels write them as
//! storage images and every reader samples them there, so the steps between
//! kernels are memory barriers that name no image. The irradiance cube and the
//! sky cube rest sampled, as every global set declares them, and the sky
//! cube's capture rests sampled for the convolution; each is opened to
//! `GENERAL` for the kernels that write it and closed again inside the pass.
//!
//! Mirrors src/metal/atmosphere.rs and src/directx/atmosphere.rs.

use ash::vk;
use concinnity_core::math::SaturatingFrom;
use concinnity_core::render::atmosphere::{
    AERIAL_SIZE, AirTables, AtmosphereFrame, AtmosphereParams, AtmosphereSettings, AtmosphereView,
    IRRADIANCE_FACE, MULTISCATTER_SIZE, SKY_CUBE_FACE, SKY_CUBE_PLAN, SKY_VIEW_HEIGHT,
    SKY_VIEW_WIDTH, SkyCubeKey, SkyCubeSchedule, SkyCubeWork, TRANSMITTANCE_HEIGHT,
    TRANSMITTANCE_WIDTH, TableReadiness,
};
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::shader_programs::ShaderProgram;

use super::allocator::{PooledBuffer, PooledImage};
use super::builtin_shaders::{
    AERIAL_PERSPECTIVE_FRAG, ATMOSPHERE_AERIAL, ATMOSPHERE_IRRADIANCE, ATMOSPHERE_MULTISCATTER,
    ATMOSPHERE_SKY_CUBE, ATMOSPHERE_SKY_VIEW, ATMOSPHERE_TRANSMITTANCE, CompileProgram,
    FULLSCREEN_VERT, SKY_ATMOSPHERE_FRAG,
};
use super::context::VkContext;
use super::descriptor_layout::{Binding, PoolSizes};
use super::owned::{
    OwnedDescriptorPool, OwnedFramebuffer, OwnedPipeline, OwnedPipelineLayout, OwnedRenderPass,
    OwnedSampler, OwnedSetLayout, VkDevice,
};
use super::pipeline_desc::{Blend, GraphicsPipelineDesc, compute_pipeline};
use super::probe_prefilter::{PrefilterGpu, PrefilterTarget, ProbePrefilterPipelines};
use super::probe_set::{self, CubeImage};
use super::record::Recorder;
use super::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use super::set_writes::SetWrites;
use super::texture::{
    GpuImage, GpuUploadContext, LayoutTransition, SubresourceRange, one_shot_submit,
    transition_image_layout_range,
};

const TABLE_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
// Absolute radiance, which a night sky takes below half float's range: the
// environment maps' irradiance cubes are full float too.
const IRRADIANCE_FORMAT: vk::Format = vk::Format::R32G32B32A32_SFLOAT;
const GROUP: u32 = 8;

// The kernels in dispatch order; each set's output binding is its own table.
const KERNELS: usize = 6;
const TRANSMITTANCE: usize = 0;
const MULTISCATTER: usize = 1;
const SKY_VIEW: usize = 2;
const AERIAL: usize = 3;
const IRRADIANCE: usize = 4;
const SKY_CUBE: usize = 5;

// The stages that sample the environment's cubes.
const CUBE_READERS: vk::PipelineStageFlags = vk::PipelineStageFlags::from_raw(
    vk::PipelineStageFlags::FRAGMENT_SHADER.as_raw()
        | vk::PipelineStageFlags::COMPUTE_SHADER.as_raw(),
);

// One table: its image and the one view every binding uses.
struct Table {
    image: PooledImage,
    view: vk::ImageView,
}

// The render targets the haze pass draws into and reads depth from, one per
// frame slot.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct AtmosphereTargets<'a> {
    pub(in crate::vulkan) hdr_resolve_views: &'a [vk::ImageView],
    pub(in crate::vulkan) depth_views: &'a [vk::ImageView],
    pub(in crate::vulkan) extent: vk::Extent2D,
    pub(in crate::vulkan) msaa: bool,
    pub(in crate::vulkan) hdr_format: vk::Format,
}

// What the sky pipeline is built against: the global set every scene pass
// binds, and the main pass it draws at the tail of.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct AtmosphereSky {
    pub(in crate::vulkan) global_set_layout: vk::DescriptorSetLayout,
    pub(in crate::vulkan) main_render_pass: vk::RenderPass,
    pub(in crate::vulkan) samples: vk::SampleCountFlags,
}

pub(in crate::vulkan) struct VkAtmosphere {
    settings: AtmosphereSettings,
    // This frame's inputs; `None` leaves the passes out of the graph.
    pub(in crate::vulkan) frame: Option<AtmosphereFrame>,
    // The constants built from them for this frame's view.
    pub(in crate::vulkan) params: Option<AtmosphereParams>,
    tables: [Table; 4],
    irradiance_image: vk::Image,
    sky_cube: SkyCube,
    params_ubos: Vec<PooledBuffer>,
    _pool: OwnedDescriptorPool,
    _sampler: OwnedSampler,
    _kernel_set_layout: OwnedSetLayout,
    kernel_layout: OwnedPipelineLayout,
    kernels: [OwnedPipeline; KERNELS],
    // Per frame slot, one set per kernel.
    kernel_sets: Vec<[vk::DescriptorSet; KERNELS]>,
    _sky_set_layout: OwnedSetLayout,
    sky_layout: OwnedPipelineLayout,
    sky_pipeline: OwnedPipeline,
    sky_sets: Vec<vk::DescriptorSet>,
    _aerial_set_layout: OwnedSetLayout,
    aerial_layout: OwnedPipelineLayout,
    aerial_pass: OwnedRenderPass,
    aerial_pipeline: OwnedPipeline,
    aerial_sets: Vec<vk::DescriptorSet>,
    framebuffers: Vec<OwnedFramebuffer>,
    // The ground the tables that depend on the air alone were built over.
    air_tables: AirTables,
    // The frames that recorded the tables, for work recorded outside the frame.
    readiness: TableReadiness,
}

// The sky drawn into a capture cube and convolved by the reflection-probe
// kernels into the environment's prefiltered cube, a few steps a frame.
struct SkyCube {
    image: vk::Image,
    prefilter: ProbePrefilterPipelines,
    gpu: PrefilterGpu,
    schedule: SkyCubeSchedule,
    // This frame's steps, planned when its constants are.
    work: SkyCubeWork,
}

// The kernels' shared set: the constants, the three tables they may read, the
// sampler, and the table the kernel writes.
fn kernel_bindings() -> [Binding; 6] {
    use vk::DescriptorType as T;
    let c = vk::ShaderStageFlags::COMPUTE;
    [
        (0, T::UNIFORM_BUFFER, c),
        (1, T::SAMPLED_IMAGE, c),
        (2, T::SAMPLED_IMAGE, c),
        (3, T::SAMPLED_IMAGE, c),
        (4, T::SAMPLER, c),
        (5, T::STORAGE_IMAGE, c),
    ]
}

// The sky's set 1: the constants, the sky-view table and its sampler.
fn sky_bindings() -> [Binding; 3] {
    use vk::DescriptorType as T;
    let f = vk::ShaderStageFlags::FRAGMENT;
    [
        (0, T::UNIFORM_BUFFER, f),
        (1, T::SAMPLED_IMAGE, f),
        (2, T::SAMPLER, f),
    ]
}

// The haze pass's set: the constants, the scene depth, the aerial volume and
// its sampler.
fn aerial_bindings() -> [Binding; 4] {
    use vk::DescriptorType as T;
    let f = vk::ShaderStageFlags::FRAGMENT;
    [
        (0, T::UNIFORM_BUFFER, f),
        (1, T::SAMPLED_IMAGE, f),
        (2, T::SAMPLED_IMAGE, f),
        (3, T::SAMPLER, f),
    ]
}

fn create_table(gpu: &GpuUploadContext<'_>, extent: [u32; 3], label: &str) -> RenderResult<Table> {
    let volume = extent[2] > 1;
    let info = vk::ImageCreateInfo::default()
        .image_type(if volume {
            vk::ImageType::TYPE_3D
        } else {
            vk::ImageType::TYPE_2D
        })
        .extent(vk::Extent3D {
            width: extent[0],
            height: extent[1],
            depth: extent[2],
        })
        .mip_levels(1)
        .array_layers(1)
        .format(TABLE_FORMAT)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::TYPE_1);
    let image = gpu
        .alloc
        .create_image(&info, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .map_err(|e| e.context(format_args!("atmosphere {label} table")))?;
    let view = image_view(
        gpu.device,
        image.image(),
        if volume {
            vk::ImageViewType::TYPE_3D
        } else {
            vk::ImageViewType::TYPE_2D
        },
        TABLE_FORMAT,
        1,
    )?;
    image.attach_view(view);
    Ok(Table { image, view })
}

fn image_view(
    device: &VkDevice,
    image: vk::Image,
    view_type: vk::ImageViewType,
    format: vk::Format,
    layers: u32,
) -> RenderResult<vk::ImageView> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(view_type)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: layers,
        });
    // SAFETY: the create-info is live for the call and names an image of this device.
    unsafe { device.create_image_view(&info, None) }
        .map_err(|e| super::error::map_vk_result(e, "atmosphere table view"))
}

// The cubes an atmosphere writes, which the caller installs as the
// environment's before any set binds them, both resting sampled once the
// atmosphere is built: the irradiance cube and the view of its six faces its
// kernel writes through, and the sky cube with a storage view per mip.
pub(in crate::vulkan) struct AtmosphereCubes {
    irradiance_image: vk::Image,
    irradiance_faces: vk::ImageView,
    sky_image: vk::Image,
    sky_mip_views: Vec<vk::ImageView>,
}

/// The environment images an atmosphere replaces.
pub(in crate::vulkan) struct EnvironmentCubes {
    pub(in crate::vulkan) irradiance: GpuImage,
    pub(in crate::vulkan) prefilter: GpuImage,
}

pub(in crate::vulkan) fn create_atmosphere_cubes(
    gpu: &GpuUploadContext<'_>,
) -> RenderResult<(EnvironmentCubes, AtmosphereCubes)> {
    let (irradiance, image, faces) = create_irradiance_cube(gpu)?;
    let mips = SKY_CUBE_PLAN.mips();
    let sky = probe_set::create_image(
        gpu.alloc,
        CubeImage {
            face_size: SKY_CUBE_FACE,
            mips,
            layers: 6,
            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
        },
    )
    .map_err(|e| e.context("atmosphere sky cube"))?;
    let sky_mip_views = probe_set::mip_storage_views(gpu.device, &sky, 0, mips)?;
    let sky_view = probe_set::create_view(
        gpu.device,
        sky.image(),
        vk::ImageViewType::CUBE,
        probe_set::range(0, mips, 0, 6),
    )?;
    let sky_image = sky.image();
    Ok((
        EnvironmentCubes {
            irradiance,
            prefilter: GpuImage::from_pooled(sky, sky_view),
        },
        AtmosphereCubes {
            irradiance_image: image,
            irradiance_faces: faces,
            sky_image,
            sky_mip_views,
        },
    ))
}

fn create_irradiance_cube(
    gpu: &GpuUploadContext<'_>,
) -> RenderResult<(GpuImage, vk::Image, vk::ImageView)> {
    let info = vk::ImageCreateInfo::default()
        .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE)
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width: IRRADIANCE_FACE,
            height: IRRADIANCE_FACE,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(6)
        .format(IRRADIANCE_FORMAT)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::TYPE_1);
    let pooled = gpu
        .alloc
        .create_image(&info, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .map_err(|e| e.context("atmosphere irradiance cube"))?;
    let faces = image_view(
        gpu.device,
        pooled.image(),
        vk::ImageViewType::TYPE_2D_ARRAY,
        IRRADIANCE_FORMAT,
        6,
    )?;
    pooled.attach_view(faces);
    let cube = image_view(
        gpu.device,
        pooled.image(),
        vk::ImageViewType::CUBE,
        IRRADIANCE_FORMAT,
        6,
    )?;
    let image = pooled.image();
    Ok((GpuImage::from_pooled(pooled, cube), image, faces))
}

fn create_sampler(device: &VkDevice) -> RenderResult<OwnedSampler> {
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
    device
        .create_sampler(&info)
        .map_err(|e| super::error::map_vk_result(e, "atmosphere sampler"))
}

fn pipeline_layout(
    device: &VkDevice,
    sets: &[vk::DescriptorSetLayout],
    label: &str,
) -> RenderResult<OwnedPipelineLayout> {
    device
        .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(sets))
        .map_err(|e| super::error::map_vk_result(e, label))
}

// The haze pass's render pass: the resolved scene, loaded and stored, resting
// sampled on either side the way the fog pass leaves it.
fn create_aerial_pass(device: &VkDevice, format: vk::Format) -> RenderResult<OwnedRenderPass> {
    let attachment = vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let color_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_ref));
    let dep_in = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::FRAGMENT_SHADER,
        )
        .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::COLOR_ATTACHMENT_READ,
        );
    let dep_out = vk::SubpassDependency::default()
        .src_subpass(0)
        .dst_subpass(vk::SUBPASS_EXTERNAL)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
    let deps = [dep_in, dep_out];
    let info = vk::RenderPassCreateInfo::default()
        .attachments(std::slice::from_ref(&attachment))
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(&deps);
    device
        .create_render_pass(&info)
        .map_err(|e| super::error::map_vk_result(e, "aerial perspective render pass"))
}

fn create_framebuffers(
    device: &VkDevice,
    render_pass: vk::RenderPass,
    views: &[vk::ImageView],
    frames: usize,
    extent: vk::Extent2D,
) -> RenderResult<Vec<OwnedFramebuffer>> {
    views
        .iter()
        .take(frames)
        .map(|&view| {
            let attachments = [view];
            let info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(extent.width.max(1))
                .height(extent.height.max(1))
                .layers(1);
            device
                .create_framebuffer(&info)
                .map_err(|e| super::error::map_vk_result(e, "aerial perspective framebuffer"))
        })
        .collect()
}

fn compile_kernel(
    device: &VkDevice,
    layout: vk::PipelineLayout,
    program: &ShaderProgram,
    hot_reload: bool,
) -> RenderResult<OwnedPipeline> {
    compute_pipeline(device, layout, &program.compile(hot_reload)?, program.label)
}

impl VkAtmosphere {
    // The tables, pipelines and sets, writing `cubes`.
    pub(in crate::vulkan) fn new(
        gpu: &GpuUploadContext<'_>,
        settings: &AtmosphereSettings,
        cubes: AtmosphereCubes,
        frames: usize,
        targets: AtmosphereTargets<'_>,
        sky: AtmosphereSky,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let device = gpu.device;
        let tables = [
            create_table(
                gpu,
                [TRANSMITTANCE_WIDTH, TRANSMITTANCE_HEIGHT, 1],
                "transmittance",
            )?,
            create_table(
                gpu,
                [MULTISCATTER_SIZE, MULTISCATTER_SIZE, 1],
                "multiple-scattering",
            )?,
            create_table(gpu, [SKY_VIEW_WIDTH, SKY_VIEW_HEIGHT, 1], "sky-view")?,
            create_table(
                gpu,
                [AERIAL_SIZE, AERIAL_SIZE, AERIAL_SIZE],
                "aerial-perspective",
            )?,
        ];
        let AtmosphereCubes {
            irradiance_image,
            irradiance_faces,
            sky_image,
            sky_mip_views,
        } = cubes;
        let prefilter = ProbePrefilterPipelines::new(device, hot_reload)?;
        let sky_gpu = PrefilterGpu::new(
            device,
            gpu.alloc,
            &prefilter,
            &SKY_CUBE_PLAN,
            PrefilterTarget {
                image: sky_image,
                cube: 0,
                mip_views: &sky_mip_views,
            },
        )?;
        one_shot_submit(device, gpu.command_pool, gpu.queue, |cmd| {
            // The sky cube and its capture rest sampled from the start, the
            // state every frame's steps open them from.
            for image in [sky_image, sky_gpu.capture_image()] {
                transition_image_layout_range(
                    device,
                    cmd,
                    image,
                    LayoutTransition {
                        old_layout: vk::ImageLayout::UNDEFINED,
                        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        aspect: vk::ImageAspectFlags::COLOR,
                    },
                    SubresourceRange {
                        base_layer: 0,
                        layer_count: 6,
                        base_mip: 0,
                        mip_count: SKY_CUBE_PLAN.mips(),
                    },
                );
            }
            for table in &tables {
                transition_image_layout_range(
                    device,
                    cmd,
                    table.image.image(),
                    LayoutTransition {
                        old_layout: vk::ImageLayout::UNDEFINED,
                        new_layout: vk::ImageLayout::GENERAL,
                        aspect: vk::ImageAspectFlags::COLOR,
                    },
                    SubresourceRange {
                        base_layer: 0,
                        layer_count: 1,
                        base_mip: 0,
                        mip_count: 1,
                    },
                );
            }
            transition_image_layout_range(
                device,
                cmd,
                irradiance_image,
                LayoutTransition {
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    aspect: vk::ImageAspectFlags::COLOR,
                },
                SubresourceRange {
                    base_layer: 0,
                    layer_count: 6,
                    base_mip: 0,
                    mip_count: 1,
                },
            );
        })?;

        let sampler = create_sampler(device)?;
        let params_ubos = (0..frames)
            .map(|_| {
                gpu.alloc.create_buffer(
                    size_of::<AtmosphereParams>() as u64,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
            })
            .collect::<RenderResult<Vec<_>>>()?;

        let kernel_set_layout = create_descriptor_set_layout(device, &kernel_bindings())?;
        let sky_set_layout = create_descriptor_set_layout(device, &sky_bindings())?;
        let aerial_set_layout = create_descriptor_set_layout(device, &aerial_bindings())?;
        let f = u32::saturating_from(frames);
        let sizes = PoolSizes::default()
            .sets(&kernel_bindings(), f * u32::saturating_from(KERNELS))
            .sets(&sky_bindings(), f)
            .sets(&aerial_bindings(), f)
            .build();
        let pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(f * (u32::saturating_from(KERNELS) + 2))
                    .pool_sizes(&sizes),
            )
            .map_err(|e| super::error::map_vk_result(e, "atmosphere descriptor pool"))?;

        let kernel_layout = pipeline_layout(
            device,
            &[kernel_set_layout.handle()],
            "atmosphere kernel layout",
        )?;
        let kernel_layout_handle = kernel_layout.handle();
        let kernels = [
            compile_kernel(
                device,
                kernel_layout_handle,
                &ATMOSPHERE_TRANSMITTANCE,
                hot_reload,
            )?,
            compile_kernel(
                device,
                kernel_layout_handle,
                &ATMOSPHERE_MULTISCATTER,
                hot_reload,
            )?,
            compile_kernel(
                device,
                kernel_layout_handle,
                &ATMOSPHERE_SKY_VIEW,
                hot_reload,
            )?,
            compile_kernel(device, kernel_layout_handle, &ATMOSPHERE_AERIAL, hot_reload)?,
            compile_kernel(
                device,
                kernel_layout_handle,
                &ATMOSPHERE_IRRADIANCE,
                hot_reload,
            )?,
            compile_kernel(
                device,
                kernel_layout_handle,
                &ATMOSPHERE_SKY_CUBE,
                hot_reload,
            )?,
        ];
        let outputs = [
            tables[0].view,
            tables[1].view,
            tables[2].view,
            tables[3].view,
            irradiance_faces,
            sky_gpu.capture_top_view(),
        ];
        let params_size = size_of::<AtmosphereParams>() as u64;
        let mut kernel_sets = Vec::with_capacity(frames);
        for ubo in &params_ubos {
            let layouts = [kernel_set_layout.handle(); KERNELS];
            let sets = alloc_descriptor_sets(device, pool.handle(), &layouts)?;
            let mut slot = [vk::DescriptorSet::null(); KERNELS];
            for (k, &set) in sets.iter().enumerate() {
                SetWrites::new(set)
                    .uniform_buffer(0, ubo.buffer(), params_size)
                    .sampled_image_in(1, tables[0].view, vk::ImageLayout::GENERAL)
                    .sampled_image_in(2, tables[1].view, vk::ImageLayout::GENERAL)
                    .sampled_image_in(3, tables[2].view, vk::ImageLayout::GENERAL)
                    .sampler(4, sampler.handle())
                    .storage_image(5, outputs[k])
                    .apply(device);
                slot[k] = set;
            }
            kernel_sets.push(slot);
        }

        let sky_layout = pipeline_layout(
            device,
            &[sky.global_set_layout, sky_set_layout.handle()],
            "atmosphere sky layout",
        )?;
        let sky_pipeline = super::sky::build_sky_pipeline(
            device,
            &SKY_ATMOSPHERE_FRAG,
            sky_layout.handle(),
            sky.main_render_pass,
            sky.samples,
            hot_reload,
        )?;
        let sky_layouts = vec![sky_set_layout.handle(); frames];
        let sky_sets = alloc_descriptor_sets(device, pool.handle(), &sky_layouts)?;
        for (set, ubo) in sky_sets.iter().zip(&params_ubos) {
            SetWrites::new(*set)
                .uniform_buffer(0, ubo.buffer(), params_size)
                .sampled_image_in(1, tables[2].view, vk::ImageLayout::GENERAL)
                .sampler(2, sampler.handle())
                .apply(device);
        }

        let aerial_layout = pipeline_layout(
            device,
            &[aerial_set_layout.handle()],
            "aerial perspective layout",
        )?;
        let aerial_pass = create_aerial_pass(device, targets.hdr_format)?;
        let aerial_pipeline = build_aerial_pipeline(
            device,
            aerial_layout.handle(),
            aerial_pass.handle(),
            targets.msaa,
            hot_reload,
        )?;
        let aerial_layouts = vec![aerial_set_layout.handle(); frames];
        let aerial_sets = alloc_descriptor_sets(device, pool.handle(), &aerial_layouts)?;
        let last_depth = targets.depth_views.len().saturating_sub(1);
        for (i, (set, ubo)) in aerial_sets.iter().zip(&params_ubos).enumerate() {
            SetWrites::new(*set)
                .uniform_buffer(0, ubo.buffer(), params_size)
                .sampled_image(1, targets.depth_views[i.min(last_depth)])
                .sampled_image_in(2, tables[3].view, vk::ImageLayout::GENERAL)
                .sampler(3, sampler.handle())
                .apply(device);
        }
        let framebuffers = create_framebuffers(
            device,
            aerial_pass.handle(),
            targets.hdr_resolve_views,
            frames,
            targets.extent,
        )?;

        Ok(Self {
            settings: *settings,
            frame: None,
            params: None,
            tables,
            irradiance_image,
            sky_cube: SkyCube {
                image: sky_image,
                prefilter,
                gpu: sky_gpu,
                schedule: SkyCubeSchedule::new(SKY_CUBE_PLAN.mips()),
                work: SkyCubeWork::default(),
            },
            params_ubos,
            _pool: pool,
            _sampler: sampler,
            _kernel_set_layout: kernel_set_layout,
            kernel_layout,
            kernels,
            kernel_sets,
            _sky_set_layout: sky_set_layout,
            sky_layout,
            sky_pipeline,
            sky_sets,
            _aerial_set_layout: aerial_set_layout,
            aerial_layout,
            aerial_pass,
            aerial_pipeline,
            aerial_sets,
            framebuffers,
            air_tables: AirTables::default(),
            readiness: TableReadiness::default(),
        })
    }

    // Re-point the haze pass at the rebuilt scene targets after a resize.
    pub(in crate::vulkan) fn rebuild(
        &mut self,
        device: &VkDevice,
        hdr_resolve_views: &[vk::ImageView],
        depth_views: &[vk::ImageView],
        extent: vk::Extent2D,
    ) -> RenderResult<()> {
        self.framebuffers = create_framebuffers(
            device,
            self.aerial_pass.handle(),
            hdr_resolve_views,
            self.params_ubos.len(),
            extent,
        )?;
        let last_depth = depth_views.len().saturating_sub(1);
        for (i, &set) in self.aerial_sets.iter().enumerate() {
            SetWrites::new(set)
                .sampled_image(1, depth_views[i.min(last_depth)])
                .apply(device);
        }
        Ok(())
    }

    // Release the framebuffers and the pooled resources ahead of the device.
    pub(in crate::vulkan) fn destroy(&mut self) {
        self.framebuffers.clear();
        self.params_ubos.clear();
        for table in &mut self.tables {
            table.image = PooledImage::null();
        }
    }

    // Rebuild every pipeline from the shader sources (shader hot reload).
    pub(in crate::vulkan) fn rebuild_pipelines(
        &mut self,
        device: &VkDevice,
        sky: AtmosphereSky,
        msaa: bool,
    ) -> RenderResult<()> {
        let layout = self.kernel_layout.handle();
        let kernels = [
            compile_kernel(device, layout, &ATMOSPHERE_TRANSMITTANCE, true)?,
            compile_kernel(device, layout, &ATMOSPHERE_MULTISCATTER, true)?,
            compile_kernel(device, layout, &ATMOSPHERE_SKY_VIEW, true)?,
            compile_kernel(device, layout, &ATMOSPHERE_AERIAL, true)?,
            compile_kernel(device, layout, &ATMOSPHERE_IRRADIANCE, true)?,
            compile_kernel(device, layout, &ATMOSPHERE_SKY_CUBE, true)?,
        ];
        let sky_pipeline = super::sky::build_sky_pipeline(
            device,
            &SKY_ATMOSPHERE_FRAG,
            self.sky_layout.handle(),
            sky.main_render_pass,
            sky.samples,
            true,
        )?;
        let aerial_pipeline = build_aerial_pipeline(
            device,
            self.aerial_layout.handle(),
            self.aerial_pass.handle(),
            msaa,
            true,
        )?;
        self.kernels = kernels;
        self.sky_pipeline = sky_pipeline;
        self.aerial_pipeline = aerial_pipeline;
        self.air_tables.invalidate();
        self.sky_cube.schedule.reset();
        Ok(())
    }
}

// `(in-scatter, 1 - T)` over the resolved scene -> `scene * T + in-scatter`.
fn build_aerial_pipeline(
    device: &VkDevice,
    layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    msaa: bool,
    hot_reload: bool,
) -> RenderResult<OwnedPipeline> {
    let vert = FULLSCREEN_VERT.compile(hot_reload)?;
    let frag = AERIAL_PERSPECTIVE_FRAG.at(msaa).compile(hot_reload)?;
    GraphicsPipelineDesc::fullscreen(
        &vert,
        &frag,
        layout,
        render_pass,
        &[Blend::PremultipliedOver],
    )
    .build(device, "aerial perspective")
}

// A compute-to-compute step inside the pass: the next kernel samples what the
// last one wrote.
fn kernel_step(rec: &Recorder<'_>) {
    rec.pipeline_barrier(
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        &[vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)],
        &[],
        &[],
    );
}

// A layout transition over `range` of `image` between sampled and GENERAL,
// ordering the sampling reads against the kernel writes.
fn image_barrier(
    image: vk::Image,
    range: vk::ImageSubresourceRange,
    from: vk::ImageLayout,
    to: vk::ImageLayout,
) -> vk::ImageMemoryBarrier<'static> {
    cube_barrier(image, from, to).subresource_range(range)
}

fn cube_barrier(
    image: vk::Image,
    from: vk::ImageLayout,
    to: vk::ImageLayout,
) -> vk::ImageMemoryBarrier<'static> {
    vk::ImageMemoryBarrier::default()
        .old_layout(from)
        .new_layout(to)
        .src_access_mask(if from == vk::ImageLayout::GENERAL {
            vk::AccessFlags::SHADER_WRITE
        } else {
            vk::AccessFlags::SHADER_READ
        })
        .dst_access_mask(if to == vk::ImageLayout::GENERAL {
            vk::AccessFlags::SHADER_WRITE
        } else {
            vk::AccessFlags::SHADER_READ
        })
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 6,
        })
}

impl VkContext {
    // This frame's atmosphere constants for `view`, written into the frame
    // slot's ring.
    pub(in crate::vulkan) fn prepare_atmosphere(&mut self, frame_idx: usize, view: AtmosphereView) {
        if let Some(atmo) = self.atmosphere.as_mut() {
            atmo.params = atmo
                .frame
                .map(|f| AtmosphereParams::new(&atmo.settings, &f, &view));
            atmo.sky_cube.work = match atmo.frame {
                Some(f) => {
                    atmo.sky_cube
                        .schedule
                        .plan(SkyCubeKey::new(&atmo.settings, &f, &view.sky_rot))
                }
                None => SkyCubeWork::default(),
            };
            if let (Some(params), Some(ubo)) = (atmo.params, atmo.params_ubos.get(frame_idx)) {
                ubo.write_val(0, &params);
            }
        }
    }

    // Build this frame's tables: the sky view, the aerial volume and the
    // irradiance cube, after the transmittance and multiple-scattering tables
    // when the ground they are built over has changed; then this frame's steps
    // of the sky cube.
    pub(in crate::vulkan) fn encode_atmosphere(
        &self,
        rec: &Recorder<'_>,
        frame_idx: usize,
    ) -> RenderResult<()> {
        let Some(atmo) = self.atmosphere.as_ref() else {
            return Ok(());
        };
        let Some(sets) = atmo.kernel_sets.get(frame_idx) else {
            return Ok(());
        };
        let dispatch = |k: usize, groups: [u32; 3]| {
            rec.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &atmo.kernels[k]);
            rec.bind_descriptor_sets(
                vk::PipelineBindPoint::COMPUTE,
                &atmo.kernel_layout,
                0,
                std::slice::from_ref(&sets[k]),
                &[],
            );
            rec.dispatch(groups[0], groups[1], groups[2]);
        };
        // The last frame's sky, haze and ambient reads finish before the
        // tables are rewritten.
        rec.pipeline_barrier(
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            &[vk::MemoryBarrier::default()],
            &[],
            &[],
        );
        atmo.readiness.note_recorded();
        let ground = atmo.params.map_or(0.0, |p| p.radii[0]);
        if atmo.air_tables.take_rebuild(ground) {
            dispatch(
                TRANSMITTANCE,
                [
                    TRANSMITTANCE_WIDTH.div_ceil(GROUP),
                    TRANSMITTANCE_HEIGHT.div_ceil(GROUP),
                    1,
                ],
            );
            kernel_step(rec);
            dispatch(MULTISCATTER, [MULTISCATTER_SIZE, MULTISCATTER_SIZE, 1]);
            kernel_step(rec);
        }
        dispatch(
            SKY_VIEW,
            [
                SKY_VIEW_WIDTH.div_ceil(GROUP),
                SKY_VIEW_HEIGHT.div_ceil(GROUP),
                1,
            ],
        );
        dispatch(
            AERIAL,
            [AERIAL_SIZE.div_ceil(GROUP), AERIAL_SIZE.div_ceil(GROUP), 1],
        );
        // The irradiance kernel reads the sky view; the cube opens for writing.
        rec.pipeline_barrier(
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            &[vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)],
            &[],
            &[cube_barrier(
                atmo.irradiance_image,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::GENERAL,
            )],
        );
        dispatch(IRRADIANCE, [1, 1, 1]);
        let cube = &atmo.sky_cube;
        let work = cube.work;
        let sky_range = probe_set::range(0, SKY_CUBE_PLAN.mips(), 0, 6);
        let open = |image| {
            image_barrier(
                image,
                sky_range,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::GENERAL,
            )
        };
        if work.capture {
            // The sky is drawn into the capture's top mip, then copied and
            // reduced; last frame's samplers of the cube and the capture
            // finish first.
            rec.pipeline_barrier(
                CUBE_READERS,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                &[],
                &[],
                &[open(cube.gpu.capture_image()), open(cube.image)],
            );
            dispatch(
                SKY_CUBE,
                [
                    SKY_CUBE_FACE.div_ceil(GROUP),
                    SKY_CUBE_FACE.div_ceil(GROUP),
                    6,
                ],
            );
            kernel_step(rec);
            self.record_prefilter_pyramid(rec.raw(), &cube.prefilter, &cube.gpu, &SKY_CUBE_PLAN);
            rec.pipeline_barrier(
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                &[],
                &[],
                &[image_barrier(
                    cube.gpu.capture_image(),
                    cube.gpu.capture_range(),
                    vk::ImageLayout::GENERAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                )],
            );
        } else if !work.is_empty() {
            rec.pipeline_barrier(
                CUBE_READERS,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                &[],
                &[],
                &[open(cube.image)],
            );
        }
        for mip in work.ggx_mips() {
            self.record_prefilter_ggx(rec.raw(), &cube.prefilter, &cube.gpu, &SKY_CUBE_PLAN, mip)?;
        }
        // Every table is read by fragment shaders from here on: the sky, the
        // haze, and the surfaces' ambient light and reflections through the
        // closed cubes.
        let irradiance = cube_barrier(
            atmo.irradiance_image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
        let sky = image_barrier(
            cube.image,
            sky_range,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
        let closing = [irradiance, sky];
        let closed = if work.is_empty() { 1 } else { 2 };
        rec.pipeline_barrier(
            vk::PipelineStageFlags::COMPUTE_SHADER,
            CUBE_READERS,
            &[vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)],
            &[],
            &closing[..closed],
        );
        Ok(())
    }

    // Blend the air in front of each opaque surface over the resolved scene.
    pub(in crate::vulkan) fn encode_aerial_perspective(
        &self,
        rec: &Recorder<'_>,
        frame_idx: usize,
    ) {
        let Some(atmo) = self.atmosphere.as_ref() else {
            return;
        };
        let (Some(framebuffer), Some(set)) = (
            atmo.framebuffers.get(frame_idx),
            atmo.aerial_sets.get(frame_idx),
        ) else {
            return;
        };
        let extent = self.targets.render_extent;
        rec.begin_render_pass(
            &atmo.aerial_pass,
            framebuffer,
            vk::Rect2D::default().extent(extent),
            &[],
        );
        rec.set_viewport(&vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        });
        rec.set_scissor(&vk::Rect2D::default().extent(extent));
        rec.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, &atmo.aerial_pipeline);
        rec.bind_descriptor_sets(
            vk::PipelineBindPoint::GRAPHICS,
            &atmo.aerial_layout,
            0,
            std::slice::from_ref(set),
            &[],
        );
        rec.draw(3, 1, 0, 0);
        rec.end_render_pass();
    }

    // The atmosphere's sky pipeline, its layout and this frame slot's set, once
    // a frame has carried its inputs.
    pub(in crate::vulkan) fn atmosphere_sky(
        &self,
        frame_idx: usize,
    ) -> Option<(vk::Pipeline, vk::PipelineLayout, vk::DescriptorSet)> {
        let atmo = self.atmosphere.as_ref()?;
        atmo.params?;
        let set = *atmo.sky_sets.get(frame_idx)?;
        Some((atmo.sky_pipeline.handle(), atmo.sky_layout.handle(), set))
    }

    // The view block's `irradiance` lane.
    pub(in crate::vulkan) fn irradiance_lane(&self) -> f32 {
        concinnity_core::render::sky::irradiance_lane(
            self.scene.prefilter_mip_count > 0,
            self.atmosphere.is_some(),
        )
    }

    // Whether work recorded outside the frame graph, such as a probe face,
    // reads tables this atmosphere has written. Always true without one.
    pub(in crate::vulkan) fn atmosphere_tables_ready(&self) -> bool {
        self.atmosphere
            .as_ref()
            .is_none_or(|atmo| atmo.readiness.ready(self.frames_in_flight))
    }
}
