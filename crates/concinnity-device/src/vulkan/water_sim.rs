//! The simulated ocean on Vulkan (see `concinnity_engine::water::ocean`):
//! the cascade arrays the water draw samples, and the `WaterSim` node that
//! fills them each frame from the shared `water_fft.hlsl` kernels.
//!
//! Both arrays rest in `GENERAL` for good, like the atmosphere's tables: the
//! kernels write them as storage images and the water draw samples them in
//! that layout, so every step is a memory barrier that names no image. The
//! node orders itself against the last frame's water draw and foam step at its
//! head and the next draw against its writes at its tail. The foam buffer is
//! the one resource that carries state from frame to frame.

use ash::vk;
use concinnity_core::math::SaturatingFrom;
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::shader::programs::ShaderProgram;
use concinnity_engine::render::uniforms::ocean::{
    OceanFftParams, ocean_mip_groups, ocean_mip_levels,
};
use concinnity_engine::water::ocean::OceanPlan;
use concinnity_engine::water::ocean::foam::FoamClock;
use concinnity_engine::water::ocean::initial::initial_spectrum;
use std::cell::Cell;

use super::allocator::{PooledBuffer, PooledImage};
use super::builtin_shaders::{self, CompileProgram};
use super::context::VkContext;
use super::descriptor_layout::{Binding, PoolSizes};
use super::owned::{
    OwnedDescriptorPool, OwnedPipeline, OwnedPipelineLayout, OwnedSampler, OwnedSetLayout, VkDevice,
};
use super::pipeline_desc::compute_pipeline;
use super::record::Recorder;
use super::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use super::set_writes::SetWrites;
use super::texture::{
    GpuUploadContext, LayoutTransition, SubresourceRange, one_shot_submit,
    transition_image_layout_range,
};

const CASCADE_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

// The kernels' set: the spectrum, the row transforms, the source mip of each
// array (read-only), the destination mip of each, and the foam.
fn kernel_bindings() -> [Binding; 7] {
    use vk::DescriptorType as T;
    let c = vk::ShaderStageFlags::COMPUTE;
    [
        (0, T::STORAGE_BUFFER, c),
        (1, T::STORAGE_BUFFER, c),
        (2, T::SAMPLED_IMAGE, c),
        (3, T::SAMPLED_IMAGE, c),
        (4, T::STORAGE_IMAGE, c),
        (5, T::STORAGE_IMAGE, c),
        (6, T::STORAGE_BUFFER, c),
    ]
}

// What the water draw binds: both arrays over every mip and layer, in
// `GENERAL`, and the sampler they are read through.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct OceanViews {
    pub displacement: vk::ImageView,
    pub slopes: vk::ImageView,
    pub sampler: vk::Sampler,
}

// One cascade array and its views: the sampled one over every mip, and one
// view per mip, which the kernels write as storage and read as sampled.
struct CascadeArray {
    image: PooledImage,
    sampled: vk::ImageView,
    mips: Vec<vk::ImageView>,
}

// The world's simulated oceans, or a stand-in pair of arrays for a world
// whose water is all summed, which the water draw binds and never samples.
pub(in crate::vulkan) struct OceanSim {
    displacement: CascadeArray,
    slopes: CascadeArray,
    sampler: OwnedSampler,
    kernels: Option<OceanKernels>,
}

struct OceanKernels {
    plan: OceanPlan,
    rows: OwnedPipeline,
    columns: OwnedPipeline,
    // The columns kernel that also steps the foam, for a whitecapping sea.
    columns_foam: OwnedPipeline,
    mip: OwnedPipeline,
    layout: OwnedPipelineLayout,
    _set_layout: OwnedSetLayout,
    _pool: OwnedDescriptorPool,
    // One set per mip level; set 0 also serves both FFT kernels.
    sets: Vec<vk::DescriptorSet>,
    _spectrum: PooledBuffer,
    _rows: PooledBuffer,
    _foam: PooledBuffer,
    foam_clock: Cell<FoamClock>,
}

impl OceanSim {
    pub(in crate::vulkan) fn new(
        gpu: &GpuUploadContext<'_>,
        plan: &OceanPlan,
        max_anisotropy: f32,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let device = gpu.device;
        let (size, layers, mips) = if plan.is_empty() {
            (1, 1, 1)
        } else {
            let size = plan.quality.size;
            (size, plan.layer_count(), ocean_mip_levels(size))
        };
        let displacement = cascade_array(gpu, size, layers, mips, "displacement")?;
        let slopes = cascade_array(gpu, size, layers, mips, "slopes")?;
        one_shot_submit(device, gpu.command_pool, gpu.queue, |cmd| {
            for array in [&displacement, &slopes] {
                transition_image_layout_range(
                    device,
                    cmd,
                    array.image.image(),
                    LayoutTransition {
                        old_layout: vk::ImageLayout::UNDEFINED,
                        new_layout: vk::ImageLayout::GENERAL,
                        aspect: vk::ImageAspectFlags::COLOR,
                    },
                    SubresourceRange {
                        base_layer: 0,
                        layer_count: layers,
                        base_mip: 0,
                        mip_count: mips,
                    },
                );
            }
        })?;
        let sampler = create_sampler(device, max_anisotropy)?;
        let kernels = if plan.is_empty() {
            None
        } else {
            Some(OceanKernels::new(
                gpu,
                plan,
                [&displacement, &slopes],
                hot_reload,
            )?)
        };
        Ok(Self {
            displacement,
            slopes,
            sampler,
            kernels,
        })
    }

    pub(in crate::vulkan) fn views(&self) -> OceanViews {
        OceanViews {
            displacement: self.displacement.sampled,
            slopes: self.slopes.sampled,
            sampler: self.sampler.handle(),
        }
    }

    // Whether any body simulates an ocean, which puts `WaterSim` in the graph.
    pub(in crate::vulkan) fn simulates(&self) -> bool {
        self.kernels.is_some()
    }
}

impl OceanKernels {
    fn new(
        gpu: &GpuUploadContext<'_>,
        plan: &OceanPlan,
        [displacement, slopes]: [&CascadeArray; 2],
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let device = gpu.device;
        let texels: Vec<[f32; 4]> = plan
            .simulations
            .iter()
            .flat_map(|sim| {
                (0..sim.cascades.count())
                    .flat_map(move |c| initial_spectrum(&sim.sea, &sim.cascades, c))
            })
            .collect();
        let spectrum_bytes: &[u8] = bytemuck::cast_slice(&texels);
        let spectrum = super::decal::upload_static_buffer(
            gpu.alloc,
            device,
            gpu.command_pool,
            gpu.queue,
            spectrum_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let rows = gpu.alloc.create_buffer(
            2 * spectrum_bytes.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let foam_bytes = (texels.len() * size_of::<f32>()) as u64;
        let foam = gpu.alloc.create_buffer(
            foam_bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let set_layout = create_descriptor_set_layout(device, &kernel_bindings())?;
        let push = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(u32::saturating_from(size_of::<OceanFftParams>()));
        let set_layouts = [set_layout.handle()];
        let layout = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(std::slice::from_ref(&push)),
            )
            .map_err(|e| super::error::map_vk_result(e, "ocean pipeline layout"))?;
        let kernel = |program: &ShaderProgram, label: &str| {
            compute_pipeline(
                device,
                layout.handle(),
                &program.compile(hot_reload)?,
                label,
            )
        };
        let rows_kernel = kernel(&builtin_shaders::WATER_FFT_ROWS, "ocean rows")?;
        let columns = kernel(&builtin_shaders::WATER_FFT_COLUMNS, "ocean columns")?;
        let columns_foam = kernel(
            &builtin_shaders::WATER_FFT_COLUMNS_FOAM,
            "ocean columns with foam",
        )?;
        let mip = kernel(&builtin_shaders::WATER_MIP, "ocean mips")?;

        let levels = displacement.mips.len();
        let sizes = PoolSizes::default()
            .sets(&kernel_bindings(), u32::saturating_from(levels))
            .build();
        let pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(u32::saturating_from(levels))
                    .pool_sizes(&sizes),
            )
            .map_err(|e| super::error::map_vk_result(e, "ocean descriptor pool"))?;
        let set_layouts = vec![set_layout.handle(); levels];
        let sets = alloc_descriptor_sets(device, pool.handle(), &set_layouts)?;
        for (level, &set) in sets.iter().enumerate() {
            let src = level.saturating_sub(1);
            SetWrites::new(set)
                .storage_buffer(0, spectrum.buffer(), spectrum_bytes.len() as u64)
                .storage_buffer(1, rows.buffer(), 2 * spectrum_bytes.len() as u64)
                .sampled_image_in(2, displacement.mips[src], vk::ImageLayout::GENERAL)
                .sampled_image_in(3, slopes.mips[src], vk::ImageLayout::GENERAL)
                .storage_image(4, displacement.mips[level])
                .storage_image(5, slopes.mips[level])
                .storage_buffer(6, foam.buffer(), foam_bytes)
                .apply(device);
        }
        Ok(Self {
            plan: plan.clone(),
            rows: rows_kernel,
            columns,
            columns_foam,
            mip,
            layout,
            _set_layout: set_layout,
            _pool: pool,
            sets,
            _spectrum: spectrum,
            _rows: rows,
            _foam: foam,
            foam_clock: Cell::new(FoamClock::default()),
        })
    }
}

// A `layers`-deep array of `size`-square cascades with `mips` levels.
fn cascade_array(
    gpu: &GpuUploadContext<'_>,
    size: u32,
    layers: u32,
    mips: u32,
    label: &str,
) -> RenderResult<CascadeArray> {
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width: size,
            height: size,
            depth: 1,
        })
        .mip_levels(mips)
        .array_layers(layers)
        .format(CASCADE_FORMAT)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::TYPE_1);
    let image = gpu
        .alloc
        .create_image(&info, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .map_err(|e| e.context(format_args!("ocean {label} array")))?;
    let sampled = array_view(gpu.device, image.image(), 0, mips, layers)?;
    image.attach_view(sampled);
    let mut views = Vec::with_capacity(mips as usize);
    for level in 0..mips {
        let view = array_view(gpu.device, image.image(), level, 1, layers)?;
        image.attach_view(view);
        views.push(view);
    }
    Ok(CascadeArray {
        image,
        sampled,
        mips: views,
    })
}

fn array_view(
    device: &VkDevice,
    image: vk::Image,
    base_mip: u32,
    mips: u32,
    layers: u32,
) -> RenderResult<vk::ImageView> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
        .format(CASCADE_FORMAT)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: base_mip,
            level_count: mips,
            base_array_layer: 0,
            layer_count: layers,
        });
    // SAFETY: the create-info is live for the call and names an image of this device.
    unsafe { device.create_image_view(&info, None) }
        .map_err(|e| super::error::map_vk_result(e, "ocean cascade view"))
}

// Trilinear, anisotropic and wrapping: a cascade tiles the sea.
fn create_sampler(device: &VkDevice, max_anisotropy: f32) -> RenderResult<OwnedSampler> {
    let aniso = max_anisotropy > 1.0;
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(aniso)
        .max_anisotropy(if aniso { max_anisotropy } else { 1.0 })
        .min_lod(0.0)
        .max_lod(vk::LOD_CLAMP_NONE);
    device
        .create_sampler(&info)
        .map_err(|e| super::error::map_vk_result(e, "ocean sampler"))
}

// One kernel's writes made visible to the next one's reads.
fn compute_step(rec: &Recorder<'_>) {
    rec.pipeline_barrier(
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        &[vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)],
        &[],
        &[],
    );
}

impl VkContext {
    // Encode the `WaterSim` node: for every simulation, evolve and transform
    // its rows, then its columns into mip 0, then reduce down the mips.
    pub(in crate::vulkan) fn encode_water_sim(&self, rec: &Recorder<'_>, time: f64) {
        let Some(k) = self.water_sim.as_ref().and_then(|o| o.kernels.as_ref()) else {
            return;
        };
        let draw_stages =
            vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER;
        // The last frame's water draw finishes reading before the arrays are
        // rewritten, and its foam is what this frame's fades.
        rec.pipeline_barrier(
            draw_stages | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            &[vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)],
            &[],
            &[],
        );
        let mut clock = k.foam_clock.get();
        let keep = clock.advance(time);
        k.foam_clock.set(clock);
        let compute = vk::ShaderStageFlags::COMPUTE;
        let levels = k.sets.len();
        for sim in &k.plan.simulations {
            let params = OceanFftParams::simulate(sim, time, keep);
            let [size, _, _, cascades] = params.grid;
            rec.bind_descriptor_sets(
                vk::PipelineBindPoint::COMPUTE,
                &k.layout,
                0,
                &k.sets[..1],
                &[],
            );
            rec.push_constants(&k.layout, compute, 0, &params);
            rec.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &k.rows);
            rec.dispatch(size, cascades, 1);
            compute_step(rec);
            let columns = if sim.foam.is_some() {
                &k.columns_foam
            } else {
                &k.columns
            };
            rec.bind_pipeline(vk::PipelineBindPoint::COMPUTE, columns);
            rec.dispatch(size, cascades, 1);
            rec.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &k.mip);
            for level in 1..levels {
                compute_step(rec);
                let reduce = params.reduce(u32::saturating_from(level));
                let groups = ocean_mip_groups(reduce.mip_size);
                rec.bind_descriptor_sets(
                    vk::PipelineBindPoint::COMPUTE,
                    &k.layout,
                    0,
                    &k.sets[level..=level],
                    &[],
                );
                rec.push_constants(&k.layout, compute, 0, &reduce);
                rec.dispatch(groups, groups, cascades);
            }
            compute_step(rec);
        }
        // The water draw samples what the kernels wrote.
        rec.pipeline_barrier(
            vk::PipelineStageFlags::COMPUTE_SHADER,
            draw_stages,
            &[vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)],
            &[],
            &[],
        );
    }
}
