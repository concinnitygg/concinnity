//! The atmosphere's clouds for the Vulkan backend (see
//! `concinnity_engine::render::environment::clouds`): the noise volume the layers are
//! carved from, built by the first frame's atmosphere pass; the shadow map and
//! the panorama that pass writes for the shaded passes and the sky's cubes; the
//! half-resolution target the march writes; and the composite that blends it
//! over the scene behind the haze, through the haze pass's render pass and
//! framebuffers.
//!
//! The volume, the panorama and the target rest in `GENERAL` for good: the
//! kernels write them as storage images and every reader samples them there,
//! so the steps between the passes are memory barriers that name no image.
//! The shadow map rests sampled, as every global set binds it, and is opened
//! to `GENERAL` for its kernel and closed again inside the atmosphere's pass.
//!
//! Mirrors src/metal/clouds.rs and src/directx/clouds.rs.

use std::cell::Cell;

use ash::vk;
use concinnity_core::math::SaturatingFrom;
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::environment::atmosphere::{
    AtmosphereFrame, AtmosphereParams, AtmosphereSettings,
};
use concinnity_engine::render::environment::clouds::{
    CloudParams, NOISE_SIZE, PANORAMA_SIZE, SHADOW_SIZE, cloud_target_size,
};

use super::allocator::{PooledBuffer, PooledImage};
use super::builtin_shaders::{
    CLOUD_COMPOSITE_FRAG, CLOUD_MARCH, CLOUD_NOISE, CLOUD_PANORAMA, CLOUD_SHADOW, CompileProgram,
    FULLSCREEN_VERT,
};
use super::descriptor_layout::{Binding, PoolSizes};
use super::owned::{
    OwnedDescriptorPool, OwnedFramebuffer, OwnedPipeline, OwnedPipelineLayout, OwnedRenderPass,
    OwnedSampler, OwnedSetLayout, VkDevice,
};
use super::pipeline_desc::{Blend, GraphicsPipelineDesc, compute_pipeline};
use super::record::Recorder;
use super::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use super::set_writes::SetWrites;
use super::texture::{
    GpuUploadContext, LayoutTransition, SubresourceRange, one_shot_submit,
    transition_image_layout_range,
};

const NOISE_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
const TARGET_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
const NOISE_GROUP: u32 = 4;
const MARCH_GROUP: u32 = 8;

// The noise kernel's set: the volume it writes.
fn noise_bindings() -> [Binding; 1] {
    [(
        0,
        vk::DescriptorType::STORAGE_IMAGE,
        vk::ShaderStageFlags::COMPUTE,
    )]
}

// The march's set: the haze pass's six bindings, then the clouds' constants,
// the noise volume and its sampler, and the target.
fn march_bindings() -> [Binding; 10] {
    use vk::DescriptorType as T;
    let c = vk::ShaderStageFlags::COMPUTE;
    [
        (0, T::UNIFORM_BUFFER, c),
        (1, T::SAMPLED_IMAGE, c),
        (2, T::SAMPLED_IMAGE, c),
        (3, T::SAMPLER, c),
        (4, T::SAMPLED_IMAGE, c),
        (5, T::SAMPLED_IMAGE, c),
        (6, T::UNIFORM_BUFFER, c),
        (7, T::SAMPLED_IMAGE, c),
        (8, T::SAMPLER, c),
        (9, T::STORAGE_IMAGE, c),
    ]
}

// The composite's set: the clouds' constants, the scene depth and the target.
fn composite_bindings() -> [Binding; 3] {
    use vk::DescriptorType as T;
    let f = vk::ShaderStageFlags::FRAGMENT;
    [
        (0, T::UNIFORM_BUFFER, f),
        (1, T::SAMPLED_IMAGE, f),
        (2, T::SAMPLED_IMAGE, f),
    ]
}

// An image of the clouds' and the one view every binding uses, resting in
// `GENERAL`.
struct Image {
    image: PooledImage,
    view: vk::ImageView,
}

fn create_image(
    gpu: &GpuUploadContext<'_>,
    extent: [u32; 3],
    format: vk::Format,
    label: &str,
) -> RenderResult<Image> {
    create_image_resting(gpu, extent, format, vk::ImageLayout::GENERAL, label)
}

// An image resting in `resting`.
fn create_image_resting(
    gpu: &GpuUploadContext<'_>,
    extent: [u32; 3],
    format: vk::Format,
    resting: vk::ImageLayout,
    label: &str,
) -> RenderResult<Image> {
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
        .format(format)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::TYPE_1);
    let image = gpu
        .alloc
        .create_image(&info, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .map_err(|e| e.context(label))?;
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image.image())
        .view_type(if volume {
            vk::ImageViewType::TYPE_3D
        } else {
            vk::ImageViewType::TYPE_2D
        })
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    // SAFETY: the create-info is live for the call and names an image of this device.
    let view = unsafe { gpu.device.create_image_view(&view_info, None) }
        .map_err(|e| super::error::map_vk_result(e, label))?;
    image.attach_view(view);
    let raw = image.image();
    one_shot_submit(gpu.device, gpu.command_pool, gpu.queue, |cmd| {
        transition_image_layout_range(
            gpu.device,
            cmd,
            raw,
            LayoutTransition {
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: resting,
                aspect: vk::ImageAspectFlags::COLOR,
            },
            SubresourceRange {
                base_layer: 0,
                layer_count: 1,
                base_mip: 0,
                mip_count: 1,
            },
        );
    })?;
    Ok(Image { image, view })
}

fn create_target(gpu: &GpuUploadContext<'_>, size: [u32; 2]) -> RenderResult<Image> {
    create_image(gpu, [size[0], size[1], 1], TARGET_FORMAT, "cloud target")
}

fn create_repeat_sampler(device: &VkDevice) -> RenderResult<OwnedSampler> {
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT);
    device
        .create_sampler(&info)
        .map_err(|e| super::error::map_vk_result(e, "cloud noise sampler"))
}

fn pipeline_layout(
    device: &VkDevice,
    set: vk::DescriptorSetLayout,
    label: &str,
) -> RenderResult<OwnedPipelineLayout> {
    device
        .create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&set)),
        )
        .map_err(|e| super::error::map_vk_result(e, label))
}

/// What the clouds borrow from the atmosphere: its per-frame constants, its
/// tables and their sampler, and the haze pass's render pass, plus the scene
/// depth they stop at.
#[derive(Clone, Copy)]
pub(in crate::vulkan) struct CloudInputs<'a> {
    pub(in crate::vulkan) atmo_ubos: &'a [PooledBuffer],
    pub(in crate::vulkan) aerial: vk::ImageView,
    pub(in crate::vulkan) transmittance: vk::ImageView,
    pub(in crate::vulkan) multiscatter: vk::ImageView,
    pub(in crate::vulkan) table_sampler: vk::Sampler,
    pub(in crate::vulkan) composite_pass: vk::RenderPass,
    pub(in crate::vulkan) depth_views: &'a [vk::ImageView],
    pub(in crate::vulkan) extent: vk::Extent2D,
    pub(in crate::vulkan) msaa: bool,
}

pub(in crate::vulkan) struct VkClouds {
    // This frame's constants, `None` until a frame carries the clouds' inputs.
    params: Option<CloudParams>,
    noise: Image,
    // Whether an atmosphere pass has built the noise volume yet.
    noise_built: Cell<bool>,
    shadow: Image,
    panorama: Image,
    target: Image,
    target_size: [u32; 2],
    ubos: Vec<PooledBuffer>,
    frames: u64,
    _pool: OwnedDescriptorPool,
    sampler: OwnedSampler,
    _noise_set_layout: OwnedSetLayout,
    noise_layout: OwnedPipelineLayout,
    noise_kernel: OwnedPipeline,
    noise_set: vk::DescriptorSet,
    _march_set_layout: OwnedSetLayout,
    march_layout: OwnedPipelineLayout,
    march_kernel: OwnedPipeline,
    march_sets: Vec<vk::DescriptorSet>,
    shadow_kernel: OwnedPipeline,
    panorama_kernel: OwnedPipeline,
    // Per frame slot, the march's set writing the shadow map and the panorama.
    map_sets: Vec<[vk::DescriptorSet; 2]>,
    _composite_set_layout: OwnedSetLayout,
    composite_layout: OwnedPipelineLayout,
    composite_pipeline: OwnedPipeline,
    composite_sets: Vec<vk::DescriptorSet>,
    // The haze pass's render pass, which the composite draws through.
    composite_pass: vk::RenderPass,
    // Per frame slot, the pre-TAA scene the composite draws into, its extent,
    // and the framebuffer over it.
    scenes: Vec<Option<(vk::ImageView, vk::Extent2D, OwnedFramebuffer)>>,
}

impl VkClouds {
    // The volume, the target, the pipelines and one set of each per frame
    // slot.
    pub(in crate::vulkan) fn new(
        gpu: &GpuUploadContext<'_>,
        inputs: CloudInputs<'_>,
        hot_reload: bool,
    ) -> RenderResult<Self> {
        let device = gpu.device;
        let frames = inputs.atmo_ubos.len();
        let noise = create_image(gpu, [NOISE_SIZE; 3], NOISE_FORMAT, "cloud noise volume")?;
        let target_size = cloud_target_size(inputs.extent.width, inputs.extent.height);
        let target = create_target(gpu, target_size)?;
        let shadow = create_image_resting(
            gpu,
            [SHADOW_SIZE, SHADOW_SIZE, 1],
            TARGET_FORMAT,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            "cloud shadow map",
        )?;
        let panorama = create_image(
            gpu,
            [PANORAMA_SIZE[0], PANORAMA_SIZE[1], 1],
            TARGET_FORMAT,
            "cloud panorama",
        )?;
        let sampler = create_repeat_sampler(device)?;
        let ubos = (0..frames)
            .map(|_| {
                gpu.alloc.create_buffer(
                    size_of::<CloudParams>() as u64,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
            })
            .collect::<RenderResult<Vec<_>>>()?;

        let noise_set_layout = create_descriptor_set_layout(device, &noise_bindings())?;
        let march_set_layout = create_descriptor_set_layout(device, &march_bindings())?;
        let composite_set_layout = create_descriptor_set_layout(device, &composite_bindings())?;
        let f = u32::saturating_from(frames);
        let sizes = PoolSizes::default()
            .sets(&noise_bindings(), 1)
            .sets(&march_bindings(), 3 * f)
            .sets(&composite_bindings(), f)
            .build();
        let pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1 + 4 * f)
                    .pool_sizes(&sizes),
            )
            .map_err(|e| super::error::map_vk_result(e, "cloud descriptor pool"))?;

        let noise_layout =
            pipeline_layout(device, noise_set_layout.handle(), "cloud noise layout")?;
        let noise_kernel = compute_pipeline(
            device,
            noise_layout.handle(),
            &CLOUD_NOISE.compile(hot_reload)?,
            CLOUD_NOISE.label,
        )?;
        let noise_set = alloc_descriptor_sets(device, pool.handle(), &[noise_set_layout.handle()])?
            .first()
            .copied()
            .ok_or_else(|| {
                concinnity_engine::render::backend::error::RenderError::Other(
                    "the cloud noise set was not allocated".into(),
                )
            })?;
        SetWrites::new(noise_set)
            .storage_image(0, noise.view)
            .apply(device);

        let march_layout =
            pipeline_layout(device, march_set_layout.handle(), "cloud march layout")?;
        let march_kernel =
            build_march_kernel(device, march_layout.handle(), inputs.msaa, hot_reload)?;
        let march_sets = alloc_descriptor_sets(
            device,
            pool.handle(),
            &vec![march_set_layout.handle(); frames],
        )?;
        let shadow_kernel = map_kernel(device, march_layout.handle(), &CLOUD_SHADOW, hot_reload)?;
        let panorama_kernel =
            map_kernel(device, march_layout.handle(), &CLOUD_PANORAMA, hot_reload)?;
        let map_sets = alloc_descriptor_sets(
            device,
            pool.handle(),
            &vec![march_set_layout.handle(); 2 * frames],
        )?
        .chunks_exact(2)
        .map(|pair| [pair[0], pair[1]])
        .collect();

        let composite_layout = pipeline_layout(
            device,
            composite_set_layout.handle(),
            "cloud composite layout",
        )?;
        let composite_pipeline = build_composite_pipeline(
            device,
            composite_layout.handle(),
            inputs.composite_pass,
            inputs.msaa,
            hot_reload,
        )?;
        let composite_sets = alloc_descriptor_sets(
            device,
            pool.handle(),
            &vec![composite_set_layout.handle(); frames],
        )?;

        let clouds = Self {
            params: None,
            noise,
            noise_built: Cell::new(false),
            shadow,
            panorama,
            target,
            target_size,
            ubos,
            frames: 0,
            _pool: pool,
            sampler,
            _noise_set_layout: noise_set_layout,
            noise_layout,
            noise_kernel,
            noise_set,
            _march_set_layout: march_set_layout,
            march_layout,
            march_kernel,
            march_sets,
            shadow_kernel,
            panorama_kernel,
            map_sets,
            _composite_set_layout: composite_set_layout,
            composite_layout,
            composite_pipeline,
            composite_sets,
            composite_pass: inputs.composite_pass,
            scenes: (0..frames).map(|_| None).collect(),
        };
        clouds.write_sets(device, &inputs);
        Ok(clouds)
    }

    // Point every set at the scene depth and the target, beside what does not
    // move.
    fn write_sets(&self, device: &VkDevice, inputs: &CloudInputs<'_>) {
        let atmo_size = size_of::<AtmosphereParams>() as u64;
        let cloud_size = size_of::<CloudParams>() as u64;
        let last_depth = inputs.depth_views.len().saturating_sub(1);
        for (i, ((ubo, atmo_ubo), (&march, maps))) in self
            .ubos
            .iter()
            .zip(inputs.atmo_ubos)
            .zip(self.march_sets.iter().zip(&self.map_sets))
            .enumerate()
        {
            let outputs = [
                (march, self.target.view),
                (maps[0], self.shadow.view),
                (maps[1], self.panorama.view),
            ];
            for (set, output) in outputs {
                SetWrites::new(set)
                    .uniform_buffer(0, atmo_ubo.buffer(), atmo_size)
                    .sampled_image(1, inputs.depth_views[i.min(last_depth)])
                    .sampled_image_in(2, inputs.aerial, vk::ImageLayout::GENERAL)
                    .sampler(3, inputs.table_sampler)
                    .sampled_image_in(4, inputs.transmittance, vk::ImageLayout::GENERAL)
                    .sampled_image_in(5, inputs.multiscatter, vk::ImageLayout::GENERAL)
                    .uniform_buffer(6, ubo.buffer(), cloud_size)
                    .sampled_image_in(7, self.noise.view, vk::ImageLayout::GENERAL)
                    .sampler(8, self.sampler.handle())
                    .storage_image(9, output)
                    .apply(device);
            }
        }
        for (i, (set, ubo)) in self.composite_sets.iter().zip(&self.ubos).enumerate() {
            SetWrites::new(*set)
                .uniform_buffer(0, ubo.buffer(), cloud_size)
                .sampled_image(1, inputs.depth_views[i.min(last_depth)])
                .sampled_image_in(2, self.target.view, vk::ImageLayout::GENERAL)
                .apply(device);
        }
    }

    // Size the target for the rebuilt scene and point the sets at it and the
    // rebuilt depth. The device is idle across a resize, so the old target is
    // free to go.
    pub(in crate::vulkan) fn rebuild(
        &mut self,
        gpu: &GpuUploadContext<'_>,
        inputs: &CloudInputs<'_>,
    ) -> RenderResult<()> {
        let size = cloud_target_size(inputs.extent.width, inputs.extent.height);
        if size != self.target_size {
            self.target = create_target(gpu, size)?;
            self.target_size = size;
        }
        self.write_sets(gpu.device, inputs);
        Ok(())
    }

    // Rebuild the pipelines from the shader sources (shader hot reload).
    pub(in crate::vulkan) fn rebuild_pipelines(
        &mut self,
        device: &VkDevice,
        composite_pass: vk::RenderPass,
        msaa: bool,
    ) -> RenderResult<()> {
        let noise_kernel = compute_pipeline(
            device,
            self.noise_layout.handle(),
            &CLOUD_NOISE.compile(true)?,
            CLOUD_NOISE.label,
        )?;
        let march_kernel = build_march_kernel(device, self.march_layout.handle(), msaa, true)?;
        let shadow_kernel = map_kernel(device, self.march_layout.handle(), &CLOUD_SHADOW, true)?;
        let panorama_kernel =
            map_kernel(device, self.march_layout.handle(), &CLOUD_PANORAMA, true)?;
        let composite_pipeline = build_composite_pipeline(
            device,
            self.composite_layout.handle(),
            composite_pass,
            msaa,
            true,
        )?;
        self.noise_kernel = noise_kernel;
        self.march_kernel = march_kernel;
        self.shadow_kernel = shadow_kernel;
        self.panorama_kernel = panorama_kernel;
        self.composite_pipeline = composite_pipeline;
        self.noise_built.set(false);
        Ok(())
    }

    // Release the pooled resources ahead of the device.
    pub(in crate::vulkan) fn destroy(&mut self) {
        self.ubos.clear();
        for image in [
            &mut self.noise,
            &mut self.shadow,
            &mut self.panorama,
            &mut self.target,
        ] {
            image.image = PooledImage::null();
        }
    }

    // This frame's constants for `settings` and `frame` over a scene `scene`
    // pixels across, written into frame slot `frame_idx`'s buffer.
    pub(in crate::vulkan) fn prepare(
        &mut self,
        settings: &AtmosphereSettings,
        frame: Option<&AtmosphereFrame>,
        scene: [u32; 2],
        frame_idx: usize,
    ) {
        self.params = match (
            settings.clouds.as_ref(),
            frame.and_then(|f| f.clouds.as_ref()),
        ) {
            (Some(clouds), Some(cloud_frame)) => Some(CloudParams::new(
                clouds,
                settings.bottom_radius,
                cloud_frame,
                scene,
                self.frames,
            )),
            _ => None,
        };
        self.frames = self.frames.wrapping_add(1);
        if let (Some(params), Some(ubo)) = (self.params, self.ubos.get(frame_idx)) {
            ubo.write_val(0, &params);
        }
    }

    // Whether this frame has the clouds' constants.
    pub(in crate::vulkan) fn live(&self) -> bool {
        self.params.is_some()
    }

    // The shadow map and the panorama the shaded passes and the sky's cubes
    // read, the panorama only when `panorama` (the sky cube redraws), inside
    // the atmosphere's pass after its aerial volume, building the noise volume
    // on the first frame. Last frame's readers have finished with both, and
    // the aerial volume's writes are visible, before the caller calls this.
    pub(in crate::vulkan) fn encode_maps(
        &self,
        rec: &Recorder<'_>,
        frame_idx: usize,
        panorama: bool,
    ) {
        let Some(sets) = self.map_sets.get(frame_idx) else {
            return;
        };
        if self.params.is_none() {
            return;
        }
        if !self.noise_built.get() {
            rec.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &self.noise_kernel);
            rec.bind_descriptor_sets(
                vk::PipelineBindPoint::COMPUTE,
                &self.noise_layout,
                0,
                std::slice::from_ref(&self.noise_set),
                &[],
            );
            let groups = NOISE_SIZE.div_ceil(NOISE_GROUP);
            rec.dispatch(groups, groups, groups);
            rec.pipeline_barrier(
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)],
                &[],
                &[],
            );
            self.noise_built.set(true);
        }
        let groups = SHADOW_SIZE.div_ceil(MARCH_GROUP);
        let shadow = |from, to| {
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
                .image(self.shadow.image.image())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
        };
        let readers =
            vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER;
        rec.pipeline_barrier(
            readers,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            &[],
            &[],
            &[shadow(
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::GENERAL,
            )],
        );
        self.dispatch(rec, &self.shadow_kernel, sets[0], [groups, groups]);
        rec.pipeline_barrier(
            vk::PipelineStageFlags::COMPUTE_SHADER,
            readers,
            &[],
            &[],
            &[shadow(
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )],
        );
        if panorama {
            self.dispatch(
                rec,
                &self.panorama_kernel,
                sets[1],
                PANORAMA_SIZE.map(|s| s.div_ceil(MARCH_GROUP)),
            );
        }
    }

    fn dispatch(
        &self,
        rec: &Recorder<'_>,
        kernel: &OwnedPipeline,
        set: vk::DescriptorSet,
        groups: [u32; 2],
    ) {
        rec.bind_pipeline(vk::PipelineBindPoint::COMPUTE, kernel);
        rec.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            &self.march_layout,
            0,
            std::slice::from_ref(&set),
            &[],
        );
        rec.dispatch(groups[0], groups[1], 1);
    }

    // The shadow map's and the panorama's views.
    pub(in crate::vulkan) fn shadow_view(&self) -> vk::ImageView {
        self.shadow.view
    }

    pub(in crate::vulkan) fn panorama_view(&self) -> vk::ImageView {
        self.panorama.view
    }

    // March the clouds into the target.
    pub(in crate::vulkan) fn encode_march(&self, rec: &Recorder<'_>, frame_idx: usize) {
        let Some(set) = self.march_sets.get(frame_idx) else {
            return;
        };
        if self.params.is_none() {
            return;
        }
        // The last frame's composite has read the target before it is
        // rewritten.
        rec.pipeline_barrier(
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            &[vk::MemoryBarrier::default()],
            &[],
            &[],
        );
        rec.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &self.march_kernel);
        rec.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            &self.march_layout,
            0,
            std::slice::from_ref(set),
            &[],
        );
        let [w, h] = self.target_size;
        rec.dispatch(w.div_ceil(MARCH_GROUP), h.div_ceil(MARCH_GROUP), 1);
        rec.pipeline_barrier(
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            &[vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)],
            &[],
            &[],
        );
    }

    // Point frame slot `frame`'s composite at the pre-TAA scene `scene`,
    // `extent` across, when it names another. The slot's fence has passed;
    // a replaced framebuffer waits out the owned-handle retire.
    pub(in crate::vulkan) fn rescene(
        &mut self,
        device: &VkDevice,
        frame: usize,
        scene: vk::ImageView,
        extent: vk::Extent2D,
    ) -> RenderResult<()> {
        let Some(slot) = self.scenes.get_mut(frame) else {
            return Ok(());
        };
        if slot
            .as_ref()
            .is_some_and(|(view, at, _)| *view == scene && *at == extent)
        {
            return Ok(());
        }
        let attachments = [scene];
        let info = vk::FramebufferCreateInfo::default()
            .render_pass(self.composite_pass)
            .attachments(&attachments)
            .width(extent.width.max(1))
            .height(extent.height.max(1))
            .layers(1);
        let framebuffer = device
            .create_framebuffer(&info)
            .map_err(|e| super::error::map_vk_result(e, "cloud composite framebuffer"))?;
        *slot = Some((scene, extent, framebuffer));
        Ok(())
    }

    // Blend the target over frame slot `frame_idx`'s pre-TAA scene, through
    // `pass`.
    pub(in crate::vulkan) fn encode_composite(
        &self,
        rec: &Recorder<'_>,
        frame_idx: usize,
        pass: &OwnedRenderPass,
    ) {
        let (Some(set), Some(Some((_, extent, framebuffer)))) = (
            self.composite_sets.get(frame_idx),
            self.scenes.get(frame_idx),
        ) else {
            return;
        };
        if self.params.is_none() {
            return;
        }
        let extent = *extent;
        rec.begin_render_pass(pass, framebuffer, vk::Rect2D::default().extent(extent), &[]);
        rec.set_viewport(&vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        });
        rec.set_scissor(&vk::Rect2D::default().extent(extent));
        rec.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, &self.composite_pipeline);
        rec.bind_descriptor_sets(
            vk::PipelineBindPoint::GRAPHICS,
            &self.composite_layout,
            0,
            std::slice::from_ref(set),
            &[],
        );
        rec.draw(3, 1, 0, 0);
        rec.end_render_pass();
    }
}

fn map_kernel(
    device: &VkDevice,
    layout: vk::PipelineLayout,
    program: &concinnity_engine::render::shader::programs::ShaderProgram,
    hot_reload: bool,
) -> RenderResult<OwnedPipeline> {
    compute_pipeline(device, layout, &program.compile(hot_reload)?, program.label)
}

fn build_march_kernel(
    device: &VkDevice,
    layout: vk::PipelineLayout,
    msaa: bool,
    hot_reload: bool,
) -> RenderResult<OwnedPipeline> {
    compute_pipeline(
        device,
        layout,
        &CLOUD_MARCH.at(msaa).compile(hot_reload)?,
        CLOUD_MARCH.label,
    )
}

// `(light, 1 - T)` over the hazed scene -> `scene * T + light`.
fn build_composite_pipeline(
    device: &VkDevice,
    layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    msaa: bool,
    hot_reload: bool,
) -> RenderResult<OwnedPipeline> {
    let vert = FULLSCREEN_VERT.compile(hot_reload)?;
    let frag = CLOUD_COMPOSITE_FRAG.at(msaa).compile(hot_reload)?;
    GraphicsPipelineDesc::fullscreen(
        &vert,
        &frag,
        layout,
        render_pass,
        &[Blend::PremultipliedOver],
    )
    .build(device, "cloud composite")
}
