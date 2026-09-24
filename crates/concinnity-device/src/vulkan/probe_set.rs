// The reflection-probe set as Vulkan binds it: one cube-map array holding every
// probe's prefiltered radiance, a cube per probe, and per-frame storage buffers
// of parallax records, one `ProbeUniforms` per installed probe. The global set
// carries all three bindings (the count UBO, the array, the records), so every
// pass that binds it -- forward, SSR, the RT resolve, the transparent pass --
// reads the same set.
//
// The array lives in GENERAL for its whole life: the convolution writes one
// probe's cube through storage views while frames sample the cubes installed
// before it, and the view a frame binds covers every cube, so a per-cube layout
// transition would leave that view spanning two layouts.

use ash::vk;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::reflection_probe::PrefilterPlan;
use concinnity_core::render::uniforms::{ProbeSet, ProbeUniforms, grown_probe_capacity};

use super::allocator::{DeviceAllocator, PooledBuffer, PooledImage};
use super::context::VkContext;
use super::descriptor_layout::{PROBE_CUBES_BINDING, PROBE_RECORDS_SSBO_BINDING};
use super::probe_prefilter::PROBE_CUBE_FORMAT;
use super::texture::GpuUploadContext;

// The layout every probe cube descriptor declares, and the one the array stays in.
pub(super) const PROBE_CUBES_LAYOUT: vk::ImageLayout = vk::ImageLayout::GENERAL;

// A cube-map array and the views the passes and the convolution bind: one
// CUBE_ARRAY view over every cube, and per cube one single-mip 2D_ARRAY storage
// view of its six layers per mip. Every view is attached to the image's lease.
pub(super) struct ProbeCubeArray {
    image: PooledImage,
    view: vk::ImageView,
    mip_views: Vec<Vec<vk::ImageView>>,
    capacity: usize,
}

impl ProbeCubeArray {
    // `capacity` cubes at the bake's face size and mip count, in GENERAL.
    pub(super) fn new(
        upload: &GpuUploadContext<'_>,
        plan: &PrefilterPlan,
        capacity: usize,
    ) -> RenderResult<ProbeCubeArray> {
        Self::create(upload, plan.face_size(), plan.mips(), capacity)
    }

    // A one-texel array holding one cube, bound by sets whose probe count is 0
    // (captures, planar mirrors) and by the global sets until a world places a
    // probe. No shader samples it.
    pub(super) fn stand_in(upload: &GpuUploadContext<'_>) -> RenderResult<ProbeCubeArray> {
        let mut array = Self::create(upload, 1, 1, 1)?;
        array.capacity = 0;
        Ok(array)
    }

    fn create(
        upload: &GpuUploadContext<'_>,
        face_size: u32,
        mips: u32,
        cubes: usize,
    ) -> RenderResult<ProbeCubeArray> {
        let layers = 6 * cubes as u32;
        let image = create_image(
            upload.alloc,
            CubeImage {
                face_size,
                mips,
                layers,
                usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            },
        )
        .map_err(|e| e.context("probe cube array"))?;
        let view = create_view(
            upload.device,
            image.image(),
            vk::ImageViewType::CUBE_ARRAY,
            range(0, mips, 0, layers),
        )?;
        image.attach_view(view);
        let mip_views = (0..cubes)
            .map(|cube| mip_storage_views(upload.device, &image, cube, mips))
            .collect::<RenderResult<Vec<_>>>()?;
        let handle = image.image();
        super::texture::one_shot_submit(upload.device, upload.command_pool, upload.queue, |cmd| {
            let barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(PROBE_CUBES_LAYOUT)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(handle)
                .subresource_range(range(0, mips, 0, layers));
            // SAFETY: `cmd` is in the recording state, the barrier it borrows is
            // live for the call, and the image belongs to this device.
            unsafe {
                upload.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    std::slice::from_ref(&barrier),
                );
            }
        })?;
        Ok(ProbeCubeArray {
            image,
            view,
            mip_views,
            capacity: cubes,
        })
    }

    // A placeholder naming no image, left behind when the context tears down.
    fn null() -> ProbeCubeArray {
        ProbeCubeArray {
            image: PooledImage::null(),
            view: vk::ImageView::null(),
            mip_views: Vec::new(),
            capacity: 0,
        }
    }

    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(super) fn view(&self) -> vk::ImageView {
        self.view
    }

    pub(super) fn image(&self) -> vk::Image {
        self.image.image()
    }

    // The single-mip storage views of cube `cube`, one per mip.
    pub(super) fn mip_views(&self, cube: usize) -> Option<&[vk::ImageView]> {
        self.mip_views.get(cube).map(Vec::as_slice)
    }
}

// The subresources of `layer_count` layers from `base_layer` at `level_count`
// mips from `base_mip`.
pub(super) fn range(
    base_mip: u32,
    level_count: u32,
    base_layer: u32,
    layer_count: u32,
) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: base_mip,
        level_count,
        base_array_layer: base_layer,
        layer_count,
    }
}

// The shape of a probe cube image: `layers / 6` cubes of `mips` levels.
#[derive(Clone, Copy)]
pub(super) struct CubeImage {
    pub(super) face_size: u32,
    pub(super) mips: u32,
    pub(super) layers: u32,
    pub(super) usage: vk::ImageUsageFlags,
}

// A cube-compatible image in the probe format, device local.
pub(super) fn create_image(alloc: &DeviceAllocator, shape: CubeImage) -> RenderResult<PooledImage> {
    let info = vk::ImageCreateInfo::default()
        .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE)
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width: shape.face_size,
            height: shape.face_size,
            depth: 1,
        })
        .mip_levels(shape.mips)
        .array_layers(shape.layers)
        .format(PROBE_CUBE_FORMAT)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(shape.usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::TYPE_1);
    alloc.create_image(&info, vk::MemoryPropertyFlags::DEVICE_LOCAL)
}

// One single-mip 2D_ARRAY storage view of cube `cube`'s six layers per mip,
// each attached to the image's lease. A cube is a six-layer array, so this is
// what lets a kernel address (x, y, face) directly.
pub(super) fn mip_storage_views(
    device: &super::owned::VkDevice,
    image: &PooledImage,
    cube: usize,
    mips: u32,
) -> RenderResult<Vec<vk::ImageView>> {
    (0..mips)
        .map(|mip| {
            let view = create_view(
                device,
                image.image(),
                vk::ImageViewType::TYPE_2D_ARRAY,
                range(mip, 1, 6 * cube as u32, 6),
            )?;
            image.attach_view(view);
            Ok(view)
        })
        .collect()
}

pub(super) fn create_view(
    device: &super::owned::VkDevice,
    image: vk::Image,
    view_type: vk::ImageViewType,
    subresources: vk::ImageSubresourceRange,
) -> RenderResult<vk::ImageView> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(view_type)
        .format(PROBE_CUBE_FORMAT)
        .subresource_range(subresources);
    // SAFETY: the create-info is live for the call and names only this device's
    // image, whose layer and mip counts cover `subresources`.
    unsafe { device.create_image_view(&info, None) }
        .map_err(|e| super::error::map_vk_result(e, "probe cube view"))
}

// A host-visible storage buffer of parallax records with room for `capacity`.
pub(super) struct ProbeRecords {
    buffer: PooledBuffer,
    capacity: usize,
}

impl ProbeRecords {
    pub(super) fn new(alloc: &DeviceAllocator, capacity: usize) -> RenderResult<ProbeRecords> {
        let capacity = capacity.max(1);
        let buffer = alloc.create_buffer(
            (capacity * size_of::<ProbeUniforms>()) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        buffer.write_slice(
            0,
            &vec![<ProbeUniforms as bytemuck::Zeroable>::zeroed(); capacity],
        );
        Ok(ProbeRecords { buffer, capacity })
    }

    pub(super) fn buffer(&self) -> vk::Buffer {
        self.buffer.buffer()
    }

    pub(super) fn descriptor(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::default()
            .buffer(self.buffer.buffer())
            .offset(0)
            .range(vk::WHOLE_SIZE)
    }
}

// Everything the probe set binds: the live array (`None` until a world places a
// probe), the stand-in bound in its place and by every set that reads no probe,
// one records buffer per frame in flight, and beside them the one-record and
// empty-header stand-ins those sets bind.
pub(super) struct ProbeSetGpu {
    pub(super) cubes: Option<ProbeCubeArray>,
    pub(super) stand_in: ProbeCubeArray,
    pub(super) records: Vec<ProbeRecords>,
    pub(super) stand_in_records: ProbeRecords,
    pub(super) stand_in_set: PooledBuffer,
}

impl ProbeSetGpu {
    pub(super) fn new(upload: &GpuUploadContext<'_>, frames: usize) -> RenderResult<ProbeSetGpu> {
        let capacity = concinnity_core::render::uniforms::DEFAULT_PROBE_RECORD_CAPACITY;
        let stand_in_set = upload.alloc.create_buffer(
            size_of::<ProbeSet>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        stand_in_set.write_val(0, &ProbeSet::EMPTY);
        Ok(ProbeSetGpu {
            cubes: None,
            stand_in: ProbeCubeArray::stand_in(upload)?,
            records: (0..frames)
                .map(|_| ProbeRecords::new(upload.alloc, capacity))
                .collect::<RenderResult<_>>()?,
            stand_in_records: ProbeRecords::new(upload.alloc, 1)?,
            stand_in_set,
        })
    }

    // The array the global sets bind: the live one, or the stand-in.
    pub(super) fn bound_cubes(&self) -> &ProbeCubeArray {
        self.cubes.as_ref().unwrap_or(&self.stand_in)
    }

    // Drop every image and buffer, retiring them through the allocator ahead of
    // the device that owns them.
    pub(super) fn release(&mut self) {
        self.cubes = None;
        self.stand_in = ProbeCubeArray::null();
        self.records.clear();
        self.stand_in_records = ProbeRecords {
            buffer: PooledBuffer::null(),
            capacity: 0,
        };
        self.stand_in_set = PooledBuffer::null();
    }
}

impl VkContext {
    // Make room for `placements` cubes, replacing the array when the list
    // outgrows it and pointing every global set at the new one. Idles the device
    // first: a submitted frame may still read the global sets, and the array
    // being replaced.
    pub(super) fn reserve_probe_cubes(
        &mut self,
        plan: &PrefilterPlan,
        placements: usize,
    ) -> RenderResult<()> {
        let have = self
            .probe
            .gpu
            .cubes
            .as_ref()
            .map_or(0, ProbeCubeArray::capacity);
        let Some(capacity) = grown_probe_capacity(have, placements, 1) else {
            return Ok(());
        };
        self.wait_idle();
        let upload = GpuUploadContext {
            alloc: &self.hw.alloc,
            device: &self.hw.device,
            command_pool: self.commands.command_pool,
            queue: self.hw.graphics_queue,
        };
        self.probe.gpu.cubes = Some(ProbeCubeArray::new(&upload, plan, capacity)?);
        self.rewrite_global_binding(PROBE_CUBES_BINDING);
        tracing::debug!("reflection probes: cube array grown to {capacity}");
        Ok(())
    }

    // Write this frame's probe count and records. The frame's records buffer
    // grows when the installed count outgrows it, to the array's capacity so it
    // grows once per placement list; its fence has retired, so the old buffer and
    // this frame's global set are free to replace and rewrite.
    pub(super) fn upload_probe_set(&mut self, frame_idx: usize) -> RenderResult<()> {
        let count = self.probe.book.count();
        self.uniforms.probe_set_ubo_buffers[frame_idx].write_val(0, &self.probe.book.header());
        let floor = self.probe.gpu.bound_cubes().capacity();
        if let Some(capacity) =
            grown_probe_capacity(self.probe.gpu.records[frame_idx].capacity, count, floor)
        {
            self.probe.gpu.records[frame_idx] = ProbeRecords::new(&self.hw.alloc, capacity)?;
            // This frame's fence has signaled, so no submission still reads its set.
            self.global_bindings().frame(frame_idx).write_binding(
                &self.hw.device,
                self.descriptors.global_sets[frame_idx],
                PROBE_RECORDS_SSBO_BINDING,
            );
            let info = self.probe.gpu.records[frame_idx].descriptor();
            self.light_cull
                .write_probe_records(&self.hw.device, frame_idx, info);
        }
        self.probe.gpu.records[frame_idx]
            .buffer
            .write_slice(0, self.probe.book.records());
        Ok(())
    }
}
