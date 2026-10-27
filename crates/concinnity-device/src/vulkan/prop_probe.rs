//! The prop reflection probe on Vulkan: the six cube faces of the scene seen
//! from the prop, rendered every frame on the frame's command buffer from the
//! frame's own records like a planar mirror, and their convolution into the
//! probe's cube of the probe array. The record and the plan are the shared
//! `probe::prop`; this module owns the capture and encodes the `PropProbe`
//! graph node. Mirrors `metal::prop_probe`.
//!
//! The faces share one cull slot per frame in flight: each face's cull and
//! packing rewrite the slot's lists after the previous face's indirect draw has
//! read them, which an explicit barrier orders. The faces render straight into
//! the layers of the capture's top mip, which the convolution then reads.

use ash::vk;
use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::render_types::{GpuDrawArgs, GpuObjectData};
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::probe::bake::CAPTURE_FACES;
use concinnity_engine::render::probe::capture::{self, PrefilterPlan};
use concinnity_engine::render::probe::prop::{PropProbeFrame, prop_probe_plan};
use concinnity_engine::render::probe::radiance::FaceStorage;
use concinnity_engine::render::uniforms::FaceCamera;

use super::allocator::PooledBuffer;
use super::compact::{CompactBuffers, DrawLists};
use super::context::{HDR_FORMAT, VkContext};
use super::descriptor_layout::{PROBE_RECORDS_SSBO_BINDING, PoolSizes, global_set};
use super::draw::ViewUniforms;
use super::global_set::{GlobalBindings, GlobalSetContents};
use super::graph_exec::GraphFrameParams;
use super::probe::{CaptureCull, FaceArea, FacePacking, FaceView};
use super::probe_prefilter::{CaptureFaces, PrefilterGpu, PrefilterTarget};
use super::resources::alloc_descriptor_sets;
use super::set_writes::SetWrites;
use super::texture::{
    GpuImage, ImageSpec, LayoutTransition, SubresourceRange, create_image, create_image_view,
    one_shot_submit, transition_image_layout_range,
};
use crate::vulkan::owned::{OwnedDescriptorPool, OwnedFramebuffer, VkDevice};

const FACE_DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

// The world's prop probe: where it captures this frame, and the capture it
// renders into. `None` capture until the first frame that rides one.
#[derive(Default)]
pub(in crate::vulkan) struct PropProbe {
    frame: Option<PropProbeFrame>,
    gpu: Option<PropProbeGpu>,
    // Set once building the capture failed, which is not retried.
    failed: bool,
}

// What the capture is built against; a change rebuilds it.
#[derive(Clone, Copy, PartialEq, Eq)]
struct CaptureKey {
    resolution: u32,
    cubes: vk::Image,
    samples: vk::SampleCountFlags,
    records: usize,
    hiz: Option<vk::ImageView>,
}

// The capture and the convolution into the probe's cube, the face targets,
// and per frame in flight the cull slot every face shares and each face's view.
struct PropProbeGpu {
    key: CaptureKey,
    plan: PrefilterPlan,
    prefilter: PrefilterGpu,
    // Held for the framebuffers: the multisampled color (with MSAA) and the
    // depth every face renders through.
    _color: Option<GpuImage>,
    _depth: GpuImage,
    framebuffers: Vec<OwnedFramebuffer>,
    frames: usize,
    // Per (face, frame), at `face * frames + frame`: the face's view block and
    // the global set binding it with that frame's lighting and probes.
    view_bufs: Vec<PooledBuffer>,
    global_sets: Vec<vk::DescriptorSet>,
    // Per frame: the cull set reading that frame's records, the packing set,
    // and the lists, status and group counts they write.
    cull_sets: Vec<vk::DescriptorSet>,
    compact_sets: Vec<vk::DescriptorSet>,
    indirect_bufs: Vec<PooledBuffer>,
    _status_bufs: Vec<PooledBuffer>,
    _group_counts: Vec<PooledBuffer>,
    // The cull's Hi-Z set with Hi-Z off, when the world runs Hi-Z.
    hiz_set: Option<vk::DescriptorSet>,
    _hiz_ubo: Option<PooledBuffer>,
    _pool: OwnedDescriptorPool,
}

// The prop probe's pool: per (face, frame) a global set, per frame a cull and
// a packing set, and one Hi-Z set (an image and a UBO) when the world runs
// Hi-Z.
fn prop_pool_sizes(frames: u32, has_hiz: u32) -> PoolSizes {
    let faces = u32::saturating_from(CAPTURE_FACES);
    PoolSizes::default()
        .sets(&global_set(), faces * frames)
        .sets(&super::init::main_cull_set(), frames)
        .sets(&super::compact::compact_set(), frames)
        .add(vk::DescriptorType::UNIFORM_BUFFER, has_hiz)
        .add(vk::DescriptorType::SAMPLED_IMAGE, has_hiz)
}

// What face set `i` of `frames` binds: its view, and frame `i % frames`'s
// lighting and probes.
fn face_contents(
    bindings: &GlobalBindings<'_>,
    view: vk::Buffer,
    frame: usize,
) -> GlobalSetContents {
    bindings
        .off_camera(
            view,
            bindings.uniforms.light_ubo_buffers[frame].buffer(),
            bindings.shadow.ubos[frame].buffer(),
            bindings.canopy(frame),
        )
        .with_probes(bindings, frame)
}

impl PropProbe {
    // Rewrite binding `binding` of every face set from what it is built with
    // now. The caller has idled the device.
    pub(in crate::vulkan) fn rewrite_global_binding(
        &self,
        device: &VkDevice,
        bindings: &GlobalBindings<'_>,
        binding: u32,
    ) {
        let Some(gpu) = self.gpu.as_ref() else {
            return;
        };
        for (i, &set) in gpu.global_sets.iter().enumerate() {
            face_contents(bindings, gpu.view_bufs[i].buffer(), i % gpu.frames)
                .write_binding(device, set, binding);
        }
    }

    // Point frame `frame`'s face sets at that frame's probe records, after its
    // records buffer was replaced. The frame's fence has retired.
    pub(in crate::vulkan) fn rewrite_probe_records(
        &self,
        device: &VkDevice,
        bindings: &GlobalBindings<'_>,
        frame: usize,
    ) {
        let Some(gpu) = self.gpu.as_ref() else {
            return;
        };
        for i in (0..gpu.global_sets.len()).filter(|i| i % gpu.frames == frame) {
            face_contents(bindings, gpu.view_bufs[i].buffer(), frame).write_binding(
                device,
                gpu.global_sets[i],
                PROBE_RECORDS_SSBO_BINDING,
            );
        }
    }
}

impl VkContext {
    // Whether this backend can capture the world's prop probe: the bindless
    // cull and packing it draws through, the convolution kernels, and a cube
    // for it.
    fn prop_probe_supported(&self) -> bool {
        self.cull.cull_kernels.is_some()
            && self.cull.compact.is_some()
            && self.cull.bindless_pipeline_layout.is_some()
            && self.probe.prefilter.is_some()
            && self.probe.gpu.cubes.is_some()
            && self.probe.book.prop_cube().is_some()
            && !self.probe.prop.failed
    }

    // This frame's prop probe, ahead of the probe records and the graph: where
    // the `PropProbe` node captures, and the probe's record, which the shaders
    // read once a frame before has filled the cube. A frame whose graph does
    // not run the node withdraws the record.
    pub(in crate::vulkan) fn prepare_prop_probe(
        &mut self,
        frame: Option<PropProbeFrame>,
        enabled: bool,
    ) {
        let Some(frame) = frame.filter(|_| enabled && self.prop_probe_supported()) else {
            self.probe.prop.frame = None;
            self.probe.book.set_prop_probe(None);
            return;
        };
        if let Err(e) = self.ensure_prop_capture(frame.resolution) {
            tracing::error!("reflection probes: the prop probe could not be built: {e}");
            self.probe.prop = PropProbe {
                failed: true,
                ..PropProbe::default()
            };
            self.probe.book.set_prop_probe(None);
            return;
        }
        let filled = self.probe.prop.frame.replace(frame).is_some();
        self.probe.book.set_prop_probe(filled.then_some(&frame));
    }

    // What a capture of `resolution`-texel faces is built against this frame.
    fn prop_capture_key(&self, resolution: u32) -> Option<CaptureKey> {
        let cubes = self.probe.gpu.cubes.as_ref()?;
        let object_buf = self.cull.object_buffers.first()?;
        Some(CaptureKey {
            resolution,
            cubes: cubes.image(),
            samples: self.targets.msaa_samples,
            records: object_buf.byte_len() / size_of::<GpuObjectData>(),
            hiz: self
                .cull
                .hiz
                .as_ref()
                .map(super::hiz::HiZResources::read_set_view),
        })
    }

    // Build the capture for faces of `resolution` texels, unless the one held
    // already matches. A new capture has not filled the cube yet. The replaced
    // capture's resources retire through the allocator and the device's
    // retire queue, after the frames still reading them.
    fn ensure_prop_capture(&mut self, resolution: u32) -> RenderResult<()> {
        let key = self
            .prop_capture_key(resolution)
            .ok_or_else(|| RenderError::Other("probe: no cube array for the prop probe".into()))?;
        if self.probe.prop.gpu.as_ref().is_some_and(|g| g.key == key) {
            return Ok(());
        }
        self.probe.prop.gpu = None;
        self.probe.prop.frame = None;
        self.probe.prop.gpu = Some(PropProbeGpu::new(self, key)?);
        Ok(())
    }

    // The `PropProbe` node: each face of the prop probe, culled against the
    // face's frustum with the prop left out and drawn from the frame's
    // records into its layer of the capture, then the convolution of the
    // capture into the probe's cube.
    pub(in crate::vulkan) fn encode_prop_probe(
        &self,
        cmd: vk::CommandBuffer,
        params: &GraphFrameParams<'_>,
    ) -> RenderResult<()> {
        let (Some(frame), Some(gpu)) = (self.probe.prop.frame, self.probe.prop.gpu.as_ref()) else {
            return Ok(());
        };
        let Some(&bindless_set) = self.cull.bindless_sets.get(params.frame_idx) else {
            return Ok(());
        };
        let f = params.frame_idx % gpu.frames;
        let eye = frame.eye();
        let size = gpu.plan.face_size();
        let extent = vk::Extent2D {
            width: size,
            height: size,
        };
        let shading = FaceCamera {
            vp: [[0.0; 4]; 4],
            view: [[0.0; 4]; 4],
            cam_pos: eye,
            elapsed: concinnity_core::math::narrow_f32(params.elapsed),
            prefilter_mip_count: self.scene.prefilter_mip_count as f32,
            sky_rot: self.state.view.sky_rot,
            irradiance: self.irradiance_lane(),
            pre_exposure: FaceStorage::PROBE.scale(),
            daylight: self.state.view.daylight,
            tile_morph: self.state.tile_morph_view(),
        };
        let packing = FacePacking {
            set: gpu.compact_sets[f],
            lists: DrawLists {
                buffer: gpu.indirect_bufs[f].buffer(),
                commands: gpu.key.records,
            },
            records: gpu.key.records,
        };
        for face in 0..CAPTURE_FACES {
            let ring = face * gpu.frames + f;
            let view = ViewUniforms::face(FaceCamera {
                vp: capture::face_view_projection(eye, face),
                view: capture::face_view_matrix(eye, face),
                ..shading
            })
            .with_stand_ins(frame.stand_ins());
            gpu.view_bufs[ring].write_val(0, &view);
            if face > 0 {
                self.order_prop_face_cull(cmd);
            }
            self.encode_probe_cull(
                cmd,
                gpu.cull_sets[f],
                gpu.hiz_set,
                packing,
                CaptureCull {
                    frustum: &capture::face_frustum(eye, face, frame.capture_distance),
                    eye,
                    exclude: frame.records,
                    capture_tiles: self.state.placement.capture_tiles,
                },
            );
            self.order_prop_face_render(cmd);
            self.encode_main_into_face(
                cmd,
                gpu.framebuffers[face].handle(),
                FaceArea::whole(extent),
                FaceView {
                    global_set: gpu.global_sets[ring],
                    storage: FaceStorage::PROBE,
                    sky_from_cube: true,
                },
                bindless_set,
                packing,
            );
        }
        self.encode_probe_pyramid(cmd, &gpu.prefilter, &gpu.plan)?;
        for mip in 1..gpu.plan.mips() {
            self.encode_probe_ggx_mip(cmd, &gpu.prefilter, &gpu.plan, mip)?;
        }
        self.encode_probe_cube_readable(cmd, &gpu.prefilter);
        Ok(())
    }

    // Order the next face's cull and packing, which rewrite the slot's status,
    // group counts and lists, after the previous face's packing wrote them and
    // its indirect draw read them.
    fn order_prop_face_cull(&self, cmd: vk::CommandBuffer) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::INDIRECT_COMMAND_READ)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
        // SAFETY: `cmd` is the frame's command buffer, recording outside a
        // render pass; the barrier names no resource.
        unsafe {
            self.hw.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        }
    }

    // Order a face's render after the last writes to its attachments: the
    // shared depth and multisampled color the previous face rendered, and the
    // capture layer the last frame's convolution read. `main_render_pass`
    // takes both in `UNDEFINED` and its external dependency makes nothing
    // available, so this barrier is what orders them.
    fn order_prop_face_render(&self, cmd: vk::CommandBuffer) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            );
        // SAFETY: `cmd` is the frame's command buffer, recording outside a
        // render pass; the barrier names no resource.
        unsafe {
            self.hw.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        }
    }
}

impl PropProbeGpu {
    fn new(ctx: &VkContext, key: CaptureKey) -> RenderResult<Self> {
        let device = &ctx.hw.device;
        let alloc = &ctx.hw.alloc;
        let missing =
            |what: &str| RenderError::Other(format!("probe: no {what} for the prop probe"));
        let pipelines = ctx
            .probe
            .prefilter
            .as_ref()
            .ok_or_else(|| missing("prefilter"))?;
        let cubes = ctx
            .probe
            .gpu
            .cubes
            .as_ref()
            .ok_or_else(|| missing("cube array"))?;
        let cube = ctx.probe.book.prop_cube().ok_or_else(|| missing("cube"))?;
        let kernels = ctx
            .cull
            .cull_kernels
            .as_ref()
            .ok_or_else(|| missing("cull"))?;
        let compact = ctx
            .cull
            .compact
            .as_ref()
            .ok_or_else(|| missing("packing"))?;
        let plan = prop_probe_plan(key.resolution);
        let size = plan.face_size();
        let frames = ctx.cull.object_buffers.len();

        let prefilter = PrefilterGpu::new(
            device,
            alloc,
            pipelines,
            &plan,
            PrefilterTarget::probe(cubes, cube, &plan)?,
            CaptureFaces::Rendered,
        )?;
        // The capture rests sampled between frames, the layout the pyramid
        // build takes it from; its faces' render passes discard the top mip.
        one_shot_submit(
            device,
            ctx.commands.command_pool,
            ctx.hw.graphics_queue,
            |cmd| {
                transition_image_layout_range(
                    device,
                    cmd,
                    prefilter.capture_image(),
                    LayoutTransition {
                        old_layout: vk::ImageLayout::UNDEFINED,
                        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        aspect: vk::ImageAspectFlags::COLOR,
                    },
                    SubresourceRange {
                        base_layer: 0,
                        layer_count: 6,
                        base_mip: 0,
                        mip_count: plan.mips(),
                    },
                );
            },
        )?;

        let target = |format, usage, samples| -> RenderResult<GpuImage> {
            let aspect = if format == FACE_DEPTH_FORMAT {
                vk::ImageAspectFlags::DEPTH
            } else {
                vk::ImageAspectFlags::COLOR
            };
            let pooled = create_image(
                alloc,
                &ImageSpec {
                    width: size,
                    height: size,
                    format,
                    tiling: vk::ImageTiling::OPTIMAL,
                    usage,
                    mem_props: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    samples,
                },
            )?;
            let view = create_image_view(device, pooled.image(), format, aspect)?;
            Ok(GpuImage::from_pooled(pooled, view))
        };
        let msaa = key.samples != vk::SampleCountFlags::TYPE_1;
        let color = msaa
            .then(|| {
                target(
                    HDR_FORMAT,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT
                        | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
                    key.samples,
                )
            })
            .transpose()?;
        let depth = target(
            FACE_DEPTH_FORMAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            key.samples,
        )?;
        let mut framebuffers = Vec::with_capacity(CAPTURE_FACES);
        for face in 0..CAPTURE_FACES {
            let layer = prefilter.capture_face_view(device, u32::saturating_from(face))?;
            let attachments: Vec<vk::ImageView> = match &color {
                Some(c) => vec![c.view, depth.view, layer],
                None => vec![layer, depth.view],
            };
            let info = vk::FramebufferCreateInfo::default()
                .render_pass(ctx.targets.main_render_pass.handle())
                .attachments(&attachments)
                .width(size)
                .height(size)
                .layers(1);
            framebuffers.push(
                device
                    .create_framebuffer(&info)
                    .map_err(|e| super::error::map_vk_result(e, "prop probe framebuffer"))?,
            );
        }

        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let ring = CAPTURE_FACES * frames;
        let view_bufs = (0..ring)
            .map(|_| {
                alloc.create_buffer(
                    size_of::<ViewUniforms>() as u64,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    host,
                )
            })
            .collect::<RenderResult<Vec<_>>>()?;

        let records = key.records;
        let object_range = (records * size_of::<GpuObjectData>()).max(4) as u64;
        let args_range = (records * size_of::<GpuDrawArgs>()).max(4) as u64;
        let status_size = (records * size_of::<u32>()).max(4) as u64;
        let group_count_size = super::compact::group_count_bytes(records);
        let device_local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let per_frame = |bytes, usage| {
            (0..frames)
                .map(|_| alloc.create_buffer(bytes, usage, device_local))
                .collect::<RenderResult<Vec<_>>>()
        };
        let indirect_bufs = per_frame(
            super::compact::indirect_buffer_bytes(records),
            super::compact::INDIRECT_USAGE,
        )?;
        let status_bufs = per_frame(status_size, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        let group_counts = per_frame(group_count_size, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        super::compact::zero_indirect_buffers(
            device,
            ctx.commands.command_pool,
            ctx.hw.graphics_queue,
            &indirect_bufs,
        )?;

        let has_hiz = u32::from(key.hiz.is_some());
        let frame_sets = u32::saturating_from(frames);
        let sizes = prop_pool_sizes(frame_sets, has_hiz).build();
        let pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&sizes)
                    .max_sets(u32::saturating_from(ring) + 2 * frame_sets + has_hiz),
            )
            .map_err(|e| super::error::map_vk_result(e, "prop probe descriptor pool"))?;

        let bindings = ctx.global_bindings();
        let global_sets = alloc_descriptor_sets(
            device,
            pool.handle(),
            &vec![ctx.descriptors.global_set_layout.handle(); ring],
        )?;
        for (i, &set) in global_sets.iter().enumerate() {
            face_contents(&bindings, view_bufs[i].buffer(), i % frames).write(device, set);
        }
        let cull_sets = alloc_descriptor_sets(
            device,
            pool.handle(),
            &vec![kernels.set_layout.handle(); frames],
        )?;
        let compact_sets = alloc_descriptor_sets(
            device,
            pool.handle(),
            &vec![compact.set_layout.handle(); frames],
        )?;
        for f in 0..frames {
            let variants = ctx.cull.variant_buffers[f].buffer();
            let draw_args = ctx.cull.draw_args_buffers[f].buffer();
            SetWrites::new(cull_sets[f])
                .storage_buffer(0, ctx.cull.object_buffers[f].buffer(), object_range)
                .storage_buffer(1, draw_args, args_range)
                .storage_buffer(2, group_counts[f].buffer(), group_count_size)
                .storage_buffer(3, status_bufs[f].buffer(), status_size)
                .storage_buffer(4, variants, super::init::VARIANT_BUFFER_BYTES)
                .apply(device);
            super::compact::write_compact_set(
                device,
                compact_sets[f],
                &CompactBuffers {
                    variants: (variants, super::init::VARIANT_BUFFER_BYTES),
                    draw_args: (draw_args, args_range),
                    status: (status_bufs[f].buffer(), status_size),
                    group_counts: (group_counts[f].buffer(), group_count_size),
                    lists: DrawLists {
                        buffer: indirect_bufs[f].buffer(),
                        commands: records,
                    },
                },
            );
        }
        let (hiz_set, hiz_ubo) = match (ctx.cull.hiz.as_ref(), key.hiz) {
            (Some(hiz), Some(view)) => {
                let (set, ubo) = super::hiz::off_camera_read_set(
                    alloc,
                    device,
                    pool.handle(),
                    hiz.read_set_layout.handle(),
                    view,
                )?;
                (Some(set), Some(ubo))
            }
            _ => (None, None),
        };

        Ok(Self {
            key,
            plan,
            prefilter,
            _color: color,
            _depth: depth,
            framebuffers,
            frames,
            view_bufs,
            global_sets,
            cull_sets,
            compact_sets,
            indirect_bufs,
            _status_bufs: status_bufs,
            _group_counts: group_counts,
            hiz_set,
            _hiz_ubo: hiz_ubo,
            _pool: pool,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The pool holds every binding of every set the prop probe allocates.
    #[test]
    fn the_prop_probe_pool_covers_its_sets() {
        let faces = u32::saturating_from(CAPTURE_FACES);
        let cull = super::super::init::main_cull_set();
        let compact = super::super::compact::compact_set();
        for frames in [1, 2, 3] {
            assert!(prop_pool_sizes(frames, 1).covers(&[
                (&global_set(), faces * frames),
                (&cull, frames),
                (&compact, frames),
            ]));
        }
    }
}
