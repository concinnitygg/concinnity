//! The compute cull: the kernel and its per-frame sets, the draw-args,
//! indirect-command and cull-status buffers, and the Hi-Z pyramid it tests
//! against.

use ash::vk;
use concinnity_core::gfx::render_types;
use concinnity_core::render::backend_init::SceneData;
use concinnity_core::render::error::RenderResult;

use super::CullPlan;
use super::bindless::BindlessPass;
use crate::vulkan::context::{VkCullKernels, VkDescriptors, VkSceneAssets, VkTargets};
use crate::vulkan::descriptor_layout::Binding;
use crate::vulkan::init::InitGpu;
use crate::vulkan::pipeline::*;
use crate::vulkan::pipeline_desc::compute_pipeline;
use crate::vulkan::resources::{alloc_descriptor_sets, create_descriptor_set_layout};
use crate::vulkan::set_writes::SetWrites;

// The compute cull; `None`/empty when the bindless path is inactive.
pub(super) struct ComputeCull {
    // Phase 1 only; the two-pass build fills in `pipeline_phase2`.
    pub(super) kernels: Option<VkCullKernels>,
    pub(super) sets: Vec<vk::DescriptorSet>,
    pub(super) draw_args_buffers: Vec<crate::vulkan::allocator::PooledBuffer>,
    pub(super) indirect_buffers: Vec<crate::vulkan::allocator::PooledBuffer>,
    pub(super) status_buffers: Vec<crate::vulkan::allocator::PooledBuffer>,
    pub(super) hiz: Option<crate::vulkan::hiz::HiZResources>,
}

pub(super) struct ComputeInputs<'a> {
    pub(super) world: &'a SceneData<'a>,
    pub(super) scene: &'a VkSceneAssets,
    pub(super) targets: &'a VkTargets,
    pub(super) descriptors: &'a VkDescriptors,
    pub(super) plan: &'a CullPlan,
    pub(super) bindless: &'a BindlessPass,
    pub(super) shader_bucket_count: usize,
    pub(super) occlusion_two_pass: bool,
}

// `N` compute-stage SSBOs at bindings `0..N`, the shape of every cull kernel's
// set 0.
pub(super) fn cull_set_bindings<const N: usize>() -> [Binding; N] {
    std::array::from_fn(|b| {
        (
            b as u32,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::ShaderStageFlags::COMPUTE,
        )
    })
}

// Set 0 of the phase-1 and phase-2 kernels: object, draw-args,
// indirect-command and cull-status SSBOs.
pub(super) fn main_cull_set() -> [Binding; 4] {
    cull_set_bindings()
}

// Build the compute cull and its Hi-Z pyramid, and write the static instance
// records into the object and draw-args buffers.
pub(super) fn build_compute_cull(
    gpu: &InitGpu<'_>,
    inputs: ComputeInputs<'_>,
) -> RenderResult<ComputeCull> {
    let InitGpu {
        hw,
        command_pool,
        frames,
        hot_reload,
    } = *gpu;
    let (device, alloc, graphics_queue) = (&hw.device, &hw.alloc, hw.graphics_queue);
    let ComputeInputs {
        world,
        scene,
        targets,
        descriptors,
        plan,
        bindless,
        shader_bucket_count,
        occlusion_two_pass,
    } = inputs;
    let CullPlan {
        n_cull,
        n_instances,
        bindless_active,
        ..
    } = *plan;
    let object_buffers = &bindless.object_buffers;
    // Per-object cull-status buffers (one u32 each), built unconditionally
    // on the bindless cull path: phase-1 cull writes them (binding 3 of the
    // cull set), and phase-2 cull (two-pass occlusion) reads them. Always
    // present so the phase-1 kernel always has a valid binding; under
    // single-pass occlusion the values are simply never read. Device-local,
    // with TRANSFER_SRC so `cull_readback` can copy one back to the host.
    // Mirrors `directx/cull.rs`.
    let cull_status_buffers = if bindless_active {
        let status_size = n_cull as u64 * std::mem::size_of::<u32>() as u64;
        let mut bufs = Vec::with_capacity(frames);
        for _ in 0..frames {
            bufs.push(alloc.create_buffer(
                status_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?);
        }
        bufs
    } else {
        Vec::new()
    };

    // Compute cull: the cull compute pipeline + per-frame draw-args /
    // indirect-command buffers + descriptor sets. Built under the same
    // condition as the bindless pass: the compute kernel writes one
    // indirect draw command per build-time object, which the bindless main
    // pass issues with a single multiDrawIndexedIndirect.
    type CullPipelineResources = (
        Option<VkCullKernels>,
        Vec<vk::DescriptorSet>,
        Vec<crate::vulkan::allocator::PooledBuffer>,
        Vec<crate::vulkan::allocator::PooledBuffer>,
        Option<crate::vulkan::hiz::HiZResources>,
    );
    let (kernels, cull_sets, draw_args_buffers, indirect_buffers, hiz): CullPipelineResources =
        if bindless_active {
            // Set 0: object SSBO + draw-args SSBO + indirect-command SSBO +
            // cull-status SSBO (binding 3: phase-1 writes the per-object cull
            // outcome for two-pass occlusion; the phase-2 kernel reads it).
            let set_layout = create_descriptor_set_layout(device, &main_cull_set())?;

            // Hi-Z occlusion resources. Built under the same gating as the cull
            // pipeline; its `read_set_layout` becomes set 1 of the cull
            // pipeline (the Hi-Z image + per-frame CullHizParams UBO).
            let depth_views: Vec<vk::ImageView> =
                targets.depth_images.iter().map(|img| img.view).collect();
            let hiz = crate::vulkan::hiz::HiZResources::new(
                crate::vulkan::hiz::HiZDeviceCtx {
                    alloc,
                    device,
                    command_pool,
                    queue: graphics_queue,
                },
                crate::vulkan::hiz::HiZTarget {
                    width: targets.render_extent.width,
                    height: targets.render_extent.height,
                    depth_views: &depth_views,
                },
                targets.msaa_samples.as_raw(),
                frames,
                occlusion_two_pass,
                hot_reload,
            )?;

            let push_range = vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(CULL_PUSH_CONSTANT_BYTES);
            let layouts = [set_layout.handle(), hiz.read_set_layout.handle()];
            let pipeline_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&layouts)
                        .push_constant_ranges(std::slice::from_ref(&push_range)),
                )
                .map_err(|e| crate::vulkan::error::map_vk_result(e, "cull pipeline layout"))?;

            let cs = compile_cull_shader(hot_reload)?;
            let pipeline = compute_pipeline(device, pipeline_layout.handle(), &cs, "cull")?;

            // Per-frame GpuDrawArgs (host-visible, rebuilt each frame) and
            // indirect-command buffers (device-local, GPU-written). `n_cull`
            // covers the static objects plus the merged instances.
            let n = n_cull as u64;
            let object_buffer_size = n * std::mem::size_of::<render_types::GpuObjectData>() as u64;
            let draw_args_size = n * std::mem::size_of::<render_types::GpuDrawArgs>() as u64;
            // One `n_cull`-command region per shader bucket: the cull kernel writes
            // every record's slot in each region and the main pass issues one
            // indirect draw per region under that bucket's pipeline.
            let indirect_size = shader_bucket_count as u64
                * n
                * std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u64;
            let mut da_buffers = Vec::with_capacity(frames);
            let mut ind_buffers = Vec::with_capacity(frames);
            for _ in 0..frames {
                da_buffers.push(alloc.create_buffer(
                    draw_args_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )?);
                ind_buffers.push(alloc.create_buffer(
                    indirect_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )?);
            }

            // One cull set per frame: that frame's object / draw-args /
            // indirect-command buffers at bindings 0 / 1 / 2.
            let set_layouts: Vec<_> = (0..frames).map(|_| set_layout.handle()).collect();
            let sets =
                alloc_descriptor_sets(device, descriptors.descriptor_pool.handle(), &set_layouts)?;
            for (i, &set) in sets.iter().enumerate() {
                SetWrites::new(set)
                    .storage_buffer(0, object_buffers[i].buffer(), object_buffer_size)
                    .storage_buffer(1, da_buffers[i].buffer(), draw_args_size)
                    .storage_buffer(2, ind_buffers[i].buffer(), indirect_size)
                    .storage_buffer(
                        3,
                        cull_status_buffers[i].buffer(),
                        n * std::mem::size_of::<u32>() as u64,
                    )
                    .apply(device);
            }

            let kernels = VkCullKernels {
                pipeline,
                pipeline_phase2: None,
                pipeline_layout,
                set_layout,
            };
            (Some(kernels), sets, da_buffers, ind_buffers, Some(hiz))
        } else {
            (None, Vec::new(), Vec::new(), Vec::new(), None)
        };

    // GPU-driven instanced merge: write each instance's `GpuObjectData`
    // record (+ `GpuDrawArgs`) once into every frame buffer, after the
    // `n_objects` static records. Instances are placed at world load and
    // never move, so these records are static -- the per-frame static fill
    // (`build_object_buffer` / `build_draw_args_buffer`) writes only
    // `[0, n_objects)`, leaving the instance tail intact. Only runs when the
    // bindless cull buffers exist (the bindless pass is active with build-time
    // geometry) and the world declares instanced props. Mirrors
    // `directx/init/mod.rs`.
    if n_instances > 0 && !object_buffers.is_empty() {
        use concinnity_core::gfx::render_types::{
            GpuDrawArgs, GpuObjectData, draw_args_flags, instance_object_records,
        };
        let records =
            instance_object_records(&world.instanced_clusters, scene.textures.len() as u32);
        // Cluster base LOD slice (absolute indices, so `base_vertex = 0`),
        // which `build_draw_args_buffer` patches per frame for the clusters
        // that declare alternates. Every instance is visible + resident +
        // cullable, so its finite per-instance world AABB is frustum /
        // distance / Hi-Z tested independently by the cull kernel.
        let mut draw_args: Vec<GpuDrawArgs> = Vec::with_capacity(records.len());
        for cluster in &world.instanced_clusters {
            for _ in &cluster.instances {
                draw_args.push(GpuDrawArgs {
                    index_count: cluster.index_count as u32,
                    index_offset: cluster.index_offset as u32,
                    base_vertex: 0,
                    flags: draw_args_flags(true, true, true),
                });
            }
        }
        let n_objects = world.draw_objects.len();
        let obj_stride = std::mem::size_of::<GpuObjectData>();
        let da_stride = std::mem::size_of::<GpuDrawArgs>();
        for (obj_buf, da_buf) in object_buffers.iter().zip(draw_args_buffers.iter()) {
            obj_buf.write_slice(n_objects * obj_stride, &records);
            da_buf.write_slice(n_objects * da_stride, &draw_args);
        }
    }
    Ok(ComputeCull {
        kernels,
        sets: cull_sets,
        draw_args_buffers,
        indirect_buffers,
        status_buffers: cull_status_buffers,
        hiz,
    })
}
