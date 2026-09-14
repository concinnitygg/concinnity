//! GPU-driven rendering: cull record sizing, the bindless static pass, ray-traced
//! reflections and the reflection composite, the compute cull with its Hi-Z
//! pyramid, the GPU-driven shadow and G-buffer passes, the probe convolution
//! kernels, and two-pass occlusion.

use ash::vk;
use concinnity_core::bake::texture::TextureImage;
use concinnity_core::gfx::render_types::{self, clone_reserve};
use concinnity_core::gfx::transform::IDENTITY;
use concinnity_core::render::backend_init::{PostSettings, SceneData, WorldShader};
use concinnity_core::render::error::RenderResult;

use super::InitGpu;
use crate::vulkan::context::{
    HDR_FORMAT, VkCull, VkDescriptors, VkGeometry, VkRayTracing, VkSceneAssets, VkShadow, VkTargets,
};
use crate::vulkan::owned::{
    OwnedDescriptorPool, OwnedPipeline, OwnedPipelineLayout, OwnedRenderPass, OwnedSetLayout,
};
use crate::vulkan::pipeline::*;
use crate::vulkan::post::gbuffer::GbufferResources;
use crate::vulkan::post::reflection_composite::ReflectionCompositeResources;
use crate::vulkan::post::rt_reflections::RtReflectionsResources;
use crate::vulkan::probe_prefilter::ProbePrefilterPipelines;
use crate::vulkan::render_pass::create_main_render_pass_two_pass;
use crate::vulkan::resources::alloc_descriptor_sets;

// How many records the GPU-driven cull buffers hold and how the bindless texture
// pool is laid out, decided from the world before any of them is built.
pub(super) struct CullPlan {
    pub(super) n_instances: usize,
    pub(super) n_cull: usize,
    pub(super) bindless_active: bool,
    pub(super) bindless_pool_size: usize,
    pub(super) bindless_uab: bool,
}

pub(super) fn plan_cull(
    gpu: &InitGpu<'_>,
    world: &SceneData<'_>,
    textures: &[TextureImage],
    descriptors: &VkDescriptors,
) -> CullPlan {
    let hw = gpu.hw;
    let draw_objects = &world.draw_objects;
    // Instanced props fold into the GPU-driven bindless cull buffers: each
    // instance becomes a `GpuObjectData` record appended after the `n_objects`
    // static records (written once at init), so the object / draw-args /
    // indirect / cull-status buffers size for the combined `n_cull` count and
    // the cull kernel tests every instance independently. Skinned objects fold
    // in after the instances (a per-frame-rebuilt tail of `n_skinned` records),
    // so `n_cull` reserves their slots too. Mirrors `directx/init`.
    let n_instances: usize = world
        .instanced_clusters
        .iter()
        .map(|c| c.instances.len())
        .sum();
    // Runtime record reserve (`[n_objects + n_instances, +n_runtime)`): the
    // worst-case resident streamed-chunk window plus the runtime-clone cap,
    // between the instances and the skinned tail; resident chunks fold in per
    // frame. 0 for a non-voxel world.
    let n_cull = draw_objects.len()
        + n_instances
        + world.n_chunk_max
        + clone_reserve(draw_objects.len())
        + world.n_skinned;

    // GPU-driven static pass: active when there is anything to drive --
    // build-time static geometry, instances, streamed chunks, or skinned
    // meshes (`n_cull > 0`). A pure-voxel world has no build-time geometry but
    // folds its chunks here. A world default Shader drives it through bucket
    // 0, built from the world's own bindless pair. Its texture pool is the
    // deduplicated [albedo..] ++ [normal-map..] image set; the helper derives
    // the same value from the texture table so the export-time precompile
    // matches.
    let bindless_active = n_cull > 0;
    // The pool is sized to the world's own texture table, which keeps every
    // index in range by construction. The shaders declare the array unsized,
    // so this length reaches the descriptor layout and nothing else: it is not
    // part of any source text and cannot make a program miss its precompiled
    // artifact.
    let bindless_pool_size = if bindless_active {
        crate::vulkan::builtins::world_pool_size(textures.len())
    } else {
        0
    };
    // The texture pool's length is the world's texture table, so it cannot be
    // clamped to the device's per-stage sampler headroom the way the probe
    // cube array is. Where it does not fit, its set layout is declared
    // update-after-bind, which moves it off `maxPerStageDescriptorSamplers`
    // (16 on MoltenVK) and onto the update-after-bind limit (1024 there). This
    // reshapes the layout, its binding flags, and the descriptor pool it is
    // allocated from, so it is resolved once here. Desktop drivers report six
    // figures and always stay on the plain path.
    let max_per_stage_samplers = super::descriptors::max_per_stage_samplers(hw);
    let pool_overflows_samplers = bindless_active
        && crate::vulkan::descriptor_layout::bindless_pool_needs_update_after_bind(
            max_per_stage_samplers,
            descriptors.probe_cube_count,
            bindless_pool_size as u32,
            descriptors.global_update_after_bind,
        );
    if pool_overflows_samplers && !hw.update_after_bind {
        tracing::warn!(
            "bindless texture pool: {bindless_pool_size} samplers exceed the device's \
             per-stage budget ({max_per_stage_samplers}) and update-after-bind is \
             unavailable"
        );
    }
    let bindless_uab = pool_overflows_samplers && hw.update_after_bind;
    CullPlan {
        n_instances,
        n_cull,
        bindless_active,
        bindless_pool_size,
        bindless_uab,
    }
}

pub(super) struct BindlessInputs<'a> {
    pub(super) world_shaders: &'a [WorldShader<'a>],
    pub(super) plan: &'a CullPlan,
    pub(super) occlusion_two_pass: bool,
    pub(super) descriptors: &'a VkDescriptors,
    pub(super) targets: &'a VkTargets,
    pub(super) scene: &'a VkSceneAssets,
    pub(super) swapchain_format: vk::Format,
}

// Build the bindless static pass. The cull state starts here; the later
// GPU-driven stages fill in the compute cull, shadow, G-buffer and two-pass
// parts.
pub(super) fn build_bindless_pass(
    gpu: &InitGpu<'_>,
    inputs: BindlessInputs<'_>,
) -> RenderResult<VkCull> {
    let InitGpu {
        hw,
        frames,
        hot_reload,
        ..
    } = *gpu;
    let (device, alloc) = (&hw.device, &hw.alloc);
    let BindlessInputs {
        world_shaders,
        plan,
        occlusion_two_pass,
        descriptors,
        targets,
        scene,
        swapchain_format,
    } = inputs;
    let CullPlan {
        n_cull,
        bindless_active,
        bindless_pool_size,
        bindless_uab,
        ..
    } = *plan;
    let probe_cube_count = descriptors.probe_cube_count;
    // Bindless static pass: bindless static main pass resources. A dedicated
    // set layout (set 1: SSBO + bindless texture pool), pipeline layout,
    // pipeline, per-frame GpuObjectData storage buffers, and one descriptor
    // set per frame. `None`/empty when the bindless pass is inactive.
    let (
        bindless_pipeline,
        bindless_pipeline_layout,
        bindless_set_layout,
        bindless_sets,
        object_buffers,
        bindless_main_spv,
    ) = if bindless_active {
        let set_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(bindless_pool_size as u32)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        // On a sampler-constrained device the pool binding is declared
        // update-after-bind so it budgets against the update-after-bind
        // sampler limit instead of the plain one. The pool is written once at
        // init, before any frame binds it, so nothing depends on the relaxed
        // update timing itself: this is purely how the layout is budgeted.
        let binding_flags = [
            vk::DescriptorBindingFlags::empty(),
            vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
        ];
        let mut flags_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);
        let mut set_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&set_bindings);
        if bindless_uab {
            set_info = set_info
                .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                .push_next(&mut flags_info);
        }
        let set_layout = device
            .create_descriptor_set_layout(&set_info)
            .map_err(|e| format!("bindless set layout: {e}"))?;

        let layouts = [descriptors.global_set_layout.handle(), set_layout.handle()];
        let pipeline_layout = device
            .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts))
            .map_err(|e| format!("bindless pipeline layout: {e}"))?;

        // The engine's own pair is the program for every bucket that
        // declares no Shader and the source of the Wireframe twin; bucket 0
        // takes the world default Shader's pair where it declares one.
        let engine_pair = compile_bindless_shaders(hot_reload, probe_cube_count)?;
        let pipeline = build_bucket_pipeline(
            device,
            BucketPipelineTargets {
                render_pass: targets.main_render_pass.handle(),
                layout: pipeline_layout.handle(),
                msaa_samples: targets.msaa_samples,
                swapchain_format,
                hot_reload,
                probe_count: probe_cube_count as usize,
            },
            0,
            world_shaders[0],
            &engine_pair,
        )?;

        // Per-frame GpuObjectData storage buffers, persistently mapped.
        // Sized for `n_cull` so the instanced merge's records fit past the
        // `n_objects` static prefix.
        let object_buffer_size =
            (n_cull * std::mem::size_of::<render_types::GpuObjectData>()) as u64;
        let mut buffers = Vec::with_capacity(frames);
        for _ in 0..frames {
            buffers.push(alloc.create_buffer(
                object_buffer_size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?);
        }

        // One bindless set per frame: binding 0 = that frame's SSBO,
        // binding 1 = the shared pool ([albedo views..] ++ [normal..]).
        let set_layouts: Vec<_> = (0..frames).map(|_| set_layout.handle()).collect();
        let sets =
            alloc_descriptor_sets(device, descriptors.descriptor_pool.handle(), &set_layouts)?;
        // Every slot the layout declares has to be written, and at the
        // ceiling there are more of them than the world fills: the world's
        // textures, then the reserved fallbacks (flat-normal, white), then
        // white again across the unused tail so a slot the shader can index
        // still names a live view. Mirrors the Metal pool's fill. Sizing
        // guarantees the world fits, so this only ever pads.
        let mut pool_infos: Vec<vk::DescriptorImageInfo> = scene
            .textures
            .iter()
            .chain(scene.fallback_textures.iter())
            .map(|img| {
                vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(img.view)
                    .sampler(scene.linear_sampler.handle())
            })
            .collect();
        if let Some(&tail) = pool_infos.last() {
            pool_infos.resize(bindless_pool_size, tail);
        }
        for (i, &set) in sets.iter().enumerate() {
            let buf_info = vk::DescriptorBufferInfo::default()
                .buffer(buffers[i].buffer())
                .offset(0)
                .range(object_buffer_size);
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&buf_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&pool_infos),
            ];
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }

        (
            Some(pipeline),
            Some(pipeline_layout),
            Some(set_layout),
            sets,
            buffers,
            engine_pair,
        )
    } else {
        (
            None,
            None,
            None,
            Vec::new(),
            Vec::new(),
            (Vec::new(), Vec::new()),
        )
    };
    Ok(VkCull {
        bindless_pipeline,
        bindless_pipeline_layout,
        bindless_set_layout,
        bindless_pool_size,
        bindless_update_after_bind: bindless_uab,
        bucket_stride: n_cull,
        bindless_main_spv,
        bindless_sets,
        object_buffers,
        occlusion_two_pass,
        hiz_prev_view_proj: IDENTITY,
        ..Default::default()
    })
}

pub(super) struct RtInputs<'a> {
    pub(super) world: &'a SceneData<'a>,
    pub(super) geometry: &'a VkGeometry,
    pub(super) scene: &'a VkSceneAssets,
    pub(super) targets: &'a VkTargets,
    pub(super) gbuffer: Option<&'a GbufferResources>,
    pub(super) descriptors: &'a VkDescriptors,
    pub(super) cull: &'a VkCull,
    pub(super) post: &'a PostSettings,
    pub(super) rt_wanted: bool,
}

pub(super) struct RtResources {
    pub(super) state: VkRayTracing,
    pub(super) reflections: Option<RtReflectionsResources>,
    pub(super) composite: Option<ReflectionCompositeResources>,
    pub(super) seethrough_mesh_indices: Vec<usize>,
    pub(super) has_seethrough_meshes: bool,
}

pub(super) fn build_rt_reflections(
    gpu: &InitGpu<'_>,
    inputs: RtInputs<'_>,
) -> RenderResult<RtResources> {
    let InitGpu {
        hw,
        command_pool,
        frames,
        hot_reload,
    } = *gpu;
    let (device, alloc) = (&hw.device, &hw.alloc);
    let RtInputs {
        world,
        geometry,
        scene,
        targets,
        gbuffer,
        descriptors,
        cull,
        post,
        rt_wanted,
    } = inputs;
    let (hdr_resolve_images, render_extent) = (&targets.hdr_resolve_images, targets.render_extent);
    // Hardware ray-traced reflections: the scene acceleration structure +
    // the inline-`rayQueryEXT` reflection pass. Built only when the world
    // requested it AND the device exposed the ray-query extensions
    // (`rt_wanted`). Reuses the SSR pre-pass G-buffer (forced on earlier) for
    // the per-pixel surface point + normal, and the bindless pool (when live)
    // for textured hit shading. Graceful-fallback throughout: no resident
    // geometry, an AS build error, or a shader compile failure leaves both
    // `None` and the graph keeps `SsrResolve`. RT takes precedence over the
    // SSR resolve in the shared graph slot, which `ReflectionPath` settles once
    // the build outcome is known.
    // Layer 2 see-through glass is opt-in per `Material` (the `see_through`
    // arg, which implies `transparent`): see-through only looks right when the
    // space behind the glass is modeled. A material that is `transparent` but
    // NOT `see_through` renders as Layer 1 (opaque, low roughness, scene
    // reflections) = tinted reflective glass that hides the interior. This list
    // drives the transparent-pass producer, the opaque-pass skip and the
    // RT-BLAS exclude together.
    let seethrough_mesh_indices: Vec<usize> = world
        .draw_objects
        .iter()
        .enumerate()
        .filter(|(_, o)| o.material.transparent != 0 && o.material.see_through != 0)
        .map(|(i, _)| i)
        .collect();

    // Whether those meshes will be rerouted, decided here because the initial
    // BLAS is built before the transparent pass that owns the mesh producer.
    // It is the same predicate that producer is gated on. The one divergence
    // is a mesh-shader compile failure, which leaves the producer absent (and
    // logs): the meshes then render opaque but stay out of this BLAS until a
    // topology refresh re-reads `seethrough_meshes_enabled` and puts them back.
    let has_seethrough_meshes = !seethrough_mesh_indices.is_empty() && hw.rt_capable;

    let (rt_accel_opt, rt_opt) = if rt_wanted {
        match crate::vulkan::raytrace::build_rt_accel(
            crate::vulkan::raytrace::RtDeviceCtx {
                alloc,
                instance: &hw.instance,
                device,
                pd: hw.physical_device,
            },
            command_pool,
            hw.graphics_queue,
            crate::vulkan::raytrace::RtSceneGeometry {
                vertex_buffer: geometry.vertex_buffer.buffer(),
                index_buffer: geometry.index_buffer.buffer(),
                draw_objects: &world.draw_objects,
                clusters: &world.instanced_clusters,
                albedo_count: scene.textures.len(),
                total_vertices: world.vertices.len(),
                exclude_seethrough: has_seethrough_meshes,
            },
            frames,
            hot_reload,
        ) {
            Ok(Some(accel)) => {
                let hdr_views: Vec<vk::ImageView> =
                    hdr_resolve_images.iter().map(|i| i.view).collect();
                // RT reads the unified G-buffer pre-pass's per-frame
                // normal+depth + roughness (built earlier whenever any consumer
                // is on); `gbuffer` is `Some` here because RT forces the
                // pre-pass on.
                let gb = gbuffer
                    .as_ref()
                    .expect("RT forces the unified G-buffer pre-pass to exist");
                let nd_views = gb.normal_depth_views();
                let rough_views = gb.roughness_views();
                let (geom_buffer, geom_size) = accel.geom_table();
                match crate::vulkan::post::rt_reflections::RtReflectionsResources::new(
                    crate::vulkan::post::rt_reflections::RtBuild {
                        alloc,
                        device,
                        width: render_extent.width,
                        height: render_extent.height,
                        frames,
                    },
                    post.rt_reflections
                        .expect("rt_wanted implies rt_settings is Some"),
                    crate::vulkan::post::rt_reflections::RtStaticInputs {
                        vertex_buffer: geometry.vertex_buffer.buffer(),
                        index_buffer: geometry.index_buffer.buffer(),
                        hdr_resolve_views: &hdr_views,
                        gbuffer_views: &nd_views,
                        roughness_views: &rough_views,
                        prefilter_view: scene.env_map.prefilter.view,
                        cube_sampler: scene.cube_sampler.handle(),
                    },
                    crate::vulkan::post::rt_reflections::RtAccelHandles {
                        tlas: accel.tlas(),
                        geom_buffer,
                        geom_size,
                        deformed_verts: accel.deformed_verts(),
                        skinned_indices: accel.skinned_indices(),
                    },
                    crate::vulkan::post::rt_reflections::RtLayoutConfig {
                        bindless_set_layout: cull.bindless_set_layout.as_ref().map(|l| l.handle()),
                        global_set_layout: descriptors.global_set_layout.handle(),
                        probe_cube_count: descriptors.probe_cube_count,
                        pool_size: cull.bindless_pool_size,
                        hot_reload,
                    },
                ) {
                    Ok(rt) => (Some(accel), Some(rt)),
                    Err(e) => {
                        tracing::warn!(
                            "RT reflections pass build failed (falling back to SSR): {e}"
                        );
                        let mut accel = accel;
                        accel.destroy(device);
                        (None, None)
                    }
                }
            }
            Ok(None) => {
                tracing::info!(
                    "RT reflections requested but no resident triangle geometry to trace; \
                     using SSR"
                );
                (None, None)
            }
            Err(e) => {
                tracing::warn!("RT acceleration-structure build failed (falling back to SSR): {e}");
                (None, None)
            }
        }
    } else {
        (None, None)
    };
    let rt_active = rt_opt.is_some();
    // Reflection composite: built whenever a resolve feeds it. Both resolves
    // write radiance+weight into their output target; this blurs by roughness
    // and composites over the scene into its own output, which then replaces
    // the raw resolve output as the scene image every downstream pass samples.
    let composite_opt = if crate::vulkan::post::reflection_composite::ReflectionPath::new(
        post.ssr.is_some(),
        rt_active,
    )
    .composite
    {
        let gb = gbuffer
            .as_ref()
            .expect("a reflection path implies the unified G-buffer pre-pass");
        Some(
            crate::vulkan::post::reflection_composite::ReflectionCompositeResources::new(
                &gpu.upload(),
                render_extent.width,
                render_extent.height,
                frames,
                post.reflection_blur_scale,
                &crate::vulkan::post::reflection_composite::CompositeInputs::new(
                    hdr_resolve_images,
                    gb,
                ),
                hot_reload,
            )?,
        )
    } else {
        None
    };
    Ok(RtResources {
        state: VkRayTracing {
            accel: rt_accel_opt,
            dynamic_mode: post.rt_dynamic,
            skinned_geometry: post.rt_skinned_geometry,
            topology_dirty: false,
            static_vertex_count: world.vertices.len(),
        },
        reflections: rt_opt,
        composite: composite_opt,
        seethrough_mesh_indices,
        has_seethrough_meshes,
    })
}

pub(super) struct CullInputs<'a> {
    pub(super) world: &'a SceneData<'a>,
    pub(super) scene: &'a VkSceneAssets,
    pub(super) targets: &'a VkTargets,
    pub(super) descriptors: &'a VkDescriptors,
    pub(super) plan: &'a CullPlan,
}

// Build the compute cull (and its Hi-Z pyramid) into the cull state, and write
// the static instance records into its buffers.
pub(super) fn build_cull_pass(
    gpu: &InitGpu<'_>,
    cull: &mut VkCull,
    inputs: CullInputs<'_>,
) -> RenderResult<()> {
    let InitGpu {
        hw,
        command_pool,
        frames,
        hot_reload,
    } = *gpu;
    let (device, alloc, graphics_queue) = (&hw.device, &hw.alloc, hw.graphics_queue);
    let CullInputs {
        world,
        scene,
        targets,
        descriptors,
        plan,
    } = inputs;
    let CullPlan {
        n_cull,
        n_instances,
        bindless_active,
        ..
    } = *plan;
    let shader_bucket_count = 1 + cull.world_pipelines.len();
    let object_buffers = &cull.object_buffers;
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
        Option<OwnedPipeline>,
        Option<OwnedPipelineLayout>,
        Option<OwnedSetLayout>,
        Vec<vk::DescriptorSet>,
        Vec<crate::vulkan::allocator::PooledBuffer>,
        Vec<crate::vulkan::allocator::PooledBuffer>,
        Option<crate::vulkan::hiz::HiZResources>,
    );
    let (
        cull_pipeline,
        cull_pipeline_layout,
        cull_set_layout,
        cull_sets,
        draw_args_buffers,
        indirect_buffers,
        hiz,
    ): CullPipelineResources = if bindless_active {
        // Set 0: object SSBO + draw-args SSBO + indirect-command SSBO +
        // cull-status SSBO (binding 3: phase-1 writes the per-object cull
        // outcome for two-pass occlusion; the phase-2 kernel reads it).
        let set_bindings: Vec<_> = (0..4u32)
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let set_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&set_bindings),
            )
            .map_err(|e| format!("cull set layout: {e}"))?;

        // Hi-Z occlusion resources. Built under the same gating as the cull
        // pipeline; its `read_set_layout` becomes set 1 of the cull
        // pipeline (sampler2D Hi-Z + per-frame CullHizParams UBO).
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
            cull.occlusion_two_pass,
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
            .map_err(|e| format!("cull pipeline layout: {e}"))?;

        let cs = compile_cull_shader(hot_reload)?;
        let pipeline = create_cull_pipeline(device, pipeline_layout.handle(), &cs)?;

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
            let obj_info = vk::DescriptorBufferInfo::default()
                .buffer(object_buffers[i].buffer())
                .offset(0)
                .range(object_buffer_size);
            let arg_info = vk::DescriptorBufferInfo::default()
                .buffer(da_buffers[i].buffer())
                .offset(0)
                .range(draw_args_size);
            let cmd_info = vk::DescriptorBufferInfo::default()
                .buffer(ind_buffers[i].buffer())
                .offset(0)
                .range(indirect_size);
            let status_info = vk::DescriptorBufferInfo::default()
                .buffer(cull_status_buffers[i].buffer())
                .offset(0)
                .range(n * std::mem::size_of::<u32>() as u64);
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&obj_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&arg_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&cmd_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&status_info)),
            ];
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }

        (
            Some(pipeline),
            Some(pipeline_layout),
            Some(set_layout),
            sets,
            da_buffers,
            ind_buffers,
            Some(hiz),
        )
    } else {
        (None, None, None, Vec::new(), Vec::new(), Vec::new(), None)
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
    cull.cull_status_buffers = cull_status_buffers;
    cull.cull_pipeline = cull_pipeline;
    cull.cull_pipeline_layout = cull_pipeline_layout;
    cull.cull_set_layout = cull_set_layout;
    cull.cull_sets = cull_sets;
    cull.draw_args_buffers = draw_args_buffers;
    cull.indirect_buffers = indirect_buffers;
    cull.hiz = hiz;
    Ok(())
}

// Build the GPU-driven shadow pass into the cull state.
pub(super) fn build_shadow_cull(
    gpu: &InitGpu<'_>,
    cull: &mut VkCull,
    shadow: &VkShadow,
    descriptors: &VkDescriptors,
    plan: &CullPlan,
) -> RenderResult<()> {
    let InitGpu {
        hw,
        frames,
        hot_reload,
        ..
    } = *gpu;
    let (device, alloc) = (&hw.device, &hw.alloc);
    let (object_buffers, draw_args_buffers) = (&cull.object_buffers, &cull.draw_args_buffers);
    // GPU-driven shadow pass resources. Built when the bindless cull path is
    // active AND shadows are enabled: a frustum + distance only cull pipeline
    // (`SHADOW_CULL`, lean 3-SSBO set: objects + draw-args + this cascade's
    // indirect buffer), one indirect buffer + cull set per (frame, cascade),
    // and a depth-only bindless graphics pipeline (shadow-global set 0 + the
    // bindless GpuObjectData set 1 + a cascade-index push constant). Each
    // re-rendered cascade then runs one cull dispatch + one
    // `cmd_draw_indexed_indirect` (static + instance prefix) + one for the
    // skinned tail, replacing the CPU per-object shadow loop.
    type ShadowCullResources = (
        Option<OwnedPipeline>,
        Option<OwnedPipelineLayout>,
        Option<OwnedSetLayout>,
        Vec<Vec<vk::DescriptorSet>>,
        Option<OwnedPipeline>,
        Option<OwnedPipelineLayout>,
        Vec<Vec<crate::vulkan::allocator::PooledBuffer>>,
    );
    let (
        shadow_cull_pipeline,
        shadow_cull_pipeline_layout,
        shadow_cull_set_layout,
        shadow_cull_sets,
        shadow_bindless_pipeline,
        shadow_bindless_pipeline_layout,
        shadow_indirect_buffers,
    ): ShadowCullResources = if plan.bindless_active
        && shadow.pipeline.is_some()
        && let Some(bl_set_layout) = cull.bindless_set_layout.as_ref()
    {
        let cascades = render_types::NUM_SHADOW_CASCADES;
        // Lean shadow cull set layout: objects(0) + draw-args(1) + commands(2).
        let sc_bindings: Vec<_> = (0..3u32)
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let sc_set_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&sc_bindings),
            )
            .map_err(|e| format!("shadow cull set layout: {e}"))?;

        let sc_push = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(CULL_PUSH_CONSTANT_BYTES);
        let sc_layouts = [sc_set_layout.handle()];
        let sc_pl = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&sc_layouts)
                    .push_constant_ranges(std::slice::from_ref(&sc_push)),
            )
            .map_err(|e| format!("shadow cull pipeline layout: {e}"))?;
        let sc_spv = compile_shadow_cull_shader(hot_reload)?;
        let sc_pipeline = create_cull_pipeline(device, sc_pl.handle(), &sc_spv)?;

        // Depth-only bindless shadow graphics pipeline: shadow-global set 0 +
        // the bindless GpuObjectData set 1 + a cascade-index push constant.
        let sb_push = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(4);
        let shadow_global_set_layout = shadow
            .global_set_layout
            .as_ref()
            .expect("the layout stage builds the shadow global set layout");
        let sb_layouts = [shadow_global_set_layout.handle(), bl_set_layout.handle()];
        let sb_pl = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&sb_layouts)
                    .push_constant_ranges(std::slice::from_ref(&sb_push)),
            )
            .map_err(|e| format!("shadow bindless pipeline layout: {e}"))?;
        let sb_spv = compile_shadow_bindless_vs(hot_reload)?;
        let sb_pipeline =
            create_shadow_pipeline(device, shadow.render_pass.handle(), sb_pl.handle(), &sb_spv)?;

        // Per-(frame, cascade) indirect buffers + cull sets. Each cull set
        // binds this frame's object + draw-args SSBOs and this cascade's
        // indirect buffer; the cull dispatch for cascade `c` binds set
        // `[frame][c]`, and the cascade's draws read buffer `[frame][c]`.
        let n = plan.n_cull as u64;
        let object_buffer_size = n * std::mem::size_of::<render_types::GpuObjectData>() as u64;
        let draw_args_size = n * std::mem::size_of::<render_types::GpuDrawArgs>() as u64;
        let indirect_size = n * std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u64;
        let mut sc_indirect_bufs: Vec<Vec<crate::vulkan::allocator::PooledBuffer>> =
            Vec::with_capacity(frames);
        let mut sc_sets: Vec<Vec<vk::DescriptorSet>> = Vec::with_capacity(frames);
        for f in 0..frames {
            let mut bufs = Vec::with_capacity(cascades);
            for _ in 0..cascades {
                bufs.push(alloc.create_buffer(
                    indirect_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )?);
            }
            let set_layouts: Vec<_> = (0..cascades).map(|_| sc_set_layout.handle()).collect();
            let sets =
                alloc_descriptor_sets(device, descriptors.descriptor_pool.handle(), &set_layouts)?;
            for (c, &set) in sets.iter().enumerate() {
                let obj_info = vk::DescriptorBufferInfo::default()
                    .buffer(object_buffers[f].buffer())
                    .offset(0)
                    .range(object_buffer_size);
                let arg_info = vk::DescriptorBufferInfo::default()
                    .buffer(draw_args_buffers[f].buffer())
                    .offset(0)
                    .range(draw_args_size);
                let cmd_info = vk::DescriptorBufferInfo::default()
                    .buffer(bufs[c].buffer())
                    .offset(0)
                    .range(indirect_size);
                let writes = [
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&obj_info)),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&arg_info)),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(2)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&cmd_info)),
                ];
                // SAFETY: `writes` and the buffer/image infos it borrows are live for the call,
                // and every set and resource it names belongs to this device.
                unsafe { device.update_descriptor_sets(&writes, &[]) };
            }
            sc_indirect_bufs.push(bufs);
            sc_sets.push(sets);
        }

        (
            Some(sc_pipeline),
            Some(sc_pl),
            Some(sc_set_layout),
            sc_sets,
            Some(sb_pipeline),
            Some(sb_pl),
            sc_indirect_bufs,
        )
    } else {
        (None, None, None, Vec::new(), None, None, Vec::new())
    };
    cull.shadow_cull_pipeline = shadow_cull_pipeline;
    cull.shadow_cull_pipeline_layout = shadow_cull_pipeline_layout;
    cull._shadow_cull_set_layout = shadow_cull_set_layout;
    cull.shadow_cull_sets = shadow_cull_sets;
    cull.shadow_bindless_pipeline = shadow_bindless_pipeline;
    cull.shadow_bindless_pipeline_layout = shadow_bindless_pipeline_layout;
    cull.shadow_indirect_buffers = shadow_indirect_buffers;
    Ok(())
}

// Build the GPU-driven G-buffer pre-pass into the cull state, and the probe
// convolution kernels the bake needs, which it returns.
pub(super) fn build_gbuffer_pass(
    gpu: &InitGpu<'_>,
    cull: &mut VkCull,
    gbuffer: Option<&GbufferResources>,
    descriptors: &VkDescriptors,
    plan: &CullPlan,
) -> RenderResult<Option<ProbePrefilterPipelines>> {
    let InitGpu {
        hw,
        frames,
        hot_reload,
        ..
    } = *gpu;
    let (device, alloc) = (&hw.device, &hw.alloc);
    let gbuffer_active = plan.bindless_active && gbuffer.is_some();
    // GPU-driven G-buffer pre-pass resources. Built when the bindless cull
    // path is active AND the G-buffer is enabled: a 3-MRT bindless pipeline +
    // the per-frame model-history ring and the snapshot kernel that fills it,
    // drawn by reusing the main pass's per-frame indirect buffer (camera
    // frustum, NO extra cull dispatch).
    type GbufferBindlessResources = (
        Option<OwnedPipeline>,
        Option<OwnedPipelineLayout>,
        Option<OwnedSetLayout>,
        Vec<vk::DescriptorSet>,
        Vec<crate::vulkan::allocator::PooledBuffer>,
        Option<crate::vulkan::post::gbuffer::ModelHistoryPipeline>,
    );
    let (
        gbuffer_bindless_pipeline,
        gbuffer_bindless_pipeline_layout,
        gbuffer_set_layout,
        gbuffer_sets,
        prev_model_buffers,
        model_history,
    ): GbufferBindlessResources = if let (true, Some(gb), Some(bl_set_layout)) =
        (gbuffer_active, gbuffer, cull.bindless_set_layout.as_ref())
    {
        let gbb = crate::vulkan::post::gbuffer::build_gbuffer_bindless(
            crate::vulkan::post::gbuffer::GbufferDeviceCtx { alloc, device },
            crate::vulkan::post::gbuffer::GbufferBindlessDescriptors {
                descriptor_pool: descriptors.descriptor_pool.handle(),
                bindless_set_layout: bl_set_layout.handle(),
            },
            crate::vulkan::post::gbuffer::GbufferBindlessRecords {
                object_buffers: &cull.object_buffers,
                draw_args_buffers: &cull.draw_args_buffers,
            },
            gb,
            crate::vulkan::post::gbuffer::GbufferBindlessScene {
                n_cull: plan.n_cull,
                frames,
            },
            hot_reload,
        )?;
        (
            Some(gbb.pipeline),
            Some(gbb.pipeline_layout),
            Some(gbb.set_layout),
            gbb.sets,
            gbb.prev_model_buffers,
            Some(gbb.history),
        )
    } else {
        (None, None, None, Vec::new(), Vec::new(), None)
    };

    // The reflection-probe convolution kernels, under the same gate the bake
    // itself needs: a probe capture renders through the bindless GPU cull, so a
    // world without the cull pipeline never bakes one and never needs them.
    let probe_prefilter = match cull.cull_pipeline.is_some() {
        true => {
            Some(crate::vulkan::probe_prefilter::ProbePrefilterPipelines::new(device, hot_reload)?)
        }
        false => None,
    };
    cull.gbuffer_bindless_pipeline = gbuffer_bindless_pipeline;
    cull.gbuffer_bindless_pipeline_layout = gbuffer_bindless_pipeline_layout;
    cull._gbuffer_set_layout = gbuffer_set_layout;
    cull.gbuffer_sets = gbuffer_sets;
    cull.prev_model_buffers = prev_model_buffers;
    cull.model_history = model_history;
    Ok(probe_prefilter)
}

// Build the two-pass Hi-Z occlusion resources into the cull state.
pub(super) fn build_two_pass_cull(
    gpu: &InitGpu<'_>,
    cull: &mut VkCull,
    plan: &CullPlan,
    msaa_samples: vk::SampleCountFlags,
) -> RenderResult<()> {
    let InitGpu {
        hw,
        frames,
        hot_reload,
        ..
    } = *gpu;
    let (device, alloc) = (&hw.device, &hw.alloc);
    let n_frames = frames as u32;
    let shader_bucket_count = 1 + cull.world_pipelines.len();
    let (object_buffers, draw_args_buffers, cull_status_buffers) = (
        &cull.object_buffers,
        &cull.draw_args_buffers,
        &cull.cull_status_buffers,
    );
    // Two-pass Hi-Z occlusion resources. Built only when the world
    // requested `occlusion_two_pass` AND the bindless cull path is active:
    // the phase-2 cull pipeline (`main_phase2`, same layout as phase 1), a
    // second set of per-frame indirect buffers `Cull2` writes / `Main2`
    // reads, a dedicated descriptor pool + per-frame phase-2 cull sets
    // (bindings 0/1/2/3 = object / draw-args / second-indirect /
    // cull-status), and the phase-1/phase-2 main render passes. The Hi-Z
    // phase-2 cull-read sets live inside `HiZResources` (built with the cull
    // pass when `occlusion_two_pass`). Mirrors `directx/init/pipelines.rs`.
    type TwoPassCullResources = (
        Option<OwnedPipeline>,
        Vec<vk::DescriptorSet>,
        Option<OwnedDescriptorPool>,
        Vec<crate::vulkan::allocator::PooledBuffer>,
        Option<OwnedRenderPass>,
        Option<OwnedRenderPass>,
    );
    let (
        cull_pipeline_phase2,
        cull_sets2,
        two_pass_pool,
        indirect_buffers2,
        main_render_pass_phase1,
        main_render_pass_phase2,
    ): TwoPassCullResources = if let (Some(set_layout), Some(pipeline_layout)) = (
        cull.cull_set_layout.as_ref(),
        cull.cull_pipeline_layout.as_ref(),
    ) && cull.occlusion_two_pass
    {
        let n = plan.n_cull as u64;
        let object_buffer_size = n * std::mem::size_of::<render_types::GpuObjectData>() as u64;
        let draw_args_size = n * std::mem::size_of::<render_types::GpuDrawArgs>() as u64;
        // Bucket-expanded exactly like the phase-1 buffers: `Main2` issues the
        // same per-bucket regions over this buffer.
        let indirect_size = shader_bucket_count as u64
            * n
            * std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u64;
        let status_size = n * std::mem::size_of::<u32>() as u64;

        // Phase-2 cull pipeline (`main_phase2` entry, shared layout).
        let cs2 = compile_cull_shader_phase2(hot_reload)?;
        let pipeline2 = create_cull_pipeline(device, pipeline_layout.handle(), &cs2)?;

        // Second indirect-command buffers (device-local, GPU-written).
        let mut ind2_buffers = Vec::with_capacity(frames);
        for _ in 0..frames {
            ind2_buffers.push(alloc.create_buffer(
                indirect_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?);
        }

        // Dedicated descriptor pool for the per-frame phase-2 cull sets
        // (4 storage buffers each), kept off the shared pool's exact sizing.
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(4 * n_frames);
        let pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(std::slice::from_ref(&pool_size))
                    .max_sets(n_frames),
            )
            .map_err(|e| format!("two-pass cull descriptor pool: {e}"))?;
        let set_layouts2: Vec<_> = (0..frames).map(|_| set_layout.handle()).collect();
        let sets2 = alloc_descriptor_sets(device, pool.handle(), &set_layouts2)?;
        for (i, &set) in sets2.iter().enumerate() {
            let obj_info = vk::DescriptorBufferInfo::default()
                .buffer(object_buffers[i].buffer())
                .offset(0)
                .range(object_buffer_size);
            let arg_info = vk::DescriptorBufferInfo::default()
                .buffer(draw_args_buffers[i].buffer())
                .offset(0)
                .range(draw_args_size);
            // Binding 2: the *second* indirect buffer (Cull2 writes it).
            let cmd_info = vk::DescriptorBufferInfo::default()
                .buffer(ind2_buffers[i].buffer())
                .offset(0)
                .range(indirect_size);
            // Binding 3: the cull-status buffer (phase 1 wrote it; read here).
            let status_info = vk::DescriptorBufferInfo::default()
                .buffer(cull_status_buffers[i].buffer())
                .offset(0)
                .range(status_size);
            let infos = [obj_info, arg_info, cmd_info, status_info];
            let writes: Vec<_> = infos
                .iter()
                .enumerate()
                .map(|(b, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(b as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();
            // SAFETY: `writes` and the buffer/image infos it borrows are live for the call, and
            // every set and resource it names belongs to this device.
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }

        // Phase-1 (STORE MSAA color) + phase-2 (LOAD color + depth) main
        // render passes, both compatible with the existing framebuffers.
        let rp1 = create_main_render_pass_two_pass(device, HDR_FORMAT, msaa_samples, false)?;
        let rp2 = create_main_render_pass_two_pass(device, HDR_FORMAT, msaa_samples, true)?;

        (
            Some(pipeline2),
            sets2,
            Some(pool),
            ind2_buffers,
            Some(rp1),
            Some(rp2),
        )
    } else {
        (None, Vec::new(), None, Vec::new(), None, None)
    };
    cull.cull_pipeline_phase2 = cull_pipeline_phase2;
    cull.cull_sets2 = cull_sets2;
    cull._two_pass_pool = two_pass_pool;
    cull.indirect_buffers2 = indirect_buffers2;
    cull.main_render_pass_phase1 = main_render_pass_phase1;
    cull.main_render_pass_phase2 = main_render_pass_phase2;
    Ok(())
}
