//! VkContext construction. `build` destructures the backend inputs, calls each
//! stage in Vulkan object creation order, and assembles the context from the
//! bundles they return:
//!
//!   bootstrap.rs    window, instance, debug messenger, surface, device, swapchain.
//!   commands.rs     command pools and buffers, timestamp reset, sync objects.
//!   targets.rs      shadow maps, render passes, HDR attachments, bloom chain.
//!   scene_data.rs   area lights, textures, samplers, geometry, uniforms, IBL.
//!   descriptors.rs  set and pipeline layouts, the descriptor pool and sets.
//!   pipelines.rs    shadow, text, composite, bloom and material pipelines.
//!   gpu_driven.rs   bindless pass, RT reflections, compute cull, Hi-Z.
//!   effects.rs      upscaler, screen-space passes, TAA, world effects.

use ash::vk;
use concinnity_core::gfx::lod;
use concinnity_core::gfx::profile;
use concinnity_core::gfx::render_types::*;
use concinnity_core::gfx::transform::IDENTITY;
use concinnity_core::render::backend_init;
use concinnity_core::render::decal;
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::reflection_probe;
use concinnity_core::render::skinned_slots;
use concinnity_core::render::slot_rewrites;

use super::context::*;
use super::texture::*;
use crate::vulkan::owned::VkDevice;

pub(in crate::vulkan) mod bootstrap;
mod commands;
mod descriptors;
mod effects;
mod gpu_driven;
mod pipelines;
mod scene_data;
mod targets;

use bootstrap::{SharedHardware, VkReuse};

// The device handles every init stage creates resources through.
struct InitGpu<'a> {
    instance: &'a ash::Instance,
    device: &'a VkDevice,
    physical_device: vk::PhysicalDevice,
    alloc: &'a super::allocator::DeviceAllocator,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    frames: usize,
    hot_reload: bool,
}

// The per-frame lighting, shadow and environment resources the global set binds,
// which fog, raymarched volumes and planar reflections bind as well.
struct GlobalBindings<'a> {
    view_ubo_buffers: &'a [super::allocator::PooledBuffer],
    view_ubo_size: u64,
    probe_set_ubo_buffers: &'a [super::allocator::PooledBuffer],
    probe_set_ubo_size: u64,
    light_ubo_buffers: &'a [super::allocator::PooledBuffer],
    light_ubo_size: u64,
    shadow_ubos: &'a [super::allocator::PooledBuffer],
    shadow_ubo_size: u64,
    local_light_buffer: &'a super::allocator::PooledBuffer,
    local_light_buffer_size: u64,
    light_cull: &'a super::light_cull::VkLightCull,
    shadow_map: &'a GpuImage,
    shadow_sampler: &'a super::owned::OwnedSampler,
    spot_shadow: &'a VkSpotShadow,
    area_light_buffer: &'a super::allocator::PooledBuffer,
    ltc_matrix_image: &'a GpuImage,
    ltc_magnitude_image: &'a GpuImage,
    ltc_sampler: &'a super::owned::OwnedSampler,
    env_map: &'a EnvironmentMapTextures,
    cube_sampler: &'a super::owned::OwnedSampler,
    transient_pool: &'a super::transient_pool::TransientImagePool,
    ssao_white: &'a GpuImage,
    linear_sampler: &'a super::owned::OwnedSampler,
}

impl VkContext {
    // Construct a fresh context, acquiring its own OS window + Vulkan
    // instance / device / surface / swapchain.
    pub(crate) fn new(init: backend_init::BackendInit<'_>) -> RenderResult<Self> {
        Self::build(init, None)
    }

    // Construct from the assembled backend inputs (see
    // `concinnity_core::render::backend_init::BackendInit` for per-field docs); the
    // Vulkan-specific behavior of each input is documented inline below.
    //
    // `reuse` is `Some` only on a live editor `reload_world` (see
    // `apply_world_reload`): the shared hardware (window / instance / device /
    // surface / swapchain / queues / capabilities / debug messenger / timestamp
    // pool) is inherited from the outgoing context instead of acquired fresh,
    // and every per-world resource below is rebuilt on it. `None` acquires it
    // all fresh, the normal launch path.
    fn build(init: backend_init::BackendInit<'_>, reuse: Option<VkReuse>) -> RenderResult<Self> {
        use concinnity_core::render::backend_init::{
            BackendInit, MediaPayloads, PostSettings, SceneData, ShadowParams, WorldFx, WorldShader,
        };
        let BackendInit {
            window,
            validation,
            frames_in_flight,
            vsync,
            clear_color,
            hot_reload,
            // Vulkan retains the presented swapchain index unconditionally, so
            // capture needs no arming here.
            capture: _,
            scene:
                SceneData {
                    vertices,
                    indices,
                    draw_objects,
                    instanced_clusters,
                    // Skinned draw-object count, threaded to size the shared
                    // GPU-cull buffers' reserved skinned tail at init
                    // (`n_objects + n_instances + n_skinned`). The skinned
                    // geometry is uploaded later by `upload_skinned`, which sets
                    // the live `self.draw.n_skinned`; this only reserves capacity.
                    n_skinned,
                    // Reserves a chunk record region in the shared cull buffers
                    // at init (`[n_objects + n_instances, +n_runtime)`);
                    // resident chunks fold into the indirect path each frame.
                    // Sets the live `self.draw.n_runtime`.
                    n_chunk_max,
                },
            shaders: world_shaders,
            media:
                MediaPayloads {
                    textures,
                    text_atlases,
                    env_map_bytes,
                    color_lut_bytes,
                },
            light_uniforms,
            // Per-scene local lights uploaded once into a static SSBO below
            // (global set 0 binding 9).
            local_lights,
            spot_shadows,
            area_lights,
            shadows:
                ShadowParams {
                    map_size: shadow_map_size,
                    update: shadow_update,
                    distance: shadow_distance,
                    cascades: shadow_cascades,
                },
            // Clamped to the device limit where the sampler is built below.
            anisotropy,
            planar_planes,
            post:
                PostSettings {
                    post_process: post_tunables,
                    taa_enabled,
                    hdr_samples,
                    ssao: ssao_settings,
                    ssr: ssr_settings,
                    ssgi: ssgi_settings,
                    rt_reflections: rt_settings,
                    rt_dynamic: rt_dynamic_mode,
                    rt_skinned_geometry,
                    reflection_blur_scale,
                    auto_exposure: auto_exposure_settings,
                    auto_exposure_bias_ev,
                    hdr_display,
                    hdr_pq,
                    temporal_upscaling,
                    upscale_scale,
                    // Only FSR is available on Vulkan (DLSS / XeSS are
                    // DirectX-only); a DX-only request logs a note and uses FSR.
                    upscale_backend,
                    occlusion_two_pass,
                },
            fx:
                WorldFx {
                    decals,
                    particles,
                    fog: fog_settings,
                    water_surfaces,
                    glass_panels,
                    sdf_volumes,
                },
            requirements: _,
        } = init;
        // Entry 0 is the world default program; entries 1.. are the
        // material-referenced shader buckets (see `world_shaders.rs`).
        let &WorldShader {
            programs: world_programs,
            // The world default program is never deferred (bucket 0 always
            // decodes at init); only the material-referenced buckets can be.
            deferred: _,
        } = world_shaders
            .first()
            .ok_or_else(|| "BackendInit carried no shaders".to_string())?;
        let (title, width, height, title_bar) = (
            window.title.as_str(),
            window.width,
            window.height,
            window.title_bar,
        );
        // Record this (main) thread so the `RenderBackend` mutation entry points
        // can `debug_assert_main_thread` against it; the Send invariant rests on
        // the context being touched from this thread alone.
        super::context::record_main_thread();

        // Temporal upscaling (FSR) consumes the velocity pre-pass's
        // render-resolution motion + depth, which TaaResources owns, so force
        // the TAA stack built when upscaling is on (the TAA *resolve* is still
        // dropped from the frame graph; only the velocity pre-pass is reused).
        let taa_enabled = taa_enabled || temporal_upscaling;
        let frames = frames_in_flight.max(1);

        // Acquire the shared hardware (fresh launch), or inherit it from the
        // outgoing context on a live editor reload. Every per-world resource
        // further below is built on it regardless of which path produced it.
        let SharedHardware {
            window,
            entry,
            instance,
            device,
            physical_device,
            surface,
            surface_loader,
            graphics_queue,
            present_queue,
            graphics_family,
            swapchain_loader,
            swapchain,
            swapchain_images,
            swapchain_format,
            swapchain_extent,
            swapchain_image_views,
            max_msaa_samples,
            hdr_mode,
            memory_budget_supported,
            rt_capable,
            update_after_bind,
            device_local_heaps,
            timestamp_query_pool,
            timestamp_period,
            alloc,
        } = match reuse {
            Some(r) => r.into_shared()?,
            None => bootstrap::acquire_hardware(bootstrap::HardwareRequest {
                title,
                width,
                height,
                title_bar,
                validation,
                frames,
                vsync,
                hdr_display,
                hdr_pq,
                temporal_upscaling,
                upscale_backend,
            })?,
        };

        // `msaa_samples` off the shared hardware is the device's ceiling for the
        // HDR format; the resolved setting is what the world actually asks for.
        // A temporal technique resolves it to one sample, which drops the
        // resolve attachment from every render pass and makes the color image
        // the scene spine. Applied after the reuse branch so a live editor
        // reload picks up a world whose AA mode differs from the outgoing one.
        let msaa_samples = super::device::resolve_sample_count(max_msaa_samples, hdr_samples);
        tracing::info!("vulkan HDR target: {}x MSAA", msaa_samples.as_raw().max(1));

        // Pair the authored tunables with the resolved HDR mode (freshly
        // negotiated or inherited on a reload), which drives the composite
        // shader's `hdr_output > 0.5` branch and its in-branch `pq_output`
        // encode flag. Mirrors `DxContext::new`.
        let post_process = hdr_mode.post_process_params(post_tunables);

        let command_pool = commands::create_command_pool(&device, graphics_family)?;
        let gpu = InitGpu {
            instance: &instance,
            device: &device,
            physical_device,
            alloc: &alloc,
            command_pool,
            queue: graphics_queue,
            frames,
            hot_reload,
        };
        let effects::Upscale {
            upscale,
            render_extent,
        } = effects::build_upscale(
            &gpu,
            swapchain_extent,
            temporal_upscaling,
            upscale_scale,
            upscale_backend,
        )?;

        commands::reset_timestamp_queries(&gpu, timestamp_query_pool)?;

        let effective_shadow_size = shadow_map_size;
        let shadow_map = targets::build_shadow_map(&gpu, effective_shadow_size)?;

        let scene_data::AreaLights {
            area_light_buffer,
            ltc_matrix_image,
            ltc_magnitude_image,
            ltc_sampler,
        } = scene_data::build_area_lights(&gpu, &area_lights)?;

        let targets::SpotShadowMap {
            slice_size: spot_shadow_slice_size,
            map: spot_shadow_map,
        } = targets::build_spot_shadow_map(&gpu, effective_shadow_size, &spot_shadows)?;

        let scene_data::SceneTextures {
            gpu_textures,
            gpu_fallbacks,
            gpu_text_atlases,
            linear_sampler,
            shadow_sampler,
            text_sampler,
            composite_sampler,
        } = scene_data::build_textures_and_samplers(&gpu, textures, &text_atlases, anisotropy)?;

        let bloom_on = post_process.bloom_intensity > 0.0;
        let rt_wanted = rt_settings.is_some() && rt_capable;
        // The unified pre-pass exists when any screen-space consumer needs it.
        // Derived once here rather than restated at the build site below: the
        // pool gate and the feature gate disagreeing would mean the pool places
        // no images while the feature expects them, or the reverse.
        let gbuffer_enabled = taa_enabled
            || ssao_settings.is_some()
            || ssr_settings.is_some()
            || ssgi_settings.is_some()
            || rt_wanted;
        let targets::RenderTargets {
            main_render_pass,
            shadow_render_pass,
            composite_render_pass,
            bloom_write_pass,
            bloom_blend_pass,
            color_images,
            depth_images,
            hdr_resolve_images,
            framebuffers,
            composite_framebuffers,
            transient_pool,
            gbuffer_pooled,
            bloom_mips,
            bloom_mip_extents,
            bloom_write_framebuffers,
            bloom_blend_framebuffers,
        } = targets::build_render_targets(
            &gpu,
            targets::RenderTargetConfig {
                msaa_samples,
                swapchain_format,
                swapchain_extent,
                swapchain_image_views: &swapchain_image_views,
                render_extent,
                ssao_enabled: ssao_settings.is_some(),
                bloom_on,
                gbuffer_enabled,
            },
        )?;

        let scene_data::SceneResources {
            vertex_buffer,
            index_buffer,
            vertex_buffer_bytes,
            index_buffer_bytes,
            view_ubo_size,
            light_ubo_size,
            shadow_ubo_size,
            local_light_buffer_size,
            view_ubo_buffers,
            probe_set_ubo_size,
            probe_set_ubo_buffers,
            light_ubo_buffers,
            shadow_ubos,
            local_light_buffer,
            shadow_light_dir,
            fog_sun_dir,
            fog_sun_color,
            shadow_uniforms,
            light_cull,
            cube_sampler,
            env_map,
            color_lut,
        } = scene_data::build_scene_resources(
            &gpu,
            scene_data::SceneInputs {
                vertices,
                indices,
                rt_capable,
                local_lights: &local_lights,
                light_uniforms: &light_uniforms,
                env_map_bytes,
                color_lut_bytes,
            },
        )?;

        let descriptors::DescriptorLayouts {
            max_per_stage_samplers,
            global_update_after_bind,
            probe_cube_count,
            global_set_layout,
            text_set_layout,
            shadow_global_set_layout,
            composite_set_layout,
            bloom_set_layout,
            shadow_pipeline_layout,
            text_pipeline_layout,
            composite_pipeline_layout,
            bloom_pipeline_layout,
        } = descriptors::build_layouts(&gpu, update_after_bind)?;

        let pipelines::MainPipelines {
            shadow_pipeline_opt,
            shadow_framebuffers_vec,
            spot_shadow,
            text_pipeline_opt,
            composite_pipeline,
            bloom_pipeline_prefilter,
            bloom_pipeline_downsample,
            bloom_pipeline_upsample,
        } = pipelines::build_main_pipelines(
            &gpu,
            pipelines::MainPipelineInputs {
                effective_shadow_size,
                shadow_render_pass: &shadow_render_pass,
                shadow_pipeline_layout: &shadow_pipeline_layout,
                shadow_global_set_layout: &shadow_global_set_layout,
                shadow_map: &shadow_map,
                spot_shadow_map,
                spot_shadow_slice_size,
                spot_shadows: &spot_shadows,
                gpu_text_atlases: &gpu_text_atlases,
                text_pipeline_layout: &text_pipeline_layout,
                composite_render_pass: &composite_render_pass,
                composite_pipeline_layout: &composite_pipeline_layout,
                bloom_write_pass: &bloom_write_pass,
                bloom_blend_pass: &bloom_blend_pass,
                bloom_pipeline_layout: &bloom_pipeline_layout,
            },
        )?;

        let effects::ScreenSpace {
            ssao_white,
            ssao_opt,
            ssr_authored,
            post_support,
            ssr_opt,
            gbuffer_opt,
            ssgi_opt,
        } = effects::build_screen_space(
            &gpu,
            effects::ScreenSpaceInputs {
                transient_pool: &transient_pool,
                gbuffer_pooled: &gbuffer_pooled,
                composite_sampler: &composite_sampler,
                cube_sampler: &cube_sampler,
                global_set_layout: &global_set_layout,
                probe_cube_count,
                render_extent,
                ssao_settings,
                ssr_settings,
                ssgi_settings,
                rt_wanted,
                gbuffer_enabled,
            },
        )?;

        let gpu_driven::CullPlan {
            n_instances,
            n_cull,
            bindless_active,
            bindless_pool_size,
            bindless_uab,
        } = gpu_driven::plan_cull(gpu_driven::CullPlanInputs {
            draw_objects: &draw_objects,
            instanced_clusters: &instanced_clusters,
            textures,
            n_chunk_max,
            n_skinned,
            max_per_stage_samplers,
            probe_cube_count,
            global_update_after_bind,
            update_after_bind,
        });

        let bindings = GlobalBindings {
            view_ubo_buffers: &view_ubo_buffers,
            view_ubo_size,
            probe_set_ubo_buffers: &probe_set_ubo_buffers,
            probe_set_ubo_size,
            light_ubo_buffers: &light_ubo_buffers,
            light_ubo_size,
            shadow_ubos: &shadow_ubos,
            shadow_ubo_size,
            local_light_buffer: &local_light_buffer,
            local_light_buffer_size,
            light_cull: &light_cull,
            shadow_map: &shadow_map,
            shadow_sampler: &shadow_sampler,
            spot_shadow: &spot_shadow,
            area_light_buffer: &area_light_buffer,
            ltc_matrix_image: &ltc_matrix_image,
            ltc_magnitude_image: &ltc_magnitude_image,
            ltc_sampler: &ltc_sampler,
            env_map: &env_map,
            cube_sampler: &cube_sampler,
            transient_pool: &transient_pool,
            ssao_white: &ssao_white,
            linear_sampler: &linear_sampler,
        };
        let descriptors::DescriptorSets {
            descriptor_pool,
            gbuffer_active,
            global_sets,
            shadow_global_sets,
        } = descriptors::build_sets(
            &gpu,
            descriptors::SetPoolInputs {
                instanced_clusters: &instanced_clusters,
                gpu_text_atlases: &gpu_text_atlases,
                bindless_active,
                has_gbuffer: gbuffer_opt.is_some(),
                has_shadow_pipeline: shadow_pipeline_opt.is_some(),
                bindless_pool_size,
                bindless_uab,
                probe_cube_count,
                global_update_after_bind,
                global_set_layout: &global_set_layout,
                shadow_global_set_layout: &shadow_global_set_layout,
            },
            &bindings,
        )?;
        let gpu_driven::BindlessPass {
            bindless_pipeline,
            bindless_pipeline_layout,
            bindless_set_layout,
            bindless_sets,
            object_buffers,
            bindless_main_spv,
        } = gpu_driven::build_bindless_pass(
            &gpu,
            gpu_driven::BindlessInputs {
                world_shaders: &world_shaders,
                bindless_active,
                bindless_pool_size,
                bindless_uab,
                n_cull,
                probe_cube_count,
                global_set_layout: &global_set_layout,
                descriptor_pool: &descriptor_pool,
                main_render_pass: &main_render_pass,
                msaa_samples,
                swapchain_format,
                gpu_textures: &gpu_textures,
                gpu_fallbacks: &gpu_fallbacks,
                linear_sampler: &linear_sampler,
            },
        )?;

        let pipelines::WorldPipelines {
            world_pipelines,
            shader_bucket_count,
        } = pipelines::build_world_pipelines(
            &gpu,
            pipelines::WorldPipelineInputs {
                world_shaders: &world_shaders,
                bindless_pipeline_layout: bindless_pipeline_layout.as_ref(),
                bindless_main_spv: &bindless_main_spv,
                main_render_pass: &main_render_pass,
                msaa_samples,
                swapchain_format,
                probe_cube_count,
            },
        )?;

        let gpu_driven::RtResources {
            seethrough_mesh_indices,
            has_seethrough_meshes,
            rt_accel_opt,
            rt_opt,
            composite_opt,
        } = gpu_driven::build_rt_reflections(
            &gpu,
            gpu_driven::RtInputs {
                draw_objects: &draw_objects,
                instanced_clusters: &instanced_clusters,
                vertices,
                vertex_buffer: &vertex_buffer,
                index_buffer: &index_buffer,
                gpu_textures: &gpu_textures,
                hdr_resolve_images: &hdr_resolve_images,
                gbuffer_opt: gbuffer_opt.as_ref(),
                env_map: &env_map,
                cube_sampler: &cube_sampler,
                global_set_layout: &global_set_layout,
                bindless_set_layout: bindless_set_layout.as_ref(),
                bindless_pool_size,
                probe_cube_count,
                render_extent,
                rt_capable,
                rt_wanted,
                rt_settings,
                ssr_authored,
                reflection_blur_scale,
            },
        )?;

        let gpu_driven::CullPass {
            cull_status_buffers,
            cull_pipeline,
            cull_pipeline_layout,
            cull_set_layout,
            cull_sets,
            draw_args_buffers,
            indirect_buffers,
            hiz,
        } = gpu_driven::build_cull_pass(
            &gpu,
            gpu_driven::CullInputs {
                draw_objects: &draw_objects,
                instanced_clusters: &instanced_clusters,
                gpu_textures: &gpu_textures,
                object_buffers: &object_buffers,
                depth_images: &depth_images,
                descriptor_pool: &descriptor_pool,
                bindless_active,
                n_cull,
                n_instances,
                shader_bucket_count,
                render_extent,
                msaa_samples,
                occlusion_two_pass,
            },
        )?;

        let gpu_driven::ShadowCull {
            shadow_cull_pipeline,
            shadow_cull_pipeline_layout,
            shadow_cull_set_layout,
            shadow_cull_sets,
            shadow_bindless_pipeline,
            shadow_bindless_pipeline_layout,
            shadow_indirect_buffers,
        } = gpu_driven::build_shadow_cull(
            &gpu,
            gpu_driven::ShadowCullInputs {
                bindless_active,
                has_shadow_pipeline: shadow_pipeline_opt.is_some(),
                bindless_set_layout: bindless_set_layout.as_ref(),
                shadow_global_set_layout: &shadow_global_set_layout,
                shadow_render_pass: &shadow_render_pass,
                descriptor_pool: &descriptor_pool,
                object_buffers: &object_buffers,
                draw_args_buffers: &draw_args_buffers,
                n_cull,
            },
        )?;

        let gpu_driven::GbufferPass {
            gbuffer_bindless_pipeline,
            gbuffer_bindless_pipeline_layout,
            gbuffer_set_layout,
            gbuffer_sets,
            prev_model_buffers,
            model_history,
            probe_prefilter,
        } = gpu_driven::build_gbuffer_pass(
            &gpu,
            gpu_driven::GbufferPassInputs {
                gbuffer_active,
                gbuffer_opt: gbuffer_opt.as_ref(),
                bindless_set_layout: bindless_set_layout.as_ref(),
                descriptor_pool: &descriptor_pool,
                object_buffers: &object_buffers,
                draw_args_buffers: &draw_args_buffers,
                n_cull,
                cull_pipeline: cull_pipeline.as_ref(),
            },
        )?;

        let gpu_driven::TwoPassCull {
            cull_pipeline_phase2,
            cull_sets2,
            two_pass_pool,
            indirect_buffers2,
            main_render_pass_phase1,
            main_render_pass_phase2,
        } = gpu_driven::build_two_pass_cull(
            &gpu,
            gpu_driven::TwoPassInputs {
                occlusion_two_pass,
                cull_set_layout: cull_set_layout.as_ref(),
                cull_pipeline_layout: cull_pipeline_layout.as_ref(),
                object_buffers: &object_buffers,
                draw_args_buffers: &draw_args_buffers,
                cull_status_buffers: &cull_status_buffers,
                n_cull,
                shader_bucket_count,
                msaa_samples,
            },
        )?;

        let descriptors::PostSets {
            text_atlas_sets,
            composite_sets,
            bloom_descriptor_pool,
            bloom_input_sets,
        } = descriptors::build_post_sets(
            &gpu,
            descriptors::PostSetInputs {
                gpu_text_atlases: &gpu_text_atlases,
                text_set_layout: &text_set_layout,
                text_sampler: &text_sampler,
                descriptor_pool: &descriptor_pool,
                composite_set_layout: &composite_set_layout,
                composite_sampler: &composite_sampler,
                composite_opt: composite_opt.as_ref(),
                hdr_resolve_images: &hdr_resolve_images,
                bloom_mips: &bloom_mips,
                bloom_set_layout: &bloom_set_layout,
                color_lut: &color_lut,
            },
        )?;

        let taa = effects::build_taa_and_wire_scene_inputs(
            &gpu,
            effects::SceneInputWiring {
                taa_enabled,
                render_extent,
                post_support: &post_support,
                composite_sampler: &composite_sampler,
                cube_sampler: &cube_sampler,
                global_set_layout: &global_set_layout,
                probe_cube_count,
                composite_sets: &composite_sets,
                bloom_input_sets: &bloom_input_sets,
                bloom_mips: &bloom_mips,
                color_lut: &color_lut,
                upscale: upscale.as_deref(),
                gbuffer_opt: gbuffer_opt.as_ref(),
                ssao_opt: ssao_opt.as_ref(),
                ssao_white: &ssao_white,
                transient_pool: &transient_pool,
            },
        )?;

        let effects::WorldEffects {
            decals_state,
            fog_resources,
            raymarch,
            planar_reflection,
            transparent,
            auto_exposure,
            auto_exposure_state,
        } = effects::build_world_effects(
            &gpu,
            effects::WorldEffectInputs {
                render_extent,
                msaa_samples,
                depth_images: &depth_images,
                hdr_resolve_images: &hdr_resolve_images,
                main_render_pass: &main_render_pass,
                shadow_render_pass: &shadow_render_pass,
                global_set_layout: &global_set_layout,
                global_update_after_bind,
                probe_cube_count,
                fog_settings: fog_settings.as_ref(),
                sdf_volumes: &sdf_volumes,
                water_surfaces: &water_surfaces,
                glass_panels: &glass_panels,
                planar_planes,
                cull_set_layout: cull_set_layout.as_ref(),
                object_buffers: &object_buffers,
                draw_args_buffers: &draw_args_buffers,
                n_cull,
                hiz: hiz.as_ref(),
                composite_opt: composite_opt.as_ref(),
                rt_accel_opt: rt_accel_opt.as_ref(),
                rt_capable,
                has_seethrough_meshes,
                seethrough_mesh_indices: &seethrough_mesh_indices,
                vertex_buffer: &vertex_buffer,
                index_buffer: &index_buffer,
                bindless_set_layout: bindless_set_layout.as_ref(),
                bindless_pool_size,
                auto_exposure_settings: auto_exposure_settings.as_ref(),
            },
            &bindings,
        )?;

        let commands::FrameCommands {
            command_buffers,
            start_command_pools,
            start_command_buffers,
            pass_command_pools,
            pass_command_buffers,
            image_available,
            render_finished,
            in_flight,
        } = commands::build_frame_commands(&gpu, graphics_family, &swapchain_images)?;
        let shadow_pipeline_layout_field = if shadow_pipeline_opt.is_some() {
            Some(shadow_pipeline_layout)
        } else {
            // SAFETY: every handle here was created from this device and is destroyed exactly once;
            // the caller has already waited for the device to go idle, so no submission still
            // references them.
            None
        };

        // Shader hot-reload: spawn a filesystem watcher over
        // `vulkan/shaders/` only under `cn debug`. The shared atomic flag
        // is also handed to the debug server elsewhere so the
        // `reload-shaders` command converges on the same trigger path.
        let (shader_reload_pending, shader_watcher) = if hot_reload {
            let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let watcher = crate::vulkan::hot_reload::spawn(std::sync::Arc::clone(&flag));
            (Some(flag), watcher)
        } else {
            (None, None)
        };

        let mut me = Self {
            instance,
            device,
            physical_device,
            alloc,
            surface,
            surface_loader,
            graphics_queue,
            present_queue,
            graphics_family,
            swapchain: super::context::SwapchainState {
                loader: swapchain_loader,
                handle: swapchain,
                images: swapchain_images,
                image_views: swapchain_image_views,
                format: swapchain_format,
                extent: swapchain_extent,
                last_present_index: None,
            },
            render_extent,
            main_render_pass,
            msaa_samples,
            color_images,
            depth_images,
            hdr_resolve_images,
            framebuffers,
            shadow: VkShadow {
                render_pass: shadow_render_pass,
                map: shadow_map,
                map_size: effective_shadow_size,
                framebuffers: shadow_framebuffers_vec,
                pipeline: shadow_pipeline_opt,
                pipeline_layout: shadow_pipeline_layout_field,
                global_set_layout: Some(shadow_global_set_layout),
                global_sets: shadow_global_sets,
                sampler: shadow_sampler,
                skinned_pipeline: None,
                skinned_pipeline_layout: None,
                ubos: shadow_ubos,
                uniforms: shadow_uniforms,
                light_dir: shadow_light_dir,
                update: shadow_update,
                distance: shadow_distance,
                cascades: shadow_cascades,
                scheduler: Default::default(),
                render_mask: 0,
            },
            spot_shadow,
            area_light: super::context::VkAreaLight {
                buffer: area_light_buffer,
                ltc_matrix: ltc_matrix_image,
                ltc_magnitude: ltc_magnitude_image,
                sampler: ltc_sampler,
            },
            textures: gpu_textures,
            fallback_textures: gpu_fallbacks,
            linear_sampler,
            light_cull,
            cull: VkCull {
                bindless_pipeline,
                bindless_pipeline_layout,
                bindless_set_layout,
                bindless_pool_size,
                bindless_update_after_bind: bindless_uab,
                world_pipelines,
                bucket_stride: n_cull,
                bindless_main_spv,
                bindless_sets,
                object_buffers,
                cull_pipeline,
                cull_pipeline_layout,
                cull_set_layout,
                cull_sets,
                draw_args_buffers,
                indirect_buffers,
                cull_status_buffers,
                occlusion_two_pass,
                cull_pipeline_phase2,
                cull_sets2,
                _two_pass_pool: two_pass_pool,
                indirect_buffers2,
                main_render_pass_phase1,
                main_render_pass_phase2,
                hiz,
                hiz_valid: false,
                hiz_prev_view_proj: IDENTITY,
                shadow_cull_pipeline,
                shadow_cull_pipeline_layout,
                _shadow_cull_set_layout: shadow_cull_set_layout,
                shadow_cull_sets,
                shadow_bindless_pipeline,
                shadow_bindless_pipeline_layout,
                shadow_indirect_buffers,
                gbuffer_bindless_pipeline,
                gbuffer_bindless_pipeline_layout,
                _gbuffer_set_layout: gbuffer_set_layout,
                gbuffer_sets,
                prev_model_buffers,
                model_history,
            },
            text: super::context::TextState {
                atlas_textures: gpu_text_atlases,
                pipeline: text_pipeline_opt,
                pipeline_layout: text_pipeline_layout,
                _sampler: text_sampler,
                upload: crate::vulkan::upload_ring::UploadRing::new(frames),
            },
            instanced: VkInstanced {
                lod_buckets: vec![Vec::new(); instanced_clusters.len()],
                any_lod: lod::any_cluster_has_lod(&instanced_clusters),
                clusters: instanced_clusters,
            },
            composite: super::context::CompositeState {
                render_pass: composite_render_pass,
                framebuffers: composite_framebuffers,
                pipeline: composite_pipeline,
                pipeline_layout: composite_pipeline_layout,
                _set_layout: composite_set_layout,
                sets: composite_sets,
                sampler: composite_sampler,
            },
            color_lut,
            bloom: super::context::BloomState {
                write_pass: bloom_write_pass,
                blend_pass: bloom_blend_pass,
                pipeline_prefilter: bloom_pipeline_prefilter,
                pipeline_downsample: bloom_pipeline_downsample,
                pipeline_upsample: bloom_pipeline_upsample,
                pipeline_layout: bloom_pipeline_layout,
                set_layout: bloom_set_layout,
                descriptor_pool: bloom_descriptor_pool,
                mips: bloom_mips,
                mip_extents: bloom_mip_extents,
                write_framebuffers: bloom_write_framebuffers,
                blend_framebuffers: bloom_blend_framebuffers,
                input_sets: bloom_input_sets,
            },
            post_process,
            taa,
            post: post_support,
            upscale,
            upscale_requested: upscale_backend,
            ssao: ssao_opt,
            ssao_white,
            transient_pool,
            ssr: ssr_opt,
            reflection_composite: composite_opt,
            ssgi: ssgi_opt,
            gbuffer: gbuffer_opt,
            model_history: Default::default(),
            rt_reflections: rt_opt,
            rt_accel: rt_accel_opt,
            rt_dynamic_mode,
            rt_skinned_geometry,
            rt_topology_dirty: false,
            rt_capable,
            update_after_bind,
            rt_static_vertex_count: vertices.len(),
            decal: super::context::DecalState {
                resources: decals_state,
                // Authored decals land in the table through `add_decal`, which
                // also writes each one's albedo descriptor.
                set: decal::DecalSet::new(crate::vulkan::decal::MAX_DECALS, frames),
            },
            lines: crate::vulkan::line::LineState::empty(),
            hdr_mode,
            vsync,
            particle: super::context::ParticleState {
                resources: None,
                records: Vec::new(),
                emitter_state: Vec::new(),
                free_slots: Vec::new(),
                last_elapsed: std::cell::Cell::new(0.0),
                frame_index: std::cell::Cell::new(0),
            },
            fog: super::context::FogState {
                resources: fog_resources,
                settings: fog_settings,
                sun_dir: fog_sun_dir,
                sun_color: fog_sun_color,
            },
            raymarch,
            transparent,
            planar_reflection,
            auto_exposure: super::context::AutoExposureState {
                resources: auto_exposure,
                settings: auto_exposure_settings,
                state: auto_exposure_state,
                bias_ev: auto_exposure_bias_ev,
                last_elapsed: 0.0,
            },
            hot_reload: super::context::HotReloadState {
                enabled: hot_reload,
                reload_pending: shader_reload_pending,
                watcher: shader_watcher,
            },
            world_shader: world_programs.cloned(),
            frame_stats: std::cell::Cell::new(profile::RenderStats::default()),
            draw_calls_accum: std::sync::atomic::AtomicU32::new(0),
            timestamp_query_pool,
            timestamp_period_ns: timestamp_period,
            device_local_heaps,
            memory_budget_supported,
            descriptors: VkDescriptors {
                global_set_layout,
                global_update_after_bind,
                probe_cube_count,
                _text_set_layout: text_set_layout,
                _descriptor_pool: descriptor_pool,
                global_sets,
                text_atlas_sets,
            },
            geometry: VkGeometry {
                vertex_buffer,
                index_buffer,
                mesh_vtx_alloc: crate::suballoc::range_alloc::RangeAllocator::new(),
                mesh_idx_alloc: crate::suballoc::range_alloc::RangeAllocator::new(),
                vertex_buffer_bytes,
                index_buffer_bytes,
            },
            chunk_stream: VkChunkStream {
                vtx_alloc: crate::suballoc::range_alloc::RangeAllocator::new(),
                idx_alloc: crate::suballoc::range_alloc::RangeAllocator::new(),
            },
            skinned: VkSkinned {
                joint_set_layout: None,
                descriptor_pool: None,
                vertex_buffer: super::allocator::PooledBuffer::null(),
                vertex_buffer_bytes: 0,
                index_buffer: super::allocator::PooledBuffer::null(),
                index_buffer_bytes: 0,
                slots: skinned_slots::SkinnedSlots::new(),
                joint_buffers: Vec::new(),
                joint_sets: Vec::new(),
                skin: None,
                deformed: Vec::new(),
                morph_delta_unique: Vec::new(),
                morph_delta_buffers: Vec::new(),
                morph_target_counts: Vec::new(),
                morph_weight_buffers: Vec::new(),
                deformed_primed: std::sync::atomic::AtomicBool::new(false),
            },
            uniforms: VkUniforms {
                view_ubo_buffers,
                probe_set_ubo_buffers,
                light_ubo_buffers,
                light_dirty: concinnity_core::render::frame_dirty::FrameDirty::new(frames),
                local_light_buffer,
                local_light_size: local_light_buffer_size,
                light_uniforms,
            },
            frame_sync: VkFrameSync {
                image_available,
                render_finished,
                in_flight,
            },
            current_frame: 0,
            frames_in_flight: frames,
            commands: VkCommands {
                command_pool,
                command_buffers,
                start_command_pools,
                start_command_buffers,
                pass_command_pools,
                pass_command_buffers,
            },
            draw: {
                let n_objects = draw_objects.len();
                super::context::DrawState {
                    n_objects,
                    objects: draw_objects,
                    graph_cache: None,
                    barrier_scratch: None,
                    n_instances,
                    // Runtime record reserve (fixed at init): the worst-case
                    // resident streamed-chunk window plus the runtime-clone budget.
                    // The cull buffers reserve `[n_objects + n_instances,
                    // +n_runtime)`; resident chunks and spawned clones fold in per
                    // frame, the unused tail is disabled.
                    n_runtime: n_chunk_max + clone_reserve(n_objects),
                    // Set in `upload_skinned` once the skin fold is built; the cull
                    // buffers reserve the tail at init via the threaded `n_skinned`
                    // capacity, but `cull_count()` reads this runtime count.
                    n_skinned: 0,
                }
            },
            view: super::context::ViewState {
                clear_color,
                scene_fade: 0.0,
                mode: Default::default(),
                show: Default::default(),
                far: 1.0,
                matrix: IDENTITY,
                sky_rot: concinnity_core::sky::SkyOrientation::IDENTITY_ROWS,
            },
            wireframe: Default::default(),
            prefilter_mip_count: env_map.prefilter_mip_count,
            cube_sampler,
            env_map,
            probe: super::context::ProbeState {
                placements: Vec::new(),
                set: concinnity_core::render::uniforms::ProbeSet::EMPTY,
                maps: Vec::new(),
                bake_queue: reflection_probe::ProbeBakeQueue::new(0),
                rendering: None,
                prefiltering: None,
                prefilter: probe_prefilter,
            },
            stream: super::context::StreamState {
                pool_rewrites: slot_rewrites::SlotRewriteQueue::new(frames),
                frame: 0,
                retires: Vec::new(),
            },
            window: Some(window),
            _entry: entry,
            // The swap-decision key for a future live reload of this context
            // (see `hot_swap_config` / `reload_world`). Normalized `frames` (>=1)
            // matches how `BackendInit::swapchain_config` clamps it.
            swapchain_config: backend_init::SwapchainConfig {
                frames_in_flight: frames,
                hdr_display,
                hdr_pq,
            },
            // A freshly built context owns its hardware outright; only
            // `apply_world_reload` flips this on the outgoing context.
            reused_by_successor: false,
            world_content_destroyed: false,
        };
        // Push every world-authored `DecalRecord` through `add_decal` so
        // its albedo descriptor lands in the reserved slot before the
        // first frame runs.
        me.upload_initial_decals(decals)?;
        // Same pattern for particle emitters: each world-authored record
        // routes through `add_particle_emitter` so its pool, counter, and
        // descriptor sets land before the first frame.
        me.upload_initial_particles(particles)?;
        // Every init upload above was synchronous (its one-shot idled the
        // queue), so init's remaining staging debris is retirable now; reclaim
        // it so the stats below report the steady footprint, not init's peak.
        me.alloc.reclaim_idle();
        tracing::info!(
            "device allocator: {} ({} allocations allowed)",
            me.alloc.stats(),
            me.alloc.max_allocations(),
        );
        crate::shader::cache::report_init();
        // Serialize the pipeline cache now that every init-built pipeline has
        // populated it, then write the segment holding it and every shader
        // artifact this init compiled; a crash mid-session then still leaves
        // the next launch warm.
        super::pipeline_cache::serialize(&me.device);
        crate::shader::pipeline_cache::report_init(super::pipeline_cache::disk_state());
        crate::shader::runtime_cache::checkpoint();
        Ok(me)
    }

    // Rebuild this backend's world content in place on its existing hardware for
    // a live editor edit: wait for the GPU to idle, hand the shared instance /
    // device / surface / swapchain / window (+ debug messenger + timestamp pool)
    // to a successor context built from `init`, then replace `self` with it. The
    // outgoing `self` is dropped by the `*self = rebuilt` assignment;
    // `reused_by_successor` (set just before it) makes that Drop free only this
    // world's content and leave the shared hardware to the successor. Only ever
    // called when `hot_swap_config` reported a config matching
    // `init.swapchain_config()`, so the swapchain (format / frames-in-flight /
    // EDR) is guaranteed unchanged. Mirrors `DxContext::apply_world_reload`.
    //
    // The old world's content is freed BEFORE the successor builds, into the
    // allocator the two contexts share, so the rebuild fills the released
    // blocks instead of holding both worlds' memory for the reload's duration.
    //
    // On a content-build failure (essentially impossible for a pre-validated
    // editor edit built from the engine's built-in shaders) the moved window is
    // closed with the dropped reuse bundle and the old world is already gone;
    // `self` keeps the shared hardware (the flag is unset on the failure path)
    // and tears it down in a normal Drop, so the caller can drop this backend
    // and mark the session failed without a leak.
    pub(in crate::vulkan) fn apply_world_reload(
        &mut self,
        init: backend_init::BackendInit<'_>,
    ) -> RenderResult<()> {
        self.wait_idle();
        // The loaders + `ash::{Entry,Instance,Device}` are dispatch-table clones
        // over the same underlying objects; the raw `vk::*` handles are `Copy`;
        // the window + (already-`Option`) debug messenger + timestamp pool are
        // MOVED out so this context's Drop leaves them for the successor.
        let reuse = VkReuse {
            window: self
                .window
                .take()
                .ok_or("apply_world_reload: window already taken")?,
            entry: self._entry.clone(),
            instance: self.instance.clone(),
            device: self.device.clone(),
            physical_device: self.physical_device,
            surface: self.surface,
            surface_loader: self.surface_loader.clone(),
            graphics_queue: self.graphics_queue,
            present_queue: self.present_queue,
            graphics_family: self.graphics_family,
            swapchain_loader: self.swapchain.loader.clone(),
            swapchain: self.swapchain.handle,
            swapchain_images: self.swapchain.images.clone(),
            swapchain_format: self.swapchain.format,
            swapchain_extent: self.swapchain.extent,
            hdr_mode: self.hdr_mode,
            memory_budget_supported: self.memory_budget_supported,
            rt_capable: self.rt_capable,
            update_after_bind: self.update_after_bind,
            device_local_heaps: self.device_local_heaps.clone(),
            timestamp_query_pool: self.timestamp_query_pool.take(),
            timestamp_period: self.timestamp_period_ns,
            alloc: self.alloc.clone(),
        };
        // Free the old world into the shared allocator BEFORE the successor
        // builds, and make the released ranges placeable now (`wait_idle`
        // above gated everything in flight): the rebuild then fills the same
        // blocks instead of doubling the device footprint for the reload's
        // duration. The flag is set first so the content pass keeps the
        // swapchain for the successor; a build failure unsets it and re-runs
        // the swapchain teardown the content pass skipped, so the
        // failed-session `Drop` still tears the shared hardware down.
        self.reused_by_successor = true;
        self.destroy_world_content();
        self.alloc.reclaim_idle();
        // The old world's pipelines, layouts and render passes queued on the
        // device's retire list as their owners dropped; the `wait_idle` above
        // means they can go now rather than after the successor's first frames.
        self.device.reclaim_idle();
        tracing::debug!("reload: old world freed: {}", self.alloc.stats());
        match VkContext::build(init, Some(reuse)) {
            Ok(rebuilt) => {
                *self = rebuilt;
                Ok(())
            }
            Err(e) => {
                self.reused_by_successor = false;
                self.destroy_swapchain_resources();
                Err(e)
            }
        }
    }
}
