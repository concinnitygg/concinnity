//! VkContext construction. `build` resolves the backend inputs, then calls each
//! stage in Vulkan object creation order. A stage returns the owned groups and
//! subsystem states it builds; a state whose render passes, layouts, pipelines
//! and descriptor sets come from different stages starts from inert
//! placeholders, and each stage fills in its part:
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
use concinnity_core::gfx::render_types::PostProcessParams;
use concinnity_core::render::backend_init::{BackendInit, PostSettings, ShadowParams, WorldShader};
use concinnity_core::render::error::RenderResult;

use super::context::*;
use super::light_cull::VkLightCull;
use super::texture::{GpuImage, GpuUploadContext};

pub(in crate::vulkan) mod bootstrap;
mod commands;
mod descriptors;
mod effects;
mod gpu_driven;
mod pipelines;
mod scene_data;
mod targets;

// The hardware and command pool every init stage creates resources through.
struct InitGpu<'a> {
    hw: &'a VkHardware,
    command_pool: vk::CommandPool,
    frames: usize,
    hot_reload: bool,
}

impl InitGpu<'_> {
    // The one-shot upload context most resource helpers take.
    fn upload(&self) -> GpuUploadContext<'_> {
        GpuUploadContext {
            alloc: &self.hw.alloc,
            device: &self.hw.device,
            command_pool: self.command_pool,
            queue: self.hw.graphics_queue,
        }
    }
}

// The feature gates every stage agrees on, resolved once from the world's post
// settings and the device.
struct Features {
    taa_enabled: bool,
    msaa_samples: vk::SampleCountFlags,
    post_process: PostProcessParams,
    bloom_on: bool,
    rt_wanted: bool,
    gbuffer_enabled: bool,
}

impl Features {
    fn resolve(hw: &VkHardware, post: &PostSettings) -> Self {
        // Temporal upscaling (FSR) consumes the velocity pre-pass's
        // render-resolution motion + depth, which TaaResources owns, so force
        // the TAA stack built when upscaling is on (the TAA *resolve* is still
        // dropped from the frame graph; only the velocity pre-pass is reused).
        let taa_enabled = post.taa_enabled || post.temporal_upscaling;
        // The device's ceiling for the HDR format, re-queried on both paths since
        // the outgoing world's AA mode may differ; the resolved setting is what
        // the world actually asks for. A temporal technique resolves it to one
        // sample, which drops the resolve attachment from every render pass and
        // makes the color image the scene spine.
        let msaa_samples = super::device::resolve_sample_count(
            super::device::get_max_usable_sample_count(&hw.instance, hw.physical_device),
            post.hdr_samples,
        );
        tracing::info!("vulkan HDR target: {}x MSAA", msaa_samples.as_raw().max(1));
        // Pair the authored tunables with the resolved HDR mode (freshly
        // negotiated or inherited on a reload), which drives the composite
        // shader's `hdr_output > 0.5` branch and its in-branch `pq_output`
        // encode flag. Mirrors `DxContext::new`.
        let post_process = hw.hdr_mode.post_process_params(post.post_process);
        let rt_wanted = post.rt_reflections.is_some() && hw.rt_capable;
        Self {
            taa_enabled,
            msaa_samples,
            post_process,
            bloom_on: post_process.bloom_intensity > 0.0,
            rt_wanted,
            // The unified pre-pass exists when any screen-space consumer needs
            // it. Derived once: the transient pool gate and the feature gate
            // disagreeing would mean the pool places no images while the feature
            // expects them, or the reverse.
            gbuffer_enabled: taa_enabled
                || post.ssao.is_some()
                || post.ssr.is_some()
                || post.ssgi.is_some()
                || rt_wanted,
        }
    }
}

// The fixed-function pass states whose render passes, layouts, pipelines and
// descriptor sets different stages create, filled in as each stage runs.
struct PassStates {
    shadow: VkShadow,
    text: TextState,
    composite: CompositeState,
    bloom: BloomState,
}

impl PassStates {
    fn new(shadows: &ShadowParams, frames: usize) -> Self {
        Self {
            shadow: VkShadow {
                render_pass: Default::default(),
                map: GpuImage::null(),
                map_size: shadows.map_size,
                framebuffers: Vec::new(),
                pipeline: None,
                pipeline_layout: None,
                global_set_layout: None,
                global_sets: Vec::new(),
                sampler: Default::default(),
                skinned_pipeline: None,
                skinned_pipeline_layout: None,
                ubos: Vec::new(),
                uniforms: concinnity_core::render::csm::empty_shadow_uniforms(),
                light_dir: [0.0; 3],
                update: shadows.update,
                distance: shadows.distance,
                cascades: shadows.cascades,
                scheduler: Default::default(),
                render_mask: 0,
            },
            text: TextState {
                atlas_textures: Vec::new(),
                pipeline: None,
                pipeline_layout: Default::default(),
                sampler: Default::default(),
                upload: super::upload_ring::UploadRing::new(frames),
            },
            composite: CompositeState::default(),
            bloom: BloomState::default(),
        }
    }
}

// The per-frame lighting, shadow and environment resources the global set binds,
// which fog, raymarched volumes and planar reflections bind as well.
struct GlobalBindings<'a> {
    uniforms: &'a VkUniforms,
    light_cull: &'a VkLightCull,
    shadow: &'a VkShadow,
    spot_shadow: &'a VkSpotShadow,
    area_light: &'a VkAreaLight,
    scene: &'a VkSceneAssets,
    targets: &'a VkTargets,
}

impl VkContext {
    // Construct a fresh context, acquiring its own OS window + Vulkan
    // instance / device / surface / swapchain.
    pub(crate) fn new(init: BackendInit<'_>) -> RenderResult<Self> {
        Self::build(init, None)
    }

    // Construct from the assembled backend inputs (see
    // `concinnity_core::render::backend_init::BackendInit` for per-field docs).
    //
    // `reuse` is `Some` only on a live editor `reload_world` (see
    // `apply_world_reload`): the hardware and swapchain are inherited from the
    // outgoing context instead of acquired fresh, and every per-world resource
    // is rebuilt on them. `None` acquires it all fresh, the normal launch path.
    fn build(
        init: BackendInit<'_>,
        reuse: Option<(VkHardware, SwapchainState)>,
    ) -> RenderResult<Self> {
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
            scene: world,
            shaders: world_shaders,
            media,
            light_uniforms,
            local_lights,
            spot_shadows,
            area_lights,
            shadows,
            anisotropy,
            planar_planes,
            post,
            fx,
            requirements: _,
        } = init;
        // Entry 0 is the world default program; entries 1.. are the
        // material-referenced shader buckets (see `world_shaders.rs`). The world
        // default program is never deferred (bucket 0 always decodes at init);
        // only the material-referenced buckets can be.
        let &WorldShader {
            programs: world_programs,
            deferred: _,
        } = world_shaders
            .first()
            .ok_or_else(|| "BackendInit carried no shaders".to_string())?;
        // Record this (main) thread so the `RenderBackend` mutation entry points
        // can `debug_assert_main_thread` against it; the Send invariant rests on
        // the context being touched from this thread alone.
        super::context::record_main_thread();
        let frames = frames_in_flight.max(1);

        let request = bootstrap::HardwareRequest {
            window,
            validation,
            frames,
            vsync,
            post: &post,
        };
        let (hw, swapchain) = match reuse {
            Some(reuse) => bootstrap::inherit_hardware(reuse, &request)?,
            None => bootstrap::acquire_hardware(&request)?,
        };
        let features = Features::resolve(&hw, &post);
        let command_pool = commands::create_command_pool(&hw)?;
        let gpu = InitGpu {
            hw: &hw,
            command_pool,
            frames,
            hot_reload,
        };

        let (upscale, render_extent) = effects::build_upscale(&gpu, swapchain.extent, &post)?;
        commands::reset_timestamp_queries(&gpu)?;
        let mut passes = PassStates::new(&shadows, frames);
        targets::build_shadow_map(&gpu, &mut passes.shadow)?;
        let area_light = scene_data::build_area_lights(&gpu, &area_lights)?;
        let spot_shadow_map =
            targets::build_spot_shadow_map(&gpu, shadows.map_size, &spot_shadows)?;
        let mut scene =
            scene_data::build_textures_and_samplers(&gpu, &media, anisotropy, &mut passes)?;
        let targets = targets::build_render_targets(
            &gpu,
            targets::TargetInputs {
                swapchain: &swapchain,
                features: &features,
                render_extent,
                ssao_enabled: post.ssao.is_some(),
            },
            &mut passes,
        )?;
        let (geometry, uniforms, light_cull) = scene_data::build_scene_resources(
            &gpu,
            scene_data::SceneInputs {
                world: &world,
                media: &media,
                local_lights: &local_lights,
                light_uniforms,
            },
            &mut scene,
            &mut passes.shadow,
        )?;
        let mut descriptors = descriptors::build_layouts(&gpu, &mut passes)?;
        let spot_shadow =
            pipelines::build_main_pipelines(&gpu, &mut passes, spot_shadow_map, &spot_shadows)?;
        scene.ssao_white = effects::build_ssao_white(&gpu)?;
        let screen = effects::build_screen_space(
            &gpu,
            effects::ScreenSpaceInputs {
                post: &post,
                features: &features,
                targets: &targets,
                scene: &scene,
                descriptors: &descriptors,
                composite_sampler: &passes.composite.sampler,
            },
        )?;

        let plan = gpu_driven::plan_cull(&gpu, &world, media.textures, &descriptors);
        descriptors::build_sets(
            &gpu,
            &mut descriptors,
            descriptors::SetPoolInputs {
                instanced_clusters: &world.instanced_clusters,
                text_atlas_count: passes.text.atlas_textures.len(),
                plan: &plan,
                has_gbuffer: screen.gbuffer.is_some(),
            },
            &GlobalBindings {
                uniforms: &uniforms,
                light_cull: &light_cull,
                shadow: &passes.shadow,
                spot_shadow: &spot_shadow,
                area_light: &area_light,
                scene: &scene,
                targets: &targets,
            },
        )?;
        descriptors::build_shadow_global_sets(&gpu, &descriptors, &mut passes.shadow)?;
        let mut cull = gpu_driven::build_bindless_pass(
            &gpu,
            gpu_driven::BindlessInputs {
                world_shaders: &world_shaders,
                plan: &plan,
                occlusion_two_pass: post.occlusion_two_pass,
                descriptors: &descriptors,
                targets: &targets,
                scene: &scene,
                swapchain_format: swapchain.format,
            },
        )?;
        pipelines::build_world_pipelines(
            &gpu,
            &mut cull,
            &world_shaders,
            &targets,
            swapchain.format,
            descriptors.probe_cube_count,
        )?;
        let rt = gpu_driven::build_rt_reflections(
            &gpu,
            gpu_driven::RtInputs {
                world: &world,
                geometry: &geometry,
                scene: &scene,
                targets: &targets,
                gbuffer: screen.gbuffer.as_ref(),
                descriptors: &descriptors,
                cull: &cull,
                post: &post,
                rt_wanted: features.rt_wanted,
            },
        )?;
        gpu_driven::build_cull_pass(
            &gpu,
            &mut cull,
            gpu_driven::CullInputs {
                world: &world,
                scene: &scene,
                targets: &targets,
                descriptors: &descriptors,
                plan: &plan,
            },
        )?;
        gpu_driven::build_shadow_cull(&gpu, &mut cull, &passes.shadow, &descriptors, &plan)?;
        let probe_prefilter = gpu_driven::build_gbuffer_pass(
            &gpu,
            &mut cull,
            screen.gbuffer.as_ref(),
            &descriptors,
            &plan,
        )?;
        gpu_driven::build_two_pass_cull(&gpu, &mut cull, &plan, targets.msaa_samples)?;
        descriptors::build_post_sets(
            &gpu,
            &mut descriptors,
            &mut passes,
            &targets,
            &scene,
            rt.composite.as_ref(),
        )?;
        let taa = effects::build_taa_and_wire_scene_inputs(
            &gpu,
            effects::SceneInputWiring {
                features: &features,
                targets: &targets,
                scene: &scene,
                descriptors: &descriptors,
                passes: &passes,
                screen: &screen,
                upscale: upscale.as_deref(),
            },
        )?;
        let world_fx = effects::build_world_effects(
            &gpu,
            effects::WorldEffectInputs {
                targets: &targets,
                descriptors: &descriptors,
                fx: &fx,
                planar_planes,
                cull: &cull,
                n_cull: plan.n_cull,
                rt: &rt,
                geometry: &geometry,
                post: &post,
            },
            &GlobalBindings {
                uniforms: &uniforms,
                light_cull: &light_cull,
                shadow: &passes.shadow,
                spot_shadow: &spot_shadow,
                area_light: &area_light,
                scene: &scene,
                targets: &targets,
            },
        )?;
        let (commands, frame_sync) = commands::build_frame_commands(&gpu, &swapchain)?;

        let mut me = Self {
            swapchain,
            targets,
            composite: passes.composite,
            shadow: passes.shadow,
            spot_shadow,
            area_light,
            scene,
            cull,
            light_cull,
            text: passes.text,
            bloom: passes.bloom,
            post_process: features.post_process,
            taa,
            post: screen.post,
            upscale,
            upscale_requested: post.upscale_backend,
            ssao: screen.ssao,
            ssr: screen.ssr,
            reflection_composite: rt.composite,
            ssgi: screen.ssgi,
            gbuffer: screen.gbuffer,
            model_history: Default::default(),
            rt_reflections: rt.reflections,
            rt: rt.state,
            decal: world_fx.decal,
            lines: crate::vulkan::line::LineState::empty(),
            fog: world_fx.fog,
            raymarch: world_fx.raymarch,
            transparent: world_fx.transparent,
            planar_reflection: world_fx.planar_reflection,
            particle: Default::default(),
            auto_exposure: world_fx.auto_exposure,
            hot_reload: HotReloadState::spawn(hot_reload),
            world_shader: world_programs.cloned(),
            frame_stats: Default::default(),
            draw_calls_accum: Default::default(),
            descriptors,
            instanced: VkInstanced::new(world.instanced_clusters),
            geometry,
            chunk_stream: Default::default(),
            skinned: VkSkinned::new(),
            uniforms,
            frame_sync,
            current_frame: 0,
            frames_in_flight: frames,
            commands,
            draw: DrawState::new(world.draw_objects, plan.n_instances, world.n_chunk_max),
            view: ViewState::new(clear_color),
            wireframe: Default::default(),
            probe: ProbeState::new(probe_prefilter),
            stream: StreamState::new(frames),
            // A freshly built context owns its hardware outright; only
            // `apply_world_reload` flips this on the outgoing context.
            reused_by_successor: false,
            world_content_destroyed: false,
            hw,
        };
        // Push every world-authored `DecalRecord` through `add_decal` so
        // its albedo descriptor lands in the reserved slot before the
        // first frame runs.
        me.upload_initial_decals(fx.decals)?;
        // Same pattern for particle emitters: each world-authored record
        // routes through `add_particle_emitter` so its pool, counter, and
        // descriptor sets land before the first frame.
        me.upload_initial_particles(fx.particles)?;
        // Every init upload above was synchronous (its one-shot idled the
        // queue), so init's remaining staging debris is retirable now; reclaim
        // it so the stats below report the steady footprint, not init's peak.
        me.hw.alloc.reclaim_idle();
        tracing::info!(
            "device allocator: {} ({} allocations allowed)",
            me.hw.alloc.stats(),
            me.hw.alloc.max_allocations(),
        );
        crate::shader::cache::report_init();
        // Serialize the pipeline cache now that every init-built pipeline has
        // populated it, then write the segment holding it and every shader
        // artifact this init compiled; a crash mid-session then still leaves
        // the next launch warm.
        super::pipeline_cache::serialize(&me.hw.device);
        crate::shader::pipeline_cache::report_init(super::pipeline_cache::disk_state());
        crate::shader::runtime_cache::checkpoint();
        Ok(me)
    }

    // Rebuild this backend's world content in place on its existing hardware for
    // a live editor edit: wait for the GPU to idle, hand the hardware and
    // swapchain to a successor context built from `init`, then replace `self`
    // with it. The outgoing `self` is dropped by the `*self = rebuilt`
    // assignment; `reused_by_successor` (set just before it) makes that Drop free
    // only this world's content and leave the shared hardware to the successor.
    // Only ever called when `hot_swap_config` reported a config matching
    // `init.swapchain_config()`, so the swapchain (format / frames-in-flight /
    // EDR) is guaranteed unchanged. Mirrors `DxContext::apply_world_reload`.
    //
    // The old world's content is freed BEFORE the successor builds, into the
    // allocator the two contexts share, so the rebuild fills the released
    // blocks instead of holding both worlds' memory for the reload's duration.
    //
    // On a content-build failure (essentially impossible for a pre-validated
    // editor edit built from the engine's built-in shaders) the moved window is
    // closed with the dropped hardware handles and the old world is already
    // gone; `self` keeps the shared hardware (the flag is unset on the failure
    // path) and tears it down in a normal Drop, so the caller can drop this
    // backend and mark the session failed without a leak.
    pub(in crate::vulkan) fn apply_world_reload(
        &mut self,
        init: BackendInit<'_>,
    ) -> RenderResult<()> {
        self.wait_idle();
        let reuse = (self.hw.hand_over()?, self.swapchain.share());
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
        self.hw.alloc.reclaim_idle();
        // The old world's pipelines, layouts and render passes queued on the
        // device's retire list as their owners dropped; the `wait_idle` above
        // means they can go now rather than after the successor's first frames.
        self.hw.device.reclaim_idle();
        tracing::debug!("reload: old world freed: {}", self.hw.alloc.stats());
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
