//! `VkContext` construction. `build` resolves the backend inputs, then calls each
//! stage in dependency order. A stage builds one owned group or subsystem state
//! whole and returns it, borrowing the finished states it depends on:
//!
//!   bootstrap.rs     window, instance, debug messenger, surface, device, swapchain.
//!   commands.rs      command pools and buffers, timestamp reset, sync objects.
//!   `scene_assets.rs`  textures, scene and cube samplers, IBL cubes, color LUT.
//!   targets.rs       main render pass, HDR attachments, transient image pool.
//!   `scene_data.rs`    area lights, geometry, uniform rings, clustered lights.
//!   shadow.rs        cascade and spot shadow states.
//!   descriptors.rs   global set layout, the shared pool, global and shadow sets.
//!   effects.rs       upscaler, screen-space passes, TAA wiring, world effects.
//!   cull/            bindless pass, compute cull, Hi-Z, GPU-driven shadow, G-buffer.
//!   `ray_tracing.rs`   RT acceleration structure, reflections, reflection composite.
//!   bloom.rs         the shared bloom chain.
//!   composite.rs     composite pass, pipeline and input sets.
//!   text.rs          text atlases, pipeline and atlas sets.

use crate::vulkan::particle::ParticleState;
use crate::vulkan::pipeline_builder::PipelineGate;
use crate::vulkan::wireframe::VkWireframe;
use ash::vk;
use concinnity_engine::gfx::render_types::PostProcessParams;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::backend::feature_plan::{BackendFacts, FeaturePlan};
use concinnity_engine::render::backend::init::{BackendInit, WorldShader};
use concinnity_engine::render::scene_state::{DrawList, SceneState};
use concinnity_engine::render::shader::reload::ShaderReload;
use concinnity_engine::settings::ResolvedGraphics;
use std::cell::Cell;

use super::context::{
    HDR_FORMAT, ProbeState, StreamState, SwapchainState, VkContext, VkCull, VkDescriptors,
    VkHardware, VkInstanced, VkShadow, VkSkinned, VkTargets, record_main_thread,
};
use super::geometry_upload::GeometryUploads;
use super::texture::GpuUploadContext;
use crate::vulkan::descriptor_layout::SCREEN_INDIRECT_BINDING;
use crate::vulkan::global_set::GlobalBindings;

mod bloom;
pub(in crate::vulkan) mod bootstrap;
mod commands;
mod composite;
mod cull;
mod descriptors;
mod effects;
mod ray_tracing;
mod scene_assets;
mod scene_data;
mod shadow;
mod targets;
mod text;

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

// The feature gates every stage agrees on, resolved once from the world's
// graphics settings and the device.
struct Features {
    taa_enabled: bool,
    msaa_samples: vk::SampleCountFlags,
    post_process: PostProcessParams,
    rt_wanted: bool,
    gbuffer_enabled: bool,
}

impl Features {
    fn resolve(
        hw: &VkHardware,
        graphics: &ResolvedGraphics,
        hdr_samples: u32,
        upscaling: bool,
    ) -> Self {
        // A built temporal upscaler consumes the velocity pre-pass's
        // render-resolution motion + depth, which TaaResources owns, so force
        // the TAA stack built while one is (the TAA *resolve* is still dropped
        // from the frame graph; only the velocity pre-pass is reused).
        let plan = FeaturePlan::resolve(
            graphics,
            BackendFacts {
                rt_capable: hw.rt_capable,
                upscaling,
            },
        );
        let taa_enabled = plan.temporal;
        // The device's ceiling for the HDR format, re-queried on both paths since
        // the outgoing world's AA mode may differ; the resolved setting is what
        // the world actually asks for. A temporal technique resolves it to one
        // sample, which drops the resolve attachment from every render pass and
        // makes the color image the scene spine.
        let msaa_samples = super::device::resolve_sample_count(
            super::device::get_max_usable_sample_count(&hw.instance, hw.physical_device),
            hdr_samples,
        );
        tracing::info!("vulkan HDR target: {}x MSAA", msaa_samples.as_raw().max(1));
        // Pair the authored tunables with the resolved HDR mode (freshly
        // negotiated or inherited on a reload), which drives the composite
        // shader's `hdr_output > 0.5` branch and its in-branch `pq_output`
        // encode flag. Mirrors `DxContext::new`.
        let post_process = hw.hdr_mode.post_process_params(graphics.post_process());
        let rt_wanted = plan.rt.is_some();
        Self {
            taa_enabled,
            msaa_samples,
            post_process,
            rt_wanted,
            // The unified pre-pass exists when any screen-space consumer needs
            // it. Derived once: the transient pool gate and the feature gate
            // disagreeing would mean the pool places no images while the feature
            // expects them, or the reverse.
            gbuffer_enabled: plan.gbuffer,
        }
    }
}

impl VkContext {
    // Construct a fresh context, acquiring its own OS window + Vulkan
    // instance / device / surface / swapchain.
    pub(crate) fn new(init: BackendInit<'_>) -> RenderResult<Self> {
        Self::build(init, None)
    }

    // Construct from the assembled backend inputs (see
    // `concinnity_engine::render::backend::init::BackendInit` for per-field docs).
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
            platform: _,
            window,
            validation,
            graphics,
            hdr_samples,
            launch,
            hot_reload,
            capture,
            embedded_surface: _,
            scene: world,
            shaders: world_shaders,
            media,
            light_uniforms,
            local_lights,
            spot_shadows,
            area_lights,
            planar,
            fx,
            requirements: _,
        } = init;
        let vsync = graphics.frame_pacing.vsync;
        let clear_color = graphics.background.color;
        let shadows = graphics.shadow_params();
        // Entry 0 is the world default program; entries 1.. are the
        // material-referenced shader buckets (see `world_shaders.rs`). The world
        // default program is never deferred (bucket 0 always decodes at init);
        // only the material-referenced buckets can be.
        let &WorldShader {
            programs: world_programs,
            ..
        } = world_shaders
            .first()
            .ok_or_else(|| RenderError::Other("BackendInit carried no shaders".to_string()))?;
        // Record this (main) thread so the `RenderBackend` mutation entry points
        // can `debug_assert_main_thread` against it; the Send invariant rests on
        // the context being touched from this thread alone.
        record_main_thread();
        let frames = graphics.frames_in_flight();

        let request = bootstrap::HardwareRequest {
            window,
            validation,
            frames,
            vsync,
            graphics: &graphics,
        };
        let (hw, swapchain) = match reuse {
            Some(reuse) => bootstrap::inherit_hardware(reuse, &request)?,
            None => bootstrap::acquire_hardware(&request)?,
        };
        let command_pool = commands::create_command_pool(&hw)?;
        let gpu = InitGpu {
            hw: &hw,
            command_pool,
            frames,
            hot_reload,
        };

        let mut upscale_requested = crate::upscale_sdk::UpscaleRequest::from_graphics(&graphics);
        let (upscale, render_extent) =
            effects::build_upscale(&gpu, swapchain.extent, &graphics, &mut upscale_requested)?;
        let features = Features::resolve(&hw, &graphics, hdr_samples, upscale.is_some());
        commands::reset_timestamp_queries(&gpu)?;
        let area_light = scene_data::build_area_lights(&gpu, &area_lights)?;
        let mut scene =
            scene_assets::build_scene_assets(&gpu, &media, graphics.texture_filtering.anisotropy)?;
        let targets = targets::build_render_targets(
            &gpu,
            targets::TargetInputs {
                swapchain: &swapchain,
                features: &features,
                render_extent,
                ssao_enabled: graphics.ssao().is_some(),
            },
        )?;
        // An atmosphere writes the irradiance and prefiltered cubes every set
        // takes ambient light and reflections from, so they replace the
        // environment map's before any set binds them.
        let atmosphere_cube = match fx.atmosphere {
            Some(_) => {
                let (images, cubes) =
                    crate::vulkan::atmosphere::create_atmosphere_cubes(&gpu.upload())?;
                scene.env_map.irradiance = images.irradiance;
                scene.env_map.prefilter = images.prefilter;
                scene.prefilter_mip_count =
                    concinnity_engine::render::environment::atmosphere::SKY_CUBE_PLAN.mips();
                Some(cubes)
            }
            None => None,
        };
        let (geometry, uniforms, light_cull) = scene_data::build_scene_resources(
            &gpu,
            scene_data::SceneInputs {
                world: &world,
                local_lights: &local_lights,
                light_uniforms,
            },
        )?;
        let shadow = shadow::build_shadow(&gpu, &shadows)?;
        let spot_shadow = shadow::build_spot_shadow(&gpu, &shadow, &spot_shadows)?;
        let probe_gpu = crate::vulkan::probe_set::ProbeSetGpu::new(&gpu.upload(), frames)?;
        for (frame, records) in probe_gpu.records.iter().enumerate() {
            light_cull.write_probe_records(&hw.device, frame, records.descriptor());
        }
        let globals = GlobalBindings {
            uniforms: &uniforms,
            light_cull: &light_cull,
            shadow: &shadow,
            spot_shadow: &spot_shadow,
            area_light: &area_light,
            scene: &scene,
            targets: &targets,
            probes: &probe_gpu,
            // SSGI is built with the effects below, which rewrite this binding.
            ssgi_indirect: None,
        };
        let plan = cull::plan_cull(&gpu, &world, media.textures);
        let descriptors = descriptors::build_descriptors(
            &gpu,
            descriptors::SetPoolInputs {
                instanced_clusters: &world.instanced_clusters,
                text_atlas_count: media.text_atlases.len(),
                plan: &plan,
            },
            &globals,
        )?;
        let screen = effects::build_screen_space(
            &gpu,
            effects::ScreenSpaceInputs {
                graphics: &graphics,
                features: &features,
                targets: &targets,
                scene: &scene,
                descriptors: &descriptors,
            },
        )?;

        let (cull, probe_prefilter) = cull::build_cull(
            &gpu,
            cull::CullInputs {
                world: &world,
                world_shaders: &world_shaders,
                plan: &plan,
                occlusion_two_pass: graphics.culling.two_pass_occlusion,
                descriptors: &descriptors,
                targets: &targets,
                scene: &scene,
                geometry: &geometry,
                shadow: &shadow,
                spot_slices: spot_shadow.count() as usize,
                gbuffer: screen.gbuffer.as_ref(),
                swapchain_format: swapchain.format,
            },
        )?;
        let rt = ray_tracing::build_rt_reflections(
            &gpu,
            ray_tracing::RtInputs {
                world: &world,
                geometry: &geometry,
                scene: &scene,
                targets: &targets,
                gbuffer: screen.gbuffer.as_ref(),
                descriptors: &descriptors,
                cull: &cull,
                graphics: &graphics,
                launch,
                post_support: &screen.post,
                rt_wanted: features.rt_wanted,
            },
        )?;
        let bloom = bloom::build_bloom(
            &gpu,
            bloom::BloomInputs {
                extent: swapchain.extent,
                post: &screen.post,
                scene: &scene,
                descriptors: &descriptors,
            },
        )?;
        let composite = composite::build_composite(
            &gpu,
            composite::CompositeInputs {
                swapchain: &swapchain,
                descriptors: &descriptors,
                scene: &scene,
                targets: &targets,
                sampler: &screen.post.sampler,
                reflection_composite: rt.composite.as_ref(),
            },
        )?;
        let text = text::build_text(&gpu, &media, &composite, &descriptors)?;
        let taa = effects::build_taa_and_wire_scene_inputs(
            &gpu,
            effects::SceneInputWiring {
                features: &features,
                targets: &targets,
                scene: &scene,
                descriptors: &descriptors,
                composite: &composite,
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
                planar,
                cull: &cull,
                n_cull: plan.n_cull,
                rt: &rt,
                geometry: &geometry,
                graphics: &graphics,
            },
            &globals,
        )?;
        let grass = build_grass(
            &gpu,
            fx.grass,
            (
                geometry.vertex_buffer.buffer(),
                geometry.vertex_buffer_bytes,
            ),
            GrassInit {
                descriptors: &descriptors,
                cull: &cull,
                targets: &targets,
                shadow: &shadow,
            },
        )?;
        let (commands, frame_sync) = commands::build_frame_commands(&gpu, &swapchain)?;
        let sky = crate::vulkan::sky::VkSky::build(
            &hw.device,
            descriptors.global_set_layout.handle(),
            targets.main_render_pass.handle(),
            targets.msaa_samples,
            media.env_map_background,
            hot_reload,
        )?;

        let atmosphere = match (fx.atmosphere, atmosphere_cube) {
            (Some(settings), Some(cube)) => {
                let hdr_resolve_views: Vec<vk::ImageView> = targets
                    .hdr_resolve_images
                    .iter()
                    .map(|img| img.view)
                    .collect();
                let depth_views: Vec<vk::ImageView> =
                    targets.depth_images.iter().map(|img| img.view).collect();
                Some(crate::vulkan::atmosphere::VkAtmosphere::new(
                    &gpu.upload(),
                    &settings,
                    cube,
                    frames,
                    crate::vulkan::atmosphere::AtmosphereTargets {
                        hdr_resolve_views: &hdr_resolve_views,
                        depth_views: &depth_views,
                        extent: render_extent,
                        msaa: targets.msaa_samples != vk::SampleCountFlags::TYPE_1,
                        hdr_format: HDR_FORMAT,
                    },
                    crate::vulkan::atmosphere::AtmosphereSky {
                        global_set_layout: descriptors.global_set_layout.handle(),
                        main_render_pass: targets.main_render_pass.handle(),
                        samples: targets.msaa_samples,
                    },
                    hot_reload,
                )?)
            }
            _ => None,
        };

        let mut me = Self {
            swapchain,
            targets,
            composite,
            shadow,
            spot_shadow,
            area_light,
            scene,
            cull,
            light_cull,
            text,
            bloom: Some(bloom),
            post_process: features.post_process,
            pre_exposure: concinnity_engine::gfx::auto_exposure::PreExposure::UNIT,
            taa,
            post: screen.post,
            upscale,
            upscale_requested,
            ssao: screen.ssao,
            ssr: screen.ssr,
            reflection_composite: rt.composite,
            ssgi: screen.ssgi,
            gbuffer: screen.gbuffer,
            rt_reflections: rt.reflections,
            rt: rt.state,
            decal: world_fx.decal,
            lines: crate::vulkan::line::LineState::empty(),
            fog: world_fx.fog,
            raymarch: world_fx.raymarch,
            sky,
            atmosphere,
            transparent: world_fx.transparent,
            water_sim: world_fx.water_sim,
            planar_reflection: world_fx.planar_reflection,
            particle: ParticleState::default(),
            grass,
            grass_lod: None,
            auto_exposure: world_fx.auto_exposure,
            hot_reload: ShaderReload::new(hot_reload),
            pipeline_gate: PipelineGate::default(),
            world_shader: world_programs.cloned(),
            frame_stats: Cell::default(),
            draw_calls_accum: Default::default(),
            descriptors,
            instanced: VkInstanced::new(world.instanced_clusters),
            geometry,
            geometry_uploads: core::cell::RefCell::new(GeometryUploads::new(frames)),
            skinned: VkSkinned::new(),
            uniforms,
            frame_sync,
            frame_capture: crate::frame_capture::FrameCaptures::new(capture),
            current_frame: 0,
            frame_input_sampled_us: vec![None; frames],
            frames_in_flight: frames,
            commands,
            state: SceneState::new(
                DrawList::with_runtime_reserve(
                    world.draw_objects,
                    plan.n_instances,
                    world.n_chunk_max,
                ),
                clear_color,
            ),
            graph_scratch: None,
            submit_scratch: Vec::new(),
            wireframe: VkWireframe::default(),
            probe: ProbeState::new(probe_prefilter, probe_gpu),
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
        me.sync_grass_prepass(false);
        me.state.planner.shadow.cadence = shadows.cadence;
        me.refresh_gates(&graphics);
        if me.ssgi.is_some() {
            me.rewrite_global_binding(SCREEN_INDIRECT_BINDING);
        }
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
        match Self::build(init, Some(reuse)) {
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

// The grass field's buffers and pipelines, extending the GPU-driven pass's
// layouts and reading its Hi-Z pyramid. A world without that pass draws no
// grass.
// What the grass draws extend and render into.
#[derive(Clone, Copy)]
struct GrassInit<'a> {
    descriptors: &'a VkDescriptors,
    cull: &'a VkCull,
    targets: &'a VkTargets,
    shadow: &'a VkShadow,
}

fn build_grass(
    gpu: &InitGpu<'_>,
    field: Option<concinnity_engine::render::environment::grass::GrassField>,
    vertices: (vk::Buffer, u64),
    passes: GrassInit<'_>,
) -> RenderResult<Option<super::grass::GrassResources>> {
    let GrassInit {
        descriptors,
        cull,
        targets,
        shadow,
    } = passes;
    let (Some(field), Some(bindless), Some(prepass), Some(hiz)) = (
        field,
        cull.bindless_set_layout.as_ref(),
        cull.prepass_layout.as_ref(),
        cull.hiz.as_ref(),
    ) else {
        return Ok(None);
    };
    let main = [descriptors.global_set_layout.handle(), bindless.handle()];
    let prepass = [main[0], main[1], prepass.set_layout.handle()];
    super::grass::GrassResources::build(
        gpu.upload(),
        field,
        super::grass::GrassPassLayouts {
            main: &main,
            prepass: &prepass,
            hiz_read: hiz.read_set_layout.handle(),
        },
        super::grass::GrassTargets {
            main_render_pass: targets.main_render_pass.handle(),
            msaa_samples: targets.msaa_samples,
            shadow_render_pass: shadow.render_pass.handle(),
            shadow_bias_clamp: gpu.hw.device.depth_bias_clamp(),
        },
        gpu.frames,
        vertices,
        gpu.hot_reload,
    )
    .map(Some)
}
