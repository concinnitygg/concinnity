//! `DxContext` construction. `build` resolves the backend inputs, then calls each
//! stage in dependency order. A stage builds one owned group or subsystem state
//! whole and returns it, borrowing the finished states it depends on:
//!
//!   bootstrap.rs     window, DXGI factory, adapter, device, queue, swapchain.
//!   heaps.rs         RTV heap with the back-buffer views, DSV and sampler heaps.
//!   descriptors.rs   shader-visible SRV heap, sampler handles, post-pass block.
//!   effects.rs       upscaler, screen-space passes, world effects.
//!   `scene_assets.rs`  area lights, IBL cubes, textures, color LUT, geometry.
//!   targets.rs       main depth, HDR scene target, transient pool.
//!   `scene_data.rs`    constant-buffer rings, clustered light binning.
//!   shadow.rs        cascade and spot shadow states.
//!   cull/            bindless pass, compute cull, Hi-Z, GPU-driven shadow, G-buffer.
//!   text.rs          text atlases and pipeline.
//!   composite.rs     composite pipeline.
//!   `ray_tracing.rs`   reflection composite, RT reflections, acceleration structure.
//!   bloom.rs         the shared bloom chain.
//!   commands.rs      command lists, frame sync, timestamp queries.
//!
//! `heap_layout.rs` holds every heap's slot layout, `pipelines.rs` the main-pass
//! root signature and PSO builders, and `adapter.rs` the adapter selection.

use concinnity_engine::components::Window;
use concinnity_engine::gfx::render_types::{FALLBACK_TEXTURE_COUNT, PostProcessParams};
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::backend::feature_plan::{BackendFacts, FeaturePlan};
use concinnity_engine::render::backend::init::{self, BackendInit, PlanarBudget, WorldShader};
use concinnity_engine::render::scene_state::{DrawList, SceneState};
use concinnity_engine::render::shader::reload::ShaderReload;
use concinnity_engine::settings::ResolvedGraphics;

use self::heap_layout::{RtvHeapLayout, SrvHeapParams};
use super::context::{
    DxContext, DxHardware, DxInstanced, ProbeState, StreamState, record_main_thread,
};
use super::geometry_upload::GeometryUploads;
use super::resources::skinning::SkinnedState;

mod adapter;

pub(in crate::directx) use adapter::select as select_adapter;
mod bloom;
mod bootstrap;
mod commands;
mod composite;
mod cull;
mod descriptors;
mod effects;
pub(in crate::directx) mod heap_layout;
mod heaps;
pub(in crate::directx) mod pipelines;
mod ray_tracing;
mod scene_assets;
mod scene_data;
pub(super) mod shadow;
mod targets;
mod text;

pub(in crate::directx) use bootstrap::swapchain_flags;

// Per-mip Hi-Z UAV slots reserved at init: the deepest pyramid the two SPD
// dispatches write at any size, so the heap layout survives a resize.
pub(in crate::directx) const HIZ_MAX_MIPS: usize =
    concinnity_engine::render::hiz::spd::MAX_MIPS as usize;

// The hardware every init stage creates resources through, and whether the
// shader compiles it makes resolve from disk for hot reload.
struct InitGpu<'a> {
    hw: &'a DxHardware,
    hot_reload: bool,
}

// The feature gates every stage agrees on, resolved once from the world's post
// settings and the device.
struct Features {
    taa_enabled: bool,
    msaa_samples: u32,
    post_process: PostProcessParams,
    gbuffer_enabled: bool,
}

impl Features {
    // The adapter ceiling clamped to what the world asks for. A temporal
    // technique resolves it to one sample, which drops the resolve step and
    // makes `hdr.color` the scene spine; every PSO bakes the result into its
    // `SampleDesc`.
    fn msaa_samples(hw: &DxHardware, hdr_samples: u32) -> u32 {
        bootstrap::resolve_sample_count(bootstrap::query_msaa_samples(&hw.device), hdr_samples)
    }

    fn resolve(
        hw: &DxHardware,
        graphics: &ResolvedGraphics,
        hdr_samples: u32,
        upscaling: bool,
    ) -> Self {
        // A built upscaler needs the velocity buffer + the TAA-velocity
        // pre-pass PSOs, both of which live inside `TaaResources`, so the TAA
        // resources are built even if the world's `AntiAliasing.mode` is off;
        // the TAA *resolve* pass is still skipped (see
        // `record_frame::seed_inputs`), because the upscaler owns the temporal
        // accumulation.
        let plan = FeaturePlan::resolve(
            graphics,
            BackendFacts {
                rt_capable: hw.rt_capable,
                upscaling,
            },
        );
        let taa_enabled = plan.temporal;
        let msaa_samples = Self::msaa_samples(hw, hdr_samples);
        // Pair the authored tunables with the resolved mode's output flags. On
        // the SDR path both stay 0.0 and the shader runs the full ACES + gamma
        // + FXAA + LUT chain unchanged. Inside the HDR branch, `pq_output`
        // picks scRGB-linear passthrough (0.0) vs SMPTE ST 2084 in-shader
        // encode (1.0). Mirrors the Metal hop in `metal/init/mod.rs`. `setup`
        // may have already downgraded the encoding when
        // `CheckColorSpaceSupport(HDR10 PQ)` came back negative, so composing
        // after `setup` returns is what makes `hw.hdr_mode` the source of truth.
        let post_process = hw.hdr_mode.post_process_params(graphics.post_process());
        Self {
            taa_enabled,
            msaa_samples,
            post_process,
            // The unified G-buffer pre-pass exists when any screen-space
            // consumer needs it: SSR / SSGI / RT, SSAO, or TAA / FSR velocity.
            // `taa_enabled` already folds in temporal upscaling.
            gbuffer_enabled: plan.gbuffer,
        }
    }
}

impl DxContext {
    // Construct a fresh context (new device + window + swapchain) from the
    // assembled backend inputs (see `concinnity_engine::render::backend::init::BackendInit`).
    pub(crate) fn new(init: BackendInit<'_>) -> RenderResult<Self> {
        Self::build(init, None)
    }

    // The shared constructor. `reuse == None` acquires fresh hardware (the
    // normal `new` path); `reuse == Some` rebuilds only the world's content on
    // the retained device + window + swapchain for a live `cn edit` world
    // reload (see `reload_world`). Everything after the device/window
    // acquisition is identical: the same pipelines, buffers, textures, and
    // targets are built from `init` either way.
    fn build(
        init: BackendInit<'_>,
        reuse: Option<(DxHardware, bootstrap::DxgiSwapchain)>,
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
        // Clamped to the D3D12 1..16 range where the sampler is built.
        let anisotropy = graphics.texture_filtering.anisotropy;
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
        let output = (window.width, window.height);
        // Record this (main) thread so the `RenderBackend` mutation entry
        // points can `debug_assert_main_thread` against it; the Send invariant
        // rests on the context being touched from this thread alone.
        record_main_thread();

        let (hw, swapchain) = open_hardware(reuse, window, &graphics, validation)?;
        // Persisted pipeline library: seeded from disk when a blob for this
        // adapter exists, consulted by every PSO creation below. No-op on the
        // reload path, where it is already installed.
        super::pso_library::install(&hw.device, hw.adapter.as_ref());
        let gpu = InitGpu {
            hw: &hw,
            hot_reload,
        };

        let plan = cull::plan_cull(&world);
        let planar = effects::plan_planar(&fx, mirror_budget(planar, plan.n_cull));
        let planar_slots = planar.assignment.slots.clone();
        let rtv = RtvHeapLayout::compute(Features::msaa_samples(&hw, hdr_samples));
        let swapchain = heaps::build_swapchain(&gpu, swapchain, &rtv, vsync)?;
        let descriptors = descriptors::build_descriptors(
            &gpu,
            &SrvHeapParams {
                n_atlases: media.text_atlases.len(),
                gbuffer_srv_extra: heap_layout::GBUFFER_TARGETS,
                planar_resolve_srv_extra: planar.planes().len(),
                // Albedo and normal maps share ONE handle-indexed pool: the real
                // textures (a 1x1 white fallback stands in when there are none)
                // followed by the reserved fallback pair.
                albedo_count: media.textures.len().max(1),
                normal_count: FALLBACK_TEXTURE_COUNT,
                frames: hw.frames(),
            },
            anisotropy,
        )?;
        let upscale = effects::build_upscale(&gpu, &descriptors, output, &graphics)?;
        let features = Features::resolve(&hw, &graphics, hdr_samples, upscale.backend.is_some());
        let mut scene =
            scene_assets::build_scene_assets(&gpu, &descriptors, &media, &area_lights, &world)?;
        // The probe set, with the stand-in array in the SRV slot until a world
        // places a probe.
        let probe_gpu = crate::directx::probe_set::ProbeSetGpu::new(
            &gpu.hw.device,
            &gpu.hw.alloc,
            hw.frames(),
        )?;
        probe_gpu.stand_in.write_srv(
            &gpu.hw.device,
            descriptors.slot_cpu(descriptors.layout.probe_cubes_srv_slot),
        );
        let targets = targets::build_targets(
            &gpu,
            targets::TargetInputs {
                descriptors: &descriptors,
                swapchain: &swapchain,
                rtv: &rtv,
                upscale: &upscale,
                features: &features,
                output,
                clear_color,
            },
        )?;
        let uniforms = scene_data::build_uniforms(&gpu, &light_uniforms, &local_lights)?;
        let light_cull = scene_data::build_light_cull(&gpu)?;
        let shadow = shadow::build_shadow(&gpu, &descriptors, &targets, &shadows)?;
        let spot_shadow = shadow::build_spot_shadow(
            &gpu,
            &descriptors,
            &targets,
            &spot_shadows,
            shadows.map_size,
        )?;

        let cull = cull::build_cull(
            &gpu,
            cull::CullInputs {
                world: &world,
                world_shaders: &world_shaders,
                plan: &plan,
                descriptors: &descriptors,
                targets: &targets,
                albedo_count: scene.textures.len(),
                shadow_enabled: shadow.map_size > 0,
                spot_slices: spot_shadow.count() as usize,
                gbuffer_enabled: features.gbuffer_enabled,
                occlusion_two_pass: graphics.culling.two_pass_occlusion,
            },
        )?;
        // The blades draw under the GPU-driven pass's two root signatures, so a
        // world without that pass draws no grass.
        let grass = match (
            fx.grass,
            cull.main_bindless_root_sig.as_ref(),
            cull.prepass_root_sig.as_ref(),
        ) {
            (Some(field), Some(main), Some(prepass)) => Some(super::grass::GrassResources::build(
                &gpu.hw.alloc,
                field,
                super::grass::GrassRootSigs { main, prepass },
                features.msaa_samples,
                hot_reload,
            )?),
            _ => None,
        };
        let probe_prefilter = cull::build_probe_prefilter(&gpu, &cull)?;
        let text = text::build_text(&gpu, &descriptors, &media, swapchain.format)?;
        let composite = composite::build_composite(&gpu, swapchain.format)?;

        let post_descriptors = descriptors::build_post_descriptors(
            &descriptors,
            &swapchain,
            &rtv,
            gpu.hw.alloc.frames_in_flight(),
        );
        let quality_slots =
            effects::build_quality_slots(&gpu, &descriptors, &swapchain, &targets, &rtv);
        let post_device = effects::post_device(&gpu, &descriptors, &post_descriptors);
        let reflection_composite =
            ray_tracing::build_reflection_composite(&post_device, &targets, &graphics)?;
        let bloom = bloom::build_bloom(&post_device, &targets, output)?;
        let taa = effects::build_taa(&post_device, &features, &targets)?;
        let ssao =
            effects::build_ssao(&gpu, &post_device, &descriptors, &targets, graphics.ssao())?;
        let ssr = effects::build_ssr(&post_device, &targets, &graphics)?;
        let ssgi = effects::build_ssgi(&post_device, &targets, &graphics)?;
        let rt_reflections =
            ray_tracing::build_rt_reflections(&gpu, &post_descriptors, &targets, &graphics);
        let gbuffer =
            effects::build_gbuffer(&gpu, &targets, &quality_slots, features.gbuffer_enabled)?;

        let decal = effects::build_decals(&gpu, &descriptors, &targets, &scene, fx.decals)?;
        let fog = effects::build_fog(&gpu, &descriptors, &targets, &shadow, fx.fog)?;
        let particle =
            effects::build_particles(&gpu, &descriptors, &targets, &scene, fx.particles)?;
        let commands = commands::build_commands(&gpu)?;
        let frame_sync = commands::build_frame_sync(&gpu)?;
        let timestamps = commands::build_timestamps(&gpu);
        let auto_exposure = effects::build_auto_exposure(&gpu, &graphics)?;
        let atmosphere = effects::build_atmosphere(
            &gpu,
            &descriptors,
            &scene,
            fx.atmosphere,
            features.msaa_samples,
        )?;
        // The atmosphere's sky cube now holds the environment's reflections.
        if let Some((_, mips)) = atmosphere.as_ref().and_then(|a| a.prefilter()) {
            scene.env_map.prefilter_mip_count = mips;
        }

        let raymarch = effects::build_raymarch(
            &gpu,
            &descriptors,
            &targets,
            &shadow,
            &scene,
            atmosphere.as_ref(),
            &fx.sdf_volumes,
        )?;
        let sky = super::sky::DxSky::build(
            &gpu.hw.device,
            features.msaa_samples,
            media.env_map_background,
            hot_reload,
        )?;
        let planar_reflection = effects::build_planar_reflection(
            &gpu,
            &descriptors,
            &targets,
            planar,
            plan.n_cull,
            clear_color,
        )?;
        let water_sim =
            effects::build_water_sim(&gpu, &descriptors, &fx.water_surfaces, &fx.ocean)?;
        let transparent = effects::build_transparent(
            &gpu,
            effects::TransparentInputs {
                descriptors: &descriptors,
                targets: &targets,
                reflection_slots: quality_slots.glass_reflection,
                post: &post_descriptors,
                reflection_divisor: graphics.rt_reflections().map_or(1, |rt| rt.divisor),
                planar_slots: &planar_slots,
                glass_panels: &fx.glass_panels,
                water_surfaces: &fx.water_surfaces,
                draw_objects: &world.draw_objects,
            },
        )?;
        let rt = ray_tracing::build_ray_tracing(
            &gpu,
            ray_tracing::RtInputs {
                world: &world,
                scene: &scene,
                reflections: rt_reflections.as_ref(),
                transparent: transparent.as_ref(),
                launch,
            },
        );

        report_init(&hw);

        let mut ctx = Self {
            post: post_descriptors,
            retired_effects: concinnity_engine::render::buffers::retire_pool::TickedRetirePool::new(
                gpu.hw.alloc.frames_in_flight() as u64 + 1,
            ),
            swapchain,
            targets,
            upscale,
            shadow,
            spot_shadow,
            scene,
            descriptors,
            geometry_uploads: std::cell::RefCell::new(GeometryUploads::new(hw.frames())),
            skinned: SkinnedState::new(),
            uniforms,
            canopy: super::canopy::DxCanopy::new(&hw.alloc, hw.frames())?,
            light_cull,
            cull,
            text,
            composite,
            bloom: Some(bloom),
            post_process: features.post_process,
            pre_exposure: concinnity_engine::gfx::auto_exposure::PreExposure::UNIT,
            gbuffer,
            taa,
            ssao,
            ssr,
            ssgi,
            disocclusion: None,
            reflection_composite,
            rt_reflections,
            rt,
            decal,
            lines: super::line::LineState::empty(),
            raymarch,
            sky,
            atmosphere,
            transparent,
            water_sim,
            wake: concinnity_engine::render::environment::water::wake::WakeHistory::default(),
            wake_frame: None,
            planar_reflection,
            fog,
            particle,
            grass,
            grass_lod: std::cell::Cell::default(),
            canopy_field: fx.canopy,
            commands,
            frame_sync,
            frame_capture: crate::frame_capture::FrameCaptures::new(capture),
            current_frame: 0,
            frame_latency_reserved: false,
            stream: StreamState::new(hw.frames()),
            state: SceneState::new(
                DrawList::with_runtime_reserve(
                    world.draw_objects,
                    plan.n_instances,
                    world.n_chunk_max,
                ),
                clear_color,
            ),
            graph_scratch: std::cell::Cell::default(),
            pass_lists: std::cell::Cell::default(),
            submission: Vec::new(),
            instanced: DxInstanced::new(world.instanced_clusters),
            wireframe: crate::directx::wireframe::DxWireframe::default(),
            diagnostics: crate::directx::context::Diagnostics::default(),
            timestamps,
            auto_exposure,
            hot_reload: ShaderReload::new(hot_reload),
            world_shader: world_programs.cloned(),
            quality_slots,
            probe: ProbeState::new(probe_prefilter, probe_gpu),
            hw,
        };
        ctx.state.planner.shadow.cadence = shadows.cadence;
        ctx.sync_disocclusion()?;
        ctx.refresh_gates(&graphics);
        Ok(ctx)
    }
}

// A live editor reload hands over the hardware and swapchain (HDR was already
// negotiated on the unchanged swapchain, so `setup` is skipped). Otherwise
// `setup` negotiates HDR: a capable adapter + a `true` toggle yields a
// `RGBA16Float` scRGB swapchain, else `hw.hdr_mode` is `Sdr`.
fn open_hardware(
    reuse: Option<(DxHardware, bootstrap::DxgiSwapchain)>,
    window: &Window,
    graphics: &ResolvedGraphics,
    validation: bool,
) -> RenderResult<(DxHardware, bootstrap::DxgiSwapchain)> {
    if let Some(reuse) = reuse {
        return Ok(reuse);
    }
    // The swapchain config the caller's reload gate compares against.
    let swapchain_config = init::SwapchainConfig {
        frames_in_flight: graphics.frames_in_flight(),
        hdr_display: graphics.display.hdr,
        hdr_pq: graphics.display.pq,
    };
    bootstrap::setup(
        bootstrap::WindowConfig {
            title: window.title.as_str(),
            width: window.width,
            height: window.height,
            title_bar: window.title_bar,
        },
        validation,
        graphics.frame_pacing.vsync,
        swapchain_config,
    )
}

// A mirror renders the culled scene, so a world with no cull records has
// nothing to mirror: its reflectors keep the probe / sky path.
fn mirror_budget(planar: PlanarBudget, n_cull: usize) -> PlanarBudget {
    if n_cull == 0 {
        PlanarBudget {
            planes: 0,
            ..planar
        }
    } else {
        planar
    }
}

// Report the shader caches, and serialize the pipeline library now that every
// init-built PSO has populated it, then write the segment holding it and every
// shader artifact this init compiled; a crash mid-session then still leaves the
// next launch warm.
fn report_init(hw: &DxHardware) {
    crate::shader::cache::report_init();
    super::pso_library::serialize();
    crate::shader::pipeline_cache::report_init(super::pso_library::disk_state());
    crate::shader::runtime_cache::checkpoint();

    let pooled = hw.alloc.stats();
    tracing::info!(
        "device allocator: {} heap(s), {} KiB reserved for {} KiB of resources",
        pooled.block_count,
        pooled.reserved_bytes / 1024,
        pooled.in_use_bytes / 1024,
    );
}

impl DxContext {
    // Rebuild the world's GPU content in place for a live `cn edit` reload:
    // idle the GPU, hand the hardware and swapchain to a successor built from
    // `init`, then replace `self` with it. The D3D12 / DXGI objects are COM
    // ref-counted, so the successor's clones keep them alive through the
    // assignment that drops the old world; the window and the fullscreen
    // restore state move over. Only ever called when the swapchain config is
    // unchanged (the caller's `hot_swap_config` gate).
    //
    // On a content-build failure (essentially impossible for a pre-validated
    // editor edit built from the engine's built-in shaders) `self.hw.win_state`
    // is left `None`; the caller drops this backend and marks the session failed.
    pub(in crate::directx) fn apply_world_reload(
        &mut self,
        init: BackendInit<'_>,
    ) -> RenderResult<()> {
        self.wait_idle();
        let swapchain = bootstrap::DxgiSwapchain {
            handle: self.swapchain.handle.clone(),
            format: self.swapchain.format,
            allow_tearing: self.swapchain.allow_tearing,
        };
        let reuse = (self.hw.hand_over()?, swapchain);
        *self = Self::build(init, Some(reuse))?;
        Ok(())
    }
}
