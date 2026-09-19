//! DxContext construction. `build` resolves the backend inputs, then calls each
//! stage in dependency order. A stage builds one owned group or subsystem state
//! whole and returns it, borrowing the finished states it depends on:
//!
//!   bootstrap.rs     window, DXGI factory, adapter, device, queue, swapchain.
//!   heaps.rs         RTV heap with the back-buffer views, DSV and sampler heaps.
//!   descriptors.rs   shader-visible SRV heap, sampler handles, post-pass block.
//!   effects.rs       upscaler, screen-space passes, world effects.
//!   scene_assets.rs  area lights, IBL cubes, textures, color LUT, geometry.
//!   targets.rs       main depth, HDR scene target, transient pool.
//!   scene_data.rs    constant-buffer rings, clustered light binning.
//!   shadow.rs        cascade and spot shadow states.
//!   cull/            bindless pass, compute cull, Hi-Z, GPU-driven shadow, G-buffer.
//!   text.rs          text atlases and pipeline.
//!   composite.rs     composite pipeline.
//!   ray_tracing.rs   reflection composite, RT reflections, acceleration structure.
//!   bloom.rs         bloom mip chain and pipelines.
//!   commands.rs      command lists, frame sync, timestamp queries.
//!
//! `heap_layout.rs` holds every heap's slot layout, `pipelines.rs` the main-pass
//! root signature and PSO builders, and `adapter.rs` the adapter selection.

use concinnity_core::gfx::render_types::{FALLBACK_TEXTURE_COUNT, PostProcessParams};
use concinnity_core::render::backend_init::{self, BackendInit, PostSettings, WorldShader};
use concinnity_core::render::error::{RenderError, RenderResult};

use self::heap_layout::{RtvHeapLayout, SrvHeapParams};
use super::context::*;
use super::hot_reload::HotReloadState;
use super::post::bloom::bloom_mip_count;
use super::resources::skinning::SkinnedState;

mod adapter;
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
mod shadow;
mod targets;
mod text;

// Maximum Hi-Z mip count we reserve descriptor slots for. 15 mips covers
// every render target up to 16384 pixels in the larger dimension; an
// 8K display sits at 13. The Hi-Z resource clamps `mip_count` against this
// so the heap layout stays anchored even when the window resizes.
pub(in crate::directx) const HIZ_MAX_MIPS: usize = 15;

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
    ssao_enabled: bool,
    gbuffer_enabled: bool,
}

impl Features {
    fn resolve(hw: &DxHardware, post: &PostSettings) -> Self {
        // FSR3 needs the velocity buffer + the TAA-velocity pre-pass
        // PSOs, both of which live inside `TaaResources`. When upscale
        // is on we force the TAA resources to be built even if the
        // world's `PostProcessConfig.aa_mode` is off; the TAA *resolve*
        // pass is still skipped (see `record_frame::seed_inputs`),
        // because FSR owns the temporal accumulation.
        let taa_enabled = post.taa_enabled || post.temporal_upscaling;
        // The adapter ceiling clamped to what the world asks for. A temporal
        // technique resolves it to one sample, which drops the resolve step and
        // makes `hdr.color` the scene spine; every PSO bakes the result into its
        // `SampleDesc`.
        let msaa_samples = bootstrap::resolve_sample_count(
            bootstrap::query_msaa_samples(&hw.device),
            post.hdr_samples,
        );
        // Pair the authored tunables with the resolved mode's output flags. On
        // the SDR path both stay 0.0 and the shader runs the full ACES + gamma
        // + FXAA + LUT chain unchanged. Inside the HDR branch, `pq_output`
        // picks scRGB-linear passthrough (0.0) vs SMPTE ST 2084 in-shader
        // encode (1.0). Mirrors the Metal hop in `metal/init/mod.rs`. `setup`
        // may have already downgraded the encoding when
        // `CheckColorSpaceSupport(HDR10 PQ)` came back negative, so composing
        // after `setup` returns is what makes `hw.hdr_mode` the source of truth.
        let post_process = hw.hdr_mode.post_process_params(post.post_process);
        // Hardware ray-tracing capability. RT reflection resources + the
        // acceleration structure are built only when the world authored
        // `ray_traced_reflections` AND the GPU reports the DXR 1.1 tier inline
        // `RayQuery` needs; otherwise the renderer falls back to SSR.
        let rt_enabled = post.rt_reflections.is_some() && hw.rt_capable;
        if post.rt_reflections.is_some() && !hw.rt_capable {
            tracing::warn!(
                "ray_traced_reflections requested but the GPU does not report DXR \
                 tier 1.1; falling back to screen-space reflections"
            );
        }
        let ssao_enabled = post.ssao.is_some();
        Self {
            taa_enabled,
            msaa_samples,
            post_process,
            ssao_enabled,
            // The unified G-buffer pre-pass exists when any screen-space
            // consumer needs it: SSR / SSGI / RT, SSAO, or TAA / FSR velocity.
            // `taa_enabled` already folds in temporal upscaling.
            gbuffer_enabled: taa_enabled
                || ssao_enabled
                || post.ssr.is_some()
                || post.ssgi.is_some()
                || rt_enabled,
        }
    }
}

impl DxContext {
    // Construct a fresh context (new device + window + swapchain) from the
    // assembled backend inputs (see `concinnity_core::render::backend_init::BackendInit`).
    pub(crate) fn new(init: BackendInit<'_>) -> RenderResult<Self> {
        Self::build(init, None)
    }

    // The shared constructor. `reuse == None` acquires fresh hardware (the
    // normal `new` path); `reuse == Some` rebuilds only the world's content on
    // the retained device + window + swapchain for a live `cn editor` world
    // reload (see `reload_world`). Everything after the device/window
    // acquisition is identical: the same pipelines, buffers, textures, and
    // targets are built from `init` either way.
    fn build(
        init: BackendInit<'_>,
        reuse: Option<(DxHardware, bootstrap::DxgiSwapchain)>,
    ) -> RenderResult<Self> {
        let BackendInit {
            window,
            validation,
            // D3D12 always renders FRAMES=3 in flight; this is retained only so
            // `hot_swap_config` can report the world's request for the reload gate.
            frames_in_flight,
            vsync,
            clear_color,
            hot_reload,
            // DirectX retains the presented back-buffer index unconditionally,
            // so capture needs no arming here.
            capture: _,
            embedded_surface: _,
            scene: world,
            shaders: world_shaders,
            media,
            light_uniforms,
            local_lights,
            spot_shadows,
            area_lights,
            shadows,
            // Clamped to the D3D12 1..16 range where the sampler is built.
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
            .ok_or_else(|| RenderError::Other("BackendInit carried no shaders".to_string()))?;
        let output = (window.width, window.height);
        // Record this (main) thread so the `RenderBackend` mutation entry
        // points can `debug_assert_main_thread` against it; the Send invariant
        // rests on the context being touched from this thread alone.
        super::context::record_main_thread();

        // The swapchain config the caller's reload gate compares against.
        let swapchain_config = backend_init::SwapchainConfig {
            frames_in_flight: frames_in_flight.max(1),
            hdr_display: post.hdr_display,
            hdr_pq: post.hdr_pq,
        };
        // A live editor reload hands over the hardware and swapchain (HDR was
        // already negotiated on the unchanged swapchain, so `setup` is skipped).
        // Otherwise `setup` negotiates HDR: a capable adapter + a `true` toggle
        // yields a `RGBA16Float` scRGB swapchain, else `hw.hdr_mode` is `Sdr`.
        let (hw, swapchain) = match reuse {
            Some(reuse) => reuse,
            None => bootstrap::setup(
                bootstrap::WindowConfig {
                    title: window.title.as_str(),
                    width: output.0,
                    height: output.1,
                    title_bar: window.title_bar,
                },
                validation,
                vsync,
                swapchain_config,
            )?,
        };
        let features = Features::resolve(&hw, &post);
        // Persisted pipeline library: seeded from disk when a blob for this
        // adapter exists, consulted by every PSO creation below. No-op on the
        // reload path, where it is already installed.
        super::pso_library::install(&hw.device, hw.adapter.as_ref());
        let gpu = InitGpu {
            hw: &hw,
            hot_reload,
        };

        let planar = effects::plan_planar(&fx, planar_planes);
        let bloom_count = bloom_mip_count(output.0, output.1) as usize;
        let rtv = RtvHeapLayout::compute(bloom_count, features.msaa_samples);
        let swapchain = heaps::build_swapchain(&gpu, swapchain, &rtv, vsync)?;
        let descriptors = descriptors::build_descriptors(
            &gpu,
            &SrvHeapParams {
                n_atlases: media.text_atlases.len(),
                bloom_count,
                ssao_srv_extra: heap_layout::SSAO_TARGETS,
                gbuffer_srv_extra: heap_layout::GBUFFER_TARGETS,
                rt_output_srv_extra: heap_layout::RT_OUTPUT_TARGETS,
                refl_composite_srv_extra: heap_layout::REFL_COMPOSITE_TARGETS,
                planar_resolve_srv_extra: planar.representatives.len(),
                // Albedo and normal maps share ONE handle-indexed pool: the real
                // textures (a 1x1 white fallback stands in when there are none)
                // followed by the reserved fallback pair.
                albedo_count: media.textures.len().max(1),
                normal_count: FALLBACK_TEXTURE_COUNT,
            },
            anisotropy,
        )?;
        let upscale = effects::build_upscale(&gpu, &descriptors, output, &post)?;
        let scene =
            scene_assets::build_scene_assets(&gpu, &descriptors, &media, &area_lights, &world)?;
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
        let uniforms = scene_data::build_uniforms(&gpu, light_uniforms, &local_lights)?;
        let light_cull = scene_data::build_light_cull(&gpu, &local_lights)?;
        let shadow = shadow::build_shadow(
            &gpu,
            &descriptors,
            &targets,
            &shadows,
            &uniforms.light_uniforms,
        )?;
        let spot_shadow = shadow::build_spot_shadow(
            &gpu,
            &descriptors,
            &targets,
            &spot_shadows,
            shadows.map_size,
        )?;

        let plan = cull::plan_cull(&world);
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
                gbuffer_enabled: features.gbuffer_enabled,
                occlusion_two_pass: post.occlusion_two_pass,
            },
        )?;
        let probe_prefilter = cull::build_probe_prefilter(&gpu, &cull)?;
        let text = text::build_text(&gpu, &descriptors, &media, swapchain.format)?;
        let composite = composite::build_composite(&gpu, swapchain.format)?;

        let post_descriptors = descriptors::build_post_descriptors(&descriptors, &swapchain, &rtv);
        let quality_slots =
            effects::build_quality_slots(&gpu, &descriptors, &swapchain, &targets, &rtv);
        let reflection_composite =
            ray_tracing::build_reflection_composite(&gpu, &quality_slots, &targets, &post)?;
        let bloom = bloom::build_bloom(
            &gpu,
            bloom::BloomInputs {
                descriptors: &descriptors,
                swapchain: &swapchain,
                rtv: &rtv,
                targets: &targets,
                output,
            },
        )?;
        let post_device = effects::post_device(&gpu, &descriptors, &post_descriptors);
        let taa = effects::build_taa(&post_device, &features, &targets)?;
        let ssao = effects::build_ssao(&gpu, &descriptors, &targets, &quality_slots, post.ssao)?;
        let ssr = effects::build_ssr(&post_device, &targets, &post)?;
        let ssgi = effects::build_ssgi(&post_device, &targets, &post)?;
        let rt_reflections =
            ray_tracing::build_rt_reflections(&gpu, &quality_slots, &targets, &post);
        let gbuffer =
            effects::build_gbuffer(&gpu, &targets, &quality_slots, features.gbuffer_enabled)?;

        let decal = effects::build_decals(&gpu, &descriptors, &targets, &scene, fx.decals)?;
        let fog = effects::build_fog(
            &gpu,
            &descriptors,
            &targets,
            &shadow,
            fx.fog,
            &uniforms.light_uniforms,
        )?;
        let particle =
            effects::build_particles(&gpu, &descriptors, &targets, &scene, fx.particles)?;
        let commands = commands::build_commands(&gpu)?;
        let frame_sync = commands::build_frame_sync(&gpu)?;
        let timestamps = commands::build_timestamps(&gpu);
        let auto_exposure = effects::build_auto_exposure(&gpu, &post)?;
        let raymarch = effects::build_raymarch(
            &gpu,
            &descriptors,
            &targets,
            &shadow,
            &scene,
            &fx.sdf_volumes,
        )?;
        let planar_reflection = effects::build_planar_reflection(
            &gpu,
            &descriptors,
            &targets,
            &planar,
            plan.n_cull,
            clear_color,
        )?;
        let transparent = effects::build_transparent(
            &gpu,
            effects::TransparentInputs {
                descriptors: &descriptors,
                targets: &targets,
                reflection_slots: quality_slots.glass_reflection,
                reflection_divisor: post.rt_reflections.map_or(1, |rt| rt.divisor),
                planar: &planar,
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
                post: &post,
            },
        );

        crate::shader::cache::report_init();
        // Serialize the pipeline library now that every init-built PSO has
        // populated it, then write the segment holding it and every shader
        // artifact this init compiled; a crash mid-session then still leaves
        // the next launch warm.
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

        Ok(Self {
            post: post_descriptors,
            swapchain,
            targets,
            upscale,
            shadow,
            spot_shadow,
            scene,
            descriptors,
            mesh_stream: Default::default(),
            chunk_stream: Default::default(),
            skinned: SkinnedState::new(),
            uniforms,
            light_cull,
            cull,
            text,
            composite,
            bloom,
            post_process: features.post_process,
            gbuffer,
            model_history: Default::default(),
            taa,
            ssao,
            ssr,
            ssgi,
            reflection_composite,
            rt_reflections,
            rt,
            decal,
            lines: super::line::LineState::empty(),
            raymarch,
            transparent,
            planar_reflection,
            fog,
            particle,
            commands,
            frame_sync,
            current_frame: 0,
            stream: StreamState::new(),
            draw: DrawState::new(world.draw_objects, plan.n_instances, world.n_chunk_max),
            instanced: DxInstanced::new(world.instanced_clusters),
            view: ViewState::new(clear_color),
            wireframe: Default::default(),
            diagnostics: Default::default(),
            timestamps,
            auto_exposure,
            hot_reload: HotReloadState::spawn(hot_reload),
            world_shader: world_programs.cloned(),
            quality_slots,
            probe: ProbeState::new(probe_prefilter),
            hw,
        })
    }
}

impl DxContext {
    // Rebuild the world's GPU content in place for a live `cn editor` reload:
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
        *self = DxContext::build(init, Some(reuse))?;
        Ok(())
    }
}
