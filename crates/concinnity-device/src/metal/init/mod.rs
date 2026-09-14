//! MtlContext construction. `build` resolves the backend inputs, then calls each
//! stage in dependency order. A stage builds one owned group or subsystem state
//! whole and returns it, borrowing the finished states it depends on:
//!
//!   bootstrap.rs     window, MTKView, device, command queues, EDR negotiation.
//!   effects.rs       upscaler, TAA, SSAO, SSR, G-buffer pre-pass, SSGI, auto-exposure.
//!   scene_assets.rs  geometry, light tables, LTC, textures, samplers, IBL, color LUT.
//!   targets.rs       depth states, HDR scene target, transient pool, bloom mips.
//!   scene_data.rs    clustered light binning.
//!   shadow.rs        cascade and spot shadow states.
//!   cull/            bindless main pass, compute cull, Hi-Z, GPU-driven shadow,
//!                    probe prefilter, instance records.
//!   arg_buffers.rs   bindless texture and sampler blocks, probe cube encoder.
//!   text.rs          text atlases and pipeline.
//!   composite.rs     composite pipeline and sampler.
//!   bloom.rs         bloom pipelines.
//!   world_fx.rs      decals, fog, particles, water, glass, planar reflections, raymarch.
//!   ray_tracing.rs   RT reflection pipelines, acceleration structure.
//!   commands.rs      frame rings, pass timing diagnostics.
//!
//! `pipelines.rs` holds the main-pass and shadow pipeline builders the stages
//! share with the shader hot reload.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::gfx::render_types::{ClusterParams, PostProcessParams};
use concinnity_core::gfx::transform::IDENTITY;
use concinnity_core::render::backend_init::{BackendInit, PostSettings};
use concinnity_core::render::error::RenderResult;
use concinnity_core::render::model_history::ModelHistory;

use self::effects::EffectSettings;
use super::context::*;
use super::frame_pacing::FrameInFlight;
use super::line::LineState;
use super::post::UpscaleState;
use super::resources::skinning::SkinnedState;

mod arg_buffers;
mod bloom;
mod bootstrap;
mod commands;
mod composite;
mod cull;
pub(super) mod effects;
pub(crate) mod pipelines;
pub(super) mod ray_tracing;
mod scene_assets;
mod scene_data;
mod shadow;
pub(super) mod targets;
mod text;
pub(super) mod world_fx;

pub(crate) use bootstrap::set_display_sync;

// The hardware every init stage creates resources through, whether the shader
// compiles it makes resolve from disk for hot reload, and the frames-in-flight
// depth the per-frame rings are sized to.
struct InitGpu<'a> {
    hw: &'a MtlHardware,
    hot_reload: bool,
    frames_in_flight: usize,
}

// The feature gates every stage agrees on, resolved once from the world's post
// settings, the drawable and the upscaler.
struct Features {
    // True when the world has 3D scene content, which builds the GPU-driven main
    // pass and the bloom pipelines.
    scene: bool,
    // The sample count every main-pass pipeline, HDR target, planar mirror and
    // probe face is built at. One whenever a temporal technique runs, which
    // drops the MSAA attachments and the resolve entirely.
    hdr_samples: u32,
    // Drawable resolution, where bloom and composite operate.
    output: (u32, u32),
    // Resolution the 3D scene and most post passes draw at: the upscaler's input
    // when MetalFX runs, the drawable resolution otherwise.
    render: (u32, u32),
    taa_enabled: bool,
    ssao_enabled: bool,
    gbuffer_enabled: bool,
    post_process: PostProcessParams,
}

impl Features {
    fn resolve(
        hw: &MtlHardware,
        post: &PostSettings,
        scene: bool,
        output: (u32, u32),
        upscale: &UpscaleState,
    ) -> Self {
        let hdr_samples = post.hdr_samples.max(1);
        tracing::info!("metal HDR target: {hdr_samples}x MSAA");
        // Render resolution comes from the scaler, which owns the clamp to the
        // device's supported range.
        let render = match &upscale.scaler {
            Some(u) => (u.input_width, u.input_height),
            None => output,
        };
        // With the MetalFX scaler doing temporal accumulation, the TAA pass
        // is bypassed but the velocity pre-pass and projection jitter stay
        // on (the scaler consumes both). `taa_enabled` is what the engine
        // carries downstream; the asset `taa` flag is ignored when upscaling
        // is on.
        let upscaling_active = upscale.scaler.is_some();
        let taa_enabled = post.taa_enabled && !upscaling_active;
        let needs_velocity = taa_enabled || upscaling_active;
        Self {
            scene,
            hdr_samples,
            output,
            render,
            taa_enabled,
            ssao_enabled: post.ssao.is_some(),
            gbuffer_enabled: EffectSettings::from_post(post).gbuffer_needed(needs_velocity),
            // Pair the authored tunables with the resolved mode's output flags.
            // On the SDR path both flags stay 0.0 and the shader runs the full
            // ACES + gamma + FXAA + LUT chain unchanged. On the HDR path
            // `hdr_output` lights up; `pq_output` further picks PQ-encode vs
            // scRGB-linear passthrough inside that branch.
            post_process: hw.hdr_mode.post_process_params(post.post_process),
        }
    }
}

impl MtlContext {
    // Create a window and Metal render pipeline from the assembled backend
    // inputs (see `concinnity_core::render::backend_init::BackendInit` for per-field docs).
    // The shadow pass is engine-internal and enabled whenever
    // `shadows.map_size > 0`.
    pub(crate) fn new(init: BackendInit<'_>) -> RenderResult<Self> {
        Self::build(init, None)
    }

    // The shared constructor. `reuse == None` creates a fresh device + command
    // queue + window (the normal `new` path); `reuse == Some` rebuilds the
    // world's content on the hardware an outgoing context handed over for a
    // live `cn editor` reload, keeping its window so a save does not recreate
    // it. Everything after the hardware is identical: the same pipelines,
    // buffers, textures, and targets are built from `init` either way.
    fn build(init: BackendInit<'_>, reuse: Option<MtlHardware>) -> RenderResult<Self> {
        let BackendInit {
            window,
            // The Metal validation layer is enabled by the CLI re-execing with
            // MTL_DEBUG_LAYER, not through this flag.
            validation: _,
            frames_in_flight,
            vsync,
            clear_color,
            hot_reload,
            capture,
            embedded_surface,
            // `n_skinned` and `n_chunk_max` are unused on Metal: the object /
            // draw-args transient rings and the cull ICB auto-grow to
            // `cull_count()` each frame (the skinned count is set later in
            // `upload_skinned`, and resident chunks fold into the per-frame
            // rebuild), so no init-time sizing is needed. DX/VK pre-size fixed
            // buffers from these.
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
            requirements,
        } = init;

        // Scene-less (UI / text only) worlds clamp the HDR / bloom / effect
        // targets to 1x1; keyed off the derived requirements rather than raw
        // vertex presence so a vertex-less world that still renders 3D content
        // (SDF volumes, water, glass) keeps full-size targets.
        let (hw, output) = bootstrap::setup(
            reuse,
            bootstrap::WindowConfig {
                title: window.title.as_str(),
                width: window.width,
                height: window.height,
                title_bar: window.title_bar,
                geometry_less: !requirements.scene,
                capture_enabled: capture,
                embedded: embedded_surface,
            },
            bootstrap::HdrRequest {
                display_requested: post.hdr_display,
                pq_requested: post.hdr_pq,
            },
            frames_in_flight,
            vsync,
        )?;
        let gpu = InitGpu {
            hw: &hw,
            hot_reload,
            frames_in_flight,
        };
        let upscale = effects::build_upscale(&gpu, output, &post);
        let features = Features::resolve(&hw, &post, requirements.scene, output, &upscale);
        let vert_desc = pipelines::make_vertex_descriptor();

        let scene = scene_assets::build_scene_assets(
            &gpu,
            scene_assets::SceneInputs {
                world: &world,
                media: &media,
                local_lights: &local_lights,
                area_lights: &area_lights,
                anisotropy,
                bindless: features.scene,
            },
        )?;
        let targets = targets::build_targets(&gpu, &features)?;
        let light_cull = scene_data::build_light_cull(&gpu, &local_lights)?;
        let shadow = shadow::build_shadow(&gpu, &vert_desc, &shadows, &light_uniforms)?;
        let spot_shadow = shadow::build_spot_shadow(&gpu, &spot_shadows, shadows.map_size)?;
        let cull = cull::build_cull(
            &gpu,
            cull::CullInputs {
                world_shaders: &world_shaders,
                vert_desc: &vert_desc,
                features: &features,
                shadow_enabled: shadow.pipeline_state.is_some(),
                occlusion_two_pass: post.occlusion_two_pass,
            },
        )?;
        let probe = cull::build_probe(&gpu, &cull)?;
        let arg_buffers = arg_buffers::build_arg_buffers(&gpu, &cull, &scene, &shadow)?;
        let text = text::build_text(&gpu, &media.text_atlases)?;
        let composite = composite::build_composite(&gpu)?;
        let bloom_pipelines = bloom::build_bloom(&gpu, features.scene)?;

        let post_device = effects::post_device(&gpu, &composite, &scene);
        let settings = EffectSettings::from_post(&post);
        let taa = effects::build_taa(&post_device, features.taa_enabled, features.render)?;
        let ssao = effects::build_ssao(&hw.allocator, &settings, features.render, hot_reload)?;
        let ssr = effects::build_ssr(&post_device, &settings, features.render)?;
        let gbuffer = effects::build_gbuffer(
            &hw.device,
            features.gbuffer_enabled,
            features.render,
            hot_reload,
        )?;
        let ssgi = effects::build_ssgi(&post_device, &settings, features.render)?;
        let auto_exposure =
            effects::build_auto_exposure(&hw.device, &settings, frames_in_flight, hot_reload)?;

        let decal = world_fx::build_decals(&gpu, fx.decals)?;
        let fog = world_fx::build_fog(&gpu, fx.fog)?;
        let particle = world_fx::build_particles(&gpu, fx.particles)?;
        let planar = world_fx::plan_planar(&fx.water_surfaces, &fx.glass_panels, planar_planes);
        let n_water = fx.water_surfaces.len();
        let water = world_fx::build_water(&gpu, &fx.water_surfaces, &planar.slots[..n_water])?;
        let glass = world_fx::build_glass(
            &gpu,
            &fx.glass_panels,
            &planar.slots[n_water..],
            &world.draw_objects,
        )?;
        let planar_reflection = world_fx::build_planar_reflection(&gpu, &planar, &features)?;
        let raymarch = world_fx::build_raymarch(&gpu, &fx.sdf_volumes)?;

        let diagnostics = commands::build_diagnostics(&gpu);
        let rt = ray_tracing::build_ray_tracing(
            &gpu,
            ray_tracing::RtInputs {
                world: &world,
                scene: &scene,
                glass: &glass,
                post: &post,
            },
        )?;
        let instanced = cull::build_instanced(world.instanced_clusters, scene.textures.len());
        let rings = commands::build_rings(&gpu);

        let ctx = Self {
            last_present_texture: None,
            cull,
            arg_buffers,
            draw: DrawState::new(world.draw_objects, &instanced),
            instanced,
            view: ViewState::new(clear_color),
            scene,
            light_uniforms,
            shadow,
            spot_shadow,
            probe,
            text,
            targets,
            composite,
            bloom_pipelines,
            post_process: features.post_process,
            taa,
            prev_view_proj: IDENTITY,
            upscale,
            ssao,
            ssr,
            gbuffer,
            ssgi,
            rt,
            lines: LineState::new(frames_in_flight),
            decal,
            fog,
            light_cull,
            cluster_params: ClusterParams::ZERO,
            particle,
            auto_exposure,
            hot_reload: HotReloadState::spawn(hot_reload),
            world_shader: world_shaders[0].programs.cloned(),
            capture,
            model_history: ModelHistory::new(),
            skinned: SkinnedState::new(),
            geometry_alloc: GeometryAllocators::default(),
            diagnostics,
            frame_pacing: FrameInFlight::new(frames_in_flight),
            frames_in_flight: frames_in_flight.max(1),
            frame_ring_index: 0,
            rings,
            water,
            planar_reflection,
            glass,
            raymarch,
            hw,
        };
        let pooled = ctx.hw.allocator.stats();
        tracing::info!(
            "device allocator: {} heap(s), {} KiB reserved for {} KiB of resources",
            pooled.block_count,
            pooled.reserved_bytes / 1024,
            pooled.in_use_bytes / 1024,
        );
        // Tally the raymarch metallib cache (the only shader-cache client on
        // Metal; everything else precompiles at build time).
        crate::shader::cache::report_init();
        crate::shader::runtime_cache::checkpoint();
        Ok(ctx)
    }

    // Re-upload a new world's GPU content onto this live context, reusing the
    // retained device + command queue + window instead of recreating them.
    // Drives the `cn editor` live SAVE: after a structural edit recompiles the
    // blobs, GraphicsSystem transplants the running backend into the rebuilt
    // world and calls this so the edit applies with no window recreation.
    //
    // The GPU is idled so no in-flight command buffer still references the old
    // content, then a fresh context is `build`t on the hardware this one hands
    // over (`MtlHardware::hand_over`) and moved into `*self`, whose drop frees
    // the old content with no window left to close. Only ever called when the
    // swapchain config is unchanged (the caller's `hot_swap_config` gate), so
    // the frames-in-flight and HDR request still match. On a build failure the
    // handed-over window closes with the dropped hardware, and the caller drops
    // this backend.
    pub(super) fn apply_world_reload(&mut self, init: BackendInit<'_>) -> RenderResult<()> {
        debug_assert_main_thread("apply_world_reload");
        self.wait_idle();
        let reuse = self.hw.hand_over()?;
        *self = MtlContext::build(init, Some(reuse))?;
        Ok(())
    }
}
