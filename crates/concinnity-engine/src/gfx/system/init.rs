// GraphicsSystem one-time setup: backend creation, draw-list build, and the
// shader / texture / streaming wiring performed on the first tick.

use concinnity_core::bake::font;
use concinnity_core::bake::texture;
use concinnity_core::components::DebugHud;
use concinnity_core::components::KeyBinding;
use concinnity_core::components::ShaderPrograms;
use concinnity_core::components::SkeletonJoint;
use concinnity_core::components::SkinnedMesh;
use concinnity_core::components::Sprite;
use concinnity_core::components::StatHud;
use concinnity_core::components::Story;
use concinnity_core::components::SubMeshRef;
use concinnity_core::components::TextInput;
use concinnity_core::components::TextLabel;
use concinnity_core::components::Transform;
use concinnity_core::components::WindowMode;
use concinnity_core::components::build_skeleton_from_joint_defs;
use concinnity_core::components::hdr_sample_count;
use concinnity_core::components::{
    BlockType, Camera3D, GraphicsConfig, HitRegion, Material, Model, PostProcessConfig, Shader,
    ShaderStage, StreamingConfig, VoxelWorld, Window,
};
use concinnity_core::ecs::FontHandle;
use concinnity_core::ecs::FrameRateCap;
use concinnity_core::ecs::MaterialHandle;
use concinnity_core::ecs::MenuOverride;
use concinnity_core::ecs::OverlayImages;
use concinnity_core::ecs::PayloadLocator;
use concinnity_core::ecs::PipelineContext;
use concinnity_core::ecs::SkinnedMeshHandle;
use concinnity_core::ecs::TextureHandle;
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::geometry::payload_joints_to_defs;
use concinnity_core::gfx::{mesh_payload, render_types};
use concinnity_core::render::{backend, backend_init, text};
use concinnity_core::resource::ColorLutTable;
use concinnity_core::resource::EnvironmentMapTable;
use concinnity_core::resource::FontTable;
use concinnity_core::resource::MaterialTable;
use concinnity_core::resource::SkinnedMeshTable;
use concinnity_core::resource::TextureTable;
use concinnity_core::window::display_mode;
use concinnity_host::store::blob::blob_path;
use concinnity_host::store::blob::payload_section_start;

use super::backend_handoff::RuntimeHandoff;
use super::blob_release::{blobs_to_release, retained_blobs};
use super::draw_geometry::{auto_seed_probe_placements, declared_probe_placements};
use super::hot_reload_sources::{
    HotReloadSources, capture_hot_reload_sources, mesh_source_map, procedural_mesh_snapshot,
    procedural_mesh_source_map,
};
use super::prop_draws::PropDrawInputs;
use super::scene_lights::gather_lights;
use super::skinned_templates::{SkinnedSkeletonEntry, SkinnedUpload};
use super::stream_plan::{StreamGeometry, StreamingSetup, plan_stream_geometry};
use super::texture_payloads::{TexturePayloads, decode_texture_payloads};
use super::world_fx::drain_world_fx;
use super::*;
use crate::app::run::LaunchRequest;
use crate::gfx::draw_list;
use crate::gfx::material_entry::MaterialEntry;
use crate::settings::system::{SettingsSlot, SettingsState};

// The resolved render config `init_render_settings` returns beside the live
// settings state: the packed post-processing config handed to the backend ctor,
// the quality ceiling (planar-reflection budget), and the drained StreamingConfig.
struct ResolvedRenderConfig {
    post: backend_init::PostSettings,
    quality_ceiling: crate::gfx::quality_preset::QualityCeiling,
    streaming_config: Option<StreamingConfig>,
    // Raw world ambient (PostProcessConfig::ambient_intensity, no user override),
    // folded into the static LightUniforms built later in init.
    world_ambient_intensity: f32,
}

// Decoded geometry for one SkinnedMesh, produced in the order cook assigned
// handles (the table index IS the `SkinnedMeshHandle` keying the animation
// correlation web): its handle, interned name id, the baked mesh, its vertices,
// LOD0 indices, the bind-pose joint defs, its morph targets, and LOD alternates.
struct SkinnedGeometry {
    handle: SkinnedMeshHandle,
    name_id: AssetId,
    mesh: SkinnedMesh,
    vertices: Vec<mesh_payload::SkinnedVertex>,
    indices: Vec<u16>,
    joint_defs: Vec<SkeletonJoint>,
    morphs: mesh_payload::PayloadMorphs,
    lod_alternates: Vec<(f32, Vec<u16>)>,
}

// Assembled skinned-mesh GPU inputs from `assemble_skinned_meshes`: the shared
// skinned vertex/index buffers, the per-slot draw objects (templates + their
// hidden pre-reserved instance copies), the per-mesh skeleton bookkeeping, the
// (template, copy) pool reservations, per-slot morph targets, and the hot-reload
// source map.
struct SkinnedMeshAssembly {
    vertices: Vec<mesh_payload::SkinnedVertex>,
    // Absolute indices into the shared skinned vertex buffer, so u32 rather
    // than the per-mesh u16 the payload carries.
    indices: Vec<u32>,
    draw_objects: Vec<render_types::SkinnedDrawObject>,
    skeletons: Vec<SkinnedSkeletonEntry>,
    pool_reservations: Vec<(usize, usize)>,
    morphs: Vec<Option<std::sync::Arc<mesh_payload::PayloadMorphs>>>,
    source_map: super::hot_reload_sources::SkinnedMeshSourceMap,
}

// The shared texture pool decoded from the TextureTable by `decode_texture_table`:
// each texture's payload locator (dense by pool slot / cook TextureHandle), the
// dev-only file-backed source map + name->slot index (cn debug hot-reload / spawn
// by name), and the pool size.
struct TextureTableDecode {
    locators: Vec<PayloadLocator>,
    source_map: super::hot_reload_sources::TextureSourceMap,
    name_to_slot: std::collections::HashMap<AssetId, usize>,
    count: usize,
}

// One world Shader as decoded at init: its compiled programs, or nothing for
// a bucket a non-start scene owns.
#[derive(Default)]
struct DecodedShader {
    programs: Option<ShaderPrograms>,
    // The payload was left undecoded because a scene other than the start scene
    // owns this bucket.
    deferred: bool,
}

// Where the streaming pump re-reads a deferred bucket's stage container: the
// blob's byte range when the world is disk-backed (`cn run`, so the bytes
// never stay RAM-resident), else a copy of the in-memory payload.
fn deferred_shader_source(
    ctx: &mut PipelineContext,
    locator: &PayloadLocator,
    blob_disk_backed: bool,
) -> Result<crate::gfx::streaming::shader::ShaderPayloadSource, String> {
    use crate::gfx::streaming::shader::ShaderPayloadSource;
    if !blob_disk_backed {
        let bytes = ctx
            .read_payload(locator)
            .map_err(|e| format!("{e:?}"))?
            .to_vec();
        return Ok(ShaderPayloadSource::Bytes(bytes));
    }
    let path = blob_path(locator.blob_index)
        .ok_or_else(|| format!("blob {}: no blob layout installed", locator.blob_index))?;
    let start = payload_section_start(&path).map_err(|e| format!("{e:?}"))?;
    Ok(ShaderPayloadSource::Disk {
        path,
        offset: start + locator.offset,
        len: locator.len,
    })
}

struct DecodedShaders {
    locators: Vec<PayloadLocator>,
    source_map: super::hot_reload_sources::ShaderStageSourceMap,
    // One entry per world Shader, in drain order == cook handle order, so a
    // baked ShaderHandle value indexes this directly. Entry 0 is the world
    // default pipeline's program.
    shaders: Vec<DecodedShader>,
}

// The text/sprite atlas pool from `decode_text_atlases`: RGBA atlases (font
// atlases first, dense by FontHandle, then the built-in fallback face when some
// text names no Font, then appended sprite/story textures) and the blob indices
// the font payloads occupy (for the blob-release step).
struct TextAtlases {
    atlases: Vec<(u32, u32, Vec<u8>)>,
    font_blob_indices: Vec<u32>,
}

// Whether any text in the world names no Font, and so has no face to draw with
// unless one is registered as the fallback.
fn font_less_text(ctx: &PipelineContext) -> bool {
    ctx.query::<TextLabel>().any(|l| l.font.is_none())
        || ctx.query::<TextInput>().any(|t| t.font.is_none())
}

impl GraphicsSystem {
    // Resolve every render setting (window, quality preset + ceiling, post-process
    // tunables, shadows, streaming caps, keymap) into a fresh settings state, sync
    // the settings-menu value labels, and return it with the config the rest of
    // init needs.
    fn init_render_settings(
        &mut self,
        ctx: &mut PipelineContext,
        launch: &LaunchRequest,
    ) -> (ResolvedRenderConfig, SettingsState) {
        let mut settings = SettingsState::new();
        // Persisted settings-menu choices override the world's authored defaults
        // below (each field is None when the user never changed that setting).
        let user_graphics = self.persisted_settings().graphics;
        settings.persisted_graphics = user_graphics.clone();

        // Detect the GPU before the backend is built so the auto-config quality
        // ceiling can influence the render targets / effect pipelines sized at
        // backend init. Held on the settings state for the menu's preset label.
        settings.gpu_profile = self.detect_gpu_profile();
        // Published so readouts outside the graphics system (the editor's Health
        // panel) can size live VRAM against the device's budget without reaching
        // for the backend itself.
        ctx.insert_resource(settings.gpu_profile);
        crate::crash::note(
            "gpu",
            &format!(
                "{:?} {:?}",
                settings.gpu_profile.vendor, settings.gpu_profile.tier
            ),
        );
        // Resolve the master quality preset. The launch's `--quality-preset` flag
        // wins first and is never persisted, so a test / CI / GPU probe can force
        // a preset (e.g. `custom` for no clamp) without touching settings.bin.
        // Otherwise the persisted choice; `None` there = never configured (a first
        // launch, or a settings file written before the preset existed): seed
        // `Auto` and persist once, which records the detection without baking any
        // per-field value (the per-field overrides keep their `None = world
        // default` meaning). `Auto` re-resolves from the detected tier each launch;
        // `Custom` / an unclassified GPU impose no ceiling.
        use crate::gfx::quality_preset::QualityPreset;
        let active_preset = launch
            .resolve_quality_preset(user_graphics.quality_preset)
            .unwrap_or_else(|| {
                self.seed_first_launch_preset();
                QualityPreset::Auto
            });
        // Hold the resolved preset as the live value the settings-menu master
        // row cycles (and that an individual quality-row change flips to Custom).
        settings.quality_preset = active_preset;
        let quality_ceiling =
            crate::gfx::quality_preset::resolve_ceiling(active_preset, &settings.gpu_profile);
        tracing::info!(
            "auto-config: GPU tier {:?}, quality preset {:?}",
            settings.gpu_profile.tier,
            active_preset,
        );

        if let Some(w) = ctx.drain::<Window>().into_iter().next() {
            settings.window_args = w;
        }
        // Capture the DebugHud chip ids (cursor, camera, sys, passes stack
        // order) so the frame step can anchor them to the top-right of the
        // window. Passes is last because it grows/shrinks with the frame's step
        // count, so keeping it at the bottom leaves the fixed-height chips
        // unshifted. The DebugHud component is queried (not drained) by its
        // system, so it is still present here; absent fields are skipped.
        self.debug_hud_chips = ctx
            .query::<DebugHud>()
            .next()
            .map(|d| {
                [d.mouse_label, d.camera_label, d.sys_label, d.passes_label]
                    .into_iter()
                    .flatten()
                    .collect()
            })
            .unwrap_or_default();
        // Capture the StatHud chip ids (fps, gpu wait, vram, ram, ev, edr strip order) so
        // the frame step can pack them tight from the top-left. Like DebugHud
        // the component is queried (not drained), so it is still present here.
        self.stat_hud_chips = ctx
            .query::<StatHud>()
            .next()
            .map(|s| {
                [
                    s.fps_label,
                    s.gpu_wait_label,
                    s.vram_label,
                    s.ram_label,
                    s.ev_label,
                    s.edr_label,
                ]
                .into_iter()
                .flatten()
                .collect()
            })
            .unwrap_or_default();
        if let Some(m) = user_graphics.window_mode {
            settings.window_args.mode = m;
        }
        // The chosen fullscreen display mode. Fullscreen-only: it never feeds
        // the windowed size, which stays the world's authored `Window` value.
        if let Some([w, h, hz]) = user_graphics.resolution {
            settings.resolution = Some(display_mode::DisplayMode {
                width: w,
                height: h,
                refresh_hz: hz,
            });
        }

        if let Some(c) = ctx.drain::<GraphicsConfig>().into_iter().next() {
            let args = c;
            settings.frames_in_flight = args.frames_in_flight as usize;
            settings.vsync = args.vsync;
            settings.fps_cap = args.fps_cap;
            self.clear_color = args.clear_color;
            self.max_frames = args.max_frames;
            settings.shadow_map_size = args.shadow_map_size;
            settings.shadow_update = args.shadow_update;
            settings.shadow_distance = args.shadow_distance;
            settings.shadow_cascades = args.shadow_cascades;
            settings.anisotropy = args.anisotropy;
        }
        // A persisted vsync choice overrides the world's value. Applied outside
        // the GraphicsConfig block (unconditional), matching window_mode /
        // resolution, so it wins over both the authored value and the default.
        if let Some(v) = user_graphics.vsync {
            settings.vsync = v;
        }
        // A persisted frame-rate cap overrides the world's value (0 = unlimited),
        // applied live by the render-step pacer. Independent of the quality preset,
        // like vsync, so no ceiling clamp.
        if let Some(v) = user_graphics.fps_cap {
            settings.fps_cap = v;
        }
        // Stats-HUD display toggles (None = shown, the default, so an existing
        // settings file keeps the FPS / VRAM chips visible). Independent of the
        // quality preset, like vsync / fps_cap.
        if let Some(v) = user_graphics.perf_stats {
            settings.perf_stats = v;
        }
        if let Some(v) = user_graphics.show_fps {
            settings.show_fps = v;
        }
        if let Some(v) = user_graphics.show_vram {
            settings.show_vram = v;
        }

        // Shadow quality knobs (GraphicsConfig-sourced). Snapshot the world's
        // authored values as the baseline a live preset change re-clamps from,
        // then apply any persisted override and otherwise clamp under the preset
        // ceiling (an explicit override wins, like the quality toggles below). The
        // resolution is restart-required -- the shadow map array is sized from
        // `settings.shadow_map_size` at backend init below -- while the cadence is read
        // by the cascade scheduler each frame.
        use crate::gfx::render_config as resolve;
        settings.authored_shadow_map_size = settings.shadow_map_size;
        settings.authored_shadow_update = settings.shadow_update;
        settings.shadow_map_size =
            resolve::shadow_map_size(settings.shadow_map_size, &user_graphics, &quality_ceiling);
        settings.shadow_update =
            resolve::shadow_update(settings.shadow_update, &user_graphics, &quality_ceiling);
        // Shadow distance (GraphicsConfig-sourced, live -- the per-frame cascade
        // split reads it). Same baseline / override / ceiling-clamp shape as the
        // shadow knobs above.
        settings.authored_shadow_distance = settings.shadow_distance;
        settings.shadow_distance =
            resolve::shadow_distance(settings.shadow_distance, &user_graphics, &quality_ceiling);
        // Shadow cascade count (GraphicsConfig-sourced, live -- the per-frame split
        // + schedule read it). Same baseline / override / ceiling-clamp shape.
        settings.authored_shadow_cascades = settings.shadow_cascades;
        settings.shadow_cascades =
            resolve::shadow_cascades(settings.shadow_cascades, &user_graphics, &quality_ceiling);
        // Anisotropy (GraphicsConfig-sourced, restart-required -- the scene sampler
        // is built from `settings.anisotropy` at backend init below). Same baseline /
        // override / ceiling-clamp shape as the shadow knobs above.
        settings.authored_anisotropy = settings.anisotropy;
        settings.anisotropy =
            resolve::anisotropy(settings.anisotropy, &user_graphics, &quality_ceiling);
        // Frames-in-flight (ring-buffer depth): a persisted override clamped to the
        // 1..3 the backends support, applied unconditionally like vsync. Restart-
        // required (the ring buffers are sized at backend init below), independent
        // of the quality preset.
        if let Some(v) = user_graphics.frames_in_flight {
            settings.frames_in_flight = (v as usize).clamp(1, 3);
        }

        // Resolve post-process tunables. The first declared PostProcessConfig
        // wins; with none declared the renderer uses the stack defaults. The
        // AA mode resolves into a TAA gate (threaded alongside the params) and
        // the composite `fxaa` flag inside `post_process` (refreshed below once
        // the override + ceiling clamp have settled the final mode).
        let post_config = ctx.drain::<PostProcessConfig>().into_iter().next();
        // Persisted slider choices override the world's values, re-applied here
        // each launch so they survive a restart, each through its `SLIDERS`
        // entry's `apply`, the same transform the live drag uses.
        let mut post_process = resolve::post_process_params(post_config.as_ref(), &user_graphics);
        // Keep a copy as the live source of truth for the slider settings to
        // read at init and mutate at runtime (PostProcessParams is Copy, so the
        // value is still passed into the backend below).
        settings.post_process = post_process;
        // Ambient (IBL) scale: the world's `PostProcessConfig.ambient_intensity`
        // overridden by any persisted choice. It rides `LightUniforms`, not
        // `PostProcessParams`, so it is held here and pushed to the backend once
        // after it is built (the world value is already seeded at backend init,
        // so this only matters for a persisted override). Clamped like
        // `PostProcessConfig::ambient_intensity`.
        // The raw world value (no override) is what the static LightUniforms are
        // built with; the override rides `set_ambient_intensity` after init.
        let world_ambient = post_config
            .as_ref()
            .map(|c| c.ambient_intensity())
            .unwrap_or(1.0);
        settings.ambient_intensity =
            resolve::ambient_intensity(post_config.as_ref(), &user_graphics);
        // Quality-feature toggles: the world's config overlaid with the user's
        // persisted choices, stored as the source of truth for the Quality-group
        // rows. A runtime toggle flips a field here, re-derives the per-feature
        // settings, and rebuilds the affected GPU resources. A world that
        // declares no config falls back to the schema defaults, which author the
        // top-tier look, so the overrides + ceiling below apply either way: the
        // preset is what settles a default world's quality.
        settings.post_config = post_config.clone().unwrap_or_default();
        // The pristine world baseline, before the user overrides + preset ceiling
        // below. A live preset change re-clamps the quality toggles from this, so
        // raising a preset restores the world's features (a ceiling never enables
        // anything the world did not author, so re-clamping the baseline is exact).
        settings.authored_post_config = settings.post_config.clone();
        resolve::overlay_quality_overrides(&mut settings.post_config, &user_graphics);
        // The active quality preset as a performance ceiling over the toggles
        // above: where the ceiling disallows a feature, force it off -- but only
        // for a toggle the user did not explicitly override, and never turning a
        // feature on. A no-op under Custom (the ceiling permits everything).
        resolve::clamp_quality_under_ceiling(
            &mut settings.post_config,
            &user_graphics,
            &quality_ceiling,
        );
        // Per-feature settings, derived from the overlaid config. Each is the
        // init-time gate the backend builds against; the same derivation feeds a
        // live rebuild (`derive_quality_settings`). RT reflections need an
        // RT-capable GPU, falling back to SSR where ray tracing is unavailable.
        // RT takes precedence over SSR where both are on (the graph builder picks
        // `RtReflections`), reusing the same SSR pre-pass G-buffer + resolve
        // target.
        let taa_enabled = settings.post_config.aa_mode.taa_enabled();
        // The composite FXAA flag follows the final (overridden + ceiling-clamped)
        // AA mode. resolve() seeded `post_process.fxaa` from the authored mode
        // before the override/clamp above, so refresh both the local copy passed
        // to the backend ctor and the live `settings.post_process` here.
        post_process.fxaa = settings.post_config.aa_mode.fxaa_flag();
        settings.post_process.fxaa = post_process.fxaa;
        let ssao_settings = SsaoSettings::from_config(&settings.post_config);
        let ssr_settings = SsrSettings::from_config(&settings.post_config);
        let rt_reflection_settings = RtReflectionSettings::from_config(&settings.post_config);
        let reflection_blur_scale = settings.post_config.reflection_blur_divisor();
        let ssgi_settings = SsgiSettings::from_config(&settings.post_config);
        // The authored `exposure_ev` becomes an additive bias on the adapted EV
        // when auto-exposure is on; otherwise the static path bakes it into
        // `post_process.exposure` (resolve()) and the bias here is unused.
        let auto_exposure_settings = settings.post_config.auto_exposure_settings();
        let auto_exposure_bias_ev = settings.post_config.exposure_ev;
        // Display-output / upscaling preferences: the world's value overridden by
        // any persisted settings-menu choice. Restart-required (the swapchain
        // format + render targets are sized once at init), so they are read here,
        // passed to the backend ctor below, and held on the settings state for the rows
        // to display + cycle. Independent of the quality preset (a user choice,
        // not a tier), so they never clamp under the ceiling or flip it to Custom.
        // HDR display output is additionally gated on the platform advertising an
        // HDR-capable surface (else it warns and falls back to the SDR composite).
        settings.hdr_display = user_graphics
            .hdr_display
            .unwrap_or_else(|| post_config.as_ref().map(|c| c.hdr_display).unwrap_or(false));
        settings.hdr_pq = user_graphics
            .hdr_pq
            .unwrap_or_else(|| post_config.as_ref().map(|c| c.hdr_pq).unwrap_or(false));
        settings.temporal_upscaling = user_graphics.temporal_upscaling.unwrap_or_else(|| {
            post_config
                .as_ref()
                .map(|c| c.temporal_upscaling)
                .unwrap_or(false)
        });
        let hdr_display = settings.hdr_display;
        let hdr_pq = settings.hdr_pq;
        let temporal_upscaling = settings.temporal_upscaling;
        // Two-pass Hi-Z occlusion + texture-streaming quality: also restart-class
        // and independent of the preset, resolved here (before the value-label sync
        // below) from the world's config overridden by any persisted choice.
        // `occlusion_two_pass` is gated on the bindless GPU-cull path being active
        // (the cull pipeline must exist). The texture pool size + per-frame upload
        // budget come from the StreamingConfig, drained here so the override lands
        // before the streamer is built later; the pool only bites where the world
        // declares streaming.
        settings.occlusion_two_pass = user_graphics.occlusion_two_pass.unwrap_or_else(|| {
            post_config
                .as_ref()
                .map(|c| c.occlusion_two_pass)
                .unwrap_or(false)
        });
        let occlusion_two_pass = settings.occlusion_two_pass;
        let mut streaming_config = ctx.drain::<StreamingConfig>().into_iter().next();
        if let Some(sc) = streaming_config.as_mut() {
            if let Some(v) = user_graphics.texture_cap {
                sc.texture_cap = v;
            }
            if let Some(v) = user_graphics.texture_budget {
                sc.texture_budget = v;
            }
        }
        settings.texture_cap = streaming_config
            .as_ref()
            .map(|c| c.texture_cap)
            .unwrap_or(96);
        settings.texture_budget = streaming_config
            .as_ref()
            .map(|c| c.texture_budget)
            .unwrap_or(4);
        // Render-scale (upscaling quality): the world's choice overridden by any
        // persisted settings-menu choice. Restart-required -- the upscaler and
        // render targets are sized from this once, here. `settings.render_scale` is
        // kept for the settings row to display and cycle.
        let world_quality = post_config
            .as_ref()
            .map(|c| c.upscale_quality)
            .unwrap_or_default();
        // A persisted render-scale choice wins; otherwise the world's choice,
        // clamped under the preset ceiling (the more aggressive of the two, so a
        // weak-tier ceiling forces more upscaling but never less).
        settings.render_scale = match user_graphics.render_scale {
            Some(v) => v,
            None => crate::gfx::quality_preset::more_aggressive_upscale(
                world_quality,
                quality_ceiling.min_upscale,
            ),
        };
        let upscale_scale = if post_config.is_some() {
            settings.render_scale.scale()
        } else {
            1.0
        };
        // Upscaler backend (Auto / FSR3 / DLSS / XeSS): the persisted choice wins,
        // else the world's value. Restart-required (the upscaler is selected +
        // built once at init); independent of the quality preset, so no ceiling
        // clamp. Resolved here (ahead of the value-label sync) so the settings row
        // shows the live value. DirectX / Vulkan honor it; Metal uses MetalFX.
        settings.upscale_backend = user_graphics.upscale_backend.unwrap_or_else(|| {
            post_config
                .as_ref()
                .map(|c| c.upscale_backend)
                .unwrap_or_default()
        });

        // Set each settings value label to its live value before the first
        // render, so a persisted/authored choice shows instead of the build's
        // placeholder. HitRegions are still present here: GraphicsSystem.init
        // runs before UiInputSystem.init, which drains them.
        // Audio / controls value labels read from the persisted settings store
        // (with the baseline default when unset); their owning systems apply the
        // value at their own init.
        let user_settings = self.persisted_settings();
        let volume_of = |stored: Option<f32>| stored.unwrap_or(crate::settings::DEFAULT_VOLUME);
        let master_volume = volume_of(user_settings.audio.master_volume);
        let music_volume = volume_of(user_settings.audio.music_volume);
        let sfx_volume = volume_of(user_settings.audio.sfx_volume);
        let voice_volume = volume_of(user_settings.audio.voice_volume);
        // Movement key map: a persisted rebind set overrides the engine default.
        // Pushed to the backend after it is built (below) and used to sync the
        // Controls-tab rebind row labels (`init_rebind_rows`).
        settings.keymap = user_settings.controls.keymap.unwrap_or_default();
        // Gamepad button map: same override rule; InputSystem loads its own
        // copy at its init, so this one only drives the rebind row labels and
        // the SettingsSystem drain.
        settings.gamepad_map = user_settings.controls.gamepad_map.unwrap_or_default();
        sync_setting_value_labels(ctx, |key| match key {
            "vsync" => Some(settings.vsync as usize),
            "fps_cap" => Some(crate::settings::fps_cap_index(settings.fps_cap)),
            "window_mode" => Some(crate::settings::window_mode_index(
                settings.window_args.mode,
            )),
            // "resolution" is a dynamic dropdown; its label is set from the
            // enumerated mode list after the backend is built.
            "render_scale" => Some(crate::settings::render_scale_index(settings.render_scale)),
            "upscale_backend" => Some(crate::settings::upscale_backend_index(
                settings.upscale_backend,
            )),
            "master_volume" => Some(crate::settings::volume_index(master_volume)),
            "music_volume" => Some(crate::settings::volume_index(music_volume)),
            "sfx_volume" => Some(crate::settings::volume_index(sfx_volume)),
            "voice_volume" => Some(crate::settings::volume_index(voice_volume)),
            // Display-output / upscaling toggles (Off/On).
            "temporal_upscaling" => Some(settings.temporal_upscaling as usize),
            "hdr_display" => Some(settings.hdr_display as usize),
            "hdr_pq" => Some(settings.hdr_pq as usize),
            // Stats-HUD display toggles (Off/On).
            "perf_stats" => Some(settings.perf_stats as usize),
            "show_fps" => Some(settings.show_fps as usize),
            "show_vram" => Some(settings.show_vram as usize),
            // Shadow quality knobs (resolution restart-required, cadence live).
            "shadow_map_size" => Some(crate::settings::shadow_resolution_index(
                settings.shadow_map_size,
            )),
            "shadow_update" => Some(crate::settings::shadow_update_index(settings.shadow_update)),
            "shadow_distance" => Some(crate::settings::shadow_distance_index(
                settings.shadow_distance,
            )),
            "shadow_cascades" => Some(crate::settings::shadow_cascades_index(
                settings.shadow_cascades,
            )),
            "anisotropy" => Some(crate::settings::anisotropy_index(settings.anisotropy)),
            // System / streaming restart rows.
            "frames_in_flight" => Some(crate::settings::frames_in_flight_index(
                settings.frames_in_flight as u32,
            )),
            "occlusion_two_pass" => Some(settings.occlusion_two_pass as usize),
            "texture_quality" => Some(crate::settings::texture_quality_index(settings.texture_cap)),
            // mouse_sensitivity is a slider now, synced by `init_sliders`.
            // Quality toggles: index 0 = Off, 1 = On, matching OFF_ON_OPTIONS.
            key if crate::settings::is_quality_toggle(key) => {
                super::quality_toggle_on(&settings.post_config, key).map(|on| on as usize)
            }
            // SSGI gather sub-quality dropdowns.
            key if super::is_quality_cycle(key) => {
                super::quality_cycle_index(&settings.post_config, key)
            }
            _ => None,
        });
        // The master "Graphics Quality" row carries the resolved tier under Auto
        // (e.g. "Auto (High)"), which the static option table cannot express, so
        // it is set directly after the generic sync above writes the bare name.
        let preset_label =
            crate::gfx::quality_preset::preset_label(active_preset, &settings.gpu_profile);
        set_setting_row_label(ctx, "graphics_quality", &preset_label);
        // Capture the slider rows and sync each handle + value label to its live
        // value (e.g. the persisted/authored exposure). Like the cycle-row sync
        // above, this runs before UiInputSystem drains the HitRegions.
        settings.init_sliders(ctx, &user_settings);
        // Capture the rebind rows and sync each value label to the live bound
        // key (persisted or default). Like the slider sync, before UiInputSystem
        // drains the HitRegions.
        settings.init_rebind_rows(ctx);
        // Capture each cycle row's value-label id, so a preset change can relabel
        // its dependent rows (and a quality-row change the master row) at runtime,
        // when the HitRegions are gone. Also before UiInputSystem drains them.
        settings.init_cycle_value_labels(ctx);
        // Capture the show_fps / show_vram row labels and apply the initial
        // gray-out from the resolved "Display performance stats" master toggle.
        // Before UiInputSystem drains the HitRegions / ScrollPanels.
        settings.capture_perf_sub_rows(ctx);
        // Capture the Resolution row's labels and apply the initial gray-out
        // from the resolved window mode (the row only applies in fullscreen).
        settings.capture_resolution_row(ctx);
        // Capture each ScrollPanel's per-element clip bands for the draw path,
        // before UiInputSystem drains the panels (init order: graphics first).
        self.init_clip_rects(ctx);
        // Upscaler backend selector, resolved above (persisted choice over the
        // world's `PostProcessConfig.upscale_backend`). Honored by the DirectX and
        // Vulkan backends (FSR3 / DLSS / XeSS); Metal always uses MetalFX, so it
        // ignores the selector.
        let upscale_backend = settings.upscale_backend;

        // Off-screen HDR sample count, resolved from the ceiling-clamped AA
        // mode and the resolved upscaling preference. Restart-class like
        // `temporal_upscaling`, its other input: the main-pass pipelines, the
        // render targets and the planar / probe faces all bake the count, so a
        // live AA toggle keeps the count this launch resolved and the next
        // launch picks up the change.
        let hdr_samples = hdr_sample_count(settings.post_config.aa_mode, temporal_upscaling);

        let post = backend_init::PostSettings {
            post_process,
            taa_enabled,
            hdr_samples,
            ssao: ssao_settings,
            ssr: ssr_settings,
            ssgi: ssgi_settings,
            rt_reflections: rt_reflection_settings,
            rt_dynamic: launch.resolve_rt_dynamic(),
            rt_skinned_geometry: launch.resolve_rt_skinned_geometry(),
            reflection_blur_scale,
            auto_exposure: auto_exposure_settings,
            auto_exposure_bias_ev,
            hdr_display,
            hdr_pq,
            temporal_upscaling,
            upscale_scale,
            upscale_backend,
            occlusion_two_pass,
        };
        (
            ResolvedRenderConfig {
                post,
                quality_ceiling,
                streaming_config,
                world_ambient_intensity: world_ambient,
            },
            settings,
        )
    }

    // Decode every SkinnedMesh resource-table entry's geometry payload (before
    // the shared blob is released) into a handle-ordered table, and publish the
    // name -> handle index + skin-selector list for the animation systems.
    // Returns the decoded geometry and the blob indices its payloads occupy (for
    // the release step), or None if any entry's baked data or payload is missing
    // or malformed, which fails init.
    fn decode_skinned_geometry(
        &self,
        ctx: &mut PipelineContext,
    ) -> Option<(Vec<SkinnedGeometry>, Vec<u32>)> {
        // Load the SkinnedMesh resource table and decode each entry's geometry
        // payload now, before the shared blob is released. The placement,
        // material references, capsule, and spawn reserve travel in the baked
        // `data_bytes`; the vertex/index geometry + skeleton in the compiled
        // payload. The table index IS the mesh's `SkinnedMeshHandle`, which keys
        // the whole animation correlation web.
        let skinned_table = ctx
            .resource::<SkinnedMeshTable>()
            .cloned()
            .unwrap_or_default();
        let mut skinned_geometry: Vec<SkinnedGeometry> = Vec::new();
        let mut skinned_blob_indices: Vec<u32> = Vec::new();
        // Interned name -> handle, published for the animation debug tool
        // calls, which address a mesh by its typed name.
        let mut skinned_name_index: std::collections::HashMap<AssetId, SkinnedMeshHandle> =
            std::collections::HashMap::new();
        for (handle, entry) in skinned_table.0.iter().enumerate() {
            let handle = SkinnedMeshHandle(handle as u32);
            let (name_id, sm): (u32, SkinnedMesh) = match postcard::from_bytes(&entry.data_bytes) {
                Ok(t) => t,
                Err(e) => {
                    tracing::error!(
                        "GraphicsSystem: SkinnedMesh handle {} baked data failed to decode: {}",
                        handle.index(),
                        e
                    );
                    return None;
                }
            };
            let name_id = AssetId(name_id);
            skinned_name_index.insert(name_id, handle);
            let locator = match &entry.payload {
                Some(l) => l.clone(),
                None => {
                    tracing::error!(
                        "GraphicsSystem: SkinnedMesh handle {} has no compiled payload",
                        handle.index()
                    );
                    return None;
                }
            };
            skinned_blob_indices.push(locator.blob_index);
            let bytes = match ctx.read_payload(&locator) {
                Ok(b) => b.to_vec(),
                Err(e) => {
                    tracing::error!(
                        "GraphicsSystem: failed to read SkinnedMesh handle {} payload: {:?}",
                        handle.index(),
                        e
                    );
                    return None;
                }
            };
            match mesh_payload::deserialize_skinned_with_lods(&bytes) {
                Ok(p) => {
                    let joint_defs = payload_joints_to_defs(p.joints);
                    skinned_geometry.push(SkinnedGeometry {
                        handle,
                        name_id,
                        mesh: sm,
                        vertices: p.vertices,
                        indices: p.indices,
                        joint_defs,
                        morphs: p.morphs,
                        lod_alternates: p.lods,
                    });
                }
                Err(e) => {
                    tracing::error!("GraphicsSystem: malformed SkinnedMesh payload: {}", e);
                    return None;
                }
            }
        }
        // Publish the name index before AnimationSystem inits (it runs after
        // GraphicsSystem) so animation debug tool calls can resolve a typed
        // mesh name to the handle keying the correlation web. The skin
        // selectors ride along for the animation reload catalog.
        ctx.insert_resource(crate::gfx::skinned_mesh_map::SkinnedMeshNameIndex(
            skinned_name_index,
        ));
        ctx.insert_resource(crate::gfx::skinned_mesh_map::SkinnedMeshSkinIndex(
            skinned_geometry.iter().map(|g| g.mesh.skin_index).collect(),
        ));
        ctx.insert_resource(crate::gfx::shape_preview::SkinnedMeshMorphNames(
            skinned_geometry
                .iter()
                .map(|g| g.morphs.names.clone())
                .collect(),
        ));
        Some((skinned_geometry, skinned_blob_indices))
    }

    // Build skinned draw objects, the shared skinned vertex/index buffers, and
    // bind-pose skeletons from the decoded SkinnedMesh geometry. Runs after the
    // material map so SkinnedMesh material references resolve. Each mesh also
    // pre-reserves `max_instances` hidden bind-pose copies for runtime spawns.
    // Returns None, failing init, if a mesh references an unknown material.
    fn assemble_skinned_meshes(
        &self,
        skinned_geometry: &[SkinnedGeometry],
        material_map: &std::collections::HashMap<MaterialHandle, MaterialEntry>,
        capture_sources: bool,
    ) -> Option<SkinnedMeshAssembly> {
        let mut skinned_vertices: Vec<mesh_payload::SkinnedVertex> = Vec::new();
        let mut skinned_indices: Vec<u32> = Vec::new();
        let mut skinned_draw_objects: Vec<render_types::SkinnedDrawObject> = Vec::new();
        // One entry per authored skinned mesh: its handle, interned name id,
        // the skinned index of its (visible) template draw object, and its
        let mut skinned_skeletons: Vec<SkinnedSkeletonEntry> = Vec::new();
        // `(template_index, instance_index)` pairs seeding the backend skinned
        // instance pool: each instance is a hidden bind-pose copy reserved from
        // SkinnedMesh.max_instances.
        let mut skinned_pool_reservations: Vec<(usize, usize)> = Vec::new();
        // Morph-target data per skinned draw object; instance copies share
        // their template's data through the Arc.
        let mut skinned_morphs: Vec<Option<std::sync::Arc<mesh_payload::PayloadMorphs>>> =
            Vec::new();
        // Asset hot-reload (`cn debug` only) needs the per-slot vertex region
        // + joint count so it can reject size + shape changes before pushing
        // to the backend. SkinnedMesh is 1:1 with its draw slot (no Prop
        // fan-out), so one entry per asset.
        let mut skinned_mesh_source_map = super::hot_reload_sources::SkinnedMeshSourceMap::new();
        for SkinnedGeometry {
            handle,
            name_id,
            mesh: sm,
            vertices: verts,
            indices: idxs,
            joint_defs,
            morphs,
            lod_alternates: lod_alts,
        } in skinned_geometry
        {
            let mat_entry =
                match crate::gfx::material_entry::resolve_material_slots(sm.material, material_map)
                {
                    Ok(entry) => entry,
                    Err(mat_id) => {
                        tracing::error!(
                            "GraphicsSystem: SkinnedMesh '{}' references unknown material {}",
                            name_id,
                            mat_id.index()
                        );
                        return None;
                    }
                };
            let (texture_slot, normal_map_slot, material) = (
                mat_entry.albedo_slot,
                mat_entry.normal_map_slot,
                mat_entry.uniforms,
            );

            let base = skinned_vertices.len() as u32;
            let index_offset = skinned_indices.len();
            skinned_vertices.extend_from_slice(verts);
            skinned_indices.extend(idxs.iter().map(|i| u32::from(*i) + base));

            // LOD alternates share this slot's vertex region. The runtime
            // skinned IB is u16, so each alternate's mesh-relative indices
            // are rebased onto the same `base` as LOD0, identical to how
            // the shadow / velocity / SSAO / SSR pre-passes already consume
            // the IB.
            let lod_slices =
                crate::gfx::draw_list::append_lod_slices(&mut skinned_indices, lod_alts, base);

            let skeleton = build_skeleton_from_joint_defs(joint_defs);
            let joint_count = skeleton.len().min(render_types::MAX_JOINTS);

            // Bind-pose (object-space) AABB over this mesh's vertices. The
            // GPU-driven skinned fold pads + transforms it per frame for culling.
            let (local_bb_min, local_bb_max) = if verts.is_empty() {
                ([0.0; 3], [0.0; 3])
            } else {
                let mut lo = [f32::INFINITY; 3];
                let mut hi = [f32::NEG_INFINITY; 3];
                for v in verts.iter() {
                    for a in 0..3 {
                        lo[a] = lo[a].min(v.pos[a]);
                        hi[a] = hi[a].max(v.pos[a]);
                    }
                }
                (lo, hi)
            };

            let mesh_morphs = (!morphs.is_empty()).then(|| std::sync::Arc::new(morphs.clone()));
            let skinned_index = skinned_draw_objects.len();
            skinned_morphs.push(mesh_morphs.clone());
            skinned_draw_objects.push(render_types::SkinnedDrawObject {
                vertex_base: base,
                vertex_count: verts.len(),
                index_offset,
                index_count: idxs.len(),
                model: sm.model_matrix(),
                texture_slot,
                normal_map_slot,
                material,
                visible: true,
                joint_count,
                local_bb_min,
                local_bb_max,
                lod_alternates: lod_slices,
            });
            if capture_sources && !sm.source.is_empty() {
                skinned_mesh_source_map.entries.push(
                    super::hot_reload_sources::SkinnedMeshSourceEntry {
                        source: sm.source.clone(),
                        skin_index: sm.skin_index,
                        skinned_index,
                        vertex_base: base,
                        vertex_count: verts.len(),
                        index_count: idxs.len(),
                        joint_count,
                    },
                );
            }
            // Pre-reserve runtime spawn copies: append `max_instances` hidden
            // bind-pose duplicates of this mesh, each with its OWN vertex region
            // in the shared skinned buffer. They must not share a region because
            // the GPU skin fold writes the deformed buffer keyed by global vertex
            // index, so two live instances at one region would clobber each
            // other's pose. A runtime skinned spawn reveals one of these without
            // growing any GPU buffer; a despawn returns it to the pool.
            for _ in 0..sm.max_instances {
                let copy_base = skinned_vertices.len() as u32;
                let copy_index_offset = skinned_indices.len();
                skinned_vertices.extend_from_slice(verts);
                skinned_indices.extend(idxs.iter().map(|i| u32::from(*i) + copy_base));
                let copy_lods = crate::gfx::draw_list::append_lod_slices(
                    &mut skinned_indices,
                    lod_alts,
                    copy_base,
                );
                let copy_skinned_index = skinned_draw_objects.len();
                skinned_morphs.push(mesh_morphs.clone());
                skinned_draw_objects.push(render_types::SkinnedDrawObject {
                    vertex_base: copy_base,
                    vertex_count: verts.len(),
                    index_offset: copy_index_offset,
                    index_count: idxs.len(),
                    model: sm.model_matrix(),
                    texture_slot,
                    normal_map_slot,
                    material,
                    // Hidden until a runtime spawn claims it.
                    visible: false,
                    joint_count,
                    local_bb_min,
                    local_bb_max,
                    lod_alternates: copy_lods,
                });
                skinned_pool_reservations.push((skinned_index, copy_skinned_index));
            }

            skinned_skeletons.push(SkinnedSkeletonEntry {
                handle: *handle,
                name_id: *name_id,
                template_index: skinned_index,
                skeleton,
                morph_names: morphs.names.clone(),
                model: sm.model_matrix(),
                capsule: sm.capsule.clone(),
                transform: Transform {
                    position: sm.position,
                    rotation_deg: sm.rotation_deg,
                    scale: sm.scale,
                },
                local_bounds: (local_bb_min, local_bb_max),
            });
        }

        Some(SkinnedMeshAssembly {
            vertices: skinned_vertices,
            indices: skinned_indices,
            draw_objects: skinned_draw_objects,
            skeletons: skinned_skeletons,
            pool_reservations: skinned_pool_reservations,
            morphs: skinned_morphs,
            source_map: skinned_mesh_source_map,
        })
    }

    // Read the shared TextureTable, collecting each texture's payload locator
    // (dense by pool slot / cook `TextureHandle`). Under `cn debug`
    // (`capture_sources`) also records the file-backed source paths + the
    // name -> slot map for the hot-reload watcher and the runtime spawn-by-name
    // path; the shipped runtime resolves every texture by handle and needs
    // neither. Returns None, failing init, if a texture lacks a payload.
    fn decode_texture_table(
        &self,
        ctx: &mut PipelineContext,
        capture_sources: bool,
    ) -> Option<TextureTableDecode> {
        // The shared texture pool comes from the blob's resource stream: cook
        // assigned each texture a dense `TextureHandle` (== its pool slot) and the
        // runtime loaded them into a `TextureTable`. Reading the table by handle
        // replaces draining a `Texture` component column and scanning names.
        let texture_table = ctx.resource::<TextureTable>().cloned().unwrap_or_default();
        // Dev-only source catalog (present under `cn debug`) so the hot-reload
        // watcher can map a texture handle back to the file that backs it.
        let texture_sources = ctx.resource::<crate::resource::TextureSources>().cloned();
        let mut texture_locators = Vec::with_capacity(texture_table.len());
        let mut asset_source_map = super::hot_reload_sources::TextureSourceMap::new();
        // Name -> pool slot, built only under `cn debug` for the runtime
        // spawn-by-name path (`TextureNameSlots`).
        let mut texture_name_to_slot: std::collections::HashMap<AssetId, usize> =
            std::collections::HashMap::new();
        for (slot, entry) in texture_table.0.iter().enumerate() {
            match &entry.payload {
                Some(l) => {
                    texture_locators.push(l.clone());
                    if capture_sources
                        && let Some(info) = texture_sources.as_ref().and_then(|s| s.0.get(slot))
                    {
                        texture_name_to_slot.insert(AssetId(info.name_id), slot);
                        if !info.source.is_empty() {
                            asset_source_map.push_texture(
                                info.source.clone(),
                                info.image_index,
                                slot,
                            );
                        }
                    }
                }
                None => {
                    tracing::error!(
                        "GraphicsSystem: Texture has no compiled payload -- did the build succeed?"
                    );
                    return None;
                }
            }
        }
        let count = texture_table.len();
        Some(TextureTableDecode {
            locators: texture_locators,
            source_map: asset_source_map,
            name_to_slot: texture_name_to_slot,
            count,
        })
    }

    // Decode the MaterialTable (dense by `MaterialHandle`) into the per-object GPU
    // uniforms + resolved texture slots the draw list indexes. Materials have no
    // payload; all data lives in the baked `data_bytes`. Returns None,
    // failing init, on any decode or resolution failure.
    fn build_material_map(
        &self,
        ctx: &mut PipelineContext,
        texture_count: usize,
    ) -> Option<std::collections::HashMap<MaterialHandle, MaterialEntry>> {
        let material_table = ctx.resource::<MaterialTable>().cloned().unwrap_or_default();
        let mut material_map: std::collections::HashMap<MaterialHandle, MaterialEntry> =
            std::collections::HashMap::with_capacity(material_table.len());
        for (material_handle, entry) in material_table.0.iter().enumerate() {
            let mat: Material = match postcard::from_bytes(&entry.data_bytes) {
                Ok(m) => m,
                Err(e) => {
                    tracing::error!(
                        "GraphicsSystem: Material handle {} failed to decode: {}",
                        material_handle,
                        e
                    );
                    return None;
                }
            };
            match crate::gfx::material_entry::of(&mat, texture_count) {
                Ok(entry) => {
                    material_map.insert(MaterialHandle(material_handle as u32), entry);
                }
                Err(field) => {
                    tracing::error!(
                        "GraphicsSystem: Material {} references an out-of-range {} texture handle (only {} textures)",
                        material_handle,
                        field,
                        texture_count
                    );
                    return None;
                }
            }
        }
        Some(material_map)
    }

    // Drain the world's Shader components, read every compiled stage
    // container, and split each into the per-stage byte sets the backend's
    // pipeline table consumes. Drain order matches cook's shader handle
    // assignment (both walk the declaration-ordered asset list), so a baked
    // `ShaderHandle` indexes the returned list directly; entry 0 drives the
    // world default pipeline. Under `cn debug` also records the default
    // shader's resolved on-disk stage source paths so the asset hot-reload
    // watcher can recompile + rebuild its pipelines on a shader save. Returns
    // None, failing init, if any payload is missing or unreadable.
    //
    // A world that declares no Shader is the common case: it gets a single
    // bucket carrying no bytes, which every backend reads as "use the engine's
    // own main-pass program".
    fn decode_shaders(
        &mut self,
        ctx: &mut PipelineContext,
        streaming: bool,
        capture_sources: bool,
    ) -> Option<DecodedShaders> {
        let world_shaders = ctx.drain::<Shader>();
        if world_shaders.is_empty() {
            return Some(DecodedShaders {
                locators: Vec::new(),
                shaders: vec![DecodedShader::default()],
                source_map: super::hot_reload_sources::ShaderStageSourceMap::new(),
            });
        }

        // Buckets a non-start scene exclusively owns skip their decode and
        // pipeline build here; the streaming pump warms them when that scene
        // pins. The backend sees them flagged `deferred` and leaves the bucket's
        // pipeline unbuilt.
        let shader_ids: Vec<AssetId> = world_shaders.iter().map(|s| s.asset_id).collect();
        self.deferred_shader_scenes =
            super::streaming::deferred_shader_buckets(ctx, streaming, &shader_ids)
                .into_iter()
                .map(|(bucket, scene)| (bucket as u32, scene))
                .collect();
        let deferred_buckets: std::collections::HashSet<u32> = self
            .deferred_shader_scenes
            .iter()
            .map(|&(bucket, _)| bucket)
            .collect();
        let blob_disk_backed = ctx.blob.disk_backed();
        let mut deferred_sources = Vec::new();

        let mut locators = Vec::with_capacity(world_shaders.len());
        let mut shaders = Vec::with_capacity(world_shaders.len());
        for (bucket, shader) in world_shaders.iter().enumerate() {
            let locator = match &shader.locator {
                Some(l) => l.clone(),
                None => {
                    tracing::error!("GraphicsSystem: Shader has no compiled payload");
                    return None;
                }
            };
            if deferred_buckets.contains(&(bucket as u32)) {
                match deferred_shader_source(ctx, &locator, blob_disk_backed) {
                    Ok(source) => {
                        deferred_sources.push(crate::gfx::streaming::shader::DeferredBucket {
                            bucket: bucket as u32,
                            source,
                        });
                        locators.push(locator);
                        shaders.push(DecodedShader {
                            deferred: true,
                            ..Default::default()
                        });
                        continue;
                    }
                    Err(e) => {
                        // Fall through to the eager decode: a bucket that
                        // cannot be deferred still has to render.
                        tracing::warn!(
                            "GraphicsSystem: shader bucket {} cannot be deferred ({}); \
                             building it at init instead",
                            bucket,
                            e
                        );
                        self.deferred_shader_scenes
                            .retain(|&(b, _)| b != bucket as u32);
                    }
                }
            }
            // Read the stage container before the blob is released -- it may
            // share one blob with the mesh/texture payloads read elsewhere in
            // init.
            let payload = match ctx.read_payload(&locator) {
                Ok(b) => match ShaderPrograms::decode(b) {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::error!("GraphicsSystem: shader payload decode: {:?}", e);
                        return None;
                    }
                },
                Err(e) => {
                    tracing::error!("GraphicsSystem: failed to read shader payload: {:?}", e);
                    return None;
                }
            };
            locators.push(locator);
            shaders.push(DecodedShader {
                programs: Some(payload),
                deferred: false,
            });
        }

        if !deferred_sources.is_empty() {
            tracing::info!(
                "GraphicsSystem: deferred {} scene-owned shader pipeline(s) past init",
                deferred_sources.len()
            );
            self.shader_warmup = Some(crate::gfx::streaming::shader::ShaderWarmup::new(
                deferred_sources,
            ));
        }

        // Capture the default Shader's declared files so the asset hot-reload
        // watcher can recompile and rebuild its pipelines on a save.
        // Material-referenced shaders past entry 0 reload via `cn build`.
        let world_default = &world_shaders[0];
        let mut shader_stage_source_map = super::hot_reload_sources::ShaderStageSourceMap::new();
        if capture_sources {
            let assets_dir = self.assets_dir();
            for stage in [ShaderStage::Vertex, ShaderStage::Fragment] {
                let Some(raw) = world_default.stage(stage) else {
                    continue;
                };
                let resolved =
                    concinnity_host::store::source::resolve_source_path(raw, assets_dir.as_deref());
                shader_stage_source_map.entries.push(
                    super::hot_reload_sources::ShaderStageSourceEntry {
                        stage,
                        resolved_path: resolved,
                    },
                );
            }
        }

        Some(DecodedShaders {
            locators,
            source_map: shader_stage_source_map,
            shaders,
        })
    }

    // Apply the persisted window mode, publish the Resolution row's mode list
    // (backend-enumerated, else the static preset fallback), apply a persisted
    // display-mode choice to the backend, and seed the frame-rate-cap resource +
    // the Resolution row's dynamic value label. Runs after the backend is built.
    fn finalize_display_modes(&mut self, ctx: &mut PipelineContext, settings: &mut SettingsState) {
        if let Some(backend) = self.backend.as_deref_mut() {
            // The window is always created as a standard titled window, so a
            // persisted or authored Borderless / Fullscreen mode is applied here.
            // No-op in embedded mode (the backend owns no window there).
            if settings.window_args.mode != WindowMode::Windowed {
                backend.set_window_mode(settings.window_args.mode);
            }
            let raw = backend.display_modes();
            settings.display_modes = if raw.is_empty() {
                display_mode::fallback_modes()
            } else {
                display_mode::normalize(raw)
            };
            settings.current_mode = backend.current_display_mode();
            if let Some(mode) = settings.resolution {
                backend.set_display_mode(mode);
            }
        }
        ctx.insert_resource(crate::ecs::DisplayModes(settings.display_modes.clone()));
        // The resolved frame-rate cap (world value or persisted override) for
        // the App-level pacer; the settings row's live change republishes it.
        ctx.insert_resource(FrameRateCap(settings.fps_cap));
        let idx = display_mode::index_of(&settings.display_modes, settings.effective_resolution());
        if let Some(m) = settings.display_modes.get(idx) {
            set_setting_row_label(ctx, "resolution", &m.label());
        }
    }

    // Decide cursor handling and push the post-build backend config: menu mode,
    // ambient scale, key map, the startup cursor grab (plain first-person worlds
    // only), and the device capability flags that gate the settings rows.
    fn finalize_backend_config(&mut self, ctx: &mut PipelineContext, settings: &SettingsState) {
        // A plain first-person world (Camera3D, no UI) captures the cursor at
        // startup. A Camera3D world that also has UI (a MainMenu's HitRegion /
        // KeyBinding) is "menu mode": capture is driven per-frame in `run_step`.
        // A UI-only world (no camera) stays free-cursor.
        let has_ui =
            ctx.query::<HitRegion>().next().is_some() || ctx.query::<KeyBinding>().next().is_some();
        let has_camera = ctx.query::<Camera3D>().next().is_some();
        self.menu_mode = has_camera && has_ui;
        // A menu / editor driver (a `MenuOverride` is present) owns cursor capture
        // per frame, so the startup auto-grab is skipped: the editor re-runs this
        // init on every live-preview rebuild, and grabbing there would re-hide and
        // decouple the OS cursor each time, desyncing the free-cursor handoff.
        let menu_driven = ctx.resource::<MenuOverride>().is_some();
        let mut device_caps = backend::DeviceCapabilities::ALL;
        if let Some(backend) = self.backend.as_deref_mut() {
            // Capability flags drive the settings-menu gating below.
            device_caps = backend.capabilities();
            // Detected GPU performance profile, logged once at init so the
            // classified tier is verifiable on each device.
            let gpu = backend.gpu_profile();
            tracing::info!(
                "GPU profile: vendor={:?} tier={:?} memory_budget={} MB unified={} discrete={}",
                gpu.vendor,
                gpu.tier,
                gpu.memory_budget_bytes / (1 << 20),
                gpu.unified_memory,
                gpu.discrete,
            );
            backend.set_menu_mode(self.menu_mode);
            // Push the effective ambient scale (world value or persisted
            // override). The backend already seeds the world value at its own
            // init, so this is the path that applies a persisted Ambient-slider
            // choice; idempotent when there is no override.
            backend.set_ambient_intensity(settings.ambient_intensity);
            // Push the movement key map (the persisted rebinds, or the default).
            // The backend decodes physical keys through it; idempotent with its
            // own default seed when there is no override.
            backend.set_keymap(&settings.keymap);
            if has_camera && !has_ui && !menu_driven {
                backend.request_cursor_capture();
            }
        }
        self.caps = device_caps;
        // Publish the flags for the systems that cannot reach the backend
        // themselves (the editor's live draw seam asks whether a rewritten draw
        // slot would land).
        ctx.insert_resource(crate::ecs::ActiveDeviceCaps(device_caps));
        // Gray out + disable settings rows whose feature the device cannot
        // provide (e.g. ray-traced reflections on a GPU without hardware ray
        // tracing). Runs while the menu HitRegions / TextLabels / ScrollPanels
        // are still present (GraphicsSystem.init runs before UiInputSystem drains
        // them); the value-label sync above already set each row's live value.
        self.apply_capability_gating(ctx);
    }

    // Read the sole EnvironmentMap (handle 0) from its resource table and capture
    // its IBL payload; extra declarations are logged and ignored. Under `cn debug`
    // (`capture_sources`) also captures the resolved HDR source path + convolution
    // sizing for the hot-reload watcher (procedural generators have no file to
    // watch). Returns (payload bytes, source), or None, failing init, if the
    // payload is unreadable.
    fn decode_environment_map(
        &self,
        ctx: &mut PipelineContext,
        capture_sources: bool,
    ) -> Option<(
        Option<Vec<u8>>,
        Option<super::hot_reload_sources::EnvironmentMapSource>,
    )> {
        let env_map_table = ctx
            .resource::<EnvironmentMapTable>()
            .cloned()
            .unwrap_or_default();
        if env_map_table.len() > 1 {
            tracing::warn!(
                "GraphicsSystem: {} EnvironmentMaps declared; only the first is used",
                env_map_table.len()
            );
        }
        let mut env_map_bytes: Option<Vec<u8>> = None;
        let mut environment_map_source: Option<super::hot_reload_sources::EnvironmentMapSource> =
            None;
        // The runtime uses handle 0. A map installed at runtime holds its
        // payload directly; a compiled one is read through its locator. An
        // entry with neither means simply "no EnvironmentMap declared".
        if let Some(entry) = env_map_table.0.first() {
            match (entry.baked_bytes(), &entry.payload) {
                (Some(baked), _) => env_map_bytes = Some(baked.to_vec()),
                (None, Some(locator)) => match ctx.read_payload(&locator.clone()) {
                    Ok(b) => env_map_bytes = Some(b.to_vec()),
                    Err(e) => {
                        tracing::error!(
                            "GraphicsSystem: failed to read EnvironmentMap payload: {:?}",
                            e
                        );
                        return None;
                    }
                },
                (None, None) => {}
            }
        }
        if capture_sources
            && let Some(info) = ctx
                .resource::<crate::resource::EnvironmentMapSources>()
                .and_then(|s| s.0.clone())
        {
            environment_map_source = Some(super::hot_reload_sources::EnvironmentMapSource {
                resolved_path: concinnity_host::store::source::resolve_source_path(
                    &info.source,
                    self.assets_dir().as_deref(),
                ),
                prefilter_face_size: info.prefilter_face_size,
                irradiance_face_size: info.irradiance_face_size,
                prefilter_samples: info.prefilter_samples,
                prefilter_clamp: info.prefilter_clamp,
            });
        }
        Some((env_map_bytes, environment_map_source))
    }

    // Read the sole ColorLut (handle 0) from its resource table and capture its
    // color-grading payload; extras are logged and ignored. Under `cn debug`
    // captures the resolved source path for the hot-reload watcher. Returns
    // (payload bytes, source), or None, failing init, if unreadable.
    fn decode_color_lut(
        &self,
        ctx: &mut PipelineContext,
        capture_sources: bool,
    ) -> Option<(
        Option<Vec<u8>>,
        Option<super::hot_reload_sources::ColorLutSource>,
    )> {
        let color_lut_table = ctx.resource::<ColorLutTable>().cloned().unwrap_or_default();
        if color_lut_table.len() > 1 {
            tracing::warn!(
                "GraphicsSystem: {} ColorLuts declared; only the first is used",
                color_lut_table.len()
            );
        }
        let mut color_lut_bytes: Option<Vec<u8>> = None;
        let mut color_lut_source: Option<super::hot_reload_sources::ColorLutSource> = None;
        // Handle 0 is the sole LUT the renderer applies; a compiled ColorLut always
        // carries a payload, so a `None` locator means "no ColorLut declared".
        if let Some(locator) = color_lut_table.locator(0) {
            match ctx.read_payload(&locator) {
                Ok(b) => color_lut_bytes = Some(b.to_vec()),
                Err(e) => {
                    tracing::error!("GraphicsSystem: failed to read ColorLut payload: {:?}", e);
                    return None;
                }
            }
        }
        if capture_sources
            && let Some(src) = ctx
                .resource::<crate::resource::ColorLutSources>()
                .and_then(|c| c.0.clone())
        {
            color_lut_source = Some(super::hot_reload_sources::ColorLutSource {
                resolved_path: concinnity_host::store::source::resolve_source_path(
                    &src,
                    self.assets_dir().as_deref(),
                ),
            });
        }
        Some((color_lut_bytes, color_lut_source))
    }

    // Build the shared text/sprite atlas pool: deserialize each Font's atlas +
    // metrics into `self.loaded_fonts` (its FontHandle == its dense atlas slot),
    // add the built-in fallback face when any text names no Font, then append
    // each distinct Sprite / Story-stage texture (resolved through
    // `texture_locators`) into `self.sprite_texture_slots`. An unresolved sprite
    // texture demotes to its tint (warned, not fatal). Returns the RGBA atlases +
    // the font payloads' blob indices, or None, failing init, on a Font decode
    // or read failure.
    fn decode_text_atlases(
        &mut self,
        ctx: &mut PipelineContext,
        texture_locators: &[PayloadLocator],
    ) -> Option<TextAtlases> {
        let font_table = ctx.resource::<FontTable>().cloned().unwrap_or_default();
        let mut text_atlas_data: Vec<(u32, u32, Vec<u8>)> = Vec::new();
        for (slot, entry) in font_table.0.iter().enumerate() {
            // A face the world baked for itself at start holds its payload
            // directly; a compiled one is read through its locator.
            let bytes = match (entry.baked_bytes(), &entry.payload) {
                (Some(baked), _) => baked.to_vec(),
                (None, Some(locator)) => match ctx.read_payload(&locator.clone()) {
                    Ok(b) => b.to_vec(),
                    Err(e) => {
                        tracing::error!(
                            "GraphicsSystem: failed to read Font handle {} payload: {:?}",
                            slot,
                            e
                        );
                        return None;
                    }
                },
                (None, None) => {
                    tracing::error!(
                        "GraphicsSystem: Font handle {} has no compiled payload -- did the build succeed?",
                        slot
                    );
                    return None;
                }
            };
            match font::deserialize(&bytes) {
                Ok((aw, ah, supersample, size_px, rgba, metrics)) => {
                    let metrics_map: text::FontMetrics =
                        metrics.into_iter().map(|m| (m.char_code, m)).collect();
                    let size_px = size_px as f32;
                    self.loaded_fonts.insert(
                        FontHandle(slot as u32),
                        text::LoadedFont {
                            atlas_slot: slot,
                            cap_px: text::derive_cap_px(&metrics_map, size_px),
                            metrics: metrics_map,
                            atlas_w: aw,
                            atlas_h: ah,
                            size_px,
                            supersample: (supersample.max(1)) as f32,
                        },
                    );
                    text_atlas_data.push((aw, ah, rgba));
                }
                Err(e) => {
                    tracing::error!("GraphicsSystem: malformed Font payload: {}", e);
                    return None;
                }
            }
        }

        // Text naming no Font has no compiled face to draw with: nothing on
        // either the cook or the code-assembly path makes one for it. Register
        // the built-in face for it to fall back to, only when some text needs
        // it: the atlas is megabytes a world that names its fonts never
        // samples.
        if font_less_text(ctx) {
            let slot = text_atlas_data.len();
            let handle = FontHandle(slot as u32);
            match crate::gfx::builtin_font::load(handle) {
                Some(builtin) => {
                    text_atlas_data.push(builtin.atlas);
                    self.loaded_fonts.insert(handle, builtin.loaded);
                    self.loaded_fonts.set_fallback(handle);
                }
                None => tracing::error!(
                    "GraphicsSystem: text naming no Font cannot draw -- the built-in face failed to decode"
                ),
            }
        }

        // Sprite textures ride the text-atlas pool: each distinct Texture a
        // Sprite references is decoded and appended after the font atlases,
        // drawn by the same pipeline (positive vertex mode = RGBA quad). A
        // Story's stage images are gathered too: the story system swaps them
        // onto the stage sprites at runtime, so they must be resident even
        // though no sprite references them yet. A texture that cannot be
        // resolved demotes its sprite to the solid tint fill, warned rather
        // than fatal.
        let sprite_texture_ids: Vec<TextureHandle> = {
            let mut ids: Vec<TextureHandle> =
                ctx.query::<Sprite>().filter_map(|s| s.texture).collect();
            for story in ctx.query::<Story>() {
                let stages = story.nodes.iter().flat_map(|n| {
                    n.pages
                        .iter()
                        .map(|p| &p.stage)
                        .chain(std::iter::once(&n.choice_stage))
                });
                for stage in stages {
                    for image in [&stage.bg, &stage.left, &stage.center, &stage.right]
                        .into_iter()
                        .flatten()
                    {
                        ids.push(image.texture);
                    }
                }
            }
            ids.sort_unstable_by_key(|id| id.0);
            ids.dedup();
            ids
        };
        for tex_id in sprite_texture_ids {
            // The texture handle is the texture's declaration-order pool slot,
            // so it indexes the locator table directly.
            let Some(locator) = texture_locators.get(tex_id.index()).cloned() else {
                tracing::warn!(
                    "GraphicsSystem: Sprite references unknown texture {:?}; drawing its tint",
                    tex_id
                );
                continue;
            };
            match ctx.read_payload(&locator) {
                Ok(bytes) => match texture::deserialize(bytes).and_then(|image| image.into_rgba8())
                {
                    Ok((w, h, rgba)) => {
                        self.sprite_texture_slots
                            .insert(tex_id, text_atlas_data.len());
                        text_atlas_data.push((w, h, rgba));
                    }
                    Err(e) => {
                        tracing::warn!("GraphicsSystem: sprite texture {:?}: {}", tex_id, e)
                    }
                },
                Err(e) => tracing::warn!(
                    "GraphicsSystem: sprite texture {:?} payload read failed: {:?}",
                    tex_id,
                    e
                ),
            }
        }

        // Tool-provided overlay images (e.g. asset thumbnails) ride the same
        // pool, keyed by the reserved handles the inserting tool chose.
        if let Some(overlay) = ctx.resource::<OverlayImages>() {
            for image in &overlay.0 {
                if image.rgba.len() != (image.width as usize) * (image.height as usize) * 4 {
                    tracing::warn!(
                        "GraphicsSystem: overlay image {:?} byte length mismatch; skipped",
                        image.handle
                    );
                    continue;
                }
                self.sprite_texture_slots
                    .insert(image.handle, text_atlas_data.len());
                text_atlas_data.push((image.width, image.height, image.rgba.clone()));
            }
        }

        let font_blob_indices: Vec<u32> = font_table.blob_indices().into_iter().collect();
        Some(TextAtlases {
            atlases: text_atlas_data,
            font_blob_indices,
        })
    }

    // Run init, marking the system failed when any step of it fails.
    pub(super) fn run_init(&mut self, ctx: &mut PipelineContext) {
        if self.try_init(ctx).is_none() {
            self.failed = true;
        }
    }

    fn try_init(&mut self, ctx: &mut PipelineContext) -> Option<()> {
        let launch = ctx.resource::<LaunchRequest>().copied().unwrap_or_default();
        let (
            ResolvedRenderConfig {
                post,
                quality_ceiling,
                streaming_config,
                world_ambient_intensity,
            },
            mut settings,
        ) = self.init_render_settings(ctx, &launch);
        // Infinite-world chunk streaming. The first declared VoxelWorld wins;
        // with none declared, no chunks stream. BlockTypes are drained here so
        // the runtime can resolve the VoxelWorld palette to chunk-mesh data.
        let voxel_world = ctx.drain::<VoxelWorld>().into_iter().next();
        let block_types: std::collections::HashMap<AssetId, BlockType> = ctx
            .drain::<BlockType>()
            .into_iter()
            .map(|bt| (bt.asset_id, bt))
            .collect();

        // Whether the blob payloads came from files on disk (`cn run`) rather
        // than an in-memory build (`cn debug`). Captured before the blobs are
        // released; the streaming subsystem uses it to pick a disk-backed
        // payload source so streamed bytes need not stay RAM-resident.
        let blob_disk_backed = ctx.blob.disk_backed();

        // `capture_sources` (cn debug) gathers the file-backed source maps the
        // hot-reload watcher consumes.
        let capture_sources = launch.dev_loop;
        let proc_mesh_args_snapshot = if capture_sources {
            procedural_mesh_snapshot(ctx)
        } else {
            std::collections::HashMap::new()
        };

        // Mesh sources owned by a scene other than the start scene skip their
        // payload decode: draw records use the blob's baked bounds, and the
        // mesh streamer decodes the payload when the owning scene pins.
        let deferred_mesh_sources =
            super::streaming::deferred_mesh_sources(ctx, streaming_config.is_some());
        let draw_list::MeshGeometry {
            meshes: mesh_geometry,
            sources: mesh_sources,
            always_resident: always_resident_meshes,
            component_handles: component_mesh_handles,
            deferred_seeds: deferred_mesh_seeds,
        } = draw_list::load_mesh_geometry(ctx, &deferred_mesh_sources, blob_disk_backed)?;

        let (skinned_geometry, skinned_blob_indices) = self.decode_skinned_geometry(ctx)?;

        // drain Model components into a name-keyed map for Prop lookup
        let models = ctx.drain::<Model>();
        let model_map: std::collections::HashMap<AssetId, Vec<SubMeshRef>> =
            models.into_iter().map(|m| (m.asset_id, m.meshes)).collect();

        // decode Room payloads before shaders/textures are read; all payloads
        // live in the same blob and must be consumed before it is released
        let (room_geometry, room_blob_indices) = draw_list::load_room_geometry(ctx)?;

        let DecodedShaders {
            locators: shader_locators,
            source_map: shader_stage_source_map,
            shaders: decoded_shaders,
        } = self.decode_shaders(ctx, streaming_config.is_some(), capture_sources)?;

        // Read the shared texture pool + the material table into the maps the
        // draw list resolves against.
        let TextureTableDecode {
            locators: texture_locators,
            source_map: asset_source_map,
            name_to_slot: texture_name_to_slot,
            count: texture_count,
        } = self.decode_texture_table(ctx, capture_sources)?;
        let material_map = self.build_material_map(ctx, texture_count)?;

        // Build skinned draw objects, the shared skinned vertex/index buffers,
        // and bind-pose skeletons from the decoded SkinnedMesh geometry. Runs
        // after the material map so SkinnedMesh material references resolve.
        let SkinnedMeshAssembly {
            vertices: skinned_vertices,
            indices: skinned_indices,
            draw_objects: skinned_draw_objects,
            skeletons: skinned_skeletons,
            pool_reservations: skinned_pool_reservations,
            morphs: skinned_morphs,
            source_map: skinned_mesh_source_map,
        } = self.assemble_skinned_meshes(&skinned_geometry, &material_map, capture_sources)?;

        let deferred_slots = super::streaming::deferred_texture_slots(
            ctx,
            streaming_config.is_some(),
            texture_locators.len(),
        );
        let TexturePayloads {
            images: texture_data,
            payloads: texture_payloads,
        } = decode_texture_payloads(ctx, &texture_locators, &deferred_slots, blob_disk_backed)?;

        // Read the sole EnvironmentMap + ColorLut payloads, then build the shared
        // text/sprite atlas pool.
        let (env_map_bytes, environment_map_source) =
            self.decode_environment_map(ctx, capture_sources)?;
        let (color_lut_bytes, color_lut_source) = self.decode_color_lut(ctx, capture_sources)?;
        let TextAtlases {
            atlases: text_atlas_data,
            font_blob_indices,
        } = self.decode_text_atlases(ctx, &texture_locators)?;

        let (light_data, light_uniforms) = gather_lights(ctx, world_ambient_intensity);

        let consumed = shader_locators
            .iter()
            .map(|l| l.blob_index)
            .chain(texture_locators.iter().map(|l| l.blob_index))
            .chain(room_blob_indices)
            .chain(font_blob_indices)
            .chain(skinned_blob_indices);
        for idx in blobs_to_release(consumed, &retained_blobs(ctx)) {
            ctx.release_blob(idx);
        }

        // A geometry-less world (e.g. text-only) is valid: the backend is
        // initialized with empty geometry buffers and only the text path runs.
        let draw_list::DrawListData {
            vertices: mut all_vertices,
            indices: mut all_indices,
            mut draw_objects,
            mut instanced_clusters,
            mesh_handle_to_draws,
            ..
        } = self.assemble_prop_draws(
            ctx,
            PropDrawInputs {
                model_map: &model_map,
                mesh_geometry: &mesh_geometry,
                room_geometry: &room_geometry,
                texture_count,
                material_map: &material_map,
                always_resident_meshes: &always_resident_meshes,
            },
        )?;

        let stream_plan = plan_stream_geometry(StreamGeometry {
            vertices: &mut all_vertices,
            indices: &mut all_indices,
            draw_objects: &mut draw_objects,
            instanced_clusters: &mut instanced_clusters,
            mesh_handle_to_draws: &mesh_handle_to_draws,
            deferred_mesh_seeds: &deferred_mesh_seeds,
            deferred_mesh_counts: &deferred_mesh_sources.counts,
            texture_count: texture_data.len(),
            config: streaming_config.as_ref(),
        });

        let draw_object_count = draw_objects.len();
        let cluster_count = instanced_clusters.len();
        let total_instances: usize = instanced_clusters.iter().map(|c| c.instances.len()).sum();

        let fx = drain_world_fx(ctx, texture_count);
        let decal_count = fx.decals.len();
        let particle_count = fx.particles.len();
        let fog_settings = fx.fog;
        settings.fog_built = fog_settings.is_some();

        // Metal is unaffected by `validation`: its layer is enabled by the CLI
        // re-execing with `MTL_DEBUG_LAYER`.
        let validation = launch.resolve_validation();
        // A shipped run leaves shader hot-reload off, so the backend never spawns
        // the filesystem watcher.
        let hot_reload = launch.dev_loop;
        let capture = launch.frame_capture();
        let embedded_surface = ctx
            .resource::<concinnity_core::render::backend_init::EmbeddedSurface>()
            .copied();

        // Declared probes win; otherwise the geometry-aware auto-seed; otherwise an
        // empty list, which lets the backend run its own coarse-AABB auto-seed.
        let mut probe_placements = declared_probe_placements(ctx);
        if probe_placements.is_empty() {
            probe_placements = auto_seed_probe_placements(
                &draw_objects,
                &all_vertices,
                &all_indices,
                &fx.water_surfaces,
                &fx.glass_panels,
            )
            .unwrap_or_default();
        }

        // Assemble the backend construction inputs, derive the world's render
        // requirements from them (a world with no 3D content drops every
        // scene-scoped feature before any backend resource is sized), and
        // hand the result to the compile-time-selected backend.
        use concinnity_core::render::backend_init::{
            BackendInit, MediaPayloads, SceneData, ShadowParams, WorldShader,
        };
        let mut backend_init = BackendInit {
            window: &settings.window_args,
            validation,
            frames_in_flight: settings.frames_in_flight,
            vsync: settings.vsync,
            clear_color: self.clear_color,
            hot_reload,
            capture,
            embedded_surface,
            scene: SceneData {
                vertices: &all_vertices,
                indices: &all_indices,
                draw_objects,
                instanced_clusters,
                // Sizes the GPU-cull buffers for the merged total; the skinned
                // geometry itself is uploaded after the build.
                n_skinned: skinned_draw_objects.len(),
                // Worst-case resident chunk count, so the GPU-cull buffers
                // reserve a chunk record region (0 for a non-voxel world).
                n_chunk_max: voxel_world
                    .as_ref()
                    .map_or(0, super::streaming::chunk_reserve_count),
            },
            // One entry per world Shader, indexed by ShaderHandle value;
            // entry 0 is the world default program.
            shaders: decoded_shaders
                .iter()
                .map(|s| WorldShader {
                    programs: s.programs.as_ref(),
                    deferred: s.deferred,
                })
                .collect(),
            media: MediaPayloads {
                textures: &texture_data,
                text_atlases: text_atlas_data,
                env_map_bytes: env_map_bytes.as_deref(),
                color_lut_bytes: color_lut_bytes.as_deref(),
            },
            light_uniforms,
            local_lights: light_data.lights,
            spot_shadows: light_data.spot_shadows,
            area_lights: light_data.area_lights,
            shadows: ShadowParams {
                map_size: settings.shadow_map_size,
                update: settings.shadow_update,
                distance: settings.shadow_distance,
                cascades: settings.shadow_cascades,
            },
            anisotropy: settings.anisotropy,
            // Restart-required: the mirror targets are allocated once at backend
            // init, so the quality ceiling scales the engine capacity here.
            planar_planes: quality_ceiling.planar_reflection_planes as usize,
            post,
            fx,
            requirements: Default::default(),
        };
        backend_init.resolve_requirements();
        match self.build_backend(ctx, backend_init) {
            Ok(backend) => self.backend = Some(backend),
            Err(e) => {
                tracing::error!("GraphicsSystem: backend build failed: {e}");
                return None;
            }
        }

        self.finalize_display_modes(ctx, &mut settings);

        // Metal bakes a cube per placement; DirectX and Vulkan no-op.
        if let Some(backend) = self.backend.as_deref_mut() {
            backend.set_reflection_probes(&probe_placements);
        }

        // The dev host resolves the world.jsonl path (discovery is authoring I/O
        // in `concinnity-cook`, which the runtime does not link); embedded preview
        // and MCP-driven runs leave it None.
        let (hot_reload_sources, texture_name_slots) = if capture_sources {
            capture_hot_reload_sources(
                HotReloadSources {
                    map: asset_source_map,
                    color_lut: color_lut_source,
                    environment_map: environment_map_source,
                    meshes: mesh_source_map(&mesh_sources, &mesh_handle_to_draws),
                    skinned_meshes: skinned_mesh_source_map,
                    procedural_meshes: procedural_mesh_source_map(
                        &proc_mesh_args_snapshot,
                        &component_mesh_handles,
                        &mesh_handle_to_draws,
                    ),
                    shader_stages: shader_stage_source_map,
                    world_jsonl_path: crate::app::dev_flags::world_jsonl_path(),
                },
                texture_name_to_slot,
            )
        } else {
            (None, None)
        };

        let skinned_upload = SkinnedUpload {
            vertices: skinned_vertices,
            indices: skinned_indices,
            draw_objects: skinned_draw_objects,
            morphs: skinned_morphs,
        };
        if let Err(e) = self.install_skinned_templates(ctx, skinned_upload, skinned_skeletons) {
            tracing::error!("GraphicsSystem: skinned upload failed: {e}");
            return None;
        }

        self.setup_streaming(StreamingSetup {
            config: streaming_config,
            plan: stream_plan,
            texture_payloads,
            texture_locators: &texture_locators,
            disk_backed: blob_disk_backed,
            deferred_mesh_seeds: &deferred_mesh_seeds,
            voxel_world,
            block_types: &block_types,
            material_map: &material_map,
        });

        self.finalize_backend_config(ctx, &settings);

        self.setup_scene_flow(ctx);

        self.park_runtime_state(
            ctx,
            RuntimeHandoff {
                draw_object_count,
                frames_in_flight: settings.frames_in_flight,
                skinned_pool_reservations: &skinned_pool_reservations,
                fog: fog_settings,
                texture_name_slots,
                hot_reload_sources,
            },
        );
        tracing::info!(
            "GraphicsSystem: ready ({}x{} \"{}\", {} frames in flight, {} draw objects, {} instanced clusters ({} instances total), {} decals, {} particle emitter(s), fog={})",
            settings.window_args.width,
            settings.window_args.height,
            settings.window_args.title,
            settings.frames_in_flight,
            draw_object_count,
            cluster_count,
            total_instances,
            decal_count,
            particle_count,
            if fog_settings.is_some() { "on" } else { "off" },
        );
        ctx.insert_resource(SettingsSlot(Some(settings)));
        Some(())
    }
}

// Set the value TextLabel of every `setting:<key>` HitRegion to the live value
// of that setting. `current_index` maps a setting key to the index of its
// active option (None for an unknown key). Runs once at init, before any
// system drains the HitRegions.
fn sync_setting_value_labels(
    ctx: &mut PipelineContext,
    current_index: impl Fn(&str) -> Option<usize>,
) {
    // (setting key, value-label id) for each settings row.
    let rows: Vec<(String, AssetId)> = ctx
        .query::<HitRegion>()
        .filter_map(|r| {
            let key = crate::settings::action::key(&r.action)?;
            Some((key.to_string(), r.label?))
        })
        .collect();

    for (key, label_id) in rows {
        let (Some(opts), Some(idx)) = (crate::settings::options(&key), current_index(&key)) else {
            continue;
        };
        if let Some(text) = opts.get(idx).copied() {
            for l in ctx.query_mut::<TextLabel>() {
                if l.asset_id == label_id {
                    l.content = text.to_string();
                    break;
                }
            }
        }
    }
}

// Set the value label of the settings row bound to `key` to `text` directly,
// for a label that is not one of the row's static `options` (the master preset
// row's "Auto (High)", or the live "Custom" flip when a quality row changes).
fn set_setting_row_label(ctx: &mut PipelineContext, key: &str, text: &str) {
    let label_id = ctx.query::<HitRegion>().find_map(|r| {
        let row_key = crate::settings::action::key(&r.action)?;
        (row_key == key).then_some(r.label).flatten()
    });
    if let Some(id) = label_id {
        for l in ctx.query_mut::<TextLabel>() {
            if l.asset_id == id {
                l.content = text.to_string();
                break;
            }
        }
    }
}
