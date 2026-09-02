// GraphicsSystem one-time setup: backend creation, draw-list build, and the
// shader / texture / streaming wiring performed on the first tick.

use crate::components::{
    BlockType, Camera3D, Decal, DirectionalLight, GlassPanel, GraphicsConfig, HitRegion, Material,
    Model, ParticleEmitter, PointLight, PostProcessConfig, PostProcessResolve, RectAreaLight,
    SdfVolume, Shader, ShaderKind, SkinnedMeshGeometry, SpotLight, StageSource, StreamingConfig,
    TextLabel, VolumetricFog, VoxelWorld, WaterSurface, Window,
};
use crate::ecs::PipelineContext;
use crate::ecs::asset_id::AssetId;
use crate::gfx::material_entry::MaterialEntry;
use crate::gfx::mesh_payload::Vertex;
use crate::gfx::{draw_list, lights, skeleton, text, transform_propagation};
use std::time::Instant;

use super::helpers::*;
use super::*;

// The resolved render settings the rest of init consumes after
// `init_render_settings` has written the remaining values onto the
// GraphicsSystem: the packed post-processing config handed to the backend ctor,
// the quality ceiling (planar-reflection budget), and the drained StreamingConfig.
struct ResolvedRenderConfig {
    post: crate::gfx::backend_init::PostSettings,
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
    handle: crate::ecs::SkinnedMeshHandle,
    name_id: AssetId,
    mesh: crate::components::SkinnedMesh,
    vertices: Vec<crate::gfx::mesh_payload::SkinnedVertex>,
    indices: Vec<u16>,
    joint_defs: Vec<crate::components::SkeletonJoint>,
    morphs: crate::gfx::mesh_payload::PayloadMorphs,
    lod_alternates: Vec<(f32, Vec<u16>)>,
}

// One skinned mesh's skeleton bookkeeping, driving the SkeletonPose +
// CharacterRig publish after the geometry upload. `template_index` is recorded
// explicitly rather than inferred from position because pre-reserved instance
// copies interleave the draw-object list (template, copies, template, ...).
struct SkinnedSkeletonEntry {
    handle: crate::ecs::SkinnedMeshHandle,
    name_id: AssetId,
    template_index: usize,
    skeleton: skeleton::Skeleton,
    morph_names: Vec<String>,
    model: [[f32; 4]; 4],
    capsule: Option<crate::components::CharacterCapsule>,
    // The authored placement and local bounds, for the editor's pick index.
    transform: crate::components::Transform,
    local_bounds: ([f32; 3], [f32; 3]),
}

// Assembled skinned-mesh GPU inputs from `assemble_skinned_meshes`: the shared
// skinned vertex/index buffers, the per-slot draw objects (templates + their
// hidden pre-reserved instance copies), the per-mesh skeleton bookkeeping, the
// (template, copy) pool reservations, per-slot morph targets, and the hot-reload
// source map.
struct SkinnedMeshAssembly {
    vertices: Vec<crate::gfx::mesh_payload::SkinnedVertex>,
    // Absolute indices into the shared skinned vertex buffer, so u32 rather
    // than the per-mesh u16 the payload carries.
    indices: Vec<u32>,
    draw_objects: Vec<crate::gfx::render_types::SkinnedDrawObject>,
    skeletons: Vec<SkinnedSkeletonEntry>,
    pool_reservations: Vec<(usize, usize)>,
    morphs: Vec<Option<std::sync::Arc<crate::gfx::mesh_payload::PayloadMorphs>>>,
    source_map: super::hot_reload_sources::SkinnedMeshSourceMap,
}

// The shared texture pool decoded from the TextureTable by `decode_texture_table`:
// each texture's payload locator (dense by pool slot / cook TextureHandle), the
// dev-only file-backed source map + name->slot index (cn debug hot-reload / spawn
// by name), and the pool size.
struct TextureTableDecode {
    locators: Vec<crate::ecs::PayloadLocator>,
    source_map: super::hot_reload_sources::TextureSourceMap,
    name_to_slot: std::collections::HashMap<AssetId, usize>,
    count: usize,
}

// The world's decoded shaders from `decode_shaders`: each stage's compiled
// payload bytes (the main vertex + fragment, an engine-internal empty shadow
// slice, and the optional instanced-vertex stage), the payload locators kept
// for the blob-release step, and the dev-only source map the hot-reload watcher
// subscribes to.
#[derive(Default)]
struct DecodedShaderBytes {
    vert: Vec<u8>,
    frag: Vec<u8>,
    vert_instanced: Vec<u8>,
    // The payload was left undecoded because a scene other than the start scene
    // owns this bucket. Recorded explicitly rather than inferred from empty stage
    // bytes: a stage the cook compiled nothing for reads as empty too.
    deferred: bool,
}

// Where the streaming pump re-reads a deferred bucket's stage container: the
// blob's byte range when the world is disk-backed (`cn run`, so the bytes
// never stay RAM-resident), else a copy of the in-memory payload.
fn deferred_shader_source(
    ctx: &mut PipelineContext,
    locator: &crate::ecs::PayloadLocator,
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
    let path = crate::blob::blob_path(locator.blob_index)
        .ok_or_else(|| format!("blob {}: no blob layout installed", locator.blob_index))?;
    let start = crate::blob::payload_section_start(&path).map_err(|e| format!("{e:?}"))?;
    Ok(ShaderPayloadSource::Disk {
        path,
        offset: start + locator.offset,
        len: locator.len,
    })
}

struct DecodedShaders {
    locators: Vec<crate::ecs::PayloadLocator>,
    source_map: super::hot_reload_sources::ShaderStageSourceMap,
    // One entry per world Shader, in drain order == cook handle order, so a
    // baked ShaderHandle value indexes this directly. Entry 0 is the world
    // default pipeline's program.
    shaders: Vec<DecodedShaderBytes>,
    shadow_bytes: Vec<u8>,
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
    ctx.query::<crate::components::TextLabel>()
        .any(|l| l.font.is_none())
        || ctx
            .query::<crate::components::TextInput>()
            .any(|t| t.font.is_none())
}

// Per-streamed-mesh data from `mesh_stream_data`: the draw-object index of each
// streamed mesh, its scoring centre, and its decoded per-mesh geometry copy.
// The three vecs are column-aligned.
struct MeshStreamData {
    draw_indices: Vec<usize>,
    centers: Vec<Vec<[f32; 3]>>,
    payloads: Vec<crate::gfx::streaming::mesh::DecodedMesh>,
}

// Per-texture-slot draw positions for the streaming scorer, which ranks each
// texture by the camera's distance to the nearest draw that samples it. Albedo
// and normal maps share one pool, so a draw contributes its position to both the
// slot it samples as albedo and the one it samples as a normal map
// (`NO_NORMAL_MAP_SLOT` = no normal map, scored by neither). `texture_count`
// sizes the outer vec so every pool slot has an entry.
fn texture_stream_centers(
    draw_objects: &[crate::gfx::render_types::DrawObject],
    texture_count: usize,
) -> Vec<Vec<[f32; 3]>> {
    let mut centers = vec![Vec::new(); texture_count];
    for obj in draw_objects {
        let pos = draw_object_position(obj);
        if let Some(slot) = centers.get_mut(obj.texture_slot) {
            slot.push(pos);
        }
        if obj.normal_map_slot != crate::gfx::render_types::NO_NORMAL_MAP_SLOT
            && let Some(slot) = centers.get_mut(obj.normal_map_slot)
        {
            slot.push(pos);
        }
    }
    centers
}

// Per-streamed-mesh data captured before `draw_objects` moves into the backend.
// Only static, frustum-cullable draws stream; skybox, rooms, and dynamic props
// (sentinel AABB) stay resident so structural geometry never pops in. Each
// payload copies the draw's region of the shared vertex/index buffers, scored by
// its AABB centre; indices are stored mesh-relative and narrowed to u16 (each
// per-mesh region fits in u16 by the build-time splitter). Draws whose
// build-time offsets fall out of range are skipped defensively.
fn mesh_stream_data(
    draw_objects: &[crate::gfx::render_types::DrawObject],
    all_vertices: &[Vertex],
    all_indices: &[u32],
    deferred_draws: &std::collections::HashSet<usize>,
) -> MeshStreamData {
    let mut draw_indices: Vec<usize> = Vec::new();
    let mut centers: Vec<Vec<[f32; 3]>> = Vec::new();
    let mut payloads: Vec<crate::gfx::streaming::mesh::DecodedMesh> = Vec::new();
    for (draw_idx, obj) in draw_objects.iter().enumerate() {
        if !obj.cullable() {
            continue;
        }
        // A deferred draw appended no geometry (its record carries baked
        // counts over an empty region): stream it with an empty payload copy;
        // the deferred source decodes the blob payload instead.
        if deferred_draws.contains(&draw_idx) {
            draw_indices.push(draw_idx);
            centers.push(vec![draw_object_position(obj)]);
            payloads.push(crate::gfx::streaming::mesh::DecodedMesh {
                vertices: Vec::new(),
                indices: Vec::new(),
            });
            continue;
        }
        let vstart = obj.vertex_offset / std::mem::size_of::<Vertex>();
        let vend = vstart + obj.vertex_count;
        let iend = obj.index_offset + obj.index_count;
        if vend > all_vertices.len() || iend > all_indices.len() {
            continue;
        }
        draw_indices.push(draw_idx);
        centers.push(vec![draw_object_position(obj)]);
        let vbase = vstart as u32;
        payloads.push(crate::gfx::streaming::mesh::DecodedMesh {
            vertices: all_vertices[vstart..vend].to_vec(),
            indices: all_indices[obj.index_offset..iend]
                .iter()
                .map(|&i| (i - vbase) as u16)
                .collect(),
        });
    }
    MeshStreamData {
        draw_indices,
        centers,
        payloads,
    }
}

impl GraphicsSystem {
    // Resolve every render setting (window, quality preset + ceiling, post-process
    // tunables, shadows, streaming caps, keymap) onto the GraphicsSystem, sync the
    // settings-menu value labels, and return the config the rest of init needs.
    fn init_render_settings(&mut self, ctx: &mut PipelineContext) -> ResolvedRenderConfig {
        // Persisted settings-menu choices override the world's authored defaults
        // below (each field is None when the user never changed that setting).
        let user_graphics = self.persisted_settings().graphics;
        self.persisted_graphics = user_graphics.clone();

        // Detect the GPU before the backend is built so the auto-config quality
        // ceiling can influence the render targets / effect pipelines sized at
        // backend init. Held on self for later (e.g. the menu's preset label).
        self.gpu_profile = self.detect_gpu_profile();
        // Published so readouts outside the graphics system (the editor's Health
        // panel) can size live VRAM against the device's budget without reaching
        // for the backend itself.
        ctx.insert_resource(self.gpu_profile);
        crate::crash::note(
            "gpu",
            &format!("{:?} {:?}", self.gpu_profile.vendor, self.gpu_profile.tier),
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
        let active_preset =
            crate::app::dev_flags::resolve_quality_preset(user_graphics.quality_preset)
                .unwrap_or_else(|| {
                    self.seed_first_launch_preset();
                    QualityPreset::Auto
                });
        // Hold the resolved preset as the live value the settings-menu master
        // row cycles (and that an individual quality-row change flips to Custom).
        self.quality_preset = active_preset;
        let quality_ceiling =
            crate::gfx::quality_preset::resolve_ceiling(active_preset, &self.gpu_profile);
        tracing::info!(
            "auto-config: GPU tier {:?}, quality preset {:?}",
            self.gpu_profile.tier,
            active_preset,
        );

        if let Some(w) = ctx.drain::<Window>().into_iter().next() {
            self.window_args = w;
        }
        // Capture the DebugHud chip ids (cursor, camera, sys, passes stack
        // order) so the frame step can anchor them to the top-right of the
        // window. Passes is last because it grows/shrinks with the frame's step
        // count, so keeping it at the bottom leaves the fixed-height chips
        // unshifted. The DebugHud component is queried (not drained) by its
        // system, so it is still present here; absent fields are skipped.
        self.debug_hud_chips = ctx
            .query::<crate::components::DebugHud>()
            .next()
            .map(|d| {
                [d.mouse_label, d.camera_label, d.sys_label, d.passes_label]
                    .into_iter()
                    .flatten()
                    .collect()
            })
            .unwrap_or_default();
        // Capture the StatHud chip ids (fps, vram, ram, ev, edr strip order) so
        // the frame step can pack them tight from the top-left. Like DebugHud
        // the component is queried (not drained), so it is still present here.
        self.stat_hud_chips = ctx
            .query::<crate::components::StatHud>()
            .next()
            .map(|s| {
                [
                    s.fps_label,
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
            self.window_args.mode = m;
        }
        // The chosen fullscreen display mode. Fullscreen-only: it never feeds
        // the windowed size, which stays the world's authored `Window` value.
        if let Some([w, h, hz]) = user_graphics.resolution {
            self.resolution = Some(crate::gfx::display_mode::DisplayMode {
                width: w,
                height: h,
                refresh_hz: hz,
            });
        }

        if let Some(c) = ctx.drain::<GraphicsConfig>().into_iter().next() {
            let args = c;
            self.frames_in_flight = args.frames_in_flight as usize;
            self.vsync = args.vsync;
            self.fps_cap = args.fps_cap;
            self.clear_color = args.clear_color;
            self.max_frames = args.max_frames;
            self.shadow_map_size = args.shadow_map_size;
            self.shadow_update = args.shadow_update;
            self.shadow_distance = args.shadow_distance;
            self.shadow_cascades = args.shadow_cascades;
            self.anisotropy = args.anisotropy;
        }
        // A persisted vsync choice overrides the world's value. Applied outside
        // the GraphicsConfig block (unconditional), matching window_mode /
        // resolution, so it wins over both the authored value and the default.
        if let Some(v) = user_graphics.vsync {
            self.vsync = v;
        }
        // A persisted frame-rate cap overrides the world's value (0 = unlimited),
        // applied live by the render-step pacer. Independent of the quality preset,
        // like vsync, so no ceiling clamp.
        if let Some(v) = user_graphics.fps_cap {
            self.fps_cap = v;
        }
        // Stats-HUD display toggles (None = shown, the default, so an existing
        // settings file keeps the FPS / VRAM chips visible). Independent of the
        // quality preset, like vsync / fps_cap.
        if let Some(v) = user_graphics.perf_stats {
            self.perf_stats = v;
        }
        if let Some(v) = user_graphics.show_fps {
            self.show_fps = v;
        }
        if let Some(v) = user_graphics.show_vram {
            self.show_vram = v;
        }

        // Shadow quality knobs (GraphicsConfig-sourced). Snapshot the world's
        // authored values as the baseline a live preset change re-clamps from,
        // then apply any persisted override and otherwise clamp under the preset
        // ceiling (an explicit override wins, like the quality toggles below). The
        // resolution is restart-required -- the shadow map array is sized from
        // `self.shadow_map_size` at backend init below -- while the cadence is read
        // by the cascade scheduler each frame.
        use crate::gfx::render_config as resolve;
        self.authored_shadow_map_size = self.shadow_map_size;
        self.authored_shadow_update = self.shadow_update;
        self.shadow_map_size =
            resolve::shadow_map_size(self.shadow_map_size, &user_graphics, &quality_ceiling);
        self.shadow_update =
            resolve::shadow_update(self.shadow_update, &user_graphics, &quality_ceiling);
        // Shadow distance (GraphicsConfig-sourced, live -- the per-frame cascade
        // split reads it). Same baseline / override / ceiling-clamp shape as the
        // shadow knobs above.
        self.authored_shadow_distance = self.shadow_distance;
        self.shadow_distance =
            resolve::shadow_distance(self.shadow_distance, &user_graphics, &quality_ceiling);
        // Shadow cascade count (GraphicsConfig-sourced, live -- the per-frame split
        // + schedule read it). Same baseline / override / ceiling-clamp shape.
        self.authored_shadow_cascades = self.shadow_cascades;
        self.shadow_cascades =
            resolve::shadow_cascades(self.shadow_cascades, &user_graphics, &quality_ceiling);
        // Anisotropy (GraphicsConfig-sourced, restart-required -- the scene sampler
        // is built from `self.anisotropy` at backend init below). Same baseline /
        // override / ceiling-clamp shape as the shadow knobs above.
        self.authored_anisotropy = self.anisotropy;
        self.anisotropy = resolve::anisotropy(self.anisotropy, &user_graphics, &quality_ceiling);
        // Frames-in-flight (ring-buffer depth): a persisted override clamped to the
        // 1..3 the backends support, applied unconditionally like vsync. Restart-
        // required (the ring buffers are sized at backend init below), independent
        // of the quality preset.
        if let Some(v) = user_graphics.frames_in_flight {
            self.frames_in_flight = (v as usize).clamp(1, 3);
        }

        // Resolve post-process tunables. The first declared PostProcessConfig
        // wins; with none declared the renderer uses the stack defaults. The
        // AA mode resolves into a TAA gate (threaded alongside the params) and
        // the composite `fxaa` flag inside `post_process` (refreshed below once
        // the override + ceiling clamp have settled the final mode).
        let post_config = ctx.drain::<PostProcessConfig>().into_iter().next();
        // Persisted slider choices override the world's values, re-applied here
        // each launch so they survive a restart. The transform / clamp is shared
        // with the live drag-apply via `settings::slider_apply_value`, so the
        // value re-applied at launch matches the value applied at drag time.
        let mut post_process = resolve::post_process_params(post_config.as_ref(), &user_graphics);
        // Keep a copy as the live source of truth for the slider settings to
        // read at init and mutate at runtime (PostProcessParams is Copy, so the
        // value is still passed into the backend below).
        self.post_process = post_process;
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
        self.ambient_intensity = resolve::ambient_intensity(post_config.as_ref(), &user_graphics);
        // Quality-feature toggles: the world's config overlaid with the user's
        // persisted choices, stored as the source of truth for the Quality-group
        // rows. A runtime toggle flips a field here, re-derives the per-feature
        // settings, and rebuilds the affected GPU resources. A world that
        // declares no config falls back to the schema defaults, which author the
        // top-tier look, so the overrides + ceiling below apply either way: the
        // preset is what settles a default world's quality.
        self.post_config = post_config.clone().unwrap_or_default();
        // The pristine world baseline, before the user overrides + preset ceiling
        // below. A live preset change re-clamps the quality toggles from this, so
        // raising a preset restores the world's features (a ceiling never enables
        // anything the world did not author, so re-clamping the baseline is exact).
        self.authored_post_config = self.post_config.clone();
        resolve::overlay_quality_overrides(&mut self.post_config, &user_graphics);
        // The active quality preset as a performance ceiling over the toggles
        // above: where the ceiling disallows a feature, force it off -- but only
        // for a toggle the user did not explicitly override, and never turning a
        // feature on. A no-op under Custom (the ceiling permits everything).
        resolve::clamp_quality_under_ceiling(
            &mut self.post_config,
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
        let taa_enabled = self.post_config.aa_mode.taa_enabled();
        // The composite FXAA flag follows the final (overridden + ceiling-clamped)
        // AA mode. resolve() seeded `post_process.fxaa` from the authored mode
        // before the override/clamp above, so refresh both the local copy passed
        // to the backend ctor and the live `self.post_process` here.
        post_process.fxaa = self.post_config.aa_mode.fxaa_flag();
        self.post_process.fxaa = post_process.fxaa;
        let ssao_settings = self.post_config.ssao_settings();
        let ssr_settings = self.post_config.ssr_settings();
        let rt_reflection_settings = self.post_config.rt_reflection_settings();
        let reflection_blur_scale = self.post_config.reflection_blur_divisor();
        let ssgi_settings = self.post_config.ssgi_settings();
        // The authored `exposure_ev` becomes an additive bias on the adapted EV
        // when auto-exposure is on; otherwise the static path bakes it into
        // `post_process.exposure` (resolve()) and the bias here is unused.
        let auto_exposure_settings = self.post_config.auto_exposure_settings();
        let auto_exposure_bias_ev = self.post_config.exposure_ev;
        // Display-output / upscaling preferences: the world's value overridden by
        // any persisted settings-menu choice. Restart-required (the swapchain
        // format + render targets are sized once at init), so they are read here,
        // passed to the backend ctor below, and held on self for the settings rows
        // to display + cycle. Independent of the quality preset (a user choice,
        // not a tier), so they never clamp under the ceiling or flip it to Custom.
        // HDR display output is additionally gated on the platform advertising an
        // HDR-capable surface (else it warns and falls back to the SDR composite).
        self.hdr_display = user_graphics
            .hdr_display
            .unwrap_or_else(|| post_config.as_ref().map(|c| c.hdr_display).unwrap_or(false));
        self.hdr_pq = user_graphics
            .hdr_pq
            .unwrap_or_else(|| post_config.as_ref().map(|c| c.hdr_pq).unwrap_or(false));
        self.temporal_upscaling = user_graphics.temporal_upscaling.unwrap_or_else(|| {
            post_config
                .as_ref()
                .map(|c| c.temporal_upscaling)
                .unwrap_or(false)
        });
        let hdr_display = self.hdr_display;
        let hdr_pq = self.hdr_pq;
        let temporal_upscaling = self.temporal_upscaling;
        // Two-pass Hi-Z occlusion + texture-streaming quality: also restart-class
        // and independent of the preset, resolved here (before the value-label sync
        // below) from the world's config overridden by any persisted choice.
        // `occlusion_two_pass` is gated on the bindless GPU-cull path being active
        // (the cull pipeline must exist). The texture pool size + per-frame upload
        // budget come from the StreamingConfig, drained here so the override lands
        // before the streamer is built later; the pool only bites where the world
        // declares streaming.
        self.occlusion_two_pass = user_graphics.occlusion_two_pass.unwrap_or_else(|| {
            post_config
                .as_ref()
                .map(|c| c.occlusion_two_pass)
                .unwrap_or(false)
        });
        let occlusion_two_pass = self.occlusion_two_pass;
        let mut streaming_config = ctx.drain::<StreamingConfig>().into_iter().next();
        if let Some(sc) = streaming_config.as_mut() {
            if let Some(v) = user_graphics.texture_cap {
                sc.texture_cap = v;
            }
            if let Some(v) = user_graphics.texture_budget {
                sc.texture_budget = v;
            }
        }
        self.texture_cap = streaming_config
            .as_ref()
            .map(|c| c.texture_cap)
            .unwrap_or(96);
        self.texture_budget = streaming_config
            .as_ref()
            .map(|c| c.texture_budget)
            .unwrap_or(4);
        // Render-scale (upscaling quality): the world's choice overridden by any
        // persisted settings-menu choice. Restart-required -- the upscaler and
        // render targets are sized from this once, here. `self.render_scale` is
        // kept for the settings row to display and cycle.
        let world_quality = post_config
            .as_ref()
            .map(|c| c.upscale_quality)
            .unwrap_or_default();
        // A persisted render-scale choice wins; otherwise the world's choice,
        // clamped under the preset ceiling (the more aggressive of the two, so a
        // weak-tier ceiling forces more upscaling but never less).
        self.render_scale = match user_graphics.render_scale {
            Some(v) => v,
            None => crate::gfx::quality_preset::more_aggressive_upscale(
                world_quality,
                quality_ceiling.min_upscale,
            ),
        };
        let upscale_scale = if post_config.is_some() {
            self.render_scale.scale()
        } else {
            1.0
        };
        // Upscaler backend (Auto / FSR3 / DLSS / XeSS): the persisted choice wins,
        // else the world's value. Restart-required (the upscaler is selected +
        // built once at init); independent of the quality preset, so no ceiling
        // clamp. Resolved here (ahead of the value-label sync) so the settings row
        // shows the live value. DirectX / Vulkan honour it; Metal uses MetalFX.
        self.upscale_backend = user_graphics.upscale_backend.unwrap_or_else(|| {
            post_config
                .as_ref()
                .map(|c| c.upscale_backend)
                .unwrap_or_default()
        });

        // Set each settings value label to its live value before the first
        // render, so a persisted/authored choice shows instead of the build's
        // placeholder. HitRegions are still present here: GraphicsSystem.init
        // runs before UiInputSystem.init, which drains them.
        let (vsync, mode, scale) = (self.vsync, self.window_args.mode, self.render_scale);
        let fps_cap_val = self.fps_cap;
        // Stats-HUD display toggles for the value-label sync (copies, so the
        // closure below does not borrow self while ctx is borrowed mutably).
        let (perf_stats_val, show_fps_val, show_vram_val) =
            (self.perf_stats, self.show_fps, self.show_vram);
        // Display-group toggle states for the value-label sync (copies, so the
        // closure below does not borrow self while ctx is borrowed mutably).
        let (display_upscaling, display_hdr, display_pq) =
            (self.temporal_upscaling, self.hdr_display, self.hdr_pq);
        // Upscaler-backend selection for the value-label sync (a copy, same
        // reason as the display tuple above).
        let upscale_backend_sel = self.upscale_backend;
        // Shadow knob states for the value-label sync (copies, same reason).
        let (shadow_size, shadow_update_val) = (self.shadow_map_size, self.shadow_update);
        let shadow_distance_val = self.shadow_distance;
        let shadow_cascades_val = self.shadow_cascades;
        let anisotropy_val = self.anisotropy;
        // System / streaming restart-row states for the value-label sync (copies).
        // `occlusion_two_pass` is already a local above.
        let (frames_in_flight_n, texture_cap_n) = (self.frames_in_flight as u32, self.texture_cap);
        // Audio / controls value labels read from the persisted settings store
        // (with the baseline default when unset); their owning systems apply the
        // value at their own init.
        let user_settings = self.persisted_settings();
        let volume_of =
            |stored: Option<f32>| stored.unwrap_or(crate::gfx::settings::DEFAULT_VOLUME);
        let master_volume = volume_of(user_settings.audio.master_volume);
        let music_volume = volume_of(user_settings.audio.music_volume);
        let sfx_volume = volume_of(user_settings.audio.sfx_volume);
        let voice_volume = volume_of(user_settings.audio.voice_volume);
        // Movement key map: a persisted rebind set overrides the engine default.
        // Pushed to the backend after it is built (below) and used to sync the
        // Controls-tab rebind row labels (`init_rebind_rows`).
        self.keymap = user_settings.controls.keymap.unwrap_or_default();
        // Gamepad button map: same override rule; InputSystem loads its own
        // copy at its init, so this one only drives the rebind row labels and
        // the SettingsState handoff.
        self.gamepad_map = user_settings.controls.gamepad_map.unwrap_or_default();
        // Snapshot of the resolved quality toggles for the value-label arm below
        // (a copy, matching the other snapshot locals, so the closure does not
        // borrow self while ctx is borrowed mutably).
        let quality_cfg = self.post_config.clone();
        sync_setting_value_labels(ctx, |key| match key {
            "vsync" => Some(vsync as usize),
            "fps_cap" => Some(crate::gfx::settings::fps_cap_index(fps_cap_val)),
            "window_mode" => Some(crate::gfx::settings::window_mode_index(mode)),
            // "resolution" is a dynamic dropdown; its label is set from the
            // enumerated mode list after the backend is built.
            "render_scale" => Some(crate::gfx::settings::render_scale_index(scale)),
            "upscale_backend" => Some(crate::gfx::settings::upscale_backend_index(
                upscale_backend_sel,
            )),
            "master_volume" => Some(crate::gfx::settings::volume_index(master_volume)),
            "music_volume" => Some(crate::gfx::settings::volume_index(music_volume)),
            "sfx_volume" => Some(crate::gfx::settings::volume_index(sfx_volume)),
            "voice_volume" => Some(crate::gfx::settings::volume_index(voice_volume)),
            // Display-output / upscaling toggles (Off/On), held on self.
            "temporal_upscaling" => Some(display_upscaling as usize),
            "hdr_display" => Some(display_hdr as usize),
            "hdr_pq" => Some(display_pq as usize),
            // Stats-HUD display toggles (Off/On), held on self.
            "perf_stats" => Some(perf_stats_val as usize),
            "show_fps" => Some(show_fps_val as usize),
            "show_vram" => Some(show_vram_val as usize),
            // Shadow quality knobs (resolution restart-required, cadence live).
            "shadow_map_size" => Some(crate::gfx::settings::shadow_resolution_index(shadow_size)),
            "shadow_update" => Some(crate::gfx::settings::shadow_update_index(shadow_update_val)),
            "shadow_distance" => Some(crate::gfx::settings::shadow_distance_index(
                shadow_distance_val,
            )),
            "shadow_cascades" => Some(crate::gfx::settings::shadow_cascades_index(
                shadow_cascades_val,
            )),
            "anisotropy" => Some(crate::gfx::settings::anisotropy_index(anisotropy_val)),
            // System / streaming restart rows.
            "frames_in_flight" => Some(crate::gfx::settings::frames_in_flight_index(
                frames_in_flight_n,
            )),
            "occlusion_two_pass" => Some(occlusion_two_pass as usize),
            "texture_quality" => Some(crate::gfx::settings::texture_quality_index(texture_cap_n)),
            // mouse_sensitivity is a slider now, synced by `init_sliders`.
            // Quality toggles: index 0 = Off, 1 = On, matching OFF_ON_OPTIONS.
            key if crate::gfx::settings::is_quality_toggle(key) => {
                super::quality_toggle_on(&quality_cfg, key).map(|on| on as usize)
            }
            // SSGI gather sub-quality dropdowns.
            key if super::is_quality_cycle(key) => super::quality_cycle_index(&quality_cfg, key),
            _ => None,
        });
        // The master "Graphics Quality" row carries the resolved tier under Auto
        // (e.g. "Auto (High)"), which the static option table cannot express, so
        // it is set directly after the generic sync above writes the bare name.
        let preset_label =
            crate::gfx::quality_preset::preset_label(active_preset, &self.gpu_profile);
        set_setting_row_label(ctx, "graphics_quality", &preset_label);
        // Capture the slider rows and sync each handle + value label to its live
        // value (e.g. the persisted/authored exposure). Like the cycle-row sync
        // above, this runs before UiInputSystem drains the HitRegions.
        self.init_sliders(ctx);
        // Capture the rebind rows and sync each value label to the live bound
        // key (persisted or default). Like the slider sync, before UiInputSystem
        // drains the HitRegions.
        self.init_rebind_rows(ctx);
        // Capture each cycle row's value-label id, so a preset change can relabel
        // its dependent rows (and a quality-row change the master row) at runtime,
        // when the HitRegions are gone. Also before UiInputSystem drains them.
        self.init_cycle_value_labels(ctx);
        // Capture the show_fps / show_vram row labels and apply the initial
        // gray-out from the resolved "Display performance stats" master toggle.
        // Before UiInputSystem drains the HitRegions / ScrollPanels.
        self.capture_perf_sub_rows(ctx);
        // Capture the Resolution row's labels and apply the initial gray-out
        // from the resolved window mode (the row only applies in fullscreen).
        self.capture_resolution_row(ctx);
        // Capture each ScrollPanel's per-element clip bands for the draw path,
        // before UiInputSystem drains the panels (init order: graphics first).
        self.init_clip_rects(ctx);
        // Upscaler backend selector, resolved above (persisted choice over the
        // world's `PostProcessConfig.upscale_backend`) and held on self for the
        // settings row. Honoured by the DirectX and Vulkan backends (FSR3 / DLSS /
        // XeSS); Metal always uses MetalFX, so it ignores the selector.
        let upscale_backend = self.upscale_backend;

        let post = crate::gfx::backend_init::PostSettings {
            post_process,
            taa_enabled,
            ssao: ssao_settings,
            ssr: ssr_settings,
            ssgi: ssgi_settings,
            rt_reflections: rt_reflection_settings,
            rt_dynamic: crate::app::dev_flags::resolve_rt_dynamic(),
            rt_skinned_geometry: crate::app::dev_flags::resolve_rt_skinned_geometry(),
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
        ResolvedRenderConfig {
            post,
            quality_ceiling,
            streaming_config,
            world_ambient_intensity: world_ambient,
        }
    }

    // Decode every SkinnedMesh resource-table entry's geometry payload (before
    // the shared blob is released) into a handle-ordered table, and publish the
    // name -> handle index + skin-selector list for the animation systems.
    // Returns the decoded geometry and the blob indices its payloads occupy (for
    // the release step), or None if any entry's baked data or payload is missing
    // or malformed (self.failed already set).
    fn decode_skinned_geometry(
        &mut self,
        ctx: &mut PipelineContext,
    ) -> Option<(Vec<SkinnedGeometry>, Vec<u32>)> {
        // Load the SkinnedMesh resource table and decode each entry's geometry
        // payload now, before the shared blob is released. The placement,
        // material references, capsule, and spawn reserve travel in the baked
        // `data_bytes`; the vertex/index geometry + skeleton in the compiled
        // payload. The table index IS the mesh's `SkinnedMeshHandle`, which keys
        // the whole animation correlation web.
        let skinned_table = ctx
            .resource::<crate::resource::SkinnedMeshTable>()
            .cloned()
            .unwrap_or_default();
        let mut skinned_geometry: Vec<SkinnedGeometry> = Vec::new();
        let mut skinned_blob_indices: Vec<u32> = Vec::new();
        // Interned name -> handle, published for the debug WS animation
        // commands, which address a mesh by its typed name.
        let mut skinned_name_index: std::collections::HashMap<
            AssetId,
            crate::ecs::SkinnedMeshHandle,
        > = std::collections::HashMap::new();
        for (handle, entry) in skinned_table.0.iter().enumerate() {
            let handle = crate::ecs::SkinnedMeshHandle(handle as u32);
            let (name_id, sm): (u32, crate::components::SkinnedMesh) =
                match postcard::from_bytes(&entry.data_bytes) {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::error!(
                            "GraphicsSystem: SkinnedMesh handle {} baked data failed to decode: {}",
                            handle.index(),
                            e
                        );
                        self.failed = true;
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
                    self.failed = true;
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
                    self.failed = true;
                    return None;
                }
            };
            match crate::gfx::mesh_payload::deserialise_skinned_with_lods(&bytes) {
                Ok(p) => {
                    let joint_defs = crate::geometry::payload_joints_to_defs(p.joints);
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
                    self.failed = true;
                    return None;
                }
            }
        }
        // Publish the name index before AnimationSystem inits (it runs after
        // GraphicsSystem) so debug WS animation commands can resolve a typed
        // mesh name to the handle keying the correlation web. The skin
        // selectors ride along for the animation reload catalogue.
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
    // Returns None (self.failed set) if a mesh references an unknown material.
    fn assemble_skinned_meshes(
        &mut self,
        skinned_geometry: &[SkinnedGeometry],
        material_map: &std::collections::HashMap<crate::ecs::MaterialHandle, MaterialEntry>,
        texture_count: usize,
        capture_sources: bool,
    ) -> Option<SkinnedMeshAssembly> {
        let mut skinned_vertices: Vec<crate::gfx::mesh_payload::SkinnedVertex> = Vec::new();
        let mut skinned_indices: Vec<u32> = Vec::new();
        let mut skinned_draw_objects: Vec<crate::gfx::render_types::SkinnedDrawObject> = Vec::new();
        // One entry per authored skinned mesh: its handle, interned name id,
        // the skinned index of its (visible) template draw object, and its
        let mut skinned_skeletons: Vec<SkinnedSkeletonEntry> = Vec::new();
        // `(template_index, instance_index)` pairs seeding the backend skinned
        // instance pool: each instance is a hidden bind-pose copy reserved from
        // SkinnedMesh.max_instances.
        let mut skinned_pool_reservations: Vec<(usize, usize)> = Vec::new();
        // Morph-target data per skinned draw object; instance copies share
        // their template's data through the Arc.
        let mut skinned_morphs: Vec<
            Option<std::sync::Arc<crate::gfx::mesh_payload::PayloadMorphs>>,
        > = Vec::new();
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
            let mat_entry = match crate::gfx::material_entry::resolve_material_slots(
                sm.material,
                sm.texture,
                material_map,
                texture_count,
            ) {
                Ok(entry) => entry,
                Err(mat_id) => {
                    tracing::error!(
                        "GraphicsSystem: SkinnedMesh '{}' references unknown material {}",
                        name_id,
                        mat_id.index()
                    );
                    self.failed = true;
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

            let skeleton = crate::components::build_skeleton_from_joint_defs(joint_defs);
            let joint_count = skeleton.len().min(crate::gfx::render_types::MAX_JOINTS);

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
            skinned_draw_objects.push(crate::gfx::render_types::SkinnedDrawObject {
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
                skinned_draw_objects.push(crate::gfx::render_types::SkinnedDrawObject {
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
                transform: crate::components::Transform {
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
    // neither. Returns None (self.failed set) if a texture lacks a payload.
    fn decode_texture_table(
        &mut self,
        ctx: &mut PipelineContext,
        capture_sources: bool,
    ) -> Option<TextureTableDecode> {
        // The shared texture pool comes from the blob's resource stream: cook
        // assigned each texture a dense `TextureHandle` (== its pool slot) and the
        // runtime loaded them into a `TextureTable`. Reading the table by handle
        // replaces draining a `Texture` component column and scanning names.
        let texture_table = ctx
            .resource::<crate::resource::TextureTable>()
            .cloned()
            .unwrap_or_default();
        // Dev-only source catalogue (present under `cn debug`) so the hot-reload
        // watcher can map a texture handle back to the file that backs it.
        let texture_sources = ctx.resource::<crate::resource::TextureSources>().cloned();
        let mut texture_locators = Vec::with_capacity(texture_table.len());
        let mut asset_source_map = super::hot_reload_sources::TextureSourceMap::new();
        // Name -> pool slot, built only under `cn debug` for the runtime
        // spawn-by-name path (`WorldReloadState`).
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
                    self.failed = true;
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
    // payload; all data lives in the baked `data_bytes`. Returns None
    // (self.failed set) on any decode or resolution failure.
    fn build_material_map(
        &mut self,
        ctx: &mut PipelineContext,
        texture_count: usize,
    ) -> Option<std::collections::HashMap<crate::ecs::MaterialHandle, MaterialEntry>> {
        let material_table = ctx
            .resource::<crate::resource::MaterialTable>()
            .cloned()
            .unwrap_or_default();
        let mut material_map: std::collections::HashMap<crate::ecs::MaterialHandle, MaterialEntry> =
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
                    self.failed = true;
                    return None;
                }
            };
            match crate::gfx::material_entry::of(&mat, texture_count) {
                Ok(entry) => {
                    material_map.insert(crate::ecs::MaterialHandle(material_handle as u32), entry);
                }
                Err(field) => {
                    tracing::error!(
                        "GraphicsSystem: Material {} references an out-of-range {} texture handle (only {} textures)",
                        material_handle,
                        field,
                        texture_count
                    );
                    self.failed = true;
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
    // None (self.failed set) if any payload is missing or unreadable.
    //
    // A world that declares no Shader is the common case: it gets a single
    // bucket carrying no bytes, which every backend reads as "use the engine's
    // own main-pass program".
    fn decode_shaders(
        &mut self,
        ctx: &mut PipelineContext,
        streaming: bool,
    ) -> Option<DecodedShaders> {
        let world_shaders = ctx.drain::<Shader>();
        if world_shaders.is_empty() {
            return Some(DecodedShaders {
                locators: Vec::new(),
                shaders: vec![DecodedShaderBytes::default()],
                source_map: super::hot_reload_sources::ShaderStageSourceMap::new(),
                shadow_bytes: Vec::new(),
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
                    self.failed = true;
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
                        shaders.push(DecodedShaderBytes {
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
                Ok(b) => match crate::components::ShaderPayload::decode(b) {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::error!("GraphicsSystem: shader payload decode: {:?}", e);
                        self.failed = true;
                        return None;
                    }
                },
                Err(e) => {
                    tracing::error!("GraphicsSystem: failed to read shader payload: {:?}", e);
                    self.failed = true;
                    return None;
                }
            };
            // A stage the cook compiled nothing for (the Vulkan inline-GLSL
            // carve-out) reads as empty bytes; the backend falls back per stage.
            let stage_bytes =
                |kind: ShaderKind| payload.stage(kind).map(<[u8]>::to_vec).unwrap_or_default();
            locators.push(locator);
            shaders.push(DecodedShaderBytes {
                vert: stage_bytes(ShaderKind::Vertex),
                frag: stage_bytes(ShaderKind::Fragment),
                vert_instanced: stage_bytes(ShaderKind::VertexInstanced),
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

        // Capture the default shader's declared stage source paths so the
        // asset hot-reload watcher can recompile + rebuild its pipelines on a
        // `.metal` / `.hlsl` / `.glsl` save. Stages whose current-platform
        // source is the embedded GLSL fallback (or whose declaration uses a
        // non-platform-compatible extension) carry no file to watch and are
        // skipped; the inline GLSL path keeps rendering at whatever was baked
        // in. Material-referenced shaders past entry 0 reload via `cn build`.
        let world_default = &world_shaders[0];
        let mut shader_stage_source_map = super::hot_reload_sources::ShaderStageSourceMap::new();
        if crate::app::dev_flags::enabled() {
            let assets_dir = self.assets_dir();
            let mut capture = |stage_opt: Option<&StageSource>, kind: ShaderKind| {
                let Some(stage) = stage_opt else {
                    return;
                };
                let Some(raw) = stage.source_for(crate::platform::current()) else {
                    return;
                };
                let resolved = super::hot_reload_sources::resolve_runtime_source_path(
                    &raw,
                    assets_dir.as_deref(),
                );
                shader_stage_source_map.entries.push(
                    super::hot_reload_sources::ShaderStageSourceEntry {
                        kind,
                        resolved_path: resolved,
                    },
                );
            };
            capture(Some(&world_default.vertex), ShaderKind::Vertex);
            capture(Some(&world_default.fragment), ShaderKind::Fragment);
            capture(
                world_default.vertex_instanced.as_ref(),
                ShaderKind::VertexInstanced,
            );
        }

        // The shadow shader is engine-internal now (compiled from
        // `shadow.metal`), so there is no per-world shadow payload. The
        // DX / Vulkan constructors still take a shadow byte slice pending their
        // own internal-shadow migration; Metal ignores it.
        let shadow_bytes: Vec<u8> = Vec::new();

        Some(DecodedShaders {
            locators,
            source_map: shader_stage_source_map,
            shaders,
            shadow_bytes,
        })
    }

    // Publish the Resolution row's mode list (backend-enumerated, else the static
    // preset fallback), apply a persisted display-mode choice to the backend, and
    // seed the frame-rate-cap resource + the Resolution row's dynamic value label.
    // Runs after the backend is built.
    fn finalize_display_modes(&mut self, ctx: &mut PipelineContext) {
        let chosen = self.resolution;
        if let Some(backend) = self.backend.as_deref_mut() {
            let raw = backend.display_modes();
            self.display_modes = if raw.is_empty() {
                crate::gfx::display_mode::fallback_modes()
            } else {
                crate::gfx::display_mode::normalize(raw)
            };
            self.current_mode = backend.current_display_mode();
            if let Some(mode) = chosen {
                backend.set_display_mode(mode);
            }
        }
        ctx.insert_resource(crate::ecs::DisplayModes(self.display_modes.clone()));
        // The resolved frame-rate cap (world value or persisted override) for
        // the App-level pacer; the settings row's live change republishes it.
        ctx.insert_resource(crate::ecs::FrameRateCap(self.fps_cap));
        let idx =
            crate::gfx::display_mode::index_of(&self.display_modes, self.effective_resolution());
        if let Some(m) = self.display_modes.get(idx) {
            set_setting_row_label(ctx, "resolution", &m.label());
        }
    }

    // Decide cursor handling and push the post-build backend config: menu mode,
    // ambient scale, key map, the startup cursor grab (plain first-person worlds
    // only), and the device capability flags that gate the settings rows.
    fn finalize_backend_config(&mut self, ctx: &mut PipelineContext) {
        // A plain first-person world (Camera3D, no UI) captures the cursor at
        // startup. A Camera3D world that also has UI (a MainMenu's HitRegion /
        // KeyBinding) is "menu mode": capture is driven per-frame in `run_step`.
        // A UI-only world (no camera) stays free-cursor.
        let has_ui = ctx.query::<HitRegion>().next().is_some()
            || ctx
                .query::<crate::components::KeyBinding>()
                .next()
                .is_some();
        let has_camera = ctx.query::<Camera3D>().next().is_some();
        self.menu_mode = has_camera && has_ui;
        // A menu / editor driver (a `MenuOverride` is present) owns cursor capture
        // per frame, so the startup auto-grab is skipped: the editor re-runs this
        // init on every live-preview rebuild, and grabbing there would re-hide and
        // decouple the OS cursor each time, desyncing the free-cursor handoff.
        let menu_driven = ctx.resource::<crate::ecs::MenuOverride>().is_some();
        let mut device_caps = crate::gfx::backend::DeviceCapabilities::ALL;
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
            backend.set_ambient_intensity(self.ambient_intensity);
            // Push the movement key map (the persisted rebinds, or the default).
            // The backend decodes physical keys through it; idempotent with its
            // own default seed when there is no override.
            backend.set_keymap(&self.keymap);
            if has_camera && !has_ui && !menu_driven {
                backend.capture_cursor();
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

    // Hand the resolved settings snapshot to SettingsSystem, which owns the live
    // SettingCommand / SceneCommand drain against the backend from here. This
    // system resolves the values (world config + persisted overrides + device
    // capabilities) at init and never re-reads its copies afterward.
    fn publish_settings_state(&mut self, ctx: &mut PipelineContext) {
        ctx.insert_resource(crate::gfx::settings_system::SettingsSlot(Some(
            crate::gfx::settings_system::SettingsState {
                keymap: self.keymap,
                rebind_rows: std::mem::take(&mut self.rebind_rows),
                gamepad_map: self.gamepad_map,
                pad_rebind_rows: std::mem::take(&mut self.pad_rebind_rows),
                sliders: std::mem::take(&mut self.sliders),
                cycle_value_labels: std::mem::take(&mut self.cycle_value_labels),
                post_process: self.post_process,
                post_config: self.post_config.clone(),
                authored_post_config: self.authored_post_config.clone(),
                ambient_intensity: self.ambient_intensity,
                quality_preset: self.quality_preset,
                gpu_profile: self.gpu_profile,
                render_scale: self.render_scale,
                upscale_backend: self.upscale_backend,
                temporal_upscaling: self.temporal_upscaling,
                hdr_display: self.hdr_display,
                hdr_pq: self.hdr_pq,
                shadow_map_size: self.shadow_map_size,
                shadow_update: self.shadow_update,
                shadow_distance: self.shadow_distance,
                shadow_cascades: self.shadow_cascades,
                anisotropy: self.anisotropy,
                authored_shadow_map_size: self.authored_shadow_map_size,
                authored_shadow_update: self.authored_shadow_update,
                authored_shadow_distance: self.authored_shadow_distance,
                authored_shadow_cascades: self.authored_shadow_cascades,
                authored_anisotropy: self.authored_anisotropy,
                vsync: self.vsync,
                fps_cap: self.fps_cap,
                perf_stats: self.perf_stats,
                show_fps: self.show_fps,
                show_vram: self.show_vram,
                perf_sub_row_labels: std::mem::take(&mut self.perf_sub_row_labels),
                window_args: self.window_args.clone(),
                display_modes: std::mem::take(&mut self.display_modes),
                resolution: self.resolution,
                current_mode: self.current_mode,
                resolution_row_labels: std::mem::take(&mut self.resolution_row_labels),
                frames_in_flight: self.frames_in_flight,
                occlusion_two_pass: self.occlusion_two_pass,
                texture_cap: self.texture_cap,
                texture_budget: self.texture_budget,
                persisted_graphics: self.persisted_graphics.clone(),
                fog_built: self.fog_built,
                settings_cache: None,
                settings_writer: None,
                scene_cmd_cursor: crate::ecs::EventCursor::default(),
                setting_cmd_cursor: crate::ecs::EventCursor::default(),
                published_hud_prefs: None,
                published_disabled_inputs: None,
            },
        )));
    }

    // Read the sole EnvironmentMap (handle 0) from its resource table and capture
    // its IBL payload; extra declarations are logged and ignored. Under `cn debug`
    // (`capture_sources`) also captures the resolved HDR source path + convolution
    // sizing for the hot-reload watcher (procedural generators have no file to
    // watch). Returns (payload bytes, source), or None (self.failed set) if the
    // payload is unreadable.
    fn decode_environment_map(
        &mut self,
        ctx: &mut PipelineContext,
        capture_sources: bool,
    ) -> Option<(
        Option<Vec<u8>>,
        Option<super::hot_reload_sources::EnvironmentMapSource>,
    )> {
        let env_map_table = ctx
            .resource::<crate::resource::EnvironmentMapTable>()
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
                        self.failed = true;
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
    // colour-grading payload; extras are logged and ignored. Under `cn debug`
    // captures the resolved source path for the hot-reload watcher. Returns
    // (payload bytes, source), or None (self.failed set) if unreadable.
    fn decode_color_lut(
        &mut self,
        ctx: &mut PipelineContext,
        capture_sources: bool,
    ) -> Option<(
        Option<Vec<u8>>,
        Option<super::hot_reload_sources::ColorLutSource>,
    )> {
        let color_lut_table = ctx
            .resource::<crate::resource::ColorLutTable>()
            .cloned()
            .unwrap_or_default();
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
                    self.failed = true;
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

    // Build the shared text/sprite atlas pool: deserialise each Font's atlas +
    // metrics into `self.loaded_fonts` (its FontHandle == its dense atlas slot),
    // add the built-in fallback face when any text names no Font, then append
    // each distinct Sprite / Story-stage texture (resolved through
    // `texture_locators`) into `self.sprite_texture_slots`. An unresolved sprite
    // texture demotes to its tint (warned, not fatal). Returns the RGBA atlases +
    // the font payloads' blob indices, or None (self.failed set) on a Font decode
    // or read failure.
    fn decode_text_atlases(
        &mut self,
        ctx: &mut PipelineContext,
        texture_locators: &[crate::ecs::PayloadLocator],
    ) -> Option<TextAtlases> {
        let font_table = ctx
            .resource::<crate::resource::FontTable>()
            .cloned()
            .unwrap_or_default();
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
                        self.failed = true;
                        return None;
                    }
                },
                (None, None) => {
                    tracing::error!(
                        "GraphicsSystem: Font handle {} has no compiled payload -- did the build succeed?",
                        slot
                    );
                    self.failed = true;
                    return None;
                }
            };
            match crate::bake::font::deserialise(&bytes) {
                Ok((aw, ah, supersample, size_px, rgba, metrics)) => {
                    let metrics_map: text::FontMetrics =
                        metrics.into_iter().map(|m| (m.char_code, m)).collect();
                    let size_px = size_px as f32;
                    self.loaded_fonts.insert(
                        crate::ecs::FontHandle(slot as u32),
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
                    self.failed = true;
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
            let handle = crate::ecs::FontHandle(slot as u32);
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
        let sprite_texture_ids: Vec<crate::ecs::TextureHandle> = {
            let mut ids: Vec<crate::ecs::TextureHandle> = ctx
                .query::<crate::components::Sprite>()
                .filter_map(|s| s.texture)
                .collect();
            for story in ctx.query::<crate::components::Story>() {
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
                Ok(bytes) => match crate::bake::texture::deserialise(bytes)
                    .and_then(|image| image.into_rgba8())
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
        if let Some(overlay) = ctx.resource::<crate::ecs::OverlayImages>() {
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

    pub(super) fn run_init(&mut self, ctx: &mut PipelineContext) {
        let ResolvedRenderConfig {
            post,
            quality_ceiling,
            streaming_config,
            world_ambient_intensity,
        } = self.init_render_settings(ctx);
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

        // Snapshot each ProceduralMesh before `load_mesh_geometry` drains
        // them, so the world.jsonl hot-reload pass can diff a freshly parsed
        // on-disk entry against the init state and re-run the generator when
        // they differ. A `None` here (hot-reload off) keeps the captured set
        // empty so the reload pass has nothing to inspect on `cn run`. Names
        // come from the interner so the reload log can read "regenerated
        // 'box_mesh'" instead of an opaque id.
        let proc_mesh_args_snapshot: std::collections::HashMap<
            AssetId,
            (String, crate::components::ProceduralMesh),
        > = if crate::app::dev_flags::enabled() {
            ctx.query::<crate::components::ProceduralMesh>()
                .filter_map(|pm| {
                    let name = crate::ecs::asset_id::name_of(pm.asset_id)?;
                    Some((pm.asset_id, (name, pm.clone())))
                })
                .collect()
        } else {
            std::collections::HashMap::new()
        };

        // Mesh sources owned by a scene other than the start scene skip their
        // payload decode: draw records use the blob's baked bounds, and the
        // mesh streamer decodes the payload when the owning scene pins.
        let deferred_mesh_sources =
            super::streaming::deferred_mesh_sources(ctx, streaming_config.is_some());
        let (
            mesh_geometry,
            mesh_sources,
            always_resident_meshes,
            component_mesh_handles,
            deferred_mesh_seeds,
        ) = match draw_list::load_mesh_geometry(ctx, &deferred_mesh_sources, blob_disk_backed) {
            Some(m) => m,
            None => {
                self.failed = true;
                return;
            }
        };

        let (skinned_geometry, skinned_blob_indices) = match self.decode_skinned_geometry(ctx) {
            Some(decoded) => decoded,
            None => return,
        };

        // drain Model components into a name-keyed map for Prop lookup
        let models = ctx.drain::<Model>();
        let model_map: std::collections::HashMap<AssetId, Vec<crate::components::SubMeshRef>> =
            models.into_iter().map(|m| (m.asset_id, m.meshes)).collect();

        // decode Room payloads before shaders/textures are read; all payloads
        // live in the same blob and must be consumed before it is released
        let (room_geometry, room_blob_indices) = match draw_list::load_room_geometry(ctx) {
            Some(r) => r,
            None => {
                self.failed = true;
                return;
            }
        };

        let DecodedShaders {
            locators: shader_locators,
            source_map: shader_stage_source_map,
            shaders: decoded_shaders,
            shadow_bytes,
        } = match self.decode_shaders(ctx, streaming_config.is_some()) {
            Some(decoded) => decoded,
            None => return,
        };
        // The world default program (ShaderHandle 0): skinned upload and the
        // DX / Vulkan single-pipeline paths consume these directly.
        let vert_bytes = decoded_shaders[0].vert.clone();
        let frag_bytes = decoded_shaders[0].frag.clone();

        // Read the shared texture pool + the material table into the maps the
        // draw list resolves against. `capture_sources` (cn debug) also gathers
        // the file-backed source maps the hot-reload watcher consumes.
        let capture_sources = crate::app::dev_flags::enabled();
        let TextureTableDecode {
            locators: texture_locators,
            source_map: asset_source_map,
            name_to_slot: texture_name_to_slot,
            count: texture_count,
        } = match self.decode_texture_table(ctx, capture_sources) {
            Some(decoded) => decoded,
            None => return,
        };
        let material_map = match self.build_material_map(ctx, texture_count) {
            Some(map) => map,
            None => return,
        };

        // Build skinned draw objects, the shared skinned vertex/index buffers,
        // and bind-pose skeletons from the decoded SkinnedMesh geometry. Runs
        // after the material map so SkinnedMesh material references resolve.
        let SkinnedMeshAssembly {
            vertices: skinned_vertices,
            indices: skinned_indices,
            draw_objects: mut skinned_draw_objects,
            skeletons: skinned_skeletons,
            pool_reservations: skinned_pool_reservations,
            morphs: mut skinned_morphs,
            source_map: skinned_mesh_source_map,
        } = match self.assemble_skinned_meshes(
            &skinned_geometry,
            &material_map,
            texture_count,
            capture_sources,
        ) {
            Some(assembly) => assembly,
            None => return,
        };

        let mut texture_data: Vec<crate::bake::texture::TextureImage> = Vec::new();
        // Raw compiled texture payloads, kept past blob release so the
        // asset-streaming subsystem can re-decode them off the main thread.
        // Left empty when the blobs are disk-backed: the streamer then re-reads
        // each payload from its blob file instead of holding a RAM copy.
        let mut texture_payloads: Vec<Vec<u8>> = Vec::new();
        // Slots owned by a scene other than the start scene enter the pool as
        // 1x1 placeholders instead of decoding: they are blocked from
        // streaming until their scene pins, at which point the streamer
        // decodes them off the main thread.
        let deferred_slots = super::streaming::deferred_texture_slots(
            ctx,
            streaming_config.is_some(),
            texture_locators.len(),
        );
        for (slot, locator) in texture_locators.iter().enumerate() {
            if deferred_slots.contains(&slot) {
                texture_data.push(crate::bake::texture::TextureImage::rgba8(
                    1,
                    1,
                    vec![0, 0, 0, 255],
                ));
                if !blob_disk_backed {
                    match ctx.read_payload(locator) {
                        Ok(b) => texture_payloads.push(b.to_vec()),
                        Err(e) => {
                            tracing::error!(
                                "GraphicsSystem: failed to read texture payload: {:?}",
                                e
                            );
                            self.failed = true;
                            return;
                        }
                    }
                }
                continue;
            }
            let tex_bytes = match ctx.read_payload(locator) {
                Ok(b) => b.to_vec(),
                Err(e) => {
                    tracing::error!("GraphicsSystem: failed to read texture payload: {:?}", e);
                    self.failed = true;
                    return;
                }
            };
            match crate::bake::texture::deserialise(&tex_bytes) {
                Ok(t) => texture_data.push(t),
                Err(e) => {
                    tracing::error!("GraphicsSystem: malformed texture payload: {}", e);
                    self.failed = true;
                    return;
                }
            }
            if !blob_disk_backed {
                texture_payloads.push(tex_bytes);
            }
        }
        if !deferred_slots.is_empty() {
            tracing::info!(
                "GraphicsSystem: deferred {} scene-owned texture payload(s) past init",
                deferred_slots.len()
            );
        }

        // Read the sole EnvironmentMap + ColorLut payloads, then build the shared
        // text/sprite atlas pool.
        let (env_map_bytes, environment_map_source) =
            match self.decode_environment_map(ctx, capture_sources) {
                Some(decoded) => decoded,
                None => return,
            };
        let (color_lut_bytes, color_lut_source) = match self.decode_color_lut(ctx, capture_sources)
        {
            Some(decoded) => decoded,
            None => return,
        };
        let TextAtlases {
            atlases: text_atlas_data,
            font_blob_indices,
        } = match self.decode_text_atlases(ctx, &texture_locators) {
            Some(decoded) => decoded,
            None => return,
        };

        // Indirect-ambient multiplier from PostProcessConfig, folded into the
        // shared LightUniforms so every backend's main pass scales its IBL /
        // flat-fallback ambient by it. 1.0 (the default) is a no-op.
        let ambient_intensity = world_ambient_intensity;
        // Lights are read, not drained: the GPU-side light data stays static
        // (built once here), but the components keep their entities so editor
        // tooling can address the authored lights by name.
        let dir_lights: Vec<DirectionalLight> = ctx.query::<DirectionalLight>().cloned().collect();
        let pt_lights: Vec<PointLight> = ctx.query::<PointLight>().cloned().collect();
        let spot_lights: Vec<SpotLight> = ctx.query::<SpotLight>().cloned().collect();
        let rect_lights: Vec<RectAreaLight> = ctx.query::<RectAreaLight>().cloned().collect();
        let light_data = lights::build_light_data(&pt_lights, &spot_lights, &rect_lights);
        let light_uniforms = lights::build_light_uniforms(
            dir_lights,
            pt_lights,
            &light_data.lights,
            ambient_intensity,
        );

        // AudioSystem inits after GraphicsSystem and reads audio-clip payloads
        // from the `AudioClipTable`, so any blob a clip lives in must survive this
        // release sweep.
        let audio_blobs = ctx
            .resource::<crate::resource::AudioClipTable>()
            .map(|table| table.blob_indices())
            .unwrap_or_default();
        // SdfVolume payloads are drained later in this same init pass (see
        // the `sdf_volumes` block below), so the release sweep here must
        // also leave their blobs resident. Without this gate, any world
        // whose SDF shader bytes happen to land alone in a blob shows
        // "failed to read fragment shader payload: FileIo; skipping" at
        // runtime and the SDF surface never draws.
        let sdf_blobs = crate::components::sdf_volume::sdf_volume_blob_indices(ctx);
        // PhysicsSystem inits after GraphicsSystem and reads the baked
        // heightfield collider grid from a heightfield ProceduralMesh's
        // payload, so those blobs must also survive this sweep.
        let terrain_blobs = crate::components::procedural_mesh::heightfield_blob_indices(ctx);
        let mut released = std::collections::HashSet::new();
        for idx in shader_locators
            .iter()
            .map(|l| l.blob_index)
            .chain(texture_locators.iter().map(|l| l.blob_index))
            .chain(room_blob_indices)
            .chain(font_blob_indices)
            .chain(skinned_blob_indices)
        {
            if !audio_blobs.contains(&idx)
                && !sdf_blobs.contains(&idx)
                && !terrain_blobs.contains(&idx)
                && released.insert(idx)
            {
                ctx.release_blob(idx);
            }
        }

        // InstancedProp components are drained because every instance becomes a
        // baked DrawObject; there is no per-frame update path yet. Drain before
        // taking Prop references because drain shifts the underlying Vec.
        let instanced_props = ctx.drain::<crate::components::InstancedProp>();

        // Entities to render, in Prop-column order, so each gets a RenderHandle +
        // GlobalTransform attached below. Enumerated through the PropInstance
        // marker the decomposition leaves on each prop, in Prop order; the Prop
        // column itself was drained by the decomposition pass at load. The
        // Transform column is not the enumeration: a SkyRotation pivot carries
        // one without being anything to draw.
        let prop_entities: Vec<crate::ecs::Entity> = ctx
            .query_with_entity::<crate::components::PropInstance>()
            .map(|(entity, _)| entity)
            .collect();

        // Build the draw-list inputs from each entity's per-instance components:
        // renderer fields from MeshRenderer/ModelRenderer, world matrices from
        // Transform/Parent. `items` / `world_mats` are column-aligned with
        // `prop_entities`.
        let resolved = transform_propagation::resolve_world_matrices(ctx);
        let entity_name: std::collections::HashMap<crate::ecs::Entity, AssetId> = ctx
            .resource::<crate::ecs::decompose::EntityByName>()
            .map(|n| n.0.iter().map(|(&id, &e)| (e, id)).collect())
            .unwrap_or_default();
        let mut items = Vec::with_capacity(prop_entities.len());
        let mut world_mats = Vec::with_capacity(prop_entities.len());
        for &entity in &prop_entities {
            let asset_id = entity_name.get(&entity).copied().unwrap_or_default();
            items.push(draw_list::decomposed_renderable_item(ctx, entity, asset_id));
            world_mats.push(
                resolved
                    .get(&entity)
                    .copied()
                    .unwrap_or(draw_list::IDENTITY4),
            );
        }

        let draw_list::DrawListData {
            vertices: all_vertices,
            indices: all_indices,
            mut draw_objects,
            instanced_clusters,
            prop_draw_indices,
            mesh_handle_to_draws,
            prop_local_bounds,
        } = match draw_list::build_draw_list(draw_list::DrawListInputs {
            items: &items,
            instanced_props: &instanced_props,
            world_mats: &world_mats,
            model_map: &model_map,
            mesh_geometry: &mesh_geometry,
            room_geometry: &room_geometry,
            texture_count,
            material_map: &material_map,
            always_resident_meshes: &always_resident_meshes,
        }) {
            Some(d) => d,
            None => {
                self.failed = true;
                return;
            }
        };

        // Give each prop entity a RenderHandle (its GPU draw slots) and a
        // GlobalTransform (its init world matrix), so the per-frame push reads
        // these. prop_entities is column-aligned with prop_draw_indices and
        // world_mats; prop_draw_indices is consumed here and then dropped.
        // When a PickIndex resource is present (the editor injects one before
        // start; a shipped runtime never does), also capture each prop's pick
        // candidate so the frame step can refresh the index from the live
        // transforms.
        self.pick_candidates.clear();
        let want_pick = ctx.resource::<crate::ecs::PickIndex>().is_some();
        for (i, &entity) in prop_entities.iter().enumerate() {
            let draws: concinnity_core::memory::InlineVec<u32> = prop_draw_indices[i]
                .iter()
                .map(|&slot| slot as u32)
                .collect();
            ctx.insert(entity, crate::components::RenderHandle { draws });
            ctx.insert(entity, crate::components::GlobalTransform(world_mats[i]));
            if want_pick {
                let (local_min, local_max) = prop_local_bounds[i];
                self.pick_candidates.push(super::PickCandidate {
                    asset_id: items[i].asset_id,
                    entity,
                    local_min,
                    local_max,
                });
            }
        }

        // Asset hot-reload mesh map: cross-reference the file-backed source
        // metadata captured at drain time with the per-Mesh draw indices
        // build_draw_list just produced. A Mesh without any draws (referenced
        // by nothing) carries no entry; the watcher would still fire on the
        // .glb change but the reload helper has nothing to push to.
        let mut mesh_source_map = super::hot_reload_sources::MeshSourceMap::new();
        if capture_sources {
            for (handle, meta) in &mesh_sources {
                if let Some(draws) = mesh_handle_to_draws.get(handle) {
                    if draws.is_empty() {
                        continue;
                    }
                    mesh_source_map
                        .entries
                        .push(super::hot_reload_sources::MeshSourceEntry {
                            source: meta.source.clone(),
                            primitive_index: meta.primitive_index,
                            lod_levels: meta.lod_levels,
                            lod_distances: meta.lod_distances.clone(),
                            draw_indices: draws.clone(),
                        });
                }
            }
        }

        // Procedural-mesh hot-reload map: same cross-reference, but the
        // "source" is the JSONL `args` object captured pre-drain rather than
        // a file path. A procedural mesh that no Prop references carries no
        // draws and is omitted; a JSONL save changing its args would be
        // observable only through a future system that introspects the args
        // map directly, which we deliberately do not maintain.
        let mut procedural_mesh_source_map =
            super::hot_reload_sources::ProceduralMeshSourceMap::new();
        if capture_sources {
            for (asset_id, (name, args)) in &proc_mesh_args_snapshot {
                let Some(handle) = component_mesh_handles.get(asset_id) else {
                    continue;
                };
                if let Some(draws) = mesh_handle_to_draws.get(handle) {
                    if draws.is_empty() {
                        continue;
                    }
                    procedural_mesh_source_map.entries.push(
                        super::hot_reload_sources::ProceduralMeshSourceEntry {
                            name: name.clone(),
                            args: args.clone(),
                            draw_indices: draws.clone(),
                        },
                    );
                }
            }
        }

        // A geometry-less world (e.g. text-only) is valid: the backend is
        // initialised with empty geometry buffers and only the text path runs.

        // Per-texture-slot draw positions for the streaming scorer, captured
        // before `draw_objects` moves into the backend.
        let texture_centers = texture_stream_centers(&draw_objects, texture_data.len());

        // Per-streamed-mesh data, also captured before `draw_objects` moves
        // into the backend.
        let MeshStreamData {
            draw_indices: mesh_stream_draw_indices,
            centers: mesh_centers,
            payloads: mesh_payloads,
        } = {
            let deferred_draws: std::collections::HashSet<usize> = deferred_mesh_seeds
                .keys()
                .filter_map(|h| mesh_handle_to_draws.get(h))
                .flatten()
                .copied()
                .collect();
            mesh_stream_data(&draw_objects, &all_vertices, &all_indices, &deferred_draws)
        };

        // Mesh streaming and LOD alternates don't yet cooperate: upload_mesh
        // writes only LOD0 to its newly-allocated region, but obj.lod_alternates
        // still carries the build-time offsets for LOD1..N. Once another stream
        // upload reuses those byte ranges, active_lod() returns offsets that
        // point at unrelated geometry and the draw renders garbage / nothing
        // (the obelisks vanish past their first LOD switch_distance). Until
        // upload_mesh learns to stream every LOD, strip the alternates from
        // every streamable draw so active_lod() always returns LOD0.
        if streaming_config.is_some() && !mesh_payloads.is_empty() {
            for &draw_idx in &mesh_stream_draw_indices {
                if let Some(obj) = draw_objects.get_mut(draw_idx) {
                    obj.lod_alternates.clear();
                }
            }
        }

        // Shrinkable seed VRAM (Metal + DirectX + Vulkan). By default
        // `build_draw_list` bakes every streamed mesh into the shared
        // vertex/index buffers, sizing them for the whole streamed set, so
        // streaming reuses space but never shrinks GPU memory. When the residency
        // cap is smaller than the streamed set, compact the resident geometry and
        // reserve a smaller seed headroom -- sized to the cap-many largest meshes
        // -- for the streamed meshes, which are placed into it on upload
        // (tolerating a transient alloc miss while freed regions await their
        // retire frame). Done before `init_backend` so the GPU buffers are born
        // small and the RT acceleration structure (built over resident draws
        // inside init) sees the final offsets.
        let mut all_vertices = all_vertices;
        let mut all_indices = all_indices;
        let mut instanced_clusters = instanced_clusters;
        let mesh_seed_region: Option<crate::gfx::mesh_seed::MeshSeedRegion> = match streaming_config
            .as_ref()
        {
            Some(cfg) if !mesh_payloads.is_empty() => {
                // A deferred mesh's payload copy is empty (its decode
                // was skipped), so its seed contribution comes from
                // the baked counts instead.
                let draw_to_handle: std::collections::HashMap<usize, usize> = mesh_handle_to_draws
                    .iter()
                    .flat_map(|(h, draws)| draws.iter().map(move |&d| (d, *h)))
                    .collect();
                let sizes: Vec<(u64, u64)> = mesh_payloads
                    .iter()
                    .zip(&mesh_stream_draw_indices)
                    .map(|(m, draw_idx)| {
                        if !m.vertices.is_empty() {
                            return (
                                (m.vertices.len() * std::mem::size_of::<Vertex>()) as u64,
                                (m.indices.len() * std::mem::size_of::<u32>()) as u64,
                            );
                        }
                        draw_to_handle
                            .get(draw_idx)
                            .and_then(|h| deferred_mesh_sources.counts.get(&(*h as u32)))
                            .map(|&(vc, ic)| {
                                (
                                    vc as u64 * std::mem::size_of::<Vertex>() as u64,
                                    ic as u64 * std::mem::size_of::<u32>() as u64,
                                )
                            })
                            .unwrap_or((0, 0))
                    })
                    .collect();
                // Deferred meshes have no baked region for the
                // full-set evict path to free; force the compaction
                // path with a whole-set headroom when the cap alone
                // would not shrink.
                let planned = crate::gfx::mesh_seed::plan_seed_bytes(&sizes, cfg.mesh_cap())
                    .or_else(|| {
                        (!deferred_mesh_seeds.is_empty()).then(|| {
                            (
                                sizes.iter().map(|s| s.0).sum(),
                                sizes.iter().map(|s| s.1).sum(),
                            )
                        })
                    });
                match planned {
                    Some((seed_vtx, seed_idx)) => {
                        let mut streamed = vec![false; draw_objects.len()];
                        for &idx in &mesh_stream_draw_indices {
                            if let Some(s) = streamed.get_mut(idx) {
                                *s = true;
                            }
                        }
                        let region = crate::gfx::mesh_seed::compact_for_streaming(
                            &mut all_vertices,
                            &mut all_indices,
                            &mut draw_objects,
                            &mut instanced_clusters,
                            &streamed,
                            seed_vtx,
                            seed_idx,
                        );
                        tracing::info!(
                            "GraphicsSystem: shrinkable seed VRAM -- {} streamed mesh(es), cap {}, seed headroom {} KiB vtx + {} KiB idx",
                            mesh_stream_draw_indices.len(),
                            cfg.mesh_cap(),
                            seed_vtx / 1024,
                            seed_idx / 1024,
                        );
                        Some(region)
                    }
                    None => None,
                }
            }
            _ => None,
        };

        let draw_object_count = draw_objects.len();
        let cluster_count = instanced_clusters.len();
        let total_instances: usize = instanced_clusters.iter().map(|c| c.instances.len()).sum();

        // Build projected-decal records from the world's `Decal` components.
        // Resolved here (rather than per-frame) because the decal set is built
        // at init and never grows: each record carries the resolved texture
        // slot and pre-inverted model matrix the fragment shader needs. The
        // Decal components are drained because the runtime keeps no per-frame
        // update path for them.
        let decal_records = {
            let decals: Vec<Decal> = ctx.drain::<Decal>();
            let refs: Vec<&Decal> = decals.iter().collect();
            crate::gfx::decal::build_decal_records(&refs, texture_count)
        };
        let decal_count = decal_records.len();

        // Build particle-emitter records from the world's `ParticleEmitter`
        // components. Same drain-at-init pattern as decals: each record carries
        // the clamped emitter tunables and the resolved texture slot. The
        // backend allocates one persistent GPU pool per record at init.
        let particle_records = {
            let emitters: Vec<ParticleEmitter> = ctx.drain::<ParticleEmitter>();
            let refs: Vec<&ParticleEmitter> = emitters.iter().collect();
            crate::gfx::particles::build_particle_records(&refs, texture_count)
        };
        let particle_count = particle_records.len();

        // Drain transparent water surfaces. Every backend builds a tessellated
        // grid + per-surface uniforms per record at init and draws them in the
        // shared transparent pass, alongside the glass panes below
        // (`metal/water.rs`, `directx/water.rs`, `vulkan/water.rs`).
        let water_surfaces: Vec<WaterSurface> = ctx.drain::<WaterSurface>();

        // Drain translucent glass panels. Every backend builds a world-space
        // quad + per-panel uniforms per record at init and draws them in the
        // shared transparent pass (`metal/glass.rs`, `directx/glass.rs`,
        // `vulkan/glass.rs`).
        let glass_panels: Vec<GlassPanel> = ctx.drain::<GlassPanel>();

        // Drain raymarched SDF volumes and pull the compiled-payload
        // fragment-shader source bytes for each. Each backend wraps the bytes
        // with the engine-shipped helpers + template and compiles a per-volume
        // pipeline at init. Volumes whose payload read fails are dropped with a
        // logged warning rather than failing the whole world build.
        let sdf_volumes: Vec<(SdfVolume, Vec<u8>, String)> = {
            let raw: Vec<SdfVolume> = ctx.drain::<SdfVolume>();
            let mut out = Vec::with_capacity(raw.len());
            for v in raw {
                let asset_id = v.asset_id;
                let label = crate::ecs::asset_id::name_of(asset_id)
                    .unwrap_or_else(|| format!("sdf_volume_{}", asset_id.0));
                let locator = match v.locator.as_ref() {
                    Some(l) => l.clone(),
                    None => {
                        tracing::warn!(
                            "SdfVolume '{}': no payload locator (fragment shader \
                             never compiled); skipping",
                            label
                        );
                        continue;
                    }
                };
                match ctx.read_payload(&locator) {
                    Ok(bytes) => {
                        let owned = bytes.to_vec();
                        out.push((v, owned, label));
                    }
                    Err(e) => {
                        tracing::warn!(
                            "SdfVolume '{}': failed to read fragment shader \
                             payload: {:?}; skipping",
                            label,
                            e
                        );
                    }
                }
            }
            out
        };

        // Resolve the world's `VolumetricFog`. The first declared instance
        // wins; later ones are silently dropped (one homogeneous medium is
        // all the fog pass models). `None` means the renderer skips the
        // fog pass; an asset with `enabled = false` also yields `None`.
        let fog_settings = {
            let fogs: Vec<VolumetricFog> = ctx.drain::<VolumetricFog>();
            fogs.into_iter().find(|f| f.enabled).map(|f| {
                crate::gfx::volumetric_fog::FogSettings::resolve(
                    f.color,
                    f.density,
                    f.height_falloff,
                    f.height_reference,
                    f.max_distance,
                    f.phase_g,
                    f.ambient,
                )
            })
        };
        let fog_enabled = fog_settings.is_some();
        self.fog_built = fog_enabled;
        // Seed the hot-reload dedupe state. Subsequent reload_volumetric_fog
        // calls compare resolved JSONL settings against this and only push
        // (and log) on a real change.
        self.last_fog_settings = fog_settings;

        // The DirectX / Vulkan debug layers: the CLI `--validation` flag if the
        // launch passed one, otherwise the build profile. Metal is unaffected
        // here: its layer is enabled by the CLI re-execing with `MTL_DEBUG_LAYER`.
        let validation = crate::app::dev_flags::resolve_validation();
        // Shader hot-reload is opted in by `cn debug` (sets the static flag
        // in `crate::app::dev_flags` before world build). Production `cn run`
        // leaves it off; the backend then never spawns the filesystem watcher
        // and shader sources stay strictly include_str!-baked.
        let hot_reload = crate::app::dev_flags::enabled();
        // Frame capture: always available under the dev loop, and armed for a
        // production run that asked for an exit screenshot.
        let capture = hot_reload || crate::app::dev_flags::capture();
        // Worst-case resident chunk count for the streaming VoxelWorld (0 for a
        // non-voxel world). Threaded into the backend so its GPU-cull buffers
        // reserve a chunk record region at init; resident chunks fold into the
        // indirect path each frame. The VoxelWorld is consumed later by
        // `setup_voxel_world_streaming`, so borrow it here.
        let n_chunk_max = voxel_world.as_ref().map_or(
            0,
            crate::gfx::graphics_system::streaming::chunk_reserve_count,
        );

        // Reflection-probe auto-seed. Computed here, before `draw_objects` moves into
        // the backend: when the world declares no `ReflectionProbe`, surface-voxelise
        // the static geometry so a watertight single-mesh interior is detected (object
        // AABBs alone would read it as a solid block). The triangle gather is
        // budget-gated, and an over-budget import falls back to coarse AABB occupancy
        // rather than to no seed at all. `None` -> the backend's own AABB auto-seed.
        //
        // The bounds the grid tiles union the reflectors with the geometry, because a
        // water surface / glass pane is not a draw object: a pool wider than every mesh
        // would otherwise sit outside the grid entirely and inherit whatever
        // `probe_set_specular`'s no-coverage fallback picked. Occupancy stays
        // geometry-only -- it answers "is this capture point inside a wall", which a
        // plane does not make true.
        let auto_seed_geometry_probes = if ctx
            .query::<crate::components::ReflectionProbe>()
            .next()
            .is_some()
        {
            None
        } else {
            let occupancy: Vec<([f32; 3], [f32; 3])> = draw_objects
                .iter()
                .map(|o| (o.bb_min, o.bb_max))
                .filter(|(mn, mx)| mn.iter().chain(mx).all(|c| c.is_finite()))
                .collect();
            let reflectors = water_surfaces
                .iter()
                .map(|w| {
                    crate::gfx::reflection_probe::reflector_bounds(
                        w.centre,
                        [w.extent[0], 0.0, w.extent[1]],
                    )
                })
                .chain(glass_panels.iter().map(|g| {
                    // A pane is an oriented quad; its longest half-side bounds it on
                    // every axis whatever its normal.
                    let r = g.half_size[0].max(g.half_size[1]);
                    crate::gfx::reflection_probe::reflector_bounds(g.centre, [r, r, r])
                }));
            let tris = gather_auto_seed_triangles(&draw_objects, &all_vertices, &all_indices)
                .unwrap_or_default();
            crate::gfx::reflection_probe::fold_world_bounds(
                occupancy.iter().copied().chain(reflectors),
            )
            .map(|(mn, mx)| {
                crate::gfx::reflection_probe::auto_seed_probes_with_geometry(
                    mn, mx, &occupancy, &tris,
                )
            })
        };

        // Planar reflection plane budget: there is no world-authored value, so the
        // engine capacity is the baseline, scaled down under the quality preset /
        // GPU tier ceiling. A lower tier renders fewer full render-res mirror passes
        // (VRAM + GPU savings); reflectors past the budget take the probe cube.
        // Restart-required like anisotropy above -- the mirror targets are allocated
        // once at backend init below.
        let planar_reflection_planes = quality_ceiling.planar_reflection_planes as usize;

        // Assemble the backend construction inputs, derive the world's render
        // requirements from them (a world with no 3D content drops every
        // scene-scoped feature before any backend resource is sized), and
        // hand the result to the compile-time-selected backend.
        use crate::gfx::backend_init::{
            BackendInit, MediaPayloads, SceneData, ShaderBytes, ShadowParams, WorldFx,
        };
        let mut backend_init = BackendInit {
            window: &self.window_args,
            validation,
            frames_in_flight: self.frames_in_flight,
            vsync: self.vsync,
            clear_color: self.clear_color,
            hot_reload,
            capture,
            scene: SceneData {
                vertices: &all_vertices,
                indices: &all_indices,
                draw_objects,
                instanced_clusters,
                // Skinned draw-object count, to size the backend's GPU-cull
                // buffers for the merged total at init. The skinned geometry is
                // uploaded later by `upload_skinned` (which consumes
                // `skinned_draw_objects`).
                n_skinned: skinned_draw_objects.len(),
                n_chunk_max,
            },
            // One entry per world Shader, indexed by ShaderHandle value;
            // entry 0 is the world default program.
            shaders: decoded_shaders
                .iter()
                .map(|s| ShaderBytes {
                    vert: &s.vert,
                    frag: &s.frag,
                    shadow: &shadow_bytes,
                    vert_instanced: &s.vert_instanced,
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
                map_size: self.shadow_map_size,
                update: self.shadow_update,
                distance: self.shadow_distance,
                cascades: self.shadow_cascades,
            },
            anisotropy: self.anisotropy,
            planar_planes: planar_reflection_planes,
            post,
            fx: WorldFx {
                decals: decal_records,
                particles: particle_records,
                fog: fog_settings,
                water_surfaces,
                glass_panels,
                sdf_volumes,
            },
            requirements: Default::default(),
        };
        backend_init.resolve_requirements();
        // Live editor swap: the rebuilt world may carry a render backend
        // transplanted out of the pre-edit world (a `PendingBackend` resource,
        // published by the `cn editor` live SAVE). Reuse it instead of building a
        // new one -- so the edit applies without recreating the OS window or
        // re-initialising the GPU device -- but only when the backend can hot-swap
        // AND the swapchain-level config (pixel format / frames-in-flight / EDR)
        // is unchanged. An incapable backend (DirectX / Vulkan, whose default
        // `hot_swap_config` is `None`) or a swapchain change routes to a full
        // rebuild instead: the transplanted backend is idled + dropped and a new
        // window is created (a rare one-frame flash). Deciding via `hot_swap_config`
        // up front leaves `backend_init` intact for that rebuild path.
        let reuse_backend = match ctx
            .resources
            .remove::<crate::ecs::PendingBackend>()
            .map(|p| p.0)
        {
            Some(backend) if backend.hot_swap_config() == Some(backend_init.swapchain_config()) => {
                Some(backend)
            }
            Some(backend) => {
                backend.wait_idle();
                None
            }
            None => None,
        };
        // Tests inject a mock backend factory through `test_hooks`; production
        // always routes to the compile-time-selected real backend. Inline (not
        // a method) because `backend_init` still borrows `self.window_args`.
        #[cfg(test)]
        let built = match reuse_backend {
            Some(mut backend) => match backend.reload_world(backend_init) {
                Ok(()) => {
                    tracing::info!(
                        "GraphicsSystem: reused live backend (world reloaded in place, window kept)"
                    );
                    Some(backend)
                }
                Err(e) => {
                    tracing::error!("GraphicsSystem: reload_world failed: {e}");
                    None
                }
            },
            None => match self.test_hooks.as_mut() {
                Some(hooks) => (hooks.backend_factory)(backend_init),
                // A test builds no real device: one that forgets its hooks
                // fails its own assertions instead of opening a window.
                None => None,
            },
        };
        #[cfg(not(test))]
        let built = match reuse_backend {
            Some(mut backend) => match backend.reload_world(backend_init) {
                Ok(()) => {
                    tracing::info!(
                        "GraphicsSystem: reused live backend (world reloaded in place, window kept)"
                    );
                    Some(backend)
                }
                Err(e) => {
                    tracing::error!("GraphicsSystem: reload_world failed: {e}");
                    None
                }
            },
            None => crate::device::init_backend(backend_init),
        };
        self.backend = built;

        if self.backend.is_none() {
            self.failed = true;
            return;
        }

        // Apply a persisted or authored non-windowed window mode at startup. The
        // window is always created as a standard titled window, so a Borderless
        // or Fullscreen choice (set in the settings menu and persisted across
        // launches) has to be applied here; otherwise the app would always start
        // windowed regardless of the saved mode. No-op for Windowed and in
        // embedded mode (the backend owns no window there).
        if self.window_args.mode != crate::components::WindowMode::Windowed
            && let Some(backend) = self.backend.as_deref_mut()
        {
            backend.set_window_mode(self.window_args.mode);
        }

        // Publish the Resolution row's mode list + frame-rate cap now the backend
        // can enumerate the display's modes.
        self.finalize_display_modes(ctx);

        // Reflection probes: hand the backend the declared `ReflectionProbe`
        // placements (Metal bakes a cube per probe; an empty list auto-seeds from
        // the scene bounds). Pushed once here, after construction; DX/VK no-op.
        // Read, not drained, for the same reason as the lights above: the
        // placements are static once pushed, but the components keep their
        // entities so editor tooling can address the authored probes by name.
        if let Some(backend) = self.backend.as_deref_mut() {
            let declared: Vec<crate::gfx::reflection_probe::ProbePlacement> = ctx
                .query::<crate::components::ReflectionProbe>()
                .map(|p| {
                    crate::gfx::reflection_probe::ProbePlacement::from_center_extents(
                        p.position,
                        p.half_extents,
                    )
                })
                .collect();
            // Declared probes win; otherwise the geometry-aware auto-seed (when the scene
            // was small enough to gather); otherwise an empty list, which lets the backend
            // run its own coarse-AABB auto-seed (the unchanged path for heavy imports).
            let placements = if !declared.is_empty() {
                declared
            } else {
                auto_seed_geometry_probes.unwrap_or_default()
            };
            backend.set_reflection_probes(&placements);
        }

        // World.jsonl path for the Prop transform reload pass. The dev host
        // (`cn debug` / `cn editor`) resolves the world path -- world.jsonl
        // discovery is authoring I/O in `concinnity-cook`, which the runtime does
        // not link -- and hands it in via `dev_flags`. Embedded preview / WS-
        // driven runs leave it None and the file watcher has no `.jsonl` to
        // subscribe to.
        let world_jsonl_path: Option<String> = if capture_sources {
            crate::app::dev_flags::world_jsonl_path()
        } else {
            None
        };

        // Asset hot-reload state. Built only when `cn debug` opted in
        // (`capture_sources`) and the world declared at least one file-backed
        // asset (texture, ColorLut, EnvironmentMap, Mesh, SkinnedMesh, or
        // world.jsonl). The constructor spawns a `notify` watcher over the
        // parent directories of every captured source path; `step` polls the
        // shared atomic at frame start.
        if capture_sources
            && (!asset_source_map.is_empty()
                || color_lut_source.is_some()
                || environment_map_source.is_some()
                || !mesh_source_map.is_empty()
                || !skinned_mesh_source_map.is_empty()
                || !procedural_mesh_source_map.is_empty()
                || !shader_stage_source_map.is_empty()
                || world_jsonl_path.is_some())
        {
            tracing::info!(
                "asset hot-reload: captured {} file-backed texture source(s), {} \
                 ColorLut source(s), {} EnvironmentMap source(s), {} Mesh \
                 source(s), {} SkinnedMesh source(s), {} ProceduralMesh source(s), \
                 {} shader stage source(s), and world.jsonl path = {:?}",
                asset_source_map.len(),
                color_lut_source.as_ref().map(|_| 1).unwrap_or(0),
                environment_map_source.as_ref().map(|_| 1).unwrap_or(0),
                mesh_source_map.len(),
                skinned_mesh_source_map.len(),
                procedural_mesh_source_map.len(),
                shader_stage_source_map.len(),
                world_jsonl_path
            );
            self.pending_hot_reload_sources = Some(super::hot_reload_sources::HotReloadSources {
                map: asset_source_map,
                color_lut: color_lut_source,
                environment_map: environment_map_source,
                meshes: mesh_source_map,
                skinned_meshes: skinned_mesh_source_map,
                procedural_meshes: procedural_mesh_source_map,
                shader_stages: shader_stage_source_map,
                world_jsonl_path,
            });
            // The texture-name -> slot map for runtime decal / emitter spawn
            // (`cn debug`), which resolves a Texture asset name to its live pool
            // slot. Captured only when hot-reload is on, so a `cn run` skips the
            // clone cost.
            self.world_reload = Some(super::WorldReloadState {
                texture_name_to_slot: texture_name_to_slot.clone(),
            });
        }

        // Upload skinned geometry to the backend and publish one SkeletonPose
        // per skinned mesh for AnimationSystem to drive. The poses are published
        // regardless of backend so the system graph is identical.
        if !skinned_skeletons.is_empty() {
            if let Some(backend) = self.backend.as_deref_mut() {
                // Metal uses `vert_bytes` + `frag_bytes` and sources the shadow
                // shader internally; `shadow_bytes` is empty (engine-internal
                // shadow). DX/VK compile their vertex/shadow paths inline.
                if let Err(e) = backend.upload_skinned(
                    &skinned_vertices,
                    &skinned_indices,
                    std::mem::take(&mut skinned_draw_objects),
                    &vert_bytes,
                    &frag_bytes,
                    &shadow_bytes,
                ) {
                    tracing::error!("GraphicsSystem: skinned geometry upload failed: {}", e);
                    self.failed = true;
                    return;
                }
                if skinned_morphs.iter().any(|m| m.is_some()) {
                    backend.upload_skinned_morphs(std::mem::take(&mut skinned_morphs));
                }
                // The hidden copies reserved above seed the engine-side
                // skinned instance pool (`RenderSlots`), published below.
            }
            let skinned_count = skinned_skeletons.len();
            let shapes = super::character_shape::collect(ctx);
            let want_pick = ctx.resource::<crate::ecs::PickIndex>().is_some();
            for SkinnedSkeletonEntry {
                handle,
                name_id,
                template_index,
                skeleton,
                morph_names,
                model,
                capsule,
                transform,
                local_bounds,
            } in skinned_skeletons
            {
                let layers = shapes
                    .get(&handle)
                    .map(|shape| super::character_shape::resolve(shape, &skeleton, &morph_names));
                let capsule = capsule.map(|c| match &layers {
                    Some(l) => {
                        super::character_shape::proportioned_capsule(&c, &skeleton, &l.proportions)
                    }
                    None => (c.half_height, c.radius),
                });
                let entity = ctx.components.spawn();
                ctx.insert(
                    entity,
                    super::character_shape::seed_pose(handle, template_index, skeleton, layers),
                );
                // Under the editor the template is pickable and movable like a
                // prop: its Transform drives the per-frame skinned model push
                // and its bounds join the pick index.
                if want_pick {
                    ctx.insert(entity, transform);
                    ctx.insert(entity, crate::components::GlobalTransform(model));
                    self.pick_candidates.push(super::PickCandidate {
                        asset_id: name_id,
                        entity,
                        local_min: local_bounds.0,
                        local_max: local_bounds.1,
                    });
                }
                // Register the template under its mesh name so a runtime
                // SpawnRequest can resolve it to this entity, the same way the
                // static spawn path resolves a named placement. The spawn then
                // clones this template's skeleton + pose into a pooled slot.
                if let Some(by_name) = ctx.resource_mut::<crate::ecs::decompose::EntityByName>() {
                    by_name.0.insert(name_id, entity);
                }
                // A mesh with a capsule gets a character rig: PhysicsSystem
                // (init runs later this tick) creates the kinematic capsule
                // from it, and the render transform follows it each frame.
                if let Some((half_height, radius)) = capsule {
                    ctx.push(crate::components::CharacterRig::new(
                        handle,
                        template_index,
                        model,
                        half_height.max(0.05),
                        radius.max(0.05),
                    ));
                }
            }
            tracing::info!("GraphicsSystem: {} skinned mesh(es) ready", skinned_count);
        }

        self.setup_texture_streaming(
            streaming_config.clone(),
            texture_payloads,
            &texture_locators,
            blob_disk_backed,
            texture_centers,
        );
        // Per-stream-id payload refs for the deferred meshes, so the worker
        // can decode them from the blob payload when their scene pins.
        let deferred_stream_payloads: std::collections::HashMap<
            usize,
            crate::gfx::streaming::mesh::DeferredMeshPayload,
        > = if deferred_mesh_seeds.is_empty() {
            Default::default()
        } else {
            use crate::gfx::streaming::mesh::DeferredMeshPayload;
            let draw_to_handle: std::collections::HashMap<usize, usize> = mesh_handle_to_draws
                .iter()
                .flat_map(|(h, draws)| draws.iter().map(move |&d| (d, *h)))
                .collect();
            let mut map = std::collections::HashMap::new();
            for (stream_id, draw_idx) in mesh_stream_draw_indices.iter().enumerate() {
                let Some(seed) = draw_to_handle
                    .get(draw_idx)
                    .and_then(|h| deferred_mesh_seeds.get(h))
                else {
                    continue;
                };
                let payload = match &seed.bytes {
                    Some(bytes) => DeferredMeshPayload::Bytes(bytes.clone()),
                    None => {
                        let Some(path) = crate::blob::blob_path(seed.locator.blob_index) else {
                            tracing::warn!(
                                "GraphicsSystem: deferred mesh blob {} has no layout to read from",
                                seed.locator.blob_index
                            );
                            continue;
                        };
                        match crate::blob::payload_section_start(&path) {
                            Ok(start) => DeferredMeshPayload::Disk {
                                path,
                                offset: start + seed.locator.offset,
                                len: seed.locator.len,
                            },
                            Err(e) => {
                                tracing::warn!(
                                    "GraphicsSystem: deferred mesh blob {} unreadable: {:?}",
                                    seed.locator.blob_index,
                                    e
                                );
                                continue;
                            }
                        }
                    }
                };
                map.insert(stream_id, payload);
            }
            tracing::info!(
                "GraphicsSystem: deferred {} scene-owned mesh payload(s) past init",
                map.len()
            );
            map
        };
        self.setup_mesh_streaming(
            streaming_config,
            super::streaming::MeshStreamSetup {
                payloads: mesh_payloads,
                centers: mesh_centers,
                draw_indices: mesh_stream_draw_indices,
                disk_backed: blob_disk_backed,
                seed_region: mesh_seed_region,
                deferred_payloads: deferred_stream_payloads,
            },
        );
        self.setup_voxel_world_streaming(voxel_world, &block_types, &material_map);

        self.finalize_backend_config(ctx);

        self.setup_scene_flow(ctx);

        // Hand the overlay build inputs assembled above (font atlases, sprite
        // slots, HUD chip ids, clip bands) to OverlaySystem, which shapes the
        // draw list from them each frame before this system submits it.
        // Seed the frame extraction's viewport from the live backend;
        // FrameInput refreshes it once InputSystem starts publishing.
        self.viewport = self
            .backend
            .as_ref()
            .map(|b| b.logical_size())
            .unwrap_or((0.0, 0.0));
        ctx.insert_resource(crate::gfx::overlay::OverlayAssets {
            fonts: std::mem::take(&mut self.loaded_fonts),
            sprite_texture_slots: std::mem::take(&mut self.sprite_texture_slots),
            debug_hud_chips: std::mem::take(&mut self.debug_hud_chips),
            stat_hud_chips: std::mem::take(&mut self.stat_hud_chips),
            clip_rects: std::mem::take(&mut self.clip_rects),
            initial_viewport: self.viewport,
        });

        self.publish_settings_state(ctx);

        // Hand the streaming pools built above to StreamingSystem: it drives
        // them each frame (against the parked backend) and publishes the
        // camera-relative view GraphicsSystem draws with. `frame_count` starts
        // at 0 in lockstep with this system's own frame clock (both tick once
        // per world step), so eviction retire-frames match the draw's frame.
        // Capture each pool's derived byte budget as the back-off valve's
        // baseline before the streamers move into the parked state, so stage 2
        // can reduce it and the release can restore it exactly.
        let texture_baseline_budget = self.texture_streamer.as_ref().and_then(|s| s.byte_budget());
        let mesh_baseline_budget = self.mesh_streamer.as_ref().and_then(|s| s.byte_budget());
        let chunk_baseline_budget = self
            .chunk_stream
            .as_ref()
            .and_then(|cs| cs.streamer.byte_budget());
        let scene_residency = self.build_scene_residency(ctx);
        ctx.insert_resource(crate::gfx::streaming_system::StreamingState {
            texture_streamer: self.texture_streamer.take(),
            mesh_streamer: self.mesh_streamer.take(),
            mesh_stream_draw_indices: std::mem::take(&mut self.mesh_stream_draw_indices),
            chunk_stream: self.chunk_stream.take(),
            shader_warmup: self.shader_warmup.take(),
            scene_residency,
            frame_count: 0,
            frames_in_flight: self.frames_in_flight,
            texture_baseline_budget,
            mesh_baseline_budget,
            chunk_baseline_budget,
            pressure_stage: crate::gfx::streaming_system::pressure::StreamPressureStage::None,
            pressure_factor: 1.0,
            last_sampled_rss: None,
            drift: Default::default(),
            last_drift_verdict: None,
            heartbeats: Default::default(),
        });

        // The recording surfaces the render-block systems take each tick: the
        // op queue backend effects accumulate into, and the slot-allocation
        // authority (draw-slot free list seeded with the build-time draw
        // count; skinned instance pool seeded with the pre-reserved copies).
        ctx.insert_resource(crate::ecs::ActiveRenderQueues(Some(
            crate::ecs::RenderQueues {
                ops: Default::default(),
                slots: crate::gfx::render_slots::RenderSlots::new(
                    draw_object_count,
                    self.caps.reuses_build_slots,
                    &skinned_pool_reservations,
                ),
            },
        )));

        // Init-time wiring is done: park the backend in the world's shared
        // slot, where each per-step user (this system's frame encode,
        // InputSystem's poll) takes and returns it.
        ctx.insert_resource(crate::ecs::ActiveRenderBackend(self.backend.take()));

        let start = Instant::now();
        self.start_time = Some(start);
        // Hand the scene flow to the shared slot SettingsSystem jumps and this
        // system ticks. `epoch` shares this system's start clock so a jump's
        // fade timing matches the render clock.
        ctx.insert_resource(crate::ecs::ActiveSceneFlow {
            flow: self.scene_flow.take(),
            epoch: start,
        });
        tracing::info!(
            "GraphicsSystem: ready ({}x{} \"{}\", {} frames in flight, {} draw objects, {} instanced clusters ({} instances total), {} decals, {} particle emitter(s), fog={})",
            self.window_args.width,
            self.window_args.height,
            self.window_args.title,
            self.frames_in_flight,
            draw_object_count,
            cluster_count,
            total_instances,
            decal_count,
            particle_count,
            if fog_enabled { "on" } else { "off" },
        );
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
            let rest = r.action.strip_prefix("setting:")?;
            let key = rest.split(':').next()?;
            Some((key.to_string(), r.label?))
        })
        .collect();

    for (key, label_id) in rows {
        let (Some(opts), Some(idx)) = (crate::gfx::settings::options(&key), current_index(&key))
        else {
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
        let row_key = r.action.strip_prefix("setting:")?.split(':').next()?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::render_types::{DrawObject, MaterialUniforms, NO_NORMAL_MAP_SLOT};

    // A draw over `[vertex_offset (bytes), +vertex_count]` / `[index_offset,
    // +index_count]` sampling `texture_slot` (+ `normal_map_slot`). A non-cullable
    // draw carries the NaN sentinel AABB, matching the skybox / dynamic path.
    fn draw(
        vertex_offset: usize,
        vertex_count: usize,
        index_offset: usize,
        index_count: usize,
        texture_slot: usize,
        normal_map_slot: usize,
        cullable: bool,
    ) -> DrawObject {
        let (bb_min, bb_max) = if cullable {
            ([0.0; 3], [1.0; 3])
        } else {
            ([f32::NAN; 3], [f32::NAN; 3])
        };
        DrawObject {
            vertex_offset,
            vertex_count,
            index_offset,
            index_count,
            base_vertex: 0,
            geometry_generation: 0,
            shader_bucket: 0,
            model: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            texture_slot,
            normal_map_slot,
            material: MaterialUniforms::DEFAULT,
            visible: true,
            resident: true,
            bb_min,
            bb_max,
            cull_distance: 0.0,
            lod_alternates: Vec::new(),
        }
    }

    fn vert(x: f32) -> Vertex {
        Vertex {
            pos: [x, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            color: [1.0, 1.0, 1.0],
            uv: [0.0, 0.0],
        }
    }

    #[test]
    fn texture_stream_centers_scores_albedo_and_normal_slots() {
        // One draw sampling slot 0 as albedo and slot 2 as its normal map.
        let objs = vec![draw(0, 1, 0, 1, 0, 2, true)];
        let centers = texture_stream_centers(&objs, 4);
        assert_eq!(centers.len(), 4);
        assert_eq!(centers[0].len(), 1);
        assert_eq!(centers[2].len(), 1);
        assert!(centers[1].is_empty());
        assert!(centers[3].is_empty());
    }

    #[test]
    fn texture_stream_centers_skips_absent_normal_map() {
        let objs = vec![draw(0, 1, 0, 1, 1, NO_NORMAL_MAP_SLOT, true)];
        let centers = texture_stream_centers(&objs, 2);
        assert_eq!(centers[1].len(), 1);
        assert!(centers[0].is_empty());
    }

    #[test]
    fn mesh_stream_data_includes_cullable_and_narrows_indices_to_u16() {
        let verts: Vec<Vertex> = (0..4).map(|i| vert(i as f32)).collect();
        // Global indices into a mesh whose vertex region starts at vertex 2.
        let indices: Vec<u32> = vec![2, 3, 2];
        // vertex_offset is a BYTE offset; vertex 2 => 2 * size_of::<Vertex>().
        let vbyte = 2 * std::mem::size_of::<Vertex>();
        let objs = vec![draw(vbyte, 2, 0, 3, 0, NO_NORMAL_MAP_SLOT, true)];
        let data = mesh_stream_data(&objs, &verts, &indices, &Default::default());
        assert_eq!(data.draw_indices, vec![0]);
        assert_eq!(data.payloads.len(), 1);
        assert_eq!(data.payloads[0].vertices.len(), 2);
        // Global indices 2,3,2 rebased mesh-relative (minus vbase 2): 0,1,0.
        assert_eq!(data.payloads[0].indices, vec![0u16, 1, 0]);
    }

    #[test]
    fn mesh_stream_data_skips_non_cullable_and_out_of_range() {
        let verts: Vec<Vertex> = (0..2).map(|i| vert(i as f32)).collect();
        let indices: Vec<u32> = vec![0, 1];
        let objs = vec![
            // Non-cullable (NaN AABB): skybox / dynamic, stays resident.
            draw(0, 2, 0, 2, 0, NO_NORMAL_MAP_SLOT, false),
            // Cullable but vertex_count overruns the 2-vertex buffer: skipped.
            draw(0, 5, 0, 2, 0, NO_NORMAL_MAP_SLOT, true),
        ];
        let data = mesh_stream_data(&objs, &verts, &indices, &Default::default());
        assert!(data.draw_indices.is_empty());
        assert!(data.payloads.is_empty());
    }
}
