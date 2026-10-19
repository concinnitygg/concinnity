// GraphicsSystem one-time setup: backend creation, draw-list build, and the
// shader / texture / streaming wiring performed on the first tick.

use concinnity_core::bake::font;
use concinnity_core::bake::texture;
use concinnity_core::components::Planet;
use concinnity_core::components::ShaderPrograms;
use concinnity_core::components::Sprite;
use concinnity_core::components::Story;
use concinnity_core::components::SubMeshRef;
use concinnity_core::components::WindowMode;
use concinnity_core::components::{
    BlockType, GraphicsConfig, Model, Shader, StreamingConfig, VoxelWorld, Window,
};
use concinnity_core::ecs::FontHandle;
use concinnity_core::ecs::OverlayImages;
use concinnity_core::ecs::PayloadLocator;
use concinnity_core::ecs::PipelineContext;
use concinnity_core::ecs::TextureHandle;
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::gfx::render_types::FarFieldTint;
use concinnity_core::render::{backend, backend_init};
use concinnity_core::resource::ColorLutTable;
use concinnity_core::resource::EnvironmentMapTable;
use concinnity_core::resource::FontTable;
use concinnity_core::resource::MaterialTable;
use concinnity_core::resource::ResourceEntry;
use concinnity_core::resource::TextureTable;

use super::backend_handoff::RuntimeHandoff;
use super::hot_reload_sources::{
    HotReloadSources, capture_hot_reload_sources, mesh_source_map, procedural_mesh_snapshot,
    procedural_mesh_source_map, skinned_mesh_source_map, texture_catalog,
};
use super::skinned_templates::SkinnedUpload;
use super::streaming::StreamingSetup;
use super::*;
use crate::app::run::LaunchRequest;
use concinnity_core::render::draw_list;
use concinnity_core::render::extract::gather_lights;
use concinnity_core::render::world_load::blob_release::{blobs_to_release, retained_blobs};
use concinnity_core::render::world_load::draw_geometry::probe_placements;
use concinnity_core::render::world_load::prop_draws::{PropDrawInputs, assemble_prop_draws};
use concinnity_core::render::world_load::skinned::{self, SkinnedAssembly};
use concinnity_core::render::world_load::stream_seed::{StreamGeometry, plan_stream_seed};
use concinnity_core::render::world_load::tables::{self, DecodedMaterials, TexturePool};
use concinnity_core::render::world_load::texture_payloads::{
    TexturePayloads, TextureSlotSkips, decode_texture_payloads,
};
use concinnity_core::render::world_load::world_fx::drain_world_fx;
use concinnity_core::render::world_load::world_scan::{
    capture_hud_chips, color_lut_strength, cursor_policy, declares_frame_captures, font_less_text,
};
use concinnity_core::settings::quality::resolve_ceiling;
use concinnity_core::settings::system::{DisplayReport, SettingsInit, SettingsSlot, SettingsState};
use concinnity_core::settings::{QualityCeiling, QualityPreset, WorldAssets, WorldSettings};

// The resolved render config `init_render_settings` returns beside the live
// settings state: the launch's ray-tracing requests, the quality ceiling
// (planar-reflection budget), and the drained StreamingConfig.
struct ResolvedRenderConfig {
    launch: backend_init::RenderLaunch,
    quality_ceiling: QualityCeiling,
    streaming_config: Option<StreamingConfig>,
    // Raw world ambient (`IndirectLighting.ambient_intensity`, no user override),
    // folded into the static LightUniforms built later in init.
    world_ambient_intensity: f32,
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
            .map_err(|e| e.to_string())?
            .to_vec();
        return Ok(ShaderPayloadSource::Bytes(bytes));
    }
    let (path, start) = super::stream_sources::blob_payload_section(locator.blob_index)?;
    Ok(ShaderPayloadSource::Disk {
        path,
        offset: start + locator.offset,
        len: locator.len,
    })
}

// The payload of the sole resource of its kind the renderer uses, from `first`
// (the table's length and its handle 0 entry): the bytes a world baked for
// itself, else read through its locator. Extra declarations are logged and
// ignored. `Some(None)` with no payload; `None`, failing init, if unreadable.
fn first_payload(
    ctx: &mut PipelineContext,
    first: Option<(usize, Option<ResourceEntry>)>,
    what: &str,
) -> Option<Option<Vec<u8>>> {
    let Some((count, entry)) = first else {
        return Some(None);
    };
    if count > 1 {
        tracing::warn!("GraphicsSystem: {count} {what}s declared; only the first is used");
    }
    let Some(entry) = entry else {
        return Some(None);
    };
    if let Some(baked) = entry.baked_bytes() {
        return Some(Some(baked.to_vec()));
    }
    let Some(locator) = entry.payload else {
        return Some(None);
    };
    match ctx.read_payload(&locator) {
        Ok(bytes) => Some(Some(bytes.to_vec())),
        Err(e) => {
            tracing::error!("GraphicsSystem: failed to read {what} payload: {e}");
            None
        }
    }
}

// How `decode_world` reads the world: whether content streams, whether the
// hot-reload source maps are gathered, and whether the blobs are files on disk.
#[derive(Clone, Copy)]
struct WorldLoad {
    streaming: bool,
    capture_sources: bool,
    blob_disk_backed: bool,
}

// Every payload the graphics load decoded before the blobs were released.
struct WorldPayloads {
    mesh: draw_list::MeshGeometry,
    deferred_mesh_sources: draw_list::DeferredMeshSources,
    proc_mesh_args_snapshot:
        hashbrown::HashMap<AssetId, (String, concinnity_core::components::ProceduralMesh)>,
    model_map: hashbrown::HashMap<AssetId, Vec<SubMeshRef>>,
    room_geometry: Vec<draw_list::RoomGeometry>,
    terrains: Vec<concinnity_core::render::terrain::LoadedTerrain>,
    shaders: DecodedShaders,
    textures: TexturePool,
    texture_count: usize,
    asset_source_map: super::hot_reload_sources::TextureSourceMap,
    texture_name_to_slot: hashbrown::HashMap<AssetId, usize>,
    materials: DecodedMaterials,
    skinned: SkinnedAssembly,
    skinned_mesh_source_map: super::hot_reload_sources::SkinnedMeshSourceMap,
    texture_payloads: TexturePayloads,
    env_map_bytes: Option<Vec<u8>>,
    environment_map_source: Option<super::hot_reload_sources::EnvironmentMapSource>,
    env_map_background: bool,
    color_lut_bytes: Option<Vec<u8>>,
    color_lut_source: Option<super::hot_reload_sources::ColorLutSource>,
    text_atlas_data: Vec<(u32, u32, Vec<u8>)>,
}

// The world's static draws, from `assemble_draws`.
struct SceneDraws {
    vertices: Vec<concinnity_core::gfx::mesh_payload::Vertex>,
    indices: Vec<u32>,
    draw_objects: Vec<concinnity_core::gfx::render_types::DrawObject>,
    instanced_clusters: Vec<concinnity_core::gfx::render_types::InstancedCluster>,
    mesh_handle_to_draws:
        hashbrown::HashMap<usize, Vec<concinnity_core::gfx::render_types::DrawIndex>>,
    far_bands: Vec<(u32, FarFieldTint)>,
    stream_seed: concinnity_core::render::world_load::stream_seed::StreamSeed,
}

struct DecodedShaders {
    locators: Vec<PayloadLocator>,
    source_map: super::shader_sources::ShaderSourceMap,
    // Hot-reloaded programs shared with the shader warmup, under capture.
    overrides: super::parked::ShaderOverrides,
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

// The GPU the backend is built for and the quality preset resolved against it.
struct DetectedQuality {
    gpu_profile: backend::GpuProfile,
    capabilities: backend::DeviceCapabilities,
    preset: QualityPreset,
    ceiling: QualityCeiling,
}

impl GraphicsSystem {
    // Detect the GPU and resolve the quality preset and its ceiling. Runs
    // before the backend is built, since the ceiling sizes its render targets
    // and effect pipelines.
    fn detect_quality(
        &self,
        ctx: &mut PipelineContext,
        launch: &LaunchRequest,
        persisted: &concinnity_core::settings::StoredSettings,
    ) -> DetectedQuality {
        let backend::DeviceProbe {
            profile: gpu_profile,
            capabilities,
        } = self.detect_device();
        ctx.insert_resource(gpu_profile);
        crate::crash::note(
            "gpu",
            &format!("{:?} {:?}", gpu_profile.vendor, gpu_profile.tier),
        );
        // `--quality-preset` wins and is never persisted; a preset never
        // chosen seeds and persists `Auto` once.
        let preset = launch
            .resolve_quality_preset(persisted.graphics.quality_preset)
            .unwrap_or_else(|| {
                let mut seeded = persisted.clone();
                seeded.graphics.quality_preset = Some(QualityPreset::Auto);
                seeded.store_changed(ctx.resources);
                QualityPreset::Auto
            });
        let ceiling = resolve_ceiling(preset, &gpu_profile);
        tracing::info!(
            "auto-config: GPU tier {:?}, quality preset {:?}",
            gpu_profile.tier,
            preset,
        );
        DetectedQuality {
            gpu_profile,
            capabilities,
            preset,
            ceiling,
        }
    }

    // Drain the world's render config, resolve it against the user's persisted
    // choices and the quality ceiling into a fresh settings state, sync the
    // settings-menu labels, and return the config the rest of init needs.
    fn init_render_settings(
        &mut self,
        ctx: &mut PipelineContext,
        launch: &LaunchRequest,
        persisted: &concinnity_core::settings::StoredSettings,
        quality: DetectedQuality,
    ) -> (ResolvedRenderConfig, SettingsState) {
        let window = ctx.drain::<Window>().into_iter().next();
        self.max_frames = launch.args.frames;
        let mut streaming_config = ctx.drain::<StreamingConfig>().into_iter().next();
        let graphics = ctx.drain::<GraphicsConfig>().into_iter().next();
        let authored = WorldSettings::new(WorldAssets {
            graphics: graphics.as_ref(),
            lut_strength: color_lut_strength(ctx),
            streaming: streaming_config.as_ref(),
        });
        self.clear_color = authored.graphics.background.color;
        let mut settings = SettingsState::from_init(SettingsInit {
            stored: persisted,
            gpu_profile: quality.gpu_profile,
            capabilities: quality.capabilities,
            preset: quality.preset,
            ceiling: &quality.ceiling,
            window,
            authored,
            // `--vsync`, `--fps-cap` and `--dlss-preset` win for this launch,
            // shown on the row and never persisted unless the row itself changes.
            session: launch.session_overrides(),
        });
        if let Some(sc) = streaming_config.as_mut() {
            sc.texture_cap = settings.resolved().texture_cap;
            sc.texture_budget = settings.resolved().texture_budget;
        }

        settings.sync_menu_labels(ctx, persisted);
        // Each ScrollPanel's per-element clip band, captured before
        // UiInputSystem drains the ScrollPanels (GraphicsSystem.init runs first).
        self.clip_rects = concinnity_core::render::overlay_maps::scroll_clip_rects(ctx);

        (
            ResolvedRenderConfig {
                launch: backend_init::RenderLaunch {
                    rt_dynamic: launch.resolve_rt_dynamic(),
                    rt_skinned_geometry: launch.resolve_rt_skinned_geometry(),
                },
                quality_ceiling: quality.ceiling,
                streaming_config,
                world_ambient_intensity: settings.authored().world_ambient(),
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
    // Drain the world's Shader components, read every compiled stage
    // container, and split each into the per-stage byte sets the backend's
    // pipeline table consumes. Drain order matches cook's shader handle
    // assignment (both walk the declaration-ordered asset list), so a baked
    // `ShaderHandle` indexes the returned list directly; entry 0 drives the
    // world default pipeline. Under hot-reload capture also records every
    // Shader's resolved on-disk files so the asset hot-reload watcher can
    // recompile it and rebuild its pipeline on a save. Returns
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
        let (shader_ids, world_shaders): (Vec<Option<AssetId>>, Vec<Shader>) =
            ctx.drain_with_ids::<Shader>().into_iter().unzip();
        if world_shaders.is_empty() {
            return Some(DecodedShaders {
                locators: Vec::new(),
                shaders: vec![DecodedShader::default()],
                source_map: Default::default(),
                overrides: Default::default(),
            });
        }

        // Buckets a non-start scene exclusively owns skip their decode and
        // pipeline build here; the streaming pump warms them when that scene
        // pins. The backend sees them flagged `deferred` and leaves the bucket's
        // pipeline unbuilt.
        self.deferred_shader_scenes =
            concinnity_core::render::world_load::scene_deferral::deferred_shader_buckets(
                ctx,
                streaming,
                &shader_ids,
            )
            .into_iter()
            .map(|(bucket, scene)| (bucket as u32, scene))
            .collect();
        let deferred_buckets: hashbrown::HashSet<u32> = self
            .deferred_shader_scenes
            .iter()
            .map(|&(bucket, _)| bucket)
            .collect();
        let blob_disk_backed = ctx.blob.disk_backed();
        let mut deferred_sources = Vec::new();
        let overrides = super::parked::ShaderOverrides::default();

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
            // Read the stage container before the blob is released: it may
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
                    tracing::error!("GraphicsSystem: failed to read shader payload: {}", e);
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
                capture_sources.then(|| overrides.clone()),
            ));
        }

        let source_map = if capture_sources {
            let assets_dir = self.assets_dir();
            super::shader_sources::ShaderSourceMap::build(
                shader_ids.iter().copied().zip(&world_shaders),
                |raw| {
                    concinnity_host::store::source::resolve_source_path(raw, assets_dir.as_deref())
                },
                concinnity_host::thread::asset_id::name_of,
            )
        } else {
            Default::default()
        };

        Some(DecodedShaders {
            locators,
            source_map,
            overrides,
            shaders,
        })
    }

    // Apply the persisted window mode, publish the Resolution row's mode list
    // (backend-enumerated, else the static preset fallback), apply a persisted
    // display-mode choice to the backend, and seed the frame-rate-cap resource +
    // the Resolution row's dynamic value label. Runs after the backend is built.
    fn finalize_display_modes(&mut self, ctx: &mut PipelineContext, settings: &mut SettingsState) {
        let reported = self.backend.as_deref_mut().map(|backend| {
            // The window is always created as a standard titled window, so a
            // persisted or authored Borderless / Fullscreen mode is applied here.
            // No-op in embedded mode (the backend owns no window there).
            let mode = settings.window().mode;
            if mode != WindowMode::Windowed {
                backend.set_window_mode(mode);
            }
            let report = DisplayReport {
                modes: backend.display_modes(),
                current: backend.current_display_mode(),
            };
            if let Some(resolution) = settings.resolution() {
                backend.set_display_mode(resolution);
            }
            report
        });
        settings.adopt_display(ctx, reported);
    }

    // Decide cursor handling and push the post-build backend config: menu mode,
    // ambient scale, key map, the startup cursor grab (plain first-person worlds
    // only), and the device capability flags that gate the settings rows.
    fn finalize_backend_config(&mut self, ctx: &mut PipelineContext, settings: &mut SettingsState) {
        // In menu mode capture is driven per-frame in `run_step`.
        let cursor = cursor_policy(ctx);
        self.extractor.set_menu_mode(cursor.menu_mode);
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
            backend.set_menu_mode(cursor.menu_mode);
            // Push the effective ambient scale (world value or persisted
            // override). The backend already seeds the world value at its own
            // init, so this is the path that applies a persisted Ambient-slider
            // choice; idempotent when there is no override.
            let ambient = concinnity_core::settings::apply::ApplyPlan::live(
                concinnity_core::settings::apply::LiveChanges {
                    ambient: true,
                    ..Default::default()
                },
            );
            if let Err(e) = backend.apply_settings(&settings.resolved().graphics, &ambient) {
                tracing::error!("GraphicsSystem: ambient push failed: {e}");
            }
            // Push the movement key map (the persisted rebinds, or the default).
            // The backend decodes physical keys through it; idempotent with its
            // own default seed when there is no override.
            backend.set_keymap(settings.keymap());
            if cursor.capture_at_start {
                backend.request_cursor_capture();
            }
        }
        self.caps = device_caps;
        // The built backend is the authority on what it can run from here on
        // (Vulkan settles ray tracing beside the upscaler it actually built).
        settings.set_capabilities(device_caps);
        // Publish the flags for the systems that cannot reach the backend
        // themselves (the editor's live draw seam asks whether a rewritten draw
        // slot would land).
        ctx.insert_resource(concinnity_core::render::queues::ActiveDeviceCaps(
            device_caps,
        ));
        // Gray out + disable settings rows whose feature the device cannot
        // provide (e.g. ray-traced reflections on a GPU without hardware ray
        // tracing). Runs while the menu HitRegions / TextLabels / ScrollPanels
        // are still present (GraphicsSystem.init runs before UiInputSystem drains
        // them); the value-label sync above already set each row's live value.
        self.apply_capability_gating(ctx);
    }

    // The sole EnvironmentMap's (handle 0) IBL payload, and, with
    // `capture_sources`, the resolved HDR source path and convolution sizing the
    // hot-reload watcher rebuilds it from (procedural generators have no file to
    // watch). None, failing init, if the payload is unreadable.
    fn decode_environment_map(
        &self,
        ctx: &mut PipelineContext,
        capture_sources: bool,
    ) -> Option<(
        Option<Vec<u8>>,
        Option<super::hot_reload_sources::EnvironmentMapSource>,
    )> {
        let first = ctx
            .resource::<EnvironmentMapTable>()
            .map(|t| (t.len(), t.0.first().cloned()));
        let env_map_bytes = first_payload(ctx, first, "EnvironmentMap")?;
        let mut environment_map_source = None;
        if capture_sources
            && let Some(info) = ctx
                .resource::<concinnity_core::resource::EnvironmentMapSources>()
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

    // The sole ColorLut's (handle 0) color-grading payload, and, with
    // `capture_sources`, its resolved source path for the hot-reload watcher.
    // None, failing init, if the payload is unreadable.
    fn decode_color_lut(
        &self,
        ctx: &mut PipelineContext,
        capture_sources: bool,
    ) -> Option<(
        Option<Vec<u8>>,
        Option<super::hot_reload_sources::ColorLutSource>,
    )> {
        let first = ctx
            .resource::<ColorLutTable>()
            .map(|t| (t.len(), t.0.first().cloned()));
        let color_lut_bytes = first_payload(ctx, first, "ColorLut")?;
        let mut color_lut_source = None;
        if capture_sources
            && let Some(src) = ctx
                .resource::<concinnity_core::resource::ColorLutSources>()
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
                            "GraphicsSystem: failed to read Font handle {} payload: {}",
                            slot,
                            e
                        );
                        return None;
                    }
                },
                (None, None) => {
                    tracing::error!(
                        "GraphicsSystem: Font handle {} has no compiled payload; did the build succeed?",
                        slot
                    );
                    return None;
                }
            };
            match font::load_face(slot, &bytes) {
                Ok((face, atlas)) => {
                    self.loaded_fonts.insert(FontHandle::new(slot as u32), face);
                    text_atlas_data.push(atlas);
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
            let handle = FontHandle::new(slot as u32);
            match crate::gfx::builtin_font::load(handle) {
                Some(builtin) => {
                    text_atlas_data.push(builtin.atlas);
                    self.loaded_fonts.insert(handle, builtin.loaded);
                    self.loaded_fonts.set_fallback(handle);
                }
                None => tracing::error!(
                    "GraphicsSystem: text naming no Font cannot draw: the built-in face failed to decode"
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

    // Decode every payload the world's graphics load reads, then release the
    // shared blobs nothing else holds. All payloads live in those blobs, so
    // each is consumed here before the release.
    fn decode_world(
        &mut self,
        ctx: &mut PipelineContext,
        load: WorldLoad,
    ) -> Option<WorldPayloads> {
        let WorldLoad {
            streaming,
            capture_sources,
            blob_disk_backed,
        } = load;
        let proc_mesh_args_snapshot = if capture_sources {
            procedural_mesh_snapshot(ctx)
        } else {
            hashbrown::HashMap::new()
        };

        // Mesh sources owned by a scene other than the start scene skip their
        // payload decode: draw records use the blob's baked bounds, and the
        // mesh streamer decodes the payload when the owning scene pins.
        let deferred_mesh_sources =
            concinnity_core::render::world_load::scene_deferral::deferred_mesh_sources(
                ctx, streaming,
            );
        let mesh = draw_list::load_mesh_geometry(
            ctx,
            &deferred_mesh_sources,
            blob_disk_backed,
            capture_sources,
        )?;

        let (skinned_geometry, skinned_blob_indices) = skinned::decode_skinned(ctx)
            .map_err(|e| tracing::error!("GraphicsSystem: {e}"))
            .ok()?;

        // drain Model components into a name-keyed map for Prop lookup
        let model_map: hashbrown::HashMap<AssetId, Vec<SubMeshRef>> = ctx
            .drain_with_ids::<Model>()
            .into_iter()
            .filter_map(|(id, m)| Some((id?, m.meshes)))
            .collect();

        // decode Room payloads before shaders/textures are read; all payloads
        // live in the same blob and must be consumed before it is released
        let (room_geometry, room_blob_indices) = draw_list::load_room_geometry(ctx)?;
        let (terrains, terrain_blob_indices) =
            concinnity_core::render::terrain::load_terrains(ctx)?;

        let shaders = self.decode_shaders(ctx, streaming, capture_sources)?;

        // Read the shared texture pool + the material table into the maps the
        // draw list resolves against.
        let empty_textures = TextureTable::default();
        let texture_table = ctx.resource::<TextureTable>().unwrap_or(&empty_textures);
        let texture_count = texture_table.len();
        let textures = tables::texture_pool(texture_table)
            .map_err(|e| tracing::error!("GraphicsSystem: {e}"))
            .ok()?;
        let (asset_source_map, texture_name_to_slot) = if capture_sources {
            texture_catalog(
                ctx.resource::<concinnity_core::resource::TextureSources>(),
                texture_count,
            )
        } else {
            Default::default()
        };
        let empty_materials = MaterialTable::default();
        let materials = tables::decode_materials(
            ctx.resource::<MaterialTable>().unwrap_or(&empty_materials),
            texture_count,
        )
        .map_err(|e| tracing::error!("GraphicsSystem: {e}"))
        .ok()?;

        // Build skinned draw objects, the shared skinned vertex/index buffers,
        // and bind-pose skeletons from the decoded SkinnedMesh geometry. Runs
        // after the material map so SkinnedMesh material references resolve.
        let skinned = skinned::assemble_skinned(&skinned_geometry, &materials.map)
            .map_err(|e| tracing::error!("GraphicsSystem: {e}"))
            .ok()?;
        let skinned_mesh_source_map = if capture_sources {
            skinned_mesh_source_map(&skinned_geometry, &skinned)
        } else {
            Default::default()
        };

        let deferred_slots =
            concinnity_core::render::world_load::scene_deferral::deferred_texture_slots(
                ctx,
                streaming,
                textures.locators.len(),
            );
        let texture_payloads = decode_texture_payloads(
            ctx,
            &textures.locators,
            TextureSlotSkips {
                deferred: &deferred_slots,
                cook_only: &textures.cook_only,
            },
            blob_disk_backed,
        )?;

        // Read the sole EnvironmentMap + ColorLut payloads, then build the shared
        // text/sprite atlas pool.
        let (env_map_bytes, environment_map_source) =
            self.decode_environment_map(ctx, capture_sources)?;
        let env_map_background = match ctx.resource::<EnvironmentMapTable>().map(|t| t.record(0)) {
            None => true,
            Some(Ok(record)) => record.background,
            Some(Err(e)) => {
                tracing::error!(
                    "GraphicsSystem: EnvironmentMap handle 0 runtime record failed to decode: {e}"
                );
                return None;
            }
        };
        let (color_lut_bytes, color_lut_source) = self.decode_color_lut(ctx, capture_sources)?;
        let TextAtlases {
            atlases: text_atlas_data,
            font_blob_indices,
        } = self.decode_text_atlases(ctx, &textures.locators)?;

        let consumed = shaders
            .locators
            .iter()
            .map(|l| l.blob_index)
            .chain(textures.locators.iter().map(|l| l.blob_index))
            .chain(room_blob_indices)
            .chain(terrain_blob_indices)
            .chain(font_blob_indices)
            .chain(skinned_blob_indices);
        for idx in blobs_to_release(consumed, &retained_blobs(ctx)) {
            ctx.release_blob(idx);
        }

        Some(WorldPayloads {
            mesh,
            deferred_mesh_sources,
            proc_mesh_args_snapshot,
            model_map,
            room_geometry,
            terrains,
            shaders,
            textures,
            texture_count,
            asset_source_map,
            texture_name_to_slot,
            materials,
            skinned,
            skinned_mesh_source_map,
            texture_payloads,
            env_map_bytes,
            environment_map_source,
            env_map_background,
            color_lut_bytes,
            color_lut_source,
            text_atlas_data,
        })
    }

    // The world's static draws: every placement's records over the shared
    // geometry buffers, each Shader's far band, and the streaming seed that
    // takes the streamed draws' geometry out of those buffers.
    fn assemble_draws(
        &mut self,
        ctx: &mut PipelineContext,
        world: &WorldPayloads,
        streaming_config: Option<&StreamingConfig>,
    ) -> Option<SceneDraws> {
        // A geometry-less world (e.g. text-only) is valid: the backend is
        // initialized with empty geometry buffers and only the text path runs.
        let draw_list::DrawListData {
            vertices: mut all_vertices,
            indices: mut all_indices,
            mut draw_objects,
            mut instanced_clusters,
            mesh_handle_to_draws,
            ..
        } = assemble_prop_draws(
            ctx,
            PropDrawInputs {
                model_map: &world.model_map,
                mesh_geometry: &world.mesh.meshes,
                room_geometry: &world.room_geometry,
                terrains: &world.terrains,
                texture_count: world.texture_count,
                material_map: &world.materials.map,
            },
            self.extractor.pick_candidates_mut(),
        )?;

        // Taken before streaming plans its geometry, so a Shader's far band is
        // seen whichever scene its draws stream with.
        let far_bands: Vec<(u32, FarFieldTint)> = draw_objects
            .iter()
            .map(|d| (d.shader_bucket, d.material.far_field))
            .collect();

        let stream_seed = plan_stream_seed(StreamGeometry {
            vertices: &mut all_vertices,
            indices: &mut all_indices,
            draw_objects: &mut draw_objects,
            instanced_clusters: &mut instanced_clusters,
            mesh_handle_to_draws: &mesh_handle_to_draws,
            deferred_mesh_seeds: &world.mesh.deferred_seeds,
            deferred_mesh_counts: &world.deferred_mesh_sources.counts,
            texture_count: world.texture_payloads.images.len(),
            config: streaming_config,
        });

        Some(SceneDraws {
            vertices: all_vertices,
            indices: all_indices,
            draw_objects,
            instanced_clusters,
            mesh_handle_to_draws,
            far_bands,
            stream_seed,
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
        let persisted = ctx
            .resource::<concinnity_core::settings::StoredSettings>()
            .cloned()
            .unwrap_or_default();
        let quality = self.detect_quality(ctx, &launch, &persisted);
        (self.debug_hud_chips, self.stat_hud_chips) = capture_hud_chips(ctx);
        let (
            ResolvedRenderConfig {
                launch: render_launch,
                quality_ceiling,
                streaming_config,
                world_ambient_intensity,
            },
            mut settings,
        ) = self.init_render_settings(ctx, &launch, &persisted, quality);
        // Infinite-world chunk streaming. The first declared VoxelWorld wins;
        // with none declared, no chunks stream. BlockTypes are drained here so
        // the runtime can resolve the VoxelWorld palette to chunk-mesh data.
        let voxel_world = ctx.drain::<VoxelWorld>().into_iter().next();
        // A planet's terrain streams through the same chunk pool, so a world
        // declares one or the other.
        let planet = ctx.query::<Planet>().next().cloned();
        let block_types: hashbrown::HashMap<AssetId, BlockType> = ctx
            .drain_with_ids::<BlockType>()
            .into_iter()
            .filter_map(|(id, bt)| Some((id?, bt)))
            .collect();

        // Whether the blob payloads came from files on disk rather than an
        // in-memory build. Captured before the blobs are
        // released; the streaming subsystem uses it to pick a disk-backed
        // payload source so streamed bytes need not stay RAM-resident.
        let blob_disk_backed = ctx.blob.disk_backed();

        // A development loop gathers the file-backed source maps the hot-reload
        // watcher consumes.
        let capture_sources = launch.dev_loop;
        let world = self.decode_world(
            ctx,
            WorldLoad {
                streaming: streaming_config.is_some(),
                capture_sources,
                blob_disk_backed,
            },
        )?;
        let (light_data, light_uniforms) = gather_lights(ctx, world_ambient_intensity);

        let SceneDraws {
            vertices: all_vertices,
            indices: all_indices,
            draw_objects,
            instanced_clusters,
            mesh_handle_to_draws,
            far_bands,
            stream_seed,
        } = self.assemble_draws(ctx, &world, streaming_config.as_ref())?;

        let draw_object_count = draw_objects.len();
        let cluster_count = instanced_clusters.len();
        let total_instances: usize = instanced_clusters.iter().map(|c| c.instances.len()).sum();

        let planet_grass = planet.as_ref().and_then(|p| {
            concinnity_core::render::world_load::world_scan::planet_grass(
                p,
                &concinnity_core::render::terrain::grass_by_id(ctx),
            )
        });
        let planet_far_field = planet_grass.as_ref().map(|g| g.far_field());
        let fx = drain_world_fx(
            ctx,
            concinnity_host::thread::asset_id::name_of,
            world.texture_count,
            &world.terrains,
            planet_grass,
            planet.as_ref(),
            quality_ceiling.ocean,
        );
        self.extractor.set_grows_grass(fx.grass.is_some());
        self.extractor.set_atmosphere(fx.atmosphere);
        let sdf_fields = if capture_sources {
            super::sdf_field_sources::SdfFieldMap::resolve(
                &fx.sdf_volumes,
                self.assets_dir().as_deref(),
            )
        } else {
            Default::default()
        };
        let decal_count = fx.decals.len();
        let particle_count = fx.particles.len();
        let fog_settings = fx.fog;
        settings.set_fog_built(fog_settings.is_some());

        // Metal is unaffected by `validation`: its layer is enabled by the CLI
        // re-execing with `MTL_DEBUG_LAYER`.
        let validation = launch.resolve_validation();
        let capture = launch.frame_capture() || declares_frame_captures(ctx);
        let embedded_surface = ctx
            .resource::<concinnity_core::render::backend_init::EmbeddedSurface>()
            .copied();

        let probe_placements = probe_placements(
            ctx,
            &draw_objects,
            &all_vertices,
            &all_indices,
            fx.water_surfaces.iter().map(|w| &w.surface),
            &fx.glass_panels,
        );

        // Assemble the backend construction inputs, derive the world's render
        // requirements from them (a world with no 3D content drops every
        // scene-scoped feature before any backend resource is sized), and
        // hand the result to the backend this process renders with.
        use concinnity_core::render::backend_init::{
            BackendInit, MediaPayloads, PlanarBudget, SceneData, WorldShader,
        };
        let mut backend_init = BackendInit {
            platform: crate::platform::current(),
            window: settings.window(),
            validation,
            graphics: settings.resolved().graphics.clone(),
            // Restart-class: the main-pass pipelines, render targets and planar /
            // probe faces all bake the count, so a live AA toggle keeps it.
            hdr_samples: settings.resolved().graphics.hdr_samples(),
            launch: render_launch,
            // A shipped run leaves shader hot-reload off, so the backend never
            // spawns the filesystem watcher.
            hot_reload: capture_sources,
            capture,
            embedded_surface,
            scene: SceneData {
                vertices: &all_vertices,
                indices: &all_indices,
                draw_objects,
                instanced_clusters,
                // Sizes the GPU-cull buffers for the merged total; the skinned
                // geometry itself is uploaded after the build.
                n_skinned: world.skinned.draw_objects.len(),
                // Worst-case resident chunk count, so the GPU-cull buffers
                // reserve a chunk record region (0 for a non-voxel world).
                n_chunk_max: concinnity_core::render::stream_pools::budget::chunk_record_reserve(
                    voxel_world.as_ref().map(|vw| vw.impostor_radius() as u32),
                    planet.is_some(),
                ),
                material_params: world.materials.params,
            },
            // One entry per world Shader, indexed by ShaderHandle value;
            // entry 0 is the world default program.
            shaders: world
                .shaders
                .shaders
                .iter()
                .zip(
                    concinnity_core::render::world_load::far_band_shaders::far_band_shaders(
                        world.shaders.shaders.len(),
                        far_bands,
                        planet_far_field.is_some(),
                    ),
                )
                .map(|(s, canopy)| WorldShader {
                    programs: s.programs.as_ref(),
                    deferred: s.deferred,
                    canopy,
                })
                .collect(),
            media: MediaPayloads {
                textures: &world.texture_payloads.images,
                text_atlases: world.text_atlas_data,
                env_map_bytes: world.env_map_bytes.as_deref(),
                env_map_background: world.env_map_background,
                color_lut_bytes: world.color_lut_bytes.as_deref(),
            },
            light_uniforms,
            local_lights: light_data.lights,
            spot_shadows: light_data.spot_shadows,
            area_lights: light_data.area_lights,
            // Restart-required: the mirror targets are allocated once at backend
            // init, so the quality ceiling scales the engine capacity here.
            planar: PlanarBudget {
                planes: quality_ceiling.planar_reflection_planes as usize,
                resolution: quality_ceiling.planar_reflection_resolution,
            },
            fx,
            requirements: Default::default(),
        };
        backend_init.resolve_requirements();
        match self.build_backend(ctx, backend_init) {
            Ok(backend) => self.backend = Some(backend),
            Err(e) => {
                tracing::error!("GraphicsSystem: backend build failed: {e}");
                // The run already resolved to a windowed one, so this machine
                // has a GPU that refused rather than no GPU at all. Leave the
                // cause where `Runtime::start` reads it, so the failure reaches
                // the caller instead of a loop that draws nothing.
                ctx.insert_resource(crate::ecs::RenderInitFailure(e));
                return None;
            }
        }

        self.finalize_display_modes(ctx, &mut settings);

        // The backend bakes a cube per placement.
        if let Some(backend) = self.backend.as_deref_mut() {
            backend.set_reflection_probes(&probe_placements);
        }

        let (hot_reload_sources, texture_name_slots) = if capture_sources {
            let (sources, slots) = capture_hot_reload_sources(
                HotReloadSources {
                    map: world.asset_source_map,
                    color_lut: world.color_lut_source,
                    environment_map: world.environment_map_source,
                    meshes: mesh_source_map(&world.mesh.sources, &mesh_handle_to_draws),
                    skinned_meshes: world.skinned_mesh_source_map,
                    procedural_meshes: procedural_mesh_source_map(
                        &world.proc_mesh_args_snapshot,
                        &world.mesh.component_handles,
                        &mesh_handle_to_draws,
                    ),
                    shaders: world.shaders.source_map,
                    shader_overrides: world.shaders.overrides,
                    sdf_fields,
                },
                world.texture_name_to_slot,
            );
            (Some(sources), Some(slots))
        } else {
            (None, None)
        };

        let skinned_upload = SkinnedUpload {
            vertices: world.skinned.vertices,
            indices: world.skinned.indices,
            draw_objects: world.skinned.draw_objects,
            morphs: world.skinned.morphs,
        };
        if let Err(e) = self.install_skinned_templates(ctx, skinned_upload, world.skinned.templates)
        {
            tracing::error!("GraphicsSystem: skinned upload failed: {e}");
            return None;
        }

        self.setup_streaming(StreamingSetup {
            config: streaming_config,
            seed: stream_seed,
            texture_payloads: world.texture_payloads.payloads,
            texture_locators: &world.textures.locators,
            cook_only_textures: &world.textures.cook_only,
            disk_backed: blob_disk_backed,
            deferred_mesh_seeds: &world.mesh.deferred_seeds,
            voxel_world,
            planet,
            planet_far_field,
            block_types: &block_types,
            material_map: &world.materials.map,
        });

        self.finalize_backend_config(ctx, &mut settings);

        self.setup_scene_flow(ctx);

        self.park_runtime_state(
            ctx,
            RuntimeHandoff {
                draw_object_count,
                frames_in_flight: settings.resolved().graphics.frames_in_flight(),
                skinned_pool_reservations: &world.skinned.pool_reservations,
                fog: fog_settings,
                texture_name_slots,
                hot_reload_sources,
            },
        );
        tracing::info!(
            "GraphicsSystem: ready ({}x{} \"{}\", {} frames in flight, {} draw objects, {} instanced clusters ({} instances total), {} decals, {} particle emitter(s), fog={})",
            settings.window().width,
            settings.window().height,
            settings.window().title,
            settings.resolved().graphics.frame_pacing.frames_in_flight,
            draw_object_count,
            cluster_count,
            total_instances,
            decal_count,
            particle_count,
            if fog_settings.is_some() { "on" } else { "off" },
        );
        ctx.insert_resource(SettingsSlot::new(settings));
        Some(())
    }
}
