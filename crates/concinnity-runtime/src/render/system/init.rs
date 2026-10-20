// GraphicsSystem one-time setup: backend creation, draw-list build, and the
// shader / texture / streaming wiring performed on the first tick.

use concinnity_core::math::SaturatingFrom;
use concinnity_engine::bake::{font, texture};
use concinnity_engine::components::{
    BlockType, GraphicsConfig, Model, Planet, Shader, ShaderPrograms, StreamingConfig, SubMeshRef,
    VoxelWorld, Window,
};
use concinnity_engine::ecs::asset_id::AssetId;
use concinnity_engine::ecs::{OverlayImages, PayloadLocator, PipelineContext};
use concinnity_engine::gfx::mesh_payload::Vertex;
use concinnity_engine::gfx::render_types::{DrawIndex, DrawObject, FarFieldTint, InstancedCluster};
use concinnity_engine::render::backend::init::EmbeddedSurface;
use concinnity_engine::render::frame::queues::ActiveDeviceCaps;
use concinnity_engine::render::world_load::stream_seed::{
    StreamGeometry, StreamSeed, plan_stream_seed,
};
use concinnity_engine::render::{backend, backend::init, draw_list};
use concinnity_engine::resource::{FontTable, MaterialTable, TextureTable};
use concinnity_engine::settings::apply::{ApplyPlan, LiveChanges};

use super::backend_handoff::RuntimeHandoff;
use super::skinned_templates::SkinnedUpload;
use super::streaming::StreamingSetup;
use super::{FrameCounter, GraphicsSystem, RenderInitFailure, flow, maps};
use crate::live_edit::hot_reload_sources::{
    HotReloadSources, procedural_mesh_snapshot, resolve_sdf_fields,
};
use concinnity_engine::launch::LaunchRequest;
use concinnity_engine::render::frame::extract::gather_lights;
use concinnity_engine::render::world_load::blob_release::{blobs_to_release, retained_blobs};
use concinnity_engine::render::world_load::draw_geometry::probe_placements;
use concinnity_engine::render::world_load::media;
use concinnity_engine::render::world_load::prop_draws::{PropDrawInputs, assemble_prop_draws};
use concinnity_engine::render::world_load::skinned::{self, SkinnedAssembly};
use concinnity_engine::render::world_load::source_maps::{
    ColorLutSource, EnvironmentMapSource, ShaderSourceMap, SkinnedMeshSourceMap, TextureNameSlots,
    TextureSourceMap, mesh_source_map, procedural_mesh_source_map, skinned_mesh_source_map,
    texture_catalog,
};
use concinnity_engine::render::world_load::tables::{self, DecodedMaterials, TexturePool};
use concinnity_engine::render::world_load::texture_payloads::{
    TexturePayloads, TextureSlotSkips, decode_texture_payloads,
};
use concinnity_engine::render::world_load::world_fx::drain_world_fx;
use concinnity_engine::render::world_load::world_scan::{
    capture_hud_chips, color_lut_strength, cursor_policy, declares_frame_captures, font_less_text,
};
use concinnity_engine::settings::quality::resolve_ceiling;
use concinnity_engine::settings::system::{
    DisplayReport, SettingsInit, SettingsSlot, SettingsState,
};
use concinnity_engine::settings::{QualityCeiling, QualityPreset, WorldAssets, WorldSettings};

mod decode;

use decode::{WorldLoad, WorldPayloads};

// The resolved render config `init_render_settings` returns beside the live
// settings state: the launch's ray-tracing requests, the quality ceiling
// (planar-reflection budget), and the drained StreamingConfig.
struct ResolvedRenderConfig {
    launch: init::RenderLaunch,
    quality_ceiling: QualityCeiling,
    streaming_config: Option<StreamingConfig>,
    // Raw world ambient (`IndirectLighting.ambient_intensity`, no user override),
    // folded into the static LightUniforms built later in init.
    world_ambient_intensity: f32,
}

// The world's static draws, from `assemble_draws`.
struct SceneDraws {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    draw_objects: Vec<DrawObject>,
    instanced_clusters: Vec<InstancedCluster>,
    mesh_handle_to_draws: hashbrown::HashMap<usize, Vec<DrawIndex>>,
    far_bands: Vec<(u32, FarFieldTint)>,
    stream_seed: StreamSeed,
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
        ctx: &mut PipelineContext<'_>,
        launch: &LaunchRequest,
        persisted: &concinnity_engine::settings::StoredSettings,
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
        let preset = launch.resolve_quality_preset(persisted, ctx.resources);
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
        ctx: &mut PipelineContext<'_>,
        launch: &LaunchRequest,
        persisted: &concinnity_engine::settings::StoredSettings,
        quality: &DetectedQuality,
    ) -> (ResolvedRenderConfig, SettingsState) {
        let window = ctx.drain::<Window>().into_iter().next();
        self.frames = FrameCounter::new(launch.args.frames);
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
            settings.apply_to_streaming(sc);
        }

        settings.sync_menu_labels(ctx, persisted);
        // Each ScrollPanel's per-element clip band, captured before
        // UiInputSystem drains the ScrollPanels (GraphicsSystem.init runs first).
        self.clip_rects = maps::scroll_clip_rects(ctx);

        (
            ResolvedRenderConfig {
                launch: init::RenderLaunch {
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

    // Apply the persisted window mode, publish the Resolution row's mode list
    // (backend-enumerated, else the static preset fallback), apply a persisted
    // display-mode choice to the backend, and seed the frame-rate-cap resource +
    // the Resolution row's dynamic value label. Runs after the backend is built.
    fn finalize_display_modes(
        &mut self,
        ctx: &mut PipelineContext<'_>,
        settings: &mut SettingsState,
    ) {
        let reported = self.backend.as_deref_mut().map(|backend| {
            // No-op in embedded mode (the backend owns no window there).
            if let Some(mode) = settings.startup_window_mode() {
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
    fn finalize_backend_config(
        &mut self,
        ctx: &mut PipelineContext<'_>,
        settings: &mut SettingsState,
    ) {
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
            let ambient = ApplyPlan::live(LiveChanges {
                ambient: true,
                ..Default::default()
            });
            if let Err(e) = backend.apply_settings(&settings.resolved().graphics, &ambient) {
                tracing::error!("GraphicsSystem: ambient push failed: {e}");
            }
            // Push the movement key map (the persisted rebinds, or the default).
            // The backend decodes physical keys through it; idempotent with its
            // own default seed when there is no override.
            backend.set_keymap(*settings.keymap());
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
        ctx.insert_resource(ActiveDeviceCaps(device_caps));
        // Gray out + disable settings rows whose feature the device cannot
        // provide (e.g. ray-traced reflections on a GPU without hardware ray
        // tracing). Runs while the menu HitRegions / TextLabels / ScrollPanels
        // are still present (GraphicsSystem.init runs before UiInputSystem drains
        // them); the value-label sync above already set each row's live value.
        self.apply_capability_gating(ctx);
    }

    // The world's static draws: every placement's records over the shared
    // geometry buffers, each Shader's far band, and the streaming seed that
    // takes the streamed draws' geometry out of those buffers.
    fn assemble_draws(
        &mut self,
        ctx: &mut PipelineContext<'_>,
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
    pub(super) fn try_init(&mut self, ctx: &mut PipelineContext<'_>) -> Option<()> {
        let launch = ctx.resource::<LaunchRequest>().copied().unwrap_or_default();
        let persisted = ctx
            .resource::<concinnity_engine::settings::StoredSettings>()
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
        ) = self.init_render_settings(ctx, &launch, &persisted, &quality);
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
            concinnity_engine::render::world_load::world_scan::planet_grass(
                p,
                &concinnity_engine::render::environment::terrain::grass_by_id(ctx),
            )
        });
        let planet_far_field = planet_grass
            .as_ref()
            .map(concinnity_engine::render::environment::grass::PlanetGrass::far_field);
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
            resolve_sdf_fields(&fx.sdf_volumes, self.assets_dir().as_deref())
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
        let embedded_surface = ctx.resource::<EmbeddedSurface>().copied();

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
        use concinnity_engine::render::backend::init::{
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
                n_chunk_max:
                    concinnity_engine::render::streaming::pools::budget::chunk_record_reserve(
                        voxel_world
                            .as_ref()
                            .map(|vw| u32::saturating_from(vw.impostor_radius())),
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
                    concinnity_engine::render::world_load::far_band_shaders::far_band_shaders(
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
                ctx.insert_resource(RenderInitFailure(e));
                return None;
            }
        }

        self.finalize_display_modes(ctx, &mut settings);

        // The backend bakes a cube per placement.
        if let Some(backend) = self.backend.as_deref_mut() {
            backend.set_reflection_probes(&probe_placements);
        }

        let (hot_reload_sources, texture_name_slots) = if capture_sources {
            let sources = HotReloadSources {
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
            };
            // Parked even when empty: the dev drive's watcher still subscribes
            // to the host's world.jsonl.
            tracing::info!(
                "asset hot-reload: captured {} file-backed texture source(s), {} \
                 ColorLut source(s), {} EnvironmentMap source(s), {} Mesh \
                 source(s), {} SkinnedMesh source(s), {} ProceduralMesh source(s), \
                 {} Shader source(s), and {} SdfVolume field(s)",
                sources.map.len(),
                usize::from(sources.color_lut.is_some()),
                usize::from(sources.environment_map.is_some()),
                sources.meshes.len(),
                sources.skinned_meshes.len(),
                sources.procedural_meshes.len(),
                sources.shaders.len(),
                sources.sdf_fields.len(),
            );
            let slots = TextureNameSlots(world.texture_name_to_slot);
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

        self.scene_flow = flow::start_flow(
            ctx,
            self.extractor.scene_visibility_mut(),
            self.backend.as_deref_mut(),
        );

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
