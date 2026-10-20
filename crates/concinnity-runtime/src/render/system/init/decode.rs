// The decode steps of graphics init: every payload the world's graphics load
// reads, decoded before the shared blobs are released.

use super::{
    AssetId, ColorLutSource, DecodedMaterials, EnvironmentMapSource, FontTable, GraphicsSystem,
    MaterialTable, Model, OverlayImages, PayloadLocator, PipelineContext, Shader, ShaderPrograms,
    ShaderSourceMap, SkinnedAssembly, SkinnedMeshSourceMap, SubMeshRef, TexturePayloads,
    TexturePool, TextureSlotSkips, TextureSourceMap, TextureTable, blobs_to_release,
    decode_texture_payloads, draw_list, font, font_less_text, media, procedural_mesh_snapshot,
    retained_blobs, skinned, skinned_mesh_source_map, tables, texture, texture_catalog,
};
use concinnity_core::math::SaturatingFrom;
use concinnity_core::render::environment::terrain::LoadedTerrain;
use concinnity_core::render::overlay::maps::{Atlas, AtlasPool};
use concinnity_core::render::world_load::scene_deferral::ShaderDeferral;

// One world Shader as decoded at init: its compiled programs, or nothing for
// a bucket a non-start scene owns.
#[derive(Default)]
pub(super) struct DecodedShader {
    pub(super) programs: Option<ShaderPrograms>,
    // The payload was left undecoded because a scene other than the start scene
    // owns this bucket.
    pub(super) deferred: bool,
}

// Where the streaming pump re-reads a deferred bucket's stage container: the
// blob's byte range when the world is disk-backed (`cn run`, so the bytes
// never stay RAM-resident), else a copy of the in-memory payload.
fn deferred_shader_source(
    ctx: &mut PipelineContext<'_>,
    locator: &PayloadLocator,
    blob_disk_backed: bool,
) -> Result<crate::render::streaming::shader::ShaderPayloadSource, String> {
    use crate::render::streaming::shader::ShaderPayloadSource;
    if !blob_disk_backed {
        let bytes = ctx
            .read_payload(locator)
            .map_err(|e| e.to_string())?
            .to_vec();
        return Ok(ShaderPayloadSource::Bytes(bytes));
    }
    let (path, start) =
        crate::render::streaming::sources::blob_payload_section(locator.blob_index)?;
    Ok(ShaderPayloadSource::Disk {
        path,
        offset: start + locator.offset,
        len: locator.len,
    })
}

// How `decode_world` reads the world: whether content streams, whether the
// hot-reload source maps are gathered, and whether the blobs are files on disk.
#[derive(Clone, Copy)]
pub(super) struct WorldLoad {
    pub(super) streaming: bool,
    pub(super) capture_sources: bool,
    pub(super) blob_disk_backed: bool,
}

// Every payload the graphics load decoded before the blobs were released.
pub(super) struct WorldPayloads {
    pub(super) mesh: draw_list::MeshGeometry,
    pub(super) deferred_mesh_sources: draw_list::DeferredMeshSources,
    pub(super) proc_mesh_args_snapshot:
        hashbrown::HashMap<AssetId, (String, concinnity_core::components::ProceduralMesh)>,
    pub(super) model_map: hashbrown::HashMap<AssetId, Vec<SubMeshRef>>,
    pub(super) room_geometry: Vec<draw_list::RoomGeometry>,
    pub(super) terrains: Vec<LoadedTerrain>,
    pub(super) shaders: DecodedShaders,
    pub(super) textures: TexturePool,
    pub(super) texture_count: usize,
    pub(super) asset_source_map: TextureSourceMap,
    pub(super) texture_name_to_slot: hashbrown::HashMap<AssetId, usize>,
    pub(super) materials: DecodedMaterials,
    pub(super) skinned: SkinnedAssembly,
    pub(super) skinned_mesh_source_map: SkinnedMeshSourceMap,
    pub(super) texture_payloads: TexturePayloads,
    pub(super) env_map_bytes: Option<Vec<u8>>,
    pub(super) environment_map_source: Option<EnvironmentMapSource>,
    pub(super) env_map_background: bool,
    pub(super) color_lut_bytes: Option<Vec<u8>>,
    pub(super) color_lut_source: Option<ColorLutSource>,
    pub(super) text_atlas_data: Vec<Atlas>,
}

pub(super) struct DecodedShaders {
    pub(super) locators: Vec<PayloadLocator>,
    pub(super) source_map: ShaderSourceMap,
    // Hot-reloaded programs shared with the shader warmup, under capture.
    pub(super) overrides: crate::live_edit::shader_overrides::ShaderOverrides,
    // One entry per world Shader, in drain order == cook handle order, so a
    // baked ShaderHandle value indexes this directly. Entry 0 is the world
    // default pipeline's program.
    pub(super) shaders: Vec<DecodedShader>,
}

// The text/sprite atlas pool from `decode_text_atlases`: RGBA atlases (font
// atlases first, dense by FontHandle, then the built-in fallback face when some
// text names no Font, then appended sprite/story textures) and the blob indices
// the font payloads occupy (for the blob-release step).
pub(super) struct TextAtlases {
    pub(super) atlases: Vec<Atlas>,
    pub(super) font_blob_indices: Vec<u32>,
}

impl GraphicsSystem {
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
    pub(super) fn decode_shaders(
        &mut self,
        ctx: &mut PipelineContext<'_>,
        streaming: bool,
        capture_sources: bool,
    ) -> Option<DecodedShaders> {
        let (shader_ids, world_shaders): (Vec<Option<AssetId>>, Vec<Shader>) =
            ctx.drain_with_ids::<Shader>().into_iter().unzip();
        if world_shaders.is_empty() {
            return Some(DecodedShaders {
                locators: Vec::new(),
                shaders: vec![DecodedShader::default()],
                source_map: ShaderSourceMap::default(),
                overrides: Default::default(),
            });
        }

        // Buckets a non-start scene exclusively owns skip their decode and
        // pipeline build here; the streaming pump warms them when that scene
        // pins. The backend sees them flagged `deferred` and leaves the bucket's
        // pipeline unbuilt.
        self.shader_deferral = ShaderDeferral::plan(ctx, streaming, &shader_ids);
        let blob_disk_backed = ctx.blob.disk_backed();
        let mut deferred_sources = Vec::new();
        let overrides = crate::live_edit::shader_overrides::ShaderOverrides::default();

        let mut locators = Vec::with_capacity(world_shaders.len());
        let mut shaders = Vec::with_capacity(world_shaders.len());
        for (bucket, shader) in world_shaders.iter().enumerate() {
            let locator = if let Some(l) = &shader.locator {
                l.clone()
            } else {
                tracing::error!("GraphicsSystem: Shader has no compiled payload");
                return None;
            };
            if self
                .shader_deferral
                .is_deferred(u32::saturating_from(bucket))
            {
                match deferred_shader_source(ctx, &locator, blob_disk_backed) {
                    Ok(source) => {
                        deferred_sources.push(crate::render::streaming::shader::DeferredBucket {
                            bucket: u32::saturating_from(bucket),
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
                        tracing::warn!(
                            "GraphicsSystem: shader bucket {} cannot be deferred ({}); \
                             building it at init instead",
                            bucket,
                            e
                        );
                        self.shader_deferral
                            .build_at_init(u32::saturating_from(bucket));
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
            self.stream_workers.shaders =
                Some(crate::render::streaming::shader::ShaderWarmup::new(
                    deferred_sources,
                    capture_sources.then(|| overrides.clone()),
                ));
        }

        let source_map = if capture_sources {
            let assets_dir = self.assets_dir();
            ShaderSourceMap::build(
                shader_ids.iter().copied().zip(&world_shaders),
                |raw| {
                    concinnity_host::store::source::resolve_source_path(raw, assets_dir.as_deref())
                },
                concinnity_host::thread::asset_id::name_of,
            )
        } else {
            ShaderSourceMap::default()
        };

        Some(DecodedShaders {
            locators,
            source_map,
            overrides,
            shaders,
        })
    }

    // The sole EnvironmentMap's (handle 0) IBL payload, and, with
    // `capture_sources`, the resolved HDR source path and convolution sizing the
    // hot-reload watcher rebuilds it from (procedural generators have no file to
    // watch). None, failing init, if the payload is unreadable.
    pub(super) fn decode_environment_map(
        &self,
        ctx: &mut PipelineContext<'_>,
        capture_sources: bool,
    ) -> Option<(Option<Vec<u8>>, Option<EnvironmentMapSource>)> {
        let env_map_bytes = media::environment_map_payload(ctx)?;
        let mut environment_map_source = None;
        if capture_sources
            && let Some(info) = ctx
                .resource::<concinnity_core::resource::EnvironmentMapSources>()
                .and_then(|s| s.0.clone())
        {
            environment_map_source = Some(EnvironmentMapSource {
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
    pub(super) fn decode_color_lut(
        &self,
        ctx: &mut PipelineContext<'_>,
        capture_sources: bool,
    ) -> Option<(Option<Vec<u8>>, Option<ColorLutSource>)> {
        let color_lut_bytes = media::color_lut_payload(ctx)?;
        let mut color_lut_source = None;
        if capture_sources
            && let Some(src) = ctx
                .resource::<concinnity_core::resource::ColorLutSources>()
                .and_then(|c| c.0.clone())
        {
            color_lut_source = Some(ColorLutSource {
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
    pub(super) fn decode_text_atlases(
        &mut self,
        ctx: &mut PipelineContext<'_>,
        texture_locators: &[PayloadLocator],
    ) -> Option<TextAtlases> {
        let font_table = ctx.resource::<FontTable>().cloned().unwrap_or_default();
        let mut pool = AtlasPool::default();
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
                    let handle = pool.push_font(atlas);
                    self.loaded_fonts.insert(handle, face);
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
            let handle = pool.next_font();
            if let Some(builtin) = crate::render::builtin_font::load(handle) {
                pool.push_font(builtin.atlas);
                self.loaded_fonts.insert(handle, builtin.loaded);
                self.loaded_fonts.set_fallback(handle);
            } else {
                tracing::error!(
                    "GraphicsSystem: text naming no Font cannot draw: the built-in face failed to decode"
                );
            }
        }

        // Sprite textures ride the text-atlas pool, appended after the font
        // atlases and drawn by the same pipeline (positive vertex mode = RGBA
        // quad). A texture that cannot be resolved demotes its sprite to the
        // solid tint fill, warned rather than fatal.
        for tex_id in media::sprite_texture_handles(ctx) {
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
                Ok(bytes) => {
                    match texture::deserialize(bytes).and_then(texture::TextureImage::into_rgba8) {
                        Ok(atlas) => pool.push_sprite(tex_id, atlas),
                        Err(e) => {
                            tracing::warn!("GraphicsSystem: sprite texture {:?}: {}", tex_id, e);
                        }
                    }
                }
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
                pool.push_overlay(image);
            }
        }

        let font_blob_indices: Vec<u32> = font_table.blob_indices().into_iter().collect();
        self.sprite_texture_slots.extend(pool.sprite_slots);
        Some(TextAtlases {
            atlases: pool.atlases,
            font_blob_indices,
        })
    }

    // Decode every payload the world's graphics load reads, then release the
    // shared blobs nothing else holds. All payloads live in those blobs, so
    // each is consumed here before the release.
    pub(super) fn decode_world(
        &mut self,
        ctx: &mut PipelineContext<'_>,
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
            concinnity_core::render::environment::terrain::load_terrains(ctx)?;

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
        let env_map_background = media::environment_background(ctx)?;
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
}
