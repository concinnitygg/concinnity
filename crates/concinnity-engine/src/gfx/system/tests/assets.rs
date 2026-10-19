use super::*;
use concinnity_core::components::{
    Decal, GlassPanel, Grass, InstanceTransform, InstancedProp, ParticleEmitter, ReflectionProbe,
    SdfVolume, Story, StoryImage, StoryNode, StoryPage, StoryStage, Terrain, TerrainLayer,
    VolumetricFog, WaterSurface, Wind,
};
use concinnity_core::ecs::ResourceKind;
use concinnity_core::render::reflection_probe::ProbePlacement;
use concinnity_core::render::world_load::texture_payloads::TextureSlotSkips;
use concinnity_core::terrain::TerrainGrid;
use concinnity_core::terrain::payload::TerrainPayload;
use concinnity_host::thread::asset_id;
use hashbrown::HashSet;
use std::collections::BTreeSet;

// A w x h font atlas in the compiled payload format, carrying one glyph so the
// cap-height derivation has metrics to read.
fn font_payload(w: u32, h: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    for v in [w, h, 1, 32] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes.extend(vec![0xFFu8; (w * h * 4) as usize]);
    // One glyph: 'H', whose height drives `derive_cap_px`.
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&(b'H' as u32).to_le_bytes());
    for v in [0u16, 0, 16, 22] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in [18.0f32, 1.0, 22.0] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

impl WorldBuilder {
    // Register a resource record of `kind` at the next handle for that kind,
    // with `bytes` as its compiled payload, and return the handle.
    fn push_resource(&mut self, kind: concinnity_core::ecs::ResourceKind, bytes: &[u8]) -> u32 {
        let locator = self.payload(bytes);
        let handle = self.kind_records(kind).len() as u32;
        self.kind_records(kind)
            .push(concinnity_core::ecs::ResourceRecord {
                resource_kind: kind,
                handle,
                payload: Some(locator),
                data_bytes: Vec::new(),
            });
        handle
    }

    // The record list backing a resource kind. Only the kinds the media tests
    // below author are routed; the quad builder owns mesh / texture / material.
    fn kind_records(
        &mut self,
        kind: concinnity_core::ecs::ResourceKind,
    ) -> &mut Vec<concinnity_core::ecs::ResourceRecord> {
        use concinnity_core::ecs::ResourceKind;
        match kind {
            ResourceKind::Texture => &mut self.texture_records,
            ResourceKind::Font => &mut self.font_records,
            ResourceKind::ColorLut => &mut self.color_lut_records,
            ResourceKind::EnvironmentMap => &mut self.env_map_records,
            ResourceKind::SkinnedMesh => &mut self.skinned_records,
            other => panic!("no record list wired for {other:?}"),
        }
    }
}

// A world assembled in code has no compiled Font for its labels to name, so the
// renderer registers the engine's built-in face and resolves those labels to it.
// Without the fallback the text draws nothing at all.
#[test]
fn text_naming_no_font_falls_back_to_the_built_in_face() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push_identified(
        AssetId(800),
        TextLabel {
            content: "Hello, world!".to_string(),
            font: None,
            visible: true,
            ..Default::default()
        },
    );
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    let labels: Vec<TextLabel> = world.context().query::<TextLabel>().cloned().collect();
    let overlay = world
        .resource::<concinnity_core::render::overlay::OverlayAssets>()
        .expect("OverlayAssets parked at init");
    let face = overlay
        .fonts
        .resolve(None)
        .expect("a label naming no font has a face to draw with");

    // The whole point: the label shapes into real glyph geometry rather than
    // being dropped for want of a font.
    let calls = text::build_text_calls(&labels, &overlay.fonts, [640.0, 360.0]);
    assert_eq!(calls.len(), 1, "one draw call for the one label");
    // The space carries no quad of its own; every other character draws one.
    let glyphs = "Hello, world!".chars().filter(|c| *c != ' ').count();
    assert_eq!(
        calls[0].vertices.len(),
        glyphs * 4,
        "four vertices per glyph quad"
    );
    assert_eq!(calls[0].atlas_slot, face.atlas_slot);

    // The face's atlas rides the shared pool like any other font's.
    let s = lock(&state);
    assert_eq!(
        s.init.as_ref().unwrap().text_atlas_count,
        1,
        "the built-in atlas is uploaded for the font-less label"
    );
    assert_eq!(
        face.atlas_slot, 0,
        "it takes the slot after the world fonts"
    );
}

// A world whose text all names a Font pays nothing for the fallback: no extra
// atlas is uploaded, and nothing resolves without a handle.
#[test]
fn a_world_whose_text_names_its_fonts_registers_no_fallback() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    let font = b.push_resource(ResourceKind::Font, &font_payload(32, 32));
    b.push_identified(
        AssetId(800),
        TextLabel {
            content: "Hello, world!".to_string(),
            font: Some(FontHandle::new(font)),
            visible: true,
            ..Default::default()
        },
    );
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    let overlay = world
        .resource::<concinnity_core::render::overlay::OverlayAssets>()
        .expect("OverlayAssets parked at init");
    assert_eq!(overlay.fonts.len(), 1);
    assert!(overlay.fonts.resolve(None).is_none());

    let s = lock(&state);
    assert_eq!(
        s.init.as_ref().unwrap().text_atlas_count,
        1,
        "the world's own font atlas, and no second one"
    );
}

// Font atlases and sprite textures share the text-atlas pool: each font's atlas
// takes its handle's leading slot, and each distinct Texture a Sprite references
// is decoded and appended after them (drawn by the same pipeline).
#[test]
fn fonts_and_sprite_textures_share_the_text_atlas_pool() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push_resource(ResourceKind::Font, &font_payload(32, 32));
    b.push_resource(ResourceKind::Font, &font_payload(64, 64));
    // A second texture (handle 1; the quad's albedo owns handle 0) for a Sprite
    // to reference. Two sprites share it, so it is decoded once.
    let sprite_tex = b.push_resource(ResourceKind::Texture, &texture_payload(4, 4));
    for id in [800u32, 801] {
        b.push_identified(
            AssetId(id),
            Sprite {
                texture: Some(TextureHandle::new(sprite_tex)),
                visible: true,
                ..Default::default()
            },
        );
    }
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    let s = lock(&state);
    assert_eq!(
        s.init.as_ref().unwrap().text_atlas_count,
        3,
        "two font atlases plus the one deduped sprite texture"
    );
    drop(s);

    let overlay = world
        .resource::<concinnity_core::render::overlay::OverlayAssets>()
        .expect("OverlayAssets parked at init");
    // A font's handle IS its atlas slot, so the pool's leading slots are the
    // fonts in handle order.
    assert_eq!(overlay.fonts.len(), 2);
    let face = |h| overlay.fonts.get(FontHandle::new(h)).expect("loaded");
    assert_eq!(face(0).atlas_slot, 0, "font handle 0 owns atlas slot 0");
    assert_eq!(face(1).atlas_slot, 1);
    assert_eq!(face(1).atlas_w, 64);
    assert_eq!(face(0).size_px, 32.0);
    // The sprite texture lands after the fonts.
    assert_eq!(
        overlay
            .sprite_texture_slots
            .get(&TextureHandle::new(sprite_tex)),
        Some(&2),
        "the sprite texture is appended after the font atlases"
    );
}

// A Sprite pointing at a texture that is not in the pool demotes to its solid
// tint fill rather than failing the world build.
#[test]
fn a_sprite_with_an_unknown_texture_keeps_its_tint() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push_identified(
        AssetId(810),
        Sprite {
            texture: Some(TextureHandle::new(99)),
            visible: true,
            ..Default::default()
        },
    );
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(!gs.failed, "an unresolvable sprite texture is not fatal");
    assert_eq!(
        lock(&state).init.as_ref().unwrap().text_atlas_count,
        0,
        "nothing was appended to the text-atlas pool"
    );
    assert!(
        world
            .resource::<concinnity_core::render::overlay::OverlayAssets>()
            .unwrap()
            .sprite_texture_slots
            .is_empty()
    );
}

// A malformed Font payload fails the world build rather than rendering blank
// text: the atlas is a build product, so a bad one means the build is broken.
#[test]
fn a_malformed_font_payload_fails_init() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push_resource(ResourceKind::Font, b"not-a-font");
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(gs.failed);
    assert!(!backend_parked(&world));
    assert!(lock(&state).init.is_none(), "backend never constructed");
}

// A texture payload that does not deserialize fails the world build, like a
// malformed Font: the pool is a build product, so bad bytes mean a broken build.
#[test]
fn a_malformed_texture_payload_fails_init() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    let bad = b.payload(b"not-a-texture");
    b.texture_records[0].payload = Some(bad);
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(gs.failed);
    assert!(!backend_parked(&world));
    assert!(lock(&state).init.is_none(), "backend never constructed");
}

// A deferred slot enters the pool as a 1x1 placeholder. Disk-backed, it keeps no
// payload bytes (the streamer re-reads the blob file); RAM-backed, every slot's
// bytes are kept for the streamer to re-decode.
#[test]
fn a_deferred_texture_slot_decodes_to_a_placeholder() {
    let mut b = scene_builder();
    b.push_resource(
        concinnity_core::ecs::ResourceKind::Texture,
        &texture_payload(4, 4),
    );
    let locators: Vec<PayloadLocator> = b
        .texture_records
        .iter()
        .filter_map(|r| r.payload.clone())
        .collect();
    let mut world = b.build();
    let deferred = HashSet::from([1]);
    let cook_only = std::collections::BTreeSet::new();
    let skips = || TextureSlotSkips {
        deferred: &deferred,
        cook_only: &cook_only,
    };

    let disk = concinnity_core::render::world_load::texture_payloads::decode_texture_payloads(
        &mut world.context(),
        &locators,
        skips(),
        true,
    )
    .expect("decodes");
    assert_eq!(disk.images.len(), 2);
    assert_eq!((disk.images[0].width(), disk.images[0].height()), (2, 2));
    assert_eq!((disk.images[1].width(), disk.images[1].height()), (1, 1));
    assert!(disk.payloads.is_empty());

    let ram = concinnity_core::render::world_load::texture_payloads::decode_texture_payloads(
        &mut world.context(),
        &locators,
        skips(),
        false,
    )
    .expect("decodes");
    assert_eq!((ram.images[1].width(), ram.images[1].height()), (1, 1));
    assert_eq!(
        ram.payloads,
        vec![texture_payload(2, 2), texture_payload(4, 4)]
    );
}

// A texture only the cook reads enters the pool as a 1x1 placeholder whether
// or not its scene is resident, and its payload is never read: RAM-backed, its
// slot keeps an empty entry so the streamer's slots stay dense.
#[test]
fn a_cook_only_texture_slot_stays_a_placeholder() {
    let mut b = scene_builder();
    b.push_resource(
        concinnity_core::ecs::ResourceKind::Texture,
        &texture_payload(4, 4),
    );
    let locators: Vec<PayloadLocator> = b
        .texture_records
        .iter()
        .filter_map(|r| r.payload.clone())
        .collect();
    let mut world = b.build();
    let cook_only = BTreeSet::from([0]);
    let skips = TextureSlotSkips {
        deferred: &HashSet::new(),
        cook_only: &cook_only,
    };
    let ram = concinnity_core::render::world_load::texture_payloads::decode_texture_payloads(
        &mut world.context(),
        &locators,
        skips,
        false,
    )
    .expect("decodes");
    assert_eq!((ram.images[0].width(), ram.images[0].height()), (1, 1));
    assert_eq!((ram.images[1].width(), ram.images[1].height()), (4, 4));
    assert_eq!(ram.payloads, vec![Vec::new(), texture_payload(4, 4)]);
}

// The EnvironmentMap and ColorLut payloads are read from their resource tables
// before the shared blob is released and handed to the backend. Both are
// singletons: the runtime uses handle 0 and logs any extras.
#[test]
fn environment_map_and_color_lut_payloads_survive_to_the_backend() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push_resource(ResourceKind::EnvironmentMap, b"ibl-cube-bytes");
    // A second map: extras are ignored, not fatal.
    b.push_resource(ResourceKind::EnvironmentMap, b"second-ibl");
    b.push_resource(ResourceKind::ColorLut, b"lut-bytes");
    b.push_resource(ResourceKind::ColorLut, b"second-lut");
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(!gs.failed, "extra singletons are ignored, not fatal");
    assert!(lock(&state).init.is_some(), "the backend was built");
}

// An EnvironmentMap whose locator does not resolve fails the world build: the
// IBL environment is a build product, so a payload that cannot be read means the
// build is broken rather than that the world wants no environment.
#[test]
fn an_unreadable_environment_map_payload_fails_init() {
    let (_state, hooks) = recording_hooks();
    let mut b = scene_builder();
    // A locator past the end of the blob's only section.
    b.env_map_records
        .push(concinnity_core::ecs::ResourceRecord {
            resource_kind: concinnity_core::ecs::ResourceKind::EnvironmentMap,
            handle: 0,
            payload: Some(PayloadLocator {
                blob_index: 0,
                offset: 1 << 30,
                len: 16,
            }),
            data_bytes: Vec::new(),
        });
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(gs.failed);
    assert!(!backend_parked(&world));
}

// A world's VolumetricFog resolves into the backend's fog pass. The first
// declared instance wins (one homogeneous medium is all the pass models).
#[test]
fn volumetric_fog_resolves_into_the_backend_init() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push(VolumetricFog {
        enabled: true,
        density: 0.05,
        ..Default::default()
    });
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(!gs.failed);
    assert!(
        lock(&state).init.as_ref().unwrap().fog,
        "the fog pass is on"
    );
    // The fog a world reload dedupes against is parked beside the backend,
    // seeded from whatever was passed into the backend constructor.
    let pushed = world.resource::<crate::gfx::system::parked::PushedFogSettings>();
    assert!(pushed.is_some_and(|fog| fog.0.is_some()));
}

// A `VolumetricFog` with `enabled = false` yields no fog pass at all, the same
// as declaring none: the renderer skips the pass rather than running it at zero
// density.
#[test]
fn disabled_volumetric_fog_skips_the_fog_pass() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push(VolumetricFog {
        enabled: false,
        ..Default::default()
    });
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(!gs.failed);
    assert!(!lock(&state).init.as_ref().unwrap().fog);
    let pushed = world.resource::<crate::gfx::system::parked::PushedFogSettings>();
    assert!(pushed.is_some_and(|fog| fog.0.is_none()));
}

// A world's declared ReflectionProbes are handed to the backend as placements,
// replacing the auto-seed a probe-less world falls back to.
#[test]
fn declared_reflection_probes_replace_the_auto_seed() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    let probes = [(0.0, None), (10.0, Some(6.0))].map(|(x, capture_distance)| ReflectionProbe {
        position: [x, 1.0, 0.0],
        half_extents: [4.0; 3],
        capture_distance,
        ..Default::default()
    });
    for probe in &probes {
        b.push(probe.clone());
    }
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(!gs.failed);
    assert!(
        lock(&state).saw(&Call::SetReflectionProbes(
            probes.iter().map(ProbePlacement::from).collect()
        )),
        "both declared placements reach the backend, capture distance and all"
    );
    // Read, not drained: draining the only component on a probe entity would
    // despawn it, and the editor's billboard drive can only address a live one.
    assert_eq!(
        world.context().query::<ReflectionProbe>().count(),
        2,
        "the probes stay resident after their placements reach the backend"
    );
}

// An InstancedProp bakes its instances into one GPU cluster rather than a draw
// object each, and the component is drained (there is no per-frame update path).
#[test]
fn instanced_prop_bakes_its_instances_into_one_cluster() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push_identified(
        AssetId(820),
        InstancedProp {
            mesh: Some(MeshHandle::new(0)),
            material: Some(MaterialHandle::new(0)),
            instances: (0..3)
                .map(|i| InstanceTransform {
                    position: [i as f32 * 2.0, 0.0, 0.0],
                    rotation_deg: [0.0; 3],
                    scale: [1.0; 3],
                })
                .collect(),
            ..Default::default()
        },
    );
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(!gs.failed);
    let s = lock(&state);
    let init = s.init.as_ref().unwrap();
    assert_eq!(init.instanced_cluster_count, 1, "one mesh = one cluster");
    assert_eq!(
        init.draw_objects.len(),
        1,
        "the instances ride the cluster, not one draw object each"
    );
    drop(s);
    assert_eq!(
        world.context().query::<InstancedProp>().count(),
        0,
        "drained"
    );
}

// The one-shot world FX (decals, emitters, water, glass, SDF volumes, grass
// and its wind) are
// resolved at init and drained: each is baked into a backend record at
// construction and has no per-frame update path, so leaving the components
// behind would only invite a second, stale build. The terrain the grass grows
// on stays, since physics builds its collider after graphics init.
#[test]
fn one_shot_world_fx_are_resolved_and_drained_at_init() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.push_identified(
        AssetId(830),
        Decal {
            texture: Some(TextureHandle::new(0)),
            size: [1.0; 3],
            visible: true,
            ..Default::default()
        },
    );
    b.push_identified(
        AssetId(831),
        ParticleEmitter {
            texture: Some(TextureHandle::new(0)),
            max_particles: 16,
            visible: true,
            ..Default::default()
        },
    );
    b.push(WaterSurface::default());
    b.push(GlassPanel::default());
    b.push(Wind::default());
    b.push_identified(AssetId(834), Grass::default());
    let ground = b.payload(
        &TerrainPayload {
            grid: TerrainGrid::new(4, [8.0, 8.0], vec![0.0; 25]).unwrap(),
            masks: vec![None],
        }
        .encode(),
    );
    b.push(Terrain {
        extent: [8.0, 8.0],
        resolution: 4,
        layers: vec![TerrainLayer {
            grass: concinnity_core::ecs::Ref::new(AssetId(834)),
            density_mask: None,
        }],
        locator: Some(ground),
        ..Default::default()
    });
    let sdf_frag = b.payload(b"sdf-fragment-bytes");
    b.push_identified(
        AssetId(832),
        SdfVolume {
            extent: [2.0; 3],
            locator: Some(sdf_frag),
            visible: true,
            ..Default::default()
        },
    );
    // A volume whose fragment shader never compiled is skipped with a warning
    // rather than failing the whole world build.
    b.push_identified(
        AssetId(833),
        SdfVolume {
            extent: [2.0; 3],
            locator: None,
            visible: true,
            ..Default::default()
        },
    );
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(!gs.failed, "a payload-less SdfVolume is not fatal");
    assert!(lock(&state).init.is_some());
    let ctx = world.context();
    assert_eq!(ctx.query::<Decal>().count(), 0);
    assert_eq!(ctx.query::<ParticleEmitter>().count(), 0);
    assert_eq!(ctx.query::<WaterSurface>().count(), 0);
    assert_eq!(ctx.query::<GlassPanel>().count(), 0);
    assert_eq!(ctx.query::<SdfVolume>().count(), 0);
    assert_eq!(ctx.query::<Wind>().count(), 0);
    assert_eq!(ctx.query::<Grass>().count(), 0);
    assert_eq!(ctx.query::<Terrain>().count(), 1);
    assert!(
        lock(&state).init.as_ref().unwrap().grass,
        "the grass field reaches the backend"
    );
}

// An SDF volume with no compiled payload, or one whose payload cannot be read, is
// skipped; a readable one keeps its bytes and its asset name as the label.
#[test]
fn sdf_volumes_without_a_readable_payload_are_skipped() {
    let mut b = WorldBuilder::new();
    let named = asset_id::intern("glass_orb");
    let frag = b.payload(b"sdf-fragment-bytes");
    b.push_identified(
        named,
        SdfVolume {
            locator: Some(frag),
            ..Default::default()
        },
    );
    b.push_identified(
        AssetId(841),
        SdfVolume {
            locator: None,
            ..Default::default()
        },
    );
    b.push_identified(
        AssetId(842),
        SdfVolume {
            locator: Some(PayloadLocator {
                blob_index: 7,
                offset: 0,
                len: 4,
            }),
            ..Default::default()
        },
    );
    let mut world = b.build();

    let volumes = concinnity_core::render::world_load::world_fx::drain_sdf_volumes(
        &mut world.context(),
        concinnity_host::thread::asset_id::name_of,
    );
    assert_eq!(volumes.len(), 1);
    assert_eq!(volumes[0].fragment_source, b"sdf-fragment-bytes");
    assert_eq!(volumes[0].label, "glass_orb");
    assert_eq!(world.context().query::<SdfVolume>().count(), 0, "drained");
}

// A backend factory that cannot build (no device, an unsupported surface) leaves
// the system failed rather than stepping on with no backend.
#[test]
fn a_backend_that_fails_to_build_marks_graphics_failed() {
    let (state, mut hooks) = recording_hooks();
    hooks.backend_factory = Box::new(|_init| {
        Err(error::RenderError::OutOfDeviceMemory(
            "no device memory".to_string(),
        ))
    });
    let mut world = scene_builder().build();
    let gs = init_graphics(&mut world, hooks);

    assert!(gs.failed);
    assert!(!backend_parked(&world));
    assert!(lock(&state).init.is_none());
    // A failed system stops the run rather than drawing. `Done` would retire
    // only this system and leave the world stepping every other one forever,
    // with no window and no frame pacing.
    let mut gs = gs;
    assert_eq!(step(&mut gs, &mut world), StepResult::Stop);
}

// A build with no backend compiled in reports `Unsupported` from the factory,
// which reaches init as any other build failure does.
#[test]
fn an_unsupported_backend_marks_graphics_failed() {
    let (_state, mut hooks) = recording_hooks();
    hooks.backend_factory =
        Box::new(|_init| Err(error::RenderError::Unsupported { op: "init_backend" }));
    let mut world = scene_builder().build();
    let gs = init_graphics(&mut world, hooks);

    assert!(gs.failed);
    assert!(!backend_parked(&world));
}

impl WorldBuilder {
    // Bake a Material into the resource stream at the next handle and return it,
    // the way cook does (Materials are a data resource; all their data lives in
    // the baked bytes).
    fn push_material(&mut self, mat: Material) -> MaterialHandle {
        let handle = self.material_records.len() as u32;
        self.material_records
            .push(concinnity_core::ecs::ResourceRecord {
                resource_kind: concinnity_core::ecs::ResourceKind::Material,
                handle,
                payload: None,
                data_bytes: postcard::to_allocvec(&mat).unwrap(),
            });
        MaterialHandle::new(handle)
    }
}

// Every texture -- albedo, normal map, emissive, packed ORM -- lives once in
// the shared pool at its own slot, so each reference resolves through the same
// handle-indexed lookup rather than a per-role pool. An unset reference falls
// back to its sentinel: slot 0 for the albedo-region maps (which the shader
// gates on) and the flat-normal fallback for the normal map.
#[test]
fn every_material_texture_reference_resolves_to_its_shared_pool_slot() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    // Three more textures beside the quad's albedo (handle 0), so each reference
    // below points at a distinct slot.
    let slots: Vec<u32> = (0..4)
        .map(|_| b.push_resource(ResourceKind::Texture, &texture_payload(2, 2)))
        .collect();
    let mat = b.push_material(Material {
        albedo: Some(TextureHandle::new(slots[0])),
        normal_map: Some(TextureHandle::new(slots[1])),
        emissive_map: Some(TextureHandle::new(slots[2])),
        orm_map: Some(TextureHandle::new(slots[3])),
        ..Default::default()
    });
    b.push_identified(
        AssetId(850),
        Prop {
            mesh: Some(MeshHandle::new(0)),
            material: Some(mat),
            ..Default::default()
        },
    );
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    let s = lock(&state);
    let init = s.init.as_ref().unwrap();
    assert_eq!(init.texture_count, 5, "one pool holds every texture");
    let draw = init
        .draw_objects
        .iter()
        .find(|d| d.texture_slot == slots[0] as usize)
        .expect("the prop's draw resolved its albedo to the pool slot");
    assert_eq!(
        draw.normal_map_slot, slots[1] as usize,
        "a normal map resolves to the same shared pool, not a separate one"
    );
    assert_eq!(draw.material.emissive_map_index, slots[2]);
    assert_eq!(draw.material.orm_map_index, slots[3]);
}

// A material whose texture handle is past the end of the pool is a resolution
// error: cook validates every reference exists, so this only fires on a corrupt
// build, and failing loudly beats sampling an unrelated texture. Each reference
// role guards independently.
#[test]
fn a_material_texture_handle_past_the_pool_fails_init() {
    let out_of_range = Some(TextureHandle::new(99));
    let cases: Vec<(&str, Material)> = vec![
        (
            "albedo",
            Material {
                albedo: out_of_range,
                ..Default::default()
            },
        ),
        (
            "normal_map",
            Material {
                normal_map: out_of_range,
                ..Default::default()
            },
        ),
        (
            "emissive_map",
            Material {
                emissive_map: out_of_range,
                ..Default::default()
            },
        ),
        (
            "orm_map",
            Material {
                orm_map: out_of_range,
                ..Default::default()
            },
        ),
    ];
    for (role, mat) in cases {
        let (_state, hooks) = recording_hooks();
        let mut b = scene_builder();
        b.push_material(mat);
        let mut world = b.build();
        assert!(
            init_graphics(&mut world, hooks).failed,
            "an out-of-range {role} handle must fail the build"
        );
    }
}

// A Material whose baked bytes do not decode fails the world build: the record
// is a build product, so a corrupt one means the build is broken.
#[test]
fn an_undecodable_material_record_fails_init() {
    let (_state, hooks) = recording_hooks();
    let mut b = scene_builder();
    b.material_records
        .push(concinnity_core::ecs::ResourceRecord {
            resource_kind: concinnity_core::ecs::ResourceKind::Material,
            handle: 1,
            payload: None,
            data_bytes: vec![0xFF; 2],
        });
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(gs.failed);
    assert!(!backend_parked(&world));
}

// The instanced vertex stage is optional -- it is only needed once a world
// declares an InstancedProp -- so its bytes are read and passed through when
// the shader's container carries the stage.
#[test]
fn an_instanced_vertex_shader_payload_reaches_the_backend() {
    let (state, hooks) = recording_hooks();
    // scene_builder()'s shader carries no instanced stage, so build the same
    // world with a three-stage shader instead.
    let mut b = WorldBuilder::new();
    b.push(Window {
        title: "mock world".to_string(),
        width: 640,
        height: 360,
        ..Default::default()
    });
    b.push(GraphicsConfig {
        background: BackgroundOverrides {
            color: Some([0.1, 0.2, 0.3, 1.0]),
        },
        ..Default::default()
    });
    b.push_shader(&["vertex_main_bindless", "fragment_main_bindless"]);
    b.push(Camera3D::bake(Default::default()));
    b.push_textured_quad(MESH, TEX, MAT, PROP);
    b.push_identified(
        AssetId(851),
        InstancedProp {
            mesh: Some(MeshHandle::new(0)),
            material: Some(MaterialHandle::new(0)),
            instances: vec![InstanceTransform {
                position: [0.0; 3],
                rotation_deg: [0.0; 3],
                scale: [1.0; 3],
            }],
            ..Default::default()
        },
    );
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);

    assert!(!gs.failed);
    assert_eq!(
        lock(&state).init.as_ref().unwrap().instanced_cluster_count,
        1
    );
}

// A Shader that never compiled (no payload locator) fails the world build
// rather than starting with a pipeline that cannot draw.
#[test]
fn a_shader_without_a_payload_fails_init() {
    let (_state, hooks) = recording_hooks();
    let mut b = WorldBuilder::new();
    b.push(Window::default());
    b.push(Shader {
        locator: None,
        ..Default::default()
    });
    let mut world = b.build();
    assert!(
        init_graphics(&mut world, hooks).failed,
        "a Shader with no compiled payload must fail the build"
    );
}

// A Story's stage images must be resident even though no Sprite references them
// yet: the story system swaps them onto the stage sprites at runtime, so they are
// gathered into the text-atlas pool alongside the sprite textures.
#[test]
fn story_stage_images_are_resident_before_any_sprite_references_them() {
    let (state, hooks) = recording_hooks();
    let mut b = scene_builder();
    // Four textures beside the quad's albedo, one per stage position.
    let slots: Vec<u32> = (0..4)
        .map(|_| b.push_resource(ResourceKind::Texture, &texture_payload(2, 2)))
        .collect();
    let image = |slot: u32| {
        Some(StoryImage {
            texture: TextureHandle::new(slot),
            ..Default::default()
        })
    };
    b.push_identified(
        AssetId(870),
        Story {
            nodes: vec![StoryNode {
                pages: vec![StoryPage {
                    stage: StoryStage {
                        bg: image(slots[0]),
                        left: image(slots[1]),
                        ..Default::default()
                    },
                    ..Default::default()
                }],
                // A choice menu carries its own stage dressing too.
                choice_stage: StoryStage {
                    center: image(slots[2]),
                    right: image(slots[3]),
                    ..Default::default()
                },
                ..Default::default()
            }],
            ..Default::default()
        },
    );
    let mut world = b.build();
    let gs = init_graphics(&mut world, hooks);
    assert!(!gs.failed);

    assert_eq!(
        lock(&state).init.as_ref().unwrap().text_atlas_count,
        4,
        "every stage image is decoded into the pool ahead of the swap"
    );
    let overlay = world
        .resource::<concinnity_core::render::overlay::OverlayAssets>()
        .unwrap();
    for slot in &slots {
        assert!(
            overlay
                .sprite_texture_slots
                .contains_key(&TextureHandle::new(*slot)),
            "stage texture {slot} never became resident"
        );
    }
}
