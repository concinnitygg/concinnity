//! The world's one-shot effect content, drained from its components at init.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::components::{
    Atmosphere, Decal, GlassPanel, Grass, NightSky, ParticleEmitter, Planet, SdfVolume,
    VolumetricFog, WaterShape, WaterSurface, Wind,
};
use crate::ecs::PipelineContext;
use crate::ecs::asset_id::AssetId;
use crate::planet::PlanetShape;
use crate::render::backend::init::{SdfVolumeSource, WorldFx};
use crate::render::environment::atmosphere::AtmosphereSettings;
use crate::render::environment::grass::{CanopyField, GrassField, GrassTerrain, PlanetGrass};
use crate::render::environment::ground::GroundCovers;
use crate::render::environment::night_sky::NightSkySettings;
use crate::render::environment::terrain::{LoadedTerrain, grass_terrains};
use crate::render::{decal, environment::volumetric_fog, particles};
use crate::water::WaterBody;
use crate::water::ocean::OceanQuality;
use crate::water::sea::shore::ShoreWaves;
use crate::wind::WindField;

/// What a planet's ground is laid with: the grass its layers grow and the
/// covers over it.
#[derive(Debug, Clone, Default)]
pub struct PlanetSurface {
    /// The grass, `None` when no layer grows any.
    pub grass: Option<PlanetGrass>,
    /// The covers.
    pub covers: GroundCovers,
}

/// Drain every effect component into the backend's effect inputs. Each record
/// is resolved once here (texture slots, inverted decal matrices, clamped
/// emitter tunables): the runtime keeps no per-frame update path for them.
/// `name_of` labels an SDF volume by its asset's name; `planet` is the
/// world's planet with what its ground is laid with.
pub fn drain_world_fx(
    ctx: &mut PipelineContext<'_>,
    name_of: fn(AssetId) -> Option<String>,
    texture_count: usize,
    terrains: &[LoadedTerrain],
    planet: Option<(&Planet, PlanetSurface)>,
    ocean_quality: OceanQuality,
) -> WorldFx {
    let decals: Vec<Decal> = ctx.drain::<Decal>();
    let decals = decal::build_decal_records(&decals.iter().collect::<Vec<_>>(), texture_count);
    let emitters = ctx.drain_with_ids::<ParticleEmitter>();
    let emitters: Vec<_> = emitters.iter().map(|(id, e)| (*id, e)).collect();
    let particles = particles::build_particle_records(&emitters, texture_count);
    let glass_panels = ctx.drain::<GlassPanel>();
    let sdf_volumes = drain_sdf_volumes(ctx, name_of);
    // The first enabled fog wins: the fog pass models one homogeneous medium.
    // `None` also covers a fog whose density cannot affect the frame.
    let fog = ctx
        .drain::<VolumetricFog>()
        .into_iter()
        .find(|f| f.enabled)
        .and_then(|f| volumetric_fog::resolve_asset(&f));
    // One wind per world, which every terrain's grass sways in and every
    // water surface takes its waves from.
    let wind = ctx.drain::<Wind>().into_iter().next();
    let wind_field = WindField::new(wind.as_ref());
    let (planet, surface) = planet.unzip();
    let planet_shape = planet.map(Planet::shape);
    let surfaces = ctx.drain::<WaterSurface>();
    if planet_shape.is_none() && surfaces.iter().any(|s| s.shape == WaterShape::PlanetSea) {
        tracing::warn!(
            "a WaterSurface is a planet_sea but the world has no Planet; it is \
             not drawn"
        );
    }
    let (water_surfaces, ocean) =
        WaterBody::resolve_all(&surfaces, &wind_field, ocean_quality, planet_shape.as_ref());
    let grass: hashbrown::HashMap<AssetId, Grass> = ctx
        .drain_with_ids::<Grass>()
        .into_iter()
        .filter_map(|(id, g)| Some((id?, g)))
        .collect();
    let grass_terrains = grass_terrains(terrains, &grass);
    let terrain_grass = GrassField::resolve(&grass_terrains, wind.as_ref());
    // The planet's beaches are wet where its sea's waves wash up them.
    let shore = water_surfaces
        .iter()
        .find_map(|b| b.sea.as_ref()?.shore.clone());
    let canopy = scene_canopy(
        planet_shape.zip(surface.as_ref()),
        terrain_grass.is_some().then_some(&grass_terrains[..]),
        shore,
        wind_field.strength,
    );
    // One field per world: a planet's grass, or the terrains'.
    let grass = match surface.and_then(|s| s.grass) {
        Some(planet) => {
            if terrain_grass.is_some() {
                tracing::warn!(
                    "a world grows grass on a Planet or on Terrains, not both; \
                     the Terrains' grass is not drawn"
                );
            }
            Some(GrassField::resolve_planet(planet, wind.as_ref()))
        }
        None => terrain_grass,
    };
    // One atmosphere per world, wrapped around the world's planet if it has
    // one. A night sky with no atmosphere is seen through a vacuum.
    let planet = planet.map(|p| (p.center.map(f64::from), p.radius));
    let night = ctx
        .drain::<NightSky>()
        .into_iter()
        .next()
        .map(|n| NightSkySettings::resolve(&n));
    let atmosphere = match ctx.drain::<Atmosphere>().into_iter().next() {
        Some(a) => Some(AtmosphereSettings::resolve(&a, planet)),
        None => night.map(|_| AtmosphereSettings::vacuum(planet)),
    }
    .map(|s| s.with_night(night));
    WorldFx {
        decals,
        particles,
        fog,
        water_surfaces,
        ocean,
        glass_panels,
        sdf_volumes,
        grass,
        canopy,
        atmosphere,
    }
}

// The scene's canopy block: the planet's ground (`planet`) when it grows
// grass, lays covers or has a `shore` washing up it, and otherwise the far
// fields of the grass `terrains` grow, if any, in a wind of mean
// `wind_speed`.
fn scene_canopy(
    planet: Option<(PlanetShape, &PlanetSurface)>,
    terrains: Option<&[GrassTerrain]>,
    shore: Option<ShoreWaves>,
    wind_speed: f32,
) -> Option<CanopyField> {
    let laid = planet
        .filter(|(_, ground)| ground.grass.is_some() || ground.covers.count > 0 || shore.is_some());
    let mut field = match (laid, terrains) {
        (Some((_, ground)), Some(terrains)) if ground.grass.is_none() => {
            tracing::warn!(
                "the Planet's covers and wet sand are not drawn beside Terrains growing grass"
            );
            CanopyField::for_terrains(terrains, wind_speed).0
        }
        (Some((shape, ground)), _) => {
            CanopyField::for_planet(shape, ground.grass.as_ref(), ground.covers, wind_speed).0
        }
        (None, Some(terrains)) => CanopyField::for_terrains(terrains, wind_speed).0,
        (None, None) => return None,
    };
    field.set_shore(shore);
    Some(field)
}

/// Drain the raymarched SDF volumes with their compiled payloads. A volume with
/// no payload, or one whose payload cannot be read, is skipped with a warning
/// rather than failing the world build. `name_of` labels each by its asset's
/// name.
fn drain_sdf_volumes(
    ctx: &mut PipelineContext<'_>,
    name_of: fn(AssetId) -> Option<String>,
) -> Vec<SdfVolumeSource> {
    let raw: Vec<(Option<AssetId>, SdfVolume)> = ctx.drain_with_ids::<SdfVolume>();
    let mut out = Vec::with_capacity(raw.len());
    for (i, (id, volume)) in raw.into_iter().enumerate() {
        let label = id
            .and_then(name_of)
            .unwrap_or_else(|| format!("sdf_volume_{i}"));
        let Some(locator) = volume.locator.clone() else {
            tracing::warn!(
                "SdfVolume '{}': no payload locator (fragment shader never compiled); skipping",
                label
            );
            continue;
        };
        match ctx.read_payload(&locator) {
            Ok(bytes) => {
                let fragment_source = bytes.to_vec();
                out.push(SdfVolumeSource {
                    volume,
                    fragment_source,
                    label,
                });
            }
            Err(e) => tracing::warn!(
                "SdfVolume '{}': failed to read fragment shader payload: {:?}; skipping",
                label,
                e
            ),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{Terrain, TerrainLayer};
    use crate::ecs::{PayloadLocator, Ref, TextureHandle, World};
    use crate::terrain::TerrainGrid;
    use crate::terrain::payload::TerrainPayload;
    use crate::test_support::OneSection;
    use alloc::boxed::Box;
    use alloc::string::ToString;
    use alloc::vec;

    const ORB: AssetId = AssetId(840);
    const GRASS: AssetId = AssetId(834);

    fn name_of(id: AssetId) -> Option<String> {
        (id == ORB).then(|| "glass_orb".to_string())
    }

    // A flat terrain growing the GRASS layer.
    fn grassy_terrain() -> LoadedTerrain {
        LoadedTerrain {
            terrain: Terrain {
                extent: [8.0, 8.0],
                resolution: 4,
                layers: vec![TerrainLayer {
                    grass: Ref::new(GRASS),
                    density_mask: None,
                    ..TerrainLayer::default()
                }],
                ..Default::default()
            },
            payload: TerrainPayload {
                grid: TerrainGrid::new(4, [8.0, 8.0], vec![0.0; 25]).unwrap(),
                masks: vec![None],
            },
            far_field: None,
        }
    }

    fn sdf(locator: Option<PayloadLocator>) -> SdfVolume {
        SdfVolume {
            extent: [2.0; 3],
            locator,
            visible: true,
            ..Default::default()
        }
    }

    // The one-shot effects are resolved and drained: each becomes a backend
    // record with no per-frame update path, so leaving the components behind
    // would only invite a second, stale build. The terrain the grass grows on
    // stays for the physics collider built after graphics init.
    #[test]
    fn the_one_shot_effects_are_resolved_and_drained() {
        let mut section = OneSection::default();
        let fragment = section.payload(b"sdf-fragment-bytes");
        let mut world = World::from_payloads(Box::new(section));
        world.push_identified(
            AssetId(830),
            Decal {
                texture: Some(TextureHandle::new(0)),
                size: [1.0; 3],
                visible: true,
                ..Default::default()
            },
        );
        world.push_identified(
            AssetId(831),
            ParticleEmitter {
                texture: Some(TextureHandle::new(0)),
                max_particles: 16,
                visible: true,
                ..Default::default()
            },
        );
        world.push(WaterSurface::default());
        world.push(GlassPanel::default());
        world.push(Wind::default());
        world.push_identified(GRASS, Grass::default());
        world.push(Terrain::default());
        world.push_identified(AssetId(832), sdf(Some(fragment)));
        world.push_identified(AssetId(833), sdf(None));

        let fx = drain_world_fx(
            &mut world.context(),
            name_of,
            1,
            &[grassy_terrain()],
            None,
            OceanQuality::FULL,
        );
        assert!(
            fx.grass.is_some(),
            "the terrain's grass resolves to a field"
        );
        assert_eq!(fx.canopy.and_then(|c| c.planet_range()), None);
        assert_eq!(
            fx.sdf_volumes.len(),
            1,
            "the payload-less volume is skipped"
        );
        let ctx = world.context();
        assert_eq!(ctx.query::<Decal>().count(), 0);
        assert_eq!(ctx.query::<ParticleEmitter>().count(), 0);
        assert_eq!(ctx.query::<WaterSurface>().count(), 0);
        assert_eq!(ctx.query::<GlassPanel>().count(), 0);
        assert_eq!(ctx.query::<SdfVolume>().count(), 0);
        assert_eq!(ctx.query::<Wind>().count(), 0);
        assert_eq!(ctx.query::<Grass>().count(), 0);
        assert_eq!(ctx.query::<Terrain>().count(), 1);
    }

    fn planet_shape() -> PlanetShape {
        Planet {
            land_share: 0.4,
            ..Planet::default()
        }
        .shape()
    }

    fn sandy() -> PlanetSurface {
        PlanetSurface {
            grass: None,
            covers: GroundCovers {
                count: 1,
                ..GroundCovers::default()
            },
        }
    }

    fn shore() -> ShoreWaves {
        let sea = crate::water::spectrum::SeaShape::new(7.0, 50_000.0).unwrap();
        ShoreWaves::new(&planet_shape(), 0.0, &sea, None).unwrap()
    }

    fn terrain_grass() -> Vec<GrassTerrain> {
        let mut grass = hashbrown::HashMap::new();
        grass.insert(GRASS, Grass::default());
        grass_terrains(&[grassy_terrain()], &grass)
    }

    // Which block the scene lays.
    #[derive(Debug, PartialEq)]
    enum Block {
        None,
        Terrains,
        Planet { layers: u32 },
    }

    fn scene_block(
        planet: Option<&PlanetSurface>,
        terrains: Option<&[GrassTerrain]>,
        shore: Option<ShoreWaves>,
    ) -> Block {
        match scene_canopy(planet.map(|g| (planet_shape(), g)), terrains, shore, 0.0) {
            None => Block::None,
            Some(c) => c
                .planet_range()
                .map_or(Block::Terrains, |r| Block::Planet { layers: r.count }),
        }
    }

    // A planet lays its ground in the block when it grows grass, lays covers
    // or has a shore, and leaves it to grassy terrains otherwise.
    #[test]
    fn the_scene_block_is_the_planets_ground_when_it_has_one() {
        let bare = PlanetSurface::default();
        let grassy = PlanetSurface {
            grass: PlanetGrass::resolve(
                planet_shape(),
                &[(Grass::default(), crate::components::GroundRule::default())],
            ),
            ..PlanetSurface::default()
        };
        assert!(grassy.grass.is_some());
        let terrains = terrain_grass();
        assert_eq!(scene_block(None, None, None), Block::None);
        assert_eq!(scene_block(Some(&bare), None, None), Block::None);
        assert_eq!(
            scene_block(Some(&sandy()), None, None),
            Block::Planet { layers: 0 }
        );
        assert_eq!(
            scene_block(Some(&bare), None, Some(shore())),
            Block::Planet { layers: 0 }
        );
        assert_eq!(
            scene_block(Some(&grassy), None, None),
            Block::Planet { layers: 1 }
        );
        assert_eq!(scene_block(None, Some(&terrains), None), Block::Terrains);
        assert_eq!(
            scene_block(Some(&bare), Some(&terrains), None),
            Block::Terrains
        );
        // Terrains growing grass keep the block from a grassless planet; a
        // planet's grass keeps it from them.
        assert_eq!(
            scene_block(Some(&sandy()), Some(&terrains), None),
            Block::Terrains
        );
        assert_eq!(
            scene_block(Some(&grassy), Some(&terrains), None),
            Block::Planet { layers: 1 }
        );
    }

    // The shore's wash rides whichever block the scene lays.
    #[test]
    fn the_scene_block_carries_the_shore() {
        let with = scene_canopy(Some((planet_shape(), &sandy())), None, Some(shore()), 0.0);
        let without = scene_canopy(Some((planet_shape(), &sandy())), None, None, 0.0);
        assert_ne!(with, without);
    }

    // A volume with no compiled payload, or one whose payload cannot be read,
    // is skipped; a readable one keeps its bytes and its asset name as label.
    #[test]
    fn sdf_volumes_without_a_readable_payload_are_skipped() {
        let mut section = OneSection::default();
        let fragment = section.payload(b"sdf-fragment-bytes");
        let mut world = World::from_payloads(Box::new(section));
        world.push_identified(ORB, sdf(Some(fragment)));
        world.push_identified(AssetId(841), sdf(None));
        world.push_identified(
            AssetId(842),
            sdf(Some(PayloadLocator {
                blob_index: 7,
                offset: 0,
                len: 4,
            })),
        );

        let volumes = drain_sdf_volumes(&mut world.context(), name_of);
        assert_eq!(volumes.len(), 1);
        assert_eq!(volumes[0].fragment_source, b"sdf-fragment-bytes");
        assert_eq!(volumes[0].label, "glass_orb");
        assert_eq!(world.context().query::<SdfVolume>().count(), 0, "drained");
    }
}
