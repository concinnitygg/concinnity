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
use crate::render::backend::init::{SdfVolumeSource, WorldFx};
use crate::render::environment::atmosphere::AtmosphereSettings;
use crate::render::environment::grass::{GrassField, PlanetGrass};
use crate::render::environment::night_sky::NightSkySettings;
use crate::render::environment::terrain::{LoadedTerrain, grass_terrains};
use crate::render::environment::water::WaterBody;
use crate::render::environment::water::ocean::OceanQuality;
use crate::render::environment::wind::WindField;
use crate::render::{decal, environment::volumetric_fog, particles};

/// Drain every effect component into the backend's effect inputs. Each record
/// is resolved once here (texture slots, inverted decal matrices, clamped
/// emitter tunables): the runtime keeps no per-frame update path for them.
/// `name_of` labels an SDF volume by its asset's name.
pub fn drain_world_fx(
    ctx: &mut PipelineContext<'_>,
    name_of: fn(AssetId) -> Option<String>,
    texture_count: usize,
    terrains: &[LoadedTerrain],
    planet_grass: Option<PlanetGrass>,
    planet: Option<&Planet>,
    ocean_quality: OceanQuality,
) -> WorldFx {
    let decals: Vec<Decal> = ctx.drain::<Decal>();
    let decals = decal::build_decal_records(&decals.iter().collect::<Vec<_>>(), texture_count);
    let emitters: Vec<ParticleEmitter> = ctx.drain::<ParticleEmitter>();
    let particles =
        particles::build_particle_records(&emitters.iter().collect::<Vec<_>>(), texture_count);
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
    let terrain_grass = GrassField::resolve(&grass_terrains(terrains, &grass), wind.as_ref());
    // One field per world: a planet's grass, or the terrains'.
    let grass = match planet_grass {
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
        atmosphere,
    }
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
            None,
            OceanQuality::FULL,
        );
        assert!(
            fx.grass.is_some(),
            "the terrain's grass resolves to a field"
        );
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
