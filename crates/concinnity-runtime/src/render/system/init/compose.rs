// The pieces of GraphicsSystem init that assemble one input each: a planet's
// ground summary, the hot-reload source catalogs, and the ready report.

use concinnity_core::math::SaturatingFrom;
use concinnity_engine::components::Planet;
use concinnity_engine::ecs::{MaterialHandle, PipelineContext};
use concinnity_engine::gfx::render_types::DrawIndex;
use concinnity_engine::planet::MAX_GROUND_COVERS;
use concinnity_engine::render::environment::ground::GroundCovers;
use concinnity_engine::render::environment::terrain::grass_by_id;
use concinnity_engine::render::material_entry::MaterialEntry;
use concinnity_engine::render::world_load::source_maps::{
    SdfFieldMap, TextureNameSlots, mesh_source_map, procedural_mesh_source_map,
};
use concinnity_engine::render::world_load::world_fx::PlanetSurface;
use concinnity_engine::render::world_load::world_scan::planet_grass;
use concinnity_engine::resource::TextureTable;
use concinnity_engine::settings::system::SettingsState;

use super::decode::WorldPayloads;
use crate::live_edit::hot_reload_sources::HotReloadSources;

// What the planet's ground looks like from afar: the mean albedo of its
// material and of each cover's, its grass, and its covers.
pub(super) fn planet_surface(
    ctx: &PipelineContext<'_>,
    planet: &Planet,
    world: &WorldPayloads,
) -> PlanetSurface {
    let textures = ctx.resource::<TextureTable>();
    let mean = |slot: usize| textures.and_then(|t| t.mean(slot));
    let entry = |h| world.materials.map.get(&h).copied();
    let seen = |h: Option<MaterialHandle>| {
        h.and_then(entry)
            .unwrap_or(MaterialEntry::UNTEXTURED)
            .mean_albedo(mean)
    };
    let mut cover_albedo = [[0.0; 3]; MAX_GROUND_COVERS];
    for (albedo, cover) in cover_albedo.iter_mut().zip(&planet.covers) {
        *albedo = seen(cover.material);
    }
    PlanetSurface {
        albedo: seen(planet.material),
        cover_albedo,
        grass: planet_grass(planet, &grass_by_id(ctx)),
        covers: GroundCovers::of(planet, entry, u32::saturating_from(world.texture_count)),
    }
}

// The file-backed sources a development loop's watcher reloads, taken out of
// the decoded world, and the texture slots its names map to.
pub(super) fn take_hot_reload_sources(
    world: &mut WorldPayloads,
    mesh_handle_to_draws: &hashbrown::HashMap<usize, Vec<DrawIndex>>,
    sdf_fields: SdfFieldMap,
) -> (HotReloadSources, TextureNameSlots) {
    let sources = HotReloadSources {
        map: std::mem::take(&mut world.asset_source_map),
        color_lut: world.color_lut_source.take(),
        environment_map: world.environment_map_source.take(),
        meshes: mesh_source_map(&world.mesh.sources, mesh_handle_to_draws),
        skinned_meshes: std::mem::take(&mut world.skinned_mesh_source_map),
        procedural_meshes: procedural_mesh_source_map(
            &world.proc_mesh_args_snapshot,
            &world.mesh.component_handles,
            mesh_handle_to_draws,
        ),
        shaders: std::mem::take(&mut world.shaders.source_map),
        shader_overrides: std::mem::take(&mut world.shaders.overrides),
        sdf_fields,
    };
    // Parked even when empty: the dev drive's watcher still subscribes to the
    // host's world.jsonl.
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
    let slots = TextureNameSlots(std::mem::take(&mut world.texture_name_to_slot));
    (sources, slots)
}

// What init built, for the ready report.
pub(super) struct ReadyCounts {
    pub(super) draw_objects: usize,
    pub(super) clusters: usize,
    pub(super) instances: usize,
    pub(super) decals: usize,
    pub(super) particles: usize,
    pub(super) fog: bool,
}

pub(super) fn log_ready(settings: &SettingsState, counts: &ReadyCounts) {
    let window = settings.window();
    tracing::info!(
        "GraphicsSystem: ready ({}x{} \"{}\", {} frames in flight, {} draw objects, {} instanced clusters ({} instances total), {} decals, {} particle emitter(s), fog={})",
        window.width,
        window.height,
        window.title,
        settings.resolved().graphics.frame_pacing.frames_in_flight,
        counts.draw_objects,
        counts.clusters,
        counts.instances,
        counts.decals,
        counts.particles,
        if counts.fog { "on" } else { "off" },
    );
}
