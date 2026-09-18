// The world's one-shot effect content, drained from its components at init.

use concinnity_core::components::{
    Decal, GlassPanel, ParticleEmitter, SdfVolume, VolumetricFog, WaterSurface,
};
use concinnity_core::ecs::PipelineContext;
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::render::backend_init::{SdfVolumeSource, WorldFx};
use concinnity_core::render::{decal, particles, volumetric_fog};
use concinnity_host::thread::asset_id;

// Drain every effect component into the backend's effect inputs. Each record
// is resolved once here (texture slots, inverted decal matrices, clamped
// emitter tunables): the runtime keeps no per-frame update path for them.
pub(super) fn drain_world_fx(ctx: &mut PipelineContext, texture_count: usize) -> WorldFx {
    let decals: Vec<Decal> = ctx.drain::<Decal>();
    let decals = decal::build_decal_records(&decals.iter().collect::<Vec<_>>(), texture_count);
    let emitters: Vec<ParticleEmitter> = ctx.drain::<ParticleEmitter>();
    let particles =
        particles::build_particle_records(&emitters.iter().collect::<Vec<_>>(), texture_count);
    let water_surfaces = ctx.drain::<WaterSurface>();
    let glass_panels = ctx.drain::<GlassPanel>();
    let sdf_volumes = drain_sdf_volumes(ctx);
    // The first enabled fog wins: the fog pass models one homogeneous medium.
    // `None` also covers a fog whose density cannot affect the frame.
    let fog = ctx
        .drain::<VolumetricFog>()
        .into_iter()
        .find(|f| f.enabled)
        .and_then(|f| volumetric_fog::resolve_asset(&f));
    WorldFx {
        decals,
        particles,
        fog,
        water_surfaces,
        glass_panels,
        sdf_volumes,
    }
}

// Drain the raymarched SDF volumes with their compiled payloads. A volume with
// no payload, or one whose payload cannot be read, is skipped with a warning
// rather than failing the world build.
pub(super) fn drain_sdf_volumes(ctx: &mut PipelineContext) -> Vec<SdfVolumeSource> {
    let raw: Vec<(Option<AssetId>, SdfVolume)> = ctx.drain_with_ids::<SdfVolume>();
    let mut out = Vec::with_capacity(raw.len());
    for (i, (id, volume)) in raw.into_iter().enumerate() {
        let label = id
            .and_then(asset_id::name_of)
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
