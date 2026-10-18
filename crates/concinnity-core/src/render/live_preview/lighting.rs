//! Live application of the world's lighting assets. The renderer reads its
//! lighting from GPU state built at load (the sun is packed into the shared
//! light uniforms, the shadow and post-process knobs are resolved once), so an
//! editor changing one of them has nothing in the ECS to write: the effect is a
//! backend call. Each function records the calls and keeps the settings state
//! in step, so the settings menu and an authoring edit never disagree about
//! what is live.
//!
//! A knob resolves the way graphics init resolves it, through
//! [`crate::settings::resolve`]: to the user's persisted settings-menu choice
//! where they made one, otherwise to the world's value under the quality
//! preset's ceiling. So editing a row the user has overridden moves the
//! authored baseline and leaves the picture alone, exactly as relaunching would.

use crate::components::{
    AmbientOcclusion, Bloom, DirectionalLight, Exposure, GraphicsConfig, IndirectLighting,
    PlanetDetail, Reflections, Shadows, Vignette, VolumetricFog,
};
use crate::ecs::{AssetFields, World};
use crate::render::ops::RenderOps;
use crate::render::queues::ActiveRenderQueues;
use crate::render::volumetric_fog;
use crate::settings::system::{SettingsSlot, SettingsState};
use crate::settings::{ResolvedGraphics, WorldAssets, WorldSettings};

use super::has_queue;

/// The `GraphicsConfig` sections this module can apply to a running world, each
/// with the fields it applies live: the shadow cadence the per-frame cascade
/// math reads, the planet terrain detail the tile streaming reads, and every
/// look-tuning scalar the settings menu exposes as a slider. The rest (shadow
/// map resolution, the feature toggles and the resolution knobs) size or gate a
/// GPU resource built at load and need the world rebuilt.
pub(crate) const LIVE_SETTINGS_FIELDS: &[(&str, &[&str])] = &[
    ("shadows", Shadows::LIVE_PARAMS),
    ("ambient_occlusion", AmbientOcclusion::LIVE_PARAMS),
    ("reflections", Reflections::LIVE_PARAMS),
    ("indirect_lighting", IndirectLighting::LIVE_PARAMS),
    ("exposure", Exposure::LIVE_PARAMS),
    ("bloom", Bloom::LIVE_PARAMS),
    ("vignette", Vignette::LIVE_PARAMS),
    ("planet_detail", PlanetDetail::LIVE_PARAMS),
];

/// The fields of `GraphicsConfig` section `section` this module applies live,
/// or `None` when it applies none of that section.
pub fn live_settings_fields(section: &str) -> Option<&'static [&'static str]> {
    LIVE_SETTINGS_FIELDS
        .iter()
        .find(|(name, _)| *name == section)
        .map(|(_, fields)| *fields)
}

/// Whether the running world has a graphics context to apply a lighting change
/// to. `false` in a headless world and before graphics init has published its
/// state, where every `apply_*` below is a no-op.
pub fn is_available(world: &World) -> bool {
    has_queue(world)
        && world
            .resource::<SettingsSlot>()
            .is_some_and(|slot| slot.0.is_some())
}

/// Whether the backend built the volumetric-fog pass, which it does only for a
/// world that declared enabled fog at load. Fog cannot be turned on live
/// without it: there is no pipeline to run.
pub fn fog_pass_built(world: &World) -> bool {
    world
        .resource::<SettingsSlot>()
        .and_then(|slot| slot.0.as_ref())
        .is_some_and(|state| state.fog_built)
}

/// Replace the running world's directional lights. `lights` is the whole set in
/// the order the renderer packed it, since the backend rewrites the array
/// wholesale. Returns whether the world took it.
pub fn apply_directional_lights(world: &mut World, lights: &[DirectionalLight]) -> bool {
    let lights = lights.to_vec();
    with_live(world, |_, ops| {
        ops.record(move |backend| backend.update_directional_lights(&lights));
    })
    .is_some()
}

/// Replace the running world's volumetric fog, or disable the pass with `None`.
/// Enabling fog on a world that started without it is refused: see
/// [`fog_pass_built`].
pub fn apply_fog(world: &mut World, fog: Option<&VolumetricFog>) -> bool {
    let settings = fog.and_then(volumetric_fog::resolve_asset);
    if settings.is_some() && !fog_pass_built(world) {
        return false;
    }
    with_live(world, |_, ops| {
        ops.record(move |backend| backend.update_fog_settings(settings));
    })
    .is_some()
}

/// Apply an edited `GraphicsConfig` to a running world. Only the fields
/// [`live_settings_fields`] names reach the picture; the rest move the authored
/// baseline the settings menu re-clamps from and take effect at the next
/// launch. Returns whether the world took it.
pub fn apply_graphics(world: &mut World, graphics: &GraphicsConfig) -> bool {
    let edited = WorldSettings::new(WorldAssets {
        graphics: Some(graphics),
        ..Default::default()
    });
    with_live(world, |state, ops| {
        let changed = adopt_live(&mut state.authored, &edited);
        if changed.shadows {
            push_shadows(state, ops);
        }
        if changed.look {
            push_look(state, ops);
        }
        if changed.planet_detail {
            push_planet_detail(state);
        }
    })
    .is_some()
}

// Which live groups an edit moved.
struct LiveChanges {
    shadows: bool,
    look: bool,
    planet_detail: bool,
}

// Move the authored baseline's live fields to the edited world's, leaving the
// toggles and resolution knobs beside them alone. The shadow map size moves
// too, as it is only read back through the resolved quality.
fn adopt_live(world: &mut WorldSettings, edited: &WorldSettings) -> LiveChanges {
    let (world, edited) = (&mut world.graphics, &edited.graphics);
    let map_size = world.shadows.map_size != edited.shadows.map_size;
    world.shadows.map_size = edited.shadows.map_size;
    let shadows = world.shadows.copy_live(&edited.shadows) | map_size;
    let look = copy_look(world, edited);
    let planet_detail =
        world.planet_detail.finest_cell_size != edited.planet_detail.finest_cell_size;
    world.planet_detail = edited.planet_detail.clone();
    LiveChanges {
        shadows,
        look,
        planet_detail,
    }
}

// Copy the live look scalars of `src` onto `dst`; whether any of them moved.
fn copy_look(dst: &mut ResolvedGraphics, src: &ResolvedGraphics) -> bool {
    dst.ambient_occlusion.copy_live(&src.ambient_occlusion)
        | dst.reflections.copy_live(&src.reflections)
        | dst.indirect_lighting.copy_live(&src.indirect_lighting)
        | dst.exposure.copy_live(&src.exposure)
        | dst.bloom.copy_live(&src.bloom)
        | dst.vignette.copy_live(&src.vignette)
}

// Re-resolve the cascade schedule and push it when it moved.
fn push_shadows(state: &mut SettingsState, ops: &mut RenderOps) {
    let resolved = state.resolve_quality(&state.persisted_graphics, &state.ceiling());
    let before = state.resolved.clone();
    state.resolved.graphics.shadows = resolved.shadows;
    state.record_change(&before, ops);
}

// Re-resolve the planet terrain detail, which the tile streaming reads from the
// resolved settings each frame.
fn push_planet_detail(state: &mut SettingsState) {
    let resolved = state.resolve_quality(&state.persisted_graphics, &state.ceiling());
    state.resolved.graphics.planet_detail = resolved.planet_detail;
}

// Re-resolve the look tuning and push the composite params, the per-feature
// sub-quality settings, and the ambient scale.
fn push_look(state: &mut SettingsState, ops: &mut RenderOps) {
    let resolved = state.resolve(&state.persisted_graphics, &state.ceiling());
    let before = state.resolved.clone();
    let held = &mut state.resolved.graphics;
    copy_look(held, &resolved.graphics);
    held.lut_strength = resolved.graphics.lut_strength;
    state.record_change(&before, ops);
}

// Run `f` against the parked settings state and the frame's op queue. `None`
// when the world has no graphics context, which leaves both slots exactly as
// they were.
fn with_live<R>(
    world: &mut World,
    f: impl FnOnce(&mut SettingsState, &mut RenderOps) -> R,
) -> Option<R> {
    let ctx = world.context();
    let mut state = SettingsSlot::take(ctx.resources)?;
    let out = ActiveRenderQueues::with(ctx.resources, |queues| f(&mut state, &mut queues.ops));
    SettingsSlot::put(ctx.resources, state);
    out
}

#[cfg(test)]
mod tests;
