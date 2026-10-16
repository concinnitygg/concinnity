//! Live application of the world's lighting assets. The renderer reads its
//! lighting from GPU state built at load -- the sun is packed into the shared
//! light uniforms, the shadow and post-process knobs are resolved once -- so an
//! editor changing one of them has nothing in the ECS to write: the effect is a
//! backend call. This is that seam. Each function records the calls into the
//! frame's op queue, where submission replays them before the next draw, and
//! keeps the settings state in step so the settings menu and an authoring edit
//! never disagree about what is live.
//!
//! A knob resolves the same way `run_init` resolves it, through the settings
//! resolver (`concinnity_core::settings::resolve`): to the user's persisted
//! settings-menu choice where they made one, otherwise to the world's value
//! under the quality preset's ceiling. So editing a row the user has overridden
//! moves the authored baseline and leaves the picture alone, exactly as
//! relaunching would.

use concinnity_core::components::{DirectionalLight, GraphicsConfig, VolumetricFog};
use concinnity_core::ecs::World;
use concinnity_core::render::ops::RenderOps;
use concinnity_core::render::volumetric_fog;
use concinnity_core::settings::{ResolvedGraphics, WorldAssets, WorldSettings};

use crate::ecs::ActiveRenderQueues;
use crate::settings::system::{SettingsSlot, SettingsState};

/// The `GraphicsConfig` sections this module can apply to a running world, each
/// with the fields it applies live: the shadow cadence the per-frame cascade
/// math reads, the planet terrain detail the tile streaming reads, and every
/// look-tuning scalar the settings menu exposes as a slider. The rest (shadow map resolution, the feature toggles and the
/// resolution knobs) size or gate a GPU resource built at load and need the
/// world rebuilt.
pub const LIVE_SETTINGS_FIELDS: &[(&str, &[&str])] = &[
    ("shadows", &["distance", "cascades", "update"]),
    ("ambient_occlusion", &["radius", "intensity"]),
    ("reflections", &["intensity", "max_distance"]),
    (
        "indirect_lighting",
        &["ambient_intensity", "ssgi_intensity", "ssgi_max_distance"],
    ),
    (
        "exposure",
        &[
            "ev",
            "auto_min_ev",
            "auto_max_ev",
            "auto_speed",
            "auto_night_ev",
            "night_vision",
        ],
    ),
    ("bloom", &["intensity", "threshold", "knee"]),
    ("vignette", &["strength"]),
    ("planet_detail", &["finest_cell_size"]),
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
    world
        .resource::<ActiveRenderQueues>()
        .is_some_and(|slot| slot.0.is_some())
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

/// Apply an edited `GraphicsConfig` to a running world. Only the fields in
/// [`LIVE_SETTINGS_FIELDS`] reach the picture; the rest move the authored
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
// toggles and resolution knobs beside them alone. The shadows section moves
// whole, as its map size is only read back through the resolved quality.
fn adopt_live(world: &mut WorldSettings, edited: &WorldSettings) -> LiveChanges {
    let (world, edited) = (&mut world.graphics, &edited.graphics);
    let mut look = false;
    macro_rules! adopt {
        ($($field:ident).+) => {
            if world.$($field).+ != edited.$($field).+ {
                world.$($field).+ = edited.$($field).+;
                look = true;
            }
        };
    }
    let (from, to) = (&edited.shadows, &world.shadows);
    let shadows = from.distance != to.distance
        || from.cascades != to.cascades
        || from.update != to.update
        || from.map_size != to.map_size;
    world.shadows = edited.shadows.clone();
    adopt!(ambient_occlusion.radius);
    adopt!(ambient_occlusion.intensity);
    adopt!(reflections.intensity);
    adopt!(reflections.max_distance);
    adopt!(indirect_lighting.ambient_intensity);
    adopt!(indirect_lighting.ssgi_intensity);
    adopt!(indirect_lighting.ssgi_max_distance);
    adopt!(exposure.ev);
    adopt!(exposure.auto_min_ev);
    adopt!(exposure.auto_max_ev);
    adopt!(exposure.auto_speed);
    adopt!(exposure.auto_night_ev);
    adopt!(exposure.night_vision);
    adopt!(bloom.intensity);
    adopt!(bloom.threshold);
    adopt!(bloom.knee);
    adopt!(vignette.strength);
    let planet_detail =
        world.planet_detail.finest_cell_size != edited.planet_detail.finest_cell_size;
    world.planet_detail = edited.planet_detail.clone();
    LiveChanges {
        shadows,
        look,
        planet_detail,
    }
}

// Re-resolve the cascade schedule and push it when it moved.
fn push_shadows(state: &mut SettingsState, ops: &mut RenderOps) {
    let user = state.persisted_graphics().clone();
    let resolved = state.resolve_quality(&user, &state.ceiling());
    let before = state.resolved.clone();
    state.resolved.graphics.shadows = resolved.shadows;
    state.record_change(&before, ops);
}

// Re-resolve the planet terrain detail, which the tile streaming reads from the
// resolved settings each frame.
fn push_planet_detail(state: &mut SettingsState) {
    let user = state.persisted_graphics().clone();
    let resolved = state.resolve_quality(&user, &state.ceiling());
    state.resolved.graphics.planet_detail = resolved.planet_detail;
}

// Re-resolve the look tuning and push the composite params, the per-feature
// sub-quality settings, and the ambient scale.
fn push_look(state: &mut SettingsState, ops: &mut RenderOps) {
    let user = state.persisted_graphics().clone();
    let resolved = state.resolve(&user, &state.ceiling());
    let before = state.resolved.clone();
    copy_live_fields(&mut state.resolved.graphics, &resolved.graphics);
    state.record_change(&before, ops);
}

// Copy the live look fields from resolved settings onto held ones.
fn copy_live_fields(dst: &mut ResolvedGraphics, src: &ResolvedGraphics) {
    dst.ambient_occlusion.radius = src.ambient_occlusion.radius;
    dst.ambient_occlusion.intensity = src.ambient_occlusion.intensity;
    dst.reflections.intensity = src.reflections.intensity;
    dst.reflections.max_distance = src.reflections.max_distance;
    let (d, s) = (&mut dst.indirect_lighting, &src.indirect_lighting);
    d.ambient_intensity = s.ambient_intensity;
    d.ssgi_intensity = s.ssgi_intensity;
    d.ssgi_max_distance = s.ssgi_max_distance;
    let (d, s) = (&mut dst.exposure, &src.exposure);
    d.ev = s.ev;
    d.auto_min_ev = s.auto_min_ev;
    d.auto_max_ev = s.auto_max_ev;
    d.auto_speed = s.auto_speed;
    d.auto_night_ev = s.auto_night_ev;
    d.night_vision = s.night_vision;
    dst.bloom = src.bloom.clone();
    dst.vignette = src.vignette.clone();
    dst.lut_strength = src.lut_strength;
}

// Run `f` against the parked settings state and the frame's op queue, taking
// both for the call and parking them again after (the handoff every recording
// system uses). `None` when the world has no graphics context, which leaves
// both slots exactly as they were.
fn with_live<R>(
    world: &mut World,
    f: impl FnOnce(&mut SettingsState, &mut RenderOps) -> R,
) -> Option<R> {
    if !is_available(world) {
        return None;
    }
    let mut queues = world.resource_mut::<ActiveRenderQueues>()?.0.take()?;
    let mut state = world.resource_mut::<SettingsSlot>()?.0.take()?;
    let out = f(&mut state, &mut queues.ops);
    if let Some(slot) = world.resource_mut::<SettingsSlot>() {
        slot.0 = Some(state);
    }
    if let Some(slot) = world.resource_mut::<ActiveRenderQueues>() {
        slot.0 = Some(queues);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::RenderQueues;
    use crate::gfx::mock_backend::{Call, MockBackend, MockState, recording_backend};
    use concinnity_core::components::{
        AaMode, AmbientOcclusion, AmbientOcclusionOverrides, ExposureOverrides,
        IndirectLightingOverrides, ShadowsOverrides,
    };
    use std::sync::{Arc, Mutex};

    struct Fixture {
        world: World,
        backend: MockBackend,
        calls: Arc<Mutex<MockState>>,
    }

    impl Fixture {
        fn new() -> Self {
            Self::with_state(SettingsState::for_tests())
        }

        fn with_state(state: SettingsState) -> Self {
            let mut world = World::new();
            world.insert_resource(ActiveRenderQueues(Some(RenderQueues {
                ops: RenderOps::default(),
                slots: crate::gfx::render_slots::RenderSlots::new(0, true, &[]),
            })));
            world.insert_resource(SettingsSlot(Some(state)));
            let (calls, backend) = recording_backend();
            Self {
                world,
                backend,
                calls,
            }
        }

        // Replay whatever the seam recorded, then report the calls it made.
        fn replay(&mut self) -> Vec<Call> {
            let mut queues = self
                .world
                .resource_mut::<ActiveRenderQueues>()
                .and_then(|slot| slot.0.take())
                .expect("the queue is parked again");
            queues.ops.replay(&mut self.backend);
            if let Some(slot) = self.world.resource_mut::<ActiveRenderQueues>() {
                slot.0 = Some(queues);
            }
            self.calls.lock().unwrap().calls.clone()
        }

        fn state(&mut self) -> &mut SettingsState {
            self.world
                .resource_mut::<SettingsSlot>()
                .and_then(|slot| slot.0.as_mut())
                .expect("parked")
        }
    }

    fn shadows() -> GraphicsConfig {
        GraphicsConfig {
            shadows: ShadowsOverrides {
                distance: Some(120),
                cascades: Some(2),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    // A world with no renderer takes nothing, and says so rather than
    // pretending: that is the signal the caller rebuilds on.
    // What this module applies live is exactly what the schema declares live:
    // a field outside these lists waits for a rebuild or a relaunch.
    #[test]
    fn the_live_fields_are_the_ones_the_schema_declares_live() {
        use concinnity_core::components::GRAPHICS_SECTIONS;
        use concinnity_core::ecs::AssetFields;
        use concinnity_core::ecs::asset_fields::ApplyClass;
        let mut declared: Vec<(String, Vec<String>)> = Vec::new();
        for param in GraphicsConfig::field_table().params {
            if param.meta.apply != Some(ApplyClass::Live) {
                continue;
            }
            let (section, key) = param.path.split_once('.').expect("a section field");
            match declared.iter_mut().find(|(s, _)| s == section) {
                Some((_, keys)) => keys.push(key.to_string()),
                None => declared.push((section.to_string(), vec![key.to_string()])),
            }
        }
        for (_, section) in GRAPHICS_SECTIONS {
            let mut listed: Vec<String> = live_settings_fields(section)
                .unwrap_or_default()
                .iter()
                .map(|k| k.to_string())
                .collect();
            listed.sort_unstable();
            let mut live = declared
                .iter()
                .find(|(s, _)| s == section)
                .map(|(_, keys)| keys.clone())
                .unwrap_or_default();
            live.sort_unstable();
            if section == "frame_pacing" {
                // Live through the settings menu, not this seam.
                assert_eq!(live, ["fps_cap", "vsync"]);
                assert!(listed.is_empty());
            } else {
                assert_eq!(listed, live, "{section}");
            }
        }
    }

    #[test]
    fn a_world_without_a_renderer_takes_nothing() {
        let mut world = World::new();
        assert!(!is_available(&world));
        assert!(!apply_directional_lights(&mut world, &[]));
        assert!(!apply_graphics(&mut world, &shadows()));
        assert!(!apply_graphics(&mut world, &GraphicsConfig::default()));
    }

    // The sun reaches the backend as the whole set, in the order it was handed.
    #[test]
    fn a_sun_edit_reaches_the_backend() {
        let mut f = Fixture::new();
        let sun = DirectionalLight {
            direction: [0.0, 1.0, 0.0],
            color: [1.0, 0.5, 0.25],
            intensity: 3.0,
        };
        assert!(apply_directional_lights(
            &mut f.world,
            std::slice::from_ref(&sun)
        ));
        assert_eq!(
            f.replay(),
            vec![Call::UpdateDirectionalLights(vec![(
                sun.direction,
                sun.color,
                sun.intensity
            )])]
        );
    }

    // The live shadow knobs push; the authored baselines move with them so a
    // later preset change re-clamps from the edited world, not the old one.
    #[test]
    fn shadow_knobs_push_and_move_their_baselines() {
        let mut f = Fixture::new();
        assert!(apply_graphics(&mut f.world, &shadows()));
        let calls = f.replay();
        let state = f.state();
        assert_eq!(
            calls,
            vec![Call::SetShadowCadence(
                state.resolved.graphics.shadow_cadence()
            )]
        );
        assert_eq!(state.resolved.graphics.shadows.distance, 120);
        assert_eq!(state.authored.graphics.shadows.distance, 120);
        assert_eq!(state.resolved.graphics.shadows.cascades, 2);
        assert_eq!(state.authored.graphics.shadows.cascades, 2);
    }

    // An edited finest planet cell size moves the resolved value the tile
    // streaming reads, under the preset's ceiling, with nothing for the
    // renderer to do.
    #[test]
    fn the_finest_planet_cell_size_applies_under_the_ceiling() {
        use concinnity_core::components::PlanetDetailOverrides;
        let mut f = Fixture::new();
        let shadows = f.state().authored.graphics.shadows.clone();
        let config = |size| GraphicsConfig {
            shadows: shadows.clone().into(),
            planet_detail: PlanetDetailOverrides {
                finest_cell_size: Some(size),
            },
            ..Default::default()
        };
        assert!(apply_graphics(&mut f.world, &config(3)));
        assert!(f.replay().is_empty());
        let state = f.state();
        assert_eq!(state.authored.graphics.planet_detail.finest_cell_size, 3);
        assert_eq!(state.resolved.graphics.planet_detail.finest_cell_size, 3);

        state.quality_preset = concinnity_core::settings::QualityPreset::Medium;
        assert!(apply_graphics(&mut f.world, &config(1)));
        let state = f.state();
        assert_eq!(state.authored.graphics.planet_detail.finest_cell_size, 1);
        assert_eq!(state.resolved.graphics.planet_detail.finest_cell_size, 2);
    }

    // Re-applying the same config records nothing: the seam is edge-triggered,
    // so a burst of edits that touch other fields never restates the cadence.
    #[test]
    fn an_unchanged_cadence_records_no_call() {
        let mut f = Fixture::new();
        apply_graphics(&mut f.world, &shadows());
        f.replay();
        f.calls.lock().unwrap().calls.clear();
        apply_graphics(&mut f.world, &shadows());
        assert!(f.replay().is_empty());
    }

    // A row the user has overridden in the settings menu keeps the user's
    // value: that is what a relaunch of the edited world would show, so the
    // edit moves the authored baseline and leaves the picture alone.
    #[test]
    fn a_persisted_override_outranks_the_edited_world() {
        let mut state = SettingsState::for_tests();
        state.persisted_graphics.shadow_distance = Some(500);
        state.resolved.graphics.shadows.distance = 500;
        let mut f = Fixture::with_state(state);

        assert!(apply_graphics(&mut f.world, &shadows()));
        let calls = f.replay();
        let [Call::SetShadowCadence(cadence)] = calls.as_slice() else {
            panic!("one cadence push expected: {calls:?}");
        };
        assert_eq!(cadence.distance, 500, "the overridden row does not move");
        assert_eq!(cadence.cascades, 2, "the rest do");
        let state = f.state();
        assert_eq!(
            state.resolved.graphics.shadows.distance, 500,
            "the user's value stands"
        );
        assert_eq!(
            state.authored.graphics.shadows.distance, 120,
            "the world's value is still recorded as the baseline"
        );
    }

    // The preset ceiling clamps an edited value exactly as it clamps an
    // authored one at launch.
    #[test]
    fn the_quality_ceiling_clamps_an_edited_knob() {
        let mut state = SettingsState::for_tests();
        state.quality_preset = concinnity_core::settings::QualityPreset::Low;
        let mut f = Fixture::with_state(state);
        let ceiling = f.state().ceiling();

        let config = GraphicsConfig {
            shadows: ShadowsOverrides {
                distance: Some(ceiling.shadow_distance + 1_000),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(apply_graphics(&mut f.world, &config));
        f.replay();
        assert_eq!(
            f.state().resolved.graphics.shadows.distance,
            ceiling.shadow_distance,
            "the ceiling holds"
        );
    }

    // A post-process edit pushes the composite params, the per-feature settings,
    // and the ambient scale, and the held copies follow.
    #[test]
    fn a_post_process_edit_pushes_every_live_knob() {
        let mut f = Fixture::new();
        let mut config = GraphicsConfig {
            indirect_lighting: IndirectLightingOverrides {
                ambient_intensity: Some(0.25),
                ssgi_intensity: Some(0.75),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(apply_graphics(&mut f.world, &config));
        config.exposure = ExposureOverrides {
            ev: Some(2.0),
            ..Default::default()
        };
        assert!(apply_graphics(&mut f.world, &config));
        let calls = f.replay();
        assert!(calls.contains(&Call::UpdatePostProcess));
        assert!(calls.contains(&Call::UpdateQualityParams));
        assert!(calls.contains(&Call::SetAmbientIntensity(0.25)));
        let state = f.state();
        assert_eq!(state.resolved.graphics.ambient_intensity(), 0.25);
        let held = &state.resolved.graphics;
        assert_eq!(held.indirect_lighting.ssgi_intensity, 0.75);
        assert_eq!(held.exposure.ev, 2.0);
        assert_eq!(
            state.authored.graphics.indirect_lighting.ssgi_intensity,
            0.75
        );
        assert_eq!(state.resolved.graphics.post_process().exposure, 4.0);
    }

    // The toggles beside the live scalars are not this seam's to move: an edit
    // that reached here with one flipped must leave the held config alone, so a
    // caller that failed to gate it cannot silently half-apply.
    #[test]
    fn a_post_process_edit_leaves_the_toggles_alone() {
        let mut f = Fixture::new();
        let config = GraphicsConfig {
            ambient_occlusion: AmbientOcclusionOverrides {
                enabled: Some(!AmbientOcclusion::default().enabled),
                radius: Some(1.0),
                ..Default::default()
            },
            ..Default::default()
        };
        let held_mode = f.state().resolved.graphics.anti_aliasing.mode;
        assert_ne!(held_mode, AaMode::Off);
        apply_graphics(&mut f.world, &config);
        let state = f.state();
        assert_eq!(
            state.resolved.graphics.ambient_occlusion.enabled,
            AmbientOcclusion::default().enabled
        );
        assert_eq!(
            state.authored.graphics.ambient_occlusion.enabled,
            AmbientOcclusion::default().enabled
        );
        assert_eq!(state.resolved.graphics.anti_aliasing.mode, held_mode);
    }

    // Fog resolves through the same clamp chain the build runs, and disabling
    // it pushes `None` rather than settings the pass would still draw.
    #[test]
    fn fog_pushes_resolved_settings_and_disables_with_none() {
        let mut f = Fixture::new();
        let fog = VolumetricFog {
            enabled: true,
            density: 0.05,
            ..Default::default()
        };
        assert!(apply_fog(&mut f.world, Some(&fog)));
        assert!(matches!(
            f.replay().as_slice(),
            [Call::UpdateFogSettings(Some(_))]
        ));

        f.calls.lock().unwrap().calls.clear();
        let off = VolumetricFog {
            enabled: false,
            ..fog
        };
        assert!(apply_fog(&mut f.world, Some(&off)));
        assert_eq!(f.replay(), vec![Call::UpdateFogSettings(None)]);
    }

    // A world that started without fog has no pass to hand it to, so enabling
    // fog is refused and nothing is recorded.
    #[test]
    fn fog_cannot_be_turned_on_where_the_pass_was_never_built() {
        let mut state = SettingsState::for_tests();
        state.fog_built = false;
        let mut f = Fixture::with_state(state);
        assert!(!fog_pass_built(&f.world));

        let fog = VolumetricFog {
            enabled: true,
            ..Default::default()
        };
        assert!(!apply_fog(&mut f.world, Some(&fog)));
        assert!(f.replay().is_empty());
        assert!(
            apply_fog(&mut f.world, None),
            "turning fog off never needs the pass"
        );
    }

    // Every name in the live-field lists is a real field of the section it
    // names, so a schema rename breaks this instead of silently sending the
    // edit to a rebuild forever.
    #[test]
    fn the_live_field_lists_name_real_args() {
        use concinnity_core::ecs::AssetFields;
        let params: Vec<String> = GraphicsConfig::field_table()
            .params
            .into_iter()
            .map(|p| p.path)
            .collect();
        for (section, fields) in LIVE_SETTINGS_FIELDS {
            for field in *fields {
                let path = format!("{section}.{field}");
                assert!(params.contains(&path), "{path}");
            }
        }
    }

    // A look edit that leaves the shadows section alone records no cadence,
    // a shadows edit records no composite update, and a sub-quality edit
    // pushes the feature scalars alone.
    #[test]
    fn an_edit_pushes_only_the_groups_it_moved() {
        let mut f = Fixture::new();
        assert!(apply_graphics(&mut f.world, &shadows()));
        assert!(
            f.replay()
                .iter()
                .all(|c| matches!(c, Call::SetShadowCadence(_)))
        );
        f.calls.lock().unwrap().calls.clear();
        let mut look = shadows();
        look.ambient_occlusion.radius = Some(2.0);
        assert!(apply_graphics(&mut f.world, &look));
        let calls = f.replay();
        assert_eq!(calls, [Call::UpdateQualityParams]);
    }
}
