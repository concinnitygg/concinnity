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

use concinnity_core::components::{
    AmbientOcclusion, Bloom, DirectionalLight, Exposure, IndirectLighting, Reflections, Shadows,
    Vignette, VolumetricFog,
};
use concinnity_core::ecs::World;
use concinnity_core::render::ops::RenderOps;
use concinnity_core::render::volumetric_fog;
use concinnity_core::settings::{QualityEffects, WorldSettings};

use crate::ecs::ActiveRenderQueues;
use crate::settings::system::{SettingsSlot, SettingsState};

/// The settings assets this module can apply to a running world, each with the
/// fields it applies live: the shadow cadence the per-frame cascade math reads,
/// and every look-tuning scalar the settings menu exposes as a slider. The
/// rest (shadow map resolution, the feature toggles and the resolution knobs)
/// size or gate a GPU resource built at load and need the world rebuilt.
pub const LIVE_SETTINGS_FIELDS: &[(&str, &[&str])] = &[
    ("Shadows", &["distance", "cascades", "update"]),
    ("AmbientOcclusion", &["radius", "intensity"]),
    ("Reflections", &["intensity", "max_distance"]),
    (
        "IndirectLighting",
        &["ambient_intensity", "ssgi_intensity", "ssgi_max_distance"],
    ),
    (
        "Exposure",
        &["ev", "auto_min_ev", "auto_max_ev", "auto_speed"],
    ),
    ("Bloom", &["intensity", "threshold", "knee"]),
    ("Vignette", &["strength"]),
];

/// The fields of `asset` this module applies live, or `None` when it applies
/// none of that type.
pub fn live_settings_fields(asset: &str) -> Option<&'static [&'static str]> {
    LIVE_SETTINGS_FIELDS
        .iter()
        .find(|(name, _)| *name == asset)
        .map(|(_, fields)| *fields)
}

/// An edited settings asset to apply to a running world. Only the fields in
/// [`LIVE_SETTINGS_FIELDS`] reach the picture; the rest move the authored
/// baseline the settings menu re-clamps from and take effect at the next
/// launch.
#[derive(Clone, Debug)]
pub enum SettingsEdit {
    /// The cascade schedule.
    Shadows(Shadows),
    /// The occlusion radius and strength.
    AmbientOcclusion(AmbientOcclusion),
    /// The reflection strength and reach.
    Reflections(Reflections),
    /// The ambient scale and the bounce strength and reach.
    IndirectLighting(IndirectLighting),
    /// The exposure offset and the auto-exposure bounds and speed.
    Exposure(Exposure),
    /// The bloom tunables.
    Bloom(Bloom),
    /// The vignette strength.
    Vignette(Vignette),
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

/// Apply an edited settings asset to a running world. Returns whether the
/// world took it.
pub fn apply_settings(world: &mut World, edit: &SettingsEdit) -> bool {
    match edit {
        SettingsEdit::Shadows(shadows) => apply_shadows(world, shadows),
        look => apply_look(world, look),
    }
}

// Apply an edited `Shadows`' cascade schedule.
fn apply_shadows(world: &mut World, shadows: &Shadows) -> bool {
    with_live(world, |state, ops| {
        state.authored.shadows = shadows.clone();
        let user = state.persisted_graphics().clone();
        let resolved = state.resolve_quality(&user, &state.ceiling());

        let quality = &mut state.graphics.quality;
        quality.shadow_map_size = resolved.shadow_map_size;
        let cadence = resolved.shadow_cadence;
        if cadence != quality.shadow_cadence {
            quality.shadow_cadence = cadence;
            ops.record(move |backend| backend.set_shadow_cadence(cadence));
        }
    })
    .is_some()
}

// Apply an edited look-tuning asset: the composite params, the per-feature
// sub-quality settings, and the ambient scale.
fn apply_look(world: &mut World, edit: &SettingsEdit) -> bool {
    with_live(world, |state, ops| {
        adopt_live_scalars(&mut state.authored, edit);
        let user = state.persisted_graphics().clone();
        let resolved = state.resolve(&user, &state.ceiling());

        let effects = &mut state.graphics.quality.effects;
        copy_live_effects(effects, &resolved.quality.effects);

        // `fxaa` follows the live AA mode, which no field here can move.
        let mut params = resolved.post_process;
        params.fxaa = effects.anti_aliasing.mode.fxaa_flag();
        state.graphics.post_process = params;
        ops.record(move |backend| backend.update_post_process(params));

        let quality = state.graphics.quality.quality_settings();
        ops.record(move |backend| backend.update_quality_params(quality));

        let ambient = resolved.ambient_intensity;
        if ambient != state.graphics.ambient_intensity {
            state.graphics.ambient_intensity = ambient;
            ops.record(move |backend| backend.set_ambient_intensity(ambient));
        }
    })
    .is_some()
}

// Move the authored baseline's live scalars to an edited asset's, leaving the
// toggles and resolution knobs beside them alone.
fn adopt_live_scalars(world: &mut WorldSettings, edit: &SettingsEdit) {
    let effects = &mut world.effects;
    match edit {
        SettingsEdit::Shadows(_) => {}
        SettingsEdit::AmbientOcclusion(src) => {
            let dst = &mut effects.ambient_occlusion;
            dst.radius = src.radius;
            dst.intensity = src.intensity;
        }
        SettingsEdit::Reflections(src) => {
            let dst = &mut effects.reflections;
            dst.intensity = src.intensity;
            dst.max_distance = src.max_distance;
        }
        SettingsEdit::IndirectLighting(src) => {
            let dst = &mut effects.indirect_lighting;
            dst.ambient_intensity = src.ambient_intensity;
            dst.ssgi_intensity = src.ssgi_intensity;
            dst.ssgi_max_distance = src.ssgi_max_distance;
        }
        SettingsEdit::Exposure(src) => {
            let dst = &mut effects.exposure;
            dst.ev = src.ev;
            dst.auto_min_ev = src.auto_min_ev;
            dst.auto_max_ev = src.auto_max_ev;
            dst.auto_speed = src.auto_speed;
        }
        SettingsEdit::Bloom(src) => world.bloom = src.clone(),
        SettingsEdit::Vignette(src) => world.vignette = src.clone(),
    }
}

// Copy the live effect scalars from resolved effects onto held ones.
fn copy_live_effects(dst: &mut QualityEffects, src: &QualityEffects) {
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
    use concinnity_core::components::AaMode;
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

    fn shadows() -> SettingsEdit {
        SettingsEdit::Shadows(Shadows {
            distance: 120,
            cascades: 2,
            ..Default::default()
        })
    }

    // A world with no renderer takes nothing, and says so rather than
    // pretending: that is the signal the caller rebuilds on.
    // What this module applies live is exactly what the schema declares live:
    // a field outside these lists waits for a rebuild or a relaunch.
    #[test]
    fn the_live_fields_are_the_ones_the_schema_declares_live() {
        use concinnity_core::components::{
            AntiAliasing, Culling, Display, FramePacing, TextureFiltering,
        };
        use concinnity_core::ecs::AssetFields;
        use concinnity_core::ecs::asset_fields::ApplyClass;
        fn live<A: AssetFields>() -> Vec<&'static str> {
            let mut keys: Vec<&str> = A::PARAMS
                .iter()
                .filter(|(_, meta)| meta.apply == Some(ApplyClass::Live))
                .map(|(key, _)| *key)
                .collect();
            keys.sort_unstable();
            keys
        }
        let listed = |asset: &str| {
            let mut keys = live_settings_fields(asset).unwrap_or_default().to_vec();
            keys.sort_unstable();
            keys
        };
        assert_eq!(listed("Shadows"), live::<Shadows>());
        assert_eq!(listed("AmbientOcclusion"), live::<AmbientOcclusion>());
        assert_eq!(listed("Reflections"), live::<Reflections>());
        assert_eq!(listed("IndirectLighting"), live::<IndirectLighting>());
        assert_eq!(listed("Exposure"), live::<Exposure>());
        assert_eq!(listed("Bloom"), live::<Bloom>());
        assert_eq!(listed("Vignette"), live::<Vignette>());
        // The frame pacing is live through the settings menu, not this seam.
        assert_eq!(live::<FramePacing>(), ["fps_cap", "vsync"]);
        for none in [
            live::<TextureFiltering>(),
            live::<AntiAliasing>(),
            live::<Display>(),
            live::<Culling>(),
        ] {
            assert!(none.is_empty(), "{none:?}");
        }
    }

    #[test]
    fn a_world_without_a_renderer_takes_nothing() {
        let mut world = World::new();
        assert!(!is_available(&world));
        assert!(!apply_directional_lights(&mut world, &[]));
        assert!(!apply_settings(&mut world, &shadows()));
        assert!(!apply_settings(
            &mut world,
            &SettingsEdit::Bloom(Bloom::default())
        ));
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
        assert!(apply_settings(&mut f.world, &shadows()));
        let calls = f.replay();
        let state = f.state();
        assert_eq!(
            calls,
            vec![Call::SetShadowCadence(
                state.graphics.quality.shadow_cadence
            )]
        );
        assert_eq!(state.graphics.quality.shadow_cadence.distance, 120);
        assert_eq!(state.authored.shadows.distance, 120);
        assert_eq!(state.graphics.quality.shadow_cadence.cascades, 2);
        assert_eq!(state.authored.shadows.cascades, 2);
    }

    // Re-applying the same config records nothing: the seam is edge-triggered,
    // so a burst of edits that touch other fields never restates the cadence.
    #[test]
    fn an_unchanged_cadence_records_no_call() {
        let mut f = Fixture::new();
        apply_settings(&mut f.world, &shadows());
        f.replay();
        f.calls.lock().unwrap().calls.clear();
        apply_settings(&mut f.world, &shadows());
        assert!(f.replay().is_empty());
    }

    // A row the user has overridden in the settings menu keeps the user's
    // value: that is what a relaunch of the edited world would show, so the
    // edit moves the authored baseline and leaves the picture alone.
    #[test]
    fn a_persisted_override_outranks_the_edited_world() {
        let mut state = SettingsState::for_tests();
        state.persisted_graphics.shadow_distance = Some(500);
        state.graphics.quality.shadow_cadence.distance = 500;
        let mut f = Fixture::with_state(state);

        assert!(apply_settings(&mut f.world, &shadows()));
        let calls = f.replay();
        let [Call::SetShadowCadence(cadence)] = calls.as_slice() else {
            panic!("one cadence push expected: {calls:?}");
        };
        assert_eq!(cadence.distance, 500, "the overridden row does not move");
        assert_eq!(cadence.cascades, 2, "the rest do");
        let state = f.state();
        assert_eq!(
            state.graphics.quality.shadow_cadence.distance, 500,
            "the user's value stands"
        );
        assert_eq!(
            state.authored.shadows.distance, 120,
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

        let config = SettingsEdit::Shadows(Shadows {
            distance: ceiling.shadow_distance + 1_000,
            ..Default::default()
        });
        assert!(apply_settings(&mut f.world, &config));
        f.replay();
        assert_eq!(
            f.state().graphics.quality.shadow_cadence.distance,
            ceiling.shadow_distance,
            "the ceiling holds"
        );
    }

    // A post-process edit pushes the composite params, the per-feature settings,
    // and the ambient scale, and the held copies follow.
    #[test]
    fn a_post_process_edit_pushes_every_live_knob() {
        let mut f = Fixture::new();
        let config = IndirectLighting {
            ambient_intensity: 0.25,
            ssgi_intensity: 0.75,
            ..Default::default()
        };
        let exposure = Exposure {
            ev: 2.0,
            ..Default::default()
        };
        assert!(apply_settings(
            &mut f.world,
            &SettingsEdit::IndirectLighting(config)
        ));
        assert!(apply_settings(
            &mut f.world,
            &SettingsEdit::Exposure(exposure)
        ));
        let calls = f.replay();
        assert!(calls.contains(&Call::UpdatePostProcess));
        assert!(calls.contains(&Call::UpdateQualityParams));
        assert!(calls.contains(&Call::SetAmbientIntensity(0.25)));
        let state = f.state();
        assert_eq!(state.graphics.ambient_intensity, 0.25);
        let held = &state.graphics.quality.effects;
        assert_eq!(held.indirect_lighting.ssgi_intensity, 0.75);
        assert_eq!(held.exposure.ev, 2.0);
        assert_eq!(
            state.authored.effects.indirect_lighting.ssgi_intensity,
            0.75
        );
        assert_eq!(state.graphics.post_process.exposure, 4.0);
    }

    // The toggles beside the live scalars are not this seam's to move: an edit
    // that reached here with one flipped must leave the held config alone, so a
    // caller that failed to gate it cannot silently half-apply.
    #[test]
    fn a_post_process_edit_leaves_the_toggles_alone() {
        let mut f = Fixture::new();
        let config = AmbientOcclusion {
            enabled: !AmbientOcclusion::default().enabled,
            ..Default::default()
        };
        let held_mode = f.state().graphics.quality.effects.anti_aliasing.mode;
        assert_ne!(held_mode, AaMode::Off);
        apply_settings(&mut f.world, &SettingsEdit::AmbientOcclusion(config));
        let state = f.state();
        assert_eq!(
            state.graphics.quality.effects.ambient_occlusion.enabled,
            AmbientOcclusion::default().enabled
        );
        assert_eq!(
            state.authored.effects.ambient_occlusion.enabled,
            AmbientOcclusion::default().enabled
        );
        assert_eq!(state.graphics.quality.effects.anti_aliasing.mode, held_mode);
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

    // Every name in the live-field lists is a real field of the asset it names,
    // so a schema rename breaks this instead of silently sending the edit to a
    // rebuild forever.
    #[test]
    fn the_live_field_lists_name_real_args() {
        let args = |asset: &str| match asset {
            "Shadows" => serde_json::to_value(Shadows::default()),
            "AmbientOcclusion" => serde_json::to_value(AmbientOcclusion::default()),
            "Reflections" => serde_json::to_value(Reflections::default()),
            "IndirectLighting" => serde_json::to_value(IndirectLighting::default()),
            "Exposure" => serde_json::to_value(Exposure::default()),
            "Bloom" => serde_json::to_value(Bloom::default()),
            "Vignette" => serde_json::to_value(Vignette::default()),
            other => panic!("{other} has no args here"),
        };
        for (asset, fields) in LIVE_SETTINGS_FIELDS {
            let args = args(asset).expect("serializes");
            for field in *fields {
                assert!(args.get(field).is_some(), "{asset}.{field}");
            }
        }
    }
}
