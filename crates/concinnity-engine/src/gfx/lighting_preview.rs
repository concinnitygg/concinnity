// src/gfx/lighting_preview.rs
//
// Live application of the world's lighting assets. The renderer reads its
// lighting from GPU state built at load -- the sun is packed into the shared
// light uniforms, the shadow and post-process knobs are resolved once -- so an
// editor changing one of them has nothing in the ECS to write: the effect is a
// backend call. This is that seam. Each function records the calls into the
// frame's op queue, where submission replays them before the next draw, and
// keeps the settings state in step so the settings menu and an authoring edit
// never disagree about what is live.
//
// A knob resolves the same way `run_init` resolves it: to the user's persisted
// settings-menu choice where they made one, otherwise to the world's value
// under the quality preset's ceiling (`gfx::render_config` holds the shared
// expressions). So editing a row the user has overridden moves the authored
// baseline and leaves the picture alone, exactly as relaunching would.

use crate::components::{DirectionalLight, GraphicsConfig, PostProcessConfig, VolumetricFog};
use crate::ecs::{ActiveRenderQueues, World};
use crate::gfx::render_config as resolve;
use crate::gfx::settings_system::{SettingsSlot, SettingsState};
use crate::gfx::volumetric_fog::FogSettings;
use concinnity_render::ops::RenderOps;

/// The `GraphicsConfig` fields this module can apply to a running world. The
/// rest (shadow map resolution, frames in flight, the sampler's anisotropy)
/// size a GPU resource at load and need the world rebuilt.
pub const LIVE_GRAPHICS_CONFIG_FIELDS: &[&str] =
    &["shadow_distance", "shadow_cascades", "shadow_update"];

/// The `PostProcessConfig` fields this module can apply to a running world:
/// every look-tuning scalar the settings menu exposes as a slider. The feature
/// toggles and the resolution knobs beside them gate render passes whose
/// pipelines and targets are built at load, so they need the world rebuilt.
pub const LIVE_POST_PROCESS_FIELDS: &[&str] = &[
    "ambient_intensity",
    "exposure_ev",
    "bloom_intensity",
    "bloom_threshold",
    "bloom_knee",
    "vignette_strength",
    "lut_strength",
    "ssao_radius",
    "ssao_intensity",
    "ssr_intensity",
    "ssr_max_distance",
    "ssgi_intensity",
    "ssgi_max_distance",
    "auto_exposure_min_ev",
    "auto_exposure_max_ev",
    "auto_exposure_speed",
];

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
    let settings = fog.filter(|f| f.enabled).map(|f| {
        FogSettings::resolve(
            f.color,
            f.density,
            f.height_falloff,
            f.height_reference,
            f.max_distance,
            f.phase_g,
            f.ambient,
        )
    });
    if settings.is_some() && !fog_pass_built(world) {
        return false;
    }
    with_live(world, |_, ops| {
        ops.record(move |backend| backend.update_fog_settings(settings));
    })
    .is_some()
}

/// Apply an edited `GraphicsConfig`'s shadow knobs. Fields outside
/// [`LIVE_GRAPHICS_CONFIG_FIELDS`] only move the authored baseline the settings
/// menu re-clamps from; they take effect at the next launch.
pub fn apply_graphics_config(world: &mut World, config: &GraphicsConfig) -> bool {
    with_live(world, |state, ops| {
        let ceiling = state.ceiling();
        let user = state.persisted_graphics().clone();

        state.authored_shadow_map_size = config.shadow_map_size;
        state.authored_anisotropy = config.anisotropy;
        state.shadow_map_size = resolve::shadow_map_size(config.shadow_map_size, &user, &ceiling);
        state.anisotropy = resolve::anisotropy(config.anisotropy, &user, &ceiling);

        state.authored_shadow_update = config.shadow_update;
        let update = resolve::shadow_update(config.shadow_update, &user, &ceiling);
        if update != state.shadow_update {
            state.shadow_update = update;
            ops.record(move |backend| backend.set_shadow_update(update));
        }
        state.authored_shadow_distance = config.shadow_distance;
        let distance = resolve::shadow_distance(config.shadow_distance, &user, &ceiling);
        if distance != state.shadow_distance {
            state.shadow_distance = distance;
            ops.record(move |backend| backend.set_shadow_distance(distance));
        }
        state.authored_shadow_cascades = config.shadow_cascades;
        let cascades = resolve::shadow_cascades(config.shadow_cascades, &user, &ceiling);
        if cascades != state.shadow_cascades {
            state.shadow_cascades = cascades;
            ops.record(move |backend| backend.set_shadow_cascades(cascades));
        }
    })
    .is_some()
}

/// Apply an edited `PostProcessConfig`'s look-tuning scalars: the composite
/// params, the per-feature sub-quality settings, and the ambient scale. Fields
/// outside [`LIVE_POST_PROCESS_FIELDS`] are ignored here.
pub fn apply_post_process_config(world: &mut World, config: &PostProcessConfig) -> bool {
    with_live(world, |state, ops| {
        let user = state.persisted_graphics().clone();

        copy_live_scalars(&mut state.authored_post_config, config);
        copy_live_scalars(&mut state.post_config, config);
        resolve::overlay_quality_scalars(&mut state.post_config, &user);

        // The composite params re-resolve from the edited config; `fxaa` follows
        // the live AA mode, which no field here can move.
        let mut params = resolve::post_process_params(Some(config), &user);
        params.fxaa = state.post_config.aa_mode.fxaa_flag();
        state.post_process = params;
        ops.record(move |backend| backend.update_post_process(params));

        let quality = crate::gfx::graphics_system::derive_quality_settings(&state.post_config);
        ops.record(move |backend| backend.update_quality_params(quality));

        let ambient = resolve::ambient_intensity(Some(config), &user);
        if ambient != state.ambient_intensity {
            state.ambient_intensity = ambient;
            ops.record(move |backend| backend.set_ambient_intensity(ambient));
        }
    })
    .is_some()
}

// Copy the fields in `LIVE_POST_PROCESS_FIELDS` from an edited config onto a
// held one, leaving the toggles and resolution knobs beside them alone.
fn copy_live_scalars(dst: &mut PostProcessConfig, src: &PostProcessConfig) {
    dst.ambient_intensity = src.ambient_intensity;
    dst.exposure_ev = src.exposure_ev;
    dst.bloom_intensity = src.bloom_intensity;
    dst.bloom_threshold = src.bloom_threshold;
    dst.bloom_knee = src.bloom_knee;
    dst.vignette_strength = src.vignette_strength;
    dst.lut_strength = src.lut_strength;
    dst.ssao_radius = src.ssao_radius;
    dst.ssao_intensity = src.ssao_intensity;
    dst.ssr_intensity = src.ssr_intensity;
    dst.ssr_max_distance = src.ssr_max_distance;
    dst.ssgi_intensity = src.ssgi_intensity;
    dst.ssgi_max_distance = src.ssgi_max_distance;
    dst.auto_exposure_min_ev = src.auto_exposure_min_ev;
    dst.auto_exposure_max_ev = src.auto_exposure_max_ev;
    dst.auto_exposure_speed = src.auto_exposure_speed;
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

    fn graphics_config() -> GraphicsConfig {
        GraphicsConfig {
            shadow_distance: 120,
            shadow_cascades: 2,
            ..Default::default()
        }
    }

    // A world with no renderer takes nothing, and says so rather than
    // pretending: that is the signal the caller rebuilds on.
    #[test]
    fn a_world_without_a_renderer_takes_nothing() {
        let mut world = World::new();
        assert!(!is_available(&world));
        assert!(!apply_directional_lights(&mut world, &[]));
        assert!(!apply_graphics_config(
            &mut world,
            &GraphicsConfig::default()
        ));
        assert!(!apply_post_process_config(
            &mut world,
            &PostProcessConfig::default()
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
        assert!(apply_graphics_config(&mut f.world, &graphics_config()));
        let calls = f.replay();
        assert!(calls.contains(&Call::SetShadowDistance(120)));
        assert!(calls.contains(&Call::SetShadowCascades(2)));
        let state = f.state();
        assert_eq!(state.shadow_distance, 120);
        assert_eq!(state.authored_shadow_distance, 120);
        assert_eq!(state.shadow_cascades, 2);
        assert_eq!(state.authored_shadow_cascades, 2);
    }

    // Re-applying the same config records nothing: the seam is edge-triggered,
    // so a burst of edits that touch other fields never restates a knob.
    #[test]
    fn an_unchanged_knob_records_no_call() {
        let mut f = Fixture::new();
        apply_graphics_config(&mut f.world, &graphics_config());
        f.replay();
        f.calls.lock().unwrap().calls.clear();
        apply_graphics_config(&mut f.world, &graphics_config());
        assert!(f.replay().is_empty());
    }

    // A row the user has overridden in the settings menu keeps the user's
    // value: that is what a relaunch of the edited world would show, so the
    // edit moves the authored baseline and leaves the picture alone.
    #[test]
    fn a_persisted_override_outranks_the_edited_world() {
        let mut state = SettingsState::for_tests();
        state.persisted_graphics.shadow_distance = Some(500);
        state.shadow_distance = 500;
        let mut f = Fixture::with_state(state);

        assert!(apply_graphics_config(&mut f.world, &graphics_config()));
        let calls = f.replay();
        assert!(
            !calls
                .iter()
                .any(|c| matches!(c, Call::SetShadowDistance(_))),
            "the overridden row does not move"
        );
        assert!(calls.contains(&Call::SetShadowCascades(2)), "the rest do");
        let state = f.state();
        assert_eq!(state.shadow_distance, 500, "the user's value stands");
        assert_eq!(
            state.authored_shadow_distance, 120,
            "the world's value is still recorded as the baseline"
        );
    }

    // The preset ceiling clamps an edited value exactly as it clamps an
    // authored one at launch.
    #[test]
    fn the_quality_ceiling_clamps_an_edited_knob() {
        let mut state = SettingsState::for_tests();
        state.quality_preset = crate::gfx::quality_preset::QualityPreset::Low;
        let mut f = Fixture::with_state(state);
        let ceiling = f.state().ceiling();

        let mut config = graphics_config();
        config.shadow_distance = ceiling.shadow_distance + 1_000;
        assert!(apply_graphics_config(&mut f.world, &config));
        f.replay();
        assert_eq!(
            f.state().shadow_distance,
            ceiling.shadow_distance,
            "the ceiling holds"
        );
    }

    // A post-process edit pushes the composite params, the per-feature settings,
    // and the ambient scale, and the held copies follow.
    #[test]
    fn a_post_process_edit_pushes_every_live_knob() {
        let mut f = Fixture::new();
        let config = PostProcessConfig {
            ambient_intensity: 0.25,
            exposure_ev: 2.0,
            ssgi_intensity: 0.75,
            ..Default::default()
        };
        assert!(apply_post_process_config(&mut f.world, &config));
        let calls = f.replay();
        assert!(calls.contains(&Call::UpdatePostProcess));
        assert!(calls.contains(&Call::UpdateQualityParams));
        assert!(calls.contains(&Call::SetAmbientIntensity(0.25)));
        let state = f.state();
        assert_eq!(state.ambient_intensity, 0.25);
        assert_eq!(state.post_config.ssgi_intensity, 0.75);
        assert_eq!(state.authored_post_config.ssgi_intensity, 0.75);
    }

    // The toggles beside the live scalars are not this seam's to move: an edit
    // that reached here with one flipped must leave the held config alone, so a
    // caller that failed to gate it cannot silently half-apply.
    #[test]
    fn a_post_process_edit_leaves_the_toggles_alone() {
        let mut f = Fixture::new();
        let config = PostProcessConfig {
            ssao: !PostProcessConfig::default().ssao,
            aa_mode: crate::components::AaMode::Fxaa,
            ..Default::default()
        };
        apply_post_process_config(&mut f.world, &config);
        let state = f.state();
        assert_eq!(state.post_config.ssao, PostProcessConfig::default().ssao);
        assert_eq!(
            state.post_config.aa_mode,
            PostProcessConfig::default().aa_mode
        );
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
        let graphics = serde_json::to_value(GraphicsConfig::default()).expect("serialises");
        for field in LIVE_GRAPHICS_CONFIG_FIELDS {
            assert!(graphics.get(field).is_some(), "GraphicsConfig.{field}");
        }
        let post = serde_json::to_value(PostProcessConfig::default()).expect("serialises");
        for field in LIVE_POST_PROCESS_FIELDS {
            assert!(post.get(field).is_some(), "PostProcessConfig.{field}");
        }
    }
}
