use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use super::super::test_support::Fixture;
use super::*;
use crate::components::{
    AaMode, AmbientOcclusion, AmbientOcclusionOverrides, ExposureOverrides,
    IndirectLightingOverrides, PlanetDetailOverrides, ShadowsOverrides,
};
use crate::render::backend::recording::Call;
use crate::settings::QualityPreset;

fn fixture() -> Fixture {
    with_state(SettingsState::for_tests())
}

fn with_state(state: SettingsState) -> Fixture {
    let mut world = World::new();
    world.insert_resource(SettingsSlot::new(state));
    Fixture::new(world)
}

fn state(f: &mut Fixture) -> &mut SettingsState {
    f.world
        .resource_mut::<SettingsSlot>()
        .and_then(|slot| slot.0.as_mut())
        .expect("parked")
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

// What this module applies live is exactly what the schema declares live: a
// field outside these lists waits for a rebuild or a relaunch.
#[test]
fn the_live_fields_are_the_ones_the_schema_declares_live() {
    use crate::components::GRAPHICS_SECTIONS;
    use crate::ecs::AssetFields;
    use crate::ecs::asset_fields::ApplyClass;
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

// Every name in the live-field lists is a real field of the section it names,
// so a schema rename breaks this instead of silently sending the edit to a
// rebuild forever.
#[test]
fn the_live_field_lists_name_real_args() {
    use crate::ecs::AssetFields;
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

// The derived live copy moves the live fields only and reports whether any did.
#[test]
fn copy_live_moves_only_the_live_fields() {
    use crate::components::Shadows;
    let mut held = Shadows::default();
    let edited = Shadows {
        distance: held.distance + 10,
        map_size: held.map_size * 2,
        ..held.clone()
    };
    let map_size = held.map_size;
    assert!(held.copy_live(&edited));
    assert_eq!(held.distance, edited.distance);
    assert_eq!(held.map_size, map_size, "map size rebuilds, it is not live");
    assert!(!held.copy_live(&edited), "nothing left to move");
}

// A world with no renderer takes nothing, and says so rather than pretending:
// that is the signal the caller rebuilds on.
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
    let mut f = fixture();
    let sun = DirectionalLight {
        direction: [0.0, 1.0, 0.0],
        color: [1.0, 0.5, 0.25],
        intensity: 3.0,
    };
    assert!(apply_directional_lights(
        &mut f.world,
        core::slice::from_ref(&sun)
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

// The live shadow knobs push; the authored baselines move with them so a later
// preset change re-clamps from the edited world, not the old one.
#[test]
fn shadow_knobs_push_and_move_their_baselines() {
    let mut f = fixture();
    assert!(apply_graphics(&mut f.world, &shadows()));
    let calls = f.replay();
    let state = state(&mut f);
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
// streaming reads, under the preset's ceiling, with nothing for the renderer
// to do.
#[test]
fn the_finest_planet_cell_size_applies_under_the_ceiling() {
    let mut f = fixture();
    let shadows = state(&mut f).authored.graphics.shadows.clone();
    let config = |size| GraphicsConfig {
        shadows: shadows.clone().into(),
        planet_detail: PlanetDetailOverrides {
            finest_cell_size: Some(size),
        },
        ..Default::default()
    };
    assert!(apply_graphics(&mut f.world, &config(3)));
    assert!(f.replay().is_empty());
    let s = state(&mut f);
    assert_eq!(s.authored.graphics.planet_detail.finest_cell_size, 3);
    assert_eq!(s.resolved.graphics.planet_detail.finest_cell_size, 3);

    s.quality_preset = QualityPreset::Medium;
    assert!(apply_graphics(&mut f.world, &config(1)));
    let s = state(&mut f);
    assert_eq!(s.authored.graphics.planet_detail.finest_cell_size, 1);
    assert_eq!(s.resolved.graphics.planet_detail.finest_cell_size, 2);
}

// Re-applying the same config records nothing: the seam is edge-triggered, so
// a burst of edits that touch other fields never restates the cadence.
#[test]
fn an_unchanged_cadence_records_no_call() {
    let mut f = fixture();
    apply_graphics(&mut f.world, &shadows());
    f.replay();
    f.clear();
    apply_graphics(&mut f.world, &shadows());
    assert!(f.replay().is_empty());
}

// A row the user has overridden in the settings menu keeps the user's value:
// that is what a relaunch of the edited world would show, so the edit moves
// the authored baseline and leaves the picture alone.
#[test]
fn a_persisted_override_outranks_the_edited_world() {
    let mut s = SettingsState::for_tests();
    s.persisted_graphics.shadow_distance = Some(500);
    s.resolved.graphics.shadows.distance = 500;
    let mut f = with_state(s);

    assert!(apply_graphics(&mut f.world, &shadows()));
    let calls = f.replay();
    let [Call::SetShadowCadence(cadence)] = calls.as_slice() else {
        panic!("one cadence push expected: {calls:?}");
    };
    assert_eq!(cadence.distance, 500, "the overridden row does not move");
    assert_eq!(cadence.cascades, 2, "the rest do");
    let s = state(&mut f);
    assert_eq!(
        s.resolved.graphics.shadows.distance, 500,
        "the user's value stands"
    );
    assert_eq!(
        s.authored.graphics.shadows.distance, 120,
        "the world's value is still recorded as the baseline"
    );
}

// The preset ceiling clamps an edited value exactly as it clamps an authored
// one at launch.
#[test]
fn the_quality_ceiling_clamps_an_edited_knob() {
    let mut s = SettingsState::for_tests();
    s.quality_preset = QualityPreset::Low;
    let mut f = with_state(s);
    let ceiling = state(&mut f).ceiling();

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
        state(&mut f).resolved.graphics.shadows.distance,
        ceiling.shadow_distance,
        "the ceiling holds"
    );
}

// A post-process edit pushes the composite params, the per-feature settings,
// and the ambient scale, and the held copies follow.
#[test]
fn a_post_process_edit_pushes_every_live_knob() {
    let mut f = fixture();
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
    let s = state(&mut f);
    assert_eq!(s.resolved.graphics.ambient_intensity(), 0.25);
    let held = &s.resolved.graphics;
    assert_eq!(held.indirect_lighting.ssgi_intensity, 0.75);
    assert_eq!(held.exposure.ev, 2.0);
    assert_eq!(s.authored.graphics.indirect_lighting.ssgi_intensity, 0.75);
    assert_eq!(s.resolved.graphics.post_process().exposure, 4.0);
}

// The toggles beside the live scalars are not this seam's to move: an edit
// that reached here with one flipped must leave the held config alone, so a
// caller that failed to gate it cannot silently half-apply.
#[test]
fn a_post_process_edit_leaves_the_toggles_alone() {
    let mut f = fixture();
    let config = GraphicsConfig {
        ambient_occlusion: AmbientOcclusionOverrides {
            enabled: Some(!AmbientOcclusion::default().enabled),
            radius: Some(1.0),
            ..Default::default()
        },
        ..Default::default()
    };
    let held_mode = state(&mut f).resolved.graphics.anti_aliasing.mode;
    assert_ne!(held_mode, AaMode::Off);
    apply_graphics(&mut f.world, &config);
    let s = state(&mut f);
    let enabled = AmbientOcclusion::default().enabled;
    assert_eq!(s.resolved.graphics.ambient_occlusion.enabled, enabled);
    assert_eq!(s.authored.graphics.ambient_occlusion.enabled, enabled);
    assert_eq!(s.resolved.graphics.anti_aliasing.mode, held_mode);
}

// A look edit that leaves the shadows section alone records no cadence, a
// shadows edit records no composite update, and a sub-quality edit pushes the
// feature scalars alone.
#[test]
fn an_edit_pushes_only_the_groups_it_moved() {
    let mut f = fixture();
    assert!(apply_graphics(&mut f.world, &shadows()));
    assert!(
        f.replay()
            .iter()
            .all(|c| matches!(c, Call::SetShadowCadence(_)))
    );
    f.clear();
    let mut look = shadows();
    look.ambient_occlusion.radius = Some(2.0);
    assert!(apply_graphics(&mut f.world, &look));
    assert_eq!(f.replay(), [Call::UpdateQualityParams]);
}

// The look copy reports a move only when a live scalar differs, and never
// touches the toggles beside them.
#[test]
fn the_look_copy_moves_only_live_scalars() {
    let held = SettingsState::for_tests().resolved.graphics;
    let mut dst = held.clone();
    assert!(!copy_look(&mut dst, &held));
    let mut src = held.clone();
    src.bloom.knee += 0.5;
    src.ambient_occlusion.enabled = !held.ambient_occlusion.enabled;
    assert!(copy_look(&mut dst, &src));
    assert_eq!(dst.bloom.knee, src.bloom.knee);
    assert_eq!(
        dst.ambient_occlusion.enabled,
        held.ambient_occlusion.enabled
    );
}

// Fog resolves through the same clamp chain the build runs, and disabling it
// pushes `None` rather than settings the pass would still draw.
#[test]
fn fog_pushes_resolved_settings_and_disables_with_none() {
    let mut f = fixture();
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

    f.clear();
    let off = VolumetricFog {
        enabled: false,
        ..fog
    };
    assert!(apply_fog(&mut f.world, Some(&off)));
    assert_eq!(f.replay(), vec![Call::UpdateFogSettings(None)]);
}

// A world that started without fog has no pass to hand it to, so enabling fog
// is refused and nothing is recorded.
#[test]
fn fog_cannot_be_turned_on_where_the_pass_was_never_built() {
    let mut s = SettingsState::for_tests();
    s.fog_built = false;
    let mut f = with_state(s);
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
