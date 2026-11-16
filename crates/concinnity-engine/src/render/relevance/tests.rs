use super::*;
use crate::components::Relevance;
use crate::gfx::render_types::ReflectionScopeGpu;
use crate::render::environment::grass::lod::GrassReach;
use crate::render::frame::plan::EffectInputs;
use crate::render::post::rt_reflections::RtReflectionSettings;
use crate::render::post::ssao::settings::SsaoSettings;
use crate::render::post::ssgi::settings::{SsgiFrame, SsgiSettings};
use crate::render::post::ssr::settings::SsrSettings;
use crate::render::view::{ScreenRect, Seen, Sighting, Subject, SubjectKind, ViewFacts};

const FOCAL: f32 = 1000.0;

// A subject of `kind` in view, `distance` meters away at its nearest point,
// seen at `nearest`.
fn subject(kind: SubjectKind, distance: f32, nearest: Option<f32>) -> Subject {
    Subject {
        kind,
        size: 1.0,
        distance,
        center_distance: distance + 0.5,
        clearance: 0.0,
        in_view: true,
        rect: ScreenRect::FULL,
        seen: Sighting {
            coverage: if nearest.is_some() { 0.1 } else { 0.0 },
            nearest,
        },
    }
}

// A still view at 1080p measured a frame ago, seeing `other` beyond the
// subjects.
fn facts(other: Option<f32>) -> ViewFacts {
    ViewFacts {
        focal_px: FOCAL,
        render_size: [1920, 1080],
        seen: Some(Seen {
            age: 1,
            drift_px: 0.0,
            sky: 0.5,
            other: Sighting {
                coverage: if other.is_some() { 0.1 } else { 0.0 },
                nearest: other,
            },
        }),
        ..ViewFacts::default()
    }
}

fn reach() -> EffectReach {
    EffectReach {
        ambient_occlusion: Some(0.5),
        indirect_lighting: Some(4.0),
        reflections: Some(100.0),
        scene: SceneReach {
            shadow_cascades: Some(200.0),
            longest_wave: Some(300.0),
            prop_probe_reflects: true,
            ..SceneReach::default()
        },
    }
}

fn margins(facts: &ViewFacts) -> Margins {
    Margins::of(&Relevance::default(), facts, &reach())
}

#[test]
fn an_effect_reaching_a_pixel_at_the_nearest_surface_is_at_its_threshold() {
    // 0.5 m at 500 m is a pixel.
    let at = margins(&facts(Some(500.0)));
    assert!((at.get(Effect::AmbientOcclusion) - 1.0).abs() < 1e-4);
    let near = margins(&facts(Some(2.0)));
    assert!((near.get(Effect::AmbientOcclusion) - 250.0).abs() < 1e-2);
    let far = margins(&facts(Some(2000.0)));
    assert!(far.get(Effect::AmbientOcclusion) < 1.0);
    // The bounce reaches further, so it holds out longer.
    assert!(far.get(Effect::IndirectLighting) > 1.0);
}

// The same rule for a pebble and a planet: the planet seen only from far off
// adds no occlusion, the prop beside the camera does.
#[test]
fn a_near_prop_keeps_what_a_far_planet_would_drop() {
    let mut view = facts(None);
    view.subjects[0] = Some(subject(SubjectKind::Planet, 150_000.0, Some(150_500.0)));
    assert!(margins(&view).get(Effect::AmbientOcclusion) < 1.0);
    assert!(margins(&view).get(Effect::ShadowCascades) < 1.0);
    view.subjects[2] = Some(subject(SubjectKind::Prop, 3.5, Some(3.9)));
    assert!(margins(&view).get(Effect::AmbientOcclusion) > 1.0);
    assert!(margins(&view).get(Effect::ShadowCascades) > 1.0);
    // Out of view, the prop counts for nothing.
    view.subjects[2] = Some(Subject {
        in_view: false,
        ..subject(SubjectKind::Prop, 3.5, None)
    });
    assert!(margins(&view).get(Effect::AmbientOcclusion) < 1.0);
}

// The occlusion, the bounce and the cascades act between surfaces, and the
// prop's probe captures what it reflects from far off: a prop near the camera
// but far over everything else asks for none of them, the reflections
// included.
#[test]
fn a_prop_far_from_everything_asks_for_nothing_between_surfaces() {
    let mut view = facts(None);
    view.subjects[0] = Some(subject(SubjectKind::Planet, 150_000.0, Some(150_500.0)));
    view.subjects[2] = Some(Subject {
        clearance: 150_000.0,
        ..subject(SubjectKind::Prop, 3.5, Some(3.9))
    });
    let floating = margins(&view);
    assert!(floating.get(Effect::AmbientOcclusion) < 1.0);
    assert!(floating.get(Effect::IndirectLighting) < 1.0);
    assert!(floating.get(Effect::ShadowCascades) < 1.0);
    assert!(floating.get(Effect::Reflections) < 1.0);
    assert_eq!(floating.reflections.prop, 0.0);
    // Where ray-traced reflections take the probe's place, the prop keeps
    // them however far it floats.
    let mut traced = reach();
    traced.scene.prop_probe_reflects = false;
    let without_probe = Margins::of(&Relevance::default(), &view, &traced);
    assert!(without_probe.reflections.prop > 1.0);
    // Within the bounce's reach of the ground, only the bounce returns.
    view.subjects[2] = Some(Subject {
        clearance: 2.0,
        ..subject(SubjectKind::Prop, 3.5, Some(3.9))
    });
    let hovering = margins(&view);
    assert!(hovering.get(Effect::AmbientOcclusion) < 1.0);
    assert!(hovering.get(Effect::IndirectLighting) > 1.0);
    assert!(hovering.get(Effect::ShadowCascades) > 1.0);
    // The reflections reach the ground too, so they return on the prop.
    assert!(hovering.reflections.prop > 1.0);
    assert!(hovering.reflections.whole < 1.0);
}

// A measurement that stands and saw none of a subject leaves it out as hidden,
// unless the subject is too small for its samples to find.
#[test]
fn a_subject_the_measurement_missed_counts_only_when_too_small_to_sample() {
    let mut view = facts(None);
    // The sea 3 m under a camera looking at the sky.
    view.subjects[1] = Some(Subject {
        size: 1.0e5,
        center_distance: 5.0e4,
        ..subject(SubjectKind::Water, 3.0, None)
    });
    assert_eq!(margins(&view).get(Effect::WaterSim), 0.0);
    assert_eq!(margins(&view).get(Effect::AmbientOcclusion), 0.0);
    // A pebble 1 cm across 10 m away spans a pixel: under any sample cell.
    view.subjects[2] = Some(Subject {
        size: 0.01,
        center_distance: 10.0,
        ..subject(SubjectKind::Prop, 10.0, None)
    });
    assert!((margins(&view).get(Effect::AmbientOcclusion) - 50.0).abs() < 1e-2);
}

#[test]
fn grass_stands_down_where_no_blade_spans_its_last_pixel_and_a_half() {
    let mut reach = reach();
    reach.scene.grass = Some(GrassReach {
        blade_height: 0.6,
        sway: 0.0,
    });
    let mut meadow = facts(None);
    // A 0.6 m blade at 400 m is 1.5 pixels.
    meadow.subjects[0] = Some(subject(SubjectKind::Planet, 1.0, Some(400.0)));
    let edge = Margins::of(&Relevance::default(), &meadow, &reach);
    assert!((edge.get(Effect::Grass) - 1.0).abs() < 1e-3);
    meadow.subjects[0] = Some(subject(SubjectKind::Planet, 1.0, Some(150_000.0)));
    assert!(Margins::of(&Relevance::default(), &meadow, &reach).get(Effect::Grass) < 1.0);
    // Ground no subject holds grows grass too; water does not.
    let mut terrain = facts(Some(10.0));
    terrain.subjects[1] = Some(subject(SubjectKind::Water, 1.0, Some(1.0)));
    let near = Margins::of(&Relevance::default(), &terrain, &reach);
    assert!((near.get(Effect::Grass) - 40.0).abs() < 1e-2);
    // No grass, no rule.
    reach.scene.grass = None;
    let bare = Margins::of(&Relevance::default(), &meadow, &reach);
    assert_eq!(bare.get(Effect::Grass), f32::INFINITY);
}

#[test]
fn nothing_in_view_to_act_on_leaves_no_margin() {
    let sky = margins(&facts(None));
    for effect in [
        Effect::AmbientOcclusion,
        Effect::IndirectLighting,
        Effect::Reflections,
        Effect::ShadowCascades,
        Effect::WaterSim,
    ] {
        assert_eq!(sky.get(effect), 0.0, "{}", effect.name());
    }
}

#[test]
fn the_water_simulation_follows_the_nearest_visible_water_alone() {
    let mut view = facts(Some(2.0));
    view.subjects[1] = Some(subject(SubjectKind::Water, 3.0, Some(100_000.0)));
    // 300 m at 100 km is 3 pixels: the water's own fade.
    assert!((margins(&view).get(Effect::WaterSim) - 1.0).abs() < 1e-3);
    view.subjects[1] = Some(subject(SubjectKind::Water, 3.0, Some(150_000.0)));
    assert!(margins(&view).get(Effect::WaterSim) < 1.0);
    // Near water seen at the surface counts for the occlusion of what lies
    // under it, but reflects nothing the reflection pass resolves.
    let mut sea = facts(None);
    sea.subjects[1] = Some(subject(SubjectKind::Water, 3.0, Some(4.0)));
    assert!(margins(&sea).get(Effect::AmbientOcclusion) > 1.0);
    assert_eq!(margins(&sea).get(Effect::Reflections), 0.0);
}

#[test]
fn a_stale_measurement_falls_back_to_where_the_subjects_lie() {
    let mut view = facts(None);
    view.subjects[1] = Some(subject(SubjectKind::Water, 3.0, Some(150_000.0)));
    assert!(margins(&view).get(Effect::WaterSim) < 1.0);
    // Turning fast: the view moved further than a sample cell since.
    view.seen.as_mut().unwrap().drift_px = 100.0;
    assert!(margins(&view).get(Effect::WaterSim) > 1.0);
    // Surfaces no subject holds were seen, so they may be anywhere now.
    let mut room = facts(Some(1000.0));
    assert!(margins(&room).get(Effect::AmbientOcclusion) < 1.0);
    room.seen.as_mut().unwrap().drift_px = 100.0;
    assert_eq!(margins(&room).get(Effect::AmbientOcclusion), f32::INFINITY);
    // Before any measurement, so may everything.
    room.seen = None;
    assert_eq!(margins(&room).get(Effect::AmbientOcclusion), f32::INFINITY);
}

#[test]
fn the_probe_follows_its_prop_s_size_on_screen() {
    let mut view = facts(None);
    assert_eq!(margins(&view).get(Effect::PropProbe), f32::INFINITY);
    // 1 m across 125 m away is 8 pixels: half the threshold.
    view.subjects[2] = Some(Subject {
        center_distance: 125.0,
        ..subject(SubjectKind::Prop, 124.5, Some(124.6))
    });
    assert!((margins(&view).get(Effect::PropProbe) - 0.5).abs() < 1e-3);
    view.subjects[2] = Some(Subject {
        in_view: false,
        ..subject(SubjectKind::Prop, 1.0, None)
    });
    assert_eq!(margins(&view).get(Effect::PropProbe), 0.0);
}

#[test]
fn the_mark_follows_parallax_and_movers_once_something_is_measured() {
    let mut view = facts(Some(10.0));
    assert_eq!(margins(&view).get(Effect::Disocclusion), 0.0);
    view.motion_px = 3.0;
    view.parallax_px = 0.1;
    assert!((margins(&view).get(Effect::Disocclusion) - 0.1).abs() < 1e-4);
    view.mover_px = 4.0;
    assert!((margins(&view).get(Effect::Disocclusion) - 4.0).abs() < 1e-4);
    view.seen = None;
    assert_eq!(margins(&view).get(Effect::Disocclusion), f32::INFINITY);
}

#[test]
fn turning_the_rules_off_or_a_threshold_to_zero_keeps_effects_on() {
    let far = facts(Some(1.0e6));
    let off = Relevance {
        enabled: false,
        ..Relevance::default()
    };
    assert_eq!(Margins::of(&off, &far, &reach()), Margins::UNBOUNDED);
    let zero = Relevance {
        min_reach_pixels: 0.0,
        ..Relevance::default()
    };
    let m = Margins::of(&zero, &far, &reach());
    assert_eq!(m.get(Effect::AmbientOcclusion), f32::INFINITY);
    assert_eq!(m.get(Effect::Reflections), f32::INFINITY);
    assert!(m.get(Effect::ShadowCascades) < 1.0);
    // An effect that does not run has no rule.
    let none = Margins::of(&Relevance::default(), &far, &EffectReach::default());
    assert_eq!(none.get(Effect::IndirectLighting), f32::INFINITY);
}

fn with(effect: Effect, margin: f32) -> Margins {
    Margins::UNBOUNDED.with(effect, margin)
}

// Each `step` seconds from `from` up to `to`, the state's update over
// `margins`, returning the last.
fn run(state: &mut RelevanceState, margins: &Margins, from: f32, to: f32) -> Relevant {
    let step = 0.05;
    let mut now = from;
    let mut relevant = state.update(margins, now);
    while now + step <= to + 1e-4 {
        now += step;
        relevant = state.update(margins, now);
    }
    relevant
}

#[test]
fn an_effect_fades_out_before_it_stands_down() {
    let mut state = RelevanceState::default();
    assert_eq!(state.relevant(), Relevant::ALL);
    assert_eq!(state.fades(), Fades::FULL);
    let failing = with(Effect::WaterSim, 0.5);
    assert!(run(&mut state, &failing, 10.0, 10.25).contains(Effect::WaterSim));
    let half = state.fades().get(Effect::WaterSim);
    assert!((half - 0.5).abs() < 1e-3, "{half}");
    assert_eq!(state.fades().get(Effect::Reflections), 1.0);
    let now = run(&mut state, &failing, 10.3, 10.55);
    assert_eq!(now, Relevant::ALL.with(Effect::WaterSim, false));
    assert_eq!(state.fades().get(Effect::WaterSim), 0.0);
}

#[test]
fn a_recovery_inside_the_fade_brings_the_effect_back_up() {
    let mut state = RelevanceState::default();
    run(&mut state, &with(Effect::Reflections, 0.5), 0.0, 0.3);
    let dimmed = state.fades().get(Effect::Reflections);
    assert!(dimmed > 0.0 && dimmed < 1.0, "{dimmed}");
    assert!(
        run(&mut state, &with(Effect::Reflections, 1.0), 0.35, 0.6).contains(Effect::Reflections)
    );
    assert_eq!(state.fades().get(Effect::Reflections), 1.0);
    // Failing again, it takes the whole fade to stand down.
    assert!(
        run(&mut state, &with(Effect::Reflections, 0.5), 0.65, 1.05).contains(Effect::Reflections)
    );
    assert!(
        !run(&mut state, &with(Effect::Reflections, 0.5), 1.15, 1.2).contains(Effect::Reflections)
    );
}

// A rule that holds only now and then, as for water the depth samples catch
// in some frames and miss in others, keeps its effect on: one that just came
// back still waits out the whole half second of failing.
#[test]
fn a_rule_that_holds_now_and_then_keeps_its_effect_on() {
    let mut state = RelevanceState::default();
    run(&mut state, &with(Effect::WaterSim, 0.0), 0.0, 0.6);
    assert!(!state.relevant().contains(Effect::WaterSim));
    let mut now = 0.65;
    for _ in 0..20 {
        assert!(
            state
                .update(&with(Effect::WaterSim, 2.0), now)
                .contains(Effect::WaterSim)
        );
        assert!(
            run(
                &mut state,
                &with(Effect::WaterSim, 0.0),
                now + 0.05,
                now + 0.3
            )
            .contains(Effect::WaterSim)
        );
        now += 0.35;
    }
    assert!(
        !run(&mut state, &with(Effect::WaterSim, 0.0), now, now + 0.5).contains(Effect::WaterSim)
    );
}

#[test]
fn an_effect_returns_only_well_past_its_threshold_and_fades_in() {
    let mut state = RelevanceState::default();
    assert!(
        !run(&mut state, &with(Effect::Disocclusion, 0.0), 0.0, 1.0).contains(Effect::Disocclusion)
    );
    // Just over the threshold it stays down, so a view resting there does
    // not toggle it.
    assert!(
        !run(&mut state, &with(Effect::Disocclusion, 1.1), 1.05, 1.5)
            .contains(Effect::Disocclusion)
    );
    assert_eq!(state.fades().get(Effect::Disocclusion), 0.0);
    let back = with(Effect::Disocclusion, 1.3);
    assert!(state.update(&back, 1.55).contains(Effect::Disocclusion));
    // It comes back showing none of itself, then fades in.
    assert_eq!(state.fades().get(Effect::Disocclusion), 0.0);
    run(&mut state, &back, 1.6, 1.7);
    let rising = state.fades().get(Effect::Disocclusion);
    assert!(rising > 0.0 && rising < 1.0, "{rising}");
    run(&mut state, &back, 1.75, 1.85);
    assert_eq!(state.fades().get(Effect::Disocclusion), 1.0);
}

#[test]
fn a_reset_brings_every_effect_back_in_full_and_a_clock_run_back_advances_nothing() {
    let mut state = RelevanceState::default();
    run(&mut state, &with(Effect::PropProbe, 0.0), 5.0, 6.0);
    assert!(!state.relevant().contains(Effect::PropProbe));
    state.reset();
    assert_eq!(state.relevant(), Relevant::ALL);
    assert_eq!(state.fades(), Fades::FULL);
    state.update(&with(Effect::PropProbe, 0.0), 5.0);
    // The world restarted: the fade starts from the new clock.
    assert!(
        state
            .update(&with(Effect::PropProbe, 0.0), 0.1)
            .contains(Effect::PropProbe)
    );
    assert_eq!(state.fades().get(Effect::PropProbe), 1.0);
    assert!(
        !run(&mut state, &with(Effect::PropProbe, 0.0), 0.15, 0.65).contains(Effect::PropProbe)
    );
}

#[test]
fn the_set_holds_each_effect_apart() {
    let none = Relevant::NONE.with(Effect::WaterSim, true);
    for effect in Effect::ALL {
        assert_eq!(
            none.contains(effect),
            effect == Effect::WaterSim,
            "{}",
            effect.name()
        );
        assert!(Relevant::ALL.contains(effect));
    }
    assert_eq!(Relevant::default(), Relevant::ALL);
}

// Fog the cascades shade, or a raymarched volume in view, keeps them on with
// nothing solid near.
#[test]
fn a_shaded_medium_near_the_camera_keeps_the_cascades() {
    let far = facts(Some(5000.0));
    assert!(margins(&far).get(Effect::ShadowCascades) < 1.0);
    let mut reach = reach();
    reach.scene.shadowed_media = Some(0.0);
    let fogged = Margins::of(&Relevance::default(), &far, &reach);
    assert_eq!(fogged.get(Effect::ShadowCascades), f32::INFINITY);
    reach.scene.shadowed_media = Some(150.0);
    let volume = Margins::of(&Relevance::default(), &far, &reach);
    assert!((volume.get(Effect::ShadowCascades) - 200.0 / 150.0).abs() < 1e-4);
}

// Wind sways planted grass even under a still camera, so the mark stays on
// while grass lies near enough for a blade's sway to cross a pixel.
#[test]
fn swaying_grass_near_the_camera_keeps_the_mark() {
    let mut reach = reach();
    reach.scene.grass = Some(GrassReach {
        blade_height: 0.5,
        sway: 0.2,
    });
    let mut meadow = facts(None);
    meadow.subjects[0] = Some(subject(SubjectKind::Planet, 1.0, Some(20.0)));
    // 0.2 m at 20 m is 10 pixels.
    let near = Margins::of(&Relevance::default(), &meadow, &reach);
    assert!((near.get(Effect::Disocclusion) - 10.0).abs() < 1e-3);
    meadow.subjects[0] = Some(subject(SubjectKind::Planet, 1.0, Some(150_000.0)));
    let orbit = Margins::of(&Relevance::default(), &meadow, &reach);
    assert!(orbit.get(Effect::Disocclusion) < 1.0);
    // No grass, no sway.
    reach.scene.grass = None;
    meadow.subjects[0] = Some(subject(SubjectKind::Planet, 1.0, Some(20.0)));
    let bare = Margins::of(&Relevance::default(), &meadow, &reach);
    assert_eq!(bare.get(Effect::Disocclusion), 0.0);
}

fn effects() -> EffectInputs {
    EffectInputs {
        ssao: Some(SsaoSettings {
            radius: 0.5,
            intensity: 1.0,
        }),
        ssr: Some(SsrSettings {
            intensity: 1.0,
            max_distance: 60.0,
        }),
        ssgi: Some((
            SsgiSettings {
                intensity: 1.0,
                max_distance: 4.0,
                rays: 1,
                gi_scale: 2,
            },
            SsgiFrame {
                frame: 0,
                history_valid: true,
                levels: 1,
            },
        )),
        rt: Some(RtReflectionSettings {
            intensity: 1.0,
            max_distance: 300.0,
            divisor: 1,
            sun_shadows: true,
            water_trace: true,
        }),
    }
}

#[test]
fn each_effect_reaches_as_far_as_its_settings_trace() {
    let scene = SceneReach {
        longest_wave: Some(168.0),
        ..SceneReach::default()
    };
    let reach = EffectReach::of(&effects(), false, scene);
    assert_eq!(reach.ambient_occlusion, Some(0.5));
    assert_eq!(reach.indirect_lighting, Some(4.0));
    // The longer of the two traces.
    assert_eq!(reach.reflections, Some(300.0));
    assert_eq!(reach.scene, scene);
    // The bounce's resolve folds the occlusion in, so it reads as far.
    let folded = EffectReach::of(&effects(), true, scene);
    assert_eq!(folded.ambient_occlusion, Some(4.0));
    let alone = EffectInputs {
        ssao: None,
        rt: None,
        ..effects()
    };
    let bounce_only = EffectReach::of(&alone, true, scene);
    assert_eq!(bounce_only.ambient_occlusion, Some(4.0));
    assert_eq!(bounce_only.reflections, Some(60.0));
    assert_eq!(
        EffectReach::of(&EffectInputs::default(), true, scene).ambient_occlusion,
        None
    );
}

// The prop's part of the screen in a 1920x1080 view: a square a tenth of the
// height across, left of center.
const PROP_RECT: ScreenRect = ScreenRect {
    min: [0.3, 0.45],
    max: [0.35, 0.55],
};

// A near mirror ball before a planet seen from orbit asks for the
// reflections alone, over its own part of the screen.
#[test]
fn the_reflections_on_a_near_prop_before_a_far_planet_cover_only_the_prop() {
    let mut view = facts(None);
    view.subjects[0] = Some(subject(SubjectKind::Planet, 150_000.0, Some(150_500.0)));
    view.subjects[2] = Some(Subject {
        rect: PROP_RECT,
        ..subject(SubjectKind::Prop, 3.5, Some(3.9))
    });
    let m = margins(&view);
    assert!(m.reflections.whole < 1.0, "{m:?}");
    assert!(m.reflections.prop > 1.0, "{m:?}");
    assert_eq!(m.reflections.rect, PROP_RECT);
    assert_eq!(m.get(Effect::Reflections), m.reflections.prop);
    // Near ground asks for them everywhere.
    view.subjects[0] = Some(subject(SubjectKind::Planet, 2.0, Some(2.0)));
    let m = margins(&view);
    assert!(m.reflections.whole > 1.0, "{m:?}");
    assert_eq!(
        m.get(Effect::Reflections),
        m.reflections.whole.max(m.reflections.prop)
    );
    // Water reflects its own way and asks for neither.
    let mut sea = facts(None);
    sea.subjects[1] = Some(subject(SubjectKind::Water, 2.0, Some(2.0)));
    let m = margins(&sea);
    assert_eq!((m.reflections.whole, m.reflections.prop), (0.0, 0.0));
    assert!(m.reflections.rect.is_empty());
}

fn reflections(whole: f32, prop: f32, rect: ScreenRect) -> Margins {
    let mut margins = Margins::UNBOUNDED;
    margins.reflections = ReflectionMargins { whole, prop, rect };
    margins
}

// Where only the prop keeps asking, the reflections elsewhere fade out while
// the prop's stay whole, and its part of the screen follows it.
#[test]
fn the_reflections_beyond_the_prop_fade_while_the_prop_keeps_them() {
    let mut state = RelevanceState::default();
    assert_eq!(state.reflection_scope().whole, 1.0);
    let prop_only = reflections(0.0, 4.0, PROP_RECT);
    run(&mut state, &prop_only, 0.0, 0.25);
    let half = state.reflection_scope();
    assert!((half.whole - 0.5).abs() < 1e-3, "{half:?}");
    assert_eq!((half.prop, half.rect), (1.0, PROP_RECT));
    run(&mut state, &prop_only, 0.3, 1.0);
    let scope = state.reflection_scope();
    assert_eq!((scope.whole, scope.prop), (0.0, 1.0));
    let moved = ScreenRect {
        min: [0.6, 0.45],
        max: [0.65, 0.55],
    };
    state.update(&reflections(0.0, 4.0, moved), 1.05);
    assert_eq!(state.reflection_scope().rect, moved);
    // Out of view it covers no part of the screen, and its share fades.
    let gone = reflections(0.0, 0.0, ScreenRect::EMPTY);
    run(&mut state, &gone, 1.1, 1.35);
    let leaving = state.reflection_scope();
    assert!(leaving.rect.is_empty(), "{leaving:?}");
    assert!((leaving.prop - 0.4).abs() < 1e-3, "{leaving:?}");
    assert_eq!(leaving.gpu([1920, 1080]).prop, 0.0);
    assert!(state.relevant().contains(Effect::Reflections));
    run(&mut state, &gone, 1.4, 2.0);
    assert_eq!(state.reflection_scope().prop, 0.0);
    assert!(!state.relevant().contains(Effect::Reflections));
}

// A pixel's share is whole everywhere, full within the prop's part of the
// screen, and falls to the share elsewhere over the ramp past it.
#[test]
fn a_pixel_s_share_ramps_from_the_prop_to_the_rest_of_the_screen() {
    let size = [1920, 1080];
    let scope = ReflectionScope {
        whole: 0.25,
        prop: 1.0,
        rect: PROP_RECT,
    };
    assert_eq!(scope.share([0.32, 0.5], size), 1.0);
    let px = |x: f32| 0.35 + x / 1920.0;
    let halfway = scope.share([px(scope::PROP_RAMP_PIXELS * 0.5), 0.5], size);
    assert!((halfway - 0.5).abs() < 1e-3, "{halfway}");
    assert_eq!(
        scope.share([px(scope::PROP_RAMP_PIXELS + 1.0), 0.5], size),
        0.25
    );
    assert_eq!(scope.share([0.9, 0.9], size), 0.25);
    // Without a prop share the block carries no rectangle at all.
    let whole = ReflectionScope { prop: 0.0, ..scope };
    assert_eq!(
        whole.gpu(size),
        ReflectionScopeGpu {
            whole: 0.25,
            ..ReflectionScopeGpu::NONE
        }
    );
    assert_eq!(ReflectionScope::FULL.share([0.9, 0.1], size), 1.0);
    assert_eq!(ReflectionScope::NONE.share([0.3, 0.5], size), 0.0);
}

// A night the exposure shows dark leaves the bounce under a display step;
// one it lifts to full keeps it, however little light reaches the ground.
#[test]
fn the_bounce_stands_down_in_a_view_the_exposure_shows_dark() {
    let mut view = facts(Some(2.0));
    let bounce = |view: &ViewFacts| margins(view).get(Effect::IndirectLighting);
    let lit = bounce(&view);
    assert!(lit > 1.0);
    // A view shown in full stands fifty times over the default.
    view.shown_light = Some(1.0);
    assert!((bounce(&view) - 50.0).abs() < 1e-3);
    view.shown_light = Some(1.0 / 128.0);
    let night = bounce(&view);
    assert!((night - (1.0 / 128.0) / 0.02).abs() < 1e-4, "{night}");
    // The occlusion is not the bounce's light.
    assert!(margins(&view).get(Effect::AmbientOcclusion) > 1.0);
    let rules = Relevance {
        min_bounce_light: 0.0,
        ..Relevance::default()
    };
    assert_eq!(
        Margins::of(&rules, &view, &reach()).get(Effect::IndirectLighting),
        lit
    );
}

// The reflections run while either part of the screen holds them: a prop
// resting between its threshold and the margin it would need to come back
// keeps them, with a scope that still shows on it.
#[test]
fn the_reflections_run_exactly_while_a_part_of_the_screen_holds_them() {
    let mut state = RelevanceState::default();
    run(&mut state, &reflections(0.0, 1.1, PROP_RECT), 0.0, 2.0);
    assert!(state.relevant().contains(Effect::Reflections));
    let scope = state.reflection_scope();
    assert_eq!((scope.whole, scope.prop), (0.0, 1.0));
    // The effect's own margin is not what holds them: the parts are.
    let unbounded_overall = reflections(0.0, 1.1, PROP_RECT);
    assert_eq!(unbounded_overall.get(Effect::Reflections), f32::INFINITY);
    run(&mut state, &reflections(0.0, 0.0, PROP_RECT), 2.05, 3.0);
    assert!(!state.relevant().contains(Effect::Reflections));
    assert_eq!(state.fades().get(Effect::Reflections), 0.0);
}
