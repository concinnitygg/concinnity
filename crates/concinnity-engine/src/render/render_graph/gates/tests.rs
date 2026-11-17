use super::*;
use crate::components::{AaMode, IndirectTechnique};
use crate::render::render_graph::{FrameGraphInputs, PassDecisions, PassInputs};

const ALL_BUILT: BuildStatus = BuildStatus {
    taa: true,
    ssao: true,
    ssr_resolve: true,
    rt_pass: true,
    ssgi: true,
    auto_exposure: true,
    gbuffer: true,
};

// Settings asking for every runtime-switched feature.
fn every_feature() -> ResolvedGraphics {
    let mut g = ResolvedGraphics::default();
    g.anti_aliasing.mode = AaMode::Taa;
    g.ambient_occlusion.enabled = true;
    g.reflections.screen_space = true;
    g.reflections.ray_traced = true;
    g.indirect_lighting.technique = IndirectTechnique::Ssgi;
    g.exposure.auto = true;
    g
}

// Settings asking for none of them.
fn no_feature() -> ResolvedGraphics {
    let mut g = ResolvedGraphics::default();
    g.anti_aliasing.mode = AaMode::Fxaa;
    g.ambient_occlusion.enabled = false;
    g.reflections.screen_space = false;
    g.reflections.ray_traced = false;
    g.indirect_lighting.technique = IndirectTechnique::Ibl;
    g.exposure.auto = false;
    g
}

// The graph inputs `gates` and `facts` select in the lit view.
fn inputs(gates: &FeatureGates, facts: &FrameFacts) -> FrameGraphInputs {
    let inputs = PassInputs {
        gates: *gates,
        facts: *facts,
        ..PassInputs::default()
    };
    PassDecisions::select(&inputs).graph_inputs(&inputs)
}

fn gates(graphics: &ResolvedGraphics) -> FeatureGates {
    FeatureGates::new(graphics, &ALL_BUILT, BackendTraits::default())
}

#[test]
fn a_built_feature_runs_only_when_the_settings_ask_for_it() {
    let on = gates(&every_feature());
    assert!(on.taa && on.ssao && on.ssr && on.rt_pass && on.ssgi && on.auto_exposure);
    let off = gates(&no_feature());
    assert!(!off.taa && !off.ssao && !off.ssr && !off.rt_pass && !off.ssgi);
    assert!(!off.auto_exposure);
    assert!(off.gbuffer, "the pre-pass targets follow the build");
}

// Each feature the settings ask for but the apply did not build stays off,
// and only that one.
#[test]
fn a_feature_the_apply_did_not_build_stays_off() {
    type Clear = fn(&mut BuildStatus);
    type Read = fn(&FeatureGates) -> bool;
    let cases: [(&str, Clear, Read); 7] = [
        ("taa", |b| b.taa = false, |g| g.taa),
        ("ssao", |b| b.ssao = false, |g| g.ssao),
        ("ssr", |b| b.ssr_resolve = false, |g| g.ssr),
        ("rt", |b| b.rt_pass = false, |g| g.rt_pass),
        ("ssgi", |b| b.ssgi = false, |g| g.ssgi),
        ("ae", |b| b.auto_exposure = false, |g| g.auto_exposure),
        ("gbuffer", |b| b.gbuffer = false, |g| g.gbuffer),
    ];
    let full = gates(&every_feature());
    for (name, clear, read) in cases {
        let mut built = ALL_BUILT;
        clear(&mut built);
        let g = FeatureGates::new(&every_feature(), &built, BackendTraits::default());
        assert!(!read(&g), "{name} without its build");
        let mut expected = full;
        match name {
            "taa" => expected.taa = false,
            "ssao" => expected.ssao = false,
            "ssr" => expected.ssr = false,
            "rt" => expected.rt_pass = false,
            "ssgi" => {
                expected.ssgi = false;
                expected.ssgi_resolve = false;
            }
            "ae" => expected.auto_exposure = false,
            _ => expected.gbuffer = false,
        }
        assert_eq!(g, expected, "{name} moves no other gate");
    }
}

// A zero SSGI intensity contributes nothing, so the trace stays off even
// while its resources are built.
#[test]
fn an_inert_ssgi_intensity_turns_its_gate_off() {
    let mut g = every_feature();
    g.indirect_lighting.ssgi_intensity = 0.0;
    assert!(!gates(&g).ssgi);
    // Still built, so the main pass keeps sampling its resolve.
    assert!(gates(&g).ssgi_resolve);
}

// Each runtime gate reaches the graph through the settings that drive it.
#[test]
fn resolved_settings_reach_the_graph() {
    let f = FrameFacts {
        bvh_live: true,
        ..FrameFacts::default()
    };
    let on = inputs(&gates(&every_feature()), &f);
    assert!(on.taa_enabled && on.ssao_enabled && on.rt_reflections_enabled);
    assert!(on.ssgi_enabled && on.auto_exposure_enabled && on.gbuffer_prepass_enabled);
    let off = inputs(&gates(&no_feature()), &f);
    assert!(!off.taa_enabled && !off.ssao_enabled && !off.rt_reflections_enabled);
    assert!(!off.ssr_enabled && !off.ssgi_enabled && !off.auto_exposure_enabled);
    assert!(!off.velocity_enabled && !off.ssr_prepass_enabled);
}
