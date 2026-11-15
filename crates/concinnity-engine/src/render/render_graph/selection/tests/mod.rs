use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use super::samples::{Rng, TOGGLES, everything, nothing, show_sets};
use super::*;
use crate::render::render_graph::build_frame_graph;
use crate::render::render_graph::gates::BackendTraits;

fn graph_inputs(inputs: &PassInputs) -> FrameGraphInputs {
    PassDecisions::select(inputs).graph_inputs(inputs)
}

fn reference_inputs(inputs: &PassInputs) -> FrameGraphInputs {
    let seeded = reference::frame_graph_inputs(&inputs.gates, &inputs.facts);
    reference::apply_view(&seeded, inputs.mode, inputs.show)
}

fn assert_matches_reference(inputs: &PassInputs, what: &dyn Fn() -> String) {
    assert_eq!(
        graph_inputs(inputs),
        reference_inputs(inputs),
        "{}: {inputs:?}",
        what()
    );
}

// Every input of the selection is either flipped by a toggle or passes
// through to the graph inputs unread. A field added to any of these structs
// fails to compile here until it is placed in one or the other.
#[test]
fn every_input_has_a_toggle_or_passes_through() {
    let PassInputs {
        gates,
        facts,
        mode: _,
        show: _,
    } = everything();
    let FeatureGates {
        taa: _,
        ssao: _,
        ssr: _,
        rt_pass: _,
        ssgi: _,
        ssgi_resolve: _,
        auto_exposure: _,
        gbuffer: _,
        traits,
    } = gates;
    let BackendTraits {
        reactive_mask: _,
        composite_scene_fixed: _,
    } = traits;
    let FrameFacts {
        hdr_width: _,
        hdr_height: _,
        hdr_sample_count: _,
        shadow: _,
        shadow_map_size: _,
        far_shadow_map_size: _,
        far_shadow_drawn: _,
        shadowed_spot_count: _,
        spot_shadow_slice_size: _,
        world_hidden: _,
        cull_ready: _,
        cull_records: _,
        two_pass_ready: _,
        hiz: _,
        shadow_receivers: _,
        view_samples: _,
        bloom_chain: _,
        bloom_intensity: _,
        upscaler: _,
        upscaler_reads_reactive: _,
        disocclusion: _,
        bvh_live: _,
        particles: _,
        late_particles: _,
        fog: _,
        decals: _,
        visible_surfaces: _,
        visible_see_through: _,
        rt_transparent_ready: _,
        planar_targets: _,
        water_planar_slot: _,
        lines: _,
        visible_volumes: _,
        atmosphere: _,
        clouds: _,
        water_sim: _,
        grass: _,
        grass_casts: _,
        clustered: _,
        prop_probe: _,
    } = facts;
    // Ten gate fields and 40 facts, less the five sizes that pass through.
    assert_eq!(TOGGLES.len(), 10 + 40 - 5);
    for (name, toggle) in TOGGLES {
        let mut flipped = nothing();
        toggle(&mut flipped);
        assert_ne!(flipped, nothing(), "{name} moves an input");
        toggle(&mut flipped);
        assert_eq!(flipped, nothing(), "{name} flips back");
    }
}

#[test]
fn the_table_has_one_row_per_pass() {
    for pass in PassId::ALL {
        let rows = PASS_TABLE.iter().filter(|row| row.pass == pass).count();
        assert_eq!(rows, 1, "{}", pass.name());
    }
}

#[test]
fn a_pass_is_decided_after_every_pass_feeding_it() {
    for (i, row) in PASS_TABLE.iter().enumerate() {
        for input in row.fed_by {
            let at = PASS_TABLE.iter().position(|r| r.pass == *input);
            assert!(
                at.is_some_and(|at| at < i),
                "{} is fed by {}, decided later",
                row.pass.name(),
                input.name()
            );
        }
    }
}

// Every pair of toggles, from nothing on and from everything on, in every view
// mode and across the show flag sets that reach each mask.
#[test]
fn every_toggle_pair_selects_what_the_hand_written_gates_did() {
    for base in [nothing(), everything()] {
        for (i, (a_name, a)) in TOGGLES.iter().enumerate() {
            for (b_name, b) in &TOGGLES[i..] {
                let mut inputs = base;
                a(&mut inputs);
                b(&mut inputs);
                for mode in ViewMode::ALL {
                    for show in show_sets() {
                        inputs.mode = mode;
                        inputs.show = show;
                        assert_matches_reference(&inputs, &|| format!("{a_name} + {b_name}"));
                    }
                }
            }
        }
    }
}

// Whole-space samples: every toggle at random, any view mode, any show flags.
#[test]
fn sampled_inputs_select_what_the_hand_written_gates_did() {
    let mut rng = Rng::new(0x9e37_79b9_7f4a_7c15);
    for n in 0..200_000 {
        let inputs = rng.sample();
        assert_matches_reference(&inputs, &|| format!("sample {n}"));
    }
}

// A decision that runs is a pass in the compiled graph, and one that does not
// is absent.
fn assert_runs_match_the_graph(inputs: &PassInputs, what: &str) {
    let decisions = PassDecisions::select(inputs);
    let graph = build_frame_graph(&decisions.graph_inputs(inputs))
        .unwrap_or_else(|e| panic!("{what}: the graph compiles: {e}"));
    for (pass, decision) in decisions.iter() {
        let in_graph = graph.passes.iter().any(|p| p.id == pass);
        assert_eq!(
            decision.runs,
            in_graph,
            "{what}: {} ({})",
            pass.name(),
            decision.reason
        );
    }
}

#[test]
fn a_running_decision_is_a_pass_in_the_graph() {
    for base in [nothing(), everything()] {
        for (i, (a_name, a)) in TOGGLES.iter().enumerate() {
            for (b_name, b) in &TOGGLES[i..] {
                let mut inputs = base;
                a(&mut inputs);
                b(&mut inputs);
                assert_runs_match_the_graph(&inputs, &format!("{a_name} + {b_name}"));
            }
        }
        for mode in ViewMode::ALL {
            for show in show_sets() {
                let inputs = PassInputs { mode, show, ..base };
                assert_runs_match_the_graph(&inputs, &format!("{mode:?} / {show:?}"));
            }
        }
    }
    let mut rng = Rng::new(0x2545_f491_4f6c_dd1d);
    for n in 0..2_000 {
        assert_runs_match_the_graph(&rng.sample(), &format!("sample {n}"));
    }
}

#[test]
fn a_hidden_world_runs_only_the_frame_passes() {
    let mut inputs = everything();
    inputs.facts.world_hidden = true;
    let decisions = PassDecisions::select(&inputs);
    for (pass, decision) in decisions.iter() {
        let frame_pass = matches!(pass, PassId::Main | PassId::Composite | PassId::HizFinal);
        assert_eq!(decision.runs, frame_pass, "{}", pass.name());
        if !frame_pass {
            assert_eq!(decision.reason, Reason::WorldHidden, "{}", pass.name());
        }
    }
}

#[test]
fn an_unmet_need_names_the_first_condition_that_failed() {
    let mut inputs = everything();
    inputs.facts.upscaler = true;
    let taa = PassDecisions::select(&inputs).get(PassId::TaaResolve);
    assert!(!taa.asked && !taa.runs);
    assert_eq!(taa.reason, Reason::Needs(Need::Lacks(Fact::Upscaler)));
    assert_eq!(taa.reason.layer(), Some(Layer::Content));
    assert_eq!(taa.reason.to_string(), "not with upscaler");

    inputs.mode = ViewMode::Unlit;
    let ssao = PassDecisions::select(&inputs).get(PassId::SsaoBlur);
    assert_eq!(ssao.reason, Reason::Needs(Need::ShadedView));
    assert_eq!(ssao.reason.layer(), Some(Layer::ViewMode));

    inputs.mode = ViewMode::Lit;
    inputs.show = ShowFlags::all().toggled(ShowFlags::FOG);
    let fog = PassDecisions::select(&inputs).get(PassId::Fog);
    assert_eq!(fog.reason, Reason::Needs(Need::Show(ShowFlags::FOG)));
    assert_eq!(fog.reason.to_string(), "needs the fog show flag");

    inputs.gates.ssao = false;
    let reason = PassDecisions::select(&inputs).get(PassId::SsaoBlur).reason;
    assert_eq!(reason.layer(), Some(Layer::Setting));
    assert_eq!(reason.to_string(), "needs ssao");
}

// The graph inputs still ask for a pass whose input is off; the decision says
// it does not run and names the input.
#[test]
fn a_pass_without_its_input_is_asked_for_but_does_not_run() {
    let mut inputs = everything();
    inputs.facts.visible_surfaces = false;
    inputs.facts.visible_see_through = false;
    let decisions = PassDecisions::select(&inputs);
    let water = decisions.get(PassId::WaterSim);
    assert!(water.asked && !water.runs);
    assert_eq!(water.reason, Reason::FedBy(PassId::Transparent));
    assert_eq!(water.reason.layer(), Some(Layer::Input));
    assert_eq!(water.reason.to_string(), "its input transparent is off");
    assert!(decisions.graph_inputs(&inputs).water_sim_enabled);
}

#[test]
fn everything_on_runs_every_pass_but_the_ones_another_replaces() {
    let decisions = PassDecisions::select(&everything());
    let off: Vec<_> = decisions
        .iter()
        .filter(|(_, d)| !d.runs)
        .map(|(pass, _)| pass.name())
        .collect();
    // The upscaler takes TAA's slot, and a live trace takes SSR's and the prop
    // probe's.
    assert_eq!(off, ["ssr_resolve", "taa_resolve", "prop_probe"]);
    assert_eq!(decisions.get(PassId::Composite).reason, Reason::Runs);
    assert_eq!(Reason::Runs.layer(), None);
}

mod frame;
mod view;
