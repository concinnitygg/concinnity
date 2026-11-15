// What the view's relevance turns off, and that every set of it still builds
// the graph its decisions describe.

use alloc::format;
use alloc::string::ToString;
use alloc::vec::Vec;

use super::super::samples::{Rng, everything, nothing};
use super::super::*;
use super::{assert_runs_match_the_graph, graph_inputs};

// Every subset of the effects, by bit.
fn every_relevant_set() -> impl Iterator<Item = Relevant> {
    (0u32..1 << Effect::COUNT).map(|bits| {
        Effect::ALL
            .iter()
            .enumerate()
            .fold(Relevant::ALL, |set, (i, &effect)| {
                set.with(effect, bits >> i & 1 == 1)
            })
    })
}

// Every pass on that anything turns on, with no live trace so the prop probe
// and the screen-space resolve have their slots, on a backend that reads the
// reflection composite only when a resolve writes it.
fn every_relevant_pass() -> PassInputs {
    let mut inputs = everything();
    inputs.facts.bvh_live = false;
    inputs.gates.rt_pass = false;
    inputs.gates.traits.composite_scene_fixed = false;
    inputs
}

fn off_for(inputs: &PassInputs, effect: Effect) -> Vec<PassId> {
    let decisions = PassDecisions::select(&PassInputs {
        relevant: Relevant::ALL.with(effect, false),
        ..*inputs
    });
    PassId::ALL
        .into_iter()
        .filter(|&pass| decisions.not_relevant() & (1 << pass as u32) != 0)
        .collect()
}

#[test]
fn each_effect_turns_off_the_passes_that_draw_it() {
    let inputs = every_relevant_pass();
    let expect = [
        (
            Effect::AmbientOcclusion,
            &[PassId::SsaoBlur, PassId::Ssgi, PassId::SsgiResolve][..],
        ),
        (Effect::IndirectLighting, &[PassId::Ssgi]),
        (Effect::Reflections, &[PassId::SsrResolve]),
        (Effect::ShadowCascades, &[PassId::Shadow]),
        (Effect::PropProbe, &[PassId::PropProbe]),
        (Effect::WaterSim, &[PassId::WaterSim]),
        (Effect::Disocclusion, &[PassId::Disocclusion]),
    ];
    for (effect, passes) in expect {
        assert_eq!(off_for(&inputs, effect), passes, "{}", effect.name());
    }
    // A live trace takes the reflection slot instead.
    let mut traced = everything();
    traced.gates.traits.composite_scene_fixed = false;
    assert_eq!(
        off_for(&traced, Effect::Reflections),
        [PassId::RtReflections]
    );
    assert_eq!(
        PassDecisions::select(&every_relevant_pass()).not_relevant(),
        0
    );
}

#[test]
fn a_pass_reading_one_turned_off_follows_it() {
    let inputs = PassInputs {
        relevant: Relevant::ALL.with(Effect::ShadowCascades, false),
        ..every_relevant_pass()
    };
    let decisions = PassDecisions::select(&inputs);
    let grass = decisions.get(PassId::GrassShadow);
    assert_eq!(grass.reason, Reason::FedBy(PassId::Shadow));
    let shadow = decisions.get(PassId::Shadow);
    assert!(!shadow.asked && !shadow.runs);
    assert_eq!(shadow.reason.layer(), Some(Layer::Relevance));
    assert_eq!(
        shadow.reason.to_string(),
        "shadow_cascades adds nothing to this view"
    );
    assert!(!decisions.get(PassId::GrassShadow).runs);
    assert!(!decisions.graph_inputs(&inputs).shadow_enabled);
}

#[test]
fn the_disocclusion_mark_leaves_the_graph_with_its_relevance() {
    let inputs = every_relevant_pass();
    assert!(graph_inputs(&inputs).disocclusion_live());
    let still = PassInputs {
        relevant: Relevant::ALL.with(Effect::Disocclusion, false),
        ..inputs
    };
    assert!(!graph_inputs(&still).disocclusion_live());
}

// Another layer that already turned a pass off names itself, not relevance.
#[test]
fn relevance_is_named_only_where_everything_else_holds() {
    let mut inputs = every_relevant_pass();
    inputs.relevant = Relevant::NONE;
    inputs.gates.ssao = false;
    inputs.show = ShowFlags::all().toggled(ShowFlags::SSGI);
    let decisions = PassDecisions::select(&inputs);
    assert_eq!(
        decisions.get(PassId::SsaoBlur).reason.layer(),
        Some(Layer::Setting)
    );
    assert_eq!(
        decisions.get(PassId::Ssgi).reason,
        Reason::Needs(Need::Show(ShowFlags::SSGI))
    );
    assert_eq!(
        decisions.get(PassId::WaterSim).reason.layer(),
        Some(Layer::Relevance)
    );
}

#[test]
fn every_relevant_set_runs_what_the_graph_holds() {
    for base in [nothing(), everything(), every_relevant_pass()] {
        for relevant in every_relevant_set() {
            assert_runs_match_the_graph(&PassInputs { relevant, ..base }, &format!("{relevant:?}"));
        }
    }
    let mut rng = Rng::new(0x51_7cc1_b727_220a);
    let sets: Vec<Relevant> = every_relevant_set().collect();
    for n in 0..2_000 {
        let pick = usize::try_from(rng.next() % sets.len() as u64).unwrap_or(0);
        let inputs = PassInputs {
            relevant: sets[pick],
            ..rng.sample()
        };
        assert_runs_match_the_graph(&inputs, &format!("sample {n}"));
    }
}

// A post chain that reads the reflection composite every frame keeps its
// resolve whatever the view.
#[test]
fn a_fixed_composite_keeps_its_reflection_resolve() {
    let mut inputs = everything();
    inputs.relevant = Relevant::ALL.with(Effect::Reflections, false);
    inputs.gates.traits.composite_scene_fixed = true;
    assert!(
        PassDecisions::select(&inputs)
            .get(PassId::RtReflections)
            .runs
    );
    inputs.gates.traits.composite_scene_fixed = false;
    let cut = PassDecisions::select(&inputs).get(PassId::RtReflections);
    assert_eq!(cut.reason.layer(), Some(Layer::Relevance));
}

// The bounce's trace feeds only its resolve, so it never runs without it.
#[test]
fn the_bounce_traces_only_where_its_resolve_runs() {
    for base in [everything(), every_relevant_pass()] {
        for relevant in every_relevant_set() {
            let decisions = PassDecisions::select(&PassInputs { relevant, ..base });
            if decisions.get(PassId::Ssgi).runs {
                assert!(decisions.get(PassId::SsgiResolve).runs, "{relevant:?}");
            }
        }
    }
}
