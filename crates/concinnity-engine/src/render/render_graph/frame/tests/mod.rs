use super::*;
use alloc::vec;
use alloc::vec::Vec;

// The production defaults, with MSAA on so the resolve-chain tests have a
// multisampled hdr target to exercise.
fn all_off() -> FrameGraphInputs {
    FrameGraphInputs {
        hdr_sample_count: 4,
        ..FrameGraphInputs::all_off()
    }
}

#[test]
fn minimum_graph_is_main_then_composite() {
    let g = build_frame_graph(&all_off()).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    assert_eq!(order, vec![PassId::Main, PassId::Composite]);
    assert!(g.passes[1].presents);
}

#[test]
fn world_hidden_collapses_to_minimum_graph() {
    // Every heavy world pass requested, but the opaque menu backdrop is up:
    // the builder must mask them all off and yield the bare Main -> Composite
    // graph, with Composite still the presenter for the overlay.
    let mut i = all_off();
    i.shadow_enabled = true;
    i.bindless_cull_enabled = true;
    i.ssr_prepass_enabled = true;
    i.ssao_enabled = true;
    i.ssgi_enabled = true;
    i.rt_reflections_enabled = true;
    i.bloom_enabled = true;
    i.taa_enabled = true;
    i.velocity_enabled = true;
    i.two_pass_occlusion_enabled = true;
    i.world_hidden = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    assert_eq!(order, vec![PassId::Main, PassId::Composite]);
    assert!(g.passes[1].presents);
}

#[test]
fn shadow_orders_before_main() {
    let mut i = all_off();
    i.shadow_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    assert_eq!(order, vec![PassId::Shadow, PassId::Main, PassId::Composite]);
}

#[test]
fn cull_orders_before_main_via_draw_args() {
    let mut i = all_off();
    i.bindless_cull_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    assert_eq!(order, vec![PassId::Cull, PassId::Main, PassId::Composite]);
    assert_eq!(g.passes[0].kind, PassKind::Compute);
}

#[test]
fn two_pass_inserts_late_chain_after_main() {
    // With bindless cull + two-pass on, the graph gains the late pass
    // prefix Cull → Main → HizEarly → CullLate → MainLate, strictly ordered.
    let mut i = all_off();
    i.bindless_cull_enabled = true;
    i.two_pass_occlusion_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    assert_eq!(
        order,
        vec![
            PassId::Cull,
            PassId::Main,
            PassId::HizEarly,
            PassId::CullLate,
            PassId::MainLate,
            PassId::Composite,
        ]
    );
    assert_eq!(g.passes[2].kind, PassKind::Compute); // HizEarly
    assert_eq!(g.passes[3].kind, PassKind::Compute); // CullLate
    assert_eq!(g.passes[4].kind, PassKind::Render); // MainLate
}

// Each cull packs its draws through one group-count scratch per frame
// slot, which the late pass reuses: the early pass packs within `Cull`, and Main and
// its pre-pass draw the early pass's lists, all before `CullLate` writes it again.
// Every feature that adds passes between them is on, so none may move a
// reader of the early pass's lists past `CullLate`.
#[test]
fn phase_ones_lists_are_drawn_before_phase_two_culls() {
    let mut i = all_off();
    for (_, enable) in GATED_FLAGS {
        enable(&mut i);
    }
    i.world_hidden = false;
    let g = build_frame_graph(&i).expect("compiles");
    let at = |id| {
        g.passes
            .iter()
            .position(|p| p.id == id)
            .unwrap_or_else(|| panic!("{id:?} in the graph"))
    };
    let cull_late = at(PassId::CullLate);
    for reader in [PassId::Cull, PassId::Main, PassId::GBufferPrepass] {
        assert!(at(reader) < cull_late, "{reader:?} runs before CullLate");
    }
    assert!(at(PassId::Cull) < at(PassId::GBufferPrepass));
}

// Index of `label` in the compiled graph's resource arena.
fn resource_of(g: &CompiledGraph, label: &str) -> usize {
    g.resources
        .iter()
        .position(|r| r.label == label)
        .unwrap_or_else(|| panic!("{label} missing from the graph"))
}

mod hdr_chain;
mod lighting;
mod occlusion;
mod prepass;
mod reactive;
mod shadows;
