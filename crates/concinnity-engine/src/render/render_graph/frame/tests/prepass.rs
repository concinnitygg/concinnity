// The G-buffer pre-pass and the ambient occlusion and shadow targets it
// feeds, and the barriers the graph derives for them.

use super::*;

#[test]
fn ssao_orders_before_main_via_ao_output() {
    let mut i = all_off();
    i.ssao_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    assert_eq!(
        order,
        vec![PassId::SsaoBlur, PassId::Main, PassId::Composite]
    );
}

#[test]
fn ao_output_barriers_are_graph_driven() {
    // Executors emit `ao_output`'s transitions from these barriers, so pin the
    // exact pair.
    use crate::render::render_graph::ResourceState;
    let mut i = all_off();
    i.ssao_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let pass = |id: PassId| g.passes.iter().find(|p| p.id == id).expect("present");

    let ssao = g.pass_barriers_for(pass(PassId::SsaoBlur), &["ao_output"]);
    assert_eq!(ssao.len(), 1, "SsaoBlur has exactly one ao_output barrier");
    assert_eq!(ssao[0].1.source_state(), ResourceState::Undefined);
    assert_eq!(ssao[0].1.to_state(), ResourceState::Write);

    let main = g.pass_barriers_for(pass(PassId::Main), &["ao_output"]);
    assert_eq!(main.len(), 1, "Main has exactly one ao_output barrier");
    assert_eq!(main[0].1.source_state(), ResourceState::Write);
    assert_eq!(main[0].1.to_state(), ResourceState::Read);
}

#[test]
fn shadow_map_barriers_are_graph_driven() {
    // Executors emit `shadow_map`'s transitions from these barriers: the
    // producer resolved against its resting state and Main's consumer. With
    // SSAO on, Main carries two barriers.
    use crate::render::render_graph::ResourceState;
    let mut i = all_off();
    i.shadow_enabled = true;
    i.ssao_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let pass = |id: PassId| g.passes.iter().find(|p| p.id == id).expect("present");

    let shadow = g.pass_barriers_for(pass(PassId::Shadow), &["shadow_map"]);
    assert_eq!(shadow.len(), 1, "Shadow has exactly one shadow_map barrier");
    assert_eq!(shadow[0].1.source_state(), ResourceState::Undefined);
    assert_eq!(shadow[0].1.to_state(), ResourceState::Write);

    let main = g.pass_barriers_for(pass(PassId::Main), &["shadow_map"]);
    assert_eq!(main.len(), 1, "Main has exactly one shadow_map barrier");
    assert_eq!(main[0].1.source_state(), ResourceState::Write);
    assert_eq!(main[0].1.to_state(), ResourceState::Read);

    // Main carries both migrated resources' barriers in one pass.
    let both = g.pass_barriers_for(pass(PassId::Main), &["shadow_map", "ao_output"]);
    assert_eq!(
        both.len(),
        2,
        "Main carries shadow_map + ao_output barriers"
    );
}

#[test]
fn fog_froxel_volume_barriers_are_graph_driven() {
    // Executors emit `fog_froxel_volume`'s transitions from these barriers: the
    // compute write's open and the fragment read's close.
    use crate::render::render_graph::ResourceState;
    let mut i = all_off();
    i.fog_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let pass = |id: PassId| g.passes.iter().find(|p| p.id == id).expect("present");

    let froxel = g.pass_barriers_for(pass(PassId::FogFroxel), &["fog_froxel_volume"]);
    assert_eq!(
        froxel.len(),
        1,
        "FogFroxel has exactly one fog_froxel_volume barrier"
    );
    assert_eq!(froxel[0].1.source_state(), ResourceState::Undefined);
    assert_eq!(froxel[0].1.to_state(), ResourceState::Write);

    let fog = g.pass_barriers_for(pass(PassId::Fog), &["fog_froxel_volume"]);
    assert_eq!(
        fog.len(),
        1,
        "Fog has exactly one fog_froxel_volume barrier"
    );
    assert_eq!(fog[0].1.source_state(), ResourceState::Write);
    assert_eq!(fog[0].1.to_state(), ResourceState::Read);
}

// SSAO, with or without SSR, triggers the one G-buffer pre-pass node.
#[test]
fn ssao_with_or_without_ssr_pins_the_gbuffer_prepass() {
    for ssr in [false, true] {
        let mut i = all_off();
        i.gbuffer_prepass_enabled = true;
        i.ssr_prepass_enabled = ssr;
        i.ssao_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(
            order,
            vec![
                PassId::GBufferPrepass,
                PassId::SsaoBlur,
                PassId::Main,
                PassId::Composite,
            ],
            "ssr: {ssr}"
        );
    }
}

#[test]
fn one_prepass_node_serves_ssr_ssao_and_velocity() {
    // SSR, SSAO and TAA all read the one GBufferPrepass output: SSAO its
    // normal+depth, TAA its motion.
    let mut i = all_off();
    i.gbuffer_prepass_enabled = true;
    i.ssr_prepass_enabled = true;
    i.ssao_enabled = true;
    i.velocity_enabled = true;
    i.taa_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    assert!(order.contains(&PassId::GBufferPrepass));
    let gb = order
        .iter()
        .position(|p| *p == PassId::GBufferPrepass)
        .unwrap();
    let ssao = order.iter().position(|p| *p == PassId::SsaoBlur).unwrap();
    let main = order.iter().position(|p| *p == PassId::Main).unwrap();
    let taa = order.iter().position(|p| *p == PassId::TaaResolve).unwrap();
    assert!(
        gb < ssao && ssao < main,
        "GBufferPrepass before SsaoBlur+Main"
    );
    assert!(gb < taa, "GBufferPrepass before TaaResolve");
}

#[test]
fn gbuffer_prepass_runs_for_velocity_only() {
    // Velocity alone (TAA, no SSR/SSAO) still triggers the pre-pass.
    let mut i = all_off();
    i.gbuffer_prepass_enabled = true;
    i.velocity_enabled = true;
    i.taa_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");

    assert!(
        g.passes
            .iter()
            .map(|p| p.id)
            .any(|x| x == PassId::GBufferPrepass)
    );
}

#[test]
fn gbuffer_prepass_orders_after_cull_via_draw_args() {
    // The GPU-driven G-buffer pre-pass reuses the main pass's per-frame
    // indirect command buffer, so it must run after Cull. With bindless cull
    // on and a G-buffer consumer active, the draw_args read edge pins
    // Cull -> GBufferPrepass (-> Main).
    let mut i = all_off();
    i.bindless_cull_enabled = true;
    i.gbuffer_prepass_enabled = true;
    i.ssao_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    let cull = order.iter().position(|p| *p == PassId::Cull).unwrap();
    let gb = order
        .iter()
        .position(|p| *p == PassId::GBufferPrepass)
        .unwrap();
    let main = order.iter().position(|p| *p == PassId::Main).unwrap();
    assert!(cull < gb, "Cull before GBufferPrepass");
    assert!(gb < main, "GBufferPrepass before Main");
}

#[test]
fn gbuffer_prepass_omitted_when_no_consumers() {
    // The flag on but no consumer active: no pre-pass node at all.
    let mut i = all_off();
    i.gbuffer_prepass_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    assert_eq!(order, vec![PassId::Main, PassId::Composite]);
}
