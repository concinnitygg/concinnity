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
    // The DirectX + Vulkan executors emit `ao_output`'s transitions from
    // these barriers (resolving them to RENDER_TARGET / COLOR_ATTACHMENT on
    // SsaoBlur and back to the sampled state on Main). Pin the exact pair
    // so the executor's stripped inline / render-pass-baked transitions
    // stay matched to what the graph derives.
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
    // The executors emit `shadow_map`'s transitions from these barriers. The
    // graph derives the producer (Undefined -> Write) + the Main consumer
    // (Write -> Read); each backend resolves the producer against the
    // resource's resting state. DirectX rests it sampled, so the producer is
    // the real PIXEL_SHADER_RESOURCE -> DEPTH_WRITE cross-frame reset (folded
    // off the old inline restore); Main's consumer replaces the encoder's
    // stripped sampled transition. With SSAO also on, Main carries both
    // shadow_map and ao_output barriers, exercising multi-resource emission
    // in one pass.
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
    // The executors emit `fog_froxel_volume`'s transitions from these
    // barriers. FogFroxel's producer (Undefined -> Write) is the compute
    // write, a real sampled -> storage open on both backends now: DirectX
    // resolves it to PIXEL_SHADER_RESOURCE -> UNORDERED_ACCESS, Vulkan to
    // SHADER_READ_ONLY -> GENERAL (both rest the volume sampled, with no
    // inline reset). Fog's consumer (Write -> Read) is the storage-write ->
    // sampled close the fragment reads through.
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
