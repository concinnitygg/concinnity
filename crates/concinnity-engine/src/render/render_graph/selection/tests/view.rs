// The view mode and show flags mask the passes they hide.

use super::super::samples::{everything, nothing};
use super::super::*;
use super::graph_inputs;
use crate::render::render_graph::build_frame_graph;

fn lit(inputs: PassInputs, mode: ViewMode, show: ShowFlags) -> FrameGraphInputs {
    graph_inputs(&PassInputs {
        mode,
        show,
        ..inputs
    })
}

// Settings that turn on every view-masked pass, with no upscaler so TAA runs.
fn every_masked_pass() -> PassInputs {
    let mut inputs = everything();
    inputs.facts.upscaler = false;
    inputs
}

#[test]
fn the_lit_view_with_every_show_flag_masks_nothing() {
    let inputs = every_masked_pass();
    let out = lit(inputs, ViewMode::Lit, ShowFlags::all());
    assert!(out.shadow_enabled && out.fog_enabled && out.bloom_enabled);
    assert!(out.ssgi_enabled && out.rt_reflections_enabled && out.lines_enabled);
    assert!(out.reactive_mask_enabled && out.taa_enabled && out.ssao_enabled);
    assert_eq!(out.shadowed_spot_count, 2);
}

#[test]
fn each_show_flag_masks_its_passes() {
    let inputs = every_masked_pass();
    let m = |f| lit(inputs, ViewMode::Lit, ShowFlags::all().toggled(f));
    let no_shadows = m(ShowFlags::SHADOWS);
    assert!(!no_shadows.shadow_enabled);
    assert_eq!(no_shadows.shadowed_spot_count, 0);
    assert!(!m(ShowFlags::FOG).fog_enabled);
    assert!(!m(ShowFlags::BLOOM).bloom_enabled);
    let no_ssgi = m(ShowFlags::SSGI);
    assert!(!no_ssgi.ssgi_enabled);
    // The main pass still samples the occlusion through the resolve.
    assert!(no_ssgi.ssgi_resolve_enabled);
    let no_ssr = m(ShowFlags::SSR);
    assert!(!no_ssr.ssr_enabled && !no_ssr.rt_reflections_enabled);
    assert!(!m(ShowFlags::LINES).lines_enabled);
    assert!(!m(ShowFlags::REACTIVE).reactive_mask_enabled);
}

#[test]
fn flat_modes_drop_the_surface_effect_stack() {
    let inputs = every_masked_pass();
    for mode in [ViewMode::Unlit, ViewMode::Wireframe] {
        let out = lit(inputs, mode, ShowFlags::all());
        assert!(!out.ssao_enabled && !out.ssr_enabled && !out.ssgi_enabled);
        assert!(!out.ssgi_resolve_enabled && !out.rt_reflections_enabled);
        assert!(!out.bloom_enabled && !out.fog_enabled && !out.decals_enabled);
        assert!(!out.taa_enabled && !out.upscale_enabled && !out.velocity_enabled);
        assert!(!out.ssr_prepass_enabled);
        // Shadows follow their show flag, not the mode.
        assert!(out.shadow_enabled);
    }
}

#[test]
fn occlusion_view_routes_ao_into_the_composite() {
    let inputs = every_masked_pass();
    let out = lit(inputs, ViewMode::Occlusion, ShowFlags::all());
    assert!(out.composite_reads_ao && out.ssao_enabled);
    assert!(!out.bloom_enabled, "bloom never overlaps the extended AO");
    // With SSAO unavailable there is nothing to route.
    let mut no_ssao = inputs;
    no_ssao.gates.ssao = false;
    let out = lit(no_ssao, ViewMode::Occlusion, ShowFlags::all());
    assert!(!out.composite_reads_ao);
}

#[test]
fn motion_view_routes_the_velocity_into_the_composite() {
    let inputs = every_masked_pass();
    let out = lit(inputs, ViewMode::Motion, ShowFlags::all());
    assert!(out.composite_reads_motion && !out.composite_reads_ao);
    assert!(!out.composite_reads_reactive);
    let reactive = lit(inputs, ViewMode::Reactive, ShowFlags::all());
    assert!(reactive.composite_reads_reactive && !reactive.composite_reads_motion);
    assert!(!lit(inputs, ViewMode::Lit, ShowFlags::all()).composite_reads_motion);
    // The composite's read keeps the pooled velocity alive to the present.
    let graph = build_frame_graph(&out).expect("the graph builds");
    let velocity = graph
        .resources
        .iter()
        .find(|r| r.label == "gbuffer_velocity")
        .expect("gbuffer_velocity present");
    assert_eq!(velocity.lifetime.last, graph.passes.len() - 1);
}

#[test]
fn channel_modes_keep_the_prepass_alive() {
    let mut inputs = nothing();
    inputs.gates.gbuffer = true;
    for mode in [
        ViewMode::Normals,
        ViewMode::Roughness,
        ViewMode::Occlusion,
        ViewMode::Depth,
    ] {
        assert!(lit(inputs, mode, ShowFlags::all()).ssr_prepass_enabled);
        let pass = PassDecisions::select(&PassInputs { mode, ..inputs });
        assert!(pass.get(PassId::GBufferPrepass).runs, "{mode:?}");
    }
    assert!(!lit(inputs, ViewMode::Lit, ShowFlags::all()).ssr_prepass_enabled);
}

#[test]
fn masked_inputs_still_compile_and_ao_read_orders_composite_last() {
    let inputs = every_masked_pass();
    for mode in ViewMode::ALL {
        let out = lit(inputs, mode, ShowFlags::all());
        let graph = build_frame_graph(&out).expect("masked graph compiles");
        assert_eq!(
            graph.passes.last().map(|p| p.id),
            Some(PassId::Composite),
            "{mode:?}"
        );
    }
    // The occlusion view's composite read extends ao_output's lifetime to the
    // last pass.
    let out = lit(inputs, ViewMode::Occlusion, ShowFlags::all());
    let graph = build_frame_graph(&out).expect("the occlusion graph compiles");
    let ao = graph
        .resources
        .iter()
        .find(|r| r.label == "ao_output")
        .expect("ao_output present");
    assert_eq!(ao.lifetime.last, graph.passes.len() - 1);
}
