// The whole graph's order, and the shadow passes' reads of the cull.

use super::*;

#[test]
fn full_graph_orders_all_passes_correctly() {
    // Everything on: every pass shows up in the expected order.
    // This is the showcase configuration.
    let mut i = all_off();
    i.shadow_enabled = true;
    i.bindless_cull_enabled = true;
    i.auto_exposure_enabled = true;
    i.bloom_enabled = true;
    i.velocity_enabled = true;
    i.taa_enabled = true;
    i.ssr_enabled = true;
    i.particles_enabled = true;
    i.fog_enabled = true;
    i.decals_enabled = true;
    i.ssr_prepass_enabled = true;
    i.ssao_enabled = true;
    i.gbuffer_prepass_enabled = true;
    i.transparent_enabled = true;
    i.raymarch_enabled = true;

    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();

    // Spot-check relative ordering rather than the exact list: with
    // many independent passes the toposort has flexibility on
    // tie-breaks.
    fn idx(order: &[PassId], p: PassId) -> usize {
        order.iter().position(|x| *x == p).expect("pass present")
    }
    // Cull / GBufferPrepass / SsaoBlur / Shadow / SSAO all precede Main.
    assert!(idx(&order, PassId::Cull) < idx(&order, PassId::Main));
    assert!(idx(&order, PassId::GBufferPrepass) < idx(&order, PassId::Main));
    assert!(idx(&order, PassId::SsaoBlur) < idx(&order, PassId::Main));
    assert!(idx(&order, PassId::Shadow) < idx(&order, PassId::Main));
    // The pre-pass precedes SsaoBlur (SSAO reads its normal+depth).
    assert!(idx(&order, PassId::GBufferPrepass) < idx(&order, PassId::SsaoBlur));
    // AutoExposure post-Main, pre-Raymarch (WAR-pinned on hdr_resolve_v1).
    assert!(idx(&order, PassId::Main) < idx(&order, PassId::AutoExposure));
    assert!(idx(&order, PassId::AutoExposure) < idx(&order, PassId::Raymarch));
    // Raymarch leads the hdr_resolve RMW chain so Decals / Fog /
    // ParticlesDraw blend on top of the raymarched color.
    assert!(idx(&order, PassId::Raymarch) < idx(&order, PassId::Decals));
    // hdr_resolve chain.
    assert!(idx(&order, PassId::Decals) < idx(&order, PassId::Fog));
    assert!(idx(&order, PassId::Fog) < idx(&order, PassId::ParticlesDraw));
    assert!(idx(&order, PassId::ParticlesDraw) < idx(&order, PassId::SsrResolve));
    // FogFroxel populates the volume Fog samples, so it must precede Fog.
    assert!(idx(&order, PassId::FogFroxel) < idx(&order, PassId::Fog));
    // The pre-pass that writes the motion vectors precedes TaaResolve.
    assert!(idx(&order, PassId::GBufferPrepass) < idx(&order, PassId::TaaResolve));
    // Post-TAA chain. Transparent slots between SsrResolve and TaaResolve.
    assert!(idx(&order, PassId::SsrResolve) < idx(&order, PassId::Transparent));
    assert!(idx(&order, PassId::Transparent) < idx(&order, PassId::TaaResolve));
    assert!(idx(&order, PassId::TaaResolve) < idx(&order, PassId::Bloom));
    assert!(idx(&order, PassId::Bloom) < idx(&order, PassId::Composite));
    // Composite is the presenter and runs last.
    assert_eq!(order.last(), Some(&PassId::Composite));
    assert!(g.passes.last().unwrap().presents);
}

#[test]
fn shadow_orders_after_cull_through_the_draw_args_read() {
    // The GPU-driven cascade path draws from an indirect command buffer the
    // cull pass fills, so Shadow has to depend on Cull. Without the edge a
    // backend that runs Cull on the async queue has nothing ordering the two.
    let mut i = all_off();
    i.shadow_enabled = true;
    i.bindless_cull_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let cull = g.pass_index(PassId::Cull).expect("Cull is in the graph");
    let shadow = g
        .pass_index(PassId::Shadow)
        .expect("Shadow is in the graph");
    assert!(
        g.depends_on(cull, shadow),
        "Shadow does not depend on Cull: {:?}",
        g.passes.iter().map(|p| p.id).collect::<Vec<_>>()
    );
    assert!(
        g.pass_precedes(cull, shadow),
        "the schedule does not order them"
    );
}

#[test]
fn spot_shadow_orders_after_cull_through_the_draw_args_read() {
    let mut i = all_off();
    i.shadowed_spot_count = 3;
    i.spot_shadow_slice_size = 512;
    i.bindless_cull_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let cull = g.pass_index(PassId::Cull).expect("Cull is in the graph");
    let spot = g
        .pass_index(PassId::SpotShadow)
        .expect("SpotShadow is in the graph");
    assert!(
        g.depends_on(cull, spot),
        "SpotShadow does not depend on Cull"
    );
    assert!(
        g.pass_precedes(cull, spot),
        "the schedule does not order them"
    );
}

#[test]
fn shadow_without_bindless_cull_declares_no_cull_read() {
    // A world drawing its cascades per-object has no cull output to read, so
    // the edge must not appear and Shadow stays the graph's first pass.
    let mut i = all_off();
    i.shadow_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let shadow = g
        .pass_index(PassId::Shadow)
        .expect("Shadow is in the graph");
    assert!(g.passes[shadow].reads.is_empty());
}
