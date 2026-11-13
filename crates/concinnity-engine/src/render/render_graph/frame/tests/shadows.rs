// The whole graph's order, and the shadow passes' reads of the cull.

use super::*;
use crate::render::render_graph::{FAR_SHADOW_MAP_RESTS, ResourceState};

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

// Every barrier the graph records on the resource labeled `label`, as
// (from, to) pairs in pass order, before and after each pass.
fn transitions(g: &CompiledGraph, label: &str) -> Vec<(ResourceState, ResourceState)> {
    let r = resource_of(g, label);
    g.passes
        .iter()
        .flat_map(|p| p.barriers_before.iter().chain(&p.barriers_after))
        .filter(|b| b.resource_index() == r)
        .map(|b| (b.source_state(), b.to_state()))
        .collect()
}

// A frame that keeps the far map has no pass for it and no transition on
// it: the map rests readable, so its readers sample it as it was left. A frame
// that redraws it opens the write from that resting read and leaves it read in
// every stage it rests in.
#[test]
fn a_kept_far_map_has_no_pass_and_no_transition() {
    for drawn in [true, false] {
        let mut i = all_off();
        i.shadow_enabled = true;
        i.bindless_cull_enabled = true;
        i.fog_enabled = true;
        i.far_shadow_map_size = 1024;
        i.far_shadow_drawn = drawn;
        let g = build_frame_graph(&i).expect("compiles");
        assert_eq!(g.pass_index(PassId::FarShadow).is_some(), drawn);
        let t = transitions(&g, "far_shadow_map");
        if drawn {
            assert_eq!(
                t,
                vec![
                    (ResourceState::Read, ResourceState::Write),
                    (ResourceState::Write, ResourceState::Read),
                ],
            );
            let far = resource_of(&g, "far_shadow_map");
            let closing = g
                .passes
                .iter()
                .flat_map(|p| p.barriers_before.iter().chain(&p.barriers_after))
                .filter(|b| b.resource_index() == far && b.to_state() == ResourceState::Read)
                .map(|b| b.read_stages())
                .next_back()
                .expect("a read closes the frame");
            assert!(closing.contains(FAR_SHADOW_MAP_RESTS));
        } else {
            assert!(t.is_empty(), "{t:?}");
        }
    }
}

// Every pass that reads the far map, in every graph the frame builder makes,
// reads it in a stage the map rests readable in: a kept frame's reads carry no
// transition, so a stage outside them would read the map unsynchronized.
#[test]
fn the_far_map_is_read_only_in_the_stages_it_rests_readable_in() {
    use crate::render::render_graph::frame::GATED_FLAGS;
    let rests = FAR_SHADOW_MAP_RESTS;
    let mut readers = 0;
    for (_, set) in GATED_FLAGS {
        for drawn in [true, false] {
            let mut i = all_off();
            i.shadow_enabled = true;
            i.far_shadow_map_size = 1024;
            i.far_shadow_drawn = drawn;
            set(&mut i);
            let g = build_frame_graph(&i).expect("compiles");
            // A flag that turns the shadows off drops the map.
            let Some(far) = g.resources.iter().position(|r| r.label == "far_shadow_map") else {
                continue;
            };
            for pass in &g.passes {
                for r in pass
                    .reads
                    .iter()
                    .filter(|r| r.version.resource_index() == far)
                {
                    readers += 1;
                    assert!(rests.contains(r.stage()), "{:?}", pass.id);
                }
            }
        }
    }
    assert!(readers > 0);
}

// The fog froxels run on the async queue whether the frame redraws the far map
// or keeps it: a kept map's reads add no transition, so nothing orders them
// after a graphics pass.
#[test]
fn fog_froxels_stay_async_while_the_far_map_is_kept() {
    use crate::render::render_graph::PassQueue;
    use crate::render::render_graph::frame::GATED_FLAGS;
    let mut i = all_off();
    for (name, set) in GATED_FLAGS {
        if !matches!(*name, "planar_reflection" | "world_hidden") && !name.starts_with("far_shadow")
        {
            set(&mut i);
        }
    }
    i.shadow_enabled = true;
    i.far_shadow_map_size = 1024;
    for drawn in [true, false] {
        i.far_shadow_drawn = drawn;
        let g = build_frame_graph(&i).expect("compiles");
        let fog = g
            .pass_index(PassId::FogFroxel)
            .expect("FogFroxel is in the graph");
        assert_eq!(
            g.passes[fog].queue,
            PassQueue::AsyncCompute,
            "drawn: {drawn}"
        );
    }
}

// The receiver reduction reads the pyramid `HizFinal` builds, so it runs after
// it, and only beside the cascades and the pyramid.
#[test]
fn the_receivers_are_measured_from_the_final_pyramid() {
    let mut i = all_off();
    i.shadow_enabled = true;
    i.bindless_cull_enabled = true;
    i.hiz_build_enabled = true;
    i.shadow_receivers_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let pos = |id| g.pass_index(id).expect("present");
    assert!(pos(PassId::HizFinal) < pos(PassId::ShadowReceivers));
    let _ = resource_of(&g, "shadow_receiver_bounds");
    for off in [
        |i: &mut FrameGraphInputs| i.shadow_enabled = false,
        |i: &mut FrameGraphInputs| i.hiz_build_enabled = false,
        |i: &mut FrameGraphInputs| i.shadow_receivers_enabled = false,
    ] {
        let mut j = i;
        off(&mut j);
        let g = build_frame_graph(&j).expect("compiles");
        assert!(g.pass_index(PassId::ShadowReceivers).is_none());
    }
}
