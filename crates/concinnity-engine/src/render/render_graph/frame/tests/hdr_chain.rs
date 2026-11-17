// The HDR chain after the main pass: exposure, decorations, temporal
// resolves, reflections and the transparent pass.

use super::*;

#[test]
fn auto_exposure_war_pinned_before_first_hdr_writer() {
    // AutoExposure reads hdr_resolve_v1. Decals writes v2 (when
    // enabled). The WAR edge from AutoExposure to Decals pins
    // AutoExposure before Decals; without it, the toposort could
    // place them in either order.
    let mut i = all_off();
    i.auto_exposure_enabled = true;
    i.decals_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    assert_eq!(
        order,
        vec![
            PassId::Main,
            PassId::AutoExposure,
            PassId::Decals,
            PassId::Composite,
        ]
    );
}

#[test]
fn full_hdr_chain_orders_decals_fog_particles_then_ssr() {
    let mut i = all_off();
    i.decals_enabled = true;
    i.fog_enabled = true;
    i.particles_enabled = true;
    i.ssr_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    assert_eq!(
        order,
        vec![
            // ParticlesSim depends on nothing, so it keeps the position its
            // declaration gave it: ahead of the raster front it can overlap.
            PassId::ParticlesSim,
            PassId::Main,
            PassId::Decals,
            PassId::FogFroxel,
            PassId::Fog,
            PassId::ParticlesDraw,
            PassId::SsrResolve,
            PassId::Composite,
        ]
    );
    // Version chain on hdr_resolve walks 1 → 2 → 3 → 4 with
    // SsrResolve reading v4. FogFroxel slots between Decals and Fog
    // (writing the froxel volume to v1) but doesn't touch hdr_resolve,
    // so the version walk skips it.
    let at = |id| &g.passes[g.pass_index(id).expect("present")];
    assert_eq!(at(PassId::Decals).writes[0].version(), 2);
    assert_eq!(at(PassId::Fog).writes[0].version(), 3);
    assert_eq!(at(PassId::ParticlesDraw).writes[0].version(), 4);
    assert_eq!(at(PassId::SsrResolve).reads[0].version(), 4);
    // The draw's pool read is the vertex-stage one the sim's transition has
    // to cover; nothing else in the frame reads it.
    let pool_read = at(PassId::ParticlesDraw)
        .reads
        .iter()
        .find(|r| g.resources[r.resource_index()].label == "particle_pool")
        .expect("the draw declares the pool read");
    assert_eq!(pool_read.stage(), ReadStages::VERTEX);
    assert!(
        at(PassId::ParticlesDraw)
            .reads
            .iter()
            .any(|r| g.resources[r.resource_index()].label == "hdr_depth"),
        "the draw declares the scene depth read its fragment tests against"
    );
}

#[test]
fn upscale_replaces_taa_and_pins_after_the_prepass() {
    // Upscale takes TaaResolve's slot when temporal upscaling is on.
    // TaaResolve must not appear in the compiled graph (the scaler
    // does temporal accumulation itself), and the pre-pass that produces
    // the motion vectors must precede Upscale via the explicit read.
    let mut i = all_off();
    i.gbuffer_prepass_enabled = true;
    i.velocity_enabled = true;
    i.upscale_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    assert!(order.contains(&PassId::Upscale));
    assert!(!order.contains(&PassId::TaaResolve));
    assert!(
        order
            .iter()
            .position(|p| *p == PassId::GBufferPrepass)
            .unwrap()
            < order.iter().position(|p| *p == PassId::Upscale).unwrap()
    );
}

#[test]
fn upscale_takes_precedence_when_both_taa_and_upscale_requested() {
    // If both flags somehow arrive set (the engine layer should
    // forbid this, but the graph is the safety net), Upscale wins
    // and TaaResolve is omitted.
    let mut i = all_off();
    i.velocity_enabled = true;
    i.taa_enabled = true;
    i.upscale_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    assert!(order.contains(&PassId::Upscale));
    assert!(!order.contains(&PassId::TaaResolve));
}

#[test]
fn velocity_taa_pinned_via_explicit_read() {
    // TaaResolve reads the velocity buffer explicitly so the toposort
    // orders the pre-pass that wrote it before TaaResolve via RAW (not
    // declaration order).
    let mut i = all_off();
    i.gbuffer_prepass_enabled = true;
    i.velocity_enabled = true;
    i.taa_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    assert!(
        order
            .iter()
            .position(|p| *p == PassId::GBufferPrepass)
            .unwrap()
            < order.iter().position(|p| *p == PassId::TaaResolve).unwrap()
    );
}

#[test]
fn both_reflection_resolves_read_the_cluster_lists() {
    // A missed ray blends the probes its cluster bins, so each resolve must
    // order after LightCull and keep the lists live until it has read them.
    for rt in [false, true] {
        let mut i = all_off();
        i.ssr_enabled = true;
        i.ssr_prepass_enabled = true;
        i.rt_reflections_enabled = rt;
        i.clustering_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let resolve = if rt {
            PassId::RtReflections
        } else {
            PassId::SsrResolve
        };
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
        let lists = resource_of(&g, "cluster_lists");
        assert!(
            g.passes[pos(resolve)]
                .reads
                .iter()
                .any(|r| r.resource_index() == lists),
            "{resolve:?} reads the cluster lists"
        );
        assert!(pos(PassId::LightCull) < pos(resolve));
        assert!(g.resources[lists].lifetime.last >= pos(resolve));
    }
}

#[test]
fn transparent_reads_the_cluster_lists_with_and_without_a_resolve() {
    // Glass and water blend the probes their cluster bins, on either side of
    // the reflection-resolve branch.
    for ssr in [false, true] {
        let mut i = all_off();
        i.ssr_enabled = ssr;
        i.ssr_prepass_enabled = ssr;
        i.transparent_enabled = true;
        i.clustering_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
        let lists = resource_of(&g, "cluster_lists");
        let trans = pos(PassId::Transparent);
        assert!(
            g.passes[trans]
                .reads
                .iter()
                .any(|r| r.resource_index() == lists),
            "Transparent reads the cluster lists (ssr: {ssr})"
        );
        assert!(pos(PassId::LightCull) < trans);
        assert!(g.resources[lists].lifetime.last >= trans);
    }
}

#[test]
fn transparent_reads_the_shadow_map_with_and_without_a_resolve() {
    // The water body is lit through the sun's cascades, so Transparent reads
    // the shadow map Main sampled, on either side of the resolve branch.
    for ssr in [false, true] {
        let mut i = all_off();
        i.shadow_enabled = true;
        i.ssr_enabled = ssr;
        i.ssr_prepass_enabled = ssr;
        i.transparent_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
        let shadow = resource_of(&g, "shadow_map");
        let trans = pos(PassId::Transparent);
        assert!(
            g.passes[trans]
                .reads
                .iter()
                .any(|r| r.resource_index() == shadow),
            "Transparent reads the shadow map (ssr: {ssr})"
        );
        assert!(pos(PassId::Shadow) < trans);
        assert!(g.resources[shadow].lifetime.last >= trans);
    }
}

#[test]
fn transparent_pinned_between_ssr_resolve_and_taa() {
    // Transparent extends the scene_pre_taa chain by one version
    // after SsrResolve, so the toposort orders SsrResolve →
    // Transparent → TaaResolve via RAW + WAW edges on the same
    // texture.
    let mut i = all_off();
    i.ssr_enabled = true;
    i.taa_enabled = true;
    i.velocity_enabled = true;
    i.transparent_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
    assert!(pos(PassId::SsrResolve) < pos(PassId::Transparent));
    assert!(pos(PassId::Transparent) < pos(PassId::TaaResolve));
}

#[test]
fn planar_mirrors_render_after_their_inputs_and_before_transparent() {
    for ssr in [false, true] {
        let mut i = all_off();
        i.bindless_cull_enabled = true;
        i.shadow_enabled = true;
        i.shadowed_spot_count = 1;
        i.ssr_enabled = ssr;
        i.transparent_enabled = true;
        i.planar_reflection_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
        let planar = pos(PassId::PlanarReflection);
        assert!(pos(PassId::Cull) < planar, "ssr: {ssr}");
        assert!(pos(PassId::Shadow) < planar, "ssr: {ssr}");
        assert!(pos(PassId::SpotShadow) < planar, "ssr: {ssr}");
        assert!(planar < pos(PassId::Transparent), "ssr: {ssr}");
        let mirrors = resource_of(&g, "planar_mirrors");
        assert!(
            g.passes[pos(PassId::Transparent)]
                .reads
                .iter()
                .any(|r| r.resource_index() == mirrors),
            "Transparent reads the mirrors (ssr: {ssr})"
        );
    }
}

#[test]
fn planar_mirrors_need_a_transparent_pass_to_read_them() {
    let mut i = all_off();
    i.planar_reflection_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    assert!(g.passes.iter().all(|p| p.id != PassId::PlanarReflection));
    assert!(g.resources.iter().all(|r| r.label != "planar_mirrors"));

    i.planar_reflection_enabled = false;
    i.transparent_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    assert!(g.passes.iter().all(|p| p.id != PassId::PlanarReflection));
}

#[test]
fn transparent_works_without_ssr() {
    // Without a reflection resolve there is no scene_pre_taa texture, so
    // Transparent RMWs hdr_resolve directly and TaaResolve reads that.
    let mut i = all_off();
    i.taa_enabled = true;
    i.velocity_enabled = true;
    i.transparent_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    assert!(order.contains(&PassId::Transparent));
    assert!(!order.contains(&PassId::SsrResolve));
    let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
    assert!(pos(PassId::Main) < pos(PassId::Transparent));
    assert!(pos(PassId::Transparent) < pos(PassId::TaaResolve));
}

// A resource is declared only where a pass writes it. The engine points
// several names at one texture depending on configuration, and declaring the
// second name anyway would give one GPU object two barrier timelines, each
// transitioning it from a state the other just left. These pin the three
// configurations where that could recur.

#[test]
fn no_reflection_resolve_means_no_scene_pre_taa_resource() {
    // With neither SSR nor RT the engine binds the pre-TAA scene name to
    // hdr_resolve itself, so a scene_pre_taa resource would be a second
    // handle on that object.
    let mut i = all_off();
    i.taa_enabled = true;
    i.velocity_enabled = true;
    i.transparent_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    assert!(
        !g.resources.iter().any(|r| r.label == "scene_pre_taa"),
        "scene_pre_taa is hdr_resolve here; it must not be declared twice"
    );
    // And it reappears the moment a pass genuinely writes it.
    i.ssr_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    assert!(g.resources.iter().any(|r| r.label == "scene_pre_taa"));
}

#[test]
fn no_temporal_pass_means_no_scene_color_resource() {
    // Neither TaaResolve nor Upscale runs, so scene_color is bound to the
    // latest pre-TAA scene texture and Composite reads that version.
    let mut i = all_off();
    i.ssr_enabled = true;
    i.bloom_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    assert!(
        !g.resources.iter().any(|r| r.label == "scene_color"),
        "scene_color is the pre-TAA scene here; it must not be declared twice"
    );
    // Bloom and Composite consume the SsrResolve output instead.
    let pre_taa = resource_of(&g, "scene_pre_taa");
    let bloom = g
        .passes
        .iter()
        .find(|p| p.id == PassId::Bloom)
        .expect("bloom present");
    assert!(
        bloom.reads.iter().any(|r| r.resource_index() == pre_taa),
        "Bloom reads the resolve output directly"
    );
    i.taa_enabled = true;
    i.velocity_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    assert!(g.resources.iter().any(|r| r.label == "scene_color"));
}

#[test]
fn glass_without_a_reflection_resolve_extends_the_hdr_chain() {
    // Transparent blends into whichever texture carries the pre-TAA scene.
    // With no resolve that is hdr_resolve, so its write must bump the
    // hdr_resolve version rather than branch a second resource -- otherwise
    // the glass blend and the decoration chain order independently over one
    // object.
    let mut i = all_off();
    i.transparent_enabled = true;
    i.decals_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let hdr = resource_of(&g, "hdr_resolve");
    let trans = g
        .passes
        .iter()
        .find(|p| p.id == PassId::Transparent)
        .expect("transparent present");
    let write = trans
        .writes
        .iter()
        .find(|w| w.resource_index() == hdr)
        .expect("Transparent writes hdr_resolve");
    // It RMWs: the version it produces is one past the decoration chain's.
    let read = trans
        .reads
        .iter()
        .find(|r| r.resource_index() == hdr)
        .expect("Transparent reads hdr_resolve");
    assert_eq!(
        write.version(),
        read.version() + 1,
        "the glass blend extends the chain it read"
    );
    // Composite sees the blended version, not the pre-glass one.
    let composite = g
        .passes
        .iter()
        .find(|p| p.id == PassId::Composite)
        .expect("composite present");
    assert!(
        composite
            .reads
            .iter()
            .any(|r| r.resource_index() == hdr && r.version() == write.version()),
        "Composite reads the post-glass version"
    );
}

#[test]
fn transparent_off_means_no_slot() {
    // The pass is omitted when nothing in the world is transparent:
    // no orphan slot, no executor stub triggered.
    let mut i = all_off();
    i.ssr_enabled = true;
    i.taa_enabled = true;
    i.velocity_enabled = true;
    // transparent_enabled left at false.
    let g = build_frame_graph(&i).expect("compiles");

    assert!(
        !g.passes
            .iter()
            .map(|p| p.id)
            .any(|x| x == PassId::Transparent)
    );
}

#[test]
fn lines_close_the_hdr_decoration_chain() {
    // The node RMWs hdr_resolve last, so the lines draw over the lit +
    // decorated scene and SSR / TAA then consume them like any other
    // scene content.
    let mut i = all_off();
    i.decals_enabled = true;
    i.particles_enabled = true;
    i.ssr_enabled = true;
    i.lines_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
    assert!(pos(PassId::Decals) < pos(PassId::Lines));
    assert!(pos(PassId::ParticlesDraw) < pos(PassId::Lines));
    assert!(pos(PassId::Lines) < pos(PassId::SsrResolve));
}

#[test]
fn lines_off_means_no_slot() {
    // A frame that published no lines omits the pass entirely.
    let g = build_frame_graph(&all_off()).expect("compiles");
    assert!(!g.passes.iter().any(|p| p.id == PassId::Lines));
}

#[test]
fn a_hidden_world_drops_the_lines() {
    // Behind an opaque menu backdrop nothing of the world is visible, so
    // the masked graph drops the lines with every other world pass.
    let mut i = all_off();
    i.lines_enabled = true;
    i.world_hidden = true;
    let g = build_frame_graph(&i).expect("compiles");
    assert!(!g.passes.iter().any(|p| p.id == PassId::Lines));
}
