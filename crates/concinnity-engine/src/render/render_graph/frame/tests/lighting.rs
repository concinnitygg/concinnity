// Indirect lighting and the passes that light the scene with it: SSGI,
// ray-traced reflections, the raymarch, the atmosphere and the ocean.

use super::*;

#[test]
fn ssgi_off_means_no_slot() {
    // IBL-only indirect lighting: the pass is omitted entirely.
    let g = build_frame_graph(&all_off()).expect("compiles");

    assert!(!g.passes.iter().map(|p| p.id).any(|x| x == PassId::Ssgi));
}

#[test]
fn rt_reflections_off_means_no_slot() {
    // No ray tracing requested: the pass is omitted entirely.
    let g = build_frame_graph(&all_off()).expect("compiles");

    assert!(
        !g.passes
            .iter()
            .map(|p| p.id)
            .any(|x| x == PassId::RtReflections)
    );
}

#[test]
fn rt_reflections_read_the_scene_in_compute_and_fragment() {
    // The classify and trace kernels read the scene in compute, the
    // composite in the same pass in its fragment.
    let mut i = all_off();
    i.rt_reflections_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let rt = g
        .passes
        .iter()
        .find(|p| p.id == PassId::RtReflections)
        .expect("present");
    assert!(!rt.reads.is_empty());
    for read in &rt.reads {
        assert!(read.stage().contains(ReadStages::COMPUTE));
    }
}

#[test]
fn rt_reflections_occupy_the_ssr_resolve_slot() {
    // RtReflections reads the post-decoration hdr_resolve and writes
    // scene_pre_taa, exactly where SsrResolve would, so it orders after
    // ParticlesDraw and before TaaResolve.
    let mut i = all_off();
    i.rt_reflections_enabled = true;
    i.particles_enabled = true;
    i.taa_enabled = true;
    i.velocity_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
    assert!(order.contains(&PassId::RtReflections));
    assert!(pos(PassId::ParticlesDraw) < pos(PassId::RtReflections));
    assert!(pos(PassId::RtReflections) < pos(PassId::TaaResolve));
}

#[test]
fn rt_reflections_take_precedence_over_ssr_resolve() {
    // RT alone inserts RtReflections, not SsrResolve.
    let mut i = all_off();
    i.rt_reflections_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    assert!(order.contains(&PassId::RtReflections));
    assert!(!order.contains(&PassId::SsrResolve));

    // With both flags set (RT available + SSR fallback authored), hardware
    // RT wins and SsrResolve is omitted; never two in the same slot.
    let mut both = all_off();
    both.ssr_enabled = true;
    both.rt_reflections_enabled = true;
    let g = build_frame_graph(&both).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    assert!(order.contains(&PassId::RtReflections));
    assert!(!order.contains(&PassId::SsrResolve));
}

#[test]
fn ssgi_reads_the_lit_scene_before_decals_and_writes_none_of_it() {
    // AutoExposure and SSGI both read hdr_resolve_v1; SSGI keeps what it
    // gathers in its own targets, so Decals RMWs to v2 after both.
    let mut i = all_off();
    i.auto_exposure_enabled = true;
    i.ssgi_enabled = true;
    i.decals_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
    assert!(pos(PassId::Main) < pos(PassId::Ssgi));
    assert!(pos(PassId::Ssgi) < pos(PassId::Decals));
    assert!(pos(PassId::AutoExposure) < pos(PassId::Decals));
    let ssgi = &g.passes[pos(PassId::Ssgi)];
    assert!(ssgi.writes.is_empty());
    let main = &g.passes[pos(PassId::Main)];
    let main_resolve = main.writes.last().expect("hdr_resolve");
    assert!(
        ssgi.reads
            .iter()
            .any(|r| r.resource_index() == main_resolve.resource_index()
                && r.version() == main_resolve.version())
    );
    let decals = &g.passes[pos(PassId::Decals)];
    assert_eq!(decals.writes[0].version(), 2);
}

#[test]
fn the_ssgi_resolve_precedes_main_which_reads_it() {
    let mut i = all_off();
    i.gbuffer_prepass_enabled = true;
    i.ssr_prepass_enabled = true;
    i.velocity_enabled = true;
    i.ssao_enabled = true;
    i.ssgi_enabled = true;
    i.ssgi_resolve_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
    assert!(pos(PassId::GBufferPrepass) < pos(PassId::SsgiResolve));
    assert!(pos(PassId::SsaoBlur) < pos(PassId::SsgiResolve));
    assert!(pos(PassId::SsgiResolve) < pos(PassId::Main));
    let resolve = &g.passes[pos(PassId::SsgiResolve)];
    let [indirect] = resolve.writes[..] else {
        panic!("the resolve writes its one output");
    };
    let main = &g.passes[pos(PassId::Main)];
    assert!(main.reads.iter().any(|r| r.resource_version() == indirect));
    // It reads the occlusion it folds in.
    let ssao = &g.passes[pos(PassId::SsaoBlur)];
    assert!(
        resolve
            .reads
            .iter()
            .any(|r| r.resource_version() == ssao.writes[0])
    );
}

// The mark is the mask's first writer, ahead of Main like the other
// pre-pass consumers, and only an upscaler that reads the mask brings it.
#[test]
fn the_disocclusion_mark_writes_the_mask_ahead_of_main_and_the_particles() {
    let mut i = all_off();
    i.gbuffer_prepass_enabled = true;
    i.velocity_enabled = true;
    i.reactive_mask_enabled = true;
    i.upscale_enabled = true;
    i.upscale_reads_reactive = true;
    i.disocclusion_enabled = true;
    i.particles_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
    assert!(pos(PassId::GBufferPrepass) < pos(PassId::Disocclusion));
    assert!(pos(PassId::Disocclusion) < pos(PassId::Main));
    assert!(pos(PassId::Disocclusion) < pos(PassId::ParticlesDraw));
    let mark = &g.passes[pos(PassId::Disocclusion)];
    let [mask] = mark.writes[..] else {
        panic!("the mark writes the mask alone");
    };
    assert_eq!(mask.version(), 1);
    let particles = &g.passes[pos(PassId::ParticlesDraw)];
    assert!(
        particles
            .writes
            .iter()
            .any(|w| w.resource_index() == mask.resource_index() && w.version() == 2)
    );

    i.upscale_reads_reactive = false;
    let g = build_frame_graph(&i).expect("compiles");
    assert!(g.pass_index(PassId::Disocclusion).is_none());
}

#[test]
fn ssgi_after_raymarch() {
    // With both on, SSGI reads the post-raymarch scene: Raymarch v1→v2,
    // SSGI reads v2, SsrResolve reads v2 too.
    let mut i = all_off();
    i.raymarch_enabled = true;
    i.ssgi_enabled = true;
    i.ssr_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
    assert!(pos(PassId::Raymarch) < pos(PassId::Ssgi));
    let raymarch = &g.passes[pos(PassId::Raymarch)];
    let ssgi = &g.passes[pos(PassId::Ssgi)];
    assert!(
        ssgi.reads
            .iter()
            .any(|r| r.resource_version() == raymarch.writes[0])
    );
}

#[test]
fn ssgi_reads_the_prepass_normals_and_motion() {
    // The accumulation reprojects through the pre-pass's motion vectors, so
    // SSGI reads both of the pre-pass channels it uses, which also pins it
    // after the pre-pass.
    let mut i = all_off();
    i.gbuffer_prepass_enabled = true;
    i.ssr_prepass_enabled = true;
    i.velocity_enabled = true;
    i.ssgi_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
    let prepass = &g.passes[pos(PassId::GBufferPrepass)];
    let ssgi = &g.passes[pos(PassId::Ssgi)];
    let [normal_depth, roughness, velocity, _depth] = prepass.writes[..] else {
        panic!("the pre-pass writes four channels");
    };
    for channel in [normal_depth, roughness, velocity] {
        assert!(
            ssgi.reads
                .iter()
                .any(|r| r.resource_index() == channel.resource_index())
        );
    }
    assert!(pos(PassId::GBufferPrepass) < pos(PassId::Ssgi));
}

#[test]
fn raymarch_off_means_no_slot() {
    // No `SdfVolume` in the world: pass is omitted, no executor stub
    // ever fires.
    let g = build_frame_graph(&all_off()).expect("compiles");

    assert!(!g.passes.iter().map(|p| p.id).any(|x| x == PassId::Raymarch));
}

#[test]
fn raymarch_pinned_between_auto_exposure_and_decals() {
    // AutoExposure reads hdr_resolve_v1 (WAR); Raymarch RMWs to v2;
    // Decals RMWs to v3. The toposort orders the three through the
    // version chain without needing declaration-order tie-breaks.
    let mut i = all_off();
    i.auto_exposure_enabled = true;
    i.raymarch_enabled = true;
    i.decals_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    assert_eq!(
        order,
        vec![
            PassId::Main,
            PassId::AutoExposure,
            PassId::Raymarch,
            PassId::Decals,
            PassId::Composite,
        ]
    );
    // Version chain on hdr_resolve walks 1 → 2 → 3.
    let raymarch = &g.passes[2];
    assert_eq!(raymarch.writes[0].version(), 2);
    let decals = &g.passes[3];
    assert_eq!(decals.writes[0].version(), 3);
}

// The atmosphere's tables are built before Main draws the sky and lights
// with them, and the haze lands on the decorated scene under the fog.
#[test]
fn the_atmosphere_feeds_main_and_hazes_under_the_fog() {
    let mut i = all_off();
    i.atmosphere_enabled = true;
    i.decals_enabled = true;
    i.fog_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    let at = |p| order.iter().position(|x| *x == p).expect("pass present");
    assert!(at(PassId::Atmosphere) < at(PassId::Main));
    assert!(at(PassId::Decals) < at(PassId::AerialPerspective));
    assert!(at(PassId::AerialPerspective) < at(PassId::Fog));
    let off = build_frame_graph(&all_off()).expect("compiles");
    assert!(
        off.passes
            .iter()
            .all(|p| p.id != PassId::Atmosphere && p.id != PassId::AerialPerspective)
    );
}

// The ocean is simulated ahead of the transparent pass, which reads its
// arrays in the water's vertex and fragment stages, with or without a
// reflection resolve; with no transparent pass there is nothing to feed.
#[test]
fn the_ocean_simulation_feeds_the_water_draw() {
    for ssr in [false, true] {
        let mut i = all_off();
        i.transparent_enabled = true;
        i.water_sim_enabled = true;
        i.ssr_enabled = ssr;
        let g = build_frame_graph(&i).expect("compiles");
        let at = |id| g.pass_index(id).expect("present");
        assert!(at(PassId::WaterSim) < at(PassId::Transparent));
        let read = g.passes[at(PassId::Transparent)]
            .reads
            .iter()
            .find(|r| g.resources[r.resource_index()].label == "water_cascades")
            .expect("the water draw declares the cascade read");
        assert_eq!(read.stage(), ReadStages::VERTEX | ReadStages::FRAGMENT);
    }
    let mut i = all_off();
    i.water_sim_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    assert!(g.pass_index(PassId::WaterSim).is_none());
}

#[test]
fn raymarch_works_without_auto_exposure_or_decals() {
    // Standalone Raymarch RMWs hdr_resolve_v1 → v2; SsrResolve reads
    // v2 instead of v1. Nothing else in the post chain.
    let mut i = all_off();
    i.raymarch_enabled = true;
    i.ssr_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    assert_eq!(
        order,
        vec![
            PassId::Main,
            PassId::Raymarch,
            PassId::SsrResolve,
            PassId::Composite,
        ]
    );
    let raymarch = &g.passes[1];
    assert_eq!(raymarch.writes[0].version(), 2);
    let ssr = &g.passes[2];
    assert_eq!(ssr.reads[0].version(), 2);
}
