// The cull, the Hi-Z pyramid and the late occlusion pass: what each reads,
// and the order the pyramid's readers and rebuilds take.

use super::*;

// The late particles draw over what Transparent left, before the temporal
// pass takes it, with or without a reflection resolve between; without a
// late emitter there is no late pass.
#[test]
fn late_particles_draw_after_transparent_and_before_taa() {
    for ssr in [false, true] {
        let mut i = all_off();
        i.particles_enabled = true;
        i.transparent_enabled = true;
        i.taa_enabled = true;
        i.ssr_enabled = ssr;
        let g = build_frame_graph(&i).expect("compiles");
        assert!(g.passes.iter().all(|p| p.id != PassId::ParticlesLateDraw));
        i.late_particles_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
        assert!(pos(PassId::ParticlesDraw) < pos(PassId::Transparent));
        assert!(
            pos(PassId::Transparent) < pos(PassId::ParticlesLateDraw),
            "ssr {ssr}"
        );
        assert!(
            pos(PassId::ParticlesLateDraw) < pos(PassId::TaaResolve),
            "ssr {ssr}"
        );
    }
}

#[test]
fn hiz_final_closes_the_frame_over_the_last_depth_version() {
    // The terminal pyramid rebuild must read the depth version every
    // decoration pass has already written and read, so main depth stays live
    // to the end of the graph. Raymarch bumps depth to v2, so HizFinal reads
    // v2 and lands after Lines / Transparent, which read the same version.
    let mut i = all_off();
    i.hiz_build_enabled = true;
    i.raymarch_enabled = true;
    i.lines_enabled = true;
    i.transparent_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
    assert!(pos(PassId::Lines) < pos(PassId::HizFinal));
    assert!(pos(PassId::Transparent) < pos(PassId::HizFinal));
    assert!(pos(PassId::HizFinal) < pos(PassId::Composite));

    let depth = resource_of(&g, "hdr_depth");
    let hizf = &g.passes[pos(PassId::HizFinal)];
    let read = hizf
        .reads
        .iter()
        .find(|r| r.resource_index() == depth)
        .expect("HizFinal reads depth");
    assert_eq!(read.version(), 2, "Main writes v1, Raymarch bumps to v2");
    // And it is the graph's last touch of depth: the resource's lifetime has
    // to reach this pass or an aliaser could reuse the memory under it.
    assert_eq!(g.resources[depth].lifetime.last, pos(PassId::HizFinal));
}

#[test]
fn cull_reads_the_pyramid_the_terminal_build_overwrites() {
    // Early cull tests against the pyramid the previous frame left. That
    // read is what gives the terminal rebuild a write-after-read edge; without
    // it the rebuild is unordered against the cull that is still sampling.
    let mut i = all_off();
    i.hiz_build_enabled = true;
    i.bindless_cull_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
    let hiz = resource_of(&g, "hiz_pyramid");
    assert!(
        g.passes[pos(PassId::Cull)]
            .reads
            .iter()
            .any(|r| r.resource_index() == hiz)
    );
    assert!(pos(PassId::Cull) < pos(PassId::HizFinal));
}

#[test]
fn grass_reads_the_pyramid_ahead_of_its_rebuilds() {
    // The grass kernel occlusion-tests its tiles against the pyramid the
    // previous frame left, so both rebuilds must wait for it, the mid-frame
    // one under two-pass occlusion included.
    for two_pass in [false, true] {
        let mut i = all_off();
        i.hiz_build_enabled = true;
        i.bindless_cull_enabled = true;
        i.two_pass_occlusion_enabled = two_pass;
        i.grass_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
        let hiz = resource_of(&g, "hiz_pyramid");
        assert!(
            g.passes[pos(PassId::Grass)]
                .reads
                .iter()
                .any(|r| r.resource_index() == hiz)
        );
        assert!(pos(PassId::Grass) < pos(PassId::HizFinal));
        if two_pass {
            assert!(pos(PassId::Grass) < pos(PassId::HizEarly));
        }
    }
}

// The bend pass stamps the field both placement kernels read, and the
// cascade's blades are drawn by the Shadow pass, so the order is bend,
// then the kernels, then their draws.
#[test]
fn grass_bends_before_it_places_and_casts_before_main() {
    let mut i = all_off();
    i.bindless_cull_enabled = true;
    i.gbuffer_prepass_enabled = true;
    i.velocity_enabled = true;
    i.grass_enabled = true;
    i.shadow_enabled = true;
    i.grass_shadow_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
    assert!(pos(PassId::GrassBend) < pos(PassId::Grass));
    assert!(pos(PassId::GrassBend) < pos(PassId::GrassShadow));
    assert!(pos(PassId::GrassShadow) < pos(PassId::Shadow));
    assert!(pos(PassId::Grass) < pos(PassId::GBufferPrepass));
    assert!(pos(PassId::Shadow) < pos(PassId::Main));
    let blades = resource_of(&g, "grass_shadow_blades");
    assert!(
        g.passes[pos(PassId::Shadow)]
            .reads
            .iter()
            .any(|r| r.resource_index() == blades)
    );

    // Without shadows, or with the cascade's draw off, nothing is placed
    // for it; the bend pass still runs for the drawn blades.
    for (shadow, cast) in [(false, true), (true, false)] {
        i.shadow_enabled = shadow;
        i.grass_shadow_enabled = cast;
        let g = build_frame_graph(&i).expect("compiles");
        assert!(g.passes.iter().all(|p| p.id != PassId::GrassShadow));
        assert!(g.passes.iter().any(|p| p.id == PassId::GrassBend));
    }
}

#[test]
fn hiz_final_off_means_no_pyramid_in_the_graph() {
    // A world without the GPU-cull path builds no pyramid, so neither the node
    // nor the resource may appear (an imported resource with no pass would
    // still take a registry entry in every backend).
    let g = build_frame_graph(&all_off()).expect("compiles");

    assert!(!g.passes.iter().map(|p| p.id).any(|x| x == PassId::HizFinal));
    assert!(!g.resources.iter().any(|r| r.label == "hiz_pyramid"));
}

#[test]
fn a_hidden_world_still_rebuilds_the_pyramid() {
    // Masking drops every world pass, but the pyramid feeds the *next* frame's
    // cull, so the terminal build survives: dropping it would leave a stale
    // pyramid the frame after the menu closes.
    let mut i = all_off();
    i.hiz_build_enabled = true;
    i.bindless_cull_enabled = true;
    i.world_hidden = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    assert_eq!(
        order,
        vec![PassId::Main, PassId::HizFinal, PassId::Composite]
    );
}

#[test]
fn depth_readers_all_sample_the_post_raymarch_version() {
    // Raymarch writes hit depth back, so every later decoration must sample
    // the version it produced. Reading Main's version instead would be a
    // write-after-read against Raymarch and pin the readers ahead of it.
    let mut i = all_off();
    i.raymarch_enabled = true;
    i.decals_enabled = true;
    i.fog_enabled = true;
    i.lines_enabled = true;
    i.transparent_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
    let depth = resource_of(&g, "hdr_depth");
    for pass in [
        PassId::Decals,
        PassId::Fog,
        PassId::Lines,
        PassId::Transparent,
    ] {
        let read = g.passes[pos(pass)]
            .reads
            .iter()
            .find(|r| r.resource_index() == depth)
            .unwrap_or_else(|| panic!("{pass:?} reads depth"));
        assert_eq!(read.version(), 2, "{pass:?}");
    }
    assert!(pos(PassId::Raymarch) < pos(PassId::Decals));
}

#[test]
fn the_msaa_color_attachment_is_declared_only_when_multisampled() {
    // Without MSAA there is no resolve step and the single color target is
    // the spine, so declaring `hdr_color` too would put two graph resources
    // on one GPU object. Every backend already reflects this: Vulkan leaves
    // `color_images` empty, DirectX and Metal leave `resolve` None.
    let mut i = all_off();
    i.hdr_sample_count = 1;
    let g = build_frame_graph(&i).expect("compiles");
    assert!(!g.resources.iter().any(|r| r.label == "hdr_color"));
    assert!(g.resources.iter().any(|r| r.label == "hdr_resolve"));

    i.hdr_sample_count = 4;
    let g = build_frame_graph(&i).expect("compiles");
    assert!(g.resources.iter().any(|r| r.label == "hdr_color"));
    assert!(g.resources.iter().any(|r| r.label == "hdr_resolve"));
}

#[test]
fn dropping_the_msaa_attachment_keeps_the_phase_order() {
    // The hdr_color write-after-write is one of three edges pinning MainLate
    // after Main; the depth and hdr_resolve writes carry the order on their
    // own, so a single-sampled two-pass frame still runs in phase order.
    let mut i = all_off();
    i.hdr_sample_count = 1;
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
}

#[test]
fn two_pass_without_bindless_cull_is_noop() {
    // Two-pass rides the bindless GPU-cull path; requesting it without
    // a bindless shader must not insert any late nodes.
    let mut i = all_off();
    i.two_pass_occlusion_enabled = true; // bindless_cull_enabled left false
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    assert_eq!(order, vec![PassId::Main, PassId::Composite]);
    assert!(!order.contains(&PassId::HizEarly));
    assert!(!order.contains(&PassId::CullLate));
    assert!(!order.contains(&PassId::MainLate));
}

// Under two-pass occlusion the late lists are drawn into the G-buffer too,
// after the late cull, and the reflection resolve reads that version rather
// than the pre-pass's; the readers ahead of Main keep the pre-pass's.
#[test]
fn the_late_lists_reach_the_gbuffer_the_resolve_reads() {
    let mut i = all_off();
    i.bindless_cull_enabled = true;
    i.two_pass_occlusion_enabled = true;
    i.gbuffer_prepass_enabled = true;
    i.ssr_prepass_enabled = true;
    i.ssr_enabled = true;
    i.ssao_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
    assert!(pos(PassId::CullLate) < pos(PassId::GBufferLate));
    assert!(pos(PassId::SsaoBlur) < pos(PassId::GBufferLate));
    assert!(pos(PassId::GBufferLate) < pos(PassId::SsrResolve));

    let normal_depth = resource_of(&g, "gbuffer_normal_depth");
    let read_version = |pass: PassId| {
        g.passes[pos(pass)]
            .reads
            .iter()
            .find(|r| r.resource_index() == normal_depth)
            .map(|r| r.version())
    };
    assert_eq!(read_version(PassId::SsaoBlur), Some(1));
    assert_eq!(read_version(PassId::SsrResolve), Some(2));
    let late = &g.passes[pos(PassId::GBufferLate)];
    for label in [
        "gbuffer_normal_depth",
        "gbuffer_roughness",
        "gbuffer_velocity",
        "gbuffer_depth",
    ] {
        let r = resource_of(&g, label);
        assert!(
            late.writes.iter().any(|w| w.resource_index() == r),
            "{label}"
        );
    }
    let draw_args_late = resource_of(&g, "draw_args_late");
    assert!(
        late.reads
            .iter()
            .any(|r| r.resource_index() == draw_args_late)
    );
}

// The model-history snapshot follows the last G-buffer pass that reads the
// previous frame's models: the late one under two-pass occlusion, else the
// pre-pass, and none without a G-buffer.
#[test]
fn the_model_history_snapshot_follows_the_last_gbuffer_pass() {
    let mut i = all_off();
    i.bindless_cull_enabled = true;
    i.ssr_prepass_enabled = true;
    i.ssr_enabled = true;
    let pass = |i: &FrameGraphInputs| model_history_pass(&build_frame_graph(i).expect("compiles"));
    assert_eq!(pass(&i), None);
    i.gbuffer_prepass_enabled = true;
    assert_eq!(pass(&i), Some(PassId::GBufferPrepass));
    i.two_pass_occlusion_enabled = true;
    assert_eq!(pass(&i), Some(PassId::GBufferLate));
}

// Without two-pass occlusion, or without a G-buffer, there is no late
// G-buffer pass.
#[test]
fn the_late_gbuffer_pass_needs_both_halves() {
    for (two_pass, gbuffer) in [(false, true), (true, false)] {
        let mut i = all_off();
        i.bindless_cull_enabled = true;
        i.two_pass_occlusion_enabled = two_pass;
        i.gbuffer_prepass_enabled = gbuffer;
        i.ssr_prepass_enabled = true;
        i.ssr_enabled = true;
        let g = build_frame_graph(&i).expect("compiles");
        assert!(
            g.passes.iter().all(|p| p.id != PassId::GBufferLate),
            "two_pass {two_pass}, gbuffer {gbuffer}"
        );
    }
}

#[test]
fn two_pass_shifts_post_chain_head_to_main_late() {
    // AutoExposure + the RMW chain must read MainLate's hdr_resolve (v2),
    // not Main's (v1), so the post stack sees the combined two-pass
    // scene. Main writes v1, MainLate writes v2, AutoExposure reads v2,
    // Decals bumps to v3.
    let mut i = all_off();
    i.bindless_cull_enabled = true;
    i.two_pass_occlusion_enabled = true;
    i.auto_exposure_enabled = true;
    i.decals_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
    let pos = |p: PassId| order.iter().position(|x| *x == p).expect("present");
    assert!(pos(PassId::MainLate) < pos(PassId::AutoExposure));
    assert!(pos(PassId::AutoExposure) < pos(PassId::Decals));
    // Version walk: MainLate RMWs hdr_resolve to v2, Decals to v3.
    let main_late = &g.passes[pos(PassId::MainLate)];
    // hdr_resolve is the last write MainLate declares (depth, color, resolve).
    assert_eq!(main_late.writes.last().unwrap().version(), 2);
    let decals = &g.passes[pos(PassId::Decals)];
    assert_eq!(decals.writes[0].version(), 3);
}

// The view samples read the pyramid `HizFinal` builds, so they run after it
// and write the imported cells the CPU reads back; only beside the pyramid,
// and only when the backend built them.
#[test]
fn the_view_is_sampled_from_the_final_pyramid() {
    let mut i = all_off();
    i.bindless_cull_enabled = true;
    i.hiz_build_enabled = true;
    i.view_samples_enabled = true;
    let g = build_frame_graph(&i).expect("compiles");
    let pos = |id| g.pass_index(id).expect("present");
    assert!(pos(PassId::HizFinal) < pos(PassId::ViewSamples));
    let cells = resource_of(&g, "view_samples");
    let writers: Vec<PassId> = g
        .passes
        .iter()
        .filter(|p| p.writes.iter().any(|w| w.resource_index() == cells))
        .map(|p| p.id)
        .collect();
    assert_eq!(writers, [PassId::ViewSamples]);
    for off in [
        |i: &mut FrameGraphInputs| i.hiz_build_enabled = false,
        |i: &mut FrameGraphInputs| i.view_samples_enabled = false,
    ] {
        let mut j = i;
        off(&mut j);
        let g = build_frame_graph(&j).expect("compiles");
        assert!(g.pass_index(PassId::ViewSamples).is_none());
    }
}
