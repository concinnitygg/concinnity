// What the gates and frame facts ask of the graph, one rule at a time.

use super::super::*;

fn facts() -> FrameFacts {
    FrameFacts {
        hdr_width: 1920,
        hdr_height: 1080,
        hdr_sample_count: 1,
        shadow_map_size: 2048,
        spot_shadow_slice_size: 512,
        ..FrameFacts::default()
    }
}

// The graph inputs in the lit view with every show flag set.
fn inputs(gates: &FeatureGates, facts: &FrameFacts) -> FrameGraphInputs {
    let inputs = PassInputs {
        gates: *gates,
        facts: *facts,
        ..PassInputs::default()
    };
    PassDecisions::select(&inputs).graph_inputs(&inputs)
}

#[test]
fn nothing_on_and_nothing_held_gates_every_pass_off() {
    let off = FeatureGates::default();
    let i = inputs(&off, &facts());
    let mut expected = FrameGraphInputs::all_off();
    expected.hdr_width = 1920;
    expected.hdr_height = 1080;
    expected.shadow_map_size = 2048;
    expected.spot_shadow_slice_size = 512;
    assert_eq!(i, expected);
}

#[test]
fn the_sizes_pass_through() {
    let f = FrameFacts {
        hdr_sample_count: 4,
        shadowed_spot_count: 3,
        ..facts()
    };
    let i = inputs(&FeatureGates::default(), &f);
    assert_eq!(
        (i.hdr_width, i.hdr_height, i.hdr_sample_count),
        (1920, 1080, 4)
    );
    assert_eq!((i.shadow_map_size, i.shadowed_spot_count), (2048, 3));
    assert_eq!(i.spot_shadow_slice_size, 512);
}

// One fact at a time moves exactly the flags its rule names.
#[test]
fn each_fact_moves_only_its_own_passes() {
    type Set = fn(&mut FrameFacts);
    type Check = fn(&FrameGraphInputs) -> bool;
    let cases: [(&str, Set, Check); 13] = [
        ("shadow", |f| f.shadow = true, |i| i.shadow_enabled),
        ("hiz", |f| f.hiz = true, |i| i.hiz_build_enabled),
        ("fog", |f| f.fog = true, |i| i.fog_enabled),
        ("decals", |f| f.decals = true, |i| i.decals_enabled),
        ("particles", |f| f.particles = true, |i| i.particles_enabled),
        ("lines", |f| f.lines = true, |i| i.lines_enabled),
        (
            "raymarch",
            |f| f.visible_volumes = true,
            |i| i.raymarch_enabled,
        ),
        (
            "atmosphere",
            |f| f.atmosphere = true,
            |i| i.atmosphere_enabled,
        ),
        ("water sim", |f| f.water_sim = true, |i| i.water_sim_enabled),
        (
            "grass shadow",
            |f| f.grass_casts = true,
            |i| i.grass_shadow_enabled,
        ),
        (
            "clustering",
            |f| f.clustered = true,
            |i| i.clustering_enabled,
        ),
        ("hidden", |f| f.world_hidden = true, |i| i.world_hidden),
        (
            "surfaces",
            |f| f.visible_surfaces = true,
            |i| i.transparent_enabled,
        ),
    ];
    let base = inputs(&FeatureGates::default(), &facts());
    for (name, set, check) in cases {
        let mut f = facts();
        set(&mut f);
        let i = inputs(&FeatureGates::default(), &f);
        assert!(check(&i), "{name} turns its pass on");
        let mut only = i;
        only.shadow_enabled = base.shadow_enabled;
        only.hiz_build_enabled = base.hiz_build_enabled;
        only.fog_enabled = base.fog_enabled;
        only.decals_enabled = base.decals_enabled;
        only.particles_enabled = base.particles_enabled;
        only.lines_enabled = base.lines_enabled;
        only.raymarch_enabled = base.raymarch_enabled;
        only.atmosphere_enabled = base.atmosphere_enabled;
        only.water_sim_enabled = base.water_sim_enabled;
        only.grass_shadow_enabled = base.grass_shadow_enabled;
        only.clustering_enabled = base.clustering_enabled;
        only.world_hidden = base.world_hidden;
        only.transparent_enabled = base.transparent_enabled;
        assert_eq!(only, base, "{name} moves no other flag");
    }
}

#[test]
fn bloom_needs_its_chain_and_a_positive_intensity() {
    let g = FeatureGates::default();
    let with = |chain, intensity| {
        let f = FrameFacts {
            bloom_chain: chain,
            bloom_intensity: intensity,
            ..facts()
        };
        inputs(&g, &f).bloom_enabled
    };
    assert!(with(true, 0.5));
    assert!(!with(true, 0.0));
    assert!(!with(false, 0.5));
}

#[test]
fn the_cull_runs_with_a_ready_pipeline_records_and_a_visible_world() {
    let g = FeatureGates::default();
    let ready = FrameFacts {
        cull_ready: true,
        cull_records: 10,
        grass: true,
        ..facts()
    };
    let i = inputs(&g, &ready);
    assert!(i.bindless_cull_enabled && i.grass_enabled);
    for f in [
        FrameFacts {
            cull_ready: false,
            ..ready
        },
        FrameFacts {
            cull_records: 0,
            ..ready
        },
        FrameFacts {
            world_hidden: true,
            ..ready
        },
    ] {
        let i = inputs(&g, &f);
        assert!(!i.bindless_cull_enabled, "{f:?}");
        assert!(!i.grass_enabled, "grass rides the cull: {f:?}");
    }
}

#[test]
fn two_pass_occlusion_needs_its_build_and_records() {
    let g = FeatureGates::default();
    let f = FrameFacts {
        two_pass_ready: true,
        cull_records: 4,
        ..facts()
    };
    assert!(inputs(&g, &f).two_pass_occlusion_enabled);
    let none = FrameFacts {
        cull_records: 0,
        ..f
    };
    assert!(!inputs(&g, &none).two_pass_occlusion_enabled);
    let unbuilt = FrameFacts {
        two_pass_ready: false,
        ..f
    };
    assert!(!inputs(&g, &unbuilt).two_pass_occlusion_enabled);
}

// The upscaler takes the temporal slot from TAA and needs motion vectors.
#[test]
fn a_running_upscaler_replaces_taa() {
    let g = FeatureGates {
        taa: true,
        ..FeatureGates::default()
    };
    let i = inputs(&g, &facts());
    assert!(i.taa_enabled && i.velocity_enabled && !i.upscale_enabled);
    let up = FrameFacts {
        upscaler: true,
        upscaler_reads_reactive: true,
        ..facts()
    };
    let i = inputs(&g, &up);
    assert!(!i.taa_enabled && i.upscale_enabled && i.velocity_enabled);
    assert!(i.upscale_reads_reactive);
    let reads_without_running = FrameFacts {
        upscaler_reads_reactive: true,
        ..facts()
    };
    assert!(!inputs(&g, &reads_without_running).upscale_reads_reactive);
}

// SSGI accumulates through motion vectors, and like SSR and the RT pass it
// needs the screen-space pre-pass.
#[test]
fn ssgi_needs_motion_and_the_pre_pass() {
    let g = FeatureGates {
        ssgi: true,
        ssgi_resolve: true,
        ..FeatureGates::default()
    };
    let i = inputs(&g, &facts());
    assert!(i.ssgi_enabled && i.velocity_enabled && i.ssr_prepass_enabled);
    assert!(i.ssgi_resolve_enabled);
}

// A built SSGI whose trace a live change turned off keeps resolving the
// occlusion the main pass samples, over a current pre-pass.
#[test]
fn a_built_ssgi_resolves_with_its_trace_off() {
    let g = FeatureGates {
        ssgi_resolve: true,
        ..FeatureGates::default()
    };
    let i = inputs(&g, &facts());
    assert!(!i.ssgi_enabled);
    assert!(i.ssgi_resolve_enabled && i.ssr_prepass_enabled);
}

#[test]
fn the_pre_pass_has_a_consumer_for_each_screen_space_feature() {
    for g in [
        FeatureGates {
            ssr: true,
            ..FeatureGates::default()
        },
        FeatureGates {
            rt_pass: true,
            ..FeatureGates::default()
        },
        FeatureGates {
            ssgi: true,
            ..FeatureGates::default()
        },
    ] {
        assert!(inputs(&g, &facts()).ssr_prepass_enabled, "{g:?}");
    }
    let ao = FeatureGates {
        ssao: true,
        gbuffer: true,
        ..FeatureGates::default()
    };
    let i = inputs(&ao, &facts());
    assert!(i.ssao_enabled && i.gbuffer_prepass_enabled && !i.ssr_prepass_enabled);
    let ae = FeatureGates {
        auto_exposure: true,
        ..FeatureGates::default()
    };
    assert!(inputs(&ae, &facts()).auto_exposure_enabled);
}

// A live trace takes the resolve slot from authored SSR; without a BVH, SSR
// covers.
#[test]
fn the_reflection_path_follows_the_bvh() {
    let g = FeatureGates {
        ssr: true,
        rt_pass: true,
        ..FeatureGates::default()
    };
    let traced = FrameFacts {
        bvh_live: true,
        ..facts()
    };
    let i = inputs(&g, &traced);
    assert!(i.rt_reflections_enabled && !i.ssr_enabled);
    let i = inputs(&g, &facts());
    assert!(!i.rt_reflections_enabled && i.ssr_enabled);
}

// A live trace's denoiser reprojects through motion vectors; without a BVH
// nothing traces, so it asks for none.
#[test]
fn a_live_trace_needs_motion() {
    let g = FeatureGates {
        rt_pass: true,
        ..FeatureGates::default()
    };
    let traced = FrameFacts {
        bvh_live: true,
        ..facts()
    };
    assert!(inputs(&g, &traced).velocity_enabled);
    assert!(!inputs(&g, &facts()).velocity_enabled);
}

// RT alone with no BVH: a backend that reads the composite as its scene every
// frame runs the empty RT node; the others skip it.
#[test]
fn a_fixed_composite_scene_runs_the_empty_rt_node() {
    let mut g = FeatureGates {
        rt_pass: true,
        ..FeatureGates::default()
    };
    assert!(!inputs(&g, &facts()).rt_reflections_enabled);
    g.traits.composite_scene_fixed = true;
    assert!(inputs(&g, &facts()).rt_reflections_enabled);
    let traced = FrameFacts {
        bvh_live: true,
        ..facts()
    };
    assert!(inputs(&g, &traced).rt_reflections_enabled);
}

#[test]
fn the_reactive_mask_follows_the_backend_trait() {
    let mut g = FeatureGates::default();
    assert!(!inputs(&g, &facts()).reactive_mask_enabled);
    g.traits.reactive_mask = true;
    assert!(inputs(&g, &facts()).reactive_mask_enabled);
}

// See-through meshes draw in the transparent pass only while their trace runs,
// and a traced transparent pass needs no planar mirror unless water holds one.
#[test]
fn see_through_meshes_and_mirrors_follow_the_transparent_trace() {
    let g = FeatureGates {
        rt_pass: true,
        ..FeatureGates::default()
    };
    let f = FrameFacts {
        visible_see_through: true,
        rt_transparent_ready: true,
        planar_targets: true,
        ..facts()
    };
    let untraced = inputs(&g, &f);
    assert!(!untraced.transparent_enabled);
    assert!(untraced.planar_reflection_enabled);

    let traced = FrameFacts {
        bvh_live: true,
        ..f
    };
    let i = inputs(&g, &traced);
    assert!(i.transparent_enabled);
    assert!(!i.planar_reflection_enabled);
    let water = FrameFacts {
        water_planar_slot: true,
        ..traced
    };
    assert!(inputs(&g, &water).planar_reflection_enabled);
    let unready = FrameFacts {
        rt_transparent_ready: false,
        ..traced
    };
    let i = inputs(&g, &unready);
    assert!(!i.transparent_enabled && i.planar_reflection_enabled);
}
