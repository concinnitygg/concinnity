use super::*;
use crate::components::ShadowUpdate;

fn profile_with_tier(tier: GpuTier) -> GpuProfile {
    GpuProfile {
        tier,
        ..GpuProfile::UNKNOWN
    }
}

#[test]
fn only_custom_imposes_no_ceiling() {
    // Custom never clamps, regardless of hardware.
    assert_eq!(
        resolve_ceiling(
            QualityPreset::Custom,
            &profile_with_tier(GpuTier::Integrated)
        ),
        QualityCeiling::NONE
    );
    // Auto on an unclassified GPU takes the mid tier rather than handing it
    // the whole stack the component defaults author.
    assert_eq!(
        resolve_ceiling(QualityPreset::Auto, &profile_with_tier(GpuTier::Unknown)),
        QualityCeiling::MEDIUM
    );
}

#[test]
fn auto_maps_tier_to_named_ceiling() {
    assert_eq!(
        resolve_ceiling(QualityPreset::Auto, &profile_with_tier(GpuTier::Integrated)),
        QualityCeiling::LOW
    );
    assert_eq!(
        resolve_ceiling(
            QualityPreset::Auto,
            &profile_with_tier(GpuTier::EntryDiscrete)
        ),
        QualityCeiling::MEDIUM
    );
    assert_eq!(
        resolve_ceiling(
            QualityPreset::Auto,
            &profile_with_tier(GpuTier::MidDiscrete)
        ),
        QualityCeiling::HIGH
    );
    assert_eq!(
        resolve_ceiling(
            QualityPreset::Auto,
            &profile_with_tier(GpuTier::HighDiscrete)
        ),
        QualityCeiling::ULTRA
    );
}

#[test]
fn named_presets_resolve_independent_of_hardware() {
    // A named preset ignores the GPU tier.
    let weak = profile_with_tier(GpuTier::Integrated);
    assert_eq!(
        resolve_ceiling(QualityPreset::Low, &weak),
        QualityCeiling::LOW
    );
    assert_eq!(
        resolve_ceiling(QualityPreset::Ultra, &weak),
        QualityCeiling::ULTRA
    );
}

#[test]
fn ceilings_are_monotonic_in_tier() {
    // Each tier permits a superset of the next-lower tier's features, so a
    // higher tier never disables something a lower tier allows.
    let order = [
        QualityCeiling::LOW,
        QualityCeiling::MEDIUM,
        QualityCeiling::HIGH,
        QualityCeiling::ULTRA,
    ];
    for pair in order.windows(2) {
        let (lo, hi) = (pair[0], pair[1]);
        for (lo_on, hi_on) in [
            (lo.ssao, hi.ssao),
            (lo.ssr, hi.ssr),
            (lo.ray_traced_reflections, hi.ray_traced_reflections),
            (lo.rt_reflection_shadows, hi.rt_reflection_shadows),
            (lo.ssgi, hi.ssgi),
            (lo.auto_exposure, hi.auto_exposure),
        ] {
            assert!(!lo_on || hi_on, "a higher tier dropped a feature");
        }
        // The AA-mode cap rises (or holds) with the tier: a higher tier
        // never permits a less aggressive (cheaper) anti-aliasing mode.
        assert!(
            crate::settings::levels::aa_mode_index(lo.aa_mode)
                <= crate::settings::levels::aa_mode_index(hi.aa_mode),
            "a higher tier capped AA at a cheaper mode"
        );
        // And never forces more aggressive upscaling than a lower tier.
        assert_eq!(
            more_aggressive_upscale(lo.min_upscale, hi.min_upscale),
            lo.min_upscale
        );
        // The SSGI sub-quality caps rise (or hold) with the tier too: a
        // higher tier never permits fewer rays / steps or a coarser gather.
        assert!(lo.ssgi_rays <= hi.ssgi_rays, "ssgi_rays cap dropped");
        assert!(
            lo.ssgi_resolution >= hi.ssgi_resolution,
            "a higher tier permitted a coarser SSGI trace"
        );
        assert!(
            lo.reflection_blur_resolution >= hi.reflection_blur_resolution,
            "a higher tier permitted a coarser reflection blur"
        );
        assert!(
            lo.rt_reflection_resolution >= hi.rt_reflection_resolution,
            "a higher tier permitted a coarser ray-traced reflection trace"
        );
        // The shadow caps rise (or hold) with the tier: a higher tier never
        // permits a smaller shadow map or forbids a cadence a lower tier
        // allowed.
        assert!(
            lo.shadow_map_size <= hi.shadow_map_size,
            "shadow_map_size cap dropped"
        );
        assert!(
            !lo.allow_every_frame_shadows || hi.allow_every_frame_shadows,
            "a higher tier forbade the EveryFrame shadow cadence"
        );
        // The anisotropy cap rises (or holds) with the tier too.
        assert!(lo.anisotropy <= hi.anisotropy, "anisotropy cap dropped");
        // The shadow-distance cap rises (or holds) with the tier too.
        assert!(
            lo.shadow_distance <= hi.shadow_distance,
            "shadow_distance cap dropped"
        );
        // The shadow cascade-count cap rises (or holds) with the tier too.
        assert!(
            lo.shadow_cascades <= hi.shadow_cascades,
            "shadow_cascades cap dropped"
        );
        // The planar reflection plane budget rises (or holds) with the tier too.
        assert!(
            lo.planar_reflection_planes <= hi.planar_reflection_planes,
            "planar plane budget cap dropped"
        );
        assert!(
            lo.planar_reflection_resolution >= hi.planar_reflection_resolution,
            "a higher tier permitted a coarser planar mirror"
        );
        // The ocean simulation is never coarser or narrower a tier up.
        assert!(lo.ocean.size <= hi.ocean.size, "ocean size dropped");
        assert!(
            lo.ocean.cascades <= hi.ocean.cascades,
            "ocean cascades dropped"
        );
        assert!(
            lo.min_planet_cell_size >= hi.min_planet_cell_size,
            "a higher tier forced coarser planet terrain"
        );
    }
}

// Planet terrain cells coarsen tier by tier, and the top tier forces nothing
// over the section default.
#[test]
fn planet_cells_coarsen_down_the_tiers() {
    let floors = [
        QualityPreset::Low,
        QualityPreset::Medium,
        QualityPreset::High,
        QualityPreset::Ultra,
    ]
    .map(|p| resolve_ceiling(p, &GpuProfile::UNKNOWN).min_planet_cell_size);
    assert_eq!(floors, [4, 2, 1, 1]);
    assert_eq!(
        floors[3],
        crate::components::PlanetDetail::default().finest_cell_size
    );
    assert_eq!(QualityCeiling::NONE.min_planet_cell_size, 0);
}

// The top tiers run the full simulation; the lowest runs two cascades of
// 128, which the simulation supports as is.
#[test]
fn the_ocean_simulation_scales_with_tier() {
    let ultra = resolve_ceiling(QualityPreset::Ultra, &GpuProfile::UNKNOWN);
    assert_eq!(ultra.ocean, OceanQuality::FULL);
    let custom = resolve_ceiling(QualityPreset::Custom, &GpuProfile::UNKNOWN);
    assert_eq!(custom.ocean, OceanQuality::FULL);
    let low = resolve_ceiling(QualityPreset::Low, &GpuProfile::UNKNOWN);
    assert_eq!(
        low.ocean,
        OceanQuality {
            size: 128,
            cascades: 2
        }
    );
    for preset in QualityPreset::ALL {
        let c = resolve_ceiling(preset, &GpuProfile::UNKNOWN);
        assert_eq!(c.ocean.clamped(), c.ocean, "{preset:?}");
    }
}

// Callers clamp with `authored.min(ceiling.field)`, so what each tier
// actually promises is the cap value itself.
#[test]
fn tier_caps_bound_the_shadow_and_texture_knobs() {
    // No ceiling (Custom / Ultra) leaves a world's authored values alone.
    let none = resolve_ceiling(QualityPreset::Custom, &GpuProfile::UNKNOWN);
    assert_eq!(none.shadow_map_size, u32::MAX);
    assert_eq!(none.anisotropy, ANISO_MAX);
    assert_eq!(none.shadow_distance, u32::MAX);
    assert_eq!(none.shadow_cascades, SHADOW_CASCADES_MAX);
    assert_eq!(
        clamp_shadow_update(ShadowUpdate::EveryFrame, &none),
        ShadowUpdate::EveryFrame
    );

    // Low caps every knob hard and forces the cheaper Hybrid cadence.
    let low = resolve_ceiling(QualityPreset::Low, &GpuProfile::UNKNOWN);
    assert_eq!(low.shadow_map_size, 1024);
    assert_eq!(low.anisotropy, 4);
    assert_eq!(low.shadow_distance, 40);
    assert_eq!(low.shadow_cascades, 2);
    assert_eq!(
        clamp_shadow_update(ShadowUpdate::EveryFrame, &low),
        ShadowUpdate::Hybrid
    );
    // The cadence clamp never raises Hybrid to EveryFrame.
    assert_eq!(
        clamp_shadow_update(ShadowUpdate::Hybrid, &none),
        ShadowUpdate::Hybrid
    );

    let medium = resolve_ceiling(QualityPreset::Medium, &GpuProfile::UNKNOWN);
    assert_eq!(medium.shadow_cascades, 3);
}

#[test]
fn planar_plane_budget_scales_with_tier_within_capacity() {
    // No ceiling (Custom / Ultra) permits the full engine capacity, matching
    // the flat pre-scaling default so capable GPUs are never downgraded.
    let none = resolve_ceiling(QualityPreset::Custom, &GpuProfile::UNKNOWN);
    assert_eq!(none.planar_reflection_planes, PLANAR_PLANES_MAX);
    let ultra = resolve_ceiling(QualityPreset::Ultra, &GpuProfile::UNKNOWN);
    assert_eq!(ultra.planar_reflection_planes, PLANAR_PLANES_MAX);
    // High (mid discrete under Auto) also runs the full budget.
    let high = resolve_ceiling(QualityPreset::High, &GpuProfile::UNKNOWN);
    assert_eq!(high.planar_reflection_planes, PLANAR_PLANES_MAX);
    // Lower tiers reduce the budget but keep at least one sharp mirror plane,
    // and never exceed the capacity every backend's allocation is sized to.
    for preset in [QualityPreset::Low, QualityPreset::Medium] {
        let c = resolve_ceiling(preset, &GpuProfile::UNKNOWN);
        assert!(
            c.planar_reflection_planes >= 1,
            "{preset:?} dropped all planes"
        );
        assert!(
            c.planar_reflection_planes < PLANAR_PLANES_MAX,
            "{preset:?} did not reduce the budget"
        );
    }
}

#[test]
fn planar_mirrors_render_at_full_resolution_from_high_up() {
    for preset in [
        QualityPreset::Custom,
        QualityPreset::Ultra,
        QualityPreset::High,
    ] {
        let c = resolve_ceiling(preset, &GpuProfile::UNKNOWN);
        assert_eq!(
            c.planar_reflection_resolution,
            PassResolution::Full,
            "{preset:?}"
        );
    }
    for preset in [QualityPreset::Medium, QualityPreset::Low] {
        let c = resolve_ceiling(preset, &GpuProfile::UNKNOWN);
        assert_eq!(
            c.planar_reflection_resolution,
            PassResolution::Half,
            "{preset:?}"
        );
    }
}

#[test]
fn ssgi_caps_clamp_down_only() {
    // The no-ceiling values are the engine maxima, so any authored value
    // stands under them.
    assert_eq!(QualityCeiling::NONE.ssgi_rays, 4);
    assert_eq!(QualityCeiling::NONE.ssgi_resolution, PassResolution::Full);
    // The coarser resolution is the higher divisor (lower quality), and an
    // equal input is returned as-is.
    assert_eq!(
        PassResolution::Full.max(PassResolution::Quarter),
        PassResolution::Quarter
    );
    assert_eq!(
        PassResolution::Half.max(PassResolution::Half),
        PassResolution::Half
    );
    // Ultra allows a second ray; Low caps hard.
    let ultra = resolve_ceiling(QualityPreset::Ultra, &GpuProfile::UNKNOWN);
    assert_eq!(ultra.ssgi_rays, 2);
    assert_eq!(ultra.ssgi_resolution, PassResolution::Full);
    let low = resolve_ceiling(QualityPreset::Low, &GpuProfile::UNKNOWN);
    assert_eq!(low.ssgi_rays, 1);
    assert_eq!(low.ssgi_resolution, PassResolution::Quarter);
}

#[test]
fn low_disables_the_expensive_effects() {
    // Resolve through the public path so the assertions are on a runtime
    // value, not the `const QualityCeiling::LOW` directly.
    let low = resolve_ceiling(QualityPreset::Low, &GpuProfile::UNKNOWN);
    assert!(!low.ssr);
    assert!(!low.ssgi);
    assert!(!low.ray_traced_reflections);
    assert!(!low.ssao);
    // Low caps anti-aliasing at FXAA: it keeps edges smooth nearly for free
    // but skips TAA's velocity pre-pass and history buffer. Auto-exposure is
    // cheap enough to keep on.
    assert_eq!(low.aa_mode, AaMode::Fxaa);
    assert!(low.auto_exposure);
}

#[test]
fn aa_mode_clamps_down_only() {
    // The no-ceiling cap is TAA (the maximum), so any authored mode stands.
    assert_eq!(QualityCeiling::NONE.aa_mode, AaMode::Taa);
    assert_eq!(
        clamp_aa_mode(AaMode::Taa, QualityCeiling::NONE.aa_mode),
        AaMode::Taa
    );
    // A lower cap forces a more expensive authored mode down to it, but
    // never raises a cheaper authored mode up.
    assert_eq!(clamp_aa_mode(AaMode::Taa, AaMode::Fxaa), AaMode::Fxaa);
    assert_eq!(clamp_aa_mode(AaMode::Fxaa, AaMode::Taa), AaMode::Fxaa);
    assert_eq!(clamp_aa_mode(AaMode::Fxaa, AaMode::Off), AaMode::Off);
    assert_eq!(clamp_aa_mode(AaMode::Off, AaMode::Taa), AaMode::Off);
    // Ultra imposes the maximum (no clamp); Low caps at FXAA.
    let ultra = resolve_ceiling(QualityPreset::Ultra, &GpuProfile::UNKNOWN);
    assert_eq!(ultra.aa_mode, AaMode::Taa);
    let low = resolve_ceiling(QualityPreset::Low, &GpuProfile::UNKNOWN);
    assert_eq!(low.aa_mode, AaMode::Fxaa);
}

#[test]
fn preset_index_and_at_round_trip() {
    for p in QualityPreset::ALL {
        assert_eq!(preset_at(preset_index(p)), p);
    }
    // Auto leads the cycle order; an out-of-range index falls back to Auto.
    assert_eq!(preset_at(0), QualityPreset::Auto);
    assert_eq!(preset_at(99), QualityPreset::Auto);
}

#[test]
fn auto_label_annotates_the_resolved_tier() {
    // Auto shows the tier it resolved to, so the user sees the auto choice.
    assert_eq!(
        preset_label(
            QualityPreset::Auto,
            &profile_with_tier(GpuTier::MidDiscrete)
        ),
        "Auto (High)"
    );
    assert_eq!(
        preset_label(QualityPreset::Auto, &profile_with_tier(GpuTier::Integrated)),
        "Auto (Low)"
    );
    // An unclassified GPU resolves to the mid tier, and says so.
    assert_eq!(
        preset_label(QualityPreset::Auto, &profile_with_tier(GpuTier::Unknown)),
        "Auto (Medium)"
    );
    // A named preset is just its own name, hardware-independent.
    assert_eq!(
        preset_label(
            QualityPreset::Ultra,
            &profile_with_tier(GpuTier::Integrated)
        ),
        "Ultra"
    );
    // The Auto suffix tracks the resolved ceiling.
    for tier in [
        GpuTier::Integrated,
        GpuTier::EntryDiscrete,
        GpuTier::MidDiscrete,
        GpuTier::HighDiscrete,
    ] {
        let profile = profile_with_tier(tier);
        let suffix = auto_tier(&profile).0;
        assert_eq!(
            preset_label(QualityPreset::Auto, &profile),
            format!("Auto ({suffix})")
        );
    }
}

#[test]
fn more_aggressive_picks_the_lower_resolution() {
    // Higher index = more aggressive (lower internal resolution).
    assert_eq!(
        more_aggressive_upscale(UpscaleQuality::Quality, UpscaleQuality::Performance),
        UpscaleQuality::Performance
    );
    assert_eq!(
        more_aggressive_upscale(UpscaleQuality::UltraPerformance, UpscaleQuality::Balanced),
        UpscaleQuality::UltraPerformance
    );
    // Equal inputs return that quality; a ceiling of Quality never raises.
    assert_eq!(
        more_aggressive_upscale(UpscaleQuality::Balanced, UpscaleQuality::Quality),
        UpscaleQuality::Balanced
    );
}
