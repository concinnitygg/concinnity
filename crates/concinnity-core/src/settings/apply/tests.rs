use super::*;
use crate::components::{
    AaMode, AmbientOcclusion, AntiAliasing, Bloom, ColorLut, Culling, Display, Exposure,
    FramePacing, IndirectLighting, IndirectTechnique, PassResolution, Reflections, Shadows,
    TextureFiltering, Vignette,
};
use crate::ecs::asset_fields::AssetFields;
use crate::settings::{
    QualityCeiling, ResolvedSettings, SettingsLayers, UserGraphicsSettings, WorldSettings,
};
use alloc::string::String;

fn resolved() -> ResolvedSettings {
    SettingsLayers::new(
        &WorldSettings::default(),
        &QualityCeiling::NONE,
        &UserGraphicsSettings::default(),
    )
    .resolve()
}

// The plan for moving from the defaults to `edit` applied to them.
fn plan_for(edit: impl FnOnce(&mut ResolvedGraphics)) -> ApplyPlan {
    let before = resolved();
    let mut after = before.clone();
    edit(&mut after.graphics);
    plan_apply(&after.diff(&before), &before.graphics, &after.graphics)
}

// A named edit to the resolved settings and what it should plan.
type Case<T> = (&'static str, fn(&mut ResolvedGraphics), T);

fn rebuilds(plan: &ApplyPlan) -> Vec<RebuildFeature> {
    plan.rebuild.iter().collect()
}

#[test]
fn an_unchanged_snapshot_plans_nothing() {
    let plan = plan_for(|_| {});
    assert_eq!(plan, ApplyPlan::default());
    assert!(!plan.touches_renderer());
}

// Each `rebuild` field of every section reallocates a feature, never only
// waits for a relaunch.
#[test]
fn every_rebuild_field_names_a_feature() {
    fn check<A: AssetFields>(asset: &'static str) {
        for (key, meta) in A::PARAMS {
            if meta.apply != Some(ApplyClass::Rebuild) {
                continue;
            }
            let mut plan = ApplyPlan::default();
            rebuild_change(
                &mut plan,
                &ParamChange {
                    asset,
                    key,
                    apply: ApplyClass::Rebuild,
                },
            );
            assert!(!plan.rebuild.is_empty(), "{asset}.{key}");
            assert!(plan.restart.is_empty(), "{asset}.{key}");
        }
    }
    check::<AmbientOcclusion>("AmbientOcclusion");
    check::<Reflections>("Reflections");
    check::<IndirectLighting>("IndirectLighting");
    check::<AntiAliasing>("AntiAliasing");
    check::<Exposure>("Exposure");
    check::<Shadows>("Shadows");
    check::<Bloom>("Bloom");
    check::<Vignette>("Vignette");
    check::<Display>("Display");
    check::<Culling>("Culling");
    check::<crate::components::PlanetDetail>("PlanetDetail");
    check::<FramePacing>("FramePacing");
    check::<TextureFiltering>("TextureFiltering");
    check::<ColorLut>("ColorLut");
}

#[test]
fn each_feature_toggle_rebuilds_its_own_feature() {
    use RebuildFeature::*;
    let cases: [Case<&[RebuildFeature]>; 7] = [
        ("ssao", |g| g.ambient_occlusion.enabled = false, &[Ssao]),
        (
            "ssr",
            |g| g.reflections.screen_space = false,
            &[Ssr, ReflectionComposite],
        ),
        (
            "rt",
            |g| g.reflections.ray_traced = false,
            &[RtReflections, ReflectionComposite],
        ),
        (
            "rt resolution",
            |g| g.reflections.ray_traced_resolution = PassResolution::Full,
            &[RtReflections],
        ),
        (
            "blur",
            |g| g.reflections.blur_resolution = PassResolution::Quarter,
            &[ReflectionComposite],
        ),
        ("ssgi rays", |g| g.indirect_lighting.ssgi_rays = 4, &[Ssgi]),
        ("auto exposure", |g| g.exposure.auto = true, &[AutoExposure]),
    ];
    for (name, edit, expected) in cases {
        let plan = plan_for(edit);
        assert_eq!(rebuilds(&plan), expected, "{name}");
        assert!(plan.restart.is_empty(), "{name}");
    }
}

// The G-buffer joins the rebuild exactly when the change moves whether any
// screen-space consumer is on.
#[test]
fn the_gbuffer_rebuilds_when_its_need_changes() {
    let none_left = plan_for(|g| {
        g.ambient_occlusion.enabled = false;
        g.reflections.screen_space = false;
        g.reflections.ray_traced = false;
        g.indirect_lighting.technique = IndirectTechnique::Ibl;
        g.anti_aliasing.mode = AaMode::Fxaa;
    });
    assert!(none_left.rebuild.contains(RebuildFeature::GBuffer));
    let one_of_many = plan_for(|g| g.ambient_occlusion.enabled = false);
    assert!(!one_of_many.rebuild.contains(RebuildFeature::GBuffer));
}

// The AA mode also moves the composite's FXAA flag, and turning auto-exposure
// off hands exposure back to the static EV, so both push the composite too.
#[test]
fn rebuilds_that_move_the_composite_push_it() {
    let aa = plan_for(|g| g.anti_aliasing.mode = AaMode::Off);
    assert_eq!(rebuilds(&aa), [RebuildFeature::Taa]);
    assert!(aa.live.post_process);
    let ae = plan_for(|g| g.exposure.auto = true);
    assert!(ae.live.post_process);
}

#[test]
fn live_changes_never_rebuild() {
    let cases: [Case<LiveChanges>; 6] = [
        (
            "bloom",
            |g| g.bloom.intensity = 0.0,
            LiveChanges {
                post_process: true,
                ..LiveChanges::default()
            },
        ),
        (
            "lut",
            |g| g.lut_strength = 0.5,
            LiveChanges {
                post_process: true,
                ..LiveChanges::default()
            },
        ),
        (
            "ambient",
            |g| g.indirect_lighting.ambient_intensity = 2.0,
            LiveChanges {
                ambient: true,
                ..LiveChanges::default()
            },
        ),
        (
            "cascades",
            |g| g.shadows.cascades = 2,
            LiveChanges {
                shadow_cadence: true,
                ..LiveChanges::default()
            },
        ),
        (
            "ssao radius",
            |g| g.ambient_occlusion.radius = 2.0,
            LiveChanges {
                feature_params: true,
                ..LiveChanges::default()
            },
        ),
        (
            "vsync",
            |g| g.frame_pacing.vsync = true,
            LiveChanges {
                vsync: true,
                ..LiveChanges::default()
            },
        ),
    ];
    for (name, edit, expected) in cases {
        let plan = plan_for(edit);
        assert_eq!(plan.live, expected, "{name}");
        assert!(plan.rebuild.is_empty() && plan.restart.is_empty(), "{name}");
    }
}

#[test]
fn restart_changes_never_reach_the_renderer() {
    let plan = plan_for(|g| {
        g.shadows.map_size = 1024;
        g.display.hdr = true;
        g.frame_pacing.frames_in_flight = 3;
        g.anti_aliasing.upscaling = true;
    });
    assert!(!plan.touches_renderer());
    let keys: Vec<(&str, &str)> = plan.restart.iter().map(|c| (c.asset, c.key)).collect();
    assert_eq!(
        keys,
        [
            ("AntiAliasing", "upscaling"),
            ("Display", "hdr"),
            ("Shadows", "map_size"),
            ("FramePacing", "frames_in_flight"),
        ]
    );
}

// The frame cap and the stats overlay are the engine's alone.
#[test]
fn engine_only_values_plan_nothing_for_the_renderer() {
    let before = resolved();
    let mut after = before.clone();
    after.graphics.frame_pacing.fps_cap = 30;
    after.show_fps = !after.show_fps;
    let plan = plan_apply(&after.diff(&before), &before.graphics, &after.graphics);
    assert!(!plan.touches_renderer());
    assert!(plan.restart.is_empty());
}

// A target recording what it was asked to do, in order.
#[derive(Default)]
struct Recorder {
    calls: Vec<String>,
    fail_rebuild: bool,
}

impl SettingsTarget for Recorder {
    fn rebuild_features(&mut self, _: &ResolvedGraphics) -> RenderResult<()> {
        self.calls.push("rebuild".into());
        if self.fail_rebuild {
            Err(crate::render::error::RenderError::Other("no".into()))
        } else {
            Ok(())
        }
    }
    fn push_feature_params(&mut self, _: &ResolvedGraphics) {
        self.calls.push("params".into());
    }
    fn push_post_process(&mut self, _: PostProcessTunables) {
        self.calls.push("post".into());
    }
    fn push_ambient(&mut self, _: f32) {
        self.calls.push("ambient".into());
    }
    fn push_shadow_cadence(&mut self, _: ShadowCadence) {
        self.calls.push("cadence".into());
    }
    fn push_vsync(&mut self, _: bool) {
        self.calls.push("vsync".into());
    }
}

#[test]
fn the_rebuild_runs_before_every_live_push() {
    let mut plan = ApplyPlan::live(LiveChanges {
        post_process: true,
        ambient: true,
        shadow_cadence: true,
        feature_params: true,
        vsync: true,
    });
    plan.rebuild.insert(RebuildFeature::Ssao);
    let mut target = Recorder::default();
    apply_plan(&mut target, &ResolvedGraphics::default(), &plan).unwrap();
    assert_eq!(
        target.calls,
        ["rebuild", "params", "post", "ambient", "cadence", "vsync"]
    );
}

#[test]
fn a_live_only_plan_runs_only_its_pushes() {
    let plan = ApplyPlan::live(LiveChanges {
        ambient: true,
        ..LiveChanges::default()
    });
    let mut target = Recorder::default();
    apply_plan(&mut target, &ResolvedGraphics::default(), &plan).unwrap();
    assert_eq!(target.calls, ["ambient"]);
}

// A failed rebuild still lets the live values land, and reports the failure.
#[test]
fn a_failed_rebuild_still_pushes_the_live_values() {
    let mut plan = ApplyPlan::live(LiveChanges {
        post_process: true,
        ..LiveChanges::default()
    });
    plan.rebuild.insert(RebuildFeature::AutoExposure);
    let mut target = Recorder {
        fail_rebuild: true,
        ..Recorder::default()
    };
    assert!(apply_plan(&mut target, &ResolvedGraphics::default(), &plan).is_err());
    assert_eq!(target.calls, ["rebuild", "params", "post"]);
}
