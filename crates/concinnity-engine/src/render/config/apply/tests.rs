use super::*;
use alloc::string::{String, ToString};

// A target recording what it was asked to do, in order.
#[derive(Default)]
struct Recorder {
    calls: Vec<String>,
    fail_rebuild: bool,
    rebuilt: RebuildSet,
}

impl SettingsTarget for Recorder {
    fn rebuild_features(&mut self, _: &ResolvedGraphics, features: RebuildSet) -> RenderResult<()> {
        self.calls.push("rebuild".into());
        self.rebuilt = features;
        if self.fail_rebuild {
            Err(RenderError::Other("no".into()))
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

#[test]
fn the_target_is_handed_exactly_the_planned_features() {
    let mut plan = ApplyPlan::default();
    plan.rebuild.insert(RebuildFeature::Ssgi);
    plan.rebuild.insert(RebuildFeature::GBuffer);
    let mut target = Recorder::default();
    apply_plan(&mut target, &ResolvedGraphics::default(), &plan).unwrap();
    assert_eq!(target.rebuilt, plan.rebuild);
}

fn set(features: &[RebuildFeature]) -> RebuildSet {
    let mut set = RebuildSet::default();
    for &f in features {
        set.insert(f);
    }
    set
}

#[test]
fn only_the_planned_features_rebuild_in_dependency_order() {
    use RebuildFeature::{AutoExposure, GBuffer, ReflectionComposite, RtReflections, Taa};
    let mut ran = Vec::new();
    rebuild_each(
        set(&[ReflectionComposite, Taa, RtReflections, GBuffer]),
        |f| {
            ran.push(f);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(ran, [GBuffer, Taa, RtReflections, ReflectionComposite]);
    assert!(!ran.contains(&AutoExposure));
}

#[test]
fn an_empty_set_rebuilds_nothing() {
    let mut ran = 0;
    rebuild_each(RebuildSet::default(), |_| {
        ran += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(ran, 0);
}

#[test]
fn a_failed_feature_does_not_stop_the_rest_and_its_error_comes_back() {
    use RebuildFeature::{AutoExposure, Ssao, Ssr};
    let mut ran = Vec::new();
    let result = rebuild_each(set(&[Ssao, Ssr, AutoExposure]), |f| {
        ran.push(f);
        match f {
            Ssao => Err(RenderError::Other("first".into())),
            Ssr => Err(RenderError::Other("second".into())),
            _ => Ok(()),
        }
    });
    assert_eq!(ran, [Ssao, Ssr, AutoExposure]);
    let err = result.expect_err("the failure is reported");
    assert!(err.to_string().contains("first"), "{err}");
}

// A backend with no live settings path keeps the default SettingsTarget: it
// accepts a plan of live pushes and refuses one that reallocates.
#[test]
fn a_default_settings_target_accepts_a_live_plan_and_refuses_a_rebuild() {
    struct Defaulted;
    impl SettingsTarget for Defaulted {}

    let graphics = ResolvedGraphics::default();
    let mut plan = ApplyPlan::live(LiveChanges {
        post_process: true,
        ambient: true,
        shadow_cadence: true,
        feature_params: true,
        vsync: true,
    });
    assert!(apply_plan(&mut Defaulted, &graphics, &plan).is_ok());
    plan.rebuild.insert(RebuildFeature::Ssao);
    assert!(matches!(
        apply_plan(&mut Defaulted, &graphics, &plan),
        Err(RenderError::Unsupported {
            op: "rebuild_features"
        })
    ));
}
