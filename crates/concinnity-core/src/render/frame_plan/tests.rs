use super::*;
use crate::gfx::render_types::LightUniforms;
use crate::render::render_graph::{PassId, ResidentUpscaler};
use crate::transform::IDENTITY;

fn resident() -> ResidentFeatures {
    ResidentFeatures {
        hdr_width: 64,
        hdr_height: 32,
        hdr_sample_count: 1,
        shadow: true,
        shadow_map_size: 1024,
        cull: true,
        hiz: true,
        model_history: true,
        ..ResidentFeatures::default()
    }
}

fn content() -> FrameContent {
    FrameContent {
        cull_records: 4,
        ..FrameContent::default()
    }
}

fn lights() -> LightState {
    LightState::new(LightUniforms::DEFAULT, &[], 2)
}

fn inputs<'a>(lights: &'a LightState, resident: ResidentFeatures) -> PlanInputs<'a> {
    PlanInputs {
        camera: CameraInputs {
            view: IDENTITY,
            position: [1.0, 2.0, 3.0],
            fov_y_radians: 1.0,
            near: 0.1,
            view_distance: None,
        },
        elapsed: 2.5,
        render_size: [64, 32],
        resident,
        content: content(),
        upscaler_jitter: None,
        lights,
        probes: &[],
        fog: None,
        pre_exposure: 1.0,
        surface: SurfaceInputs::default(),
        effects: EffectInputs::default(),
    }
}

fn fog() -> FogSettings {
    FogSettings {
        color: [1.0; 3],
        density: 0.1,
        height_falloff: 0.1,
        height_reference: 0.0,
        max_distance: 100.0,
        phase_g: 0.0,
        ambient: 0.1,
    }
}

#[test]
fn a_still_frame_samples_pixel_centers_and_runs_the_cull() {
    let lights = lights();
    let mut planner = FramePlanner::default();
    let view = SceneView::new([0.0; 4]);
    let plan = planner.plan(&view, &inputs(&lights, resident())).unwrap();
    assert_eq!(plan.camera.vp, plan.camera.cur_vp);
    assert_eq!(plan.view.vp, plan.camera.vp);
    assert!(plan.graph.pass(PassId::Cull).is_some());
    assert_eq!(
        plan.records,
        Some(HistoryMode::Stale),
        "nothing reads motion"
    );
    assert!(!plan.motion_read);
}

#[test]
fn taa_jitters_the_raster_and_tracks_the_model_history() {
    let lights = lights();
    let mut planner = FramePlanner::default();
    let view = SceneView::new([0.0; 4]);
    let taa = ResidentFeatures {
        taa: true,
        ..resident()
    };
    let plan = planner.plan(&view, &inputs(&lights, taa)).unwrap();
    assert_ne!(plan.camera.vp, plan.camera.cur_vp);
    assert_eq!(plan.records, Some(HistoryMode::Track));
    assert!(plan.motion_read);
    planner.finish(plan);
    assert_eq!(planner.jitter_phase(), 1);
}

#[test]
fn an_upscaler_takes_its_own_jitter() {
    let lights = lights();
    let mut planner = FramePlanner::default();
    let view = SceneView::new([0.0; 4]);
    let up = ResidentFeatures {
        upscaler: Some(ResidentUpscaler::default()),
        ..resident()
    };
    let mut frame = inputs(&lights, up);
    frame.upscaler_jitter = Some([0.25, -0.125]);
    let plan = planner.plan(&view, &frame).unwrap();
    assert_eq!(plan.camera.jitter_px, [0.25, -0.125]);
}

#[test]
fn fog_is_laid_along_the_unjittered_view() {
    let lights = lights();
    let mut planner = FramePlanner::default();
    let view = SceneView::new([0.0; 4]);
    let mut frame = inputs(
        &lights,
        ResidentFeatures {
            taa: true,
            ..resident()
        },
    );
    frame.fog = Some(fog());
    let plan = planner.plan(&view, &frame).unwrap();
    let blocks = plan.fog.unwrap();
    let expected = fog().params(
        plan.camera.inv_cur_vp,
        [1.0, 2.0, 3.0],
        lights.sun().direction,
        lights.sun().color,
        [64.0, 32.0],
        1.0,
    );
    assert_eq!(blocks.params.inv_vp, expected.inv_vp);
    assert_ne!(plan.camera.inv_cur_vp, plan.camera.inv_vp);
    assert_eq!(plan.atmosphere.inv_vp, plan.camera.inv_cur_vp);
    assert_eq!(plan.atmosphere.depth_inv_vp, plan.camera.inv_vp);
}

#[test]
fn fog_that_contributes_nothing_has_no_blocks() {
    let lights = lights();
    let mut planner = FramePlanner::default();
    let view = SceneView::new([0.0; 4]);
    let mut frame = inputs(&lights, resident());
    frame.fog = Some(FogSettings {
        density: 0.0,
        ..fog()
    });
    assert!(planner.plan(&view, &frame).unwrap().fog.is_none());
}

#[test]
fn a_hidden_world_packs_no_records() {
    let lights = lights();
    let mut planner = FramePlanner::default();
    let view = SceneView::new([0.0; 4]);
    let mut frame = inputs(&lights, resident());
    frame.content.world_hidden = true;
    assert_eq!(planner.plan(&view, &frame).unwrap().records, None);
}

#[test]
fn a_finished_frame_reuses_its_graph_and_feeds_the_occlusion_test() {
    let lights = lights();
    let mut planner = FramePlanner::default();
    let view = SceneView::new([0.0; 4]);
    let plan = planner.plan(&view, &inputs(&lights, resident())).unwrap();
    let passes = plan.graph.passes.len();
    let cur_vp = plan.camera.cur_vp;
    planner.finish(plan);
    assert!(planner.temporal().occlusion().valid());
    assert_eq!(planner.temporal().occlusion().prev_view_proj(), cur_vp);
    let again = planner.plan(&view, &inputs(&lights, resident())).unwrap();
    assert_eq!(again.graph.passes.len(), passes);
}

#[test]
fn the_view_block_turns_reflections_on_only_with_a_composite() {
    let lights = lights();
    let mut planner = FramePlanner::default();
    planner.gates.ssr = true;
    let view = SceneView::new([0.0; 4]);
    let plan = planner.plan(&view, &inputs(&lights, resident())).unwrap();
    assert_eq!(plan.view.reflections_enabled, 0.0);
    let composite = ResidentFeatures {
        reflection_composite: true,
        ..resident()
    };
    let plan = planner.plan(&view, &inputs(&lights, composite)).unwrap();
    assert_eq!(plan.view.reflections_enabled, 1.0);
}

#[test]
fn the_traced_reflection_block_waits_for_a_live_bvh() {
    use crate::render::post::rt_reflections::RtReflectionSettings;
    let lights = lights();
    let mut planner = FramePlanner::default();
    let view = SceneView::new([0.0; 4]);
    let mut frame = inputs(&lights, resident());
    frame.effects.rt = Some(RtReflectionSettings::resolve(1.0, 50.0));
    assert!(planner.plan(&view, &frame).unwrap().effects.rt.is_none());
    frame.content.bvh_live = true;
    assert!(planner.plan(&view, &frame).unwrap().effects.rt.is_some());
}
