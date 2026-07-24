// src/scene_flow.rs
//
// Platform-agnostic active-scene state and transition logic. Scenes are pure
// content groupings; changes are imperative jumps (UI actions, Reactions), so
// this module tracks only which scene is active and drives fade transitions.
// The SceneControl trait decouples this module from any specific backend;
// callers supply a concrete backend that implements the two mutation methods.

use crate::ecs::asset_id::AssetId;

const FADE_HALF_SECS: f32 = 0.3;

pub struct SceneFlow {
    // Every Scene declared in the world, in declaration order.
    pub scenes: Vec<AssetId>,
    pub current: AssetId,
    pub fade: FadePhase,
    // Clear colour before any fade was applied (restored after fade-in).
    pub base_clear_color: [f32; 4],
}

pub enum FadePhase {
    None,
    // Fading clear_color toward black; next is the scene to activate mid-fade.
    ToBlack { started_at: f32, next: AssetId },
    // New scene is active; fading clear_color back from black.
    FromBlack { started_at: f32 },
}

// Backend operations required to drive scene visibility and fade transitions.
pub trait SceneControl {
    fn update_visibility(&mut self, draw_idx: usize, visible: bool);
    fn update_clear_color(&mut self, color: [f32; 4]);
}

// Set draw-object visibility according to which scene is currently active.
// Props with no scene association (prop_scene[i] == None) are always visible.
pub fn set_scene_visibility<B: SceneControl + ?Sized>(
    prop_draw_indices: &[Vec<usize>],
    prop_scene: &[Option<AssetId>],
    active_scene: AssetId,
    backend: &mut B,
) {
    for (prop_idx, scene_opt) in prop_scene.iter().enumerate() {
        let visible = match scene_opt {
            None => true,
            Some(s) => *s == active_scene,
        };
        if let Some(draw_idxs) = prop_draw_indices.get(prop_idx) {
            for &draw_idx in draw_idxs {
                backend.update_visibility(draw_idx, visible);
            }
        }
    }
}

// Advance any in-flight fade transition, updating the clear colour and
// switching visibility to the target scene mid-fade.
pub fn tick_transitions<B: SceneControl + ?Sized>(
    flow_opt: &mut Option<SceneFlow>,
    prop_draw_indices: &[Vec<usize>],
    prop_scene: &[Option<AssetId>],
    elapsed: f32,
    backend: &mut B,
) {
    let flow = match flow_opt {
        Some(f) => f,
        None => return,
    };

    match flow.fade {
        FadePhase::ToBlack { started_at, next } => {
            let t = ((elapsed - started_at) / FADE_HALF_SECS).clamp(0.0, 1.0);
            let [r, g, b, a] = flow.base_clear_color;
            backend.update_clear_color([r * (1.0 - t), g * (1.0 - t), b * (1.0 - t), a]);
            if t >= 1.0 {
                flow.current = next;
                flow.fade = FadePhase::FromBlack {
                    started_at: elapsed,
                };
                set_scene_visibility(prop_draw_indices, prop_scene, next, backend);
                tracing::debug!("SceneFlow: switched to scene {}", next);
            }
        }
        FadePhase::FromBlack { started_at } => {
            let t = ((elapsed - started_at) / FADE_HALF_SECS).clamp(0.0, 1.0);
            let [r, g, b, a] = flow.base_clear_color;
            backend.update_clear_color([r * t, g * t, b * t, a]);
            if t >= 1.0 {
                backend.update_clear_color(flow.base_clear_color);
                flow.fade = FadePhase::None;
            }
        }
        FadePhase::None => {}
    }
}

// Imperatively jump to a named scene. Ignored with a warning if the target
// scene is not declared, or no scenes exist.
pub fn jump_to_scene<B: SceneControl + ?Sized>(
    flow_opt: &mut Option<SceneFlow>,
    prop_draw_indices: &[Vec<usize>],
    prop_scene: &[Option<AssetId>],
    elapsed: f32,
    target_scene: AssetId,
    transition: &str,
    backend: &mut B,
) {
    let flow = match flow_opt {
        Some(f) => f,
        None => {
            tracing::warn!(
                "SceneCommand: jump to {} ignored -- no Scene assets in world",
                target_scene
            );
            return;
        }
    };

    if !flow.scenes.contains(&target_scene) {
        tracing::warn!(
            "SceneCommand: jump to {} ignored -- no Scene with that name",
            target_scene
        );
        return;
    }

    if target_scene == flow.current {
        return;
    }

    match transition {
        "FadeBlack" => {
            flow.fade = FadePhase::ToBlack {
                started_at: elapsed,
                next: target_scene,
            };
        }
        _ => {
            flow.current = target_scene;
            flow.fade = FadePhase::None;
            set_scene_visibility(prop_draw_indices, prop_scene, target_scene, backend);
            tracing::debug!("SceneCommand: cut to scene {}", target_scene);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Minimal SceneControl implementation that records every call.
    #[derive(Default)]
    struct TestBackend {
        visibility: Vec<(usize, bool)>,
        clear_colors: Vec<[f32; 4]>,
    }

    impl SceneControl for TestBackend {
        fn update_visibility(&mut self, draw_idx: usize, visible: bool) {
            self.visibility.push((draw_idx, visible));
        }
        fn update_clear_color(&mut self, color: [f32; 4]) {
            self.clear_colors.push(color);
        }
    }

    fn make_flow(scenes: &[AssetId]) -> SceneFlow {
        SceneFlow {
            scenes: scenes.to_vec(),
            current: scenes[0],
            fade: FadePhase::None,
            base_clear_color: [1.0, 1.0, 1.0, 1.0],
        }
    }

    #[test]
    fn set_visibility_active_scene_visible_others_hidden() {
        // Three props: one in "a", one with no scene, one in "b".
        let indices: Vec<Vec<usize>> = vec![vec![0], vec![1], vec![2]];
        let scenes: Vec<Option<AssetId>> = vec![Some(AssetId(0)), None, Some(AssetId(1))];
        let mut backend = TestBackend::default();
        set_scene_visibility(&indices, &scenes, AssetId(0), &mut backend);

        assert!(
            backend.visibility.contains(&(0, true)),
            "prop in 'a' should be visible"
        );
        assert!(
            backend.visibility.contains(&(1, true)),
            "scene-less prop always visible"
        );
        assert!(
            backend.visibility.contains(&(2, false)),
            "prop in 'b' should be hidden"
        );
    }

    #[test]
    fn set_visibility_no_scene_always_visible_regardless_of_active() {
        let indices: Vec<Vec<usize>> = vec![vec![0]];
        let scenes: Vec<Option<AssetId>> = vec![None];
        let mut backend = TestBackend::default();
        set_scene_visibility(&indices, &scenes, AssetId(99), &mut backend);
        assert_eq!(backend.visibility, vec![(0, true)]);
    }

    #[test]
    fn tick_without_fade_is_a_no_op() {
        let mut opt = Some(make_flow(&[AssetId(0), AssetId(1)]));
        let mut backend = TestBackend::default();
        tick_transitions(&mut opt, &[], &[], 999.0, &mut backend);
        assert!(backend.visibility.is_empty());
        assert!(backend.clear_colors.is_empty());
        assert_eq!(opt.as_ref().unwrap().current, AssetId(0));
    }

    #[test]
    fn tick_fade_to_black_darkens_clear_color() {
        let mut flow = make_flow(&[AssetId(0), AssetId(1)]);
        flow.base_clear_color = [1.0, 0.0, 0.0, 1.0];
        flow.fade = FadePhase::ToBlack {
            started_at: 0.0,
            next: AssetId(1),
        };
        let mut opt = Some(flow);
        let mut backend = TestBackend::default();
        // elapsed = FADE_HALF_SECS / 2 → t = 0.5
        tick_transitions(&mut opt, &[], &[], FADE_HALF_SECS * 0.5, &mut backend);
        assert_eq!(backend.clear_colors.len(), 1);
        let [r, _g, _b, a] = backend.clear_colors[0];
        assert!((r - 0.5).abs() < 1e-5, "red should be half-dimmed");
        assert!((a - 1.0).abs() < 1e-5, "alpha unchanged");
        // Still in ToBlack, no scene switch yet.
        assert!(matches!(
            opt.as_ref().unwrap().fade,
            FadePhase::ToBlack { .. }
        ));
    }

    #[test]
    fn tick_fade_to_black_completes_and_enters_from_black() {
        let mut flow = make_flow(&[AssetId(0), AssetId(1)]);
        flow.fade = FadePhase::ToBlack {
            started_at: 0.0,
            next: AssetId(1),
        };
        let mut opt = Some(flow);
        let mut backend = TestBackend::default();
        // elapsed = FADE_HALF_SECS → t = 1.0, scene switches
        tick_transitions(&mut opt, &[], &[], FADE_HALF_SECS, &mut backend);
        let f = opt.as_ref().unwrap();
        assert_eq!(f.current, AssetId(1));
        assert!(matches!(f.fade, FadePhase::FromBlack { .. }));
    }

    #[test]
    fn tick_fade_from_black_restores_clear_color() {
        let mut flow = make_flow(&[AssetId(0)]);
        flow.base_clear_color = [1.0, 1.0, 1.0, 1.0];
        flow.fade = FadePhase::FromBlack { started_at: 0.0 };
        let mut opt = Some(flow);
        let mut backend = TestBackend::default();
        // elapsed = FADE_HALF_SECS → t = 1.0, fade ends
        tick_transitions(&mut opt, &[], &[], FADE_HALF_SECS, &mut backend);
        assert!(matches!(opt.as_ref().unwrap().fade, FadePhase::None));
        // The final clear_color call restores the base color.
        assert_eq!(*backend.clear_colors.last().unwrap(), [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn jump_to_scene_no_flow_is_no_op() {
        let mut opt: Option<SceneFlow> = None;
        let mut backend = TestBackend::default();
        jump_to_scene(&mut opt, &[], &[], 0.0, AssetId(99), "Cut", &mut backend);
        assert!(backend.visibility.is_empty());
    }

    #[test]
    fn jump_to_unknown_scene_is_no_op() {
        let mut opt = Some(make_flow(&[AssetId(0), AssetId(1)]));
        let mut backend = TestBackend::default();
        jump_to_scene(&mut opt, &[], &[], 0.0, AssetId(99), "Cut", &mut backend);
        assert_eq!(opt.as_ref().unwrap().current, AssetId(0));
        assert!(backend.visibility.is_empty());
    }

    #[test]
    fn jump_to_scene_same_scene_is_no_op() {
        let mut opt = Some(make_flow(&[AssetId(0), AssetId(1)]));
        let mut backend = TestBackend::default();
        jump_to_scene(&mut opt, &[], &[], 0.0, AssetId(0), "Cut", &mut backend);
        assert_eq!(opt.as_ref().unwrap().current, AssetId(0));
        assert!(backend.visibility.is_empty());
    }

    #[test]
    fn jump_to_scene_cut_switches_immediately() {
        let indices: Vec<Vec<usize>> = vec![vec![0], vec![1]];
        let scenes: Vec<Option<AssetId>> = vec![Some(AssetId(0)), Some(AssetId(1))];
        let mut opt = Some(make_flow(&[AssetId(0), AssetId(1)]));
        let mut backend = TestBackend::default();
        jump_to_scene(
            &mut opt,
            &indices,
            &scenes,
            1.0,
            AssetId(1),
            "Cut",
            &mut backend,
        );
        assert_eq!(opt.as_ref().unwrap().current, AssetId(1));
        assert!(matches!(opt.as_ref().unwrap().fade, FadePhase::None));
        assert!(backend.visibility.contains(&(1, true)));
    }

    #[test]
    fn jump_to_scene_fade_black_starts_to_black_phase() {
        let mut opt = Some(make_flow(&[AssetId(0), AssetId(1)]));
        let mut backend = TestBackend::default();
        jump_to_scene(
            &mut opt,
            &[],
            &[],
            5.0,
            AssetId(1),
            "FadeBlack",
            &mut backend,
        );
        // current not changed yet; scene switches mid-fade
        assert_eq!(opt.as_ref().unwrap().current, AssetId(0));
        assert!(matches!(
            opt.as_ref().unwrap().fade,
            FadePhase::ToBlack { started_at, next } if (started_at - 5.0).abs() < 1e-6 && next == AssetId(1)
        ));
    }
}
