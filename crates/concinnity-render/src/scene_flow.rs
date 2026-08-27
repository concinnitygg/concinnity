//! Platform-agnostic active-scene state and transition logic. Scenes are pure
//! content groupings; changes are imperative jumps (UI actions, Behaviors), so
//! this module tracks only which scene is active and drives fade transitions.
//! The SceneControl trait decouples this module from any specific backend;
//! callers supply a concrete backend that implements the two mutation methods.

use crate::ecs::asset_id::AssetId;
use alloc::vec::Vec;

const FADE_HALF_SECS: f32 = 0.3;

/// The active scene and any transition in flight.
pub struct SceneFlow {
    /// Every Scene declared in the world, in declaration order.
    pub scenes: Vec<AssetId>,
    /// The scene currently active.
    pub current: AssetId,
    /// Where the transition fade stands.
    pub fade: FadePhase,
}

/// A scene transition's fade phase.
pub enum FadePhase {
    /// No transition in flight.
    None,
    /// Fading the composited image toward black; next is the scene to activate
    /// mid-fade.
    ToBlack {
        /// When the fade-out started, in world seconds.
        started_at: f32,
        /// The scene being faded to.
        next: AssetId,
    },
    /// New scene is active; fading the composited image back from black.
    FromBlack {
        /// When the fade-in started, in world seconds.
        started_at: f32,
    },
}

/// Backend operations required to drive scene visibility and fade transitions.
pub trait SceneControl {
    /// Show or hide one draw slot.
    fn update_visibility(&mut self, draw_idx: usize, visible: bool);
    /// Fade the composited image to black by `fade` in `[0, 1]`: 0 leaves the
    /// frame untouched, 1 renders it fully black. Applied in the composite pass
    /// so the whole image fades, not just the pixels no geometry covers.
    fn set_fade(&mut self, fade: f32);
}

/// Per-prop scene visibility pairs, flattened so a per-frame refresh during a
/// fade reuses its buffers instead of allocating a slot list per prop: prop
/// `i`'s draw slots are the `spans[i]` range of `draws`, its scene is
/// `scenes[i]` (`None` = always visible).
#[derive(Default)]
pub struct SceneVisibility {
    draws: Vec<usize>,
    spans: Vec<(u32, u32)>,
    scenes: Vec<Option<AssetId>>,
}

impl SceneVisibility {
    /// Forget every prop, retaining the buffers for the next refresh.
    pub fn clear(&mut self) {
        self.draws.clear();
        self.spans.clear();
        self.scenes.clear();
    }

    /// Start the next prop's entry; its draw slots follow via
    /// [`SceneVisibility::push_draw`].
    pub fn begin_prop(&mut self, scene: Option<AssetId>) {
        self.spans.push((self.draws.len() as u32, 0));
        self.scenes.push(scene);
    }

    /// Add one draw slot to the prop most recently begun.
    pub fn push_draw(&mut self, draw_idx: usize) {
        debug_assert!(!self.spans.is_empty(), "push_draw before begin_prop");
        self.draws.push(draw_idx);
        if let Some(span) = self.spans.last_mut() {
            span.1 += 1;
        }
    }

    /// Every prop's `(draw slots, scene)`, in insertion order.
    pub fn props(&self) -> impl Iterator<Item = (&[usize], Option<AssetId>)> + '_ {
        self.spans
            .iter()
            .zip(self.scenes.iter())
            .map(|(&(start, len), scene)| {
                (&self.draws[start as usize..(start + len) as usize], *scene)
            })
    }
}

/// Set draw-object visibility according to which scene is currently active.
/// Props with no scene association (scene == None) are always visible.
pub fn set_scene_visibility<B: SceneControl + ?Sized>(
    visibility: &SceneVisibility,
    active_scene: AssetId,
    backend: &mut B,
) {
    for (draw_idxs, scene_opt) in visibility.props() {
        let visible = match scene_opt {
            None => true,
            Some(s) => s == active_scene,
        };
        for &draw_idx in draw_idxs {
            backend.update_visibility(draw_idx, visible);
        }
    }
}

/// Advance any in-flight fade transition, updating the composite fade and
/// switching visibility to the target scene mid-fade.
pub fn tick_transitions<B: SceneControl + ?Sized>(
    flow_opt: &mut Option<SceneFlow>,
    visibility: &SceneVisibility,
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
            backend.set_fade(t);
            if t >= 1.0 {
                flow.current = next;
                flow.fade = FadePhase::FromBlack {
                    started_at: elapsed,
                };
                set_scene_visibility(visibility, next, backend);
                tracing::debug!("SceneFlow: switched to scene {}", next);
            }
        }
        FadePhase::FromBlack { started_at } => {
            let t = ((elapsed - started_at) / FADE_HALF_SECS).clamp(0.0, 1.0);
            backend.set_fade(1.0 - t);
            if t >= 1.0 {
                flow.fade = FadePhase::None;
            }
        }
        FadePhase::None => {}
    }
}

/// Imperatively jump to a named scene. Ignored with a warning if the target
/// scene is not declared, or no scenes exist.
pub fn jump_to_scene<B: SceneControl + ?Sized>(
    flow_opt: &mut Option<SceneFlow>,
    visibility: &SceneVisibility,
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
            set_scene_visibility(visibility, target_scene, backend);
            tracing::debug!("SceneCommand: cut to scene {}", target_scene);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::vec;
    // Minimal SceneControl implementation that records every call.
    #[derive(Default)]
    struct TestBackend {
        visibility: Vec<(usize, bool)>,
        fades: Vec<f32>,
    }

    impl SceneControl for TestBackend {
        fn update_visibility(&mut self, draw_idx: usize, visible: bool) {
            self.visibility.push((draw_idx, visible));
        }
        fn set_fade(&mut self, fade: f32) {
            self.fades.push(fade);
        }
    }

    fn make_flow(scenes: &[AssetId]) -> SceneFlow {
        SceneFlow {
            scenes: scenes.to_vec(),
            current: scenes[0],
            fade: FadePhase::None,
        }
    }

    fn vis(props: &[(&[usize], Option<AssetId>)]) -> SceneVisibility {
        let mut v = SceneVisibility::default();
        for (draws, scene) in props {
            v.begin_prop(*scene);
            for &d in *draws {
                v.push_draw(d);
            }
        }
        v
    }

    #[test]
    fn set_visibility_active_scene_visible_others_hidden() {
        // Three props: one in "a", one with no scene, one in "b".
        let visibility = vis(&[
            (&[0], Some(AssetId(0))),
            (&[1], None),
            (&[2], Some(AssetId(1))),
        ]);
        let mut backend = TestBackend::default();
        set_scene_visibility(&visibility, AssetId(0), &mut backend);

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
        let visibility = vis(&[(&[0], None)]);
        let mut backend = TestBackend::default();
        set_scene_visibility(&visibility, AssetId(99), &mut backend);
        assert_eq!(backend.visibility, vec![(0, true)]);
    }

    #[test]
    fn tick_without_fade_is_a_no_op() {
        let mut opt = Some(make_flow(&[AssetId(0), AssetId(1)]));
        let mut backend = TestBackend::default();
        tick_transitions(&mut opt, &SceneVisibility::default(), 999.0, &mut backend);
        assert!(backend.visibility.is_empty());
        assert!(backend.fades.is_empty());
        assert_eq!(opt.as_ref().unwrap().current, AssetId(0));
    }

    #[test]
    fn tick_fade_to_black_ramps_the_fade_up() {
        let mut flow = make_flow(&[AssetId(0), AssetId(1)]);
        flow.fade = FadePhase::ToBlack {
            started_at: 0.0,
            next: AssetId(1),
        };
        let mut opt = Some(flow);
        let mut backend = TestBackend::default();
        // elapsed = FADE_HALF_SECS / 2 → t = 0.5
        tick_transitions(
            &mut opt,
            &SceneVisibility::default(),
            FADE_HALF_SECS * 0.5,
            &mut backend,
        );
        assert_eq!(backend.fades.len(), 1);
        assert!(
            (backend.fades[0] - 0.5).abs() < 1e-5,
            "half way to black at the midpoint of the first half"
        );
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
        tick_transitions(
            &mut opt,
            &SceneVisibility::default(),
            FADE_HALF_SECS,
            &mut backend,
        );
        let f = opt.as_ref().unwrap();
        assert_eq!(f.current, AssetId(1));
        assert!(matches!(f.fade, FadePhase::FromBlack { .. }));
    }

    #[test]
    fn tick_fade_from_black_clears_the_fade() {
        let mut flow = make_flow(&[AssetId(0)]);
        flow.fade = FadePhase::FromBlack { started_at: 0.0 };
        let mut opt = Some(flow);
        let mut backend = TestBackend::default();
        // elapsed = FADE_HALF_SECS → t = 1.0, fade ends
        tick_transitions(
            &mut opt,
            &SceneVisibility::default(),
            FADE_HALF_SECS,
            &mut backend,
        );
        assert!(matches!(opt.as_ref().unwrap().fade, FadePhase::None));
        // The last push leaves the image un-faded.
        assert_eq!(*backend.fades.last().unwrap(), 0.0);
    }

    // The fade-in half runs the fade back down, so a frame partway through it
    // is partially, not fully, black.
    #[test]
    fn tick_fade_from_black_ramps_the_fade_down() {
        let mut flow = make_flow(&[AssetId(0)]);
        flow.fade = FadePhase::FromBlack { started_at: 0.0 };
        let mut opt = Some(flow);
        let mut backend = TestBackend::default();
        tick_transitions(
            &mut opt,
            &SceneVisibility::default(),
            FADE_HALF_SECS * 0.25,
            &mut backend,
        );
        assert_eq!(backend.fades.len(), 1);
        assert!((backend.fades[0] - 0.75).abs() < 1e-5);
        assert!(matches!(
            opt.as_ref().unwrap().fade,
            FadePhase::FromBlack { .. }
        ));
    }

    #[test]
    fn jump_to_scene_no_flow_is_no_op() {
        let mut opt: Option<SceneFlow> = None;
        let mut backend = TestBackend::default();
        jump_to_scene(
            &mut opt,
            &SceneVisibility::default(),
            0.0,
            AssetId(99),
            "Cut",
            &mut backend,
        );
        assert!(backend.visibility.is_empty());
    }

    #[test]
    fn jump_to_unknown_scene_is_no_op() {
        let mut opt = Some(make_flow(&[AssetId(0), AssetId(1)]));
        let mut backend = TestBackend::default();
        jump_to_scene(
            &mut opt,
            &SceneVisibility::default(),
            0.0,
            AssetId(99),
            "Cut",
            &mut backend,
        );
        assert_eq!(opt.as_ref().unwrap().current, AssetId(0));
        assert!(backend.visibility.is_empty());
    }

    #[test]
    fn jump_to_scene_same_scene_is_no_op() {
        let mut opt = Some(make_flow(&[AssetId(0), AssetId(1)]));
        let mut backend = TestBackend::default();
        jump_to_scene(
            &mut opt,
            &SceneVisibility::default(),
            0.0,
            AssetId(0),
            "Cut",
            &mut backend,
        );
        assert_eq!(opt.as_ref().unwrap().current, AssetId(0));
        assert!(backend.visibility.is_empty());
    }

    #[test]
    fn jump_to_scene_cut_switches_immediately() {
        let visibility = vis(&[(&[0], Some(AssetId(0))), (&[1], Some(AssetId(1)))]);
        let mut opt = Some(make_flow(&[AssetId(0), AssetId(1)]));
        let mut backend = TestBackend::default();
        jump_to_scene(&mut opt, &visibility, 1.0, AssetId(1), "Cut", &mut backend);
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
            &SceneVisibility::default(),
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
