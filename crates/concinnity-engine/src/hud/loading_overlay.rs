// src/hud/loading_overlay.rs
//
// Scene-loading overlay: shows the LoadingOverlay's screen while the scene a
// jump targets still streams content, drives its progress bar and label, and
// fades the backdrop out to reveal the scene once it is resident. Runs after
// StreamingSystem (it reads the residency status published this tick) and
// before UiInputSystem (its screen commands apply the same tick); the elements
// it writes are drawn by the next overlay build, like every HUD system.

use crate::components::{LoadingOverlay, ScreenCommand, Sprite, TextLabel};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, StepResult, System};
use crate::gfx::scene_flow::FadePhase;
use crate::gfx::scene_residency::SceneLoadState;
use std::time::Instant;

// Seconds the backdrop takes to fade out over the freshly resident scene.
const FADE_OUT_SECS: f32 = 0.35;

#[derive(Debug)]
enum Phase {
    Hidden,
    Shown,
    FadingOut { started: Instant },
}

#[derive(Debug)]
pub struct LoadingOverlaySystem {
    screen: Option<AssetId>,
    backdrop: Option<AssetId>,
    track: Option<AssetId>,
    fill: Option<AssetId>,
    label: Option<AssetId>,
    phase: Phase,
    // The backdrop's authored tint alpha, captured on first show so the
    // fade-out animation can restore it for the next load.
    backdrop_alpha: Option<f32>,
}

impl LoadingOverlaySystem {
    // Build the overlay from a world's `LoadingOverlay` request component.
    pub fn new(config: LoadingOverlay) -> Self {
        Self {
            screen: config.screen,
            backdrop: config.backdrop,
            track: config.track,
            fill: config.fill,
            label: config.label,
            phase: Phase::Hidden,
            backdrop_alpha: None,
        }
    }

    fn sprite_mut<'a>(ctx: &'a mut PipelineContext, id: Option<AssetId>) -> Option<&'a mut Sprite> {
        let id = id?;
        ctx.query_mut::<Sprite>().find(|s| s.asset_id == id)
    }

    fn set_visible(ctx: &mut PipelineContext, id: Option<AssetId>, visible: bool) {
        if let Some(sprite) = Self::sprite_mut(ctx, id) {
            sprite.visible = visible;
        }
    }

    fn set_backdrop_alpha(&self, ctx: &mut PipelineContext, alpha: f32) {
        if let Some(sprite) = Self::sprite_mut(ctx, self.backdrop) {
            sprite.tint[3] = alpha;
        }
    }

    // Restore the overlay's elements to their shown state: the backdrop at its
    // authored alpha (captured here on first show), the bar visible.
    fn show(&mut self, ctx: &mut PipelineContext, progress: f32) {
        if self.backdrop_alpha.is_none() {
            self.backdrop_alpha = Self::sprite_mut(ctx, self.backdrop).map(|s| s.tint[3]);
        }
        if let Some(alpha) = self.backdrop_alpha {
            self.set_backdrop_alpha(ctx, alpha);
        }
        Self::set_visible(ctx, self.track, true);
        if let Some(id) = self.label
            && let Some(label) = ctx.query_mut::<TextLabel>().find(|l| l.asset_id == id)
        {
            label.visible = true;
        }
        self.update_bar(ctx, progress);
    }

    // Size the fill against the track and refresh the percentage label.
    fn update_bar(&self, ctx: &mut PipelineContext, progress: f32) {
        let progress = progress.clamp(0.0, 1.0);
        let track = self
            .track
            .and_then(|id| ctx.query::<Sprite>().find(|s| s.asset_id == id))
            .map(|s| (s.x, s.width));
        if let Some(fill) = Self::sprite_mut(ctx, self.fill) {
            if let Some((x, width)) = track {
                fill.x = x;
                fill.width = width * progress;
            }
            fill.visible = progress > 0.0;
        }
        if let Some(id) = self.label
            && let Some(label) = ctx.query_mut::<TextLabel>().find(|l| l.asset_id == id)
        {
            label.content = format!("Loading {:.0}%", progress * 100.0);
        }
    }

    // Hide the bar and label, leaving only the backdrop to fade out.
    fn hide_bar(&self, ctx: &mut PipelineContext) {
        Self::set_visible(ctx, self.track, false);
        Self::set_visible(ctx, self.fill, false);
        if let Some(id) = self.label
            && let Some(label) = ctx.query_mut::<TextLabel>().find(|l| l.asset_id == id)
        {
            label.visible = false;
        }
    }

    // Whether the overlay's screen currently draws on top of the stack.
    // `None` when the screen is not in the stack at all.
    fn screen_on_top(&self, ctx: &PipelineContext) -> Option<bool> {
        let screen = self.screen?;
        let stack = ctx.resource::<crate::ecs::ScreenStack>()?;
        let mine = *stack.layers.get(&screen)?;
        Some(stack.layers.values().all(|&layer| layer <= mine))
    }
}

impl System for LoadingOverlaySystem {
    fn access(&self) -> crate::ecs::Access {
        crate::ecs::Access::new()
            .writes_components(crate::component_mask![
                crate::components::Sprite,
                crate::components::TextLabel,
            ])
            .reads_resources(crate::resource_mask![
                crate::ecs::ActiveSceneFlow,
                crate::ecs::SceneResidencyStatus,
                crate::ecs::ScreenStack,
            ])
            .writes_resources(crate::resource_mask![crate::components::ScreenCommand])
    }

    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        // The scene the player is headed to: the fade target mid-jump, else
        // the active scene.
        let target = ctx
            .resource::<crate::ecs::ActiveSceneFlow>()
            .and_then(|slot| slot.flow.as_ref())
            .map(|flow| match flow.fade {
                FadePhase::ToBlack { next, .. } => next,
                _ => flow.current,
            });
        let status = target.and_then(|t| {
            ctx.resource::<crate::ecs::SceneResidencyStatus>()
                .and_then(|s| s.scenes.iter().find(|(id, ..)| *id == t))
                .copied()
        });
        let loading = matches!(status, Some((_, SceneLoadState::Loading, _)));
        let progress = status.map(|(_, _, p)| p).unwrap_or(1.0);

        match self.phase {
            Phase::Hidden => {
                // Defer to an already-open pausing screen (a menu): the
                // overlay fronts a load only when nothing else covers the
                // world.
                let menu_up = ctx
                    .resource::<crate::ecs::ScreenStack>()
                    .is_some_and(|s| s.pauses_world);
                if loading
                    && !menu_up
                    && let Some(screen) = self.screen
                {
                    ctx.events_mut::<ScreenCommand>()
                        .send(ScreenCommand::Push(screen));
                    self.show(ctx, progress);
                    self.phase = Phase::Shown;
                }
            }
            Phase::Shown => {
                if loading {
                    // A scene jump dismisses every open screen; if that took
                    // the overlay down mid-load (a fresh jump), re-open it.
                    if self.screen_on_top(ctx).is_none()
                        && let Some(screen) = self.screen
                    {
                        ctx.events_mut::<ScreenCommand>()
                            .send(ScreenCommand::Push(screen));
                    }
                    self.update_bar(ctx, progress);
                } else {
                    self.hide_bar(ctx);
                    self.phase = Phase::FadingOut {
                        started: Instant::now(),
                    };
                }
            }
            Phase::FadingOut { started } => {
                if loading {
                    // A new jump began before the reveal finished.
                    self.show(ctx, progress);
                    self.phase = Phase::Shown;
                    return StepResult::Continue;
                }
                let faded = 1.0 - started.elapsed().as_secs_f32() / FADE_OUT_SECS;
                if faded > 0.0 {
                    let base = self.backdrop_alpha.unwrap_or(1.0);
                    self.set_backdrop_alpha(ctx, base * faded);
                    return StepResult::Continue;
                }
                // Fully faded: pop the screen once it is on top (beneath a
                // later-pushed screen it stays, invisible, until then).
                match self.screen_on_top(ctx) {
                    Some(true) => {
                        ctx.events_mut::<ScreenCommand>().send(ScreenCommand::Hide);
                        self.phase = Phase::Hidden;
                    }
                    Some(false) => self.set_backdrop_alpha(ctx, 0.0),
                    // Dismissed externally (a scene jump's screen clear).
                    None => self.phase = Phase::Hidden,
                }
            }
        }
        StepResult::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::Screen;
    use crate::ecs::World;
    use crate::gfx::scene_flow::SceneFlow;

    const SCREEN: AssetId = AssetId(1);
    const BACKDROP: AssetId = AssetId(2);
    const TRACK: AssetId = AssetId(3);
    const FILL: AssetId = AssetId(4);
    const LABEL: AssetId = AssetId(5);
    const SCENE: AssetId = AssetId(9);

    fn overlay_world() -> World {
        let mut world = World::new();
        world.add_component(LoadingOverlay {
            screen: Some(SCREEN),
            backdrop: Some(BACKDROP),
            track: Some(TRACK),
            fill: Some(FILL),
            label: Some(LABEL),
        });
        world.add_component(Screen {
            asset_id: SCREEN,
            ..Default::default()
        });
        world.add_component(Sprite {
            asset_id: BACKDROP,
            tint: [0.0, 0.0, 0.0, 1.0],
            screen: Some(SCREEN),
            ..Default::default()
        });
        world.add_component(Sprite {
            asset_id: TRACK,
            x: 400.0,
            width: 480.0,
            screen: Some(SCREEN),
            ..Default::default()
        });
        world.add_component(Sprite {
            asset_id: FILL,
            x: 400.0,
            width: 0.0,
            visible: false,
            screen: Some(SCREEN),
            ..Default::default()
        });
        world.add_component(TextLabel {
            asset_id: LABEL,
            screen: Some(SCREEN),
            ..Default::default()
        });
        world.insert_resource(crate::ecs::ActiveSceneFlow {
            flow: Some(SceneFlow {
                scenes: vec![SCENE],
                current: SCENE,
                fade: FadePhase::None,
            }),
            epoch: Instant::now(),
        });
        world
    }

    fn set_status(world: &mut World, state: SceneLoadState, progress: f32) {
        world.insert_resource(crate::ecs::SceneResidencyStatus {
            scenes: vec![(SCENE, state, progress)],
        });
    }

    fn shown(world: &World) -> bool {
        world
            .resource::<crate::ecs::ScreenStack>()
            .is_some_and(|s| s.layers.contains_key(&SCREEN))
    }

    fn sprite(world: &World, id: AssetId) -> &Sprite {
        world.query::<Sprite>().find(|s| s.asset_id == id).unwrap()
    }

    #[test]
    fn loading_scene_shows_the_overlay_and_drives_the_bar() {
        let mut world = overlay_world();
        set_status(&mut world, SceneLoadState::Loading, 0.25);
        world.start().unwrap();
        world.step();

        assert!(shown(&world), "overlay screen should be on the stack");
        let fill = sprite(&world, FILL);
        assert!(fill.visible);
        assert!((fill.width - 480.0 * 0.25).abs() < 1e-3);
        let label = world
            .query::<TextLabel>()
            .find(|l| l.asset_id == LABEL)
            .unwrap();
        assert_eq!(label.content, "Loading 25%");
    }

    #[test]
    fn resident_scene_never_shows_the_overlay() {
        let mut world = overlay_world();
        set_status(&mut world, SceneLoadState::Resident, 1.0);
        world.start().unwrap();
        world.step();
        assert!(!shown(&world));
    }

    #[test]
    fn missing_residency_status_never_shows_the_overlay() {
        let mut world = overlay_world();
        world.start().unwrap();
        world.step();
        assert!(!shown(&world));
    }

    #[test]
    fn an_open_pausing_screen_defers_the_overlay() {
        let mut world = overlay_world();
        world.add_component(Screen {
            asset_id: AssetId(20),
            initial: true,
            ..Default::default()
        });
        set_status(&mut world, SceneLoadState::Loading, 0.5);
        world.start().unwrap();
        // Two steps: the first publishes the initial screen's stack, the
        // second is the earliest the overlay could react to it.
        world.step();
        world.step();
        assert!(!shown(&world), "overlay must not cover an open menu");
    }

    #[test]
    fn resident_target_fades_the_overlay_out_and_hides_it() {
        let mut world = overlay_world();
        set_status(&mut world, SceneLoadState::Loading, 0.5);
        world.start().unwrap();
        world.step();
        assert!(shown(&world));

        // The load completes: the bar hides first, then the backdrop fades.
        set_status(&mut world, SceneLoadState::Resident, 1.0);
        world.step();
        assert!(!sprite(&world, FILL).visible);
        assert!(!sprite(&world, TRACK).visible);
        assert!(shown(&world), "backdrop still fading");

        // Past the fade the screen pops off the stack.
        std::thread::sleep(std::time::Duration::from_secs_f32(FADE_OUT_SECS + 0.1));
        world.step();
        world.step();
        assert!(!shown(&world));

        // A later jump restores the full overlay, opaque again.
        set_status(&mut world, SceneLoadState::Loading, 0.1);
        world.step();
        assert!(shown(&world));
        assert_eq!(sprite(&world, BACKDROP).tint[3], 1.0);
        assert!(sprite(&world, TRACK).visible);
    }
}
