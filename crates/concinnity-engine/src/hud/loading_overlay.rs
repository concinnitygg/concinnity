//! Scene-loading overlay: shows the `LoadingOverlay`'s screen while the scene a
//! jump targets still streams content, drives its progress bar and label, and
//! fades the backdrop out to reveal the scene once it is resident. Runs after
//! `StreamingSystem` (it reads the residency status published this tick) and
//! before `UiInputSystem` (its screen commands apply the same tick); the elements
//! it writes are drawn by the next overlay build, like every HUD system.

use crate::ecs::{Access, FrameTime, PipelineContext, Ref, ScreenStack, StepResult, System, World};
use alloc::format;

use crate::components::Screen;
use crate::components::{ScreenCommand, Sprite, TextLabel};
use crate::ecs::asset_id::AssetId;
use crate::render::scene::flow::{ActiveSceneFlow, FadePhase};
use crate::render::streaming::scene_residency::{SceneLoadState, SceneResidencyStatus};

// Seconds the backdrop takes to fade out over the freshly resident scene.
const FADE_OUT_SECS: f32 = 0.35;

#[derive(Debug)]
enum Phase {
    Hidden,
    Shown,
    // Seconds of frame time since the fade-out began.
    FadingOut { secs: f32 },
}

/// The scene-loading overlay, built from a world's `LoadingOverlay` request
/// component.
#[derive(Debug)]
pub(crate) struct LoadingOverlaySystem {
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
    /// The system for a world that declares a `LoadingOverlay`.
    pub(crate) fn for_world(world: &World) -> Option<Self> {
        world.query::<LoadingOverlay>().next().map(Self::new)
    }

    /// Build the overlay from a world's `LoadingOverlay` request component.
    pub(crate) fn new(config: &LoadingOverlay) -> Self {
        Self {
            screen: config.screen.map(Ref::id),
            backdrop: config.backdrop.map(Ref::id),
            track: config.track.map(Ref::id),
            fill: config.fill.map(Ref::id),
            label: config.label.map(Ref::id),
            phase: Phase::Hidden,
            backdrop_alpha: None,
        }
    }

    fn sprite_mut<'a>(
        ctx: &'a mut PipelineContext<'_>,
        id: Option<AssetId>,
    ) -> Option<&'a mut Sprite> {
        ctx.get_mut_by_id::<Sprite>(id?)
    }

    fn set_visible(ctx: &mut PipelineContext<'_>, id: Option<AssetId>, visible: bool) {
        if let Some(sprite) = Self::sprite_mut(ctx, id) {
            sprite.visible = visible;
        }
    }

    fn set_backdrop_alpha(&self, ctx: &mut PipelineContext<'_>, alpha: f32) {
        if let Some(sprite) = Self::sprite_mut(ctx, self.backdrop) {
            sprite.tint[3] = alpha;
        }
    }

    // Restore the overlay's elements to their shown state: the backdrop at its
    // authored alpha (captured here on first show), the bar visible.
    fn show(&mut self, ctx: &mut PipelineContext<'_>, progress: f32) {
        if self.backdrop_alpha.is_none() {
            self.backdrop_alpha = Self::sprite_mut(ctx, self.backdrop).map(|s| s.tint[3]);
        }
        if let Some(alpha) = self.backdrop_alpha {
            self.set_backdrop_alpha(ctx, alpha);
        }
        Self::set_visible(ctx, self.track, true);
        if let Some(id) = self.label
            && let Some(label) = ctx.get_mut_by_id::<TextLabel>(id)
        {
            label.visible = true;
        }
        self.update_bar(ctx, progress);
    }

    // Size the fill against the track and refresh the percentage label.
    fn update_bar(&self, ctx: &mut PipelineContext<'_>, progress: f32) {
        let progress = progress.clamp(0.0, 1.0);
        let track = self
            .track
            .and_then(|id| ctx.get_by_id::<Sprite>(id))
            .map(|s| (s.x, s.width));
        if let Some(fill) = Self::sprite_mut(ctx, self.fill) {
            if let Some((x, width)) = track {
                fill.x = x;
                fill.width = width * progress;
            }
            fill.visible = progress > 0.0;
        }
        if let Some(id) = self.label
            && let Some(label) = ctx.get_mut_by_id::<TextLabel>(id)
        {
            label.content = format!("Loading {:.0}%", progress * 100.0);
        }
    }

    // Hide the bar and label, leaving only the backdrop to fade out.
    fn hide_bar(&self, ctx: &mut PipelineContext<'_>) {
        Self::set_visible(ctx, self.track, false);
        Self::set_visible(ctx, self.fill, false);
        if let Some(id) = self.label
            && let Some(label) = ctx.get_mut_by_id::<TextLabel>(id)
        {
            label.visible = false;
        }
    }

    // Whether the overlay's screen currently draws on top of the stack.
    // `None` when the screen is not in the stack at all.
    fn screen_on_top(&self, ctx: &PipelineContext<'_>) -> Option<bool> {
        let screen = self.screen?;
        let stack = ctx.resource::<ScreenStack>()?;
        let mine = *stack.layers.get(&screen)?;
        Some(stack.layers.values().all(|&layer| layer <= mine))
    }
}

impl System for LoadingOverlaySystem {
    fn access(&self) -> Access {
        Access::new()
            .writes_components(crate::component_mask![Sprite, TextLabel])
            .reads_resources(crate::resource_mask![
                crate::ecs::EntityById,
                ActiveSceneFlow,
                SceneResidencyStatus,
                ScreenStack,
                FrameTime,
            ])
            .writes_resources(crate::resource_mask![ScreenCommand])
    }

    fn step(&mut self, ctx: &mut PipelineContext<'_>) -> StepResult {
        // The scene the player is headed to: the fade target mid-jump, else
        // the active scene.
        let target = ctx
            .resource::<ActiveSceneFlow>()
            .and_then(|slot| slot.flow.as_ref())
            .map(|flow| match flow.fade {
                FadePhase::ToBlack { next, .. } => next,
                _ => flow.current,
            });
        let status = target.and_then(|t| {
            ctx.resource::<SceneResidencyStatus>()
                .and_then(|s| s.scenes.iter().find(|(id, ..)| *id == t))
                .copied()
        });
        let loading = matches!(status, Some((_, SceneLoadState::Loading, _)));
        let progress = status.map_or(1.0, |(_, _, p)| p);

        match self.phase {
            Phase::Hidden => {
                // Defer to an already-open pausing screen (a menu): the
                // overlay fronts a load only when nothing else covers the
                // world.
                let menu_up = ctx
                    .resource::<ScreenStack>()
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
                    self.phase = Phase::FadingOut { secs: 0.0 };
                }
            }
            Phase::FadingOut { ref mut secs } => {
                if loading {
                    // A new jump began before the reveal finished.
                    self.show(ctx, progress);
                    self.phase = Phase::Shown;
                    return StepResult::Continue;
                }
                *secs += ctx
                    .resource::<FrameTime>()
                    .copied()
                    .unwrap_or_default()
                    .dt
                    .max(0.0);
                let faded = fade_remaining(*secs);
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

// The backdrop's remaining opacity fraction `elapsed` into the fade-out; at or
// below zero once the fade has finished.
fn fade_remaining(elapsed_secs: f32) -> f32 {
    1.0 - elapsed_secs / FADE_OUT_SECS
}

/// Requests the scene-loading overlay: a full-window backdrop with a progress
/// bar, shown while a scene jump waits for its streamed content and faded out
/// once the destination scene is fully resident.
///
/// The overlay is assembled from ordinary UI elements the fields reference: a
/// [Screen](#screen) that hosts them, a backdrop [Sprite](#sprite) covering
/// the canvas, a progress-bar track and fill [Sprite](#sprite) pair, and a
/// [`TextLabel`](#textlabel) above the bar. Each frame the engine widens the
/// fill to the destination scene's load progress and rewrites the label with
/// the percentage; restyle any piece by declaring it yourself and pointing the
/// overlay's field at it.
///
/// While the overlay's screen is up the world pauses and its render is
/// skipped, exactly like an opaque menu, but streaming keeps running so the
/// load it reports can finish. Scenes whose content is already resident jump
/// without the overlay ever appearing.
///
/// Every rendering world that declares [Scene](#scene)s and a
/// [`StreamingConfig`](#streamingconfig) receives a `LoadingOverlay` at start
/// when it declares none, and any field left unset receives the piece it
/// names, so the example below is only needed to restyle them. Declare an
/// [`EngineDefaults`](#enginedefaults) with `"loading_overlay": false` to leave
/// the world without one.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, crate::ecs::AssetFields)]
#[serde(default)]
pub struct LoadingOverlay {
    /// [Screen](#screen) the overlay shows while a scene loads. Its
    /// `pauses_world` (on by default) freezes the world beneath the overlay so
    /// the destination scene starts fresh when revealed.
    pub screen: Option<Ref<Screen>>,
    /// Backdrop [Sprite](#sprite) covering the canvas. Its tint alpha is
    /// animated to reveal the scene once loading completes; an opaque tint
    /// hides the still-loading world completely.
    pub backdrop: Option<Ref<Sprite>>,
    /// Progress-bar track [Sprite](#sprite); its width is the bar's full
    /// extent the fill is measured against.
    pub track: Option<Ref<Sprite>>,
    /// Progress-bar fill [Sprite](#sprite); the engine sets its width to the
    /// track width times the destination scene's load progress each frame.
    pub fill: Option<Ref<Sprite>>,
    /// [`TextLabel`](#textlabel) rewritten each frame with the load percentage.
    pub label: Option<Ref<TextLabel>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::Screen;
    use crate::ecs::World;
    use crate::render::scene::flow::SceneFlow;
    use alloc::vec;

    // The overlay, then the UI system that applies the screen commands it
    // sends, in the engine's run order.
    fn start(world: &mut World) {
        world.add_system(
            crate::ecs::Phase::PreRender,
            "LoadingOverlaySystem",
            LoadingOverlaySystem::new(&overlay_config()),
        );
        crate::test_support::start_with(
            world,
            crate::ecs::Phase::PreRender,
            "UiInputSystem",
            crate::ui::UiInputSystem::new(),
        );
    }

    const SCREEN: AssetId = AssetId(1);
    const BACKDROP: AssetId = AssetId(2);
    const TRACK: AssetId = AssetId(3);
    const FILL: AssetId = AssetId(4);
    const LABEL: AssetId = AssetId(5);
    const SCENE: AssetId = AssetId(9);

    fn overlay_config() -> LoadingOverlay {
        LoadingOverlay {
            screen: Some(Ref::new(SCREEN)),
            backdrop: Some(Ref::new(BACKDROP)),
            track: Some(Ref::new(TRACK)),
            fill: Some(Ref::new(FILL)),
            label: Some(Ref::new(LABEL)),
        }
    }

    // The overlay's screen, elements and scene flow.
    fn overlay_world() -> World {
        let mut world = World::new();
        world.push_identified(SCREEN, Screen::default());
        world.push_identified(
            BACKDROP,
            Sprite {
                tint: [0.0, 0.0, 0.0, 1.0],
                screen: Some(Ref::new(SCREEN)),
                ..Default::default()
            },
        );
        world.push_identified(
            TRACK,
            Sprite {
                x: 400.0,
                width: 480.0,
                screen: Some(Ref::new(SCREEN)),
                ..Default::default()
            },
        );
        world.push_identified(
            FILL,
            Sprite {
                x: 400.0,
                width: 0.0,
                visible: false,
                screen: Some(Ref::new(SCREEN)),
                ..Default::default()
            },
        );
        world.push_identified(
            LABEL,
            TextLabel {
                screen: Some(Ref::new(SCREEN)),
                ..Default::default()
            },
        );
        world.insert_resource(ActiveSceneFlow::new(Some(SceneFlow {
            scenes: vec![SCENE],
            current: SCENE,
            fade: FadePhase::None,
        })));
        world
    }

    fn set_status(world: &mut World, state: SceneLoadState, progress: f32) {
        world.insert_resource(SceneResidencyStatus {
            scenes: vec![(SCENE, state, progress)],
        });
    }

    fn shown(world: &World) -> bool {
        world
            .resource::<ScreenStack>()
            .is_some_and(|s| s.layers.contains_key(&SCREEN))
    }

    fn sprite(world: &World, id: AssetId) -> &Sprite {
        world.get_by_id::<Sprite>(id).unwrap()
    }

    #[test]
    fn loading_scene_shows_the_overlay_and_drives_the_bar() {
        let mut world = overlay_world();
        set_status(&mut world, SceneLoadState::Loading, 0.25);
        start(&mut world);
        world.step();

        assert!(shown(&world), "overlay screen should be on the stack");
        let fill = sprite(&world, FILL);
        assert!(fill.visible);
        assert!((fill.width - 480.0 * 0.25).abs() < 1e-3);
        let label = world.get_by_id::<TextLabel>(LABEL).unwrap();
        assert_eq!(label.content, "Loading 25%");
    }

    #[test]
    fn resident_scene_never_shows_the_overlay() {
        let mut world = overlay_world();
        set_status(&mut world, SceneLoadState::Resident, 1.0);
        start(&mut world);
        world.step();
        assert!(!shown(&world));
    }

    #[test]
    fn missing_residency_status_never_shows_the_overlay() {
        let mut world = overlay_world();
        start(&mut world);
        world.step();
        assert!(!shown(&world));
    }

    #[test]
    fn an_open_pausing_screen_defers_the_overlay() {
        let mut world = overlay_world();
        world.push_identified(
            AssetId(20),
            Screen {
                initial: true,
                ..Default::default()
            },
        );
        set_status(&mut world, SceneLoadState::Loading, 0.5);
        start(&mut world);
        // Two steps: the first publishes the initial screen's stack, the
        // second is the earliest the overlay could react to it.
        world.step();
        world.step();
        assert!(!shown(&world), "overlay must not cover an open menu");
    }

    #[test]
    fn fade_remaining_runs_from_one_to_zero_over_the_fade() {
        assert_eq!(fade_remaining(0.0), 1.0);
        let half = fade_remaining(FADE_OUT_SECS / 2.0);
        assert!((half - 0.5).abs() < 1e-6);
        assert!(fade_remaining(FADE_OUT_SECS) <= 0.0);
    }

    #[test]
    fn resident_target_fades_the_overlay_out_and_hides_it() {
        let mut world = overlay_world();
        set_status(&mut world, SceneLoadState::Loading, 0.5);
        start(&mut world);
        world.step();
        assert!(shown(&world));

        // The load completes: the bar hides first, then the backdrop fades.
        set_status(&mut world, SceneLoadState::Resident, 1.0);
        world.step();
        assert!(!sprite(&world, FILL).visible);
        assert!(!sprite(&world, TRACK).visible);
        assert!(shown(&world), "backdrop still fading");

        // The backdrop fades by frame time, then the screen pops off the stack.
        world.insert_resource(FrameTime {
            dt: FADE_OUT_SECS * 0.6,
            elapsed: 0.0,
        });
        world.step();
        assert!((sprite(&world, BACKDROP).tint[3] - 0.4).abs() < 1e-4);
        assert!(shown(&world), "backdrop still fading");
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
