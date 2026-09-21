//! Whether a world runs windowed or headless, resolved once per run.
//!
//! This replaces asking the world for a marker component. Four things decide
//! it, and only two of them are the world's own, so the answer is computed
//! before the world starts and published as a resource the render band's gates
//! read.

use concinnity_core::components::AppConfig;
use concinnity_core::ecs::World;

/// How a world runs: with a window and a renderer, or with neither.
///
/// Resolved once by [`resolve`], before `World::start` drains the columns it
/// reads, and published as a world resource. Every system in the render band
/// is gated on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderMode {
    /// A window and a renderer, stepped against the display.
    Rendered,
    /// No window and no renderer: the simulation systems on a fixed virtual
    /// timestep.
    Headless,
}

impl RenderMode {
    /// Whether this mode builds a renderer.
    pub fn renders(self) -> bool {
        self == RenderMode::Rendered
    }
}

/// Why a world resolved the way it did, for the one log line that says so.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadlessReason {
    /// The world declares `AppConfig.headless`.
    Requested,
    /// No render backend compiles into this build.
    NoBackend,
    /// The machine exposes no usable GPU.
    NoGpu,
    /// The world holds nothing to draw.
    NothingToDraw,
}

impl HeadlessReason {
    fn as_str(self) -> &'static str {
        match self {
            HeadlessReason::Requested => "the world asks for it (AppConfig.headless)",
            HeadlessReason::NoBackend => "this build compiles no render backend",
            HeadlessReason::NoGpu => "this machine exposes no usable GPU",
            HeadlessReason::NothingToDraw => "the world holds nothing to draw",
        }
    }
}

/// Resolve how `world` runs, and why when the answer is headless.
///
/// The checks run cheapest-first, so the GPU probe only happens for a world
/// that would otherwise open a window. Must be called before
/// [`World::start`](concinnity_core::ecs::World::start), which drains several
/// of the columns [`World::renders`] reads.
pub fn resolve(world: &World) -> (RenderMode, Option<HeadlessReason>) {
    let headless = |r| (RenderMode::Headless, Some(r));

    if world
        .query::<AppConfig>()
        .next()
        .is_some_and(|c| c.headless)
    {
        return headless(HeadlessReason::Requested);
    }
    if !world.renders() {
        return headless(HeadlessReason::NothingToDraw);
    }
    if !crate::device::AVAILABLE {
        return headless(HeadlessReason::NoBackend);
    }
    if crate::device::probe_gpu_profile().is_none() {
        return headless(HeadlessReason::NoGpu);
    }
    (RenderMode::Rendered, None)
}

/// The world's half of the resolution, with no GPU probe: whether the content
/// asks to be seen.
///
/// This is what an authoring tool wants. `cn list --systems` reports the
/// schedule a world implies, which is a property of the world and must not
/// change with the machine the listing happens to run on; [`resolve`] answers
/// for this machine and is what a run uses.
pub fn from_content(world: &World) -> RenderMode {
    let asked_for_headless = world
        .query::<AppConfig>()
        .next()
        .is_some_and(|c| c.headless);
    if !asked_for_headless && world.renders() {
        RenderMode::Rendered
    } else {
        RenderMode::Headless
    }
}

/// Resolve as [`resolve`] does and log the outcome, so a run that quietly did
/// not open a window says why it did not.
pub fn resolve_and_report(world: &World) -> RenderMode {
    let (mode, reason) = resolve(world);
    if let Some(reason) = reason {
        tracing::info!("Running headless: {}", reason.as_str());
    }
    mode
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::{PhysicsConfig, Prop, TextLabel};

    // The README case: content alone decides, with no GraphicsConfig anywhere.
    // What the resolution lands on depends on whether this build has a backend
    // and the host a GPU, so the assertion is on the REASON: a world with
    // something to draw is never turned down for having nothing to draw.
    #[test]
    fn a_world_with_something_to_draw_is_not_refused_for_being_empty() {
        let mut world = World::new();
        world.add_component(TextLabel {
            content: "Hello, world!".into(),
            ..Default::default()
        });

        let (_, reason) = resolve(&world);
        assert_ne!(reason, Some(HeadlessReason::NothingToDraw));
    }

    // A simulation-only world has no window to open, so it resolves headless
    // without ever touching the GPU.
    #[test]
    fn a_world_with_nothing_to_draw_runs_headless() {
        let mut world = World::new();
        world.add_component(PhysicsConfig::default());

        assert_eq!(
            resolve(&world),
            (RenderMode::Headless, Some(HeadlessReason::NothingToDraw))
        );
    }

    // An empty world holds nothing to draw either: no window opens for it.
    #[test]
    fn an_empty_world_runs_headless() {
        assert_eq!(
            resolve(&World::new()),
            (RenderMode::Headless, Some(HeadlessReason::NothingToDraw))
        );
    }

    // The opt-out, and the whole point of it: a world full of geometry stays
    // headless because it asked to. Checked first, so it never reaches the
    // content test or the GPU probe.
    #[test]
    fn app_config_headless_overrides_a_world_full_of_geometry() {
        let mut world = World::new();
        world.add_component(Prop::default());
        world.add_component(AppConfig {
            headless: true,
            ..Default::default()
        });

        assert_eq!(
            resolve(&world),
            (RenderMode::Headless, Some(HeadlessReason::Requested))
        );
    }

    // An AppConfig is how a world sets its home and budgets; declaring one
    // must not turn the renderer off by itself.
    #[test]
    fn an_app_config_that_does_not_ask_for_headless_leaves_the_choice_to_content() {
        let mut world = World::new();
        world.add_component(Prop::default());
        world.add_component(AppConfig::default());

        let (_, reason) = resolve(&world);
        assert_ne!(reason, Some(HeadlessReason::Requested));
    }

    // The authoring answer reads the world and nothing else, so it says
    // "rendered" for a world with geometry on any machine -- including a build
    // with no backend, where `resolve` would say otherwise.
    #[test]
    fn from_content_answers_for_the_world_not_the_machine() {
        let mut world = World::new();
        world.add_component(Prop::default());
        assert_eq!(from_content(&world), RenderMode::Rendered);

        let mut world = World::new();
        world.add_component(PhysicsConfig::default());
        assert_eq!(from_content(&world), RenderMode::Headless);
    }

    // The opt-out still wins there: a world that ships headless lists the
    // schedule it will actually run.
    #[test]
    fn from_content_honors_the_headless_opt_out() {
        let mut world = World::new();
        world.add_component(Prop::default());
        world.add_component(AppConfig {
            headless: true,
            ..Default::default()
        });

        assert_eq!(from_content(&world), RenderMode::Headless);
    }
}
