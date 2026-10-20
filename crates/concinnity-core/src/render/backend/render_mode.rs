//! Whether a world runs windowed or headless, resolved once per run.
//!
//! This replaces asking the world for a marker component. Four things decide
//! it, and only two of them are the world's own, so the answer is computed
//! before the world starts and published as a resource the render band's gates
//! read. A world with something to draw needs a renderer: on a build or a
//! machine that cannot give it one, it does not quietly run headless; it fails
//! to start unless it declares `AppConfig.headless`.

use crate::components::AppConfig;
use crate::ecs::World;
use crate::render::backend::error::RenderError;

/// How a world runs: with a window and a renderer, or with neither.
///
/// Resolved once by [`resolve_with`], before `World::start` drains the columns it
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
        self == Self::Rendered
    }

    /// Whether the mode published in `world` builds a renderer. A world with
    /// none published was never started by a host that renders.
    pub fn renders_in(world: &World) -> bool {
        world.resource::<Self>().is_some_and(|m| m.renders())
    }
}

/// Why a world resolved headless, for the one log line that says so.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadlessReason {
    /// The world declares `AppConfig.headless`.
    Requested,
    /// The world holds nothing to draw.
    NothingToDraw,
}

impl HeadlessReason {
    /// Why, in words, for the log line that says so.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Requested => "the world asks for it (AppConfig.headless)",
            Self::NothingToDraw => "the world holds nothing to draw",
        }
    }
}

/// Why a world with something to draw cannot get a renderer here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoRenderer {
    /// No render backend compiles into this build.
    NoBackend,
    /// The machine exposes no usable GPU.
    NoGpu,
}

impl NoRenderer {
    /// What the run reports when it refuses to start.
    pub fn message(self) -> &'static str {
        match self {
            Self::NoBackend => {
                "this build compiles no render backend; set AppConfig.headless to run \
                 the world without one"
            }
            Self::NoGpu => {
                "no usable GPU was found; set AppConfig.headless to run the world \
                 without one"
            }
        }
    }
}

/// How a world resolved for this build and machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resolution {
    /// A window and a renderer.
    Rendered,
    /// No window, for the reason given.
    Headless(HeadlessReason),
    /// The world has something to draw and nothing here can draw it.
    Unavailable(NoRenderer),
}

impl Resolution {
    /// The mode a run takes; a world that cannot get its renderer opens no
    /// window either, and fails at start.
    pub fn mode(self) -> RenderMode {
        match self {
            Self::Rendered => RenderMode::Rendered,
            Self::Headless(_) | Self::Unavailable(_) => RenderMode::Headless,
        }
    }

    /// The mode a world starts in, or why it cannot start: it has something
    /// to draw and nothing here can draw it.
    pub fn startable(self) -> Result<RenderMode, RenderError> {
        match self {
            Self::Unavailable(why) => Err(RenderError::Other(why.message().into())),
            resolved => Ok(resolved.mode()),
        }
    }
}

/// Resolve how `world` runs given whether the build has a render `backend`
/// and, asked only when the content needs one, whether the machine has a
/// usable `gpu`. Must be called before
/// [`World::start`](crate::ecs::World::start), which drains several of the
/// columns [`World::renders`] reads.
pub fn resolve_with(world: &World, backend: bool, gpu: impl FnOnce() -> bool) -> Resolution {
    if world
        .query::<AppConfig>()
        .next()
        .is_some_and(|c| c.headless)
    {
        return Resolution::Headless(HeadlessReason::Requested);
    }
    if !world.renders() {
        return Resolution::Headless(HeadlessReason::NothingToDraw);
    }
    if !backend {
        return Resolution::Unavailable(NoRenderer::NoBackend);
    }
    if !gpu() {
        return Resolution::Unavailable(NoRenderer::NoGpu);
    }
    Resolution::Rendered
}

/// The world's half of the resolution, with no build or machine input:
/// whether the content asks to be seen.
///
/// This is what an authoring tool wants: the schedule a world implies is a
/// property of the world and must not change with the machine a listing
/// happens to run on.
pub fn from_content(world: &World) -> RenderMode {
    resolve_with(world, true, || true).mode()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{PhysicsConfig, Prop, TextLabel};

    #[test]
    fn only_an_unavailable_renderer_refuses_to_start() {
        assert_eq!(Resolution::Rendered.startable(), Ok(RenderMode::Rendered));
        assert_eq!(
            Resolution::Headless(HeadlessReason::NothingToDraw).startable(),
            Ok(RenderMode::Headless)
        );
        assert!(matches!(
            Resolution::Unavailable(NoRenderer::NoGpu).startable(),
            Err(RenderError::Other(_))
        ));
    }

    // The README case: content alone decides, with no settings asset anywhere.
    #[test]
    fn a_world_with_something_to_draw_is_not_refused_for_being_empty() {
        let mut world = World::new();
        world.add_component(TextLabel {
            content: "Hello, world!".into(),
            ..Default::default()
        });

        assert_eq!(resolve_with(&world, true, || true), Resolution::Rendered);
    }

    // A simulation-only world has no window to open, so it resolves headless
    // without ever touching the GPU.
    #[test]
    fn a_world_with_nothing_to_draw_runs_headless() {
        let mut world = World::new();
        world.add_component(PhysicsConfig::default());

        assert_eq!(
            resolve_with(&world, true, || true),
            Resolution::Headless(HeadlessReason::NothingToDraw)
        );
    }

    // An empty world holds nothing to draw either: no window opens for it.
    #[test]
    fn an_empty_world_runs_headless() {
        assert_eq!(
            resolve_with(&World::new(), true, || true),
            Resolution::Headless(HeadlessReason::NothingToDraw)
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
            resolve_with(&world, true, || true),
            Resolution::Headless(HeadlessReason::Requested)
        );
    }

    // An AppConfig is how a world sets its home and budgets; declaring one
    // must not turn the renderer off by itself.
    #[test]
    fn an_app_config_that_does_not_ask_for_headless_leaves_the_choice_to_content() {
        let mut world = World::new();
        world.add_component(Prop::default());
        world.add_component(AppConfig::default());

        assert_ne!(
            resolve_with(&world, true, || true),
            Resolution::Headless(HeadlessReason::Requested)
        );
    }

    // The authoring answer reads the world and nothing else, so it says
    // "rendered" for a world with geometry on any machine -- including a build
    // with no backend.
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

    // Content that draws on a machine with no GPU, or a build with no backend,
    // is refused rather than run headless; the opt-out still runs it, and a
    // world with nothing to draw never needs one.
    #[test]
    fn content_that_draws_needs_a_renderer_unless_it_opts_out() {
        let mut world = World::new();
        world.add_component(Prop::default());
        assert_eq!(
            resolve_with(&world, true, || false),
            Resolution::Unavailable(NoRenderer::NoGpu)
        );
        assert_eq!(
            resolve_with(&world, false, || true),
            Resolution::Unavailable(NoRenderer::NoBackend)
        );
        assert_eq!(resolve_with(&world, true, || true), Resolution::Rendered);
        assert_eq!(
            Resolution::Unavailable(NoRenderer::NoGpu).mode(),
            RenderMode::Headless
        );

        world.add_component(AppConfig {
            headless: true,
            ..Default::default()
        });
        assert_eq!(
            resolve_with(&world, false, || false),
            Resolution::Headless(HeadlessReason::Requested)
        );

        let mut physics = World::new();
        physics.add_component(PhysicsConfig::default());
        assert_eq!(
            resolve_with(&physics, false, || false),
            Resolution::Headless(HeadlessReason::NothingToDraw)
        );
    }
}
