//! The machine's half of how a world runs: the build's render backend and the
//! GPU probe, fed to `concinnity_core::render::render_mode::resolve_with`.

use concinnity_core::ecs::World;
use concinnity_core::render::render_mode::{Resolution, resolve_with};

/// Resolve how `world` runs on this build and machine, logging a headless
/// outcome so a run that did not open a window says why. The GPU probe only
/// happens for a world that would otherwise open a window.
pub fn resolve_and_report(world: &World) -> Resolution {
    let resolution = resolve_with(world, crate::device::AVAILABLE, || {
        crate::device::probe_device().is_some()
    });
    if let Resolution::Headless(reason) = resolution {
        tracing::info!("Running headless: {}", reason.as_str());
    }
    resolution
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::{PhysicsConfig, TextLabel};
    use concinnity_core::render::render_mode::HeadlessReason;

    // What a world with something to draw lands on depends on this build's
    // backend and the host's GPU, so the assertion is on the reason: it is
    // never turned down for having nothing to draw. One with nothing to draw
    // never reaches the probe.
    #[test]
    fn only_content_that_draws_reaches_the_machine() {
        let mut world = World::new();
        world.add_component(TextLabel {
            content: "Hello, world!".into(),
            ..Default::default()
        });
        assert_ne!(
            resolve_and_report(&world),
            Resolution::Headless(HeadlessReason::NothingToDraw)
        );

        let mut physics = World::new();
        physics.add_component(PhysicsConfig::default());
        assert_eq!(
            resolve_and_report(&physics),
            Resolution::Headless(HeadlessReason::NothingToDraw)
        );
    }
}
