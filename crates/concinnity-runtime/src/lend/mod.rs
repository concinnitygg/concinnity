//! What this host lends core: the system table, the `SystemHost` seams core's
//! gates build their systems through, and the resources only this host can
//! publish at world start.
//!
//! The table holds core's systems (`CoreSystems`, each entry defined once in
//! concinnity-engine) with this crate's own among them. Table order is run order.
//! A core system's phase, edges, gate and description come from its core entry;
//! this file decides only where this crate's systems sit. A new core system is
//! listed here in its run position after its entry is added to `CoreSystems`.
//!
//! Entries are in phase order (see core's `Phase` for what each band holds),
//! so a system registered with `World::add_system` anchors to a phase, never an
//! entry.
//!
//! This crate's edges, and why:
//!   * Graphics after the overlay, the sky, the drains and streaming: the
//!     frame it submits shows the tick's input, simulation and cameras. Only
//!     the frame report follows it.
//!
//! Event-carried couplings (`RootMotionEvent`, `GroundProbes`, `SettingCommand`) are
//! order-robust thanks to the event store's two-frame retention.

use concinnity_core::fanout::Lent;
use concinnity_engine::audio::out::AudioOut;
use concinnity_engine::behavior::BehaviorStore;
use concinnity_engine::ecs::{
    CoreSystems, Phase, PipelineContext, SystemEntry, SystemHost, SystemTable, TransientSaves,
    World, boxed,
};
use concinnity_engine::input::system::PadSource;
use concinnity_engine::planet::StartBuilder;
use concinnity_engine::settings::system::SettingsStore;
use concinnity_engine::story::StoryStore;
use concinnity_host::store::paths::StateTree;
use concinnity_host::thread::jobs::{self, JobPool};

use crate::render::system::GraphicsSystem;

#[cfg(debug_assertions)]
mod access_check;
mod ground;
mod resources;

// What this host lends core's systems: the job pool, the planet's ground
// worker, the kira audio output, gilrs gamepads, the file stores behind
// behavior state, settings and story saves, and stdout for a benchmark's
// report.
struct Host;

impl SystemHost for Host {
    type Pool = JobPool;

    const POOL: Lent<JobPool> = Lent::new(jobs::pool_for);

    const GROUND: StartBuilder = ground::start_ground_worker;

    fn behavior_store(world: &World) -> Option<Box<dyn BehaviorStore>> {
        let tree = world.resource::<StateTree>()?;
        Some(Box::new(crate::persist::behavior::FileStore::new(tree)))
    }

    fn audio_out() -> Option<Box<dyn AudioOut>> {
        crate::audio::open()
    }

    fn pads() -> Option<Box<dyn PadSource>> {
        Some(Box::new(crate::gamepad::GamepadSource::default()))
    }

    fn settings_store(world: &World) -> Option<Box<dyn SettingsStore>> {
        let tree = world.resource::<StateTree>()?;
        Some(Box::new(crate::persist::settings::FileStore::new(tree)))
    }

    fn story_store(world: &World) -> Option<Box<dyn StoryStore>> {
        let tree = world.resource::<StateTree>()?;
        let transient = world.resource::<TransientSaves>().is_some_and(|t| t.0);
        Some(Box::new(crate::persist::story::FileStore::new(
            tree, transient,
        )))
    }

    fn report(text: &str) {
        print!("{text}");
    }
}

type Core = CoreSystems<Host>;

const GRAPHICS: SystemEntry = SystemEntry {
    name: "GraphicsSystem",
    present_when: "the world runs with a window",
    phase: Phase::PostRender,
    gate: |world| boxed(GraphicsSystem::for_world(world)),
    after: &[
        "OverlaySystem",
        "SkyRotationSystem",
        "SpawnSystem",
        "SettingsSystem",
        "StreamingSystem",
    ],
    before: &["FrameReportSystem"],
};

// Runs once at world start, after the gates have built the systems and before
// their `init`. The defaults the world is completed with land earlier still
// (`complete_world`), so a Prop one of them injects decomposes here like any
// other.
fn before_init(ctx: &mut PipelineContext<'_>) {
    // Every context touch a stepping system makes is asserted against what it
    // declared; the hook is installed before the first step can happen.
    #[cfg(debug_assertions)]
    access_check::install_hook();
    resources::publish(ctx);
    concinnity_engine::ecs::decompose::run(ctx);
}

/// The system table a world starts from: one entry per system, in run order,
/// plus the load-time passes that bracket them.
pub const SYSTEMS: &SystemTable = &SystemTable {
    entries: &[
        Core::OVERLAY,
        Core::INPUT,
        Core::SKY_ROTATION,
        Core::PLANET,
        Core::BEHAVIOR,
        Core::PHYSICS,
        Core::CAMERA3D,
        Core::THIRD_PERSON,
        Core::CAMERA_TRACK,
        Core::ANIMATION,
        Core::STORY,
        Core::AUDIO,
        Core::TEXT_INPUT,
        Core::STAT_HUD,
        Core::DEBUG_HUD,
        Core::FPS_COUNTER,
        Core::SPAWN,
        Core::SETTINGS,
        Core::STREAMING,
        Core::LOADING_OVERLAY,
        Core::UI_INPUT,
        GRAPHICS,
        Core::FRAME_REPORT,
    ],
    // Graphics runs at the tail of the tick, but its init builds the window,
    // the backend and the payload indexes later inits read: it inits right
    // after the entries its own init reads.
    init_first: &[
        "OverlaySystem",
        "SkyRotationSystem",
        "BehaviorSystem",
        "SpawnSystem",
        "SettingsSystem",
        "StreamingSystem",
        "GraphicsSystem",
    ],
    complete_world: Some(concinnity_engine::defaults::run),
    before_init: Some(before_init),
    prepare_events: Some(concinnity_engine::ecs::access_id::prepare_events),
};

#[cfg(test)]
mod tests;
