//! Concinnity is a graphics application framework: construct an [`App`],
//! populate its [`World`] with assets, and run it on the engine's runtime
//! loop.
//!
//! The runtime plays compiled blobs. Worlds are authored either as
//! `world.jsonl` declarations or as typed specs, and compiled in memory with
//! the [`world`] module (behind the default-on `world` feature). The feature
//! gates authoring worlds from source, not worlds themselves: the [`World`]
//! type is always present, so a shipped player that only loads prebuilt
//! blobs can depend on this crate with `default-features = false`, which
//! reduces the dependency set to the runtime tier.
//!
//! # Creating an application
//!
//! A [`World`] is constructed first, then handed to an [`App`], which runs
//! it. This is the `cn init` starter world written in code: a window and a
//! line of text.
//!
//! ```no_run
//! use concinnity::assets::{GraphicsConfig, TextLabel};
//! use concinnity::{App, World};
//!
//! fn main() -> std::io::Result<()> {
//!     let mut world = World::new();
//!     world.add_component(GraphicsConfig::default());
//!     world.add_component(TextLabel {
//!         content: "Hello, world!".to_string(),
//!         ..Default::default()
//!     });
//!
//!     let mut app = App::new();
//!     app.load_world(world);
//!     app.run()
//! }
//! ```
//!
//! A [`GraphicsConfig`](assets::GraphicsConfig) is what gives the app a
//! window. Without one the world runs headless, which is what a test or a
//! simulation-only tool wants.
//!
//! # Loading compiled blobs and running
//!
//! A shipped app plays the blobs a prior build wrote under the state dir:
//!
//! ```no_run
//! use concinnity::{App, init_logging};
//!
//! fn main() -> std::io::Result<()> {
//!     init_logging();
//!     let mut app = App::new();
//!     app.load_blob().expect("compiled blobs under the state dir");
//!     app.run()
//! }
//! ```
//!
//! For a packaged binary whose content dir sits beside the executable,
//! [`run_from`] wraps the same flow with the state root pinned there.
//!
//! # Authoring a world in code
//!
//! With the `world` feature, typed asset specs or `world.jsonl` content
//! compile straight into a runnable [`World`]; see [`world`] for the
//! examples.

pub use concinnity_engine::ecs::World;
pub use concinnity_engine::{
    App, PipelineMode, RunOptions, init_logging, run_from, set_writable_state_dir,
};
pub use concinnity_memory::install_global_allocator;

/// The runtime asset vocabulary (`Application`, `Camera3D`, `Room`,
/// `DirectionalLight`, ...), each addable to a [`World`] as a component.
pub mod assets {
    pub use concinnity_engine::assets::*;
}

/// State-tree anchoring: where `.concinnity/` (blobs, caches, saves,
/// settings) resolves from.
pub mod paths {
    pub use concinnity_store::paths::*;
}

#[cfg(feature = "world")]
pub mod world;

#[cfg(test)]
mod tests {
    use super::assets::TextLabel;
    use super::{App, World};

    // The documented headless case: the same starter world without a
    // GraphicsConfig, which is what lets a test or a simulation-only tool
    // drive a world with no window.
    #[test]
    fn a_world_without_graphics_starts_headless() {
        let mut world = World::new();
        world.add_component(TextLabel {
            content: "Hello, world!".to_string(),
            ..Default::default()
        });

        let mut app = App::new();
        app.load_world(world);
        assert_eq!(app.start(), Ok(()));
    }
}
