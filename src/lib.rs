//! Concinnity is a graphics application framework: construct an [`App`],
//! populate its [`World`] with components, and run it on the engine's
//! runtime loop.
//!
//! # Creating an application
//!
//! A [`World`] is constructed first, then handed to an [`App`], which runs
//! it.
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
//!     App::from_world(world).run()
//! }
//! ```
//!
//! A [`GraphicsConfig`](assets::GraphicsConfig) is what gives the app a
//! window.

pub use concinnity_engine::App;
pub use concinnity_engine::ecs::World;
pub use concinnity_memory::install_global_allocator;

/// The runtime asset vocabulary (`Application`, `Camera3D`, `Room`,
/// `DirectionalLight`, ...), each addable to a [`World`] as a component.
pub mod assets {
    pub use concinnity_engine::assets::*;
}

#[cfg(feature = "cook")]
pub mod cook;

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

        let mut app = App::from_world(world);
        assert_eq!(app.start(), Ok(()));
    }
}
