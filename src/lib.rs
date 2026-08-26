//! Concinnity is a graphics application framework: construct an `App`,
//! populate its [`World`] with components, and run it on the engine's
//! runtime loop.
//!
//! # Running an application
//!
//! The most common use-case for a client will be running an `App` from
//! locally saved binary data.
//!
//! ```no_run
//! use concinnity::App;
//!
//! fn main() {
//!     App::from_blob("my_game.cnb")
//!         .expect("my_game.cnb holds a compiled world")
//!         .run()
//!         .expect("the app runs");
//! }
//! ```
//!
//! The `cook` module manages this world binary data.
//!
//! # Features
//!
//! The default build is the runtime alone: the world loop, the renderer, and
//! the [`components`] vocabulary, which is all the example needs.
//!
//! `std` is that runtime, and it is on by default. Turning it off leaves
//! [`components`] and a [`World`] to build with it, so a `no_std` crate can
//! assemble world content where no runtime exists. What it drops is everything
//! that runs: there is no `App` and no `cook`, and a [`World`] carries
//! components but nothing to step them.
//!
//! `cook` adds the `cook` module, which compiles authored assets into a
//! runnable [`World`] in process, or writes them to a blob file for a shipped
//! application to play. It carries the authoring-only half of the vocabulary
//! (textures, meshes, prefabs, menus) and pulls in the asset importers (glTF,
//! FBX, textures, fonts), so an application that only plays an
//! already-compiled world should leave it off.
//!
//! `vulkan` selects the Vulkan backend where the platform default is Metal or
//! DirectX.

#![cfg_attr(not(feature = "std"), no_std)]

// The test harness is a std program whichever tier is built, so the `no_std`
// build still gets one.
#[cfg(all(test, not(feature = "std")))]
extern crate std;

#[cfg(feature = "std")]
mod app;
mod world;

#[cfg(feature = "std")]
pub use app::App;
pub use world::World;

/// The runtime component vocabulary (`Camera3D`, `Room`, `DirectionalLight`,
/// `Transform`, ...): every type a [`World`] holds, each addable with
/// [`add_component`](World::add_component).
///
/// A type an application authors but a world never stores -- a `Texture`, a
/// `Prefab`, a `MainMenu` -- belongs to the `cook` namespace instead. Where a
/// name is in both, the authoring form is `cook::X` and the runtime form
/// `components::X`; glob this module and path-qualify `cook`, since the twinned
/// names are ambiguous under two globs.
pub mod components {
    // The curated stored group, globbed rather than named one by one. Curation
    // is at the source (`concinnity_core::components::stored`), where a type
    // and its registry group are both in view; `namespaces` pins the two to
    // each other.
    pub use concinnity_core::components::stored::*;
}

#[cfg(feature = "cook")]
pub mod cook;

#[cfg(test)]
mod namespaces;

#[cfg(test)]
mod tests {
    use super::World;
    use super::components::TextLabel;

    // The starter world's first two lines, on whichever tier is built. Without
    // `std` this is the whole of what the facade offers, so it is also the
    // check that a world is constructible with no runtime behind it.
    #[test]
    fn a_world_holds_the_components_added_to_it() {
        let mut world = World::new();
        world.add_component(TextLabel {
            content: "Hello, world!".into(),
            ..Default::default()
        });

        let inner = world.inner();
        assert_eq!(inner.component_count(), 1);
        assert_eq!(
            inner.query::<TextLabel>().next().unwrap().content,
            "Hello, world!"
        );
    }

    // The documented headless case: the same starter world without a
    // GraphicsConfig, which is what lets a test or a simulation-only tool
    // drive a world with no window.
    #[cfg(feature = "std")]
    #[test]
    fn a_world_without_graphics_starts_headless() {
        let mut world = World::new();
        world.add_component(TextLabel {
            content: "Hello, world!".into(),
            ..Default::default()
        });

        let mut app = super::App::from_world(world);
        assert_eq!(app.inner_mut().start(), Ok(()));
    }
}
