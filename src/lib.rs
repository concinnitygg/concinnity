//! Concinnity is a graphics application framework. A [`World`] holds the
//! components describing what exists -- a camera, lights, geometry, text -- and
//! an `App` runs that world on the engine's loop. Behaviour is declared as data
//! rather than assembled from calls, so an application's job is to hand over a
//! world and let the runtime drive it.
//!
//! # Running a compiled world
//!
//! A shipped application usually plays a world that was compiled ahead of time.
//! That needs nothing but the runtime, which is the crate's default build:
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
//! Worlds are authored and compiled by the `cook` module, which is where to go
//! next: it declares assets as typed values, resolves the references between
//! them, and either compiles a [`World`] in memory or writes one out for a
//! build like the above to play. It sits behind the `cook` feature, off by
//! default.
#![cfg_attr(feature = "cook", doc = " Start at [`cook`](mod@cook).")]
//!
//! # Assembling a world in code
//!
//! Compiling is not always needed. Every type in the [`components`] vocabulary
//! can be handed straight to [`World::add_component`], so an application built
//! only from those runs on the default feature set alone -- no authoring step
//! and no file on disk.
//!
//! Adding a [`GraphicsConfig`](components::GraphicsConfig) is what opens a
//! window. A world without one still runs, with everything but the rendering,
//! which is how a test or a simulation-only tool drives one:
//!
//! ```no_run
//! use concinnity::components::{PhysicsConfig, TriggerVolume};
//! use concinnity::{App, World};
//!
//! fn main() {
//!     let mut world = World::new();
//!     world.add_component(PhysicsConfig::default());
//!     world.add_component(TriggerVolume {
//!         position: [0.0, 1.0, 0.0],
//!         ..Default::default()
//!     });
//!
//!     App::from_world(world).run().expect("the app runs");
//! }
//! ```
//!
//! # Choosing between the two
//!
//! Add components directly when a world is small, or when it is decided at
//! runtime and there is nothing to prepare in advance. Reach for the `cook`
//! module when an asset needs work before it can run: an image or model to read
//! from disk, a room to generate geometry for, a prefab to expand into the
//! components it stands for. Both end at the same place, an `App` holding a
//! [`World`], and one application can use both.
//!
//! # Features
//!
//! The default build is the runtime alone: the world loop, the renderer, and
//! the [`components`] vocabulary, which is all the examples above need.
//!
//! `std` is that runtime, and it is on by default. Turning it off leaves
//! [`components`] and a [`World`] to build with it, so a `no_std` crate can
//! assemble world content where no runtime exists. What it drops is everything
//! that runs: there is no `App` and no `cook`, and a [`World`] carries
//! components but nothing to step them.
//!
//! `cook` adds the `cook` module described above. It carries the authoring half
//! of the vocabulary (textures, meshes, prefabs, menus) and pulls in the
//! importers that read them (glTF, FBX, images, fonts), so an application that
//! only plays an already-compiled world should leave it off.
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
