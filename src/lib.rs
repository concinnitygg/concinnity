//! Concinnity is a graphics application framework. A [`World`] holds the
//! components describing what exists -- a camera, lights, geometry, text -- and
//! an [`App`] runs that world on the engine's loop. Behaviour is declared as data
//! rather than assembled from calls, so an application's job is to hand over a
//! world and let the runtime drive it.
//!
//! # Running a compiled world
//!
//! A shipped application usually plays a world that was compiled ahead of time.
//! That needs nothing but the runtime, which is the crate's default build:
//!
//! ```rust,no_run
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
//! What is not a component but can be computed -- generated geometry,
//! image-based lighting, the built-in font -- is baked by the [`bake`](mod@bake)
//! module's functions and handed to the world's data-entry methods
//! ([`World::add_mesh`], [`World::add_material`],
//! [`World::add_environment_map`]), which return the handle a component
//! references the result by. Every build of the crate carries `bake`; only an
//! asset that has to be read from a file needs the `cook` importers.
//!
//! Adding a [`GraphicsConfig`](components::GraphicsConfig) is what opens a
//! window. A world without one still runs, with everything but the rendering,
//! which is how a test or a simulation-only tool drives one:
//!
//! [`App::into_headless`] is the other way to that: it runs any world on the
//! headless loop, including one authored to be seen.
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
//! # Composing vs Cooking
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
//! `--features cook` adds the `cook` module described above. It carries the authoring half
//! of the vocabulary (textures, meshes, prefabs, menus) and pulls in the
//! importers that read them (glTF, FBX, images, fonts), so an application that
//! only plays an already-compiled world should leave it off.
//!
//! `--features metal`, `--features directx` and `--features vulkan` each name
//! one rendering backend. A backend the target does not have is inert, so a
//! build can name more than one and get the one that applies; where two do,
//! Vulkan wins. `native` names all three and is what the default build
//! carries: Metal on macOS, DirectX on Windows, Vulkan elsewhere.
//!
//! `--no-default-features --features std` keeps the operating system and drops
//! the renderer: no GPU code is in the graph at all, and every world runs on
//! the headless loop, which is what a simulation-only tool or a build host with
//! no GPU wants.
//!
//! `--features player` builds `concinnity-run`, the standalone binary that plays
//! a compiled world, and `--features editor` builds both it and the `concinnity`
//! command-line tool. Both are for working on an application rather than for
//! linking one: nothing in the library's own API depends on either, and with
//! neither enabled this crate builds no build dependency at all.
//!
//! `--no-default-features` leaves the core `no_std` runtime: the component
//! vocabulary, a [`World`] to hold it, and the headless loop that steps it,
//! with no operating system underneath. It is the same loop
//! [`App::into_headless`] selects on any tier, so what a test runs is what a
//! `no_std` build runs. [`App::run`] reports the same [`Error`] everywhere,
//! so what runs a world ports between tiers unchanged.

#![cfg_attr(not(feature = "std"), no_std)]

// The world vocabulary is heap-backed on every tier, so the facade names
// `alloc` rather than `std` wherever it holds one.
extern crate alloc;

// The test harness is a std program whichever tier is built, so the `no_std`
// build still gets one.
#[cfg(all(test, not(feature = "std")))]
extern crate std;

mod app;
pub mod bake;
mod driver;
mod error;
mod world;

pub use app::App;
pub use error::Error;
pub use world::World;

// The status a failed call reports, carried by [`Error`] and returned by the
// world's own systems.
pub use concinnity_core::result::CnResult;

// The dense per-kind handles the world's data-entry methods return and a
// component's reference fields hold, and the bound on what `add_mesh` takes.
pub use concinnity_core::ecs::{BakedMesh, EnvironmentMapHandle, MaterialHandle, MeshHandle};

// The identity a component names another by (a `Prop`'s `parent`, a `Model`'s
// meshes). A world built here assigns these itself: an authoring build interns
// them from declared names, which is not a step a Rust-built world takes.
pub use concinnity_core::ecs::asset_id::AssetId;

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

#[cfg(all(test, feature = "cook"))]
mod bake_parity_tests;
#[cfg(test)]
mod namespaces;
#[cfg(test)]
mod test_support;

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

    // The documented headless case, on whichever tier is built: the same
    // starter world without a GraphicsConfig, which is what lets a test or a
    // simulation-only tool drive a world with no window.
    #[test]
    fn a_world_without_graphics_starts_headless() {
        let mut app = super::App::from_world(starter_world());
        assert_eq!(app.inner_mut().start(), Ok(()));
    }

    // Without `std` the same app runs to completion in process: the headless
    // loop steps the world it was handed until nothing is left to step. The
    // std tier's `run` opens a window and waits for it, so it has no place in
    // a test.
    #[cfg(not(feature = "std"))]
    #[test]
    fn a_headless_app_runs_the_world_it_was_given() {
        let app = super::App::from_world(starter_world());
        assert_eq!(app.run(), Ok(()));
    }

    fn starter_world() -> World {
        let mut world = World::new();
        world.add_component(TextLabel {
            content: "Hello, world!".into(),
            ..Default::default()
        });
        world
    }
}
