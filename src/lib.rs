//! Concinnity is a graphics application framework. A [`World`] holds the
//! components describing what exists -- a camera, lights, geometry, text -- and
//! an [`App`] runs that world on the engine's loop. Behavior is declared as data
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
//! build like the above to play. It sits behind the `cook` feature, on by
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
//! [`World::add_environment_map`], [`World::add_font`]), which return the
//! handle a component references the result by. Every build of the crate
//! carries `bake`; only an asset that has to be read from a file needs the
//! `cook` importers.
//!
//! A world with something to draw opens a window. One without draws nothing,
//! so it runs headless: everything but the rendering, on a fixed virtual
//! timestep. A machine with no GPU runs headless either way.
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
//! [`AppConfig::headless`](components::AppConfig::headless) keeps a world that
//! could draw from opening a window; [`App::into_headless`] does the same for
//! one run.
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
//! `std` is the runtime: the world loop, the renderer, and the [`components`]
//! vocabulary, which is all the examples above need. The renderer is the one
//! the target draws with: Metal on macOS, DirectX on Windows, Vulkan on Linux.
//! `--features vulkan` swaps in Vulkan wherever the target has it.
//!
//! `cook` adds the `cook` module described above and is on by default. It
//! carries the authoring half of the vocabulary (textures, meshes, prefabs,
//! menus) and pulls in the importers that read them (glTF, FBX, images,
//! fonts), so an application that only plays an already-compiled world can
//! drop it with `--no-default-features --features std`.
//!
//! `player`, on by default, builds `concinnity-run`, the standalone binary that
//! plays a compiled world, and `--features editor` builds both it and the
//! `concinnity` command-line tool. Both are for working on an application
//! rather than for linking one: nothing in the library's own API depends on
//! either.
//!
//! `heap-tracking`, on by default, installs the counting allocator as the
//! program's `#[global_allocator]`, so every application gets heap figures (the
//! stats HUD's allocation chip, memory drift reports, crash-report heap totals)
//! without declaring anything. A program that wants an allocator of its own
//! turns it off and declares one, wrapping it to keep the counts:
//! `concinnity_core::install_global_allocator!(mimalloc::MiMalloc);`.
//!
//! `--no-default-features` leaves the core `no_std` runtime: the component
//! vocabulary, a [`World`] to hold it, and the headless loop that steps it,
//! with no operating system underneath. It is the same loop
//! [`App::into_headless`] selects on any tier, so what a test runs is what a
//! `no_std` build runs. [`App::run`] reports the same [`WorldError`]
//! everywhere, so what runs a world ports between tiers unchanged.
//!
//! # Errors
//!
//! Each step fails with the error of the part that ran it: [`BakeError`] from
//! [`bake`](mod@bake), [`WorldError`] from a world or a run, `StartupError`
//! from `App::from_blob`, and `WorldBuildError` from the `cook` module. All
//! of them are [`core::error::Error`], so a `Box<dyn Error>` carries any mix
//! of them through one `?`.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "heap-tracking")]
concinnity_core::install_global_allocator!();

pub use concinnity_engine::ecs::World;

/// A runnable application: a [`World`] plus the loop that runs it.
#[cfg(feature = "std")]
pub use concinnity_runtime::App;

/// A runnable application: a [`World`] plus the headless loop that runs it.
#[cfg(not(feature = "std"))]
pub use concinnity_engine::App;

pub use concinnity_engine::bake::payload::BakeError;
pub use concinnity_engine::error::{AssetError, PayloadError, WorldError};
#[cfg(feature = "std")]
pub use concinnity_runtime::{StartupError, WorldLoadError};

// Exported at the crate root because `#[macro_export]` puts them there.
// `system` is where both are documented.
pub use concinnity_engine::{component_mask, declare_components};

pub use concinnity_engine::ecs::{EnvironmentMapHandle, FontHandle, MaterialHandle, MeshHandle};

// The identity a component names another by, and the typed reference field
// that holds one (a `Prop`'s `parent`).
pub use concinnity_engine::ecs::Ref;
pub use concinnity_engine::ecs::asset_id::AssetId;

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
    // Curation is at the source (`concinnity_engine::components::stored`),
    // where a type and its registry group are both in view.
    pub use concinnity_engine::components::stored::*;
}

/// What a launch asks of the run, and the flags it is read from.
pub mod launch {
    pub use concinnity_engine::launch::{
        LAUNCH_FLAGS, LaunchArgs, LaunchArgsError, LaunchCommand, LaunchFlag,
        QUALITY_PRESET_VALUES, usage,
    };
    #[cfg(feature = "std")]
    pub use concinnity_runtime::launch_args_from_env as from_env;
}

/// What a standalone player binary is made of: where an app keeps its state,
/// the crash hooks, and the run over a compiled world on disk. Requires the
/// `player` feature.
#[cfg(feature = "player")]
pub mod player {
    pub use concinnity_host::store::paths::{StateTree, per_user_base, tree_for_exe};
    pub use concinnity_runtime::crash::install as install_crash_hooks;
    pub use concinnity_runtime::{BlobSource, run_from};
}

/// The heap figures the counting allocator keeps. `stats` is `None` in a
/// program without it, which the `heap-tracking` feature installs.
pub mod memory {
    pub use concinnity_core::memory::{MemStats, stats};
}

/// Math an application lays its own geometry out with.
pub mod math {
    /// Double-precision 3-vectors, for positions too far out for f32: on and
    /// around a [`Planet`](crate::components::Planet), for one.
    pub use concinnity_core::math::dvec3;
}

pub mod system {
    //! Writing a system: code that runs every tick over a [`World`](crate::World).
    //!
    //! A world's own behavior is data -- components describe what exists, and the
    //! engine's systems run over them. A system is the other half: Rust that runs
    //! on the same tick as the engine's own, over the same world, with the same
    //! borrow of it.
    //!
    //! Implement [`System`] and register it with
    //! [`World::add_system`](crate::World::add_system), naming the [`Phase`] it
    //! runs in:
    //!
    //! ```
    //! use concinnity::components::TextLabel;
    //! use concinnity::system::{Phase, PipelineContext, StepResult, System};
    //! use concinnity::World;
    //!
    //! #[derive(Debug)]
    //! struct Ticker {
    //!     ticks: u32,
    //! }
    //!
    //! impl System for Ticker {
    //!     fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
    //!         self.ticks += 1;
    //!         for label in ctx.query_mut::<TextLabel>() {
    //!             label.content = format!("tick {}", self.ticks);
    //!         }
    //!         StepResult::Continue
    //!     }
    //! }
    //!
    //! let mut world = World::new();
    //! world.add_component(TextLabel::default());
    //! world.add_system(Phase::Late, "Ticker", Ticker { ticks: 0 });
    //! ```
    //!
    //! # State
    //!
    //! A system owns whatever it needs: the value registered is the value stepped,
    //! so per-system state is ordinary struct fields. State several systems share
    //! belongs in a resource -- any `Send` type, published with
    //! [`PipelineContext::insert_resource`] and read back by type.
    //!
    //! Per-entity state belongs in a component of your own. The
    //! [`components`](crate::components) vocabulary is the engine's and does not
    //! grow, but an application declares its own types beside it with
    //! [`declare_components!`](crate::declare_components), and they behave like any
    //! other component from there on:
    //!
    //! ```
    //! use concinnity::components::Transform;
    //! use concinnity::system::{PipelineContext, StepResult, System};
    //! use concinnity::{World, declare_components};
    //!
    //! #[derive(Debug)]
    //! struct Velocity([f32; 3]);
    //!
    //! declare_components!(Velocity);
    //!
    //! #[derive(Debug)]
    //! struct Motion;
    //!
    //! impl System for Motion {
    //!     fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
    //!         let moving: Vec<_> = ctx
    //!             .join2::<Velocity, Transform>()
    //!             .map(|(entity, v, _)| (entity, v.0))
    //!             .collect();
    //!         for (entity, velocity) in moving {
    //!             if let Some(t) = ctx.get_mut::<Transform>(entity) {
    //!                 for axis in 0..3 {
    //!                     t.position[axis] += velocity[axis];
    //!                 }
    //!             }
    //!         }
    //!         StepResult::Continue
    //!     }
    //! }
    //!
    //! let mut world = World::new();
    //! let mover = world.spawn();
    //! world.insert(mover, Velocity([1.0, 0.0, 0.0]));
    //! world.insert(mover, Transform::default());
    //! ```
    //!
    //! A declared component is runtime-only: it has no authoring name and no blob
    //! record, so it cannot be written in a world file or survive a save. Seed it
    //! from Rust, with [`World::spawn`](crate::World::spawn) and
    //! [`World::insert`](crate::World::insert) before the run or
    //! [`PipelineContext::push`] during it.
    //!
    //! # Declaring access
    //!
    //! [`System::access`] says what a step touches. The default claims everything,
    //! which is always correct and always safe; declaring narrower access is what
    //! turns on the debug-build check that a step touched only what it said it
    //! would. Build the masks with [`component_mask!`](crate::component_mask):
    //!
    //! ```
    //! # use concinnity::components::{TextLabel, Transform};
    //! # use concinnity::component_mask;
    //! # use concinnity::system::Access;
    //! # fn declare() -> Access {
    //! Access::new()
    //!     .reads_components(component_mask![Transform])
    //!     .writes_components(component_mask![TextLabel])
    //! # }
    //! ```
    //!
    //! # Moving the camera
    //!
    //! A system that moves the [`Camera3D`](crate::components::Camera3D), or
    //! what it follows, somewhere it did not travel to, as at a respawn or a shot
    //! change, calls [`raise_camera_cut`] on the same tick. The first frame drawn
    //! from the moved camera then starts its temporal history afresh instead of
    //! blending with the view before the jump:
    //!
    //! ```
    //! use concinnity::components::Camera3D;
    //! use concinnity::system::{PipelineContext, StepResult, System, raise_camera_cut};
    //!
    //! #[derive(Debug)]
    //! struct Respawn;
    //!
    //! impl System for Respawn {
    //!     fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
    //!         for camera in ctx.query_mut::<Camera3D>() {
    //!             camera.set_pose([0.0, 1.7, 0.0], 0.0, 0.0);
    //!         }
    //!         raise_camera_cut(ctx);
    //!         StepResult::Done
    //!     }
    //! }
    //! ```

    pub use concinnity_engine::ecs::{
        Access, ComponentId, ComponentMask, ComponentSlot, Entity, Phase, PipelineContext,
        StepResult, System,
    };
    pub use concinnity_engine::render::temporal::history_reset::raise_camera_cut;
}

pub mod bake {
    //! Bake payloads from typed values, entirely in memory.
    //!
    //! Everything here is pure computation over the value it is handed -- a
    //! generator's geometry, an image-based-lighting convolution, the built-in
    //! face's glyph atlas -- so it needs no files, no importers, and no operating
    //! system: this module is available on every build of the crate,
    //! `--no-default-features` included.
    //!
    //! The baked payloads go into a [`World`](crate::World) through its data-entry
    //! methods ([`add_mesh`](crate::World::add_mesh),
    //! [`add_environment_map`](crate::World::add_environment_map),
    //! [`add_font`](crate::World::add_font)), which hand back the handle a
    //! component references the result by. Each bake returns its own payload type,
    //! so a payload only fits the method for its kind:
    //!
    //! ```no_run
    //! use concinnity::components::{DirectionalLight, ProceduralMesh, Prop};
    //! use concinnity::{App, World, bake};
    //!
    //! fn main() -> Result<(), Box<dyn std::error::Error>> {
    //!     let mut world = World::new();
    //!     world.add_component(DirectionalLight::default());
    //!
    //!     let mesh = ProceduralMesh {
    //!         generator: "box".into(),
    //!         ..Default::default()
    //!     };
    //!     let mesh = world.add_mesh(bake::procedural_mesh(mesh)?)?;
    //!     let stone = world.add_material(bake::Material::default());
    //!
    //!     world.add_component(Prop {
    //!         mesh: Some(mesh),
    //!         material: Some(stone),
    //!         ..Default::default()
    //!     });
    //!
    //!     App::from_world(world).run()?;
    //!     Ok(())
    //! }
    //! ```
    //!
    //! # Baking vs cooking
    //!
    //! Reach for this module when everything a world needs can be computed: the
    //! built-in generators, the built-in font, image-based lighting from a
    //! generator. Reach for the `cook` module when an asset has to be *read* -- a
    //! model or texture from disk, a shader to compile, a prefab to expand --
    //! which is what its importers are for. A value this module cannot bake (a
    //! `source` naming a file, a generator that decodes an image) is refused with
    //! an error naming the cook module.

    /// The bakeable types that [`components`](crate::components) does not
    /// carry.
    ///
    /// `EnvironmentMap`, `Font`, `Material` and `Mesh` are resources: a world
    /// reaches each by handle, so a value that uses one holds the handle its
    /// `World::add_*` method returned rather than the value itself. `Camera3D`
    /// and `CameraTrack` are the authored forms of the components of the same
    /// names, which those components' own `bake` turns into the runtime value.
    /// The `cook` module carries these same types under its own namespace,
    /// along with the ones that need an importer.
    pub use concinnity_engine::components::cook::{
        Camera3D, CameraTrack, EnvironmentMap, Font, Material, Mesh, VertexData,
    };

    pub use concinnity_engine::bake::payload::{
        BakeError, EnvironmentMapPayload, FontPayload, MeshPayload, font, mesh, procedural_mesh,
    };

    /// Convolve an [`EnvironmentMap`]'s generator into its
    /// image-based-lighting payload, for
    /// [`World::add_environment_map`](crate::World::add_environment_map).
    ///
    /// This is the expensive bake -- hundreds of millions of float operations
    /// at default sizes. On the std tier the convolutions are spread over the
    /// engine's job pool; without it they run on the calling thread.
    ///
    /// # Errors
    ///
    /// Returns an error if the map names a `source` file, its generator is
    /// missing or unknown, or its sizes are out of range.
    pub fn environment_map(map: &EnvironmentMap) -> Result<EnvironmentMapPayload, BakeError> {
        #[cfg(feature = "std")]
        let rows = concinnity_host::thread::jobs::pool();
        #[cfg(not(feature = "std"))]
        let rows = &concinnity_core::fanout::Inline;
        concinnity_engine::bake::payload::environment_map(map, rows)
    }
}

#[cfg(feature = "cook")]
pub mod cook {
    //! Compile authored worlds into runnable [`World`](crate::World)s, entirely in memory.
    //!
    //! Requires the `cook` feature, which is on by default. An application that only
    //! plays compiled worlds can drop it to shed the importers this module needs.
    //!
    //! An application declares what it wants -- a texture from this file, a room of
    //! that size -- and the runtime plays only the finished result. This module is
    //! the step between: it checks the declarations, expands the ones that stand
    //! for several assets, prepares each asset's data, and then either assembles a
    //! [`World`](crate::World) to run straight away or writes the result to a file for a later
    //! run to play. Prepared payloads are cached under the build root a host
    //! installs -- the dev CLI's is `.concinnity/` inside the project -- so a
    //! second compile with unchanged sources skips the expensive work. A process
    //! that installed none has nowhere to cache and prepares them every time.
    //!
    //! # The authoring vocabulary
    //!
    //! This module is also the other half of the asset vocabulary: the types a
    //! world declares and the cook consumes, which never reach a running world as
    //! components. Those are the build-only assets it expands (`Prefab`,
    //! `MainMenu`, `CharacterSchema`, ...) and the resources it prepares
    //! (`Texture`, `Mesh`, `Material`, `Font`, ...). The stored half is
    //! [`components`](crate::components).
    //!
    //! Five assets are authored as something other than what they bake into, and
    //! are named here for the asset they declare: `cook::AppConfig`,
    //! `cook::Camera3D`, `cook::File`, `cook::Room`, and `cook::Spawner` are the
    //! authored forms of the components of the same name. That makes those five
    //! names ambiguous when both namespaces are glob-imported, so glob
    //! [`components`](crate::components) and path-qualify this module.
    //!
    //! # Declaring a world in code
    //!
    //! Each asset is declared by its own authored struct, so the type is carried
    //! by the value rather than spelled as a string:
    //!
    //! ```no_run
    //! use concinnity::App;
    //! use concinnity::components::DirectionalLight;
    //! use concinnity::cook;
    //!
    //! fn main() {
    //!     let world = cook::world()
    //!         .add("sun", &DirectionalLight {
    //!             color: [1.0, 0.96, 0.86],
    //!             direction: [-0.35, 0.85, 0.35],
    //!             intensity: 2.2,
    //!             ..Default::default()
    //!         })
    //!         .add("room", &cook::Room {
    //!             size: Some([16.0, 20.0, 5.0]),
    //!             ..Default::default()
    //!         })
    //!         .compile()
    //!         .expect("the declared world compiles");
    //!
    //!     App::from_world(world).run().expect("the app runs");
    //! }
    //! ```
    //!
    //! A field that references another asset holds a resolved handle rather than
    //! a name, so the name is given alongside the value with
    //! [`reference`](WorldBuilder::reference):
    //!
    //! ```no_run
    //! # use concinnity::components::Prop;
    //! # use concinnity::cook;
    //! cook::world()
    //!     .add("stone", &cook::Material { roughness: 0.9, ..Default::default() })
    //!     .add("pillar", &Prop::default())
    //!     .reference("material", "stone");
    //! ```
    //!
    //! Most assets are plain runtime components, so one needing no preparation can
    //! equally be added straight to a [`World`](crate::World) with
    //! [`add_component`](crate::World::add_component). A [`Room`] is what this module is
    //! for: its geometry is generated here, and its texture names become the
    //! handles the runtime reads, neither of which exists beforehand.
    //!
    //! # Compiling ahead of time
    //!
    //! [`write_blob`](WorldBuilder::write_blob) takes the same declarations and
    //! writes them to a file instead of building a world. That file is a *blob*:
    //! the compiled form of a world, holding the components and the prepared asset
    //! data together. Producing one moves the preparation to a build tool, off
    //! every launch. The shipped application plays it with
    //! [`App::from_blob`](crate::App::from_blob) and needs neither this module nor
    //! the importers behind it.
    //!
    //! ```no_run
    //! # use concinnity::components::DirectionalLight;
    //! # use concinnity::cook;
    //! cook::world()
    //!     .add("sun", &DirectionalLight::default())
    //!     .write_blob("data/0")
    //!     .expect("the world is written to data/0");
    //! ```

    pub use concinnity_cook::authoring::registry::Authored;
    pub use concinnity_cook::{WorldBuildError, WorldBuilder};

    // The authoring vocabulary, from the two crates that own its halves: the
    // compiled resources and the five diverging args schemas from the runtime
    // crate, the build-only assets from the authoring one.
    pub use concinnity_cook::authoring::registry::build_only::{
        CameraShot, CharacterModel, CharacterSchema, Include, KeyPolarity, LightRig, MainMenu,
        MainMenuItem, MaterialPalette, OptionSelect, PaletteEntry, Panel, PanelSection, Prefab,
        PrefabEntry, PrefabKind, ProportionGroup, SceneImport, SchemaJoint, SchemaKey,
        SchemaRegion, SettingsProfile, ShapePreset, Slider, StoryImport, SynthParams,
        SynthesizedTarget,
    };
    pub use concinnity_engine::components::cook::*;

    /// Start an empty world, cooking shaders for every backend this build can
    /// render with, so a world compiled in memory runs in the same process and
    /// a blob it writes plays on whichever one a launch selects.
    pub fn world() -> WorldBuilder {
        concinnity_cook::world(concinnity_runtime::platform::compiled_set())
    }
}
