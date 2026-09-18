//! Compile authored worlds into runnable [`World`]s, entirely in memory.
//!
//! Requires the `cook` feature, which is off by default: the importers this
//! module needs are build-time weight a shipped application does not carry.
//!
//! An application declares what it wants -- a texture from this file, a room of
//! that size -- and the runtime plays only the finished result. This module is
//! the step between: it checks the declarations, expands the ones that stand
//! for several assets, prepares each asset's data, and then either assembles a
//! [`World`] to run straight away or writes the result to a file for a later
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
//!         .add("sun", DirectionalLight {
//!             color: [1.0, 0.96, 0.86],
//!             direction: [-0.35, 0.85, 0.35],
//!             intensity: 2.2,
//!         })
//!         .add("room", cook::Room {
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
//!     .add("stone", cook::Material { roughness: 0.9, ..Default::default() })
//!     .add("pillar", Prop::default())
//!     .reference("material", "stone");
//! ```
//!
//! Most assets are plain runtime components, so one needing no preparation can
//! equally be added straight to a [`World`] with
//! [`add_component`](World::add_component). A [`Room`] is what this module is
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
//!     .add("sun", DirectionalLight::default())
//!     .write_blob("data/0")
//!     .expect("the world is written to data/0");
//! ```

use std::path::Path;

pub use concinnity_cook::authoring::registry::Authored;

// The authoring vocabulary, from the two crates that own its halves: the
// compiled resources and the five diverging args schemas from the runtime
// crate, the build-only assets from the authoring one.
pub use concinnity_cook::authoring::registry::build_only::{
    CameraShot, CharacterModel, CharacterSchema, KeyPolarity, LightRig, MainMenu, MainMenuItem,
    MaterialPalette, OptionSelect, PaletteEntry, Panel, PanelSection, Prefab, PrefabEntry,
    PrefabKind, ProportionGroup, SceneImport, SchemaJoint, SchemaKey, SchemaRegion,
    SettingsProfile, ShapePreset, Slider, StoryImport, SynthParams, SynthesizedTarget,
};
pub use concinnity_core::components::cook::*;

use crate::{World, error};

/// A world under construction: typed authored assets, compiled together into
/// a runnable [`World`] or a blob file.
pub struct WorldBuilder(concinnity_cook::WorldBuilder);

/// Start an empty world.
pub fn world() -> WorldBuilder {
    // Shaders are cooked for the backend the runtime linked in beside this
    // module consumes, so a world compiled in memory runs in the same process.
    WorldBuilder(concinnity_cook::world(
        concinnity_engine::platform::current(),
    ))
}

impl WorldBuilder {
    /// Resolve bare `source` filenames under `dir`, the way a build resolves
    /// them under a state tree's `assets/`. Without one only a path that stands
    /// on its own resolves, since nothing here guesses a root.
    pub fn assets_in(&mut self, dir: impl Into<std::path::PathBuf>) -> &mut Self {
        self.0.assets_in(dir);
        self
    }

    /// The asset search root this build resolves against, if one was named.
    pub fn assets_dir(&self) -> Option<&Path> {
        self.0.assets_dir()
    }

    /// Declare `value` under the `$id` `id`, the name a reference to it uses.
    /// The asset type comes from the value's own [`Authored`] impl, so it
    /// cannot disagree with the fields.
    pub fn add<T: Authored>(&mut self, id: impl Into<String>, value: T) -> &mut Self {
        self.0.add(id, value);
        self
    }

    /// The assets declared so far, as `(id, type)` pairs in declaration
    /// order. Declaration order is load-bearing for scenes: the first `Scene`
    /// is the one active at world start.
    pub fn declared(&self) -> impl Iterator<Item = (&str, &str)> {
        self.0.declared()
    }

    /// Point a reference field of the asset just added at `target`, by its
    /// `$id`.
    ///
    /// A reference on an authored struct holds a resolved handle (a dense
    /// index the compile assigns in declaration order), so the typed value
    /// cannot carry the name it points at. This writes the name into the
    /// pending declaration, where the compile resolves it.
    ///
    /// ```no_run
    /// # use concinnity::components::{CharacterShape, ShapeSlider};
    /// # use concinnity::cook;
    /// cook::world()
    ///     .add(
    ///         "hero_shape",
    ///         CharacterShape {
    ///             sliders: vec![ShapeSlider { name: "weight".into(), value: 0.4 }],
    ///             ..Default::default()
    ///         },
    ///     )
    ///     .reference("target", "hero");
    /// ```
    pub fn reference(&mut self, field: &str, target: impl Into<String>) -> &mut Self {
        self.0.reference(field, target);
        self
    }

    /// Compile every declared asset into a runnable [`World`].
    pub fn compile(&self) -> Result<World, crate::Error> {
        self.0
            .compile()
            .map(World::from_inner)
            .map_err(error::from_cook)
    }

    /// Compile every declared asset and write it to the blob file at `path`.
    /// Payloads too large for one blob spill into siblings named by index, so
    /// a world written to `data/0` may also write `data/1`, `data/2`, ...
    /// [`App::from_blob`](crate::App::from_blob) reads that layout back.
    pub fn write_blob(&self, path: impl AsRef<Path>) -> Result<(), crate::Error> {
        self.0.write_blob(path).map_err(error::from_cook)
    }
}

// The wrapper is the published surface, so what it is checked for is that a
// build reaches the facade's own types: a `crate::World` out of a compile, and
// each of the failures mapped onto the variant of `crate::Error` that carries
// it.
#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::DirectionalLight;

    #[test]
    fn a_compiled_world_is_the_facade_world() {
        let world = world()
            .add(
                "sun",
                DirectionalLight {
                    intensity: 2.2,
                    ..Default::default()
                },
            )
            .add(
                "room",
                Room {
                    size: Some([16.0, 20.0, 5.0]),
                    ..Default::default()
                },
            )
            .compile()
            .expect("typed specs compile");

        let sun = world
            .inner()
            .query::<DirectionalLight>()
            .next()
            .expect("the sun compiled into a component");
        assert_eq!(sun.intensity, 2.2);
    }

    // The declarations are readable through the wrapper without paying for a
    // compile, and in the order they were made.
    #[test]
    fn declared_reports_names_and_types_in_order() {
        let mut spec = world();
        spec.assets_in("project/assets")
            .add("menu", concinnity_core::components::Scene::default())
            .add("sun", DirectionalLight::default());
        assert_eq!(spec.assets_dir(), Some(Path::new("project/assets")));
        let declared: Vec<_> = spec.declared().collect();
        assert_eq!(declared, [("menu", "Scene"), ("sun", "DirectionalLight")]);
    }

    #[test]
    fn a_rejected_world_is_a_validation_error() {
        let mut spec = world();
        // A slider on nothing: the shape names a target that was never
        // declared, which validation rejects.
        spec.add(
            "orphan",
            concinnity_core::components::CharacterShape::default(),
        )
        .reference("target", "no_such_body");
        let err = spec.compile().expect_err("an unresolved reference fails");
        let crate::Error::Validation(errs) = err else {
            panic!("expected a validation failure, got {err:?}");
        };
        assert!(
            errs.iter().any(|e| e.contains("no_such_body")),
            "got: {errs:?}"
        );
    }

    // Naming a reference with nothing to attach it to is the caller's
    // mistake, surfaced at compile rather than silently dropped.
    #[test]
    fn a_reference_before_any_asset_is_a_build_error() {
        let err = world()
            .reference("target", "hero")
            .compile()
            .expect_err("nothing to reference");
        assert!(
            matches!(
                err,
                crate::Error::Build {
                    kind: std::io::ErrorKind::InvalidInput,
                    ..
                }
            ),
            "{err:?}"
        );
        assert!(err.to_string().contains("before any asset"), "{err}");
    }

    // The ahead-of-time path goes through the wrapper too, and lands at the
    // path the caller named.
    #[test]
    fn write_blob_writes_a_world_at_the_named_path() {
        let tree = concinnity_testing::TempTree::new();
        let primary = tree.join("data/0");

        world()
            .add("sun", DirectionalLight::default())
            .write_blob(&primary)
            .expect("the world is written");

        assert!(primary.exists(), "the blob was written");
    }
}
