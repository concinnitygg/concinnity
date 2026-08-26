//! Compile authored worlds into runnable [`World`]s, entirely in memory.
//!
//! Requires the `cook` feature, which is off by default: the importers this
//! module needs are build-time weight a shipped application does not carry.
//!
//! The runtime plays compiled worlds, never authored declarations. This module
//! owns the in-process compile step: validate and expand the declarations,
//! compile each asset's payload, then either assemble the components into a
//! [`World`] or write them to a blob file. Source-backed assets are cached
//! under `.concinnity/cache/` (relative to the current directory), so a second
//! run with unchanged sources skips the expensive decode.
//!
//! # Declaring a world in code
//!
//! Each asset is declared by its own authored struct, so the type is carried
//! by the value rather than spelled as a string:
//!
//! ```no_run
//! use concinnity::App;
//! use concinnity::assets::{DirectionalLight, RoomArgs};
//! use concinnity::cook;
//!
//! fn main() {
//!     let world = cook::world()
//!         .add("sun", DirectionalLight {
//!             color: [1.0, 0.96, 0.86],
//!             direction: [-0.35, 0.85, 0.35],
//!             intensity: 2.2,
//!         })
//!         .add("room", RoomArgs {
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
//! # use concinnity::assets::{Material, Prop};
//! # use concinnity::cook;
//! cook::world()
//!     .add("stone", Material { roughness: 0.9, ..Default::default() })
//!     .add("pillar", Prop::default())
//!     .reference("material", "stone");
//! ```
//!
//! Most assets are plain runtime components, so an asset needing no compile
//! step can equally be added straight to a [`World`] with
//! [`add_component`](World::add_component). The cook is what a
//! [`RoomArgs`](crate::assets::RoomArgs) needs: its geometry is generated into
//! the blob, and its texture names resolve to handles, neither of which exists
//! before the compile.
//!
//! # Compiling ahead of time
//!
//! [`write_blob`](WorldBuilder::write_blob) compiles the same declarations to
//! a blob file instead of a world, so the authoring cost is paid once by a
//! build tool rather than on every launch. The shipped application plays that
//! file with [`App::from_blob`](crate::App::from_blob) and needs neither this
//! module nor the importers behind it.
//!
//! ```no_run
//! # use concinnity::assets::DirectionalLight;
//! # use concinnity::cook;
//! cook::world()
//!     .add("sun", DirectionalLight::default())
//!     .write_blob("data/0")
//!     .expect("the world is written to data/0");
//! ```

use std::path::Path;

use concinnity_cook::pipeline::PipelineResult;
use concinnity_cook::world::LoadedWorld;
use concinnity_cook::{build_compiled, check::report_validation_errors, prepare_world};
use concinnity_engine::blob::BlobData;
use concinnity_engine::ecs::ComponentAsset;

use concinnity_world::registry::{asset_line, set_reference};

pub use concinnity_world::registry::Authored;

use crate::World;

/// A world under construction: typed authored assets, compiled together into
/// a runnable [`World`] or a blob file.
#[derive(Default)]
pub struct WorldBuilder {
    // Finished world lines, serialized as each asset is added.
    lines: Vec<String>,
    // Name and type per line, so the declaration order can be inspected
    // without re-reading the lines.
    declared: Vec<(String, &'static str)>,
    // The first serialization failure, held until the compile so the call
    // chain stays borrow-friendly.
    error: Option<std::io::Error>,
}

/// Start an empty world.
pub fn world() -> WorldBuilder {
    WorldBuilder::default()
}

impl WorldBuilder {
    /// Declare `value` under `name`. The asset type comes from the value's
    /// own [`Authored`] impl, so it cannot disagree with the fields.
    pub fn add<T: Authored>(&mut self, name: impl Into<String>, value: T) -> &mut Self {
        let name = name.into();
        match asset_line(&name, &value) {
            Ok(line) => {
                self.lines.push(line);
                self.declared.push((name, T::TYPE));
            }
            Err(e) => {
                self.error.get_or_insert(e);
            }
        }
        self
    }

    /// The assets declared so far, as `(name, type)` pairs in declaration
    /// order. Declaration order is load-bearing for scenes: the first `Scene`
    /// is the one active at world start.
    pub fn declared(&self) -> impl Iterator<Item = (&str, &str)> {
        self.declared.iter().map(|(n, t)| (n.as_str(), *t))
    }

    /// Point a reference field of the asset just added at `target`, by name.
    ///
    /// A reference on an authored struct holds a resolved handle (a dense
    /// index the compile assigns in declaration order), so the typed value
    /// cannot carry the name it points at. This writes the name into the
    /// pending declaration, where the compile resolves it.
    ///
    /// ```no_run
    /// # use concinnity::assets::{CharacterShape, ShapeSlider};
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
        let invalid = |msg: String| std::io::Error::new(std::io::ErrorKind::InvalidInput, msg);
        let Some(line) = self.lines.pop() else {
            self.error.get_or_insert(invalid(format!(
                "reference(\"{field}\") before any asset was added"
            )));
            return self;
        };
        match set_reference(&line, field, &target.into()) {
            Ok(patched) => self.lines.push(patched),
            Err(e) => {
                self.error.get_or_insert(e);
            }
        }
        self
    }

    /// Compile every declared asset into a runnable [`World`].
    pub fn compile(&self) -> std::io::Result<World> {
        let mut result = self.build()?;

        let payload_sections: Vec<Option<Vec<u8>>> = std::mem::take(&mut result.payloads)
            .into_iter()
            .map(Some)
            .collect();
        let mut world = concinnity_engine::ecs::World::from_blob(BlobData::new(payload_sections));

        for def in &result.defs {
            let mut component = ComponentAsset::from_baked(def).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("asset construction failed: {e:?}"),
                )
            })?;
            if let Some(locator) = &def.payload {
                component.inject_locator(locator.clone());
            }
            world.add(component);
        }

        // Load the compiled resource stream into the per-kind tables the
        // systems read by handle. Kinds that have left the component registry
        // (textures, audio clips, fonts, colour LUTs, environment maps) live
        // here, not in `defs`, so without this the renderer sees an empty
        // texture pool and every material's albedo handle resolves out of
        // range. Same call the runtime makes when it loads a blob file.
        concinnity_engine::resource::install_resource_tables(&mut world, &mut result.resources);

        Ok(World::from_inner(world))
    }

    /// Compile every declared asset and write it to the blob file at `path`.
    /// Payloads too large for one blob spill into siblings named by index, so
    /// a world written to `data/0` may also write `data/1`, `data/2`, ...
    /// [`App::from_blob`](crate::App::from_blob) reads that layout back.
    pub fn write_blob(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let result = self.build()?;
        concinnity_cook::pipeline::write_blobs_to(&result, path.as_ref())?;
        Ok(())
    }

    // Validate, expand and compile the declarations. The shared front half of
    // `compile` and `write_blob`: both need every payload built, and differ
    // only in where the result lands.
    fn build(&self) -> std::io::Result<PipelineResult> {
        if let Some(e) = &self.error {
            return Err(std::io::Error::new(e.kind(), e.to_string()));
        }

        // The cook ships no shader compilers of its own; hand it this build's
        // before any ShaderStage is compiled.
        concinnity_shader::install();

        // Bare `source` filenames resolve under the installed state root's
        // `assets/`. An embedder that installed no state root has no tree to
        // search, so only paths that stand on their own resolve.
        let assets_dir = concinnity_cook::paths::assets_dir();
        let loaded: LoadedWorld = prepare_world(&self.lines.concat(), assets_dir.as_deref())
            .map_err(|errs| report_validation_errors(&errs))?;
        build_compiled(loaded.assets, assets_dir.as_deref(), None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::components::{Camera3D, DirectionalLight};

    // The typed path: authored structs instead of string-keyed specs, across
    // all three shapes (args override, pass-through component, resource).
    #[test]
    fn typed_builder_compiles_a_world() {
        use concinnity_engine::components::{DirectionalLight, RoomArgs};

        let world = world()
            .add(
                "sun",
                DirectionalLight {
                    color: [1.0, 0.96, 0.86],
                    direction: [-0.35, 0.85, 0.35],
                    intensity: 2.2,
                },
            )
            .add(
                "room",
                RoomArgs {
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
        // Room is `compiled`: the cook generated its geometry into the blob.
        let room = world
            .inner()
            .query::<concinnity_engine::components::Room>()
            .next()
            .expect("the room compiled into a component");
        assert_eq!(room.half_width, 8.0, "size is halved by the bake");
        assert!(room.locator.is_some(), "generated geometry is in the blob");
    }

    #[test]
    fn compile_reports_validation_errors() {
        let mut spec = world();
        // A slider on nothing: the shape names a target that was never
        // declared, which validation rejects.
        spec.add(
            "orphan",
            concinnity_engine::components::CharacterShape::default(),
        )
        .reference("target", "no_such_body");
        let err = spec.compile().expect_err("an unresolved reference fails");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn an_empty_world_yields_the_injected_defaults() {
        // An empty authored world still compiles: the pipeline injects the
        // engine defaults (DebugHud et al), so the world is valid but carries
        // no authored scene.
        let world = world().compile().expect("an empty world compiles");
        assert!(world.inner().query::<Camera3D>().next().is_none());
    }

    // A reference field holds a resolved handle, so the typed value cannot
    // name what it points at; the builder names it and the compile resolves
    // it exactly as it resolves an authored reference.
    #[test]
    fn a_named_reference_resolves_to_its_handle() {
        use concinnity_engine::components::{Material, ProceduralMesh, Prop};

        let world = world()
            .add(
                "floor_mat",
                Material {
                    roughness: 0.8,
                    ..Default::default()
                },
            )
            .add(
                "floor_mesh",
                ProceduralMesh {
                    generator: "plane".into(),
                    half_width: 4.0,
                    half_depth: 4.0,
                    ..Default::default()
                },
            )
            .add("floor", Prop::default())
            .reference("mesh", "floor_mesh")
            .reference("material", "floor_mat")
            .compile()
            .expect("a named reference compiles");

        let prop = world
            .inner()
            .query::<Prop>()
            .next()
            .expect("the prop compiled into a component");
        // The handle types are not part of the public surface (a reference
        // is only ever named), so compare the resolved indices.
        assert_eq!(prop.mesh.map(|h| h.index()), Some(0));
        assert_eq!(prop.material.map(|h| h.index()), Some(0));
    }

    // `declared` reports names and types in declaration order, which is what
    // lets a caller check scene ordering before paying for a compile.
    #[test]
    fn declared_reports_names_and_types_in_order() {
        let mut spec = world();
        spec.add("menu", concinnity_engine::components::Scene::default())
            .add("sun", DirectionalLight::default());
        let declared: Vec<_> = spec.declared().collect();
        assert_eq!(declared, [("menu", "Scene"), ("sun", "DirectionalLight")]);
    }

    // Naming a reference with nothing to attach it to is the caller's
    // mistake, surfaced at compile rather than silently dropped.
    #[test]
    fn a_reference_before_any_asset_is_a_compile_error() {
        let err = world()
            .reference("target", "hero")
            .compile()
            .expect_err("nothing to reference");
        assert!(err.to_string().contains("before any asset"), "{err}");
    }

    // The ahead-of-time path: the same declarations land in a blob file whose
    // name the caller chose, and that file is a world the runtime can read.
    #[test]
    fn write_blob_writes_a_readable_world_at_the_named_path() {
        use concinnity_engine::ecs::ComponentSlot;

        let dir = std::env::temp_dir().join("concinnity-cook-write-blob");
        let primary = dir.join("data").join("0");
        let _ = std::fs::remove_dir_all(&dir);

        world()
            .add(
                "sun",
                DirectionalLight {
                    intensity: 3.5,
                    ..Default::default()
                },
            )
            .write_blob(&primary)
            .expect("the world is written");

        let (meta, _) = concinnity_engine::blob::read_cnb(&primary.to_string_lossy())
            .expect("the written blob parses");
        assert!(
            meta.defs
                .iter()
                .any(|d| d.discriminant == DirectionalLight::DISCRIMINANT),
            "the sun is in the def table"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
