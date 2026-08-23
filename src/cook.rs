//! Compile authored worlds into runnable [`World`]s, entirely in memory.
//!
//! Requires the `cook` feature, which is off by default: the importers this
//! module needs are build-time weight a shipped application does not carry.
//!
//! The runtime plays compiled blobs, never `world.jsonl` source declarations.
//! This module owns the in-process compile step: validate and expand the
//! declarations, compile each asset's payload, then assemble the components
//! into a [`World`] backed by the compiled blob. Source-backed assets are
//! cached under `.concinnity/cache/` (relative to the current directory), so
//! a second run with unchanged sources skips the expensive decode.
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
//! fn main() -> std::io::Result<()> {
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
//!         .compile()?;
//!     App::from_world(world).run()
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
//! step can equally be added straight to a [`World`] with `add_component`.
//! The cook is what a [`RoomArgs`](crate::assets::RoomArgs) needs: its
//! geometry is generated into the blob, and its texture names resolve to
//! handles, neither of which exists before the compile.
//!
//! # Compiling a world.jsonl file
//!
//! A world authored as `world.jsonl` (the `cn` CLI's medium) compiles the
//! same way:
//!
//! ```no_run
//! use concinnity::App;
//! use concinnity::cook::compile_world;
//!
//! fn main() -> std::io::Result<()> {
//!     let content = std::fs::read_to_string("world.jsonl")?;
//!     App::from_world(compile_world(&content)?).run()
//! }
//! ```

use concinnity_cook::world::LoadedWorld;
use concinnity_cook::{build_compiled, check::report_validation_errors, prepare_world};
use concinnity_engine::blob::BlobData;
use concinnity_engine::ecs::{ComponentAsset, World};

use concinnity_world::registry::{asset_line, set_reference};

pub use concinnity_world::registry::Authored;

/// A world under construction: typed authored assets, compiled together into
/// a runnable [`World`].
#[derive(Default)]
pub struct WorldBuilder {
    // Finished world lines, serialized as each asset is added.
    lines: Vec<String>,
    // Name and type per line, so the declaration order can be inspected
    // without re-reading the lines.
    declared: Vec<(String, &'static str)>,
    // The first serialization failure, held until compile so the call chain
    // stays borrow-friendly.
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

    /// Point a reference field of the asset just added at `target`, by name.
    ///
    /// A reference on an authored struct holds a resolved handle (a dense
    /// index the compile assigns in declaration order), so the typed value
    /// cannot carry the name it points at. This writes the name into the
    /// pending declaration, where the compile resolves it exactly as it
    /// resolves the same reference in a `world.jsonl`.
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

    /// The assets declared so far, as `(name, type)` pairs in declaration
    /// order. Declaration order is load-bearing for scenes: the first `Scene`
    /// is the one active at world start.
    pub fn declared(&self) -> impl Iterator<Item = (&str, &str)> {
        self.declared.iter().map(|(n, t)| (n.as_str(), *t))
    }

    /// Compile every declared asset into a runnable [`World`].
    pub fn compile(&self) -> std::io::Result<World> {
        if let Some(e) = &self.error {
            return Err(std::io::Error::new(e.kind(), e.to_string()));
        }
        compile_world(&self.lines.concat())
    }
}

/// Compile world.jsonl content into a runnable [`World`] entirely in memory:
/// validate and expand the declarations, compile each asset's payload, then
/// assemble the components into a world backed by the compiled blob.
pub fn compile_world(content: &str) -> std::io::Result<World> {
    // The cook ships no shader compilers of its own; hand it this build's
    // before any ShaderStage is compiled.
    concinnity_shader::install();

    let loaded: LoadedWorld =
        prepare_world(content).map_err(|errs| report_validation_errors(&errs))?;

    let mut result = build_compiled(loaded.assets, None)?;

    let payload_sections: Vec<Option<Vec<u8>>> = result.payloads.into_iter().map(Some).collect();
    let mut world = World::from_blob(BlobData::new(payload_sections));

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

    // Load the compiled blob's resource stream into the per-kind tables the
    // systems read by handle. Kinds that have left the component registry
    // (textures, audio clips, fonts, colour LUTs, environment maps) live here,
    // not in `defs`, so without this the renderer sees an empty texture pool
    // and every material's albedo handle resolves out of range. Same call the
    // shipped runtime's `App::load_blob` makes.
    concinnity_engine::resource::install_resource_tables(&mut world, &mut result.resources);

    Ok(world)
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::assets::{Camera3D, DirectionalLight};

    // The typed path: authored structs instead of string-keyed specs, across
    // all three shapes (args override, pass-through component, resource).
    #[test]
    fn typed_builder_compiles_a_world() {
        use concinnity_engine::assets::{DirectionalLight, RoomArgs};

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
            .query::<DirectionalLight>()
            .next()
            .expect("the sun compiled into a component");
        assert_eq!(sun.intensity, 2.2);
        // Room is `compiled`: the cook generated its geometry into the blob.
        let room = world
            .query::<concinnity_engine::assets::Room>()
            .next()
            .expect("the room compiled into a component");
        assert_eq!(room.half_width, 8.0, "size is halved by the bake");
        assert!(room.locator.is_some(), "generated geometry is in the blob");
    }

    #[test]
    fn compile_world_reports_validation_errors() {
        let err = compile_world(r#"{"name": "x", "type": "NotARegisteredType", "args": {}}"#)
            .expect_err("an unknown asset type fails validation");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn an_empty_world_yields_the_injected_defaults() {
        // An empty authored world still compiles: the pipeline injects the
        // engine defaults (DebugHud et al), so the world is valid but carries
        // no authored scene.
        let world = world().compile().expect("an empty world compiles");
        assert!(world.query::<Camera3D>().next().is_none());
    }

    // A reference field holds a resolved handle, so the typed value cannot
    // name what it points at; the builder names it and the compile resolves
    // it exactly as it resolves a world.jsonl reference.
    #[test]
    fn a_named_reference_resolves_to_its_handle() {
        use concinnity_engine::assets::{Material, ProceduralMesh, Prop};

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
            .query::<Prop>()
            .next()
            .expect("the prop compiled into a component");
        // The handle types are not part of the public surface (a reference
        // is only ever named), so compare the resolved indices.
        assert_eq!(prop.mesh.map(|h| h.index()), Some(0));
        assert_eq!(prop.material.map(|h| h.index()), Some(0));
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

    // `declared` reports names and types in declaration order, which is what
    // lets a caller check scene ordering before paying for a compile.
    #[test]
    fn declared_reports_names_and_types_in_order() {
        let mut spec = world();
        spec.add("menu", concinnity_engine::assets::Scene::default())
            .add("sun", DirectionalLight::default());
        let declared: Vec<_> = spec.declared().collect();
        assert_eq!(declared, [("menu", "Scene"), ("sun", "DirectionalLight")]);
    }
}
