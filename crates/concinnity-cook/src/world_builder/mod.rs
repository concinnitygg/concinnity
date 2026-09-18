//! Declaring a world in typed authored values and compiling it in memory.
//!
//! The rest of this crate's build API takes a world as authored text; this
//! takes it as the authored structs themselves, so the asset type is carried
//! by the value rather than spelled as a string. The result is either a
//! runnable [`World`] or a blob file, from the same declarations.

use std::path::{Path, PathBuf};

use concinnity_core::ecs::{ComponentAsset, ResourceRecord, World};
use concinnity_core::platform::Platform;
use concinnity_host::store::blob::BlobData;

use crate::authoring::registry::{Authored, asset_line, set_reference};
use crate::build_only::{LoadedWorld, prepare_world};
use crate::pipeline::{PipelineResult, build_compiled, write_blobs_to};

mod error;

pub use error::WorldBuildError;
use error::from_io;

/// A world under construction: typed authored assets, compiled together into
/// a runnable [`World`] or a blob file.
pub struct WorldBuilder {
    // The backend shaders are compiled for. Named by the caller, since nothing
    // here resolves a backend of its own.
    platform: Platform,
    // Finished world lines, serialized as each asset is added.
    lines: Vec<String>,
    // The search root a bare `source` filename resolves under, when the
    // embedder named one.
    assets_dir: Option<PathBuf>,
    // Id and type per line, so the declaration order can be inspected
    // without re-reading the lines.
    declared: Vec<(String, &'static str)>,
    // The first declaration failure, held as the kind and message a
    // [`WorldBuildError::Build`] is rebuilt from at the compile, so the call
    // chain stays borrow-friendly.
    error: Option<(std::io::ErrorKind, String)>,
}

/// Start an empty world, cooked for `platform`.
pub fn world(platform: Platform) -> WorldBuilder {
    WorldBuilder {
        platform,
        lines: Vec::new(),
        assets_dir: None,
        declared: Vec::new(),
        error: None,
    }
}

impl WorldBuilder {
    /// Resolve bare `source` filenames under `dir`, the way a build resolves
    /// them under a state tree's `assets/`. Without one only a path that stands
    /// on its own resolves, since nothing here guesses a root.
    pub fn assets_in(&mut self, dir: impl Into<PathBuf>) -> &mut Self {
        self.assets_dir = Some(dir.into());
        self
    }

    /// The asset search root this build resolves against, if one was named.
    pub fn assets_dir(&self) -> Option<&Path> {
        self.assets_dir.as_deref()
    }

    /// Declare `value` under the `$id` `id`, the name a reference to it uses.
    /// The asset type comes from the value's own [`Authored`] impl, so it
    /// cannot disagree with the fields.
    pub fn add<T: Authored>(&mut self, id: impl Into<String>, value: T) -> &mut Self {
        let id = id.into();
        match asset_line(&id, &value) {
            Ok(line) => {
                self.lines.push(line);
                self.declared.push((id, T::TYPE));
            }
            Err(e) => {
                self.error.get_or_insert((e.kind(), e.to_string()));
            }
        }
        self
    }

    /// The assets declared so far, as `(id, type)` pairs in declaration
    /// order. Declaration order is load-bearing for scenes: the first `Scene`
    /// is the one active at world start.
    pub fn declared(&self) -> impl Iterator<Item = (&str, &str)> {
        self.declared.iter().map(|(n, t)| (n.as_str(), *t))
    }

    /// Point a reference field of the asset just added at `target`, by its
    /// `$id`.
    ///
    /// A reference on an authored struct holds a resolved handle (a dense
    /// index the compile assigns in declaration order), so the typed value
    /// cannot carry the name it points at. This writes the name into the
    /// pending declaration, where the compile resolves it.
    pub fn reference(&mut self, field: &str, target: impl Into<String>) -> &mut Self {
        let Some(line) = self.lines.pop() else {
            self.error.get_or_insert((
                std::io::ErrorKind::InvalidInput,
                format!("reference(\"{field}\") before any asset was added"),
            ));
            return self;
        };
        match set_reference(&line, field, &target.into()) {
            Ok(patched) => self.lines.push(patched),
            Err(e) => {
                self.error.get_or_insert((e.kind(), e.to_string()));
            }
        }
        self
    }

    /// Compile every declared asset into a runnable [`World`].
    pub fn compile(&self) -> Result<World, WorldBuildError> {
        let mut result = self.build()?;

        let payload_sections: Vec<Option<Vec<u8>>> = std::mem::take(&mut result.payloads)
            .into_iter()
            .map(Some)
            .collect();
        let mut world = World::from_payloads(Box::new(BlobData::new(payload_sections)));

        for def in &result.defs {
            let mut component = ComponentAsset::from_baked(def)?;
            if let Some(locator) = &def.payload {
                component.inject_locator(locator.clone());
            }
            world.add(component, def.name);
        }

        // Load the compiled resource stream into the per-kind tables the
        // systems read by handle. Kinds that have left the component registry
        // (textures, audio clips, fonts, color LUTs, environment maps) live
        // here, not in `defs`, so without this the renderer sees an empty
        // texture pool and every material's albedo handle resolves out of
        // range. Same call the runtime makes when it loads a blob file.
        log_resource_footprint(&result.resources);
        concinnity_core::resource::install_tables(&mut world, &mut result.resources);

        Ok(world)
    }

    /// Compile every declared asset and write it to the blob file at `path`.
    /// Payloads too large for one blob spill into siblings named by index, so
    /// a world written to `data/0` may also write `data/1`, `data/2`, ...
    pub fn write_blob(&self, path: impl AsRef<Path>) -> Result<(), WorldBuildError> {
        let result = self.build()?;
        write_blobs_to(&result, path.as_ref()).map_err(from_io)?;
        Ok(())
    }

    // Validate, expand and compile the declarations. The shared front half of
    // `compile` and `write_blob`: both need every payload built, and differ
    // only in where the result lands.
    fn build(&self) -> Result<PipelineResult, WorldBuildError> {
        if let Some((kind, message)) = &self.error {
            return Err(WorldBuildError::Build {
                kind: *kind,
                message: message.clone(),
            });
        }

        // Bare `source` filenames resolve under the root the embedder named
        // (`assets_in`). Without one there is no tree to search, so only paths
        // that stand on their own resolve.
        let assets_dir = self.assets_dir.clone();
        let loaded: LoadedWorld = prepare_world(&self.lines.concat(), assets_dir.as_deref())
            .map_err(WorldBuildError::Validation)?;
        build_compiled(loaded.assets, assets_dir.as_deref(), None, self.platform).map_err(from_io)
    }
}

// The compiled-resource footprint an in-memory build installs: the payload
// bytes each record references plus the data-resource bytes the tables hold
// directly. A coarse figure toward the memory budget, matching what the
// runtime reports when it loads a blob file.
fn log_resource_footprint(records: &[ResourceRecord]) {
    if records.is_empty() {
        return;
    }
    let total: u64 = records
        .iter()
        .map(|r| r.data_bytes.len() as u64 + r.payload.as_ref().map_or(0, |p| p.len))
        .sum();
    tracing::info!(
        "Resource tables: {} record(s), {} MiB compiled",
        records.len(),
        total / (1024 * 1024)
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::components::{Camera3D, DirectionalLight};

    // None of these declarations carries a shader, so the platform only names
    // a backend the compile never reaches; Metal is what the rest of this
    // crate's tests name.
    fn builder() -> WorldBuilder {
        world(Platform::Metal)
    }

    // The typed path: authored structs instead of string-keyed specs, across
    // all three shapes (args override, pass-through component, resource).
    #[test]
    fn typed_builder_compiles_a_world() {
        use concinnity_core::components::cook::Room;

        let world = builder()
            .add(
                "sun",
                DirectionalLight {
                    color: [1.0, 0.96, 0.86],
                    direction: [-0.35, 0.85, 0.35],
                    intensity: 2.2,
                },
            )
            // `Room` here is the authored form, not the component of the same
            // name the query below reads back.
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
            .query::<DirectionalLight>()
            .next()
            .expect("the sun compiled into a component");
        assert_eq!(sun.intensity, 2.2);
        // Room is `compiled`: the cook generated its geometry into the blob.
        let room = world
            .query::<concinnity_core::components::Room>()
            .next()
            .expect("the room compiled into a component");
        assert_eq!(room.half_width, 8.0, "size is halved by the bake");
        assert!(room.locator.is_some(), "generated geometry is in the blob");
    }

    #[test]
    fn compile_reports_validation_errors() {
        let mut spec = builder();
        // A slider on nothing: the shape names a target that was never
        // declared, which validation rejects.
        spec.add(
            "orphan",
            concinnity_core::components::CharacterShape::default(),
        )
        .reference("target", "no_such_body");
        let err = spec.compile().expect_err("an unresolved reference fails");
        let WorldBuildError::Validation(errs) = err else {
            panic!("expected a validation failure, got {err:?}");
        };
        assert!(
            errs.iter().any(|e| e.contains("no_such_body")),
            "got: {errs:?}"
        );
    }

    #[test]
    fn an_empty_world_yields_the_injected_defaults() {
        // An empty authored world still compiles: the pipeline injects the
        // engine defaults (DebugHud et al), so the world is valid but carries
        // no authored scene.
        let world = builder().compile().expect("an empty world compiles");
        assert!(world.query::<Camera3D>().next().is_none());
    }

    // A reference field holds a resolved handle, so the typed value cannot
    // name what it points at; the builder names it and the compile resolves
    // it exactly as it resolves an authored reference.
    #[test]
    fn a_named_reference_resolves_to_its_handle() {
        use concinnity_core::components::{Material, ProceduralMesh, Prop};

        let world = builder()
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

    // `declared` reports names and types in declaration order, which is what
    // lets a caller check scene ordering before paying for a compile.
    #[test]
    fn declared_reports_names_and_types_in_order() {
        let mut spec = builder();
        spec.add("menu", concinnity_core::components::Scene::default())
            .add("sun", DirectionalLight::default());
        let declared: Vec<_> = spec.declared().collect();
        assert_eq!(declared, [("menu", "Scene"), ("sun", "DirectionalLight")]);
    }

    // The search root is what a bare `source` filename resolves under, so a
    // caller can read back the one it named.
    #[test]
    fn the_named_asset_root_is_readable() {
        let mut spec = builder();
        assert!(spec.assets_dir().is_none());
        spec.assets_in("project/assets");
        assert_eq!(spec.assets_dir(), Some(Path::new("project/assets")));
    }

    // Naming a reference with nothing to attach it to is the caller's
    // mistake, surfaced at compile rather than silently dropped.
    #[test]
    fn a_reference_before_any_asset_is_a_compile_error() {
        let err = builder()
            .reference("target", "hero")
            .compile()
            .expect_err("nothing to reference");
        assert!(
            matches!(
                err,
                WorldBuildError::Build {
                    kind: std::io::ErrorKind::InvalidInput,
                    ..
                }
            ),
            "{err:?}"
        );
        assert!(err.to_string().contains("before any asset"), "{err}");
    }

    // The ahead-of-time path: the same declarations land in a blob file whose
    // name the caller chose, and that file is a world the runtime can read.
    #[test]
    fn write_blob_writes_a_readable_world_at_the_named_path() {
        use concinnity_core::ecs::ComponentSlot;

        let tree = concinnity_testing::TempTree::new();
        let primary = tree.join("data/0");

        builder()
            .add(
                "sun",
                DirectionalLight {
                    intensity: 3.5,
                    ..Default::default()
                },
            )
            .write_blob(&primary)
            .expect("the world is written");

        let (meta, _) = concinnity_host::store::blob::read_cnb(&primary.to_string_lossy())
            .expect("the written blob parses");
        assert!(
            meta.defs
                .iter()
                .any(|d| d.discriminant == DirectionalLight::DISCRIMINANT),
            "the sun is in the def table"
        );
    }
}
