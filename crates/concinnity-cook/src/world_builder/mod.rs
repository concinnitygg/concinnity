//! Declaring a world in typed authored values and compiling it in memory.
//!
//! The rest of this crate's build API takes a world as authored text; this
//! takes it as the authored structs themselves, so the asset type is carried
//! by the value rather than spelled as a string. The result is either a
//! runnable [`World`] or a blob file, from the same declarations.

use std::path::{Path, PathBuf};

use concinnity_core::ecs::{ComponentAsset, ResourceRecord, World};
use concinnity_core::platform::PlatformSet;
use concinnity_host::store::blob::BlobData;

use crate::authoring::registry::{Authored, RegisteredType, asset_entry, set_reference};
use crate::authoring::world::{
    anonymous_label, entry_handle, entry_handles, entry_id, find_entry, parse_world_jsonl,
    write_world_jsonl,
};
use crate::build_only::{LoadedWorld, merge_args, prepare_world};
use crate::pipeline::{PipelineResult, build_compiled, write_blobs_to};

mod error;

pub use error::WorldBuildError;
use error::from_io;

/// A world under construction: typed authored assets, compiled together into
/// a runnable [`World`] or a blob file.
pub struct WorldBuilder {
    // The backend shaders are compiled for. Named by the caller, since nothing
    // here resolves a backend of its own.
    targets: PlatformSet,
    // The declared entries, built as each asset is added.
    entries: Vec<serde_json::Value>,
    // The search root a bare `source` filename resolves under, when the
    // embedder named one.
    assets_dir: Option<PathBuf>,
    // Id and type per entry, so the declaration order can be inspected
    // without re-reading the entries.
    declared: Vec<(String, &'static str)>,
    // The first declaration failure, held as the kind and message a
    // [`WorldBuildError::Build`] is rebuilt from at the compile, so the call
    // chain stays borrow-friendly.
    error: Option<(std::io::ErrorKind, String)>,
}

/// Start an empty world, cooked for every platform in `targets`.
pub fn world(targets: impl Into<PlatformSet>) -> WorldBuilder {
    WorldBuilder {
        targets: targets.into(),
        entries: Vec::new(),
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
        match asset_entry(&id, &value) {
            Ok(entry) => {
                self.entries.push(entry);
                self.declared.push((id, T::TYPE));
            }
            Err(e) => {
                self.error.get_or_insert((e.kind(), e.to_string()));
            }
        }
        self
    }

    /// Declare every entry of world file text, one `["Type", {args}]` per
    /// line, in file order after the assets already declared. An entry
    /// without a `$id` is listed by its anonymous label, such as `Prop#0`.
    pub fn add_jsonl(&mut self, text: &str) -> &mut Self {
        let entries = match parse_world_jsonl(text) {
            Ok(entries) => entries,
            Err(e) => {
                self.error
                    .get_or_insert((std::io::ErrorKind::InvalidData, e.to_string()));
                return self;
            }
        };
        for entry in entries {
            if !self.push_entry(entry) {
                break;
            }
        }
        self
    }

    /// Lay world file text over the assets already declared. An entry with
    /// the `$id` of a declared asset, or with no `$id` and the type of an
    /// anonymous one, patches the first such asset: objects merge field by
    /// field, and anything else, `null` included, replaces the field. Any
    /// other entry is declared as [`add_jsonl`](Self::add_jsonl) declares it.
    pub fn overlay_jsonl(&mut self, text: &str) -> &mut Self {
        let entries = match parse_world_jsonl(text) {
            Ok(entries) => entries,
            Err(e) => {
                self.error
                    .get_or_insert((std::io::ErrorKind::InvalidData, e.to_string()));
                return self;
            }
        };
        for entry in entries {
            let ty = entry.get("type").and_then(|t| t.as_str()).unwrap_or("");
            let handle = entry_id(&entry).map_or_else(|| anonymous_label(ty, 0), str::to_string);
            let Some(index) = find_entry(&self.entries, &handle) else {
                if !self.push_entry(entry) {
                    break;
                }
                continue;
            };
            let target = &mut self.entries[index];
            if target.get("type") != entry.get("type") {
                self.error.get_or_insert((
                    std::io::ErrorKind::InvalidData,
                    format!("overlay of \"{handle}\" as a {ty}: it is declared as another type"),
                ));
                break;
            }
            let args = merge_args(&target["args"], &entry["args"]);
            target["args"] = args;
        }
        self
    }

    /// The args of the declared asset `handle` names (its `$id`, or the label
    /// of an anonymous entry, such as `Planet#0`) as authored, references
    /// still by name.
    pub fn declared_args(&self, handle: &str) -> Option<&serde_json::Value> {
        find_entry(&self.entries, handle).and_then(|i| self.entries[i].get("args"))
    }

    // Declare one parsed entry, or note why it cannot be and refuse it.
    fn push_entry(&mut self, entry: serde_json::Value) -> bool {
        let ty = entry.get("type").and_then(|t| t.as_str()).unwrap_or("");
        let Some(registered) = RegisteredType::parse(ty) else {
            self.error.get_or_insert((
                std::io::ErrorKind::InvalidData,
                format!("unknown asset type '{ty}'"),
            ));
            return false;
        };
        self.entries.push(entry);
        let id = entry_handle(&self.entries, self.entries.len() - 1).unwrap_or_default();
        self.declared.push((id, registered.as_str()));
        true
    }

    /// Remove the declared asset `handle` names: its `$id`, or the label of
    /// an anonymous entry, such as `RigidBody#0`. Naming nothing is an error
    /// at compile.
    pub fn remove(&mut self, handle: &str) -> &mut Self {
        let Some(index) = find_entry(&self.entries, handle) else {
            self.error.get_or_insert((
                std::io::ErrorKind::NotFound,
                format!("remove(\"{handle}\"): no asset by that name"),
            ));
            return self;
        };
        self.entries.remove(index);
        self.declared.remove(index);
        // Anonymous labels count their type's entries, so later ones shift.
        for ((id, _), handle) in self.declared.iter_mut().zip(entry_handles(&self.entries)) {
            *id = handle.unwrap_or_default();
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
    /// pending declaration, where the compile resolves it. A dotted `field`
    /// reaches into a nested object, such as a behavior's `on.enter`, and a
    /// numeric step into a list, such as a planet's `layers.0.grass`.
    pub fn reference(&mut self, field: &str, target: impl Into<String>) -> &mut Self {
        let Some(entry) = self.entries.last_mut() else {
            self.error.get_or_insert((
                std::io::ErrorKind::InvalidInput,
                format!("reference(\"{field}\") before any asset was added"),
            ));
            return self;
        };
        if let Err(e) = set_reference(entry, field, &target.into()) {
            self.error.get_or_insert((e.kind(), e.to_string()));
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
        let text = write_world_jsonl(&self.entries).map_err(from_io)?;
        let loaded: LoadedWorld =
            prepare_world(&text, assets_dir.as_deref()).map_err(WorldBuildError::Validation)?;
        build_compiled(loaded.assets, assets_dir.as_deref(), self.targets).map_err(from_io)
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
    let total = records
        .iter()
        .map(ResourceRecord::compiled_len)
        .sum::<u64>();
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
        world(concinnity_core::platform::Platform::Metal)
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

    const BASE: &str = r#"["GraphicsConfig",{"bloom":{"intensity":0.3,"threshold":1.2}}]
["Camera3D",{"$id":"cam","pitch":0.05,"controller":{"move_speed":5.0}}]
["Camera3D",{"$id":"other","pitch":0.5}]"#;

    // An entry without an `$id` patches the anonymous one of its type, nested
    // objects field by field.
    #[test]
    fn an_overlay_merges_into_the_entry_of_its_type() {
        let mut spec = builder();
        spec.add_jsonl(BASE).overlay_jsonl(
            r#"["GraphicsConfig",{"bloom":{"intensity":0.0},"frame_pacing":{"fps_cap":0}}]"#,
        );
        let graphics = spec.declared_args("GraphicsConfig#0").expect("declared");
        assert_eq!(graphics["bloom"]["intensity"], 0.0);
        assert_eq!(graphics["bloom"]["threshold"], 1.2);
        assert_eq!(graphics["frame_pacing"]["fps_cap"], 0);
        assert_eq!(spec.declared().count(), 3);
    }

    // An `$id` picks its entry among several of a type, `null` clears a field,
    // and an entry naming nothing declared is added.
    #[test]
    fn an_overlay_patches_by_id_and_adds_the_rest() {
        let mut spec = builder();
        spec.add_jsonl(BASE).overlay_jsonl(
            "[\"Camera3D\",{\"$id\":\"cam\",\"controller\":null}]\n\
             [\"FrameReport\",{\"warmup_seconds\":2.5}]",
        );
        let cam = spec.declared_args("cam").expect("declared");
        assert_eq!(cam["controller"], serde_json::Value::Null);
        assert_eq!(cam["pitch"], 0.05);
        assert_eq!(spec.declared_args("other").expect("declared")["pitch"], 0.5);
        assert_eq!(
            spec.declared().last(),
            Some(("FrameReport#0", "FrameReport"))
        );
        spec.compile().expect("the overlaid world compiles");
    }

    #[test]
    fn an_overlay_cannot_change_an_entrys_type() {
        let mut spec = builder();
        spec.add_jsonl(BASE)
            .overlay_jsonl(r#"["DirectionalLight",{"$id":"cam"}]"#);
        assert!(spec.compile().is_err());
        assert!(spec.declared_args("nothing").is_none());
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

    // World file text declares its entries in file order after the typed
    // ones, an anonymous entry under its label, and compiles like them.
    #[test]
    fn world_file_text_declares_its_entries_in_order() {
        let mut spec = builder();
        spec.add("sun", DirectionalLight::default()).add_jsonl(
            "// a comment line\n\
             [\"Material\",{\"$id\":\"stone\",\"roughness\":0.9}]\n\
             \n\
             [\"Material\",{\"roughness\":0.3}]\n",
        );
        let declared: Vec<_> = spec.declared().collect();
        assert_eq!(
            declared,
            [
                ("sun", "DirectionalLight"),
                ("stone", "Material"),
                ("Material#0", "Material")
            ]
        );
        let world = spec.compile().expect("the text compiles");
        assert!(world.query::<DirectionalLight>().next().is_some());
    }

    // A malformed line and an unknown type each fail the compile with what
    // was wrong, rather than dropping the entry.
    #[test]
    fn bad_world_file_text_is_a_compile_error() {
        let err = builder()
            .add_jsonl("{\"not\":\"a line\"}\n")
            .compile()
            .expect_err("an object is not a line");
        assert!(err.to_string().contains("line 1"), "{err}");

        let err = builder()
            .add_jsonl("[\"NoSuchAsset\",{}]\n")
            .compile()
            .expect_err("the type is unknown");
        assert!(err.to_string().contains("NoSuchAsset"), "{err}");
    }

    // Removing by `$id` or by anonymous label drops that entry, and the
    // anonymous entries after it are relabeled.
    #[test]
    fn remove_drops_the_named_entry() {
        let mut spec = builder();
        spec.add_jsonl(
            "[\"Material\",{\"roughness\":0.1}]\n\
             [\"Material\",{\"$id\":\"stone\"}]\n\
             [\"Material\",{\"roughness\":0.3}]\n",
        )
        .remove("Material#0")
        .remove("stone");
        let declared: Vec<_> = spec.declared().collect();
        assert_eq!(declared, [("Material#0", "Material")]);
        spec.compile().expect("what remains compiles");

        let err = builder()
            .remove("missing")
            .compile()
            .expect_err("nothing is named missing");
        assert!(err.to_string().contains("missing"), "{err}");
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
