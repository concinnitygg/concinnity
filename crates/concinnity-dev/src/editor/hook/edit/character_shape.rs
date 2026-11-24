//! `EditorHook`: the `CharacterShape` panel's actions. The panel edits the shape
//! whose target is the selected skinned mesh or character model (or the
//! selected shape entry); its rows are the target's schema laid over what
//! that mesh exposes in the live world (`character_shape.rs`). Every commit
//! (a slider release, Reset, Randomize, a preset) writes the entry's args
//! through `form::assemble` + `form::validate` and `mark_changed` once; the
//! per-frame drag preview lives in `drag/shape.rs`.

use concinnity_cook::authoring::registry::RegisteredType;
use concinnity_cook::authoring::registry::build_only::{CharacterSchema, ShapePreset};
use concinnity_cook::authoring::world::replace_args;
use concinnity_cook::compile::character::builtin_schema;
use concinnity_engine::components::CharacterCapsule;
use concinnity_engine::ecs::World;
use concinnity_engine::render::live_preview::shape::{self as shape_preview, ShapeTarget};
use concinnity_host::thread::asset_id;

use concinnity_engine::ecs::SkinnedMeshHandle;

use crate::editor::asset_handle::AssetHandle;
use crate::editor::hook::{EditorHook, declared_id, entry_type, short_status};
use crate::editor::panels::character_shape::{self, Row, Rows, ShapeValues};
use crate::editor::panels::character_shape_panel::{self, ShapeAction, ShapeLayout, ShapeView};
use crate::editor::panels::form;
use crate::editor::ui::BuiltPanel;

// What the panel edits: the selected skinned mesh and the shape entry
// targeting it, if the world has one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::editor::hook) struct ShapeBinding {
    pub(in crate::editor::hook) mesh: String,
    pub(in crate::editor::hook) shape_idx: Option<usize>,
}

// Owned per-tick data backing a `ShapeView`.
pub(in crate::editor::hook) struct ShapeData {
    pub(in crate::editor::hook) binding: Option<ShapeBinding>,
    pub(in crate::editor::hook) target: ShapeTarget,
    pub(in crate::editor::hook) derived: Rows,
    pub(in crate::editor::hook) presets: Vec<ShapePreset>,
    pub(in crate::editor::hook) preset_names: Vec<String>,
    pub(in crate::editor::hook) rows: Vec<Row>,
    pub(in crate::editor::hook) values: Vec<f32>,
    pub(in crate::editor::hook) status: Option<String>,
}

// What the panel shows, read from the entries, the selection, and the live
// mesh they name, kept until one of those changes.
pub(in crate::editor::hook) struct ShapeCache {
    revision: u64,
    active: Option<AssetHandle>,
    mesh: Option<String>,
    // The live mesh the rows were read from, and whether it had a pose.
    live: Option<(SkinnedMeshHandle, bool)>,
    shape: Option<String>,
    status: Option<String>,
    // One value per slider row; a drag writes its working values here.
    pub(in crate::editor::hook) values: Vec<f32>,
    layout: ShapeLayout,
}

// The skinned mesh the running world loaded as `mesh`, and whether it has a
// pose yet.
fn live_mesh(world: &World, mesh: &str) -> Option<(SkinnedMeshHandle, bool)> {
    let handle = shape_preview::mesh_handle(world, asset_id::lookup(mesh)?)?;
    Some((handle, shape_preview::posed(world, handle)))
}

impl EditorHook {
    // The binding the selection implies: a selected SkinnedMesh or
    // CharacterModel and the first shape targeting it, or a selected
    // CharacterShape and its target.
    pub(super) fn shape_binding(&self) -> Option<ShapeBinding> {
        let active = self.selection.active()?;
        let idx = self.handle_index(active)?;
        let name = self.handle_name(active)?;
        let entry = &self.entries[idx];
        match entry_type(entry)? {
            RegisteredType::SkinnedMesh | RegisteredType::CharacterModel => Some(ShapeBinding {
                shape_idx: self.entries.iter().position(|e| {
                    entry_type(e) == Some(RegisteredType::CharacterShape)
                        && e.pointer("/args/target").and_then(|t| t.as_str()) == Some(&name)
                }),
                mesh: name,
            }),
            RegisteredType::CharacterShape => {
                let mesh = entry.pointer("/args/target")?.as_str()?.to_string();
                Some(ShapeBinding {
                    mesh,
                    shape_idx: Some(idx),
                })
            }
            _ => None,
        }
    }

    // What `mesh` exposes to a shape: its live pose's joints and published
    // morph names when the preview world has it, else whatever the entry
    // authors inline.
    pub(super) fn shape_target(&self, world: &World, mesh: &str) -> ShapeTarget {
        if let Some(t) = asset_id::lookup(mesh)
            .and_then(|id| shape_preview::mesh_handle(world, id))
            .and_then(|h| shape_preview::target(world, h))
        {
            return t;
        }
        let Some(idx) = self.entry_index_named(mesh) else {
            return ShapeTarget::default();
        };
        let args = self.entry_args(idx);
        let strings = |v: Option<&serde_json::Value>| -> Vec<String> {
            v.and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|s| s.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default()
        };
        ShapeTarget {
            morph_names: strings(args.get("morph_target_names")),
            joint_names: args
                .get("skeleton")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|j| j.get("name").and_then(|n| n.as_str()).map(String::from))
                        .collect()
                })
                .unwrap_or_default(),
        }
    }

    fn entry_index_named(&self, name: &str) -> Option<usize> {
        concinnity_cook::authoring::world::find_entry(&self.entries, name)
    }

    // The schema that lays out `mesh`'s panel: a CharacterModel's named
    // schema (a CharacterSchema entry, or the bundled humanoid); a plain
    // SkinnedMesh gets the bundled humanoid's, which places whatever keys
    // and joints it shares and leaves the rest under Other.
    pub(in crate::editor::hook) fn shape_schema(&self, mesh: &str) -> CharacterSchema {
        let Some(idx) = self.entry_index_named(mesh) else {
            return builtin_schema::humanoid().clone();
        };
        if entry_type(&self.entries[idx]) != Some(RegisteredType::CharacterModel) {
            return builtin_schema::humanoid().clone();
        }
        let name = self
            .entry_args(idx)
            .get("schema")
            .and_then(|v| v.as_str())
            .unwrap_or(builtin_schema::HUMANOID_SCHEMA)
            .to_string();
        self.entry_index_named(&name)
            .filter(|&i| entry_type(&self.entries[i]) == Some(RegisteredType::CharacterSchema))
            .and_then(|i| {
                serde_json::from_value(serde_json::Value::Object(self.entry_args(i))).ok()
            })
            .unwrap_or_else(|| builtin_schema::humanoid().clone())
    }

    // The authored slider / proportion values of shape entry `idx`.
    pub(in crate::editor::hook) fn shape_values(&self, idx: usize) -> ShapeValues {
        let args = self.entry_args(idx);
        let parse = |key: &str| args.get(key).cloned().unwrap_or(serde_json::Value::Null);
        ShapeValues {
            sliders: serde_json::from_value(parse("sliders")).unwrap_or_default(),
            proportions: serde_json::from_value(parse("proportions")).unwrap_or_default(),
        }
    }

    // The authored capsule of `mesh`, the base the preview scales.
    pub(in crate::editor::hook) fn mesh_capsule(&self, mesh: &str) -> Option<CharacterCapsule> {
        let idx = self.entry_index_named(mesh)?;
        let capsule = self.entry_args(idx).get("capsule")?.clone();
        serde_json::from_value(capsule).ok()
    }

    pub(in crate::editor::hook) fn shape_data(&self, world: &World) -> ShapeData {
        let binding = self.shape_binding();
        let Some(b) = &binding else {
            return ShapeData {
                binding,
                target: ShapeTarget::default(),
                derived: Rows::default(),
                presets: Vec::new(),
                preset_names: Vec::new(),
                rows: Vec::new(),
                values: Vec::new(),
                status: Some("Select a SkinnedMesh".to_string()),
            };
        };
        let target = self.shape_target(world, &b.mesh);
        let schema = self.shape_schema(&b.mesh);
        let derived =
            character_shape::derive_rows(&schema, &target.morph_names, &target.joint_names);
        let rows = character_shape::rows(&derived, schema.presets.len(), b.shape_idx.is_some());
        // A drag in flight shows its working values, not the committed ones.
        let values: Vec<f32> = match (&self.shape.drag, b.shape_idx) {
            (Some(d), _) => derived.sliders.iter().map(|r| d.values.get(r)).collect(),
            (None, Some(idx)) => {
                let v = self.shape_values(idx);
                derived.sliders.iter().map(|r| v.get(r)).collect()
            }
            (None, None) => Vec::new(),
        };
        let preset_names = schema.presets.iter().map(|p| p.name.clone()).collect();
        ShapeData {
            binding,
            target,
            derived,
            presets: schema.presets,
            preset_names,
            rows,
            values,
            status: self.shape.status.clone(),
        }
    }

    // Bring what the panel shows up to date, and build its rows again when
    // what they lay out changed.
    pub(in crate::editor::hook) fn prepare_shape(
        &mut self,
        world: &mut World,
        built: &mut BuiltPanel,
    ) {
        if self.shape_cache_stale(world) {
            let data = self.shape_data(world);
            let live = data
                .binding
                .as_ref()
                .and_then(|b| live_mesh(world, &b.mesh));
            // Unbound, the status says what to select; bound, it is the
            // hook's own, read each frame.
            let prompt = if data.binding.is_none() {
                data.status
            } else {
                None
            };
            let shape = data
                .binding
                .as_ref()
                .and_then(|b| b.shape_idx)
                .and_then(|i| self.entries.get(i))
                .and_then(declared_id)
                .map(String::from);
            self.shape.cache = Some(ShapeCache {
                revision: self.entries.revision(),
                active: self.selection.active().cloned(),
                mesh: data.binding.map(|b| b.mesh),
                live,
                shape,
                status: prompt,
                values: data.values,
                layout: ShapeLayout {
                    rows: data.rows,
                    sections: data.derived.sections,
                    sliders: data.derived.sliders,
                    presets: data.preset_names,
                },
            });
        }
        let Some(cache) = &self.shape.cache else {
            return;
        };
        if self
            .shape
            .shown
            .as_ref()
            .is_none_or(|ids| ids.layout != cache.layout)
        {
            self.shape.shown = Some(character_shape_panel::build_rows(
                world,
                built,
                &cache.layout,
            ));
        }
    }

    // Whether the panel's cached rows no longer follow the entries, the
    // selection, or the live mesh they were read from.
    fn shape_cache_stale(&self, world: &World) -> bool {
        let Some(cache) = &self.shape.cache else {
            return true;
        };
        cache.revision != self.entries.revision()
            || cache.active.as_ref() != self.selection.active()
            || cache.mesh.as_deref().and_then(|m| live_mesh(world, m)) != cache.live
    }

    pub(in crate::editor::hook) fn make_shape_view(&self) -> Option<ShapeView<'_>> {
        let cache = self.shape.cache.as_ref()?;
        Some(ShapeView {
            shape: cache.shape.as_deref(),
            status: self.shape.status.as_deref().or(cache.status.as_deref()),
            values: &cache.values,
        })
    }

    // Act on what the user asked of the panel.
    pub(in crate::editor::hook) fn apply_shape_action(
        &mut self,
        action: ShapeAction,
        world: &mut World,
    ) {
        match action {
            ShapeAction::Change { slider, value } => {
                self.drag_shape_slider(slider, value, world);
                return;
            }
            ShapeAction::Commit { slider, value } => {
                self.release_shape_slider(slider, value);
                return;
            }
            _ => {}
        }
        let data = self.shape_data(world);
        let shape_idx = data.binding.as_ref().and_then(|b| b.shape_idx);
        match action {
            ShapeAction::Reset => {
                if let Some(idx) = shape_idx {
                    let mut values = self.shape_values(idx);
                    values.reset(&data.derived.sliders, &data.target.joint_names);
                    self.commit_shape(idx, &values);
                }
            }
            ShapeAction::Randomize => {
                if let Some(idx) = shape_idx {
                    self.shape.seed = self.shape.seed.wrapping_add(1);
                    let mut values = self.shape_values(idx);
                    values.randomize(
                        &data.derived.sliders,
                        &data.target.joint_names,
                        self.shape.seed,
                    );
                    self.commit_shape(idx, &values);
                }
            }
            ShapeAction::Preset(p) => {
                if let (Some(idx), Some(preset)) = (shape_idx, data.presets.get(p)) {
                    let mut values = self.shape_values(idx);
                    values.apply_preset(preset);
                    self.commit_shape(idx, &values);
                }
            }
            ShapeAction::Add => {
                if let Some(b) = &data.binding
                    && b.shape_idx.is_none()
                {
                    self.add_shape(&b.mesh);
                }
            }
            ShapeAction::Change { .. } | ShapeAction::Commit { .. } => {}
        }
    }

    // Write `values` into shape entry `idx` as ONE edit.
    pub(in crate::editor::hook) fn commit_shape(&mut self, idx: usize, values: &ShapeValues) {
        self.shape.status = None;
        let mut existing = self.entry_args(idx);
        let Ok(sliders) = serde_json::to_value(&values.sliders) else {
            return;
        };
        let Ok(proportions) = serde_json::to_value(&values.proportions) else {
            return;
        };
        existing.insert("sliders".to_string(), sliders);
        existing.insert("proportions".to_string(), proportions);
        let args = form::assemble("CharacterShape", Some(&existing), &[], &[]);
        let name = self
            .entries
            .get(idx)
            .and_then(declared_id)
            .unwrap_or("CharacterShape");
        if let Err(e) = form::validate("CharacterShape", name, &args) {
            self.shape.status = Some(short_status(&e));
            return;
        }
        if let Some(entry) = self.entries.get_mut(idx) {
            replace_args(entry, serde_json::Value::Object(args));
        }
        self.mark_changed();
    }

    // Add a shape targeting `mesh` with no sliders or proportions yet.
    pub(super) fn add_shape(&mut self, mesh: &str) {
        let name = self.unique_name("CharacterShape");
        self.entries.push(serde_json::json!({
            "type": "CharacterShape", "args": { "$id": name, "target": mesh },
        }));
        self.mark_changed();
    }
}
