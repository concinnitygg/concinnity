// src/editor/hook/character_shape_edit.rs
//
// EditorHook: the CharacterShape panel's actions. The panel edits the shape
// whose target is the selected skinned mesh or character model (or the
// selected shape entry); its rows are the target's schema laid over what
// that mesh exposes in the live world (`character_shape.rs`). Every commit
// (a slider release, Reset, Randomize, a preset) writes the entry's args
// through `form::assemble` + `form::validate` and `mark_changed` once; the
// per-frame drag preview lives in `shape_drag.rs`.

use super::*;
use concinnity_world::registry::build_only::CharacterSchema;
use concinnity_world::registry::build_only::ShapePreset;

use crate::components::CharacterCapsule;
use crate::editor::character_shape::{self, Row, Rows, ShapeValues};
use crate::editor::character_shape_panel::{self, ShapeAction, ShapeView};
use crate::gfx::shape_preview::{self, ShapeTarget};
use concinnity_cook::character::builtin_schema;

// What the panel edits: the selected skinned mesh and the shape entry
// targeting it, if the world has one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ShapeBinding {
    pub(super) mesh: String,
    pub(super) shape_idx: Option<usize>,
}

// Owned per-tick data backing a `ShapeView`.
pub(super) struct ShapeData {
    pub(super) binding: Option<ShapeBinding>,
    pub(super) target: ShapeTarget,
    pub(super) derived: Rows,
    pub(super) presets: Vec<ShapePreset>,
    pub(super) preset_names: Vec<String>,
    pub(super) rows: Vec<Row>,
    pub(super) values: Vec<f32>,
    pub(super) status: Option<String>,
}

impl EditorHook {
    // The binding the selection implies: a selected SkinnedMesh or
    // CharacterModel and the first shape targeting it, or a selected
    // CharacterShape and its target.
    pub(super) fn shape_binding(&self) -> Option<ShapeBinding> {
        let name = self.selection.active()?;
        let idx = self
            .entries
            .iter()
            .position(|e| entry_name(e) == Some(name))?;
        let entry = &self.entries[idx];
        match entry_type(entry)? {
            "SkinnedMesh" | "CharacterModel" => Some(ShapeBinding {
                mesh: name.to_string(),
                shape_idx: self.entries.iter().position(|e| {
                    entry_type(e) == Some("CharacterShape")
                        && e.pointer("/args/target").and_then(|t| t.as_str()) == Some(name)
                }),
            }),
            "CharacterShape" => {
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
        if let Some(t) = crate::ecs::asset_id::lookup(mesh)
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
        self.entries
            .iter()
            .position(|e| entry_name(e) == Some(name))
    }

    // The schema that lays out `mesh`'s panel: a CharacterModel's named
    // schema (a CharacterSchema entry, or the bundled humanoid); a plain
    // SkinnedMesh gets the bundled humanoid's, which places whatever keys
    // and joints it shares and leaves the rest under Other.
    pub(super) fn shape_schema(&self, mesh: &str) -> CharacterSchema {
        let Some(idx) = self.entry_index_named(mesh) else {
            return builtin_schema::humanoid().clone();
        };
        if entry_type(&self.entries[idx]) != Some("CharacterModel") {
            return builtin_schema::humanoid().clone();
        }
        let name = self
            .entry_args(idx)
            .get("schema")
            .and_then(|v| v.as_str())
            .unwrap_or(builtin_schema::HUMANOID_SCHEMA)
            .to_string();
        self.entry_index_named(&name)
            .filter(|&i| entry_type(&self.entries[i]) == Some("CharacterSchema"))
            .and_then(|i| {
                serde_json::from_value(serde_json::Value::Object(self.entry_args(i))).ok()
            })
            .unwrap_or_else(|| builtin_schema::humanoid().clone())
    }

    // The authored slider / proportion values of shape entry `idx`.
    pub(super) fn shape_values(&self, idx: usize) -> ShapeValues {
        let args = self.entry_args(idx);
        let parse = |key: &str| args.get(key).cloned().unwrap_or(serde_json::Value::Null);
        ShapeValues {
            sliders: serde_json::from_value(parse("sliders")).unwrap_or_default(),
            proportions: serde_json::from_value(parse("proportions")).unwrap_or_default(),
        }
    }

    // The authored capsule of `mesh`, the base the preview scales.
    pub(super) fn mesh_capsule(&self, mesh: &str) -> Option<CharacterCapsule> {
        let idx = self.entry_index_named(mesh)?;
        let capsule = self.entry_args(idx).get("capsule")?.clone();
        serde_json::from_value(capsule).ok()
    }

    pub(super) fn shape_data(&self, world: &World) -> ShapeData {
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
        let values: Vec<f32> = match (&self.shape_drag, b.shape_idx) {
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
            status: self.shape_status.clone(),
        }
    }

    pub(super) fn make_shape_view<'a>(
        &'a self,
        d: &'a ShapeData,
        mouse: [f32; 2],
    ) -> ShapeView<'a> {
        let shape = d
            .binding
            .as_ref()
            .and_then(|b| b.shape_idx)
            .and_then(|i| self.entries.get(i))
            .and_then(entry_name);
        ShapeView {
            rows: &d.rows,
            sections: &d.derived.sections,
            sliders: &d.derived.sliders,
            presets: &d.preset_names,
            values: &d.values,
            scroll: self.shape_scroll.min(d.rows.len().saturating_sub(1)),
            dragging: self.shape_drag.as_ref().map(|d| d.slider),
            shape,
            status: d.status.as_deref(),
            mouse,
        }
    }

    // Keep the row count the panel sizes itself by current with the live
    // world, once a frame ahead of the draw.
    pub(super) fn sample_shape_rows(&mut self, world: &World) {
        if !self.shape_open {
            return;
        }
        self.shape_rows = self.shape_data(world).rows.len();
        let max = self
            .shape_rows
            .saturating_sub(character_shape_panel::window(
                self.effective_size(PanelKey::CharacterShape)[1],
            ));
        self.shape_scroll = self.shape_scroll.min(max);
    }

    pub(super) fn scroll_shape(&mut self, delta: f32) {
        let window =
            character_shape_panel::window(self.effective_size(PanelKey::CharacterShape)[1]);
        let max = self.shape_rows.saturating_sub(window);
        self.shape_scroll = scroll_step(self.shape_scroll, delta, max);
    }

    // Route a resolved panel click.
    pub(super) fn apply_shape_action(
        &mut self,
        action: ShapeAction,
        data: &ShapeData,
        mouse: [f32; 2],
        world: &mut World,
    ) {
        match action {
            ShapeAction::Reset => {
                if let Some(idx) = data.binding.as_ref().and_then(|b| b.shape_idx) {
                    let mut values = self.shape_values(idx);
                    values.reset(&data.derived.sliders, &data.target.joint_names);
                    self.commit_shape(idx, &values);
                }
            }
            ShapeAction::Randomize => {
                if let Some(idx) = data.binding.as_ref().and_then(|b| b.shape_idx) {
                    self.shape_seed = self.shape_seed.wrapping_add(1);
                    let mut values = self.shape_values(idx);
                    values.randomize(
                        &data.derived.sliders,
                        &data.target.joint_names,
                        self.shape_seed,
                    );
                    self.commit_shape(idx, &values);
                }
            }
            ShapeAction::Preset(p) => {
                if let (Some(idx), Some(preset)) = (
                    data.binding.as_ref().and_then(|b| b.shape_idx),
                    data.presets.get(p),
                ) {
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
            ShapeAction::Drag { slider, rect } => {
                self.begin_shape_drag(data, slider, rect, mouse, world)
            }
            ShapeAction::Consume => {}
        }
    }

    // Write `values` into shape entry `idx` as ONE edit.
    pub(super) fn commit_shape(&mut self, idx: usize, values: &ShapeValues) {
        self.shape_status = None;
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
            .and_then(entry_name)
            .unwrap_or("CharacterShape");
        if let Err(e) = form::validate("CharacterShape", name, &args) {
            self.shape_status = Some(short_status(&e));
            return;
        }
        if let Some(obj) = self.entries.get_mut(idx).and_then(|e| e.as_object_mut()) {
            obj.insert("args".to_string(), serde_json::Value::Object(args));
        }
        self.mark_changed();
    }

    // Add a shape targeting `mesh` with no sliders or proportions yet.
    pub(super) fn add_shape(&mut self, mesh: &str) {
        let name = self.unique_name("CharacterShape");
        self.entries.push(serde_json::json!({
            "name": name, "type": "CharacterShape", "args": { "target": mesh },
        }));
        self.mark_changed();
    }
}
