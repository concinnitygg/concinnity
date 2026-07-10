// src/assets/prop.rs

use crate::assets::Prop;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, CompanionSpec, Component};

impl Component for Prop {
    const NAME: &'static str = "Prop";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(mut args: Self) -> Self {
        args.cull_distance = args.cull_distance.max(0.0);
        args.is_held = false;
        args
    }

    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }

    fn companions(_args: &serde_json::Value, _world: &[serde_json::Value]) -> Vec<CompanionSpec> {
        vec![CompanionSpec {
            name: "GraphicsConfig",
            asset_type: "GraphicsConfig",
            args: serde_json::json!({}),
        }]
    }
}

impl crate::check::cross_reference::CrossReferenced for Prop {
    fn cross_refs(
        name: &str,
        args: &serde_json::Value,
    ) -> Vec<crate::check::cross_reference::CrossRef> {
        use crate::check::cross_reference::{CrossRef, RefKind};
        let arg = |key: &str| args.get(key).and_then(|v| v.as_str()).unwrap_or("");
        let mut refs = Vec::new();

        // A Model takes precedence over a Mesh; only the one in effect is checked.
        let model_ref = arg("model");
        let mesh_ref = arg("mesh");
        if !model_ref.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Model,
                target: model_ref.to_string(),
                error: format!(
                    "Prop '{}': model '{}' not found, add a Model asset with that name",
                    name, model_ref
                ),
            });
        } else if !mesh_ref.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::MeshSource,
                target: mesh_ref.to_string(),
                error: format!(
                    "Prop '{}': mesh '{}' not found, add a Mesh, ProceduralMesh, or File (obj) asset with that name",
                    name, mesh_ref
                ),
            });
        }

        let mat_ref = arg("material");
        if !mat_ref.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Material,
                target: mat_ref.to_string(),
                error: format!(
                    "Prop '{}': material '{}' not found, add a Material asset with that name",
                    name, mat_ref
                ),
            });
        }

        let tex_ref = arg("texture");
        if !tex_ref.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Texture,
                target: tex_ref.to_string(),
                error: format!(
                    "Prop '{}': texture '{}' not found, add a Texture asset with that name",
                    name, tex_ref
                ),
            });
        }

        let parent_ref = arg("parent");
        if !parent_ref.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Prop,
                target: parent_ref.to_string(),
                error: format!(
                    "Prop '{}': parent '{}' not found, add a Prop asset with that name",
                    name, parent_ref
                ),
            });
        }

        refs
    }
}
