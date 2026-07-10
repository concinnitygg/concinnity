// src/assets/model.rs

use crate::assets::Model;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, Component};

impl Component for Model {
    const NAME: &'static str = "Model";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }

    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }
}

impl crate::check::cross_reference::CrossReferenced for Model {
    fn cross_refs(
        name: &str,
        args: &serde_json::Value,
    ) -> Vec<crate::check::cross_reference::CrossRef> {
        use crate::check::cross_reference::{CrossRef, RefKind};
        let mut refs = Vec::new();

        if let Some(meshes) = args.get("meshes").and_then(|v| v.as_array()) {
            for (i, sub) in meshes.iter().enumerate() {
                let sub_mesh = sub.get("mesh").and_then(|v| v.as_str()).unwrap_or("");
                if sub_mesh.is_empty() {
                    refs.push(CrossRef::Issue(format!(
                        "Model '{}': submesh[{}] is missing a 'mesh' field",
                        name, i
                    )));
                } else {
                    refs.push(CrossRef::Resolve {
                        kind: RefKind::MeshSource,
                        target: sub_mesh.to_string(),
                        error: format!(
                            "Model '{}': submesh[{}] mesh '{}' not found, add a Mesh, ProceduralMesh, or File (obj) asset with that name",
                            name, i, sub_mesh
                        ),
                    });
                }

                let sub_mat = sub.get("material").and_then(|v| v.as_str()).unwrap_or("");
                if !sub_mat.is_empty() {
                    refs.push(CrossRef::Resolve {
                        kind: RefKind::Material,
                        target: sub_mat.to_string(),
                        error: format!(
                            "Model '{}': submesh[{}] material '{}' not found, add a Material asset with that name",
                            name, i, sub_mat
                        ),
                    });
                }
            }
        }

        refs
    }
}
