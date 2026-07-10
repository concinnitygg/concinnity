// src/assets/instanced_prop.rs

use crate::assets::InstancedProp;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, CompanionSpec, Component};

impl Component for InstancedProp {
    const NAME: &'static str = "InstancedProp";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn from_args(mut args: Self) -> Self {
        args.cull_distance = args.cull_distance.max(0.0);
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
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

impl crate::check::cross_reference::CrossReferenced for InstancedProp {
    fn cross_refs(
        name: &str,
        args: &serde_json::Value,
    ) -> Vec<crate::check::cross_reference::CrossRef> {
        use crate::check::cross_reference::{CrossRef, RefKind};
        let arg = |key: &str| args.get(key).and_then(|v| v.as_str()).unwrap_or("");
        let mut refs = Vec::new();

        let mesh_ref = arg("mesh");
        if mesh_ref.is_empty() {
            refs.push(CrossRef::Issue(format!(
                "InstancedProp '{}': `mesh` field is required",
                name
            )));
        } else {
            refs.push(CrossRef::Resolve {
                kind: RefKind::MeshSource,
                target: mesh_ref.to_string(),
                error: format!(
                    "InstancedProp '{}': mesh '{}' not found, add a Mesh, ProceduralMesh, VoxelChunk, or File (obj) asset with that name",
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
                    "InstancedProp '{}': material '{}' not found, add a Material asset with that name",
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
                    "InstancedProp '{}': texture '{}' not found, add a Texture asset with that name",
                    name, tex_ref
                ),
            });
        }

        refs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{InstanceTransform, InstancedPropGeometry};

    fn empty() -> InstancedProp {
        InstancedProp {
            asset_id: AssetId::default(),
            mesh: None,
            material: None,
            texture: None,
            instances: Vec::new(),
            cull_distance: 0.0,
        }
    }

    #[test]
    fn instance_model_matrix_default_is_identity() {
        let mut p = empty();
        p.instances.push(InstanceTransform::default());
        let m = p.instance_model_matrix(0).unwrap();
        assert_eq!(m[3], [0.0, 0.0, 0.0, 1.0]);
        assert!((m[0][0] - 1.0).abs() < 1e-5);
        assert!((m[1][1] - 1.0).abs() < 1e-5);
        assert!((m[2][2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn instance_model_matrix_translates() {
        let mut p = empty();
        p.instances.push(InstanceTransform {
            position: [5.0, -2.0, 3.0],
            ..InstanceTransform::default()
        });
        let m = p.instance_model_matrix(0).unwrap();
        assert_eq!(m[3], [5.0, -2.0, 3.0, 1.0]);
    }

    #[test]
    fn instance_model_matrix_scales() {
        let mut p = empty();
        p.instances.push(InstanceTransform {
            scale: [2.0, 3.0, 4.0],
            ..InstanceTransform::default()
        });
        let m = p.instance_model_matrix(0).unwrap();
        // diagonal entries should be the scale factors (no rotation)
        assert!((m[0][0] - 2.0).abs() < 1e-5);
        assert!((m[1][1] - 3.0).abs() < 1e-5);
        assert!((m[2][2] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn instance_model_matrix_out_of_range_returns_none() {
        let p = empty();
        assert!(p.instance_model_matrix(0).is_none());
    }

    #[test]
    fn from_args_clamps_negative_cull_distance() {
        let args = InstancedProp {
            cull_distance: -5.0,
            ..InstancedProp::default()
        };
        let p = InstancedProp::from_args(args);
        assert_eq!(p.cull_distance, 0.0);
    }
}
