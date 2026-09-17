// Companion-asset declarations. Some assets imply others must exist to
// function: anything that renders needs a GraphicsConfig, and a GraphicsConfig
// in turn needs a Window. Which types render is the registry's `renders` flag;
// GraphicsConfig's own companions are declared here. The injection pass in
// `companion.rs` applies the resulting specs to the world.
//
// This is build-time-only authoring logic; the asset data structs live in
// concinnity-core alongside their runtime `Component` impls.

use crate::authoring::registry::RegisteredType;

// A companion asset implied by the presence of another asset in the world. The
// injection pass adds one only if no asset of the companion's `asset_type` is
// already present.
#[derive(Debug, Clone)]
pub(crate) struct CompanionSpec {
    // Default name for the injected asset (e.g. "GraphicsConfig").
    pub name: &'static str,
    // The asset type to inject.
    pub asset_type: RegisteredType,
    // JSON args for the injected asset.
    pub args: serde_json::Value,
}

// The lone GraphicsConfig companion shared by every renderable asset: its
// presence is the marker that a world renders.
fn graphics_config_marker() -> Vec<CompanionSpec> {
    vec![CompanionSpec {
        name: "GraphicsConfig",
        asset_type: RegisteredType::GraphicsConfig,
        args: serde_json::json!({}),
    }]
}

// GraphicsConfig is the marker that a world renders: its presence gates the
// internal GraphicsSystem at runtime and pulls in the Window that system needs.
fn graphics_config_companions() -> Vec<CompanionSpec> {
    vec![CompanionSpec {
        name: "Window",
        asset_type: RegisteredType::Window,
        args: serde_json::json!({}),
    }]
}

// Companion specs implied by one asset of the given type. GraphicsConfig
// declares the render stack; every other type flagged `renders` in the registry
// implies the GraphicsConfig marker. Remaining types imply none.
pub(crate) fn companions_for(asset_type: RegisteredType) -> Vec<CompanionSpec> {
    if asset_type == RegisteredType::GraphicsConfig {
        graphics_config_companions()
    } else if asset_type.renders() {
        graphics_config_marker()
    } else {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderable_assets_imply_graphics_config() {
        for ty in [
            RegisteredType::Prop,
            RegisteredType::Sprite,
            RegisteredType::TextLabel,
            RegisteredType::VoxelWorld,
            RegisteredType::WaterSurface,
            RegisteredType::InstancedProp,
            RegisteredType::SkinnedMesh,
            RegisteredType::EnvironmentMap,
            RegisteredType::MainMenu,
        ] {
            let specs = companions_for(ty);
            assert!(
                specs
                    .iter()
                    .any(|c| c.asset_type == RegisteredType::GraphicsConfig),
                "{} should imply a GraphicsConfig companion",
                ty.as_str()
            );
        }
    }

    #[test]
    fn a_non_rendering_type_implies_no_companions() {
        assert!(companions_for(RegisteredType::Window).is_empty());
        assert!(companions_for(RegisteredType::Mesh).is_empty());
    }

    #[test]
    fn graphics_config_injects_a_window() {
        let specs = companions_for(RegisteredType::GraphicsConfig);
        assert!(specs.iter().any(|c| c.asset_type == RegisteredType::Window));
        assert!(!specs.iter().any(|c| c.asset_type == RegisteredType::Shader));
    }
}
