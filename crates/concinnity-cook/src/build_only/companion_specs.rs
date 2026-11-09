// Companion-asset declarations. Some assets imply others must exist to
// function: anything that renders needs a Window to render into, and a Window
// the GraphicsConfig its renderer is configured by. Which types render is the
// registry's `renders` flag, the same one the runtime resolves a windowed run
// from. The injection pass in `companion.rs` applies the resulting specs to
// the world.
//
// This is build-time-only authoring logic; the asset data structs live in
// concinnity-engine alongside their runtime `Component` impls.

use crate::authoring::registry::RegisteredType;

// A companion asset implied by the presence of another asset in the world. The
// injection pass adds one only if no asset of the companion's `asset_type` is
// already present.
#[derive(Debug, Clone)]
pub(crate) struct CompanionSpec {
    // Default name for the injected asset (e.g. "Window").
    pub name: &'static str,
    // The asset type to inject.
    pub asset_type: RegisteredType,
    // JSON args for the injected asset.
    pub args: serde_json::Value,
}

// The lone companion shared by every renderable asset: the window it draws
// into.
fn window_companion() -> Vec<CompanionSpec> {
    vec![CompanionSpec {
        name: "Window",
        asset_type: RegisteredType::Window,
        args: serde_json::json!({}),
    }]
}

// The settings a Window's renderer is built with: every field unset, so each
// follows the quality tier until the world sets it.
fn graphics_companion() -> Vec<CompanionSpec> {
    vec![CompanionSpec {
        name: "GraphicsConfig",
        asset_type: RegisteredType::GraphicsConfig,
        args: serde_json::json!({}),
    }]
}

// Companion specs implied by one asset of the given type. A Window implies a
// GraphicsConfig; every other type flagged `renders` in the registry implies a
// Window, which then implies its GraphicsConfig on the next round. The rest
// imply none.
pub(crate) fn companions_for(asset_type: RegisteredType) -> Vec<CompanionSpec> {
    if asset_type == RegisteredType::Window {
        graphics_companion()
    } else if asset_type.renders() {
        window_companion()
    } else {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderable_assets_imply_a_window() {
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
            RegisteredType::Screen,
        ] {
            let specs = companions_for(ty);
            assert!(
                specs.iter().any(|c| c.asset_type == RegisteredType::Window),
                "{} should imply a Window companion",
                ty.as_str()
            );
            assert!(
                !specs.iter().any(|c| c.asset_type == RegisteredType::Shader),
                "{} should not imply a Shader",
                ty.as_str()
            );
        }
    }

    #[test]
    fn a_non_rendering_type_implies_no_companions() {
        assert!(companions_for(RegisteredType::Mesh).is_empty());
        assert!(companions_for(RegisteredType::PhysicsConfig).is_empty());
    }

    // Content decides whether a world renders, so the settings tuning the
    // renderer imply no Window of their own.
    #[test]
    fn a_settings_asset_implies_no_window() {
        assert!(companions_for(RegisteredType::GraphicsConfig).is_empty());
        assert!(companions_for(RegisteredType::StreamingConfig).is_empty());
    }

    #[test]
    fn a_window_implies_its_graphics_config() {
        let specs = companions_for(RegisteredType::Window);
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].asset_type, RegisteredType::GraphicsConfig);
        assert_eq!(specs[0].args, serde_json::json!({}));
    }
}
