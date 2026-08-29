// src/world/companion_specs.rs
//
// Companion-asset declarations. Some assets imply others must exist to
// function: anything that renders needs a GraphicsConfig, and a GraphicsConfig
// in turn needs a Window. Which types render is the registry's `renders` flag;
// GraphicsConfig's own companions are declared here. The injection pass in
// `companion.rs` applies the resulting specs to the world.
//
// This is build-time-only authoring logic; the asset data structs live in
// concinnity-core alongside their runtime `Component` impls.

// A companion asset implied by the presence of another asset in the world. The
// injection pass adds one only if no asset of the companion's `asset_type` is
// already present (case-insensitive, underscores stripped).
#[derive(Debug, Clone)]
pub(crate) struct CompanionSpec {
    // Default name for the injected asset (e.g. "GraphicsConfig").
    pub name: &'static str,
    // The asset type to inject.
    pub asset_type: &'static str,
    // JSON args for the injected asset.
    pub args: serde_json::Value,
}

// The lone GraphicsConfig companion shared by every renderable asset: its
// presence is the marker that a world renders.
fn graphics_config_marker() -> Vec<CompanionSpec> {
    vec![CompanionSpec {
        name: "GraphicsConfig",
        asset_type: "GraphicsConfig",
        args: serde_json::json!({}),
    }]
}

// GraphicsConfig is the marker that a world renders: its presence gates the
// internal GraphicsSystem at runtime and pulls in the Window that system needs.
fn graphics_config_companions() -> Vec<CompanionSpec> {
    vec![CompanionSpec {
        name: "Window",
        asset_type: "Window",
        args: serde_json::json!({}),
    }]
}

// Companion specs implied by one asset of the given normalized type.
// GraphicsConfig declares the render stack; every other type flagged `renders`
// in the registry implies the GraphicsConfig marker. Remaining types imply none.
pub(crate) fn companions_for(type_norm: &str) -> Vec<CompanionSpec> {
    if type_norm == "graphicsconfig" {
        graphics_config_companions()
    } else if crate::registry::type_renders(type_norm) {
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
            "prop",
            "sprite",
            "textlabel",
            "voxelworld",
            "watersurface",
            "instancedprop",
            "skinnedmesh",
        ] {
            let specs = companions_for(ty);
            assert!(
                specs.iter().any(|c| c.asset_type == "GraphicsConfig"),
                "{ty} should imply a GraphicsConfig companion"
            );
        }
    }

    #[test]
    fn unknown_type_implies_no_companions() {
        assert!(companions_for("window").is_empty());
        assert!(companions_for("mesh").is_empty());
    }

    #[test]
    fn graphics_config_injects_a_window() {
        let specs = companions_for("graphicsconfig");
        assert!(specs.iter().any(|c| c.asset_type == "Window"));
        assert!(!specs.iter().any(|c| c.asset_type == "Shader"));
    }
}
