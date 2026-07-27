// src/world/companion_specs.rs
//
// Companion-asset declarations. Some assets imply others must exist to
// function: anything that renders needs a GraphicsConfig, and a GraphicsConfig
// in turn needs a Window (and the bundled default shader set when the world
// declares none). Which types render is the registry's `renders` flag;
// GraphicsConfig's own companions are declared here. The injection pass in
// `companion.rs` applies the resulting specs to the world.
//
// This is build-time-only authoring logic; the asset data structs live in
// concinnity-asset and their runtime `Component` impls in concinnity-core.

// A companion asset implied by the presence of another asset in the world. The
// injection pass adds one only if no asset of the companion's `asset_type` is
// already present (case-insensitive, underscores stripped).
#[derive(Debug, Clone)]
pub struct CompanionSpec {
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
// internal GraphicsSystem at runtime and pulls in the assets that system
// needs: a Window and, when the world declares no Shader of its own, the
// bundled default shader set.
fn graphics_config_companions(world: &[serde_json::Value]) -> Vec<CompanionSpec> {
    let mut specs = vec![CompanionSpec {
        name: "Window",
        asset_type: "Window",
        args: serde_json::json!({}),
    }];

    // Inject the bundled default Shader only when the world declares no
    // Shader at all. A world with even one custom Shader owns its pipeline
    // and is left alone.
    let has_shader = world.iter().any(|v| {
        v.get("type")
            .and_then(|t| t.as_str())
            .map(|s| s.to_lowercase().replace('_', "") == "shader")
            .unwrap_or(false)
    });
    if !has_shader {
        // The GPU-instanced vertex stage is only needed when the world has an
        // InstancedProp; without it the backend cannot build the instanced
        // main/SSR/SSAO/velocity pipelines and instanced clusters silently
        // never rasterize. Declare it on demand so worlds without instancing
        // don't pay for an extra shader compile.
        let has_instanced = world.iter().any(|v| {
            v.get("type")
                .and_then(|t| t.as_str())
                .map(|s| s.to_lowercase().replace('_', "") == "instancedprop")
                .unwrap_or(false)
        });
        let mut args = serde_json::json!({
            "vertex": {"sources": {"metal": "default.metal", "hlsl": "default_vert.hlsl"}},
            "fragment": {"sources": {"metal": "default.metal", "hlsl": "default_frag.hlsl"}}
        });
        if has_instanced {
            args["vertex_instanced"] = serde_json::json!({
                "sources": {"metal": "default.metal", "hlsl": "default_vert_instanced.hlsl"}
            });
        }
        specs.push(CompanionSpec {
            name: "default_shader",
            asset_type: "Shader",
            args,
        });
    }
    specs
}

// Companion specs implied by one asset of the given normalized type. `world`
// is the pre-injection asset list, read by types whose companions depend on
// world state (GraphicsConfig's default shader set). GraphicsConfig itself
// declares the render stack; every other type flagged `renders` in the
// registry implies the GraphicsConfig marker. Remaining types imply none.
pub(crate) fn companions_for(
    type_norm: &str,
    _args: &serde_json::Value,
    world: &[serde_json::Value],
) -> Vec<CompanionSpec> {
    if type_norm == "graphicsconfig" {
        graphics_config_companions(world)
    } else if crate::registry::type_renders(type_norm) {
        graphics_config_marker()
    } else {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
            let specs = companions_for(ty, &json!({}), &[]);
            assert!(
                specs.iter().any(|c| c.asset_type == "GraphicsConfig"),
                "{ty} should imply a GraphicsConfig companion"
            );
        }
    }

    #[test]
    fn unknown_type_implies_no_companions() {
        assert!(companions_for("window", &json!({}), &[]).is_empty());
        assert!(companions_for("mesh", &json!({}), &[]).is_empty());
    }

    #[test]
    fn graphics_config_injects_window_and_default_shader() {
        let specs = companions_for("graphicsconfig", &json!({}), &[]);
        assert!(specs.iter().any(|c| c.asset_type == "Window"));
        let shader = specs
            .iter()
            .find(|c| c.asset_type == "Shader")
            .expect("default Shader spec");
        assert!(shader.args["vertex"].is_object());
        assert!(shader.args["fragment"].is_object());
        // No instanced stage without an InstancedProp in the world.
        assert!(shader.args.get("vertex_instanced").is_none());
    }

    #[test]
    fn graphics_config_skips_default_shaders_when_world_has_one() {
        let world = vec![json!({"type": "Shader", "args": {"vertex": {"source": "x.metal"}}})];
        let specs = companions_for("graphicsconfig", &json!({}), &world);
        assert!(specs.iter().any(|c| c.asset_type == "Window"));
        assert!(!specs.iter().any(|c| c.asset_type == "Shader"));
    }

    #[test]
    fn graphics_config_adds_instanced_stage_when_world_has_instancing() {
        let world = vec![json!({"type": "InstancedProp", "args": {}})];
        let specs = companions_for("graphicsconfig", &json!({}), &world);
        let shader = specs
            .iter()
            .find(|c| c.asset_type == "Shader")
            .expect("default Shader spec");
        assert!(shader.args["vertex_instanced"].is_object());
    }
}
