// src/assets/graphics_config.rs

use crate::assets::GraphicsConfig;
use crate::ecs::{AssetOrigin, CompanionSpec, Component};

impl Component for GraphicsConfig {
    const NAME: &'static str = "GraphicsConfig";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }

    // GraphicsConfig is the marker that a world renders: its presence gates the
    // internal GraphicsSystem at runtime and pulls in the assets that system
    // needs: a Window and, when the world declares no ShaderStage of its own,
    // the bundled default shader set.
    fn companions(_args: &serde_json::Value, world: &[serde_json::Value]) -> Vec<CompanionSpec> {
        let mut specs = vec![CompanionSpec {
            name: "Window",
            asset_type: "Window",
            args: serde_json::json!({}),
        }];

        // Inject the bundled default vertex + fragment ShaderStages as a set
        // only when the world declares no ShaderStage at all. A world with even
        // one custom ShaderStage owns its pipeline and is left alone.
        let has_shader = world.iter().any(|v| {
            v.get("type")
                .and_then(|t| t.as_str())
                .map(|s| s.to_lowercase().replace('_', "") == "shaderstage")
                .unwrap_or(false)
        });
        if !has_shader {
            specs.push(CompanionSpec {
                name: "default_vertex_shader",
                asset_type: "ShaderStage",
                args: serde_json::json!({
                    "kind": "vertex",
                    "sources": {"metal": "default.metal", "hlsl": "default_vert.hlsl"}
                }),
            });
            specs.push(CompanionSpec {
                name: "default_fragment_shader",
                asset_type: "ShaderStage",
                args: serde_json::json!({
                    "kind": "fragment",
                    "sources": {"metal": "default.metal", "hlsl": "default_frag.hlsl"}
                }),
            });
            // The GPU-instanced vertex stage is only needed when the world has
            // an InstancedProp; without it the backend cannot build the
            // instanced main/SSR/SSAO/velocity pipelines and instanced clusters
            // silently never rasterize. Inject it on demand so worlds without
            // instancing don't pay for an extra shader compile.
            let has_instanced = world.iter().any(|v| {
                v.get("type")
                    .and_then(|t| t.as_str())
                    .map(|s| s.to_lowercase().replace('_', "") == "instancedprop")
                    .unwrap_or(false)
            });
            if has_instanced {
                specs.push(CompanionSpec {
                    name: "default_instanced_vertex_shader",
                    asset_type: "ShaderStage",
                    args: serde_json::json!({
                        "kind": "vertex_instanced",
                        "sources": {"metal": "default.metal", "hlsl": "default_vert_instanced.hlsl"}
                    }),
                });
            }
        }
        specs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::ShadowUpdate;

    #[test]
    fn shadow_update_defaults_to_hybrid() {
        assert_eq!(
            GraphicsConfig::default().shadow_update,
            ShadowUpdate::Hybrid
        );
        assert_eq!(ShadowUpdate::default(), ShadowUpdate::Hybrid);
    }

    #[test]
    fn shadow_update_round_trips_via_snake_case_json() {
        let cfg: GraphicsConfig =
            serde_json::from_str(r#"{"shadow_update":"every_frame"}"#).expect("parse");
        assert_eq!(cfg.shadow_update, ShadowUpdate::EveryFrame);
        // Omitting the field falls back to the hybrid default.
        let cfg: GraphicsConfig =
            serde_json::from_str(r#"{"shadow_map_size":1024}"#).expect("parse");
        assert_eq!(cfg.shadow_update, ShadowUpdate::Hybrid);
    }

    #[test]
    fn vsync_defaults_off_and_round_trips() {
        // Omitted -> uncapped (false).
        let cfg: GraphicsConfig =
            serde_json::from_str(r#"{"shadow_map_size":1024}"#).expect("parse");
        assert!(!cfg.vsync);
        // Explicit true is honoured.
        let cfg: GraphicsConfig = serde_json::from_str(r#"{"vsync":true}"#).expect("parse");
        assert!(cfg.vsync);
    }

    #[test]
    fn fps_cap_defaults_to_unlimited_and_round_trips() {
        // Omitted -> 0 (uncapped).
        assert_eq!(GraphicsConfig::default().fps_cap, 0);
        let cfg: GraphicsConfig =
            serde_json::from_str(r#"{"shadow_map_size":1024}"#).expect("parse");
        assert_eq!(cfg.fps_cap, 0);
        // Explicit cap is honoured.
        let cfg: GraphicsConfig = serde_json::from_str(r#"{"fps_cap":60}"#).expect("parse");
        assert_eq!(cfg.fps_cap, 60);
    }

    #[test]
    fn shadow_distance_defaults_to_80_and_round_trips() {
        assert_eq!(GraphicsConfig::default().shadow_distance, 80);
        let cfg: GraphicsConfig =
            serde_json::from_str(r#"{"shadow_distance":160}"#).expect("parse");
        assert_eq!(cfg.shadow_distance, 160);
        let cfg: GraphicsConfig =
            serde_json::from_str(r#"{"shadow_map_size":1024}"#).expect("parse");
        assert_eq!(cfg.shadow_distance, 80);
    }

    #[test]
    fn shadow_cascades_defaults_to_4_and_round_trips() {
        assert_eq!(GraphicsConfig::default().shadow_cascades, 4);
        let cfg: GraphicsConfig = serde_json::from_str(r#"{"shadow_cascades":2}"#).expect("parse");
        assert_eq!(cfg.shadow_cascades, 2);
        let cfg: GraphicsConfig =
            serde_json::from_str(r#"{"shadow_map_size":1024}"#).expect("parse");
        assert_eq!(cfg.shadow_cascades, 4);
    }

    #[test]
    fn anisotropy_defaults_to_8_and_round_trips() {
        // The default matches the value the backends historically hardcoded.
        assert_eq!(GraphicsConfig::default().anisotropy, 8);
        // An authored value is honoured; omitting the field falls back to 8.
        let cfg: GraphicsConfig = serde_json::from_str(r#"{"anisotropy":16}"#).expect("parse");
        assert_eq!(cfg.anisotropy, 16);
        let cfg: GraphicsConfig =
            serde_json::from_str(r#"{"shadow_map_size":1024}"#).expect("parse");
        assert_eq!(cfg.anisotropy, 8);
    }

    #[test]
    fn shadow_update_round_trips_through_args() {
        let cfg = GraphicsConfig {
            shadow_update: ShadowUpdate::EveryFrame,
            ..Default::default()
        };
        assert_eq!(
            GraphicsConfig::from_args(cfg.to_args()).shadow_update,
            ShadowUpdate::EveryFrame
        );
    }
}
