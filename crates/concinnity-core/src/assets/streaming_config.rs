// src/assets/streaming_config.rs

use crate::assets::StreamingConfig;
use crate::ecs::{AssetOrigin, Component};

impl Component for StreamingConfig {
    const NAME: &'static str = "StreamingConfig";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }

    fn from_args(args: Self) -> Self {
        args
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_a_moderate_budget_and_a_full_cap() {
        let c = StreamingConfig::default();
        assert_eq!(c.texture_budget, 4);
        assert_eq!(c.texture_cap, 96);
        assert_eq!(c.budget(), 4);
        assert_eq!(c.cap(), 96);
        assert_eq!(c.mesh_budget(), 4);
        assert_eq!(c.mesh_cap(), 4096);
    }

    #[test]
    fn zero_budget_and_cap_are_floored_at_one() {
        let c = StreamingConfig {
            texture_budget: 0,
            texture_cap: 0,
            mesh_budget: 0,
            mesh_cap: 0,
        };
        // A 0 here would otherwise stall streaming forever.
        assert_eq!(c.budget(), 1);
        assert_eq!(c.cap(), 1);
        assert_eq!(c.mesh_budget(), 1);
        assert_eq!(c.mesh_cap(), 1);
    }

    #[test]
    fn deserialises_from_jsonl_args_with_defaults_for_omitted_fields() {
        let c: StreamingConfig =
            serde_json::from_str(r#"{"texture_budget":2,"mesh_budget":2}"#).expect("parse");
        assert_eq!(c.texture_budget, 2);
        assert_eq!(c.mesh_budget, 2);
        // Omitted fields fall back to the defaults.
        assert_eq!(c.texture_cap, 96);
        assert_eq!(c.mesh_cap, 4096);

        // An empty object is all defaults.
        let c: StreamingConfig = serde_json::from_str("{}").expect("parse");
        assert_eq!(c.texture_budget, 4);
        assert_eq!(c.texture_cap, 96);
        assert_eq!(c.mesh_budget, 4);
        assert_eq!(c.mesh_cap, 4096);
    }

    #[test]
    fn round_trips_through_args() {
        let c = StreamingConfig {
            texture_budget: 7,
            texture_cap: 32,
            mesh_budget: 3,
            mesh_cap: 64,
        };
        let back = StreamingConfig::from_args(c.to_args());
        assert_eq!(back.texture_budget, 7);
        assert_eq!(back.texture_cap, 32);
        assert_eq!(back.mesh_budget, 3);
        assert_eq!(back.mesh_cap, 64);
    }
}
