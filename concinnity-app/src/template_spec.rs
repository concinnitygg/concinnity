// World-template bridging on top of the core spec bridge.
//
// The generic `AssetSpec` -> `serde_json::Value` conversion lives in
// `concinnity_core::template_spec` (core owns `serde_json` and every std
// consumer depends on it). This module re-exports those primitives so the app's
// public API (`concinnity_app::spec_to_value`, ...) stays stable, and adds the
// world-template convenience the authoring layer uses.

use concinnity_templates::WorldTemplate;
use serde_json::Value;

pub use concinnity_core::template_spec::{arg_value_to_json, spec_args, spec_to_value};

// A world template's assets as world-line entries, in application order.
pub fn world_template_entries(t: &WorldTemplate) -> Vec<Value> {
    t.assets().iter().map(spec_to_value).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_template_entries_yields_world_lines() {
        let template = concinnity_templates::by_name("showcase").expect("showcase template");
        let entries = world_template_entries(template);
        assert!(!entries.is_empty());
        for entry in &entries {
            assert!(entry.get("name").is_some());
            assert!(entry.get("type").is_some());
            assert!(entry.get("args").is_some());
        }
    }
}
