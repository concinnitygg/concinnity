// World-template bridging on top of the world spec bridge.
//
// The generic `AssetSpec` -> `serde_json::Value` conversion lives in
// `concinnity_cook::authoring::spec`; this module adds the world-template
// convenience the editor uses on top of it.

pub(crate) use concinnity_cook::authoring::spec::spec_args;
use concinnity_cook::authoring::spec::spec_to_value;
use concinnity_cook::authoring::template::WorldTemplate;
use serde_json::Value;

/// A world template's assets as world-line entries, in application order.
pub(crate) fn world_template_entries(t: &WorldTemplate) -> Vec<Value> {
    t.assets().iter().map(spec_to_value).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_template_entries_yields_world_lines() {
        let template = concinnity_cook::authoring::template::TEMPLATES
            .first()
            .expect("at least one template");
        let entries = world_template_entries(template);
        assert!(!entries.is_empty());
        for entry in &entries {
            assert!(entry.get("name").is_some());
            assert!(entry.get("type").is_some());
            assert!(entry.get("args").is_some());
        }
    }
}
