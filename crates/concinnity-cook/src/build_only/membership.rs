// Scene and screen membership on generated assets.
//
// A Prop belongs to a Scene and an overlay element to a Screen through an
// explicit `scene` / `screen` arg. An expander that emits content into a host
// stamps that arg onto every child it generates, so membership never depends on
// how an asset happens to be named.

use crate::authoring::registry::RegisteredType;
use crate::build_only::expand::registered_type;

// The overlay element types that belong to a Screen, plus the UI shorthands
// that carry a `screen` through to the elements they expand into. Those
// shorthands expand after the passes that emit them, so stamping the shorthand
// is enough to place its own children.
const SCREEN_SCOPED: [RegisteredType; 8] = [
    RegisteredType::Sprite,
    RegisteredType::TextLabel,
    RegisteredType::TextInput,
    RegisteredType::HitRegion,
    RegisteredType::ScrollPanel,
    RegisteredType::Panel,
    RegisteredType::Slider,
    RegisteredType::OptionSelect,
];

/// Set `screen` on every generated overlay element that does not already name
/// one. Anything else in the list -- the Screen itself, a Font, a KeyBinding --
/// is left alone, as is an empty `screen`.
pub(crate) fn scope_to_screen(children: &mut [serde_json::Value], screen: &str) {
    scope(children, "screen", screen, &SCREEN_SCOPED);
}

/// Set `scene` on every generated Prop that does not already name one. An empty
/// `scene` leaves the props unbound, which makes them visible in every scene.
pub(crate) fn scope_to_scene(children: &mut [serde_json::Value], scene: &str) {
    scope(children, "scene", scene, &[RegisteredType::Prop]);
}

fn scope(children: &mut [serde_json::Value], key: &str, host: &str, types: &[RegisteredType]) {
    if host.is_empty() {
        return;
    }
    for child in children.iter_mut() {
        if !registered_type(child).is_some_and(|ty| types.contains(&ty)) {
            continue;
        }
        let Some(entry) = child.as_object_mut() else {
            continue;
        };
        let args = entry
            .entry("args")
            .or_insert_with(|| serde_json::Value::Object(Default::default()));
        if let serde_json::Value::Object(args) = args
            && !args.contains_key(key)
        {
            args.insert(key.to_string(), serde_json::Value::String(host.to_string()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn arg(v: &serde_json::Value, key: &str) -> Option<String> {
        v.get("args")?.get(key)?.as_str().map(str::to_string)
    }

    #[test]
    fn stamps_the_screen_onto_overlay_elements_only() {
        let mut children = vec![
            json!({"type":"Screen","args":{"$id":"m"}}),
            json!({"type":"Sprite","args":{"$id":"m_dim"}}),
            json!({"type":"TextLabel","args":{"$id":"m_title"}}),
            json!({"type":"HitRegion","args":{"$id":"m_btn"}}),
            json!({"type":"Font","args":{"$id":"m_font"}}),
        ];
        scope_to_screen(&mut children, "m");
        assert_eq!(arg(&children[0], "screen"), None, "the Screen itself");
        for child in &children[1..4] {
            assert_eq!(arg(child, "screen").as_deref(), Some("m"));
        }
        assert_eq!(arg(&children[4], "screen"), None, "a Font has no screen");
    }

    // A shorthand forwards the screen to the elements it expands into, so the
    // stamp has to reach the shorthand itself, not only primitives.
    #[test]
    fn stamps_the_screen_onto_ui_shorthands() {
        let mut children = vec![
            serde_json::json!({"type":"Slider","args":{"$id":"row"}}),
            serde_json::json!({"type":"OptionSelect","args":{"$id":"pick"}}),
            serde_json::json!({"type":"Panel","args":{"$id":"card"}}),
        ];
        scope_to_screen(&mut children, "settings");
        for child in &children {
            assert_eq!(arg(child, "screen").as_deref(), Some("settings"));
        }
    }

    #[test]
    fn an_authored_host_is_never_overwritten() {
        let mut children = vec![json!({"type":"Sprite","args":{"$id":"s","screen":"other"}})];
        scope_to_screen(&mut children, "m");
        assert_eq!(arg(&children[0], "screen").as_deref(), Some("other"));
    }

    #[test]
    fn an_empty_host_stamps_nothing() {
        let mut children = vec![json!({"type":"Prop","args":{"$id":"p"}})];
        scope_to_scene(&mut children, "");
        assert_eq!(arg(&children[0], "scene"), None);
    }

    #[test]
    fn a_child_without_args_gains_them() {
        let mut children = vec![json!({"type":"Prop","args":{"$id":"p"}})];
        scope_to_scene(&mut children, "level");
        assert_eq!(arg(&children[0], "scene").as_deref(), Some("level"));
    }

    #[test]
    fn scene_scoping_skips_types_that_are_not_props() {
        let mut children = vec![
            json!({"type":"Prop","args":{"$id":"p"}}),
            json!({"type":"Material","args":{"$id":"m"}}),
            json!({"type":"InstancedProp","args":{"$id":"i"}}),
        ];
        scope_to_scene(&mut children, "level");
        assert_eq!(arg(&children[0], "scene").as_deref(), Some("level"));
        assert_eq!(arg(&children[1], "scene"), None);
        assert_eq!(arg(&children[2], "scene"), None);
    }
}
