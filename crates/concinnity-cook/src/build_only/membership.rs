// Scene membership on generated assets.
//
// A Prop belongs to a Scene through an explicit `scene` arg. An expander that
// emits props into a scene stamps that arg onto every prop it generates, so
// membership never depends on how an asset happens to be named.

use crate::authoring::registry::RegisteredType;
use crate::build_only::expand::{registered_type, set_arg_if_absent};

/// Set `scene` on every generated Prop that does not already name one. An empty
/// `scene` leaves the props unbound, which makes them visible in every scene.
pub(crate) fn scope_to_scene(children: &mut [serde_json::Value], scene: &str) {
    if scene.is_empty() {
        return;
    }
    for child in children.iter_mut() {
        if registered_type(child) == Some(RegisteredType::Prop) {
            set_arg_if_absent(child, "scene", scene);
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
    fn an_empty_scene_stamps_nothing() {
        let mut children = vec![json!({"type":"Prop","args":{"$id":"p"}})];
        scope_to_scene(&mut children, "");
        assert_eq!(arg(&children[0], "scene"), None);
    }

    #[test]
    fn a_child_without_args_gains_them() {
        let mut children = vec![json!({"type":"Prop"})];
        scope_to_scene(&mut children, "level");
        assert_eq!(arg(&children[0], "scene").as_deref(), Some("level"));
    }

    #[test]
    fn an_authored_scene_is_never_overwritten() {
        let mut children = vec![json!({"type":"Prop","args":{"$id":"p","scene":"other"}})];
        scope_to_scene(&mut children, "level");
        assert_eq!(arg(&children[0], "scene").as_deref(), Some("other"));
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
