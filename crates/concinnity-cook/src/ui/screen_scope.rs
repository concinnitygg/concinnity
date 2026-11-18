// Screen membership on generated UI entries. An overlay element belongs to a
// Screen through an explicit `screen` arg, so a shorthand that expands into a
// screen stamps it onto every element it generates.

use crate::authoring::registry::RegisteredType;
use crate::authoring::spec::AssetSpec;
use crate::build_only::expand::{registered_type, set_arg_if_absent};

// The overlay element types that belong to a Screen, plus the UI shorthands
// that carry a `screen` through to the elements they expand into. Those
// shorthands expand after the passes that emit them, so stamping the shorthand
// is enough to place its own children.
const SCREEN_SCOPED: [RegisteredType; 19] = [
    RegisteredType::Sprite,
    RegisteredType::TextLabel,
    RegisteredType::TextField,
    RegisteredType::HitRegion,
    RegisteredType::ScrollPanel,
    RegisteredType::Layout,
    RegisteredType::List,
    RegisteredType::Grid,
    RegisteredType::ValueSlider,
    RegisteredType::Tabs,
    RegisteredType::Checkbox,
    RegisteredType::Button,
    RegisteredType::TextArea,
    RegisteredType::Menu,
    RegisteredType::Modal,
    RegisteredType::Toasts,
    RegisteredType::Panel,
    RegisteredType::Slider,
    RegisteredType::OptionSelect,
];

// The asset types a Screen can hold as children: every one that takes a
// `screen`, which adds the key bindings a generator leaves global.
pub(super) const SCREEN_CHILDREN: [RegisteredType; SCREEN_SCOPED.len() + 1] = {
    let mut all = [RegisteredType::KeyBinding; SCREEN_SCOPED.len() + 1];
    let mut i = 0;
    while i < SCREEN_SCOPED.len() {
        all[i] = SCREEN_SCOPED[i];
        i += 1;
    }
    all
};

// Whether an entry of `asset_type` joins the screen its generator expands into.
fn joins_screen(asset_type: Option<RegisteredType>) -> bool {
    asset_type.is_some_and(|ty| SCREEN_SCOPED.contains(&ty))
}

/// Set `screen` on every generated overlay element that does not already name
/// one. Anything else in the list -- the Screen itself, a Font, a `KeyBinding` --
/// is left alone, as is an empty `screen`.
pub(super) fn scope_to_screen(children: &mut [AssetSpec], screen: &str) {
    if screen.is_empty() {
        return;
    }
    for child in children.iter_mut() {
        if joins_screen(RegisteredType::parse(child.asset_type))
            && !child.fields.iter().any(|(key, _)| key == "screen")
        {
            child.fields.push(("screen".to_string(), screen.into()));
        }
    }
}

/// `scope_to_screen` for entries already in their world.jsonl form.
pub(crate) fn scope_entries_to_screen(children: &mut [serde_json::Value], screen: &str) {
    if screen.is_empty() {
        return;
    }
    for child in children.iter_mut() {
        if joins_screen(registered_type(child)) {
            set_arg_if_absent(child, "screen", screen);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authoring::spec::{ArgValue, AssetSpec};

    fn screen_of(spec: &AssetSpec) -> Option<&ArgValue> {
        spec.fields
            .iter()
            .find(|(key, _)| key == "screen")
            .map(|(_, v)| v)
    }

    fn spec(name: &str, ty: &'static str) -> AssetSpec {
        AssetSpec::new(name, ty)
    }

    #[test]
    fn stamps_the_screen_onto_overlay_elements_only() {
        let mut children = vec![
            spec("m", "Screen"),
            spec("m_dim", "Sprite"),
            spec("m_title", "TextLabel"),
            spec("m_btn", "HitRegion"),
            spec("m_font", "Font"),
        ];
        scope_to_screen(&mut children, "m");
        assert_eq!(screen_of(&children[0]), None, "the Screen itself");
        for child in &children[1..4] {
            assert_eq!(screen_of(child), Some(&ArgValue::from("m")));
        }
        assert_eq!(screen_of(&children[4]), None, "a Font has no screen");
    }

    // A shorthand forwards the screen to the elements it expands into, so the
    // stamp has to reach the shorthand itself, not only primitives.
    #[test]
    fn stamps_the_screen_onto_ui_shorthands() {
        let mut children = vec![
            spec("row", "Slider"),
            spec("pick", "OptionSelect"),
            spec("card", "Panel"),
        ];
        scope_to_screen(&mut children, "settings");
        for child in &children {
            assert_eq!(screen_of(child), Some(&ArgValue::from("settings")));
        }
    }

    #[test]
    fn an_authored_screen_is_never_overwritten() {
        let mut children = vec![spec("s", "Sprite").set("screen", "other")];
        scope_to_screen(&mut children, "m");
        assert_eq!(screen_of(&children[0]), Some(&ArgValue::from("other")));
        assert_eq!(children[0].fields.len(), 1);
    }

    #[test]
    fn an_empty_screen_stamps_nothing() {
        let mut children = vec![spec("s", "Sprite")];
        scope_to_screen(&mut children, "");
        assert_eq!(screen_of(&children[0]), None);
    }

    #[test]
    fn entries_take_the_same_screen_as_specs() {
        let mut entries = vec![
            serde_json::json!({"type":"Screen","args":{"$id":"m"}}),
            serde_json::json!({"type":"Sprite"}),
            serde_json::json!({"type":"TextLabel","args":{"screen":"other"}}),
        ];
        scope_entries_to_screen(&mut entries, "m");
        assert!(
            entries[0]["args"].get("screen").is_none(),
            "the Screen itself"
        );
        assert_eq!(entries[1]["args"]["screen"], "m");
        assert_eq!(entries[2]["args"]["screen"], "other");
    }
}
