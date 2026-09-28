// Carries a generated action into a spec's args. Generated HitRegions and
// KeyBindings name their targets by `$id`, resolved when the args deserialize
// into a `UiAction`.

use concinnity_core::components::NamedAction;
use serde_json::Value;

use crate::authoring::spec::ArgValue;

impl From<NamedAction> for ArgValue {
    fn from(action: NamedAction) -> Self {
        json_arg(serde_json::to_value(action).expect("an action serializes to JSON"))
    }
}

fn json_arg(value: Value) -> ArgValue {
    match value {
        Value::Null => ArgValue::Null,
        Value::Bool(b) => ArgValue::Bool(b),
        Value::Number(n) => match n.as_i64() {
            Some(i) => ArgValue::Int(i),
            None => ArgValue::Float(n.as_f64().unwrap_or_default()),
        },
        Value::String(s) => ArgValue::Str(s),
        Value::Array(items) => ArgValue::Array(items.into_iter().map(json_arg).collect()),
        Value::Object(map) => {
            ArgValue::Object(map.into_iter().map(|(k, v)| (k, json_arg(v))).collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use concinnity_core::components::{AuthoredAction, ScreenCommand, StoryCommand, UiAction};
    use concinnity_core::ecs::asset_id::AssetId;

    use super::*;
    use crate::authoring::spec::json::arg_value_to_json;

    // A generated action reaches the args in its authored form, which reads
    // back as the runtime action it names.
    #[test]
    fn a_generated_action_reads_back_as_its_runtime_action() {
        let resolved = |action: NamedAction| {
            let json = arg_value_to_json(&ArgValue::from(action));
            let text = json.to_string().replace("\"pause\"", "9");
            serde_json::from_str::<UiAction>(&text).unwrap_or_else(|e| panic!("{text}: {e}"))
        };
        assert_eq!(resolved(AuthoredAction::Quit), UiAction::Quit);
        assert_eq!(
            resolved(AuthoredAction::Show("pause".into())),
            UiAction::Screen(ScreenCommand::Show(AssetId(9)))
        );
        assert_eq!(
            resolved(AuthoredAction::Story(StoryCommand::Choose(2))),
            UiAction::Story(StoryCommand::Choose(2))
        );
        assert_eq!(
            resolved(AuthoredAction::GroupToggle(3)),
            UiAction::GroupToggle(3)
        );
    }
}
