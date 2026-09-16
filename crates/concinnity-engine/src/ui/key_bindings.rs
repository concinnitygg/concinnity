// Screen toggle keys and KeyBindings, matched against the frame's pressed key.

use concinnity_core::components::{KeyBinding, ScreenCommand};
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::ecs::{PipelineContext, StepResult};

use super::intent::UiIntent;
use super::{UiInputSystem, fire_action};

impl UiInputSystem {
    // A Screen's `toggle_key` opens / closes it from anywhere, ahead of ordinary
    // KeyBindings and immune to the typing suppression (so a console screen's
    // own key still closes it while its field has focus); a matched toggle
    // consumes the key. Otherwise the first in-scope binding for the key fires,
    // unless a field is typing or Enter was consumed as confirm. Runs before
    // the region pass so an Escape-toggled pause beats a same-frame click.
    pub(super) fn dispatch_key_bindings(
        &self,
        intent: &UiIntent,
        typing: bool,
        ctx: &mut PipelineContext,
    ) -> Option<StepResult> {
        let name = intent.pressed_key?;
        let toggles = self.screens.toggles_for_key(name);
        for id in &toggles {
            ctx.events_mut::<ScreenCommand>()
                .send(ScreenCommand::Toggle(*id));
        }
        let toggled_key = !toggles.is_empty();
        if toggled_key || typing || intent.enter_confirm {
            return None;
        }
        let binding = matching_binding(&self.bindings, name, self.screens.top())?;
        // KeyBindings carry no label (no settings row binds a key).
        fire_action(&binding.action, None, ctx)
    }
}

// The binding a pressed key fires: the first one for `name` with an action,
// skipping bindings scoped to a screen other than `top`.
fn matching_binding<'a>(
    bindings: &'a [KeyBinding],
    name: &str,
    top: Option<AssetId>,
) -> Option<&'a KeyBinding> {
    bindings.iter().find(|kb| {
        let scoped_out = kb.screen.is_some() && kb.screen != top;
        kb.key == name && !kb.action.is_empty() && !scoped_out
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(key: &str, action: &str, screen: Option<u32>) -> KeyBinding {
        KeyBinding {
            key: key.to_string(),
            action: action.to_string(),
            screen: screen.map(AssetId),
        }
    }

    #[test]
    fn first_match_wins() {
        let bindings = [binding("Space", "a", None), binding("Space", "b", None)];
        let hit = matching_binding(&bindings, "Space", None).map(|kb| kb.action.as_str());
        assert_eq!(hit, Some("a"));
    }

    #[test]
    fn other_keys_do_not_match() {
        let bindings = [binding("Enter", "a", None)];
        assert!(matching_binding(&bindings, "Space", None).is_none());
    }

    #[test]
    fn empty_action_is_skipped() {
        let bindings = [binding("Space", "", None), binding("Space", "b", None)];
        let hit = matching_binding(&bindings, "Space", None).map(|kb| kb.action.as_str());
        assert_eq!(hit, Some("b"));
    }

    #[test]
    fn scoped_binding_matches_only_under_its_top_screen() {
        let bindings = [
            binding("Escape", "scoped", Some(7)),
            binding("Escape", "global", None),
        ];
        let under = |top| matching_binding(&bindings, "Escape", top).map(|kb| kb.action.as_str());
        assert_eq!(under(Some(AssetId(7))), Some("scoped"));
        assert_eq!(under(Some(AssetId(8))), Some("global"));
        assert_eq!(under(None), Some("global"));
    }
}
