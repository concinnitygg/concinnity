// Screen toggle keys and KeyBindings, matched against the frame's pressed key.

use concinnity_core::components::{KeyBinding, ScreenCommand, UiAction};
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
        let action = matching_binding(&self.bindings, name, self.screens.top())?;
        // KeyBindings carry no label (no settings row binds a key).
        fire_action(action, None, ctx)
    }
}

// The action a pressed key fires: the first binding for `name` with an action,
// skipping bindings scoped to a screen other than `top`.
fn matching_binding<'a>(
    bindings: &'a [KeyBinding],
    name: &str,
    top: Option<AssetId>,
) -> Option<&'a UiAction> {
    bindings.iter().find_map(|kb| {
        let scoped_out = kb.screen.is_some() && kb.screen != top;
        if kb.key == name && !scoped_out {
            kb.action.as_ref()
        } else {
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // A distinguishable stand-in action per binding.
    fn action(n: usize) -> UiAction {
        UiAction::GroupToggle(n)
    }

    fn binding(key: &str, action: Option<UiAction>, screen: Option<u32>) -> KeyBinding {
        KeyBinding {
            key: key.to_string(),
            action,
            screen: screen.map(AssetId),
        }
    }

    #[test]
    fn first_match_wins() {
        let bindings = [
            binding("Space", Some(action(1)), None),
            binding("Space", Some(action(2)), None),
        ];
        assert_eq!(matching_binding(&bindings, "Space", None), Some(&action(1)));
    }

    #[test]
    fn other_keys_do_not_match() {
        let bindings = [binding("Enter", Some(action(1)), None)];
        assert!(matching_binding(&bindings, "Space", None).is_none());
    }

    #[test]
    fn empty_action_is_skipped() {
        let bindings = [
            binding("Space", None, None),
            binding("Space", Some(action(2)), None),
        ];
        assert_eq!(matching_binding(&bindings, "Space", None), Some(&action(2)));
    }

    #[test]
    fn scoped_binding_matches_only_under_its_top_screen() {
        let bindings = [
            binding("Escape", Some(action(7)), Some(7)),
            binding("Escape", Some(action(0)), None),
        ];
        let under = |top| matching_binding(&bindings, "Escape", top);
        assert_eq!(under(Some(AssetId(7))), Some(&action(7)));
        assert_eq!(under(Some(AssetId(8))), Some(&action(0)));
        assert_eq!(under(None), Some(&action(0)));
    }
}
