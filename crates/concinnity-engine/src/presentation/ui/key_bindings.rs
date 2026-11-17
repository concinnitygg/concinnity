// Screen toggle keys and KeyBindings, matched against the frame's pressed key.

use crate::components::{KeyBinding, KeyMods, ScreenCommand};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, Ref, StepResult};

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
        ctx: &mut PipelineContext<'_>,
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
        let (id, binding) =
            matching_binding(&self.bindings, name, intent.pressed_mods, |screen| {
                self.screens.binds_keys(screen)
            })?;
        let action = binding.action.as_ref()?;
        // KeyBindings carry no label (no settings row binds a key).
        fire_action(action, *id, None, ctx)
    }
}

// The action a press of `name` with `mods` held fires, skipping bindings on a
// screen `binds` says is not live: the first binding naming exactly those
// modifiers, else the first that names none.
fn matching_binding<'a>(
    bindings: &'a [(Option<AssetId>, KeyBinding)],
    name: &str,
    mods: KeyMods,
    binds: impl Fn(Option<AssetId>) -> bool + Copy,
) -> Option<&'a (Option<AssetId>, KeyBinding)> {
    let live = || {
        bindings.iter().filter(move |(_, kb)| {
            kb.key == name && binds(kb.screen.map(Ref::id)) && kb.action.is_some()
        })
    };
    live()
        .find(|(_, kb)| kb.mods == Some(mods))
        .or_else(|| live().find(|(_, kb)| kb.mods.is_none()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::UiAction;
    use crate::ecs::Ref;
    use alloc::string::ToString;
    use alloc::vec::Vec;

    // A distinguishable stand-in action per binding.
    fn action(n: usize) -> UiAction {
        UiAction::Screenshot(u32::try_from(n).unwrap_or(0))
    }

    fn binding(key: &str, action: Option<UiAction>, screen: Option<u32>) -> KeyBinding {
        KeyBinding {
            key: key.to_string(),
            mods: None,
            action,
            screen: screen.map(AssetId).map(Ref::new),
        }
    }

    fn chord(key: &str, mods: KeyMods, action: UiAction) -> KeyBinding {
        KeyBinding {
            mods: Some(mods),
            ..binding(key, Some(action), None)
        }
    }

    // Live while `top` is on top, as a stack-only world binds them.
    fn under(top: Option<AssetId>) -> impl Fn(Option<AssetId>) -> bool + Copy {
        move |screen: Option<AssetId>| screen.is_none() || screen == top
    }

    // The action the first live binding fires, ids aside.
    fn matching(
        bindings: &[KeyBinding],
        name: &str,
        mods: KeyMods,
        binds: impl Fn(Option<AssetId>) -> bool + Copy,
    ) -> Option<UiAction> {
        let held: Vec<_> = bindings.iter().cloned().map(|b| (None, b)).collect();
        matching_binding(&held, name, mods, binds).and_then(|(_, kb)| kb.action.clone())
    }

    fn plain(bindings: &[KeyBinding], name: &str) -> Option<UiAction> {
        matching(bindings, name, KeyMods::NONE, under(None))
    }

    #[test]
    fn first_match_wins() {
        let bindings = [
            binding("Space", Some(action(1)), None),
            binding("Space", Some(action(2)), None),
        ];
        assert_eq!(plain(&bindings, "Space"), Some(action(1)));
    }

    #[test]
    fn other_keys_do_not_match() {
        let bindings = [binding("Enter", Some(action(1)), None)];
        assert!(plain(&bindings, "Space").is_none());
    }

    #[test]
    fn empty_action_is_skipped() {
        let bindings = [
            binding("Space", None, None),
            binding("Space", Some(action(2)), None),
        ];
        assert_eq!(plain(&bindings, "Space"), Some(action(2)));
    }

    #[test]
    fn scoped_binding_matches_only_under_its_top_screen() {
        let bindings = [
            binding("Escape", Some(action(7)), Some(7)),
            binding("Escape", Some(action(0)), None),
        ];
        let with_top = |top| matching(&bindings, "Escape", KeyMods::NONE, under(top));
        assert_eq!(with_top(Some(AssetId(7))), Some(action(7)));
        assert_eq!(with_top(Some(AssetId(8))), Some(action(0)));
        assert_eq!(with_top(None), Some(action(0)));
    }

    #[test]
    fn a_chord_wins_over_the_bare_key_only_with_its_modifiers_held() {
        let shot = UiAction::Screenshot(1);
        let burst = UiAction::Screenshot(8);
        let bindings = [
            binding("P", Some(shot.clone()), None),
            chord("P", KeyMods::SHIFT, burst.clone()),
        ];
        let with = |mods| matching(&bindings, "P", mods, under(None));
        assert_eq!(with(KeyMods::NONE), Some(shot.clone()));
        assert_eq!(with(KeyMods::SHIFT), Some(burst));
        assert_eq!(
            with(KeyMods::CTRL),
            Some(shot.clone()),
            "a bare binding takes any modifiers"
        );
        assert_eq!(with(KeyMods::CTRL.with_shift()), Some(shot));
    }

    #[test]
    fn a_chord_never_fires_without_its_modifiers() {
        let bindings = [chord("P", KeyMods::SHIFT, UiAction::Screenshot(8))];
        assert!(plain(&bindings, "P").is_none());
        let ctrl = matching(&bindings, "P", KeyMods::CTRL, under(None));
        assert!(ctrl.is_none());
    }
}
