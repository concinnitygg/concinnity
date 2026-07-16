// The screen stack: which Screens are active, in what order, and what each
// one's policies are. Pure bookkeeping -- UiInputSystem owns applying the
// resulting element-visibility and focus changes to the world.

use crate::assets::{Screen, ScreenCommand, ScreenInput};
use crate::ecs::asset_id::AssetId;
use std::collections::HashMap;

// Multiplier that turns a Screen's authored `layer` into a draw-layer band:
// stack position orders screens within a band, the authored layer orders the
// bands. Clamped so a wild authored value cannot collide with the editor's
// override range.
const LAYER_BAND: i32 = 4096;
const LAYER_MAX: i32 = 1000;

// The per-screen policies drained from a `Screen` asset at init.
#[derive(Debug, Clone)]
pub(crate) struct ScreenMeta {
    pub input: ScreenInput,
    pub pauses_world: bool,
    pub toggle_key: String,
    pub focus: Option<AssetId>,
    pub layer: i32,
}

impl ScreenMeta {
    pub(crate) fn from_asset(s: &Screen) -> Self {
        Self {
            input: s.input,
            pauses_world: s.pauses_world,
            toggle_key: s.toggle_key.clone(),
            focus: s.focus,
            layer: s.layer,
        }
    }
}

// The element-visibility and top-of-stack changes one command produced;
// UiInputSystem applies them to the world.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct ScreenTransition {
    // Screens whose elements hide (left the stack).
    pub hidden: Vec<AssetId>,
    // Screens whose elements show (entered the stack).
    pub shown: Vec<AssetId>,
    // The screen newly on top, when the top changed to one (a pop revealing a
    // lower screen announces it too, so e.g. its music cue re-fires).
    pub new_top: Option<AssetId>,
    // True whenever the top changed, including to "no screen".
    pub top_changed: bool,
}

// Active-screen bookkeeping: every declared Screen's policies plus the active
// stack (bottom to top). Dispatch warns when a `screen:*` action resolves to
// an unknown id.
#[derive(Debug, Default)]
pub(crate) struct ScreenRegistry {
    known: HashMap<AssetId, ScreenMeta>,
    stack: Vec<AssetId>,
}

impl ScreenRegistry {
    pub(crate) fn register(&mut self, id: AssetId, meta: ScreenMeta) {
        self.known.insert(id, meta);
    }

    pub(crate) fn is_known(&self, id: AssetId) -> bool {
        self.known.contains_key(&id)
    }

    pub(crate) fn meta(&self, id: AssetId) -> Option<&ScreenMeta> {
        self.known.get(&id)
    }

    pub(crate) fn top(&self) -> Option<AssetId> {
        self.stack.last().copied()
    }

    // The topmost screen that captures input, seen through any passthrough
    // screens above it. HitRegions fire only for this screen (or, when it is
    // `None`, for screen-less regions).
    pub(crate) fn top_capture(&self) -> Option<AssetId> {
        self.stack
            .iter()
            .rev()
            .find(|id| {
                self.known
                    .get(id)
                    .is_none_or(|m| m.input == ScreenInput::Capture)
            })
            .copied()
    }

    // True while any active screen pauses the world (physics / animation /
    // gameplay freeze, menu frame-rate cap).
    pub(crate) fn pauses_world(&self) -> bool {
        self.stack
            .iter()
            .any(|id| self.known.get(id).is_some_and(|m| m.pauses_world))
    }

    // True while any active screen captures input: gameplay keys are
    // suppressed even when the world keeps simulating beneath the screen.
    pub(crate) fn captures_input(&self) -> bool {
        self.top_capture().is_some()
    }

    // Screens whose non-empty `toggle_key` matches the pressed key name.
    pub(crate) fn toggles_for_key(&self, key_name: &str) -> Vec<AssetId> {
        let mut ids: Vec<AssetId> = self
            .known
            .iter()
            .filter(|(_, m)| !m.toggle_key.is_empty() && m.toggle_key == key_name)
            .map(|(id, _)| *id)
            .collect();
        ids.sort_by_key(|id| id.0);
        ids
    }

    // Draw layer per active screen: the authored layer picks a band, the stack
    // position orders screens within it. Screen-less HUD elements sit at 0, so
    // a default (layer 0) screen draws above the HUD and a negative layer
    // below it.
    pub(crate) fn layers(&self) -> HashMap<AssetId, i32> {
        self.stack
            .iter()
            .enumerate()
            .map(|(i, id)| {
                let band = self
                    .known
                    .get(id)
                    .map(|m| m.layer.clamp(-LAYER_MAX, LAYER_MAX))
                    .unwrap_or(0);
                (*id, band * LAYER_BAND + (i as i32) + 1)
            })
            .collect()
    }

    // Apply one command to the stack, returning the changes to enact, or
    // `None` when the command was a no-op (or its target unknown).
    pub(crate) fn apply(&mut self, cmd: ScreenCommand) -> Option<ScreenTransition> {
        let old_top = self.top();
        let old_stack = self.stack.clone();
        match cmd {
            ScreenCommand::Show(id) => {
                if !self.is_known(id) {
                    tracing::warn!("ScreenCommand::Show: unknown screen {}", id);
                    return None;
                }
                if old_top == Some(id) {
                    return None;
                }
                // Replace the top (menu navigation); an empty stack just opens.
                self.stack.pop();
                self.stack.retain(|s| *s != id);
                self.stack.push(id);
            }
            ScreenCommand::Push(id) => {
                if !self.is_known(id) {
                    tracing::warn!("ScreenCommand::Push: unknown screen {}", id);
                    return None;
                }
                if old_top == Some(id) {
                    return None;
                }
                // Already-open screens are raised rather than duplicated.
                self.stack.retain(|s| *s != id);
                self.stack.push(id);
            }
            ScreenCommand::Hide => {
                self.stack.pop()?;
            }
            ScreenCommand::Toggle(id) => {
                if !self.is_known(id) {
                    tracing::warn!("ScreenCommand::Toggle: unknown screen {}", id);
                    return None;
                }
                if old_top == Some(id) {
                    self.stack.pop();
                } else {
                    // Opening by toggle navigates like Show (replace the top),
                    // so an Escape-toggled menu returns from a settings
                    // sub-screen to the menu, and from the menu to the world.
                    // Stacking on top is the explicit `screen:push:` action.
                    self.stack.pop();
                    self.stack.retain(|s| *s != id);
                    self.stack.push(id);
                }
            }
            ScreenCommand::Clear => {
                if self.stack.is_empty() {
                    return None;
                }
                self.stack.clear();
            }
        }
        let new_top = self.top();
        Some(ScreenTransition {
            hidden: old_stack
                .iter()
                .filter(|s| !self.stack.contains(s))
                .copied()
                .collect(),
            shown: self
                .stack
                .iter()
                .filter(|s| !old_stack.contains(s))
                .copied()
                .collect(),
            new_top: (new_top != old_top).then_some(new_top).flatten(),
            top_changed: new_top != old_top,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> ScreenMeta {
        ScreenMeta {
            input: ScreenInput::Capture,
            pauses_world: true,
            toggle_key: String::new(),
            focus: None,
            layer: 0,
        }
    }

    fn registry(ids: &[u32]) -> ScreenRegistry {
        let mut r = ScreenRegistry::default();
        for &i in ids {
            r.register(AssetId(i), meta());
        }
        r
    }

    #[test]
    fn show_replaces_the_top() {
        let mut r = registry(&[1, 2]);
        r.apply(ScreenCommand::Show(AssetId(1))).unwrap();
        let t = r.apply(ScreenCommand::Show(AssetId(2))).unwrap();
        assert_eq!(r.top(), Some(AssetId(2)));
        assert_eq!(t.hidden, vec![AssetId(1)]);
        assert_eq!(t.shown, vec![AssetId(2)]);
        assert_eq!(t.new_top, Some(AssetId(2)));
    }

    #[test]
    fn push_stacks_and_hide_pops_to_reveal() {
        let mut r = registry(&[1, 2]);
        r.apply(ScreenCommand::Show(AssetId(1))).unwrap();
        let t = r.apply(ScreenCommand::Push(AssetId(2))).unwrap();
        assert!(t.hidden.is_empty(), "the lower screen stays visible");
        assert_eq!(r.top(), Some(AssetId(2)));

        let t = r.apply(ScreenCommand::Hide).unwrap();
        assert_eq!(t.hidden, vec![AssetId(2)]);
        assert!(t.shown.is_empty(), "revealed screen was already visible");
        assert_eq!(t.new_top, Some(AssetId(1)), "reveal is announced");
        assert_eq!(r.top(), Some(AssetId(1)));
    }

    #[test]
    fn toggle_pops_when_topmost_and_navigates_otherwise() {
        let mut r = registry(&[1, 2]);
        r.apply(ScreenCommand::Toggle(AssetId(1))).unwrap();
        assert_eq!(r.top(), Some(AssetId(1)));
        // Open elsewhere: toggle navigates (replaces the top), like Show.
        let t = r.apply(ScreenCommand::Toggle(AssetId(2))).unwrap();
        assert_eq!(r.top(), Some(AssetId(2)));
        assert_eq!(t.hidden, vec![AssetId(1)]);
        // Topmost: toggle pops it, back to the world.
        let t = r.apply(ScreenCommand::Toggle(AssetId(2))).unwrap();
        assert_eq!(t.hidden, vec![AssetId(2)]);
        assert_eq!(r.top(), None);
    }

    #[test]
    fn toggle_over_a_pushed_screen_replaces_it() {
        let mut r = registry(&[1, 2]);
        r.apply(ScreenCommand::Show(AssetId(1))).unwrap();
        r.apply(ScreenCommand::Push(AssetId(2))).unwrap();
        // Toggling the buried menu closes the pushed screen and raises the
        // menu, never duplicating it in the stack.
        let t = r.apply(ScreenCommand::Toggle(AssetId(1))).unwrap();
        assert_eq!(r.top(), Some(AssetId(1)));
        assert_eq!(t.hidden, vec![AssetId(2)]);
        assert_eq!(r.layers().len(), 1);
    }

    #[test]
    fn hide_on_empty_stack_and_repeat_show_are_no_ops() {
        let mut r = registry(&[1]);
        assert!(r.apply(ScreenCommand::Hide).is_none());
        r.apply(ScreenCommand::Show(AssetId(1))).unwrap();
        assert!(r.apply(ScreenCommand::Show(AssetId(1))).is_none());
        assert!(r.apply(ScreenCommand::Push(AssetId(1))).is_none());
    }

    #[test]
    fn unknown_ids_are_rejected() {
        let mut r = registry(&[1]);
        assert!(r.apply(ScreenCommand::Show(AssetId(9))).is_none());
        assert!(r.apply(ScreenCommand::Toggle(AssetId(9))).is_none());
        assert_eq!(r.top(), None);
    }

    #[test]
    fn clear_empties_the_stack() {
        let mut r = registry(&[1, 2]);
        r.apply(ScreenCommand::Show(AssetId(1))).unwrap();
        r.apply(ScreenCommand::Push(AssetId(2))).unwrap();
        let t = r.apply(ScreenCommand::Clear).unwrap();
        assert_eq!(r.top(), None);
        assert!(t.top_changed && t.new_top.is_none());
        let mut hidden = t.hidden.clone();
        hidden.sort_by_key(|id| id.0);
        assert_eq!(hidden, vec![AssetId(1), AssetId(2)]);
        assert!(r.apply(ScreenCommand::Clear).is_none());
    }

    #[test]
    fn top_capture_sees_through_passthrough_screens() {
        let mut r = registry(&[1]);
        let mut pass = meta();
        pass.input = ScreenInput::Passthrough;
        pass.pauses_world = false;
        r.register(AssetId(2), pass);

        r.apply(ScreenCommand::Show(AssetId(1))).unwrap();
        r.apply(ScreenCommand::Push(AssetId(2))).unwrap();
        assert_eq!(r.top(), Some(AssetId(2)));
        assert_eq!(r.top_capture(), Some(AssetId(1)));
        assert!(r.captures_input());

        // Alone, a passthrough screen captures nothing.
        r.apply(ScreenCommand::Clear).unwrap();
        r.apply(ScreenCommand::Push(AssetId(2))).unwrap();
        assert_eq!(r.top_capture(), None);
        assert!(!r.captures_input());
        assert!(!r.pauses_world());
    }

    #[test]
    fn pauses_world_follows_any_active_pausing_screen() {
        let mut r = registry(&[1]);
        let mut hud = meta();
        hud.pauses_world = false;
        hud.input = ScreenInput::Passthrough;
        r.register(AssetId(2), hud);

        r.apply(ScreenCommand::Push(AssetId(2))).unwrap();
        assert!(!r.pauses_world());
        r.apply(ScreenCommand::Push(AssetId(1))).unwrap();
        assert!(r.pauses_world());
        r.apply(ScreenCommand::Hide).unwrap();
        assert!(!r.pauses_world());
    }

    #[test]
    fn layers_band_by_authored_layer_and_order_by_stack() {
        let mut r = registry(&[1, 2]);
        let mut under_hud = meta();
        under_hud.layer = -1;
        r.register(AssetId(3), under_hud);

        r.apply(ScreenCommand::Push(AssetId(1))).unwrap();
        r.apply(ScreenCommand::Push(AssetId(2))).unwrap();
        r.apply(ScreenCommand::Push(AssetId(3))).unwrap();
        let layers = r.layers();
        assert_eq!(layers[&AssetId(1)], 1);
        assert_eq!(layers[&AssetId(2)], 2);
        // Negative authored layer sinks below the screen-less HUD's 0 even
        // though the screen was pushed last.
        assert_eq!(layers[&AssetId(3)], -LAYER_BAND + 3);
    }

    #[test]
    fn toggle_keys_match_by_name() {
        let mut r = ScreenRegistry::default();
        let mut m = meta();
        m.toggle_key = "Backtick".to_string();
        r.register(AssetId(4), m);
        assert_eq!(r.toggles_for_key("Backtick"), vec![AssetId(4)]);
        assert!(r.toggles_for_key("Escape").is_empty());
    }
}
