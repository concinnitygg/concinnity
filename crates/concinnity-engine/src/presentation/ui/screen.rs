// The screen stack: which Screens are active, in what order, and what each
// one's policies are. Pure bookkeeping: UiInputSystem owns applying the
// resulting element-visibility and focus changes to the world.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use concinnity_core::math::SaturatingFrom;

use crate::components::ScreenCommand;
use crate::components::{TextField, Vocabulary};
use crate::ecs::Ref;
use crate::ecs::asset_id::AssetId;
use crate::gfx::overlay::OverlayTransform;
use hashbrown::HashMap;

// Multiplier that turns a Screen's authored `layer` into a draw-layer band:
// stack position orders screens within a band, the authored layer orders the
// bands. Clamped so a wild authored value cannot collide with the editor's
// override range.
const LAYER_BAND: i32 = 4096 * POPUP_LAYERS;
const LAYER_MAX: i32 = 1000;

// Draw layers each screen spans: its content, then the popups it opens over
// that content (menus, a modal, toasts), in `Popup` order.
const POPUP_LAYERS: i32 = 4;

/// A layer a screen draws over its own content, so a menu, modal, or toast
/// covers the screen's elements without a screen of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Popup {
    Menu = 1,
    Modal = 2,
    Toast = 3,
}

impl Popup {
    // The draw layer of this popup over a screen drawn at `screen_layer`.
    pub(crate) fn over(self, screen_layer: i32) -> i32 {
        screen_layer.saturating_add(self as i32)
    }
}

// The per-screen policies drained from a `Screen` asset at init.
#[derive(Debug, Clone)]
pub(crate) struct ScreenMeta {
    pub input: ScreenInput,
    pub(crate) space: ScreenSpace,
    pub(crate) persistent: bool,
    pub(crate) pauses_world: bool,
    pub(crate) toggle_key: String,
    pub focus: Option<AssetId>,
    pub layer: i32,
}

impl ScreenMeta {
    pub(crate) fn from_asset(s: &Screen) -> Self {
        Self {
            input: s.input,
            space: s.space,
            persistent: s.persistent,
            pauses_world: s.pauses_world,
            toggle_key: s.toggle_key.clone(),
            focus: s.focus.map(Ref::id),
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
    pub(crate) new_top: Option<AssetId>,
    // True whenever the top changed, including to "no screen".
    pub(crate) top_changed: bool,
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

    // Whether `id` is a persistent screen: open from load, outside the stack.
    pub(crate) fn is_persistent(&self, id: AssetId) -> bool {
        self.known.get(&id).is_some_and(|m| m.persistent)
    }

    // Whether an element of `screen` takes input now: the topmost capturing
    // screen's own, a persistent screen layered above it, or, while no
    // capturing screen is open, any persistent screen's (or a screen-less
    // element's).
    pub(crate) fn receives_input(&self, screen: Option<AssetId>) -> bool {
        match self.top_capture() {
            Some(top) => screen == Some(top) || screen.is_some_and(|id| self.above(id, top)),
            None => screen.is_none_or(|id| self.is_persistent(id)),
        }
    }

    // Whether persistent screen `id` is layered above screen `top`.
    fn above(&self, id: AssetId, top: AssetId) -> bool {
        let layer = |id| self.known.get(&id).map(|m| m.layer);
        self.is_persistent(id) && layer(id) > layer(top)
    }

    // Whether a key binding on `screen` is live: one on no screen always is,
    // one on a persistent screen while no capturing screen is open or while it
    // is layered above the one that is, and any other while its screen is on
    // top.
    pub(crate) fn binds_keys(&self, screen: Option<AssetId>) -> bool {
        match screen {
            None => true,
            Some(id) if self.is_persistent(id) => {
                self.top_capture().is_none_or(|top| self.above(id, top))
            }
            Some(id) => self.top() == Some(id),
        }
    }

    // Every screen drawn now: the persistent ones, then the stack's.
    pub(crate) fn open(&self) -> Vec<AssetId> {
        let mut open: Vec<AssetId> = self
            .known
            .iter()
            .filter(|(_, m)| m.persistent)
            .map(|(id, _)| *id)
            .collect();
        open.sort_unstable();
        open.extend(self.stack.iter().copied());
        open
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
                    .get(*id)
                    .is_none_or(|m| m.input == ScreenInput::Capture)
            })
            .copied()
    }

    // Whether elements of `screen` are laid out on the reference canvas: a
    // screen-less element and a window-space screen's elements use window
    // pixels.
    pub(crate) fn on_canvas(&self, screen: Option<AssetId>) -> bool {
        screen.is_some_and(|id| {
            self.known
                .get(&id)
                .is_none_or(|m| m.space == ScreenSpace::Canvas)
        })
    }

    // The cursor at window `(x, y)` in `screen`'s layout space: reference
    // space through `overlay` on the canvas, window pixels otherwise.
    pub(crate) fn cursor(
        &self,
        screen: Option<AssetId>,
        overlay: &OverlayTransform,
        x: f32,
        y: f32,
    ) -> (f32, f32) {
        if self.on_canvas(screen) {
            overlay.inverse(x, y)
        } else {
            (x, y)
        }
    }

    // Every declared window-space screen, in id order.
    pub(crate) fn window_screens(&self) -> Vec<AssetId> {
        let mut ids: Vec<AssetId> = self
            .known
            .iter()
            .filter(|(_, m)| m.space == ScreenSpace::Window)
            .map(|(id, _)| *id)
            .collect();
        ids.sort_by_key(|id| id.0);
        ids
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
            .filter(|(_, m)| !m.persistent && !m.toggle_key.is_empty() && m.toggle_key == key_name)
            .map(|(id, _)| *id)
            .collect();
        ids.sort_by_key(|id| id.0);
        ids
    }

    // Draw layer per open screen: the authored layer picks a band, the stack
    // position orders stack screens within it, a screen's popups between it and
    // the next. A persistent screen sits at the
    // bottom of its band, so a default (layer 0) stack screen draws above the
    // default HUD and a negative layer below it.
    pub(crate) fn layers(&self) -> BTreeMap<AssetId, i32> {
        let band = |id: &AssetId| {
            self.known
                .get(id)
                .map_or(0, |m| m.layer.clamp(-LAYER_MAX, LAYER_MAX))
                * LAYER_BAND
        };
        // A persistent screen in the HUD band takes layer 0, which an absent
        // entry already means, so a world of HUD screens alone stays unlayered.
        let persistent = self
            .known
            .iter()
            .filter(|(_, m)| m.persistent)
            .map(|(id, _)| (*id, band(id)))
            .filter(|(_, layer)| *layer != 0);
        let stacked = self.stack.iter().enumerate().map(|(i, id)| {
            let position = i32::saturating_from(i).saturating_add(1);
            (*id, band(id) + position.saturating_mul(POPUP_LAYERS))
        });
        persistent.chain(stacked).collect()
    }

    // Apply one command to the stack, returning the changes to enact, or
    // `None` when the command was a no-op (or its target unknown).
    pub(crate) fn apply(&mut self, cmd: ScreenCommand) -> Option<ScreenTransition> {
        if let ScreenCommand::Show(id) | ScreenCommand::Push(id) | ScreenCommand::Toggle(id) = cmd
            && self.is_persistent(id)
        {
            tracing::warn!("{cmd:?}: screen {id} is persistent and always open");
            return None;
        }
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
                    // Stacking on top is the explicit `push` action.
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

/// How a [Screen](#screen) treats input while it is active.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Vocabulary,
)]
#[serde(rename_all = "lowercase")]
pub enum ScreenInput {
    /// The screen owns input while it is the topmost capturing screen:
    /// gameplay input is suppressed and lower screens' [`HitRegion`](#hitregion)s
    /// stop firing.
    #[default]
    #[vocab("capture")]
    Capture,
    /// The screen only draws; input passes through to whatever is beneath it.
    #[vocab("passthrough")]
    Passthrough,
}

/// Where a [Screen](#screen)'s elements are laid out.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Vocabulary,
)]
#[serde(rename_all = "lowercase")]
pub enum ScreenSpace {
    /// The 1280 x 720 reference canvas, scaled uniformly to the window through
    /// each element's `fit`, so the screen keeps its proportions at any window
    /// size.
    #[default]
    #[vocab("canvas")]
    Canvas,
    /// Window pixels: elements keep their authored size and position at any
    /// window size.
    #[vocab("window")]
    Window,
}

/// One layer of the screen stack, as a running world holds it: the policies of
/// a [Screen](#screen) whose elements name it in their `screen` field.
///
/// ```rust
/// # use concinnity_engine::components::Screen;
/// Screen {
///     toggle_key: "Escape".into(),
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct Screen {
    /// When true, this screen is shown as soon as the world loads.
    pub initial: bool,
    /// Seconds to fade the screen in when it's shown. 0 shows it instantly.
    pub fade_in_secs: f32,
    /// `InputKey` that toggles this screen open / closed from anywhere, by the same
    /// canonical key names a [`KeyBinding`](#keybinding) uses (e.g. "Escape",
    /// "Backtick"). Empty leaves the screen action-driven only.
    pub toggle_key: String,
    /// Input policy while the screen is active.
    pub input: ScreenInput,
    /// Where the screen's elements are laid out: the reference canvas, or
    /// window pixels.
    pub space: ScreenSpace,
    /// Open from load and outside the stack: no navigation shows or hides it,
    /// its elements keep the visibility they are given, and they take input
    /// while no capturing screen is open, or while it is layered above the
    /// one that is.
    pub persistent: bool,
    /// When true (the default), the world pauses beneath this screen while it
    /// is active: gameplay input, physics, and animation freeze.
    #[asset(default = true)]
    pub pauses_world: bool,
    /// [`TextField`](#textfield) that receives keyboard focus whenever this
    /// screen reaches the top of the stack.
    pub focus: Option<Ref<TextField>>,
    /// Draw-order bias against the always-on HUD and other screens. Screens
    /// default above the HUD in stack order; a negative layer draws beneath
    /// the HUD, a higher layer stays above later-pushed screens.
    pub layer: i32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    fn meta() -> ScreenMeta {
        ScreenMeta {
            input: ScreenInput::Capture,
            space: ScreenSpace::Canvas,
            persistent: false,
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
        let mut hidden = t.hidden;
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
        assert_eq!(layers[&AssetId(1)], POPUP_LAYERS);
        assert_eq!(layers[&AssetId(2)], 2 * POPUP_LAYERS);
        // Negative authored layer sinks below the screen-less HUD's 0 even
        // though the screen was pushed last.
        assert_eq!(layers[&AssetId(3)], -LAYER_BAND + 3 * POPUP_LAYERS);
        // A screen's popups stay below the screen pushed over it.
        assert!(Popup::Toast.over(layers[&AssetId(1)]) < layers[&AssetId(2)]);
        assert!(Popup::Toast.over(0) < layers[&AssetId(1)], "the HUD's too");
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

    #[test]
    fn input_names_parse_in_lowercase() {
        let i = |s: &str| serde_json::from_str::<ScreenInput>(s).unwrap();
        assert_eq!(i(r#""capture""#), ScreenInput::Capture);
        assert_eq!(i(r#""passthrough""#), ScreenInput::Passthrough);
        assert_eq!(
            serde_json::to_string(&ScreenInput::Passthrough).unwrap(),
            r#""passthrough""#
        );
    }

    #[test]
    fn window_space_screens_lay_out_in_window_pixels() {
        let mut r = registry(&[1]);
        let mut window = meta();
        window.space = ScreenSpace::Window;
        r.register(AssetId(2), window);

        assert!(r.on_canvas(Some(AssetId(1))));
        assert!(!r.on_canvas(Some(AssetId(2))));
        assert!(!r.on_canvas(None), "a screen-less element");
        assert!(
            r.on_canvas(Some(AssetId(9))),
            "an unknown screen keeps the canvas"
        );
        assert_eq!(r.window_screens(), vec![AssetId(2)]);
    }

    #[test]
    fn space_names_parse_in_lowercase() {
        let s = |s: &str| serde_json::from_str::<ScreenSpace>(s).unwrap();
        assert_eq!(s(r#""canvas""#), ScreenSpace::Canvas);
        assert_eq!(s(r#""window""#), ScreenSpace::Window);
    }

    fn persistent() -> ScreenMeta {
        let mut hud = meta();
        hud.persistent = true;
        hud.input = ScreenInput::Passthrough;
        hud.pauses_world = false;
        hud
    }

    #[test]
    fn navigation_never_reaches_a_persistent_screen() {
        let mut r = registry(&[1]);
        let mut hud = persistent();
        hud.toggle_key = "F2".to_string();
        r.register(AssetId(2), hud);
        assert!(r.apply(ScreenCommand::Show(AssetId(2))).is_none());
        assert!(r.apply(ScreenCommand::Push(AssetId(2))).is_none());
        assert!(r.apply(ScreenCommand::Toggle(AssetId(2))).is_none());
        assert!(r.toggles_for_key("F2").is_empty());
        r.apply(ScreenCommand::Show(AssetId(1))).unwrap();
        r.apply(ScreenCommand::Clear).unwrap();
        assert_eq!(r.top(), None);
        assert!(!r.pauses_world() && !r.captures_input());
    }

    #[test]
    fn a_persistent_screen_draws_at_the_bottom_of_its_band() {
        let mut r = registry(&[1]);
        r.register(AssetId(2), persistent());
        let mut under = persistent();
        under.layer = -1;
        r.register(AssetId(3), under);
        r.apply(ScreenCommand::Show(AssetId(1))).unwrap();
        let layers = r.layers();
        assert!(!layers.contains_key(&AssetId(2)), "the HUD band is layer 0");
        assert_eq!(layers[&AssetId(3)], -LAYER_BAND);
        assert_eq!(layers[&AssetId(1)], POPUP_LAYERS);
    }

    #[test]
    fn persistent_screens_take_input_while_no_capturing_screen_is_open() {
        let mut r = registry(&[1]);
        r.register(AssetId(2), persistent());
        assert!(r.receives_input(Some(AssetId(2))));
        assert!(r.receives_input(None));
        assert!(!r.receives_input(Some(AssetId(1))), "a closed stack screen");
        r.apply(ScreenCommand::Show(AssetId(1))).unwrap();
        assert!(r.receives_input(Some(AssetId(1))));
        assert!(!r.receives_input(Some(AssetId(2))));
        assert!(!r.receives_input(None));
    }

    #[test]
    fn a_persistent_screen_layered_above_a_capture_still_takes_input() {
        let mut r = registry(&[1]);
        let mut over = persistent();
        over.layer = 5;
        r.register(AssetId(2), over);
        r.apply(ScreenCommand::Show(AssetId(1))).unwrap();
        assert!(r.receives_input(Some(AssetId(2))));
        assert!(r.receives_input(Some(AssetId(1))));
        assert!(
            r.binds_keys(Some(AssetId(2))),
            "its key bindings stay live too"
        );
    }
}
