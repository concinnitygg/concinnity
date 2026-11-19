//! The toast stack: the engine `Toasts` widget the editor's notifications
//! (`notify.rs`) show on, above the viewport's bottom-right corner, and the
//! events its clicks send back.

use concinnity_engine::components::{LayoutAnchor, Toasts, UiEvent, UiValue};
use concinnity_engine::ecs::World;
use concinnity_engine::ecs::asset_id::AssetId;

use super::notify::MAX_VISIBLE;
use super::theme;
use super::ui::{UiBuilder, state};

const CARD_W: f32 = 320.0;
const PAD: f32 = 8.0;
const CARD_H: f32 = PAD * 2.0 + 2.0 * theme::LINE_H;
const MARGIN: f32 = 14.0;

// The events the stack's clicks send.
const DISMISS: &str = "editor.toast.dismiss";
const MORE: &str = "editor.toast.more";

// What a click on the stack asked for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ToastClick {
    // The card of this key was clicked away.
    Dismissed(String),
    // The line counting the waiting cards was clicked.
    More,
}

// What a toast event reports, if `event` is one.
pub(crate) fn click(event: &UiEvent) -> Option<ToastClick> {
    match (event.name.as_str(), &event.value) {
        (DISMISS, Some(UiValue::Text(key))) => Some(ToastClick::Dismissed(key.clone())),
        (MORE, _) => Some(ToastClick::More),
        _ => None,
    }
}

pub(crate) fn build(ui: &mut UiBuilder<'_>) -> Option<AssetId> {
    ui.toasts(Toasts {
        anchor: LayoutAnchor::BottomRight,
        margin: [MARGIN, MARGIN],
        width: CARD_W,
        card_height: CARD_H,
        max_visible: MAX_VISIBLE,
        on_dismiss: DISMISS.to_string(),
        on_more: MORE.to_string(),
        ..Default::default()
    })
}

// Show or hide the stack.
pub(crate) fn set_visible(world: &mut World, id: AssetId, visible: bool) {
    if let Some(t) = world.get_mut_by_id::<Toasts>(id) {
        state::set_if_changed(&mut t.visible, visible);
    }
}

// Whether `(x, y)` lands on the shown stack in a window of size `vp`.
pub(crate) fn covers(world: &World, id: AssetId, x: f32, y: f32, vp: [f32; 2]) -> bool {
    world
        .get_by_id::<Toasts>(id)
        .is_some_and(|t| t.covers(x, y, vp))
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_engine::components::{ShownToast, ToastTone};

    fn event(name: &str, value: Option<UiValue>) -> UiEvent {
        UiEvent {
            name: name.into(),
            element: None,
            value,
        }
    }

    #[test]
    fn clicks_name_the_card_or_the_waiting_line() {
        assert_eq!(
            click(&event(DISMISS, Some(UiValue::Text("note.x".into())))),
            Some(ToastClick::Dismissed("note.x".into()))
        );
        assert_eq!(click(&event(MORE, None)), Some(ToastClick::More));
        assert_eq!(click(&event(DISMISS, None)), None);
    }

    #[test]
    fn the_stack_covers_its_cards_only_while_shown() {
        let mut world = World::new();
        let id = build(&mut UiBuilder::new(&mut world, AssetId(9), None)).unwrap();
        let vp = [1280.0, 720.0];
        if let Some(t) = world.get_mut_by_id::<Toasts>(id) {
            t.shown.push(ShownToast {
                text: "saved".into(),
                tone: ToastTone::Info,
                left: 4.0,
                key: String::new(),
                progress: None,
                age: 0.0,
            });
        }
        let at = [vp[0] - MARGIN - 10.0, vp[1] - MARGIN - 10.0];
        assert!(covers(&world, id, at[0], at[1], vp));
        assert!(!covers(&world, id, 10.0, 10.0, vp));
        set_visible(&mut world, id, false);
        assert!(!covers(&world, id, at[0], at[1], vp));
    }
}
