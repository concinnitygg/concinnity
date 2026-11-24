//! The toast stack's drive (`hook/drive/notify.rs`): what a frame feeds the
//! stack, the actions a clicked-away card runs, the line that opens the
//! Console, the replay into a fresh world, and the new-fault latch that keeps
//! a behavior fault from toasting every frame.

use concinnity_engine::components::{Toast, Toasts, UiEvent, UiValue};
use concinnity_engine::ecs::{EventCursor, World};

use crate::editor::behavior::panel::Status;
use crate::editor::hook::EditorHook;
use crate::editor::hook::tests::fixtures::{entry, hook, selected, toasted, world_with_fields};
use crate::editor::install;
use crate::editor::notify;
use crate::editor::panels::registry::PanelKey;
use crate::editor::ui::BuiltUi;

// The toasts sent to the world so far.
fn sent(world: &World) -> Vec<Toast> {
    let mut cursor = EventCursor::default();
    world
        .events::<Toast>()
        .map(|e| e.read(&mut cursor).cloned().collect())
        .unwrap_or_default()
}

fn click(h: &mut EditorHook, world: &mut World, name: &str, value: Option<UiValue>) {
    world.events_mut::<UiEvent>().send(UiEvent {
        name: name.into(),
        element: None,
        value,
    });
    h.read_ui_events(world);
}

// A pushed message reaches the stack once, and the stack hides with the HUD.
#[test]
fn a_frame_feeds_what_was_pushed_and_the_stack_follows_the_hud() {
    let mut h = hook(Vec::new());
    let mut world = World::new();
    install::editor_ui(&mut world);
    h.notifier.success("saved");
    h.drive_toasts(&mut world, true);
    h.drive_toasts(&mut world, true);
    let toasts = sent(&world);
    assert_eq!(toasts.len(), 1, "sent once");
    assert_eq!(toasts[0].text, "saved");
    let id = world.resource::<BuiltUi>().unwrap().toasts.unwrap();
    h.drive_toasts(&mut world, false);
    assert!(!world.get_by_id::<Toasts>(id).unwrap().visible);
}

// Clicking a card away runs its action: an error's opens the Console.
#[test]
fn a_dismissed_error_opens_the_console() {
    let mut h = hook(Vec::new());
    let mut world = World::new();
    h.notifier
        .error_with("cook failed", notify::Action::OpenConsole);
    h.drive_toasts(&mut world, true);
    let key = sent(&world)[0].key.clone();
    click(
        &mut h,
        &mut world,
        "editor.toast.dismiss",
        Some(UiValue::Text(key)),
    );
    assert!(h.open[PanelKey::Console]);
    assert!(toasted(&mut h).is_empty(), "the card is gone");
}

// A failed build's toast names an asset; clicking it selects that asset.
#[test]
fn a_build_failure_toast_selects_the_asset_it_names() {
    let mut h = hook(vec![entry("rock", "Prop"), entry("tree", "Prop")]);
    let mut world = world_with_fields();
    h.notifier.error_with(
        "Cook failed: Prop 'tree': unknown mesh 'oak'",
        notify::Action::SelectAsset("tree".into()),
    );
    h.drive_toasts(&mut world, true);
    let key = sent(&world)[0].key.clone();
    click(
        &mut h,
        &mut world,
        "editor.toast.dismiss",
        Some(UiValue::Text(key)),
    );
    assert_eq!(selected(&h), ["tree"]);
}

// A name the world no longer resolves has nothing to select, so the click
// opens the Console, where the whole failure is listed.
#[test]
fn a_build_failure_toast_for_a_vanished_asset_opens_the_console() {
    let mut h = hook(vec![entry("rock", "Prop")]);
    let mut world = world_with_fields();
    h.notifier.error_with(
        "Cook failed: Prop 'ghost': unknown mesh",
        notify::Action::SelectAsset("ghost".into()),
    );
    h.drive_toasts(&mut world, true);
    let key = sent(&world)[0].key.clone();
    click(
        &mut h,
        &mut world,
        "editor.toast.dismiss",
        Some(UiValue::Text(key)),
    );
    assert!(selected(&h).is_empty());
    assert!(h.open[PanelKey::Console]);
}

// The line counting the waiting cards opens the Console.
#[test]
fn the_waiting_line_opens_the_console() {
    let mut h = hook(Vec::new());
    let mut world = World::new();
    click(&mut h, &mut world, "editor.toast.more", None);
    assert!(h.open[PanelKey::Console]);
}

// A world the panels were just built into shows every live toast again.
#[test]
fn a_fresh_world_is_replayed_the_live_toasts() {
    let mut h = hook(Vec::new());
    let mut world = World::new();
    h.notifier.success("saved");
    h.drive_toasts(&mut world, true);
    let mut fresh = World::new();
    h.replay_toasts(&mut fresh);
    assert_eq!(sent(&fresh)[0].text, "saved");
}

// A behavior fault pushed on commit dedupes against the persisting fault, so
// per-keystroke re-checks of the same complaint do not re-toast.
#[test]
fn behavior_fault_toasts_only_on_a_new_fault() {
    let mut h = hook(Vec::new());
    h.behavior.status = Some(Status::message("a behavior needs a name"));
    let mut world = World::new();
    let mut fed = |h: &mut EditorHook| {
        let before = sent(&world).len();
        h.drive_toasts(&mut world, true);
        sent(&world).len() - before
    };
    h.notify_behavior_fault(None);
    assert_eq!(fed(&mut h), 1, "a fresh fault toasts");
    let prev = h.behavior_fault_message();
    h.notify_behavior_fault(prev.as_deref());
    assert_eq!(fed(&mut h), 0, "the same fault stays quiet");
    h.notify_behavior_fault(Some("a different complaint"));
    assert_eq!(fed(&mut h), 1, "a changed fault toasts again");
}
