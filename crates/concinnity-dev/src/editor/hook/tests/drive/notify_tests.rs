// src/editor/hook/tests/drive/notify_tests.rs
//
// The toast stack's drive (`hook/drive/notify.rs`): the presses a card claims
// and the ones that fall through to what is behind it, the hidden state the
// drive settles into once the queue empties, and the new-fault latch that
// keeps a behavior fault from toasting every frame.

use concinnity_core::ecs::World;

use crate::editor::behavior::panel::Status;
use crate::editor::hook::tests::fixtures::hook;

use crate::editor::notify;

use crate::editor::toast_overlay;

// The toast stack claims only presses on its live cards; everywhere else the
// press falls through to normal routing. Clicking a card runs its action and
// dismisses it.
#[test]
fn toast_presses_claim_cards_and_fall_through_elsewhere() {
    let mut h = hook(Vec::new());
    let vp = [1280.0, 720.0];
    let mut world = World::new();
    // Empty queue: any press falls straight through.
    assert!(!h.try_toast_press(vp[0] - 20.0, vp[1] - 20.0, vp, &mut world));
    h.notifier
        .error_with("cook failed", notify::Action::OpenConsole);
    h.drive_toasts(&mut world, vp, true, [0.0, 0.0]);
    assert!(!h.toasts_hidden, "a live toast draws");
    // While live, the stack's ids join the layer map above the modal band.
    let card_id = toast_overlay::all_sprite_ids()[0];
    assert!(h.compute_layers().contains_key(&card_id));
    // A press off the stack still falls through.
    assert!(!h.try_toast_press(10.0, 10.0, vp, &mut world));
    // A press on the newest card claims it, runs its action, and dismisses.
    let r = toast_overlay::card_rect(vp, 0, 0);
    assert!(h.try_toast_press(r[0] + 5.0, r[1] + 5.0, vp, &mut world));
    assert!(h.console_open, "the error's action opened the Console");
    assert!(h.notifier.is_empty(), "the card dismissed");
}

// After the last toast goes, the drive hides the overlay once and then does
// nothing per frame; the layer map drops the stack's ids again.
#[test]
fn toast_drive_settles_hidden_when_the_queue_empties() {
    let mut h = hook(Vec::new());
    let vp = [1280.0, 720.0];
    let mut world = World::new();
    h.notifier.success("saved");
    h.drive_toasts(&mut world, vp, true, [0.0, 0.0]);
    assert!(!h.toasts_hidden);
    h.notifier.click_card(0);
    h.drive_toasts(&mut world, vp, true, [0.0, 0.0]);
    assert!(h.toasts_hidden, "one hide pass after the stack empties");
    let card_id = toast_overlay::all_sprite_ids()[0];
    assert!(!h.compute_layers().contains_key(&card_id));
}

// A behavior fault pushed on commit dedupes against the persisting fault, so
// per-keystroke re-checks of the same complaint do not re-toast.
#[test]
fn behavior_fault_toasts_only_on_a_new_fault() {
    let mut h = hook(Vec::new());
    h.behavior_status = Some(Status::message("a behavior needs a name"));
    h.notify_behavior_fault(None);
    assert!(!h.notifier.is_empty(), "a fresh fault toasts");
    h.notifier.click_card(0);
    let prev = h.behavior_fault_message();
    h.notify_behavior_fault(prev);
    assert!(h.notifier.is_empty(), "the same fault stays quiet");
    h.notify_behavior_fault(Some("a different complaint".to_string()));
    assert!(!h.notifier.is_empty(), "a changed fault toasts again");
}
