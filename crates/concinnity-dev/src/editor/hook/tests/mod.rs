// src/editor/hook/tests/mod.rs
//
// The hook's tests, mirroring the modules they cover: one companion per
// subject, named after it, under the same group directory the subject sits in.
// So `edit/console_tests.rs` here covers `hook/edit/console.rs`, and a subject
// at the hook root has its companion at this level. `fixtures` holds what more
// than one companion needs -- a hook over a given entry list, the worlds a tick
// reads its input and typed fields from, and the entry literals.
//
// A module small and pure enough to read alongside its own assertions keeps an
// inline `#[cfg(test)] mod tests` beside its code instead, which is the shape
// the rest of the crate uses: `camera_pose`, `fly`, `drive/axes` and
// `drive/outline`. The one companion here that covers a family rather than a
// single module is `camera_tests`, whose three drives are interlocked -- a
// bookmark recall rides the glide and cancels a tumble.

pub(super) mod fixtures;

mod behavior_keys_tests;
mod camera_tests;
mod drop_floor_tests;
mod duplicate_tests;
mod editing_tests;
mod edits_tests;
mod hide_tests;
mod layout_tests;
mod panels_tests;
mod pick_tests;
mod routing_tests;
mod sim_control_tests;
mod worlds_start_tests;

mod drag;
mod drive;
mod edit;
