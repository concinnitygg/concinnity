//! Process-wide "world.jsonl changed" / "these world Shaders changed" /
//! "Markdown story source changed" / "Animation source changed" signals (dev
//! sessions only). Set by the asset hot-reload watcher and the `reload-assets`
//! debug tool call; consumed by the per-frame reload poll in
//! `super::state::run_frame` and, for animations, by `super::animation`. They
//! live in the debug tree rather than the engine because nothing in the engine
//! references them: the reload passes that read them (`super::passes`,
//! `super::animation`) are driven entirely from `DebugHook::tick`.

use concinnity_core::ecs::asset_id::AssetId;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, PoisonError};

// "world.jsonl changed" signal. Consumed by the world.jsonl reload poll to
// re-apply `Prop` transform edits in place via `backend.update_model`. V1
// covers transform-only edits (position / rotation / scale); add / remove and
// non-transform arg changes are detected and logged but not applied.
static PENDING_WORLD: AtomicBool = AtomicBool::new(false);

// The world Shaders whose files changed. The watcher marks the Shaders that
// read a saved file and `reload-assets` marks every one; the shader reload
// poll takes the set and recompiles just those. Kept apart from `PENDING_WORLD`
// so a shader save does not also kick the Prop-diff and procedural-mesh passes.
static PENDING_SHADERS: Mutex<PendingShaders> = Mutex::new(PendingShaders {
    all: false,
    ids: BTreeSet::new(),
});

// Which world Shaders a reload was asked for.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct PendingShaders {
    // Every Shader in the catalog, whatever `ids` holds.
    pub all: bool,
    pub ids: BTreeSet<AssetId>,
}

impl PendingShaders {
    pub(crate) fn is_empty(&self) -> bool {
        !self.all && self.ids.is_empty()
    }

    // Whether the Shader `id` is asked for.
    pub(crate) fn wants(&self, id: AssetId) -> bool {
        self.all || self.ids.contains(&id)
    }
}

fn pending_shaders() -> std::sync::MutexGuard<'static, PendingShaders> {
    PENDING_SHADERS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

// Raise the "world.jsonl changed" flag. Called by the asset hot-reload watcher
// when a `.jsonl` save fires and by the `reload-assets` debug tool call.
pub(crate) fn set_pending_world() {
    PENDING_WORLD.store(true, Ordering::SeqCst);
}

// Swap the "world.jsonl changed" flag to `false`, returning whether it was
// set. The reload poll calls this at frame start; a `true` result kicks the
// Prop-transform re-apply pass.
pub(crate) fn take_pending_world() -> bool {
    PENDING_WORLD.swap(false, Ordering::SeqCst)
}

// "Animation source changed" signal. Consumed by the animation clip reload in
// `super::animation`, which re-imports every file-backed clip from source.
static PENDING_ANIMATIONS: AtomicBool = AtomicBool::new(false);

// Raise the "Animation source changed" flag. Called by the asset hot-reload
// watcher and the `reload-assets` debug tool call.
pub(crate) fn set_pending_animations() {
    PENDING_ANIMATIONS.store(true, Ordering::SeqCst);
}

// Swap the "Animation source changed" flag to `false`, returning whether it
// was set. `super::animation::reload_clips_if_pending` calls this; a `true`
// result kicks the per-clip re-import pass.
pub(crate) fn take_pending_animations() -> bool {
    PENDING_ANIMATIONS.swap(false, Ordering::SeqCst)
}

// "Markdown story source changed" signal. Consumed by the story reload poll
// to re-expand the world's `StoryImport`s and hand each freshly compiled
// `Story` graph to the running story system. Kept separate from
// `PENDING_WORLD` so a dialogue save does not also kick the procedural-mesh
// and fog passes.
static PENDING_STORIES: AtomicBool = AtomicBool::new(false);

// Mark the Shaders `ids` for recompiling. Called by the asset hot-reload
// watcher with the Shaders that read a saved file.
pub(crate) fn mark_shaders_pending(ids: impl IntoIterator<Item = AssetId>) {
    pending_shaders().ids.extend(ids);
}

// Mark every Shader for recompiling. Called by the `reload-assets` handler.
pub(crate) fn mark_all_shaders_pending() {
    pending_shaders().all = true;
}

// Take the pending Shader set, leaving it empty. The reload poll calls this at
// frame start and recompiles whatever it names.
pub(crate) fn take_pending_shaders() -> PendingShaders {
    std::mem::take(&mut *pending_shaders())
}

// Raise the "Markdown story source changed" flag. Called by the asset
// hot-reload watcher when a `.md` save fires in a watched story-source
// directory.
pub(crate) fn set_pending_stories() {
    PENDING_STORIES.store(true, Ordering::SeqCst);
}

// Swap the "Markdown story source changed" flag to `false`, returning whether
// it was set. The reload poll calls this at frame start; a `true` result
// kicks the story re-expansion pass.
pub(crate) fn take_pending_stories() -> bool {
    PENDING_STORIES.swap(false, Ordering::SeqCst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_flag_round_trips() {
        // These flags are process-global; serialize on the shared lock so the
        // reload-driving tests elsewhere don't observe a mid-toggle value.
        let _guard = crate::test_support::lock();
        // Capture and restore so this test does not leak state into others.
        let prior = take_pending_world();
        assert!(!take_pending_world());
        set_pending_world();
        assert!(take_pending_world());
        assert!(!take_pending_world());
        if prior {
            set_pending_world();
        }
    }

    #[test]
    fn stories_flag_round_trips() {
        let _guard = crate::test_support::lock();
        let prior = take_pending_stories();
        assert!(!take_pending_stories());
        set_pending_stories();
        assert!(take_pending_stories());
        assert!(!take_pending_stories());
        if prior {
            set_pending_stories();
        }
    }

    #[test]
    fn pending_shaders_collect_marks_until_taken() {
        let _guard = crate::test_support::lock();
        let prior = take_pending_shaders();
        assert!(take_pending_shaders().is_empty());
        mark_shaders_pending([AssetId(3), AssetId(1)]);
        mark_shaders_pending([AssetId(3)]);
        let taken = take_pending_shaders();
        assert!(!taken.all);
        assert_eq!(
            taken.ids.into_iter().collect::<Vec<_>>(),
            [AssetId(1), AssetId(3)]
        );
        assert!(take_pending_shaders().is_empty());

        mark_all_shaders_pending();
        let all = take_pending_shaders();
        assert!(all.wants(AssetId(99)));
        assert!(take_pending_shaders().is_empty());

        mark_shaders_pending(prior.ids);
        if prior.all {
            mark_all_shaders_pending();
        }
    }

    #[test]
    fn a_pending_set_wants_only_what_it_names_unless_all() {
        let some = PendingShaders {
            all: false,
            ids: [AssetId(2)].into(),
        };
        assert!(some.wants(AssetId(2)));
        assert!(!some.wants(AssetId(5)));
        assert!(!some.is_empty());
        assert!(PendingShaders::default().is_empty());
    }
}
