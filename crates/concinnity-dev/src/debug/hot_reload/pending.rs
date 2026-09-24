//! Process-wide "world.jsonl changed" / "world-loaded Shader stage changed" /
//! "Markdown story source changed" / "Animation source changed" signals (dev
//! sessions only). Set by the asset hot-reload watcher and the `reload-assets`
//! debug tool call; consumed by the per-frame reload poll in
//! `super::state::run_frame` and, for animations, by `super::animation`. They
//! live in the debug tree rather than the engine because nothing in the engine
//! references them: the reload passes that read them (`super::passes`,
//! `super::animation`) are driven entirely from `DebugHook::tick`.

use std::sync::atomic::{AtomicBool, Ordering};

// "world.jsonl changed" signal. Consumed by the world.jsonl reload poll to
// re-apply `Prop` transform edits in place via `backend.update_model`. V1
// covers transform-only edits (position / rotation / scale); add / remove and
// non-transform arg changes are detected and logged but not applied.
static PENDING_WORLD: AtomicBool = AtomicBool::new(false);

// "world-loaded Shader stage source changed" signal. Consumed by the
// shader-stage reload poll to re-compile each captured Shader stage source
// and rebuild the affected backend pipelines (main / instanced / shadow) into
// temporaries before swapping them in. Kept separate from `PENDING_WORLD` so a
// shader save does not also kick the Prop-diff and procedural-mesh passes.
static PENDING_SHADER_STAGES: AtomicBool = AtomicBool::new(false);

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

// Raise the "world-loaded Shader stage source changed" flag. Called by the
// asset hot-reload watcher when a captured `.hlsl` file is saved and by the
// debug `reload-assets` handler.
pub(crate) fn set_pending_shader_stages() {
    PENDING_SHADER_STAGES.store(true, Ordering::SeqCst);
}

// Swap the "world-loaded Shader stage source changed" flag to `false`,
// returning whether it was set. The reload poll calls this at frame start; a
// `true` result kicks the per-stage recompile + pipeline rebuild pass.
pub(crate) fn take_pending_shader_stages() -> bool {
    PENDING_SHADER_STAGES.swap(false, Ordering::SeqCst)
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
    fn shader_stages_flag_round_trips() {
        let _guard = crate::test_support::lock();
        let prior = take_pending_shader_stages();
        assert!(!take_pending_shader_stages());
        set_pending_shader_stages();
        assert!(take_pending_shader_stages());
        assert!(!take_pending_shader_stages());
        if prior {
            set_pending_shader_stages();
        }
    }
}
