// src/debug/hot_reload/driver.rs
//
// Per-frame drive of the asset / shader / world.jsonl hot-reload passes,
// shared by every dev session. A plain `cn editor` runs the driver as its own
// per-frame hook; `cn debug` (and `cn editor --debug-port`) drive it from
// inside `DebugServer::tick`, which layers the WebSocket-only concerns
// (runtime spawn commands, camera motion) around it. Each session constructs
// exactly one driver, so a reload is never applied twice.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crate::debug_hook::DebugHook;
use crate::ecs::{SystemAsset, World};

use super::state::{AssetHotReloadState, FrameHotReloadEffects, run_frame};

pub(crate) struct HotReloadDriver {
    // Reload catalogue + filesystem watcher + in-flight decode handles. Armed
    // from the GraphicsSystem's init-captured sources on the first tick that
    // finds them, and re-armed whenever a fresh capture appears (the editor's
    // live preview rebuild re-runs init), so the catalogue never goes stale
    // against the current backend slots.
    state: Option<AssetHotReloadState>,
    // The editor's toast queue, when driving inside an editor session: the
    // reload passes report apply results through it. `None` under a bare
    // `cn debug`, which has no toast surface.
    notifier: Option<crate::editor::notify::Notifier>,
}

impl HotReloadDriver {
    pub(crate) fn new() -> Self {
        Self {
            state: None,
            notifier: None,
        }
    }

    // Report reload results through an editor session's toast queue as well
    // as the log.
    pub(crate) fn with_notifier(mut self, notifier: crate::editor::notify::Notifier) -> Self {
        self.notifier = Some(notifier);
        self
    }

    // The shared "reload requested" flag of the armed state, for the debug
    // WS `reload-assets` command. `None` until a tick arms the state; the
    // caller must re-query after ticks since a re-arm swaps the flag.
    pub(crate) fn pending(&self) -> Option<Arc<AtomicBool>> {
        self.state.as_ref().map(|s| Arc::clone(&s.pending))
    }

    // Rebuild the reload state from a freshly captured source catalogue.
    // Dropping the previous state stops its watcher and abandons any
    // in-flight decode aimed at the replaced world's slots.
    pub(crate) fn arm(
        &mut self,
        sources: crate::gfx::graphics_system::hot_reload_sources::HotReloadSources,
    ) {
        self.state = Some(AssetHotReloadState::from_sources(sources));
    }

    // Run the reload passes once for this frame and apply their ECS
    // side-effects. A world with no GraphicsSystem (or no captured sources)
    // is a cheap no-op.
    pub(crate) fn drive(&mut self, world: &mut World) {
        let mut effects = None;
        // The backend lives in the world's parked slot (disjoint from the
        // system list), so both are borrowed at once for the apply passes.
        let (systems, mut backend) = world.systems_and_render_backend();
        for system in systems {
            match system {
                SystemAsset::GraphicsSystem(gs) => {
                    // Arm (or re-arm after a world rebuild) from the init-
                    // captured sources; must precede the apply-parts borrow
                    // of `gs`.
                    if let Some(sources) = gs.take_hot_reload_sources() {
                        self.arm(sources);
                    }
                    if let (Some(state), Some(backend)) = (self.state.as_mut(), backend.take()) {
                        let mut apply = gs.hot_reload_apply_parts(backend);
                        effects = Some(run_frame(state, &mut apply, self.notifier.as_ref()));
                    }
                }
                SystemAsset::AnimationSystem(anim) => {
                    crate::anim_reload::reload_clips_if_pending(anim);
                }
                _ => {}
            }
        }
        if let Some(effects) = effects {
            apply_effects(world, effects);
        }
    }
}

impl DebugHook for HotReloadDriver {
    fn tick(&mut self, world: &mut World) {
        self.drive(world);
    }
}

// Apply the ECS side-effects one reload pass produced, once the system borrow
// is released.
pub(crate) fn apply_effects(world: &mut World, effects: FrameHotReloadEffects) {
    // Splice any skeleton-shape changes into the ECS-owned `SkeletonPose`
    // components so `AnimationSystem` produces right-sized output going
    // forward.
    if !effects.skeleton_updates.is_empty() {
        let index_to_new: std::collections::HashMap<usize, crate::gfx::skinning::Skeleton> =
            effects
                .skeleton_updates
                .into_iter()
                .map(|u| (u.skinned_index, u.new_skeleton))
                .collect();
        let mut applied = 0usize;
        for pose in world.query_mut::<crate::components::SkeletonPose>() {
            if let Some(new_skel) = index_to_new.get(&pose.skinned_index) {
                pose.skeleton = new_skel.clone();
                pose.joint_matrices = pose.skeleton.bind_skinning_matrices();
                pose.updated = true;
                applied += 1;
            }
        }
        tracing::info!(
            "asset hot-reload: applied skeleton-shape change to {} SkeletonPose component(s)",
            applied
        );
    }

    // Hand freshly re-compiled story graphs to the story system, which swaps
    // them in while keeping the play position. The drive runs before the
    // world step, so the swap lands the same frame.
    for story in effects.story_updates {
        world
            .events_mut::<crate::components::StoryReload>()
            .send(crate::components::StoryReload { story });
    }
}
