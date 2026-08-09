// src/gfx/animation/mod.rs
//
// Skeletal animation playback. An internal system (not a declarable asset):
// `World::start` constructs one whenever the world contains any `Animation`
// or `AnimGraph` component, then it produces fresh skinning matrices for each
// `SkeletonPose` every frame.
//
// Each target `SkinnedMesh` gets a bucket of clips driven in one of two
// modes: `Flat` blends every clip by a live weight vector (startup fade-in +
// runtime crossfades; see `flat`), while `Graph` walks a compiled animation
// state machine whose transitions are driven by the target's `AnimParams`
// component (see `graph`). Runtime debug commands for both modes are drained
// in `commands`.

mod commands;
mod flat;
mod graph;
mod ik;
mod root;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use crate::assets::{Animation, SkeletonPose};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, SkinnedMeshHandle, StepResult, System};
use crate::gfx::skinning::{self, AnimationClip};
use crate::jobs;

use flat::{ClipEntry, FlatState, Transition};
use graph::GraphTarget;

// Per-`SkinnedMesh` bucket: the clips targeting it plus the mode that drives
// them. Clip storage is mode-independent so hot-reload can swap a clip in
// place either way.
struct TargetState {
    clips: Vec<ClipEntry>,
    mode: TargetMode,
}

// How a bucket's clips are driven each frame.
enum TargetMode {
    // Weighted blend of every clip (the default).
    Flat(FlatState),
    // A compiled `AnimGraph` state machine owns the bucket.
    Graph(GraphTarget),
}

// One hot-reload entry for a file-backed `Animation`. Captured at init
// alongside the runtime clip; consulted by the per-step reload pass when the
// shared `PENDING_ANIMATIONS` flag fires (see
// [`crate::app::dev_flags::take_pending_animations`]). Inline-authored
// animations (no `source`) carry no entry; there's no file to watch and
// the build pipeline never expanded one.
//
// `pub` (with public fields) because the editor crate's hot-reload drive reads
// these to re-import the clip from source, then pushes the result back through
// `AnimationSystem::apply_reloaded_clip`. The GLB decode itself lives in the
// editor crate; the runtime crate only stores the catalogue.
#[derive(Debug, Clone)]
pub struct AnimationReloadEntry {
    // Target `SkinnedMesh` handle, also the key into
    // [`AnimationSystem::targets`] where this clip lives.
    pub target: SkinnedMeshHandle,
    // Position in the target bucket's `clips`. Set at init when the clip is
    // first pushed; stable for the process lifetime since the Vec is
    // neither rebuilt nor trimmed.
    pub clip_index: usize,
    // `.glb` source path verbatim from the asset declaration; used as-is by
    // the GLB parser at reload time.
    pub source: String,
    // The target mesh's [`SkinnedMesh::skin_index`]: the clip re-imports
    // against the same skeleton the build cooked it against.
    pub skin_index: u32,
    // Mirrors [`Animation::animation_index`].
    pub animation_index: u32,
    // Mirrors [`Animation::animation_name`] (precedence over index when
    // non-empty).
    pub animation_name: String,
    // Mirrors [`Animation::sample_rate`]; the FBX reload path bakes at the
    // same rate the build used.
    pub sample_rate: f32,
    // Mirrors [`Animation::weight`]; the .glb has nothing equivalent, so
    // it's carried through the reload unchanged.
    pub weight: f32,
    // Mirrors [`Animation::looping`]; same rationale as `weight`.
    pub looping: bool,
}

// Skeletal animation playback behavior. Constructed internally by
// `World::start` when the world declares any `Animation` or `AnimGraph`;
// never a world-declared asset, so it carries no config.
pub struct AnimationSystem {
    // Per-target clip buckets keyed by the `SkinnedMesh` handle they animate.
    // Ordered so per-frame iteration (and the RootMotion events it emits) is
    // deterministic across runs.
    targets: BTreeMap<SkinnedMeshHandle, TargetState>,
    // Interned-name -> handle index snapshotted at init, so the debug WS
    // animation commands (which address a mesh by name) can find the bucket.
    name_index: crate::gfx::skinned_mesh_map::SkinnedMeshNameIndex,
    // Wall-clock origin, captured on the first step.
    start: Option<Instant>,
    // Clip time `t` of the previous step, for the graph clocks' delta time.
    last_step_secs: Option<f32>,
    // When a menu opened (and froze playback), if currently paused. On resume
    // the origin `start` is shifted forward by the paused span so clip time `t`
    // is continuous across the pause: the animation freezes on its current pose
    // and resumes from it, with no jump.
    pause_anchor: Option<Instant>,
    // One entry per file-backed Animation, captured at init under
    // `cn debug`. Empty when hot-reload is off or every clip is inline.
    reload_entries: Vec<AnimationReloadEntry>,
}

impl std::fmt::Debug for AnimationSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnimationSystem")
            .field("targets", &self.targets.len())
            .field("reload_entries", &self.reload_entries.len())
            .finish()
    }
}

impl Default for AnimationSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl AnimationSystem {
    // Fresh playback state with no clips. Clips and graphs are drained from
    // the world's components in [`System::init`].
    pub fn new() -> Self {
        Self {
            targets: BTreeMap::new(),
            name_index: Default::default(),
            start: None,
            last_step_secs: None,
            pause_anchor: None,
            reload_entries: Vec::new(),
        }
    }

    // The file-backed clips captured at init under `cn debug`. The editor
    // crate's hot-reload drive reads these to re-import each clip from source.
    // Empty when hot-reload is off or every clip is inline.
    // `dead_code` allowed while the runtime crate still carries the legacy
    // binary; the only caller is the editor crate (external). Removed when the
    // binary moves out of the runtime crate.
    pub fn reload_entries(&self) -> &[AnimationReloadEntry] {
        &self.reload_entries
    }

    // Swap a freshly re-imported `clip` into the bucket slot identified by
    // `target` + `clip_index`, restoring its declared `weight`. Returns false
    // if the target bucket disappeared or the slot index is out of range
    // (a half-applied reload is impossible: nothing is mutated on miss). The
    // editor crate calls this after decoding the source GLB; the runtime crate
    // does no decoding of its own.
    pub fn apply_reloaded_clip(
        &mut self,
        target: SkinnedMeshHandle,
        clip_index: usize,
        clip: AnimationClip,
        weight: f32,
    ) -> bool {
        let Some(bucket) = self.targets.get_mut(&target) else {
            return false;
        };
        let Some(slot) = bucket.clips.get_mut(clip_index) else {
            return false;
        };
        slot.clip = clip;
        slot.declared_weight = weight;
        // A graph compiled this clip's duration into any member playing it;
        // keep those in sync so wrap / phase / exit-time math tracks the new
        // clip. The compiled loop mode is left as resolved at compile time.
        if let TargetMode::Graph(g) = &mut bucket.mode {
            let duration = bucket.clips[clip_index].clip.duration;
            g.graph.refresh_clip_duration(clip_index, duration);
        }
        true
    }
}

// The animation origin to use on the frame a pause ends. Shifting the original
// origin forward by the paused span (now - anchor) holds clip time
// `t = now - origin` exactly where it was when the pause began, so playback
// resumes from the frozen pose with no jump. Split out so the continuity
// property is unit-testable without a live system.
fn resumed_origin(start: Instant, anchor: Instant, now: Instant) -> Instant {
    start + now.saturating_duration_since(anchor)
}

impl System for AnimationSystem {
    fn init(&mut self, ctx: &mut PipelineContext) {
        // Clips accumulate per target mesh; how a bucket's clips combine is
        // decided below (graph if the world declares one, weighted blend
        // otherwise).
        let capture_sources = crate::app::dev_flags::enabled();
        // Interned-name -> handle index published by GraphicsSystem (which
        // loaded the SkinnedMesh table before this system inits), kept for the
        // debug WS animation commands. The correlation web itself is keyed by
        // the authored `target` handles directly.
        self.name_index = ctx
            .resource::<crate::gfx::skinned_mesh_map::SkinnedMeshNameIndex>()
            .cloned()
            .unwrap_or_default();
        let skin_index = ctx
            .resource::<crate::gfx::skinned_mesh_map::SkinnedMeshSkinIndex>()
            .cloned()
            .unwrap_or_default();
        // Animation asset id -> (target bucket, clip slot), for resolving
        // graph clip references onto bucket indices.
        let mut clip_slots: HashMap<AssetId, (SkinnedMeshHandle, usize)> = HashMap::new();
        let mut count = 0usize;
        for anim in ctx.drain::<Animation>() {
            let Some(target) = anim.target else {
                tracing::warn!("AnimationSystem: Animation has no target SkinnedMesh, ignored");
                continue;
            };
            let weight = anim.weight;
            let fade_in_secs = anim.fade_in_secs.max(0.0);
            let state = self.targets.entry(target).or_insert_with(|| TargetState {
                clips: Vec::new(),
                mode: TargetMode::Flat(FlatState::default()),
            });
            let clip_index = state.clips.len();
            state.clips.push(ClipEntry {
                clip: anim.to_clip(),
                declared_weight: weight,
                fade_in_secs,
            });
            clip_slots.insert(anim.asset_id, (target, clip_index));
            // Each new clip starts at full declared weight unless it requests
            // a fade-in, in which case it begins at zero and ramps up.
            let initial = if fade_in_secs > 0.0 { 0.0 } else { weight };
            if let TargetMode::Flat(flat) = &mut state.mode {
                flat.current_weights.push(initial);
            }
            if capture_sources && !anim.source.is_empty() {
                self.reload_entries.push(AnimationReloadEntry {
                    target,
                    clip_index,
                    source: anim.source.clone(),
                    skin_index: skin_index.get(target),
                    animation_index: anim.animation_index,
                    animation_name: anim.animation_name.clone(),
                    sample_rate: anim.sample_rate,
                    weight,
                    looping: anim.looping,
                });
            }
            count += 1;
        }

        // Graphs take ownership of their target's bucket; each publishes an
        // `AnimParams` component seeded with its declared defaults.
        let graph_count = graph::install_graphs(&mut self.targets, ctx, &clip_slots);

        // Build a startup transition for any flat bucket whose clips requested
        // a fade-in. The transition runs from zero to the declared weights over
        // the bucket's longest fade-in; clips with `fade_in_secs == 0` start
        // already at their declared weight via `current_weights`, so the lerp
        // leaves them alone. Graph buckets ignore fade-in (the graph owns
        // weights outright).
        for state in self.targets.values_mut() {
            let TargetMode::Flat(flat) = &mut state.mode else {
                continue;
            };
            let max_fade = state
                .clips
                .iter()
                .fold(0.0f32, |m, c| m.max(c.fade_in_secs));
            if max_fade > 0.0 {
                let source = flat.current_weights.clone();
                let target: Vec<f32> = state.clips.iter().map(|c| c.declared_weight).collect();
                flat.transition = Some(Transition {
                    source_weights: source,
                    target_weights: target,
                    // Start the ramp on the first step (negative until then,
                    // overwritten in `step`).
                    start_secs: 0.0,
                    duration_secs: max_fade,
                });
            }
        }
        tracing::info!(
            "AnimationSystem: {} clip(s) across {} target mesh(es); {} graph(s); {} \
             file-backed clip(s) captured for hot-reload",
            count,
            self.targets.len(),
            graph_count,
            self.reload_entries.len()
        );
    }

    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        // Asset hot-reload of file-backed clips (`cn debug` only) is driven
        // from the binary's `DebugHook::tick` via `reload_clips_if_pending`,
        // not here. `cn run` has no debug hook, so this step is reload-free.

        let now = Instant::now();

        // Freeze while a menu is open: skip all sampling so animation stops
        // consuming CPU/GPU behind the menu, recording when the pause began.
        // The flag is published by GraphicsSystem, which runs first this tick.
        let paused = ctx
            .resource::<crate::ecs::MenuActive>()
            .is_some_and(|m| m.0);
        if paused {
            self.pause_anchor.get_or_insert(now);
            return StepResult::Continue;
        }
        // Resuming: advance the origin by the paused span so clip time `t` stays
        // continuous -- the animation resumes from the exact pose it froze on,
        // with no jump. (A pause before the first step has no origin yet, so it
        // just defers the capture below.)
        if let Some(anchor) = self.pause_anchor.take()
            && let Some(start) = self.start.as_mut()
        {
            *start = resumed_origin(*start, anchor, now);
        }

        let start = *self.start.get_or_insert(now);
        let t = (now - start).as_secs_f32();
        // Graph clocks advance by delta time; the origin shift above keeps
        // `t` continuous across a pause, so the first post-pause delta stays
        // one frame long.
        let dt = t - self.last_step_secs.replace(t).unwrap_or(t);

        // First-frame fix-up: the startup transition built in `init` has
        // `start_secs == 0.0`. We don't know the wall-clock origin until the
        // first step, so re-anchor any in-flight transition that hasn't yet
        // started elapsing.
        for state in self.targets.values_mut() {
            if let TargetMode::Flat(flat) = &mut state.mode
                && let Some(tr) = flat.transition.as_mut()
                && tr.start_secs == 0.0
            {
                tr.start_secs = t;
            }
        }

        // Runtime commands (`cn debug` WS `anim-crossfade` / `anim-param` /
        // `anim-state`) are drained from the binary's `DebugHook::tick` via
        // `apply_runtime_commands`, not here.

        // Advance each bucket's driver before sampling: flat buckets move
        // their weight transitions, graph buckets sync `AnimParams` and step
        // their cursor. Each advance also yields the frame's root-motion
        // displacement (mesh-local), published as one `RootMotion` event per
        // target that actually moved; the rig drive in PhysicsSystem
        // consumes them next frame.
        for (target, state) in &mut self.targets {
            let TargetState { clips, mode } = state;
            let delta = match mode {
                TargetMode::Flat(flat) => {
                    flat::advance_weights(flat, t);
                    root::flat_root_delta(clips, &flat.current_weights, t - dt, t)
                }
                TargetMode::Graph(g) => {
                    let before = g.cursor.clone();
                    graph::step_target(g, *target, ctx, dt);
                    crate::gfx::anim_graph::cursor_root_delta(
                        &g.graph,
                        &before,
                        &g.cursor,
                        &g.params,
                        &|i| &clips[i].clip,
                    )
                }
            };
            if delta != [0.0; 3] {
                ctx.events_mut::<crate::assets::RootMotion>()
                    .send(crate::assets::RootMotion {
                        target: *target,
                        delta,
                    });
            }
        }

        // Foot-pinning inputs for this frame: per graph target with IK
        // chains, the rig transform and each chain's ground pin (probe hits
        // answered by PhysicsSystem earlier this tick).
        let ik_frames = ik::frame_inputs(&self.targets, ctx);

        // Each `SkeletonPose` is sampled and skinned independently, so the
        // per-pose work fans across the job pool and joins before returning.
        let targets = &self.targets;
        let poses = ctx.query_slice_mut::<SkeletonPose>();
        jobs::pool().parallel_for(poses, |pose| {
            let Some(state) = targets.get(&pose.mesh_id) else {
                return;
            };
            // Split borrows: the scratch buffers and outputs are written
            // while the skeleton is read.
            let crate::assets::SkeletonPose {
                skeleton,
                scratch,
                joint_matrices,
                morph_weights,
                updated,
                ..
            } = pose;
            match &state.mode {
                TargetMode::Flat(flat) => match state.clips.as_slice() {
                    [] => return,
                    [single] => {
                        // One-clip buckets ignore weight and play at full
                        // strength; the blend would be a no-op anyway.
                        single.clip.sample_into(t, skeleton, &mut scratch.locals)
                    }
                    many => {
                        // Incremental normalized fold: the first clip seeds
                        // the accumulator (regardless of weight, so an
                        // all-zero bucket falls back to it), later clips at
                        // weight 0 are skipped without sampling.
                        let mut fold = skinning::PoseBlend::new(&mut scratch.locals);
                        for (i, entry) in many.iter().enumerate() {
                            let w = flat.current_weights.get(i).copied().unwrap_or(1.0);
                            if fold.seeded() && w <= 0.0 {
                                continue;
                            }
                            entry.clip.sample_into(t, skeleton, &mut scratch.clip);
                            fold.add(&scratch.clip, w);
                        }
                    }
                },
                TargetMode::Graph(g) => crate::gfx::anim_graph::sample_graph_pose_into(
                    &g.graph,
                    &g.cursor,
                    &g.params,
                    |i| &state.clips[i].clip,
                    skeleton,
                    scratch,
                ),
            }
            if let TargetMode::Graph(g) = &state.mode
                && let Some(frame) = ik_frames.get(&pose.mesh_id)
            {
                ik::apply_chains(skeleton, scratch, &g.chains, frame);
            }
            skeleton.skinning_matrices_into(&scratch.locals, joint_matrices);
            *updated = true;

            // Morph weights follow the same flat blend as the pose; clips
            // without a morph track do not dilute the result. Graph-driven
            // targets do not sample morph tracks.
            if let TargetMode::Flat(flat) = &state.mode {
                let acc = &mut scratch.morph;
                acc.clear();
                let mut weight_sum = 0.0f32;
                for (i, entry) in state.clips.iter().enumerate() {
                    if entry.clip.morph_keys.is_empty() {
                        continue;
                    }
                    let w = if state.clips.len() == 1 {
                        1.0
                    } else {
                        flat.current_weights.get(i).copied().unwrap_or(0.0)
                    };
                    if w <= 0.0 {
                        continue;
                    }
                    entry.clip.sample_morph_weights_into(
                        t,
                        entry.clip.looping,
                        &mut scratch.weights,
                    );
                    if acc.len() < scratch.weights.len() {
                        acc.resize(scratch.weights.len(), 0.0);
                    }
                    for (a, s) in acc.iter_mut().zip(scratch.weights.iter()) {
                        *a += s * w;
                    }
                    weight_sum += w;
                }
                if weight_sum > 0.0 {
                    for a in acc.iter_mut() {
                        *a /= weight_sum;
                    }
                    morph_weights.clear();
                    morph_weights.extend_from_slice(acc);
                }
            }
        });

        // Refresh the ground-probe rays from the posed foot positions for
        // PhysicsSystem to answer next frame.
        ik::refresh_rays(&self.targets, ctx);

        StepResult::Continue
    }
}
