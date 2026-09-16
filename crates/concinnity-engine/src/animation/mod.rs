//! Skeletal animation playback. An internal system (not a declarable asset):
//! `World::start` constructs one whenever the world contains any `Animation`
//! or `AnimationGraph` component, then it produces fresh skinning matrices for each
//! `SkeletonPose` every frame.
//!
//! Each target `SkinnedMesh` gets a bucket of clips driven in one of two
//! modes: `Flat` blends every clip by a live weight vector (startup fade-in +
//! runtime crossfades; see `flat`), while `Graph` walks a compiled animation
//! state machine whose transitions are driven by the target's `AnimationParams`
//! component (see `graph`). Runtime debug commands for both modes are drained
//! in `commands`.

mod commands;
mod flat;
mod graph;
mod ik;
mod morph;
mod root;
pub mod runtime_queue;
#[cfg(test)]
mod tests;

use concinnity_core::animation::anim_graph;
use concinnity_core::animation::pose_blend::PoseBlend;
use concinnity_core::animation::skeleton::AnimationClip;
use concinnity_core::components::Animation;
use concinnity_core::components::AnimationParams;
use concinnity_core::components::CharacterRig;
use concinnity_core::components::GroundProbes;
use concinnity_core::components::RootMotionEvent;
use concinnity_core::components::SkeletonPose;
use concinnity_core::ecs::asset_id::AssetId;
use concinnity_core::ecs::{
    Access, FrameTime, MenuActive, PipelineContext, SkinnedMeshHandle, StepResult, System,
};
use concinnity_host::thread::jobs;
use flat::{ClipEntry, FlatState, Transition};
use graph::GraphTarget;
use std::collections::{BTreeMap, HashMap};

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
    // A compiled `AnimationGraph` state machine owns the bucket.
    Graph(GraphTarget),
}

/// One hot-reload entry for a file-backed `Animation`. Captured at init
/// alongside the runtime clip; consulted by the per-step reload pass when the
/// dev tooling raises its pending-animations flag. Inline-authored
/// animations (no `source`) carry no entry; there's no file to watch and
/// the build pipeline never expanded one.
///
/// `pub` (with public fields) because the editor crate's hot-reload drive reads
/// these to re-import the clip from source, then pushes the result back through
/// `AnimationSystem::apply_reloaded_clip`. The GLB decode itself lives in the
/// editor crate; the runtime crate only stores the catalog.
#[derive(Debug, Clone)]
pub struct AnimationReloadEntry {
    /// EntityTarget `SkinnedMesh` handle, also the key into
    /// `AnimationSystem::targets` where this clip lives.
    pub target: SkinnedMeshHandle,
    /// Position in the target bucket's `clips`. Set at init when the clip is
    /// first pushed; stable for the process lifetime since the Vec is
    /// neither rebuilt nor trimmed.
    pub clip_index: usize,
    /// `.glb` source path verbatim from the asset declaration; used as-is by
    /// the GLB parser at reload time.
    pub source: String,
    /// The target mesh's `skin_index`: the clip re-imports
    /// against the same skeleton the build cooked it against.
    pub skin_index: u32,
    /// Mirrors [`Animation::animation_index`].
    pub animation_index: u32,
    /// Mirrors [`Animation::animation_name`] (precedence over index when
    /// non-empty).
    pub animation_name: String,
    /// Mirrors [`Animation::sample_rate`]; the FBX reload path bakes at the
    /// same rate the build used.
    pub sample_rate: f32,
    /// Mirrors [`Animation::weight`]; the .glb has nothing equivalent, so
    /// it's carried through the reload unchanged.
    pub weight: f32,
    /// Mirrors [`Animation::looping`]; same rationale as `weight`.
    pub looping: bool,
}

/// Skeletal animation playback behavior. Constructed internally by
/// `World::start` when the world declares any `Animation` or `AnimationGraph`;
/// never a world-declared asset, so it carries no config.
pub struct AnimationSystem {
    // Per-target clip buckets keyed by the `SkinnedMesh` handle they animate.
    // Ordered so per-frame iteration (and the RootMotionEvent events it emits) is
    // deterministic across runs.
    targets: BTreeMap<SkinnedMeshHandle, TargetState>,
    // Interned-name -> handle index snapshotted at init, so the animation
    // debug tool calls (which address a mesh by name) can find the bucket.
    name_index: crate::gfx::skinned_mesh_map::SkinnedMeshNameIndex,
    // Clip time `t` in seconds: the frame time of every unpaused step, so
    // playback freezes on its pose behind a menu and resumes from it.
    clip_secs: f32,
    // One entry per file-backed Animation, captured at init under
    // `cn debug`. Empty when hot-reload is off or every clip is inline.
    reload_entries: Vec<AnimationReloadEntry>,
    // Per-target IK solve inputs, refreshed in place each frame so the pin
    // buffers persist across frames.
    ik_frames: std::collections::HashMap<SkinnedMeshHandle, ik::IkFrame>,
    // Foot-position scratch for the probe-ray refresh, reused across targets.
    ik_feet_scratch: Vec<[f32; 3]>,
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
    /// Fresh playback state with no clips. Clips and graphs are drained from
    /// the world's components in [`System::init`].
    pub fn new() -> Self {
        Self {
            targets: BTreeMap::new(),
            name_index: Default::default(),
            clip_secs: 0.0,
            reload_entries: Vec::new(),
            ik_frames: std::collections::HashMap::new(),
            ik_feet_scratch: Vec::new(),
        }
    }

    /// The file-backed clips captured at init under `cn debug`. The editor
    /// crate's hot-reload drive reads these to re-import each clip from source.
    /// Empty when hot-reload is off or every clip is inline.
    pub fn reload_entries(&self) -> &[AnimationReloadEntry] {
        &self.reload_entries
    }

    /// Swap a freshly re-imported `clip` into the bucket slot identified by
    /// `target` + `clip_index`, restoring its declared `weight`. Returns false
    /// if the target bucket disappeared or the slot index is out of range
    /// (a half-applied reload is impossible: nothing is mutated on miss). The
    /// editor crate calls this after decoding the source GLB; the runtime crate
    /// does no decoding of its own.
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

impl System for AnimationSystem {
    fn access(&self) -> Access {
        Access::new()
            .reads_components(crate::component_mask![CharacterRig])
            .writes_components(crate::component_mask![
                SkeletonPose,
                AnimationParams,
                GroundProbes,
            ])
            .reads_resources(crate::resource_mask![MenuActive, FrameTime])
            .writes_resources(crate::resource_mask![RootMotionEvent])
    }

    fn init(&mut self, ctx: &mut PipelineContext) {
        // Clips accumulate per target mesh; how a bucket's clips combine is
        // decided below (graph if the world declares one, weighted blend
        // otherwise).
        let capture_sources = crate::app::dev_flags::enabled();
        // Interned-name -> handle index published by GraphicsSystem (which
        // loaded the SkinnedMesh table before this system inits), kept for the
        // animation debug tool calls. The correlation web itself is keyed by
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
        // `AnimationParams` component seeded with its declared defaults.
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
                    // The ramp starts with the clip clock.
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

        // Freeze while a menu is open: skip all sampling so animation stops
        // consuming CPU/GPU behind the menu, and let none of the paused frames'
        // time reach the clip clock, so resuming continues from the frozen pose.
        // The flag is published by OverlaySystem, which runs first this tick.
        let paused = ctx.resource::<MenuActive>().is_some_and(|m| m.0);
        if paused {
            return StepResult::Continue;
        }
        let dt = ctx
            .resource::<FrameTime>()
            .copied()
            .unwrap_or_default()
            .dt
            .max(0.0);
        self.clip_secs += dt;
        let t = self.clip_secs;

        // Runtime commands (the `anim-crossfade` / `anim-param` / `anim-state`
        // debug tool calls) are drained from the editor's `DebugHook::tick` via
        // `apply_runtime_commands`, not here.

        // Advance each bucket's driver before sampling: flat buckets move
        // their weight transitions, graph buckets sync `AnimationParams` and step
        // their cursor. Each advance also yields the frame's root-motion
        // displacement (mesh-local), published as one `RootMotionEvent` event per
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
                    anim_graph::cursor_root_delta(&g.graph, &before, &g.cursor, &g.params, &|i| {
                        &clips[i].clip
                    })
                }
            };
            if delta != [0.0; 3] {
                ctx.events_mut::<RootMotionEvent>().send(RootMotionEvent {
                    target: *target,
                    delta,
                });
            }
        }

        // Foot-pinning inputs for this frame: per graph target with IK
        // chains, the rig transform and each chain's ground pin (probe hits
        // answered by PhysicsSystem earlier this tick).
        ik::frame_inputs(&self.targets, ctx, &mut self.ik_frames);
        let ik_frames = &self.ik_frames;

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
            let SkeletonPose {
                skeleton,
                scratch,
                joint_matrices,
                morph_weights,
                morph_base,
                proportions,
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
                        let mut fold = PoseBlend::new(&mut scratch.locals);
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
                TargetMode::Graph(g) => anim_graph::sample_graph_pose_into(
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
            // The shape's proportion layer re-shapes the posed locals; the
            // inverse bind matrices stay as authored.
            proportions.apply(&mut scratch.locals);
            skeleton.skinning_matrices_into(&scratch.locals, joint_matrices);
            *updated = true;

            // Morph weights follow the same flat blend as the pose, added onto
            // the shape's base layer. Graph-driven targets do not sample
            // morph tracks.
            if let TargetMode::Flat(flat) = &state.mode {
                morph::update_weights(&state.clips, flat, t, morph_base, scratch, morph_weights);
            }
        });

        // Refresh the ground-probe rays from the posed foot positions for
        // PhysicsSystem to answer next frame.
        ik::refresh_rays(&self.targets, ctx, &mut self.ik_feet_scratch);

        StepResult::Continue
    }
}
