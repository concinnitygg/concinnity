// src/metal/probe.rs
//
// Scene-captured reflection probes. Each declared `ReflectionProbe` (or an
// auto-seeded grid when a world declares none) is baked into its own cube,
// DISTINCT from `env_map`: the specular reflection term box-projects against the
// probe's influence box and samples its cube, so glossy surfaces and windows
// reflect the actual surrounding geometry instead of the imported (often foreign)
// HDR sky, while the skybox + diffuse irradiance keep sampling `env_map` so the
// visible sky is never replaced by a capture.
//
// Each cube mirrors the main pass exactly -- it reuses the GPU-driven bindless
// cull + the three main-pass geometry sub-paths (`encode_main_into_face`) so the
// folded static + instanced + skinned geometry, and the skybox (a non-cullable
// draw object), all render into each face. The six faces are rendered through
// the cube view-projections in `gfx::reflection_probe` (orientation unit-tested
// there) into the six slices of a capture cube, then convolved into the probe's
// prefiltered radiance cube by the compute kernels in `probe_prefilter.slang`.
// Nothing is read back: the whole bake stays on the GPU timeline. The build-time
// CPU convolution in `bake::environment_map` still serves imported HDR environment
// maps, and the two agree on the roughness ramp and the firefly clamp through the
// shared `PrefilterPlan`.
//
// The bake is STAGGERED, ASYNCHRONOUS, and PIPELINED across frames so the render
// thread NEVER blocks on a capture, walking a `ProbeBakeQueue` cursor so a not-yet-
// baked probe falls back to the sky until its turn. Each probe passes through three
// phases (`gfx::reflection_probe::BakePhase`, driven by the pure `next_bake_action`
// transition table, called once per pipeline slot per frame):
//   * Rendering    -- six cube faces submitted to the GPU WITHOUT
//                     `waitUntilCompleted`; a completion handler flags GPU
//                     completion. The faces draw from a RESERVED ring slot
//                     (`bake_ring_slot`) the frame never overwrites, so the bake's
//                     CPU-written bindless buffers stay valid across the async work.
//   * Prefiltering -- the capture's draw resources are released, the probe cube is
//                     allocated, and the convolution runs as compute dispatches: the
//                     clamped mirror mip plus the capture's source pyramid in the
//                     first frame (all cheap), then ONE GGX mip per frame after it,
//                     so no frame pays the whole convolution.
//   * (install)    -- the finished cube is installed into `probe.maps` +
//                     `probe.set`. No upload: the cube was written in place.
// The Rendering and Prefiltering phases run in PARALLEL across two slots
// (`probe.rendering` / `probe.prefiltering`): once a probe's faces are captured its
// draw resources (the reserved ring slot included) are freed, so the NEXT probe starts
// rendering while the prior probe's cube convolves -- shortening the warm-up vs
// serialising render-then-convolve per probe. Only ONE probe renders at a time (so a
// single reserved ring slot suffices, GPU lifetime unchanged) and only ONE convolves
// at a time (so installs stay in queue order, keeping `probe.maps` aligned with the
// placement list). A re-placement (`set_reflection_probes`) parks BOTH slots' GPU
// resources in a frame-tagged retire pool so they outlive any still-running work.
//
// Known simplifications (documented intentionally):
//   * The scene is captured lit by whatever environment is live at bake time
//     (single bounce): surfaces carry the old env's ambient. The dominant,
//     visible change is that reflections now show real geometry.
//   * Captured before that frame's shadow map is populated, so the probe bakes
//     direct + ambient lighting without contact shadows.
#![deny(unsafe_op_in_unsafe_fn)]

use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer as _, MTLCommandBuffer as _, MTLCommandBufferStatus, MTLCommandQueue as _,
    MTLDevice as _, MTLPixelFormat, MTLResourceOptions, MTLTexture, MTLTextureType,
    MTLTextureUsage,
};

use super::context::{HDR_SAMPLE_COUNT, MtlContext};
use super::descriptors::TextureDesc;
use super::probe_prefilter::{PrefilterGpu, create_capture_cube};
use crate::gfx::reflection_probe::{self, BakeAction, BakePhase, BakeSignals, PrefilterPlan};

// What a runtime capture bakes: face size, mip count, GGX sample count and
// firefly clamp. Shared with the DirectX and Vulkan backends (and with the
// build-time CPU convolution's roughness ramp) so a probe looks the same
// whichever backend captured it.
const PLAN: PrefilterPlan = PrefilterPlan::RUNTIME;
// Cube faces per probe. Rendered one per frame (spread) so a single bake never adds
// the cost of six full-scene captures to one frame.
const PROBE_FACE_COUNT: usize = 6;

// The two pipelined bake slots. One probe renders its six cube faces on the GPU
// (`RenderingBake`, owning the reserved-ring-slot buffers + capture targets) while a
// PRIOR probe's capture convolves into its cube (`PrefilteringBake`, owning the two
// cubes and their per-mip views). Overlapping the convolution with the next probe's
// render shortens the bake warm-up. Only ONE probe holds the reserved ring slot at a
// time (the rendering one), so the GPU-resource lifetime is identical to a single
// in-flight bake; at most one probe convolves at a time, which keeps installs in queue
// order (`probe.maps` is appended one cube per probe).
pub(in crate::metal) struct RenderingBake {
    // Placement index being captured; its cube lands at `probe.maps[index]` on install.
    index: usize,
    placement: reflection_probe::ProbePlacement,
    // Set by the LAST face's completion handler once every face has been submitted.
    done: Arc<AtomicBool>,
    // The next of `PROBE_FACE_COUNT` faces to submit (one per frame, so no single frame
    // pays the whole capture).
    cursor: usize,
    // Capture vantage, snapshotted at start so the six faces are temporally consistent.
    eye: [f32; 3],
    near: f32,
    far: f32,
    elapsed: f32,
    // Loop-invariant buffers + targets shared across the six faces (reserved ring slot).
    gpu: BakeGpu,
    // The record set `gpu`'s object + draw-args buffers were built for. Runtime
    // spawns grow the live draw list while the faces render, so every face culls
    // and draws against this snapshot instead.
    counts: crate::metal::context::DrawRecordCounts,
}

pub(in crate::metal) struct PrefilteringBake {
    // Placement index; its convolved cube lands at `probe.maps[index]` on install.
    index: usize,
    placement: reflection_probe::ProbePlacement,
    // The capture and the probe cube being written, plus their per-mip views.
    gpu: PrefilterGpu,
    // The next destination mip to convolve. Starts at 1: mip 0 is the clamped copy,
    // dispatched with the source pyramid when the slot is filled.
    cursor: u32,
}

// A bake's GPU resources parked behind the frames-in-flight fence when a
// re-placement or a failure interrupts it. Each payload holds at least one
// heap-placed allocation (the probe cube) whose memory would otherwise be handed
// to the next request while a submitted command buffer is still reading it, so
// neither slot may simply drop.
// Never matched on: a payload is held for its lifetime, then dropped by the pool.
#[expect(
    dead_code,
    reason = "payloads are held to defer their free, never read"
)]
pub(in crate::metal) enum RetiredBake {
    Capture(BakeGpu),
    Prefilter(PrefilterGpu),
}

// The GPU resources of one capture, built once at the start and reused across all
// six faces: the shared MSAA pair, the capture cube each face resolves a slice of,
// and the reserved-slot bindless buffers (+ a skinned deformed buffer). Held
// resident for the whole asynchronous capture; the cube outlives it, moving into
// the prefiltering slot as the convolution's source.
pub(in crate::metal) struct BakeGpu {
    msaa_color: Retained<ProtocolObject<dyn MTLTexture>>,
    msaa_depth: Retained<ProtocolObject<dyn MTLTexture>>,
    capture: Retained<ProtocolObject<dyn MTLTexture>>,
    object_buffer: Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    draw_args: Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    tex_args: Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    joint_bufs: Vec<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    morph_weight_bufs: Vec<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    deformed: Option<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
}

impl MtlContext {
    // Set the reflection-probe placements (declared `ReflectionProbe` assets,
    // converted to `ProbePlacement`s by the graphics system). An empty list
    // auto-seeds a grid from the scene bounds, so existing scenes still get local
    // reflections without authoring. Resets the staggered bake so the next
    // eligible frames re-bake from scratch; capped at `MAX_PROBES`.
    pub(in crate::metal) fn set_reflection_probes(
        &mut self,
        declared: &[reflection_probe::ProbePlacement],
    ) {
        use concinnity_core::render::uniforms::MAX_PROBES;
        let mut placements: Vec<reflection_probe::ProbePlacement> = if declared.is_empty() {
            match self.scene_world_bounds() {
                Some((mn, mx)) => {
                    // Object AABBs as occupancy so a probe is not auto-captured from
                    // inside a wall; skip degenerate (non-finite) boxes.
                    let occupancy: Vec<([f32; 3], [f32; 3])> = self
                        .draw
                        .objects
                        .iter()
                        .map(|o| (o.bb_min, o.bb_max))
                        .filter(|(mn, mx)| mn.iter().chain(mx).all(|c| c.is_finite()))
                        .collect();
                    reflection_probe::auto_seed_probes(mn, mx, &occupancy)
                }
                None => Vec::new(),
            }
        } else {
            declared.to_vec()
        };
        if placements.len() > MAX_PROBES {
            tracing::warn!(
                "reflection probes: {} placements, capping at MAX_PROBES={}",
                placements.len(),
                MAX_PROBES
            );
            placements.truncate(MAX_PROBES);
        }
        self.probe.placements = placements;
        self.probe.maps.clear();
        self.probe.set = concinnity_core::render::uniforms::ProbeSet::EMPTY;
        self.probe.bake_queue = reflection_probe::ProbeBakeQueue::new(self.probe.placements.len());
        // Park both slots' GPU resources instead of dropping them: their command
        // buffers may still be reading the reserved-slot buffers, the capture cube or
        // the probe cube being convolved, so defer the free until the frames-in-flight
        // fence guarantees those command buffers have retired.
        self.retire_in_flight_bakes();
    }

    // The reserved transient-ring slot the asynchronous bake builds its bindless
    // buffers into: one past the frame's range `[0, frames_in_flight)`. The frame
    // never writes this slot, so the bake's CPU-written buffers stay valid across
    // its `waitUntilCompleted`-free capture. The bake-relevant rings are sized
    // `frames_in_flight + 1` in `init` to make room for it.
    fn bake_ring_slot(&self) -> usize {
        self.frames_in_flight
    }

    // Park whatever both bake slots hold behind the frames-in-flight fence. Each
    // payload carries a heap-placed cube whose memory would be handed to the next
    // allocation while a submitted command buffer is still reading it if it simply
    // dropped.
    fn retire_in_flight_bakes(&mut self) {
        let frame = self.frame_ring_index;
        if let Some(bake) = self.probe.rendering.take() {
            self.probe
                .retire_pool
                .push(frame, RetiredBake::Capture(bake.gpu));
        }
        if let Some(bake) = self.probe.prefiltering.take() {
            self.probe
                .retire_pool
                .push(frame, RetiredBake::Prefilter(bake.gpu));
        }
    }

    // Advance the asynchronous reflection-probe bake by one step. Called every
    // frame from `draw_frame_inner` after the frames-in-flight fence; cheap once the
    // queue drains and nothing is in flight. Drives the pure `next_bake_action`
    // transition table: start the next probe's capture, move a finished capture into
    // the convolution, convolve one mip, or install a finished cube. Never blocks the
    // render thread. Non-fatal: a failure keeps the current state.
    pub(in crate::metal) fn bake_pending_probes(
        &mut self,
        elapsed: f32,
        near: f32,
        far: f32,
    ) -> Result<(), String> {
        // Free any parked (interrupted) bake resources the fence now guarantees have
        // retired on the GPU.
        self.probe
            .retire_pool
            .collect(self.frame_ring_index, self.frames_in_flight as u64);

        // Permanent ineligibility: the capture renders through the bindless ICB
        // (needs the GPU-driven static path); a world with no real geometry keeps
        // the sky; and a probe only adds value over a real environment (a world on
        // the 1x1 grey fallback has no prefilter chain). The environment is built at
        // init, so its readiness never changes for a world. None of these can become
        // eligible later, so abandon the queue rather than re-checking it forever.
        // Under normal play no bake is in flight here (the gate is stable from the
        // first frame), but a debug shader hot-reload can flip `self.bindless` false
        // after a bake started, so park any in-flight work behind the fence rather
        // than leaking it (its command buffers may still be reading those resources).
        if !self.bindless
            || self.geometry_less
            || self.env_map.prefilter_mip_count <= 1
            || self.probe.prefilter.is_none()
        {
            self.retire_in_flight_bakes();
            self.probe.bake_queue.abort();
            return Ok(());
        }

        // Two pipelined slots advance independently each frame, the pure
        // `next_bake_action` transition table called once per slot. Every transition
        // that can fail routes through `fail_bake` (abandon the rest, keep what baked):
        // the queue cursor advanced when a probe started, so leaving it pending after a
        // mid-bake failure would desync `probe.maps` from the placement list.

        // Prefiltering slot: convolve one mip, or install the finished cube. Doing this
        // FIRST frees the slot so the rendering slot can hand its finished capture over
        // this same frame.
        let prefiltering_occupied = self.probe.prefiltering.is_some();
        let more_mips = self
            .probe
            .prefiltering
            .as_ref()
            .is_some_and(|p| p.cursor < PLAN.mips());
        match reflection_probe::next_bake_action(
            if prefiltering_occupied {
                BakePhase::Prefiltering
            } else {
                BakePhase::Idle
            },
            BakeSignals {
                more_mips,
                // Nothing to wait for: the convolution dispatches and every frame
                // that will sample the cube share one queue, so FIFO completion
                // already orders the writes before the reads, and Metal owns the
                // command buffers' lifetime rather than this bake.
                mips_done: true,
                ..Default::default()
            },
        ) {
            BakeAction::PrefilterMip => {
                if let Err(e) = self.probe_prefilter_next_mip() {
                    self.fail_bake(e);
                    return Ok(());
                }
            }
            BakeAction::Install => {
                if let Err(e) = self.probe_install() {
                    self.fail_bake(e);
                    return Ok(());
                }
            }
            _ => {}
        }
        // The prefiltering slot is free this frame if it was empty or we just installed
        // it (the install is the only transition that empties it).
        let prefiltering_free = self.probe.prefiltering.is_none();

        // Rendering slot: submit one cube face per frame; once the GPU signals all six
        // done AND the prefiltering slot is free, hand the capture over (moving to the
        // prefiltering slot); or, when no probe is rendering, start the next pending
        // placement. Gating `StartPrefilter` on the prefiltering slot being free keeps at
        // most one probe convolving, so installs stay in queue order -- and the next
        // probe's render overlaps the prior probe's convolution, so the warm-up no longer
        // serialises render-then-convolve per probe.
        let rendering_occupied = self.probe.rendering.is_some();
        // `done` only matters once every face is submitted; the completion handler is
        // attached on the last face, so it cannot be set while faces remain.
        let more_faces = self
            .probe
            .rendering
            .as_ref()
            .is_some_and(|r| r.cursor < PROBE_FACE_COUNT);
        let done = self
            .probe
            .rendering
            .as_ref()
            .is_some_and(|r| r.done.load(Ordering::Acquire));
        // Transient ineligibility: geometry may still be streaming in on the first
        // frames. A zero cull keeps the queue cursor so a later frame retries rather than
        // starting an empty capture.
        let eligible = self.cull_count() > 0;
        match reflection_probe::next_bake_action(
            if rendering_occupied {
                BakePhase::Rendering
            } else {
                BakePhase::Idle
            },
            BakeSignals {
                faces_done: done && prefiltering_free,
                queue_pending: self.probe.bake_queue.pending(),
                eligible,
                more_faces,
                ..Default::default()
            },
        ) {
            BakeAction::RenderFace => {
                if let Err(e) = self.probe_render_next_face() {
                    self.fail_bake(e);
                }
            }
            BakeAction::StartPrefilter => {
                if let Err(e) = self.probe_begin_prefilter() {
                    self.fail_bake(e);
                }
            }
            BakeAction::StartNext => {
                if let Err(e) = self.probe_start_next(near, far, elapsed) {
                    self.fail_bake(e);
                }
            }
            BakeAction::PrefilterMip | BakeAction::Install | BakeAction::Idle => {}
        }
        Ok(())
    }

    // Abandon the rest of the bake after an unrecoverable error, keeping the cubes
    // already installed. The queue cursor advanced when the current probe started,
    // so aborting (cursor -> end) is what keeps `probe.maps` aligned with the
    // placement list; the sky covers the remaining placements.
    fn fail_bake(&mut self, e: String) {
        tracing::warn!(
            "reflection probe bake failed, keeping {} baked: {e}",
            self.probe.maps.len()
        );
        // Abandon BOTH slots: a prefiltering-slot (install) failure leaves `probe.maps`
        // short by one, so a later rendering probe would install at a gapped index and
        // desync the box alignment. Both slots' resources go behind the fence rather
        // than dropping: their command buffers may still be reading them.
        self.retire_in_flight_bakes();
        self.probe.bake_queue.abort();
    }

    // Begin baking the next pending placement: build the reserved-slot bindless
    // buffers + capture targets ONCE (they are loop-invariant across the six faces),
    // and enter `Rendering` with the face cursor at 0. No face is submitted here; the
    // faces follow one per frame via `probe_render_next_face`, so a single frame never
    // pays the cost of all six full-scene captures.
    fn probe_start_next(&mut self, near: f32, far: f32, elapsed: f32) -> Result<(), String> {
        let Some(index) = self.probe.bake_queue.take_next() else {
            return Ok(());
        };
        // Note: unlike the install-time check, `index == probe.maps.len()` does NOT hold
        // here -- with the pipeline this probe can START rendering while the PRIOR probe
        // is still prefiltering (not yet installed), so `probe.maps` may be one entry
        // behind. The box-alignment invariant is enforced at install instead, where the
        // single-prefiltering rule guarantees installs land in queue order.
        let placement = self.probe.placements[index];
        let eye = placement.position;
        let slot = self.bake_ring_slot();

        // Build into the reserved ring slot (the frame never touches it), so these
        // CPU-written buffers stay valid for the whole asynchronous capture. They are
        // frustum-independent (only the per-face view/projection differs), so they are
        // built once and reused by every face.
        let object_buffer = self
            .build_object_buffer(slot)?
            .ok_or("probe: no static geometry to bake")?;
        let draw_args = self
            .build_draw_args_buffer(eye, slot)?
            .ok_or("probe: no draw args to bake")?;
        let counts = self.draw_record_counts();
        let tex_args = self
            .build_bindless_texture_args(slot)?
            .ok_or("probe: no bindless texture args")?;
        let joint_bufs = self.build_joint_buffers(slot)?;
        let morph_weight_bufs = self.build_morph_weight_buffers(slot)?;
        // The folded skinned tail draws compute-deformed vertices. The frame's
        // deformed ring is overwritten every frame, so an async capture needs its
        // OWN deformed buffer (Shared storage -- a Private one page-faults in this
        // cross-command-buffer producer/consumer pattern, like the frame's). `None`
        // for static worlds.
        let deformed: Option<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>> =
            if self.draw.n_skinned > 0 {
                match self.skinned.deformed.first().map(|b| b.length()) {
                    Some(len) if len > 0 => Some(
                        self.device
                            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
                            .ok_or("probe: failed to allocate deformed buffer")?,
                    ),
                    _ => None,
                }
            } else {
                None
            };

        // One reused MSAA colour + depth pair (faces render serially across frames),
        // and the capture cube each face resolves its own slice of.
        let msaa_color = make_msaa_color(&self.device, PLAN.face_size())?;
        let msaa_depth = make_msaa_depth(&self.device, PLAN.face_size())?;
        let capture = create_capture_cube(&self.device, &PLAN)?;

        self.probe.rendering = Some(RenderingBake {
            index,
            placement,
            done: Arc::new(AtomicBool::new(false)),
            cursor: 0,
            eye,
            near,
            far,
            elapsed,
            gpu: BakeGpu {
                msaa_color,
                msaa_depth,
                capture,
                object_buffer,
                draw_args,
                tex_args,
                joint_bufs,
                morph_weight_bufs,
                deformed,
            },
            counts,
        });
        Ok(())
    }

    // Submit the in-flight capture's next cube face (one per frame). On the LAST face
    // a completion handler is attached (before that face's commit, as Metal requires)
    // to flag GPU completion: single-queue FIFO completion means every face is done
    // when this one is. The shared `cull.icb` is GPU-written, so Metal hazard-tracks
    // it: each face's cull (and the frame's own cull) waits for the prior read,
    // ordering the reuse correctly with no explicit barrier or `waitUntilCompleted`.
    fn probe_render_next_face(&mut self) -> Result<(), String> {
        let Some(bake) = self.probe.rendering.as_ref() else {
            return Err("probe: render face with no capture in flight".into());
        };
        let (face, eye, near, far, elapsed, counts) = (
            bake.cursor,
            bake.eye,
            bake.near,
            bake.far,
            bake.elapsed,
            bake.counts,
        );
        let attach_done = face + 1 == PROBE_FACE_COUNT;

        // The shared ICB is otherwise sized from the frame's live draw list, which
        // this capture's snapshot does not follow; size it for the snapshot before
        // the face's cull encodes into it.
        self.ensure_icb_capacity(counts.total)?;

        let vp = reflection_probe::face_view_projection(eye, face, near, far);
        let view = reflection_probe::face_view_matrix(eye, face);
        let frustum = crate::gfx::frustum::Frustum::from_view_projection(vp);

        let RenderingBake { done, gpu, .. } = self
            .probe
            .rendering
            .as_ref()
            .expect("probe capture was just checked");

        // Cull command buffer: fills the shared ICB for this face's frustum.
        let cull_cb = self
            .command_queue
            .commandBuffer()
            .ok_or("probe: failed to get cull command buffer")?;
        // Skin once, on the first face: the deformed vertices are a pure function of
        // the bind pose + joint palettes (both loop-invariant), so the pose is
        // identical for every face. FIFO + hazard tracking on the Shared deformed
        // buffer order that single write before every face render reads it.
        if face == 0
            && let Some(def) = gpu.deformed.as_ref()
        {
            self.encode_main_skin(
                &cull_cb,
                def,
                crate::metal::raytrace::MainSkinBuffers {
                    joints: &gpu.joint_bufs,
                    morph_weights: &gpu.morph_weight_bufs,
                },
            )?;
        }
        self.encode_cull(
            &cull_cb,
            &gpu.object_buffer,
            &gpu.draw_args,
            &frustum,
            eye,
            counts,
        )?;
        cull_cb.commit();

        // Render command buffer: reads the ICB into this face. Instances fold into
        // the bindless ICB, so the legacy prepared set draws nothing here (empty).
        let render_cb = self
            .command_queue
            .commandBuffer()
            .ok_or("probe: failed to get render command buffer")?;
        let prepared = super::instanced::PreparedInstances {
            clusters: Vec::new(),
        };
        self.encode_main_into_face(
            &render_cb,
            crate::metal::draw::main::FaceTargets {
                color_msaa: &gpu.msaa_color,
                depth_msaa: &gpu.msaa_depth,
                resolve: &gpu.capture,
                resolve_slice: face,
            },
            crate::metal::draw::main::MainPassCamera {
                elapsed,
                vp,
                view,
                cam_pos: eye,
            },
            crate::metal::draw::main::DrawInputs {
                visible: &[],
                prepared_instances: &prepared,
                skinned_joint_bufs: &gpu.joint_bufs,
            },
            crate::metal::draw::main::GpuFrameBuffers {
                object_buffer: Some(&gpu.object_buffer),
                bindless_tex_args: Some(&gpu.tex_args),
                deformed_skinned: gpu.deformed.as_ref(),
                counts,
            },
            // Probe cube bake reuses the main cull ICB (no per-face mirror cull).
            None,
        )?;
        if attach_done {
            let flag = Arc::clone(done);
            let handler = block2::RcBlock::new(
                move |cb: NonNull<ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>| {
                    // SAFETY: the completion handler is invoked by Metal with a valid
                    // command-buffer pointer.
                    let cb = unsafe { cb.as_ref() };
                    if cb.status() == MTLCommandBufferStatus::Error {
                        tracing::error!(
                            "reflection probe face bake faulted (async): {:?}",
                            cb.error()
                        );
                    }
                    flag.store(true, Ordering::Release);
                },
            );
            // SAFETY: addCompletedHandler copies the block, so the RcBlock may drop
            // here; it must be added before the commit below.
            unsafe {
                render_cb.addCompletedHandler(block2::RcBlock::as_ptr(&handler));
            }
        }
        render_cb.commit();

        // Advance the cursor (a separate mutable borrow now the render borrows ended).
        if let Some(RenderingBake { cursor, .. }) = &mut self.probe.rendering {
            *cursor += 1;
        }
        Ok(())
    }

    // The GPU has finished the in-flight capture: release the capture's draw
    // resources (the reserved ring slot included, so the next probe can start
    // rendering), take ownership of the capture cube, allocate the probe cube it
    // convolves into, and dispatch the cheap half of the convolution -- the clamped
    // mirror mip plus the capture's source pyramid. The bake moves to Prefiltering
    // with the mip cursor at 1.
    fn probe_begin_prefilter(&mut self) -> Result<(), String> {
        let RenderingBake {
            index,
            placement,
            gpu,
            ..
        } = self
            .probe
            .rendering
            .take()
            .ok_or("probe: prefilter with no bake in flight")?;
        let BakeGpu { capture, .. } = gpu;
        // Everything but the capture cube drops here -- safe, the GPU is done with
        // all of it (the last face's completion handler flagged `done`, observed
        // Acquire before this call).

        let prefilter_gpu = PrefilterGpu::new(&self.allocator, capture, &PLAN)?;
        let cmd_buf = self
            .command_queue
            .commandBuffer()
            .ok_or("probe: failed to get prefilter command buffer")?;
        self.encode_probe_pyramid(&cmd_buf, &prefilter_gpu, &PLAN)?;
        cmd_buf.commit();

        self.probe.prefiltering = Some(PrefilteringBake {
            index,
            placement,
            gpu: prefilter_gpu,
            cursor: 1,
        });
        Ok(())
    }

    // Convolve one destination mip of the in-flight probe cube (one per frame, so no
    // frame pays the whole convolution). The dispatches run on their own command
    // buffers with no `waitUntilCompleted`: each reads the capture pyramid and writes
    // a mip nothing else touches, and single-queue FIFO ordering puts every one of
    // them after the pyramid build that produced their source.
    fn probe_prefilter_next_mip(&mut self) -> Result<(), String> {
        let (cursor, gpu) = {
            let bake = self
                .probe
                .prefiltering
                .as_ref()
                .ok_or("probe: convolve with no bake in flight")?;
            (bake.cursor, &bake.gpu)
        };
        let cmd_buf = self
            .command_queue
            .commandBuffer()
            .ok_or("probe: failed to get convolution command buffer")?;
        self.encode_probe_ggx_mip(&cmd_buf, gpu, &PLAN, cursor)?;
        cmd_buf.commit();

        if let Some(bake) = self.probe.prefiltering.as_mut() {
            bake.cursor += 1;
        }
        Ok(())
    }

    // Every mip is dispatched: install the probe cube into `probe.maps` + `probe.set`
    // (the specular reflection source), leaving `env_map` / the sky untouched. There
    // is nothing to upload -- the cube was written in place -- and nothing to wait
    // for: the dispatches and the frames that will sample the cube share one queue,
    // so the reads are already ordered after the writes.
    fn probe_install(&mut self) -> Result<(), String> {
        let PrefilteringBake {
            index,
            placement: p,
            gpu,
            ..
        } = self
            .probe
            .prefiltering
            .take()
            .ok_or("probe: install with no bake in flight")?;
        debug_assert_eq!(index, self.probe.maps.len());
        self.probe.maps.push(super::context::ProbeCube {
            prefilter: gpu.into_probe_cube(),
        });
        self.probe.set.probes[index] = concinnity_core::render::uniforms::ProbeUniforms {
            box_min: [p.box_min[0], p.box_min[1], p.box_min[2], 1.0],
            box_max: [p.box_max[0], p.box_max[1], p.box_max[2], 0.0],
            probe_pos: [p.position[0], p.position[1], p.position[2], 0.0],
        };
        self.probe.set.count = self.probe.maps.len() as u32;
        tracing::info!(
            "reflection probes: baked {}/{}",
            index + 1,
            self.probe.placements.len()
        );
        Ok(())
    }

    // World-space bounds over every static draw object, skipping degenerate
    // (non-finite) AABBs. `None` for an empty scene. Folded instances + skinned
    // objects sit inside the static extent for the scenes this bakes, so the
    // static objects' union is a good probe-centring volume.
    pub(in crate::metal) fn scene_world_bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        reflection_probe::fold_world_bounds(self.draw.objects.iter().map(|o| (o.bb_min, o.bb_max)))
    }
}

// MSAA HDR colour face: RGBA16Float, 4x, render-target only -- matches the main
// pipeline's attachment format + sample count so `self.pipeline_state` binds.
fn make_msaa_color(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    size: u32,
) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, String> {
    let desc = TextureDesc {
        kind: MTLTextureType::Type2DMultisample,
        format: MTLPixelFormat::RGBA16Float,
        width: size as usize,
        height: size as usize,
        sample_count: HDR_SAMPLE_COUNT as usize,
        usage: MTLTextureUsage::RenderTarget,
        ..Default::default()
    }
    .build();
    device
        .newTextureWithDescriptor(&desc)
        .ok_or_else(|| "probe: failed to create MSAA colour face".into())
}

// MSAA depth face: Depth32Float, 4x, render-target only. Cleared per face and
// discarded -- the probe consumes only the resolved colour.
fn make_msaa_depth(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    size: u32,
) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, String> {
    let desc = TextureDesc {
        kind: MTLTextureType::Type2DMultisample,
        format: MTLPixelFormat::Depth32Float,
        width: size as usize,
        height: size as usize,
        sample_count: HDR_SAMPLE_COUNT as usize,
        usage: MTLTextureUsage::RenderTarget,
        ..Default::default()
    }
    .build();
    device
        .newTextureWithDescriptor(&desc)
        .ok_or_else(|| "probe: failed to create MSAA depth face".into())
}
