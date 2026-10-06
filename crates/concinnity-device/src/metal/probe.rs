//! Scene-captured reflection probes. Each declared `ReflectionProbe` (or an
//! auto-seeded grid when a world declares none) is baked into its own cube,
//! DISTINCT from `env_map`: the specular reflection term box-projects against the
//! probe's influence box and samples its cube, so glossy surfaces and windows
//! reflect the actual surrounding geometry instead of the imported (often foreign)
//! HDR sky, while the background + diffuse irradiance keep sampling `env_map` so
//! the visible sky is never replaced by a capture.
//!
//! Each cube mirrors the main pass exactly -- it reuses the GPU-driven bindless
//! cull + the three main-pass geometry sub-paths (`encode_main_into_face`) so the
//! folded static + instanced + skinned geometry render into each face, with the
//! environment drawn behind them as the main pass draws it. The six faces are
//! rendered through the cube view-projections in `gfx::reflection_probe`
//! (orientation unit-tested there) into the six slices of a capture cube, then
//! convolved into the probe's
//! prefiltered radiance cube by the compute kernels in `probe_prefilter.hlsl`.
//! Nothing is read back: the whole bake stays on the GPU timeline. The build-time
//! CPU convolution in `bake::environment_map` still serves imported HDR environment
//! maps, and the two agree on the roughness ramp and the firefly clamp through the
//! shared `PrefilterPlan`.
//!
//! The bake is STAGGERED, ASYNCHRONOUS, and PIPELINED across frames so the render
//! thread NEVER blocks on a capture, walking a `ProbeBakeQueue` cursor so a not-yet-
//! baked probe falls back to the sky until its turn. Each probe passes through three
//! phases, sequenced by the shared `ProbeBake` this module serves as a
//! `ProbeBakeDevice`:
//!   * Rendering    -- six cube faces submitted to the GPU WITHOUT
//!     `waitUntilCompleted`; a completion handler flags GPU
//!     completion. The faces draw their object and draw-args records
//!     from a RESERVED ring slot (`bake_ring_slot`) the frame never
//!     overwrites, so they stay valid across the async work, and sample
//!     textures through the frame's own bindless arguments.
//!   * Prefiltering -- the capture's draw resources are released, and the
//!     convolution into the probe's cube of the array runs as compute dispatches: the
//!     clamped mirror mip plus the capture's source pyramid in the
//!     first frame (all cheap), then ONE GGX mip per frame after it,
//!     so no frame pays the whole convolution.
//!   * (install)    -- the probe's record joins the probe book, which is what
//!     makes the shaders read its cube. No upload: the cube was written in place.
//!
//! The Rendering and Prefiltering phases run in PARALLEL across the bake's two
//! slots: once a probe's faces are captured its
//! draw resources (the reserved ring slot included) are freed, so the NEXT probe starts
//! rendering while the prior probe's cube convolves -- shortening the warm-up vs
//! serializing render-then-convolve per probe. Only ONE probe renders at a time (so a
//! single reserved ring slot suffices, GPU lifetime unchanged) and only ONE convolves
//! at a time (so installs stay in queue order, keeping the book's records aligned with
//! the placement list). A re-placement (`set_reflection_probes`) parks BOTH slots' GPU
//! resources in a frame-tagged retire pool so they outlive any still-running work.
//!
//! Known simplifications (documented intentionally):
//!   * The scene is captured lit by whatever environment is live at bake time
//!     (single bounce): surfaces carry the old env's ambient. The dominant,
//!     visible change is that reflections now show real geometry.
//!   * Captured before that frame's shadow map is populated, so the probe bakes
//!     direct + ambient lighting without contact shadows.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::probe_bake::{
    CAPTURE_FACES, ProbeBake, ProbeBakeDevice, capture_ring_slot,
};
use concinnity_core::render::probe_book::ProbeBook;
use concinnity_core::render::reflection_probe::{self, PrefilterPlan, ProbePlacement};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer as _, MTLCommandBuffer as _, MTLCommandQueue as _, MTLDevice as _, MTLPixelFormat,
    MTLResourceOptions, MTLTexture, MTLTextureType, MTLTextureUsage,
};
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::context::MtlContext;
use super::descriptors::TextureDesc;
use super::error::allocation_failed;
use super::probe_prefilter::{PrefilterGpu, create_capture_cube};

// What a runtime capture bakes: face size, mip count, GGX sample count and
// firefly clamp. Shared with the DirectX and Vulkan backends (and with the
// build-time CPU convolution's roughness ramp) so a probe looks the same
// whichever backend captured it.
pub(super) const PLAN: PrefilterPlan = PrefilterPlan::RUNTIME;

// The two pipelined bake slots. One probe renders its six cube faces on the GPU
// (`RenderingBake`, owning the reserved-ring-slot buffers + capture targets) while a
// PRIOR probe's capture convolves into its cube (`PrefilteringBake`, owning the two
// cubes and their per-mip views). Overlapping the convolution with the next probe's
// render shortens the bake warm-up. Only ONE probe holds the reserved ring slot at a
// time (the rendering one), so the GPU-resource lifetime is identical to a single
// in-flight bake; at most one probe convolves at a time, which keeps installs in queue
// order (the probe book's records gain one record per probe).
pub(crate) struct RenderingBake {
    // Set by the LAST face's completion handler once every face has been submitted.
    done: Arc<AtomicBool>,
    // Capture vantage, snapshotted at start so the six faces are temporally consistent.
    eye: [f32; 3],
    capture_distance: Option<f32>,
    elapsed: f32,
    // Loop-invariant buffers + targets shared across the six faces (reserved ring slot).
    gpu: BakeGpu,
    // The record set `gpu`'s object + draw-args buffers were built for. Runtime
    // spawns grow the live draw list while the faces render, so every face culls
    // and draws against this snapshot instead.
    counts: crate::metal::context::DrawRecordCounts,
}

pub(crate) struct PrefilteringBake {
    // The capture and the per-mip views of it and of the cube being written.
    gpu: PrefilterGpu,
}

// The bake's two slots: a capture rendering its faces and a capture convolving
// into its cube.
pub(in crate::metal) type MtlProbeBake = ProbeBake<RenderingBake, PrefilteringBake>;

// What the frame hands the bake: the time a capture starting this frame
// animates its faces with, and the frame's bindless texture arguments every face
// samples through.
pub(crate) struct ProbeFrame<'f> {
    pub(in crate::metal) elapsed: f32,
    pub(in crate::metal) tex_args: Option<&'f Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
}

// Probe resources parked behind the frames-in-flight fence: a bake's when a
// re-placement or a failure interrupts it, and a cube array a larger placement
// list replaced. A submitted command buffer may still be reading any of them,
// and the array is reached through an argument buffer the command buffer does
// not retain, so none may simply drop.
// Never matched on: a payload is held for its lifetime, then dropped by the pool.
#[expect(
    dead_code,
    reason = "payloads are held to defer their free, never read"
)]
pub(in crate::metal) enum RetiredBake {
    Capture(BakeGpu),
    Prefilter(PrefilterGpu),
    CubeArray(super::probe_set::ProbeCubeArray),
}

// The GPU resources of one capture, built once at the start and reused across all
// six faces: the shared MSAA pair, the capture cube each face resolves a slice of,
// and the reserved-slot bindless buffers (+ a skinned deformed buffer). Held
// resident for the whole asynchronous capture; the cube outlives it, moving into
// the prefiltering slot as the convolution's source.
pub(in crate::metal) struct BakeGpu {
    // `None` at one sample, where each face draws straight into its cube slice.
    msaa_color: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
    depth: Retained<ProtocolObject<dyn MTLTexture>>,
    capture: Retained<ProtocolObject<dyn MTLTexture>>,
    object_buffer: Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    material_params: Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    draw_args: Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    joint_bufs: Vec<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    morph_weight_bufs: Vec<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    deformed: Option<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
}

impl MtlContext {
    // Set the reflection-probe placements (declared `ReflectionProbe` assets,
    // converted to `ProbePlacement`s by the graphics system). An empty list
    // auto-seeds a grid from the scene bounds, so existing scenes still get local
    // reflections without authoring. Resets the staggered bake so the next
    // eligible frames re-bake from scratch, and grows the cube array when the
    // list outgrows it; a world whose array cannot grow keeps the sky.
    pub(in crate::metal) fn set_reflection_probes(
        &mut self,
        declared: &[reflection_probe::ProbePlacement],
    ) {
        let placements = reflection_probe::resolve_placements(
            declared,
            self.state.draw.objects.iter().map(|o| (o.bb_min, o.bb_max)),
        );
        let placed = self.with_probe_bake(|bake, ctx| bake.place(ctx, placements));
        crate::probe_report::report_probe_placement(placed);
    }

    // The reserved transient-ring slot the asynchronous bake builds its bindless
    // buffers into: one past the frame's range `[0, frames_in_flight)`. The frame
    // never writes this slot, so the bake's CPU-written buffers stay valid across
    // its `waitUntilCompleted`-free capture. The bake-relevant rings are sized
    // `frames_in_flight + 1` in `init` to make room for it.
    fn bake_ring_slot(&self) -> usize {
        capture_ring_slot(self.frames_in_flight)
    }

    // Advance the prefiltering half of the asynchronous reflection-probe bake by one
    // step: convolve one mip, or install a finished cube. Called every frame from
    // `draw_frame_inner` after the frames-in-flight fence and before the frame
    // builds its argument buffers, so an installed cube is sampled that same frame.
    // Cheap once the queue drains and nothing is in flight. Never blocks the render
    // thread; a failure abandons the rest of the bake and keeps what baked.
    pub(in crate::metal) fn advance_probe_prefilter(&mut self) {
        // Free any parked (interrupted) bake resources the fence now guarantees have
        // retired on the GPU.
        self.probe
            .retire_pool
            .collect(self.frame_ring_index, self.frames_in_flight as u64);
        let report = self.with_probe_bake(|bake, ctx| bake.advance_prefilter(ctx));
        crate::probe_report::report_probe_bake(report);
    }

    // Advance the capture half of the bake by one step: submit the next cube face,
    // hand a captured probe to the prefiltering slot, or start the next pending
    // placement. Called once the frame has built its bindless texture arguments,
    // which every face samples through: they name the textures live this frame,
    // and the fence keeps their ring slot intact until the face retires.
    pub(in crate::metal) fn advance_probe_capture(&mut self, frame: &ProbeFrame<'_>) {
        let report = self.with_probe_bake(|bake, ctx| bake.advance_capture(ctx, frame));
        crate::probe_report::report_probe_bake(report);
    }

    // Run `f` over the bake with this context as its device. The slots are lent
    // to `f`, so the context reads them as empty for the call.
    fn with_probe_bake<R>(&mut self, f: impl FnOnce(&mut MtlProbeBake, &mut Self) -> R) -> R {
        let mut bake = std::mem::take(&mut self.probe.bake);
        let out = f(&mut bake, self);
        self.probe.bake = bake;
        out
    }

    // Build the capture of the next placement: the reserved-slot bindless buffers
    // + capture targets ONCE (they are loop-invariant across the six faces). No
    // face is submitted here; the faces follow one per frame via
    // `record_probe_face`, so a single frame never pays the cost of all six
    // full-scene captures.
    fn start_probe_capture(
        &mut self,
        frame: &ProbeFrame<'_>,
        placement: ProbePlacement,
    ) -> RenderResult<RenderingBake> {
        let ProbeFrame { elapsed, .. } = *frame;
        let eye = placement.position;
        let slot = self.bake_ring_slot();

        // Build into the reserved ring slot (the frame never touches it), so these
        // CPU-written buffers stay valid for the whole asynchronous capture. They are
        // frustum-independent (only the per-face view/projection differs), so they are
        // built once and reused by every face.
        let object_buffer = self
            .build_object_buffer(slot)?
            .ok_or_else(|| RenderError::Other("probe: no static geometry to bake".into()))?;
        let material_params = self.rings.material_params.buffer(&self.hw.device, slot)?;
        let draw_args = self
            .build_draw_args_buffer(
                eye,
                slot,
                concinnity_core::render::model_history::HistoryMode::Untracked,
            )?
            .ok_or_else(|| RenderError::Other("probe: no draw args to bake".into()))?;
        let counts = self.draw_record_counts();
        let joint_bufs = self.build_joint_buffers(slot)?;
        let morph_weight_bufs = self.build_morph_weight_buffers(slot)?;
        // The folded skinned tail draws compute-deformed vertices. The frame's
        // deformed ring is overwritten every frame, so an async capture needs its
        // OWN deformed buffer (Shared storage -- a Private one page-faults in this
        // cross-command-buffer producer/consumer pattern, like the frame's). `None`
        // for static worlds.
        let deformed: Option<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>> =
            if self.state.draw.n_skinned > 0 {
                match self.skinned.deformed.first().map(|b| b.length()) {
                    Some(len) if len > 0 => Some(
                        self.hw
                            .device
                            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
                            .ok_or_else(|| allocation_failed("probe deformed buffer"))?,
                    ),
                    _ => None,
                }
            } else {
                None
            };

        // One reused color + depth pair (faces render serially across frames),
        // and the capture cube each face resolves its own slice of. The sample
        // count is the main pipelines' -- a face binds them -- so a
        // single-sample world skips the color attachment entirely.
        let samples = self.targets.hdr.sample_count;
        let msaa_color = (samples > 1)
            .then(|| make_face_color(&self.hw.device, PLAN.face_size(), samples))
            .transpose()?;
        let depth = make_face_depth(&self.hw.device, PLAN.face_size(), samples)?;
        let capture = create_capture_cube(&self.hw.device, &PLAN)?;

        Ok(RenderingBake {
            done: Arc::new(AtomicBool::new(false)),
            eye,
            capture_distance: placement.capture_distance,
            elapsed,
            gpu: BakeGpu {
                msaa_color,
                depth,
                capture,
                object_buffer,
                material_params,
                draw_args,
                joint_bufs,
                morph_weight_bufs,
                deformed,
            },
            counts,
        })
    }

    // Submit the in-flight capture's next cube face (one per frame). On the LAST face
    // a completion handler is attached (before that face's commit, as Metal requires)
    // to flag GPU completion: single-queue FIFO completion means every face is done
    // when this one is. The shared `cull.icb` is GPU-written, so Metal hazard-tracks
    // it: each face's cull (and the frame's own cull) waits for the prior read,
    // ordering the reuse correctly with no explicit barrier or `waitUntilCompleted`.
    fn record_probe_face(
        &mut self,
        bake: &RenderingBake,
        face: usize,
        tex_args: &Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    ) -> RenderResult<()> {
        let (eye, elapsed, counts) = (bake.eye, bake.elapsed, bake.counts);
        let attach_done = face + 1 == CAPTURE_FACES;

        // The shared ICB is otherwise sized from the frame's live draw list, which
        // this capture's snapshot does not follow; size it for the snapshot before
        // the face's cull encodes into it.
        self.ensure_icb_capacity(counts.total)?;

        let vp = reflection_probe::face_view_projection(eye, face);
        let view = reflection_probe::face_view_matrix(eye, face);
        let frustum = reflection_probe::face_frustum(eye, face, bake.capture_distance);

        let RenderingBake { done, gpu, .. } = bake;

        // Cull command buffer: fills the shared ICB for this face's frustum.
        let cull_cb =
            self.hw.command_queue.commandBuffer().ok_or_else(|| {
                RenderError::Other("probe: failed to get cull command buffer".into())
            })?;
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
        super::fault_log::attach_fault_logger(&cull_cb, "reflection probe cull");
        cull_cb.commit();

        // Render command buffer: reads the ICB into this face.
        let render_cb = self.hw.command_queue.commandBuffer().ok_or_else(|| {
            RenderError::Other("probe: failed to get render command buffer".into())
        })?;
        self.encode_main_into_face(
            &render_cb,
            crate::metal::draw::main::FaceTargets {
                color_msaa: gpu.msaa_color.as_deref(),
                depth: &gpu.depth,
                resolve: &gpu.capture,
                resolve_slice: face,
            },
            crate::metal::draw::main::MainPassCamera {
                elapsed,
                vp,
                view,
                cam_pos: eye,
            },
            crate::metal::draw::main::GpuFrameBuffers {
                object_buffer: Some(&gpu.object_buffer),
                material_params: Some(&gpu.material_params),
                bindless_tex_args: Some(tex_args),
                deformed_skinned: gpu.deformed.as_ref(),
                counts,
            },
            crate::metal::draw::main::FacePass::PROBE,
        )?;
        super::fault_log::attach_fault_logger(&render_cb, "reflection probe face");
        if attach_done {
            let flag = Arc::clone(done);
            let handler = block2::RcBlock::new(
                move |_: NonNull<ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>| {
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
        Ok(())
    }

    // Commit one convolution step on its own command buffer, with no
    // `waitUntilCompleted`. Mip 0 is the clamped mirror mip plus the capture's
    // source pyramid; each later mip one GGX convolution, which reads the pyramid
    // and writes a mip nothing else touches, so single-queue FIFO ordering puts
    // every one of them after the pyramid build that produced their source.
    fn record_prefilter_mip(&self, bake: &PrefilteringBake, mip: u32) -> RenderResult<()> {
        let cmd_buf = self.hw.command_queue.commandBuffer().ok_or_else(|| {
            RenderError::Other("probe: failed to get convolution command buffer".into())
        })?;
        let label = if mip == 0 {
            self.encode_probe_pyramid(&cmd_buf, &bake.gpu, &PLAN)?;
            "reflection probe pyramid"
        } else {
            self.encode_probe_ggx_mip(&cmd_buf, &bake.gpu, &PLAN, mip)?;
            "reflection probe convolution"
        };
        super::fault_log::attach_fault_logger(&cmd_buf, label);
        cmd_buf.commit();
        Ok(())
    }
}

impl ProbeBakeDevice for MtlContext {
    type Capture = RenderingBake;
    type Prefilter = PrefilteringBake;
    type Frame<'f> = ProbeFrame<'f>;

    fn book(&mut self) -> &mut ProbeBook {
        &mut self.probe.book
    }

    // The capture renders through the bindless ICB (needs the GPU-driven static
    // path), and a world with no real geometry keeps the sky. Neither changes
    // after init.
    fn capture_supported(&self) -> bool {
        self.cull.bindless && !self.targets.geometry_less && self.probe.prefilter.is_some()
    }

    // Geometry may still be streaming in on the first frames: a zero cull would
    // start an empty capture.
    fn capture_ready(&self, _prefilter_in_flight: bool) -> bool {
        self.cull_count() > 0
    }

    fn reserve_cubes(&mut self, count: usize) -> RenderResult<()> {
        self.reserve_probe_cubes(&PLAN, count)
    }

    fn start_capture(
        &mut self,
        frame: &ProbeFrame<'_>,
        _index: usize,
        placement: ProbePlacement,
    ) -> RenderResult<RenderingBake> {
        self.start_probe_capture(frame, placement)
    }

    fn render_face(
        &mut self,
        frame: &ProbeFrame<'_>,
        capture: &mut RenderingBake,
        face: usize,
    ) -> RenderResult<()> {
        let tex_args = frame
            .tex_args
            .ok_or_else(|| RenderError::Other("probe: no bindless texture args".into()))?;
        self.record_probe_face(capture, face, tex_args)
    }

    // The completion handler on the last face flags it, and single-queue FIFO
    // completion means every face is done when that one is.
    fn capture_retired(&self, capture: &RenderingBake) -> bool {
        capture.done.load(Ordering::Acquire)
    }

    // Everything but the capture cube drops here -- safe, the GPU is done with all
    // of it -- and the convolution takes the cube it reads and the cube of the
    // array it writes.
    fn begin_prefilter(
        &mut self,
        index: usize,
        capture: RenderingBake,
    ) -> RenderResult<PrefilteringBake> {
        let gpu = PrefilterGpu::new(
            capture.gpu.capture,
            self.probe.cubes.texture(),
            index,
            &PLAN,
        )?;
        Ok(PrefilteringBake { gpu })
    }

    fn prefilter_mip(&mut self, prefilter: &mut PrefilteringBake, mip: u32) -> RenderResult<()> {
        self.record_prefilter_mip(prefilter, mip)
    }

    // Nothing to wait for: the convolution dispatches and every frame that will
    // sample the cube share one queue, so FIFO completion already orders the
    // writes before the reads, and Metal owns the command buffers' lifetime
    // rather than this bake.
    fn prefilter_retired(&self, _prefilter: &PrefilteringBake) -> bool {
        true
    }

    // The views were the bake's only hold on the cube; the array keeps it.
    fn finish_prefilter(&mut self, prefilter: PrefilteringBake) {
        drop(prefilter);
    }

    // Park both slots' GPU resources behind the frames-in-flight fence instead of
    // dropping them: their command buffers may still be reading the reserved-slot
    // buffers, the capture cube or the cube being convolved, and each payload
    // carries a heap-placed cube whose memory would otherwise be handed to the
    // next allocation.
    fn abandon(&mut self, capture: Option<RenderingBake>, prefilter: Option<PrefilteringBake>) {
        let frame = self.frame_ring_index;
        if let Some(bake) = capture {
            self.probe
                .retire_pool
                .push(frame, RetiredBake::Capture(bake.gpu));
        }
        if let Some(bake) = prefilter {
            self.probe
                .retire_pool
                .push(frame, RetiredBake::Prefilter(bake.gpu));
        }
    }
}

// Multisample HDR color face: RGBA16Float, render-target only -- matches the
// main pipeline's attachment format + sample count so `self.pipeline_state`
// binds. Built only when the world resolved to more than one sample.
fn make_face_color(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    size: u32,
    sample_count: u32,
) -> RenderResult<Retained<ProtocolObject<dyn MTLTexture>>> {
    let desc = TextureDesc {
        kind: MTLTextureType::Type2DMultisample,
        format: MTLPixelFormat::RGBA16Float,
        width: size as usize,
        height: size as usize,
        sample_count: sample_count as usize,
        usage: MTLTextureUsage::RenderTarget,
        ..Default::default()
    }
    .build();
    device
        .newTextureWithDescriptor(&desc)
        .ok_or_else(|| allocation_failed("probe color face"))
}

// Depth face: Depth32Float, render-target only, at the main pipelines' sample
// count. Cleared per face and discarded -- the probe consumes only the color.
fn make_face_depth(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    size: u32,
    sample_count: u32,
) -> RenderResult<Retained<ProtocolObject<dyn MTLTexture>>> {
    let desc = TextureDesc {
        kind: if sample_count > 1 {
            MTLTextureType::Type2DMultisample
        } else {
            MTLTextureType::Type2D
        },
        format: MTLPixelFormat::Depth32Float,
        width: size as usize,
        height: size as usize,
        sample_count: sample_count as usize,
        usage: MTLTextureUsage::RenderTarget,
        ..Default::default()
    }
    .build();
    device
        .newTextureWithDescriptor(&desc)
        .ok_or_else(|| allocation_failed("probe depth face"))
}
