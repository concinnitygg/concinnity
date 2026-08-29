// src/metal/particle.rs
//
// GPU-compute particle system on Metal. Each `ParticleEmitter` declared in
// the world produces one persistent `ParticleEmitterGpuState` carrying a pool
// of `Particle` slots and an atomic spawn-counter buffer. Each frame the
// renderer:
//
//   1. Computes the per-emitter spawn budget CPU-side (a fractional
//      accumulator drives integer particle spawns per dispatch).
//   2. Writes that budget into this frame's slot of the atomic counter
//      buffer.
//   3. Dispatches the `particle_simulate` compute kernel to age + integrate +
//      respawn the pool.
//   4. Dispatches the `particle_vertex`/`particle_fragment` render pipeline
//      with `instance_count = max_particles`, drawing one camera-facing
//      billboard quad per live particle.
//
// The render pass alpha-blends into `hdr_resolve` after the volumetric fog
// pass and before SSR, so particles appear in screen-space reflections and
// are temporally stabilised by TAA.
#![deny(unsafe_op_in_unsafe_fn)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlendFactor, MTLBuffer, MTLCommandBuffer as _, MTLComputeCommandEncoder as _,
    MTLComputePassDescriptor, MTLComputePipelineState, MTLDevice as _, MTLLibrary as _,
    MTLLoadAction, MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder as _,
    MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLRenderPipelineState,
    MTLResourceOptions, MTLSamplerAddressMode, MTLSamplerDescriptor, MTLSamplerMinMagFilter,
    MTLSamplerState, MTLSize, MTLStoreAction,
};

use crate::gfx::particles::{ParticleEmitterRecord, ParticleSpawnState};

use super::context::MtlContext;
use super::encode::{ComputeEncode, RenderEncode};
use super::pipeline::ns_str;
use super::scoped_encoder::ScopedEncoder;
// GPU-free repr(C) structs; live in `core::render` so their layout tests
// count toward coverage. Re-exported so this file's existing paths are unchanged.
use concinnity_core::render::uniforms::GpuParticle;
use concinnity_core::render::uniforms::ParticleView;

// Byte stride between an emitter's per-frame spawn-counter slots. The counter
// itself is one `u32`; the padding buys the 256-byte buffer-offset alignment
// `setBuffer:offset:atIndex:` requires on every Metal GPU family.
const SPAWN_COUNTER_STRIDE: usize = 256;

// Byte offset of spawn-counter slot `slot` inside an emitter's counter buffer.
fn spawn_counter_offset(slot: usize) -> usize {
    slot * SPAWN_COUNTER_STRIDE
}

// Size of an emitter's spawn-counter buffer: one slot per frame in flight.
fn spawn_counter_bytes(frames_in_flight: usize) -> usize {
    frames_in_flight.max(1) * SPAWN_COUNTER_STRIDE
}

// Slot the next frame writes. Rotating over the frames-in-flight depth is what
// makes the CPU-side reset safe: the frame-pacing semaphore has already retired
// the frame that last used this slot, so no in-flight `particle_simulate` is
// still decrementing it. A depth of 1 pins every frame to slot 0, which is
// equally safe -- the CPU waits on the previous frame's completion there.
fn next_counter_slot(slot: usize, frames_in_flight: usize) -> usize {
    (slot + 1) % frames_in_flight.max(1)
}

// Per-emitter persistent GPU state. The pool buffer lives in shared storage
// so the CPU can zero-init it once; the counter buffer's slot for the frame
// being built is rewritten with that frame's integer spawn budget.
pub(super) struct ParticleEmitterGpuState {
    // Particle pool: `record.max_particles` slots of `GpuParticle`.
    pub pool: Retained<ProtocolObject<dyn MTLBuffer>>,
    // One `u32` atomic counter per frame in flight, `SPAWN_COUNTER_STRIDE`
    // apart. The compute kernel decrements the frame's slot as threads claim
    // spawn slots; the CPU resets that slot to `spawn_budget` before the
    // dispatch. Slots rotate so the reset never lands in a counter an
    // in-flight dispatch is still claiming against.
    pub spawn_counter: Retained<ProtocolObject<dyn MTLBuffer>>,
    // Carry-over spawn fraction. Combined with `dt` and the emitter's
    // `spawn_rate` to produce the integer spawn budget for each dispatch.
    pub spawn_state: ParticleSpawnState,
}

// Pair of pipelines driving the particle system: the compute kernel that
// ages + integrates + respawns the pool, and the render pipeline that draws
// each live particle as a camera-facing billboard quad. Built only when the
// world declared at least one `ParticleEmitter`.
pub(super) struct ParticlePipelines {
    pub simulate: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub render: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    pub sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
}

// All particle-system state grouped into one feature unit: the per-emitter
// records (with their tombstone free-list), the parallel per-emitter GPU
// pools, the shared compute + render pipelines, and the per-frame timing
// bookkeeping. `records` and `emitter_state` are parallel: the dispatch
// loop walks both in lockstep, skipping `None` pairs. `pipelines` is built
// lazily at init (≥1 declared emitter) or on the first runtime
// [`MtlContext::add_emitter`].
pub(crate) struct ParticleState {
    // One slot per emitter; `None` slots are tombstones from
    // [`MtlContext::remove_emitter`], reused by the next add via `free_slots`.
    pub records: Vec<Option<ParticleEmitterRecord>>,
    // Per-emitter persistent GPU state, parallel to `records`; `None` matches
    // a tombstoned record.
    pub emitter_state: Vec<Option<ParticleEmitterGpuState>>,
    pub free_slots: Vec<usize>,
    pub pipelines: Option<ParticlePipelines>,
    // Last frame's `elapsed`; the diff drives spawn budgets + integration.
    pub last_elapsed: f32,
    // Frame counter mixed into the compute kernel's per-thread RNG seed.
    pub frame_index: u32,
    // Spawn-counter slot the last prepared frame wrote; advanced by
    // `next_counter_slot` once per prepared frame.
    pub counter_slot: usize,
}

// The per-frame particle inputs `prepare_particle_pass` derives on `&mut self`
// for the read-only `encode_particles` to consume.
pub(in crate::metal) struct ParticleFrame {
    // Seconds since the previous prepared frame; drives ageing + integration.
    pub dt: f32,
    // Monotonic frame counter, mixed into the kernel's per-thread RNG seed.
    pub frame_index: u32,
    // Spawn-counter slot this frame's budgets were written to. The compute
    // dispatch binds each emitter's counter at this slot's byte offset.
    pub counter_slot: usize,
    // Integer spawn budget per emitter slot, parallel to `records`.
    pub spawn_budgets: Vec<u32>,
}

impl MtlContext {
    // Encode the per-emitter compute + render passes. A no-op when no
    // emitters are declared; `record.visible` and `max_particles == 0`
    // filtering is done at `build_particle_records` time, so any emitter
    // that reached this point is drawn.
    //
    // `elapsed` is the same value the rest of the frame already computed:
    // the previous-frame snapshot lives in `particle.last_elapsed`, and the
    // diff is the frame `dt` driving spawn rates + integration.
    // pub(in crate::metal) so the render-graph executor in
    // metal/graph_exec.rs can dispatch this pass from a CompiledGraph.
    // Bundles ParticlesSim (compute) + ParticlesDraw (render); the
    // graph only adds a node for `PassId::ParticlesDraw`, but the
    // bundled sim sub-pass keeps its own per-pass timing slot via the
    // inline `diagnostics.pass_timing.attach_compute` call below.
    // Mutate the per-frame particle state (dt against
    // `particle.last_elapsed`, monotonic `particle.frame_index`,
    // per-emitter spawn budgets) and write each emitter's spawn-counter
    // slot in place. Returns the [`ParticleFrame`] the read-only
    // `encode_particles` then consumes. Split out so `encode_particles` can
    // take `&self` and run on a parallel-recording worker; the mutating
    // prelude stays on the frame's main `&mut self` path inside
    // `execute_graph`, which runs it exactly once per paced frame -- the
    // counter-slot rotation depends on that.
    pub(in crate::metal) fn prepare_particle_pass(
        &mut self,
        elapsed: f32,
    ) -> Option<ParticleFrame> {
        self.particle.pipelines.as_ref()?;
        if self.particle.records.is_empty() || self.particle.emitter_state.is_empty() {
            return None;
        }
        let dt = (elapsed - self.particle.last_elapsed).max(0.0);
        self.particle.last_elapsed = elapsed;
        self.particle.frame_index = self.particle.frame_index.wrapping_add(1);
        let frame_index = self.particle.frame_index;
        let counter_slot = next_counter_slot(self.particle.counter_slot, self.frames_in_flight);
        self.particle.counter_slot = counter_slot;
        let offset = spawn_counter_offset(counter_slot);
        let mut budgets = Vec::with_capacity(self.particle.records.len());
        for (rec_slot, gpu_slot) in self
            .particle
            .records
            .iter()
            .zip(self.particle.emitter_state.iter_mut())
        {
            let budget = match (rec_slot.as_ref(), gpu_slot.as_mut()) {
                (Some(rec), Some(gpu)) => {
                    let spawn = gpu
                        .spawn_state
                        .take_budget(dt, rec.spawn_rate, rec.max_particles);
                    // Reset this frame's counter slot to its budget. Shared
                    // storage means the kernel sees the write immediately;
                    // the slot rotation is what keeps it clear of the
                    // dispatches still in flight.
                    // SAFETY: `spawn_counter` is a shared-storage buffer of
                    // `spawn_counter_bytes(frames_in_flight)`, so `contents()` is a live CPU
                    // mapping of it and `offset` -- a slot index below that depth, times the
                    // stride -- keeps a whole `u32` in bounds and 4-byte aligned.
                    unsafe {
                        let dst = gpu.spawn_counter.contents().as_ptr().add(offset) as *mut u32;
                        dst.write(spawn);
                    }
                    spawn
                }
                _ => 0,
            };
            budgets.push(budget);
        }
        Some(ParticleFrame {
            dt,
            frame_index,
            counter_slot,
            spawn_budgets: budgets,
        })
    }

    pub(in crate::metal) fn encode_particles(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        frame: &ParticleFrame,
        vp: [[f32; 4]; 4],
        frustum: &crate::gfx::frustum::Frustum,
    ) -> Result<u32, String> {
        let Some(pipelines) = self.particle.pipelines.as_ref() else {
            return Ok(0);
        };
        if self.particle.records.is_empty() || self.particle.emitter_state.is_empty() {
            return Ok(0);
        }
        let ParticleFrame {
            dt,
            frame_index,
            counter_slot,
            spawn_budgets,
        } = frame;
        let (dt, frame_index) = (*dt, *frame_index);
        let counter_offset = spawn_counter_offset(*counter_slot);
        let last_tex = self.textures.len().saturating_sub(1);

        // Visibility-cull per emitter for the *render* pass only. The compute
        // simulation still ticks every pool so particles spawn / age / die
        // while the camera looks away: that way the emitter is in a
        // realistic mid-life state the moment the camera turns back. The
        // compute cost is per-slot work in a single threadgroup, so leaving
        // it un-culled is cheap. Tombstoned (None) slots are always invisible.
        let visible: Vec<bool> = self
            .particle
            .records
            .iter()
            .map(|slot| match slot {
                Some(r) => {
                    let (mn, mx) = r.aabb();
                    frustum.intersects_aabb(mn, mx)
                }
                None => false,
            })
            .collect();

        // Camera basis for camera-facing billboards: rows 0 and 1 of the view
        // matrix's 3×3 are the world-space right and up vectors (the view
        // matrix is column-major, so we read those rows out element-wise).
        let v = self.view.matrix;
        let cam_right = [v[0][0], v[1][0], v[2][0]];
        let cam_up = [v[0][1], v[1][1], v[2][1]];
        let view = ParticleView {
            vp,
            cam_right,
            _pad0: 0.0,
            cam_up,
            _pad1: 0.0,
        };

        // Compute: age + integrate + respawn each pool in turn. One
        // dispatch per emitter; cheap enough to not bother packing them.
        {
            let sim_desc = MTLComputePassDescriptor::new();
            if let Some(t) = &self.diagnostics.pass_timing {
                t.attach_compute(&sim_desc, super::pass_timing::PassId::ParticlesSim);
            }
            // Guard drops at the end of this block, ending the compute pass
            // before the render encoder below opens.
            let enc = ScopedEncoder::new(
                cmd_buf
                    .computeCommandEncoderWithDescriptor(&sim_desc)
                    .ok_or("failed to get particle compute encoder")?,
                "particles: simulate",
            );
            enc.set_pipeline(&pipelines.simulate);
            for (i, (rec_slot, gpu_slot)) in self
                .particle
                .records
                .iter()
                .zip(self.particle.emitter_state.iter())
                .enumerate()
            {
                let (rec, gpu) = match (rec_slot.as_ref(), gpu_slot.as_ref()) {
                    (Some(r), Some(g)) => (r, g),
                    _ => continue,
                };
                let spawn_budget = spawn_budgets.get(i).copied().unwrap_or(0);
                let params = rec.params(dt, spawn_budget, frame_index);
                enc.set_buffer(gpu.pool.as_ref(), 0, 0);
                enc.set_buffer(gpu.spawn_counter.as_ref(), counter_offset, 1);
                enc.set_value(&params, 2);
                let grid = MTLSize {
                    width: rec.max_particles as usize,
                    height: 1,
                    depth: 1,
                };
                // 64-thread groups: a multiple of the SIMD width on every Apple
                // GPU since A11 and small enough that a thin pool still
                // dispatches efficiently.
                let tg = MTLSize {
                    width: 64,
                    height: 1,
                    depth: 1,
                };
                enc.dispatchThreads_threadsPerThreadgroup(grid, tg);
            }
        }

        // Render: one alpha-blended quad per live particle, drawn into
        // `hdr_resolve`. Caller has already ended the previous render
        // pass (fog), so we open a fresh Load/Store pass here. When every
        // emitter culls out we skip the render encoder entirely.
        if !visible.iter().any(|v| *v) {
            return Ok(0);
        }
        let pass_desc = MTLRenderPassDescriptor::new();
        // SAFETY: plain descriptor property setters; the subscripted slots are ones this descriptor
        // declares.
        unsafe {
            let ca = pass_desc.colorAttachments().objectAtIndexedSubscript(0);
            ca.setTexture(Some(self.hdr_targets.hdr_resolve.as_ref()));
            ca.setLoadAction(MTLLoadAction::Load);
            ca.setStoreAction(MTLStoreAction::Store);
        }

        if let Some(t) = &self.diagnostics.pass_timing {
            t.attach_render(&pass_desc, super::pass_timing::PassId::ParticlesDraw);
        }
        let enc = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&pass_desc)
                .ok_or("failed to get particle render encoder")?,
            "particles: draw",
        );
        enc.set_pipeline(&pipelines.render);
        enc.set_vertex_value(&view, 1);
        enc.set_fragment_sampler(&pipelines.sampler, 0);

        let mut draw_calls: u32 = 0;
        for (i, (rec_slot, gpu_slot)) in self
            .particle
            .records
            .iter()
            .zip(self.particle.emitter_state.iter())
            .enumerate()
        {
            if !visible[i] {
                continue;
            }
            let (rec, gpu) = match (rec_slot.as_ref(), gpu_slot.as_ref()) {
                (Some(r), Some(g)) => (r, g),
                _ => continue,
            };
            // Spawn budget and frame seed only matter to the compute kernel,
            // but we share the uniform layout so the render path passes its
            // own zero-budget copy. `dt` is irrelevant to the vertex shader
            // (it reads `age` / `lifetime` straight from the pool).
            let params = rec.params(0.0, 0, frame_index);
            let slot = rec.texture_slot.min(last_tex);
            enc.set_vertex_buffer(gpu.pool.as_ref(), 0, 0);
            enc.set_vertex_value(&params, 2);
            enc.set_fragment_texture(self.textures[slot].as_ref(), 0);
            // SAFETY: the four strip vertices are generated from `[[vertex_id]]` in the shader.
            unsafe {
                enc.drawPrimitives_vertexStart_vertexCount_instanceCount(
                    MTLPrimitiveType::TriangleStrip,
                    0,
                    4,
                    rec.max_particles as usize,
                );
            }
            draw_calls += 1;
        }

        Ok(draw_calls)
    }
}

// Build the particle compute + render pipelines plus the shared sampler.
// Returned only when the world declares at least one `ParticleEmitter`.
pub(super) fn build_particle_pipelines(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> Result<ParticlePipelines, String> {
    // Compute kernel, from `particle_simulate.slang`. The render pair below
    // splices the same `{PARTICLE_TYPES}` fragment, so both halves stride one
    // declaration of the pool record and the per-emitter uniform.
    let sim_lib = super::slang_shaders::PARTICLE_SIMULATE.library(device, hot_reload)?;
    let sim_fn = sim_lib
        .newFunctionWithName(&ns_str("particle_simulate"))
        .ok_or("particle_simulate not found")?;
    let simulate = device
        .newComputePipelineStateWithFunction_error(&sim_fn)
        .map_err(|e| format!("failed to create particle_simulate pipeline: {:?}", e))?;

    // Render pipeline. No vertex descriptor: the vertex shader reads from the
    // particle pool storage buffer directly via `[[vertex_id]]` + `[[instance_id]]`.
    // Each entry compiles to its own metallib, so the two stages come from
    // separate libraries and pair by semantic.
    let vert_fn = super::slang_shaders::entry_function(
        device,
        &super::slang_shaders::PARTICLE_VERT,
        hot_reload,
    )?;
    let frag_fn = super::slang_shaders::entry_function(
        device,
        &super::slang_shaders::PARTICLE_FRAG,
        hot_reload,
    )?;
    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexFunction(Some(&vert_fn));
    desc.setFragmentFunction(Some(&frag_fn));
    desc.setRasterSampleCount(1);
    // SAFETY: plain descriptor property setters; the subscripted slots are ones this descriptor
    // declares.
    unsafe {
        let ca = desc.colorAttachments().objectAtIndexedSubscript(0);
        ca.setPixelFormat(MTLPixelFormat::RGBA16Float);
        ca.setBlendingEnabled(true);
        ca.setSourceRGBBlendFactor(MTLBlendFactor::SourceAlpha);
        ca.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
        ca.setSourceAlphaBlendFactor(MTLBlendFactor::SourceAlpha);
        ca.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    }
    let render = device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .map_err(|e| format!("failed to create particle render pipeline: {:?}", e))?;

    // Sampler: linear-clamp, same envelope the decal pass uses.
    let sampler = {
        let sdesc = MTLSamplerDescriptor::new();
        sdesc.setMinFilter(MTLSamplerMinMagFilter::Linear);
        sdesc.setMagFilter(MTLSamplerMinMagFilter::Linear);
        sdesc.setSAddressMode(MTLSamplerAddressMode::ClampToEdge);
        sdesc.setTAddressMode(MTLSamplerAddressMode::ClampToEdge);
        device
            .newSamplerStateWithDescriptor(&sdesc)
            .ok_or("failed to create particle sampler state")?
    };

    Ok(ParticlePipelines {
        simulate,
        render,
        sampler,
    })
}

// Allocate the per-emitter GPU state for one record: a zero-initialised
// particle pool plus an atomic counter buffer holding one `u32` slot per frame
// in flight. Both buffers use shared storage so the CPU can reset the spawn
// counter each frame without a staging copy.
pub(super) fn build_emitter_gpu_state(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    record: &ParticleEmitterRecord,
    frames_in_flight: usize,
) -> Result<ParticleEmitterGpuState, String> {
    let slots = record.max_particles as usize;
    let pool_bytes = slots * std::mem::size_of::<GpuParticle>();
    let pool = device
        .newBufferWithLength_options(pool_bytes, MTLResourceOptions::StorageModeShared)
        .ok_or("failed to allocate particle pool buffer")?;
    // Zero-init: every slot starts dead (`lifetime = 0`).
    // SAFETY: `pool` was just allocated with `pool_bytes` bytes of shared storage, so `contents()`
    // is a live CPU mapping of exactly that many bytes.
    unsafe {
        let dst = pool.contents().as_ptr() as *mut u8;
        std::ptr::write_bytes(dst, 0, pool_bytes);
    }

    let counter_bytes = spawn_counter_bytes(frames_in_flight);
    let spawn_counter = device
        .newBufferWithLength_options(counter_bytes, MTLResourceOptions::StorageModeShared)
        .ok_or("failed to allocate particle spawn counter")?;
    // Zero every slot: a frame that skips its dispatch leaves its slot at
    // whatever the last dispatch decremented it to.
    // SAFETY: `spawn_counter` was just allocated with `counter_bytes` bytes of shared storage, so
    // `contents()` is a live CPU mapping of exactly that many bytes.
    unsafe {
        let dst = spawn_counter.contents().as_ptr() as *mut u8;
        std::ptr::write_bytes(dst, 0, counter_bytes);
    }

    Ok(ParticleEmitterGpuState {
        pool,
        spawn_counter,
        spawn_state: ParticleSpawnState::default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_slots_are_stride_aligned_and_distinct() {
        let offsets: Vec<usize> = (0..3).map(spawn_counter_offset).collect();
        assert_eq!(offsets, vec![0, 256, 512]);
        // Every slot must clear the 256-byte buffer-offset alignment Metal
        // requires and leave a whole `u32` inside the allocation.
        for (slot, offset) in offsets.iter().enumerate() {
            assert_eq!(offset % SPAWN_COUNTER_STRIDE, 0, "slot {slot} misaligned");
            assert!(offset + std::mem::size_of::<u32>() <= spawn_counter_bytes(3));
        }
    }

    #[test]
    fn counter_buffer_holds_one_slot_per_frame_in_flight() {
        assert_eq!(spawn_counter_bytes(3), 3 * SPAWN_COUNTER_STRIDE);
        assert_eq!(spawn_counter_bytes(1), SPAWN_COUNTER_STRIDE);
        // A zero depth would otherwise allocate nothing and divide by zero in
        // `next_counter_slot`; both clamp to a single slot.
        assert_eq!(spawn_counter_bytes(0), SPAWN_COUNTER_STRIDE);
        assert_eq!(next_counter_slot(0, 0), 0);
    }

    #[test]
    fn counter_slot_cycles_over_the_frames_in_flight_depth() {
        let depth = 3;
        let mut slot = 0;
        let mut seen = Vec::new();
        for _ in 0..depth {
            slot = next_counter_slot(slot, depth);
            assert!(slot < depth, "slot {slot} outside the allocated depth");
            seen.push(slot);
        }
        // And the cycle repeats rather than drifting.
        let first = seen[0];
        assert_eq!(next_counter_slot(slot, depth), first);
        // A full cycle visits every slot exactly once, so a frame's reset is
        // `depth` frames removed from the last dispatch that read the slot.
        seen.sort_unstable();
        assert_eq!(seen, (0..depth).collect::<Vec<_>>());
    }

    #[test]
    fn single_frame_in_flight_pins_slot_zero() {
        // Depth 1 means the CPU already waits on the previous frame's
        // completion, so reusing one slot cannot race.
        assert_eq!(next_counter_slot(0, 1), 0);
        assert_eq!(next_counter_slot(5, 1), 0);
    }
}
