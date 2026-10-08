//! GPU-compute particle system on Metal. Each `ParticleEmitter` declared in
//! the world produces one persistent `ParticleEmitterGpuState` carrying a pool
//! of `Particle` slots. Each frame the renderer:
//!
//!   1. Computes the per-emitter spawn run CPU-side (a fractional accumulator
//!      drives integer particle spawns per dispatch, into the pool slots a ring
//!      cursor names).
//!   2. Dispatches the `particle_simulate` compute kernel to age + integrate +
//!      respawn the pool.
//!   3. Dispatches the `particle_vertex`/`particle_fragment` render pipeline
//!      with `instance_count = max_particles`, drawing one camera-facing
//!      billboard quad per live particle.
//!
//! The render pass alpha-blends into `hdr_resolve` after the volumetric fog
//! pass and before SSR, so particles appear in screen-space reflections and
//! are temporally stabilized by TAA. It attaches no depth buffer; the fragment
//! tests the resolved scene depth itself, so opaque geometry hides a sprite.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::gfx::frustum::Frustum;
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::particles::{
    ParticleEmitterRecord, ParticleSpawnState, ParticleSpawns, spawn_seed,
};
use concinnity_core::render::reactive_mask::ReactiveWrite;
use concinnity_core::render::uniforms::ParticleView;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLBlendFactor, MTLBuffer, MTLCommandBuffer as _, MTLComputeCommandEncoder as _,
    MTLComputePassDescriptor, MTLComputePipelineState, MTLDevice as _, MTLLoadAction,
    MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder as _, MTLRenderPassDescriptor,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLResourceOptions, MTLSamplerAddressMode,
    MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSamplerState, MTLSize, MTLStoreAction,
};
// GPU-free repr(C) structs; live in `core::render` so their layout tests
// count toward coverage. Re-exported so this file's existing paths are unchanged.
use concinnity_core::render::uniforms::GpuParticle;

use super::builtin_shaders::compute_pipeline;
use super::context::MtlContext;
use super::encode::{ComputeEncode, RenderEncode};
use super::error::allocation_failed;
use super::scoped_encoder::ScopedEncoder;

// Per-emitter persistent GPU state. The pool buffer lives in shared storage
// so the CPU can zero-init it once.
pub(super) struct ParticleEmitterGpuState {
    // Particle pool: `record.max_particles` slots of `GpuParticle`.
    pub pool: Retained<ProtocolObject<dyn MTLBuffer>>,
    // Carry-over spawn fraction and ring cursor, which turn `dt` and the
    // emitter's `spawn_rate` into each dispatch's spawn run.
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
    // Frame counter mixed into the compute kernel's per-spawn RNG seed.
    pub frame_index: u32,
}

// The per-frame particle inputs `prepare_particle_pass` derives on `&mut self`
// for the read-only encode halves to consume.
pub(in crate::metal) struct ParticleFrame {
    // Seconds since the previous prepared frame; drives ageing + integration.
    pub dt: f32,
    // Monotonic frame counter, mixed into the kernel's per-spawn RNG seed.
    pub frame_index: u32,
    // Spawn run per emitter slot, parallel to `records`.
    pub spawns: Vec<ParticleSpawns>,
}

impl MtlContext {
    // Mutate the per-frame particle state (dt against
    // `particle.last_elapsed`, monotonic `particle.frame_index`,
    // per-emitter spawn runs). Returns the [`ParticleFrame`] the read-only
    // `encode_particles_sim` and `encode_particles_draw` then consume. Split out
    // so both halves take `&self` and run on parallel-recording workers; the
    // mutating prelude stays on the frame's main `&mut self` path inside
    // `execute_graph`, which runs it exactly once per paced frame.
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
        let spawns = self
            .particle
            .records
            .iter()
            .zip(self.particle.emitter_state.iter_mut())
            .map(
                |(rec_slot, gpu_slot)| match (rec_slot.as_ref(), gpu_slot.as_mut()) {
                    (Some(rec), Some(gpu)) => gpu.spawn_state.take_spawns(dt, rec),
                    _ => ParticleSpawns::default(),
                },
            )
            .collect();
        Some(ParticleFrame {
            dt,
            frame_index,
            spawns,
        })
    }

    // Encode the `ParticlesSim` node: age + integrate + respawn every live
    // emitter's pool in place. One dispatch per emitter; cheap enough to not
    // bother packing them, and their resources are disjoint. A no-op when no
    // emitter has ever existed in this session or every slot is tombstoned.
    //
    // Records into its own command buffer, one per graph node. The pools it
    // writes are read only by the draw, and Metal's implicit hazard tracking
    // plus the executor's FIFO commit order is what makes the write visible
    // there. `frame` is the state `prepare_particle_pass` advanced on
    // `&mut self`; this method takes `&self` so it can record on a worker.
    pub(in crate::metal) fn encode_particles_sim(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        frame: &ParticleFrame,
    ) -> RenderResult<()> {
        let Some(pipelines) = self.particle.pipelines.as_ref() else {
            return Ok(());
        };
        if self.particle.records.is_empty() || self.particle.emitter_state.is_empty() {
            return Ok(());
        }
        let sim_desc = MTLComputePassDescriptor::new();
        if let Some(t) = &self.diagnostics.pass_timing {
            t.attach_compute(&sim_desc, super::pass_timing::PassId::ParticlesSim);
        }
        let enc = ScopedEncoder::new(
            cmd_buf
                .computeCommandEncoderWithDescriptor(&sim_desc)
                .ok_or_else(|| {
                    RenderError::Other("failed to get particle compute encoder".into())
                })?,
            ns_string!("particles: simulate"),
        );
        enc.set_pipeline(&pipelines.simulate);
        // Every live pool ticks, visible or not, so an off-screen emitter stays
        // in a realistic mid-life state for when the camera turns back. The cost
        // is per-slot work in a single threadgroup, so leaving it un-culled is
        // cheap.
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
            let spawns = frame.spawns.get(i).copied().unwrap_or_default();
            let params = rec.params(frame.dt, spawns, spawn_seed(frame.frame_index, i));
            enc.set_buffer(gpu.pool.as_ref(), 0, 0);
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
        Ok(())
    }

    // Encode the `ParticlesDraw` node: one alpha-blended camera-facing quad per
    // live particle, blend-written into `hdr_resolve`. The vertex stage reads
    // the pool `encode_particles_sim` wrote. Returns the draw-call count.
    pub(in crate::metal) fn encode_particles_draw(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        frame: &ParticleFrame,
        vp: [[f32; 4]; 4],
        frustum: &Frustum,
        reactive: ReactiveWrite,
    ) -> RenderResult<u32> {
        let Some(pipelines) = self.particle.pipelines.as_ref() else {
            return Ok(0);
        };
        if self.particle.records.is_empty() || self.particle.emitter_state.is_empty() {
            return Ok(0);
        }
        let frame_index = frame.frame_index;
        let last_tex = self.scene.textures.len().saturating_sub(1);

        // Visibility-cull per emitter, for the draw alone: the simulation above
        // ticked every pool. Tombstoned (None) slots are always invisible.
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
        let v = self.state.view.matrix;
        let cam_right = [v[0][0], v[1][0], v[2][0]];
        let cam_up = [v[0][1], v[1][1], v[2][1]];
        let view = ParticleView {
            vp,
            cam_right,
            _pad0: 0.0,
            cam_up,
            _pad1: 0.0,
        };

        // A fresh Load/Store pass: the prior render pass ended with its own
        // command buffer. When every emitter culls out we skip the encoder
        // entirely.
        if !visible.iter().any(|v| *v) {
            return Ok(0);
        }
        let pass_desc = MTLRenderPassDescriptor::new();
        // SAFETY: plain descriptor property setters; the subscripted slots are ones this descriptor
        // declares.
        unsafe {
            let ca = pass_desc.colorAttachments().objectAtIndexedSubscript(0);
            ca.setTexture(Some(self.targets.hdr.hdr_resolve.as_ref()));
            ca.setLoadAction(MTLLoadAction::Load);
            ca.setStoreAction(MTLStoreAction::Store);
        }
        self.attach_reactive_mask(&pass_desc, reactive);

        if let Some(t) = &self.diagnostics.pass_timing {
            t.attach_render(&pass_desc, super::pass_timing::PassId::ParticlesDraw);
        }
        let enc = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&pass_desc)
                .ok_or_else(|| {
                    RenderError::Other("failed to get particle render encoder".into())
                })?,
            ns_string!("particles: draw"),
        );
        enc.set_pipeline(&pipelines.render);
        enc.set_vertex_value(&view, 1);
        enc.set_fragment_sampler(&pipelines.sampler, 0);
        // Resolved scene depth at texture(2) for the manual depth test.
        enc.set_fragment_texture(self.targets.hdr.depth_resolve.as_ref(), 2);

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
            // The spawn run and frame seed only matter to the compute kernel,
            // but we share the uniform layout so the render path passes its
            // own copy with no spawns. `dt` is irrelevant to the vertex shader
            // (it reads `age` / `lifetime` straight from the pool).
            let params = rec.params(0.0, ParticleSpawns::default(), frame_index);
            let slot = rec.texture_slot.min(last_tex);
            enc.set_vertex_buffer(gpu.pool.as_ref(), 0, 0);
            enc.set_vertex_value(&params, 2);
            enc.set_fragment_texture(self.scene.textures[slot].as_ref(), 0);
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
) -> RenderResult<ParticlePipelines> {
    // Compute kernel, from `particle_simulate.hlsl`. The render pair below
    // splices the same `{PARTICLE_TYPES}` fragment, so both halves stride one
    // declaration of the pool record and the per-emitter uniform.
    let simulate = compute_pipeline(
        device,
        &super::builtin_shaders::PARTICLE_SIMULATE,
        hot_reload,
    )?;

    // Render pipeline. No vertex descriptor: the vertex shader reads from the
    // particle pool storage buffer directly via `[[vertex_id]]` + `[[instance_id]]`.
    // Each entry compiles to its own metallib, so the two stages come from
    // separate libraries and pair by semantic.
    let vert_fn = super::builtin_shaders::entry_function(
        device,
        &super::builtin_shaders::PARTICLE_VERT,
        hot_reload,
    )?;
    let frag_fn = super::builtin_shaders::entry_function(
        device,
        &super::builtin_shaders::PARTICLE_FRAG,
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
    super::reactive_mask::declare_target(&desc);
    let render = device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .map_err(|e| RenderError::ShaderCompile(format!("particle render pipeline: {e:?}")))?;

    // Sampler: linear-clamp, same envelope the decal pass uses.
    let sampler = {
        let sdesc = MTLSamplerDescriptor::new();
        sdesc.setMinFilter(MTLSamplerMinMagFilter::Linear);
        sdesc.setMagFilter(MTLSamplerMinMagFilter::Linear);
        sdesc.setSAddressMode(MTLSamplerAddressMode::ClampToEdge);
        sdesc.setTAddressMode(MTLSamplerAddressMode::ClampToEdge);
        device
            .newSamplerStateWithDescriptor(&sdesc)
            .ok_or_else(|| RenderError::Other("failed to create particle sampler state".into()))?
    };

    Ok(ParticlePipelines {
        simulate,
        render,
        sampler,
    })
}

// Allocate the per-emitter GPU state for one record: a zero-initialized
// particle pool in shared storage.
pub(super) fn build_emitter_gpu_state(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    record: &ParticleEmitterRecord,
) -> RenderResult<ParticleEmitterGpuState> {
    let slots = record.max_particles as usize;
    let pool_bytes = slots * std::mem::size_of::<GpuParticle>();
    let pool = device
        .newBufferWithLength_options(pool_bytes, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| allocation_failed("the particle pool buffer"))?;
    // Zero-init: every slot starts dead (`lifetime = 0`).
    // SAFETY: `pool` was just allocated with `pool_bytes` bytes of shared storage, so `contents()`
    // is a live CPU mapping of exactly that many bytes.
    unsafe {
        let dst = pool.contents().as_ptr() as *mut u8;
        std::ptr::write_bytes(dst, 0, pool_bytes);
    }

    Ok(ParticleEmitterGpuState {
        pool,
        spawn_state: ParticleSpawnState::default(),
    })
}
