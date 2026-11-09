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

use concinnity_engine::gfx::frustum::Frustum;
use concinnity_engine::protocol::ParticleBurst;
use concinnity_engine::render::backend::EmitterPass;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::particles::burst::PARTICLE_FRAME_ROWS;
use concinnity_engine::render::particles::{EmitterGpu, ParticleEmitterRecord, ParticleFrame};
use concinnity_engine::render::shader::programs::abi::particle::SIMULATE_GROUP;
use concinnity_engine::render::shader::programs::abi::particle::metal::{draw, simulate};
use concinnity_engine::render::temporal::reactive_mask::ReactiveWrite;
use concinnity_engine::render::uniforms::ParticleView;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLBlendFactor, MTLBuffer, MTLCommandBuffer as _, MTLComputePassDescriptor,
    MTLComputePipelineState, MTLDevice as _, MTLLoadAction, MTLPixelFormat, MTLPrimitiveType,
    MTLRenderCommandEncoder as _, MTLRenderPassDescriptor, MTLRenderPipelineDescriptor,
    MTLRenderPipelineState, MTLResourceOptions, MTLSamplerAddressMode, MTLSamplerDescriptor,
    MTLSamplerMinMagFilter, MTLSamplerState, MTLStoreAction,
};
// GPU-free repr(C) structs; live in `core::render` so their layout tests
// count toward coverage. Re-exported so this file's existing paths are unchanged.
use concinnity_engine::render::uniforms::GpuParticle;

use super::builtin_shaders::compute_pipeline;
use super::context::MtlContext;
use super::dispatch::dispatch_threads;
use super::encode::{ComputeEncode, RenderEncode};
use super::error::allocation_failed;
use super::scoped_encoder::ScopedEncoder;

// Per-emitter persistent GPU state: the particle pool, `record.max_particles`
// slots of `GpuParticle` in shared storage so the CPU can zero-init it once.
pub(super) struct ParticleEmitterGpuState {
    pub pool: Retained<ProtocolObject<dyn MTLBuffer>>,
}

// Pair of pipelines driving the particle system: the compute kernel that
// ages + integrates + respawns the pool, and the render pipeline that draws
// each live particle as a camera-facing billboard quad. Built only when the
// world declared at least one `ParticleEmitter`.
pub(super) struct ParticlePipelines {
    pub simulate: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub render: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    // The same for a lit emitter.
    pub render_lit: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    pub sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
    // The frame's rows every simulation reads, one buffer per ring slot.
    pub rows: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

// Each emitter's GPU pool, by the id the scene's particle system hands out,
// and the shared compute + render pipelines, built on the first
// `RenderBackend::add_emitter`, authored emitters included.
#[derive(Default)]
pub(crate) struct ParticleState {
    pub gpu: EmitterGpu<ParticleEmitterGpuState>,
    pub pipelines: Option<ParticlePipelines>,
    // The ring slot this frame's rows went to.
    pub rows_slot: usize,
}

impl MtlContext {
    // Advance the particle clock and every emitter's spawn run, once per paced
    // frame on the main `&mut self` path, so the `&self` sim and draw halves can
    // record on parallel workers against one frame.
    pub(in crate::metal) fn prepare_particle_pass(
        &mut self,
        elapsed: f32,
        bursts: &[ParticleBurst],
        ring_slot: usize,
    ) -> Option<ParticleFrame> {
        let pipelines = self.particle.pipelines.as_ref()?;
        let frame = self.state.particles.prepare(elapsed, bursts)?;
        let slot = ring_slot % pipelines.rows.len().max(1);
        let rows = self.state.particles.rows();
        if let Some(buffer) = pipelines.rows.get(slot) {
            let bytes = bytemuck::cast_slice::<[f32; 4], u8>(rows);
            // SAFETY: the buffer is shared storage sized for `PARTICLE_FRAME_ROWS` rows, which
            // the rows never outgrow, and this ring slot's last reader retired before the frame
            // was paced in.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    buffer.contents().as_ptr().cast::<u8>(),
                    bytes.len().min(buffer.length()),
                );
            }
        }
        self.particle.rows_slot = slot;
        Some(frame)
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
        frame: ParticleFrame,
    ) -> RenderResult<()> {
        let Some(pipelines) = self.particle.pipelines.as_ref() else {
            return Ok(());
        };
        if self.state.particles.is_empty() {
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
        if let Some(rows) = pipelines.rows.get(self.particle.rows_slot) {
            enc.set_buffer(rows, 0, simulate::FRAME_ROWS.index);
        }
        // Every live pool ticks, visible or not, so an off-screen emitter stays
        // in a realistic mid-life state for when the camera turns back. The cost
        // is per-slot work in a single threadgroup, so leaving it un-culled is
        // cheap.
        for emitter in self.state.particles.emitters(&frame) {
            let Some(gpu) = self.particle.gpu.get(emitter.id) else {
                continue;
            };
            let (rec, params) = (emitter.record, emitter.params);
            enc.set_buffer(gpu.pool.as_ref(), 0, simulate::POOL.index);
            enc.set_value(&params, simulate::PARAMS.index);
            dispatch_threads(&enc, [rec.max_particles, 1, 1], SIMULATE_GROUP);
        }
        Ok(())
    }

    // Encode the `ParticlesDraw` node, or with `late`, the pre-TAA scene, the
    // `ParticlesLateDraw` one: one alpha-blended camera-facing quad per live
    // particle of the emitters that draw at that stage, blend-written into
    // `hdr_resolve` or the scene `late` names. The vertex stage reads the pool
    // `encode_particles_sim` wrote. Returns the draw-call count.
    pub(in crate::metal) fn encode_particles_draw(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        frame: ParticleFrame,
        vp: [[f32; 4]; 4],
        frustum: &Frustum,
        reactive: ReactiveWrite,
        late: Option<&ProtocolObject<dyn objc2_metal::MTLTexture>>,
    ) -> RenderResult<u32> {
        let Some(pipelines) = self.particle.pipelines.as_ref() else {
            return Ok(0);
        };
        let last_tex = self.scene.textures.len().saturating_sub(1);

        // Visibility-cull per emitter, for the draw alone: the simulation above
        // ticked every pool.
        let mut visible = self
            .state
            .particles
            .visible(&frame, frustum, late.is_some())
            .peekable();

        let view = ParticleView::new(
            vp,
            &self.state.view.matrix,
            self.targets.hdr.height as f32,
            self.pre_exposure.value(),
            self.state.lights.uniforms(),
            &self.state.view.daylight,
        );

        // A fresh Load/Store pass: the prior render pass ended with its own
        // command buffer. When every emitter culls out we skip the encoder
        // entirely.
        if visible.peek().is_none() {
            return Ok(0);
        }
        let pass_desc = MTLRenderPassDescriptor::new();
        // SAFETY: plain descriptor property setters; the subscripted slots are ones this descriptor
        // declares.
        unsafe {
            let ca = pass_desc.colorAttachments().objectAtIndexedSubscript(0);
            ca.setTexture(Some(
                late.unwrap_or_else(|| self.targets.hdr.hdr_resolve.as_ref()),
            ));
            ca.setLoadAction(MTLLoadAction::Load);
            ca.setStoreAction(MTLStoreAction::Store);
        }
        self.attach_reactive_mask(&pass_desc, reactive);

        if let Some(t) = &self.diagnostics.pass_timing {
            let id = if late.is_some() {
                super::pass_timing::PassId::ParticlesLateDraw
            } else {
                super::pass_timing::PassId::ParticlesDraw
            };
            t.attach_render(&pass_desc, id);
        }
        let enc = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&pass_desc)
                .ok_or_else(|| {
                    RenderError::Other("failed to get particle render encoder".into())
                })?,
            ns_string!("particles: draw"),
        );
        let mut lit = false;
        enc.set_pipeline(&pipelines.render);
        enc.set_vertex_value(&view, draw::VIEW.index);
        enc.set_fragment_sampler(&pipelines.sampler, draw::ALBEDO_SAMPLER.index);
        // Resolved scene depth at texture(2) for the manual depth test.
        enc.set_fragment_texture(
            self.targets.hdr.depth_resolve.as_ref(),
            draw::SCENE_DEPTH.index,
        );

        let mut draw_calls: u32 = 0;
        for emitter in visible {
            let Some(gpu) = self.particle.gpu.get(emitter.id) else {
                continue;
            };
            let rec = emitter.record;
            let params = emitter.params;
            if rec.lit != lit {
                lit = rec.lit;
                enc.set_pipeline(if lit {
                    &pipelines.render_lit
                } else {
                    &pipelines.render
                });
            }
            let slot = rec.texture_slot.min(last_tex);
            enc.set_vertex_buffer(gpu.pool.as_ref(), 0, draw::POOL.index);
            enc.set_vertex_value(&params, draw::PARAMS.index);
            enc.set_fragment_texture(self.scene.textures[slot].as_ref(), draw::ALBEDO.index);
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
    ring_slots: usize,
) -> RenderResult<ParticlePipelines> {
    // Compute kernel, from `particle_simulate.hlsl`. The render pair below
    // splices the same `{PARTICLE_TYPES}` fragment, so both halves stride one
    // declaration of the pool record and the per-emitter uniform.
    let simulate = compute_pipeline(
        device,
        &super::builtin_shaders::PARTICLE_SIMULATE,
        hot_reload,
    )?;

    let render = render_pipeline(
        device,
        &super::builtin_shaders::PARTICLE_VERT,
        &super::builtin_shaders::PARTICLE_FRAG,
        hot_reload,
    )?;
    let render_lit = render_pipeline(
        device,
        &super::builtin_shaders::PARTICLE_LIT_VERT,
        &super::builtin_shaders::PARTICLE_LIT_FRAG,
        hot_reload,
    )?;

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

    let row_bytes = PARTICLE_FRAME_ROWS * size_of::<[f32; 4]>();
    let rows = (0..ring_slots.max(1))
        .map(|_| {
            device
                .newBufferWithLength_options(row_bytes, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| allocation_failed("the particle frame rows"))
        })
        .collect::<RenderResult<_>>()?;
    Ok(ParticlePipelines {
        simulate,
        render,
        render_lit,
        sampler,
        rows,
    })
}

// The render pipeline of the `vert` and `frag` pair. No vertex descriptor:
// the vertex shader reads from the particle pool storage buffer directly via
// `[[vertex_id]]` + `[[instance_id]]`. Each entry compiles to its own
// metallib, so the two stages come from separate libraries and pair by
// semantic.
fn render_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    vert: &super::builtin_shaders::ShaderProgram,
    frag: &super::builtin_shaders::ShaderProgram,
    hot_reload: bool,
) -> RenderResult<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
    let vert_fn = super::builtin_shaders::entry_function(device, vert, hot_reload)?;
    let frag_fn = super::builtin_shaders::entry_function(device, frag, hot_reload)?;
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
    device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .map_err(|e| RenderError::ShaderCompile(format!("particle render pipeline: {e:?}")))
}

// Allocate the per-emitter GPU state for one record: a zero-initialized
// particle pool in shared storage.
pub(super) fn build_emitter_gpu_state(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    record: &ParticleEmitterRecord,
) -> RenderResult<ParticleEmitterGpuState> {
    let slots = record.max_particles as usize;
    let pool_bytes = slots * size_of::<GpuParticle>();
    let pool = device
        .newBufferWithLength_options(pool_bytes, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| allocation_failed("the particle pool buffer"))?;
    // Zero-init: every slot starts dead (`lifetime = 0`).
    // SAFETY: `pool` was just allocated with `pool_bytes` bytes of shared storage, so `contents()`
    // is a live CPU mapping of exactly that many bytes.
    unsafe {
        let dst = pool.contents().as_ptr().cast::<u8>();
        std::ptr::write_bytes(dst, 0, pool_bytes);
    }

    Ok(ParticleEmitterGpuState { pool })
}

impl EmitterPass for MtlContext {
    // Build the compute + render pipelines on first use, so a world that never
    // had an emitter pays no pipeline cost until the first add.
    fn ensure_emitter_pass(&mut self) -> RenderResult<()> {
        if self.particle.pipelines.is_none() {
            let pipelines = build_particle_pipelines(
                &self.hw.device,
                self.hot_reload.enabled,
                self.frames_in_flight + 1,
            )?;
            self.particle.pipelines = Some(pipelines);
        }
        Ok(())
    }

    // Allocate the emitter's pool + atomic counter buffer, zeroed so every slot
    // starts dead.
    fn create_emitter(&mut self, id: usize, record: &ParticleEmitterRecord) -> RenderResult<()> {
        let state = build_emitter_gpu_state(&self.hw.device, record)?;
        self.particle.gpu.set(id, state);
        Ok(())
    }

    // Drop the emitter's pool: Metal keeps the underlying buffers alive until
    // any in-flight command buffer referencing them completes, so this is safe
    // between encode passes.
    fn destroy_emitter(&mut self, id: usize) {
        drop(self.particle.gpu.take(id));
    }
}
