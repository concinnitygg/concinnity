//! Skinned-mesh GPU resources: buffer setup (`upload_skinned`),
//! per-frame pose updates, hot-reload of skinned geometry, and skeleton
//! joint-count changes.
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::mesh_payload;
use concinnity_engine::gfx::mesh_payload::SkinnedVertex;
use concinnity_engine::gfx::render_types::{SkinnedDrawObject, SkinnedIndex};
use concinnity_engine::render::backend::SkinnedDraws;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::{rt_accel::geom, slots::geometry_repack};
use concinnity_engine::transform::IDENTITY;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer as _, MTLComputePipelineState, MTLDevice, MTLResourceOptions};

use crate::metal::context::{MtlContext, bytes_of_slice, write_buffer_region, write_buffer_slice};
use crate::metal::error::allocation_failed;

// Upload a skinned index slice, sized by `skinned_index_buffer_bytes` (which
// the DirectX and Vulkan hosts size the same buffer with).
pub(super) fn upload_skinned_index_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    indices: &[u32],
    label: &str,
) -> RenderResult<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>> {
    let buffer = device
        .newBufferWithLength_options(
            geom::skinned_index_buffer_bytes(indices.len()),
            MTLResourceOptions::StorageModeShared,
        )
        .ok_or_else(|| allocation_failed(format_args!("{label} skinned index buffer")))?;
    write_buffer_slice(&buffer, indices).map_err(|e| e.context(label))?;
    Ok(buffer)
}

// All skinned-mesh rendering state grouped into one feature unit: the shared
// skinned vertex / index buffers, the per-mesh draw
// objects, and the current + previous joint-palette matrices. All `None` /
// empty until `upload_skinned` runs; with no `SkinnedMesh` in the world the
// skinned passes are skipped entirely. Skinned geometry draws through the
// GPU-driven pass from the pre-skinned `deformed` buffer, so there is no
// skinned main pipeline. (The G-buffer pre-pass skinned pipeline lives on
// `GBufferState` with its siblings.)
pub(crate) struct SkinnedState {
    // Shared vertex buffer holding every skinned mesh's `SkinnedVertex` data.
    pub vertex_buffer: Option<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    // Shared index buffer for skinned geometry.
    pub index_buffer: Option<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    // GPU-driven fold: the `rt_skin` compute pipeline that deforms
    // bind-pose vertices into the per-frame `deformed` buffer, built here
    // independently of RT (which keeps its own pipeline) so a skinned world with
    // no ray tracing still gets the pre-skin. `None` until `upload_skinned` runs
    // on a bindless world with static geometry.
    pub skin_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    // One deformed-vertex buffer per frame-in-flight (56-byte static `Vertex`
    // layout, mirroring the skinned VB's global indexing so the skinned
    // index buffer addresses it directly with `base_vertex = 0`). The per-frame
    // skin compute writes this frame's slot; the main-pass skinned ICB tail
    // reads it. `StorageModeShared`, NOT Private: written and read in separate
    // command buffers under the parallel per-pass encoder, where a Private
    // buffer GPU-page-faults (the RT deformed buffer hit the same and uses
    // Shared). Empty until `upload_skinned` allocates them.
    pub deformed: Vec<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    // First-frame priming gate for the GPU-driven G-buffer velocity:
    // the previous-frame deformed buffer is unposed on frame 0 (and after a
    // deformed-ring rebuild in `upload_skinned`), so reading it would emit a
    // garbage skinned motion vector for one frame. While `false`, the G-buffer
    // skinned tail binds the CURRENT deformed buffer as the previous one (zero
    // skinned motion); set `true` after the first skinned-tail draw. Atomic, not
    // `Cell`: the G-buffer pass encodes on a render-graph worker thread (the
    // parallel per-pass encoder shares `&self` across rayon workers), so any
    // interior mutation reachable from `encode_pass_into` must be atomic, like
    // `draw_calls_accum`. Reset to `false` on the main thread in `upload_skinned`.
    pub deformed_primed: std::sync::atomic::AtomicBool,
    // Per-object morph-target bindings, parallel to the scene's skinned slots;
    // `None` for a mesh without morph targets. Instance copies share their
    // template's entry buffer.
    pub morphs: Vec<Option<MorphBinding>>,
}

impl SkinnedState {
    pub(crate) fn new() -> Self {
        Self {
            vertex_buffer: None,
            index_buffer: None,
            skin_pipeline: None,
            deformed: Vec::new(),
            deformed_primed: std::sync::atomic::AtomicBool::new(false),
            morphs: Vec::new(),
        }
    }
}

// GPU-resident morph data for one skinned mesh: the packed sparse buffer
// (`PayloadMorphs::packed_words`: per-vertex offsets, then 28-byte
// `MorphEntry`s) and its target count.
#[derive(Clone)]
pub(crate) struct MorphBinding {
    pub buffer: Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    pub target_count: u32,
}

impl SkinnedDraws for MtlContext {
    // Build the GPU pipelines + buffers for skeletally animated meshes.
    //
    // Called once by `GraphicsSystem` after `MtlContext::new`, only when the
    // world declares at least one `SkinnedMesh`. Skinned geometry draws through
    // the GPU-driven pass: the pre-skin kernel deforms it into a per-frame
    // buffer and the cull records draw it as rigid geometry. With no skinned
    // meshes this is never called and every skinned pass is skipped.
    fn upload_skinned(
        &mut self,
        vertices: &[SkinnedVertex],
        indices: &[u32],
        draw_objects: Vec<SkinnedDrawObject>,
    ) -> RenderResult<()> {
        if draw_objects.is_empty() || vertices.is_empty() || indices.is_empty() {
            return Ok(());
        }

        // SAFETY: the pointer and length describe the live `vertices` allocation, and Metal copies
        // those bytes into the new buffer before the call returns.
        let skinned_vertex_buffer = unsafe {
            let ptr = std::ptr::NonNull::new(vertices.as_ptr() as *mut _)
                .ok_or_else(|| RenderError::Other("skinned vertex slice is empty".into()))?;
            self.hw
                .device
                .newBufferWithBytes_length_options(
                    ptr,
                    size_of_val(vertices),
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or_else(|| allocation_failed("skinned vertex buffer"))?
        };
        let skinned_index_buffer =
            upload_skinned_index_buffer(&self.hw.device, indices, "upload_skinned")?;

        // Seed each object's joint matrices to identity (bind pose) so the
        // mesh renders undeformed until the first `update_skinned_pose`.
        self.state.skinned.joint_matrices = draw_objects
            .iter()
            .map(|o| vec![IDENTITY; o.joint_count.max(1)])
            .collect();

        // GPU-driven skinned fold: build the per-frame pre-skin so skinned
        // objects draw as rigid deformed geometry through the unified cull.
        // A build failure is a startup error rather than a degraded render,
        // as on every host. The skin pipeline is built independently of RT;
        // RT keeps its own skin pipeline + deformed buffer.
        if self.cull.bindless {
            let skin_pipeline = crate::metal::raytrace::build_rt_skin_pipeline(
                &self.hw.device,
                self.hot_reload.enabled,
            )?;
            // One deformed buffer per frame-in-flight (the skin write and the
            // main-pass read live in separate command buffers, so a per-frame
            // ring lets frames pipeline without the next frame's skin racing this
            // frame's draw). Sized to every skinned vertex (global indexing,
            // base 0). Shared storage: a Private buffer page-faults in this
            // cross-command-buffer producer/consumer pattern (see the RT path).
            let stride = crate::metal::raytrace::VERTEX_STRIDE;
            let deformed_bytes = (vertices.len() * stride).max(stride);
            let mut deformed = Vec::with_capacity(self.frames_in_flight);
            for _ in 0..self.frames_in_flight {
                let buf = self
                    .hw
                    .device
                    .newBufferWithLength_options(
                        deformed_bytes,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or_else(|| allocation_failed("skinned deformed-vertex buffer"))?;
                deformed.push(buf);
            }
            self.skinned.skin_pipeline = Some(skin_pipeline);
            self.skinned.deformed = deformed;
            // Fresh deformed ring: the previous-frame slots are unposed until a
            // frame writes them, so re-arm the G-buffer velocity priming gate.
            // Main-thread store (this runs outside the per-pass fan-out).
            self.skinned
                .deformed_primed
                .store(false, std::sync::atomic::Ordering::Relaxed);
            // The count `cull_count()` reads: the skinned records ride the
            // unified cull + bindless ICB.
            self.state.draw.n_skinned = draw_objects.len();
        }

        self.skinned.vertex_buffer = Some(skinned_vertex_buffer);
        self.skinned.index_buffer = Some(skinned_index_buffer);
        self.state.skinned.draw_objects = draw_objects;
        // A whole new skinned set: nothing in the model-history ring was
        // written for these records.
        let n_cull = self.cull_count();
        self.state.model_history.get_mut().reset(n_cull);
        Ok(())
    }

    // Upload morph-target entry buffers for the skinned draw objects.
    // `morphs[i]` pairs with draw object `i`; instance copies share their
    // template's `Arc`, so each unique entry set becomes one GPU buffer.
    fn upload_skinned_morphs(
        &mut self,
        morphs: &[Option<std::sync::Arc<mesh_payload::PayloadMorphs>>],
    ) -> RenderResult<()> {
        use std::collections::HashMap;

        let mut by_source: HashMap<usize, MorphBinding> = HashMap::new();
        let mut bindings: Vec<Option<MorphBinding>> = Vec::with_capacity(morphs.len());
        let mut weights: Vec<Vec<f32>> = Vec::with_capacity(morphs.len());
        for m in morphs {
            let binding = match m {
                None => None,
                Some(data) => {
                    let key = std::sync::Arc::as_ptr(data) as usize;
                    let entry = if let Some(b) = by_source.get(&key) {
                        b.clone()
                    } else {
                        let words = data.packed_words();
                        let bytes = bytes_of_slice(&words);
                        // SAFETY: the pointer and length describe the live `bytes` allocation,
                        // and Metal copies those bytes into the new buffer before the call
                        // returns.
                        let buffer = unsafe {
                            let ptr = std::ptr::NonNull::new(bytes.as_ptr() as *mut _).ok_or_else(
                                || RenderError::Other("morph entry slice is empty".to_string()),
                            )?;
                            self.hw
                                .device
                                .newBufferWithBytes_length_options(
                                    ptr,
                                    bytes.len(),
                                    MTLResourceOptions::StorageModeShared,
                                )
                                .ok_or_else(|| allocation_failed("morph entry buffer"))?
                        };
                        let b = MorphBinding {
                            buffer,
                            target_count: u32::saturating_from(data.target_count()),
                        };
                        by_source.insert(key, b.clone());
                        b
                    };
                    Some(entry)
                }
            };
            weights.push(vec![
                0.0;
                binding.as_ref().map_or(0, |b| b.target_count as usize)
            ]);
            bindings.push(binding);
        }
        self.skinned.morphs = bindings;
        self.state.skinned.morph_weights = weights;
        Ok(())
    }

    // Overwrite a `SkinnedMesh` draw slot's vertex + index data in the
    // shared skinned vertex / index buffers in place. Driven by asset
    // hot-reload (`cn run -f` only).
    //
    // Like [`MtlContext::update_mesh_geometry`], this rewrites a live buffer
    // region with no in-flight fence. There is no steady-state skinned
    // streamer (skinned meshes upload once at init and are never evicted), so
    // the only caller is the human-paced `cn run -f` hot-reload, which stalls
    // for the reload: no frame racing this write is plausibly in flight.
    // Production (non-debug) callers must not take this path.
    //
    // The slot's vertex region starts at
    // `vertex_base * size_of::<SkinnedVertex>()` and is `vertices.len()`
    // vertices wide; the index region lives at the slot's init-time
    // `index_offset` / `index_count`. Indices are rebased onto `vertex_base`
    // before writing (the existing init-time draws followed the same
    // rebasing convention). New `verts` / `idxs` must match the slot's
    // init-time count -- size-changing reloads route through
    // [`Self::rebuild_skinned_geometry`] instead. The shader
    // libraries + pipelines stay untouched on every skinned reload path;
    // joint-count changes resize the per-slot joint-matrix buffers via
    // [`Self::update_skinned_skeleton`].
    fn update_skinned_mesh_geometry(
        &mut self,
        skinned_index: SkinnedIndex,
        vertex_base: u32,
        vertices: &[SkinnedVertex],
        indices: &[u16],
    ) -> RenderResult<()> {
        let v_buf = self.skinned.vertex_buffer.as_ref().ok_or_else(|| {
            RenderError::Other(
                "update_skinned_mesh_geometry: no skinned vertex buffer (was upload_skinned called?)"
                    .to_string(),
            )
        })?;
        let i_buf = self.skinned.index_buffer.as_ref().ok_or_else(|| {
            RenderError::Other(
                "update_skinned_mesh_geometry: no skinned index buffer (was upload_skinned called?)"
                    .to_string(),
            )
        })?;
        let write = geometry_repack::place_skinned_update(
            &self.state.skinned.draw_objects,
            skinned_index,
            vertex_base,
            vertices.len(),
            indices,
            v_buf.length(),
        )?;
        write_buffer_region(
            v_buf,
            usize::saturating_from(write.vertex_offset),
            bytes_of_slice(vertices),
        )?;
        write_buffer_region(
            i_buf,
            usize::saturating_from(write.index_offset),
            bytes_of_slice(&write.indices),
        )?;
        Ok(())
    }
}
