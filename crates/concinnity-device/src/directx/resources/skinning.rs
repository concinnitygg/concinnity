// src/directx/resources/skinning.rs
//
// Skinned-mesh resources for DxContext: the skinned shadow pipeline (built
// lazily by `upload_skinned` the first time a SkinnedMesh is uploaded), the
// skinned geometry upload, and the per-frame joint / morph-weight uploads.
// The per-slot CPU records these uploads read live in `gfx::skinned_slots`.

use concinnity_core::gfx::mesh_payload;
use concinnity_core::gfx::mesh_payload::{SkinnedVertex, Vertex};
use concinnity_core::gfx::render_types::*;
use concinnity_core::gfx::transform::IDENTITY;
use concinnity_core::render::rt_geom;
use concinnity_core::render::shadow_bias;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::super::allocator::PooledBuffer;
use super::super::com;
use super::super::context::*;
use super::super::pipeline::{serialize_and_create_root_sig, skinned_input_layout};
use super::super::slang_builtins;
use super::super::texture::*;
use crate::directx::slang_builtins::SlangCompile;
// Skinned shadow pipeline builders
//
// These mirror the shadow PSO builder in init/pipelines.rs but use the skinned
// vertex layout (80-byte SkinnedVertex with joint indices + weights). Skinned
// main-pass draws ride the GPU-driven pass through the skin fold.

// The depth-only skinned shadow vertex, the engine's own.
fn compile_skinned_shadow_shader(hot_reload: bool) -> Result<Vec<u8>, String> {
    slang_builtins::SKINNED_SHADOW_VERT.compile(hot_reload)
}

// Same as the shadow root signature but with one extra root SRV at slot [2]
// (t0) carrying the per-object joint matrices. Used by the skinned shadow PSO.
fn create_skinned_shadow_root_signature(
    device: &ID3D12Device,
) -> Result<ID3D12RootSignature, String> {
    let params = [
        // [0] Root constants: model mat4 (16) + cascade_idx + 3 pad = 20 DWORDs at b0
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 20,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        },
        // [1] Root CBV: shadow UBO (light_vps[4] + cascade_splits) at b1
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 1,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        },
        // [2] Root SRV: per-object joint matrices (t0, VS-only)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        },
    ];

    serialize_and_create_root_sig(device, &params, "skinned shadow root sig")
}

// Shadow-pass PSO for skinned geometry: the skinned shadow vertex shader
// (80-byte layout, depth-only). Uses the skinned shadow root signature.
fn create_skinned_shadow_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
) -> Result<ID3D12PipelineState, String> {
    let layout = skinned_input_layout();
    let pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: com::borrowed(root_sig),
        VS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: vs.as_ptr() as _,
            BytecodeLength: vs.len(),
        },
        InputLayout: D3D12_INPUT_LAYOUT_DESC {
            pInputElementDescs: layout.as_ptr(),
            NumElements: layout.len() as u32,
        },
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 0,
        DSVFormat: DXGI_FORMAT_D32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        SampleMask: u32::MAX,
        RasterizerState: D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE,
            FrontCounterClockwise: true.into(),
            DepthBias: shadow_bias::RASTER_CONSTANT as i32,
            DepthBiasClamp: shadow_bias::RASTER_CLAMP,
            SlopeScaledDepthBias: shadow_bias::RASTER_SLOPE,
            DepthClipEnable: true.into(),
            ..Default::default()
        },
        DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: true.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
            DepthFunc: D3D12_COMPARISON_FUNC_LESS,
            StencilEnable: false.into(),
            ..Default::default()
        },
        BlendState: D3D12_BLEND_DESC {
            ..Default::default()
        },
        ..Default::default()
    };

    // SAFETY: `desc` outlives this synchronous call, and so do the root signature, shader bytecode
    // and input-element array whose raw pointers it borrows.
    unsafe { crate::directx::pso_library::create_graphics(device, &pso_desc) }
        .map_err(|e| format!("create skinned shadow PSO: {e}"))
}
impl DxContext {
    // Upload skinned-mesh geometry, build the skinned shadow pipeline and the
    // main-pass skin fold.
    //
    // Called once at init by `GraphicsSystem` when the world declares at least
    // one `SkinnedMesh`. The joint matrices live in per-(frame, object) upload
    // buffers the skinned passes bind as a root SRV. With no skinned meshes
    // this is never called and every skinned pass is skipped.
    pub(crate) fn upload_skinned(
        &mut self,
        vertices: &[SkinnedVertex],
        indices: &[u32],
        draw_objects: Vec<SkinnedDrawObject>,
    ) -> Result<(), String> {
        if draw_objects.is_empty() || vertices.is_empty() || indices.is_empty() {
            return Ok(());
        }
        if draw_objects.len() > MAX_SKINNED_OBJECTS {
            return Err(format!(
                "skinned: {} skinned meshes exceeds MAX_SKINNED_OBJECTS ({})",
                draw_objects.len(),
                MAX_SKINNED_OBJECTS
            ));
        }
        self.wait_idle();

        let skinned_shadow_vs = compile_skinned_shadow_shader(self.hot_reload.enabled)?;

        // Skinned shadow pipeline: built only when the static shadow pass is
        // active, so a skinned mesh casts a correctly deformed shadow.
        let (skinned_shadow_root_sig, skinned_shadow_pso) = if self.shadow_pso.is_some() {
            let sr = dump_on_err(
                self.diagnostics.info_queue.as_ref(),
                create_skinned_shadow_root_signature(&self.device),
            )?;
            let sp = dump_on_err(
                self.diagnostics.info_queue.as_ref(),
                create_skinned_shadow_pso(&self.device, &sr, &skinned_shadow_vs),
            )?;
            (Some(sr), Some(sp))
        } else {
            (None, None)
        };

        // Shared skinned vertex/index buffers (DEFAULT heap, GPU-copied once).
        let vtx_bytes = bytemuck::cast_slice(vertices);
        let idx_bytes = bytemuck::cast_slice(indices);
        // GENERIC_READ (rather than the narrower VERTEX_AND_CONSTANT_BUFFER /
        // INDEX_BUFFER) so these stay both vertex/index-bindable for the skinned
        // main + shadow passes AND shader-readable as raw root SRVs for the RT
        // skin compute dispatch (bind-pose VB) and the RT reflection trace (u32
        // IB). GENERIC_READ is a superset of both, so no per-frame transition on
        // these shared resources is needed.
        let skinned_vertex_buffer =
            upload_buffer(&self.alloc, vtx_bytes, D3D12_RESOURCE_STATE_GENERIC_READ)?;
        // Never zero-length: the ray-traced hit path binds this buffer as a raw
        // word array and no backend accepts a zero-length binding.
        let skinned_index_buffer = upload_buffer_padded(
            &self.alloc,
            idx_bytes,
            rt_geom::skinned_index_buffer_bytes(indices.len()) as u64,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        self.skinned.vertex_buffer_view = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: com::gpu_va(&skinned_vertex_buffer),
            SizeInBytes: vtx_bytes.len() as u32,
            StrideInBytes: std::mem::size_of::<SkinnedVertex>() as u32,
        };
        self.skinned.index_buffer_view = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: com::gpu_va(&skinned_index_buffer),
            SizeInBytes: idx_bytes.len() as u32,
            Format: DXGI_FORMAT_R32_UINT,
        };

        // Per-(frame, object) joint-matrix upload buffers, each MAX_JOINTS
        // float4x4 matrices, persistently mapped.
        //
        // The buffer is seeded with `MAX_JOINTS` identity matrices once at
        // creation. `upload_joint_matrices` later overwrites only the first
        // `mats.len()` slots each frame; anything past the live pose count
        // keeps the identity seed, so a vertex whose `joints.{xyzw}` indexes
        // past the live range degenerates into an LBS of identity matrices
        // (i.e. its bind-pose position) instead of reading uninitialized
        // UPLOAD-heap memory and producing an arbitrary spike. The seed is
        // also what the renderer wants on frame 0 before the first pose
        // arrives: every joint is identity, so the mesh shows in bind pose.
        let joint_buf_bytes = (MAX_JOINTS * std::mem::size_of::<[[f32; 4]; 4]>()) as u64;
        let identity_seed: Vec<[[f32; 4]; 4]> = vec![IDENTITY; MAX_JOINTS];
        let mut joint_buffers: Vec<Vec<PooledBuffer>> = Vec::with_capacity(FRAMES);
        let mut joint_ptrs: Vec<Vec<*mut u8>> = Vec::with_capacity(FRAMES);
        for _ in 0..FRAMES {
            let mut frame_bufs: Vec<PooledBuffer> = Vec::with_capacity(draw_objects.len());
            let mut frame_ptrs: Vec<*mut u8> = Vec::with_capacity(draw_objects.len());
            for _ in 0..draw_objects.len() {
                let buf = create_buffer(
                    &self.alloc,
                    joint_buf_bytes,
                    D3D12_HEAP_TYPE_UPLOAD,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                )
                .map_err(|e| format!("skinned joint buf: {e}"))?;
                let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
                // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload,
                // and the source is a separate allocation, so the ranges cannot overlap.
                unsafe {
                    buf.Map(0, None, Some(&mut ptr))
                        .map_err(|e| format!("map skinned joint buf: {e}"))?;
                    std::ptr::copy_nonoverlapping(
                        identity_seed.as_ptr() as *const u8,
                        ptr as *mut u8,
                        joint_buf_bytes as usize,
                    );
                }
                frame_bufs.push(buf);
                frame_ptrs.push(ptr as *mut u8);
            }
            joint_buffers.push(frame_bufs);
            joint_ptrs.push(frame_ptrs);
        }

        // Seed each object's joint matrices to identity (bind pose) so the mesh
        // renders undeformed until the first `update_skinned_pose`.
        self.skinned.slots.joint_matrices = draw_objects
            .iter()
            .map(|o| vec![IDENTITY; o.joint_count.max(1)])
            .collect();

        self.skinned.shadow_pso = skinned_shadow_pso;
        self.skinned.shadow_root_sig = skinned_shadow_root_sig;
        self.skinned.vertex_buffer = Some(skinned_vertex_buffer);
        self.skinned.index_buffer = Some(skinned_index_buffer);
        self.skinned.joint_buffers = joint_buffers;
        self.skinned.joint_ptrs = joint_ptrs;
        self.skinned.slots.draw_objects = draw_objects;
        // A whole new skinned set: nothing in the model-history ring was written
        // for these records.
        self.model_history.borrow_mut().reset(self.cull_count());

        // Morph targets are attached by a later `upload_skinned_morphs`; until
        // then every object is morphless (a re-upload / hot-reload resets here).
        let n_objects = self.skinned.slots.draw_objects.len();
        self.skinned.morph_delta_buffers = (0..n_objects).map(|_| None).collect();
        self.skinned.morph_target_counts = vec![0; n_objects];
        self.skinned.slots.morph_weights = vec![Vec::new(); n_objects];
        self.skinned.morph_weight_buffers = Vec::new();
        self.skinned.morph_weight_ptrs = Vec::new();

        // GPU-driven main-pass skinning: build the `rt_skin` compute pipeline
        // (reused independently of RT) + one UAV-writable deformed-vertex buffer
        // per frame-in-flight, sized to all skinned verts. Each frame
        // `encode_skin` poses the bind-pose verts into this frame's buffer and
        // the main pass's 2nd ExecuteIndirect draws the skinned records the cull
        // buffers reserved at init via the threaded `n_skinned` capacity.
        // Setting `self.draw.n_skinned` here engages the fold. Every skinned draw
        // rides the GPU-driven pass, so a build failure is a startup error, as
        // on Metal.
        {
            let stride = std::mem::size_of::<Vertex>();
            let deformed_bytes = (vertices.len() * stride).max(stride) as u64;
            let mut deformed_buffers: Vec<ID3D12Resource> = Vec::with_capacity(FRAMES);
            let mut deformed_vbvs: Vec<D3D12_VERTEX_BUFFER_VIEW> = Vec::with_capacity(FRAMES);
            for _ in 0..FRAMES {
                let buf =
                    create_uav_buffer(&self.device, deformed_bytes, D3D12_RESOURCE_STATE_COMMON)?;
                let vbv = D3D12_VERTEX_BUFFER_VIEW {
                    BufferLocation: com::gpu_va(&buf),
                    SizeInBytes: deformed_bytes as u32,
                    StrideInBytes: stride as u32,
                };
                deformed_buffers.push(buf);
                deformed_vbvs.push(vbv);
            }
            // Move COMMON -> VERTEX_AND_CONSTANT_BUFFER so the per-frame skin
            // pass's VERTEX -> UAV -> VERTEX transition cycle is valid from frame 0.
            // SAFETY: the command list is in the recording state, and every resource, descriptor
            // and slice these commands name is live for the call.
            one_shot_submit(&self.device, &self.command_queue, |cmd| unsafe {
                let barriers: Vec<D3D12_RESOURCE_BARRIER> = deformed_buffers
                    .iter()
                    .map(|b| {
                        transition_barrier(
                            b,
                            D3D12_RESOURCE_STATE_COMMON,
                            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                        )
                    })
                    .collect();
                cmd.ResourceBarrier(&barriers);
            })?;
            let skin = super::super::raytrace::build_rt_skin_pipeline(
                &self.device,
                self.hot_reload.enabled,
            )
            .map_err(|e| format!("skinned: main-pass skin fold build failed: {e}"))?;
            self.skinned.skin_pipeline = Some(skin);
            self.skinned.deformed_buffers = deformed_buffers;
            self.skinned.deformed_vbvs = deformed_vbvs;
            // Fresh ring: no slot has been posed yet, so the G-buffer velocity
            // must treat the previous deformed buffer as the current one until a
            // full frame has primed it.
            self.skinned
                .deformed_primed
                .store(false, std::sync::atomic::Ordering::Relaxed);
            self.draw.n_skinned = self.skinned.slots.draw_objects.len();
        }

        Ok(())
    }

    // Overwrite a `SkinnedMesh` draw slot's vertex + index data in the shared
    // skinned vertex / index buffers in place. Driven by asset hot-reload
    // (`cn debug` only). The slot's vertex region starts at
    // `vertex_base * size_of::<SkinnedVertex>()` and is `vertices.len()`
    // vertices wide; the index region lives at the slot's init-time
    // `index_offset` / `index_count`. Indices are rebased onto `vertex_base`
    // before writing (matching the init-time `upload_skinned` rebasing).
    // `indices.len()` must match init-time; size-changing reloads route
    // through `rebuild_skinned_geometry`. Joint-count
    // changes resize the per-slot joint-matrix buffers via
    // `update_skinned_skeleton`. Pipelines stay untouched.
    // Mirrors `MtlContext::update_skinned_mesh_geometry`.
    pub(crate) fn update_skinned_mesh_geometry(
        &mut self,
        skinned_index: usize,
        vertex_base: u32,
        vertices: &[SkinnedVertex],
        indices: &[u16],
    ) -> Result<(), String> {
        let obj = self
            .skinned
            .slots
            .draw_objects
            .get(skinned_index)
            .ok_or_else(|| {
                format!(
                    "update_skinned_mesh_geometry: skinned object {} out of range",
                    skinned_index
                )
            })?;
        if indices.len() != obj.index_count {
            return Err(format!(
                "update_skinned_mesh_geometry: skinned {} expects {} indices, got {} \
                 (in-place path is size-matched only; size changes route through \
                 rebuild_skinned_geometry)",
                skinned_index,
                obj.index_count,
                indices.len()
            ));
        }
        let v_buf = self.skinned.vertex_buffer.clone().ok_or(
            "update_skinned_mesh_geometry: no skinned vertex buffer (was upload_skinned called?)",
        )?;
        let i_buf = self.skinned.index_buffer.clone().ok_or(
            "update_skinned_mesh_geometry: no skinned index buffer (was upload_skinned called?)",
        )?;
        // Check the vertex region fits inside the live buffer. The shared
        // buffer was sized once at `upload_skinned` to hold every skinned
        // mesh's vertices; vertex_base + vertices.len() must stay within that
        // region or a neighboring slot would be overwritten.
        let v_byte_off = (vertex_base as usize) * std::mem::size_of::<SkinnedVertex>();
        let v_byte_len = std::mem::size_of_val(vertices);
        let v_buf_len = self.skinned.vertex_buffer_view.SizeInBytes as usize;
        if v_byte_off + v_byte_len > v_buf_len {
            return Err(format!(
                "update_skinned_mesh_geometry: vertex region [{}, {}) overruns skinned \
                 vertex buffer length {}",
                v_byte_off,
                v_byte_off + v_byte_len,
                v_buf_len
            ));
        }
        let i_byte_off = (obj.index_offset * std::mem::size_of::<u32>()) as u64;
        let rebased: Vec<u32> = indices
            .iter()
            .map(|&i| u32::from(i) + vertex_base)
            .collect();

        self.wait_idle();

        let vert_bytes = bytemuck::cast_slice(vertices);
        self.write_geometry_region(
            &v_buf,
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            v_byte_off as u64,
            vert_bytes,
        )?;
        let idx_bytes = bytemuck::cast_slice(&rebased);
        self.write_geometry_region(
            &i_buf,
            D3D12_RESOURCE_STATE_INDEX_BUFFER,
            i_byte_off,
            idx_bytes,
        )?;
        Ok(())
    }

    // The CPU-side skinned entry points the `RenderBackend` impl forwards to.
    // Each is the `SkinnedSlots` operation of the same name; the behavior and
    // its contract are documented there, once for all three backends.
    pub(crate) fn update_skinned_skeleton(
        &mut self,
        skinned_index: usize,
        new_joint_count: usize,
    ) -> Result<(), String> {
        self.skinned
            .slots
            .update_skeleton(skinned_index, new_joint_count)
    }

    pub(crate) fn update_skinned_pose(&mut self, skinned_index: usize, matrices: &[[[f32; 4]; 4]]) {
        self.skinned.slots.update_pose(skinned_index, matrices);
    }

    pub(crate) fn reveal_skinned_instance(&mut self, instance_index: usize, model: [[f32; 4]; 4]) {
        self.skinned
            .slots
            .reveal(instance_index, model, &mut self.model_history.borrow_mut());
    }

    pub(crate) fn retire_skinned_draw_object(&mut self, skinned_index: usize) {
        self.skinned.slots.retire(skinned_index);
    }

    pub(crate) fn update_skinned_models(&mut self, updates: &[(u32, [[f32; 4]; 4])]) {
        self.skinned.slots.update_models(updates);
    }

    // Copy this frame's skinning matrices into the per-frame joint buffers.
    // Called from `record_frame` before the skinned shadow + main passes.
    pub(in crate::directx) fn upload_joint_matrices(&self, frame_idx: usize) {
        let Some(frame_ptrs) = self.skinned.joint_ptrs.get(frame_idx) else {
            return;
        };
        for (i, mats) in self.skinned.slots.joint_matrices.iter().enumerate() {
            let Some(&dst) = frame_ptrs.get(i) else {
                continue;
            };
            let n = mats.len().min(MAX_JOINTS);
            // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload, and
            // the source is a separate allocation, so the ranges cannot overlap.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    mats.as_ptr() as *const u8,
                    dst,
                    n * std::mem::size_of::<[[f32; 4]; 4]>(),
                );
            }
        }
    }

    // GPU virtual address of skinned object `i`'s joint buffer for `frame_idx`.
    pub(in crate::directx) fn skinned_joint_gva(&self, frame_idx: usize, i: usize) -> u64 {
        com::gpu_va(&self.skinned.joint_buffers[frame_idx][i])
    }

    // Attach morph-target buffers (`PayloadMorphs::packed_words`) to the skinned
    // draw objects. `morphs[i]` pairs with draw object `i`; instance copies share
    // their template's `Arc`, so each unique entry set becomes one GPU buffer. Allocates the per-frame
    // weight upload buffers (one f32 per target per object) when any object
    // carries morphs. Called once after `upload_skinned`.
    pub(in crate::directx) fn upload_skinned_morphs(
        &mut self,
        morphs: Vec<Option<std::sync::Arc<mesh_payload::PayloadMorphs>>>,
    ) -> Result<(), String> {
        use std::collections::HashMap;

        let n = self.skinned.slots.draw_objects.len();
        let mut delta_buffers: Vec<Option<PooledBuffer>> = Vec::with_capacity(n);
        let mut target_counts: Vec<u32> = Vec::with_capacity(n);
        let mut weights: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut by_source: HashMap<usize, (PooledBuffer, u32)> = HashMap::new();

        for m in morphs.iter().take(n) {
            match m {
                None => {
                    delta_buffers.push(None);
                    target_counts.push(0);
                    weights.push(Vec::new());
                }
                Some(data) => {
                    let key = std::sync::Arc::as_ptr(data) as usize;
                    let (buf, count) = match by_source.get(&key) {
                        Some(entry) => entry.clone(),
                        None => {
                            let words = data.packed_words();
                            let bytes: &[u8] = bytemuck::cast_slice(&words);
                            let buf = upload_buffer(
                                &self.alloc,
                                bytes,
                                D3D12_RESOURCE_STATE_GENERIC_READ,
                            )?;
                            let count = data.target_count() as u32;
                            by_source.insert(key, (buf.clone(), count));
                            (buf, count)
                        }
                    };
                    delta_buffers.push(Some(buf));
                    weights.push(vec![0.0; count as usize]);
                    target_counts.push(count);
                }
            }
        }
        // Pad the tail morphless if `morphs` was shorter than `draw.objects`.
        while delta_buffers.len() < n {
            delta_buffers.push(None);
            target_counts.push(0);
            weights.push(Vec::new());
        }

        // Per-(frame, object) weight upload buffers, one f32 per target (>= 1 so
        // every slot has a valid GVA), persistently mapped and zero-seeded. Only
        // allocated when some object carries morphs.
        let (mut weight_buffers, mut weight_ptrs) = (Vec::new(), Vec::new());
        if target_counts.iter().any(|&c| c > 0) {
            for _ in 0..FRAMES {
                let mut frame_bufs: Vec<PooledBuffer> = Vec::with_capacity(n);
                let mut frame_ptrs: Vec<*mut u8> = Vec::with_capacity(n);
                for count in &target_counts {
                    let bytes = ((*count).max(1) as u64) * std::mem::size_of::<f32>() as u64;
                    let buf = create_buffer(
                        &self.alloc,
                        bytes,
                        D3D12_HEAP_TYPE_UPLOAD,
                        D3D12_RESOURCE_STATE_GENERIC_READ,
                    )
                    .map_err(|e| format!("morph weight buf: {e}"))?;
                    let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
                    // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this
                    // payload, and the source is a separate allocation, so the ranges cannot
                    // overlap.
                    unsafe {
                        buf.Map(0, None, Some(&mut ptr))
                            .map_err(|e| format!("map morph weight buf: {e}"))?;
                        std::ptr::write_bytes(ptr as *mut u8, 0, bytes as usize);
                    }
                    frame_bufs.push(buf);
                    frame_ptrs.push(ptr as *mut u8);
                }
                weight_buffers.push(frame_bufs);
                weight_ptrs.push(frame_ptrs);
            }
        }

        self.skinned.morph_delta_buffers = delta_buffers;
        self.skinned.morph_target_counts = target_counts;
        self.skinned.slots.morph_weights = weights;
        self.skinned.morph_weight_buffers = weight_buffers;
        self.skinned.morph_weight_ptrs = weight_ptrs;
        Ok(())
    }

    pub(in crate::directx) fn update_morph_weights(
        &mut self,
        skinned_index: usize,
        weights: &[f32],
    ) {
        self.skinned
            .slots
            .update_morph_weights(skinned_index, weights);
    }

    // Copy this frame's morph weights into the per-frame weight buffers. Called
    // from `record_frame` alongside `upload_joint_matrices`. A no-op when no
    // object carries morphs (the buffers are empty).
    pub(in crate::directx) fn upload_morph_weights(&self, frame_idx: usize) {
        let Some(frame_ptrs) = self.skinned.morph_weight_ptrs.get(frame_idx) else {
            return;
        };
        for (i, w) in self.skinned.slots.morph_weights.iter().enumerate() {
            let (Some(&dst), false) = (frame_ptrs.get(i), w.is_empty()) else {
                continue;
            };
            // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload, and
            // the source is a separate allocation, so the ranges cannot overlap.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    w.as_ptr() as *const u8,
                    dst,
                    w.len() * std::mem::size_of::<f32>(),
                );
            }
        }
    }

    // GPU virtual address of skinned object `i`'s morph weight buffer for
    // `frame_idx`, or `None` when no weight buffers are allocated.
    pub(in crate::directx) fn morph_weight_gva(&self, frame_idx: usize, i: usize) -> Option<u64> {
        let buf = self.skinned.morph_weight_buffers.get(frame_idx)?.get(i)?;
        Some(com::gpu_va(buf))
    }
}
