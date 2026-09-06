// src/directx/resources.rs
//
// Runtime GPU resource management for DxContext: texture-pool slot updates,
// mesh upload/eviction, chunk streaming, and skinned-mesh upload. Also owns
// the skinned shadow pipeline (built lazily by `upload_skinned` the first time
// a SkinnedMesh is uploaded), mirroring metal/resources/skinning.rs.
use concinnity_core::gfx::transform::IDENTITY;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::allocator::PooledBuffer;
use crate::gfx::backend::ChunkMesh;
use crate::gfx::mesh_payload::{SkinnedVertex, Vertex};
use crate::gfx::render_types::*;
use crate::gfx::shadow_bias;

use super::com;
use super::context::*;
use super::init::pipelines::{BucketPipelineTargets, build_bucket_pipeline};
use super::pipeline::{serialize_and_create_root_sig, skinned_input_layout};
use super::slang_builtins;
use super::texture::*;
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
    // CPU descriptor handle for CBV/SRV/UAV heap `slot`.
    fn srv_slot_cpu(&self, slot: usize) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        // SAFETY: a property query on a live descriptor heap; it only reads.
        let base = unsafe {
            self.descriptors
                .srv_heap
                .GetCPUDescriptorHandleForHeapStart()
        };
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: base.ptr + slot * self.descriptors.srv_descriptor_size,
        }
    }

    // Re-point every per-frame flat-pool copy that samples texture-pool `slot`.
    // The swapped resource has exactly one descriptor per frame copy (index ==
    // its handle), shared by albedo + normal sampling and by the RT hit shader,
    // so one re-point per copy refreshes every consumer at once.
    fn rewrite_bound_texture_srvs(&self, slot: usize) {
        let resource = &self.descriptors.textures[slot];
        for f in 0..FRAMES {
            write_texture_srv(
                &self.device,
                resource,
                self.srv_slot_cpu(self.flat_pool_slot(f, slot)),
            );
        }
    }

    // Heap slot of pool index `slot` in frame `frame`'s flat-pool copy.
    fn flat_pool_slot(&self, frame: usize, slot: usize) -> usize {
        self.descriptors.flat_pool_base_slot + frame * self.descriptors.flat_pool_len + slot
    }

    // Re-point every SRV that samples texture-pool `slot`. Only legal under a
    // device drain.
    fn rewrite_texture_slot(&self, slot: usize) {
        self.rewrite_bound_texture_srvs(slot);
    }

    // Whether replacing pool `slot` must drain the device first: true when an
    // SRV that samples the slot may be dereferenced by pending command lists
    // AND cannot wait for the per-frame propagation. The flat-pool copies
    // propagate per frame, so only a world with nothing to GPU-drive (which
    // draws nothing) answers yes.
    fn streamed_slot_needs_drain(&self) -> bool {
        !(self.cull.main_bindless_pso.is_some() && self.cull_count() > 0)
    }

    // Replace texture-pool `slot` with a freshly decoded texture.
    //
    // The asset-streaming subsystem calls this to bring a texture resident
    // after init. Like Vulkan -- and unlike Metal, whose bind paths re-read
    // the texture pool every frame -- the D3D12 per-object / per-cluster SRVs
    // are baked into the descriptor heap at init, so a streamed swap must
    // rewrite every heap slot that samples this pool index (as an albedo or a
    // normal map). The streaming fast path never stalls the device: the
    // upload is submitted without waiting (the in-order queue executes it
    // before any later frame's lists), the build-time pairs are re-pointed
    // immediately (undereferenced while the bindless pass drives every draw),
    // the per-frame flat-pool copies re-point one per frame as their fences
    // retire, and the old resource plus upload transients are parked on
    // `stream.retires` until every consumer provably moved off them. When a
    // pending-referenced SRV samples the slot (see `streamed_slot_needs_drain`)
    // the swap instead drains the device and rewrites everything in place,
    // matching the hot-reload paths below.
    pub(crate) fn update_texture_slot(
        &mut self,
        slot: usize,
        image: &crate::bake::texture::TextureImage,
    ) -> Result<(), String> {
        if slot >= self.descriptors.textures.len() {
            return Err(format!(
                "update_texture_slot: slot {} out of range (pool size {})",
                slot,
                self.descriptors.textures.len()
            ));
        }
        if self.streamed_slot_needs_drain() {
            self.wait_idle();
            let texture = upload_texture_image(&self.alloc, image)?;
            self.descriptors.textures[slot] = texture;
            self.rewrite_texture_slot(slot);
            // The full rewrite covered every flat-pool copy, so any propagation
            // queued for this slot is already satisfied.
            self.stream.pool_rewrites.remove(slot);
            return Ok(());
        }
        let (texture, in_flight) = upload_texture_image_deferred(&self.alloc, image)?;
        let old = std::mem::replace(&mut self.descriptors.textures[slot], texture);
        self.stream.pool_rewrites.queue(slot);
        // `+ 1`: the swap lands between frames, after the previous frame's
        // submit, so the first frame fence that covers the upload submission
        // is the one signalled by the NEXT draw -- waited FRAMES ticks after
        // that draw's own tick.
        self.stream
            .retires
            .push(super::texture::StreamedUploadRetire {
                texture: old,
                upload: in_flight.upload,
                allocator: in_flight.allocator,
                cmd: in_flight.cmd,
                retire_at: self.stream.frame + FRAMES as u64 + 1,
            });
        Ok(())
    }

    // Per-frame streamed-texture upkeep, called at the top of `draw_frame`
    // right after frame slot `frame`'s fence wait: re-point this frame's
    // flat-pool copy at any swapped slots (legal now -- the wait retired every
    // list that dereferences this copy), then release retires whose covering
    // fence has signalled (dropping the entry releases the COM references).
    pub(super) fn apply_streamed_texture_rewrites(&mut self, frame: usize) {
        self.stream.frame += 1;
        if !self.stream.pool_rewrites.is_empty() {
            let last = self.descriptors.textures.len().saturating_sub(1);
            for slot in self.stream.pool_rewrites.begin_frame() {
                let resource = &self.descriptors.textures[slot.min(last)];
                write_texture_srv(
                    &self.device,
                    resource,
                    self.srv_slot_cpu(self.flat_pool_slot(frame, slot)),
                );
            }
        }
        let now = self.stream.frame;
        self.stream.retires.retain(|r| r.retire_at > now);
    }

    // Reset texture-pool `slot` to a 1x1 mid-grey placeholder.
    //
    // Used by the asset-streaming subsystem to mark a slot whose texture is
    // not yet resident; a later `update_texture_slot` brings the real texture
    // back. The grey is distinct from the white no-texture fallback so a
    // not-yet-streamed slot reads differently under inspection.
    pub(crate) fn evict_texture_slot(&mut self, slot: usize) -> Result<(), String> {
        let grey = crate::bake::texture::TextureImage::rgba8(1, 1, vec![128, 128, 128, 255]);
        self.update_texture_slot(slot, &grey)
    }

    // Replace the live colour-grading LUT with a fresh `size³` RGBA8 payload.
    // Driven by asset hot-reload (`cn debug` only) when the file-backed
    // `ColorLut` source is saved. Reuses the SRV heap slot the composite pass
    // already binds, so the new texture is picked up on the next `draw_frame`
    // with no pipeline or descriptor-table change. `wait_idle` first
    // guarantees no in-flight command list still references the old texture
    // (or the now-stale SRV) before it is overwritten and dropped. Mirrors
    // `MtlContext::update_color_lut`.
    pub(crate) fn update_color_lut(&mut self, size: u32, data: &[u8]) -> Result<(), String> {
        self.wait_idle();
        let srv_cpu = self.color_lut.srv_cpu;
        let srv_gpu = self.color_lut.srv_gpu;
        let new_lut = upload_color_lut(&self.alloc, size, data, srv_cpu, srv_gpu)?;
        self.color_lut = new_lut;
        Ok(())
    }

    // Swap the live IBL cubemap pair for a freshly precomputed envmap payload.
    // Driven by asset hot-reload (`cn debug` only). Re-uploads into the same
    // SRV heap slots [1] (irradiance) + [2] (prefilter) the init path wrote,
    // so every pipeline that references those slots keeps working without a
    // descriptor-table rebind. The new payload may declare different mip /
    // face sizes than the original; `EnvironmentMapTextures` is replaced
    // wholesale and the next frame's `ViewUniforms` picks up the new
    // `prefilter_mip_count` from `self.env_map`. `wait_idle` first guarantees
    // no in-flight command list still references the old cubes (or the
    // now-stale SRVs) before they are overwritten and dropped. Mirrors
    // `MtlContext::update_environment_map`.
    pub(crate) fn update_environment_map(&mut self, payload: &[u8]) -> Result<(), String> {
        let view = crate::bake::environment_map::deserialise(payload)
            .map_err(|e| format!("envmap hot-reload payload malformed: {e}"))?;
        self.wait_idle();
        let irr_srv_cpu = self.env_map.irradiance.srv_cpu;
        let irr_srv_gpu = self.env_map.irradiance.srv_gpu;
        let pre_srv_cpu = self.env_map.prefilter.srv_cpu;
        let pre_srv_gpu = self.env_map.prefilter.srv_gpu;
        let new_env = upload_environment_map(
            &self.alloc,
            EnvironmentMapPayload {
                irradiance_face: view.irradiance_face,
                irradiance_bytes: view.irradiance_bytes,
                prefilter_face: view.prefilter_face,
                mip_bytes: &view.prefilter_mip_bytes,
            },
            EnvironmentMapDescriptors {
                irr_srv_cpu,
                irr_srv_gpu,
                pre_srv_cpu,
                pre_srv_gpu,
            },
        )?;
        self.env_map = new_env;
        Ok(())
    }

    // Append a new draw object that re-uses an existing slot's geometry
    // region (vertex / index offsets, base_vertex, LOD alternates) with a
    // fresh model matrix, texture / normal-map slots, material, and cull
    // distance. Driven by `world.jsonl` hot-reload (`cn debug` only) when a
    // newly authored Prop references a Mesh / Model already present in the
    // init world. The clone is non-cullable (sentinel AABB) and joins
    // `draw.always` since the init-time BVH cannot refit; the dynamically added
    // prop is drawn every frame, like a streamed `VoxelWorld` chunk -- and
    // through the same runtime reserve in the cull records, so it needs no
    // descriptors of its own. Mirrors `MtlContext::clone_static_draw_object`.
    pub(crate) fn clone_static_draw_object(
        &mut self,
        src_draw_idx: usize,
        model: [[f32; 4]; 4],
        dst: crate::gfx::draw_slot::SlotAlloc,
    ) -> Result<(), String> {
        if runtime_reserve_full(&self.draw.objects, self.draw.n_objects, self.draw.n_runtime) {
            return Err(format!(
                "clone_static_draw_object: the runtime draw reserve ({}) is full",
                self.draw.n_runtime
            ));
        }
        let src = self.draw.objects.get(src_draw_idx).ok_or_else(|| {
            format!(
                "clone_static_draw_object: src draw {} out of range",
                src_draw_idx
            )
        })?;
        // A runtime spawn duplicates the template, swapping only the transform:
        // copy the source's material, pool slots, and cull distance.
        let texture_slot = src.texture_slot;
        let normal_map_slot = src.normal_map_slot;
        let material = src.material;
        let cull_distance = src.cull_distance;
        let obj = DrawObject {
            vertex_offset: src.vertex_offset,
            vertex_count: src.vertex_count,
            index_offset: src.index_offset,
            index_count: src.index_count,
            base_vertex: src.base_vertex,
            geometry_generation: src.geometry_generation,
            model,
            texture_slot,
            normal_map_slot,
            material,
            visible: true,
            resident: true,
            // Sentinel AABB so the init-time BVH cull skips the new draw:
            // it joins `draw.always` and is drawn every frame regardless of
            // camera position. Matches the runtime-streamed chunk pattern.
            bb_min: [f32::NAN; 3],
            bb_max: [f32::NAN; 3],
            cull_distance,
            lod_alternates: src.lod_alternates.clone(),
            shader_bucket: src.shader_bucket,
        };

        // Write at the engine-allocated destination slot.
        match dst {
            crate::gfx::draw_slot::SlotAlloc::Reuse(slot) => {
                self.draw.objects[slot] = obj;
                // Seed the velocity prepass's previous-model snapshot so a
                // recycled slot does not ghost from the prior occupant's
                // transform for one frame. A slot past the snapshot's end (one
                // appended beyond the build-time object count) falls back to its
                // own current model in the prepass, so the guard is enough.
                if let Some(gbuffer) = &self.gbuffer {
                    let mut prev = gbuffer.prev_models.borrow_mut();
                    if slot < prev.len() {
                        prev[slot] = model;
                    }
                }
            }
            crate::gfx::draw_slot::SlotAlloc::Append(slot) => {
                debug_assert_eq!(
                    slot,
                    self.draw.objects.len(),
                    "appended draw slot must match the draw-object count"
                );
                self.draw.objects.push(obj);
            }
        }
        // The cloned prop joins the RT-relevant draw set; the next RT update folds
        // it into the BVH (it reuses the source mesh's geometry slice, so only
        // this clone's BLAS is built).
        self.rt_topology_dirty = true;
        Ok(())
    }

    // Copy `data` into a sub-region of a DEFAULT-heap geometry buffer.
    //
    // `dest` is a buffer currently in `usage_state` (the vertex or index
    // buffer). The copy goes through a temporary UPLOAD-heap staging buffer
    // and a one-shot command list that transitions the resource
    // `usage_state -> COPY_DEST -> usage_state` around a `CopyBufferRegion`.
    // The caller must `wait_idle` first: the COPY_DEST transition covers the
    // whole resource, so no in-flight command list may still reference it.
    fn write_geometry_region(
        &self,
        dest: &ID3D12Resource,
        usage_state: D3D12_RESOURCE_STATES,
        offset: u64,
        data: &[u8],
    ) -> Result<(), String> {
        if data.is_empty() {
            return Ok(());
        }
        let upload = create_buffer(
            &self.alloc,
            data.len() as u64,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
        // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live local
        // that receives the mapping.
        unsafe { upload.Map(0, None, Some(&mut ptr)) }
            .map_err(|e| format!("mesh region map: {e}"))?;
        // SAFETY: the mapping covers an UPLOAD-heap buffer created to hold this payload, and the
        // source is a separate allocation, so the ranges cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
            upload.Unmap(0, None);
        }
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        one_shot_submit(&self.device, &self.command_queue, |cmd| unsafe {
            let to_dst = transition_barrier(dest, usage_state, D3D12_RESOURCE_STATE_COPY_DEST);
            cmd.ResourceBarrier(&[to_dst]);
            cmd.CopyBufferRegion(dest, offset, &*upload, 0, data.len() as u64);
            let back = transition_barrier(dest, D3D12_RESOURCE_STATE_COPY_DEST, usage_state);
            cmd.ResourceBarrier(&[back]);
        })
    }

    // Upload a streamed mesh's geometry into the shared vertex and index
    // buffers, place it via the sub-allocators, and mark the draw resident.
    //
    // The mesh-streaming subsystem calls this to bring a mesh resident after
    // init. The geometry is placed wherever the allocators find free space
    // (not the build-time region), so `DrawObject::vertex_offset` /
    // `index_offset` are rewritten here. `vertices` / `indices` must match the
    // fixed `vertex_count` / `index_count` recorded by `build_draw_list`.
    //
    // `indices` are mesh-relative (0-based); they are rebased onto the chosen
    // vertex region before upload, so the D3D12 draw can keep a 0 base-vertex.
    // `frame` reclaims deferred frees that have retired by then. `wait_idle`
    // runs first so the whole-resource COPY_DEST transition races no in-flight
    // command list (see `write_geometry_region`).
    pub(crate) fn upload_mesh(
        &mut self,
        draw_idx: usize,
        vertices: &[Vertex],
        indices: &[u16],
        frame: u64,
    ) -> Result<(), String> {
        let obj = self
            .draw
            .objects
            .get(draw_idx)
            .ok_or_else(|| format!("upload_mesh: draw object {} out of range", draw_idx))?;
        let (vertex_count, index_count) = (obj.vertex_count, obj.index_count);
        if vertices.len() != vertex_count {
            return Err(format!(
                "upload_mesh: draw {} expects {} vertices, got {}",
                draw_idx,
                vertex_count,
                vertices.len()
            ));
        }
        if indices.len() != index_count {
            return Err(format!(
                "upload_mesh: draw {} expects {} indices, got {}",
                draw_idx,
                index_count,
                indices.len()
            ));
        }

        // Reclaim frees whose in-flight frames have retired, then place the
        // geometry. build_draw_list never emits a zero-length mesh, so an
        // empty allocation request is treated as a hard error.
        self.mesh_stream.vtx_alloc.reclaim(frame);
        self.mesh_stream.idx_alloc.reclaim(frame);
        let v_len = std::mem::size_of_val(vertices);
        // Static IB is u32 (the per-scene total can exceed u16); per-mesh
        // indices come in as u16 (each mesh fits in u16, enforced by the
        // build-time splitter) and get widened on write below. Size the
        // allocation against the u32 stride. Mirrors metal's upload_mesh.
        let i_len = indices.len() * std::mem::size_of::<u32>();
        let v_off = self
            .mesh_stream
            .vtx_alloc
            .alloc(v_len as u64)
            .ok_or_else(|| {
                format!(
                    "upload_mesh: draw {}: no free vertex space for {} bytes",
                    draw_idx, v_len
                )
            })? as usize;
        let i_off = match self.mesh_stream.idx_alloc.alloc(i_len as u64) {
            Some(o) => o as usize,
            None => {
                // hand the vertex region back so a half-failed upload leaks no
                // space (frame 0: it was never written or drawn)
                self.mesh_stream
                    .vtx_alloc
                    .free(v_off as u64, v_len as u64, 0);
                return Err(format!(
                    "upload_mesh: draw {}: no free index space for {} bytes",
                    draw_idx, i_len
                ));
            }
        };

        self.wait_idle();

        // Vertices copy verbatim. Indices are mesh-relative, so rebase them to
        // the vertex region the allocator chose: v_off is always a multiple of
        // size_of::<Vertex>() (every seed region and allocation is), so the
        // base is an exact vertex index.
        let vert_bytes = bytemuck::cast_slice(vertices);
        self.write_geometry_region(
            &self.geometry.vertex_buffer,
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            v_off as u64,
            vert_bytes,
        )?;
        let base = (v_off / std::mem::size_of::<Vertex>()) as u32;
        // Widen u16 → u32 while rebasing onto the chosen vertex region.
        let rebased: Vec<u32> = indices.iter().map(|&i| u32::from(i) + base).collect();
        let idx_bytes = bytemuck::cast_slice(&rebased);
        self.write_geometry_region(
            &self.geometry.index_buffer,
            D3D12_RESOURCE_STATE_INDEX_BUFFER,
            i_off as u64,
            idx_bytes,
        )?;

        let obj = &mut self.draw.objects[draw_idx];
        obj.vertex_offset = v_off;
        obj.index_offset = i_off / std::mem::size_of::<u32>();
        obj.resident = true;
        // The mesh joins the RT-relevant draw set at a freshly allocated region;
        // the next RT update builds its BLAS over the new slice.
        self.rt_topology_dirty = true;
        Ok(())
    }

    // Overwrite a `Mesh` draw slot's vertex / index data in place. Driven by
    // asset hot-reload (`cn debug` only). New `vertices` / `indices` are
    // written at the draw object's existing offsets in the shared vertex /
    // index buffers, so the slot's count must match init-time; size-changing
    // reloads need `rebuild_static_geometry`, not this call. Each entry in
    // `lod_alternates` is written to the matching slot's pre-allocated LOD
    // region; LOD counts and per-LOD index counts must match init-time too.
    // Per-LOD `switch_distance`s are re-stored so JSON-side tweaks to
    // `lod_distances` propagate without a process restart. `wait_idle` is
    // folded into each `write_geometry_region` call (the whole-resource
    // COPY_DEST transition needs no in-flight command list referencing the
    // buffer). Mirrors `MtlContext::update_mesh_geometry`.
    pub(crate) fn update_mesh_geometry(
        &mut self,
        draw_idx: usize,
        vertices: &[Vertex],
        indices: &[u16],
        lod_alternates: &[(f32, Vec<u16>)],
    ) -> Result<(), String> {
        let obj = self.draw.objects.get(draw_idx).ok_or_else(|| {
            format!(
                "update_mesh_geometry: draw object {} out of range",
                draw_idx
            )
        })?;
        if vertices.len() != obj.vertex_count {
            return Err(format!(
                "update_mesh_geometry: draw {} expects {} vertices, got {} \
                 (in-place path is size-matched only; size changes route through \
                 rebuild_static_geometry)",
                draw_idx,
                obj.vertex_count,
                vertices.len()
            ));
        }
        if indices.len() != obj.index_count {
            return Err(format!(
                "update_mesh_geometry: draw {} expects {} indices, got {} \
                 (in-place path is size-matched only; size changes route through \
                 rebuild_static_geometry)",
                draw_idx,
                obj.index_count,
                indices.len()
            ));
        }
        if lod_alternates.len() != obj.lod_alternates.len() {
            return Err(format!(
                "update_mesh_geometry: draw {} expects {} LOD alternate(s), got {} \
                 (LOD-count changes need rebuild_static_geometry)",
                draw_idx,
                obj.lod_alternates.len(),
                lod_alternates.len()
            ));
        }
        for (lod_idx, ((_, alt_idx), slice)) in lod_alternates
            .iter()
            .zip(obj.lod_alternates.iter())
            .enumerate()
        {
            if alt_idx.len() != slice.index_count {
                return Err(format!(
                    "update_mesh_geometry: draw {} LOD{} expects {} indices, got {} \
                     (LOD size changes need rebuild_static_geometry)",
                    draw_idx,
                    lod_idx + 1,
                    slice.index_count,
                    alt_idx.len()
                ));
            }
        }
        let v_off = obj.vertex_offset as u64;
        let i_off_bytes = (obj.index_offset * std::mem::size_of::<u32>()) as u64;
        // Static draws keep indices absolute (base_vertex == 0), so rebase
        // mesh-relative u16 indices onto the slot's vertex_offset and widen to
        // u32 before writing, matching the shared u32 index buffer and the
        // streaming upload_mesh path. `v_off` is always a multiple of
        // size_of::<Vertex>() (every region build_draw_list emits starts on a
        // vertex boundary).
        let base = (obj.vertex_offset / std::mem::size_of::<Vertex>()) as u32;
        let lod_byte_offsets: Vec<u64> = obj
            .lod_alternates
            .iter()
            .map(|s| (s.index_offset * std::mem::size_of::<u32>()) as u64)
            .collect();

        self.wait_idle();

        let vert_bytes = bytemuck::cast_slice(vertices);
        self.write_geometry_region(
            &self.geometry.vertex_buffer,
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            v_off,
            vert_bytes,
        )?;
        let rebased: Vec<u32> = indices.iter().map(|&i| u32::from(i) + base).collect();
        let idx_bytes = bytemuck::cast_slice(&rebased);
        self.write_geometry_region(
            &self.geometry.index_buffer,
            D3D12_RESOURCE_STATE_INDEX_BUFFER,
            i_off_bytes,
            idx_bytes,
        )?;
        // LOD alternate slots were laid out at init alongside LOD0 in the
        // same shared index buffer. Each alternate shares LOD0's vertex
        // region (LOD decimation never touches vertices), so rebase onto the
        // same `base`.
        for ((_, alt_idx), &alt_off_bytes) in lod_alternates.iter().zip(lod_byte_offsets.iter()) {
            let alt_rebased: Vec<u32> = alt_idx.iter().map(|&i| u32::from(i) + base).collect();
            let alt_bytes = bytemuck::cast_slice(&alt_rebased);
            self.write_geometry_region(
                &self.geometry.index_buffer,
                D3D12_RESOURCE_STATE_INDEX_BUFFER,
                alt_off_bytes,
                alt_bytes,
            )?;
        }
        // Refresh the per-LOD switch distances so JSON-side tweaks to
        // `lod_distances` propagate without a process restart.
        let slot = &mut self.draw.objects[draw_idx];
        for ((switch_distance, _), slice) in
            lod_alternates.iter().zip(slot.lod_alternates.iter_mut())
        {
            slice.switch_distance = *switch_distance;
        }
        // The slot now holds different triangles at the same offsets, so its RT
        // BLAS traces the pre-reload positions. Nothing else in the geometry
        // signature moved, so bump the generation (which the signature carries)
        // and flag the topology: the next RT update rebuilds this slot's BLAS
        // rather than reusing the stale one.
        slot.geometry_generation = slot.geometry_generation.wrapping_add(1);
        self.rt_topology_dirty = true;
        Ok(())
    }

    // Return a streamed mesh's geometry region to the sub-allocators and mark
    // the draw non-resident so it is skipped in every pass.
    //
    // `retire_frame` is the frame from which the freed region may be reused:
    // pass `current_frame + frames_in_flight` for a runtime eviction so a
    // still-in-flight command list never has its region overwritten by a
    // later `upload_mesh`, and `0` at init, where nothing has been drawn.
    // The region is not zeroed: the draw leaves the RT-relevant set here, so
    // the next RT update retires its BLAS rather than tracing the vacated
    // bytes, and every raster pass skips a non-resident draw.
    pub(crate) fn evict_mesh(&mut self, draw_idx: usize, retire_frame: u64) -> Result<(), String> {
        let obj = self
            .draw
            .objects
            .get(draw_idx)
            .ok_or_else(|| format!("evict_mesh: draw object {} out of range", draw_idx))?;
        let v_off = obj.vertex_offset as u64;
        let v_len = (obj.vertex_count * std::mem::size_of::<Vertex>()) as u64;
        let i_off = (obj.index_offset * std::mem::size_of::<u32>()) as u64;
        let i_len = (obj.index_count * std::mem::size_of::<u32>()) as u64;
        self.mesh_stream.vtx_alloc.free(v_off, v_len, retire_frame);
        self.mesh_stream.idx_alloc.free(i_off, i_len, retire_frame);
        self.draw.objects[draw_idx].resident = false;
        // The mesh leaves the RT-relevant draw set; the next RT update drops its
        // BLAS (deferred-freed once in-flight traces retire).
        self.rt_topology_dirty = true;
        Ok(())
    }

    // Seed the streamed-mesh sub-allocators with one reserved headroom block
    // (byte ranges in the shared vertex / index buffers), for the
    // shrinkable-seed path.
    //
    // The streamed geometry is not baked into the buffers at build time;
    // instead the buffers carry one zeroed headroom region (sized to the
    // cap-many resident meshes) at these offsets, which `compact_for_streaming`
    // appended before init. `retire_frame 0`: nothing has been drawn yet, so
    // the space is allocatable immediately -- mirrors `setup_chunk_streaming`'s
    // seeding. From then on `upload_mesh` / `evict_mesh` place and free streamed
    // meshes within it. Mirrors `MtlContext::seed_mesh_streaming`.
    pub(crate) fn seed_mesh_streaming(
        &mut self,
        vtx_offset: u64,
        vtx_bytes: u64,
        idx_offset: u64,
        idx_bytes: u64,
    ) {
        self.mesh_stream.vtx_alloc.free(vtx_offset, vtx_bytes, 0);
        self.mesh_stream.vtx_alloc.reclaim(0);
        self.mesh_stream.idx_alloc.free(idx_offset, idx_bytes, 0);
        self.mesh_stream.idx_alloc.reclaim(0);
    }

    // Grow the shared vertex/index buffers by a headroom region for streamed
    // `VoxelWorld` chunks and seed the chunk sub-allocators with it. The chunk
    // material's texture slots ride each chunk's cull record.
    //
    // Called once at init by `GraphicsSystem` when a `VoxelWorld` is present.
    // The build-time geometry is copied verbatim into the start of the new
    // (larger) DEFAULT-heap buffers; chunks are placed in the appended
    // headroom by `add_chunk_mesh`. This runs before the first frame, so no
    // in-flight command list references the replaced buffers.
    pub(crate) fn setup_chunk_streaming(
        &mut self,
        chunk_vtx_bytes: usize,
        chunk_idx_bytes: usize,
    ) -> Result<(), String> {
        self.wait_idle();
        let old_v_len = self.geometry.vertex_buffer_view.SizeInBytes as u64;
        let old_i_len = self.geometry.index_buffer_view.SizeInBytes as u64;
        let new_v_len = old_v_len + chunk_vtx_bytes as u64;
        let new_i_len = old_i_len + chunk_idx_bytes as u64;

        // Buffers are created in COMMON; the CopyBufferRegion below implicitly
        // promotes the destination COMMON -> COPY_DEST.
        let new_vbuf = create_buffer(
            &self.alloc,
            new_v_len,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON,
        )?;
        let new_ibuf = create_buffer(
            &self.alloc,
            new_i_len,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON,
        )?;

        // Copy the build-time geometry into the start of the grown buffers so
        // every existing draw's offsets stay valid.
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        one_shot_submit(&self.device, &self.command_queue, |cmd| unsafe {
            let v_src = transition_barrier(
                &self.geometry.vertex_buffer,
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            let i_src = transition_barrier(
                &self.geometry.index_buffer,
                D3D12_RESOURCE_STATE_INDEX_BUFFER,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            cmd.ResourceBarrier(&[v_src, i_src]);
            cmd.CopyBufferRegion(&*new_vbuf, 0, &*self.geometry.vertex_buffer, 0, old_v_len);
            cmd.CopyBufferRegion(&*new_ibuf, 0, &*self.geometry.index_buffer, 0, old_i_len);
            let v_dst = transition_barrier(
                &new_vbuf,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            );
            let i_dst = transition_barrier(
                &new_ibuf,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_INDEX_BUFFER,
            );
            cmd.ResourceBarrier(&[v_dst, i_dst]);
        })?;

        self.geometry.vertex_buffer_view = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: com::gpu_va(&new_vbuf),
            SizeInBytes: new_v_len as u32,
            StrideInBytes: std::mem::size_of::<Vertex>() as u32,
        };
        self.geometry.index_buffer_view = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: com::gpu_va(&new_ibuf),
            SizeInBytes: new_i_len as u32,
            // Static IB is u32 (matches the `Format` chosen in init/mod.rs).
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT,
        };
        self.geometry.vertex_buffer = new_vbuf;
        self.geometry.index_buffer = new_ibuf;

        // Seed the chunk allocators with the appended headroom. retire_frame 0:
        // nothing has been drawn, so the space is reusable immediately.
        self.chunk_stream
            .vtx_alloc
            .free(old_v_len, chunk_vtx_bytes as u64, 0);
        self.chunk_stream
            .idx_alloc
            .free(old_i_len, chunk_idx_bytes as u64, 0);
        Ok(())
    }

    // Place one streamed chunk's geometry in the chunk headroom region and
    // write its `DrawObject` at the engine-allocated destination slot.
    //
    // The chunk is non-cullable and joins the `draw.always` set: the streaming
    // window already bounds the resident chunk count. Indices stay
    // mesh-relative (0-based) and the draw passes the vertex region's base as
    // `base_vertex`, so a chunk placed past the 65 535-vertex `u16` index
    // range still renders. `frame` reclaims retired deferred frees first.
    // `wait_idle` runs before the geometry copy so the whole-resource
    // COPY_DEST transition races no in-flight command list.
    pub(crate) fn add_chunk_mesh(
        &mut self,
        mesh: ChunkMesh<'_>,
        dst: crate::gfx::draw_slot::SlotAlloc,
    ) -> crate::gfx::error::RenderResult<()> {
        let ChunkMesh {
            verts: vertices,
            idxs: indices,
            model,
            texture_slot,
            normal_map_slot,
            material,
            frame,
        } = mesh;
        if vertices.is_empty() || indices.is_empty() {
            return Err("add_chunk_mesh: empty chunk geometry".into());
        }
        self.chunk_stream.vtx_alloc.reclaim(frame);
        self.chunk_stream.idx_alloc.reclaim(frame);

        let v_len = std::mem::size_of_val(vertices);
        // Static IB is u32; chunk indices come in as u16 and get widened on
        // write. Size the allocation against the u32 stride.
        let i_len = indices.len() * std::mem::size_of::<u32>();
        let v_off = self
            .chunk_stream
            .vtx_alloc
            .alloc(v_len as u64)
            .ok_or_else(|| {
                crate::gfx::error::RenderError::OutOfDeviceMemory(format!(
                    "add_chunk_mesh: no free chunk vertex space for {} bytes",
                    v_len
                ))
            })? as usize;
        let i_off = match self.chunk_stream.idx_alloc.alloc(i_len as u64) {
            Some(o) => o as usize,
            None => {
                self.chunk_stream
                    .vtx_alloc
                    .free(v_off as u64, v_len as u64, 0);
                return Err(crate::gfx::error::RenderError::OutOfDeviceMemory(format!(
                    "add_chunk_mesh: no free chunk index space for {} bytes",
                    i_len
                )));
            }
        };

        self.wait_idle();

        // Vertices and indices both copy verbatim: the indices stay 0-based and
        // the draw fixes them up with `base_vertex`.
        let vert_bytes = bytemuck::cast_slice(vertices);
        self.write_geometry_region(
            &self.geometry.vertex_buffer,
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            v_off as u64,
            vert_bytes,
        )?;
        // Chunk indices stay mesh-relative; the draw fixes them up with
        // `base_vertex`. Widen u16 → u32 to match the static IB's stride.
        let widened: Vec<u32> = indices.iter().map(|&i| u32::from(i)).collect();
        let idx_bytes = bytemuck::cast_slice(&widened);
        self.write_geometry_region(
            &self.geometry.index_buffer,
            D3D12_RESOURCE_STATE_INDEX_BUFFER,
            i_off as u64,
            idx_bytes,
        )?;

        // v_off is a multiple of size_of::<Vertex>() (the headroom start and
        // every alloc are), so the base is an exact vertex index.
        let base_vertex = (v_off / std::mem::size_of::<Vertex>()) as i32;
        let obj = DrawObject {
            vertex_offset: v_off,
            vertex_count: vertices.len(),
            index_offset: i_off / std::mem::size_of::<u32>(),
            index_count: indices.len(),
            base_vertex,
            geometry_generation: 0,
            model,
            texture_slot,
            normal_map_slot,
            material,
            visible: true,
            resident: true,
            // Non-cullable: degenerate AABB disables frustum/distance culling.
            bb_min: [f32::NAN; 3],
            bb_max: [f32::NAN; 3],
            cull_distance: 0.0,
            // Streamed chunks always render at the build-time mesh; no LOD.
            lod_alternates: Vec::new(),
            // Streamed chunks render through the world default program.
            shader_bucket: 0,
        };

        // Write at the engine-allocated destination slot.
        let draw_idx = match dst {
            crate::gfx::draw_slot::SlotAlloc::Reuse(slot) => {
                self.draw.objects[slot] = obj;
                slot
            }
            crate::gfx::draw_slot::SlotAlloc::Append(slot) => {
                debug_assert_eq!(
                    slot,
                    self.draw.objects.len(),
                    "appended draw slot must match the draw-object count"
                );
                self.draw.objects.push(obj);
                slot
            }
        };
        // Seed the G-buffer pre-pass's previous-model snapshot for a recycled
        // slot so a fresh chunk does not inherit the removed chunk's transform
        // and ghost for one frame. A fresh append is past the snapshot's end
        // and the pre-pass falls back to the current model itself.
        if let Some(gbuffer) = &self.gbuffer {
            let mut prev = gbuffer.prev_models.borrow_mut();
            if draw_idx < prev.len() {
                prev[draw_idx] = model;
            }
        }
        // A new resident chunk changes the RT-relevant draw set; the next RT
        // update folds it into the BVH (building just this chunk's BLAS).
        self.rt_topology_dirty = true;
        Ok(())
    }

    // Free a streamed chunk's geometry region and retire its `DrawObject`
    // slot for reuse.
    //
    // `retire_frame` is `current_frame + frames_in_flight` so an in-flight
    // command list never has the freed region overwritten by a later
    // `add_chunk_mesh`. The slot stays in `draw.objects` / `draw.always` but
    // is marked non-resident and invisible, so every pass skips it. The region
    // is not zeroed -- a non-resident draw is skipped everywhere and an
    // `alloc` hands back exactly `size` bytes that `add_chunk_mesh` fully
    // overwrites.
    pub(crate) fn remove_chunk_mesh(
        &mut self,
        draw_idx: usize,
        retire_frame: u64,
    ) -> Result<(), String> {
        let obj =
            self.draw.objects.get(draw_idx).ok_or_else(|| {
                format!("remove_chunk_mesh: draw object {} out of range", draw_idx)
            })?;
        let v_off = obj.vertex_offset as u64;
        let v_len = (obj.vertex_count * std::mem::size_of::<Vertex>()) as u64;
        let i_off = (obj.index_offset * std::mem::size_of::<u32>()) as u64;
        let i_len = (obj.index_count * std::mem::size_of::<u32>()) as u64;
        self.chunk_stream.vtx_alloc.free(v_off, v_len, retire_frame);
        self.chunk_stream.idx_alloc.free(i_off, i_len, retire_frame);
        let obj = &mut self.draw.objects[draw_idx];
        obj.visible = false;
        obj.resident = false;
        // The removed chunk leaves the RT-relevant draw set; the next RT update
        // drops its BLAS (deferred-freed once in-flight traces retire).
        self.rt_topology_dirty = true;
        Ok(())
    }

    // Rewrite a resident chunk's model matrix.
    //
    // Used by camera-relative rendering: when the camera crosses into a new
    // chunk the render origin follows it, so every resident chunk is rebased
    // onto the new origin. Only the model matrix changes -- the geometry stays
    // where it was uploaded.
    pub(crate) fn set_chunk_model(
        &mut self,
        draw_idx: usize,
        model: [[f32; 4]; 4],
    ) -> Result<(), String> {
        let obj = self
            .draw
            .objects
            .get_mut(draw_idx)
            .ok_or_else(|| format!("set_chunk_model: draw object {} out of range", draw_idx))?;
        obj.model = model;
        Ok(())
    }
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
            crate::gfx::rt_geom::skinned_index_buffer_bytes(indices.len()) as u64,
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
        // (i.e. its bind-pose position) instead of reading uninitialised
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
        self.skinned.joint_matrices = draw_objects
            .iter()
            .map(|o| vec![IDENTITY; o.joint_count.max(1)])
            .collect();

        self.skinned.shadow_pso = skinned_shadow_pso;
        self.skinned.shadow_root_sig = skinned_shadow_root_sig;
        self.skinned.vertex_buffer = Some(skinned_vertex_buffer);
        self.skinned.index_buffer = Some(skinned_index_buffer);
        self.skinned.joint_buffers = joint_buffers;
        self.skinned.joint_ptrs = joint_ptrs;
        self.skinned.draw_objects = draw_objects;

        // Morph targets are attached by a later `upload_skinned_morphs`; until
        // then every object is morphless (a re-upload / hot-reload resets here).
        let n_objects = self.skinned.draw_objects.len();
        self.skinned.morph_delta_buffers = (0..n_objects).map(|_| None).collect();
        self.skinned.morph_target_counts = vec![0; n_objects];
        self.skinned.morph_weights = vec![Vec::new(); n_objects];
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
            let skin =
                super::raytrace::build_rt_skin_pipeline(&self.device, self.hot_reload.enabled)
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
            self.draw.n_skinned = self.skinned.draw_objects.len();
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
        // region or a neighbouring slot would be overwritten.
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

    // Update a skinned slot's joint count and resize its per-slot CPU
    // joint-matrix buffer to match. Driven by asset hot-reload (`cn debug`
    // only) when a re-imported `.glb`'s skeleton has a different joint
    // count than the slot was initialised with. New entries are seeded to
    // identity so the slot renders undeformed until the next
    // `update_skinned_pose` writes the new pose. The shared skinned
    // pipelines + per-frame GPU joint buffers stay untouched; the GPU
    // buffers are sized for `MAX_JOINTS` at init, so a joint-count change
    // only resizes the CPU-side `skinned_joint_matrices[skinned_index]`
    // Vec (and `SkinnedDrawObject.joint_count`); the next
    // `upload_joint_matrices` writes the new (capped at `MAX_JOINTS`)
    // count of matrices into the per-frame ring. The velocity pre-pass
    // reads the previous-frame pose from `(frame_idx + FRAMES - 1) %
    // FRAMES` of the same ring rather than a separate CPU mirror, so no
    // "prev" array needs resizing; joints past the previous pose's
    // length retain the init identity seed (or stale prior data) for one
    // post-reload frame and then catch up. Mirrors
    // `MtlContext::update_skinned_skeleton`.
    pub(crate) fn update_skinned_skeleton(
        &mut self,
        skinned_index: usize,
        new_joint_count: usize,
    ) -> Result<(), String> {
        let obj = self
            .skinned
            .draw_objects
            .get_mut(skinned_index)
            .ok_or_else(|| {
                format!(
                    "update_skinned_skeleton: skinned object {} out of range",
                    skinned_index
                )
            })?;
        let capped = new_joint_count.min(MAX_JOINTS);
        obj.joint_count = capped;
        let size = capped.max(1);
        if let Some(slot) = self.skinned.joint_matrices.get_mut(skinned_index) {
            slot.resize(size, IDENTITY);
        }
        Ok(())
    }

    // Replace the skinning matrices for one skinned object. Called each frame
    // from `GraphicsSystem` with the pose `AnimationSystem` computed. Out-of-
    // range indices are ignored.
    pub(crate) fn update_skinned_pose(&mut self, skinned_index: usize, matrices: &[[[f32; 4]; 4]]) {
        if let Some(slot) = self.skinned.joint_matrices.get_mut(skinned_index) {
            slot.clear();
            slot.extend_from_slice(matrices);
            if slot.is_empty() {
                slot.push(IDENTITY);
            }
        }
    }

    // Reveal the pre-reserved skinned instance at `instance_index` (the
    // engine's instance pool decided which): show it at `model` and reset its
    // joint palette to the bind pose so it does not flash its previous
    // occupant's last frame (the owning `SkeletonPose`'s first pose push
    // replaces it next frame). The copy's deformed region is already valid
    // because `encode_skin` folds every pre-reserved copy each frame. A no-op
    // if the index is out of range. Mirrors the Metal path.
    pub(crate) fn reveal_skinned_instance(&mut self, instance_index: usize, model: [[f32; 4]; 4]) {
        let Some(obj) = self.skinned.draw_objects.get_mut(instance_index) else {
            return;
        };
        obj.model = model;
        obj.visible = true;
        if let Some(palette) = self.skinned.joint_matrices.get_mut(instance_index) {
            palette.iter_mut().for_each(|m| *m = IDENTITY);
        }
    }

    // Hide a skinned object; the engine's instance pool recycles the slot. A
    // no-op if the index is out of range. Mirrors the Metal path.
    pub(crate) fn retire_skinned_draw_object(&mut self, skinned_index: usize) {
        if let Some(obj) = self.skinned.draw_objects.get_mut(skinned_index) {
            obj.visible = false;
        }
    }

    // Push the model-to-world matrices of the given skinned objects, one
    // `(skinned index, matrix)` entry per moved instance. The per-frame cull
    // records and the legacy skinned draw both read `obj.model` directly, so
    // this only writes the fields. Out-of-range indices have no effect.
    pub(crate) fn update_skinned_models(&mut self, updates: &[(u32, [[f32; 4]; 4])]) {
        for &(skinned_index, model) in updates {
            if let Some(obj) = self.skinned.draw_objects.get_mut(skinned_index as usize) {
                obj.model = model;
            }
        }
    }

    // Copy this frame's skinning matrices into the per-frame joint buffers.
    // Called from `record_frame` before the skinned shadow + main passes.
    pub(super) fn upload_joint_matrices(&self, frame_idx: usize) {
        let Some(frame_ptrs) = self.skinned.joint_ptrs.get(frame_idx) else {
            return;
        };
        for (i, mats) in self.skinned.joint_matrices.iter().enumerate() {
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
    pub(super) fn skinned_joint_gva(&self, frame_idx: usize, i: usize) -> u64 {
        com::gpu_va(&self.skinned.joint_buffers[frame_idx][i])
    }

    // Attach morph-target buffers (`PayloadMorphs::packed_words`) to the skinned
    // draw objects. `morphs[i]` pairs with draw object `i`; instance copies share
    // their template's `Arc`, so each unique entry set becomes one GPU buffer. Allocates the per-frame
    // weight upload buffers (one f32 per target per object) when any object
    // carries morphs. Called once after `upload_skinned`.
    pub(super) fn upload_skinned_morphs(
        &mut self,
        morphs: Vec<Option<std::sync::Arc<crate::gfx::mesh_payload::PayloadMorphs>>>,
    ) -> Result<(), String> {
        use std::collections::HashMap;

        let n = self.skinned.draw_objects.len();
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
        self.skinned.morph_weights = weights;
        self.skinned.morph_weight_buffers = weight_buffers;
        self.skinned.morph_weight_ptrs = weight_ptrs;
        Ok(())
    }

    // Replace one skinned object's morph weights. Out-of-range indices and
    // objects without morph targets are ignored; extra weights are dropped.
    pub(super) fn update_morph_weights(&mut self, skinned_index: usize, weights: &[f32]) {
        if let Some(slot) = self.skinned.morph_weights.get_mut(skinned_index) {
            for (i, w) in slot.iter_mut().enumerate() {
                *w = weights.get(i).copied().unwrap_or(0.0);
            }
        }
    }

    // Copy this frame's morph weights into the per-frame weight buffers. Called
    // from `record_frame` alongside `upload_joint_matrices`. A no-op when no
    // object carries morphs (the buffers are empty).
    pub(super) fn upload_morph_weights(&self, frame_idx: usize) {
        let Some(frame_ptrs) = self.skinned.morph_weight_ptrs.get(frame_idx) else {
            return;
        };
        for (i, w) in self.skinned.morph_weights.iter().enumerate() {
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
    pub(super) fn morph_weight_gva(&self, frame_idx: usize, i: usize) -> Option<u64> {
        let buf = self.skinned.morph_weight_buffers.get(frame_idx)?.get(i)?;
        Some(com::gpu_va(buf))
    }
}

// World-Shader runtime hot-swap (RenderBackend::update_world_shader_pipelines)

// cn-debug-only runtime-mutation surface; dead from the FFI lib crate's roots,
// live in the concinnity binary. See the note on the analogous block in
// [directx/particle.rs].
impl DxContext {
    // Rebuild bucket 0 of the GPU-driven main pass from a freshly compiled world
    // Shader and hot-swap it, for the live-reload path (`reload_shader_stages`
    // -> here). Buckets past 0 and the shadow, G-buffer and cull pipelines are
    // engine-internal or scene-owned and are not rebuilt here.
    //
    // The replacement is built first; a compile / PSO-create failure
    // early-returns with the live pipeline untouched, mirroring
    // `reload_shaders`. Mirrors `MtlContext::update_world_shader_pipelines`.
    pub(crate) fn update_world_shader_pipelines(
        &mut self,
        programs: &concinnity_core::components::ShaderPrograms,
    ) -> Result<(), String> {
        let new_main = self.build_world_main_pso(Some(programs), &self.bindless_main_shaders)?;
        // Drain the GPU before the swap releases the displaced PSO: a command
        // list does not keep one alive, and the debug reload drive does not
        // wait for us.
        self.wait_idle();
        self.cull.main_bindless_pso = Some(new_main);
        self.world_shader = Some(programs.clone());
        self.invalidate_wireframe_pipelines();
        Ok(())
    }

    // Bucket 0's PSO against the live bindless root signature: the world default
    // Shader's pair where `world` declares one, `engine_default` otherwise.
    // Errors when the GPU-driven pass is not live, which means the world has
    // nothing to draw.
    pub(super) fn build_world_main_pso(
        &self,
        world: Option<&concinnity_core::components::ShaderPrograms>,
        engine_default: &super::init::pipelines::BindlessMainShaders,
    ) -> Result<ID3D12PipelineState, String> {
        let root_sig = self
            .cull
            .main_bindless_root_sig
            .as_ref()
            .ok_or_else(|| "the GPU-driven main pass is not live".to_string())?;
        build_bucket_pipeline(
            &self.device,
            self.diagnostics.info_queue.as_ref(),
            BucketPipelineTargets {
                root_sig,
                msaa_samples: self.hdr.msaa_samples,
                engine_default,
                hot_reload: self.hot_reload.enabled,
            },
            0,
            crate::gfx::backend_init::WorldShader {
                programs: world,
                deferred: false,
            },
        )
    }
}
