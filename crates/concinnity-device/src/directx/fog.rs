//! Volumetric fog for the D3D12 backend. Frostbite-style froxel volume:
//!
//!   * The `fog_froxel_kernel` compute pass (`encode_fog_froxel`) populates a
//!     screen-aligned `(80 × 45 × 64)` 3D `RGBA16Float` volume of
//!     `(scattered_rgb, 1 - T)` across the view frustum. One thread per
//!     (x, y) tile; each thread walks Z front-to-back, accumulating the
//!     per-slab scatter + transmittance with a CSM shadow tap per slice.
//!
//!   * The fullscreen `Fog` render pass (`encode_fog`) samples the volume by
//!     `(screen_uv, view_z)` instead of marching per-pixel and composites
//!     `(scattered, 1 - T)` over the resolved HDR target with the standard
//!     `over` blend (`final = scene * T + scattered`).
//!
//! Runs between the projected-decal pass and the SSR resolve so the fog wraps
//! the decal-stamped scene and SSR reflects through it; TAA history then
//! reprojects the integrated fog color and transmittance.
//!
//! Mirrors src/metal/fog.rs.

use concinnity_core::math::SaturatingFrom;
use concinnity_engine::gfx::render_types::{FogFroxelParams, FogParams};
use concinnity_engine::render::backend::error::{RenderError, RenderResult};
use concinnity_engine::render::environment::volumetric_fog;
use concinnity_engine::render::frame::plan::FogBlocks;
use concinnity_engine::render::render_graph::{FOG_FROXEL_X, FOG_FROXEL_Y, FOG_FROXEL_Z};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::allocator::{DeviceAllocator, PooledBuffer};
use super::com;
use crate::directx::builtin_shaders;
use crate::directx::builtin_shaders::CompileProgram;
use crate::directx::context::{DxContext, align256, dump_on_err};
use crate::directx::descriptor_slot::{DescriptorTables, SrvSlot};
use crate::directx::error::map_hresult;
use crate::directx::pso::{Blend, GraphicsPso, Raster, compute_pso};
use crate::directx::root_sig::{RootSig, SamplerState, Visibility};
use crate::directx::texture::HDR_FORMAT;

// Volumetric fog. All fields `None`/default until the world declares a
// `VolumetricFog`; the fog pass is skipped while `resources` is `None`. The
// settings are cached so the per-frame encoder can build its `FogParams`
// without re-resolving the asset. The sun it lights with is the light state's.
pub(in crate::directx) struct FogState {
    pub resources: Option<FogResources>,
    pub settings: Option<volumetric_fog::FogSettings>,
}

// Compile the fog vertex + fragment shaders; the MSAA define keeps the
// fragment shader's depth SRV declaration in sync with the resource's
// sample count. Used by [`FogResources::new`] at init and by shader hot-
// reload to rebuild the fog PSO.
pub(in crate::directx) fn compile_fog_shaders(
    msaa_samples: u32,
    hot_reload: bool,
) -> RenderResult<(Vec<u8>, Vec<u8>)> {
    let vs = builtin_shaders::FULLSCREEN_VERT.compile(hot_reload)?;
    let ps = builtin_shaders::FOG_FRAG
        .at(msaa_samples > 1)
        .compile(hot_reload)?;
    Ok((vs, ps))
}

// Compile the froxel-volume compute kernel.
pub(in crate::directx) fn compile_fog_froxel_shader(hot_reload: bool) -> RenderResult<Vec<u8>> {
    builtin_shaders::FOG_FROXEL.compile(hot_reload)
}

// Rebuild the fog PSO against fresh shader source. Called from the DirectX
// shader hot-reload pass; reuses the existing root signature.
pub(in crate::directx) fn rebuild_fog_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    msaa_samples: u32,
    hot_reload: bool,
    info_queue: Option<&ID3D12InfoQueue>,
) -> RenderResult<ID3D12PipelineState> {
    let (vs, ps) = compile_fog_shaders(msaa_samples, hot_reload)?;
    dump_on_err(info_queue, create_fog_pso(device, root_sig, &vs, &ps))
}

// Rebuild the froxel compute PSO against fresh shader source.
pub(in crate::directx) fn rebuild_fog_froxel_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    hot_reload: bool,
    info_queue: Option<&ID3D12InfoQueue>,
) -> RenderResult<ID3D12PipelineState> {
    let cs = compile_fog_froxel_shader(hot_reload)?;
    dump_on_err(info_queue, create_fog_froxel_pso(device, root_sig, &cs))
}

// Fog render-pass root signature:
//   [0] root CBV b0   FogParams         (per-frame)
//   [1] root CBV b1   FogFroxelParams   (per-frame)
//   [2] table  t0     scene depth SRV (Texture2D[MS]<float>)
//   [3] table  t1     froxel volume SRV (Texture3D<float4>)
// Static linear-clamp sampler s0 for the trilinear volume sample.
fn create_fog_root_signature(device: &ID3D12Device) -> RenderResult<ID3D12RootSignature> {
    // The fullscreen pass uses SV_VertexID; no input assembler is needed.
    RootSig::new()
        .cbv(0, Visibility::Pixel)
        .cbv(1, Visibility::Pixel)
        .srv_table(0, 1, Visibility::Pixel)
        .srv_table(1, 1, Visibility::Pixel)
        .static_sampler(
            SamplerState::LinearClampTransparentBorder,
            0,
            Visibility::Pixel,
        )
        .build(device, "fog root sig")
}

// Froxel compute root signature:
//   [0] root CBV b0   FogParams         (per-frame)
//   [1] root CBV b1   FogFroxelParams   (per-frame)
//   [2] root CBV b2   ShadowUniforms    (per-frame, shared with Main / Shadow)
//   [3] table  t0     shadow map SRV (Texture2DArray<float>)
//   [4] table  u0     froxel volume UAV (RWTexture3D<float4>)
// Static comparison sampler s0 for the shadow tap, matching the existing
// `shadow_sampler_gpu` static sampler. Clamp on every axis so cascades that
// fail their NDC bounds check fall back to 1.0 via the explicit
// `if (any(uv < 0.0)...)` guard in the kernel anyway.
fn create_fog_froxel_root_signature(device: &ID3D12Device) -> RenderResult<ID3D12RootSignature> {
    RootSig::new()
        .cbv(0, Visibility::All)
        .cbv(1, Visibility::All)
        .cbv(2, Visibility::All)
        .srv_table(0, 1, Visibility::All)
        .uav_table(0, 1, Visibility::All)
        .static_sampler(SamplerState::ShadowCompare, 0, Visibility::All)
        .build(device, "fog froxel root sig")
}

// PSO for the fog pass. Writes the resolved HDR target with `(scattered,
// 1 - T)` over scene blending: the fragment emits the in-scattered color
// at `1 - transmittance` alpha and the blend resolves to
// `scene * T + scattered`. No depth attachment; the shader handles the
// depth-based ray-length termination itself.
fn create_fog_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
    ps: &[u8],
) -> RenderResult<ID3D12PipelineState> {
    GraphicsPso::fullscreen(root_sig, vs, ps, HDR_FORMAT, Blend::PremultipliedOver)
        .raster(Raster {
            depth_clip: false,
            ..Raster::default()
        })
        .build(device, "fog")
}

// Compute PSO for the froxel kernel.
fn create_fog_froxel_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    cs: &[u8],
) -> RenderResult<ID3D12PipelineState> {
    compute_pso(device, root_sig, cs, "fog froxel")
}

// Create the 3D `RGBA16Float` froxel volume. Rests in `PIXEL_SHADER_RESOURCE`
// between frames: the graph's FogFroxel producer barrier transitions it to
// `UNORDERED_ACCESS` for the compute write, and the Fog consumer barrier returns
// it to `PIXEL_SHADER_RESOURCE` for the trilinear sample. Both transitions are
// graph-driven (no inline cross-frame reset); creating it sampled makes frame 0's
// producer barrier (sampled -> UAV) start from the resource's real state.
fn create_fog_froxel_volume(
    device: &ID3D12Device,
    uav_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> RenderResult<ID3D12Resource> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE3D,
        Width: u64::from(FOG_FROXEL_X),
        Height: FOG_FROXEL_Y,
        DepthOrArraySize: u16::saturating_from(FOG_FROXEL_Z),
        MipLevels: 1,
        Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut tex_opt: Option<ID3D12Resource> = None;
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    unsafe {
        device.CreateCommittedResource(
            &raw const heap_props,
            D3D12_HEAP_FLAG_NONE,
            &raw const desc,
            // Rest sampled; the graph drives both transitions (see the fn doc).
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            None,
            &raw mut tex_opt,
        )
    }
    .map_err(|e| map_hresult(e.code(), "create fog froxel volume"))?;
    let resource = tex_opt
        .ok_or_else(|| RenderError::Other("create fog froxel volume returned None".to_string()))?;

    let uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE3D,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Texture3D: D3D12_TEX3D_UAV {
                MipSlice: 0,
                FirstWSlice: 0,
                WSize: FOG_FROXEL_Z,
            },
        },
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe {
        device.CreateUnorderedAccessView(&resource, None, Some(&raw const uav_desc), uav_cpu);
    }

    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE3D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture3D: D3D12_TEX3D_SRV {
                MostDetailedMip: 0,
                MipLevels: 1,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe { device.CreateShaderResourceView(&resource, Some(&raw const srv_desc), srv_cpu) };

    Ok(resource)
}

// Owned by `DxContext` exactly when the world declared a `VolumetricFog`:
// the fog pipeline + per-frame uniform rings + the froxel volume + the
// kernel that populates it. The depth SRV the fog pass reads through t0
// is shared with the projected-decal pass.
pub(in crate::directx) struct FogResources {
    pub(in crate::directx) root_sig: ID3D12RootSignature,
    pub(in crate::directx) pso: ID3D12PipelineState,

    pub(in crate::directx) froxel_root_sig: ID3D12RootSignature,
    pub(in crate::directx) froxel_pso: ID3D12PipelineState,

    // Per-frame FogParams ring (176-byte block, persistently mapped).
    pub(in crate::directx) params_ubo_resources: Vec<PooledBuffer>,
    pub(in crate::directx) params_ubo_ptrs: Vec<*mut u8>,

    // Per-frame FogFroxelParams ring (96-byte block, persistently mapped).
    pub(in crate::directx) froxel_params_ubo_resources: Vec<PooledBuffer>,
    pub(in crate::directx) froxel_params_ubo_ptrs: Vec<*mut u8>,

    // 3D `RGBA16Float` volume the kernel writes and the fragment shader
    // samples. The handle backs the graph-driven UAV ↔ PIXEL_SHADER_RESOURCE
    // producer + consumer barriers (resolved by the executor's barrier
    // registry); the shader reads/writes go through the heap-stored UAV + SRV
    // descriptors.
    pub(in crate::directx) volume_resource: ID3D12Resource,
    pub(in crate::directx) volume_uav_gpu: SrvSlot,
    pub(in crate::directx) volume_srv_gpu: SrvSlot,

    // Heap GPU handle of the main-depth SRV. Bound at fog pass t0; the graph
    // declares the Fog pass's depth read, so the executor puts the resource in a
    // shader-resource state before this pass and restores DEPTH_WRITE at the end
    // of the frame.
    pub(in crate::directx) depth_srv_gpu: SrvSlot,

    // Heap GPU handle of the shadow map array SRV. Bound at the froxel
    // kernel's t0 so each slab can do a CSM tap. Shared with the rest of
    // the engine; the resource is transitioned to PIXEL_SHADER_RESOURCE
    // by `encode_shadow_pass` ahead of every later pass, including this one.
    pub(in crate::directx) shadow_srv_gpu: SrvSlot,
}

// CPU + GPU descriptor handles for the froxel volume: the compute kernel's UAV
// (write) and the fog pass's SRV (read).
#[derive(Clone, Copy)]
pub(in crate::directx) struct FogVolumeDescriptors {
    pub uav_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub uav_gpu: SrvSlot,
    pub srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub srv_gpu: SrvSlot,
}

// GPU descriptor handles for the scene depth + shadow map the fog shaders sample.
#[derive(Clone, Copy)]
pub(in crate::directx) struct FogShaderResourceHandles {
    pub depth_srv_gpu: SrvSlot,
    pub shadow_srv_gpu: SrvSlot,
}

// Device-level fog build config: MSAA sample count (compiled into the depth SRV
// declarations) and the shader hot-reload toggle.
#[derive(Clone, Copy)]
pub(in crate::directx) struct FogDeviceParams {
    pub msaa_samples: u32,
    pub hot_reload: bool,
}

impl FogResources {
    // Build the fog pipeline + per-frame uniform rings + the froxel volume
    // + the compute kernel. Called from `DxContext::new` only when the
    // world declared a `VolumetricFog`. The depth SRV is already written
    // into the heap by the decal-init path (the projected-decal pass
    // writes the main-depth SRV unconditionally so runtime `add_decal`
    // works from a world that started empty); the fog pass reuses the
    // same descriptor. `volume.uav_cpu` / `volume.srv_cpu` are dedicated
    // SRV-heap slots reserved by init for the froxel volume.
    pub(in crate::directx) fn new(
        alloc: &DeviceAllocator,
        volume: FogVolumeDescriptors,
        shader_resources: FogShaderResourceHandles,
        params: FogDeviceParams,
        info_queue: Option<&ID3D12InfoQueue>,
    ) -> RenderResult<Self> {
        let device = alloc.device();
        let FogVolumeDescriptors {
            uav_cpu: volume_uav_cpu,
            uav_gpu: volume_uav_gpu,
            srv_cpu: volume_srv_cpu,
            srv_gpu: volume_srv_gpu,
        } = volume;
        let FogShaderResourceHandles {
            depth_srv_gpu,
            shadow_srv_gpu,
        } = shader_resources;
        let FogDeviceParams {
            msaa_samples,
            hot_reload,
        } = params;
        let (vs, ps) = compile_fog_shaders(msaa_samples, hot_reload)?;
        let cs = compile_fog_froxel_shader(hot_reload)?;

        let root_sig = dump_on_err(info_queue, create_fog_root_signature(device))?;
        let pso = dump_on_err(info_queue, create_fog_pso(device, &root_sig, &vs, &ps))?;

        let froxel_root_sig = dump_on_err(info_queue, create_fog_froxel_root_signature(device))?;
        let froxel_pso = dump_on_err(
            info_queue,
            create_fog_froxel_pso(device, &froxel_root_sig, &cs),
        )?;

        let volume_resource = create_fog_froxel_volume(device, volume_uav_cpu, volume_srv_cpu)?;

        // Per-frame FogParams ring.
        let params_ubo_size = align256(size_of::<FogParams>() as u64);
        let frames = alloc.frames_in_flight();
        let mut params_ubo_resources: Vec<PooledBuffer> = Vec::with_capacity(frames);
        let mut params_ubo_ptrs: Vec<*mut u8> = Vec::with_capacity(frames);
        for _ in 0..frames {
            let buf = alloc.alloc_buffer(
                params_ubo_size,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
            // local that receives the mapping.
            unsafe { buf.Map(0, None, Some(&raw mut ptr)) }
                .map_err(|e| map_hresult(e.code(), "map fog params ubo"))?;
            params_ubo_ptrs.push(ptr.cast::<u8>());
            params_ubo_resources.push(buf);
        }

        // Per-frame FogFroxelParams ring.
        let froxel_ubo_size = align256(size_of::<FogFroxelParams>() as u64);
        let mut froxel_params_ubo_resources: Vec<PooledBuffer> = Vec::with_capacity(frames);
        let mut froxel_params_ubo_ptrs: Vec<*mut u8> = Vec::with_capacity(frames);
        for _ in 0..frames {
            let buf = alloc.alloc_buffer(
                froxel_ubo_size,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            // SAFETY: the resource is a live CPU-visible buffer, and the out-parameter is a live
            // local that receives the mapping.
            unsafe { buf.Map(0, None, Some(&raw mut ptr)) }
                .map_err(|e| map_hresult(e.code(), "map fog froxel params ubo"))?;
            froxel_params_ubo_ptrs.push(ptr.cast::<u8>());
            froxel_params_ubo_resources.push(buf);
        }

        Ok(Self {
            root_sig,
            pso,
            froxel_root_sig,
            froxel_pso,
            params_ubo_resources,
            params_ubo_ptrs,
            froxel_params_ubo_resources,
            froxel_params_ubo_ptrs,
            volume_resource,
            volume_uav_gpu,
            volume_srv_gpu,
            depth_srv_gpu,
            shadow_srv_gpu,
        })
    }
}

impl DxContext {
    // Hot-reload entry point for the volumetric-fog tunables. Writes the new
    // `Option<FogSettings>` into `self.fog.settings`; the next frame's
    // `encode_fog_froxel` + `encode_fog` re-derive `FogParams` /
    // `FogFroxelParams` from it (both bail to a no-op when it is `None`, so a
    // `None` here disables the pass). Mirrors `MtlContext::update_fog_settings`.
    //
    // If the world started with no `VolumetricFog` (so `self.fog.resources` is
    // `None` and the froxel + fog PSOs were never built), a `Some` update logs
    // once and is dropped: re-enabling fog mid-run requires a relaunch.
    //
    // The only caller is the bin-only `cn run -f` world hot-reload
    // (`debug::hot_reload::passes`), reached through the `RenderBackend`
    // vtable. Mirrors the other bin-only runtime-mutation seams on this
    // backend.
    pub(crate) fn update_fog_settings(&mut self, settings: Option<volumetric_fog::FogSettings>) {
        if settings.is_some() && self.fog.resources.is_none() {
            tracing::warn!(
                "VolumetricFog hot-reload: world started without fog, so the fog \
                 pipeline was never built: re-enabling fog mid-run is not \
                 supported (relaunch required). Ignoring update."
            );
            return;
        }
        self.fog.settings = settings;
    }

    // Encode the volumetric-fog froxel-volume compute pass. Populates the
    // 3D `(scattered, 1 - T)` volume the fog fragment shader samples. The
    // caller (`execute_graph`) seeds this PassId before `Fog` so the RAW
    // edge in the shared graph orders the dispatch correctly.
    pub(in crate::directx) fn encode_fog_froxel(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        blocks: &FogBlocks,
        shadow_ubo_gva: u64,
    ) {
        let Some(fog) = &self.fog.resources else {
            return;
        };
        let FogBlocks {
            params,
            froxel: froxel_params,
        } = blocks;
        // Write per-frame `FogParams` + `FogFroxelParams` into their ring
        // slots. The `Fog` render pass below reads from the same slot, so
        // both passes see the same params this frame.
        // SAFETY: the destination is the persistent mapping of an UPLOAD-heap constant buffer that
        // init sized for `FogParams`, and the source is a separate live value, so the ranges
        // cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                std::ptr::from_ref::<FogParams>(params).cast::<u8>(),
                fog.params_ubo_ptrs[frame_idx],
                size_of::<FogParams>(),
            );
        }
        // SAFETY: the froxel ring slot is a separate UPLOAD-heap mapping sized for
        // `FogFroxelParams`, and the source is a separate live value.
        unsafe {
            std::ptr::copy_nonoverlapping(
                std::ptr::from_ref::<FogFroxelParams>(froxel_params).cast::<u8>(),
                fog.froxel_params_ubo_ptrs[frame_idx],
                size_of::<FogFroxelParams>(),
            );
        }
        let params_gva = com::gpu_va(&fog.params_ubo_resources[frame_idx]);
        let froxel_params_gva = com::gpu_va(&fog.froxel_params_ubo_resources[frame_idx]);

        // The shadow map's cascade tap is a declared read of this pass, so the
        // Shadow consumer barrier's stage union already carries the non-pixel
        // shader-resource state the compute kernel needs. The volume stays in
        // `UNORDERED_ACCESS`; the graph's Fog consumer barrier is what returns it
        // to a shader-resource state for the render pass.

        // SAFETY: the command list is recording, and the froxel root signature is owned by `fog`.
        unsafe { cmd.SetComputeRootSignature(&fog.froxel_root_sig) };
        // SAFETY: the froxel PSO is owned by `fog` and matches the root signature above.
        unsafe { cmd.SetPipelineState(&fog.froxel_pso) };
        // SAFETY: the descriptor heap is owned by `self.descriptors` and outlives the frame.
        unsafe { cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]) };
        // SAFETY: `params_gva` addresses this frame's `FogParams` ring slot written above.
        unsafe { cmd.SetComputeRootConstantBufferView(0, params_gva) };
        // SAFETY: `froxel_params_gva` addresses this frame's `FogFroxelParams` ring slot.
        unsafe { cmd.SetComputeRootConstantBufferView(1, froxel_params_gva) };
        // SAFETY: `shadow_ubo_gva` addresses this frame's live shadow constant buffer.
        unsafe { cmd.SetComputeRootConstantBufferView(2, shadow_ubo_gva) };
        // SAFETY: the shadow map's SRV lives in the heap set above, already in a non-pixel
        // shader-resource state.
        unsafe { cmd.set_compute_srv_table(3, fog.shadow_srv_gpu) };
        // SAFETY: the froxel volume's UAV lives in that heap, and the volume is in
        // UNORDERED_ACCESS.
        unsafe { cmd.set_compute_srv_table(4, fog.volume_uav_gpu) };

        // 8×8 threadgroups, one thread per (x, y) froxel.
        let groups_x = FOG_FROXEL_X.div_ceil(8);
        let groups_y = FOG_FROXEL_Y.div_ceil(8);
        // SAFETY: the froxel pipeline and every root argument it reads are bound above.
        unsafe { cmd.Dispatch(groups_x, groups_y, 1) };

        // The shadow map stays in the state the graph's read run put it in; the
        // executor's end-of-frame restore returns it to resting.
    }

    // Encode the volumetric-fog pass. Samples the 3D froxel volume the
    // `FogFroxel` compute pass populated this frame. Caller has already
    // ended the main HDR pass + the projected-decal pass (if any), so
    // `depth.resource` (MSAA when MSAA is on) holds the scene depth and
    // the resolved scene target holds the resolved scene + decal color.
    // The pass alpha-blends `(scattered, 1 - T)` over the resolved HDR
    // target.
    pub(in crate::directx) fn encode_fog(&self, cmd: &ID3D12GraphicsCommandList, frame_idx: usize) {
        let Some(fog) = &self.fog.resources else {
            return;
        };

        // `FogParams` / `FogFroxelParams` were uploaded by `encode_fog_froxel`
        // for this frame's slot, so we only read their GVAs here.
        let params_gva = com::gpu_va(&fog.params_ubo_resources[frame_idx]);
        let froxel_params_gva = com::gpu_va(&fog.froxel_params_ubo_resources[frame_idx]);

        // Main depth and the froxel volume are both graph resources, so both
        // transitions are already in place: the executor emits this pass's depth
        // consumer barrier and the froxel volume's UNORDERED_ACCESS →
        // PIXEL_SHADER_RESOURCE close before this command list.

        // The scene spine is a graph resource: this pass declares its
        // read-modify-write, so the executor has already put it in RENDER_TARGET.
        let scene_rtv = self.hdr_scene_rtv();

        let w = self.targets.extent.render_width;
        let h = self.targets.extent.render_height;
        let vp_state = D3D12_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: w as f32,
            Height: h as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        let scissor = RECT {
            left: 0,
            top: 0,
            right: i32::saturating_from(w),
            bottom: i32::saturating_from(h),
        };
        // SAFETY: the command list is recording, and the executor has put the scene behind
        // `scene_rtv` in RENDER_TARGET.
        unsafe { cmd.OMSetRenderTargets(1, Some(&raw const scene_rtv), false, None) };
        // SAFETY: the viewport slice is a local that outlives the call.
        unsafe { cmd.RSSetViewports(&[vp_state]) };
        // SAFETY: the scissor slice is a local that outlives the call.
        unsafe { cmd.RSSetScissorRects(&[scissor]) };
        // SAFETY: setting the topology only needs a recording command list.
        unsafe {
            cmd.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
        }

        // SAFETY: the fog PSO is owned by `fog` and outlives the recording.
        unsafe { cmd.SetPipelineState(&fog.pso) };
        // SAFETY: the fog root signature is owned by `fog` and matches the PSO above.
        unsafe { cmd.SetGraphicsRootSignature(&fog.root_sig) };
        // SAFETY: the descriptor heap is owned by `self.descriptors` and outlives the frame.
        unsafe { cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]) };
        // SAFETY: `params_gva` addresses the `FogParams` slot `encode_fog_froxel` wrote this frame.
        unsafe { cmd.SetGraphicsRootConstantBufferView(0, params_gva) };
        // SAFETY: `froxel_params_gva` addresses this frame's live `FogFroxelParams` slot.
        unsafe { cmd.SetGraphicsRootConstantBufferView(1, froxel_params_gva) };
        // SAFETY: the depth SRV lives in the heap set above, and the depth is in a
        // shader-resource state for this pass.
        unsafe { cmd.set_graphics_srv_table(2, fog.depth_srv_gpu) };
        // SAFETY: the froxel volume SRV lives in that heap, and the executor closed the volume to
        // PIXEL_SHADER_RESOURCE.
        unsafe { cmd.set_graphics_srv_table(3, fog.volume_srv_gpu) };
        // SAFETY: the target, pipeline and every root argument the draw reads are bound above.
        unsafe { cmd.DrawInstanced(3, 1, 0, 0) };

        // The scene spine, main depth and the froxel volume are all graph
        // resources; nothing here is left to restore.
    }
}
