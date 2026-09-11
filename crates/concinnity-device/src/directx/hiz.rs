// src/directx/hiz.rs
//
// Hi-Z (depth-mip pyramid) build pass used by the GPU-cull compute kernel for
// occlusion culling. Each frame, after the main depth buffer has been written
// by the graph, we copy/reduce it into a Texture2D mip chain (R32_FLOAT, MAX
// reduction). The *next* frame's `Cull` pass projects each `DrawObject` AABB
// through the previous frame's view-projection, picks the Hi-Z mip whose
// texels are roughly the size of the projected rect, and culls the AABB when
// its nearest projected depth is behind the rasterized occluder depth.
//
// Three compute kernels build it (see `src/shaders/hiz_build.slang`):
//
//   * `hiz_spd_single`: reduce a single-sample main depth into mips 0..6.
//   * `hiz_spd_msaa`  : the same for an MSAA main depth, taking the MAX over
//                       every sample so the result is conservative.
//   * `hiz_spd_tail`  : continue from mip 6 into mips 7..12.
//
// Each workgroup reduces a 64x64 tile through seven levels, so the whole
// pyramid is two dispatches with one barrier between them rather than one
// dispatch and one barrier per mip. `core::render::hiz_spd::Plan` decides the
// dispatch geometry; Vulkan builds its pyramid from the same plan.
//
// The pyramid is *not* a graph node; it runs inline on the outer "end" cmd
// list after `execute_graph` returns (see `directx/draw/mod.rs`). Treating
// it as an end-of-frame action keeps it off the graph's RMW chain on the
// main depth attachment (decals, fog, and SSAO/SSR pre-passes already share
// that target).

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::directx::com;
use crate::directx::context::dump_on_err;
use crate::directx::pipeline::serialize_desc_and_create;
use crate::directx::slang_builtins;
use crate::directx::slang_builtins::SlangCompile;
use crate::directx::texture::uav_barrier;

use concinnity_core::render::hiz_spd::{self, Plan};
use concinnity_core::render::uniforms::HizSpdParams;

// DWORD count of the `HizSpdParams` cbuffer.
const HIZ_PARAMS_DWORDS: u32 = (std::mem::size_of::<HizSpdParams>() / 4) as u32;

// UAV descriptors one SPD dispatch binds, one per level it can write.
const HIZ_SPD_UAVS: u32 = hiz_spd::LEVELS;

// Compute pipelines + texture + per-mip descriptors for the Hi-Z build. Built
// alongside the GPU-cull pipeline (same gating condition: bindless main pass
// active with build-time static geometry).
pub(super) struct HiZResources {
    pub(super) root_sig: ID3D12RootSignature,
    // The tail binds no depth source, so it drops the SRV table the phase-1
    // signature carries; a shader that never reads t0 would leave the table
    // unbound and the two cannot share one signature.
    pub(super) tail_root_sig: ID3D12RootSignature,
    pub(super) spd_single_pso: ID3D12PipelineState,
    pub(super) spd_msaa_pso: ID3D12PipelineState,
    pub(super) spd_tail_pso: ID3D12PipelineState,

    // R32_FLOAT 2D texture with a full mip chain. UAV-writable; the cull
    // kernel reads it via `Texture2D<float>.Load(int3(x, y, mip))`. Held
    // only to keep the resource alive; the per-mip CPU UAV handles below
    // reference it.
    pub(super) texture: ID3D12Resource,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) mip_count: u32,
    // Resting resource state between Hi-Z build pulses. The encoder
    // transitions to `UNORDERED_ACCESS` for the duration of the build and
    // back to this state afterwards so the next frame's cull dispatch can
    // sample it.
    pub(super) rest_state: D3D12_RESOURCE_STATES,

    // CPU descriptor handle for the SRV covering the whole mip chain. Used by
    // the GPU-cull kernel (4 corner Loads at a picked mip).
    pub(super) srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub(super) srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    // GPU descriptor handle of the depth-source SRV the init kernel binds at
    // t0 (the main-depth SRV the decal/fog passes also share).
    pub(super) depth_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    // Per-mip UAV CPU/GPU descriptor pairs. Length = `mip_count`.
    pub(super) mip_uav_cpus: Vec<D3D12_CPU_DESCRIPTOR_HANDLE>,
    pub(super) mip_uav_gpus: Vec<D3D12_GPU_DESCRIPTOR_HANDLE>,
}

// Compiled Hi-Z kernels: spd_single, spd_msaa, spd_tail bytecode.
type HizShaders = (Vec<u8>, Vec<u8>, Vec<u8>);

pub(in crate::directx) fn compile_hiz_shaders(hot_reload: bool) -> Result<HizShaders, String> {
    let single = slang_builtins::HIZ_SPD_SINGLE.compile(hot_reload)?;
    let msaa = slang_builtins::HIZ_SPD_MSAA.compile(hot_reload)?;
    let tail = slang_builtins::HIZ_SPD_TAIL.compile(hot_reload)?;
    Ok((single, msaa, tail))
}

// Root signatures for the two SPD dispatches. Both take the params as root
// constants at b0 and a table of `HIZ_SPD_UAVS` contiguous per-mip UAVs at
// u0..u6; phase 1 adds the main-depth SRV at t0 ahead of it. The per-mip UAVs
// sit contiguously in the heap, which is what lets one range cover a whole
// dispatch's levels: phase 1 bases its table on mip 0, the tail on mip 6.
fn hiz_root_params(
    srv_range: &D3D12_DESCRIPTOR_RANGE,
    uav_range: &D3D12_DESCRIPTOR_RANGE,
    with_srv: bool,
) -> Vec<D3D12_ROOT_PARAMETER> {
    let table = |range: &D3D12_DESCRIPTOR_RANGE| D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                NumDescriptorRanges: 1,
                pDescriptorRanges: range,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    };
    let mut params = vec![D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Constants: D3D12_ROOT_CONSTANTS {
                ShaderRegister: 0,
                RegisterSpace: 0,
                Num32BitValues: HIZ_PARAMS_DWORDS,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }];
    if with_srv {
        params.push(table(srv_range));
    }
    params.push(table(uav_range));
    params
}

pub(in crate::directx) fn create_hiz_root_signature(
    device: &ID3D12Device,
) -> Result<ID3D12RootSignature, String> {
    create_hiz_signature(device, true, "hiz spd root sig")
}

pub(in crate::directx) fn create_hiz_tail_root_signature(
    device: &ID3D12Device,
) -> Result<ID3D12RootSignature, String> {
    create_hiz_signature(device, false, "hiz spd tail root sig")
}

fn create_hiz_signature(
    device: &ID3D12Device,
    with_srv: bool,
    label: &str,
) -> Result<ID3D12RootSignature, String> {
    let srv_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 0, // t0
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let uav_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
        NumDescriptors: HIZ_SPD_UAVS,
        BaseShaderRegister: 0, // u0..u6
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let params = hiz_root_params(&srv_range, &uav_range, with_srv);
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
        ..Default::default()
    };
    serialize_desc_and_create(device, &desc, label)
}

fn create_hiz_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    cs: &[u8],
    label: &str,
) -> Result<ID3D12PipelineState, String> {
    let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: com::borrowed(root_sig),
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: cs.as_ptr() as _,
            BytecodeLength: cs.len(),
        },
        ..Default::default()
    };
    // SAFETY: `desc` outlives this synchronous call, and so do the root signature, shader bytecode
    // and input-element array whose raw pointers it borrows.
    unsafe { crate::directx::pso_library::create_compute(device, &desc) }
        .map_err(|e| format!("create {label} PSO: {e}"))
}

// Mip count for a Hi-Z of size (w, h): `floor(log2(max(w, h))) + 1`. Power-
// of-two sources end exactly at 1x1; non-power-of-two sources stop one mip
// short of true 1x1 in the smaller dimension, which is fine; the cull
// kernel clamps to the actual mip dims.
pub(super) fn hiz_mip_count(width: u32, height: u32) -> u32 {
    let m = width.max(height).max(1);
    32 - m.leading_zeros()
}

// Pyramid depth the two SPD dispatches actually write, which is what the
// texture is allocated with and what the cull is told. Never more than the
// reserved descriptor slots.
fn hiz_plan_mip_count(width: u32, height: u32, uav_slots: usize) -> u32 {
    let requested = hiz_mip_count(width, height).min(uav_slots as u32);
    Plan::new(width, height, requested, 1).mip_count()
}

// Write a per-mip UAV into every reserved slot. Slots past the last live mip
// repeat it: an SPD dispatch binds a fixed-length table, and D3D12 requires
// each descriptor in a bound range to be valid even where the kernel's
// `level_count` stops it from writing through them.
fn write_hiz_mip_uavs(
    device: &ID3D12Device,
    tex: &ID3D12Resource,
    mip_count: u32,
    slots: &[D3D12_CPU_DESCRIPTOR_HANDLE],
) {
    for (slot, &cpu) in slots.iter().enumerate() {
        write_hiz_mip_uav(device, tex, (slot as u32).min(mip_count - 1), cpu);
    }
}

// Create the Hi-Z texture (R32_FLOAT, full mip chain, UAV + SRV capable)
// plus the resource. Resting state is `NON_PIXEL_SHADER_RESOURCE` so the
// next-frame cull dispatch can sample it without a transition.
fn create_hiz_texture(
    device: &ID3D12Device,
    width: u32,
    height: u32,
    mip_count: u32,
) -> Result<ID3D12Resource, String> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: width as u64,
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: mip_count as u16,
        Format: DXGI_FORMAT_R32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut tex: Option<ID3D12Resource> = None;
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            None,
            &mut tex,
        )
    }
    .map_err(|e| format!("create hiz texture: {e}"))?;
    tex.ok_or_else(|| "create hiz texture returned None".to_string())
}

// Write the all-mips SRV that the cull kernel and the downsample kernel
// share. `MipLevels: u32::MAX` means "every mip".
pub(in crate::directx) fn write_hiz_srv(
    device: &ID3D12Device,
    tex: &ID3D12Resource,
    mip_count: u32,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_FLOAT,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MostDetailedMip: 0,
                MipLevels: mip_count,
                PlaneSlice: 0,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe { device.CreateShaderResourceView(tex, Some(&desc), srv_cpu) };
}

// Write a UAV pointing at a single mip slice.
pub(in crate::directx) fn write_hiz_mip_uav(
    device: &ID3D12Device,
    tex: &ID3D12Resource,
    mip: u32,
    uav_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_FLOAT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_UAV {
                MipSlice: mip,
                PlaneSlice: 0,
            },
        },
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe { device.CreateUnorderedAccessView(tex, None, Some(&desc), uav_cpu) };
}

// Device handles the Hi-Z builder submits against, plus the shader hot-reload
// toggle. They always travel together through `new`.
#[derive(Clone, Copy)]
pub(super) struct HiZDeviceCtx<'a> {
    pub device: &'a ID3D12Device,
    pub info_queue: Option<&'a ID3D12InfoQueue>,
    pub hot_reload: bool,
}

// The Hi-Z render target: its dimensions plus every descriptor handle the
// downsample + cull kernels bind. The all-mips SRV has a (CPU, GPU) pair;
// the per-mip UAVs have one pair per mip.
pub(super) struct HiZTarget {
    pub width: u32,
    pub height: u32,
    pub srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    pub depth_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    pub mip_uav_cpus: Vec<D3D12_CPU_DESCRIPTOR_HANDLE>,
    pub mip_uav_gpus: Vec<D3D12_GPU_DESCRIPTOR_HANDLE>,
}

impl HiZResources {
    // Build the Hi-Z resource + every PSO. Called from the init path when
    // the bindless static pass + cull pipeline are active. Each of the
    // supplied descriptor handles points at a pre-reserved slot in the
    // SRV heap; the resource owns the descriptors but not the heap.
    pub(super) fn new(ctx: HiZDeviceCtx, target: HiZTarget) -> Result<Self, String> {
        let HiZDeviceCtx {
            device,
            info_queue,
            hot_reload,
        } = ctx;
        let HiZTarget {
            width,
            height,
            srv_cpu,
            srv_gpu,
            depth_srv_gpu,
            mip_uav_cpus,
            mip_uav_gpus,
        } = target;
        let mip_count = hiz_plan_mip_count(width, height, mip_uav_cpus.len());
        if mip_count == 0 {
            return Err("hiz: zero mip count".into());
        }
        let (spd_single_cs, spd_msaa_cs, spd_tail_cs) = compile_hiz_shaders(hot_reload)?;
        let root_sig = dump_on_err(info_queue, create_hiz_root_signature(device))?;
        let tail_root_sig = dump_on_err(info_queue, create_hiz_tail_root_signature(device))?;
        let spd_single_pso = dump_on_err(
            info_queue,
            create_hiz_pso(device, &root_sig, &spd_single_cs, "hiz spd_single"),
        )?;
        let spd_msaa_pso = dump_on_err(
            info_queue,
            create_hiz_pso(device, &root_sig, &spd_msaa_cs, "hiz spd_msaa"),
        )?;
        let spd_tail_pso = dump_on_err(
            info_queue,
            create_hiz_pso(device, &tail_root_sig, &spd_tail_cs, "hiz spd_tail"),
        )?;

        let texture = create_hiz_texture(device, width, height, mip_count)?;
        write_hiz_srv(device, &texture, mip_count, srv_cpu);
        write_hiz_mip_uavs(device, &texture, mip_count, &mip_uav_cpus);
        Ok(Self {
            root_sig,
            tail_root_sig,
            spd_single_pso,
            spd_msaa_pso,
            spd_tail_pso,
            texture,
            width,
            height,
            mip_count,
            rest_state: D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            srv_cpu,
            srv_gpu,
            depth_srv_gpu,
            mip_uav_cpus,
            mip_uav_gpus,
        })
    }

    // Recreate the texture at new render-target dimensions. Re-uses the
    // existing descriptor heap slots; the live cull-kernel binding stays
    // valid because the GPU descriptor handles point at the same slots.
    pub(super) fn resize_to(
        &mut self,
        device: &ID3D12Device,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        let new_mip_count = hiz_plan_mip_count(width, height, self.mip_uav_cpus.len());
        let texture = create_hiz_texture(device, width, height, new_mip_count)?;
        write_hiz_srv(device, &texture, new_mip_count, self.srv_cpu);
        write_hiz_mip_uavs(device, &texture, new_mip_count, &self.mip_uav_cpus);
        self.texture = texture;
        self.width = width;
        self.height = height;
        self.mip_count = new_mip_count;
        self.rest_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        Ok(())
    }

    // Swap the freshly-rebuilt PSOs into the live resources. Used by the
    // shader hot-reload pass.
    pub(super) fn swap_pipelines(
        &mut self,
        spd_single_pso: ID3D12PipelineState,
        spd_msaa_pso: ID3D12PipelineState,
        spd_tail_pso: ID3D12PipelineState,
    ) {
        self.spd_single_pso = spd_single_pso;
        self.spd_msaa_pso = spd_msaa_pso;
        self.spd_tail_pso = spd_tail_pso;
    }
}

impl crate::directx::context::DxContext {
    // Encode the Hi-Z build pass on `cmd`. Runs as the graph's `HizBuild`
    // (mid-frame, phase-1 depth) or `HizFinal` (terminal, the frame's last depth
    // version) node, so the executor has already put main depth in a
    // shader-resource state and the pyramid in `UNORDERED_ACCESS`; only the
    // per-mip chain below is this encoder's. A no-op when bindless cull isn't
    // active (no Hi-Z resource was built).
    pub(in crate::directx) fn encode_hiz_build(&self, cmd: &ID3D12GraphicsCommandList) {
        let Some(hiz) = self.cull.hiz.as_ref() else {
            return;
        };
        let sample_count = self.hdr.msaa_samples.max(1);
        let plan = Plan::new(hiz.width, hiz.height, hiz.mip_count, sample_count);

        // Phase 1: main depth into mips 0..6. The UAV table starts at mip 0.
        let pso = match self.hdr.msaa_samples > 1 {
            true => &hiz.spd_msaa_pso,
            false => &hiz.spd_single_pso,
        };
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.SetComputeRootSignature(&hiz.root_sig);
            cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]);
            cmd.SetPipelineState(pso);
            set_hiz_constants(cmd, &plan.phase1.params);
            cmd.SetComputeRootDescriptorTable(1, hiz.depth_srv_gpu);
            cmd.SetComputeRootDescriptorTable(2, hiz.mip_uav_gpus[0]);
            cmd.Dispatch(plan.phase1.groups.0, plan.phase1.groups.1, 1);
        }

        let Some(tail) = plan.tail else {
            return;
        };
        // Phase 2 reads the mip 6 phase 1 just wrote, so the pyramid needs one
        // write -> read barrier here. It is the only one the build takes; the
        // graph owns the transitions on either side of the node.
        // SAFETY: the command list is in the recording state, and the resource this barrier names
        // is live for the call.
        unsafe { cmd.ResourceBarrier(&[uav_barrier(&hiz.texture)]) };
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.SetComputeRootSignature(&hiz.tail_root_sig);
            cmd.SetPipelineState(&hiz.spd_tail_pso);
            set_hiz_constants(cmd, &tail.params);
            cmd.SetComputeRootDescriptorTable(1, hiz.mip_uav_gpus[tail.base_mip as usize]);
            cmd.Dispatch(tail.groups.0, tail.groups.1, 1);
        }
    }
}

// Push one dispatch's params into the root constants at b0.
//
// SAFETY: the caller holds a command list in the recording state whose bound root signature
// declares `HIZ_PARAMS_DWORDS` 32-bit constants at parameter 0, and `params` outlives the call.
unsafe fn set_hiz_constants(cmd: &ID3D12GraphicsCommandList, params: &HizSpdParams) {
    // SAFETY: forwarded from this function's own contract.
    unsafe {
        cmd.SetComputeRoot32BitConstants(
            0,
            HIZ_PARAMS_DWORDS,
            params as *const HizSpdParams as *const std::ffi::c_void,
            0,
        )
    };
}
