// src/directx/probe_prefilter.rs
//
// The convolution half of a runtime reflection-probe bake on DirectX: the three
// compute PSOs built from `probe_prefilter.slang`, their root signatures, the two
// cube resources one bake works between, and the dispatches that turn six
// captured faces into the prefiltered radiance cube the specular term samples.
// Mirrors `metal::probe_prefilter` and `vulkan::probe_prefilter`.
//
// The capture cube collects the six rendered faces (one array slice each) and
// carries a mip chain the `probe_downsample` kernel fills; the probe cube is the
// result, mip 0 a firefly-clamped copy of the capture and every mip after it a
// GGX convolution at that mip's roughness. Both are R16G16B16A16_FLOAT: the faces
// are rendered as halfs, the clamp caps luminance well inside the format's range,
// and it halves what a probe costs against the R32G32B32A32 cube the CPU
// convolution used to upload.
//
// Nothing reads back. The whole convolution stays on the direct queue, so the
// frames that sample the finished cube are ordered after the dispatches that
// wrote it by submission order alone.
//
// Resource states, which the barriers below are the whole of: the capture arrives
// in COPY_DEST (the per-face copies write it), moves to UNORDERED_ACCESS for the
// pyramid build, then to NON_PIXEL_SHADER_RESOURCE for the GGX dispatches that
// sample it. The probe cube sits in UNORDERED_ACCESS for every dispatch that
// writes it and moves to PIXEL_SHADER_RESOURCE at install.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use concinnity_core::render::reflection_probe::PrefilterPlan;
use concinnity_core::render::uniforms::ProbePrefilterParams;

use super::com;
use super::context::DxContext;
use super::pipeline::serialize_desc_and_create;
use super::slang_builtins::SlangCompile;

/// Colour format of both cubes.
pub(in crate::directx) const PROBE_CUBE_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R16G16B16A16_FLOAT;

/// Upper bound on a probe cube's mip count, sizing the descriptor block the SRV
/// heap reserves for one bake. `PrefilterPlan` stops at 4x4 faces, so 12 covers a
/// capture up to 8192 px on an edge.
pub(in crate::directx) const PROBE_MAX_MIPS: usize = 12;

// Threadgroup tile, matching the kernels' `[numthreads(8, 8, 1)]`. The third
// dispatch dimension is the six cube faces, one invocation deep.
const PREFILTER_TILE: u32 = 8;

// `ProbePrefilterParams` as 32-bit root constants at b0.
const PREFILTER_PARAM_DWORDS: u32 = (size_of::<ProbePrefilterParams>() / 4) as u32;

/// The convolution PSOs and the two root signatures they bind. Built once at init
/// and reused by every bake.
pub(in crate::directx) struct ProbePrefilterPipelines {
    // The mirror-mip copy and the pyramid reduction bind the same shape (root
    // constants plus a two-descriptor UAV table), so they share a root signature;
    // the GGX kernel adds an SRV table and a static sampler.
    mip_root: ID3D12RootSignature,
    ggx_root: ID3D12RootSignature,
    mip0: ID3D12PipelineState,
    downsample: ID3D12PipelineState,
    ggx: ID3D12PipelineState,
}

/// Whether the device can load `R16G16B16A16_FLOAT` through an unordered-access
/// view. `probe_mip0` and `probe_downsample` both read their source mip as a
/// `RWTexture2DArray<float4>`, and D3D12 guarantees typed UAV loads for only four
/// formats without this optional capability. A device without it cannot run the
/// convolution at all, so the caller leaves the pipelines unbuilt and probes fall
/// back to the sky prefilter.
pub(in crate::directx) fn typed_uav_load_supported(device: &ID3D12Device) -> bool {
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    // SAFETY: a query on a live COM object; the descriptor it reads and the out-parameters it fills
    // are live locals that outlive the call.
    let ok = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS,
            &mut options as *mut _ as *mut std::ffi::c_void,
            size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS>() as u32,
        )
    };
    ok.is_ok() && options.TypedUAVLoadAdditionalFormats.as_bool()
}

impl ProbePrefilterPipelines {
    pub(in crate::directx) fn new(device: &ID3D12Device, hot_reload: bool) -> Result<Self, String> {
        use super::slang_builtins;
        let mip_root = create_mip_root_signature(device)?;
        let ggx_root = create_ggx_root_signature(device)?;
        let mip0 = create_pso(
            device,
            &mip_root,
            &slang_builtins::PROBE_MIP0.compile(hot_reload)?,
            "probe_mip0",
        )?;
        let downsample = create_pso(
            device,
            &mip_root,
            &slang_builtins::PROBE_DOWNSAMPLE.compile(hot_reload)?,
            "probe_downsample",
        )?;
        let ggx = create_pso(
            device,
            &ggx_root,
            &slang_builtins::PROBE_GGX.compile(hot_reload)?,
            "probe_ggx",
        )?;
        Ok(Self {
            mip_root,
            ggx_root,
            mip0,
            downsample,
            ggx,
        })
    }
}

/// The two cube resources one bake convolves between. Their descriptors live in
/// the SRV heap's reserved probe-prefilter block, which [`PrefilterGpu::new`]
/// rewrites for each bake. One block for every bake is what serialises them on
/// this backend: `bake_pending_probes` starts a capture only once the prefiltering
/// slot is empty, and the install that empties it is gated on the fence covering
/// the prior bake's last dispatch, so nothing in flight still binds the block when
/// it is rewritten.
pub(in crate::directx) struct PrefilterGpu {
    capture: ID3D12Resource,
    probe: ID3D12Resource,
    mips: u32,
}

impl PrefilterGpu {
    /// Allocate both cubes and write every descriptor the bake's dispatches bind.
    /// The capture starts in COPY_DEST so the per-face copies can write it, the
    /// probe cube in UNORDERED_ACCESS so the first dispatch can.
    pub(in crate::directx) fn new(
        ctx: &DxContext,
        plan: &PrefilterPlan,
    ) -> Result<PrefilterGpu, String> {
        let mips = plan.mips();
        if mips as usize > PROBE_MAX_MIPS {
            return Err(format!(
                "probe: {mips} mips exceeds the {PROBE_MAX_MIPS} descriptors reserved for one bake"
            ));
        }
        let capture = create_cube(
            &ctx.device,
            plan.face_size(),
            mips,
            D3D12_RESOURCE_STATE_COPY_DEST,
            "probe capture cube",
        )?;
        let probe = create_cube(
            &ctx.device,
            plan.face_size(),
            mips,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            "probe cube",
        )?;

        let device = &ctx.device;
        let d = &ctx.descriptors;
        write_cube_srv(device, &capture, mips, ctx.probe_capture_srv_cpu());
        for mip in 0..mips {
            write_cube_mip_uav(
                device,
                &capture,
                mip,
                cpu_slot(ctx, d.probe_capture_uav_base_slot + mip as usize),
            );
            write_cube_mip_uav(
                device,
                &probe,
                mip,
                cpu_slot(ctx, d.probe_cube_uav_base_slot + mip as usize),
            );
        }
        // The mirror-mip copy binds capture mip 0 and probe mip 0 as one contiguous
        // pair, which neither per-cube block can supply, so it gets its own.
        write_cube_mip_uav(device, &capture, 0, cpu_slot(ctx, d.probe_mip0_pair_slot));
        write_cube_mip_uav(device, &probe, 0, cpu_slot(ctx, d.probe_mip0_pair_slot + 1));

        Ok(PrefilterGpu {
            capture,
            probe,
            mips,
        })
    }

    /// The capture resource the six face copies write into.
    pub(in crate::directx) fn capture(&self) -> &ID3D12Resource {
        &self.capture
    }

    /// The probe cube being written.
    pub(in crate::directx) fn probe(&self) -> &ID3D12Resource {
        &self.probe
    }

    /// Mip levels both cubes carry.
    pub(in crate::directx) fn mips(&self) -> u32 {
        self.mips
    }

    /// The finished cube, handed to the probe pool at install. The capture drops
    /// with the rest of `self`.
    pub(in crate::directx) fn into_probe_cube(self) -> ID3D12Resource {
        self.probe
    }
}

impl DxContext {
    /// CPU handle of the capture cube's all-mips SRV slot.
    pub(in crate::directx) fn probe_capture_srv_cpu(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        cpu_slot(self, self.descriptors.probe_capture_srv_slot)
    }

    /// Record the cheap half of the convolution: the capture moves from the
    /// per-face copies' COPY_DEST into UNORDERED_ACCESS, the mirror mip is copied
    /// through with the firefly clamp, the source pyramid is reduced level by
    /// level, and the capture ends in NON_PIXEL_SHADER_RESOURCE for the GGX
    /// dispatches that follow.
    ///
    /// All of it goes in one command list: the reductions are a few taps per texel
    /// and each depends on the one before, so spreading them over frames would only
    /// lengthen the bake.
    pub(in crate::directx) fn encode_probe_pyramid(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        gpu: &PrefilterGpu,
        plan: &PrefilterPlan,
    ) -> Result<(), String> {
        let pipelines = self
            .probe
            .prefilter
            .as_ref()
            .ok_or("probe: prefilter pipelines missing")?;
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe {
            cmd.ResourceBarrier(&[super::texture::transition_barrier(
                gpu.capture(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            cmd.SetComputeRootSignature(&pipelines.mip_root);
        }
        let d = &self.descriptors;
        self.dispatch_prefilter(
            cmd,
            &pipelines.mip0,
            gpu_slot(self, d.probe_mip0_pair_slot),
            None,
            &plan.mip0_params(),
            plan.face_size(),
        );
        for mip in 1..plan.mips() {
            // Each level reads the one the previous dispatch wrote.
            uav_barrier(cmd, gpu.capture());
            self.dispatch_prefilter(
                cmd,
                &pipelines.downsample,
                gpu_slot(self, d.probe_capture_uav_base_slot + (mip - 1) as usize),
                None,
                &plan.downsample_params(mip),
                plan.mip_face_size(mip),
            );
        }
        // SAFETY: as above.
        unsafe {
            cmd.ResourceBarrier(&[super::texture::transition_barrier(
                gpu.capture(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            )]);
        }
        Ok(())
    }

    /// Record the GGX convolution producing probe-cube mip `dst_mip`, sampling the
    /// finished pyramid. Nothing else writes that mip, and the capture is read-only
    /// from here on, so consecutive mips need no barrier between them.
    pub(in crate::directx) fn encode_probe_ggx_mip(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        plan: &PrefilterPlan,
        dst_mip: u32,
    ) -> Result<(), String> {
        let pipelines = self
            .probe
            .prefilter
            .as_ref()
            .ok_or("probe: prefilter pipelines missing")?;
        // SAFETY: the command list is in the recording state and the root signature is live.
        unsafe { cmd.SetComputeRootSignature(&pipelines.ggx_root) };
        let d = &self.descriptors;
        self.dispatch_prefilter(
            cmd,
            &pipelines.ggx,
            gpu_slot(self, d.probe_cube_uav_base_slot + dst_mip as usize),
            Some(gpu_slot(self, d.probe_capture_srv_slot)),
            &plan.ggx_params(dst_mip),
            plan.mip_face_size(dst_mip),
        );
        Ok(())
    }

    // Bind, push and dispatch one prefilter kernel over a `size`-square cube face,
    // six faces deep. The kernels bounds-guard against `dst_size`, so the
    // rounded-up remainder returns early. `srv_table` is the GGX kernel's sampled
    // capture; the mip kernels bind none.
    fn dispatch_prefilter(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        pso: &ID3D12PipelineState,
        uav_table: D3D12_GPU_DESCRIPTOR_HANDLE,
        srv_table: Option<D3D12_GPU_DESCRIPTOR_HANDLE>,
        params: &ProbePrefilterParams,
        size: u32,
    ) {
        let groups = size.div_ceil(PREFILTER_TILE).max(1);
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call; the root-constant count matches the
        // signature's, both derived from `ProbePrefilterParams`.
        unsafe {
            cmd.SetPipelineState(pso);
            cmd.SetComputeRoot32BitConstants(
                0,
                PREFILTER_PARAM_DWORDS,
                params as *const ProbePrefilterParams as *const std::ffi::c_void,
                0,
            );
            match srv_table {
                Some(srv) => {
                    cmd.SetComputeRootDescriptorTable(1, srv);
                    cmd.SetComputeRootDescriptorTable(2, uav_table);
                }
                None => cmd.SetComputeRootDescriptorTable(1, uav_table),
            }
            cmd.Dispatch(groups, groups, 6);
        }
    }
}

// Root signature for the mirror-copy and downsample kernels: root constants at b0
// plus one two-descriptor UAV table (u0 the source mip, u1 the destination). The
// per-mip UAVs are contiguous in the heap, which is what lets one range cover the
// pair.
fn create_mip_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature, String> {
    let uav_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
        NumDescriptors: 2,
        BaseShaderRegister: 0, // u0..u1
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let params = [root_constants(), descriptor_table(&uav_range)];
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
        ..Default::default()
    };
    serialize_desc_and_create(device, &desc, "probe prefilter mip root sig")
}

// Root signature for the GGX kernel: root constants at b0, the sampled capture
// pyramid at t0, the destination mip at u0, and the linear-clamp mipmapped
// sampler at s0 as a static sampler (a shader sampler needs no heap of its own
// when it never varies).
fn create_ggx_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature, String> {
    let srv_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 0, // t0
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let uav_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
        NumDescriptors: 1,
        BaseShaderRegister: 0, // u0
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let params = [
        root_constants(),
        descriptor_table(&srv_range),
        descriptor_table(&uav_range),
    ];
    // The solid-angle lod the kernel computes is fractional, so the trilinear
    // filter is what makes the level selection continuous.
    let sampler = D3D12_STATIC_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        ComparisonFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        BorderColor: D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK,
        MinLOD: 0.0,
        MaxLOD: f32::MAX,
        ShaderRegister: 0,
        RegisterSpace: 0,
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        ..Default::default()
    };
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        NumStaticSamplers: 1,
        pStaticSamplers: &sampler,
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
    };
    serialize_desc_and_create(device, &desc, "probe prefilter ggx root sig")
}

fn root_constants() -> D3D12_ROOT_PARAMETER {
    D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Constants: D3D12_ROOT_CONSTANTS {
                ShaderRegister: 0,
                RegisterSpace: 0,
                Num32BitValues: PREFILTER_PARAM_DWORDS,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

fn descriptor_table(range: &D3D12_DESCRIPTOR_RANGE) -> D3D12_ROOT_PARAMETER {
    D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                NumDescriptorRanges: 1,
                pDescriptorRanges: range,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

fn create_pso(
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
    // SAFETY: `desc` outlives this synchronous call, and so do the root signature and shader
    // bytecode whose raw pointers it borrows.
    unsafe { super::pso_library::create_compute(device, &desc) }
        .map_err(|e| format!("create {label} PSO: {e}"))
}

// A cube resource: six array slices, `mips` levels, UAV + SRV capable. Committed
// rather than pooled: the suballocator refuses GPU-written descs, because a placed
// resource needs re-initialising every time it claims memory and the pool does not
// do that.
fn create_cube(
    device: &ID3D12Device,
    face_size: u32,
    mips: u32,
    state: D3D12_RESOURCE_STATES,
    label: &str,
) -> Result<ID3D12Resource, String> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: face_size as u64,
        Height: face_size,
        DepthOrArraySize: 6,
        MipLevels: mips as u16,
        Format: PROBE_CUBE_FORMAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let mut cube: Option<ID3D12Resource> = None;
    // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the new
    // COM object lands in a binding that owns it.
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            state,
            None,
            &mut cube,
        )
    }
    .map_err(|e| format!("create {label}: {e}"))?;
    cube.ok_or_else(|| format!("create {label} returned None"))
}

// All-mips TEXTURECUBE SRV, the shape a sampler reads.
fn write_cube_srv(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    mips: u32,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: PROBE_CUBE_FORMAT,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURECUBE,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            TextureCube: D3D12_TEXCUBE_SRV {
                MostDetailedMip: 0,
                MipLevels: mips,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe { device.CreateShaderResourceView(resource, Some(&desc), srv_cpu) };
}

// Single-mip TEXTURE2DARRAY UAV over all six faces. A cube is a six-slice array,
// so this is what lets a kernel address (x, y, face) directly.
fn write_cube_mip_uav(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    mip: u32,
    uav_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: PROBE_CUBE_FORMAT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Texture2DArray: D3D12_TEX2D_ARRAY_UAV {
                MipSlice: mip,
                FirstArraySlice: 0,
                ArraySize: 6,
                PlaneSlice: 0,
            },
        },
    };
    // SAFETY: the view descriptor and the resource it names are live for the call, and the
    // destination handle addresses a slot this context reserved for the view in a heap it owns.
    unsafe { device.CreateUnorderedAccessView(resource, None, Some(&desc), uav_cpu) };
}

// Order one dispatch's writes to `resource` before the next dispatch's reads.
fn uav_barrier(cmd: &ID3D12GraphicsCommandList, resource: &ID3D12Resource) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: com::borrowed(resource),
            }),
        },
    };
    // SAFETY: the command list is in the recording state, and the barrier and the resource it
    // borrows are live for the call.
    unsafe { cmd.ResourceBarrier(&[barrier]) };
}

fn cpu_slot(ctx: &DxContext, slot: usize) -> D3D12_CPU_DESCRIPTOR_HANDLE {
    // SAFETY: a property query on a live descriptor heap; it only reads.
    let base = unsafe {
        ctx.descriptors
            .srv_heap
            .GetCPUDescriptorHandleForHeapStart()
    };
    D3D12_CPU_DESCRIPTOR_HANDLE {
        ptr: base.ptr + slot * ctx.descriptors.srv_descriptor_size,
    }
}

fn gpu_slot(ctx: &DxContext, slot: usize) -> D3D12_GPU_DESCRIPTOR_HANDLE {
    // SAFETY: a property query on a live descriptor heap; it only reads.
    let base = unsafe {
        ctx.descriptors
            .srv_heap
            .GetGPUDescriptorHandleForHeapStart()
    };
    D3D12_GPU_DESCRIPTOR_HANDLE {
        ptr: base.ptr + (slot * ctx.descriptors.srv_descriptor_size) as u64,
    }
}
