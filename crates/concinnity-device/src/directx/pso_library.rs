// The device's ID3D12PipelineLibrary, persisted across launches.
//
// Mirrors `vulkan::pipeline_cache`: one library per adapter, loaded from
// `pipeline-cache/dx-<vendor>-<device>-<rev>.bin` at init, consulted by every
// PSO creation through `create_graphics` / `create_compute`, serialized at the
// end of init and at shutdown. D3D12 gives no content addressing, so each PSO
// is stored under a name derived from its desc: the stage bytecode plus every
// fixed-function discriminator, hashed. A shader edit or state change is a
// different name and simply misses; the driver validates the blob itself and
// answers a stale one with an adapter/driver-mismatch HRESULT, on which the
// file is deleted and the launch proceeds cold. Nothing here can fail init.
//
// The library lives in a thread-local rather than on `DxContext` because every
// PSO build funnels through the two wrappers here, `DxContext` is pinned to
// the main thread, and a live editor `reload_world` reuses the device: the
// library stays installed across the successor context.

use sha2::{Digest, Sha256};
use std::cell::RefCell;
use windows::Win32::Foundation::{
    D3D12_ERROR_ADAPTER_NOT_FOUND, D3D12_ERROR_DRIVER_VERSION_MISMATCH, E_INVALIDARG,
};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_COMPUTE_PIPELINE_STATE_DESC, D3D12_DEPTH_STENCILOP_DESC,
    D3D12_GRAPHICS_PIPELINE_STATE_DESC, D3D12_RENDER_TARGET_BLEND_DESC, D3D12_SHADER_BYTECODE,
    ID3D12Device, ID3D12Device1, ID3D12PipelineLibrary, ID3D12PipelineState,
};
use windows::Win32::Graphics::Dxgi::{DXGI_ERROR_UNSUPPORTED, IDXGIAdapter3};
use windows::core::{Interface, PCWSTR};

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

struct State {
    // Declared before `disk`: D3D12 parses the initial blob in place, so the
    // library must drop before its backing bytes.
    library: ID3D12PipelineLibrary,
    file: String,
    disk: Option<Vec<u8>>,
    warm: bool,
}

// Create the adapter's pipeline library, seeded from disk when a blob exists
// and the driver accepts it. No-op when already installed (the `reload_world`
// path, whose successor context reuses the device).
pub(super) fn install(device: &ID3D12Device, adapter: Option<&IDXGIAdapter3>) {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        if state.is_some() {
            return;
        }
        let Some(adapter) = adapter else {
            return;
        };
        let Ok(desc) = (unsafe { adapter.GetDesc1() }) else {
            return;
        };
        let Ok(device1) = device.cast::<ID3D12Device1>() else {
            tracing::info!("pipeline library: ID3D12Device1 unavailable, PSOs build uncached");
            return;
        };
        let file = format!(
            "dx-{:04x}-{:04x}-{:02x}.bin",
            desc.VendorId, desc.DeviceId, desc.Revision
        );
        let mut disk = crate::pipeline_cache::load(&file);
        let mut created = create_library(&device1, disk.as_deref());
        // E_INVALIDARG covers a corrupt blob; the two mismatch codes are a
        // driver update or a GPU swap that kept the file name. All three drop
        // the blob and retry empty.
        let rejected = matches!(
            &created,
            Err(e) if disk.is_some() && [
                E_INVALIDARG,
                D3D12_ERROR_ADAPTER_NOT_FOUND,
                D3D12_ERROR_DRIVER_VERSION_MISMATCH,
            ].contains(&e.code())
        );
        if rejected {
            tracing::warn!("pipeline library: driver rejected {file}, rebuilding cold");
            crate::pipeline_cache::delete(&file);
            disk = None;
            created = create_library(&device1, None);
        }
        match created {
            Ok(library) => {
                let warm = disk.is_some();
                *state = Some(State {
                    library,
                    file,
                    disk,
                    warm,
                });
            }
            Err(e) if e.code() == DXGI_ERROR_UNSUPPORTED => {
                tracing::info!(
                    "pipeline library: unsupported by this runtime, PSOs build uncached"
                );
            }
            Err(e) => {
                tracing::warn!("pipeline library: create failed ({e}), PSOs build uncached");
            }
        }
    });
}

fn create_library(
    device1: &ID3D12Device1,
    blob: Option<&[u8]>,
) -> windows::core::Result<ID3D12PipelineLibrary> {
    unsafe { device1.CreatePipelineLibrary(blob.unwrap_or(&[])) }
}

// `CreateGraphicsPipelineState` through the library: a stored PSO under this
// desc's derived name loads without a driver compile; a miss creates and
// stores. Timed for the init tally either way.
//
// # Safety
// Same contract as `ID3D12Device::CreateGraphicsPipelineState`.
pub(super) unsafe fn create_graphics(
    device: &ID3D12Device,
    desc: &D3D12_GRAPHICS_PIPELINE_STATE_DESC,
) -> windows::core::Result<ID3D12PipelineState> {
    let started = std::time::Instant::now();
    let result = STATE.with(|state| match &*state.borrow() {
        None => unsafe { device.CreateGraphicsPipelineState(desc) },
        Some(state) => {
            let name = wide(&graphics_name(desc));
            let loaded: windows::core::Result<ID3D12PipelineState> = unsafe {
                state
                    .library
                    .LoadGraphicsPipeline(PCWSTR(name.as_ptr()), desc)
            };
            loaded.or_else(|_| {
                let pso: ID3D12PipelineState = unsafe { device.CreateGraphicsPipelineState(desc) }?;
                let _ = unsafe { state.library.StorePipeline(PCWSTR(name.as_ptr()), &pso) };
                Ok(pso)
            })
        }
    });
    crate::pipeline_cache::note_creation(started.elapsed().as_micros() as u64);
    result
}

// `CreateComputePipelineState` counterpart of [`create_graphics`].
//
// # Safety
// Same contract as `ID3D12Device::CreateComputePipelineState`.
pub(super) unsafe fn create_compute(
    device: &ID3D12Device,
    desc: &D3D12_COMPUTE_PIPELINE_STATE_DESC,
) -> windows::core::Result<ID3D12PipelineState> {
    let started = std::time::Instant::now();
    let result = STATE.with(|state| match &*state.borrow() {
        None => unsafe { device.CreateComputePipelineState(desc) },
        Some(state) => {
            let name = wide(&compute_name(desc));
            let loaded: windows::core::Result<ID3D12PipelineState> = unsafe {
                state
                    .library
                    .LoadComputePipeline(PCWSTR(name.as_ptr()), desc)
            };
            loaded.or_else(|_| {
                let pso: ID3D12PipelineState = unsafe { device.CreateComputePipelineState(desc) }?;
                let _ = unsafe { state.library.StorePipeline(PCWSTR(name.as_ptr()), &pso) };
                Ok(pso)
            })
        }
    });
    crate::pipeline_cache::note_creation(started.elapsed().as_micros() as u64);
    result
}

// Whether `install` seeded the library from a driver-accepted disk blob;
// names the init tally's disk state.
pub(super) fn disk_state() -> &'static str {
    STATE.with(|state| match &*state.borrow() {
        Some(s) if s.warm => "warm",
        Some(_) => "cold",
        None => "absent",
    })
}

// Persist the library if it accumulated new content since the last write. Called at
// the end of init (so a crash later in the session cannot lose the warm-up)
// and again from `shutdown`.
pub(super) fn serialize() {
    STATE.with(|state| {
        if let Some(state) = state.borrow_mut().as_mut() {
            let size = unsafe { state.library.GetSerializedSize() };
            if size == 0 {
                return;
            }
            let mut bytes = vec![0u8; size];
            let serialized = unsafe { state.library.Serialize(&mut bytes) };
            if serialized.is_ok()
                && crate::pipeline_cache::store_if_grown(&state.file, &bytes, state.disk.as_deref())
            {
                state.disk = Some(bytes);
            }
        }
    });
}

// Serialize, then release the library. Called from the final context's Drop
// (never the outgoing context of a `reload_world`, which keeps the device).
pub(super) fn shutdown() {
    serialize();
    STATE.with(|state| {
        state.borrow_mut().take();
    });
}

fn wide(name: &str) -> Vec<u16> {
    name.encode_utf16().chain(std::iter::once(0)).collect()
}

// The library name for a graphics desc: a hash over the stage bytecode and
// every fixed-function discriminator the backend varies, so any change to
// shaders or state is a different name and misses rather than loading a PSO
// built for other state. The root signature is deliberately excluded (it is a
// pointer, and no two backend PSOs share stage bytecode across root
// signatures).
fn graphics_name(desc: &D3D12_GRAPHICS_PIPELINE_STATE_DESC) -> String {
    let mut h = Sha256::new();
    for stage in [&desc.VS, &desc.PS, &desc.DS, &desc.HS, &desc.GS] {
        hash_bytecode(&mut h, stage);
    }
    hash_u32s(
        &mut h,
        &[
            desc.BlendState.AlphaToCoverageEnable.0 as u32,
            desc.BlendState.IndependentBlendEnable.0 as u32,
            desc.SampleMask,
        ],
    );
    for rt in &desc.BlendState.RenderTarget {
        hash_blend(&mut h, rt);
    }
    hash_u32s(
        &mut h,
        &[
            desc.RasterizerState.FillMode.0 as u32,
            desc.RasterizerState.CullMode.0 as u32,
            desc.RasterizerState.FrontCounterClockwise.0 as u32,
            desc.RasterizerState.DepthBias as u32,
            desc.RasterizerState.DepthBiasClamp.to_bits(),
            desc.RasterizerState.SlopeScaledDepthBias.to_bits(),
            desc.RasterizerState.DepthClipEnable.0 as u32,
            desc.RasterizerState.MultisampleEnable.0 as u32,
            desc.RasterizerState.AntialiasedLineEnable.0 as u32,
            desc.RasterizerState.ForcedSampleCount,
            desc.RasterizerState.ConservativeRaster.0 as u32,
            desc.DepthStencilState.DepthEnable.0 as u32,
            desc.DepthStencilState.DepthWriteMask.0 as u32,
            desc.DepthStencilState.DepthFunc.0 as u32,
            desc.DepthStencilState.StencilEnable.0 as u32,
            desc.DepthStencilState.StencilReadMask as u32,
            desc.DepthStencilState.StencilWriteMask as u32,
        ],
    );
    hash_stencil_op(&mut h, &desc.DepthStencilState.FrontFace);
    hash_stencil_op(&mut h, &desc.DepthStencilState.BackFace);
    hash_u32s(&mut h, &[desc.InputLayout.NumElements]);
    if !desc.InputLayout.pInputElementDescs.is_null() {
        let elements = unsafe {
            std::slice::from_raw_parts(
                desc.InputLayout.pInputElementDescs,
                desc.InputLayout.NumElements as usize,
            )
        };
        for e in elements {
            if !e.SemanticName.is_null() {
                let semantic =
                    unsafe { std::ffi::CStr::from_ptr(e.SemanticName.0 as *const _) }.to_bytes();
                h.update((semantic.len() as u64).to_le_bytes());
                h.update(semantic);
            }
            hash_u32s(
                &mut h,
                &[
                    e.SemanticIndex,
                    e.Format.0 as u32,
                    e.InputSlot,
                    e.AlignedByteOffset,
                    e.InputSlotClass.0 as u32,
                    e.InstanceDataStepRate,
                ],
            );
        }
    }
    hash_u32s(
        &mut h,
        &[
            desc.IBStripCutValue.0 as u32,
            desc.PrimitiveTopologyType.0 as u32,
            desc.NumRenderTargets,
        ],
    );
    for format in &desc.RTVFormats {
        hash_u32s(&mut h, &[format.0 as u32]);
    }
    hash_u32s(
        &mut h,
        &[
            desc.DSVFormat.0 as u32,
            desc.SampleDesc.Count,
            desc.SampleDesc.Quality,
            desc.NodeMask,
            desc.Flags.0 as u32,
        ],
    );
    format!("g{:x}", Truncated(h.finalize()))
}

// The library name for a compute desc: the kernel bytecode is the identity.
fn compute_name(desc: &D3D12_COMPUTE_PIPELINE_STATE_DESC) -> String {
    let mut h = Sha256::new();
    hash_bytecode(&mut h, &desc.CS);
    hash_u32s(&mut h, &[desc.NodeMask, desc.Flags.0 as u32]);
    format!("c{:x}", Truncated(h.finalize()))
}

// First 16 digest bytes as lowercase hex: ample collision margin for a few
// hundred names, half the string churn in the library's name table.
struct Truncated(sha2::digest::Output<Sha256>);

impl std::fmt::LowerHex for Truncated {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in self.0.iter().take(16) {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

fn hash_bytecode(h: &mut Sha256, bytecode: &D3D12_SHADER_BYTECODE) {
    h.update((bytecode.BytecodeLength as u64).to_le_bytes());
    if !bytecode.pShaderBytecode.is_null() && bytecode.BytecodeLength > 0 {
        h.update(unsafe {
            std::slice::from_raw_parts(
                bytecode.pShaderBytecode as *const u8,
                bytecode.BytecodeLength,
            )
        });
    }
}

fn hash_u32s(h: &mut Sha256, values: &[u32]) {
    for value in values {
        h.update(value.to_le_bytes());
    }
}

fn hash_blend(h: &mut Sha256, rt: &D3D12_RENDER_TARGET_BLEND_DESC) {
    hash_u32s(
        h,
        &[
            rt.BlendEnable.0 as u32,
            rt.LogicOpEnable.0 as u32,
            rt.SrcBlend.0 as u32,
            rt.DestBlend.0 as u32,
            rt.BlendOp.0 as u32,
            rt.SrcBlendAlpha.0 as u32,
            rt.DestBlendAlpha.0 as u32,
            rt.BlendOpAlpha.0 as u32,
            rt.LogicOp.0 as u32,
            rt.RenderTargetWriteMask as u32,
        ],
    );
}

fn hash_stencil_op(h: &mut Sha256, op: &D3D12_DEPTH_STENCILOP_DESC) {
    hash_u32s(
        h,
        &[
            op.StencilFailOp.0 as u32,
            op.StencilDepthFailOp.0 as u32,
            op.StencilPassOp.0 as u32,
            op.StencilFunc.0 as u32,
        ],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Dxgi::Common::{
        DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT,
    };

    fn graphics_desc(vs: &[u8], ps: &[u8]) -> D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            VS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: vs.as_ptr() as *const _,
                BytecodeLength: vs.len(),
            },
            PS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: ps.as_ptr() as *const _,
                BytecodeLength: ps.len(),
            },
            ..Default::default()
        }
    }

    #[test]
    fn the_name_is_stable_for_identical_descs() {
        let (vs, ps) = ([1u8, 2, 3], [4u8, 5]);
        assert_eq!(
            graphics_name(&graphics_desc(&vs, &ps)),
            graphics_name(&graphics_desc(&vs, &ps))
        );
    }

    #[test]
    fn every_discriminator_changes_the_name() {
        let (vs, ps) = ([1u8, 2, 3], [4u8, 5]);
        let base = graphics_name(&graphics_desc(&vs, &ps));
        assert_ne!(base, graphics_name(&graphics_desc(&[9u8], &ps)), "vs");
        assert_ne!(base, graphics_name(&graphics_desc(&vs, &[9u8])), "ps");

        let mut rtv = graphics_desc(&vs, &ps);
        rtv.RTVFormats[0] = DXGI_FORMAT_R16G16B16A16_FLOAT;
        assert_ne!(base, graphics_name(&rtv), "rtv format");

        let mut dsv = graphics_desc(&vs, &ps);
        dsv.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        assert_ne!(base, graphics_name(&dsv), "dsv format");

        let mut msaa = graphics_desc(&vs, &ps);
        msaa.SampleDesc.Count = 4;
        assert_ne!(base, graphics_name(&msaa), "sample count");

        let mut blend = graphics_desc(&vs, &ps);
        blend.BlendState.RenderTarget[0].BlendEnable = true.into();
        assert_ne!(base, graphics_name(&blend), "blend");
    }

    #[test]
    fn bytecode_boundaries_cannot_be_confused() {
        // Without length prefixes ("ab", "c") and ("a", "bc") would collide.
        assert_ne!(
            graphics_name(&graphics_desc(b"ab", b"c")),
            graphics_name(&graphics_desc(b"a", b"bc"))
        );
    }

    #[test]
    fn compute_and_graphics_names_never_collide() {
        let kernel = [7u8, 7, 7];
        let compute = compute_name(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: kernel.as_ptr() as *const _,
                BytecodeLength: kernel.len(),
            },
            ..Default::default()
        });
        assert!(compute.starts_with('c'));
        assert!(graphics_name(&graphics_desc(&kernel, &[])).starts_with('g'));
    }
}
