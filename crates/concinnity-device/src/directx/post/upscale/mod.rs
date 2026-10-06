//! Temporal upscaling for the D3D12 backend. The engine renders the 3D scene at
//! a fraction of drawable size and the `PassId::Upscale` pass reconstructs a
//! drawable-resolution image the bloom + composite stack consumes.
//!
//! Three interchangeable backends sit behind the `UpscaleBackend` trait:
//!   fsr   AMD FidelityFX FSR3 (cross-vendor; needs the Agility SDK bundled)
//!   dlss  NVIDIA DLSS via raw NGX (RTX only; cfg(ngx_sdk_bundled))
//!   xess  Intel XeSS (cross-vendor DP4a + Arc XMX)
//! The API-independent half of each lives in `crate::upscale_sdk`; these files
//! hold the D3D12 resources, DLL loading and command recording. `build_upscaler`
//! constructs the first backend that initializes, in the shared fallback order,
//! and `encode_upscale` drives whichever is active.

use std::cell::Cell;
use std::ffi::{CStr, c_void};

use concinnity_core::components::UpscalerBackend;
use concinnity_core::render::error::{RenderError, RenderResult};
use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};
use windows::core::PCSTR;

use crate::directx::descriptor_slot::SrvSlot;
use crate::directx::error::map_hresult;
use crate::directx::graph_exec::GraphFrameParams;
use crate::directx::texture::transition_barrier;
use crate::upscale_sdk::{
    Availability, ResolvedBackend, SdkLibrary, UpscaleCamera, UpscaleExtent, build_first_available,
};

#[cfg(ngx_sdk_bundled)]
mod dlss;
mod fsr;
mod xess;

// Temporal upscaling state. `backend` is `Some` only when the world's
// `PostProcessConfig.temporal_upscaling` is on and a vendor context was created.
// `requested` is the backend the world asked for, kept so a resize rebuilds the
// same one. `jitter` is the current frame's sub-pixel projection offset in
// render pixels, set by the frame stages before the parallel fan-out.
pub(in crate::directx) struct UpscaleState {
    pub backend: Option<Box<dyn UpscaleBackend>>,
    pub requested: UpscalerBackend,
    pub jitter: Cell<[f32; 2]>,
}

// One temporal-upscaling backend. `encode_upscale` transitions the scene /
// depth / motion inputs and the output texture, then calls `dispatch`; each
// backend records its vendor upscale onto the supplied command list.
pub(in crate::directx) trait UpscaleBackend: Send {
    // The render and output sizes the backend was created for.
    fn extent(&self) -> UpscaleExtent;
    // The output texture the bloom + composite stack samples as the scene.
    fn output(&self) -> &UpscaleOutput;
    // Sub-pixel jitter for this frame's index, shared with the camera
    // projection so the jittered VP and the upscale agree.
    fn jitter_offset(&self, frame_index: u32) -> [f32; 2];
    // Record the upscale onto `cmd`. The inputs are in
    // NON_PIXEL_SHADER_RESOURCE and the output in UNORDERED_ACCESS.
    fn dispatch(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        inputs: UpscaleInputs<'_>,
        camera: UpscaleCamera,
    ) -> RenderResult<()>;
}

// Render-resolution inputs the upscale consumes for one frame.
#[derive(Clone, Copy)]
pub(in crate::directx) struct UpscaleInputs<'a> {
    pub color: &'a ID3D12Resource,
    pub depth: &'a ID3D12Resource,
    pub motion_vectors: &'a ID3D12Resource,
}

// The heap slots of the upscaler's output texture (UAV write + SRV read),
// reserved at init so a resize rebuilds into the same ones.
#[derive(Clone, Copy)]
pub(in crate::directx) struct UpscalerDescriptors {
    pub uav_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub srv_gpu: SrvSlot,
}

// The device an upscaler is created on, and the queue NGX's one-shot feature
// creation is submitted to.
#[derive(Clone, Copy)]
pub(in crate::directx) struct UpscaleDevice<'a> {
    pub device: &'a ID3D12Device,
    #[cfg_attr(
        not(ngx_sdk_bundled),
        expect(dead_code, reason = "only NGX feature creation submits work")
    )]
    pub command_queue: &'a ID3D12CommandQueue,
}

// What every backend is created from.
#[derive(Clone, Copy)]
struct UpscalerTarget<'a> {
    gpu: UpscaleDevice<'a>,
    extent: UpscaleExtent,
    descriptors: UpscalerDescriptors,
}

const UPSCALE_OUTPUT_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R16G16B16A16_FLOAT;

// The output-resolution RGBA16Float texture a backend writes through a UAV and
// the post stack samples through an SRV. Created in UNORDERED_ACCESS; between
// frames it rests in PIXEL_SHADER_RESOURCE once a dispatch has run.
pub(in crate::directx) struct UpscaleOutput {
    resource: ID3D12Resource,
    descriptors: UpscalerDescriptors,
    is_psr: Cell<bool>,
}

impl UpscaleOutput {
    fn create(
        device: &ID3D12Device,
        (width, height): (u32, u32),
        descriptors: UpscalerDescriptors,
    ) -> RenderResult<Self> {
        let heap_props = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: width as u64,
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: UPSCALE_OUTPUT_FORMAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut created: Option<ID3D12Resource> = None;
        // SAFETY: the create descriptor and every pointer it borrows are live for the call, and the
        // new COM object lands in a binding that owns it.
        unsafe {
            device.CreateCommittedResource(
                &heap_props,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                None,
                &mut created,
            )
        }
        .map_err(|e| map_hresult(e.code(), "create upscale output texture"))?;
        let resource = created.ok_or_else(|| {
            RenderError::Other("create upscale output texture returned None".into())
        })?;
        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: UPSCALE_OUTPUT_FORMAT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_UAV {
                    MipSlice: 0,
                    PlaneSlice: 0,
                },
            },
        };
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: UPSCALE_OUTPUT_FORMAT,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 1,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 0.0,
                },
            },
        };
        // SAFETY: the view descriptors and the resource they name are live for the calls, and the
        // destination handles address slots init reserved for these views in a heap it owns.
        unsafe {
            device.CreateUnorderedAccessView(&resource, None, Some(&uav), descriptors.uav_cpu);
            device.CreateShaderResourceView(&resource, Some(&srv), descriptors.srv_cpu);
        }
        Ok(Self {
            resource,
            descriptors,
            is_psr: Cell::new(false),
        })
    }

    pub(in crate::directx) fn resource(&self) -> &ID3D12Resource {
        &self.resource
    }

    pub(in crate::directx) fn descriptors(&self) -> UpscalerDescriptors {
        self.descriptors
    }

    pub(in crate::directx) fn srv_gpu(&self) -> SrvSlot {
        self.descriptors.srv_gpu
    }
}

// A vendor runtime DLL, found beside the executable or on PATH. It is never
// unloaded, so every entry point resolved from it stays valid for the process.
struct Dll(HMODULE);

impl Dll {
    fn open(name: &CStr) -> Option<Self> {
        // SAFETY: `name` is NUL-terminated and live for the call, and a failed load is returned as
        // an error rather than a handle.
        unsafe { LoadLibraryA(PCSTR(name.as_ptr().cast())) }
            .ok()
            .map(Self)
    }
}

impl SdkLibrary for Dll {
    fn symbol(&self, name: &CStr) -> Option<*const c_void> {
        // SAFETY: `self.0` is a loaded module and `name` a NUL-terminated string live for the call.
        unsafe { GetProcAddress(self.0, PCSTR(name.as_ptr().cast())) }.map(|f| f as *const c_void)
    }
}

// The SDKs this build bundles. FFX additionally needs the Agility SDK's newer
// D3D12 runtime beside the executable, which only an opt-in build bundles.
fn availability() -> Availability {
    Availability {
        dlss: cfg!(ngx_sdk_bundled),
        xess: cfg!(xess_sdk_bundled),
        fsr: cfg!(agility_sdk_configured),
    }
}

// Construct the upscaler for the requested backend at `upscale_scale` of
// `output`, falling through the shared order whenever one cannot initialize
// (DLL miss, unsupported GPU, context-init failure). `None` renders at native
// resolution.
pub(in crate::directx) fn build_upscaler(
    gpu: UpscaleDevice<'_>,
    output: (u32, u32),
    upscale_scale: f32,
    descriptors: UpscalerDescriptors,
    requested: UpscalerBackend,
) -> RenderResult<Option<Box<dyn UpscaleBackend>>> {
    let target = UpscalerTarget {
        gpu,
        extent: UpscaleExtent::resolve(output, upscale_scale),
        descriptors,
    };
    let (built, _) = build_first_available(requested, availability(), output, |candidate| {
        Ok(match candidate {
            ResolvedBackend::Fsr => fsr::FsrUpscaler::try_new(target)?.map(boxed),
            ResolvedBackend::Xess => xess::XessUpscaler::try_new(target)?.map(boxed),
            #[cfg(ngx_sdk_bundled)]
            ResolvedBackend::Dlss => dlss::DlssUpscaler::try_new(target)?.map(boxed),
            _ => None,
        })
    })?;
    Ok(built)
}

fn boxed(backend: impl UpscaleBackend + 'static) -> Box<dyn UpscaleBackend> {
    Box::new(backend)
}

impl crate::directx::context::DxContext {
    // Encode the temporal upscale onto `cmd`, the `PassId::Upscale` command
    // list. Runs after SSR resolve / fog / particles (so the input is the fully
    // decorated scene) and before bloom + composite, which sample the output.
    pub(in crate::directx) fn encode_upscale(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        params: &GraphFrameParams<'_>,
    ) -> RenderResult<()> {
        let Some(upscaler) = &self.upscale.backend else {
            return Ok(());
        };
        static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::info!(
                "temporal upscaling: first encode_upscale firing ({})",
                upscaler.extent()
            );
        }
        // Init builds the G-buffer whenever upscaling is on, since it owns the
        // velocity + depth the upscalers read.
        let gb = self.gbuffer.as_ref().ok_or_else(|| {
            RenderError::Other(
                "Upscale enabled but G-buffer resources (velocity / depth) are missing".into(),
            )
        })?;
        let output = upscaler.output();

        // Inputs: the scene the post stack consumes (`post_scene_target` is the
        // one place that choice lives; the executor already moved it to a
        // non-pixel shader resource state, as `Upscale` is a compute node), the
        // G-buffer velocity (a graph resource), and the G-buffer depth, which
        // is not a graph resource and so is flipped here.
        let scene = self.post_scene_target().clone();

        // The output needs flipping back to UNORDERED_ACCESS only once a
        // previous dispatch left it in PIXEL_SHADER_RESOURCE for the post stack.
        let barriers = [
            transition_barrier(
                &gb.depth,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            ),
            transition_barrier(
                output.resource(),
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ),
        ];
        let count = if output.is_psr.get() { 2 } else { 1 };
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe { cmd.ResourceBarrier(&barriers[..count]) };

        upscaler.dispatch(
            cmd,
            UpscaleInputs {
                color: &scene,
                depth: &gb.depth,
                motion_vectors: &gb.velocity,
            },
            UpscaleCamera::new(
                self.upscale.jitter.get(),
                params.elapsed,
                params.near,
                params.fov_y_radians,
            ),
        )?;

        // Give the G-buffer depth back, and hand the output to the post stack.
        // The output is `scene_color` under upscaling, and the one driven
        // resource the graph's resting model cannot express: its between-frames
        // state depends on whether a previous frame dispatched.
        let after = [
            transition_barrier(
                &gb.depth,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
            ),
            transition_barrier(
                output.resource(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            ),
        ];
        output.is_psr.set(true);
        // SAFETY: the command list is in the recording state, and every resource, descriptor and
        // slice these commands name is live for the call.
        unsafe { cmd.ResourceBarrier(&after) };
        Ok(())
    }
}
