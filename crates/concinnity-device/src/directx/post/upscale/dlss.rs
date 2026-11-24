//! NVIDIA DLSS temporal upscaling for the D3D12 backend, through Streamline
//! (see `crate::upscale_sdk::dlss`); RTX only. Available when the hardware's
//! Streamline session reports DLSS on its adapter.

use concinnity_engine::gfx::jitter;
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::output::dlss::DlssPreset;
use concinnity_engine::render::output::upscaler::VendorUpscaler;
use concinnity_engine::render::temporal::history_reset::UpscalerResetLatch;
use concinnity_engine::render::temporal::reactive_mask::ReactiveReader;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use super::{UpscaleBackend, UpscaleInputs, UpscaleOutput, UpscalerTarget};
use crate::upscale_sdk::dlss::{DlssFeature, DlssResources};
use crate::upscale_sdk::streamline::abi::Resource;
use crate::upscale_sdk::{UpscaleCamera, UpscaleExtent};

const LABEL: &str = "DLSS";

// A texture described for Streamline in `state`, which it is in when the
// evaluate's commands run.
fn resource(texture: &ID3D12Resource, state: D3D12_RESOURCE_STATES) -> Resource {
    Resource::texture(texture.as_raw(), state.0.cast_unsigned())
}

// The DLSS viewport and the output texture it writes.
pub(super) struct DlssUpscaler {
    feature: DlssFeature,
    output: UpscaleOutput,
    reset: UpscalerResetLatch,
}

impl DlssUpscaler {
    // `Ok(None)` when DLSS is unavailable: no Streamline session, or DLSS
    // does not run on this adapter. A feature library older than the
    // requested render preset runs the default one.
    pub(super) fn try_new(target: UpscalerTarget<'_>) -> RenderResult<Option<Self>> {
        let UpscalerTarget {
            gpu,
            extent,
            descriptors,
            dlss_preset,
        } = target;
        let Some(sl) = gpu.streamline else {
            return Ok(None);
        };
        let adapter = sl.adapter();
        let Some(feature) = DlssFeature::new(sl.session(), &adapter, extent, dlss_preset, LABEL)
        else {
            return Ok(None);
        };
        let output = UpscaleOutput::create(gpu.device, extent.output, descriptors)?;
        Ok(Some(Self {
            feature,
            output,
            reset: UpscalerResetLatch::default(),
        }))
    }
}

impl UpscaleBackend for DlssUpscaler {
    fn vendor(&self) -> VendorUpscaler {
        VendorUpscaler::Dlss
    }

    fn extent(&self) -> UpscaleExtent {
        self.feature.extent()
    }

    fn output(&self) -> &UpscaleOutput {
        &self.output
    }

    // DLSS prescribes no jitter sequence; the engine's Halton (2, 3) drives
    // both the projection and the evaluate.
    fn jitter_offset(&self, frame_index: u32) -> [f32; 2] {
        jitter::offset(frame_index)
    }

    fn dispatch(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        inputs: UpscaleInputs<'_>,
        camera: UpscaleCamera,
    ) -> RenderResult<()> {
        // The inputs are in NON_PIXEL_SHADER_RESOURCE (or that and the pixel
        // state, as the graph chose). They are tagged PIXEL_SHADER_RESOURCE,
        // which is what Streamline's texture-read state maps to on D3D12, so it
        // records no transitions of its own around the evaluate.
        let read = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        let resources = [
            resource(inputs.color, read),
            resource(inputs.depth, read),
            resource(inputs.motion_vectors, read),
            resource(
                self.output.resource(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ),
        ];
        let [color, depth, motion, output] = &resources;
        let reset = crate::upscale_reset::consume(&self.reset);
        self.feature.evaluate(
            cmd.as_raw(),
            &DlssResources {
                color,
                depth,
                motion,
                output,
            },
            &camera,
            reset,
        )
    }

    fn request_history_reset(&self) {
        self.reset.request();
    }

    fn reactive_reader(&self) -> ReactiveReader {
        ReactiveReader::Dlss
    }

    fn dlss_preset(&self) -> Option<DlssPreset> {
        Some(self.feature.preset())
    }
}
