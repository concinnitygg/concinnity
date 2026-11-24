//! NVIDIA DLSS temporal upscaling for the Vulkan backend, through Streamline
//! (see `crate::upscale_sdk::dlss`); RTX on Windows only. Available when the
//! device was created under a Streamline session that reports DLSS on its
//! physical device; Streamline enabled the extensions DLSS needs then.
//!
//! Vulkan has no query for an image's description, so each tagged image
//! carries its view, layout, format, size and usage, and the depth its aspect.

use std::cell::Cell;

use ash::vk;
use concinnity_engine::gfx::jitter;
use concinnity_engine::render::backend::error::RenderResult;
use concinnity_engine::render::output::dlss::DlssPreset;
use concinnity_engine::render::output::upscaler::VendorUpscaler;
use concinnity_engine::render::temporal::history_reset::UpscalerResetLatch;
use concinnity_engine::render::temporal::reactive_mask::ReactiveReader;

use super::{
    OutputWrites, UpscaleBackend, UpscaleImage, UpscaleInputs, UpscaleOutput, UpscalerGpu,
};
use crate::upscale_sdk::dlss::{DlssFeature, DlssResources};
use crate::upscale_sdk::streamline::abi::{AdapterInfo, Resource, SubresourceRange};
use crate::upscale_sdk::{UpscaleCamera, UpscaleExtent};

const LABEL: &str = "DLSS (Vulkan)";

// `img` described for Streamline in `layout`, which it is in when the
// evaluate's commands run.
fn resource(img: &UpscaleImage, layout: vk::ImageLayout, usage: vk::ImageUsageFlags) -> Resource {
    use ash::vk::Handle;
    let mut resource = Resource::texture(
        img.image.as_raw() as *mut std::ffi::c_void,
        layout.as_raw().cast_unsigned(),
    );
    resource.view = img.view.as_raw() as *mut std::ffi::c_void;
    resource.width = img.width;
    resource.height = img.height;
    resource.native_format = img.format.as_raw().cast_unsigned();
    resource.mip_levels = 1;
    resource.array_layers = 1;
    resource.usage = usage.as_raw();
    resource
}

// The DLSS viewport and the output image it writes.
pub(super) struct DlssUpscaler {
    // `None` once `destroy` released it.
    feature: Option<DlssFeature>,
    extent: UpscaleExtent,
    output: UpscaleOutput,
    jitter: Cell<[f32; 2]>,
    reset: UpscalerResetLatch,
}

impl DlssUpscaler {
    // `Ok(None)` when DLSS is unavailable: the device was not created under
    // Streamline, or DLSS does not run on its physical device. A feature
    // library older than the requested render preset runs the default one.
    pub(super) fn try_new(
        gpu: UpscalerGpu<'_>,
        extent: UpscaleExtent,
        dlss_preset: DlssPreset,
    ) -> RenderResult<Option<Self>> {
        use ash::vk::Handle;
        let Some(sl) = gpu.device.streamline() else {
            return Ok(None);
        };
        let adapter =
            AdapterInfo::vk_physical_device(gpu.physical_device.as_raw() as *mut std::ffi::c_void);
        let Some(feature) = DlssFeature::new(sl, &adapter, extent, dlss_preset, LABEL) else {
            return Ok(None);
        };
        // NGX clears the output inside its evaluate as well as writing it.
        let output = UpscaleOutput::create(gpu, extent.output, OutputWrites::storage_and_clear())?;
        Ok(Some(Self {
            feature: Some(feature),
            extent,
            output,
            jitter: Cell::new([0.0, 0.0]),
            reset: UpscalerResetLatch::default(),
        }))
    }
}

impl UpscaleBackend for DlssUpscaler {
    fn vendor(&self) -> VendorUpscaler {
        VendorUpscaler::Dlss
    }

    fn extent(&self) -> UpscaleExtent {
        self.extent
    }

    fn output(&self) -> &UpscaleOutput {
        &self.output
    }

    // DLSS prescribes no jitter sequence; the engine's Halton (2, 3) drives
    // both the projection and the evaluate.
    fn jitter_offset(&self, frame_index: u32) -> [f32; 2] {
        jitter::offset(frame_index)
    }

    fn jitter(&self) -> &Cell<[f32; 2]> {
        &self.jitter
    }

    fn dispatch(
        &self,
        cmd: vk::CommandBuffer,
        inputs: UpscaleInputs<'_>,
        camera: UpscaleCamera,
    ) -> RenderResult<()> {
        use ash::vk::Handle;
        let Some(feature) = &self.feature else {
            return Ok(());
        };
        let read = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        let sampled = vk::ImageUsageFlags::SAMPLED;
        let depth_range = SubresourceRange::single(inputs.depth.aspect.as_raw());
        let resources = [
            resource(inputs.color, read, sampled),
            resource(inputs.depth, read, sampled).with_subresource(&depth_range),
            resource(inputs.motion, read, sampled),
            resource(
                &self.output.as_upscale_image(),
                vk::ImageLayout::GENERAL,
                self.output.writes.image_usage(),
            ),
        ];
        let [color, depth, motion, output] = &resources;
        let reset = crate::upscale_reset::consume(&self.reset);
        feature.evaluate(
            cmd.as_raw() as *mut std::ffi::c_void,
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
        self.feature.as_ref().map(DlssFeature::preset)
    }

    fn destroy(&mut self) {
        self.feature = None;
        self.output.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tagged_image_carries_its_vulkan_description() {
        use ash::vk::Handle;
        let img = UpscaleImage {
            image: vk::Image::from_raw(0x10),
            view: vk::ImageView::from_raw(0x20),
            format: vk::Format::D32_SFLOAT,
            width: 640,
            height: 360,
            aspect: vk::ImageAspectFlags::DEPTH,
        };
        let r = resource(
            &img,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageUsageFlags::SAMPLED,
        );
        assert_eq!(r.native as u64, 0x10);
        assert_eq!(r.view as u64, 0x20);
        assert_eq!((r.width, r.height), (640, 360));
        assert_eq!(
            r.native_format,
            vk::Format::D32_SFLOAT.as_raw().cast_unsigned()
        );
        assert_eq!(
            r.state,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                .as_raw()
                .cast_unsigned()
        );
        assert_eq!(r.usage, vk::ImageUsageFlags::SAMPLED.as_raw());
    }
}
