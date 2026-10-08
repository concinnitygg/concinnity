//! Intel XeSS, whose D3D12 and Vulkan entry points share their result codes,
//! quality and init-flag enums, context lifetime, velocity scale, and the
//! leading and trailing runs of their init and execute parameters. A backend
//! adds its own create, build, init and execute entry points and the resource
//! fields of its API.
//!
//! Layouts match `inc/xess/{xess.h,xess_d3d12.h,xess_vk.h}` of XeSS SDK 3.0.1.
//! `XESS_PACK_B()` is `pack(8)`, a no-op on x86_64 where no field is aligned
//! wider than 8, so `#[repr(C)]` matches byte for byte; the tests pin it.
#![expect(
    non_camel_case_types,
    reason = "the XeSS bindings keep the SDK's own C type names"
)]

use std::ffi::c_void;
use std::ptr;

use concinnity_core::components::UpscaleQuality;
use concinnity_core::render::depth::{CAMERA_DEPTH, DepthMapping};
use concinnity_core::render::history_reset::UpscalerResetLatch;
use concinnity_core::render::reactive_mask::ReactiveReader;

use super::{SdkLibrary, UpscaleExtent, entry_point};

/// `xess_result_t`: zero is success, negative an error, positive a warning.
pub(crate) const XESS_RESULT_SUCCESS: i32 = 0;

// xess_quality_settings_t.
const XESS_QUALITY_SETTING_ULTRA_PERFORMANCE: i32 = 100;
const XESS_QUALITY_SETTING_PERFORMANCE: i32 = 101;
const XESS_QUALITY_SETTING_BALANCED: i32 = 102;
const XESS_QUALITY_SETTING_QUALITY: i32 = 103;
const XESS_QUALITY_SETTING_AA: i32 = 106;

// xess_init_flags_t.
const XESS_INIT_FLAG_INVERTED_DEPTH: u32 = 1 << 1;
const XESS_INIT_FLAG_RESPONSIVE_PIXEL_MASK: u32 = 1 << 3;
const XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE: u32 = 1 << 8;

pub(crate) type xess_context_handle_t = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct xess_2d_t {
    x: u32,
    y: u32,
}

impl From<(u32, u32)> for xess_2d_t {
    fn from((x, y): (u32, u32)) -> Self {
        Self { x, y }
    }
}

/// The fields `xess_d3d12_init_params_t` and `xess_vk_init_params_t` open
/// with, `outputResolution` through `visibleNodeMask`.
#[repr(C)]
pub(crate) struct XessInitHead {
    output_resolution: xess_2d_t,
    quality_setting: i32,
    init_flags: u32,
    creation_node_mask: u32,
    visible_node_mask: u32,
}

impl XessInitHead {
    /// The head for an upscaler of `extent` reading the camera's depth.
    pub(crate) fn new(extent: UpscaleExtent) -> Self {
        Self {
            output_resolution: extent.output.into(),
            quality_setting: xess_quality(UpscaleQuality::nearest(extent.scale)),
            init_flags: xess_init_flags(CAMERA_DEPTH),
            creation_node_mask: 0,
            visible_node_mask: 0,
        }
    }

    /// The init flags, which pipeline building takes as well.
    pub(crate) fn init_flags(&self) -> u32 {
        self.init_flags
    }
}

/// The fields `xess_d3d12_execute_params_t` and `xess_vk_execute_params_t`
/// share after their textures, `jitterOffsetX` through `outputColorBase`.
#[repr(C)]
pub(crate) struct XessExecuteFrame {
    jitter_offset_x: f32,
    jitter_offset_y: f32,
    exposure_scale: f32,
    reset_history: u32,
    input_width: u32,
    input_height: u32,
    input_color_base: xess_2d_t,
    input_motion_vector_base: xess_2d_t,
    input_depth_base: xess_2d_t,
    input_responsive_mask_base: xess_2d_t,
    reserved0: xess_2d_t,
    output_color_base: xess_2d_t,
}

impl XessExecuteFrame {
    // One frame at `jitter` (render pixels, within `[-0.5, 0.5]`) over the
    // whole render and output extents.
    fn new(jitter: [f32; 2], reset: bool, extent: UpscaleExtent) -> Self {
        let origin = xess_2d_t { x: 0, y: 0 };
        Self {
            jitter_offset_x: jitter[0],
            jitter_offset_y: jitter[1],
            exposure_scale: 1.0,
            reset_history: u32::from(reset),
            input_width: extent.render.0,
            input_height: extent.render.1,
            input_color_base: origin,
            input_motion_vector_base: origin,
            input_depth_base: origin,
            input_responsive_mask_base: origin,
            reserved0: origin,
            output_color_base: origin,
        }
    }
}

type PfnXessDestroyContext = unsafe extern "C" fn(ctx: xess_context_handle_t) -> i32;
type PfnXessSetVelocityScale =
    unsafe extern "C" fn(ctx: xess_context_handle_t, x: f32, y: f32) -> i32;
type PfnXessSetMaxResponsiveMaskValue =
    unsafe extern "C" fn(ctx: xess_context_handle_t, value: f32) -> i32;

// The XeSS quality preset nearest the engine preset, native anti-aliasing at
// native resolution. It hints XeSS's model selection; the render size itself is
// passed with every execute.
fn xess_quality(quality: Option<UpscaleQuality>) -> i32 {
    match quality {
        None => XESS_QUALITY_SETTING_AA,
        Some(UpscaleQuality::Quality) => XESS_QUALITY_SETTING_QUALITY,
        Some(UpscaleQuality::Balanced) => XESS_QUALITY_SETTING_BALANCED,
        Some(UpscaleQuality::Performance) => XESS_QUALITY_SETTING_PERFORMANCE,
        Some(UpscaleQuality::UltraPerformance) => XESS_QUALITY_SETTING_ULTRA_PERFORMANCE,
    }
}

// HDR linear color and render-resolution motion vectors scaled to pixels by
// the velocity scale, so: XeSS's own auto-exposure (the scene is un-exposed
// until after the upscale), inverted depth when the near plane is device depth
// 1, and the engine's reactive mask as the responsive pixel mask, which every
// execute must then supply.
const fn xess_init_flags(depth: DepthMapping) -> u32 {
    let flags = XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE | XESS_INIT_FLAG_RESPONSIVE_PIXEL_MASK;
    if depth.reversed {
        flags | XESS_INIT_FLAG_INVERTED_DEPTH
    } else {
        flags
    }
}

/// The entry points both APIs share.
#[derive(Clone, Copy)]
pub(crate) struct XessCommonApi {
    destroy_context: PfnXessDestroyContext,
    set_velocity_scale: PfnXessSetVelocityScale,
    set_max_responsive_mask_value: PfnXessSetMaxResponsiveMaskValue,
}

impl XessCommonApi {
    pub(crate) fn resolve(library: &impl SdkLibrary) -> Option<Self> {
        // SAFETY: each type is the prototype `xess.h` declares for that export.
        unsafe {
            Some(Self {
                destroy_context: entry_point(library, c"xessDestroyContext")?,
                set_velocity_scale: entry_point(library, c"xessSetVelocityScale")?,
                set_max_responsive_mask_value: entry_point(
                    library,
                    c"xessSetMaxResponsiveMaskValue",
                )?,
            })
        }
    }
}

/// A XeSS context for one render-to-output extent, with the runtime library
/// its entry points come from. Destroyed on drop, before the library is
/// released.
pub(crate) struct XessContext<L> {
    handle: xess_context_handle_t,
    api: XessCommonApi,
    extent: UpscaleExtent,
    reset: UpscalerResetLatch,
    _library: L,
}

impl<L> XessContext<L> {
    /// Take ownership of `handle`, a context `library` created for `extent`.
    pub(crate) fn adopt(
        library: L,
        api: XessCommonApi,
        handle: xess_context_handle_t,
        extent: UpscaleExtent,
    ) -> Self {
        Self {
            handle,
            api,
            extent,
            reset: UpscalerResetLatch::default(),
            _library: library,
        }
    }

    pub(crate) fn handle(&self) -> xess_context_handle_t {
        self.handle
    }

    pub(crate) fn extent(&self) -> UpscaleExtent {
        self.extent
    }

    /// The shared execute fields of this frame at `jitter`. The first frame
    /// after creation, and the first after [`Self::request_history_reset`],
    /// resets XeSS's history.
    pub(crate) fn frame(&self, jitter: [f32; 2]) -> XessExecuteFrame {
        XessExecuteFrame::new(
            jitter,
            crate::upscale_reset::consume(&self.reset),
            self.extent,
        )
    }

    /// Discard XeSS's history on the next frame.
    pub(crate) fn request_history_reset(&self) {
        self.reset.request();
    }

    /// Scale the engine's UV-space motion vectors into the render pixels XeSS
    /// reads. A failure is logged under `label` and otherwise ignored.
    pub(crate) fn set_velocity_scale(&self, label: &str) {
        let (w, h) = self.extent.render;
        // SAFETY: `self.handle` is the live context; the call takes only scalars besides it.
        let rc = unsafe { (self.api.set_velocity_scale)(self.handle, w as f32, h as f32) };
        if rc != XESS_RESULT_SUCCESS {
            tracing::warn!("{label}: xessSetVelocityScale returned {rc} (non-fatal)");
        }
    }

    /// Clip the responsive mask at the value the engine caps XeSS's reactive
    /// input at. A failure is logged under `label` and otherwise ignored.
    pub(crate) fn set_responsive_mask_cap(&self, label: &str) {
        let Some(cap) = ReactiveReader::Xess.cap() else {
            return;
        };
        // SAFETY: `self.handle` is the live context; the call takes only a scalar besides it.
        let rc = unsafe { (self.api.set_max_responsive_mask_value)(self.handle, cap) };
        if rc != XESS_RESULT_SUCCESS {
            tracing::warn!("{label}: xessSetMaxResponsiveMaskValue returned {rc} (non-fatal)");
        }
    }

    /// Destroy the context now; dropping it afterwards does nothing more. The
    /// device it was created on must be idle and still alive.
    pub(crate) fn destroy(&mut self) {
        if self.handle.is_null() {
            return;
        }
        // SAFETY: `self.handle` is the live, non-null context, destroyed exactly once (nulled
        // straight after) while the library its entry point came from is still loaded.
        unsafe { (self.api.destroy_context)(self.handle) };
        self.handle = ptr::null_mut();
    }
}

impl<L> Drop for XessContext<L> {
    fn drop(&mut self) {
        self.destroy();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    #[test]
    fn shared_xess_layouts_match_sdk_v301() {
        assert_eq!(size_of::<xess_2d_t>(), 8);
        assert_eq!(offset_of!(xess_2d_t, x), 0);
        assert_eq!(offset_of!(xess_2d_t, y), 4);

        assert_eq!(size_of::<XessInitHead>(), 24);
        assert_eq!(align_of::<XessInitHead>(), 4);
        assert_eq!(offset_of!(XessInitHead, output_resolution), 0);
        assert_eq!(offset_of!(XessInitHead, quality_setting), 8);
        assert_eq!(offset_of!(XessInitHead, init_flags), 12);
        assert_eq!(offset_of!(XessInitHead, creation_node_mask), 16);
        assert_eq!(offset_of!(XessInitHead, visible_node_mask), 20);

        type F = XessExecuteFrame;
        assert_eq!(size_of::<F>(), 72);
        assert_eq!(align_of::<F>(), 4);
        assert_eq!(offset_of!(F, jitter_offset_x), 0);
        assert_eq!(offset_of!(F, jitter_offset_y), 4);
        assert_eq!(offset_of!(F, exposure_scale), 8);
        assert_eq!(offset_of!(F, reset_history), 12);
        assert_eq!(offset_of!(F, input_width), 16);
        assert_eq!(offset_of!(F, input_height), 20);
        assert_eq!(offset_of!(F, input_color_base), 24);
        assert_eq!(offset_of!(F, input_motion_vector_base), 32);
        assert_eq!(offset_of!(F, input_depth_base), 40);
        assert_eq!(offset_of!(F, input_responsive_mask_base), 48);
        assert_eq!(offset_of!(F, reserved0), 56);
        assert_eq!(offset_of!(F, output_color_base), 64);
    }

    #[test]
    fn xess_constants_match_sdk_v301() {
        assert_eq!(XESS_QUALITY_SETTING_ULTRA_PERFORMANCE, 100);
        assert_eq!(XESS_QUALITY_SETTING_PERFORMANCE, 101);
        assert_eq!(XESS_QUALITY_SETTING_BALANCED, 102);
        assert_eq!(XESS_QUALITY_SETTING_QUALITY, 103);
        assert_eq!(XESS_QUALITY_SETTING_AA, 106);
        assert_eq!(XESS_INIT_FLAG_INVERTED_DEPTH, 2);
        assert_eq!(XESS_INIT_FLAG_RESPONSIVE_PIXEL_MASK, 8);
        assert_eq!(XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE, 256);
    }

    #[test]
    fn quality_follows_the_nearest_engine_preset() {
        assert_eq!(xess_quality(None), XESS_QUALITY_SETTING_AA);
        assert_eq!(
            xess_quality(Some(UpscaleQuality::Quality)),
            XESS_QUALITY_SETTING_QUALITY
        );
        assert_eq!(
            xess_quality(Some(UpscaleQuality::UltraPerformance)),
            XESS_QUALITY_SETTING_ULTRA_PERFORMANCE
        );
        let native = XessInitHead::new(UpscaleExtent::resolve((64, 64), 1.0));
        assert_eq!(native.quality_setting, XESS_QUALITY_SETTING_AA);
    }

    // The inverted-depth bit is set exactly when the depth is reversed, and
    // the camera's depth is.
    #[test]
    fn the_depth_flag_follows_the_depth_mapping() {
        for reversed in [false, true] {
            for infinite in [false, true] {
                let flags = xess_init_flags(DepthMapping { reversed, infinite });
                assert_eq!((flags & XESS_INIT_FLAG_INVERTED_DEPTH) != 0, reversed);
                assert_ne!(flags & XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE, 0);
                assert_ne!(flags & XESS_INIT_FLAG_RESPONSIVE_PIXEL_MASK, 0);
            }
        }
        assert_ne!(
            xess_init_flags(CAMERA_DEPTH) & XESS_INIT_FLAG_INVERTED_DEPTH,
            0
        );
    }

    #[test]
    fn a_frame_covers_the_render_extent_from_the_origin() {
        let extent = UpscaleExtent::resolve((1920, 1080), 0.5);
        let frame = XessExecuteFrame::new([0.25, -0.25], true, extent);
        assert_eq!((frame.input_width, frame.input_height), (960, 540));
        assert_eq!(frame.reset_history, 1);
        assert_eq!(frame.exposure_scale, 1.0);
        assert_eq!(
            (frame.jitter_offset_x, frame.jitter_offset_y),
            (0.25, -0.25)
        );
        assert_eq!(
            XessExecuteFrame::new([0.0; 2], false, extent).reset_history,
            0
        );
    }
}
