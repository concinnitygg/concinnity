//! The hardware a context builds on: the NSWindow + MTKView, the device with its
//! command queues and allocator, the EDR negotiation, and the initial HDR target
//! sizing decision (geometry-less worlds clamp to 1x1; otherwise the drawable
//! size wins, falling back to the requested width/height before the drawable
//! exists).
#![deny(unsafe_op_in_unsafe_fn)]

use concinnity_core::render::backend_init::{EmbeddedSurface, SwapchainConfig};
use concinnity_core::render::error::{RenderError, RenderResult};
use concinnity_core::render::hdr_output;
use concinnity_core::render::hdr_output::HdrOutputMode;
use objc2::MainThreadOnly;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_app_kit::{NSApplication, NSAutoresizingMaskOptions, NSScreen, NSView, NSWindow};
use objc2_core_graphics::{
    CGColorSpace, kCGColorSpaceDisplayP3_PQ, kCGColorSpaceExtendedLinearDisplayP3,
};
use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice, MTLPixelFormat};
use objc2_metal_kit::MTKView;
use objc2_quartz_core::CAMetalLayer;

use crate::metal::allocator::DeviceAllocator;
use crate::metal::context::{MtlHardware, WindowState};
use crate::metal::graph_queues::GraphQueues;

pub(crate) struct WindowSetup {
    pub window: Option<Retained<NSWindow>>,
    pub mtk_view: Retained<MTKView>,
    pub pump_events: bool,
    pub initial_w: u32,
    pub initial_h: u32,
    // Shared native-fullscreen flag, kept in sync by `window_delegate`. False
    // in embedded mode (no NSWindow).
    pub fullscreen: std::sync::Arc<std::sync::atomic::AtomicBool>,
    // NSWindowDelegate tracking the fullscreen transition; None in embedded
    // mode. The caller stores it so the window's weak delegate stays attached.
    pub window_delegate: Option<Retained<crate::appkit::window_delegate::WindowDelegate>>,
    // Resolved swapchain color-output mode. `Sdr` when the world did not
    // request HDR or the active display lacks EDR headroom; `Hdr` when the
    // CAMetalLayer was configured with `RGBA16Float` + extended-linear
    // Display P3 color space + `wantsExtendedDynamicRangeContent = true`.
    pub hdr_mode: HdrOutputMode,
}

// The window's own configuration: its title, requested size, whether the title
// bar is drawn, whether the world is geometry-less (clamps the initial HDR
// targets to 1x1), whether frame capture is enabled (drives `framebufferOnly`
// on the MTKView), and the host view to embed into instead of a window.
pub(crate) struct WindowConfig<'a> {
    pub title: &'a str,
    pub width: u32,
    pub height: u32,
    pub title_bar: bool,
    pub geometry_less: bool,
    pub capture_enabled: bool,
    pub embedded: Option<EmbeddedSurface>,
}

// The world's HDR-output request, resolved against the active display's EDR
// headroom to pick the swapchain color-output mode.
#[derive(Clone, Copy)]
pub(crate) struct HdrRequest {
    pub display_requested: bool,
    pub pq_requested: bool,
}

// Acquire the hardware a context builds on, with the initial scene target size.
// A live reload adopts the handed-over device, queues, allocator and window,
// re-resolving the HDR mode on the inherited view; a fresh build creates them.
pub(super) fn setup(
    reuse: Option<MtlHardware>,
    config: WindowConfig,
    hdr: HdrRequest,
    frames_in_flight: usize,
    vsync: bool,
) -> RenderResult<(MtlHardware, (u32, u32))> {
    // all Metal and AppKit calls must happen on the main thread
    let mtm = objc2::MainThreadMarker::new().ok_or_else(|| {
        RenderError::Other("MtlContext::new must be called from the main thread".into())
    })?;
    let swapchain_config = SwapchainConfig {
        frames_in_flight: frames_in_flight.max(1),
        hdr_display: hdr.display_requested,
        hdr_pq: hdr.pq_requested,
    };
    let title_bar = config.title_bar;

    let (hw, initial_w, initial_h) = match reuse {
        Some(mut hw) => {
            let view = &hw
                .window
                .as_ref()
                .ok_or_else(|| {
                    RenderError::Other("reload_world: handed-over hardware has no window".into())
                })?
                .view;
            let (hdr_mode, initial_w, initial_h) = reconfigure_view(mtm, view, config, hdr);
            hw.swap_pixel_format = swap_pixel_format(hdr_mode);
            hw.hdr_mode = hdr_mode;
            hw.swapchain_config = swapchain_config;
            (hw, initial_w, initial_h)
        }
        None => {
            let device = MTLCreateSystemDefaultDevice()
                .ok_or_else(|| RenderError::Other("no default Metal device".into()))?;
            let command_queue = device
                .newCommandQueue()
                .ok_or_else(|| RenderError::Other("failed to create Metal command queue".into()))?;
            // The block pool the world's persistent buffers and textures are
            // placed in, per context: a live reload hands its successor a
            // fresh one, so the outgoing context's heaps go with it.
            let allocator = DeviceAllocator::new(&device, frames_in_flight);
            let WindowSetup {
                window,
                mtk_view,
                pump_events,
                initial_w,
                initial_h,
                fullscreen,
                window_delegate,
                hdr_mode,
            } = setup_window_and_view(mtm, &device, config, hdr)?;
            // Second queue + per-queue events for the render graph's
            // two-queue schedule. Falls back to a single-queue submission
            // when the device will not create them.
            let graph_queues = GraphQueues::new(&device);
            if graph_queues.is_none() {
                tracing::warn!(
                    "metal: no async-compute queue, submitting the render graph on one queue"
                );
            }
            let window = WindowState {
                appkit: crate::appkit::AppKitWindow::new(crate::appkit::AppKitWindowParts {
                    window,
                    // The shared layer drives the view through NSView alone;
                    // the MTKView below stays for drawable acquisition.
                    view: Retained::into_super(mtk_view.clone()),
                    title_bar,
                    pump_events,
                    fullscreen,
                    window_delegate,
                }),
                view: mtk_view,
                was_visible: false,
            };
            let hw = MtlHardware {
                device,
                allocator,
                command_queue,
                graph_queues,
                swap_pixel_format: swap_pixel_format(hdr_mode),
                hdr_mode,
                swapchain_config,
                window: Some(window),
            };
            (hw, initial_w, initial_h)
        }
    };
    if let Some(w) = &hw.window {
        // Honor the requested vsync on the backing CAMetalLayer (default
        // CAMetalLayer presentation is display-synced).
        set_display_sync(&w.view, vsync);
    }
    Ok((hw, (initial_w, initial_h)))
}

pub(crate) fn setup_window_and_view(
    mtm: objc2::MainThreadMarker,
    device: &ProtocolObject<dyn MTLDevice>,
    config: WindowConfig,
    hdr: HdrRequest,
) -> RenderResult<WindowSetup> {
    // Resolve the swapchain color-output mode. EDR support is per-display,
    // so the answer depends on which screen the window will land on. In
    // windowed mode we use `NSWindow::screen()` after attaching; in embedded
    // mode (preview) we fall back to the main screen since the parent NSView
    // does not always have a window at this point. The asset toggle is the
    // outer gate: a world that did not opt in stays SDR even on a capable
    // panel.
    let max_edr = measure_max_edr(mtm);
    let hdr_mode = HdrOutputMode::resolve(hdr.display_requested, hdr.pq_requested, max_edr);
    if hdr.display_requested && !hdr_mode.is_hdr() {
        tracing::warn!(
            "HDR display requested but the active display reports max EDR \
             multiplier {:.3}: falling back to SDR (BGRA8Unorm) output",
            max_edr
        );
    } else if let HdrOutputMode::Hdr { encoding, .. } = hdr_mode {
        tracing::info!(
            "HDR display output enabled: max EDR multiplier {:.3}, encoding={:?}",
            max_edr,
            encoding,
        );
    }

    // A window of our own always pumps events; an embedded view pumps only when
    // the host asks, since the host usually dispatches input itself.
    let embedded = config.embedded;
    let pump_events = embedded.is_none_or(|s| s.pump_events);
    let (window, mtk_view, fullscreen, window_delegate) = if let Some(surface) = embedded {
        // Embedded mode: attach an MTKView as a subview of the host's NSView.
        // SAFETY: `EmbeddedSurface` requires the host to keep its view alive for
        // the world's lifetime, which covers this borrow, and `surface.view` is
        // non-null by construction. This runs on the main thread (`mtm`).
        let parent: &NSView = unsafe { surface.view.cast::<NSView>().as_ref() };
        let bounds = parent.bounds();
        let mtk_view = MTKView::initWithFrame_device(MTKView::alloc(mtm), bounds, Some(device));
        configure_mtk_view(&mtk_view, hdr_mode, config.capture_enabled);
        mtk_view.setAutoresizingMask(
            NSAutoresizingMaskOptions::ViewWidthSizable
                | NSAutoresizingMaskOptions::ViewHeightSizable,
        );
        parent.addSubview(&mtk_view);
        // No NSWindow we own in embedded mode, so no fullscreen delegate; the
        // flag stays false (set_window_mode is a no-op without self.window).
        (
            None,
            mtk_view,
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            None,
        )
    } else {
        // Windowed mode: create a new NSWindow containing the MTKView.
        let window = crate::appkit::chrome::create_window(
            mtm,
            config.title,
            config.width,
            config.height,
            config.title_bar,
        )
        .map_err(RenderError::Other)?;
        let content_rect = window.contentRectForFrameRect(window.frame());
        let mtk_view =
            MTKView::initWithFrame_device(MTKView::alloc(mtm), content_rect, Some(device));
        configure_mtk_view(&mtk_view, hdr_mode, config.capture_enabled);
        window.setContentView(Some(&mtk_view));
        // Track native-fullscreen state authoritatively (the style-mask bit
        // lags the animated transition) so the settings menu's Window Mode row
        // never toggles the wrong way.
        let (delegate, fullscreen) =
            crate::appkit::window_delegate::attach_fullscreen_delegate(mtm, &window);
        NSApplication::sharedApplication(mtm).activate();
        window.makeKeyAndOrderFront(None);
        (Some(window), mtk_view, fullscreen, Some(delegate))
    };

    let drawable = mtk_view.drawableSize();
    let (initial_w, initial_h) = initial_target_size((drawable.width, drawable.height), &config);

    Ok(WindowSetup {
        window,
        mtk_view,
        pump_events,
        initial_w,
        initial_h,
        fullscreen,
        window_delegate,
        hdr_mode,
    })
}

// Re-resolve the swapchain color-output mode on the view a live world reload
// (`cn editor` SAVE) inherits, returning it with the initial target size. The
// reload is only chosen when the swapchain REQUEST is unchanged, but the
// display's actual EDR headroom can still have changed since the original build
// (a monitor plugged/unplugged, HDR toggled), which flips the resolved mode and
// therefore the drawable pixel format. Re-running `configure_mtk_view` keeps the
// CAMetalLayer's format in sync with the pipelines `build` rebuilds from it; no
// window is created, activated or re-delegated, so a save does not steal focus.
pub(crate) fn reconfigure_view(
    mtm: objc2::MainThreadMarker,
    mtk_view: &MTKView,
    config: WindowConfig,
    hdr: HdrRequest,
) -> (HdrOutputMode, u32, u32) {
    let max_edr = measure_max_edr(mtm);
    let hdr_mode = HdrOutputMode::resolve(hdr.display_requested, hdr.pq_requested, max_edr);
    configure_mtk_view(mtk_view, hdr_mode, config.capture_enabled);
    let drawable = mtk_view.drawableSize();
    let (initial_w, initial_h) = initial_target_size((drawable.width, drawable.height), &config);
    (hdr_mode, initial_w, initial_h)
}

// Initial HDR target sizing from the view's drawable size. The drawable may not
// exist yet (especially in embedded mode before the parent view finishes
// layout), so the requested width/height stands in and draw_frame resizes the
// targets once the actual drawable size differs. A geometry-less world (e.g.
// text-only) renders no 3D content into the off-screen HDR / bloom / effect
// targets, so they are allocated at 1x1 rather than paying for a full MSAA HDR
// color + depth + bloom chain (tens of MB) for a trivial 2D world.
fn initial_target_size(drawable: (f64, f64), config: &WindowConfig) -> (u32, u32) {
    if config.geometry_less {
        return (1, 1);
    }
    let pick = |live: f64, requested: u32| {
        if live > 0.0 {
            live as u32
        } else {
            requested.max(1)
        }
    };
    (
        pick(drawable.0, config.width),
        pick(drawable.1, config.height),
    )
}

// Largest extended-range color-component multiplier the system thinks any
// attached screen can drive. SDR panels report `1.0`; HDR panels report
// `2.0`+. With no screens at all (a head-less unit test or detached embedded
// preview) the function returns `1.0` so the resolver stays on the SDR
// path.
//
// We query the *potential* headroom rather than the *current* headroom
// because `measure_max_edr` runs before the CAMetalLayer is configured for
// EDR: at that point macOS hasn't yet allocated any HDR headroom for our
// window, so `maximumExtendedDynamicRangeColorComponentValue` returns the
// idle value (commonly `1.0`) regardless of the panel's capabilities. The
// `Potential` API reports what the panel CAN do once HDR content is on
// screen, which is what we need at gate time. A separate live readout
// could later poll the dynamic value each frame for an in-game brightness
// monitor; the static gate only needs to know whether HDR is a possibility.
pub(crate) fn measure_max_edr(mtm: objc2::MainThreadMarker) -> f32 {
    let screens = NSScreen::screens(mtm);
    let mut best: f64 = 1.0;
    for i in 0..screens.count() {
        let s = screens.objectAtIndex(i);
        let v = s.maximumPotentialExtendedDynamicRangeColorComponentValue();
        if v > best {
            best = v;
        }
    }
    best as f32
}

// The drawable is now the composite/post-pass target only -- main pass renders
// into an off-screen RGBA16Float MSAA target with its own depth. The drawable
// therefore has no depth attachment and no MSAA.
//
// In HDR mode the swapchain color attachment is widened from BGRA8Unorm to
// RGBA16Float and the underlying CAMetalLayer is reconfigured for extended
// dynamic-range output: extended-linear Display P3 color space, EDR content
// enabled. The post-process fragment then writes linear extended-range values
// straight through (no tonemap / gamma).
fn configure_mtk_view(mtk_view: &MTKView, hdr_mode: HdrOutputMode, capture_enabled: bool) {
    mtk_view.setPaused(true);
    mtk_view.setEnableSetNeedsDisplay(false);
    let swap_fmt = swap_pixel_format(hdr_mode);
    mtk_view.setColorPixelFormat(swap_fmt);
    mtk_view.setDepthStencilPixelFormat(MTLPixelFormat::Invalid);
    mtk_view.setSampleCount(1);
    // The drawable defaults to `framebufferOnly` (write-as-attachment only),
    // which forbids using it as a blit source. Under the capture-enabled
    // (`cn debug`) path the `screenshot` command blits the last presented
    // drawable back to the host, so switch it off there. Left at the default
    // in production so a normal `cn run` pays nothing for a debug-only feature.
    if capture_enabled {
        mtk_view.setFramebufferOnly(false);
    }

    if let HdrOutputMode::Hdr { encoding, .. } = hdr_mode {
        configure_hdr_layer(mtk_view, encoding);
    }
}

// Pixel format the swapchain attachment uses. BGRA8Unorm is the historical
// SDR default; RGBA16Float gives the EDR path the headroom to drive values
// past SDR reference white without crushing precision.
pub(crate) fn swap_pixel_format(hdr_mode: HdrOutputMode) -> MTLPixelFormat {
    if hdr_mode.is_hdr() {
        MTLPixelFormat::RGBA16Float
    } else {
        MTLPixelFormat::BGRA8Unorm
    }
}

// Turn display sync (vsync) on or off on the MTKView's backing CAMetalLayer.
// `displaySyncEnabled` true locks presentation to the display refresh; false
// presents as soon as a frame is ready (uncapped, possible tearing). Used both
// at init (to honor GraphicsConfig.vsync) and at runtime (settings menu). A nil
// or non-CAMetalLayer layer is treated as a no-op, matching configure_hdr_layer.
pub(crate) fn set_display_sync(mtk_view: &MTKView, on: bool) {
    let Some(layer) = mtk_view.layer() else {
        return;
    };
    if let Some(metal_layer) = layer.downcast_ref::<CAMetalLayer>() {
        metal_layer.setDisplaySyncEnabled(on);
    }
}

// Apply EDR layer flags to the MTKView's backing CAMetalLayer. MTKView's
// `layer` property is documented to be a CAMetalLayer, but `NSView::layer()`
// returns the parent CALayer type: we cast through the runtime-safe `cast`
// path and only flip the EDR + color-space switches once we have it. A nil
// layer is reported and treated as a no-op (the renderer then falls back to
// the standard sRGB SDR path silently; we already log warn-level above when
// EDR is requested but not achievable).
fn configure_hdr_layer(mtk_view: &MTKView, encoding: hdr_output::HdrEncoding) {
    let Some(layer) = mtk_view.layer() else {
        tracing::warn!(
            "HDR display requested but MTKView has no backing layer: falling back to SDR"
        );
        return;
    };
    // The MTKView documents its layer is always CAMetalLayer. The runtime
    // class check inside `downcast_ref` keeps this safe in the unlikely event
    // that a future MTKView build hands us something else.
    let metal_layer: &CAMetalLayer = match layer.downcast_ref::<CAMetalLayer>() {
        Some(l) => l,
        None => {
            tracing::warn!(
                "HDR display requested but the MTKView layer is not a CAMetalLayer: falling \
                 back to SDR"
            );
            return;
        }
    };
    // Pick the swapchain color space by encoding:
    //   - ExtendedLinear → kCGColorSpaceExtendedLinearDisplayP3. The shader
    //     writes linear values where `1.0` is SDR reference white; the
    //     compositor handles the panel-side encode.
    //   - Pq → kCGColorSpaceDisplayP3_PQ. The shader emits PQ-encoded values
    //     directly; the panel decodes via the PQ EOTF. Same Display P3
    //     primaries as the linear path so the gamut situation is unchanged.
    let (name, label): (_, &str) = match encoding {
        hdr_output::HdrEncoding::ExtendedLinear => (
            // SAFETY: the CoreGraphics color-space name is a framework-owned static that outlives
            // this borrow.
            unsafe { kCGColorSpaceExtendedLinearDisplayP3 },
            "kCGColorSpaceExtendedLinearDisplayP3",
        ),
        hdr_output::HdrEncoding::Pq => (
            // SAFETY: the CoreGraphics color-space name is a framework-owned static that outlives
            // this borrow.
            unsafe { kCGColorSpaceDisplayP3_PQ },
            "kCGColorSpaceDisplayP3_PQ",
        ),
    };
    let colorspace = CGColorSpace::with_name(Some(name));
    match colorspace.as_deref() {
        Some(cs) => metal_layer.setColorspace(Some(cs)),
        None => tracing::warn!(
            "{} unavailable: leaving CAMetalLayer at default color space (HDR output may \
             look desaturated)",
            label
        ),
    }
    metal_layer.setWantsExtendedDynamicRangeContent(true);
    // CAMetalLayer.pixelFormat is normally set by MTKView::setColorPixelFormat
    // above, but make it explicit so a future refactor that drops the MTKView
    // hop does not silently bring the layer back to BGRA8Unorm.
    metal_layer.setPixelFormat(MTLPixelFormat::RGBA16Float);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(width: u32, height: u32, geometry_less: bool) -> WindowConfig<'static> {
        WindowConfig {
            title: "",
            width,
            height,
            title_bar: true,
            geometry_less,
            capture_enabled: false,
            embedded: None,
        }
    }

    #[test]
    fn live_drawable_sets_the_target_size() {
        assert_eq!(
            initial_target_size((2048.0, 1536.0), &config(1024, 768, false)),
            (2048, 1536)
        );
    }

    #[test]
    fn missing_drawable_falls_back_to_the_requested_size() {
        assert_eq!(
            initial_target_size((0.0, 0.0), &config(1024, 768, false)),
            (1024, 768)
        );
    }

    #[test]
    fn requested_size_is_at_least_one_pixel() {
        assert_eq!(
            initial_target_size((0.0, 0.0), &config(0, 0, false)),
            (1, 1)
        );
    }

    #[test]
    fn geometry_less_world_clamps_to_one_pixel() {
        assert_eq!(
            initial_target_size((2048.0, 1536.0), &config(1024, 768, true)),
            (1, 1)
        );
    }
}
