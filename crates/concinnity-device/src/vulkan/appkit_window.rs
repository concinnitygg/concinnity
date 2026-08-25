// src/vulkan/appkit_window.rs
//
// The Vulkan backend's macOS window: a thin adapter over the shared native
// AppKit layer (crate::appkit) the Metal backend also uses, so the two
// NSView-rendering backends share one window/input/display-mode implementation
// with identical behavior (event pump, cursor capture/confinement, window modes,
// Resolution-row mode switching). GLFW (window.rs) remains the windowing layer
// on Linux only.
//
// Metal renders through an `MTKView`, which owns its `CAMetalLayer` and sizes
// the drawable itself. Vulkan has no MTKView: it hosts a bare `CAMetalLayer` on
// a plain `NSView` and hands that layer to `vkCreateMetalSurfaceEXT`. The layer
// is therefore ours to size, which `framebuffer_size` does from the view bounds
// and the window's backing scale each time the swapchain asks.

use ash::vk;
use objc2::MainThreadOnly;
use objc2::rc::Retained;
use objc2_app_kit::{NSApplication, NSView};
use objc2_foundation::NSSize;
use objc2_quartz_core::CAMetalLayer;

use crate::appkit::{AppKitWindow, AppKitWindowParts, chrome, window_delegate};
use crate::components::WindowMode;
use crate::gfx::display_mode::DisplayMode;
use crate::gfx::input::RenderInput;
use crate::gfx::keymap::KeyMap;

pub(crate) struct AppKitVkWindow {
    win: AppKitWindow,
    // The presentation layer `vkCreateMetalSurfaceEXT` was handed. Retained so
    // it outlives the surface, and re-sized by `framebuffer_size`.
    layer: Retained<CAMetalLayer>,
}

// SAFETY: Only ever used on the thread that created it, like the GLFW and Win32 windows.
unsafe impl Send for AppKitVkWindow {}

impl AppKitVkWindow {
    pub(crate) fn new(
        title: &str,
        width: u32,
        height: u32,
        mode: &WindowMode,
        _resizable: bool,
        title_bar: bool,
    ) -> Result<Self, String> {
        let mtm = objc2::MainThreadMarker::new()
            .ok_or_else(|| "the Vulkan window must be created on the main thread".to_string())?;
        let window = chrome::create_window(mtm, title, width, height, title_bar)?;
        let content_rect = window.contentRectForFrameRect(window.frame());

        // A layer-hosting NSView: assigning the layer before `wantsLayer` makes
        // AppKit adopt ours instead of creating its own backing layer.
        let view = NSView::initWithFrame(NSView::alloc(mtm), content_rect);
        let layer = CAMetalLayer::new();
        // MoltenVK assigns the layer's `device` from the VkPhysicalDevice at
        // surface creation, so no MTLDevice is set here.
        view.setLayer(Some(&layer));
        view.setWantsLayer(true);
        window.setContentView(Some(&view));
        let (delegate, fullscreen) = window_delegate::attach_fullscreen_delegate(mtm, &window);
        NSApplication::sharedApplication(mtm).activate();
        window.makeKeyAndOrderFront(None);

        let mut this = Self {
            win: AppKitWindow::new(AppKitWindowParts {
                window: Some(window),
                view,
                title_bar,
                // The Vulkan path is always the windowed CLI path; the embedded
                // host-owned-view mode is Metal only.
                pump_events: true,
                fullscreen,
                window_delegate: Some(delegate),
            }),
            layer,
        };
        this.sync_drawable_size();
        this.set_window_mode(*mode);
        Ok(this)
    }

    // Push the view's current pixel size onto the layer. AppKit sizes the layer's
    // frame with the view but never its `drawableSize`, which is what MoltenVK
    // reports through `vkGetPhysicalDeviceSurfaceCapabilitiesKHR`; without this
    // the swapchain would stay at the creation size across a resize or a move to
    // a display with a different backing scale.
    fn sync_drawable_size(&self) -> (i32, i32) {
        let view = self.win.view();
        let bounds = view.bounds().size;
        // The window's backing scale, or the layer's current one before the view
        // is attached (during init the window is not on a screen yet).
        let scale = match self.win.window().map(|w| w.backingScaleFactor()) {
            Some(s) if s > 0.0 => s,
            _ => self.layer.contentsScale(),
        };
        let (w, h) = (bounds.width * scale, bounds.height * scale);
        self.layer.setContentsScale(scale);
        self.layer
            .setDrawableSize(NSSize::new(w.max(1.0), h.max(1.0)));
        (w as i32, h as i32)
    }

    // Drain pending AppKit events; true when the window should close.
    pub(crate) fn poll(&mut self) -> bool {
        if let Some(mtm) = objc2::MainThreadMarker::new() {
            self.win.pump_ns_events(mtm);
        }
        // Hold the chosen display mode while fullscreen and restore the desktop
        // mode on leaving it, mirroring the Metal backend's per-frame reconcile.
        self.win.reconcile_display_mode();
        self.win.closed()
    }

    pub(crate) fn take_input(&mut self) -> RenderInput {
        self.win.take_input()
    }

    pub(crate) fn capture_cursor(&mut self) {
        self.win.capture_cursor();
    }

    pub(crate) fn set_ui_cursor_hidden(&mut self, hidden: bool) {
        self.win.set_ui_cursor_hidden(hidden);
    }

    pub(crate) fn set_menu_mode(&mut self, on: bool) {
        self.win.set_menu_mode(on);
    }

    pub(crate) fn set_camera_capture(&mut self, capture: bool) {
        self.win.set_camera_capture(capture);
    }

    pub(crate) fn cursor_outside_window(&self) -> bool {
        self.win.cursor_outside_window()
    }

    pub(crate) fn set_keymap(&mut self, keymap: &KeyMap) {
        self.win.set_keymap(keymap);
    }

    pub(crate) fn set_window_mode(&mut self, mode: WindowMode) {
        self.win.set_window_mode(mode);
    }

    pub(crate) fn set_window_size(&mut self, width: u32, height: u32) {
        self.win.set_window_size(width, height);
    }

    pub(crate) fn display_modes(&self) -> Vec<DisplayMode> {
        self.win.display_modes()
    }

    pub(crate) fn current_display_mode(&self) -> Option<DisplayMode> {
        self.win.current_display_mode()
    }

    pub(crate) fn set_display_mode(&mut self, mode: DisplayMode) {
        self.win.set_display_mode(mode);
    }

    // The framebuffer size in pixels, the extent the swapchain is sized to.
    // Also the point at which the layer's drawable size is refreshed, so a
    // resize reaches MoltenVK's reported surface capabilities before the
    // swapchain is rebuilt from them.
    pub(crate) fn framebuffer_size(&self) -> (i32, i32) {
        self.sync_drawable_size()
    }

    // The overlay coordinate space: the view's size in points, larger than
    // `framebuffer_size` by the backing scale on a retina display. Comes from
    // the shared AppKit layer, which reports the cursor in the same units.
    pub(crate) fn logical_size(&self) -> (f32, f32) {
        self.win.logical_size()
    }

    // Create the presentation surface from the hosted CAMetalLayer.
    // `_entry` keeps the signature shared with the GLFW / Win32 windows.
    pub(crate) fn create_surface(
        &mut self,
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> Result<vk::SurfaceKHR, String> {
        let info =
            vk::MetalSurfaceCreateInfoEXT::default().layer(Retained::as_ptr(&self.layer).cast());
        let loader = ash::ext::metal_surface::Instance::new(entry, instance);
        // SAFETY: the create-info and every slice it borrows are live for the call, and each handle
        // it names belongs to this device.
        unsafe { loader.create_metal_surface(&info, None) }
            .map_err(|e| format!("vkCreateMetalSurfaceEXT: {e}"))
    }

    // Vulkan instance extensions required for surface creation on macOS.
    pub(crate) fn required_instance_extensions(&self) -> Vec<String> {
        [ash::khr::surface::NAME, ash::ext::metal_surface::NAME]
            .into_iter()
            .map(|n| n.to_str().unwrap_or_default().to_string())
            .collect()
    }
}
