// Native AppKit windowing, input, and display-mode switching, shared by every
// backend that renders into an NSView on macOS: Metal always, and Vulkan (via
// vulkan/appkit_window.rs) instead of GLFW, so the two backends get one
// window/input implementation with identical behavior. GLFW remains the
// windowing layer on Linux only.
//
// `AppKitWindow` works through `NSView` and never names the concrete view
// subclass, so Metal keeps its `MTKView` (it needs the drawable) and Vulkan
// creates a plain `CAMetalLayer`-backed view for `VK_EXT_metal_surface`.
// Counterpart of the `win32` module on Windows.

pub(crate) mod chrome;
pub(crate) mod display_mode;
pub(crate) mod input;
pub(crate) mod window;
pub(crate) mod window_delegate;

pub(crate) use window::{AppKitWindow, AppKitWindowParts};
