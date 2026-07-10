// Native Win32 windowing, input, and display-mode switching, shared by every
// backend that renders into an HWND on Windows: DirectX always, and Vulkan
// (via vulkan/win32_window.rs) instead of GLFW, so the two backends get one
// window/input implementation with identical behavior. GLFW remains the
// windowing layer on Linux only.

pub(crate) mod display_mode;
pub(crate) mod input;
pub(crate) mod window;
