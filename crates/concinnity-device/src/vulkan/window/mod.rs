// The platform window VkContext owns: the shared native Win32 layer on Windows
// and the shared native AppKit layer on macOS (one window/input implementation
// with the DirectX and Metal backends respectively), GLFW on the desktop Unix
// tier. The gate matches the manifest's, so a target with no window layer
// fails naming this alias rather than naming a missing crate.

#[cfg(target_os = "macos")]
mod appkit;
#[cfg(all(unix, not(target_vendor = "apple"), not(target_os = "android")))]
mod glfw;
#[cfg(all(unix, not(target_vendor = "apple"), not(target_os = "android")))]
mod glfw_clipboard;
#[cfg(target_os = "windows")]
mod win32;

#[cfg(target_os = "macos")]
pub(super) use self::appkit::AppKitVkWindow as PlatformWindow;
#[cfg(all(unix, not(target_vendor = "apple"), not(target_os = "android")))]
pub(super) use self::glfw::GlfwWindow as PlatformWindow;
#[cfg(target_os = "windows")]
pub(super) use self::win32::Win32Window as PlatformWindow;
