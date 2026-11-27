// The platform window VkContext owns: the shared Win32 layer on Windows, the
// shared AppKit layer on macOS, GLFW on desktop Unix. Each answers
// `WindowControl` through `control` / `control_mut`.

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
