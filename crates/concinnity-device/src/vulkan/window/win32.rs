//! The Vulkan backend's Windows window: the shared native Win32 window
//! (`crate::win32`) the DirectX backend also presents to, with the surface
//! created directly through `VK_KHR_win32_surface`. GLFW (window/glfw.rs)
//! remains the windowing layer on Linux only.

use ash::vk;
use concinnity_engine::components::WindowMode;
use concinnity_engine::render::backend::WindowControl;
use concinnity_engine::render::backend::error::{RenderError, RenderResult};

pub(crate) use crate::win32::platform_window::Win32Window;

impl Win32Window {
    // Create the window windowed, then apply a non-default creation mode.
    pub(crate) fn new(
        title: &str,
        width: u32,
        height: u32,
        mode: WindowMode,
        _resizable: bool,
        title_bar: bool,
    ) -> RenderResult<Self> {
        let mut this = Self::create(title, width, height, title_bar).map_err(RenderError::Other)?;
        if !matches!(mode, WindowMode::Windowed) {
            this.set_window_mode(mode);
        }
        Ok(this)
    }

    // The window as the engine drives it.
    pub(crate) fn control(&self) -> &dyn WindowControl {
        self
    }

    pub(crate) fn control_mut(&mut self) -> &mut dyn WindowControl {
        self
    }

    // Create the presentation surface for this window via
    // VK_KHR_win32_surface.
    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "same receiver as the GLFW and AppKit windows the shared Vulkan code calls"
    )]
    pub(crate) fn create_surface(
        &mut self,
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> RenderResult<vk::SurfaceKHR> {
        // SAFETY: passing None asks for the handle of the current process image, which is always
        // valid.
        let hinstance = unsafe { windows::Win32::System::LibraryLoader::GetModuleHandleW(None) }
            .map_err(|e| RenderError::Other(format!("GetModuleHandleW: {e}")))?;
        let info = vk::Win32SurfaceCreateInfoKHR::default()
            .hinstance(hinstance.0 as isize)
            .hwnd(self.state().hwnd.0 as isize);
        let loader = ash::khr::win32_surface::Instance::new(entry, instance);
        // SAFETY: `info` borrows the module handle and HWND for the call; both name live Win32
        // objects owned by this window.
        unsafe { loader.create_win32_surface(&info, None) }
            .map_err(|e| crate::vulkan::error::map_vk_result(e, "vkCreateWin32SurfaceKHR"))
    }

    // Vulkan instance extensions required for surface creation on Windows.
    #[expect(
        clippy::unused_self,
        reason = "every window backend answers through its window"
    )]
    pub(crate) fn required_instance_extensions(&self) -> Vec<String> {
        [ash::khr::surface::NAME, ash::khr::win32_surface::NAME]
            .into_iter()
            .map(|n| n.to_str().unwrap_or_default().to_string())
            .collect()
    }
}
