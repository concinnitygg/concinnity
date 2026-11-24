//! Streamline on Vulkan. The instance and device are loaded through the
//! interposer's `vkGetInstanceProcAddr`, which forwards to the system loader
//! except for the calls Streamline hooks: instance and device creation, where
//! it adds what its features need, and the swapchain and present, which it
//! needs every frame. Streamline exists only on Windows.

use concinnity_engine::render::backend::error::RenderResult;

/// The live session the device was created under.
#[cfg(windows)]
pub(in crate::vulkan) type Session = crate::upscale_sdk::streamline::Streamline;

/// No session can exist off Windows.
#[cfg(not(windows))]
#[derive(Clone, Debug)]
pub(in crate::vulkan) enum Session {}

/// Initialize Streamline ahead of the instance, `None` when it is
/// unavailable.
pub(in crate::vulkan) fn init() -> Option<Session> {
    #[cfg(windows)]
    {
        use crate::upscale_sdk::streamline::abi::RenderApi;
        Session::init(RenderApi::Vulkan)
    }
    #[cfg(not(windows))]
    {
        None
    }
}

/// The Vulkan loader to create the instance through: the interposer's when a
/// session is live, else the system's.
pub(in crate::vulkan) fn load_entry(session: Option<&Session>) -> RenderResult<ash::Entry> {
    match session {
        #[cfg(windows)]
        Some(session) => Ok(entry_through(session)),
        #[cfg(not(windows))]
        Some(never) => match *never {},
        None => super::loader::load_entry(),
    }
}

#[cfg(windows)]
fn entry_through(session: &Session) -> ash::Entry {
    let address = session.vk_get_instance_proc_addr();
    // SAFETY: the interposer exports `vkGetInstanceProcAddr` with the registry's prototype, and
    // the library is never unloaded.
    let get_instance_proc_addr = unsafe {
        std::mem::transmute::<*const std::ffi::c_void, ash::vk::PFN_vkGetInstanceProcAddr>(address)
    };
    // SAFETY: the function is the loader entry point for every other Vulkan call, as a system
    // loader's export is.
    unsafe {
        ash::Entry::from_static_fn(ash::StaticFn {
            get_instance_proc_addr,
        })
    }
}
