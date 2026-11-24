//! One Streamline session: `slInit` for a graphics API, bound to the device a
//! backend creates next, and `slShutdown` once the last handle to it drops.
//! Streamline binds its plugins to a single device per initialization, so at
//! most one session is live in the process.

use std::ffi::{CStr, c_char, c_void};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use super::abi::{
    AdapterInfo, FEATURE_DLSS, Feature, FeatureVersion, LOG_TYPE_ERROR, LOG_TYPE_WARN,
    PREFERENCE_DISABLE_CL_STATE_TRACKING, PREFERENCE_USE_MANUAL_HOOKING, Preferences, RenderApi,
    SDK_VERSION, SlResult,
};
use super::interposer::{Interposer, interposer};
use super::nvidia::nvidia_adapter_present;

/// The engine's identity for Streamline and NGX. A GUID-like project id
/// stands in for an NVIDIA-assigned application id.
const PROJECT_ID: &CStr = c"5f2e1a64-9c3b-4d7e-8a1f-2b6c0d9e7f30";
// NUL-terminated for the C side.
const ENGINE_VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");

/// The features every session loads.
const FEATURES: [Feature; 1] = [FEATURE_DLSS];

// Whether a session is live in the process.
static LIVE: AtomicBool = AtomicBool::new(false);

struct Session {
    api: &'static Interposer,
}

impl Drop for Session {
    fn drop(&mut self) {
        // SAFETY: the session initialized Streamline and is the only one live, so this is the
        // matching shutdown; every handle into it has dropped.
        let result = unsafe { (self.api.shutdown)() };
        if !result.is_ok() {
            tracing::warn!("Streamline: slShutdown returned {}", result.name());
        }
        LIVE.store(false, Ordering::Release);
    }
}

/// A shared handle to the live Streamline session.
#[derive(Clone)]
pub(crate) struct Streamline(Arc<Session>);

impl std::fmt::Debug for Streamline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Streamline")
    }
}

impl Streamline {
    /// Initialize Streamline for `render_api`, before the device it will bind
    /// to is created. `None` (logged) when the machine has no NVIDIA adapter,
    /// the interposer is unavailable, another session is live, or `slInit`
    /// fails; the backend then runs without it.
    pub(crate) fn init(render_api: RenderApi) -> Option<Self> {
        if !nvidia_adapter_present() {
            tracing::info!("Streamline not loaded: no NVIDIA adapter");
            return None;
        }
        let api = interposer()
            .inspect_err(|reason| tracing::info!("Streamline unavailable: {reason}"))
            .ok()?;
        if LIVE.swap(true, Ordering::AcqRel) {
            tracing::warn!("Streamline: a session is already live; this device runs without it");
            return None;
        }
        let directory: Vec<u16> = super::wide(&api.directory);
        let plugin_paths = [directory.as_ptr()];
        let mut prefs = Preferences::new(render_api);
        prefs.paths_to_plugins = plugin_paths.as_ptr();
        prefs.num_paths_to_plugins = 1;
        prefs.log_message_callback = Some(log_message);
        prefs.flags = preference_flags(render_api);
        prefs.features_to_load = FEATURES.as_ptr();
        prefs.num_features_to_load = 1;
        prefs.engine_version = ENGINE_VERSION.as_ptr().cast();
        prefs.project_id = PROJECT_ID.as_ptr();
        // SAFETY: `prefs` and every buffer it points at are live for the call, which copies them.
        let result = unsafe { (api.init)(&raw const prefs, SDK_VERSION) };
        if !result.is_ok() {
            LIVE.store(false, Ordering::Release);
            tracing::warn!("Streamline: slInit returned {}", result.name());
            return None;
        }
        tracing::info!(
            "Streamline: initialized for {render_api:?} from {}",
            api.directory.display()
        );
        Some(Self(Arc::new(Session { api })))
    }

    pub(crate) fn api(&self) -> &'static Interposer {
        self.0.api
    }

    /// Bind the session to the D3D12 device just created.
    ///
    /// # Safety
    ///
    /// `device` must be a live `ID3D12Device`.
    pub(crate) unsafe fn set_d3d_device(&self, device: *mut c_void) -> bool {
        // SAFETY: the caller guarantees the device.
        let result = unsafe { (self.api().set_d3d_device)(device) };
        logged(result, "slSetD3DDevice")
    }

    /// Replace the DXGI or D3D interface `*interface` with Streamline's proxy
    /// for it. The proxy holds its own reference to the original, and the
    /// caller owns the one reference to the proxy.
    ///
    /// # Safety
    ///
    /// `*interface` must be a live COM interface the caller holds a reference
    /// to, which stays the caller's to release.
    pub(crate) unsafe fn upgrade_interface(&self, interface: &mut *mut c_void) -> bool {
        // SAFETY: the caller guarantees the interface.
        let result = unsafe { (self.api().upgrade_interface)(interface) };
        logged(result, "slUpgradeInterface")
    }

    /// Whether `feature` runs on `adapter`, or why not.
    pub(crate) fn feature_supported(
        &self,
        feature: Feature,
        adapter: &AdapterInfo,
    ) -> Result<(), SlResult> {
        // SAFETY: `adapter` and whatever it points at are live for the call.
        let result = unsafe { (self.api().is_feature_supported)(feature, adapter) };
        if result.is_ok() { Ok(()) } else { Err(result) }
    }

    /// The versions of `feature`'s plugin and of the NGX library it loaded.
    pub(crate) fn feature_version(&self, feature: Feature) -> Option<FeatureVersion> {
        let mut version = FeatureVersion::default();
        // SAFETY: `version` is a live local the call fills.
        let result = unsafe { (self.api().get_feature_version)(feature, &raw mut version) };
        logged(result, "slGetFeatureVersion").then_some(version)
    }

    /// The feature function `name` as the pointer type `F`.
    ///
    /// # Safety
    ///
    /// `F` must be the function pointer type the feature's header declares for
    /// `name`; it stays valid while the session is live.
    pub(crate) unsafe fn feature_function<F: Copy>(
        &self,
        feature: Feature,
        name: &CStr,
    ) -> Option<F> {
        const {
            assert!(size_of::<F>() == size_of::<*mut c_void>());
        }
        let mut function: *mut c_void = std::ptr::null_mut();
        // SAFETY: `name` is NUL-terminated and `function` a live local the call fills.
        let result =
            unsafe { (self.api().get_feature_function)(feature, name.as_ptr(), &raw mut function) };
        if !logged(result, "slGetFeatureFunction") || function.is_null() {
            return None;
        }
        // SAFETY: `F` is pointer-sized (checked above) and, per the caller's contract, the type of
        // the non-null function returned.
        Some(unsafe { std::mem::transmute_copy::<*mut c_void, F>(&function) })
    }

    /// The interposer's `vkGetInstanceProcAddr`, which an instance and device
    /// are loaded through so Streamline sees their creation and presents.
    #[cfg(backend_vk)]
    pub(crate) fn vk_get_instance_proc_addr(&self) -> *const c_void {
        self.api().vk_get_instance_proc_addr
    }
}

/// A viewport id no other upscaler in the process uses, so a successor built
/// before its predecessor drops never shares (or frees) its resources.
pub(crate) fn next_viewport() -> u32 {
    static NEXT: AtomicU32 = AtomicU32::new(0);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

// D3D12 hooks only the interfaces the engine upgrades itself; Vulkan loads
// through the interposer, which hooks what it needs. Command-list state is
// the engine's to restore either way, and over-the-air updates are off so a
// build runs the libraries it shipped.
fn preference_flags(render_api: RenderApi) -> u64 {
    match render_api {
        RenderApi::D3D12 => PREFERENCE_DISABLE_CL_STATE_TRACKING | PREFERENCE_USE_MANUAL_HOOKING,
        RenderApi::Vulkan => PREFERENCE_DISABLE_CL_STATE_TRACKING,
    }
}

// Whether `result` is success, logging it otherwise.
fn logged(result: SlResult, call: &str) -> bool {
    if !result.is_ok() {
        tracing::warn!("Streamline: {call} returned {}", result.name());
    }
    result.is_ok()
}

// Streamline's log, forwarded at its own severity.
unsafe extern "C" fn log_message(ty: u32, msg: *const c_char) {
    if msg.is_null() {
        return;
    }
    // SAFETY: Streamline passes a NUL-terminated message live for the callback.
    let msg = unsafe { CStr::from_ptr(msg) }.to_string_lossy();
    let msg = msg.trim_end();
    match ty {
        LOG_TYPE_ERROR => tracing::error!("Streamline: {msg}"),
        LOG_TYPE_WARN => tracing::warn!("Streamline: {msg}"),
        _ => tracing::debug!("Streamline: {msg}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d3d12_hooks_manually_and_vulkan_through_the_loader() {
        let d3d12 = preference_flags(RenderApi::D3D12);
        let vulkan = preference_flags(RenderApi::Vulkan);
        assert_ne!(d3d12 & PREFERENCE_USE_MANUAL_HOOKING, 0);
        assert_eq!(vulkan & PREFERENCE_USE_MANUAL_HOOKING, 0);
        assert_ne!(d3d12 & PREFERENCE_DISABLE_CL_STATE_TRACKING, 0);
        assert_ne!(vulkan & PREFERENCE_DISABLE_CL_STATE_TRACKING, 0);
    }

    #[test]
    fn the_engine_version_is_the_crate_version_nul_terminated() {
        assert_eq!(
            ENGINE_VERSION.strip_suffix('\0'),
            Some(env!("CARGO_PKG_VERSION"))
        );
    }

    #[test]
    fn viewports_are_never_reused() {
        let a = next_viewport();
        let b = next_viewport();
        assert_ne!(a, b);
    }
}
