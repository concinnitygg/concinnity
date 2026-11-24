//! Loading `sl.interposer.dll`, the Streamline core every NVIDIA feature runs
//! through. It is found beside the executable or in the runtime library
//! directory, loaded only once its Authenticode signature verifies and names
//! NVIDIA as the signer, and never unloaded, so the entry points resolved from
//! it stay valid for the life of the process. Its plugins load from the same
//! directory.

use std::ffi::{CStr, c_void};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use windows::Win32::Foundation::{HMODULE, HWND};
use windows::Win32::Security::Cryptography::{CERT_NAME_SIMPLE_DISPLAY_TYPE, CertGetNameStringW};
use windows::Win32::Security::WinTrust::{
    WINTRUST_ACTION_GENERIC_VERIFY_V2, WINTRUST_DATA, WINTRUST_DATA_0, WINTRUST_FILE_INFO,
    WTD_CHOICE_FILE, WTD_REVOKE_NONE, WTD_STATEACTION_CLOSE, WTD_STATEACTION_VERIFY, WTD_UI_NONE,
    WTHelperGetProvSignerFromChain, WTHelperProvDataFromStateData, WinVerifyTrust,
};
use windows::Win32::System::LibraryLoader::{
    GetProcAddress, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW,
};
use windows::core::{PCSTR, PCWSTR};

use super::abi::{
    PfnSlEvaluateFeature, PfnSlFreeResources, PfnSlGetFeatureFunction, PfnSlGetFeatureVersion,
    PfnSlGetNewFrameToken, PfnSlInit, PfnSlIsFeatureSupported, PfnSlSetConstants,
    PfnSlSetD3DDevice, PfnSlShutdown, PfnSlUpgradeInterface,
};
use super::wide;
use crate::upscale_sdk::{SdkLibrary, entry_point};

/// The Streamline core's file name.
pub(crate) const LIBRARY: &str = "sl.interposer.dll";

/// The subject NVIDIA signs its Streamline binaries as.
const NVIDIA_SIGNER: &str = "NVIDIA Corporation";

/// The core API, resolved from the loaded interposer.
pub(crate) struct Interposer {
    pub(crate) init: PfnSlInit,
    pub(crate) shutdown: PfnSlShutdown,
    pub(crate) is_feature_supported: PfnSlIsFeatureSupported,
    pub(crate) evaluate_feature: PfnSlEvaluateFeature,
    pub(crate) free_resources: PfnSlFreeResources,
    pub(crate) get_feature_version: PfnSlGetFeatureVersion,
    pub(crate) upgrade_interface: PfnSlUpgradeInterface,
    pub(crate) set_constants: PfnSlSetConstants,
    pub(crate) get_feature_function: PfnSlGetFeatureFunction,
    pub(crate) get_new_frame_token: PfnSlGetNewFrameToken,
    pub(crate) set_d3d_device: PfnSlSetD3DDevice,
    /// The `vkGetInstanceProcAddr` export, which a Vulkan instance and device
    /// are loaded through.
    #[cfg(backend_vk)]
    pub(crate) vk_get_instance_proc_addr: *const c_void,
    /// The directory the interposer was loaded from, where its plugins are.
    pub(crate) directory: PathBuf,
}

// SAFETY: the table holds plain function pointers into a library that is never unloaded, and a
// path; none of it is tied to the thread that resolved it.
unsafe impl Send for Interposer {}
// SAFETY: as above; the table is never written after it is built.
unsafe impl Sync for Interposer {}

/// The interposer, loaded on first use from beside the executable or the
/// runtime library directory, or why it could not be: absent, unsigned, or
/// missing an entry point.
pub(crate) fn interposer() -> Result<&'static Interposer, &'static str> {
    static INTERPOSER: OnceLock<Result<Interposer, String>> = OnceLock::new();
    INTERPOSER
        .get_or_init(|| {
            let searched: Vec<PathBuf> = crate::runtime_libraries::search_dirs().collect();
            let directory = searched
                .iter()
                .find(|dir| dir.join(LIBRARY).is_file())
                .ok_or_else(|| {
                    let dirs: Vec<String> =
                        searched.iter().map(|d| d.display().to_string()).collect();
                    format!("{LIBRARY} not found in {}", dirs.join(" or "))
                })?;
            Interposer::load(directory.clone())
        })
        .as_ref()
        .map_err(String::as_str)
}

struct Module(HMODULE);

impl SdkLibrary for Module {
    fn symbol(&self, name: &CStr) -> Option<*const c_void> {
        // SAFETY: `self.0` is a loaded module and `name` a NUL-terminated string live for the call.
        unsafe { GetProcAddress(self.0, PCSTR(name.as_ptr().cast())) }.map(|f| f as *const c_void)
    }
}

impl Interposer {
    fn load(directory: PathBuf) -> Result<Self, String> {
        let path = directory.join(LIBRARY);
        verify_signature(&path)?;
        let wide = wide(&path);
        // Its own dependencies resolve from its directory and System32 only.
        let search = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32;
        // SAFETY: `wide` is a NUL-terminated absolute path live for the call.
        let module = unsafe { LoadLibraryExW(PCWSTR(wide.as_ptr()), None, search) }
            .map_err(|e| format!("loading {}: {e}", path.display()))?;
        let module = Module(module);
        // Each export as the prototype `sl_core_api.h` (or the Vulkan registry,
        // for the loader export) declares for its name.
        macro_rules! export {
            ($name:literal) => {
                // SAFETY: the field's type is the prototype declared for `$name`, and the module is
                // never unloaded.
                unsafe { entry_point(&module, $name) }.ok_or_else(|| {
                    format!("{LIBRARY} has no `{}` export", $name.to_string_lossy())
                })?
            };
        }
        Ok(Self {
            init: export!(c"slInit"),
            shutdown: export!(c"slShutdown"),
            is_feature_supported: export!(c"slIsFeatureSupported"),
            evaluate_feature: export!(c"slEvaluateFeature"),
            free_resources: export!(c"slFreeResources"),
            get_feature_version: export!(c"slGetFeatureVersion"),
            upgrade_interface: export!(c"slUpgradeInterface"),
            set_constants: export!(c"slSetConstants"),
            get_feature_function: export!(c"slGetFeatureFunction"),
            get_new_frame_token: export!(c"slGetNewFrameToken"),
            set_d3d_device: export!(c"slSetD3DDevice"),
            #[cfg(backend_vk)]
            vk_get_instance_proc_addr: export!(c"vkGetInstanceProcAddr"),
            directory,
        })
    }
}

// Whether `path` carries a valid Authenticode signature whose signer is NVIDIA,
// checked by `WinVerifyTrust` without UI or a revocation fetch, as Streamline's
// integration guide asks before the interposer is loaded.
fn verify_signature(path: &Path) -> Result<(), String> {
    let wide = wide(path);
    let mut file = WINTRUST_FILE_INFO {
        cbStruct: u32::try_from(size_of::<WINTRUST_FILE_INFO>()).unwrap_or(u32::MAX),
        pcwszFilePath: PCWSTR(wide.as_ptr()),
        ..Default::default()
    };
    let mut data = WINTRUST_DATA {
        cbStruct: u32::try_from(size_of::<WINTRUST_DATA>()).unwrap_or(u32::MAX),
        dwUIChoice: WTD_UI_NONE,
        fdwRevocationChecks: WTD_REVOKE_NONE,
        dwUnionChoice: WTD_CHOICE_FILE,
        Anonymous: WINTRUST_DATA_0 {
            pFile: &raw mut file,
        },
        dwStateAction: WTD_STATEACTION_VERIFY,
        ..Default::default()
    };
    let mut action = WINTRUST_ACTION_GENERIC_VERIFY_V2;
    // SAFETY: `data` and the file info and path it points at are live locals for the call.
    let status = unsafe {
        WinVerifyTrust(
            HWND::default(),
            &raw mut action,
            (&raw mut data).cast::<c_void>(),
        )
    };
    let signer = (status == 0).then(|| signer_name(&data)).flatten();
    data.dwStateAction = WTD_STATEACTION_CLOSE;
    // SAFETY: as above; closing releases the state the verify call opened.
    unsafe {
        WinVerifyTrust(
            HWND::default(),
            &raw mut action,
            (&raw mut data).cast::<c_void>(),
        )
    };
    if status != 0 {
        return Err(format!(
            "{} is not validly signed (WinVerifyTrust {status:#x})",
            path.display()
        ));
    }
    signed_by_nvidia(signer.as_deref()).map_err(|reason| format!("{}: {reason}", path.display()))
}

// Why `signer` is not NVIDIA, or nothing when it is.
fn signed_by_nvidia(signer: Option<&str>) -> Result<(), String> {
    match signer {
        Some(NVIDIA_SIGNER) => Ok(()),
        Some(other) => Err(format!("signed by {other}, not {NVIDIA_SIGNER}")),
        None => Err("its signer could not be read".to_string()),
    }
}

// The display name of the certificate that signed the file `data` verified,
// read from the state the verify left open.
fn signer_name(data: &WINTRUST_DATA) -> Option<String> {
    // SAFETY: the state handle is the one a successful verify opened and has not closed.
    let provider = unsafe { WTHelperProvDataFromStateData(data.hWVTStateData) };
    if provider.is_null() {
        return None;
    }
    // SAFETY: `provider` is the live provider data of that state.
    let signer = unsafe { WTHelperGetProvSignerFromChain(provider, 0, false, 0) };
    // SAFETY: a non-null signer from the provider data is live while the state is open.
    let signer = unsafe { signer.as_ref() }?;
    if signer.csCertChain == 0 || signer.pasCertChain.is_null() {
        return None;
    }
    // SAFETY: the chain holds `csCertChain` entries, checked non-empty above; the first is the
    // signing certificate.
    let leaf = unsafe { &*signer.pasCertChain };
    let mut name = [0u16; 256];
    // SAFETY: `leaf.pCert` is the certificate context the chain holds, and `name` a live buffer
    // of the length passed.
    let written = unsafe {
        CertGetNameStringW(
            leaf.pCert,
            CERT_NAME_SIMPLE_DISPLAY_TYPE,
            0,
            None,
            Some(&mut name),
        )
    } as usize;
    // The count includes the terminating NUL.
    let len = written.checked_sub(1)?;
    Some(String::from_utf16_lossy(name.get(..len)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_an_nvidia_signer_passes() {
        assert!(signed_by_nvidia(Some("NVIDIA Corporation")).is_ok());
        assert_eq!(
            signed_by_nvidia(Some("Contoso Ltd")).unwrap_err(),
            "signed by Contoso Ltd, not NVIDIA Corporation"
        );
        assert!(signed_by_nvidia(None).is_err());
    }
}
