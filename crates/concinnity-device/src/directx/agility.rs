// The Agility SDK: Microsoft's newer D3D12 runtime, which FSR on DirectX
// needs. Where a `D3D12/` directory holding `D3D12Core.dll` sits beside the
// executable or in the runtime library directory, the process's D3D12 is
// switched to it through a device factory before any device exists; elsewhere,
// or on a system whose `d3d12.dll` does not expose `ID3D12SDKConfiguration1`
// (which the device factory needs), the OS runtime stays.

use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use windows::Win32::Foundation::E_NOINTERFACE;
use windows::Win32::Graphics::Direct3D12::{
    CLSID_D3D12SDKConfiguration, D3D12GetInterface, ID3D12DeviceFactory, ID3D12SDKConfiguration1,
};
use windows::core::PCSTR;

// The `D3D12SDKVersion` of the vendored Agility package, the middle number of
// its release.
const SDK_VERSION: u32 = 619;

// The directory the runtime sits in, beside the executable.
const SDK_DIR: &str = "D3D12";

// What switching to the Agility runtime came to.
enum Activation {
    // No runtime is vendored, so the OS runtime is the one there is.
    Absent,
    Active(PathBuf),
    Refused { dir: PathBuf, reason: String },
}

fn activation() -> &'static Activation {
    static ACTIVATION: OnceLock<Activation> = OnceLock::new();
    ACTIVATION.get_or_init(|| {
        let Some(dir) = crate::runtime_libraries::search_dirs()
            .map(|dir| dir.join(SDK_DIR))
            .find(|dir| dir.join("D3D12Core.dll").is_file())
        else {
            return Activation::Absent;
        };
        match apply(&dir) {
            Ok(()) => Activation::Active(dir),
            Err(e) => Activation::Refused {
                dir,
                reason: e.to_string(),
            },
        }
    })
}

// The Agility runtime the process's D3D12 runs on, switching to it on the first
// call; `None` when the OS runtime stays. The first call has to come before any
// device is created, which is why both the device probe and the backend make
// it first. It can come before logging starts, so the outcome is logged by
// `log_outcome` instead.
pub(in crate::directx) fn activate() -> Option<&'static Path> {
    match activation() {
        Activation::Active(dir) => Some(dir),
        Activation::Absent | Activation::Refused { .. } => None,
    }
}

// Log which D3D12 runtime the process runs on.
pub(in crate::directx) fn log_outcome() {
    match activation() {
        Activation::Absent => {}
        Activation::Active(dir) => {
            tracing::info!("d3d12: Agility SDK runtime from {}", dir.display());
        }
        Activation::Refused { dir, reason } => tracing::warn!(
            "d3d12: the Agility SDK in {} could not be used ({reason}); running on the OS D3D12 \
             runtime, without FSR",
            dir.display()
        ),
    }
}

// Make the runtime in `dir` the process's D3D12.
fn apply(dir: &Path) -> windows::core::Result<()> {
    let path = sdk_path(dir).ok_or_else(|| windows::core::Error::from(E_NOINTERFACE))?;
    let clsid = CLSID_D3D12SDKConfiguration;
    let mut config: Option<ID3D12SDKConfiguration1> = None;
    // SAFETY: `clsid` and `config` are live locals; the call fills `config` with the interface the
    // CLSID names.
    unsafe { D3D12GetInterface(&raw const clsid, &raw mut config) }?;
    let config = config.ok_or_else(|| windows::core::Error::from(E_NOINTERFACE))?;
    // SAFETY: `path` is a NUL-terminated string live for the call.
    let factory: ID3D12DeviceFactory =
        unsafe { config.CreateDeviceFactory(SDK_VERSION, PCSTR(path.as_ptr().cast())) }?;
    // SAFETY: a call on a live factory; no device exists yet, which is when it may switch.
    unsafe { factory.ApplyToGlobalState() }
}

// `dir` as the NUL-terminated path, with a trailing separator, a device factory
// takes; `None` for a path that is not UTF-8 or holds a NUL.
fn sdk_path(dir: &Path) -> Option<CString> {
    let mut path = dir.to_str()?.to_string();
    if !path.ends_with('\\') {
        path.push('\\');
    }
    CString::new(path).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sdk_version_is_the_vendored_release() {
        let release = concinnity_vendor::sdks::AGILITY.release;
        assert_eq!(
            release.split('.').nth(1),
            Some(SDK_VERSION.to_string().as_str()),
            "{release}"
        );
    }

    #[test]
    fn the_sdk_path_ends_in_a_separator() {
        let path = sdk_path(Path::new("C:\\app\\D3D12")).unwrap();
        assert_eq!(path.to_str().unwrap(), "C:\\app\\D3D12\\");
        let path = sdk_path(Path::new("C:\\app\\D3D12\\")).unwrap();
        assert_eq!(path.to_str().unwrap(), "C:\\app\\D3D12\\");
    }
}
