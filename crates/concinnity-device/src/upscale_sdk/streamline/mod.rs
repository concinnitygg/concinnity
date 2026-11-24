//! NVIDIA Streamline, the runtime every NVIDIA RTX feature is reached
//! through. Its core, `sl.interposer.dll`, is loaded at runtime from beside the
//! executable after its signature verifies, so nothing of it is needed to
//! build. A backend initializes a session before creating its device and hands
//! that device over; the session then hooks the backend's presentation, which
//! Streamline needs every frame. Only a machine with an NVIDIA adapter gets a
//! session.

pub(crate) mod abi;
mod interposer;
mod nvidia;
mod session;

use std::os::windows::ffi::OsStrExt;
use std::path::Path;

pub(crate) use session::{Streamline, next_viewport};

/// Whether DLSS can be offered: the machine has an NVIDIA adapter and the
/// interposer beside the executable loads.
pub(crate) fn available() -> bool {
    nvidia::nvidia_adapter_present() && interposer::interposer().is_ok()
}

// `path` as a NUL-terminated UTF-16 string.
fn wide(path: &Path) -> Vec<u16> {
    path.as_os_str().encode_wide().chain([0]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_paths_are_nul_terminated() {
        let w = wide(Path::new("C:\\a"));
        assert_eq!(w.last(), Some(&0));
        assert_eq!(w.len(), "C:\\a".len() + 1);
    }
}
