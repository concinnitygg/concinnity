//! Whether the machine has an NVIDIA adapter. Streamline's features run only
//! on NVIDIA hardware, so the session, and the presentation hooks that come
//! with it, are set up only where one is present.

use std::sync::OnceLock;

use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, IDXGIFactory1,
};

const NVIDIA_VENDOR_ID: u32 = 0x10de;

/// Whether any hardware adapter is NVIDIA's, enumerated once per process.
pub(crate) fn nvidia_adapter_present() -> bool {
    static PRESENT: OnceLock<bool> = OnceLock::new();
    *PRESENT.get_or_init(enumerate)
}

fn enumerate() -> bool {
    // SAFETY: the call takes no arguments, and the new COM object lands in a binding that owns
    // it.
    let Ok(factory) = (unsafe { CreateDXGIFactory1::<IDXGIFactory1>() }) else {
        return false;
    };
    (0..)
        // SAFETY: an enumeration query on a live factory; an index past the last adapter is an
        // error that ends the walk.
        .map_while(|index| unsafe { factory.EnumAdapters1(index) }.ok())
        // SAFETY: a property query on a live adapter.
        .filter_map(|adapter| unsafe { adapter.GetDesc1() }.ok())
        .any(|desc| is_nvidia_hardware(desc.VendorId, desc.Flags))
}

// Whether an adapter with these `DXGI_ADAPTER_DESC1` fields is NVIDIA hardware.
fn is_nvidia_hardware(vendor_id: u32, flags: u32) -> bool {
    vendor_id == NVIDIA_VENDOR_ID && flags & DXGI_ADAPTER_FLAG_SOFTWARE.0.cast_unsigned() == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_nvidia_hardware_counts() {
        let software = DXGI_ADAPTER_FLAG_SOFTWARE.0.cast_unsigned();
        assert!(is_nvidia_hardware(0x10de, 0));
        assert!(!is_nvidia_hardware(0x10de, software));
        assert!(!is_nvidia_hardware(0x1002, 0));
        assert!(!is_nvidia_hardware(0x8086, 0));
    }
}
