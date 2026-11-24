//! Streamline on D3D12, integrated with manual hooking: the device stays
//! native and is handed to Streamline once created, and only the DXGI factory
//! the swapchain is created from is replaced with Streamline's proxy, so the
//! swapchain presents through it.

use std::ffi::c_void;

use windows::Win32::Graphics::Direct3D12::ID3D12Device;
use windows::Win32::Graphics::Dxgi::{IDXGIAdapter1, IDXGIFactory4};
use windows::core::Interface;

use crate::upscale_sdk::streamline::Streamline;
use crate::upscale_sdk::streamline::abi::{AdapterInfo, FEATURE_DLSS, RenderApi};

/// A Streamline session bound to this backend's device.
#[derive(Clone)]
pub(in crate::directx) struct DxStreamline {
    session: Streamline,
    adapter_luid: [u8; 8],
}

/// Initialize Streamline ahead of the device, `None` when it is unavailable.
pub(in crate::directx) fn init() -> Option<Streamline> {
    Streamline::init(RenderApi::D3D12)
}

impl DxStreamline {
    /// Bind `session` to the device just created on `adapter`. `None` (and the
    /// session shut down) when Streamline refuses the device or DLSS does not
    /// run on it, so a device that cannot use it presents without its hooks.
    pub(in crate::directx) fn bind(
        session: Streamline,
        device: &ID3D12Device,
        adapter: &IDXGIAdapter1,
    ) -> Option<Self> {
        // SAFETY: a property query on a live adapter.
        let luid = unsafe { adapter.GetDesc1() }.ok()?.AdapterLuid;
        let mut adapter_luid = [0u8; 8];
        adapter_luid[..4].copy_from_slice(&luid.LowPart.to_le_bytes());
        adapter_luid[4..].copy_from_slice(&luid.HighPart.to_le_bytes());
        // SAFETY: `device` is a live D3D12 device.
        if !unsafe { session.set_d3d_device(device.as_raw()) } {
            return None;
        }
        let bound = Self {
            session,
            adapter_luid,
        };
        if let Err(result) = bound
            .session
            .feature_supported(FEATURE_DLSS, &bound.adapter())
        {
            tracing::info!(
                "Streamline released: DLSS does not run on this adapter ({})",
                result.name()
            );
            return None;
        }
        Some(bound)
    }

    /// Streamline's proxy for `factory`, through which a created swapchain
    /// presents via Streamline. `None` leaves the native factory in use.
    pub(in crate::directx) fn proxy_factory(
        &self,
        factory: &IDXGIFactory4,
    ) -> Option<IDXGIFactory4> {
        let mut raw: *mut c_void = factory.as_raw();
        // SAFETY: `raw` is the live factory `factory` holds a reference to, which the proxy takes
        // its own reference on.
        if !unsafe { self.session.upgrade_interface(&mut raw) } || raw == factory.as_raw() {
            return None;
        }
        // SAFETY: the upgrade replaced `raw` with a new proxy whose single reference the caller
        // owns, and the proxy implements every `IDXGIFactory` version up to 7.
        Some(unsafe { IDXGIFactory4::from_raw(raw) })
    }

    pub(in crate::directx) fn session(&self) -> &Streamline {
        &self.session
    }

    /// The adapter the device was created on, as Streamline names it.
    pub(in crate::directx) fn adapter(&self) -> AdapterInfo {
        AdapterInfo::luid(&self.adapter_luid)
    }
}
