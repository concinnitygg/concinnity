// The core depth conventions in D3D12 terms.

use concinnity_core::render::depth::DepthConvention;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D32_FLOAT;

// The optimized clear a `D32_FLOAT` depth resource is created with: the value
// its passes clear it to under `convention`, so the clear takes the fast path.
pub(in crate::directx) fn optimized_clear(convention: DepthConvention) -> D3D12_CLEAR_VALUE {
    D3D12_CLEAR_VALUE {
        Format: DXGI_FORMAT_D32_FLOAT,
        Anonymous: D3D12_CLEAR_VALUE_0 {
            DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
                Depth: convention.clear(),
                Stencil: 0,
            },
        },
    }
}
