// The core depth convention in D3D12 terms.

use concinnity_core::render::depth::{DEPTH_CLEAR, DEPTH_INCLUSIVE_COMPARE};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D32_FLOAT;

use crate::directx::pso::CompareOp;

// The optimized clear a `D32_FLOAT` depth resource is created with: the value
// its passes clear it to, so the clear takes the fast path.
pub(in crate::directx) fn optimized_clear() -> D3D12_CLEAR_VALUE {
    D3D12_CLEAR_VALUE {
        Format: DXGI_FORMAT_D32_FLOAT,
        Anonymous: D3D12_CLEAR_VALUE_0 {
            DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
                Depth: DEPTH_CLEAR,
                Stencil: 0,
            },
        },
    }
}

// The shadow compare samplers' test: lit where the reference depth is no
// farther from the light than the stored caster.
pub(in crate::directx) fn shadow_sample_compare() -> D3D12_COMPARISON_FUNC {
    CompareOp::depth(DEPTH_INCLUSIVE_COMPARE).raw()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reversed depth: a sample is lit where its reference is at or nearer the
    // light than the stored caster.
    #[test]
    fn the_shadow_sampler_passes_at_or_nearer_the_light() {
        assert_eq!(shadow_sample_compare(), D3D12_COMPARISON_FUNC_GREATER_EQUAL);
    }
}
