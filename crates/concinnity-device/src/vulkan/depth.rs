// The core depth conventions in Vulkan terms: compare ops for pipeline depth
// state and clear values for render-pass begin.

use ash::vk;
use concinnity_core::render::depth::{DepthCompare, DepthConvention};

pub(in crate::vulkan) const fn compare_op(compare: DepthCompare) -> vk::CompareOp {
    match compare {
        DepthCompare::Less => vk::CompareOp::LESS,
        DepthCompare::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
        DepthCompare::Greater => vk::CompareOp::GREATER,
        DepthCompare::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
    }
}

// The depth attachment's clear for a target drawn under `convention`.
pub(in crate::vulkan) const fn clear_value(convention: DepthConvention) -> vk::ClearValue {
    vk::ClearValue {
        depth_stencil: vk::ClearDepthStencilValue {
            depth: convention.clear(),
            stencil: 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_compare_maps_to_its_vulkan_op() {
        assert_eq!(compare_op(DepthCompare::Less), vk::CompareOp::LESS);
        assert_eq!(
            compare_op(DepthCompare::LessEqual),
            vk::CompareOp::LESS_OR_EQUAL
        );
        assert_eq!(compare_op(DepthCompare::Greater), vk::CompareOp::GREATER);
        assert_eq!(
            compare_op(DepthCompare::GreaterEqual),
            vk::CompareOp::GREATER_OR_EQUAL
        );
    }
}
