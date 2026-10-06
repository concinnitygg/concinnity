// The core depth convention in Vulkan terms: compare ops for pipeline depth
// state and the shadow sampler, and the clear value for render-pass begin.

use ash::vk;
use concinnity_core::render::depth::{DEPTH_CLEAR, DEPTH_INCLUSIVE_COMPARE, DepthCompare};

pub(in crate::vulkan) const fn compare_op(compare: DepthCompare) -> vk::CompareOp {
    match compare {
        DepthCompare::Less => vk::CompareOp::LESS,
        DepthCompare::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
        DepthCompare::Greater => vk::CompareOp::GREATER,
        DepthCompare::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
    }
}

// Every depth attachment's clear.
pub(in crate::vulkan) const CLEAR_VALUE: vk::ClearValue = vk::ClearValue {
    depth_stencil: vk::ClearDepthStencilValue {
        depth: DEPTH_CLEAR,
        stencil: 0,
    },
};

// The shadow compare sampler's test: lit where the reference depth is no
// farther from the light than the stored caster.
pub(in crate::vulkan) const SHADOW_SAMPLE_COMPARE: vk::CompareOp =
    compare_op(DEPTH_INCLUSIVE_COMPARE);

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

    // Reversed depth: a sample is lit where its reference is at or nearer the
    // light than the stored caster.
    #[test]
    fn the_shadow_sampler_passes_at_or_nearer_the_light() {
        assert_eq!(SHADOW_SAMPLE_COMPARE, vk::CompareOp::GREATER_OR_EQUAL);
    }
}
