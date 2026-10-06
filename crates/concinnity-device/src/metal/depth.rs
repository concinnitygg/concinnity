// The core depth convention in Metal terms: the depth-stencil states the passes
// bind, the clear depth their attachments load with, and the compare the
// shadow sampler takes.

use concinnity_core::render::depth::{
    DEPTH_CLEAR, DEPTH_INCLUSIVE_COMPARE, DEPTH_WRITE_COMPARE, DepthCompare,
};
use concinnity_core::render::error::{RenderError, RenderResult};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCompareFunction, MTLDepthStencilDescriptor, MTLDepthStencilState, MTLDevice};

// A depth test and whether it writes. Built into an `MTLDepthStencilState`
// once; a pass binds the state rather than this.
#[derive(Clone, Copy, Debug)]
pub(in crate::metal) struct Depth {
    compare: DepthCompare,
    write: bool,
}

impl Depth {
    // The opaque-geometry test: nearer fragments pass and write. The camera's
    // opaque passes and every shadow caster draw with it.
    pub(in crate::metal) const fn write() -> Self {
        Self {
            compare: DEPTH_WRITE_COMPARE,
            write: true,
        }
    }

    // A pass whose shader writes a depth no farther than the rasterized one:
    // equal depth passes too.
    pub(in crate::metal) const fn write_inclusive() -> Self {
        Self {
            compare: DEPTH_INCLUSIVE_COMPARE,
            write: true,
        }
    }

    // Tested against depth another pass wrote, without writing it.
    pub(in crate::metal) const fn read_only() -> Self {
        Self {
            compare: DEPTH_INCLUSIVE_COMPARE,
            write: false,
        }
    }

    pub(in crate::metal) fn state(
        self,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> RenderResult<Retained<ProtocolObject<dyn MTLDepthStencilState>>> {
        let desc = MTLDepthStencilDescriptor::new();
        desc.setDepthCompareFunction(compare_function(self.compare));
        desc.setDepthWriteEnabled(self.write);
        device
            .newDepthStencilStateWithDescriptor(&desc)
            .ok_or_else(|| RenderError::Other("failed to create depth stencil state".into()))
    }
}

const fn compare_function(compare: DepthCompare) -> MTLCompareFunction {
    match compare {
        DepthCompare::Less => MTLCompareFunction::Less,
        DepthCompare::LessEqual => MTLCompareFunction::LessEqual,
        DepthCompare::Greater => MTLCompareFunction::Greater,
        DepthCompare::GreaterEqual => MTLCompareFunction::GreaterEqual,
    }
}

// The depth every depth attachment clears to.
pub(in crate::metal) const CLEAR_DEPTH: f64 = DEPTH_CLEAR as f64;

// The shadow compare sampler's test: lit where the reference depth is no
// farther from the light than the stored caster.
pub(in crate::metal) const SHADOW_SAMPLE_COMPARE: MTLCompareFunction =
    compare_function(DEPTH_INCLUSIVE_COMPARE);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_compare_maps_to_its_metal_function() {
        assert_eq!(
            compare_function(DepthCompare::Less),
            MTLCompareFunction::Less
        );
        assert_eq!(
            compare_function(DepthCompare::LessEqual),
            MTLCompareFunction::LessEqual
        );
        assert_eq!(
            compare_function(DepthCompare::Greater),
            MTLCompareFunction::Greater
        );
        assert_eq!(
            compare_function(DepthCompare::GreaterEqual),
            MTLCompareFunction::GreaterEqual
        );
    }

    // Reversed depth: a sample is lit where its reference is at or nearer the
    // light than the stored caster, i.e. greater or equal.
    #[test]
    fn the_shadow_sampler_passes_at_or_nearer_the_light() {
        assert_eq!(SHADOW_SAMPLE_COMPARE, MTLCompareFunction::GreaterEqual);
        assert_eq!(CLEAR_DEPTH, 0.0);
    }
}
