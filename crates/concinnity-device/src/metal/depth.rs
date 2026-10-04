// The core depth conventions in Metal terms: the depth-stencil states the passes
// bind and the clear depth their attachments load with.

use concinnity_core::render::depth::{DepthCompare, DepthConvention};
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
    // The opaque-geometry test against the camera's depth: nearer fragments
    // pass and write.
    pub(in crate::metal) const fn camera_write() -> Self {
        Self {
            compare: DepthConvention::Camera.write_compare(),
            write: true,
        }
    }

    // A camera-depth pass whose shader writes a depth no farther than the
    // rasterized one: equal depth passes too.
    pub(in crate::metal) const fn camera_write_inclusive() -> Self {
        Self {
            compare: DepthConvention::Camera.inclusive_compare(),
            write: true,
        }
    }

    // Tested against the camera's depth without writing it.
    pub(in crate::metal) const fn camera_read_only() -> Self {
        Self {
            compare: DepthConvention::Camera.inclusive_compare(),
            write: false,
        }
    }

    // A shadow caster: nearer the light passes and writes.
    pub(in crate::metal) const fn shadow_write() -> Self {
        Self {
            compare: DepthConvention::Shadow.write_compare(),
            write: true,
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

// The depth an attachment clears to for a target drawn under `convention`.
pub(in crate::metal) fn clear_depth(convention: DepthConvention) -> f64 {
    f64::from(convention.clear())
}

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
}
