// Compute dispatches sized by a kernel's thread group.

use concinnity_engine::render::shader::programs::abi::group::ThreadGroup;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

fn size([width, height, depth]: [u32; 3]) -> MTLSize {
    MTLSize {
        width: width as usize,
        height: height as usize,
        depth: depth as usize,
    }
}

// One thread per element of a `threads` grid, in `group` groups. Metal trims
// the last group on each axis to the grid.
pub(super) fn dispatch_threads(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    threads: [u32; 3],
    group: ThreadGroup,
) {
    enc.dispatchThreads_threadsPerThreadgroup(size(threads), size(group.size));
}

// `groups` whole `group` groups, for a kernel whose threads share group
// memory or that sizes its own grid.
pub(super) fn dispatch_groups(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    groups: [u32; 3],
    group: ThreadGroup,
) {
    enc.dispatchThreadgroups_threadsPerThreadgroup(size(groups), size(group.size));
}

// The threads one `group` group runs, for an indirect dispatch.
pub(super) fn threadgroup(group: ThreadGroup) -> MTLSize {
    size(group.size)
}
