// Safe encoder commands for the Metal backend.
//
// objc2 marks every generated message send `unsafe`, which put an `unsafe`
// block around each run of encoder binds. The preconditions those blocks
// restated are discharged here instead: the resource arguments are borrowed
// objc2 objects, so the borrow itself proves the resource outlives the call,
// and the argument-table index and inline-constant size are checked against
// Metal's documented limits before the send.
//
// Draw and dispatch commands are deliberately absent. Their correctness
// depends on bound state no borrow can express (whether the vertex range or
// the index values address the bound buffers), which is the same line objc2
// itself draws: it marks `useResource` and `dispatchThreads` safe and the
// draws unsafe. Those sites keep an explicit `unsafe` block.
#![deny(unsafe_op_in_unsafe_fn)]

use bytemuck::NoUninit;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLAccelerationStructure, MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLDepthStencilState, MTLRenderCommandEncoder, MTLRenderPipelineState, MTLSamplerState,
    MTLTexture,
};

// Argument-table sizes a single graphics or compute function may address, from
// the Metal feature-set tables. A bind past the end of a table writes outside
// it, so every index is checked against these.
const BUFFER_TABLE_LEN: usize = 31;
const TEXTURE_TABLE_LEN: usize = 128;
const SAMPLER_TABLE_LEN: usize = 16;

// Largest inline-constant block `setBytes` accepts. Larger data belongs in a
// buffer.
const MAX_INLINE_BYTES: usize = 4096;

fn check_buffer_index(index: usize) {
    assert!(
        index < BUFFER_TABLE_LEN,
        "buffer index {index} exceeds the {BUFFER_TABLE_LEN}-entry argument table"
    );
}

fn check_texture_index(index: usize) {
    assert!(
        index < TEXTURE_TABLE_LEN,
        "texture index {index} exceeds the {TEXTURE_TABLE_LEN}-entry argument table"
    );
}

fn check_sampler_index(index: usize) {
    assert!(
        index < SAMPLER_TABLE_LEN,
        "sampler index {index} exceeds the {SAMPLER_TABLE_LEN}-entry argument table"
    );
}

fn check_inline_len(len: usize) {
    assert!(
        len <= MAX_INLINE_BYTES,
        "inline constant of {len} bytes exceeds the {MAX_INLINE_BYTES}-byte limit"
    );
}

// Safe binding commands on a render command encoder.
pub(super) trait RenderEncode {
    fn set_pipeline(&self, pso: &ProtocolObject<dyn MTLRenderPipelineState>);
    fn set_depth_stencil(&self, state: &ProtocolObject<dyn MTLDepthStencilState>);
    fn set_vertex_buffer(
        &self,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        offset: usize,
        index: usize,
    );
    fn set_fragment_buffer(
        &self,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        offset: usize,
        index: usize,
    );
    // Upload `value` as an inline constant. The pointer and length both come
    // from the one reference, so they cannot disagree, and `NoUninit` rules out
    // copying padding.
    fn set_vertex_value<T: NoUninit>(&self, value: &T, index: usize);
    fn set_fragment_value<T: NoUninit>(&self, value: &T, index: usize);
    fn set_fragment_texture(&self, texture: &ProtocolObject<dyn MTLTexture>, index: usize);
    fn set_fragment_sampler(&self, sampler: &ProtocolObject<dyn MTLSamplerState>, index: usize);
    fn set_fragment_acceleration_structure(
        &self,
        structure: &ProtocolObject<dyn MTLAccelerationStructure>,
        index: usize,
    );
}

impl RenderEncode for ProtocolObject<dyn MTLRenderCommandEncoder> {
    fn set_pipeline(&self, pso: &ProtocolObject<dyn MTLRenderPipelineState>) {
        self.setRenderPipelineState(pso);
    }

    fn set_depth_stencil(&self, state: &ProtocolObject<dyn MTLDepthStencilState>) {
        self.setDepthStencilState(Some(state));
    }

    fn set_vertex_buffer(
        &self,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        offset: usize,
        index: usize,
    ) {
        check_buffer_index(index);
        // SAFETY: `buffer` outlives the call through its borrow, and the index
        // is within the buffer argument table.
        unsafe { self.setVertexBuffer_offset_atIndex(Some(buffer), offset, index) };
    }

    fn set_fragment_buffer(
        &self,
        buffer: &ProtocolObject<dyn MTLBuffer>,
        offset: usize,
        index: usize,
    ) {
        check_buffer_index(index);
        // SAFETY: `buffer` outlives the call through its borrow, and the index
        // is within the buffer argument table.
        unsafe { self.setFragmentBuffer_offset_atIndex(Some(buffer), offset, index) };
    }

    fn set_vertex_value<T: NoUninit>(&self, value: &T, index: usize) {
        check_buffer_index(index);
        let len = size_of::<T>();
        check_inline_len(len);
        // SAFETY: the pointer and length both describe `value`, which is live
        // for the call and holds no uninitialised bytes, and the index is
        // within the buffer argument table.
        unsafe {
            self.setVertexBytes_length_atIndex(std::ptr::NonNull::from(value).cast(), len, index);
        }
    }

    fn set_fragment_value<T: NoUninit>(&self, value: &T, index: usize) {
        check_buffer_index(index);
        let len = size_of::<T>();
        check_inline_len(len);
        // SAFETY: the pointer and length both describe `value`, which is live
        // for the call and holds no uninitialised bytes, and the index is
        // within the buffer argument table.
        unsafe {
            self.setFragmentBytes_length_atIndex(std::ptr::NonNull::from(value).cast(), len, index);
        }
    }

    fn set_fragment_texture(&self, texture: &ProtocolObject<dyn MTLTexture>, index: usize) {
        check_texture_index(index);
        // SAFETY: `texture` outlives the call through its borrow, and the index
        // is within the texture argument table.
        unsafe { self.setFragmentTexture_atIndex(Some(texture), index) };
    }

    fn set_fragment_sampler(&self, sampler: &ProtocolObject<dyn MTLSamplerState>, index: usize) {
        check_sampler_index(index);
        // SAFETY: `sampler` outlives the call through its borrow, and the index
        // is within the sampler argument table.
        unsafe { self.setFragmentSamplerState_atIndex(Some(sampler), index) };
    }

    fn set_fragment_acceleration_structure(
        &self,
        structure: &ProtocolObject<dyn MTLAccelerationStructure>,
        index: usize,
    ) {
        check_buffer_index(index);
        // SAFETY: `structure` outlives the call through its borrow, and the
        // index is within the buffer argument table it shares.
        unsafe { self.setFragmentAccelerationStructure_atBufferIndex(Some(structure), index) };
    }
}

// Safe binding commands on a compute command encoder.
pub(super) trait ComputeEncode {
    fn set_pipeline(&self, pso: &ProtocolObject<dyn MTLComputePipelineState>);
    fn set_buffer(&self, buffer: &ProtocolObject<dyn MTLBuffer>, offset: usize, index: usize);
    // Upload `value` as an inline constant. The pointer and length both come
    // from the one reference, so they cannot disagree, and `NoUninit` rules out
    // copying padding.
    fn set_value<T: NoUninit>(&self, value: &T, index: usize);
    fn set_texture(&self, texture: &ProtocolObject<dyn MTLTexture>, index: usize);
    fn set_sampler(&self, sampler: &ProtocolObject<dyn MTLSamplerState>, index: usize);
}

impl ComputeEncode for ProtocolObject<dyn MTLComputeCommandEncoder> {
    fn set_pipeline(&self, pso: &ProtocolObject<dyn MTLComputePipelineState>) {
        self.setComputePipelineState(pso);
    }

    fn set_buffer(&self, buffer: &ProtocolObject<dyn MTLBuffer>, offset: usize, index: usize) {
        check_buffer_index(index);
        // SAFETY: `buffer` outlives the call through its borrow, and the index
        // is within the buffer argument table.
        unsafe { self.setBuffer_offset_atIndex(Some(buffer), offset, index) };
    }

    fn set_value<T: NoUninit>(&self, value: &T, index: usize) {
        check_buffer_index(index);
        let len = size_of::<T>();
        check_inline_len(len);
        // SAFETY: the pointer and length both describe `value`, which is live
        // for the call and holds no uninitialised bytes, and the index is
        // within the buffer argument table.
        unsafe {
            self.setBytes_length_atIndex(std::ptr::NonNull::from(value).cast(), len, index);
        }
    }

    fn set_texture(&self, texture: &ProtocolObject<dyn MTLTexture>, index: usize) {
        check_texture_index(index);
        // SAFETY: `texture` outlives the call through its borrow, and the index
        // is within the texture argument table.
        unsafe { self.setTexture_atIndex(Some(texture), index) };
    }

    fn set_sampler(&self, sampler: &ProtocolObject<dyn MTLSamplerState>, index: usize) {
        check_sampler_index(index);
        // SAFETY: `sampler` outlives the call through its borrow, and the index
        // is within the sampler argument table.
        unsafe { self.setSamplerState_atIndex(Some(sampler), index) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_bounds_accept_the_last_slot_and_reject_the_next() {
        check_buffer_index(BUFFER_TABLE_LEN - 1);
        check_texture_index(TEXTURE_TABLE_LEN - 1);
        check_sampler_index(SAMPLER_TABLE_LEN - 1);
        check_inline_len(MAX_INLINE_BYTES);
        for over in [
            std::panic::catch_unwind(|| check_buffer_index(BUFFER_TABLE_LEN)),
            std::panic::catch_unwind(|| check_texture_index(TEXTURE_TABLE_LEN)),
            std::panic::catch_unwind(|| check_sampler_index(SAMPLER_TABLE_LEN)),
            std::panic::catch_unwind(|| check_inline_len(MAX_INLINE_BYTES + 1)),
        ] {
            assert!(over.is_err());
        }
    }
}
