// Declarative construction for the two Metal descriptor shapes the backend
// builds repeatedly: texture descriptors and vertex descriptors. Each builder
// owns the unsafe objc2 property setters those shapes require.

use objc2::rc::Retained;
use objc2_metal::{
    MTLPixelFormat, MTLStorageMode, MTLTextureDescriptor, MTLTextureType, MTLTextureUsage,
    MTLVertexDescriptor, MTLVertexFormat, MTLVertexStepFunction,
};

// A texture allocation's shape. Fields default to Metal's own descriptor
// defaults, so a site states only what it diverges on.
pub(super) struct TextureDesc {
    pub kind: MTLTextureType,
    pub format: MTLPixelFormat,
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub mip_count: usize,
    pub sample_count: usize,
    pub array_length: usize,
    pub usage: MTLTextureUsage,
    pub storage: MTLStorageMode,
}

impl Default for TextureDesc {
    fn default() -> Self {
        Self {
            kind: MTLTextureType::Type2D,
            format: MTLPixelFormat::RGBA8Unorm,
            width: 1,
            height: 1,
            depth: 1,
            mip_count: 1,
            sample_count: 1,
            array_length: 1,
            usage: MTLTextureUsage::ShaderRead,
            storage: MTLStorageMode::Private,
        }
    }
}

impl TextureDesc {
    pub(super) fn build(&self) -> Retained<MTLTextureDescriptor> {
        let desc = MTLTextureDescriptor::new();
        desc.setTextureType(self.kind);
        desc.setPixelFormat(self.format);
        // SAFETY: plain setter on a fresh descriptor; the width is in range for the property.
        unsafe { desc.setWidth(self.width) };
        // SAFETY: the height is likewise an in-range extent from the caller's shape.
        unsafe { desc.setHeight(self.height) };
        // SAFETY: the depth is an in-range extent; Metal's default of 1 unless stated.
        unsafe { desc.setDepth(self.depth) };
        // SAFETY: the mip count is an in-range level count for this descriptor.
        unsafe { desc.setMipmapLevelCount(self.mip_count) };
        // SAFETY: the sample count is an in-range value the caller chose for this shape.
        unsafe { desc.setSampleCount(self.sample_count) };
        // SAFETY: the array length is in range; Metal's default of 1 unless stated.
        unsafe { desc.setArrayLength(self.array_length) };
        desc.setUsage(self.usage);
        desc.setStorageMode(self.storage);
        desc
    }
}

// One vertex attribute of a vertex descriptor.
pub(super) struct VertexAttr {
    pub index: usize,
    pub format: MTLVertexFormat,
    pub offset: usize,
    pub buffer_index: usize,
}

// One buffer layout of a vertex descriptor.
pub(super) struct VertexLayout {
    pub buffer_index: usize,
    pub stride: usize,
    pub step: MTLVertexStepFunction,
}

// Build a vertex descriptor from declarative attribute and layout lists.
pub(super) fn vertex_descriptor(
    attrs: &[VertexAttr],
    layouts: &[VertexLayout],
) -> Retained<MTLVertexDescriptor> {
    let desc = MTLVertexDescriptor::new();
    for a in attrs {
        // SAFETY: `a.index` is an attribute slot this descriptor declares via the list.
        let slot = unsafe { desc.attributes().objectAtIndexedSubscript(a.index) };
        slot.setFormat(a.format);
        // SAFETY: the offset is the attribute's in-range byte position, as the list declares.
        unsafe { slot.setOffset(a.offset) };
        // SAFETY: the buffer index is an in-range slot the layout list also declares.
        unsafe { slot.setBufferIndex(a.buffer_index) };
    }
    for l in layouts {
        // SAFETY: `l.buffer_index` is a layout slot this descriptor declares via the list.
        let slot = unsafe { desc.layouts().objectAtIndexedSubscript(l.buffer_index) };
        // SAFETY: the stride is an in-range per-vertex size from the declared layout.
        unsafe { slot.setStride(l.stride) };
        slot.setStepFunction(l.step);
    }
    desc
}
