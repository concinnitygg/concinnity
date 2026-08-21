// Declarative construction for the two Metal descriptor shapes the backend
// builds repeatedly: texture descriptors and vertex descriptors. Each builder
// owns the one unsafe block the objc2 property setters require.

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
        // SAFETY: plain descriptor property setters on a freshly created
        // descriptor, all values in range for their properties.
        unsafe {
            desc.setTextureType(self.kind);
            desc.setPixelFormat(self.format);
            desc.setWidth(self.width);
            desc.setHeight(self.height);
            desc.setDepth(self.depth);
            desc.setMipmapLevelCount(self.mip_count);
            desc.setSampleCount(self.sample_count);
            desc.setArrayLength(self.array_length);
            desc.setUsage(self.usage);
            desc.setStorageMode(self.storage);
        }
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
    // SAFETY: plain descriptor property setters; every subscripted slot is one
    // this descriptor declares via the lists.
    unsafe {
        for a in attrs {
            let slot = desc.attributes().objectAtIndexedSubscript(a.index);
            slot.setFormat(a.format);
            slot.setOffset(a.offset);
            slot.setBufferIndex(a.buffer_index);
        }
        for l in layouts {
            let slot = desc.layouts().objectAtIndexedSubscript(l.buffer_index);
            slot.setStride(l.stride);
            slot.setStepFunction(l.step);
        }
    }
    desc
}
