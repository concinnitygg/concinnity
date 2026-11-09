// Resource descriptions the graph tests declare their textures and buffers with.

use super::types::{BufferDesc, BufferUsage, PixelFormat, TextureDesc, TextureSize, TextureUsage};

pub(super) fn tex() -> TextureDesc {
    TextureDesc::texture_2d(
        TextureSize::Drawable,
        TextureSize::Drawable,
        PixelFormat::Rgba16Float,
        TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
    )
}

pub(super) fn buf() -> BufferDesc {
    BufferDesc {
        size_bytes: None,
        usage: BufferUsage::STORAGE,
    }
}
