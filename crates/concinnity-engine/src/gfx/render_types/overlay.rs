//! Overlay layouts: text and line vertices and the text draw call.

use alloc::vec::Vec;

/// Compact vertex type used exclusively by the text render pass.
/// 32 bytes: screen-pixel position, atlas UV, text color, and sampling mode.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct TextVertex {
    /// Screen-space position in pixels (x from left, y from top).
    pub pos: [f32; 2],
    /// Normalized UV into the bound atlas texture. A negative u marks a
    /// solid-fill quad (no sampling; alpha carried in v).
    pub uv: [f32; 2],
    /// Linear-space RGB color: text color, fill color, or texture tint.
    pub color: [f32; 3],
    /// 0 = SDF glyph sampling; > 0 = textured quad (RGBA sample multiplied by
    /// `color`), with the value itself as the quad's alpha multiplier.
    pub mode: f32,
}

/// Compact vertex type used exclusively by the line pass. 32 bytes: the
/// expanded ribbon corner in world space, its signed position across the ribbon
/// width (for the shader's edge fade), and the corner color.
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::NoUninit)]
#[repr(C)]
pub struct LineVertex {
    /// World-space position of this ribbon corner. Already offset off the line
    /// center by the CPU expansion, so the shader only applies the camera VP.
    pub pos: [f32; 3],
    /// Signed offset across the ribbon: -1 on one edge, +1 on the other. The
    /// fragment fades the outer pixel from it, so lines antialias without MSAA.
    pub edge: f32,
    /// Linear-space RGBA. Alpha carries the distance fade, interpolated along
    /// the ribbon.
    pub color: [f32; 4],
}

/// Uniforms pushed to the text vertex shader once per text draw call.
/// Carries the framebuffer size so the shader can convert pixel coords to NDC.
#[derive(Copy, Clone, bytemuck::NoUninit)]
#[repr(C)]
pub struct TextUniforms {
    /// Framebuffer width in pixels.
    pub win_width: f32,
    /// Framebuffer height in pixels.
    pub win_height: f32,
    /// Padding so the field layout matches the shader-side struct.
    pub _pad: [f32; 2],
}

/// One text draw call: quads for all visible characters sharing one atlas texture.
pub struct TextDrawCall {
    /// One quad (4 vertices, 6 indices) per character.
    pub vertices: Vec<TextVertex>,
    /// Triangle indices into `vertices`.
    pub indices: Vec<u16>,
    /// Index into the backend's text atlas texture array.
    pub atlas_slot: usize,
    /// Optional clip rectangle in window pixels `[x, y, width, height]`. When
    /// set, the backend scissors this call to the rect so a scrollable UI panel's
    /// off-band rows do not bleed over its chrome. `None` draws unclipped.
    pub clip_rect: Option<[f32; 4]>,
    /// Draw layer for overlay ordering: calls are stable-sorted by this (ascending,
    /// so higher draws on top) before submission when a `HudLayers` override is
    /// active. `0` for everything by default, which leaves insertion order intact.
    pub layer: i32,
}
