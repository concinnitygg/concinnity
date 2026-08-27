//! In-engine mouse cursor geometry. A `follow_cursor` Sprite is drawn as a
//! classic arrow pointer rather than a plain quad: a filled polygon with a
//! contrasting outline so it stays legible over any scene. Like the rest of the
//! UI overlay, it rides the text pass's sentinel-UV solid-fill path (u < 0), so
//! it needs no new pipeline and renders on every backend. The arrow's diagonal
//! edges are real geometry, not a stair-stepped stack of quads.

// `build_cursor_calls` below is test-only, and it is the only Vec here.
use crate::components::Sprite;
use crate::ecs::CursorShape;
use crate::render_types::{TextDrawCall, TextVertex};
#[cfg(test)]
use alloc::vec::Vec;
use concinnity_core::gfx::overlay::OverlayTransform;

// Arrow silhouette in a normalised space: tip (the hotspot) at the origin,
// pointing down-right, height 1.0 and width ~0.62. Vertices run clockwise
// around the boundary in screen space (y grows downward):
//   V0 tip, V1 left-edge foot, V2 inner notch, V3 tail tip,
//   V4 tail heel, V5 barb root, V6 right barb.
const ARROW: [(f32, f32); 7] = [
    (0.00, 0.00),
    (0.00, 0.86),
    (0.21, 0.65),
    (0.35, 1.00),
    (0.50, 0.93),
    (0.35, 0.59),
    (0.62, 0.59),
];

// Triangulation of the arrow: a fan over the head (tip to the two barbs) plus
// the small tail quad. Indices reference ARROW.
const ARROW_TRIS: [[u16; 3]; 5] = [[0, 1, 2], [0, 2, 5], [0, 5, 6], [2, 3, 4], [2, 4, 5]];

// A double-headed resize arrow centred on the hotspot, pointing east/west; the
// other resize axes are this same silhouette rotated (see `cursor_geometry`). A
// shaft rectangle capped by a triangular head at each end. y grows downward.
//   V0 left tip, V1/V2 left head top/bottom, V3..V6 shaft corners,
//   V7/V8 right head top/bottom, V9 right tip.
const RESIZE_ARROW: [(f32, f32); 10] = [
    (-0.50, 0.00),
    (-0.24, -0.22),
    (-0.24, 0.22),
    (-0.24, -0.08),
    (0.24, -0.08),
    (0.24, 0.08),
    (-0.24, 0.08),
    (0.24, -0.22),
    (0.24, 0.22),
    (0.50, 0.00),
];

// The two head triangles and the two shaft triangles. Indices reference RESIZE_ARROW.
const RESIZE_TRIS: [[u16; 3]; 4] = [[0, 1, 2], [3, 4, 5], [3, 5, 6], [9, 7, 8]];

// Eight unit directions used to stamp the outline around the fill, giving an
// even border ring of one outline-width radius.
const OUTLINE_OFFSETS: [(f32, f32); 8] = [
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (0.707, 0.707),
    (-0.707, 0.707),
    (0.707, -0.707),
    (-0.707, -0.707),
];

// Outline width as a fraction of the cursor height, floored at one pixel.
const OUTLINE_RATIO: f32 = 0.085;
// Arrow height in pixels when a cursor sprite leaves its size unset.
const DEFAULT_CURSOR_PX: f32 = 22.0;
// The cursor sorts above every other overlay layer: a screen stack or the
// editor lifts its elements above 0, and at layer 0 the arrow would sort
// beneath the menu it points at.
const CURSOR_LAYER: i32 = i32::MAX;

// Build the cursor draw calls (one mesh per visible `follow_cursor` sprite) at
// the pointer, drawing `shape`'s silhouette (the arrow, or a resize double-arrow
// over a `cn editor` panel edge). Each sprite's tint is the fill colour and its
// `height` the cursor height; `width` is ignored so the silhouette keeps its
// aspect ratio. The height is authored in the reference canvas, so it is scaled
// by the overlay factor for `viewport` to stay proportional with the menu it
// belongs to; the pointer stays at the live cursor position. Returns empty when
// no font atlas is loaded (the text pipeline is inactive then).
#[cfg(test)]
pub(crate) fn build_cursor_calls(
    sprites: &[Sprite],
    pointer: (f32, f32),
    shape: CursorShape,
    default_atlas_slot: Option<usize>,
    viewport: [f32; 2],
) -> Vec<TextDrawCall> {
    let mut out = crate::call_buffer::TextCallBuffer::default();
    build_cursor_calls_into(
        &mut out,
        sprites,
        pointer,
        shape,
        default_atlas_slot,
        viewport,
    );
    out.take()
}

/// `build_cursor_calls`, appending onto an existing draw list. Sprites without
/// `follow_cursor` are skipped, so the caller can pass its whole sprite slice.
pub fn build_cursor_calls_into(
    out: &mut crate::call_buffer::TextCallBuffer,
    sprites: &[Sprite],
    pointer: (f32, f32),
    shape: CursorShape,
    default_atlas_slot: Option<usize>,
    viewport: [f32; 2],
) {
    let atlas_slot = match default_atlas_slot {
        Some(s) => s,
        None => return,
    };
    let overlay_scale = OverlayTransform::from_viewport(viewport).scale();
    let sil = cursor_geometry(shape);
    for s in sprites {
        if !s.follow_cursor || !s.visible {
            continue;
        }
        let alpha = s.tint[3];
        if alpha <= 0.0 {
            continue;
        }
        let size = if s.height > 0.0 {
            s.height
        } else {
            DEFAULT_CURSOR_PX
        } * overlay_scale;
        let fill = [s.tint[0], s.tint[1], s.tint[2]];
        let outline = outline_color(fill);
        let outline_w = (size * OUTLINE_RATIO).max(1.0);

        let (vertices, indices) = out.geometry();
        let mut call = TextDrawCall {
            vertices,
            indices,
            atlas_slot,
            // The cursor is never clipped: it draws on top of everything.
            clip_rect: None,
            layer: CURSOR_LAYER,
        };
        // Outline first so the fill, appended after, composites on top of it
        // (the overlay draws indexed triangles in order, with no depth test).
        for (dx, dy) in OUTLINE_OFFSETS {
            let o = (pointer.0 + dx * outline_w, pointer.1 + dy * outline_w);
            push_shape(&mut call, o, size, outline, alpha, &sil);
        }
        push_shape(&mut call, pointer, size, fill, alpha, &sil);
        out.calls.push(call);
    }
}

// A cursor silhouette: its boundary vertices, triangulation, and the rotation
// (cos, sin) applied to place it on its axis.
struct Silhouette {
    verts: &'static [(f32, f32)],
    tris: &'static [[u16; 3]],
    rot: (f32, f32),
}

// The silhouette for `shape`. The arrow is unrotated (hotspot at its tip); each
// resize cursor is the shared horizontal double-arrow rotated onto its axis
// (hotspot at its centre).
fn cursor_geometry(shape: CursorShape) -> Silhouette {
    const DIAG: f32 = core::f32::consts::FRAC_1_SQRT_2;
    let arrow = || Silhouette {
        verts: &ARROW[..],
        tris: &ARROW_TRIS[..],
        rot: (1.0, 0.0),
    };
    let resize = |rot| Silhouette {
        verts: &RESIZE_ARROW[..],
        tris: &RESIZE_TRIS[..],
        rot,
    };
    match shape {
        CursorShape::Default => arrow(),
        CursorShape::ResizeEW => resize((1.0, 0.0)),
        CursorShape::ResizeNS => resize((0.0, 1.0)),
        CursorShape::ResizeNWSE => resize((DIAG, DIAG)),
        CursorShape::ResizeNESW => resize((DIAG, -DIAG)),
    }
}

// Append one silhouette (rotated and scaled by `size`, hotspot at `origin`) to a
// draw call.
fn push_shape(
    call: &mut TextDrawCall,
    origin: (f32, f32),
    size: f32,
    color: [f32; 3],
    alpha: f32,
    sil: &Silhouette,
) {
    let base = call.vertices.len() as u16;
    let (c, s) = sil.rot;
    for &(nx, ny) in sil.verts {
        let rx = nx * c - ny * s;
        let ry = nx * s + ny * c;
        call.vertices.push(TextVertex {
            pos: [origin.0 + rx * size, origin.1 + ry * size],
            // sentinel u < 0 -> solid-fill path; v carries alpha
            uv: [-1.0, alpha],
            color,
            mode: 0.0,
        });
    }
    for tri in sil.tris {
        call.indices.push(base + tri[0]);
        call.indices.push(base + tri[1]);
        call.indices.push(base + tri[2]);
    }
}

// Pick an outline that contrasts the fill: a near-black border under a light
// cursor, a near-white border under a dark one. Keeps any tint legible.
fn outline_color(fill: [f32; 3]) -> [f32; 3] {
    let luma = 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2];
    if luma > 0.5 {
        [0.05, 0.05, 0.06]
    } else {
        [0.95, 0.95, 0.96]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::asset_id::AssetId;

    fn cursor(tint: [f32; 4], height: f32) -> Sprite {
        Sprite {
            asset_id: AssetId::default(),
            x: 0.0,
            y: 0.0,
            width: height,
            height,
            texture: None,
            tint,
            follow_cursor: true,
            visible: true,
            screen: None,
            fit: crate::components::SpriteFit::Fit,
            corner_radius: 0.0,
            border_width: 0.0,
            border_color: [0.0, 0.0, 0.0, 1.0],
        }
    }

    #[test]
    fn no_fonts_means_no_calls() {
        let c = cursor([1.0, 1.0, 1.0, 1.0], 22.0);
        assert!(
            build_cursor_calls(
                core::slice::from_ref(&c),
                (10.0, 10.0),
                CursorShape::Default,
                None,
                [0.0, 0.0]
            )
            .is_empty()
        );
    }

    #[test]
    fn builds_outline_then_fill_mesh() {
        let c = cursor([1.0, 1.0, 1.0, 1.0], 22.0);
        let calls = build_cursor_calls(
            core::slice::from_ref(&c),
            (100.0, 50.0),
            CursorShape::Default,
            Some(0),
            [0.0, 0.0],
        );
        assert_eq!(calls.len(), 1);
        // Eight outline stamps plus one fill, seven vertices each.
        assert_eq!(calls[0].vertices.len(), 9 * ARROW.len());
        assert_eq!(calls[0].indices.len(), 9 * ARROW_TRIS.len() * 3);
        // The tip of the fill arrow (last stamp's first vertex) sits exactly on
        // the pointer; the outline stamps are offset off it.
        let tip = calls[0].vertices[8 * ARROW.len()];
        assert_eq!(tip.pos, [100.0, 50.0]);
        // Fill keeps the sprite tint; outline does not.
        assert_eq!(tip.color, [1.0, 1.0, 1.0]);
        assert_ne!(calls[0].vertices[0].color, [1.0, 1.0, 1.0]);
        // Every vertex uses the solid-fill sentinel and carries the alpha.
        for v in &calls[0].vertices {
            assert!(v.uv[0] < 0.0);
            assert!((v.uv[1] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn invisible_or_transparent_cursor_is_skipped() {
        let mut hidden = cursor([1.0, 1.0, 1.0, 1.0], 22.0);
        hidden.visible = false;
        assert!(
            build_cursor_calls(
                core::slice::from_ref(&hidden),
                (0.0, 0.0),
                CursorShape::Default,
                Some(0),
                [0.0, 0.0]
            )
            .is_empty()
        );
        let clear = cursor([1.0, 1.0, 1.0, 0.0], 22.0);
        assert!(
            build_cursor_calls(
                core::slice::from_ref(&clear),
                (0.0, 0.0),
                CursorShape::Default,
                Some(0),
                [0.0, 0.0]
            )
            .is_empty()
        );
    }

    #[test]
    fn outline_contrasts_the_fill() {
        // Light fill -> dark outline, dark fill -> light outline.
        assert!(outline_color([1.0, 1.0, 1.0])[0] < 0.5);
        assert!(outline_color([0.0, 0.0, 0.0])[0] > 0.5);
    }

    #[test]
    fn unset_height_falls_back_to_default_size() {
        let c = cursor([1.0, 1.0, 1.0, 1.0], 0.0);
        let calls = build_cursor_calls(
            core::slice::from_ref(&c),
            (0.0, 0.0),
            CursorShape::Default,
            Some(0),
            [0.0, 0.0],
        );
        // The lowest vertex (tail tip, ny = 1.0) reaches the default height.
        let max_y = calls[0]
            .vertices
            .iter()
            .map(|v| v.pos[1])
            .fold(f32::MIN, f32::max);
        assert!((max_y - DEFAULT_CURSOR_PX).abs() < OUTLINE_PX_TOLERANCE);
    }

    #[test]
    fn arrow_scales_with_the_overlay() {
        // At twice the reference size the overlay scale is 2.0, so the arrow
        // height doubles while the tip stays on the pointer. Measure the fill
        // arrow (the last stamp) so the outline ring's extra width is excluded.
        let c = cursor([1.0, 1.0, 1.0, 1.0], 22.0);
        let calls = build_cursor_calls(
            core::slice::from_ref(&c),
            (0.0, 0.0),
            CursorShape::Default,
            Some(0),
            [2560.0, 1440.0],
        );
        let fill = &calls[0].vertices[8 * ARROW.len()..];
        let max_y = fill.iter().map(|v| v.pos[1]).fold(f32::MIN, f32::max);
        // Tail tip (ny = 1.0) at pointer y = 0 reaches the doubled height.
        assert!((max_y - 44.0).abs() < 1e-3, "max_y={max_y}");
    }

    // A resize shape draws the double-arrow silhouette centred on the pointer
    // (both tips equidistant), unlike the arrow whose hotspot is its tip.
    #[test]
    fn resize_shape_draws_a_centered_double_arrow() {
        let c = cursor([1.0, 1.0, 1.0, 1.0], 20.0);
        let calls = build_cursor_calls(
            core::slice::from_ref(&c),
            (100.0, 100.0),
            CursorShape::ResizeEW,
            Some(0),
            [0.0, 0.0],
        );
        assert_eq!(calls.len(), 1);
        // Eight outline stamps plus one fill, ten vertices each.
        assert_eq!(calls[0].vertices.len(), 9 * RESIZE_ARROW.len());
        assert_eq!(calls[0].indices.len(), 9 * RESIZE_TRIS.len() * 3);
        // The fill's two tips (V0 left, V9 right) sit either side of the pointer.
        let fill = &calls[0].vertices[8 * RESIZE_ARROW.len()..];
        let left = fill[0].pos;
        let right = fill[9].pos;
        assert!(
            left[0] < 100.0 && right[0] > 100.0,
            "tips straddle the pointer x"
        );
        assert!((left[1] - 100.0).abs() < 1e-4 && (right[1] - 100.0).abs() < 1e-4);
        assert!(
            ((100.0 - left[0]) - (right[0] - 100.0)).abs() < 1e-4,
            "the pointer is centred between the tips"
        );
    }

    // The vertical resize cursor is the horizontal one rotated onto the y axis:
    // its tips straddle the pointer in y, not x.
    #[test]
    fn resize_ns_rotates_onto_the_vertical_axis() {
        let c = cursor([1.0, 1.0, 1.0, 1.0], 20.0);
        let calls = build_cursor_calls(
            core::slice::from_ref(&c),
            (100.0, 100.0),
            CursorShape::ResizeNS,
            Some(0),
            [0.0, 0.0],
        );
        let fill = &calls[0].vertices[8 * RESIZE_ARROW.len()..];
        let top = fill[0].pos;
        let bottom = fill[9].pos;
        assert!((top[0] - 100.0).abs() < 1e-4 && (bottom[0] - 100.0).abs() < 1e-4);
        assert!(
            top[1] < 100.0 && bottom[1] > 100.0,
            "tips straddle the pointer y"
        );
    }

    // The outline stamp pushes the tail a fraction of a pixel past the fill
    // height, so allow a small tolerance in the size check.
    const OUTLINE_PX_TOLERANCE: f32 = 2.0;
}
