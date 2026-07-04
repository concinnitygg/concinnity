// src/gfx/sprite.rs
//
// Sprite quad assembly. Piggybacks on the text render pass: a plain Sprite is
// emitted as a TextDrawCall containing a single quad with the sentinel UV
// (u < 0) the text shader interprets as a solid-coloured fill (alpha carried
// in v); a Sprite with a `texture` is emitted with real 0..1 UVs and a
// positive vertex `mode` so the shader samples the sprite's texture, which
// lives in the same atlas pool as the font atlases. Either way, screen-space
// rectangles need no pipeline of their own.

use std::collections::HashMap;

use crate::assets::{Sprite, SpriteFit};
use crate::ecs::asset_id::AssetId;
use crate::gfx::render_types::{TextDrawCall, TextVertex};
use concinnity_core::gfx::overlay::{OverlayTransform, UI_REFERENCE_SIZE};

// A view-owned sprite that spans the whole reference canvas is a full-screen
// backdrop (e.g. a menu dim): it is stretched to fill the live window rather
// than uniform-scaled, and an opaque one hides the scene behind it.
pub(crate) fn covers_canvas(s: &Sprite) -> bool {
    let [ref_w, ref_h] = UI_REFERENCE_SIZE;
    s.view.is_some()
        && s.x <= 0.0
        && s.y <= 0.0
        && s.x + s.width >= ref_w
        && s.y + s.height >= ref_h
}

// Build a TextDrawCall per visible Sprite. `default_atlas_slot` is the atlas
// a solid-fill call binds (the shader does not sample for sentinel-UV verts,
// but the backend still expects a valid slot). Pass the slot of any loaded
// font; returns an empty list when there are no fonts (the text pipeline
// isn't initialised in that case). `texture_slots` maps a Sprite's Texture
// asset to its slot in the atlas pool; a textured sprite whose texture never
// made it there falls back to a solid fill. `viewport` is the live logical
// window size: view-owned sprites are overlay UI authored in the reference
// canvas and are mapped onto the window so menus scale with it; HUD / scene
// sprites (view == None) keep literal window pixels.
pub(crate) fn build_sprite_calls(
    sprites: &[&Sprite],
    default_atlas_slot: Option<usize>,
    texture_slots: &HashMap<AssetId, usize>,
    viewport: [f32; 2],
    clips: &HashMap<AssetId, [f32; 4]>,
) -> Vec<TextDrawCall> {
    let fill_slot = match default_atlas_slot {
        Some(s) => s,
        None => return Vec::new(),
    };
    let overlay = OverlayTransform::from_viewport(viewport);
    let cover = OverlayTransform::cover_from_viewport(viewport);
    let [vw, vh] = viewport;
    let mut calls = Vec::new();
    for s in sprites {
        if !s.visible {
            continue;
        }
        let [r, g, b, a] = s.tint;
        if a <= 0.0 {
            continue;
        }
        let (x0, y0, x1, y1) = if s.view.is_some() {
            if s.fit == SpriteFit::Cover {
                // Full-bleed stage imagery: uniform fill, centered crop. The
                // canvas edges map at or beyond the window edges, so edge-
                // anchored content (a bottom-anchored portrait) stays flush.
                let (ax, ay) = cover.forward(s.x, s.y);
                let (bx, by) = cover.forward(s.x + s.width, s.y + s.height);
                (ax, ay, bx, by)
            } else if covers_canvas(s) && vw > 0.0 && vh > 0.0 {
                // A view-owned sprite spanning the whole reference canvas is a
                // full-screen backdrop (e.g. a menu dim): always fill the live
                // window instead of uniform-scaling, which would letterbox it.
                (0.0, 0.0, vw, vh)
            } else {
                let (ax, ay) = overlay.forward(s.x, s.y);
                let (bx, by) = overlay.forward(s.x + s.width, s.y + s.height);
                (ax, ay, bx, by)
            }
        } else {
            (s.x, s.y, s.x + s.width, s.y + s.height)
        };
        let texture_slot = s.texture.and_then(|t| texture_slots.get(&t).copied());
        // UVs derive from the vertex position inside the rect so arbitrary
        // boundary geometry (the rounded-corner path) samples correctly; for
        // the plain quad this reproduces the 0..1 corner UVs exactly.
        let (w, h) = ((x1 - x0).max(f32::EPSILON), (y1 - y0).max(f32::EPSILON));
        let v = |x: f32, y: f32, alpha: f32| match texture_slot {
            // Textured quad: real UVs, tint in color, alpha in mode.
            Some(_) => TextVertex {
                pos: [x, y],
                uv: [(x - x0) / w, (y - y0) / h],
                color: [r, g, b],
                mode: alpha,
            },
            // Solid fill: sentinel u < 0, alpha carried in v.
            None => TextVertex {
                pos: [x, y],
                uv: [-1.0, alpha],
                color: [r, g, b],
                mode: 0.0,
            },
        };
        // The corner radius is authored in the sprite's own pixel space; map
        // it through the same scale the rect took (the overlay transforms are
        // uniform; the full-canvas stretch takes the smaller axis).
        let scale = if s.width > 0.0 && s.height > 0.0 {
            ((x1 - x0) / s.width).min((y1 - y0) / s.height)
        } else {
            1.0
        };
        let radius = (s.corner_radius * scale).min(w / 2.0).min(h / 2.0);
        let (vertices, indices) = if radius > 0.5 {
            rounded_rect_geometry(x0, y0, x1, y1, radius, a, v)
        } else {
            (
                vec![v(x0, y0, a), v(x1, y0, a), v(x1, y1, a), v(x0, y1, a)],
                vec![0, 1, 2, 0, 2, 3],
            )
        };
        calls.push(TextDrawCall {
            vertices,
            indices,
            atlas_slot: texture_slot.unwrap_or(fill_slot),
            clip_rect: clips
                .get(&s.asset_id)
                .map(|b| crate::gfx::text::band_to_window(&overlay, *b)),
        });
    }
    calls
}

// Arc steps per rounded corner. Six segments keep a 10-15 px UI radius
// visually smooth once the feathered edge blends the silhouette.
const CORNER_SEGMENTS: usize = 6;
// Width (window pixels) of the soft edge ring. The solid interior stops this
// far inside the authored boundary and fades to transparent at it, so the
// silhouette never grows past the authored rect.
const EDGE_FEATHER: f32 = 1.25;

// Tessellate a rounded rectangle in window space: a solid convex polygon
// inset one feather width inside the authored boundary, fanned from its first
// point, plus a fading ring out to the boundary for anti-aliasing. Vertex
// alpha carries the fade (both sprite modes read per-vertex alpha).
fn rounded_rect_geometry(
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    radius: f32,
    alpha: f32,
    mut v: impl FnMut(f32, f32, f32) -> TextVertex,
) -> (Vec<TextVertex>, Vec<u16>) {
    use std::f32::consts::{FRAC_PI_2, PI};
    // Corner arc centers in polygon order, each with its start angle; y grows
    // downward so the arcs sweep clockwise around the boundary.
    let corners = [
        (x0 + radius, y0 + radius, PI),
        (x1 - radius, y0 + radius, 1.5 * PI),
        (x1 - radius, y1 - radius, 0.0),
        (x0 + radius, y1 - radius, FRAC_PI_2),
    ];
    let mut boundary = Vec::with_capacity(4 * (CORNER_SEGMENTS + 1));
    for &(cx, cy, start) in &corners {
        for i in 0..=CORNER_SEGMENTS {
            let t = start + (i as f32 / CORNER_SEGMENTS as f32) * FRAC_PI_2;
            boundary.push((cx, cy, t.cos(), t.sin()));
        }
    }
    let m = boundary.len();
    let inner_r = (radius - EDGE_FEATHER).max(0.0);
    let mut vertices = Vec::with_capacity(2 * m);
    for &(cx, cy, cos, sin) in &boundary {
        vertices.push(v(cx + inner_r * cos, cy + inner_r * sin, alpha));
    }
    for &(cx, cy, cos, sin) in &boundary {
        vertices.push(v(cx + radius * cos, cy + radius * sin, 0.0));
    }
    let mut indices = Vec::with_capacity(3 * (m - 2) + 6 * m);
    for i in 1..m - 1 {
        indices.extend([0, i as u16, (i + 1) as u16]);
    }
    for i in 0..m {
        let j = (i + 1) % m;
        let (i, j, m) = (i as u16, j as u16, m as u16);
        indices.extend([i, j, m + j, i, m + j, m + i]);
    }
    (vertices, indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::asset_id::AssetId;

    fn no_clips() -> std::collections::HashMap<AssetId, [f32; 4]> {
        std::collections::HashMap::new()
    }

    fn no_slots() -> HashMap<AssetId, usize> {
        HashMap::new()
    }

    fn sprite(x: f32, y: f32, w: f32, h: f32, tint: [f32; 4]) -> Sprite {
        Sprite {
            asset_id: AssetId::default(),
            x,
            y,
            width: w,
            height: h,
            texture: None,
            tint,
            follow_cursor: false,
            visible: true,
            view: None,
            fit: SpriteFit::Fit,
            corner_radius: 0.0,
        }
    }

    #[test]
    fn no_fonts_means_no_calls() {
        let s = sprite(0.0, 0.0, 100.0, 100.0, [1.0, 0.0, 0.0, 1.0]);
        assert!(build_sprite_calls(&[&s], None, &no_slots(), [0.0, 0.0], &no_clips()).is_empty());
    }

    #[test]
    fn visible_sprite_emits_quad_with_sentinel_uv() {
        let s = sprite(10.0, 20.0, 100.0, 50.0, [0.5, 0.5, 0.5, 0.75]);
        let calls = build_sprite_calls(&[&s], Some(0), &no_slots(), [0.0, 0.0], &no_clips());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].vertices.len(), 4);
        assert_eq!(calls[0].indices, vec![0, 1, 2, 0, 2, 3]);
        for v in &calls[0].vertices {
            assert!(v.uv[0] < 0.0, "sentinel u should be negative");
            assert!((v.uv[1] - 0.75).abs() < 1e-5, "alpha carried in v");
            assert_eq!(v.color, [0.5, 0.5, 0.5]);
        }
        assert_eq!(calls[0].vertices[0].pos, [10.0, 20.0]);
        assert_eq!(calls[0].vertices[2].pos, [110.0, 70.0]);
    }

    #[test]
    fn textured_sprite_emits_real_uvs_and_its_slot() {
        let mut s = sprite(10.0, 20.0, 100.0, 50.0, [1.0, 0.9, 0.8, 0.75]);
        s.texture = Some(AssetId(42));
        let mut slots = no_slots();
        slots.insert(AssetId(42), 3);
        let calls = build_sprite_calls(&[&s], Some(0), &slots, [0.0, 0.0], &no_clips());
        assert_eq!(calls.len(), 1);
        // The call binds the sprite texture's atlas slot, not the font's.
        assert_eq!(calls[0].atlas_slot, 3);
        let vs = &calls[0].vertices;
        assert_eq!(vs[0].uv, [0.0, 0.0]);
        assert_eq!(vs[1].uv, [1.0, 0.0]);
        assert_eq!(vs[2].uv, [1.0, 1.0]);
        assert_eq!(vs[3].uv, [0.0, 1.0]);
        for v in vs {
            // Tint in color, alpha in the mode flag (> 0 = textured).
            assert_eq!(v.color, [1.0, 0.9, 0.8]);
            assert!((v.mode - 0.75).abs() < 1e-5);
        }
    }

    #[test]
    fn rounded_sprite_tessellates_with_a_feathered_edge() {
        let mut s = sprite(100.0, 100.0, 400.0, 200.0, [0.1, 0.2, 0.3, 0.9]);
        s.corner_radius = 20.0;
        let calls = build_sprite_calls(&[&s], Some(0), &no_slots(), [0.0, 0.0], &no_clips());
        assert_eq!(calls.len(), 1);
        let vs = &calls[0].vertices;
        // An inner solid ring and an outer transparent ring, 4 corner arcs of
        // CORNER_SEGMENTS + 1 points each.
        let ring = 4 * (CORNER_SEGMENTS + 1);
        assert_eq!(vs.len(), 2 * ring);
        for v in &vs[..ring] {
            assert!(
                (v.uv[1] - 0.9).abs() < 1e-5,
                "inner ring carries the tint alpha"
            );
        }
        for v in &vs[ring..] {
            assert!(v.uv[1].abs() < 1e-5, "outer ring fades to transparent");
        }
        // Every vertex stays inside the authored rect, and the outer ring
        // reaches the rect edges at the flat sides.
        for v in vs {
            assert!(v.pos[0] >= 100.0 - 1e-3 && v.pos[0] <= 500.0 + 1e-3);
            assert!(v.pos[1] >= 100.0 - 1e-3 && v.pos[1] <= 300.0 + 1e-3);
        }
        let min_x = vs.iter().map(|v| v.pos[0]).fold(f32::MAX, f32::min);
        assert!((min_x - 100.0).abs() < 1e-3);
        // The corner point itself is never touched: the arc cuts it off.
        assert!(!vs.iter().any(|v| v.pos == [100.0, 100.0]));
    }

    #[test]
    fn rounded_view_sprite_scales_its_radius_with_the_window() {
        // A 2x window doubles the radius: the outer ring's leftmost point
        // sits at the transformed rect's left edge, and the top-left corner
        // arc starts (2 * radius) transformed pixels down from the rect top.
        let mut s = sprite(100.0, 100.0, 400.0, 200.0, [0.1, 0.2, 0.3, 0.9]);
        s.view = Some(AssetId(7));
        s.corner_radius = 20.0;
        let calls = build_sprite_calls(
            &[&s],
            Some(0),
            &no_slots(),
            [2.0 * UI_REFERENCE_SIZE[0], 2.0 * UI_REFERENCE_SIZE[1]],
            &no_clips(),
        );
        let vs = &calls[0].vertices;
        let min_x = vs.iter().map(|v| v.pos[0]).fold(f32::MAX, f32::min);
        let top_left_arc_y = vs
            .iter()
            .filter(|v| (v.pos[0] - min_x).abs() < 1e-3)
            .map(|v| v.pos[1])
            .fold(f32::MAX, f32::min);
        assert!((min_x - 200.0).abs() < 1e-3);
        assert!((top_left_arc_y - (200.0 + 40.0)).abs() < 1e-3);
    }

    #[test]
    fn textured_sprite_without_a_loaded_texture_falls_back_to_fill() {
        let mut s = sprite(0.0, 0.0, 10.0, 10.0, [0.2, 0.3, 0.4, 1.0]);
        s.texture = Some(AssetId(42));
        // The texture never made it into the atlas pool: solid-fill sentinel.
        let calls = build_sprite_calls(&[&s], Some(5), &no_slots(), [0.0, 0.0], &no_clips());
        assert_eq!(calls[0].atlas_slot, 5);
        assert!(calls[0].vertices[0].uv[0] < 0.0);
        assert_eq!(calls[0].vertices[0].mode, 0.0);
    }

    #[test]
    fn invisible_sprite_is_skipped() {
        let mut s = sprite(0.0, 0.0, 100.0, 100.0, [1.0, 1.0, 1.0, 1.0]);
        s.visible = false;
        assert!(
            build_sprite_calls(&[&s], Some(0), &no_slots(), [0.0, 0.0], &no_clips()).is_empty()
        );
    }

    #[test]
    fn zero_alpha_sprite_is_skipped() {
        let s = sprite(0.0, 0.0, 100.0, 100.0, [1.0, 1.0, 1.0, 0.0]);
        assert!(
            build_sprite_calls(&[&s], Some(0), &no_slots(), [0.0, 0.0], &no_clips()).is_empty()
        );
    }

    #[test]
    fn view_owned_sprite_scales_to_window() {
        // A view-owned (overlay) sprite is authored in the reference canvas and
        // uniformly scaled onto the window. At twice the reference size the
        // rect doubles and stays centered.
        let mut s = sprite(100.0, 100.0, 200.0, 100.0, [1.0, 1.0, 1.0, 1.0]);
        s.view = Some(AssetId(7));
        let calls = build_sprite_calls(&[&s], Some(0), &no_slots(), [2560.0, 1440.0], &no_clips());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].vertices[0].pos, [200.0, 200.0]);
        assert_eq!(calls[0].vertices[2].pos, [600.0, 400.0]);
    }

    #[test]
    fn view_owned_full_canvas_backdrop_fills_window() {
        // A view-owned sprite spanning the whole reference canvas is a
        // full-screen backdrop: it fills the live window rather than letterboxing.
        let mut s = sprite(0.0, 0.0, 1280.0, 720.0, [0.0, 0.0, 0.0, 0.5]);
        s.view = Some(AssetId(7));
        let calls = build_sprite_calls(&[&s], Some(0), &no_slots(), [2560.0, 1440.0], &no_clips());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].vertices[0].pos, [0.0, 0.0]);
        assert_eq!(calls[0].vertices[2].pos, [2560.0, 1440.0]);
    }

    #[test]
    fn cover_sprite_fills_the_window_and_crops_the_overflow() {
        // On a 4:3 window the 16:9 canvas covers by the height ratio
        // (768/720): vertical edges land exactly on the window edges,
        // horizontal overflow is cropped equally on both sides.
        let mut s = sprite(0.0, 0.0, 1280.0, 720.0, [1.0, 1.0, 1.0, 1.0]);
        s.view = Some(AssetId(7));
        s.fit = SpriteFit::Cover;
        let calls = build_sprite_calls(&[&s], Some(0), &no_slots(), [1024.0, 768.0], &no_clips());
        let scale = 768.0 / 720.0;
        let overflow = (1280.0 * scale - 1024.0) / 2.0;
        let vs = &calls[0].vertices;
        assert!(
            (vs[0].pos[0] - -overflow).abs() < 1e-3,
            "x0={}",
            vs[0].pos[0]
        );
        assert!((vs[0].pos[1]).abs() < 1e-3, "y0={}", vs[0].pos[1]);
        assert!(
            (vs[2].pos[0] - (1024.0 + overflow)).abs() < 1e-3,
            "x1={}",
            vs[2].pos[0]
        );
        assert!((vs[2].pos[1] - 768.0).abs() < 1e-3, "y1={}", vs[2].pos[1]);
    }

    #[test]
    fn cover_sprite_anchored_to_the_canvas_bottom_stays_flush() {
        // A bottom-anchored partial-canvas sprite (a character portrait): its
        // bottom edge maps exactly to the window bottom on a window taller
        // than the reference aspect.
        let mut s = sprite(400.0, 100.0, 480.0, 620.0, [1.0, 1.0, 1.0, 1.0]);
        s.view = Some(AssetId(7));
        s.fit = SpriteFit::Cover;
        let calls = build_sprite_calls(&[&s], Some(0), &no_slots(), [1024.0, 768.0], &no_clips());
        let bottom = calls[0].vertices[2].pos[1];
        assert!((bottom - 768.0).abs() < 1e-3, "bottom={bottom}");
    }

    #[test]
    fn view_less_sprite_keeps_literal_pixels() {
        // A HUD / scene sprite (view == None) is never overlay-scaled.
        let s = sprite(10.0, 20.0, 100.0, 50.0, [0.5, 0.5, 0.5, 1.0]);
        let calls = build_sprite_calls(&[&s], Some(0), &no_slots(), [2560.0, 1440.0], &no_clips());
        assert_eq!(calls[0].vertices[0].pos, [10.0, 20.0]);
        assert_eq!(calls[0].vertices[2].pos, [110.0, 70.0]);
    }

    #[test]
    fn clipped_element_carries_window_space_clip_rect() {
        // A view-owned sprite whose id is in the clips map gets a clip_rect
        // mapped from the reference-space band through the overlay; one not in
        // the map stays unclipped.
        let mut s = sprite(100.0, 100.0, 50.0, 50.0, [1.0, 1.0, 1.0, 1.0]);
        s.asset_id = AssetId(7);
        s.view = Some(AssetId(1));
        let mut clips = no_clips();
        // Reference band [200,200] size [200,60] at a 2x viewport (1280x720 ->
        // 2560x1440, scale 2 about the centre): forward(200,200)=(400,400),
        // forward(400,260)=(800,520) -> clip [400,400,400,120].
        clips.insert(AssetId(7), [200.0, 200.0, 200.0, 60.0]);
        let calls = build_sprite_calls(&[&s], Some(0), &no_slots(), [2560.0, 1440.0], &clips);
        let clip = calls[0].clip_rect.expect("clipped sprite has a clip rect");
        assert!((clip[0] - 400.0).abs() < 1e-3, "x={}", clip[0]);
        assert!((clip[1] - 400.0).abs() < 1e-3, "y={}", clip[1]);
        assert!((clip[2] - 400.0).abs() < 1e-3, "w={}", clip[2]);
        assert!((clip[3] - 120.0).abs() < 1e-3, "h={}", clip[3]);

        // A sprite not in the clips map is unclipped.
        let mut other = sprite(0.0, 0.0, 10.0, 10.0, [1.0, 1.0, 1.0, 1.0]);
        other.asset_id = AssetId(9);
        other.view = Some(AssetId(1));
        let calls = build_sprite_calls(&[&other], Some(0), &no_slots(), [2560.0, 1440.0], &clips);
        assert!(calls[0].clip_rect.is_none());
    }
}
