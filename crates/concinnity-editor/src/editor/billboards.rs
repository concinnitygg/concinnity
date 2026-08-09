// src/editor/billboards.rs
//
// Editor-only viewport billboards: a constant screen-size icon over every
// authored asset that has a world position but no rendered geometry (lights,
// trigger volumes, probes, cameras), so those assets can be seen, picked, and
// moved like anything else. This module is the pure half: the registry-driven
// eligibility, the glyph / tint derivation, the projection and hit-test math,
// the trigger-volume outline layout, and the injected sprite / label pools.
// The hook (`hook/billboard_drive.rs`) resolves entries to live entities and
// drives the per-frame placement.
//
// Occlusion: billboards live in the window-space overlay, so they always draw
// on top of the 3D scene (no depth test against it) and under the floating
// panels. Always-on-top is the v1 choice; a depth-aware fade would need the
// scene depth buffer on the CPU or a shader change, neither of which an icon
// earns yet.

use super::outlines::shapes::{BOX_EDGES, EDGES};
use super::registry::ID_BASE;
use super::theme;
use super::widget;
use crate::assets::Sprite;
use crate::ecs::World;
use crate::ecs::asset_id::AssetId;
use concinnity_world::registry::ComponentType;

// Reserved id family: the next free block after the panel families below 0x1000.
// Icons at +0x00, their glyph labels at +0x40, and the drag ghost's dotted box
// run at +0x100 (0x10 per edge).
const BILLBOARD_BASE: u32 = ID_BASE + 0x1000;

// Icon pool size, bounding the per-frame sprite cost; entries past the pool
// simply draw no icon (they stay pickable through the Assets tree).
pub(crate) const MAX_BILLBOARDS: usize = 48;

const ICON_PX: f32 = 18.0;
const ICON_RADIUS: f32 = 5.0;
const ICON_BORDER_W: f32 = 1.5;
// Slop beyond the icon's half-extent for the press hit test.
pub(crate) const PICK_RADIUS_PX: f32 = ICON_PX * 0.5 + 4.0;
// The glyph draws a step smaller than panel text to fit the chip.
const GLYPH_SCALE: f32 = 0.6;
const GLYPH_HALF: f32 = 10.0 * GLYPH_SCALE;

const ICON_TINT: [f32; 4] = [0.08, 0.08, 0.10, 0.85];
// Non-active selected icons ring in the same dimmed accent as the mesh
// selection's non-active rings.
const MEMBER_TINT: [f32; 4] = [0.20, 0.30, 0.45, 1.0];

// The dotted screen-space box outline: a dotted run per box edge. Its one
// tenant is the drag-out placement ghost; entity extents draw through the
// renderer's line pass instead (`editor/outlines`), whose box topology
// (`BOX_EDGES` / `EDGES`) this shares.
pub(crate) const EDGE_SEGMENTS: usize = 6;
const SEGMENT_PX: f32 = 3.0;

fn icon_id(i: usize) -> AssetId {
    AssetId(BILLBOARD_BASE + i as u32)
}

fn glyph_id(i: usize) -> AssetId {
    AssetId(BILLBOARD_BASE + 0x40 + i as u32)
}

fn box_segment_id(edge: usize, seg: usize) -> AssetId {
    AssetId(BILLBOARD_BASE + 0x100 + edge as u32 * 0x10 + seg as u32)
}

pub(crate) fn all_sprite_ids() -> Vec<AssetId> {
    let mut out: Vec<AssetId> = (0..MAX_BILLBOARDS).map(icon_id).collect();
    for edge in 0..BOX_EDGES {
        for seg in 0..EDGE_SEGMENTS {
            out.push(box_segment_id(edge, seg));
        }
    }
    out
}

pub(crate) fn all_label_ids() -> Vec<AssetId> {
    (0..MAX_BILLBOARDS).map(glyph_id).collect()
}

// Billboard-eligible: a registered component type that never renders (the
// registry's `renders` flag) but declares a world position in its authored
// args. Derived from the registry metadata, so a new asset type earns an icon
// by declaring those, not by editing a list here. Memoized per type: the
// answer is fixed for the process, and the default-args probe behind it
// serializes a whole default struct, too heavy for a per-entry per-frame call.
pub(crate) fn eligible(ty: &str) -> bool {
    thread_local! {
        static CACHE: std::cell::RefCell<std::collections::HashMap<String, bool>> =
            std::cell::RefCell::new(std::collections::HashMap::new());
    }
    CACHE.with(|c| {
        if let Some(&known) = c.borrow().get(ty) {
            return known;
        }
        let fresh = ComponentType::parse(ty).is_some_and(|ct| {
            !ct.renders() && position_of(&super::form::working_args(ty, None)).is_some()
        });
        c.borrow_mut().insert(ty.to_string(), fresh);
        fresh
    })
}

// The asset's authored world position: a 3-number `position` (or, for panel
// lights, `centre`) arg.
pub(crate) fn position_of(args: &serde_json::Map<String, serde_json::Value>) -> Option<[f32; 3]> {
    ["position", "centre"]
        .into_iter()
        .find_map(|k| vec3(args, k))
}

// A 3-number array arg as an [f32; 3], if present and well formed.
pub(crate) fn vec3(
    args: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Option<[f32; 3]> {
    let a = args.get(key)?.as_array()?;
    if a.len() != 3 {
        return None;
    }
    let mut out = [0.0f32; 3];
    for (o, v) in out.iter_mut().zip(a) {
        *o = v.as_f64()? as f32;
    }
    Some(out)
}

// The icon's glyph: the type name's capitals and digits (PointLight -> "PL",
// Camera3D -> "C3"), so every type reads distinctly without a glyph table.
pub(crate) fn glyph(ty: &str) -> String {
    let caps: String = ty
        .chars()
        .filter(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
        .take(2)
        .collect();
    if caps.is_empty() {
        ty.chars().take(1).collect::<String>().to_ascii_uppercase()
    } else {
        caps
    }
}

// A stable per-type hue from an FNV-1a hash of the type name, shared with the
// extent outlines so an entity's wireframe matches its icon.
pub(crate) fn hue_deg(ty: &str) -> f32 {
    let mut h: u32 = 0x811c_9dc5;
    for b in ty.bytes() {
        h ^= u32::from(b);
        h = h.wrapping_mul(0x0100_0193);
    }
    (h % 360) as f32
}

// A stable per-type accent: the hashed hue at fixed saturation / value, so
// colours never need registering either.
pub(crate) fn tint(ty: &str) -> [f32; 4] {
    let [r, g, b] = hsv_to_rgb(hue_deg(ty), 0.55, 0.95);
    [r, g, b, 1.0]
}

pub(crate) fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match (h / 60.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [r + m, g + m, b + m]
}

// Project a world point to window space, returning its screen position and
// view-space depth. The same projection (and view-matrix convention) as the
// gizmo's layout; `None` at or behind the camera plane or with a degenerate
// viewport.
pub(crate) fn project(
    view: &[[f32; 4]; 4],
    fov_y_radians: f32,
    viewport: [f32; 2],
    p: [f32; 3],
) -> Option<([f32; 2], f32)> {
    let (vw, vh) = (viewport[0], viewport[1]);
    if vw <= 0.0 || vh <= 0.0 {
        return None;
    }
    let tan_half = (fov_y_radians * 0.5).tan();
    if tan_half <= 0.0 || !tan_half.is_finite() {
        return None;
    }
    let v = [
        view[0][0] * p[0] + view[1][0] * p[1] + view[2][0] * p[2] + view[3][0],
        view[0][1] * p[0] + view[1][1] * p[1] + view[2][1] * p[2] + view[3][1],
        view[0][2] * p[0] + view[1][2] * p[1] + view[2][2] * p[2] + view[3][2],
    ];
    let depth = -v[2];
    if depth <= 1e-4 {
        return None;
    }
    let aspect = vw / vh;
    Some((
        [
            (v[0] / (depth * tan_half * aspect) + 1.0) * 0.5 * vw,
            (1.0 - v[1] / (depth * tan_half)) * 0.5 * vh,
        ],
        depth,
    ))
}

// The billboard under the mouse: within the pick radius, nearest camera
// distance first (overlapping icons resolve like overlapping meshes). Spots
// are `(screen, camera distance)` pairs; returns the winning index.
pub(crate) fn hit(spots: &[([f32; 2], f32)], mouse: [f32; 2]) -> Option<usize> {
    spots
        .iter()
        .enumerate()
        .filter(|(_, (screen, _))| {
            (screen[0] - mouse[0]).abs() <= PICK_RADIUS_PX
                && (screen[1] - mouse[1]).abs() <= PICK_RADIUS_PX
        })
        .min_by(|a, b| a.1.1.total_cmp(&b.1.1))
        .map(|(i, _)| i)
}

// Whether a billboard at camera distance `dist` beats the nearest mesh hit
// (both measured along the pick ray from the camera). Ties go to the
// billboard: it is the smaller target.
pub(crate) fn beats_mesh(dist: f32, mesh_t: Option<f32>) -> bool {
    mesh_t.is_none_or(|t| dist <= t)
}

// The dotted-outline segment centers of an oriented box: the 8 corners of
// `half_extents` through `model` (column-major), `EDGE_SEGMENTS` dots per
// edge. `None` when any corner is at or behind the camera plane (only
// possible with the camera against the box).
pub(crate) fn box_outline(
    view: &[[f32; 4]; 4],
    fov_y_radians: f32,
    viewport: [f32; 2],
    model: &[[f32; 4]; 4],
    half_extents: [f32; 3],
) -> Option<Vec<[f32; 2]>> {
    let world_corners = super::outlines::shapes::box_corners(model, half_extents);
    let mut corners = [[0.0f32; 2]; 8];
    for (corner, world) in corners.iter_mut().zip(world_corners) {
        *corner = project(view, fov_y_radians, viewport, world)?.0;
    }
    let mut out = Vec::with_capacity(BOX_EDGES * EDGE_SEGMENTS);
    for (a, b) in EDGES {
        for seg in 0..EDGE_SEGMENTS {
            // Dots span the edge inclusive of both ends, so adjacent edges
            // meet at shared corners.
            let f = seg as f32 / (EDGE_SEGMENTS - 1) as f32;
            out.push([
                corners[a][0] + (corners[b][0] - corners[a][0]) * f,
                corners[a][1] + (corners[b][1] - corners[a][1]) * f,
            ]);
        }
    }
    Some(out)
}

// One placed icon for the frame.
pub(crate) struct Icon {
    pub screen: [f32; 2],
    pub tint: [f32; 4],
    pub glyph: String,
    pub selected: bool,
    pub active: bool,
}

// The injected pools, all hidden: icon chips with their glyph labels and the
// box-outline dot run. Per-frame placement recolours the mutable parts.
pub(crate) fn sprites() -> Vec<Sprite> {
    let mut out: Vec<Sprite> = (0..MAX_BILLBOARDS)
        .map(|i| Sprite {
            asset_id: icon_id(i),
            tint: ICON_TINT,
            corner_radius: ICON_RADIUS,
            border_width: ICON_BORDER_W,
            visible: false,
            ..Default::default()
        })
        .collect();
    for edge in 0..BOX_EDGES {
        for seg in 0..EDGE_SEGMENTS {
            out.push(Sprite {
                asset_id: box_segment_id(edge, seg),
                visible: false,
                ..Default::default()
            });
        }
    }
    out
}

// Show one icon chip + glyph per entry (icons past the pool are dropped) and
// hide the rest. Selection reads as the ring: the active member in the full
// accent, other members dimmed, unselected icons in their type tint.
pub(crate) fn place_icons(world: &mut World, icons: &[Icon]) {
    for i in 0..MAX_BILLBOARDS {
        let placed = icons.get(i);
        if let Some(s) = sprite_mut(world, icon_id(i)) {
            match placed {
                Some(icon) => {
                    s.x = icon.screen[0] - ICON_PX * 0.5;
                    s.y = icon.screen[1] - ICON_PX * 0.5;
                    s.width = ICON_PX;
                    s.height = ICON_PX;
                    s.border_width = if icon.active { 2.5 } else { ICON_BORDER_W };
                    s.border_color = if icon.active {
                        theme::ACCENT_TINT
                    } else if icon.selected {
                        MEMBER_TINT
                    } else {
                        icon.tint
                    };
                    s.visible = true;
                }
                None => s.visible = false,
            }
        }
        match placed {
            Some(icon) => {
                if let Some(l) = widget::label_mut(world, glyph_id(i)) {
                    l.content = icon.glyph.clone();
                    l.x = icon.screen[0];
                    l.y = icon.screen[1] - GLYPH_HALF;
                    l.scale = GLYPH_SCALE;
                    l.color = theme::LABEL;
                    l.visible = true;
                }
            }
            None => widget::set_label_visible(world, glyph_id(i), false),
        }
    }
}

// Lay the dotted box outline over its projected segment centers, tinted with
// the owning type's accent; unused segments hide.
pub(crate) fn place_box_outline(world: &mut World, centers: &[[f32; 2]], tint: [f32; 4]) {
    let mut idx = 0;
    for edge in 0..BOX_EDGES {
        for seg in 0..EDGE_SEGMENTS {
            if let Some(s) = sprite_mut(world, box_segment_id(edge, seg)) {
                match centers.get(idx) {
                    Some(c) => {
                        s.x = c[0] - SEGMENT_PX * 0.5;
                        s.y = c[1] - SEGMENT_PX * 0.5;
                        s.width = SEGMENT_PX;
                        s.height = SEGMENT_PX;
                        s.tint = tint;
                        s.visible = true;
                    }
                    None => s.visible = false,
                }
            }
            idx += 1;
        }
    }
}

pub(crate) fn hide_outline(world: &mut World) {
    place_box_outline(world, &[], [0.0; 4]);
}

pub(crate) fn hide(world: &mut World) {
    for id in all_sprite_ids() {
        if let Some(s) = sprite_mut(world, id) {
            s.visible = false;
        }
    }
    for id in all_label_ids() {
        widget::set_label_visible(world, id, false);
    }
}

fn sprite_mut(world: &mut World, id: AssetId) -> Option<&mut Sprite> {
    world.query_mut::<Sprite>().find(|s| s.asset_id == id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::gfx::camera::view_matrix;

    const VP: [f32; 2] = [1280.0, 720.0];
    const FOV: f32 = core::f32::consts::FRAC_PI_2;

    #[test]
    fn eligibility_follows_the_registry_metadata() {
        // Positioned non-rendering types earn an icon.
        for ty in [
            "PointLight",
            "SpotLight",
            "RectAreaLight",
            "TriggerVolume",
            "ReflectionProbe",
            "Camera3D",
        ] {
            assert!(eligible(ty), "{ty} is billboard-eligible");
        }
        // Rendering types are covered by the mesh pick; positionless types
        // have nowhere to draw; unknown strings are not component types.
        for ty in ["Prop", "Sprite", "DirectionalLight", "Spawner", "NotAType"] {
            assert!(!eligible(ty), "{ty} is not billboard-eligible");
        }
    }

    #[test]
    fn position_reads_position_or_centre() {
        let args = |json: serde_json::Value| json.as_object().unwrap().clone();
        assert_eq!(
            position_of(&args(serde_json::json!({"position": [1, 2, 3]}))),
            Some([1.0, 2.0, 3.0])
        );
        assert_eq!(
            position_of(&args(serde_json::json!({"centre": [4.0, 5.0, 6.0]}))),
            Some([4.0, 5.0, 6.0])
        );
        assert_eq!(position_of(&args(serde_json::json!({"radius": 2.0}))), None);
        assert_eq!(
            position_of(&args(serde_json::json!({"position": [1, 2]}))),
            None,
            "a malformed vec is not a position"
        );
    }

    #[test]
    fn glyphs_derive_from_the_type_name() {
        assert_eq!(glyph("PointLight"), "PL");
        assert_eq!(glyph("TriggerVolume"), "TV");
        assert_eq!(glyph("Camera3D"), "C3");
        assert_eq!(glyph("RectAreaLight"), "RA");
        assert_eq!(glyph("lowercase"), "L", "caseless names fall back");
    }

    #[test]
    fn tints_are_opaque_and_distinct_across_the_eligible_set() {
        let types = [
            "PointLight",
            "SpotLight",
            "RectAreaLight",
            "TriggerVolume",
            "ReflectionProbe",
            "Camera3D",
        ];
        let tints: Vec<[f32; 4]> = types.iter().map(|t| tint(t)).collect();
        for (i, a) in tints.iter().enumerate() {
            assert_eq!(a[3], 1.0, "{} tint is opaque", types[i]);
            for (j, b) in tints.iter().enumerate().skip(i + 1) {
                assert_ne!(a, b, "{} and {} share a tint", types[i], types[j]);
            }
        }
    }

    #[test]
    fn project_centers_a_point_ahead_and_rejects_behind() {
        let view = view_matrix([0.0; 3], 0.0, 0.0);
        let (screen, depth) = project(&view, FOV, VP, [0.0, 0.0, -5.0]).unwrap();
        assert!((screen[0] - 640.0).abs() < 0.5);
        assert!((screen[1] - 360.0).abs() < 0.5);
        assert!((depth - 5.0).abs() < 1e-3);
        assert_eq!(project(&view, FOV, VP, [0.0, 0.0, 5.0]), None);
        assert_eq!(project(&view, FOV, [0.0, 720.0], [0.0, 0.0, -5.0]), None);
    }

    #[test]
    fn hit_takes_the_nearest_icon_within_the_radius() {
        let spots = [([100.0, 100.0], 10.0), ([104.0, 100.0], 5.0)];
        // Both icons cover the cursor: the nearer (second) one wins.
        assert_eq!(hit(&spots, [102.0, 100.0]), Some(1));
        // Outside every icon's radius: no hit.
        assert_eq!(hit(&spots, [100.0, 200.0]), None);
        // Only the first icon covers a cursor at its far edge.
        assert_eq!(hit(&spots, [100.0 - PICK_RADIUS_PX, 100.0]), Some(0));
    }

    #[test]
    fn billboard_vs_mesh_prefers_the_nearer_hit() {
        assert!(beats_mesh(5.0, None), "no mesh under the cursor");
        assert!(beats_mesh(5.0, Some(8.0)), "billboard in front");
        assert!(!beats_mesh(5.0, Some(2.0)), "mesh in front");
        assert!(beats_mesh(5.0, Some(5.0)), "ties go to the smaller target");
    }

    #[test]
    fn box_outline_covers_every_edge_and_rejects_behind() {
        let view = view_matrix([0.0; 3], 0.0, 0.0);
        let model = crate::assets::Transform {
            position: [0.0, 0.0, -10.0],
            ..Default::default()
        }
        .model_matrix();
        let centers = box_outline(&view, FOV, VP, &model, [1.0, 1.0, 1.0]).unwrap();
        assert_eq!(centers.len(), BOX_EDGES * EDGE_SEGMENTS);
        // Every dot projects inside the box's screen footprint around center.
        for c in &centers {
            assert!((c[0] - 640.0).abs() < 100.0, "{c:?}");
            assert!((c[1] - 360.0).abs() < 100.0, "{c:?}");
        }
        let behind = crate::assets::Transform {
            position: [0.0, 0.0, 10.0],
            ..Default::default()
        }
        .model_matrix();
        assert_eq!(box_outline(&view, FOV, VP, &behind, [1.0; 3]), None);
    }

    #[test]
    fn id_family_is_contiguous_and_unique() {
        let sprites = all_sprite_ids();
        let labels = all_label_ids();
        assert_eq!(sprites.len(), MAX_BILLBOARDS + BOX_EDGES * EDGE_SEGMENTS);
        assert_eq!(labels.len(), MAX_BILLBOARDS);
        let unique: std::collections::HashSet<_> = sprites.iter().chain(labels.iter()).collect();
        assert_eq!(unique.len(), sprites.len() + labels.len());
    }
}
