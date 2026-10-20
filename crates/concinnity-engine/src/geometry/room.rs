// Room interior geometry.
//
// Each face is two CCW triangles. Vertex color distinguishes surfaces:
// warm dark gray floor, light gray ceiling, slightly different grays per wall.
// UV coordinates tile at one repeat per meter.

use alloc::vec::Vec;
use concinnity_core::math::SaturatingFrom;

use super::Vert;
use crate::ecs::{PayloadLocator, TextureHandle};

/// Build room geometry from explicit extents.
///
/// Returns `(vertices, indices)` where each vertex is `(pos, normal, color, uv)`.
/// Winding is CCW when viewed from inside. UV coordinates tile at one repeat
/// per meter. Normals point inward so diffuse lighting is correct for a camera
/// inside the room.
pub fn build_room_geometry(
    half_width: f32,
    half_depth: f32,
    floor_y: f32,
    ceiling_y: f32,
) -> (Vec<Vert>, Vec<u16>) {
    let mut verts: Vec<Vert> = Vec::new();
    let mut idxs: Vec<u16> = Vec::new();

    let (xn, xp) = (-half_width, half_width);
    let (yn, yp) = (floor_y, ceiling_y);
    let (zn, zp) = (-half_depth, half_depth);

    let width = half_width * 2.0;
    let depth = half_depth * 2.0;
    let height = yp - yn;

    let mut push_quad = |a: [f32; 3],
                         b: [f32; 3],
                         c: [f32; 3],
                         d: [f32; 3],
                         normal: [f32; 3],
                         color: [f32; 3],
                         uv_a: [f32; 2],
                         uv_b: [f32; 2],
                         uv_c: [f32; 2],
                         uv_d: [f32; 2]| {
        let base = u16::saturating_from(verts.len());
        verts.extend_from_slice(&[
            (a, normal, color, uv_a),
            (b, normal, color, uv_b),
            (c, normal, color, uv_c),
            (d, normal, color, uv_d),
        ]);
        idxs.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
    };

    let floor_color = [0.25, 0.22, 0.20];
    let ceiling_color = [0.70, 0.70, 0.72];
    let wall_n_color = [0.45, 0.45, 0.48];
    let wall_s_color = [0.40, 0.40, 0.43];
    let wall_e_color = [0.50, 0.48, 0.46];
    let wall_w_color = [0.42, 0.44, 0.46];

    // floor: normal points up into the room
    push_quad(
        [xn, yn, zn],
        [xp, yn, zn],
        [xp, yn, zp],
        [xn, yn, zp],
        [0.0, 1.0, 0.0],
        floor_color,
        [0.0, 0.0],
        [width, 0.0],
        [width, depth],
        [0.0, depth],
    );

    // ceiling: normal points down into the room
    push_quad(
        [xn, yp, zp],
        [xp, yp, zp],
        [xp, yp, zn],
        [xn, yp, zn],
        [0.0, -1.0, 0.0],
        ceiling_color,
        [0.0, 0.0],
        [width, 0.0],
        [width, depth],
        [0.0, depth],
    );

    // north wall (+Z face); normal points toward -Z
    push_quad(
        [xp, yn, zp],
        [xn, yn, zp],
        [xn, yp, zp],
        [xp, yp, zp],
        [0.0, 0.0, -1.0],
        wall_n_color,
        [0.0, height],
        [width, height],
        [width, 0.0],
        [0.0, 0.0],
    );

    // south wall (-Z face); normal points toward +Z
    push_quad(
        [xn, yn, zn],
        [xp, yn, zn],
        [xp, yp, zn],
        [xn, yp, zn],
        [0.0, 0.0, 1.0],
        wall_s_color,
        [0.0, height],
        [width, height],
        [width, 0.0],
        [0.0, 0.0],
    );

    // east wall (+X face); normal points toward -X
    push_quad(
        [xp, yn, zn],
        [xp, yn, zp],
        [xp, yp, zp],
        [xp, yp, zn],
        [-1.0, 0.0, 0.0],
        wall_e_color,
        [0.0, height],
        [depth, height],
        [depth, 0.0],
        [0.0, 0.0],
    );

    // west wall (-X face); normal points toward +X
    push_quad(
        [xn, yn, zp],
        [xn, yn, zn],
        [xn, yp, zn],
        [xn, yp, zp],
        [1.0, 0.0, 0.0],
        wall_w_color,
        [0.0, height],
        [depth, height],
        [depth, 0.0],
        [0.0, 0.0],
    );

    (verts, idxs)
}

/// A self-contained room (floor, ceiling, four walls), with optional texturing.
///
/// Prefer `Room` over a [`ProceduralMesh`](#proceduralmesh) (generator `"room"`) +
/// [Prop](#prop) pair for a shorter declaration. The room is placed at the world
/// origin.
///
/// Dimensions can be given as `size: [width, depth, height]` (full extents) or
/// as `half_width`, `half_depth`, and `ceiling_height` individually.
///
/// `texture`, `wall_texture`, `floor_texture`, and `ceiling_texture` are checked
/// in that order; the first set value wins. Generator names such as `"brick"` or
/// `"concrete"` resolve to a matching [Texture](#texture) at build time.
///
/// ```rust
/// # use concinnity_engine::components::cook::Room as RoomArgs;
/// RoomArgs {
///     size: Some([16.0, 20.0, 3.5]),
///     ..Default::default()
/// };
/// ```
#[derive(
    Debug,
    Clone,
    serde::Serialize,
    serde::Deserialize,
    crate::ecs::AssetFields,
    crate::ecs::AssetDefault,
)]
#[serde(default)]
pub struct RoomArgs {
    /// Half the room's width along X, in world units. Ignored when `size` is set.
    #[asset(default = 8.0)]
    pub half_width: f32,
    /// Half the room's depth along Z, in world units. Ignored when `size` is set.
    #[asset(default = 10.0)]
    pub half_depth: f32,
    /// Floor-to-ceiling height in world units. Ignored when `size` is set.
    #[asset(default = 3.5)]
    pub ceiling_height: f32,
    /// Shorthand for the full dimensions `[width, depth, height]`. When set, it
    /// overrides `half_width`, `half_depth`, and `ceiling_height`.
    pub size: Option<[f32; 3]>,
    /// [Texture](#texture) applied to all surfaces. Falls back to `wall_texture`
    /// when unset. Generator names such as `"brick"` or `"concrete"` resolve to
    /// a matching texture at build time.
    pub texture: Option<TextureHandle>,
    /// [Texture](#texture) for the walls. Currently all surfaces share one
    /// texture; per-surface texturing is reserved for a future update.
    pub wall_texture: Option<TextureHandle>,
    /// [Texture](#texture) for the floor (see `wall_texture`).
    pub floor_texture: Option<TextureHandle>,
    /// [Texture](#texture) for the ceiling (see `wall_texture`).
    pub ceiling_texture: Option<TextureHandle>,
    /// Number of level-of-detail versions to generate, including the original.
    /// `1` (the default) generates no alternates.
    #[asset(default = 1)]
    pub lod_levels: u32,
    /// Camera distances at which to switch to each lower-detail version. Empty
    /// lets the build choose defaults.
    pub lod_distances: Vec<f32>,
}

/// The runtime `Room`: the dimensions resolved from
/// [`cook::Room`](crate::components::cook::Room)'s authored fields, and its payload
/// locator.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Room {
    /// Half the room's width in world units.
    pub half_width: f32,
    /// Half the room's depth in world units.
    pub half_depth: f32,
    /// Floor-to-ceiling height in world units.
    pub ceiling_height: f32,
    /// Texture applied to every surface unless a surface overrides it.
    pub texture: Option<TextureHandle>,
    /// Texture for the four walls.
    pub wall_texture: Option<TextureHandle>,
    /// Texture for the floor.
    pub floor_texture: Option<TextureHandle>,
    /// Texture for the ceiling.
    pub ceiling_texture: Option<TextureHandle>,
    /// The generated geometry's place in the blob, injected at load.
    pub locator: Option<PayloadLocator>,
}

impl Room {
    /// Returns the first set texture reference across all texture fields.
    pub fn effective_texture(&self) -> Option<TextureHandle> {
        [
            self.texture,
            self.wall_texture,
            self.floor_texture,
            self.ceiling_texture,
        ]
        .into_iter()
        .flatten()
        .next()
    }
}

impl Room {
    /// Translate the authored args into the runtime room: resolve the `size`
    /// shorthand into half extents. Run by cook at build time (the baked blob
    /// record carries the result).
    #[expect(
        clippy::needless_pass_by_value,
        reason = "the cook registry calls every component's `bake` with owned args"
    )]
    pub fn bake(args: RoomArgs) -> Self {
        let (half_width, half_depth, ceiling_height) = if let Some([w, d, h]) = args.size {
            (w / 2.0, d / 2.0, h)
        } else {
            (args.half_width, args.half_depth, args.ceiling_height)
        };
        Self {
            half_width,
            half_depth,
            ceiling_height,
            texture: args.texture,
            wall_texture: args.wall_texture,
            floor_texture: args.floor_texture,
            ceiling_texture: args.ceiling_texture,
            locator: None,
        }
    }
}

#[cfg(test)]
mod runtime_tests {
    use super::*;

    #[test]
    fn effective_texture_returns_texture_field_first() {
        let room = Room {
            half_width: 8.0,
            half_depth: 10.0,
            ceiling_height: 3.5,
            texture: Some(TextureHandle::new(1)),
            wall_texture: Some(TextureHandle::new(2)),
            floor_texture: None,
            ceiling_texture: None,
            locator: None,
        };
        assert_eq!(room.effective_texture(), Some(TextureHandle::new(1)));
    }

    #[test]
    fn effective_texture_falls_back_to_wall_texture() {
        let room = Room {
            half_width: 8.0,
            half_depth: 10.0,
            ceiling_height: 3.5,
            texture: None,
            wall_texture: Some(TextureHandle::new(7)),
            floor_texture: None,
            ceiling_texture: None,
            locator: None,
        };
        assert_eq!(room.effective_texture(), Some(TextureHandle::new(7)));
    }

    #[test]
    fn effective_texture_returns_none_when_all_unset() {
        let room = Room::bake(RoomArgs::default());
        assert_eq!(room.effective_texture(), None);
    }

    #[test]
    fn from_args_resolves_size_shorthand() {
        let args = RoomArgs {
            size: Some([16.0, 20.0, 3.5]),
            ..RoomArgs::default()
        };
        let room = Room::bake(args);
        assert_eq!(room.half_width, 8.0);
        assert_eq!(room.half_depth, 10.0);
        assert_eq!(room.ceiling_height, 3.5);
    }

    #[test]
    fn from_args_uses_explicit_half_extents_when_no_size() {
        let args = RoomArgs {
            half_width: 5.0,
            half_depth: 7.0,
            ceiling_height: 4.0,
            size: None,
            ..RoomArgs::default()
        };
        let room = Room::bake(args);
        assert_eq!(room.half_width, 5.0);
        assert_eq!(room.half_depth, 7.0);
        assert_eq!(room.ceiling_height, 4.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::math::vec3::{dot, length};

    fn bounds(verts: &[Vert]) -> ([f32; 3], [f32; 3]) {
        let mut mn = [f32::INFINITY; 3];
        let mut mx = [f32::NEG_INFINITY; 3];
        for (pos, ..) in verts {
            for k in 0..3 {
                mn[k] = mn[k].min(pos[k]);
                mx[k] = mx[k].max(pos[k]);
            }
        }
        (mn, mx)
    }

    #[test]
    fn room_geometry_is_six_quads_spanning_the_requested_extents() {
        let (verts, idxs) = build_room_geometry(3.0, 4.0, 0.0, 2.5);
        assert_eq!(verts.len(), 6 * 4);
        assert_eq!(idxs.len(), 6 * 6);
        assert!(idxs.iter().all(|&i| (i as usize) < verts.len()));
        let (mn, mx) = bounds(&verts);
        assert_eq!(mn, [-3.0, 0.0, -4.0]);
        assert_eq!(mx, [3.0, 2.5, 4.0]);
    }

    #[test]
    fn room_normals_point_into_the_interior() {
        // Every face's normal is a unit axis pointing at the room center, which
        // for an origin-centered box means normal . position is negative.
        let (verts, _) = build_room_geometry(3.0, 4.0, -1.0, 1.0);
        for (pos, normal, ..) in &verts {
            let len = length(*normal);
            assert!((len - 1.0).abs() < 1e-6, "normal {normal:?} is not unit");
            let dot = dot(*pos, *normal);
            assert!(dot < 0.0, "normal {normal:?} is not inward at {pos:?}");
        }
    }

    #[test]
    fn room_uvs_tile_once_per_meter() {
        let (verts, _) = build_room_geometry(3.0, 4.0, 0.0, 2.5);
        // The floor quad is emitted first and tiles across the full 6x8 extent.
        let floor_uvs: Vec<[f32; 2]> = verts[..4].iter().map(|v| v.3).collect();
        assert_eq!(
            floor_uvs,
            alloc::vec![[0.0, 0.0], [6.0, 0.0], [6.0, 8.0], [0.0, 8.0]]
        );
        // The north wall (third quad) tiles width by ceiling height.
        let wall_uvs: Vec<[f32; 2]> = verts[8..12].iter().map(|v| v.3).collect();
        assert_eq!(
            wall_uvs,
            alloc::vec![[0.0, 2.5], [6.0, 2.5], [6.0, 0.0], [0.0, 0.0]]
        );
    }
}
