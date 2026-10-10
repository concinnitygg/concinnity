//! Backend-agnostic resolution of a `Grass` field into what the grass pass
//! dispatches and draws: where the blades grow, what each blade looks like, and
//! the wind they sway in, turned into the per-frame [`GrassParams`] every grass
//! stage reads. Pure CPU; the backend owns the buffers and pipelines.
//!
//! Where blades grow ([`GrassPatch`]) and what they look like
//! ([`GrassBladeLook`]) are separate records, so the look can be grown over any
//! placement the pass learns to read.

pub mod tiles;

use crate::components::{Grass, Wind};
use crate::gfx::frustum::Frustum;
use crate::render::uniforms::grass::{GRASS_ARGS_SLOTS, GrassParams};
use crate::render::wind::WindField;
use tiles::{GRASS_DRAW_DISTANCE, GRASS_TILE_SIZE, GrassGrid, GroundRect, TileRange};

/// The flat rectangle a field grows on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrassPatch {
    /// World-space center; its height is the ground the blades root at.
    pub center: [f32; 3],
    /// Half-width and half-depth `[x, z]`, in meters.
    pub extent: [f32; 2],
}

impl GrassPatch {
    /// The patch's footprint on the ground plane.
    pub fn rect(&self) -> GroundRect {
        GroundRect::centered([self.center[0], self.center[2]], self.extent)
    }
}

/// What every blade of a field looks like, wherever it grows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrassBladeLook {
    /// Mean height, in meters.
    pub height: f32,
    /// Height variation as a fraction of `height`.
    pub height_variance: f32,
    /// Width at the root, in meters.
    pub width: f32,
    /// Clump diameter, in meters.
    pub clump_size: f32,
    /// Resistance to bending, in [0, 1].
    pub stiffness: f32,
    /// Linear RGB at the root.
    pub root_color: [f32; 3],
    /// Linear RGB at the tip.
    pub tip_color: [f32; 3],
    /// Hue and brightness variation, in [0, 1].
    pub color_variation: f32,
}

/// The camera a frame's grass is placed and culled for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrassCamera {
    /// World-space camera position.
    pub position: [f32; 3],
    /// This frame's unjittered view-projection, column-major.
    pub vp: [[f32; 4]; 4],
}

/// A drawable grass field: its patch, blade look, grid and wind, resolved once
/// from the world's assets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrassField {
    /// Where the blades grow.
    pub patch: GrassPatch,
    /// What each blade looks like.
    pub look: GrassBladeLook,
    /// The cell grid the density resolves to.
    pub grid: GrassGrid,
    /// The wind the blades sway in.
    pub wind: WindField,
    /// Blades the visible-blade buffer is sized for.
    pub capacity: u32,
}

impl GrassField {
    /// The field the first visible entry of `grass` describes, swaying in
    /// `wind`. `None` when no visible field places a blade.
    pub fn resolve(grass: &[Grass], wind: Option<&Wind>) -> Option<Self> {
        let g = grass.iter().find(|g| g.visible)?;
        let g = crate::components::validate::grass(g.clone());
        let grid = GrassGrid::for_density(g.density)?;
        let patch = GrassPatch {
            center: g.center,
            extent: g.extent,
        };
        if patch.rect().area() <= 0.0 {
            return None;
        }
        Some(Self {
            patch,
            look: GrassBladeLook {
                height: g.height,
                height_variance: g.height_variance,
                width: g.width,
                clump_size: g.clump_size,
                stiffness: g.stiffness,
                root_color: g.root_color,
                tip_color: g.tip_color,
                color_variation: g.color_variation,
            },
            grid,
            wind: WindField::new(wind),
            capacity: tiles::blade_capacity(&patch.rect(), &grid, GRASS_DRAW_DISTANCE),
        })
    }

    /// The tiles a camera at `position` can see blades in.
    pub fn tiles(&self, position: [f32; 3]) -> TileRange {
        tiles::tiles_in_reach(
            &self.patch.rect(),
            [position[0], position[2]],
            GRASS_DRAW_DISTANCE,
        )
    }

    /// The kernel dispatch for a camera at `position`.
    pub fn dispatch(&self, position: [f32; 3]) -> [u32; 3] {
        self.tiles(position).dispatch(&self.grid)
    }

    /// This frame's parameters for `camera`, filling draw-argument slot
    /// `args_slot`.
    pub fn frame_params(&self, camera: &GrassCamera, args_slot: u32) -> GrassParams {
        let rect = self.patch.rect();
        let tiles = self.tiles(camera.position);
        let frustum = Frustum::from_camera(camera.vp, Some(GRASS_DRAW_DISTANCE));
        let look = &self.look;
        let rgb = |c: [f32; 3]| [c[0], c[1], c[2], 0.0];
        let [wind, wind_gust] = self.wind.gpu_rows();
        GrassParams {
            patch_rect: [rect.min[0], rect.min[1], rect.max[0], rect.max[1]],
            ground_y: self.patch.center[1],
            tile_size: GRASS_TILE_SIZE,
            cell_size: self.grid.cell_size(),
            cells_per_side: self.grid.cells_per_side,
            tile_origin: tiles.origin,
            tile_count: tiles.count,
            cam_pos: camera.position,
            draw_distance: GRASS_DRAW_DISTANCE,
            frustum: frustum
                .planes
                .map(|p| [p.normal[0], p.normal[1], p.normal[2], p.d]),
            height: look.height,
            height_variance: look.height_variance,
            width: look.width,
            clump_size: look.clump_size,
            stiffness: look.stiffness,
            color_variation: look.color_variation,
            capacity: self.capacity,
            args_slot: args_slot % GRASS_ARGS_SLOTS as u32,
            root_color: rgb(look.root_color),
            tip_color: rgb(look.tip_color),
            wind,
            wind_gust,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> [[f32; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    fn field() -> GrassField {
        GrassField::resolve(
            &[Grass {
                center: [10.0, 2.0, -4.0],
                extent: [20.0, 8.0],
                density: 100.0,
                ..Grass::default()
            }],
            None,
        )
        .unwrap()
    }

    #[test]
    fn the_first_visible_field_wins() {
        let hidden = Grass {
            visible: false,
            height: 9.0,
            ..Grass::default()
        };
        let shown = Grass {
            height: 0.8,
            ..Grass::default()
        };
        let f = GrassField::resolve(&[hidden, shown], None).unwrap();
        assert_eq!(f.look.height, 0.8);
    }

    #[test]
    fn a_field_that_places_nothing_resolves_to_none() {
        assert_eq!(GrassField::resolve(&[], None), None);
        let empty = Grass {
            density: 0.0,
            ..Grass::default()
        };
        assert_eq!(GrassField::resolve(&[empty], None), None);
        let flat = Grass {
            extent: [0.0, 5.0],
            ..Grass::default()
        };
        assert_eq!(GrassField::resolve(&[flat], None), None);
    }

    #[test]
    fn the_field_carries_the_world_wind() {
        let wind = Wind {
            strength: 6.0,
            ..Wind::default()
        };
        let f = GrassField::resolve(&[Grass::default()], Some(&wind)).unwrap();
        assert_eq!(f.wind.strength, 6.0);
        assert_eq!(field().wind, WindField::STILL);
    }

    #[test]
    fn frame_params_place_the_patch_and_the_tiles_in_reach() {
        let f = field();
        let p = f.frame_params(
            &GrassCamera {
                position: [10.0, 3.0, -4.0],
                vp: identity(),
            },
            1,
        );
        assert_eq!(p.patch_rect, [-10.0, -12.0, 30.0, 4.0]);
        assert_eq!(p.ground_y, 2.0);
        assert_eq!(p.tile_size, GRASS_TILE_SIZE);
        assert_eq!(p.cells_per_side, 40);
        assert_eq!(p.cell_size, 0.1);
        // x in [-10, 30) is tiles -3..=7; z in [-12, 4) is tiles -3..=0.
        assert_eq!(p.tile_origin, [-3, -3]);
        assert_eq!(p.tile_count, [11, 4]);
        assert_eq!(p.cam_pos, [10.0, 3.0, -4.0]);
        assert_eq!(p.draw_distance, GRASS_DRAW_DISTANCE);
        assert_eq!(p.capacity, f.capacity);
        assert_eq!(p.args_slot, 1);
        assert_eq!(p.root_color[3], 0.0);
        assert_eq!(f.dispatch([10.0, 3.0, -4.0]), [25, 11, 4]);
    }

    #[test]
    fn the_args_slot_wraps_onto_the_two_slots() {
        let cam = GrassCamera {
            position: [0.0; 3],
            vp: identity(),
        };
        assert_eq!(field().frame_params(&cam, 2).args_slot, 0);
        assert_eq!(field().frame_params(&cam, 7).args_slot, 1);
    }

    #[test]
    fn the_frustum_planes_are_the_cameras() {
        let p = field().frame_params(
            &GrassCamera {
                position: [0.0; 3],
                vp: identity(),
            },
            0,
        );
        let expected = Frustum::from_camera(identity(), Some(GRASS_DRAW_DISTANCE));
        for (row, plane) in p.frustum.iter().zip(expected.planes.iter()) {
            assert_eq!(&row[..3], &plane.normal[..]);
            assert_eq!(row[3], plane.d);
        }
    }
}
