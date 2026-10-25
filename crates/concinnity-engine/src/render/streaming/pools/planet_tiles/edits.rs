// New edits over a planet's resident tiles: which tiles they touch, and those
// tiles built again in place, coarsest and nearest first.

use concinnity_core::math::dvec3;

use super::{MAX_IN_FLIGHT, PlanetTiles, Tile, TileBuild, in_flight};
use crate::planet::{DVec3, EditArea, PlanetShape, TileId};

/// Where a tile lies, for the areas edits change, and whether edits changed
/// its ground since its mesh was asked for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Touch {
    // The area holding all of the tile.
    pub(super) area: EditArea,
    // Edits changed its ground since its mesh was asked for.
    pub(super) stale: bool,
    // A worker is building it again.
    pub(super) rebuilding: bool,
}

impl Touch {
    pub(super) fn new(tile: TileId) -> Self {
        Self {
            area: EditArea::of_tile(tile),
            stale: false,
            rebuilding: false,
        }
    }
}

impl PlanetTiles {
    // Bring the tiles to `shape`, whose edits changed the ground within
    // `area` and nowhere else: every tile that reaches into it, built or
    // being built, stands stale.
    pub(super) fn touch(&mut self, shape: &PlanetShape, area: EditArea) {
        self.lod.shape.clone_from(shape);
        self.bounds.reshape_within(shape.clone(), &area);
        for tile in self.tiles.values_mut() {
            let (Tile::Pending { touch } | Tile::Resident { touch, .. }) = tile;
            if touch.area.overlaps(&area) {
                touch.stale = true;
                self.any_stale = true;
            }
        }
    }

    // Ask again for the stale resident tiles a worker is not building, the
    // coarsest first and then the nearest to the camera at `eye`, as many as
    // the in-flight bound leaves room for.
    pub(super) fn rebuild_touched(&mut self, eye: DVec3) {
        if !self.any_stale {
            return;
        }
        let up = self.lod.shape.up_at(eye);
        let mut waiting = false;
        self.rebuilds.clear();
        for (&id, tile) in &self.tiles {
            match tile {
                Tile::Resident { touch, .. } if touch.stale && !touch.rebuilding => {
                    let near = dvec3::dot(up, touch.area.center);
                    self.rebuilds.push((id.level, -near, id));
                }
                Tile::Pending { touch } if touch.stale => waiting = true,
                _ => {}
            }
        }
        self.rebuilds
            .sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
        let room = MAX_IN_FLIGHT.saturating_sub(in_flight(&self.tiles));
        for &(_, _, id) in self.rebuilds.iter().take(room) {
            if let Some(Tile::Resident { touch, .. }) = self.tiles.get_mut(&id) {
                touch.stale = false;
                touch.rebuilding = true;
                self.mail.request(TileBuild {
                    tile: id,
                    shape: self.lod.shape.clone(),
                    generation: self.generation,
                });
            }
        }
        self.any_stale = waiting || self.rebuilds.len() > room;
    }
}

#[cfg(test)]
mod tests {
    use super::super::{TileCamera, TileMaterial};
    use super::*;
    use crate::planet::{
        CubeFace, EditPage, EditPages, LocalFrame, PAGE_SAMPLES, PlanetEdited, PlanetFrame,
    };
    use crate::render::frame::ops::RenderOps;
    use crate::render::slots::render_slots::RenderSlots;
    use crate::render::streaming::pools::state::PlanetView;
    use alloc::sync::Arc;
    use alloc::vec::Vec;

    fn shape() -> PlanetShape {
        PlanetShape {
            center: [0.0, -50_000.0, 0.0],
            radius: 50_000.0,
            amplitude: 30.0,
            feature_size: 2_000.0,
            octaves: 6,
            seed: 5,
            continents: crate::planet::Continents::NONE,
            edits: None,
        }
    }

    // A raised page over `tile`.
    fn raised(tile: TileId) -> Arc<EditPages> {
        let page = EditPage::from_samples(&[3.0; PAGE_SAMPLES], &[1.0; PAGE_SAMPLES]).unwrap();
        Arc::new(EditPages::new([(tile, page)]))
    }

    // The resident tiles down one column of the quadtree over the top face's
    // middle, and one tile far from it on another face.
    fn resident_tiles(tiles: &mut PlanetTiles, slots: &mut RenderSlots) -> Vec<TileId> {
        let mut column = Vec::new();
        let mut at = TileId::root(CubeFace(2));
        for _ in 0..8 {
            column.push(at);
            at = at.children()[3];
        }
        column.push(TileId {
            face: CubeFace(3),
            level: 4,
            x: 2,
            y: 9,
        });
        for &id in &column {
            tiles.tiles.insert(
                id,
                Tile::Resident {
                    draw: slots.allocate_draw().slot(),
                    origin: [0.0; 3],
                    visible: true,
                    seams: crate::planet::TileSeams::default(),
                    touch: Touch::new(id),
                },
            );
        }
        column
    }

    // The tiles chosen for the view as they stand, so no plan moves them.
    fn hold_the_plan(tiles: &mut PlanetTiles, shape: &PlanetShape) {
        let camera = view(shape, None).camera;
        tiles.planned = Some(camera.view(&LocalFrame::AUTHORED));
        tiles.residency_changed = false;
    }

    fn view(shape: &PlanetShape, edited: Option<PlanetEdited>) -> PlanetView {
        PlanetView {
            frame: PlanetFrame {
                shape: shape.clone(),
                frame: LocalFrame::AUTHORED,
            },
            camera: TileCamera {
                position: [0.0, 40.0, 0.0],
                points_per_radian: 2_000.0,
                finest_cell_size: crate::planet::FINEST_CELL_SIZE,
            },
            edited,
        }
    }

    // Edits over a small area build again only the resident tiles at every
    // level that reach into it, coarsest first, each staying drawn until its
    // new mesh lands in its place; tiles elsewhere are left as they are.
    #[test]
    fn only_the_touched_tiles_are_built_again() {
        let plain = shape();
        let frame = PlanetFrame {
            shape: plain.clone(),
            frame: LocalFrame::AUTHORED,
        };
        let mut tiles = PlanetTiles::new(&frame, TileMaterial::from_entry(None));
        let mut ops = RenderOps::default();
        let mut slots = RenderSlots::new(0, true, &[]);
        let column = resident_tiles(&mut tiles, &mut slots);
        let deepest = column[7];
        let edited = plain.with_edits(Some(raised(deepest)));
        let area = EditArea {
            center: deepest.direction(0.5, 0.5),
            angle: 1e-6,
        };
        let change = PlanetEdited { revision: 1, area };
        hold_the_plan(&mut tiles, &plain);
        tiles.step(&view(&edited, Some(change)), &mut ops, &mut slots, (0, 2));
        let builds: Vec<TileBuild> = tiles.mail.take_requests().collect();
        let requested: Vec<TileId> = builds.iter().map(|b| b.tile).collect();
        let rebuilt: Vec<TileId> = column[..8].to_vec();
        assert!(
            rebuilt.iter().all(|t| requested.contains(t)),
            "{requested:?}"
        );
        assert!(!requested.contains(&column[8]), "the far tile is left");
        let levels: Vec<u8> = requested
            .iter()
            .filter(|t| rebuilt.contains(t))
            .map(|t| t.level)
            .collect();
        assert!(levels.windows(2).all(|w| w[0] <= w[1]), "coarsest first");
        assert!(
            tiles
                .tiles
                .values()
                .all(|t| matches!(t, Tile::Resident { .. }))
        );
        assert_eq!(tiles.stats().0, 9, "every tile stays drawn meanwhile");

        // The new meshes land in place: the same tiles, drawn as they were,
        // from the edited shape.
        for build in builds {
            if column.contains(&build.tile) {
                tiles.mail.deliver(build.build());
            }
        }
        tiles.step(&view(&edited, Some(change)), &mut ops, &mut slots, (1, 3));
        for &id in &column {
            let Some(Tile::Resident { visible, touch, .. }) = tiles.tiles.get(&id) else {
                panic!("{id:?} is no longer resident");
            };
            assert!(*visible && !touch.stale && !touch.rebuilding, "{id:?}");
        }
        assert!(!tiles.any_stale);
    }

    // Edits that change while a tile is being built again build it once
    // more when the first new mesh lands; a change the tiles did not follow
    // touches every tile.
    #[test]
    fn edits_during_a_rebuild_build_the_tile_again() {
        let plain = shape();
        let frame = PlanetFrame {
            shape: plain.clone(),
            frame: LocalFrame::AUTHORED,
        };
        let mut tiles = PlanetTiles::new(&frame, TileMaterial::from_entry(None));
        let mut ops = RenderOps::default();
        let mut slots = RenderSlots::new(0, true, &[]);
        let column = resident_tiles(&mut tiles, &mut slots);
        hold_the_plan(&mut tiles, &plain);
        let area = EditArea {
            center: column[7].direction(0.5, 0.5),
            angle: 1e-6,
        };
        let first = plain.with_edits(Some(raised(column[7])));
        tiles.step(
            &view(&first, Some(PlanetEdited { revision: 1, area })),
            &mut ops,
            &mut slots,
            (0, 2),
        );
        let second = plain.with_edits(Some(raised(column[6])));
        tiles.step(
            &view(&second, Some(PlanetEdited { revision: 2, area })),
            &mut ops,
            &mut slots,
            (1, 3),
        );
        let Some(Tile::Resident { touch, .. }) = tiles.tiles.get(&column[7]) else {
            panic!("resident");
        };
        assert!(touch.stale && touch.rebuilding);

        let skipped = plain.with_edits(Some(raised(column[5])));
        tiles.step(
            &view(&skipped, Some(PlanetEdited { revision: 9, area })),
            &mut ops,
            &mut slots,
            (2, 4),
        );
        let Some(Tile::Resident { touch, .. }) = tiles.tiles.get(&column[8]) else {
            panic!("resident");
        };
        assert!(
            touch.stale || touch.rebuilding,
            "a missed change touches all"
        );
    }
}
