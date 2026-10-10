//! The `std`-side driver for a planet's terrain tiles.
//!
//! Worker threads build each tile's mesh from the planet's shape; the meshes
//! ride the chunk pool like a voxel world's chunks. Which tiles to build, keep,
//! show and drop is the quadtree policy in `concinnity_core::planet`
//! ([`TileLod`]); this module applies it to what is resident each frame.
//!
//! A tile's vertices are relative to its own origin, held in double precision,
//! so its model matrix is the only thing that changes when the simulated frame
//! moves: every resident tile is re-placed in the new frame on the frame it
//! moves.

use std::collections::BTreeMap;
use std::sync::mpsc::{Receiver, Sender};

use concinnity_core::gfx::render_types::{DrawIndex, MaterialUniforms};
use concinnity_core::planet::{
    DVec3, LocalFrame, MAX_RESIDENT_TILES, PlanetFrame, TileId, TileLod, TileMesh, tile_mesh,
};
use concinnity_core::render::backend::ChunkMesh;
use concinnity_core::render::error;
use concinnity_core::render::ops::{OpFailure, RenderOps};

use super::worker::Worker;
use crate::gfx::render_slots::RenderSlots;

// Threads building tile meshes. A tile takes about a millisecond; a fresh
// camera wants a few hundred.
const WORKERS: usize = 3;

// Tiles being built at once, so a jump does not queue work it will discard.
const MAX_IN_FLIGHT: usize = 24;

// The finest cells a tile reaches under the camera, in meters.
const FINEST_CELL_M: f64 = 1.0;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Tile {
    // A worker is building it.
    Pending,
    // Its mesh is in the chunk pool at `draw`, placed relative to `origin`.
    Resident {
        draw: DrawIndex,
        origin: DVec3,
        visible: bool,
    },
}

// One frame's changes to the resident tiles.
#[derive(Debug, Default, PartialEq)]
struct TilePlan {
    show: Vec<TileId>,
    hide: Vec<TileId>,
    evict: Vec<TileId>,
    request: Vec<TileId>,
}

// What the camera at authored point `camera` makes of `tiles`: the resident
// tiles whose visibility flips, the tiles no longer worth keeping, and the
// tiles to build next, coarsest and nearest first, within the in-flight and
// residency bounds.
fn plan(lod: &TileLod, tiles: &BTreeMap<TileId, Tile>, camera: DVec3) -> TilePlan {
    let resident = |t: TileId| matches!(tiles.get(&t), Some(Tile::Resident { .. }));
    let selection = lod.select(camera, &resident);
    let drawn: std::collections::BTreeSet<TileId> = selection.draw.iter().copied().collect();
    let mut keep: std::collections::BTreeSet<TileId> = lod.retained(camera).into_iter().collect();
    keep.extend(drawn.iter().copied());

    let mut out = TilePlan::default();
    for (&id, tile) in tiles {
        match *tile {
            Tile::Resident { visible, .. } => {
                let want = drawn.contains(&id);
                if !keep.contains(&id) {
                    out.evict.push(id);
                } else if want && !visible {
                    out.show.push(id);
                } else if !want && visible {
                    out.hide.push(id);
                }
            }
            Tile::Pending if !keep.contains(&id) => out.evict.push(id),
            Tile::Pending => {}
        }
    }

    let pending = tiles.values().filter(|t| **t == Tile::Pending).count();
    let held = tiles.len() - out.evict.len();
    let room = MAX_IN_FLIGHT
        .saturating_sub(pending)
        .min(MAX_RESIDENT_TILES.saturating_sub(held));
    let mut wanted: Vec<(u8, f64, TileId)> = selection
        .wanted
        .iter()
        .filter(|id| !tiles.contains_key(id))
        .map(|&id| {
            let b = concinnity_core::planet::tile_bounds(&lod.shape, id);
            let d = (0..3)
                .map(|i| (camera[i] - b.center[i]).powi(2))
                .sum::<f64>();
            (id.level, d, id)
        })
        .collect();
    wanted.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
    out.request = wanted.into_iter().take(room).map(|(_, _, id)| id).collect();
    out
}

/// A planet's streamed terrain: its tiles, the workers that build them, and
/// the material they draw with.
pub(crate) struct PlanetTiles {
    lod: TileLod,
    workers: Vec<Worker<TileId>>,
    results: Receiver<(TileId, TileMesh)>,
    next_worker: usize,
    tiles: BTreeMap<TileId, Tile>,
    // The frame every resident tile's model matrix places it in.
    placed_in: LocalFrame,
    texture_slot: usize,
    normal_map_slot: usize,
    material: MaterialUniforms,
}

impl std::fmt::Debug for PlanetTiles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlanetTiles")
            .field("tiles", &self.tiles.len())
            .finish_non_exhaustive()
    }
}

/// How one tile draws: the albedo and normal-map slots and the material
/// scalars every tile of a planet shares.
#[derive(Clone, Copy)]
pub(crate) struct TileMaterial {
    pub(crate) texture_slot: usize,
    pub(crate) normal_map_slot: usize,
    pub(crate) material: MaterialUniforms,
}

/// The largest mesh a tile builds: `(vertices, indices)`.
pub(crate) fn tile_mesh_size() -> (usize, usize) {
    let n = concinnity_core::planet::TILE_CELLS as usize;
    ((n + 1) * (n + 1) + 4 * n, n * n * 6 + 4 * n * 6)
}

impl PlanetTiles {
    pub(crate) fn new(planet: &PlanetFrame, material: TileMaterial) -> Self {
        let lod = TileLod::new(planet.shape, FINEST_CELL_M);
        let (result_tx, results) = std::sync::mpsc::channel::<(TileId, TileMesh)>();
        let workers = (0..WORKERS)
            .map(|i| {
                let (request_tx, request_rx) = std::sync::mpsc::channel::<TileId>();
                let results = result_tx.clone();
                let shape = lod.shape;
                Worker::spawn(
                    &format!("cn-planet-tiles-{i}"),
                    request_rx,
                    request_tx,
                    move |rx| build_tiles(shape, rx, results),
                )
            })
            .collect();
        Self {
            lod,
            workers,
            results,
            next_worker: 0,
            tiles: BTreeMap::new(),
            placed_in: planet.frame,
            texture_slot: material.texture_slot,
            normal_map_slot: material.normal_map_slot,
            material: material.material,
        }
    }

    /// `(resident, pending)` tile counts.
    pub(crate) fn stats(&self) -> (usize, usize) {
        let pending = self.tiles.values().filter(|t| **t == Tile::Pending).count();
        (self.tiles.len() - pending, pending)
    }

    /// Forget a tile whose add the backend refused, freeing its slot.
    pub(crate) fn roll_back(&mut self, tile: TileId, slots: &mut RenderSlots) {
        if let Some(Tile::Resident { draw, .. }) = self.tiles.remove(&tile) {
            slots.free_draw(draw);
        }
    }

    /// One frame: take the finished meshes, re-place every tile if the frame
    /// moved, and show, hide, drop and request tiles for the camera at local
    /// point `camera`.
    pub(crate) fn step(
        &mut self,
        planet: &PlanetFrame,
        camera: [f32; 3],
        ops: &mut RenderOps,
        slots: &mut RenderSlots,
        frame: u64,
        retire_frame: u64,
    ) {
        if planet.frame != self.placed_in {
            self.placed_in = planet.frame;
            for tile in self.tiles.values() {
                if let Tile::Resident { draw, origin, .. } = *tile {
                    let model = planet.frame.model_at(origin);
                    ops.record(move |backend| {
                        if let Err(e) = backend.set_chunk_model(draw, model) {
                            tracing::warn!("StreamingSystem: planet tile rebase: {e}");
                        }
                    });
                }
            }
        }
        self.add_finished(planet, ops, slots, frame);

        let plan = plan(&self.lod, &self.tiles, planet.frame.to_world(camera));
        for (ids, visible) in [(plan.show, true), (plan.hide, false)] {
            for id in ids {
                if let Some(Tile::Resident {
                    draw, visible: v, ..
                }) = self.tiles.get_mut(&id)
                {
                    *v = visible;
                    let draw = *draw;
                    ops.record(move |backend| backend.update_visibility(draw, visible));
                }
            }
        }
        for id in plan.evict {
            if let Some(Tile::Resident { draw, .. }) = self.tiles.remove(&id) {
                slots.free_draw(draw);
                ops.record(move |backend| {
                    if let Err(e) = backend.remove_chunk_mesh(draw, retire_frame) {
                        tracing::warn!("StreamingSystem: planet tile remove: {e}");
                    }
                });
            }
        }
        for id in plan.request {
            let worker = &self.workers[self.next_worker % self.workers.len()];
            self.next_worker = self.next_worker.wrapping_add(1);
            if worker.send(id) {
                self.tiles.insert(id, Tile::Pending);
            }
        }
    }

    // Put every finished mesh still wanted into the chunk pool, hidden until
    // the plan shows it.
    fn add_finished(
        &mut self,
        planet: &PlanetFrame,
        ops: &mut RenderOps,
        slots: &mut RenderSlots,
        frame: u64,
    ) {
        while let Ok((id, mesh)) = self.results.try_recv() {
            if self.tiles.get(&id) != Some(&Tile::Pending) {
                continue;
            }
            let dst = slots.allocate_draw();
            let draw = dst.slot();
            self.tiles.insert(
                id,
                Tile::Resident {
                    draw,
                    origin: mesh.origin,
                    visible: false,
                },
            );
            let model = planet.frame.model_at(mesh.origin);
            let (texture_slot, normal_map_slot, material) =
                (self.texture_slot, self.normal_map_slot, self.material);
            ops.record_with(move |backend, out| {
                let added = backend.add_chunk_mesh(
                    ChunkMesh {
                        verts: &mesh.vertices,
                        idxs: &mesh.indices,
                        model,
                        texture_slot,
                        normal_map_slot,
                        material,
                        frame,
                    },
                    dst,
                );
                match added {
                    Ok(()) => backend.update_visibility(draw, false),
                    Err(e) => {
                        tracing::warn!("StreamingSystem: planet tile add: {e}");
                        out.memory_pressure |=
                            matches!(e, error::RenderError::OutOfDeviceMemory(_));
                        out.failures.push(OpFailure::PlanetTileAdd { tile: id });
                    }
                }
            });
        }
    }
}

// A worker: build every requested tile's mesh and send it back, until the
// streamer is dropped.
fn build_tiles(
    shape: concinnity_core::planet::PlanetShape,
    requests: Receiver<TileId>,
    results: Sender<(TileId, TileMesh)>,
) {
    while let Ok(id) = requests.recv() {
        if results.send((id, tile_mesh(&shape, id))).is_err() {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::planet::{CubeFace, PlanetShape};

    fn lod() -> TileLod {
        TileLod::new(
            PlanetShape {
                center: [0.0, -50_000.0, 0.0],
                radius: 50_000.0,
                amplitude: 30.0,
                feature_size: 2_000.0,
                octaves: 6,
                seed: 5,
            },
            FINEST_CELL_M,
        )
    }

    const CAMERA: DVec3 = [0.0, 40.0, 0.0];

    fn resident(draw: u32, visible: bool) -> Tile {
        Tile::Resident {
            draw: DrawIndex::from_usize(draw as usize),
            origin: [0.0; 3],
            visible,
        }
    }

    // A fresh planet asks for the six faces first, nearest first, and no more
    // at once than the in-flight bound.
    #[test]
    fn a_fresh_planet_asks_for_its_faces_first() {
        let p = plan(&lod(), &BTreeMap::new(), CAMERA);
        assert_eq!(p.request.len(), MAX_IN_FLIGHT);
        let mut first: Vec<TileId> = p.request[..6].to_vec();
        assert_eq!(
            first[0],
            TileId::root(CubeFace(2)),
            "the face under the camera"
        );
        first.sort();
        assert_eq!(first, CubeFace::ALL.map(TileId::root).to_vec());
        assert!(p.show.is_empty() && p.hide.is_empty() && p.evict.is_empty());
    }

    // With only the roots resident, the roots show; the next level is asked
    // for, nearest first.
    #[test]
    fn resident_roots_show_and_their_children_follow() {
        let l = lod();
        let tiles: BTreeMap<TileId, Tile> = CubeFace::ALL
            .iter()
            .enumerate()
            .map(|(i, &f)| (TileId::root(f), resident(i as u32, false)))
            .collect();
        let p = plan(&l, &tiles, CAMERA);
        assert_eq!(p.show.len(), 6);
        assert!(p.request.iter().all(|t| t.level == 1));
        assert_eq!(p.request[0].face, CubeFace(2), "the face under the camera");
    }

    // A tile nobody wants any more is dropped, built or not; a shown tile the
    // camera no longer draws is hidden.
    #[test]
    fn unwanted_tiles_go_and_undrawn_tiles_hide() {
        let l = lod();
        let mut tiles: BTreeMap<TileId, Tile> = CubeFace::ALL
            .iter()
            .enumerate()
            .map(|(i, &f)| (TileId::root(f), resident(i as u32, true)))
            .collect();
        // A deep tile on the far side of the planet.
        let far = TileId {
            face: CubeFace(3),
            level: 6,
            x: 3,
            y: 3,
        };
        tiles.insert(far, resident(40, true));
        let gone = TileId { x: 4, ..far };
        tiles.insert(gone, Tile::Pending);
        // Every child of the top face resident, so the top root hides.
        for (i, c) in TileId::root(CubeFace(2)).children().iter().enumerate() {
            tiles.insert(*c, resident(10 + i as u32, false));
        }
        let p = plan(&l, &tiles, CAMERA);
        assert!(p.evict.contains(&far) && p.evict.contains(&gone));
        assert_eq!(
            p.hide,
            vec![TileId::root(CubeFace(2))],
            "the top root gives way"
        );
        assert_eq!(p.show.len(), 4, "to its four children");
    }

    #[test]
    fn the_largest_tile_mesh_is_what_a_tile_builds() {
        let mesh = tile_mesh(&lod().shape, TileId::root(CubeFace(0)));
        assert_eq!((mesh.vertices.len(), mesh.indices.len()), tile_mesh_size());
    }
}
