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
    DVec3, HORIZON_SHADOW_REACH, Horizon, HorizonOccluder, LocalFrame, MAX_RESIDENT_TILES,
    PlanetFrame, TileBoundsCache, TileId, TileLod, TileMesh, TileMorphFrame, TileSeams, TileView,
    tile_mesh,
};
use concinnity_core::render::backend::{ChunkMesh, PlanetChunk};
use concinnity_core::render::error;
use concinnity_core::render::ops::{OpFailure, RenderOps};

use super::worker::Worker;
use crate::gfx::render_slots::RenderSlots;

// Threads building tile meshes. A fresh camera wants a thousand or more.
const WORKERS: usize = 3;

// Tiles being built at once, so a jump does not queue work it will discard.
const MAX_IN_FLIGHT: usize = 96;

// Frames a choice of tiles holds after a tile arrives or leaves, so a stream
// of arrivals does not choose again every frame.
const ARRIVAL_REPLAN_FRAMES: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Tile {
    // A worker is building it.
    Pending,
    // Its mesh is in the chunk pool at `draw`, placed relative to `origin`,
    // matching its neighbors along `seams`.
    Resident {
        draw: DrawIndex,
        origin: DVec3,
        visible: bool,
        seams: TileSeams,
    },
}

// One frame's changes to the resident tiles.
#[derive(Debug, Default, PartialEq)]
struct TilePlan {
    show: Vec<TileId>,
    hide: Vec<TileId>,
    seams: Vec<(TileId, TileSeams)>,
    evict: Vec<TileId>,
    request: Vec<TileId>,
}

// What the camera `view` makes of `tiles`: the resident tiles whose
// visibility flips, the drawn ones whose seams change, the tiles no longer
// worth keeping, and the
// tiles to build next, coarsest and nearest first, within the in-flight and
// residency bounds. A tile the quadtree draws stays hidden while it lies
// wholly behind the planet's horizon.
fn plan(
    lod: &TileLod,
    tiles: &BTreeMap<TileId, Tile>,
    view: &TileView,
    bounds: &mut TileBoundsCache,
) -> TilePlan {
    let camera = view.eye;
    let resident = |t: TileId| matches!(tiles.get(&t), Some(Tile::Resident { .. }));
    let selection = lod.select(view, &resident, bounds);
    // Both lists sorted, so the walk over the (sorted) tiles below meets
    // their entries in order and looks nothing up.
    let drawn: Vec<TileId> = selection.draw.iter().map(|d| d.id).collect();
    let drawn_seams: Vec<TileSeams> = selection.draw.iter().map(|d| d.seams).collect();
    let mut keep = selection.retained;
    keep.sort_unstable();
    let (mut drawn, mut keep) = (Sorted::new(&drawn), Sorted::new(&keep));
    let horizon = Horizon::new(
        camera,
        lod.shape.center,
        lod.shape.radius,
        HORIZON_SHADOW_REACH,
    );

    let mut out = TilePlan::default();
    for (&id, tile) in tiles {
        match *tile {
            Tile::Resident { visible, seams, .. } => {
                let kept = keep.contains(id);
                let drawn_at = drawn.position(id);
                if let Some(new) = drawn_at.map(|k| drawn_seams[k])
                    && new != seams
                {
                    out.seams.push((id, new));
                }
                let want = drawn_at.is_some()
                    && !horizon.is_some_and(|h| {
                        let b = bounds.get(id);
                        h.hides_sphere(b.center, b.radius)
                    });
                if !kept {
                    out.evict.push(id);
                } else if want && !visible {
                    out.show.push(id);
                } else if !want && visible {
                    out.hide.push(id);
                }
            }
            Tile::Pending if !keep.contains(id) => out.evict.push(id),
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
            let b = bounds.get(id);
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

// Whether tiles chosen for `planned` still serve `view`: the same ranges,
// and a camera within the drift the choice left room for.
fn still_serves(lod: &TileLod, planned: &TileView, view: &TileView) -> bool {
    let distance = (0..3)
        .map(|i| (view.eye[i] - planned.eye[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    planned.range_widths == view.range_widths && distance < lod.drift(planned)
}

// The morph inputs for the camera `view`, last frame's at authored point
// `prev_eye`, in the frame `planet` is drawn in.
fn morph_frame(
    lod: &TileLod,
    planet: &PlanetFrame,
    view: &TileView,
    prev_eye: DVec3,
) -> TileMorphFrame {
    TileMorphFrame {
        eye: planet.frame.to_local(view.eye),
        prev_eye: planet.frame.to_local(prev_eye),
        level0_range: lod.range(0, view) as f32,
    }
}

// What the last choice of tiles was made for, and what has happened since.
struct PlanState {
    view: Option<TileView>,
    frames: u32,
    residency_changed: bool,
}

impl PlanState {
    // Whether the tiles must be chosen again for `view`: never chosen, the
    // view no longer served, or tiles arrived or left and the choice has
    // held its frames.
    fn needs_plan(&self, lod: &TileLod, view: &TileView) -> bool {
        let arrivals_due = self.residency_changed && self.frames >= ARRIVAL_REPLAN_FRAMES;
        arrivals_due
            || !self
                .view
                .is_some_and(|planned| still_serves(lod, &planned, view))
    }
}

// A sorted list asked about ascending ids: each answer resumes where the
// last one stopped.
struct Sorted<'a> {
    ids: &'a [TileId],
    at: usize,
}

impl<'a> Sorted<'a> {
    fn new(ids: &'a [TileId]) -> Self {
        Self { ids, at: 0 }
    }

    // Whether `id`, at or past every id asked about before, is listed.
    fn contains(&mut self, id: TileId) -> bool {
        self.position(id).is_some()
    }

    // Where `id`, at or past every id asked about before, is listed.
    fn position(&mut self, id: TileId) -> Option<usize> {
        while self.ids.get(self.at).is_some_and(|&t| t < id) {
            self.at += 1;
        }
        (self.ids.get(self.at) == Some(&id)).then_some(self.at)
    }
}

/// A planet's streamed terrain: its tiles, the workers that build them, and
/// the material they draw with.
pub(crate) struct PlanetTiles {
    lod: TileLod,
    // The finest cell size `lod` was built for.
    finest_cell_size: f64,
    workers: Vec<Worker<TileId>>,
    results: Receiver<(TileId, TileMesh)>,
    next_worker: usize,
    tiles: BTreeMap<TileId, Tile>,
    bounds: TileBoundsCache,
    // The view the tiles were last chosen for: a view it still serves keeps
    // that choice.
    planned: Option<TileView>,
    // Frames since the tiles were last chosen.
    planned_frames: u32,
    // Whether a tile has arrived or left since.
    residency_changed: bool,
    // The camera last frame, in authored coordinates,
    // and the morph inputs the renderer last received.
    morph_eye: Option<DVec3>,
    morph_sent: Option<TileMorphFrame>,
    // The frame every resident tile's model matrix places it in.
    placed_in: LocalFrame,
    // The frame the renderer's horizon occluder was last placed in.
    occluder_in: Option<LocalFrame>,
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

/// Where the camera a planet's tiles are chosen for is, and how large the
/// ground looks from it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TileCamera {
    /// The camera, in the simulated frame.
    pub(crate) position: [f32; 3],
    /// How many logical points a radian of its view spans.
    pub(crate) points_per_radian: f64,
    /// How wide the finest terrain cells get, in meters.
    pub(crate) finest_cell_size: f64,
}

impl TileCamera {
    // The view its tiles are chosen for, from the simulated frame `frame`.
    fn view(&self, frame: &LocalFrame) -> TileView {
        TileView::new(frame.to_world(self.position), self.points_per_radian)
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
    ((n + 1) * (n + 1), n * n * 6)
}

impl PlanetTiles {
    /// Tiles of `planet` drawn with `material`.
    pub(crate) fn new(planet: &PlanetFrame, material: TileMaterial) -> Self {
        let lod = TileLod::new(planet.shape, concinnity_core::planet::FINEST_CELL_SIZE);
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
            finest_cell_size: concinnity_core::planet::FINEST_CELL_SIZE,
            workers,
            results,
            next_worker: 0,
            tiles: BTreeMap::new(),
            bounds: TileBoundsCache::new(planet.shape),
            planned: None,
            planned_frames: 0,
            residency_changed: false,
            morph_eye: None,
            morph_sent: None,
            placed_in: planet.frame,
            occluder_in: None,
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
            self.residency_changed = true;
        }
    }

    /// One frame: take the finished meshes, re-place every tile if the frame
    /// moved, and show, hide, drop and request tiles for `camera`.
    pub(crate) fn step(
        &mut self,
        planet: &PlanetFrame,
        camera: TileCamera,
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
        if self.occluder_in != Some(planet.frame) {
            self.occluder_in = Some(planet.frame);
            let occluder = HorizonOccluder {
                center: planet.center(),
                radius: planet.shape.radius as f32,
            };
            ops.record(move |backend| backend.set_horizon_occluder(Some(occluder)));
        }
        self.add_finished(planet, ops, slots, frame);

        if camera.finest_cell_size != self.finest_cell_size {
            self.finest_cell_size = camera.finest_cell_size;
            self.lod = TileLod::new(planet.shape, camera.finest_cell_size);
            self.planned = None;
        }
        let view = camera.view(&planet.frame);
        self.planned_frames = self.planned_frames.saturating_add(1);
        let due = PlanState {
            view: self.planned,
            frames: self.planned_frames,
            residency_changed: self.residency_changed,
        };
        if due.needs_plan(&self.lod, &view) {
            self.planned = Some(view);
            self.planned_frames = 0;
            self.residency_changed = false;
            let plan = plan(&self.lod, &self.tiles, &view, &mut self.bounds);
            self.apply(plan, ops, slots, retire_frame);
        }
        self.send_morph(planet, &view, ops);
    }

    // Show, hide, re-seam, drop and request tiles as `plan` says.
    fn apply(
        &mut self,
        plan: TilePlan,
        ops: &mut RenderOps,
        slots: &mut RenderSlots,
        retire_frame: u64,
    ) {
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
        for (id, new) in plan.seams {
            if let Some(Tile::Resident { draw, seams, .. }) = self.tiles.get_mut(&id) {
                *seams = new;
                let draw = *draw;
                ops.record(move |backend| backend.set_planet_tile_seams(draw, new));
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

    // Hand the renderer the camera, this frame and last, in the frame the
    // tiles are drawn in, whenever that changes. The morph follows the live
    // camera: the choice of tiles left room for it to stray this far.
    fn send_morph(&mut self, planet: &PlanetFrame, view: &TileView, ops: &mut RenderOps) {
        if self.planned.is_none() {
            return;
        }
        let morph = morph_frame(&self.lod, planet, view, self.morph_eye.unwrap_or(view.eye));
        self.morph_eye = Some(view.eye);
        if self.morph_sent != Some(morph) {
            self.morph_sent = Some(morph);
            ops.record(move |backend| backend.set_tile_morph(Some(morph)));
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
            self.residency_changed = true;
            let dst = slots.allocate_draw();
            let draw = dst.slot();
            self.tiles.insert(
                id,
                Tile::Resident {
                    draw,
                    origin: mesh.origin,
                    visible: false,
                    seams: TileSeams::default(),
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
                        planet_tile: Some(PlanetChunk {
                            tile: id,
                            radial: mesh.radial,
                            morph_extent: mesh.morph_extent,
                        }),
                        bounds: Some(mesh.bounds),
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
        let mesh = tile_mesh(&shape, id);
        if results.send((id, mesh)).is_err() {
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
            concinnity_core::planet::FINEST_CELL_SIZE,
        )
    }

    const CAMERA: DVec3 = [0.0, 40.0, 0.0];

    // The camera's view of the ground decides how far its levels reach.
    #[test]
    fn the_camera_s_view_sets_its_ranges() {
        let cells = f64::from(concinnity_core::planet::TILE_CELLS)
            * concinnity_core::planet::RANGE_CELL_POINTS;
        let camera = |ppr| TileCamera {
            position: [0.0, 40.0, 0.0],
            points_per_radian: ppr,
            finest_cell_size: 1.0,
        };
        let frame = LocalFrame::AUTHORED;
        let view = camera(5.5 * cells).view(&frame);
        assert_eq!(view.range_widths, 5.5);
        assert_eq!(view.eye, CAMERA);
        let wider = camera(5.8 * cells).view(&frame);
        assert!(wider.range_widths > view.range_widths);
        assert!(!still_serves(&lod(), &view, &wider));
    }

    fn view(eye: DVec3) -> TileView {
        TileView::nearest(eye)
    }

    fn resident(draw: u32, visible: bool) -> Tile {
        Tile::Resident {
            draw: DrawIndex::from_usize(draw as usize),
            origin: [0.0; 3],
            visible,
            seams: TileSeams::default(),
        }
    }

    // A fresh planet asks for the six faces first, nearest first, and no more
    // at once than the in-flight bound.
    #[test]
    fn a_fresh_planet_asks_for_its_faces_first() {
        let l = lod();
        let p = plan(
            &l,
            &BTreeMap::new(),
            &view(CAMERA),
            &mut TileBoundsCache::new(l.shape),
        );
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
    // for first, nearest first.
    #[test]
    fn resident_roots_show_and_their_children_follow() {
        let l = lod();
        let tiles: BTreeMap<TileId, Tile> = CubeFace::ALL
            .iter()
            .enumerate()
            .map(|(i, &f)| (TileId::root(f), resident(i as u32, false)))
            .collect();
        let p = plan(
            &l,
            &tiles,
            &view(CAMERA),
            &mut TileBoundsCache::new(l.shape),
        );
        // From 40 m up the horizon is 3.6 km out: the sides and the bottom
        // lie wholly behind it.
        assert_eq!(p.show, vec![TileId::root(CubeFace(2))]);
        assert_eq!(p.request[0].level, 1);
        assert!(
            p.request.windows(2).all(|w| w[0].level <= w[1].level),
            "coarsest first"
        );
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
        let p = plan(
            &l,
            &tiles,
            &view(CAMERA),
            &mut TileBoundsCache::new(l.shape),
        );
        assert!(p.evict.contains(&far) && p.evict.contains(&gone));
        assert_eq!(
            p.hide,
            CubeFace::ALL.map(TileId::root).to_vec(),
            "the top root gives way, the rest are behind the horizon"
        );
        assert_eq!(p.show.len(), 4, "to its four children");
    }

    // Tiles drawn by the quadtree but wholly behind the horizon stay hidden;
    // those within shadow reach of the camera show, below the horizon or not.
    #[test]
    fn tiles_behind_the_horizon_stay_hidden() {
        let l = lod();
        let mut bounds = TileBoundsCache::new(l.shape);
        let camera = [0.0, 32.0, 0.0];
        let sel = l.select(&view(camera), &|_| true, &mut bounds);
        let tiles: BTreeMap<TileId, Tile> = sel
            .wanted
            .iter()
            .enumerate()
            .map(|(i, &t)| (t, resident(i as u32, false)))
            .collect();
        let p = plan(&l, &tiles, &view(camera), &mut bounds);
        let horizon = Horizon::new(camera, l.shape.center, l.shape.radius, 0.0).unwrap();
        let mut hidden = 0;
        for t in sel.draw.iter().map(|d| d.id) {
            let b = bounds.get(t);
            let near = (0..3)
                .map(|i| (b.center[i] - camera[i]).powi(2))
                .sum::<f64>()
                .sqrt()
                - b.radius
                <= HORIZON_SHADOW_REACH;
            let behind = horizon.hides_sphere(b.center, b.radius);
            assert_eq!(p.show.contains(&t), near || !behind, "{t:?}");
            hidden += usize::from(!p.show.contains(&t));
        }
        assert!(hidden > 0, "the far side is hidden");
        assert!(p.show.len() > hidden / 4, "the near side shows");
    }

    #[test]
    fn a_sorted_list_answers_ascending_questions() {
        let ids: Vec<TileId> = (0..6)
            .step_by(2)
            .map(|x| TileId {
                face: CubeFace(1),
                level: 3,
                x,
                y: 0,
            })
            .collect();
        let mut sorted = Sorted::new(&ids);
        let asked: Vec<bool> = (0..7)
            .map(|x| sorted.contains(TileId { x, ..ids[0] }))
            .collect();
        assert_eq!(asked, [true, false, true, false, true, false, false]);
    }

    #[test]
    fn the_largest_tile_mesh_is_what_a_tile_builds() {
        let mesh = tile_mesh(&lod().shape, TileId::root(CubeFace(0)));
        assert_eq!((mesh.vertices.len(), mesh.indices.len()), tile_mesh_size());
    }

    // A choice of tiles holds while the camera drifts less than a twentieth
    // of its height (or of the finest range near the ground) and the ranges
    // stay put.
    #[test]
    fn a_plan_serves_until_the_camera_drifts_or_the_ranges_change() {
        let l = lod();
        let top = l.shape.radius + l.shape.amplitude;
        let at = |height: f64| view([0.0, top - 50_000.0 + height, 0.0]);
        let ground = at(2.0);
        let finest = l.range(l.max_level, &ground);
        let nudge = |v: TileView, by: f64| TileView {
            eye: [v.eye[0] + by, v.eye[1], v.eye[2]],
            ..v
        };
        assert!(still_serves(&l, &ground, &ground));
        assert!(still_serves(&l, &ground, &nudge(ground, finest * 0.049)));
        assert!(!still_serves(&l, &ground, &nudge(ground, finest * 0.051)));
        let aloft = at(5_000.0);
        assert!(still_serves(&l, &aloft, &nudge(aloft, 249.0)));
        assert!(!still_serves(&l, &aloft, &nudge(aloft, 251.0)));
        let wider = TileView {
            range_widths: ground.range_widths + 0.5,
            ..ground
        };
        assert!(!still_serves(&l, &ground, &wider));
    }

    #[test]
    fn arrivals_wait_their_frames_before_a_new_plan() {
        let l = lod();
        let v = view(CAMERA);
        let state = |view, frames, residency_changed| PlanState {
            view,
            frames,
            residency_changed,
        };
        assert!(state(None, 0, false).needs_plan(&l, &v));
        assert!(!state(Some(v), 1, false).needs_plan(&l, &v));
        assert!(!state(Some(v), 100, false).needs_plan(&l, &v));
        assert!(!state(Some(v), ARRIVAL_REPLAN_FRAMES - 1, true).needs_plan(&l, &v));
        assert!(state(Some(v), ARRIVAL_REPLAN_FRAMES, true).needs_plan(&l, &v));
        let moved = view([CAMERA[0] + 1_000.0, CAMERA[1], CAMERA[2]]);
        assert!(state(Some(v), 1, false).needs_plan(&l, &moved));
    }
}
