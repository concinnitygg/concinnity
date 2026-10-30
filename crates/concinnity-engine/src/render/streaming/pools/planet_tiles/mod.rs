//! A planet's terrain tiles.
//!
//! The host's workers build each tile's mesh from the planet's shape; the
//! meshes ride the chunk pool like a voxel world's chunks. Which tiles to build, keep,
//! show and drop is the quadtree policy in `crate::planet`
//! ([`TileLod`]); this module applies it to what is resident each frame.
//!
//! A tile's vertices are relative to its own origin, held in double precision,
//! so its model matrix is the only thing that changes when the simulated frame
//! moves: every resident tile is re-placed in the new frame on the frame it
//! moves. A planet reshaped while the world runs drops every tile and streams
//! the new shape's in, coarsest first; a mesh built for the old shape is
//! discarded when it arrives. New edits over an area build again only the
//! tiles they touch, at every level, coarsest first, each drawn as it was
//! until its new mesh takes its place.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use concinnity_core::math::{dvec3, narrow_f32};

mod edits;

use super::state::PlanetView;
use super::{LoadPool, Mailbox};
use crate::gfx::render_types::{self, DrawIndex, MaterialUniforms};
use crate::planet::{
    DVec3, EditArea, HORIZON_SHADOW_REACH, Horizon, HorizonOccluder, LocalFrame,
    MAX_RESIDENT_TILES, PlanetFrame, PlanetShape, TileBoundsCache, TileId, TileLod, TileMesh,
    TileMorphFrame, TileSeams, TileView, tile_mesh,
};
use crate::render::backend::error;
use crate::render::backend::{ChunkMesh, PlanetChunk};
use crate::render::environment::grass::CanopyRange;
use crate::render::frame::ops::{OpFailure, RenderOps};
use crate::render::material_entry::MaterialEntry;
use crate::render::slots::render_slots::RenderSlots;
use edits::Touch;

// Tiles being built at once, so a jump does not queue work it will discard.
const MAX_IN_FLIGHT: usize = 96;

// Frames a choice of tiles holds after a tile arrives or leaves, so a stream
// of arrivals does not choose again every frame.
const ARRIVAL_REPLAN_FRAMES: u32 = 4;

/// One tile mesh to build: `tile` of a planet of `shape`, for the pool's
/// `generation` of shapes.
#[derive(Debug, Clone)]
pub struct TileBuild {
    /// The tile.
    pub tile: TileId,
    /// The planet's surface.
    pub shape: PlanetShape,
    /// Which of the pool's shapes it is built for.
    pub generation: u32,
}

impl TileBuild {
    /// The mesh, built on the calling thread.
    pub fn build(self) -> TileBuilt {
        TileBuilt {
            tile: self.tile,
            generation: self.generation,
            mesh: tile_mesh(&self.shape, self.tile),
        }
    }
}

/// A built tile mesh, for the generation of shapes it was asked for.
#[derive(Debug, Clone)]
pub struct TileBuilt {
    /// The tile.
    pub tile: TileId,
    /// Which of the pool's shapes it was built for.
    pub generation: u32,
    /// The mesh.
    pub mesh: TileMesh,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Tile {
    // A worker is building it.
    Pending {
        touch: Touch,
    },
    // Its mesh is in the chunk pool at `draw`, placed relative to `origin`,
    // matching its neighbors along `seams`.
    Resident {
        draw: DrawIndex,
        origin: DVec3,
        visible: bool,
        seams: TileSeams,
        touch: Touch,
    },
}

impl Tile {
    // A worker is building it: its first mesh, or a new one for edits.
    fn in_flight(&self) -> bool {
        match self {
            Self::Pending { .. } => true,
            Self::Resident { touch, .. } => touch.rebuilding,
        }
    }
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
    failed: &BTreeSet<TileId>,
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
    let horizon = Horizon::of_planet(camera, &lod.shape, HORIZON_SHADOW_REACH);

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
            Tile::Pending { .. } if !keep.contains(id) => out.evict.push(id),
            Tile::Pending { .. } => {}
        }
    }

    let pending = in_flight(tiles);
    let held = tiles.len() - out.evict.len();
    let room = MAX_IN_FLIGHT
        .saturating_sub(pending)
        .min(MAX_RESIDENT_TILES.saturating_sub(held));
    let mut wanted: Vec<(u8, f64, TileId)> = selection
        .wanted
        .iter()
        .filter(|id| !tiles.contains_key(id) && !failed.contains(id))
        .map(|&id| {
            let b = bounds.get(id);
            let to = dvec3::sub(camera, b.center);
            let d = dvec3::dot(to, to);
            (id.level, d, id)
        })
        .collect();
    wanted.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
    out.request = wanted.into_iter().take(room).map(|(_, _, id)| id).collect();
    out
}

// How many tiles a worker is building.
fn in_flight(tiles: &BTreeMap<TileId, Tile>) -> usize {
    tiles.values().filter(|t| t.in_flight()).count()
}

// Whether tiles chosen for `planned` still serve `view`: the same ranges,
// and a camera within the drift the choice left room for.
#[expect(clippy::float_cmp, reason = "range widths change only with the plan")]
fn still_serves(lod: &TileLod, planned: &TileView, view: &TileView) -> bool {
    let distance = dvec3::length(dvec3::sub(view.eye, planned.eye));
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
        level0_range: narrow_f32(lod.range(0, view)),
        coarsest_level: lod.min_level,
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

/// A planet's streamed terrain: its tiles, the builds requested and finished,
/// and the material they draw with.
pub struct PlanetTiles {
    lod: TileLod,
    // The finest cell size `lod` was built for.
    finest_cell_size: f64,
    mail: Mailbox<TileBuild, TileBuilt>,
    // Counts the shapes the tiles were built for: a mesh of another is stale.
    generation: u32,
    tiles: BTreeMap<TileId, Tile>,
    // Tiles whose build failed for good, never requested again.
    failed: BTreeSet<TileId>,
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
    // The planet the renderer's horizon occluder was last placed for.
    occluder_for: Option<PlanetFrame>,
    // The last change of the edits the tiles were brought to.
    edit_revision: u64,
    // Whether any resident tile waits to be built again for new edits, and
    // the tiles asked for again this frame.
    any_stale: bool,
    rebuilds: Vec<(u8, f64, TileId)>,
    texture_slot: usize,
    normal_map_slot: usize,
    material: MaterialUniforms,
}

impl core::fmt::Debug for PlanetTiles {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PlanetTiles")
            .field("tiles", &self.tiles.len())
            .finish_non_exhaustive()
    }
}

/// Where the camera a planet's tiles are chosen for is, and how large the
/// ground looks from it.
#[derive(Debug, Clone, Copy)]
pub struct TileCamera {
    /// The camera, in the simulated frame.
    pub position: [f32; 3],
    /// How many logical points a radian of its view spans.
    pub points_per_radian: f64,
    /// How wide the finest terrain cells get, in meters.
    pub finest_cell_size: f64,
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
pub struct TileMaterial {
    /// The albedo slot.
    pub texture_slot: usize,
    /// The normal-map slot.
    pub normal_map_slot: usize,
    /// The material scalars.
    pub material: MaterialUniforms,
}

impl TileMaterial {
    /// The slots and scalars of `entry`, or the engine defaults for a world
    /// naming no material.
    pub fn from_entry(entry: Option<MaterialEntry>) -> Self {
        match entry {
            Some(entry) => Self {
                texture_slot: entry.albedo_slot,
                normal_map_slot: entry.normal_map_slot,
                material: entry.uniforms,
            },
            None => Self {
                texture_slot: 0,
                normal_map_slot: render_types::NO_NORMAL_MAP_SLOT,
                material: MaterialUniforms::DEFAULT,
            },
        }
    }

    /// A planet's ground material: its entry, tinted at range by the canopy's
    /// far field when it has one.
    pub fn for_planet(entry: Option<MaterialEntry>, far_field: Option<&CanopyRange>) -> Self {
        let mut material = Self::from_entry(entry);
        if let Some(far) = far_field {
            material.material.far_field = render_types::FarFieldTint::of(far.packed());
        }
        material
    }
}

/// The largest mesh a tile builds: `(vertices, indices)`.
pub fn tile_mesh_size() -> (usize, usize) {
    let n = crate::planet::TILE_CELLS as usize;
    ((n + 1) * (n + 1), n * n * 6)
}

impl LoadPool for PlanetTiles {
    type Request = TileBuild;
    type Result = TileBuilt;

    fn mailbox(&mut self) -> &mut Mailbox<TileBuild, TileBuilt> {
        &mut self.mail
    }

    // A refused new mesh is asked for again when the tile next stands
    // stale.
    fn request_refused(&mut self, build: TileBuild) {
        let tile = build.tile;
        if build.generation != self.generation {
            return;
        }
        match self.tiles.get_mut(&tile) {
            Some(Tile::Pending { .. }) => {
                self.tiles.remove(&tile);
            }
            Some(Tile::Resident { touch, .. }) => {
                touch.rebuilding = false;
                touch.stale = true;
                self.any_stale = true;
            }
            None => {}
        }
    }

    // A new mesh that failed for good leaves the old one drawn.
    fn load_failed(&mut self, build: TileBuild) {
        let tile = build.tile;
        if build.generation != self.generation {
            return;
        }
        match self.tiles.get_mut(&tile) {
            Some(Tile::Pending { .. }) => {
                tracing::warn!("planet tile {tile:?} failed to build for good");
                self.tiles.remove(&tile);
                self.failed.insert(tile);
            }
            Some(Tile::Resident { touch, .. }) => {
                tracing::warn!("planet tile {tile:?} failed to build again for its edits");
                touch.rebuilding = false;
            }
            None => {}
        }
    }
}

impl PlanetTiles {
    /// Tiles of `planet` drawn with `material`.
    pub fn new(planet: &PlanetFrame, material: TileMaterial) -> Self {
        let lod = TileLod::new(planet.shape.clone(), crate::planet::FINEST_CELL_SIZE);
        Self {
            lod,
            finest_cell_size: crate::planet::FINEST_CELL_SIZE,
            mail: Mailbox::default(),
            generation: 0,
            tiles: BTreeMap::new(),
            failed: BTreeSet::new(),
            bounds: TileBoundsCache::new(planet.shape.clone()),
            planned: None,
            planned_frames: 0,
            residency_changed: false,
            morph_eye: None,
            morph_sent: None,
            placed_in: planet.frame,
            occluder_for: None,
            edit_revision: 0,
            any_stale: false,
            rebuilds: Vec::new(),
            texture_slot: material.texture_slot,
            normal_map_slot: material.normal_map_slot,
            material: material.material,
        }
    }

    /// `(resident, pending)` tile counts.
    pub fn stats(&self) -> (usize, usize) {
        let pending = self
            .tiles
            .values()
            .filter(|t| matches!(t, Tile::Pending { .. }))
            .count();
        (self.tiles.len() - pending, pending)
    }

    /// Forget a tile whose add the backend refused, freeing its slot.
    pub fn roll_back(&mut self, tile: TileId, slots: &mut RenderSlots) {
        if let Some(Tile::Resident { draw, .. }) = self.tiles.remove(&tile) {
            slots.free_draw(draw);
            self.residency_changed = true;
        }
    }

    /// One frame: take the finished meshes, re-place every tile if the frame
    /// moved, and show, hide, drop and request tiles for `camera`. The
    /// planet's edits last changed as `edited` says.
    #[expect(
        clippy::float_cmp,
        reason = "change detection: any change must re-upload"
    )]
    pub fn step(
        &mut self,
        planet: &PlanetView,
        ops: &mut RenderOps,
        slots: &mut RenderSlots,
        frames: (u64, u64),
    ) {
        let (frame, retire_frame) = frames;
        let (camera, edited) = (planet.camera, planet.edited);
        let planet = &planet.frame;
        if !planet.shape.same_generated(&self.lod.shape) {
            self.reshape(planet.shape.clone(), ops, slots, retire_frame);
        } else if planet.shape != self.lod.shape {
            let area = edited
                .filter(|e| e.revision == self.edit_revision.wrapping_add(1))
                .map_or(EditArea::EVERYWHERE, |e| e.area);
            self.touch(&planet.shape, area);
        }
        if let Some(e) = edited {
            self.edit_revision = e.revision;
        }
        if planet.frame != self.placed_in {
            self.placed_in = planet.frame;
            for tile in self.tiles.values() {
                if let Tile::Resident { draw, origin, .. } = *tile {
                    let model = planet.frame.model_at(origin);
                    ops.record(move |backend| {
                        if let Err(e) = backend.set_chunk_model(draw, model) {
                            tracing::warn!("planet tile rebase: {e}");
                        }
                    });
                }
            }
        }
        if self.occluder_for.as_ref() != Some(planet) {
            self.occluder_for = Some(planet.clone());
            let occluder = HorizonOccluder::of(planet);
            ops.record(move |backend| backend.set_horizon_occluder(Some(occluder)));
        }
        self.add_finished(planet, ops, slots, (frame, retire_frame));

        if camera.finest_cell_size != self.finest_cell_size {
            self.finest_cell_size = camera.finest_cell_size;
            self.lod = TileLod::new(planet.shape.clone(), camera.finest_cell_size);
            self.planned = None;
        }
        let view = camera.view(&planet.frame);
        self.rebuild_touched(view.eye);
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
            let plan = plan(
                &self.lod,
                &self.tiles,
                &self.failed,
                &view,
                &mut self.bounds,
            );
            self.apply(plan, ops, slots, retire_frame);
        }
        self.send_morph(planet, &view, ops);
    }

    // Drop every tile of the old shape for a planet reshaped to `shape`; the
    // next plan asks for the new shape's, and the renderer brings what it
    // keeps of the shape along.
    fn reshape(
        &mut self,
        shape: PlanetShape,
        ops: &mut RenderOps,
        slots: &mut RenderSlots,
        retire_frame: u64,
    ) {
        for tile in core::mem::take(&mut self.tiles).into_values() {
            if let Tile::Resident { draw, .. } = tile {
                slots.free_draw(draw);
                ops.record(move |backend| {
                    if let Err(e) = backend.remove_chunk_mesh(draw, retire_frame) {
                        tracing::warn!("planet tile remove: {e}");
                    }
                });
            }
        }
        self.failed.clear();
        self.generation = self.generation.wrapping_add(1);
        self.lod = TileLod::new(shape.clone(), self.finest_cell_size);
        self.bounds = TileBoundsCache::new(shape.clone());
        self.planned = None;
        self.residency_changed = true;
        self.morph_sent = None;
        self.any_stale = false;
        ops.record(move |backend| {
            if let Some(live) = backend.live_edit() {
                live.reshape_planet(&shape);
            }
        });
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
                        tracing::warn!("planet tile remove: {e}");
                    }
                });
            }
        }
        for id in plan.request {
            self.mail.request(TileBuild {
                tile: id,
                shape: self.lod.shape.clone(),
                generation: self.generation,
            });
            self.tiles.insert(
                id,
                Tile::Pending {
                    touch: Touch::new(id),
                },
            );
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
    // the plan shows it; a tile's new mesh for edits takes its old one's
    // place, drawn as it was.
    fn add_finished(
        &mut self,
        planet: &PlanetFrame,
        ops: &mut RenderOps,
        slots: &mut RenderSlots,
        (frame, retire_frame): (u64, u64),
    ) {
        for built in self.mail.take_results() {
            let (id, mesh) = (built.tile, built.mesh);
            if built.generation != self.generation {
                continue;
            }
            let (touch, replaces) = match self.tiles.get(&id) {
                Some(Tile::Pending { touch }) => (*touch, None),
                Some(&Tile::Resident {
                    draw,
                    visible,
                    seams,
                    touch,
                    ..
                }) if touch.rebuilding => (touch, Some((draw, visible, seams))),
                _ => continue,
            };
            if replaces.is_none() {
                self.residency_changed = true;
            }
            let (visible, seams) =
                replaces.map_or_else(|| (false, TileSeams::default()), |r| (r.1, r.2));
            let dst = slots.allocate_draw();
            let draw = dst.slot();
            self.tiles.insert(
                id,
                Tile::Resident {
                    draw,
                    origin: mesh.origin,
                    visible,
                    seams,
                    touch: Touch {
                        rebuilding: false,
                        ..touch
                    },
                },
            );
            let old = replaces.map(|r| r.0);
            if let Some(old) = old {
                slots.free_draw(old);
            }
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
                            origin_altitude: mesh.origin_altitude,
                        }),
                        bounds: Some(mesh.bounds),
                    },
                    dst,
                );
                match added {
                    Ok(()) => {
                        backend.update_visibility(draw, visible);
                        if seams != TileSeams::default() {
                            backend.set_planet_tile_seams(draw, seams);
                        }
                    }
                    Err(e) => {
                        tracing::warn!("planet tile add: {e}");
                        out.memory_pressure |=
                            matches!(e, error::RenderError::OutOfDeviceMemory(_));
                        out.failures.push(OpFailure::PlanetTileAdd { tile: id });
                    }
                }
                if let Some(old) = old
                    && let Err(e) = backend.remove_chunk_mesh(old, retire_frame)
                {
                    tracing::warn!("planet tile remove: {e}");
                }
            });
        }
    }
}

#[cfg(test)]
mod tests;
