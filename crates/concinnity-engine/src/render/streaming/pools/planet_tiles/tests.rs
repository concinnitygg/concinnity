use super::*;

fn pending(tile: TileId) -> Tile {
    Tile::Pending {
        touch: Touch::new(tile),
    }
}
use crate::planet::{CubeFace, tile_mesh};
use alloc::vec;
use concinnity_core::math::SaturatingFrom;

fn lod() -> TileLod {
    TileLod::new(
        PlanetShape {
            center: [0.0, -50_000.0, 0.0],
            radius: 50_000.0,
            amplitude: 30.0,
            feature_size: 2_000.0,
            octaves: 6,
            seed: 5,
            continents: crate::planet::Continents::NONE,
            edits: None,
        },
        crate::planet::FINEST_CELL_SIZE,
    )
}

const CAMERA: DVec3 = [0.0, 40.0, 0.0];

// The camera's view of the ground decides how far its levels reach.
#[test]
fn the_camera_s_view_sets_its_ranges() {
    let cells = f64::from(crate::planet::TILE_CELLS) * crate::planet::RANGE_CELL_POINTS;
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
        touch: Touch::new(TileId::root(CubeFace(0))),
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
        &BTreeSet::new(),
        &view(CAMERA),
        &mut TileBoundsCache::new(l.shape.clone()),
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
        .map(|(i, &f)| (TileId::root(f), resident(u32::saturating_from(i), false)))
        .collect();
    let p = plan(
        &l,
        &tiles,
        &BTreeSet::new(),
        &view(CAMERA),
        &mut TileBoundsCache::new(l.shape.clone()),
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
        .map(|(i, &f)| (TileId::root(f), resident(u32::saturating_from(i), true)))
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
    tiles.insert(gone, pending(gone));
    // Every child of the top face resident, so the top root hides.
    for (i, c) in TileId::root(CubeFace(2)).children().iter().enumerate() {
        tiles.insert(*c, resident(10 + u32::saturating_from(i), false));
    }
    let p = plan(
        &l,
        &tiles,
        &BTreeSet::new(),
        &view(CAMERA),
        &mut TileBoundsCache::new(l.shape.clone()),
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
    let mut bounds = TileBoundsCache::new(l.shape.clone());
    let camera = [0.0, 32.0, 0.0];
    let sel = l.select(&view(camera), &|_| true, &mut bounds);
    let tiles: BTreeMap<TileId, Tile> = sel
        .wanted
        .iter()
        .enumerate()
        .map(|(i, &t)| (t, resident(u32::saturating_from(i), false)))
        .collect();
    let p = plan(&l, &tiles, &BTreeSet::new(), &view(camera), &mut bounds);
    let horizon = Horizon::of_planet(camera, &l.shape, 0.0).unwrap();
    let mut hidden = 0;
    for t in sel.draw.iter().map(|d| d.id) {
        let b = bounds.get(t);
        let near = dvec3::length(dvec3::sub(b.center, camera)) - b.radius <= HORIZON_SHADOW_REACH;
        let behind = horizon.hides_sphere(b.center, b.radius);
        assert_eq!(p.show.contains(&t), near || !behind, "{t:?}");
        hidden += usize::from(!p.show.contains(&t));
    }
    assert!(hidden > 0, "the far side is hidden");
    assert!(p.show.len() > hidden / 4, "the near side shows");
}

// A tile whose build failed for good leaves tracking and is never asked
// for again; a refused one is asked for again.
#[test]
fn a_failed_tile_is_never_requested_again() {
    let l = lod();
    let top = TileId::root(CubeFace(2));
    let frame = PlanetFrame {
        shape: l.shape.clone(),
        frame: LocalFrame::AUTHORED,
    };
    let mut tiles = PlanetTiles::new(&frame, TileMaterial::from_entry(None));
    let build = TileBuild {
        tile: top,
        shape: l.shape.clone(),
        generation: 0,
    };
    tiles.tiles.insert(top, pending(top));
    tiles.load_failed(build.clone());
    assert!(!tiles.tiles.contains_key(&top));
    let p = plan(
        &l,
        &tiles.tiles,
        &tiles.failed,
        &view(CAMERA),
        &mut TileBoundsCache::new(l.shape.clone()),
    );
    assert!(!p.request.contains(&top));

    let mut refused = PlanetTiles::new(&frame, TileMaterial::from_entry(None));
    refused.tiles.insert(top, pending(top));
    refused.request_refused(build);
    assert!(refused.failed.is_empty());
}

// A reshaped planet drops every tile, tells the renderer, and discards a
// mesh built for the old shape when it arrives.
#[test]
fn a_reshape_drops_every_tile_and_discards_stale_meshes() {
    use crate::render::backend::recording::{Call, recording_backend};
    let l = lod();
    let frame = PlanetFrame {
        shape: l.shape.clone(),
        frame: LocalFrame::AUTHORED,
    };
    let mut tiles = PlanetTiles::new(&frame, TileMaterial::from_entry(None));
    let mut ops = RenderOps::default();
    let mut slots = RenderSlots::new(0, true, &[]);
    let top = TileId::root(CubeFace(2));
    tiles.tiles.insert(top, pending(top));
    let stale = TileBuild {
        tile: top,
        shape: l.shape.clone(),
        generation: tiles.generation,
    }
    .build();
    let reshaped = PlanetShape {
        seed: l.shape.seed + 1,
        ..l.shape
    };
    tiles.reshape(reshaped.clone(), &mut ops, &mut slots, 0);
    assert!(tiles.tiles.is_empty());
    assert_eq!(tiles.lod.shape, reshaped);

    tiles.tiles.insert(top, pending(top));
    tiles.mail.deliver(stale);
    let moved = PlanetFrame {
        shape: reshaped,
        frame: frame.frame,
    };
    tiles.add_finished(&moved, &mut ops, &mut slots, (0, 2));
    assert_eq!(
        tiles.tiles.get(&top),
        Some(&pending(top)),
        "stale mesh dropped"
    );

    let (calls, mut backend) = recording_backend();
    ops.replay(&mut backend);
    assert!(calls.lock().calls.contains(&Call::ReshapePlanet));
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

// A material entry lends its slots and scalars; no entry draws with the
// engine defaults.
#[test]
fn a_tile_material_comes_from_its_entry_or_the_defaults() {
    let mut uniforms = MaterialUniforms::DEFAULT;
    uniforms.roughness = 0.25;
    let entry = MaterialEntry {
        albedo_slot: 5,
        normal_map_slot: 6,
        uniforms,
        shader_bucket: 0,
    };
    let named = TileMaterial::from_entry(Some(entry));
    assert_eq!((named.texture_slot, named.normal_map_slot), (5, 6));
    assert_eq!(named.material.roughness, 0.25);
    let unnamed = TileMaterial::from_entry(None);
    assert_eq!(
        (unnamed.texture_slot, unnamed.normal_map_slot),
        (0, render_types::NO_NORMAL_MAP_SLOT)
    );
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
    let top = l.shape.radius + l.shape.highest_height();
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
