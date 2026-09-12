// src/editor/hook/tests/drive/billboard_tests.rs
//
// The billboard drive's picking (`hook/drive/billboard.rs`): the light a
// billboard click selects and the transform it seeds, which hit wins when a
// billboard and a mesh overlap, and the session's hide and lock flags applying
// to billboards as they do to meshes.

use concinnity_core::components::PointLight;
use concinnity_core::components::Transform;
use concinnity_core::ecs::World;
use concinnity_host::thread::asset_id;

use crate::editor::hook::tests::fixtures::{click_at, entry, hook, pick_world};

use crate::editor::viewport::billboards;

// Billboard test rig: the pick rig plus a PointLight entity indexed by name
// (as the loaders' name -> entity index would) and the injected billboard
// pools.
fn billboard_world(
    light_pos: [f32; 3],
    picks: Vec<(asset_id::AssetId, [f32; 3], [f32; 3])>,
) -> World {
    let mut world = pick_world([0.0; 3], picks);
    for s in billboards::sprites() {
        world.add_component(s);
    }
    let entity = world.push(PointLight {
        position: light_pos,
        ..Default::default()
    });
    let id = asset_id::intern("lamp");
    let mut by_name = std::collections::BTreeMap::new();
    by_name.insert(id, entity);
    world.insert_resource(concinnity_core::ecs::EntityByName(by_name));
    world
}

fn lamp_entry(pos: [f32; 3]) -> serde_json::Value {
    serde_json::json!({"name": "lamp", "type": "PointLight", "args": {"position": pos}})
}

// Clicking a light's billboard selects it by name through the normal pick
// flow, and the tick seeds the Transform the gizmo needs onto its entity.
#[test]
fn billboard_click_selects_the_light_and_seeds_its_transform() {
    asset_id::reset_interner();
    let mut world = billboard_world([0.0, 0.0, -5.0], Vec::new());
    let mut h = hook(vec![lamp_entry([0.0, 0.0, -5.0])]);

    // The light projects to the viewport center (camera at origin facing -Z).
    click_at(&mut world, &mut h, [640.0, 360.0]);
    assert_eq!(h.selection.active(), Some("lamp"), "the icon press selects");
    assert!(!h.panel_open && !h.form_open(), "an icon press opens no UI");

    // The seeded Transform mirrors the authored position, so the gizmo's
    // member resolve works on the light.
    let entity = world
        .resource::<concinnity_core::ecs::EntityByName>()
        .unwrap()
        .0
        .values()
        .next()
        .copied()
        .unwrap();
    let t = world.get::<Transform>(entity).unwrap();
    assert_eq!(t.position, [0.0, 0.0, -5.0]);
    assert!(
        h.gizmo_layout(&world, [1280.0, 720.0]).is_some(),
        "the translate gizmo anchors on the selected light"
    );
}

// A mesh AABB in front of the billboard's anchor keeps the press; one behind
// it loses to the icon.
#[test]
fn billboard_and_mesh_overlap_prefers_the_nearer_hit() {
    asset_id::reset_interner();
    let wall = asset_id::intern("wall");
    // Wall at depth 2..3, light at depth 5: the wall is nearer.
    let mut world = billboard_world(
        [0.0, 0.0, -5.0],
        vec![(wall, [-1.0, -1.0, -3.0], [1.0, 1.0, -2.0])],
    );
    let mut h = hook(vec![entry("wall", "Sprite"), lamp_entry([0.0, 0.0, -5.0])]);
    click_at(&mut world, &mut h, [640.0, 360.0]);
    assert_eq!(h.selection.active(), Some("wall"), "the nearer mesh wins");

    // Wall at depth 9..10, light at depth 5: the icon is nearer.
    asset_id::reset_interner();
    let wall = asset_id::intern("wall");
    let mut world = billboard_world(
        [0.0, 0.0, -5.0],
        vec![(wall, [-1.0, -1.0, -10.0], [1.0, 1.0, -9.0])],
    );
    let mut h = hook(vec![entry("wall", "Sprite"), lamp_entry([0.0, 0.0, -5.0])]);
    click_at(&mut world, &mut h, [640.0, 360.0]);
    assert_eq!(h.selection.active(), Some("lamp"), "the nearer icon wins");
}

// Editor-hidden billboards neither draw nor pick; locked ones stay visible
// but pass the press through, both matching the mesh pick's rules.
#[test]
fn hidden_and_locked_billboards_follow_the_pick_rules() {
    asset_id::reset_interner();
    let mut world = billboard_world([0.0, 0.0, -5.0], Vec::new());
    let mut h = hook(vec![lamp_entry([0.0, 0.0, -5.0])]);
    h.locked_assets.insert("lamp".to_string());
    click_at(&mut world, &mut h, [640.0, 360.0]);
    assert_eq!(h.selection.active(), None, "a locked icon is pick-through");
    assert!(h.marquee.is_some(), "the click fell through to empty space");

    asset_id::reset_interner();
    let mut world = billboard_world([0.0, 0.0, -5.0], Vec::new());
    let mut h = hook(vec![lamp_entry([0.0, 0.0, -5.0])]);
    h.hidden_assets.insert("lamp".to_string());
    click_at(&mut world, &mut h, [640.0, 360.0]);
    assert_eq!(h.selection.active(), None, "a hidden asset draws no icon");
    assert!(h.marquee.is_some(), "the click fell through to empty space");
}
