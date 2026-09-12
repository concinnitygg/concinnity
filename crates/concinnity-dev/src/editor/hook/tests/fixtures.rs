// src/editor/hook/tests/fixtures.rs
//
// The fixtures every hook test companion shares: a hook over a given entry
// list, the worlds a tick reads its input and typed fields from, the entry
// literals, and the seeded asset tree the panel rows come from. Nothing here
// asserts anything -- each companion brings in what it needs.

use crate::editor::modal;
use concinnity_cook::authoring::world::write_world_jsonl;
use concinnity_core::components::Camera3D;
use concinnity_core::components::FrameInput;
use concinnity_core::components::InputKey;
use concinnity_core::components::TextInput;
use concinnity_core::components::Transform;
use concinnity_core::ecs::Entity;
use concinnity_core::ecs::PickEntry;
use concinnity_core::ecs::PickIndex;
use concinnity_core::ecs::World;
use concinnity_host::thread::asset_id;

use crate::debug_hook::DebugHook;
use crate::editor::behavior;
use crate::editor::behavior::panel::BehaviorAction;
use crate::editor::hook::{EditorHook, entry_name, entry_type};

use crate::editor::panels::asset_tree::{self, TreeGroup};

use crate::editor::panels::form_panel;

use crate::editor::panels::panel::{self, PanelAction};
use crate::editor::panels::registry::{self, PanelKey};

use crate::editor::viewport::gizmo;
use crate::editor::viewport::highlight;
use crate::editor::viewport::marquee;
use crate::editor::widget;

pub(in crate::editor::hook) fn hook(entries: Vec<serde_json::Value>) -> EditorHook {
    EditorHook::new("unused.jsonl".to_string(), entries)
}

// The shared title-bar / close-button rects the routing derives for a panel
// (the shared geometry lives in `widget`).
pub(in crate::editor::hook) fn title_rect_of(
    h: &EditorHook,
    key: PanelKey,
    vp: [f32; 2],
) -> [f32; 4] {
    let o = h.origin(key, vp);
    widget::title_rect(o, registry::panel(key).size(h)[0])
}

pub(in crate::editor::hook) fn close_rect_of(
    h: &EditorHook,
    key: PanelKey,
    vp: [f32; 2],
) -> [f32; 4] {
    widget::close_rect(title_rect_of(h, key, vp))
}

use crate::test_support::isolate_state_dir;

// A world holding just a FrameInput, for driving `tick` directly.
pub(in crate::editor::hook) fn world_with_input(input: FrameInput) -> World {
    let mut world = World::new();
    world.add_component(input);
    world
}

// A world with the injected typed fields, for the add / edit flow (the
// combo filter, the form's name heading, and its arg-input pool).
pub(in crate::editor::hook) fn world_with_fields() -> World {
    let mut world = World::new();
    for id in panel::all_field_ids()
        .into_iter()
        .chain(form_panel::all_field_ids())
    {
        world.add_component(TextInput {
            asset_id: id,
            ..Default::default()
        });
    }
    world
}

pub(in crate::editor::hook) fn set_field(world: &mut World, id: asset_id::AssetId, text: &str) {
    for t in world.query_mut::<TextInput>() {
        if t.asset_id == id {
            t.content = text.to_string();
            break;
        }
    }
}

pub(in crate::editor::hook) fn entry(name: &str, ty: &str) -> serde_json::Value {
    serde_json::json!({"name": name, "type": ty, "args": {}})
}

pub(in crate::editor::hook) fn entry_with_args(
    name: &str,
    ty: &str,
    args: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({"name": name, "type": ty, "args": args})
}

// Seed the cooked tree the panel rows come from, without paying for a real
// world expansion: the working entries under `World`, plus any generated groups
// the test needs. Mirrors what `refresh_tree_if_needed` builds, and clears
// `tree_stale` so the frame drive does not overwrite it.
pub(in crate::editor::hook) fn seed_tree(h: &mut EditorHook, extra: Vec<TreeGroup>) {
    let world_group = TreeGroup {
        label: asset_tree::WORLD_GROUP.to_string(),
        assets: h
            .entries
            .iter()
            .map(|e| asset_tree::TreeAsset {
                name: entry_name(e).unwrap_or_default().to_string(),
                asset_type: entry_type(e).unwrap_or_default().to_string(),
                badge: asset_tree::Badge::Authored,
                promote: None,
            })
            .collect(),
    };
    h.tree_groups = std::iter::once(world_group).chain(extra).collect();
    h.tree_unfolded = (0..h.tree_groups.len()).collect();
    h.tree_stale = false;
}

// One generated group, as a scene import's or injection pass's output would
// appear: every asset promotable from the entry the expansion produced.
pub(in crate::editor::hook) fn generated_group(label: &str, assets: &[(&str, &str)]) -> TreeGroup {
    TreeGroup {
        label: label.to_string(),
        assets: assets
            .iter()
            .map(|(name, ty)| asset_tree::TreeAsset {
                name: name.to_string(),
                asset_type: ty.to_string(),
                badge: asset_tree::Badge::Imported,
                promote: Some(serde_json::json!({
                    "name": name, "type": ty, "args": {},
                })),
            })
            .collect(),
    }
}

// The (group, index) a row click on `name` resolves to.
pub(in crate::editor::hook) fn row_of(h: &EditorHook, name: &str) -> (usize, usize) {
    h.tree_groups
        .iter()
        .enumerate()
        .find_map(|(gi, g)| {
            g.assets
                .iter()
                .position(|a| a.name == name)
                .map(|ai| (gi, ai))
        })
        .unwrap_or_else(|| panic!("{name} is not in the seeded tree"))
}

// Click the tree row for `name` (the panel's plain select-and-edit action).
pub(in crate::editor::hook) fn click_row(h: &mut EditorHook, name: &str, world: &mut World) {
    let (g, i) = row_of(h, name);
    h.apply_panel(PanelAction::SelectRow(g, i), world);
}

// Overwrite the world's FrameInput in place (tick reads the live component).
pub(in crate::editor::hook) fn set_input(world: &mut World, input: FrameInput) {
    if let Some(i) = world.query_mut::<FrameInput>().last() {
        *i = input;
    } else {
        world.add_component(input);
    }
}

// Viewport picking test rig: a camera at `cam_pos` facing -Z (yaw 0, pitch 0),
// the injected typed fields (the pick flows open the edit form), and a
// PickIndex resource carrying the given (id, bb_min, bb_max) entries.
pub(in crate::editor::hook) fn pick_world(
    cam_pos: [f32; 3],
    picks: Vec<(asset_id::AssetId, [f32; 3], [f32; 3])>,
) -> World {
    let mut world = world_with_fields();
    world.add_component(Camera3D {
        position: cam_pos,
        view_matrix: concinnity_core::gfx::camera::view_matrix(cam_pos, 0.0, 0.0),
        fov_y_degrees: 90.0,
        near: 0.05,
        far: 200.0,
        yaw: 0.0,
        pitch: 0.0,
        desired_move: [0.0; 3],
        jump_requested: false,
        interact_requested: false,
        controller: None,
    });
    for s in highlight::outline_sprites() {
        world.add_component(s);
    }
    world.add_component(marquee::rect_sprite());
    world.insert_resource(PickIndex {
        entries: picks
            .into_iter()
            .map(|(asset_id, bb_min, bb_max)| PickEntry {
                asset_id,
                bb_min,
                bb_max,
            })
            .collect(),
    });
    world
}

pub(in crate::editor::hook) fn click_at(world: &mut World, h: &mut EditorHook, pos: [f32; 2]) {
    click_at_mod(world, h, pos, false);
}

pub(in crate::editor::hook) fn click_at_mod(
    world: &mut World,
    h: &mut EditorHook,
    pos: [f32; 2],
    shift: bool,
) {
    set_input(
        world,
        FrameInput {
            viewport: [1280.0, 720.0],
            mouse_x: pos[0],
            mouse_y: pos[1],
            left_click: true,
            left_button_down: true,
            shift,
            ..Default::default()
        },
    );
    h.tick(world);
}

// A button-up tick at `pos`: ends an armed marquee (or gizmo drag).
pub(in crate::editor::hook) fn release_at(world: &mut World, h: &mut EditorHook, pos: [f32; 2]) {
    set_input(
        world,
        FrameInput {
            viewport: [1280.0, 720.0],
            mouse_x: pos[0],
            mouse_y: pos[1],
            ..Default::default()
        },
    );
    h.tick(world);
}

// A held-button move tick at `pos`: advances an armed marquee or gizmo drag.
pub(in crate::editor::hook) fn drag_to(world: &mut World, h: &mut EditorHook, pos: [f32; 2]) {
    set_input(
        world,
        FrameInput {
            viewport: [1280.0, 720.0],
            mouse_x: pos[0],
            mouse_y: pos[1],
            left_button_down: true,
            ..Default::default()
        },
    );
    h.tick(world);
}

pub(in crate::editor::hook) fn drag_input(pos: [f32; 2], held: bool) -> FrameInput {
    FrameInput {
        viewport: [1280.0, 720.0],
        mouse_x: pos[0],
        mouse_y: pos[1],
        left_button_down: held,
        ..Default::default()
    }
}

pub(in crate::editor::hook) fn behavior(name: &str, args: serde_json::Value) -> serde_json::Value {
    serde_json::json!({"name": name, "type": "Behavior", "args": args})
}

// An open Behavior panel over `entries`, with its value field injected.
pub(in crate::editor::hook) fn behavior_session(
    entries: Vec<serde_json::Value>,
) -> (EditorHook, World) {
    let mut world = World::new();
    for id in behavior::panel::all_field_ids() {
        world.add_component(TextInput {
            asset_id: id,
            ..Default::default()
        });
    }
    let mut h = hook(entries);
    registry::panel(PanelKey::Behavior).toggle(&mut h, &mut world);
    (h, world)
}

pub(in crate::editor::hook) fn select_behavior(h: &mut EditorHook, world: &mut World, label: &str) {
    let i = behavior_row(h, label);
    h.apply_behavior_action(BehaviorAction::Select(i), world, [0.0, 0.0]);
}

// The args of the open behavior, for asserting on what an action wrote.
pub(in crate::editor::hook) fn open_args(h: &EditorHook) -> serde_json::Value {
    h.behavior_args()
}

// One press of the header's removal chip: the first arms it, the second carries
// it out.
pub(in crate::editor::hook) fn press_remove(h: &mut EditorHook, world: &mut World) {
    h.apply_behavior_action(BehaviorAction::Remove, world, [0.0, 0.0]);
}

// Type into the name field, as the engine's text-input system would.
pub(in crate::editor::hook) fn type_name(world: &mut World, text: &str) {
    widget::seed_field(world, behavior::panel::NAME_INPUT, text);
}

pub(in crate::editor::hook) fn story_key_input(key: InputKey) -> FrameInput {
    FrameInput {
        captured_key: Some(key),
        viewport: [1280.0, 720.0],
        ..Default::default()
    }
}

// Two props with live Transforms at `s1` / `s2` (AABB half-extent `half`),
// wired like `gizmo_rig`, for the multi-select flows.
pub(in crate::editor::hook) fn two_prop_rig(
    s1: [f32; 3],
    s2: [f32; 3],
    half: f32,
) -> (World, Entity, Entity, EditorHook) {
    asset_id::reset_interner();
    let a = asset_id::intern("box_a");
    let b = asset_id::intern("box_b");
    let bb = |s: [f32; 3]| {
        (
            [s[0] - half, s[1] - half, s[2] - half],
            [s[0] + half, s[1] + half, s[2] + half],
        )
    };
    let (min1, max1) = bb(s1);
    let (min2, max2) = bb(s2);
    let mut world = pick_world([0.0; 3], vec![(a, min1, max1), (b, min2, max2)]);
    let transform = |p: [f32; 3]| Transform {
        position: p,
        rotation_deg: [0.0; 3],
        scale: [1.0; 3],
    };
    let e1 = world.push(transform(s1));
    let e2 = world.push(transform(s2));
    let mut by_name = std::collections::BTreeMap::new();
    by_name.insert(a, e1);
    by_name.insert(b, e2);
    world.insert_resource(concinnity_core::ecs::EntityByName(by_name));
    for s in gizmo::sprites() {
        world.add_component(s);
    }
    let entry = |name: &str, p: [f32; 3]| serde_json::json!({ "name": name, "type": "Prop", "args": { "position": p } });
    let h = hook(vec![entry("box_a", s1), entry("box_b", s2)]);
    (world, e1, e2, h)
}

// Side-by-side props with a clear gap (so neither center ray clips the other
// box): box_a projects around [200, 600], box_b around [424, 598], both in
// the panel-free lower-left screen region.
pub(in crate::editor::hook) const SIDE_A: [f32; 3] = [-6.11, -3.3, -5.0];

pub(in crate::editor::hook) const SIDE_B: [f32; 3] = [-3.0, -3.3, -5.0];

pub(in crate::editor::hook) fn playing_hook(entries: Vec<serde_json::Value>) -> EditorHook {
    let mut h = hook(entries);
    h.sim.toggle_play();
    h
}

pub(in crate::editor::hook) fn shape_world_entries() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({"name": "body", "type": "SkinnedMesh", "args": {
            "morph_target_names": ["jaw+", "jaw-", "muscle"],
            "skeleton": [{"name": "root", "parent": -1}, {"name": "thigh_l", "parent": 0}]
        }}),
        serde_json::json!({"name": "body_shape", "type": "CharacterShape", "args": {
            "target": "body", "sliders": [{"name": "muscle", "value": 0.3}]
        }}),
    ]
}

// A world whose GraphicsConfig pulls in companions, so the expansion has
// something to show without needing a scene file on disk.
pub(in crate::editor::hook) fn expandable_hook() -> EditorHook {
    isolate_state_dir();
    let mut h = hook(vec![serde_json::json!({
        "name": "gfx", "type": "GraphicsConfig", "args": {}
    })]);
    h.panel_open = true;
    h
}

// The first asset the build generates that an authored line could override,
// as (group, index, name).
pub(in crate::editor::hook) fn a_promotable_asset(h: &EditorHook) -> (usize, usize, String) {
    h.tree_groups
        .iter()
        .enumerate()
        .find_map(|(gi, g)| {
            g.assets
                .iter()
                .enumerate()
                .find_map(|(ai, a)| a.promote.as_ref().map(|_| (gi, ai, a.name.clone())))
        })
        .expect("an injected default can be promoted")
}

// Escape arrives as its own one-frame pulse rather than as a `InputKey`.
pub(in crate::editor::hook) fn behavior_escape_input() -> FrameInput {
    FrameInput {
        escape: true,
        viewport: [1280.0, 720.0],
        ..Default::default()
    }
}

pub(in crate::editor::hook) fn press_behavior_key(
    h: &mut EditorHook,
    world: &mut World,
    key: InputKey,
) {
    h.behavior_keys(world, &story_key_input(key));
}

// The outline row index of the first row with `label`.
pub(in crate::editor::hook) fn behavior_row(h: &EditorHook, label: &str) -> usize {
    h.behavior_rows()
        .iter()
        .position(|r| r.label == label)
        .unwrap_or_else(|| panic!("no `{label}` row"))
}

// A project rooted at `dir`, sharing the machine-wide build cache. The caller
// holds the process lock: this moves the session-wide project.
pub(in crate::editor::hook) fn open_project(dir: &std::path::Path) {
    crate::project::open(
        concinnity_host::store::paths::StateTree::at(dir).with_cache(
            concinnity_testing::shared_cache_dir("concinnity-dev-tests-cache"),
        ),
    );
}

pub(in crate::editor::hook) fn prop_entry(name: &str) -> serde_json::Value {
    serde_json::json!({"name": name, "type": "Prop", "args": {}})
}

// Write a world file with `entries` and pin its mtime, so a listing's order is
// the one under test rather than whatever the filesystem's resolution gives.
pub(in crate::editor::hook) fn write_world(
    dir: &std::path::Path,
    name: &str,
    entries: &[serde_json::Value],
    at_secs: u64,
) -> std::path::PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    let path = dir.join(format!("{name}.jsonl"));
    std::fs::write(&path, write_world_jsonl(entries).unwrap()).unwrap();
    let file = std::fs::File::options().write(true).open(&path).unwrap();
    file.set_modified(std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(at_secs))
        .unwrap();
    path
}

pub(in crate::editor::hook) fn hook_at(
    path: &std::path::Path,
    entries: Vec<serde_json::Value>,
) -> EditorHook {
    let mut h = EditorHook::new(path.to_string_lossy().into_owned(), entries);
    h.refresh_worlds();
    h
}

// The naming prompt's field, which is the only place a world is named now.
pub(in crate::editor::hook) fn world_with_name_field() -> World {
    let mut world = World::new();
    for id in modal::all_field_ids() {
        world.add_component(TextInput {
            asset_id: id,
            ..Default::default()
        });
    }
    world
}

pub(in crate::editor::hook) fn set_world_name(world: &mut World, text: &str) {
    widget::seed_field(world, modal::NAME_INPUT, text);
}

pub(in crate::editor::hook) fn world_row_index(h: &EditorHook, name: &str) -> usize {
    h.worlds_rows
        .iter()
        .position(|r| r.name == name)
        .unwrap_or_else(|| panic!("{name} is not listed"))
}

pub(in crate::editor::hook) fn world_names(h: &EditorHook) -> Vec<String> {
    h.worlds_rows.iter().map(|r| r.name.clone()).collect()
}

// Press one of the open dialog's buttons.
pub(in crate::editor::hook) fn press_modal(h: &mut EditorHook, world: &mut World, label: &str) {
    let i = button_index(h, label);
    let state = h.modal.as_ref().unwrap();
    let (count, field) = (state.buttons.len(), state.field);
    let r = modal::button_rect(modal::panel_rect(VP, field), count, i);
    let input = FrameInput {
        left_click: true,
        mouse_x: r[0] + 2.0,
        mouse_y: r[1] + 2.0,
        viewport: VP,
        ..Default::default()
    };
    assert!(h.route_modal_click(&input, VP, world));
}

pub(in crate::editor::hook) fn button_index(h: &EditorHook, label: &str) -> usize {
    let buttons = &h.modal.as_ref().expect("a dialog is open").buttons;
    buttons
        .iter()
        .position(|b| b.label == label)
        .unwrap_or_else(|| panic!("no '{label}' button"))
}

// The viewport every panel test lays out against.
pub(in crate::editor::hook) const VP: [f32; 2] = [1280.0, 720.0];
