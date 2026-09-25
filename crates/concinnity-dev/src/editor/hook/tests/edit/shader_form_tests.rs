//! The Shader form (`hook/edit/shader_form.rs`) on the generic add / edit
//! form: creating a Shader with its starters, its Materials and its files;
//! the name validated live; renaming with every reference following; adding
//! and dropping the vertex file; assigning Materials; and moving the world
//! default. Each commit is one undo step.

use concinnity_core::components::ShaderStage;
use concinnity_core::ecs::World;
use concinnity_core::gfx::render_types::MAX_SHADER_BUCKETS;
use std::path::Path;

use super::shader_fixtures::{args, click, in_project, key, material, shader, shader_names, write};
use crate::editor::hook::EditorHook;
use crate::editor::hook::FormTarget;
use crate::editor::hook::tests::fixtures::{hook, press_modal, set_field, world_with_fields};
use crate::editor::modal;
use crate::editor::panels::form_extras::{ExtraControl, ExtraRow};
use crate::editor::panels::form_panel::{self, FormAction};
use crate::editor::panels::registry::PanelKey;
use crate::editor::panels::shader_list::{self, RowKind};
use crate::editor::panels::shader_templates;

// A world with the form's fields and the dialog's, so a leave question can be
// answered while the form is open.
fn form_world() -> World {
    let mut world = world_with_fields();
    for id in modal::all_field_ids() {
        world.push_identified(id, concinnity_core::components::TextInput::default());
    }
    world
}

fn plain_material(name: &str) -> serde_json::Value {
    serde_json::json!({"type": "Material", "args": {"$id": name}})
}

fn type_name(world: &mut World, name: &str) {
    set_field(world, form_panel::NAME_INPUT, name);
}

fn rows(h: &EditorHook, world: &World) -> Vec<ExtraRow> {
    h.form_extras_data(world).0
}

fn blocked(h: &EditorHook, world: &World) -> Option<String> {
    h.form_extras_data(world).1
}

// Press the row captioned `caption` that is a checkbox (or, with `choice`, a
// choice).
fn press(h: &mut EditorHook, world: &mut World, caption: &str, choice: bool) {
    let row = rows(h, world)
        .into_iter()
        .find(|r| {
            r.caption == caption && matches!(r.control, ExtraControl::Choice { .. }) == choice
        })
        .unwrap_or_else(|| panic!("no '{caption}' row"));
    h.apply_form(FormAction::PressExtra(row.id), world);
}

fn detail(h: &EditorHook, world: &World, caption: &str) -> Option<String> {
    rows(h, world)
        .into_iter()
        .find(|r| r.caption == caption)
        .and_then(|r| r.detail)
}

fn confirm(h: &mut EditorHook, world: &mut World) {
    h.apply_form(FormAction::Confirm, world);
}

fn toasts(h: &EditorHook) -> Vec<String> {
    h.notifier
        .stack()
        .cards
        .into_iter()
        .map(|c| c.message)
        .collect()
}

// "+ New Shader" opens the form empty; a blank name blocks it, and Create
// writes each file from its starter, assigns the Materials picked, and opens
// the fragment file. Undo takes back the entry and the assignment in one step
// and leaves the files.
#[test]
fn a_new_shader_is_created_through_its_form() {
    in_project(|root| {
        let shaders = root.join("assets/shaders");
        std::fs::create_dir_all(&shaders).unwrap();
        std::fs::write(shaders.join("water.hlsl"), "someone else's").unwrap();
        let mut h = hook(vec![plain_material("sea"), plain_material("rock")]);
        let mut world = form_world();

        click(&mut h, &mut world, RowKind::New);
        assert_eq!(h.form.selected_type.as_deref(), Some("Shader"));
        assert_eq!(h.form.target, FormTarget::New);
        assert_eq!(h.form.host, PanelKey::Shaders);
        assert!(
            h.form.fields.is_empty(),
            "the file fields are the form's own"
        );

        type_name(&mut world, "   ");
        let reason = blocked(&h, &world).unwrap();
        assert!(reason.contains("Enter a name"), "{reason}");
        confirm(&mut h, &mut world);
        assert!(h.entries.iter().all(|e| e["type"] != "Shader"));
        assert!(h.form_open(), "the form stays open with the reason");

        type_name(&mut world, " water ");
        assert_eq!(blocked(&h, &world), None);
        assert_eq!(
            detail(&h, &world, "fragment").as_deref(),
            Some("assets/shaders/water_1.hlsl"),
            "the preview numbers past a file already there"
        );
        press(&mut h, &mut world, "vertex", false);
        press(&mut h, &mut world, "fragment", true);
        press(&mut h, &mut world, "vertex", true);
        press(&mut h, &mut world, "sea", false);
        confirm(&mut h, &mut world);

        assert!(!h.form_open());
        assert_eq!(shader_names(&h), ["water"]);
        let water = args(&h, "water");
        assert_eq!(water["fragment"], "assets/shaders/water_1.hlsl");
        assert_eq!(water["vertex"], "assets/shaders/water_vertex.hlsl");
        let read = |p: &str| std::fs::read_to_string(root.join(p)).unwrap();
        assert_eq!(
            read("assets/shaders/water_1.hlsl"),
            shader_templates::FRAGMENT[1].text
        );
        assert_eq!(
            read("assets/shaders/water_vertex.hlsl"),
            shader_templates::VERTEX[1].text
        );
        assert_eq!(read("assets/shaders/water.hlsl"), "someone else's");
        assert_eq!(args(&h, "sea")["shader"], "water");
        assert!(args(&h, "rock").get("shader").is_none());
        let src = h.shaders.source.as_ref().unwrap();
        assert_eq!(src.key, key("water", ShaderStage::Fragment));
        assert!(
            toasts(&h)
                .iter()
                .any(|t| t.contains("Added Shader 'water'"))
        );

        h.undo(&mut world);
        assert!(shader_names(&h).is_empty());
        assert!(args(&h, "sea").get("shader").is_none());
        assert!(!h.can_undo(), "one step");
        assert!(
            root.join("assets/shaders/water_1.hlsl").exists(),
            "files stay"
        );
    });
}

// A name another entry declares, a reserved label, or a world at the Shader
// limit blocks the form with the reason on it, and a confirm adds nothing.
#[test]
fn a_taken_name_or_the_limit_blocks_the_form() {
    let mut h = hook(vec![
        shader("lit", Path::new("/cn-none/lit.hlsl")),
        plain_material("rock"),
    ]);
    let mut world = form_world();
    h.open_form(&mut world, "Shader".to_string(), FormTarget::New);
    for (name, why) in [
        ("lit", "already taken"),
        ("rock", "already taken"),
        ("Shader#3", "reserved"),
    ] {
        type_name(&mut world, name);
        let reason = blocked(&h, &world).unwrap();
        assert!(reason.contains(why), "{name}: {reason}");
    }
    confirm(&mut h, &mut world);
    assert_eq!(shader_names(&h), ["lit"]);
    assert!(h.form.error.as_deref().unwrap().contains("reserved"));

    let full: Vec<serde_json::Value> = (0..MAX_SHADER_BUCKETS)
        .map(|i| shader(&format!("s{i}"), Path::new("/cn-none/s.hlsl")))
        .collect();
    let mut h = hook(full);
    h.open_form(&mut world, "Shader".to_string(), FormTarget::New);
    type_name(&mut world, "more");
    assert!(blocked(&h, &world).unwrap().contains("No room"));
    confirm(&mut h, &mut world);
    assert_eq!(shader_list::shader_count(&h.entries), MAX_SHADER_BUCKETS);
    assert!(!h.dirty);
}

// A heading opens its Shader's form; renaming there sets the `$id` and every
// Material naming it follows, in one undo step. Its file keeps its name, and a
// source panel open on it follows the name through the rename and its undo.
#[test]
fn a_rename_through_the_form_rewrites_every_reference() {
    let dir = tempfile::tempdir().unwrap();
    let lit = write(dir.path(), "lit.hlsl");
    let water = write(dir.path(), "water.hlsl");
    let mut h = hook(vec![
        shader("lit", &lit),
        shader("water", &water),
        material("sea", "water"),
        material("pond", "water"),
        material("stone", "lit"),
    ]);
    let mut world = form_world();
    h.open_shader_file(key("water", ShaderStage::Fragment));

    click(&mut h, &mut world, RowKind::Header(1));
    assert!(h.form.target.is_edit());
    assert_eq!(
        crate::editor::widget::field_text(&world, form_panel::NAME_INPUT),
        "water",
        "seeded with the current name"
    );
    type_name(&mut world, "lit");
    assert!(blocked(&h, &world).unwrap().contains("already taken"));
    type_name(&mut world, "ocean");
    confirm(&mut h, &mut world);
    assert_eq!(shader_names(&h), ["lit", "ocean"]);
    assert_eq!(args(&h, "sea")["shader"], "ocean");
    assert_eq!(args(&h, "pond")["shader"], "ocean");
    assert_eq!(args(&h, "stone")["shader"], "lit");
    assert_eq!(
        args(&h, "ocean")["fragment"],
        water.to_string_lossy().as_ref()
    );
    assert_eq!(h.shaders.source.as_ref().unwrap().key.shader, "ocean");
    assert!(
        toasts(&h)
            .iter()
            .any(|t| t == "Renamed Shader 'water' to 'ocean' and the 2 references to it")
    );

    h.undo(&mut world);
    assert_eq!(shader_names(&h), ["lit", "water"]);
    assert_eq!(args(&h, "sea")["shader"], "water");
    assert_eq!(h.shaders.source.as_ref().unwrap().key.shader, "water");
    assert!(!h.can_undo(), "the rename was one step");
}

// Checking vertex on an existing Shader adds a starter vertex file and opens
// it; unchecking drops the declaration and leaves the file, closing it in the
// source panel first (Cancel on the unsaved-changes question keeps
// everything, Discard goes on with the Apply).
#[test]
fn stages_add_and_drop_the_vertex_file() {
    in_project(|root| {
        let lit = write(root, "lit.hlsl");
        let mut h = hook(vec![shader("lit", &lit)]);
        let mut world = form_world();

        click(&mut h, &mut world, RowKind::Header(0));
        assert!(!rows(&h, &world).iter().any(|r| r.caption == "Starters"));
        press(&mut h, &mut world, "vertex", false);
        assert_eq!(
            detail(&h, &world, "vertex").as_deref(),
            Some("assets/shaders/lit_vertex.hlsl")
        );
        confirm(&mut h, &mut world);
        let declared = args(&h, "lit")["vertex"].as_str().unwrap().to_string();
        assert_eq!(declared, "assets/shaders/lit_vertex.hlsl");
        let path = root.join(&declared);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            shader_templates::VERTEX[0].text
        );
        let vertex = key("lit", ShaderStage::Vertex);
        assert_eq!(h.shaders.source.as_ref().unwrap().key, vertex);

        h.shaders.source.as_mut().unwrap().area.type_char('x');
        click(&mut h, &mut world, RowKind::Header(0));
        press(&mut h, &mut world, "vertex", false);
        confirm(&mut h, &mut world);
        assert!(h.modal.is_some(), "asked over the unsaved vertex file");
        press_modal(&mut h, &mut world, "Cancel");
        assert!(args(&h, "lit").get("vertex").is_some(), "Cancel keeps it");
        assert!(h.form_open() && h.shaders.source.is_some());

        confirm(&mut h, &mut world);
        press_modal(&mut h, &mut world, "Discard");
        assert!(args(&h, "lit").get("vertex").is_none());
        assert!(h.shaders.source.is_none());
        assert!(!h.form_open());
        assert!(path.exists(), "the file stays");

        h.undo(&mut world);
        assert_eq!(args(&h, "lit")["vertex"], declared.as_str());
    });
}

// Used by lists every Material: checking one assigns it (from another Shader
// too), unchecking one falls it back, all in the Apply's one undo step.
#[test]
fn used_by_assigns_and_unassigns_in_one_step() {
    let mut h = hook(vec![
        shader("lit", Path::new("/cn-none/lit.hlsl")),
        shader("water", Path::new("/cn-none/water.hlsl")),
        material("a", "water"),
        material("b", "lit"),
        plain_material("c"),
    ]);
    let mut world = form_world();
    click(&mut h, &mut world, RowKind::Header(1));
    assert_eq!(detail(&h, &world, "b").as_deref(), Some("names 'lit'"));
    for m in ["a", "b", "c"] {
        press(&mut h, &mut world, m, false);
    }
    confirm(&mut h, &mut world);
    assert!(args(&h, "a").get("shader").is_none());
    assert_eq!(args(&h, "b")["shader"], "water");
    assert_eq!(args(&h, "c")["shader"], "water");

    h.undo(&mut world);
    assert_eq!(args(&h, "a")["shader"], "water");
    assert_eq!(args(&h, "b")["shader"], "lit");
    assert!(args(&h, "c").get("shader").is_none());
    assert!(!h.can_undo(), "one step");
}

// World default moves the Shader first, saying what that does; unchecking it
// on the default moves it behind the next one. The rest keep their order.
#[test]
fn the_world_default_moves_through_the_form() {
    let mut h = hook(vec![
        shader("a", Path::new("/cn-none/a.hlsl")),
        plain_material("m"),
        shader("b", Path::new("/cn-none/b.hlsl")),
        shader("c", Path::new("/cn-none/c.hlsl")),
    ]);
    let mut world = form_world();
    click(&mut h, &mut world, RowKind::Header(2));
    press(&mut h, &mut world, "World default", false);
    assert!(
        rows(&h, &world)
            .iter()
            .any(|r| r.caption == "Materials naming no Shader switch to it")
    );
    confirm(&mut h, &mut world);
    assert_eq!(shader_names(&h), ["c", "a", "b"]);
    assert!(h.shader_rows()[0].badge.is_some(), "listed as the default");

    click(&mut h, &mut world, RowKind::Header(0));
    press(&mut h, &mut world, "World default", false);
    assert!(
        rows(&h, &world)
            .iter()
            .any(|r| r.caption == "'a' becomes the world default")
    );
    confirm(&mut h, &mut world);
    assert_eq!(shader_names(&h), ["a", "c", "b"]);

    h.undo(&mut world);
    h.undo(&mut world);
    assert_eq!(shader_names(&h), ["a", "b", "c"]);
}

// The Shaders panel's form shows while that panel does; a form the Assets
// panel opens afterwards belongs to the Assets panel again.
#[test]
fn the_form_belongs_to_the_panel_that_opened_it() {
    let mut h = hook(vec![shader("lit", Path::new("/cn-none/lit.hlsl"))]);
    let mut world = form_world();
    let edit = crate::editor::panels::registry::panel(PanelKey::Edit);
    h.shaders.open = true;
    click(&mut h, &mut world, RowKind::Header(0));
    assert!(edit.is_open(&h), "shown beside the Shaders panel");
    h.shaders.open = false;
    assert!(!edit.is_open(&h));

    h.open_form(&mut world, "PointLight".to_string(), FormTarget::New);
    assert_eq!(h.form.host, PanelKey::Assets);
    h.panel_open = true;
    assert!(edit.is_open(&h));
}
