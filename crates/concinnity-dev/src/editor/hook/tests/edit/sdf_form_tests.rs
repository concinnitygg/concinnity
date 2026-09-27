//! An SdfVolume form's extras (`hook/edit/sdf_form.rs`): the Shaders panel's
//! create form switching to a volume when a field kind is chosen, the volume
//! and its starter file a create writes, where the new volume lands, the
//! Shader limit leaving the field kinds open, and an edited volume listing who
//! else reads its field.

use concinnity_core::ecs::World;
use concinnity_core::gfx::render_types::MAX_SHADER_BUCKETS;
use std::path::Path;

use super::shader_fixtures::{
    args, blocked, click, confirm, detail, form_world, in_project, press, rows, shader, toasts,
    type_name,
};
use crate::editor::hook::tests::fixtures::{VP, hook, pick_world};
use crate::editor::hook::{EditorHook, FormTarget};
use crate::editor::panels::form_extras::ExtraControl;
use crate::editor::panels::registry::PanelKey;
use crate::editor::panels::sdf_templates;
use crate::editor::panels::shader_list::RowKind;
use crate::editor::panels::shader_source::SourceKey;
use crate::editor::widget;

fn kind(h: &EditorHook, world: &World) -> usize {
    match rows(h, world)
        .into_iter()
        .find(|r| r.caption == "Kind")
        .expect("a Kind row")
        .control
    {
        ExtraControl::Choice { selected, .. } => selected,
        _ => panic!("the Kind row is a choice"),
    }
}

fn field_keys(h: &EditorHook) -> Vec<&str> {
    h.form.fields.iter().map(|f| f.key.as_str()).collect()
}

// Choosing "SDF field" in "+ New Shader"'s form turns it into a volume's,
// keeping the typed name and the panel; Create writes the volume and its
// starter file and opens the file. Undo takes back the volume, not the file.
#[test]
fn a_field_kind_creates_a_volume_with_its_starter_file() {
    in_project(|root| {
        let mut h = hook(Vec::new());
        let mut world = form_world();
        click(&mut h, &mut world, RowKind::New);
        type_name(&mut world, "blob");
        assert_eq!(kind(&h, &world), 0);

        press(&mut h, &mut world, "Kind", true);
        assert_eq!(h.form.selected_type.as_deref(), Some("SdfVolume"));
        assert_eq!(h.form.target, FormTarget::New);
        assert_eq!(h.form.host, PanelKey::Shaders);
        assert_eq!(
            widget::field_text(&world, crate::editor::panels::form_panel::NAME_INPUT),
            "blob"
        );
        assert_eq!(kind(&h, &world), 1);
        let keys = field_keys(&h);
        assert!(keys.contains(&"center"), "{keys:?}");
        assert!(!keys.contains(&"fragment_shader") && !keys.contains(&"volumetric"));
        assert_eq!(
            detail(&h, &world, "Field file").as_deref(),
            Some("shaders/blob.hlsl")
        );
        assert_eq!(blocked(&h, &world), None);
        confirm(&mut h, &mut world);

        assert!(!h.form_open());
        let blob = args(&h, "blob");
        assert_eq!(blob["fragment_shader"], "shaders/blob.hlsl");
        assert_eq!(blob["volumetric"], false);
        let file = root.join("assets/shaders/blob.hlsl");
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            sdf_templates::SURFACE[0].text
        );
        let src = h.shaders.source.as_ref().expect("the file is open");
        assert!(matches!(&src.key, SourceKey::Field { path } if Path::new(path) == file));
        assert_eq!(src.area.text(), sdf_templates::SURFACE[0].text);
        assert!(
            toasts(&h)
                .iter()
                .any(|t| t == "Added SDF field 'blob' (shaders/blob.hlsl)")
        );

        h.undo(&mut world);
        assert!(h.entries.iter().all(|e| e["type"] != "SdfVolume"));
        assert!(!h.can_undo(), "one step");
        assert!(file.exists(), "the file stays");
    });
}

// The volumetric kind sets the flag and starts from a volumetric starter; one
// more press is a Shader again.
#[test]
fn the_kinds_cycle_through_volumetric_and_back_to_a_shader() {
    in_project(|root| {
        let mut h = hook(Vec::new());
        let mut world = form_world();
        click(&mut h, &mut world, RowKind::New);
        type_name(&mut world, "fog");
        press(&mut h, &mut world, "Kind", true);
        press(&mut h, &mut world, "Kind", true);
        assert_eq!(h.form.selected_type.as_deref(), Some("SdfVolume"));
        assert_eq!(kind(&h, &world), 2);
        press(&mut h, &mut world, "Starter", true);
        confirm(&mut h, &mut world);
        assert_eq!(args(&h, "fog")["volumetric"], true);
        assert_eq!(
            std::fs::read_to_string(root.join("assets/shaders/fog.hlsl")).unwrap(),
            sdf_templates::VOLUMETRIC[1].text
        );

        click(&mut h, &mut world, RowKind::New);
        press(&mut h, &mut world, "Kind", true);
        press(&mut h, &mut world, "Kind", true);
        press(&mut h, &mut world, "Kind", true);
        assert_eq!(h.form.selected_type.as_deref(), Some("Shader"));
        assert_eq!(kind(&h, &world), 0);
    });
}

// A new volume lands where the create menu would land a click at the middle
// of the view: here on the face of the wall ahead of the camera.
#[test]
fn a_new_volume_lands_at_the_middle_of_the_view() {
    let wall = concinnity_core::ecs::asset_id::AssetId(0x7a11);
    let mut world = pick_world(
        [0.0; 3],
        vec![(wall, [-10.0, -10.0, -3.0], [10.0, 10.0, -2.0])],
    );
    let mut h = hook(Vec::new());
    h.viewport = VP;
    click(&mut h, &mut world, RowKind::New);
    press(&mut h, &mut world, "Kind", true);
    let center = h.form.args["center"].clone();
    assert_eq!(center[0], 0.0, "{center}");
    assert_eq!(center[2], -2.0, "{center}");
}

// At the Shader limit "+ New Shader" opens straight onto a field kind, whose
// create the limit does not block.
#[test]
fn the_shader_limit_leaves_the_field_kinds_open() {
    let full: Vec<serde_json::Value> = (0..MAX_SHADER_BUCKETS)
        .map(|i| shader(&format!("s{i}"), Path::new("/cn-none/s.hlsl")))
        .collect();
    let mut h = hook(full);
    let mut world = form_world();
    let last = h.shader_rows().last().cloned().unwrap();
    assert_eq!(last.kind, RowKind::New);
    click(&mut h, &mut world, RowKind::New);
    assert_eq!(h.form.selected_type.as_deref(), Some("SdfVolume"));
    assert_eq!(h.form.host, PanelKey::Shaders);
    type_name(&mut world, "blob");
    assert_eq!(blocked(&h, &world), None);
}

// A volume row opens that volume's own form, which keeps every field and
// names the other volumes reading its file.
#[test]
fn a_volume_row_opens_its_form_naming_who_shares_the_field() {
    let volume = |name: &str| {
        serde_json::json!({"type": "SdfVolume", "args": {
            "$id": name, "fragment_shader": "/cn-none/blob.hlsl",
        }})
    };
    let mut h = hook(vec![volume("left"), volume("right")]);
    let mut world = form_world();
    click(
        &mut h,
        &mut world,
        RowKind::FieldVolume("right".to_string()),
    );
    assert_eq!(h.form.selected_type.as_deref(), Some("SdfVolume"));
    assert!(matches!(h.form.target, FormTarget::Entry(_)));
    assert_eq!(h.form.host, PanelKey::Shaders);
    assert!(field_keys(&h).contains(&"fragment_shader"));
    let captions: Vec<String> = rows(&h, &world).into_iter().map(|r| r.caption).collect();
    assert_eq!(captions, ["Field file", "Also read by", "left"]);
}
