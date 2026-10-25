use super::*;

fn object(v: &Value) -> Map<String, Value> {
    v.as_object().cloned().unwrap()
}

// A planet authoring a sea, two covers and a grass layer.
fn authored() -> Map<String, Value> {
    working(&object(&json!({
        "$id": "world",
        "seed": 95,
        "land_share": 0.4,
        "sea": {"level": 1.5, "water": "sea"},
        "material": "turf",
        "covers": [
            {"material": "seabed", "rule": {"altitude": [null, -7.0], "altitude_edge": 2.5}},
            {"material": "sand"}
        ],
        "layers": [{"grass": "meadow", "rule": {"altitude": [2.5, null]}}]
    })))
}

fn keys(rows: &[Row]) -> Vec<&str> {
    rows.iter()
        .filter_map(|r| r.field().map(|f| f.key.as_str()))
        .collect()
}

fn row_of<'a>(rows: &'a [Row], key: &str) -> &'a Row {
    rows.iter()
        .find(|r| r.field().is_some_and(|f| f.key == key))
        .unwrap_or_else(|| panic!("no row for {key}"))
}

// Every arg a Planet authors in JSON has a row, down to each ground rule's
// leaves and the sea's: a field the panel cannot reach breaks this test.
#[test]
fn every_planet_arg_has_a_row() {
    let rows = rows(Some(&authored()));
    let keys = keys(&rows);
    let leaf_paths = |prefix: &str, v: &Value| -> Vec<String> {
        match v {
            Value::Object(m) => m.keys().map(|k| format!("{prefix}.{k}")).collect(),
            _ => vec![prefix.to_string()],
        }
    };
    let mut expected: Vec<String> = Vec::new();
    for (key, default) in form::base_args(TY) {
        match key.as_str() {
            "center" => expected.extend((0..3).map(|i| format!("center.{i}"))),
            // The edits file is the paint tools' to write, not a row's.
            "covers" | "layers" | "sea" | "edits" => {}
            _ => expected.extend(leaf_paths(&key, &default)),
        }
    }
    let sea = serde_json::to_value(concinnity_engine::components::PlanetSea::default()).unwrap();
    expected.extend(leaf_paths("sea", &sea));
    for rule in ["covers.0.rule", "covers.1.rule", "layers.0.rule"] {
        for leaf in ["altitude.0", "altitude.1", "altitude_edge"] {
            expected.push(format!("{rule}.{leaf}"));
        }
        for leaf in ["slope.0", "slope.1", "slope_edge"] {
            expected.push(format!("{rule}.{leaf}"));
        }
    }
    expected.extend(["covers.0.material", "covers.1.material", "layers.0.grass"].map(String::from));
    for path in &expected {
        assert!(keys.contains(&path.as_str()), "no row edits {path}");
    }
    // A cover's and a layer's every field is in the expected list too.
    let cover =
        serde_json::to_value(concinnity_engine::components::GroundCover::default()).unwrap();
    assert_eq!(leaf_paths("c", &cover).len(), 2, "material + rule");
    let rule = rule_defaults();
    assert_eq!(rule.len(), 4, "two bands and two edges");
}

// A planet's density masks are refused by the cook, so the panel offers none.
#[test]
fn a_layer_offers_no_density_mask() {
    let rows = rows(Some(&authored()));
    assert!(!keys(&rows).iter().any(|k| k.contains("density_mask")));
}

#[test]
fn a_world_without_a_planet_shows_only_the_add_row() {
    let rows = rows(None);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].kind, RowKind::AddPlanet);
}

// The sections run in order, each under its heading.
#[test]
fn sections_run_in_order() {
    let rows = rows(Some(&authored()));
    let headers: Vec<Section> = rows
        .iter()
        .filter_map(|r| match r.kind {
            RowKind::Header(s) => Some(s),
            _ => None,
        })
        .collect();
    assert_eq!(
        headers,
        [
            Section::Shape,
            Section::Coasts,
            Section::Sea,
            Section::Ground,
            Section::Grass,
            Section::Gravity
        ]
    );
}

// The seed's row rerolls; the sea's heading removes it; a planet without one
// shows the add row in its place.
#[test]
fn the_seed_rerolls_and_the_sea_comes_and_goes() {
    let rows = rows(Some(&authored()));
    assert_eq!(row_of(&rows, "seed").buttons[0], Some(Button::Reroll));
    let sea = rows
        .iter()
        .find(|r| r.kind == RowKind::Header(Section::Sea))
        .unwrap();
    assert_eq!(sea.buttons[0], Some(Button::RemoveSea));
    assert_eq!(row_of(&rows, "sea.level").field().unwrap().initial, "1.5");
    let water = row_of(&rows, "sea.water").field().unwrap();
    assert_eq!(water.initial, "sea");
    assert!(matches!(water.kind, FieldKind::Ref { .. }));

    let dry = press(&authored(), Button::RemoveSea, None).unwrap();
    let rows = rows_of(&dry);
    assert!(rows.iter().any(|r| r.kind == RowKind::AddSea));
    assert!(!keys(&rows).contains(&"sea.level"));
    let wet = with_sea(&dry, "ocean");
    assert_eq!(value_at(&wet, "sea.water"), Some(&json!("ocean")));
}

fn rows_of(args: &Map<String, Value>) -> Vec<Row> {
    rows(Some(&working(args)))
}

// The lists show their length against the cap, each entry its own heading
// with the buttons that make sense for it.
#[test]
fn list_rows_carry_their_buttons() {
    let rows = rows(Some(&authored()));
    let covers = rows
        .iter()
        .find(|r| {
            matches!(
                r.kind,
                RowKind::List {
                    list: List::Covers,
                    ..
                }
            )
        })
        .unwrap();
    assert_eq!(covers.caption, "covers (2/4)");
    assert_eq!(covers.buttons[0], Some(Button::Add(List::Covers)));
    let item = |i: usize| {
        rows.iter()
            .find(|r| {
                r.kind
                    == RowKind::Item {
                        list: List::Covers,
                        index: i,
                    }
            })
            .unwrap()
    };
    assert_eq!(
        item(0).buttons,
        [Some(Button::Remove(List::Covers, 0)), None]
    );
    assert_eq!(
        item(1).buttons,
        [
            Some(Button::Remove(List::Covers, 1)),
            Some(Button::Raise(List::Covers, 1))
        ]
    );
}

// A full list offers no add, and a press past the cap changes nothing.
#[test]
fn covers_stop_at_the_cap() {
    let mut args = authored();
    while let Some(next) = press(&args, Button::Add(List::Covers), None) {
        args = next;
    }
    assert_eq!(args["covers"].as_array().unwrap().len(), MAX_COVERS);
    let rows = rows_of(&args);
    let head = rows
        .iter()
        .find(|r| {
            matches!(
                r.kind,
                RowKind::List {
                    list: List::Covers,
                    ..
                }
            )
        })
        .unwrap();
    assert_eq!(head.buttons[0], None);
}

#[test]
fn list_presses_move_and_remove_entries() {
    let args = authored();
    let raised = press(&args, Button::Raise(List::Covers, 1), None).unwrap();
    assert_eq!(value_at(&raised, "covers.0.material"), Some(&json!("sand")));
    assert!(press(&args, Button::Raise(List::Covers, 0), None).is_none());
    let removed = press(&args, Button::Remove(List::Covers, 0), None).unwrap();
    assert_eq!(removed["covers"].as_array().unwrap().len(), 1);
    assert!(press(&args, Button::Remove(List::Covers, 5), None).is_none());
    // A layer needs grass to grow.
    assert!(press(&args, Button::Add(List::Layers), None).is_none());
    let grown = press(&args, Button::Add(List::Layers), Some("meadow")).unwrap();
    assert_eq!(value_at(&grown, "layers.1.grass"), Some(&json!("meadow")));
    assert_eq!(
        value_at(&grown, "layers.1.rule.slope_edge"),
        Some(&json!(5.0))
    );
}

#[test]
fn reroll_draws_a_new_seed_from_the_old_one() {
    let a = press(&authored(), Button::Reroll, None).unwrap();
    let b = press(&authored(), Button::Reroll, None).unwrap();
    assert_eq!(a["seed"], b["seed"], "the same seed rerolls the same way");
    assert_ne!(a["seed"], json!(95));
    assert_ne!(next_seed(0), 0);
}

// Typed boxes land at their paths; an empty bound opens it; a box that does
// not parse keeps what was there.
#[test]
fn assemble_writes_the_boxes() {
    let args = authored();
    let rows = rows(Some(&args));
    let mut texts: Vec<String> = rows
        .iter()
        .map(|r| r.field().map(|f| f.initial.clone()).unwrap_or_default())
        .collect();
    let at = |key: &str| {
        rows.iter()
            .position(|r| r.field().is_some_and(|f| f.key == key))
            .unwrap()
    };
    texts[at("amplitude")] = "55".into();
    texts[at("covers.0.rule.altitude.1")] = String::new();
    texts[at("covers.1.rule.slope.1")] = "30".into();
    texts[at("radius")] = "not a number".into();
    let out = assemble(&args, &rows, &texts);
    assert_eq!(out["amplitude"], json!(55.0));
    assert_eq!(
        value_at(&out, "covers.0.rule.altitude.1"),
        Some(&Value::Null)
    );
    assert_eq!(value_at(&out, "covers.1.rule.slope.1"), Some(&json!(30.0)));
    assert_eq!(out["radius"], args["radius"]);
    assert_eq!(out["$id"], json!("world"), "the id is kept");
    assert_eq!(
        value_at(&out, "covers.0.material"),
        Some(&json!("seabed")),
        "refs untouched"
    );
}

// The panel's output is a Planet the schema accepts.
#[test]
fn assembled_args_validate() {
    let args = authored();
    let rows = rows(Some(&args));
    let texts: Vec<String> = rows
        .iter()
        .map(|r| r.field().map(|f| f.initial.clone()).unwrap_or_default())
        .collect();
    let out = assemble(&args, &rows, &texts);
    form::validate(TY, "world", &out).expect("a valid Planet");
}

#[test]
fn a_required_reference_offers_no_none() {
    let rows = rows(Some(&authored()));
    let mut water = row_of(&rows, "sea.water").field().unwrap().clone();
    set_ref_options(&mut water, false, &["lake".into(), "sea".into()]);
    assert_eq!(water.variants, ["lake", "sea"]);
    assert_eq!(water.variant_idx, 1);
    assert_eq!(next_ref(&water), Some(json!("lake")));
    let mut material = row_of(&rows, "material").field().unwrap().clone();
    set_ref_options(&mut material, true, &["turf".into()]);
    assert_eq!(material.variants, [form::NONE_LABEL, "turf"]);
    assert_eq!(
        next_ref(&material),
        Some(Value::Null),
        "past the last is none"
    );
}

#[test]
fn set_value_writes_only_existing_paths() {
    let mut args = authored();
    assert!(set_value(
        &mut args,
        "covers.1.rule.altitude.0",
        json!(-2.0)
    ));
    assert_eq!(
        value_at(&args, "covers.1.rule.altitude.0"),
        Some(&json!(-2.0))
    );
    assert!(!set_value(&mut args, "covers.9.material", json!("x")));
    assert!(!set_value(&mut args, "nothing.here", json!(1)));
}

// A planet with every list full shows exactly the most rows the pools hold.
#[test]
fn a_full_planet_fills_the_row_pool() {
    let mut args = authored();
    while let Some(next) = press(&args, Button::Add(List::Covers), None) {
        args = next;
    }
    while let Some(next) = press(&args, Button::Add(List::Layers), Some("meadow")) {
        args = next;
    }
    assert_eq!(rows_of(&args).len(), MAX_ROWS);
}

// Only what the edit changed is written back: the defaults the working args
// fill in stay out of the authored line.
#[test]
fn an_edit_writes_back_only_what_changed() {
    let authored = object(&json!({
        "seed": 95,
        "sea": {"level": 0.0, "water": "sea"},
        "covers": [{"material": "sand"}]
    }));
    let before = working(&authored);
    let rerolled = press(&before, Button::Reroll, None).unwrap();
    let out = authored_edit(&authored, &before, &rerolled);
    assert_eq!(
        out["covers"], authored["covers"],
        "no rule defaults written"
    );
    assert!(!out.contains_key("radius"));
    assert_ne!(out["seed"], json!(95));
    let dry = press(&before, Button::RemoveSea, None).unwrap();
    assert!(!authored_edit(&authored, &before, &dry).contains_key("sea"));
}
