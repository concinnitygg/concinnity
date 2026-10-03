use super::*;

fn asset(name: &str, asset_type: RegisteredType, args: serde_json::Value) -> WorldJsonlAsset {
    WorldJsonlAsset {
        id: name.to_string(),
        asset_type,
        args,
    }
}

#[test]
fn prop_typed_and_structured_refs_are_unioned() {
    // `material` is a typed field; `mesh` is structured (polymorphic
    // MeshSource, kept in the hand impl).
    let refs = referenced_names(&asset(
        "p",
        RegisteredType::Prop,
        serde_json::json!({"mesh":"box","material":"mat"}),
    ));
    assert!(refs.contains(&"box".to_string()));
    assert!(refs.contains(&"mat".to_string()));
}

#[test]
fn material_texture_slots_come_from_the_resource_registry() {
    let refs = referenced_names(&asset(
        "m",
        RegisteredType::Material,
        serde_json::json!({"albedo":"tex_a","normal_map":"tex_n"}),
    ));
    assert_eq!(refs, vec!["tex_a".to_string(), "tex_n".to_string()]);
}

#[test]
fn model_submesh_list_is_structured() {
    let refs = referenced_names(&asset(
        "mdl",
        RegisteredType::Model,
        serde_json::json!({"meshes":[{"mesh":"m0","material":"mat0"},{"mesh":"m1"}]}),
    ));
    assert!(refs.contains(&"m0".to_string()));
    assert!(refs.contains(&"mat0".to_string()));
    assert!(refs.contains(&"m1".to_string()));
}

#[test]
fn list_and_nested_references_are_derived() {
    let refs = referenced_names(&asset(
        "c",
        RegisteredType::VoxelChunk,
        serde_json::json!({"palette":["stone","dirt"]}),
    ));
    assert_eq!(refs, vec!["stone".to_string(), "dirt".to_string()]);
    let refs = referenced_names(&asset(
        "cam",
        RegisteredType::Camera3D,
        serde_json::json!({"controller":{"follow":{"target":"hero"}}}),
    ));
    assert_eq!(refs, vec!["hero".to_string()]);
}

#[test]
fn null_and_absent_fields_are_omitted() {
    let refs = referenced_names(&asset(
        "p",
        RegisteredType::Prop,
        serde_json::json!({"model":null}),
    ));
    assert!(refs.is_empty());
}

#[test]
fn a_type_without_references_has_no_refs() {
    let light = asset(
        "x",
        RegisteredType::PointLight,
        serde_json::json!({"model": "m"}),
    );
    assert!(referenced_names(&light).is_empty());
}

#[test]
fn a_retarget_rewrites_only_fields_that_may_name_the_type() {
    let mut material = serde_json::json!({"type": "Material", "args": {
        "$id": "m", "shader": "water", "albedo": "water",
    }});
    assert_eq!(
        retarget_references(&mut material, "Shader", "water", Some("sea")),
        1
    );
    assert_eq!(material["args"]["shader"], "sea");
    assert_eq!(material["args"]["albedo"], "water", "a texture slot");
    assert_eq!(retarget_references(&mut material, "Shader", "sea", None), 1);
    assert!(material["args"].get("shader").is_none());
    let mut unknown = serde_json::json!({"type": "Nope", "args": {"shader": "sea"}});
    assert_eq!(retarget_references(&mut unknown, "Shader", "sea", None), 0);
}

// A rename follows the structured references too, and only the strings the
// build reads as references: a label that happens to read the old name
// stays.
#[test]
fn a_rename_follows_structured_references() {
    let mesh = serde_json::json!({"generator": "box"});
    let mut prop = serde_json::json!({"type": "Prop", "args": {
        "$id": "p", "mesh": "box", "material": "box",
    }});
    assert_eq!(
        rename_references(
            &mut prop,
            RegisteredType::ProceduralMesh,
            &mesh,
            "box",
            "crate"
        ),
        1
    );
    assert_eq!(prop["args"]["mesh"], "crate");
    assert_eq!(prop["args"]["material"], "box", "a Material slot");

    let mut behavior = serde_json::json!({"type": "Behavior", "args": {
        "$id": "b",
        "on": {"enter": "gate"},
        "do": [
            {"despawn": {"target": {"named": "gate"}}},
            {"log": {"text": "gate"}},
        ],
    }});
    let volume = serde_json::json!({});
    let moved = rename_references(
        &mut behavior,
        RegisteredType::TriggerVolume,
        &volume,
        "gate",
        "door",
    );
    assert_eq!(moved, 2);
    assert_eq!(behavior["args"]["on"]["enter"], "door");
    assert_eq!(
        behavior["args"]["do"][0]["despawn"]["target"]["named"],
        "door"
    );
    assert_eq!(
        behavior["args"]["do"][1]["log"]["text"], "gate",
        "plain text"
    );
}

// Rename the entry declaring `from` to `to` the way an editing tool does: its
// `$id` first, then every entry's references to it. Returns the references
// moved.
fn rename_in(world: &mut [serde_json::Value], from: &str, to: &str) -> usize {
    let at = world
        .iter()
        .position(|e| e["args"]["$id"] == from)
        .unwrap_or_else(|| panic!("no entry declares {from}"));
    world[at]["args"]["$id"] = to.into();
    let target = RegisteredType::parse(world[at]["type"].as_str().unwrap()).unwrap();
    let args = world[at]["args"].clone();
    world
        .iter_mut()
        .map(|e| rename_references(e, target, &args, from, to))
        .sum()
}

fn entry(ty: &str, args: serde_json::Value) -> serde_json::Value {
    serde_json::json!({"type": ty, "args": args})
}

// Every field a build-only schema (or a Prop's `prefab`) names another asset by
// follows that asset's rename, including a Prefab's entries; text that merely
// reads the old name stays.
#[test]
fn a_rename_follows_every_build_only_reference() {
    let mut world = vec![
        entry("Screen", serde_json::json!({"$id": "pause"})),
        entry("Font", serde_json::json!({"$id": "body"})),
        entry("Scene", serde_json::json!({"$id": "level"})),
        entry("Texture", serde_json::json!({"$id": "stone"})),
        entry("DirectionalLight", serde_json::json!({"$id": "sun"})),
        entry("PointLight", serde_json::json!({"$id": "fill"})),
        entry("Model", serde_json::json!({"$id": "mdl", "meshes": []})),
        entry("Material", serde_json::json!({"$id": "mat"})),
        entry(
            "ProceduralMesh",
            serde_json::json!({"$id": "box", "generator": "box"}),
        ),
        entry("Prop", serde_json::json!({"$id": "base"})),
        entry("CharacterSchema", serde_json::json!({"$id": "sk"})),
        entry(
            "Prefab",
            serde_json::json!({"$id": "lantern", "props": [
                {"name": "body", "mesh": "box", "material": "mat", "parent": "base"},
                {"name": "shade", "model": "mdl"},
            ]}),
        ),
        entry(
            "Prefab",
            serde_json::json!({"$id": "post", "props": [
                {"name": "top", "kind": "prefab", "prefab": "lantern"},
            ]}),
        ),
        entry(
            "Prop",
            serde_json::json!({"$id": "lamp_1", "prefab": "lantern"}),
        ),
        entry(
            "Panel",
            serde_json::json!({
                "$id": "card", "screen": "pause", "title": "pause", "title_font": "body",
            }),
        ),
        entry(
            "OptionSelect",
            serde_json::json!({"$id": "vsync", "screen": "pause", "font": "body"}),
        ),
        entry(
            "Slider",
            serde_json::json!({"$id": "gain", "screen": "pause", "font": "body"}),
        ),
        entry(
            "MainMenu",
            serde_json::json!({"$id": "menu", "font": "body"}),
        ),
        entry(
            "CharacterModel",
            serde_json::json!({"$id": "hero", "schema": "sk"}),
        ),
        entry(
            "SceneImport",
            serde_json::json!({"$id": "city", "source": "a.glb", "scene": "level"}),
        ),
        entry(
            "LightRig",
            serde_json::json!({"$id": "rig", "lights": ["sun", "fill"]}),
        ),
        entry(
            "MaterialPalette",
            serde_json::json!({"$id": "pal", "entries": [
                {"alias": "wall", "albedo": "stone", "normal_map": "stone"},
            ]}),
        ),
    ];

    for (from, to, moved) in [
        ("pause", "paused", 3),
        ("body", "face", 4),
        ("level", "stage", 1),
        ("stone", "rock", 2),
        ("sun", "key", 1),
        ("fill", "bounce", 1),
        ("mdl", "shade_model", 1),
        ("mat", "brass", 1),
        ("box", "crate", 1),
        ("base", "plinth", 1),
        ("sk", "humanoid", 1),
        ("lantern", "lamp", 2),
    ] {
        assert_eq!(rename_in(&mut world, from, to), moved, "renaming {from}");
    }

    let args = |id: &str| {
        world
            .iter()
            .find(|e| e["args"]["$id"] == id)
            .map(|e| e["args"].clone())
            .unwrap_or_else(|| panic!("no entry declares {id}"))
    };
    let card = args("card");
    assert_eq!(
        (&card["screen"], &card["title_font"]),
        (&"paused".into(), &"face".into())
    );
    assert_eq!(card["title"], "pause", "plain text");
    for id in ["vsync", "gain"] {
        assert_eq!(
            (&args(id)["screen"], &args(id)["font"]),
            (&"paused".into(), &"face".into())
        );
    }
    assert_eq!(args("menu")["font"], "face");
    assert_eq!(args("hero")["schema"], "humanoid");
    assert_eq!(args("city")["scene"], "stage");
    assert_eq!(args("rig")["lights"], serde_json::json!(["key", "bounce"]));
    let wall = &args("pal")["entries"][0];
    assert_eq!(
        (&wall["albedo"], &wall["normal_map"]),
        (&"rock".into(), &"rock".into())
    );
    let lamp = args("lamp");
    assert_eq!(lamp["props"][0]["mesh"], "crate");
    assert_eq!(lamp["props"][0]["material"], "brass");
    assert_eq!(lamp["props"][0]["parent"], "plinth");
    assert_eq!(lamp["props"][0]["name"], "body", "an entry name");
    assert_eq!(lamp["props"][1]["model"], "shade_model");
    assert_eq!(args("post")["props"][0]["prefab"], "lamp");
    assert_eq!(args("lamp_1")["prefab"], "lamp");
}

// A Prefab entry's mesh is a reference only where the entry builds a prop from
// it: a model takes precedence, and a light entry reads no mesh.
#[test]
fn a_prefab_mesh_is_a_reference_only_where_the_entry_draws_it() {
    let mesh = serde_json::json!({"generator": "box"});
    let mut prefab = entry(
        "Prefab",
        serde_json::json!({"$id": "p", "props": [
            {"name": "a", "mesh": "box", "model": "m"},
            {"name": "b", "kind": "point_light", "mesh": "box"},
            {"name": "c", "kind": "prop", "mesh": "box"},
        ]}),
    );
    let moved = rename_references(
        &mut prefab,
        RegisteredType::ProceduralMesh,
        &mesh,
        "box",
        "crate",
    );
    assert_eq!(moved, 1);
    let props = &prefab["args"]["props"];
    assert_eq!(
        (&props[0]["mesh"], &props[1]["mesh"]),
        (&"box".into(), &"box".into())
    );
    assert_eq!(props[2]["mesh"], "crate");
}

// The count is exactly what a rename would move, and an entry holding no
// string that reads the name counts none.
#[test]
fn counting_references_matches_what_a_rename_moves() {
    let mesh = serde_json::json!({"generator": "box"});
    let prop = entry(
        "Prop",
        serde_json::json!({"$id": "p", "mesh": "box", "material": "box"}),
    );
    assert_eq!(
        count_references(&prop, RegisteredType::ProceduralMesh, &mesh, "box"),
        1
    );
    let panel = entry(
        "Panel",
        serde_json::json!({"$id": "c", "screen": "pause", "title": "pause"}),
    );
    let screen = serde_json::json!({});
    assert_eq!(
        count_references(&panel, RegisteredType::Screen, &screen, "pause"),
        1
    );
    assert_eq!(
        count_references(&panel, RegisteredType::Screen, &screen, "menu"),
        0
    );

    let mut untouched = entry(
        "Behavior",
        serde_json::json!({"$id": "b", "on": {"enter": "gate"}}),
    );
    let before = untouched.clone();
    assert_eq!(
        rename_references(
            &mut untouched,
            RegisteredType::TriggerVolume,
            &screen,
            "door",
            "exit"
        ),
        0
    );
    assert_eq!(untouched, before);
}

// A renamed world still expands and validates: each generated asset carries the
// new name, which resolves.
#[test]
fn a_renamed_world_still_builds() {
    let mut world = vec![
        entry("GraphicsConfig", serde_json::json!({"$id": "gfx"})),
        entry("Screen", serde_json::json!({"$id": "pause"})),
        entry("Font", serde_json::json!({"$id": "body", "size_px": 20})),
        entry(
            "Texture",
            serde_json::json!({"$id": "stone", "generator": "brick"}),
        ),
        entry("PointLight", serde_json::json!({"$id": "fill"})),
        entry(
            "ProceduralMesh",
            serde_json::json!({"$id": "box", "generator": "box"}),
        ),
        entry(
            "Prefab",
            serde_json::json!({"$id": "lantern", "props": [
                {"name": "body", "mesh": "box"},
            ]}),
        ),
        entry(
            "Prop",
            serde_json::json!({"$id": "lamp_1", "prefab": "lantern"}),
        ),
        entry(
            "Panel",
            serde_json::json!({
                "$id": "card", "screen": "pause", "title": "Paused", "title_font": "body",
            }),
        ),
        entry(
            "Slider",
            serde_json::json!({
                "$id": "gain", "setting": "exposure", "screen": "pause", "font": "body",
            }),
        ),
        entry(
            "LightRig",
            serde_json::json!({"$id": "rig", "lights": ["fill"]}),
        ),
        entry(
            "MaterialPalette",
            serde_json::json!({"$id": "pal", "entries": [
                {"alias": "wall", "albedo": "stone"},
            ]}),
        ),
    ];
    for (from, to) in [
        ("pause", "paused"),
        ("body", "face"),
        ("stone", "rock"),
        ("fill", "bounce"),
        ("box", "crate"),
        ("lantern", "lamp"),
    ] {
        rename_in(&mut world, from, to);
    }
    let text: String = world
        .iter()
        .map(|e| format!("{}\n", serde_json::json!([e["type"], e["args"]])))
        .collect();
    let loaded = crate::build_only::prepare_world(text.as_str(), None)
        .unwrap_or_else(|e| panic!("the renamed world builds: {e:?}"));
    let args = |id: &str| {
        &loaded
            .assets
            .iter()
            .find(|a| a.id == id)
            .unwrap_or_else(|| panic!("no asset {id}"))
            .args
    };
    assert_eq!(args("card_bg")["screen"], "paused");
    assert_eq!(args("card_title")["font"], "face");
    assert_eq!(args("gain_label")["font"], "face");
    assert_eq!(args("lamp_1_body")["mesh"], "crate");
    assert_eq!(args("pal_wall")["albedo"], "rock");
}

// An action's target is a reference like any other: a rename of the Screen or
// Scene it names follows into a HitRegion's and a KeyBinding's action, a menu
// item's, and a menu's Back override, and an action naming something else is
// left alone.
#[test]
fn a_rename_follows_action_targets() {
    let mut world = vec![
        entry("Screen", serde_json::json!({"$id": "pause"})),
        entry("Scene", serde_json::json!({"$id": "level"})),
        entry(
            "HitRegion",
            serde_json::json!({"$id": "go", "action": {"scene": "level"}}),
        ),
        entry(
            "KeyBinding",
            serde_json::json!({"$id": "esc", "action": {"toggle": "pause"}}),
        ),
        entry(
            "MainMenu",
            serde_json::json!({
                "$id": "menu",
                "items": [
                    {"label": "Play", "action": {"scene": "level"}},
                    {"label": "Pause", "action": {"push": "pause"}},
                    {"label": "Options", "action": "settings"},
                ],
                "settings_back_action": {"show": "pause"},
            }),
        ),
    ];
    assert_eq!(rename_in(&mut world, "pause", "paused"), 3);
    assert_eq!(rename_in(&mut world, "level", "stage"), 2);
    assert_eq!(
        world[2]["args"]["action"],
        serde_json::json!({"scene": "stage"})
    );
    assert_eq!(
        world[3]["args"]["action"],
        serde_json::json!({"toggle": "paused"})
    );
    let menu = &world[4]["args"];
    assert_eq!(
        menu["items"][0]["action"],
        serde_json::json!({"scene": "stage"})
    );
    assert_eq!(
        menu["items"][1]["action"],
        serde_json::json!({"push": "paused"})
    );
    assert_eq!(menu["items"][2]["action"], "settings");
    assert_eq!(
        menu["settings_back_action"],
        serde_json::json!({"show": "paused"})
    );
}
