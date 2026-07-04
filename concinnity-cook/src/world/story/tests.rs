use super::emit::{CHOICE_BOX_RADIUS, DIALOG_BOX_RADIUS};
use super::helpers::{slug, wrap_text};
use super::*;

const CROSSROADS: &str = r#"---
title: The Crossroads
characters:
  ayame: { name: "Ayame", color: [1.0, 0.85, 0.8] }
  keeper: Innkeeper
---

# inn

You wake at a roadside inn. A note rests on the pillow.

**keeper:** Slept well? Someone left that for you.

# The Crossroads

The signpost points two ways.

- [Into the wood](#wood)
- [Toward the shore](#the-crossroads)

# wood

**ayame:** You came. I wasn't sure you would.

[The morning comes.](#ending)

# ending

You walk together toward the morning.
"#;

// Fixed portrait-shaped dimensions so emission tests need no image files.
fn stub_dims(_path: &str) -> Result<(u32, u32), String> {
    Ok((456, 700))
}

fn find<'a>(entries: &'a [serde_json::Value], name: &str) -> &'a serde_json::Value {
    entries
        .iter()
        .find(|e| asset_name(e) == name)
        .unwrap_or_else(|| panic!("missing entry '{}'", name))
}

fn action(entry: &serde_json::Value) -> &str {
    entry["args"]["action"].as_str().unwrap_or("")
}

#[test]
fn passes_through_without_imports() {
    let mut assets = vec![serde_json::json!({"name":"x","type":"Logger","args":{}})];
    expand_stories(&mut assets).unwrap();
    assert_eq!(assets.len(), 1);
    assert_eq!(assets[0]["type"], "Logger");
}

#[test]
fn missing_source_is_an_error() {
    let mut assets = vec![serde_json::json!({
        "name": "story", "type": "StoryImport", "args": {}
    })];
    let err = expand_stories(&mut assets).unwrap_err();
    assert!(err.contains("missing `source`"));
}

#[test]
fn parses_frontmatter_nodes_and_flow() {
    let story = parse_story(CROSSROADS).unwrap();
    assert_eq!(story.title, "The Crossroads");
    assert_eq!(story.characters["ayame"].name, "Ayame");
    assert_eq!(story.characters["ayame"].color, [1.0, 0.85, 0.8]);
    assert_eq!(story.characters["keeper"].name, "Innkeeper");
    assert_eq!(story.characters["keeper"].color, [1.0, 1.0, 1.0]);

    let slugs: Vec<&str> = story.nodes.iter().map(|n| n.slug.as_str()).collect();
    assert_eq!(slugs, ["inn", "the-crossroads", "wood", "ending"]);

    let inn = &story.nodes[0];
    assert_eq!(inn.pages.len(), 2);
    assert_eq!(inn.pages[1].speaker.as_deref(), Some("keeper"));

    let crossroads = &story.nodes[1];
    assert_eq!(crossroads.choices.len(), 2);
    assert_eq!(crossroads.choices[0].target, "wood");
    assert_eq!(crossroads.choices[1].target, "the-crossroads");

    let wood = &story.nodes[2];
    assert_eq!(wood.pages[1].jump.as_deref(), Some("ending"));
    assert_eq!(wood.pages[1].text, "The morning comes.");
}

#[test]
fn emits_the_compiled_graph_and_stage() {
    let story = parse_story(CROSSROADS).unwrap();
    let entries = emit_story("story", &story, true, 30.0, &stub_dims).unwrap();

    // The title screen is initial and starts the story system.
    let title = find(&entries, "story_title");
    assert_eq!(title["args"]["initial"], true);
    assert_eq!(
        action(find(&entries, "story_title_start_btn")),
        "story:start"
    );

    // The compiled graph takes the import's own name. Speakers resolve
    // to display name + color, jump and choice targets to node indices,
    // and the reveal speed rides along.
    let graph = &find(&entries, "story")["args"];
    assert_eq!(graph["title"], "The Crossroads");
    assert_eq!(graph["text_speed"], 30.0);
    let nodes = graph["nodes"].as_array().unwrap();
    assert_eq!(nodes.len(), 4);
    let plate = &nodes[0]["pages"][1]["speaker"];
    assert_eq!(plate["name"], "Innkeeper");
    assert_eq!(nodes[1]["choices"][0]["label"], "Into the wood");
    assert_eq!(nodes[1]["choices"][0]["target"], 2);
    assert_eq!(nodes[1]["choices"][1]["target"], 1);
    assert_eq!(nodes[2]["pages"][1]["text"], "The morning comes.");
    assert_eq!(nodes[2]["pages"][1]["jump"], 3);

    // One stage view carries the whole story: backdrop, portrait slots,
    // dialog furniture, the advance region, and one button per option of
    // the widest choice menu (disabled until the story reaches one).
    assert_eq!(find(&entries, "story_stage")["args"]["initial"], false);
    assert_eq!(find(&entries, "story_stage_bg")["args"]["fit"], "cover");
    assert_eq!(
        find(&entries, "story_stage_center")["args"]["visible"],
        false
    );
    assert_eq!(
        action(find(&entries, "story_stage_advance")),
        "story:advance"
    );
    assert_eq!(
        find(&entries, "story_advance_key")["args"]["action"],
        "story:advance"
    );
    let opt0 = find(&entries, "story_stage_opt0_btn");
    assert_eq!(action(opt0), "story:choose:0");
    assert_eq!(
        find(&entries, "story_stage_opt0_lbl")["args"]["visible"],
        false
    );

    // The scaffold block names the generated stage assets; the build
    // resolves these to ids so the compiled blob never needs the names.
    let scaffold = &graph["scaffold"];
    assert_eq!(scaffold["view"], "story_stage");
    assert_eq!(scaffold["ending"], "story_ending");
    assert_eq!(scaffold["text_label"], "story_stage_text");
    assert_eq!(scaffold["option_boxes"][0], "story_stage_opt0_box");
    assert_eq!(scaffold["options"][0], "story_stage_opt0_lbl");
    assert_eq!(scaffold["options"].as_array().unwrap().len(), 2);

    // Each option slot gets its own rounded box behind the label, and the
    // dialog box sits nearly flush with the canvas bottom with the name
    // plate inside its bounds.
    let opt0_box = find(&entries, "story_stage_opt0_box");
    assert_eq!(opt0_box["args"]["corner_radius"], CHOICE_BOX_RADIUS);
    assert_eq!(opt0_box["args"]["tint"][3], 0.0);
    let dialog = &find(&entries, "story_stage_box")["args"];
    assert_eq!(dialog["corner_radius"], DIALOG_BOX_RADIUS);
    let box_bottom = dialog["y"].as_f64().unwrap() + dialog["height"].as_f64().unwrap();
    assert!((700.0..=720.0).contains(&box_bottom));
    let name = &find(&entries, "story_stage_name")["args"];
    assert!(name["y"].as_f64().unwrap() > dialog["y"].as_f64().unwrap());
    let text = &find(&entries, "story_stage_text")["args"];
    assert!(text["y"].as_f64().unwrap() > name["y"].as_f64().unwrap());
    assert!(!entries.iter().any(|e| asset_name(e).contains("_panel")));

    // Quick row, advance marker, overlay furniture, and slot rows all
    // exist and start hidden; the title screen offers Load.
    assert_eq!(
        action(find(&entries, "story_stage_qauto_btn")),
        "story:auto"
    );
    assert_eq!(action(find(&entries, "story_stage_qlog_btn")), "story:log");
    assert_eq!(
        action(find(&entries, "story_stage_qskip_btn")),
        "story:skip"
    );
    assert_eq!(
        action(find(&entries, "story_stage_qsave_btn")),
        "story:save"
    );
    assert_eq!(find(&entries, "story_stage_marker")["args"]["tint"][3], 0.0);
    assert_eq!(find(&entries, "story_stage_dim")["args"]["tint"][3], 0.0);
    assert_eq!(find(&entries, "story_stage_history")["args"]["content"], "");
    assert_eq!(
        action(find(&entries, "story_stage_slot2_btn")),
        "story:slot:2"
    );
    assert_eq!(
        find(&entries, "story_stage_slot0_box")["args"]["tint"][3],
        0.0
    );
    assert_eq!(action(find(&entries, "story_title_load_btn")), "story:load");
    assert_eq!(scaffold["slot_labels"].as_array().unwrap().len(), 3);
    assert_eq!(scaffold["advance_marker"], "story_stage_marker");
    assert_eq!(scaffold["title"], "story_title");
    assert!(
        entries
            .iter()
            .any(|e| asset_name(e) == "story_stage_opt1_btn")
    );
    assert!(
        !entries
            .iter()
            .any(|e| asset_name(e) == "story_stage_opt2_btn")
    );

    // No per-page views or audio cues remain.
    assert!(!entries.iter().any(|e| asset_name(e).contains("_n_")));
    assert!(!entries.iter().any(|e| type_norm(e) == "audiocue"));

    // The ending returns to the title screen.
    assert_eq!(
        action(find(&entries, "story_ending_back_btn")),
        "view:show:story_title"
    );
}

#[test]
fn no_title_screen_makes_the_stage_initial() {
    let story = parse_story(CROSSROADS).unwrap();
    let entries = emit_story("story", &story, false, 45.0, &stub_dims).unwrap();
    assert!(!entries.iter().any(|e| asset_name(e) == "story_title"));
    assert_eq!(find(&entries, "story_stage")["args"]["initial"], true);
    let back = find(&entries, "story_ending_back_btn");
    assert_eq!(action(back), "story:start");
    assert_eq!(
        find(&entries, "story_ending_back_lbl")["args"]["content"],
        "Restart"
    );
}

// The dialog furniture is bottom-anchored, its labels centered/aligned,
// and the title menu buttons follow their labels so the story can lay the
// menu out at runtime.
#[test]
fn dialog_is_bottom_anchored_and_title_buttons_follow() {
    let story = parse_story(CROSSROADS).unwrap();
    let entries = emit_story("story", &story, true, 30.0, &stub_dims).unwrap();

    // The box, its text, its marker, and the quick row all hug the bottom.
    assert_eq!(find(&entries, "story_stage_box")["args"]["fit"], "bottom");
    assert_eq!(find(&entries, "story_stage_text")["args"]["fit"], "bottom");
    assert_eq!(
        find(&entries, "story_stage_marker")["args"]["fit"],
        "bottom"
    );
    assert_eq!(
        find(&entries, "story_stage_qauto_btn")["args"]["fit"],
        "bottom"
    );

    // The heading and menu buttons center on their anchor with real
    // metrics; each title button's region follows its label.
    assert_eq!(
        find(&entries, "story_title_heading")["args"]["align"],
        "center"
    );
    let start = find(&entries, "story_title_start_btn");
    assert_eq!(start["args"]["follow_label"], true);
    assert_eq!(start["args"]["label"], "story_title_start_lbl");
    assert_eq!(
        find(&entries, "story_title_start_lbl")["args"]["align"],
        "center"
    );
    let scaffold = &find(&entries, "story")["args"]["scaffold"];
    assert_eq!(scaffold["start_label"], "story_title_start_lbl");
    assert_eq!(scaffold["quit_label"], "story_title_quit_lbl");
}

// A `background` frontmatter draws the title menu over a full-bleed image
// (a cover-fit textured sprite + one Texture entry).
#[test]
fn menu_background_becomes_a_cover_sprite() {
    let src = CROSSROADS.replace(
        "title: The Crossroads\n",
        "title: The Crossroads\nbackground: assets/menu.png\n",
    );
    let story = parse_story(&src).unwrap();
    assert_eq!(story.background.as_deref(), Some("assets/menu.png"));
    let entries = emit_story("story", &story, true, 30.0, &stub_dims).unwrap();
    let bg = &find(&entries, "story_title_bg")["args"];
    assert_eq!(bg["fit"], "cover");
    let texture = bg["texture"].as_str().unwrap();
    let tex = find(&entries, texture);
    assert_eq!(type_norm(tex), "texture");
    assert_eq!(tex["args"]["source"], "assets/menu.png");
}

#[test]
fn expands_from_file_and_replaces_the_import() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("story.md");
    std::fs::write(&path, CROSSROADS).unwrap();
    let mut assets = vec![serde_json::json!({
        "name": "story", "type": "StoryImport",
        "args": {"source": path.to_str().unwrap()}
    })];
    expand_stories(&mut assets).unwrap();
    assert!(!assets.iter().any(|v| type_norm(v) == "storyimport"));
    assert!(assets.iter().any(|v| type_norm(v) == "view"));
    assert!(assets.iter().any(|v| type_norm(v) == "hitregion"));
    assert!(assets.iter().any(|v| type_norm(v) == "font"));
}

#[test]
fn generated_name_collision_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("story.md");
    std::fs::write(&path, CROSSROADS).unwrap();
    let mut assets = vec![
        serde_json::json!({
            "name": "story", "type": "StoryImport",
            "args": {"source": path.to_str().unwrap()}
        }),
        serde_json::json!({"name":"story_title","type":"View","args":{}}),
    ];
    let err = expand_stories(&mut assets).unwrap_err();
    assert!(err.contains("collides"));
}

const SCORED: &str = r#"---
title: T
---

# inn

[music](assets/theme.ogg)

First page.

Second page.

[sound](assets/door.wav)

The door creaks.

# crossroads

[music](assets/tense.ogg)

- [Left](#inn)
- [Right](#crossroads)
"#;

#[test]
fn media_directives_parse_and_propagate() {
    let story = parse_story(SCORED).unwrap();
    let inn = &story.nodes[0];
    // Music persists from its directive across the following pages.
    assert_eq!(inn.pages[0].music.as_deref(), Some("assets/theme.ogg"));
    assert_eq!(inn.pages[1].music.as_deref(), Some("assets/theme.ogg"));
    // The one-shot attaches only to the page directly after it.
    assert!(inn.pages[1].sounds.is_empty());
    assert_eq!(inn.pages[2].sounds, ["assets/door.wav"]);
    // A pages-free choice node still carries the current music.
    let crossroads = &story.nodes[1];
    assert!(crossroads.pages.is_empty());
    assert_eq!(crossroads.choice_music.as_deref(), Some("assets/tense.ogg"));
}

#[test]
fn media_directives_compile_to_deduped_clip_names() {
    let story = parse_story(SCORED).unwrap();
    let entries = emit_story("s", &story, true, 45.0, &stub_dims).unwrap();

    // Three distinct audio files -> three AudioClip entries; the theme is
    // deduplicated to one clip despite two pages sharing its music.
    let clips: Vec<&serde_json::Value> = entries
        .iter()
        .filter(|e| type_norm(e) == "audioclip")
        .collect();
    assert_eq!(clips.len(), 3);
    assert_eq!(clips[0]["args"]["source"], "assets/theme.ogg");

    // Pages carry the deduplicated clip names in the compiled graph.
    let nodes = &find(&entries, "s")["args"]["nodes"];
    assert_eq!(nodes[0]["pages"][0]["music"], clips[0]["name"]);
    assert_eq!(nodes[0]["pages"][1]["music"], clips[0]["name"]);

    // The one-shot lands on its page only.
    assert_eq!(nodes[0]["pages"][2]["sounds"][0], clips[1]["name"]);
    assert_eq!(nodes[0]["pages"][1]["sounds"].as_array().unwrap().len(), 0);

    // The choice menu carries the tense track.
    assert_eq!(nodes[1]["choice_music"], clips[2]["name"]);
}

#[test]
fn bg_directive_parses_propagates_and_emits_textured_backdrops() {
    let src = "---\ntitle: T\n---\n\n# inn\n\n![bg](assets/inn.png)\n\nFirst.\n\nSecond.\n\n# out\n\n![bg](assets/street.png)\n\n- [Stay](#inn)\n";
    let story = parse_story(src).unwrap();
    // The backdrop persists across the pages after its directive.
    assert_eq!(
        story.nodes[0].pages[0].stage.bg.as_deref(),
        Some("assets/inn.png")
    );
    assert_eq!(
        story.nodes[0].pages[1].stage.bg.as_deref(),
        Some("assets/inn.png")
    );
    // A pages-free choice node carries the current backdrop.
    assert_eq!(
        story.nodes[1].choice_stage.bg.as_deref(),
        Some("assets/street.png")
    );

    let entries = emit_story("s", &story, true, 45.0, &stub_dims).unwrap();
    // Two distinct images -> two Texture entries.
    let textures: Vec<&serde_json::Value> = entries
        .iter()
        .filter(|e| type_norm(e) == "texture")
        .collect();
    assert_eq!(textures.len(), 2);
    assert_eq!(textures[0]["args"]["source"], "assets/inn.png");

    // Both pages of the node share the deduplicated texture in the
    // compiled graph; a full-canvas rectangle places the backdrop.
    let nodes = &find(&entries, "s")["args"]["nodes"];
    let bg = &nodes[0]["pages"][0]["stage"]["bg"];
    assert_eq!(bg["texture"], textures[0]["name"]);
    assert_eq!(bg["width"], 1280.0);
    assert_eq!(
        nodes[0]["pages"][1]["stage"]["bg"]["texture"],
        textures[0]["name"]
    );
    assert_eq!(
        nodes[1]["choice_stage"]["bg"]["texture"],
        textures[1]["name"]
    );

    // The title screen keeps its flat fill (no texture key at all).
    assert!(
        find(&entries, "s_title_bg")["args"]
            .get("texture")
            .is_none()
    );
}

#[test]
fn stacked_directives_in_one_paragraph_all_apply() {
    // Adjacent directive lines form one Markdown paragraph; all apply.
    let src = "---\ntitle: T\n---\n\n# a\n\n![bg](x.png)\n[music](m.ogg)\n[sound](s.wav)\n\nhi\n";
    let story = parse_story(src).unwrap();
    let page = &story.nodes[0].pages[0];
    assert_eq!(page.stage.bg.as_deref(), Some("x.png"));
    assert_eq!(page.music.as_deref(), Some("m.ogg"));
    assert_eq!(page.sounds, ["s.wav"]);
}

#[test]
fn jump_link_mixed_with_directives_is_an_error() {
    let src = "---\ntitle: T\n---\n\n# a\n\n![bg](x.png)\n[go](#a)\n\nhi\n";
    let err = parse_story(src).unwrap_err();
    assert!(err.contains("cannot share"), "{err}");
}

#[test]
fn portraits_persist_and_bg_change_clears_them() {
    let src = "---\ntitle: T\n---\n\n# a\n\n![bg](room.png)\n![left](ana.png)\n\nOne.\n\n![right](ben.png)\n![center](cho.png)\n\nTwo.\n\n![bg](street.png)\n\nThree.\n";
    let story = parse_story(src).unwrap();
    let pages = &story.nodes[0].pages;
    // Page one: left portrait only.
    assert_eq!(pages[0].stage.left.as_deref(), Some("ana.png"));
    assert_eq!(pages[0].stage.center, None);
    assert_eq!(pages[0].stage.right, None);
    // Page two: the right and center portraits join; the left persists.
    assert_eq!(pages[1].stage.left.as_deref(), Some("ana.png"));
    assert_eq!(pages[1].stage.center.as_deref(), Some("cho.png"));
    assert_eq!(pages[1].stage.right.as_deref(), Some("ben.png"));
    // Page three: the scene change cleared every portrait.
    assert_eq!(pages[2].stage.bg.as_deref(), Some("street.png"));
    assert_eq!(pages[2].stage.left, None);
    assert_eq!(pages[2].stage.center, None);
    assert_eq!(pages[2].stage.right, None);
}

#[test]
fn portraits_compile_at_native_size_and_bottom_anchor() {
    let src = "---\ntitle: T\n---\n\n# a\n\n![left](ana.png)\n\nhi\n";
    let story = parse_story(src).unwrap();
    // stub_dims reports 456x700: placed at native pixel size against the
    // 720 canvas, bottom-anchored; the cover-fit stage sprites put the
    // canvas bottom at the window bottom.
    let entries = emit_story("s", &story, true, 45.0, &stub_dims).unwrap();
    let p = &find(&entries, "s")["args"]["nodes"][0]["pages"][0]["stage"]["left"];
    assert_eq!(p["width"], 456.0);
    assert_eq!(p["height"], 700.0);
    assert_eq!(p["y"], 20.0);
    let x = p["x"].as_f64().unwrap() as f32;
    assert!((x - (320.0 - 456.0 / 2.0)).abs() < 1e-3);
    // The portrait image becomes a Texture entry like a backdrop.
    assert_eq!(p["texture"], find(&entries, "s_img0")["name"]);
}

#[test]
fn oversized_portrait_scales_down_to_the_canvas_height() {
    let src = "---\ntitle: T\n---\n\n# a\n\n![center](big.png)\n\nhi\n";
    let story = parse_story(src).unwrap();
    let dims = |_: &str| Ok((900u32, 1440u32));
    let entries = emit_story("s", &story, true, 45.0, &dims).unwrap();
    let p = &find(&entries, "s")["args"]["nodes"][0]["pages"][0]["stage"]["center"];
    // Taller than the canvas: clamped to 720 with the width following.
    assert_eq!(p["height"], 720.0);
    assert_eq!(p["width"], 450.0);
    assert_eq!(p["y"], 0.0);
    // Centered on the canvas.
    let x = p["x"].as_f64().unwrap() as f32;
    assert!((x - (640.0 - 450.0 / 2.0)).abs() < 1e-3);
}

#[test]
fn unknown_image_role_is_an_error() {
    let src = "---\ntitle: T\n---\n\n# a\n\n![portrait](x.png)\n\nhi\n";
    let err = parse_story(src).unwrap_err();
    assert!(err.contains("'portrait'"), "{err}");
    assert!(err.contains("![left]"), "{err}");
}

#[test]
fn non_image_bg_target_is_an_error() {
    let src = "---\ntitle: T\n---\n\n# a\n\n![bg](x.gif)\n\nhi\n";
    let err = parse_story(src).unwrap_err();
    assert!(err.contains("png/jpg/jpeg"), "{err}");
}

#[test]
fn image_mixed_with_text_is_an_error() {
    let src = "---\ntitle: T\n---\n\n# a\n\nlook: ![bg](x.png)\n\nhi\n";
    let err = parse_story(src).unwrap_err();
    assert!(err.contains("stand alone"), "{err}");
}

#[test]
fn trailing_bg_directive_is_an_error() {
    let src = "---\ntitle: T\n---\n\n# a\n\nhi\n\n![bg](x.png)\n";
    let err = parse_story(src).unwrap_err();
    assert!(err.contains("needs a following"), "{err}");
}

#[test]
fn trailing_media_directive_is_an_error() {
    let src = "---\ntitle: T\n---\n\n# a\n\nhi\n\n[sound](x.wav)\n";
    let err = parse_story(src).unwrap_err();
    assert!(err.contains("needs a following"), "{err}");
}

#[test]
fn bad_audio_label_is_an_error() {
    let src = "---\ntitle: T\n---\n\n# a\n\n[loop](x.ogg)\n\nhi\n";
    let err = parse_story(src).unwrap_err();
    assert!(err.contains("`music`"), "{err}");
    assert!(err.contains("loop"), "{err}");
}

#[test]
fn non_audio_link_target_is_an_error() {
    let src = "---\ntitle: T\n---\n\n# a\n\n[music](x.txt)\n\nhi\n";
    let err = parse_story(src).unwrap_err();
    assert!(err.contains("neither"), "{err}");
}

#[test]
fn choice_to_audio_file_is_an_error() {
    let src = "---\ntitle: T\n---\n\n# a\n\n- [music](x.ogg)\n";
    let err = parse_story(src).unwrap_err();
    assert!(err.contains("must link to a `#heading`"), "{err}");
}

#[test]
fn block_style_characters_parse() {
    let src = "---\ntitle: T\ncharacters:\n  ayame:\n    name: Ayame\n    color: [1.0, 0.85, 0.8]\n  keeper: Innkeeper\n---\n\n# a\n\n**ayame:** hi\n\n**keeper:** bye\n";
    let story = parse_story(src).unwrap();
    assert_eq!(story.characters["ayame"].name, "Ayame");
    assert_eq!(story.characters["ayame"].color, [1.0, 0.85, 0.8]);
    assert_eq!(story.characters["keeper"].name, "Innkeeper");
    assert_eq!(story.characters["keeper"].color, [1.0, 1.0, 1.0]);
}

#[test]
fn unquoted_flow_map_name_parses() {
    let src = "---\ntitle: T\ncharacters:\n  a: { name: Ayame Doe, color: [1, 1, 1] }\n---\n\n# n\n\n**a:** hi\n";
    let story = parse_story(src).unwrap();
    assert_eq!(story.characters["a"].name, "Ayame Doe");
}

#[test]
fn block_character_missing_name_is_an_error() {
    let src = "---\ntitle: T\ncharacters:\n  ayame:\n    color: [1, 1, 1]\n---\n\n# a\n\nhi\n";
    let err = parse_story(src).unwrap_err();
    assert!(err.contains("missing `name`"), "{err}");
    assert!(err.contains("ayame"), "{err}");
}

#[test]
fn block_character_unknown_key_is_an_error() {
    let src =
        "---\ntitle: T\ncharacters:\n  ayame:\n    name: Ayame\n    voice: low\n---\n\n# a\n\nhi\n";
    let err = parse_story(src).unwrap_err();
    assert!(err.contains("unknown character key 'voice'"), "{err}");
}

#[test]
fn fields_under_a_plain_character_are_an_error() {
    // `keeper: Innkeeper` is complete; a deeper line under it has no open
    // block to attach to.
    let src =
        "---\ntitle: T\ncharacters:\n  keeper: Innkeeper\n    color: [1, 1, 1]\n---\n\n# a\n\nhi\n";
    let err = parse_story(src).unwrap_err();
    assert!(err.contains("need an `id:` line"), "{err}");
}

#[test]
fn dangling_link_target_is_an_error() {
    let src = "---\ntitle: T\n---\n\n# a\n\n[go](#nowhere)\n";
    let err = parse_story(src).unwrap_err();
    assert!(err.contains("#nowhere"), "{err}");
}

#[test]
fn undeclared_speaker_is_an_error() {
    let src = "---\ntitle: T\n---\n\n# a\n\n**ghost:** boo\n";
    let err = parse_story(src).unwrap_err();
    assert!(err.contains("ghost"), "{err}");
    assert!(err.contains("line 7"), "{err}");
}

#[test]
fn duplicate_heading_is_an_error() {
    let src = "---\ntitle: T\n---\n\n# a\n\nhi\n\n# a\n\nbye\n";
    let err = parse_story(src).unwrap_err();
    assert!(err.contains("duplicate"), "{err}");
}

#[test]
fn missing_title_is_an_error() {
    let src = "---\ncharacters:\n---\n\n# a\n\nhi\n";
    let err = parse_story(src).unwrap_err();
    assert!(err.contains("title"), "{err}");
}

#[test]
fn content_before_first_heading_is_an_error() {
    let src = "---\ntitle: T\n---\n\nno node yet\n";
    let err = parse_story(src).unwrap_err();
    assert!(err.contains("before the first"), "{err}");
}

#[test]
fn empty_node_is_an_error() {
    let src = "---\ntitle: T\n---\n\n# a\n\n# b\n\nhi\n";
    let err = parse_story(src).unwrap_err();
    assert!(err.contains("empty"), "{err}");
}

#[test]
fn unsupported_constructs_are_errors() {
    let base = "---\ntitle: T\n---\n\n# a\n\n";
    for (body, needle) in [
        ("```\ncode\n```\n", "script blocks use"),
        ("```rust\nlet x = 1;\n```\n", "script blocks use"),
        ("> quoted\n", "block quotes"),
        ("## sub\n", "headings"),
        ("*soft*\n", "emphasis"),
        ("1. [go](#a)\n", "bullet list"),
        ("see `code`\n", "inline code"),
    ] {
        let err = parse_story(&format!("{base}{body}")).unwrap_err();
        assert!(err.contains(needle), "{body:?} -> {err}");
    }
}

#[test]
fn mixed_link_and_text_is_an_error() {
    let src = "---\ntitle: T\n---\n\n# a\n\ngo [here](#a) now\n";
    let err = parse_story(src).unwrap_err();
    assert!(err.contains("stand alone"), "{err}");
}

#[test]
fn content_after_choices_is_an_error() {
    let src = "---\ntitle: T\n---\n\n# a\n\n- [go](#a)\n\nafterthought\n";
    let err = parse_story(src).unwrap_err();
    assert!(err.contains("last content"), "{err}");
}

#[test]
fn script_blocks_parse_ops_gates_and_choice_conditions() {
    let src = "---\ntitle: T\n---\n\n# a\n\n```story\nset asked\nclear shy\nif asked -> #b\nif not shy -> #b\n```\n\nOne.\n\n- [Go](#b \"if asked\")\n- [Stay](#a \"if not shy\")\n- [Leave](#b)\n\n# b\n\nDone.\n";
    let story = parse_story(src).unwrap();
    let page = &story.nodes[0].pages[0];
    // Ops and gates attach to the page after the block, in order. Flag
    // syntax compiles to the numeric model: set = 1, clear = 0, a bare
    // test = "not zero", a `not` test = "zero".
    assert_eq!(page.ops.len(), 2);
    assert_eq!(page.ops[0].name, "asked");
    assert_eq!((page.ops[0].value, page.ops[0].add), (1, false));
    assert_eq!((page.ops[1].value, page.ops[1].add), (0, false));
    assert_eq!(page.gates.len(), 2);
    assert_eq!(page.gates[0].target, "b");
    assert_eq!(
        (page.gates[0].condition.op, page.gates[0].condition.value),
        ("ne", 0)
    );
    assert_eq!(
        (page.gates[1].condition.op, page.gates[1].condition.value),
        ("eq", 0)
    );
    // Link titles gate their choices; an untitled choice is unconditional.
    let choices = &story.nodes[0].choices;
    let c = choices[0].condition.as_ref().unwrap();
    assert_eq!(c.name, "asked");
    assert_eq!((c.op, c.value), ("ne", 0));
    assert_eq!(choices[1].condition.as_ref().unwrap().op, "eq");
    assert!(choices[2].condition.is_none());
}

#[test]
fn script_numeric_ops_and_comparisons_parse() {
    let src = "---\ntitle: T\n---\n\n# a\n\n```story\nset gold = 5\nadd gold -2\nif gold >= 3 -> #b\nif gold == 0 -> #b\nif gold != 1 -> #b\nif gold < 2 -> #b\nif gold <= 2 -> #b\nif gold > 2 -> #b\n```\n\nOne.\n\n- [Buy](#b \"if gold >= 3\")\n- [Beg](#b)\n\n# b\n\nDone.\n";
    let story = parse_story(src).unwrap();
    let page = &story.nodes[0].pages[0];
    assert_eq!(
        (
            page.ops[0].name.as_str(),
            page.ops[0].value,
            page.ops[0].add
        ),
        ("gold", 5, false)
    );
    assert_eq!((page.ops[1].value, page.ops[1].add), (-2, true));
    let ops: Vec<&str> = page.gates.iter().map(|g| g.condition.op).collect();
    assert_eq!(ops, vec!["ge", "eq", "ne", "lt", "le", "gt"]);
    assert_eq!(page.gates[0].condition.value, 3);
    let c = story.nodes[0].choices[0].condition.as_ref().unwrap();
    assert_eq!((c.name.as_str(), c.op, c.value), ("gold", "ge", 3));
}

#[test]
fn script_numeric_errors_are_strict() {
    let bad_amount = "---\ntitle: T\n---\n\n# a\n\n```story\nadd gold two\n```\n\nhi\n";
    assert!(parse_story(bad_amount).unwrap_err().contains("integer"));
    let missing_amount = "---\ntitle: T\n---\n\n# a\n\n```story\nadd gold\n```\n\nhi\n";
    assert!(parse_story(missing_amount).unwrap_err().contains("amount"));
    let bad_value = "---\ntitle: T\n---\n\n# a\n\n```story\nset gold = lots\n```\n\nhi\n";
    assert!(parse_story(bad_value).unwrap_err().contains("integer"));
    let bad_cmp = "---\ntitle: T\n---\n\n# a\n\n```story\nif gold >= many -> #a\n```\n\nhi\n";
    assert!(parse_story(bad_cmp).unwrap_err().contains("integer"));
}

#[test]
fn script_before_choices_attaches_to_the_menu() {
    let src = "---\ntitle: T\n---\n\n# a\n\nHi.\n\n```story\nset ready\nif done -> #b\n```\n\n- [Go](#b)\n\n# b\n\nDone.\n";
    let story = parse_story(src).unwrap();
    let node = &story.nodes[0];
    assert!(node.pages[0].ops.is_empty());
    assert_eq!(node.choice_ops.len(), 1);
    assert_eq!(node.choice_ops[0].name, "ready");
    assert_eq!(node.choice_gates.len(), 1);
    assert_eq!(node.choice_gates[0].target, "b");
}

#[test]
fn script_errors_are_strict() {
    // A malformed line, a bad flag name, a dangling gate target, a
    // trailing block with nothing to attach to, and a titled
    // non-choice link are all build errors.
    let bad_line = "---\ntitle: T\n---\n\n# a\n\n```story\nraise x\n```\n\nhi\n";
    assert!(parse_story(bad_line).unwrap_err().contains("is not `set"));
    let bad_flag = "---\ntitle: T\n---\n\n# a\n\n```story\nset Bad Flag\n```\n\nhi\n";
    assert!(
        parse_story(bad_flag)
            .unwrap_err()
            .contains("not a variable name")
    );
    let dangling = "---\ntitle: T\n---\n\n# a\n\n```story\nif x -> #nowhere\n```\n\nhi\n";
    assert!(
        parse_story(dangling)
            .unwrap_err()
            .contains("matches no heading")
    );
    let trailing = "---\ntitle: T\n---\n\n# a\n\nhi\n\n```story\nset x\n```\n";
    assert!(
        parse_story(trailing)
            .unwrap_err()
            .contains("needs a following")
    );
    let titled_jump = "---\ntitle: T\n---\n\n# a\n\n[Go](#a \"if x\")\n\nhi\n";
    assert!(
        parse_story(titled_jump)
            .unwrap_err()
            .contains("only supported on choices")
    );
}

#[test]
fn script_state_compiles_into_the_graph() {
    let src = "---\ntitle: T\n---\n\n# a\n\n```story\nset asked\nif asked -> #b\n```\n\nOne.\n\n- [Go](#b \"if asked\")\n\n# b\n\nDone.\n";
    let story = parse_story(src).unwrap();
    let entries = emit_story("s", &story, true, 45.0, &stub_dims).unwrap();
    let nodes = &find(&entries, "s")["args"]["nodes"];
    let page = &nodes[0]["pages"][0];
    assert_eq!(page["ops"][0]["name"], "asked");
    assert_eq!(page["ops"][0]["value"], 1);
    assert_eq!(page["ops"][0]["add"], false);
    // Gate targets compile to node indices like every other jump.
    assert_eq!(page["gates"][0]["target"], 1);
    assert_eq!(page["gates"][0]["op"], "ne");
    assert_eq!(page["gates"][0]["value"], 0);
    assert_eq!(nodes[0]["choices"][0]["condition"]["name"], "asked");
    assert_eq!(nodes[0]["choices"][0]["condition"]["op"], "ne");
}

#[test]
fn heading_names_cannot_collide_with_generated_views() {
    // Node names no longer mint views (the whole story plays inside one
    // stage view), so headings named after the scaffolding are fine.
    let src = "---\ntitle: T\n---\n\n# title\n\nhi\n\n# stage\n\nbye\n\n# ending\n\nfin\n";
    let story = parse_story(src).unwrap();
    let entries = emit_story("s", &story, true, 45.0, &stub_dims).unwrap();
    let nodes = &find(&entries, "s")["args"]["nodes"];
    assert_eq!(nodes.as_array().unwrap().len(), 3);
    // Exactly the three scaffolding views exist.
    let views: Vec<String> = entries
        .iter()
        .filter(|e| type_norm(e) == "view")
        .map(asset_name)
        .collect();
    assert_eq!(views, ["s_title", "s_stage", "s_ending"]);
}

#[test]
fn slug_matches_github_style_anchors() {
    assert_eq!(slug("The Crossroads"), "the-crossroads");
    assert_eq!(slug("  Inn's  Door  "), "inns-door");
    assert_eq!(slug("wood"), "wood");
}

#[test]
fn wrap_text_wraps_on_word_boundaries() {
    let wrapped = wrap_text("one two three four five", 9);
    assert_eq!(wrapped, "one two\nthree\nfour five");
    // Authored hard breaks survive.
    assert_eq!(wrap_text("a\nb", 80), "a\nb");
}
