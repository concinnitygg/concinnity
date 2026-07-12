use super::*;
use crate::assets::{Story, StoryChoice, StoryNode, StoryPage, StoryScaffold, StorySpeaker, View};
use crate::ecs::World;
use crate::ecs::asset_id::intern;
use crate::ecs::{AudioClipHandle, TextureHandle};

fn page(text: &str) -> StoryPage {
    StoryPage {
        text: text.to_string(),
        ..Default::default()
    }
}

fn label_named(name: &str) -> TextLabel {
    TextLabel {
        asset_id: intern(name),
        ..Default::default()
    }
}

fn sprite_named(name: &str) -> Sprite {
    Sprite {
        asset_id: intern(name),
        ..Default::default()
    }
}

// The build-resolved scaffold references for a story named "s".
fn scaffold() -> StoryScaffold {
    StoryScaffold {
        view: Some(intern("s_stage")),
        ending: Some(intern("s_ending")),
        bg: Some(intern("s_stage_bg")),
        left: Some(intern("s_stage_left")),
        center: Some(intern("s_stage_center")),
        right: Some(intern("s_stage_right")),
        dialog_box: Some(intern("s_stage_box")),
        name_label: Some(intern("s_stage_name")),
        text_label: Some(intern("s_stage_text")),
        option_boxes: vec![intern("s_stage_opt0_box")],
        options: vec![intern("s_stage_opt0_lbl")],
        start_label: None,
        quit_label: None,
        continue_label: None,
        title: None,
        load_label: None,
        pause: None,
        settings: None,
        settings_label: None,
        advance_marker: Some(intern("s_stage_marker")),
        log_label: Some(intern("s_stage_qlog_lbl")),
        auto_label: Some(intern("s_stage_qauto_lbl")),
        skip_label: Some(intern("s_stage_qskip_lbl")),
        save_label: Some(intern("s_stage_qsave_lbl")),
        overlay_dim: Some(intern("s_stage_dim")),
        backlog_label: Some(intern("s_stage_history")),
        slot_title: Some(intern("s_stage_slot_title")),
        slot_boxes: vec![intern("s_stage_slot0_box")],
        slot_labels: vec![intern("s_stage_slot0_lbl")],
    }
}

// Add the stage sprites and labels the scaffold references, all on the
// stage view, so `StageIds::from_scaffold` resolves for any story world.
fn add_stage_furniture(world: &mut World) {
    for sprite in [
        "s_stage_bg",
        "s_stage_left",
        "s_stage_center",
        "s_stage_right",
        "s_stage_box",
        "s_stage_opt0_box",
        "s_stage_marker",
        "s_stage_dim",
        "s_stage_slot0_box",
    ] {
        world.add_component(Sprite {
            view: Some(intern("s_stage")),
            ..sprite_named(sprite)
        });
    }
    for label in [
        "s_stage_name",
        "s_stage_text",
        "s_stage_opt0_lbl",
        "s_stage_qlog_lbl",
        "s_stage_qauto_lbl",
        "s_stage_qskip_lbl",
        "s_stage_qsave_lbl",
        "s_stage_history",
        "s_stage_slot_title",
        "s_stage_slot0_lbl",
    ] {
        world.add_component(TextLabel {
            view: Some(intern("s_stage")),
            ..label_named(label)
        });
    }
}

// A world with the stage scaffolding the build expansion would generate
// for a story named "s", plus the compiled graph itself.
fn story_world(story: Story) -> World {
    let mut world = World::new_empty();
    let mut story = story;
    story.asset_id = intern("s");
    story.scaffold = scaffold();
    world.add_component(story);
    for view in ["s_stage", "s_ending"] {
        world.add_component(View {
            asset_id: intern(view),
            initial: view == "s_stage",
            fade_in_secs: 0.0,
        });
    }
    add_stage_furniture(&mut world);
    world
}

// A story world whose slot overlay has `rows` visible slot rows (the shared
// scaffold has one), so the scrolling window over the logical slots can be
// exercised.
fn multi_slot_world(story: Story, rows: usize) -> World {
    let mut world = World::new_empty();
    let mut story = story;
    story.asset_id = intern("s");
    let mut sc = scaffold();
    sc.slot_boxes = (0..rows)
        .map(|i| intern(&format!("s_stage_slot{i}_box")))
        .collect();
    sc.slot_labels = (0..rows)
        .map(|i| intern(&format!("s_stage_slot{i}_lbl")))
        .collect();
    story.scaffold = sc;
    world.add_component(story);
    for view in ["s_stage", "s_ending"] {
        world.add_component(View {
            asset_id: intern(view),
            initial: view == "s_stage",
            fade_in_secs: 0.0,
        });
    }
    add_stage_furniture(&mut world);
    // Extra rows beyond slot0, which `add_stage_furniture` already provides.
    for i in 1..rows {
        world.add_component(Sprite {
            view: Some(intern("s_stage")),
            ..sprite_named(&format!("s_stage_slot{i}_box"))
        });
        world.add_component(TextLabel {
            view: Some(intern("s_stage")),
            ..label_named(&format!("s_stage_slot{i}_lbl"))
        });
    }
    world
}

// A world whose initial view is the generated title menu. The four menu
// button labels start at distinct emitted positions; the story lays them out
// (contiguous, centered, only the applicable buttons) from the save state on
// the first `ViewShown`. Mirrors the title-screen scaffold the build produces.
fn title_menu_world(story: Story) -> World {
    let mut world = World::new_empty();
    let mut story = story;
    story.asset_id = intern("s");
    let mut sc = scaffold();
    sc.title = Some(intern("s_title"));
    sc.start_label = Some(intern("s_title_start_lbl"));
    sc.continue_label = Some(intern("s_title_continue_lbl"));
    sc.load_label = Some(intern("s_title_load_lbl"));
    sc.quit_label = Some(intern("s_title_quit_lbl"));
    story.scaffold = sc;
    world.add_component(story);
    // The title menu is the initial screen; the stage and ending are inactive.
    for (view, initial) in [("s_title", true), ("s_stage", false), ("s_ending", false)] {
        world.add_component(View {
            asset_id: intern(view),
            initial,
            fade_in_secs: 0.0,
        });
    }
    add_stage_furniture(&mut world);
    // The four title buttons at distinct emitted y's (any values that differ
    // from the runtime stack positions, so a relayout is observable).
    for (name, y, text) in [
        ("s_title_start_lbl", 100.0, "Start"),
        ("s_title_continue_lbl", 200.0, "Continue"),
        ("s_title_load_lbl", 300.0, "Load"),
        ("s_title_quit_lbl", 400.0, "Quit"),
    ] {
        world.add_component(TextLabel {
            view: Some(intern("s_title")),
            content: text.to_string(),
            y,
            ..label_named(name)
        });
    }
    world
}

// A story world with the pause menu + settings + title screens the engine
// defaults inject, each with a member sprite the view system can show/hide so
// the active screen is observable. Mirrors the scaffold the build produces once
// the pause menu is injected and its views are threaded into the story.
fn story_world_with_pause(story: Story) -> World {
    let mut world = World::new_empty();
    let mut story = story;
    story.asset_id = intern("s");
    let mut sc = scaffold();
    sc.pause = Some(intern("s_pause"));
    sc.settings = Some(intern("s_settings"));
    sc.title = Some(intern("s_title"));
    sc.start_label = Some(intern("s_title_start_lbl"));
    sc.quit_label = Some(intern("s_title_quit_lbl"));
    sc.settings_label = Some(intern("s_title_settings_lbl"));
    story.scaffold = sc;
    world.add_component(story);
    for (view, initial) in [
        ("s_stage", true),
        ("s_ending", false),
        ("s_pause", false),
        ("s_settings", false),
        ("s_title", false),
    ] {
        world.add_component(View {
            asset_id: intern(view),
            initial,
            fade_in_secs: 0.0,
        });
    }
    // One member sprite per menu screen, so its visibility tracks whether the
    // screen is the active view.
    for (name, view) in [
        ("s_pause_dim", "s_pause"),
        ("s_settings_dim", "s_settings"),
        ("s_title_bg", "s_title"),
    ] {
        world.add_component(Sprite {
            view: Some(intern(view)),
            ..sprite_named(name)
        });
    }
    for lbl in [
        "s_title_start_lbl",
        "s_title_quit_lbl",
        "s_title_settings_lbl",
    ] {
        world.add_component(TextLabel {
            view: Some(intern("s_title")),
            ..label_named(lbl)
        });
    }
    add_stage_furniture(&mut world);
    world
}

fn sprite_visible(world: &World, name: &str) -> bool {
    let id = intern(name);
    world
        .query::<Sprite>()
        .find(|s| s.asset_id == id)
        .map(|s| s.visible)
        .unwrap_or(false)
}

fn label_content(world: &World, name: &str) -> String {
    let id = intern(name);
    world
        .query::<TextLabel>()
        .find(|l| l.asset_id == id)
        .map(|l| l.content.clone())
        .unwrap_or_default()
}

fn label_y(world: &World, name: &str) -> f32 {
    let id = intern(name);
    world
        .query::<TextLabel>()
        .find(|l| l.asset_id == id)
        .map(|l| l.y)
        .unwrap_or_default()
}

fn label_color(world: &World, name: &str) -> [f32; 3] {
    let id = intern(name);
    world
        .query::<TextLabel>()
        .find(|l| l.asset_id == id)
        .map(|l| l.color)
        .unwrap_or_default()
}

fn two_page_story() -> Story {
    Story {
        title: "T".to_string(),
        text_speed: 0.0,
        nodes: vec![StoryNode {
            slug: "a".to_string(),
            pages: vec![
                StoryPage {
                    speaker: Some(StorySpeaker {
                        name: "Ayame".to_string(),
                        color: [1.0, 0.85, 0.8],
                    }),
                    ..page("First page.")
                },
                page("Second page."),
            ],
            ..Default::default()
        }],
        ..Default::default()
    }
}

// The stage being the world's initial view auto-starts the story: the
// first page and its name plate appear without any command.
#[test]
fn initial_stage_view_auto_starts() {
    let mut world = story_world(two_page_story());
    world.start().unwrap();
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "First page.");
    assert_eq!(label_content(&world, "s_stage_name"), "Ayame");
}

// Advance walks pages and reaching past the last page shows the ending.
#[test]
fn advance_walks_pages_to_the_ending() {
    let mut world = story_world(two_page_story());
    world.start().unwrap();
    world.step();

    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Advance);
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "Second page.");
    // The narration page has no speaker: the plate empties.
    assert_eq!(label_content(&world, "s_stage_name"), "");

    // Advancing past the last page shows the ending view; once the
    // active view moves off the stage, further advances are ignored.
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Advance);
    world.step();
    world.step();
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Advance);
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "Second page.");
}

// A node ending in choices swaps the stage into menu mode: option labels
// fill, stage advances go inert, and picking an option jumps to its
// target node.
#[test]
fn choices_fill_buttons_and_choose_jumps() {
    let story = Story {
        title: "T".to_string(),
        text_speed: 0.0,
        nodes: vec![
            StoryNode {
                slug: "a".to_string(),
                pages: vec![page("Pick.")],
                choices: vec![StoryChoice {
                    label: "Go".to_string(),
                    target: 1,
                    condition: None,
                }],
                ..Default::default()
            },
            StoryNode {
                slug: "b".to_string(),
                pages: vec![page("Picked.")],
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    let mut world = story_world(story);
    world.start().unwrap();
    world.step();

    // A click over a (still hidden) choice button fires both the advance
    // region and the button. The advance opens the menu; the same tick's
    // choose is absorbed, so the click that turned the page cannot also
    // pick an option.
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Advance);
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Choose(0));
    world.step();
    assert_eq!(label_content(&world, "s_stage_opt0_lbl"), "Go");
    assert_eq!(label_content(&world, "s_stage_text"), "");
    let opt_box = intern("s_stage_opt0_box");
    let shown = world
        .query::<Sprite>()
        .find(|s| s.asset_id == opt_box)
        .unwrap();
    assert!(shown.visible);
    assert!(shown.tint[3] > 0.0, "occupied slot's box is opaque");

    // Advancing during a choice is ignored.
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Advance);
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "");

    // Choosing jumps to the target node and restores page mode.
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Choose(0));
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "Picked.");
    // Hidden furniture goes transparent (view re-activation force-shows
    // members, so `visible` cannot carry menu state).
    let alpha = world
        .query::<Sprite>()
        .find(|s| s.asset_id == opt_box)
        .unwrap()
        .tint[3];
    assert_eq!(alpha, 0.0);
}

// Page stage dressing applies to the stage sprites: the backdrop samples
// the page's texture and a portrait slot shows at its compiled rectangle.
#[test]
fn stage_dressing_applies_to_sprites() {
    let mut story = two_page_story();
    story.nodes[0].pages[0].stage = StoryStage {
        bg: Some(StoryImage {
            texture: TextureHandle(intern("s_img0").0),
            x: 0.0,
            y: 0.0,
            width: 1280.0,
            height: 720.0,
        }),
        center: Some(StoryImage {
            texture: TextureHandle(intern("s_img1").0),
            x: 412.0,
            y: 20.0,
            width: 456.0,
            height: 700.0,
        }),
        ..Default::default()
    };
    let mut world = story_world(story);
    world.start().unwrap();
    world.step();

    let bg = intern("s_stage_bg");
    let sprite = world.query::<Sprite>().find(|s| s.asset_id == bg).unwrap();
    assert_eq!(sprite.texture, Some(TextureHandle(intern("s_img0").0)));
    let center = intern("s_stage_center");
    let sprite = world
        .query::<Sprite>()
        .find(|s| s.asset_id == center)
        .unwrap();
    assert!(sprite.visible);
    assert_eq!(sprite.width, 456.0);
    assert_eq!(sprite.y, 20.0);

    // The next page has no dressing: the portrait hides and the backdrop
    // falls back to its flat fill.
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Advance);
    world.step();
    let sprite = world
        .query::<Sprite>()
        .find(|s| s.asset_id == center)
        .unwrap();
    assert_eq!(sprite.tint[3], 0.0);
    let sprite = world.query::<Sprite>().find(|s| s.asset_id == bg).unwrap();
    assert_eq!(sprite.texture, None);
}

// Page audio is requested through PlayCue events the audio system plays.
#[test]
fn page_audio_sends_play_cues() {
    let mut story = two_page_story();
    // The story graph carries pre-resolved AudioClipHandles (cook resolves the
    // clip names at build time); a hand-built graph sets them directly.
    story.nodes[0].pages[0].music = Some(AudioClipHandle(0));
    story.nodes[0].pages[0].sounds = vec![AudioClipHandle(1)];
    let mut world = story_world(story);
    world.start().unwrap();
    world.step();

    let mut cursor = crate::ecs::EventCursor::default();
    let cues: Vec<PlayCue> = world
        .events_mut::<PlayCue>()
        .read(&mut cursor)
        .into_iter()
        .copied()
        .collect();
    assert_eq!(cues.len(), 2);
    assert_eq!(cues[0].clip, AudioClipHandle(0));
    assert_eq!(cues[0].kind, CueKind::Music);
    assert_eq!(cues[1].kind, CueKind::Sound);
}

// With a non-zero speed the page reveals over time instead of appearing
// whole; an advance mid-reveal completes it rather than turning the page.
#[test]
fn typewriter_reveals_and_advance_completes() {
    let mut story = two_page_story();
    // Slow enough that no character appears within the test's runtime.
    story.text_speed = 0.0001;
    let mut world = story_world(story);
    world.start().unwrap();
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "");

    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Advance);
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "First page.");

    // The page is complete: the next advance turns it, and the new page
    // starts its own reveal.
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Advance);
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "");
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Advance);
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "Second page.");
}

// Page ops raise flags and a gate on the next page redirects play past
// it while the flag is set.
#[test]
fn ops_raise_flags_and_gates_redirect() {
    use crate::assets::{StoryGate, StoryOp};
    let story = Story {
        title: "T".to_string(),
        text_speed: 0.0,
        nodes: vec![
            StoryNode {
                slug: "a".to_string(),
                pages: vec![
                    StoryPage {
                        ops: vec![StoryOp {
                            name: "asked".to_string(),
                            value: 1,
                            add: false,
                        }],
                        ..page("Intro.")
                    },
                    StoryPage {
                        gates: vec![StoryGate {
                            name: "asked".to_string(),
                            op: CmpOp::Ne,
                            value: 0,
                            target: 1,
                        }],
                        ..page("Skipped.")
                    },
                ],
                ..Default::default()
            },
            StoryNode {
                slug: "b".to_string(),
                pages: vec![page("Landed.")],
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    let mut world = story_world(story);
    world.start().unwrap();
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "Intro.");
    // The intro's op set `asked`; the second page's gate fires instead
    // of showing it.
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Advance);
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "Landed.");
}

// A gated option stays off the menu; the button slots fill from the
// visible options, and picking maps back to the right target.
#[test]
fn gated_choices_filter_and_remap() {
    use crate::assets::StoryCondition;
    let story = Story {
        title: "T".to_string(),
        text_speed: 0.0,
        nodes: vec![
            StoryNode {
                slug: "a".to_string(),
                pages: vec![page("Pick.")],
                choices: vec![
                    StoryChoice {
                        label: "Secret".to_string(),
                        target: 1,
                        condition: Some(StoryCondition {
                            name: "secret".to_string(),
                            op: CmpOp::Ne,
                            value: 0,
                        }),
                    },
                    StoryChoice {
                        label: "Plain".to_string(),
                        target: 2,
                        condition: None,
                    },
                ],
                ..Default::default()
            },
            StoryNode {
                slug: "hidden".to_string(),
                pages: vec![page("Never.")],
                ..Default::default()
            },
            StoryNode {
                slug: "plain".to_string(),
                pages: vec![page("Plainly.")],
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    let mut world = story_world(story);
    world.start().unwrap();
    world.step();
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Advance);
    world.step();
    // The gated option is hidden, so the first button is "Plain".
    assert_eq!(label_content(&world, "s_stage_opt0_lbl"), "Plain");
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Choose(0));
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "Plainly.");
}

// A menu whose gate passes redirects play instead of opening.
#[test]
fn menu_gates_redirect_past_the_menu() {
    use crate::assets::{StoryGate, StoryOp};
    let story = Story {
        title: "T".to_string(),
        text_speed: 0.0,
        nodes: vec![
            StoryNode {
                slug: "a".to_string(),
                pages: vec![StoryPage {
                    ops: vec![StoryOp {
                        name: "skip".to_string(),
                        value: 1,
                        add: false,
                    }],
                    ..page("Once.")
                }],
                choices: vec![StoryChoice {
                    label: "Never shown".to_string(),
                    target: 0,
                    condition: None,
                }],
                choice_gates: vec![StoryGate {
                    name: "skip".to_string(),
                    op: CmpOp::Ne,
                    value: 0,
                    target: 1,
                }],
                ..Default::default()
            },
            StoryNode {
                slug: "b".to_string(),
                pages: vec![page("Landed.")],
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    let mut world = story_world(story);
    world.start().unwrap();
    world.step();
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Advance);
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "Landed.");
    // The menu never opened: its first button was never filled.
    assert_eq!(label_content(&world, "s_stage_opt0_lbl"), "");
}

// The save file round-trips position and variables; a missing file
// reads as None and an unreadable one starts fresh.
#[test]
fn story_save_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let auto = save_file(dir.path());
    assert!(read_save(&auto).is_none());
    let save = StorySave {
        slug: "meadow".to_string(),
        page: 2,
        vars: [("asked".to_string(), 1), ("gold".to_string(), 7)]
            .into_iter()
            .collect(),
    };
    write_save(&auto, &save).unwrap();
    let back = read_save(&auto).expect("save exists");
    assert_eq!(back.slug, "meadow");
    assert_eq!(back.page, 2);
    assert_eq!(back.vars.get("gold"), Some(&7));
    // Corruption falls back to a fresh start rather than a panic.
    std::fs::write(&auto, b"not cbor").unwrap();
    assert!(read_save(&auto).is_none());
    // Slot files sit beside the auto-save, one per slot.
    let slot = slot_file(dir.path(), 1);
    write_save(&slot, &save).unwrap();
    assert_eq!(read_save(&slot).unwrap().slug, "meadow");
}

// Continue without any save behaves exactly like Start.
#[test]
fn continue_without_a_save_starts_fresh() {
    let mut world = story_world(two_page_story());
    world.start().unwrap();
    world.step();
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Continue);
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "First page.");
}

// A jump page overrides the next-page order.
#[test]
fn jump_pages_target_their_node() {
    let story = Story {
        title: "T".to_string(),
        text_speed: 0.0,
        nodes: vec![
            StoryNode {
                slug: "a".to_string(),
                pages: vec![StoryPage {
                    jump: Some(2),
                    ..page("Jump away.")
                }],
                ..Default::default()
            },
            StoryNode {
                slug: "skipped".to_string(),
                pages: vec![page("Never seen.")],
                ..Default::default()
            },
            StoryNode {
                slug: "c".to_string(),
                pages: vec![page("Landed.")],
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    let mut world = story_world(story);
    world.start().unwrap();
    world.step();
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Advance);
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "Landed.");
}

// A hot-reloaded graph swaps in without losing the play position: the
// revised page under the cursor re-renders in full.
#[test]
fn reload_swaps_the_graph_in_place() {
    let mut world = story_world(two_page_story());
    world.start().unwrap();
    world.step();
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Advance);
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "Second page.");

    let mut edited = two_page_story();
    edited.nodes[0].pages[1] = page("Second page, revised.");
    edited.scaffold = scaffold();
    world
        .events_mut::<StoryReload>()
        .send(StoryReload { story: edited });
    world.step();
    assert_eq!(
        label_content(&world, "s_stage_text"),
        "Second page, revised."
    );

    // A reload for some other story's stage is ignored.
    let mut foreign = two_page_story();
    foreign.nodes[0].pages[1] = page("Not ours.");
    world
        .events_mut::<StoryReload>()
        .send(StoryReload { story: foreign });
    world.step();
    assert_eq!(
        label_content(&world, "s_stage_text"),
        "Second page, revised."
    );
}

// Deleting the node under the cursor restarts the story from the top.
#[test]
fn reload_with_the_node_deleted_restarts() {
    let mut world = story_world(two_page_story());
    world.start().unwrap();
    world.step();

    let mut edited = two_page_story();
    edited.nodes[0].slug = "renamed".to_string();
    edited.nodes[0].pages[0] = page("Fresh start.");
    edited.scaffold = scaffold();
    world
        .events_mut::<StoryReload>()
        .send(StoryReload { story: edited });
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "Fresh start.");
}

// A reload while a menu is open re-renders the revised options and stays
// in choice mode.
#[test]
fn reload_refreshes_an_open_menu() {
    let choice_story = |label: &str| Story {
        title: "T".to_string(),
        text_speed: 0.0,
        nodes: vec![
            StoryNode {
                slug: "a".to_string(),
                pages: vec![page("Pick.")],
                choices: vec![StoryChoice {
                    label: label.to_string(),
                    target: 1,
                    condition: None,
                }],
                ..Default::default()
            },
            StoryNode {
                slug: "b".to_string(),
                pages: vec![page("Picked.")],
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    let mut world = story_world(choice_story("Go"));
    world.start().unwrap();
    world.step();
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Advance);
    world.step();
    assert_eq!(label_content(&world, "s_stage_opt0_lbl"), "Go");

    let mut edited = choice_story("Go north");
    edited.scaffold = scaffold();
    world
        .events_mut::<StoryReload>()
        .send(StoryReload { story: edited });
    world.step();
    assert_eq!(label_content(&world, "s_stage_opt0_lbl"), "Go north");

    // Still a menu: advancing stays inert, choosing works.
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Advance);
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "");
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Choose(0));
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "Picked.");
}

// Numeric ops accumulate across pages and a comparison gate fires once
// the threshold is met.
#[test]
fn numeric_ops_accumulate_and_comparisons_gate() {
    use crate::assets::{StoryGate, StoryOp};
    let add_trip = StoryOp {
        name: "trips".to_string(),
        value: 1,
        add: true,
    };
    let story = Story {
        title: "T".to_string(),
        text_speed: 0.0,
        nodes: vec![
            StoryNode {
                slug: "a".to_string(),
                pages: vec![
                    StoryPage {
                        ops: vec![add_trip.clone()],
                        ..page("One.")
                    },
                    StoryPage {
                        ops: vec![add_trip.clone()],
                        ..page("Two.")
                    },
                    StoryPage {
                        gates: vec![StoryGate {
                            name: "trips".to_string(),
                            op: CmpOp::Ge,
                            value: 2,
                            target: 1,
                        }],
                        ..page("Never shown.")
                    },
                ],
                ..Default::default()
            },
            StoryNode {
                slug: "counted".to_string(),
                pages: vec![page("Counted.")],
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    let mut world = story_world(story);
    world.start().unwrap();
    world.step();
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Advance);
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "Two.");
    // trips == 2 now: the third page's gate redirects.
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Advance);
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "Counted.");
}

// The quick row fills in page mode; toggling a mode re-tints its label
// and the same click's advance half is suppressed.
#[test]
fn quick_row_toggles_and_suppresses_the_same_click_advance() {
    let mut world = story_world(two_page_story());
    world.start().unwrap();
    world.step();
    assert_eq!(label_content(&world, "s_stage_qauto_lbl"), "Auto");
    assert_eq!(label_content(&world, "s_stage_qsave_lbl"), "Save");

    // The button click fires both its action and the full-canvas
    // advance; the page must not turn.
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::ToggleAuto);
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Advance);
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "First page.");
    let auto_id = intern("s_stage_qauto_lbl");
    let color = world
        .query::<TextLabel>()
        .find(|l| l.asset_id == auto_id)
        .unwrap()
        .color;
    assert_eq!(color, QUICK_ACTIVE);
}

// The backlog overlay lists shown pages; a plain click dismisses it and
// restores the covered page.
#[test]
fn backlog_lists_history_and_a_click_dismisses() {
    let mut world = story_world(two_page_story());
    world.start().unwrap();
    world.step();
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Advance);
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "Second page.");

    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::ToggleLog);
    world.step();
    let history = label_content(&world, "s_stage_history");
    assert!(history.contains("Ayame: First page."));
    assert!(history.contains("Second page."));
    // The covered page furniture is cleared under the dim.
    assert_eq!(label_content(&world, "s_stage_text"), "");
    assert_eq!(label_content(&world, "s_stage_qauto_lbl"), "");

    // While the overlay is up, an advance dismisses it (and only that).
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Advance);
    world.step();
    assert_eq!(label_content(&world, "s_stage_history"), "");
    assert_eq!(label_content(&world, "s_stage_text"), "Second page.");
}

// Slot bookkeeping: summaries name the saved position, and any existing
// slot lights the title's Load.
#[test]
fn slot_summaries_and_slot_presence() {
    let dir = tempfile::tempdir().unwrap();
    let mut story = two_page_story();
    story.save_key = "q".to_string();
    story.scaffold = scaffold();
    let mut sys = StorySystem::new(story);
    sys.save_dir = dir.path().to_path_buf();
    sys.ids = StageIds::from_scaffold(&scaffold());
    assert!(!sys.any_slot_save());
    assert!(sys.slot_summary(0).contains("empty"));

    let save = StorySave {
        slug: "a".to_string(),
        page: 1,
        vars: BTreeMap::new(),
    };
    write_save(&slot_file(dir.path(), 0), &save).unwrap();
    assert!(sys.any_slot_save());
    let summary = sys.slot_summary(0);
    assert!(summary.contains("a"), "{summary}");
    assert!(summary.contains("page 2"), "{summary}");
}

// The save overlay writes the picked slot and dismisses; play resumes
// from a loaded slot.
#[test]
fn save_overlay_writes_and_load_resumes() {
    let dir = tempfile::tempdir().unwrap();
    let mut story = two_page_story();
    story.save_key = "slot_test".to_string();
    let mut world = story_world(story);
    world.start().unwrap();
    // Point the world-constructed system's saves at the temp directory
    // (the process-global state root must stay untouched: tests share
    // it).
    for system in world.systems_mut() {
        if let crate::ecs::SystemAsset::StorySystem(s) = system {
            s.save_dir = dir.path().to_path_buf();
        }
    }
    world.step();
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Advance);
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "Second page.");

    // Open the save overlay and pick slot 1.
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::OpenSave);
    world.step();
    assert!(label_content(&world, "s_stage_slot_title").contains("Save"));
    assert!(label_content(&world, "s_stage_slot0_lbl").contains("empty"));
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Slot(0));
    world.step();
    // Overlay dismissed, page restored, slot written.
    assert_eq!(label_content(&world, "s_stage_text"), "Second page.");
    let saved = read_save(&slot_file(dir.path(), 0)).expect("slot written");
    assert_eq!(saved.page, 1);

    // Restart, then load the slot back: play resumes at page 2.
    world.events_mut::<StoryCommand>().send(StoryCommand::Start);
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "First page.");
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::OpenLoad);
    world.step();
    assert!(label_content(&world, "s_stage_slot0_lbl").contains("page 2"));
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Slot(0));
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "Second page.");
}

// Holding the skip modifier (Control) fast-forwards: the current page snaps to
// full and the Skip control reads engaged; releasing the key clears it. A slow
// reveal makes the instant snap observable.
#[test]
fn holding_ctrl_fast_forwards_and_lights_skip() {
    let mut story = two_page_story();
    story.text_speed = 0.0001;
    let mut world = story_world(story);
    world.start().unwrap();
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "");

    // Hold Control: the page reveals at once and Skip lights up.
    world.add_component(FrameInput {
        ctrl: true,
        ..Default::default()
    });
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "First page.");
    assert_eq!(label_color(&world, "s_stage_qskip_lbl"), QUICK_ACTIVE);

    // Release: Skip returns to idle.
    world.add_component(FrameInput::default());
    world.step();
    assert_eq!(label_color(&world, "s_stage_qskip_lbl"), QUICK_IDLE);
}

// The slot overlay shows a fixed window of rows and scrolls it over the full
// slot set: the wheel shifts which logical slots the rows show, the title
// reports the visible range, and a picked row maps through the scroll offset.
#[test]
fn slot_overlay_scrolls_the_window_over_all_slots() {
    let dir = tempfile::tempdir().unwrap();
    let mut story = two_page_story();
    story.save_key = "scroll_test".to_string();
    let mut world = multi_slot_world(story, 3);
    world.start().unwrap();
    for system in world.systems_mut() {
        if let crate::ecs::SystemAsset::StorySystem(s) = system {
            s.save_dir = dir.path().to_path_buf();
        }
    }
    // A save in a far slot (index 6) so a scrolled row can show and load it.
    let save = StorySave {
        slug: "a".to_string(),
        page: 1,
        vars: BTreeMap::new(),
    };
    write_save(&slot_file(dir.path(), 6), &save).unwrap();
    world.step();

    // Open the load overlay: the window starts at slot 1, showing three rows.
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::OpenLoad);
    world.step();
    assert!(label_content(&world, "s_stage_slot0_lbl").contains("Slot 1"));
    assert!(label_content(&world, "s_stage_slot_title").contains("1-3 / 10"));

    // Wheel down six rows: the window shifts and row 0 shows the far slot.
    world.add_component(FrameInput {
        scroll_delta: SLOT_SCROLL_UNIT * 6.0,
        ..Default::default()
    });
    world.step();
    assert!(label_content(&world, "s_stage_slot0_lbl").contains("Slot 7"));
    assert!(label_content(&world, "s_stage_slot0_lbl").contains("page 2"));
    assert!(label_content(&world, "s_stage_slot_title").contains("7-9 / 10"));

    // Picking row 0 now loads the far slot (index 6): play resumes at page 2.
    world.add_component(FrameInput::default());
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Slot(0));
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "Second page.");
}

// Saving from a menu shown over the story (the injected pause menu) raises the
// hidden stage and opens the slot overlay on it, even though a different view
// -- not the stage -- is active. Without the raise the slot furniture (all
// stage-view members) would stay hidden behind the menu.
#[test]
fn pause_menu_save_raises_stage_and_opens_slots() {
    let dir = tempfile::tempdir().unwrap();
    let mut story = two_page_story();
    story.save_key = "pause_save".to_string();
    let mut world = story_world(story);
    world.start().unwrap();
    for system in world.systems_mut() {
        if let crate::ecs::SystemAsset::StorySystem(s) = system {
            s.save_dir = dir.path().to_path_buf();
        }
    }
    world.step();
    // A pause menu becomes the active view over the started story.
    world.events_mut::<ViewShown>().send(ViewShown {
        view: intern("s_pause"),
    });
    world.step();

    // Save from the pause menu opens the slot overlay on the raised stage.
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::OpenSave);
    world.step();
    assert!(label_content(&world, "s_stage_slot_title").contains("Save"));

    // Picking a slot writes the current position.
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Slot(0));
    world.step();
    let saved = read_save(&slot_file(dir.path(), 0)).expect("slot written");
    assert_eq!(saved.page, 0);
}

// Escape opens the pause over the stage; Escape (or Resume) again returns to
// the stage rather than dismissing to no view (which would render nothing).
#[test]
fn pause_toggle_returns_to_the_stage_not_a_blank_view() {
    let mut world = story_world_with_pause(two_page_story());
    world.start().unwrap();
    world.step();
    assert!(sprite_visible(&world, "s_stage_bg"), "stage starts visible");

    // Escape from the stage opens the pause menu over it.
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::TogglePause);
    world.step();
    world.step();
    assert!(
        sprite_visible(&world, "s_pause_dim"),
        "pause menu shown on Escape"
    );
    assert!(
        !sprite_visible(&world, "s_stage_bg"),
        "stage hidden behind the pause"
    );

    // Escape again returns to the stage (the reported blank-screen bug).
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::TogglePause);
    world.step();
    world.step();
    assert!(
        sprite_visible(&world, "s_stage_bg"),
        "resume returns to the stage, not a blank view"
    );
    assert!(!sprite_visible(&world, "s_pause_dim"));
}

// The settings screen returns, on Back, to whichever menu opened it: the pause
// menu when opened mid-game, the title when opened from the title.
#[test]
fn settings_back_returns_to_the_opener() {
    let mut world = story_world_with_pause(two_page_story());
    world.start().unwrap();
    world.step();

    // Pause -> Settings -> Back returns to the pause menu.
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::TogglePause);
    world.step();
    world.step();
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::OpenSettings);
    world.step();
    world.step();
    assert!(sprite_visible(&world, "s_settings_dim"), "settings shown");
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::CloseSettings);
    world.step();
    world.step();
    assert!(
        sprite_visible(&world, "s_pause_dim"),
        "Back from settings returns to the pause menu"
    );

    // Title -> Settings -> Back returns to the title.
    world
        .events_mut::<ViewCommand>()
        .send(ViewCommand::Show(intern("s_title")));
    world.step();
    world.step();
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::OpenSettings);
    world.step();
    world.step();
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::CloseSettings);
    world.step();
    world.step();
    assert!(
        sprite_visible(&world, "s_title_bg"),
        "Back from settings opened at the title returns to the title"
    );
}

// Returning to the title (Quit-to-Title, or the ending's Back) ends the
// playthrough, so the title's Load opens the slot overlay. Regression: with
// `started` left set, Load fell through open_load's not-page-mode guard and was
// a dead button on every title visit after the first play.
#[test]
fn title_load_works_after_returning_to_the_title() {
    let dir = tempfile::tempdir().unwrap();
    let mut story = two_page_story();
    story.save_key = "replay".to_string();
    let mut world = story_world_with_pause(story);
    world.start().unwrap();
    for system in world.systems_mut() {
        if let crate::ecs::SystemAsset::StorySystem(s) = system {
            s.save_dir = dir.path().to_path_buf();
        }
    }
    world.step(); // stage active, started == true
    let save = StorySave {
        slug: "a".to_string(),
        page: 1,
        vars: BTreeMap::new(),
    };
    write_save(&slot_file(dir.path(), 0), &save).unwrap();

    // Quit to the title (a plain view:show the pause menu's Quit fires).
    world
        .events_mut::<ViewCommand>()
        .send(ViewCommand::Show(intern("s_title")));
    world.step();
    world.step();
    assert!(sprite_visible(&world, "s_title_bg"), "title shown");

    // Load from the title now opens the slot overlay (dead before the fix).
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::OpenLoad);
    world.step();
    world.step();
    assert!(
        label_content(&world, "s_stage_slot_title").contains("Load"),
        "load overlay opens from the title after a playthrough"
    );
}

// A choice button's hit region overlaps a slot row's (always-active) hit
// region, so a click on the option fires both Choose and a stray Slot. The
// slot command is a no-op outside the slot overlay and must not suppress
// the choose, or the first option would stop working.
#[test]
fn choice_survives_a_spurious_slot_click() {
    let story = Story {
        title: "T".to_string(),
        text_speed: 0.0,
        nodes: vec![
            StoryNode {
                slug: "a".to_string(),
                pages: vec![page("Pick.")],
                choices: vec![StoryChoice {
                    label: "Go".to_string(),
                    target: 1,
                    condition: None,
                }],
                ..Default::default()
            },
            StoryNode {
                slug: "b".to_string(),
                pages: vec![page("Picked.")],
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    let mut world = story_world(story);
    world.start().unwrap();
    world.step();
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Advance);
    world.step();
    assert_eq!(label_content(&world, "s_stage_opt0_lbl"), "Go");

    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Choose(0));
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Slot(1));
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "Picked.");
}

// Each menu button's follow-label hit region captures its fixed offset to its
// label in `UiInputSystem::init`, which runs after `StorySystem::init`. So the
// story must NOT lay the title menu out during its own init: moving the labels
// then would slide them out from under that capture, pinning every hit region
// at its emitted position instead of tracking the runtime layout (the cursor
// would then hit each button in the wrong place). Guard both halves: init
// leaves the emitted label positions untouched, and the first title
// `ViewShown` (announced during that same init) lays out only the applicable
// buttons.
#[test]
fn title_menu_lays_out_on_first_shown_not_at_init() {
    let dir = tempfile::tempdir().unwrap();
    let mut story = two_page_story();
    story.save_key = "titletest".to_string();
    let mut world = title_menu_world(story);
    world.start().unwrap();

    // After init the emitted positions are untouched: had init laid the menu
    // out, the follow-region offsets would already be baked in wrong.
    assert_eq!(label_y(&world, "s_title_start_lbl"), 100.0);
    assert_eq!(label_y(&world, "s_title_quit_lbl"), 400.0);

    // Point saves at an empty directory so the layout is the two-button case.
    for system in world.systems_mut() {
        if let crate::ecs::SystemAsset::StorySystem(s) = system {
            s.save_dir = dir.path().to_path_buf();
        }
    }

    // The first step consumes the initial title `ViewShown` and lays the menu
    // out: with no save data, only Start and Quit, stacked contiguously and
    // centered (no gap where Continue and Load would sit).
    world.step();
    let start = label_y(&world, "s_title_start_lbl");
    let quit = label_y(&world, "s_title_quit_lbl");
    assert_eq!(start, TITLE_MENU_CENTER_Y - TITLE_MENU_SPACING / 2.0);
    assert_eq!(quit, TITLE_MENU_CENTER_Y + TITLE_MENU_SPACING / 2.0);
    assert_eq!(quit - start, TITLE_MENU_SPACING);
    assert_eq!(label_content(&world, "s_title_start_lbl"), "Start");
    assert_eq!(label_content(&world, "s_title_quit_lbl"), "Quit");
    // The absent buttons are emptied so their follow-regions go inert.
    assert_eq!(label_content(&world, "s_title_continue_lbl"), "");
    assert_eq!(label_content(&world, "s_title_load_lbl"), "");
}

// Redirect the world's StorySystem saves at a temp directory (the process
// global state root must stay untouched: tests share it).
fn point_saves(world: &mut World, dir: &std::path::Path) {
    for system in world.systems_mut() {
        if let crate::ecs::SystemAsset::StorySystem(s) = system {
            s.save_dir = dir.to_path_buf();
        }
    }
}

// Backdate the story clock so the next step sees `secs` of elapsed time,
// driving the timed reader-assist paths (typewriter / auto / skip) without a
// real sleep. `last_step` is a private field, reachable from this in-crate
// test module.
fn backdate_clock(world: &mut World, secs: f32) {
    for system in world.systems_mut() {
        if let crate::ecs::SystemAsset::StorySystem(s) = system {
            s.last_step = Some(Instant::now() - std::time::Duration::from_secs_f32(secs));
        }
    }
}

// A story world whose stage carries two option slots (the shared scaffold has
// one), so an unoccupied slot is observable when a menu has fewer choices.
fn two_option_world(story: Story) -> World {
    let mut world = World::new_empty();
    let mut story = story;
    story.asset_id = intern("s");
    let mut sc = scaffold();
    sc.option_boxes = vec![intern("s_stage_opt0_box"), intern("s_stage_opt1_box")];
    sc.options = vec![intern("s_stage_opt0_lbl"), intern("s_stage_opt1_lbl")];
    story.scaffold = sc;
    world.add_component(story);
    for view in ["s_stage", "s_ending"] {
        world.add_component(View {
            asset_id: intern(view),
            initial: view == "s_stage",
            fade_in_secs: 0.0,
        });
    }
    add_stage_furniture(&mut world);
    // The second option slot beyond slot0 that add_stage_furniture provides.
    world.add_component(Sprite {
        view: Some(intern("s_stage")),
        ..sprite_named("s_stage_opt1_box")
    });
    world.add_component(TextLabel {
        view: Some(intern("s_stage")),
        ..label_named("s_stage_opt1_lbl")
    });
    world
}

// A single choice node linking to a landing node (shared by several menu
// tests). `choice_sounds` is optional so an audio-free variant stays silent.
fn one_choice_story(label: &str, sounds: Vec<AudioClipHandle>) -> Story {
    Story {
        title: "T".to_string(),
        text_speed: 0.0,
        nodes: vec![
            StoryNode {
                slug: "a".to_string(),
                pages: vec![page("Pick.")],
                choices: vec![StoryChoice {
                    label: label.to_string(),
                    target: 1,
                    condition: None,
                }],
                choice_sounds: sounds,
                ..Default::default()
            },
            StoryNode {
                slug: "b".to_string(),
                pages: vec![page("Picked.")],
                ..Default::default()
            },
        ],
        ..Default::default()
    }
}

// With a positive speed the page reveals character by character as clip time
// advances, rather than appearing whole.
#[test]
fn typewriter_reveals_the_page_over_time() {
    let mut story = two_page_story();
    story.text_speed = 100.0;
    let mut world = story_world(story);
    world.start().unwrap();
    world.step();
    // First step's dt is zero, so nothing has revealed yet.
    assert_eq!(label_content(&world, "s_stage_text"), "");

    // Half a second at 100 cps budgets far more than the page's length.
    backdate_clock(&mut world, 0.5);
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "First page.");
}

// The Skip toggle snaps the current page to full immediately and lights the
// Skip control.
#[test]
fn toggle_skip_snaps_the_current_page_to_full() {
    let mut story = two_page_story();
    story.text_speed = 0.0001;
    let mut world = story_world(story);
    world.start().unwrap();
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "");

    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::ToggleSkip);
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "First page.");
    assert_eq!(label_color(&world, "s_stage_qskip_lbl"), QUICK_ACTIVE);
}

// A quick-row toggle is inert outside page mode: during a choice menu the row
// is cleared and a ToggleSkip must not repaint or engage it.
#[test]
fn quick_toggle_is_inert_during_a_choice_menu() {
    let mut world = story_world(one_choice_story("Go", Vec::new()));
    world.start().unwrap();
    world.step();
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Advance);
    world.step();
    // The menu cleared the quick row.
    assert_eq!(label_content(&world, "s_stage_qskip_lbl"), "");
    // ToggleSkip out of page mode does nothing, so the row stays cleared.
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::ToggleSkip);
    world.step();
    assert_eq!(label_content(&world, "s_stage_qskip_lbl"), "");
}

// Skip mode turns fully revealed pages at its rapid cadence once enough clip
// time has passed.
#[test]
fn skip_mode_turns_pages_at_its_cadence() {
    let mut world = story_world(two_page_story());
    world.start().unwrap();
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "First page.");

    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::ToggleSkip);
    // A full second exceeds the skip page cadence.
    backdate_clock(&mut world, 1.0);
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "Second page.");
}

// Auto mode turns a fully revealed page after its reading-time delay elapses.
#[test]
fn auto_mode_turns_a_read_page_after_the_delay() {
    let mut world = story_world(two_page_story());
    world.start().unwrap();
    world.step();

    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::ToggleAuto);
    // Ten seconds far exceeds the base pause plus per-character reading time.
    backdate_clock(&mut world, 10.0);
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "Second page.");
}

// A node with no pages, only choices, opens its menu directly on entry.
#[test]
fn pageless_node_enters_its_choice_menu_directly() {
    let story = Story {
        title: "T".to_string(),
        text_speed: 0.0,
        nodes: vec![
            StoryNode {
                slug: "a".to_string(),
                pages: Vec::new(),
                choices: vec![StoryChoice {
                    label: "Go".to_string(),
                    target: 1,
                    condition: None,
                }],
                ..Default::default()
            },
            StoryNode {
                slug: "b".to_string(),
                pages: vec![page("Picked.")],
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    let mut world = story_world(story);
    world.start().unwrap();
    world.step();
    assert_eq!(label_content(&world, "s_stage_opt0_lbl"), "Go");
}

// A passing choice gate on a pageless node redirects play past the menu on
// entry.
#[test]
fn pageless_choice_gate_redirects_on_entry() {
    use crate::assets::StoryGate;
    let story = Story {
        title: "T".to_string(),
        text_speed: 0.0,
        nodes: vec![
            StoryNode {
                slug: "a".to_string(),
                pages: Vec::new(),
                choices: vec![StoryChoice {
                    label: "Never".to_string(),
                    target: 0,
                    condition: None,
                }],
                choice_gates: vec![StoryGate {
                    name: "x".to_string(),
                    op: CmpOp::Eq,
                    value: 0,
                    target: 1,
                }],
                ..Default::default()
            },
            StoryNode {
                slug: "b".to_string(),
                pages: vec![page("Landed.")],
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    let mut world = story_world(story);
    world.start().unwrap();
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "Landed.");
}

// A first-page gate redirects the moment its node is entered, before the page
// is shown.
#[test]
fn first_page_gate_redirects_on_node_entry() {
    use crate::assets::StoryGate;
    let story = Story {
        title: "T".to_string(),
        text_speed: 0.0,
        nodes: vec![
            StoryNode {
                slug: "a".to_string(),
                pages: vec![StoryPage {
                    gates: vec![StoryGate {
                        name: "x".to_string(),
                        op: CmpOp::Eq,
                        value: 0,
                        target: 1,
                    }],
                    ..page("Skipped.")
                }],
                ..Default::default()
            },
            StoryNode {
                slug: "b".to_string(),
                pages: vec![page("Landed.")],
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    let mut world = story_world(story);
    world.start().unwrap();
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "Landed.");
}

// A pageless node whose every choice is gated off falls through to the next
// node, like a menu-less node.
#[test]
fn all_gated_choices_fall_through() {
    use crate::assets::StoryCondition;
    let story = Story {
        title: "T".to_string(),
        text_speed: 0.0,
        nodes: vec![
            StoryNode {
                slug: "a".to_string(),
                pages: Vec::new(),
                choices: vec![StoryChoice {
                    label: "Secret".to_string(),
                    target: 2,
                    condition: Some(StoryCondition {
                        name: "secret".to_string(),
                        op: CmpOp::Ne,
                        value: 0,
                    }),
                }],
                ..Default::default()
            },
            StoryNode {
                slug: "b".to_string(),
                pages: vec![page("Fell through.")],
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    let mut world = story_world(story);
    world.start().unwrap();
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "Fell through.");
}

// Two pageless nodes gating to each other form a cycle; the hop budget stops
// the loop instead of spinning forever, landing on no page.
#[test]
fn gate_loop_is_stopped_by_the_hop_limit() {
    use crate::assets::StoryGate;
    let dummy = || StoryChoice {
        label: "x".to_string(),
        target: 0,
        condition: None,
    };
    let gate_to = |target: u32| StoryGate {
        name: "x".to_string(),
        op: CmpOp::Eq,
        value: 0,
        target,
    };
    let story = Story {
        title: "T".to_string(),
        text_speed: 0.0,
        nodes: vec![
            StoryNode {
                slug: "a".to_string(),
                pages: Vec::new(),
                choices: vec![dummy()],
                choice_gates: vec![gate_to(1)],
                ..Default::default()
            },
            StoryNode {
                slug: "b".to_string(),
                pages: Vec::new(),
                choices: vec![dummy()],
                choice_gates: vec![gate_to(0)],
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    let mut world = story_world(story);
    world.start().unwrap();
    // Must terminate rather than hang; nothing lands on the stage.
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "");
}

// A page reached under an active skip run is snapped to full on arrival by
// render_page, rather than starting its own reveal.
#[test]
fn skip_snaps_a_freshly_entered_page_to_full() {
    let mut story = two_page_story();
    story.text_speed = 0.0001;
    let mut world = story_world(story);
    world.start().unwrap();
    world.step();

    // Skip on: snaps page 1 now.
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::ToggleSkip);
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "First page.");

    // The skip cadence turns to page 2, which render_page snaps on arrival
    // despite the slow speed.
    backdate_clock(&mut world, 1.0);
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "Second page.");
}

// Entering a choice menu fires the node's one-shot choice sounds as PlayCue
// events.
#[test]
fn choice_menu_fires_its_one_shot_sounds() {
    let click = AudioClipHandle(0);
    let mut world = story_world(one_choice_story("Go", vec![click]));
    world.start().unwrap();
    world.step();
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Advance);
    world.step();

    let mut cursor = crate::ecs::EventCursor::default();
    let cues: Vec<PlayCue> = world
        .events_mut::<PlayCue>()
        .read(&mut cursor)
        .into_iter()
        .copied()
        .collect();
    assert!(
        cues.iter()
            .any(|c| c.clip == click && c.kind == CueKind::Sound),
        "choice sound cue fired: {cues:?}"
    );
}

// A menu with fewer options than slots leaves the spare slots blank and their
// boxes transparent, while the occupied slot's box stays opaque.
#[test]
fn unoccupied_option_slots_blank_and_go_transparent() {
    let mut world = two_option_world(one_choice_story("Go", Vec::new()));
    world.start().unwrap();
    world.step();
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Advance);
    world.step();

    assert_eq!(label_content(&world, "s_stage_opt0_lbl"), "Go");
    assert_eq!(label_content(&world, "s_stage_opt1_lbl"), "");
    let box1 = world
        .query::<Sprite>()
        .find(|s| s.asset_id == intern("s_stage_opt1_box"))
        .unwrap();
    assert_eq!(box1.tint[3], 0.0, "empty slot box is transparent");
    let box0 = world
        .query::<Sprite>()
        .find(|s| s.asset_id == intern("s_stage_opt0_box"))
        .unwrap();
    assert!(box0.tint[3] > 0.0, "occupied slot box is opaque");
}

// Continue resumes play from a written auto-save (the real-save path, distinct
// from the fresh-start fallback).
#[test]
fn continue_resumes_from_a_written_auto_save() {
    let dir = tempfile::tempdir().unwrap();
    let mut story = two_page_story();
    story.save_key = "cont".to_string();
    let mut world = story_world(story);
    world.start().unwrap();
    point_saves(&mut world, dir.path());
    world.step();

    // Point the auto-save at page 2.
    write_save(
        &save_file(dir.path()),
        &StorySave {
            slug: "a".to_string(),
            page: 1,
            vars: BTreeMap::new(),
        },
    )
    .unwrap();
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Continue);
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "Second page.");
}

// A save naming an unknown slug starts fresh from the top instead of resuming.
#[test]
fn continue_from_an_unknown_slug_starts_fresh() {
    let dir = tempfile::tempdir().unwrap();
    let mut story = two_page_story();
    story.save_key = "stale".to_string();
    let mut world = story_world(story);
    world.start().unwrap();
    point_saves(&mut world, dir.path());
    world.step();

    write_save(
        &save_file(dir.path()),
        &StorySave {
            slug: "ghost".to_string(),
            page: 5,
            vars: BTreeMap::new(),
        },
    )
    .unwrap();
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Continue);
    world.step();
    assert_eq!(label_content(&world, "s_stage_text"), "First page.");
}

// A save pointing at a node with no pages falls back to a fresh start.
#[test]
fn continue_into_a_pageless_node_starts_fresh() {
    let dir = tempfile::tempdir().unwrap();
    let story = Story {
        title: "T".to_string(),
        text_speed: 0.0,
        save_key: "pageless".to_string(),
        nodes: vec![
            StoryNode {
                slug: "a".to_string(),
                pages: Vec::new(),
                choices: vec![StoryChoice {
                    label: "Go".to_string(),
                    target: 1,
                    condition: None,
                }],
                ..Default::default()
            },
            StoryNode {
                slug: "b".to_string(),
                pages: vec![page("Picked.")],
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    let mut world = story_world(story);
    world.start().unwrap();
    point_saves(&mut world, dir.path());
    world.step();

    write_save(
        &save_file(dir.path()),
        &StorySave {
            slug: "a".to_string(),
            page: 0,
            vars: BTreeMap::new(),
        },
    )
    .unwrap();
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Continue);
    world.step();
    // The pageless save cannot resume, so play restarts at the same node's
    // menu.
    assert_eq!(label_content(&world, "s_stage_opt0_lbl"), "Go");
}

// Reaching the ending drops the auto-save so the next launch starts fresh.
#[test]
fn reaching_the_ending_clears_the_auto_save() {
    let dir = tempfile::tempdir().unwrap();
    let mut story = two_page_story();
    story.save_key = "endclear".to_string();
    let mut world = story_world(story);
    world.start().unwrap();
    point_saves(&mut world, dir.path());
    world.step();
    assert!(
        read_save(&save_file(dir.path())).is_some(),
        "the first page auto-saved"
    );

    // Advance to page 2, then past it to the ending, which clears the save.
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Advance);
    world.step();
    world
        .events_mut::<StoryCommand>()
        .send(StoryCommand::Advance);
    world.step();
    assert!(
        read_save(&save_file(dir.path())).is_none(),
        "the ending cleared the auto-save"
    );
}
