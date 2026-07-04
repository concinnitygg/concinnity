// src/story.rs
//
// Story playback: drives a compiled `Story` graph through the stage view its
// build expansion generated. An internal system (not a declarable asset):
// `World::start` constructs one whenever the world contains a `Story`. The
// whole story plays inside one view: this system fills the dialogue and
// name-plate labels (revealing text at the story's speed), swaps the backdrop
// and portrait sprite textures, shows the choice menu when a node ends in
// one, and asks the audio system to play page music and one-shots.

use std::time::Instant;

use crate::assets::{
    CueKind, PlayCue, Sprite, Story, StoryCommand, StoryImage, StoryStage, TextLabel, ViewCommand,
    ViewShown,
};
use crate::ecs::asset_id::{AssetId, intern};
use crate::ecs::{PipelineContext, StepResult, System};

// The stage scaffolding's menu font size; choice labels center by the same
// glyph-width estimate the build uses (no font metrics at either stage).
const MENU_FONT_PX: f32 = 28.0;
const CHOICE_BUTTON_X: f32 = 280.0;
const CHOICE_BUTTON_W: f32 = 720.0;

// The generated stage assets this system mutates, resolved once from the
// story's name prefix (`<name>_stage_*`).
struct StageIds {
    view: AssetId,
    ending_view: AssetId,
    bg: AssetId,
    left: AssetId,
    center: AssetId,
    right: AssetId,
    dialog_box: AssetId,
    name: AssetId,
    text: AssetId,
    panel: AssetId,
    // One label per choice slot. The buttons' hit regions live inside
    // UiInputSystem and stay active the whole time; mode guards here make an
    // out-of-menu choose (or an in-menu advance) a no-op, so the overlap
    // with the full-canvas advance region resolves without touching them.
    options: Vec<AssetId>,
}

// Dialogue reveal state for the current page.
#[derive(Default)]
struct Typewriter {
    full: Vec<char>,
    shown: usize,
    budget: f32,
}

impl Typewriter {
    fn done(&self) -> bool {
        self.shown >= self.full.len()
    }
    fn text(&self) -> String {
        self.full[..self.shown].iter().collect()
    }
}

pub struct StorySystem {
    story: Story,
    ids: Option<StageIds>,
    // Current position in the graph.
    node: usize,
    page: usize,
    started: bool,
    in_choice: bool,
    typewriter: Typewriter,
    last_step: Option<Instant>,
    // The active view as announced by UiInputSystem; stage input (advance,
    // choose) is ignored while a menu or the title screen is up.
    active_view: Option<AssetId>,
    command_cursor: crate::ecs::EventCursor,
    view_shown_cursor: crate::ecs::EventCursor,
}

impl std::fmt::Debug for StorySystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StorySystem")
            .field("nodes", &self.story.nodes.len())
            .field("node", &self.node)
            .field("page", &self.page)
            .field("started", &self.started)
            .finish()
    }
}

impl StorySystem {
    pub fn new(story: Story) -> Self {
        Self {
            story,
            ids: None,
            node: 0,
            page: 0,
            started: false,
            in_choice: false,
            typewriter: Typewriter::default(),
            last_step: None,
            active_view: None,
            command_cursor: crate::ecs::EventCursor::default(),
            view_shown_cursor: crate::ecs::EventCursor::default(),
        }
    }

    fn start(&mut self, ctx: &mut PipelineContext) {
        self.started = true;
        self.exit_choice_ui(ctx);
        let view = self.ids.as_ref().expect("resolved at init").view;
        if self.active_view != Some(view) {
            ctx.events_mut::<ViewCommand>()
                .send(ViewCommand::Show(view));
        }
        self.enter_node(0, ctx);
    }

    // Move play to a node: its first page, or straight to its choice menu
    // when it has no pages. A node with neither falls through in document
    // order; running past the last node ends the story.
    fn enter_node(&mut self, index: usize, ctx: &mut PipelineContext) {
        let mut index = index;
        loop {
            let Some(node) = self.story.nodes.get(index) else {
                self.show_ending(ctx);
                return;
            };
            if !node.pages.is_empty() {
                self.node = index;
                self.page = 0;
                self.apply_page(ctx);
                return;
            }
            if !node.choices.is_empty() {
                self.node = index;
                self.enter_choice(ctx);
                return;
            }
            index += 1;
        }
    }

    fn advance(&mut self, ctx: &mut PipelineContext) {
        let Some((view, text_id)) = self.ids.as_ref().map(|i| (i.view, i.text)) else {
            return;
        };
        if !self.started || self.in_choice || self.active_view != Some(view) {
            return;
        }
        // A click mid-reveal completes the page instead of leaving it.
        if !self.typewriter.done() {
            self.typewriter.shown = self.typewriter.full.len();
            let text = self.typewriter.text();
            set_label(ctx, text_id, |l| l.content = text);
            return;
        }
        let node = &self.story.nodes[self.node];
        let jump = node.pages[self.page].jump;
        let more_pages = self.page + 1 < node.pages.len();
        let has_choices = !node.choices.is_empty();
        if let Some(jump) = jump {
            self.enter_node(jump as usize, ctx);
        } else if more_pages {
            self.page += 1;
            self.apply_page(ctx);
        } else if has_choices {
            self.enter_choice(ctx);
        } else {
            self.enter_node(self.node + 1, ctx);
        }
    }

    fn choose(&mut self, option: usize, ctx: &mut PipelineContext) {
        let Some(view) = self.ids.as_ref().map(|i| i.view) else {
            return;
        };
        if !self.started || !self.in_choice || self.active_view != Some(view) {
            return;
        }
        let Some(choice) = self.story.nodes[self.node].choices.get(option) else {
            return;
        };
        let target = choice.target as usize;
        self.exit_choice_ui(ctx);
        self.enter_node(target, ctx);
    }

    // Fill the stage for the current page: name plate, dialogue reveal,
    // backdrop and portraits, page audio.
    fn apply_page(&mut self, ctx: &mut PipelineContext) {
        let (name_id, text_id) = {
            let ids = self.ids.as_ref().expect("resolved at init");
            (ids.name, ids.text)
        };
        let page = self.story.nodes[self.node].pages[self.page].clone();

        let (speaker, color) = match &page.speaker {
            Some(s) => (s.name.clone(), s.color),
            None => (String::new(), [1.0, 1.0, 1.0]),
        };
        set_label(ctx, name_id, |l| {
            l.content = speaker;
            l.color = color;
        });

        self.typewriter = Typewriter {
            full: page.text.chars().collect(),
            shown: 0,
            budget: 0.0,
        };
        if self.story.text_speed <= 0.0 {
            self.typewriter.shown = self.typewriter.full.len();
        }
        let text = self.typewriter.text();
        set_label(ctx, text_id, |l| l.content = text);

        apply_stage(
            ctx,
            self.ids.as_ref().expect("resolved at init"),
            &page.stage,
        );
        play(ctx, page.music.as_deref(), CueKind::Music);
        for sound in &page.sounds {
            play(ctx, Some(sound), CueKind::Sound);
        }
    }

    // Show the current node's choice menu over its stage dressing; stage
    // clicks are inert until an option is picked.
    fn enter_choice(&mut self, ctx: &mut PipelineContext) {
        self.in_choice = true;
        let node = self.story.nodes[self.node].clone();
        let ids = self.ids.as_ref().expect("resolved at init");

        apply_stage(ctx, ids, &node.choice_stage);
        play(ctx, node.choice_music.as_deref(), CueKind::Music);
        for sound in &node.choice_sounds {
            play(ctx, Some(sound), CueKind::Sound);
        }

        set_label(ctx, ids.name, |l| l.content.clear());
        set_label(ctx, ids.text, |l| l.content.clear());
        set_sprite(ctx, ids.dialog_box, |s| s.visible = false);
        set_sprite(ctx, ids.panel, |s| {
            s.visible = true;
            s.tint = [0.0, 0.0, 0.0, 0.55];
        });
        for (i, label_id) in ids.options.iter().enumerate() {
            match node.choices.get(i) {
                Some(choice) => {
                    let text = choice.label.clone();
                    let width = est_text_width(&text);
                    set_label(ctx, *label_id, |l| {
                        l.content = text;
                        l.visible = true;
                        l.x = CHOICE_BUTTON_X + ((CHOICE_BUTTON_W - width) / 2.0).max(0.0);
                    });
                }
                None => set_label(ctx, *label_id, |l| l.visible = false),
            }
        }
    }

    // Put the stage back into page mode (idempotent).
    fn exit_choice_ui(&mut self, ctx: &mut PipelineContext) {
        self.in_choice = false;
        let ids = self.ids.as_ref().expect("resolved at init");
        set_sprite(ctx, ids.dialog_box, |s| s.visible = true);
        set_sprite(ctx, ids.panel, |s| s.visible = false);
        for label_id in &ids.options {
            set_label(ctx, *label_id, |l| l.visible = false);
        }
    }

    fn show_ending(&mut self, ctx: &mut PipelineContext) {
        let ids = self.ids.as_ref().expect("resolved at init");
        ctx.events_mut::<ViewCommand>()
            .send(ViewCommand::Show(ids.ending_view));
    }

    // Reveal more of the current page at the story's characters-per-second.
    fn tick_typewriter(&mut self, ctx: &mut PipelineContext, dt: f32) {
        if self.typewriter.done() || !self.started || self.in_choice {
            return;
        }
        self.typewriter.budget += dt * self.story.text_speed;
        let step = self.typewriter.budget as usize;
        if step == 0 {
            return;
        }
        self.typewriter.budget -= step as f32;
        self.typewriter.shown = (self.typewriter.shown + step).min(self.typewriter.full.len());
        let text = self.typewriter.text();
        let id = self.ids.as_ref().expect("resolved at init").text;
        set_label(ctx, id, |l| l.content = text);
    }
}

impl System for StorySystem {
    fn init(&mut self, _ctx: &mut PipelineContext) {
        // Resolve the generated stage assets by the name convention the build
        // expansion guarantees. Interning is idempotent: these are the same
        // ids the blob loader gave the components.
        let names = crate::ecs::asset_id::name_table();
        let prefix = names
            .get(self.story.asset_id.0 as usize)
            .cloned()
            .unwrap_or_default();
        let stage = format!("{}_stage", prefix);
        let max_choices = self
            .story
            .nodes
            .iter()
            .map(|n| n.choices.len())
            .max()
            .unwrap_or(0);
        self.ids = Some(StageIds {
            view: intern(&stage),
            ending_view: intern(&format!("{}_ending", prefix)),
            bg: intern(&format!("{}_bg", stage)),
            left: intern(&format!("{}_left", stage)),
            center: intern(&format!("{}_center", stage)),
            right: intern(&format!("{}_right", stage)),
            dialog_box: intern(&format!("{}_box", stage)),
            name: intern(&format!("{}_name", stage)),
            text: intern(&format!("{}_text", stage)),
            panel: intern(&format!("{}_panel", stage)),
            options: (0..max_choices)
                .map(|i| intern(&format!("{}_opt{}_lbl", stage, i)))
                .collect(),
        });
        tracing::info!(
            "StorySystem: '{}', {} node(s), text speed {} cps",
            self.story.title,
            self.story.nodes.len(),
            self.story.text_speed,
        );
    }

    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        let now = Instant::now();
        let dt = self
            .last_step
            .map(|t| (now - t).as_secs_f32())
            .unwrap_or(0.0);
        self.last_step = Some(now);

        // Track the active view; auto-start when the stage itself is the
        // world's initial view (no title screen).
        let shown: Vec<AssetId> = match ctx.events::<ViewShown>() {
            Some(events) => events
                .read(&mut self.view_shown_cursor)
                .into_iter()
                .map(|e| e.view)
                .collect(),
            None => Vec::new(),
        };
        for view in shown {
            self.active_view = Some(view);
            if !self.started && Some(view) == self.ids.as_ref().map(|i| i.view) {
                self.start(ctx);
            }
        }

        let commands: Vec<StoryCommand> = match ctx.events::<StoryCommand>() {
            Some(events) => events
                .read(&mut self.command_cursor)
                .into_iter()
                .cloned()
                .collect(),
            None => Vec::new(),
        };
        // One click over a choice button fires both the full-canvas advance
        // region and the button. The mode guards absorb the wrong-mode one,
        // but a command that flips the mode must not let the click's other
        // half act in the new mode the same tick (advancing into a menu must
        // not also pick an option).
        let mut mode_flipped = false;
        for command in commands {
            let was_in_choice = self.in_choice;
            match command {
                StoryCommand::Start => self.start(ctx),
                StoryCommand::Advance if !mode_flipped => self.advance(ctx),
                StoryCommand::Choose(i) if !mode_flipped => self.choose(i, ctx),
                StoryCommand::Advance | StoryCommand::Choose(_) => {}
            }
            mode_flipped |= self.in_choice != was_in_choice;
        }

        self.tick_typewriter(ctx, dt);
        StepResult::Continue
    }
}

// Mutate the first component with the given asset id; a missing id (a stage
// asset the world never declared) is a silent no-op.
fn set_label(ctx: &mut PipelineContext, id: AssetId, apply: impl FnOnce(&mut TextLabel)) {
    if let Some(label) = ctx.query_mut::<TextLabel>().find(|l| l.asset_id == id) {
        apply(label);
    }
}

fn set_sprite(ctx: &mut PipelineContext, id: AssetId, apply: impl FnOnce(&mut Sprite)) {
    if let Some(sprite) = ctx.query_mut::<Sprite>().find(|s| s.asset_id == id) {
        apply(sprite);
    }
}

// Apply a page's stage dressing: the backdrop keeps its dark fill when the
// page has no image; portraits hide when their slot is empty.
fn apply_stage(ctx: &mut PipelineContext, ids: &StageIds, stage: &StoryStage) {
    match &stage.bg {
        Some(image) => {
            let texture = intern(&image.texture);
            set_sprite(ctx, ids.bg, |s| {
                s.texture = Some(texture);
                s.tint = [1.0, 1.0, 1.0, 1.0];
            });
        }
        None => set_sprite(ctx, ids.bg, |s| {
            s.texture = None;
            s.tint = [0.05, 0.06, 0.09, 1.0];
        }),
    }
    for (slot, image) in [
        (ids.left, &stage.left),
        (ids.center, &stage.center),
        (ids.right, &stage.right),
    ] {
        apply_portrait(ctx, slot, image.as_ref());
    }
}

fn apply_portrait(ctx: &mut PipelineContext, slot: AssetId, image: Option<&StoryImage>) {
    match image {
        Some(image) => {
            let texture = intern(&image.texture);
            let (x, y, w, h) = (image.x, image.y, image.width, image.height);
            set_sprite(ctx, slot, |s| {
                s.visible = true;
                s.texture = Some(texture);
                s.x = x;
                s.y = y;
                s.width = w;
                s.height = h;
            });
        }
        None => set_sprite(ctx, slot, |s| s.visible = false),
    }
}

fn play(ctx: &mut PipelineContext, clip: Option<&str>, kind: CueKind) {
    let Some(clip) = clip else { return };
    ctx.events_mut::<PlayCue>().send(PlayCue {
        clip: intern(clip),
        kind,
        volume: 1.0,
    });
}

// The build centers labels by the same estimate (0.6 em per glyph).
fn est_text_width(text: &str) -> f32 {
    text.chars().count() as f32 * 0.6 * MENU_FONT_PX
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{Story, StoryChoice, StoryNode, StoryPage, StorySpeaker, View};
    use crate::ecs::World;

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

    // A world with the stage scaffolding the build expansion would generate
    // for a story named "s", plus the compiled graph itself.
    fn story_world(story: Story) -> World {
        let mut world = World::new_empty();
        let mut story = story;
        story.asset_id = intern("s");
        world.add_component(story);
        for view in ["s_stage", "s_ending"] {
            world.add_component(View {
                asset_id: intern(view),
                initial: view == "s_stage",
                fade_in_secs: 0.0,
            });
        }
        for sprite in [
            "s_stage_bg",
            "s_stage_left",
            "s_stage_center",
            "s_stage_right",
            "s_stage_box",
            "s_stage_panel",
        ] {
            world.add_component(Sprite {
                view: Some(intern("s_stage")),
                ..sprite_named(sprite)
            });
        }
        for label in ["s_stage_name", "s_stage_text", "s_stage_opt0_lbl"] {
            world.add_component(TextLabel {
                view: Some(intern("s_stage")),
                ..label_named(label)
            });
        }
        world
    }

    fn label_content(world: &World, name: &str) -> String {
        let id = intern(name);
        world
            .query::<TextLabel>()
            .find(|l| l.asset_id == id)
            .map(|l| l.content.clone())
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
        let panel = intern("s_stage_panel");
        assert!(
            world
                .query::<Sprite>()
                .find(|s| s.asset_id == panel)
                .unwrap()
                .visible
        );

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
        assert!(
            !world
                .query::<Sprite>()
                .find(|s| s.asset_id == panel)
                .unwrap()
                .visible
        );
    }

    // Page stage dressing applies to the stage sprites: the backdrop samples
    // the page's texture and a portrait slot shows at its compiled rectangle.
    #[test]
    fn stage_dressing_applies_to_sprites() {
        let mut story = two_page_story();
        story.nodes[0].pages[0].stage = StoryStage {
            bg: Some(StoryImage {
                texture: "s_img0".to_string(),
                x: 0.0,
                y: 0.0,
                width: 1280.0,
                height: 720.0,
            }),
            center: Some(StoryImage {
                texture: "s_img1".to_string(),
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
        assert_eq!(sprite.texture, Some(intern("s_img0")));
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
        assert!(!sprite.visible);
        let sprite = world.query::<Sprite>().find(|s| s.asset_id == bg).unwrap();
        assert_eq!(sprite.texture, None);
    }

    // Page audio is requested through PlayCue events the audio system plays.
    #[test]
    fn page_audio_sends_play_cues() {
        let mut story = two_page_story();
        story.nodes[0].pages[0].music = Some("s_clip0".to_string());
        story.nodes[0].pages[0].sounds = vec!["s_clip1".to_string()];
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
        assert_eq!(cues[0].clip, intern("s_clip0"));
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
}
