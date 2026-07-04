// src/story.rs
//
// Story playback: drives a compiled `Story` graph through the stage view its
// build expansion generated. An internal system (not a declarable asset):
// `World::start` constructs one whenever the world contains a `Story`. The
// whole story plays inside one view: this system fills the dialogue and
// name-plate labels (revealing text at the story's speed), swaps the backdrop
// and portrait sprite textures, shows the choice menu when a node ends in
// one, and asks the audio system to play page music and one-shots.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::assets::{
    CueKind, PlayCue, Sprite, Story, StoryCommand, StoryGate, StoryImage, StoryStage, TextLabel,
    ViewCommand, ViewShown,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, StepResult, System};

// The stage scaffolding's menu font size; choice labels center by the same
// glyph-width estimate the build uses (no font metrics at either stage).
const MENU_FONT_PX: f32 = 28.0;
const CHOICE_BUTTON_X: f32 = 280.0;
const CHOICE_BUTTON_W: f32 = 720.0;

// The generated stage assets this system mutates, taken from the story's
// build-resolved scaffold references. An unset optional slot (e.g. a story
// with no choice panel) makes its mutations a no-op.
struct StageIds {
    view: AssetId,
    ending_view: AssetId,
    bg: Option<AssetId>,
    left: Option<AssetId>,
    center: Option<AssetId>,
    right: Option<AssetId>,
    dialog_box: Option<AssetId>,
    name: Option<AssetId>,
    text: Option<AssetId>,
    panel: Option<AssetId>,
    // One label per choice slot. The buttons' hit regions live inside
    // UiInputSystem and stay active the whole time; mode guards here make an
    // out-of-menu choose (or an in-menu advance) a no-op, so the overlap
    // with the full-canvas advance region resolves without touching them.
    options: Vec<AssetId>,
    // The title screen's Continue label, hidden while no save exists.
    continue_label: Option<AssetId>,
}

// A story's persisted position and raised flags, auto-written page by page
// and resumed by `story:continue`. Position is kept by node slug (stable
// across story edits, unlike an index).
#[derive(serde::Serialize, serde::Deserialize)]
struct StorySave {
    slug: String,
    page: u32,
    flags: Vec<String>,
}

fn save_file(dir: &Path, key: &str) -> PathBuf {
    dir.join(format!("story_{}.bin", key))
}

fn read_save(dir: &Path, key: &str) -> Option<StorySave> {
    let bytes = std::fs::read(save_file(dir, key)).ok()?;
    match ciborium::from_reader(&bytes[..]) {
        Ok(save) => Some(save),
        Err(e) => {
            tracing::warn!("StorySystem: save unreadable, starting fresh: {e}");
            None
        }
    }
}

fn write_save(dir: &Path, key: &str, save: &StorySave) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut bytes = Vec::new();
    ciborium::into_writer(save, &mut bytes).map_err(std::io::Error::other)?;
    std::fs::write(save_file(dir, key), bytes)
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
    // Raised story flags; cleared on start, mutated by page and menu ops.
    flags: HashSet<String>,
    // Where the auto-save lives (the project data directory).
    save_dir: PathBuf,
    // The open menu's option indices into the node's choices (conditions
    // filter gated options out); button i picks menu[i].
    menu: Vec<usize>,
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
            flags: HashSet::new(),
            save_dir: concinnity_core::paths::data_dir(),
            menu: Vec::new(),
            typewriter: Typewriter::default(),
            last_step: None,
            active_view: None,
            command_cursor: crate::ecs::EventCursor::default(),
            view_shown_cursor: crate::ecs::EventCursor::default(),
        }
    }

    fn start(&mut self, ctx: &mut PipelineContext) {
        self.started = true;
        self.flags.clear();
        self.exit_choice_ui(ctx);
        let view = self.ids.as_ref().expect("resolved at init").view;
        if self.active_view != Some(view) {
            ctx.events_mut::<ViewCommand>()
                .send(ViewCommand::Show(view));
        }
        self.enter_node(0, ctx);
    }

    // Resume from the saved position; a missing, stale, or unreadable save
    // starts fresh.
    fn continue_story(&mut self, ctx: &mut PipelineContext) {
        let save = if self.story.save_key.is_empty() {
            None
        } else {
            read_save(&self.save_dir, &self.story.save_key)
        };
        let Some(save) = save else {
            self.start(ctx);
            return;
        };
        let Some(node) = self.story.nodes.iter().position(|n| n.slug == save.slug) else {
            self.start(ctx);
            return;
        };
        if self.story.nodes[node].pages.is_empty() {
            self.start(ctx);
            return;
        }
        self.started = true;
        self.flags = save.flags.into_iter().collect();
        self.exit_choice_ui(ctx);
        let view = self.ids.as_ref().expect("resolved at init").view;
        if self.active_view != Some(view) {
            ctx.events_mut::<ViewCommand>()
                .send(ViewCommand::Show(view));
        }
        self.node = node;
        self.page = (save.page as usize).min(self.story.nodes[node].pages.len() - 1);
        self.apply_page(ctx);
    }

    // Auto-save the current position and flags; the title screen's Continue
    // lights up once one exists.
    fn persist_position(&mut self, ctx: &mut PipelineContext) {
        if self.story.save_key.is_empty() {
            return;
        }
        let mut flags: Vec<String> = self.flags.iter().cloned().collect();
        flags.sort();
        let save = StorySave {
            slug: self.story.nodes[self.node].slug.clone(),
            page: self.page as u32,
            flags,
        };
        if let Err(e) = write_save(&self.save_dir, &self.story.save_key, &save) {
            tracing::warn!("StorySystem: save failed: {e}");
            return;
        }
        let continue_label = self.ids.as_ref().and_then(|i| i.continue_label);
        set_label(ctx, continue_label, |l| l.content = "Continue".to_string());
    }

    // A finished story starts fresh next time: drop the save and dim
    // Continue.
    fn clear_save(&mut self, ctx: &mut PipelineContext) {
        if self.story.save_key.is_empty() {
            return;
        }
        let _ = std::fs::remove_file(save_file(&self.save_dir, &self.story.save_key));
        let continue_label = self.ids.as_ref().and_then(|i| i.continue_label);
        set_label(ctx, continue_label, |l| l.content.clear());
    }

    fn flag_passes(&self, flag: &str, negate: bool) -> bool {
        self.flags.contains(flag) != negate
    }

    // The first gate whose condition passes, if any: its target node.
    fn passing_gate(&self, gates: &[StoryGate]) -> Option<usize> {
        gates
            .iter()
            .find(|g| self.flag_passes(&g.flag, g.negate))
            .map(|g| g.target as usize)
    }

    // The node's choice indices whose conditions pass right now.
    fn visible_choices(&self, node: usize) -> Vec<usize> {
        self.story.nodes[node]
            .choices
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                c.condition
                    .as_ref()
                    .is_none_or(|cond| self.flag_passes(&cond.flag, cond.negate))
            })
            .map(|(i, _)| i)
            .collect()
    }

    // Move play to a node: its first page, or straight to its choice menu
    // when it has no pages. A node with neither falls through in document
    // order; running past the last node ends the story. Gates on the arrived
    // page (or menu) redirect first; the hop budget stops a gate cycle from
    // spinning forever.
    fn enter_node(&mut self, index: usize, ctx: &mut PipelineContext) {
        let mut index = index;
        let mut hops = 0;
        loop {
            hops += 1;
            if hops > 64 {
                tracing::warn!("StorySystem: story gates form a loop; stopping");
                return;
            }
            let Some(node) = self.story.nodes.get(index) else {
                self.show_ending(ctx);
                return;
            };
            if !node.pages.is_empty() {
                if let Some(target) = self.passing_gate(&node.pages[0].gates) {
                    index = target;
                    continue;
                }
                self.node = index;
                self.page = 0;
                self.apply_page(ctx);
                return;
            }
            if !node.choices.is_empty() {
                if let Some(target) = self.passing_gate(&node.choice_gates) {
                    index = target;
                    continue;
                }
                if !self.visible_choices(index).is_empty() {
                    self.node = index;
                    self.enter_choice(ctx);
                    return;
                }
                // Every option is gated off: fall through like a menu-less
                // node.
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
        // Gates on whatever comes next redirect before it shows.
        let next_redirect = if jump.is_none() && more_pages {
            self.passing_gate(&node.pages[self.page + 1].gates)
        } else if jump.is_none() && has_choices {
            self.passing_gate(&node.choice_gates)
        } else {
            None
        };
        if let Some(jump) = jump {
            self.enter_node(jump as usize, ctx);
        } else if let Some(target) = next_redirect {
            self.enter_node(target, ctx);
        } else if more_pages {
            self.page += 1;
            self.apply_page(ctx);
        } else if has_choices && !self.visible_choices(self.node).is_empty() {
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
        let Some(choice) = self
            .menu
            .get(option)
            .and_then(|&i| self.story.nodes[self.node].choices.get(i))
        else {
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
        for op in &page.ops {
            if op.clear {
                self.flags.remove(&op.flag);
            } else {
                self.flags.insert(op.flag.clone());
            }
        }

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
        play(ctx, page.music, CueKind::Music);
        for sound in &page.sounds {
            play(ctx, Some(*sound), CueKind::Sound);
        }
        self.persist_position(ctx);
    }

    // Show the current node's choice menu over its stage dressing; stage
    // clicks are inert until an option is picked. Gated options are left off
    // the menu, and the button slots fill from the visible options in order.
    fn enter_choice(&mut self, ctx: &mut PipelineContext) {
        self.in_choice = true;
        let node = self.story.nodes[self.node].clone();
        for op in &node.choice_ops {
            if op.clear {
                self.flags.remove(&op.flag);
            } else {
                self.flags.insert(op.flag.clone());
            }
        }
        self.menu = self.visible_choices(self.node);
        let menu = self.menu.clone();
        let ids = self.ids.as_ref().expect("resolved at init");

        apply_stage(ctx, ids, &node.choice_stage);
        play(ctx, node.choice_music, CueKind::Music);
        for sound in &node.choice_sounds {
            play(ctx, Some(*sound), CueKind::Sound);
        }

        set_label(ctx, ids.name, |l| l.content.clear());
        set_label(ctx, ids.text, |l| l.content.clear());
        // Hidden stage furniture renders nothing (zero alpha, empty text)
        // rather than relying on `visible`: view re-activation (a pause
        // overlay dismissing back to the stage) force-shows every member.
        set_sprite(ctx, ids.dialog_box, |s| s.tint = [0.0, 0.0, 0.0, 0.0]);
        set_sprite(ctx, ids.panel, |s| {
            s.visible = true;
            s.tint = [0.0, 0.0, 0.0, 0.55];
        });
        for (i, label_id) in ids.options.iter().enumerate() {
            match menu.get(i).map(|&c| &node.choices[c]) {
                Some(choice) => {
                    let text = choice.label.clone();
                    let width = est_text_width(&text);
                    set_label(ctx, Some(*label_id), |l| {
                        l.content = text;
                        l.visible = true;
                        l.x = CHOICE_BUTTON_X + ((CHOICE_BUTTON_W - width) / 2.0).max(0.0);
                    });
                }
                None => set_label(ctx, Some(*label_id), |l| l.content.clear()),
            }
        }
    }

    // Put the stage back into page mode (idempotent).
    fn exit_choice_ui(&mut self, ctx: &mut PipelineContext) {
        self.in_choice = false;
        let ids = self.ids.as_ref().expect("resolved at init");
        set_sprite(ctx, ids.dialog_box, |s| s.tint = [0.0, 0.0, 0.0, 0.55]);
        set_sprite(ctx, ids.panel, |s| s.tint = [0.0, 0.0, 0.0, 0.0]);
        for label_id in &ids.options {
            set_label(ctx, Some(*label_id), |l| l.content.clear());
        }
    }

    fn show_ending(&mut self, ctx: &mut PipelineContext) {
        self.clear_save(ctx);
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
    fn init(&mut self, ctx: &mut PipelineContext) {
        // The scaffold references were resolved to ids at build time, like
        // every other cross-reference, so this works identically for a
        // compiled blob (`cn run`) and the interpreted debug path.
        let scaffold = &self.story.scaffold;
        match (scaffold.view, scaffold.ending) {
            (Some(view), Some(ending_view)) => {
                self.ids = Some(StageIds {
                    view,
                    ending_view,
                    bg: scaffold.bg,
                    left: scaffold.left,
                    center: scaffold.center,
                    right: scaffold.right,
                    dialog_box: scaffold.dialog_box,
                    name: scaffold.name_label,
                    text: scaffold.text_label,
                    panel: scaffold.panel,
                    options: scaffold.options.clone(),
                    continue_label: scaffold.continue_label,
                });
                // Continue lights up only when a resumable save exists. View
                // activation force-shows every member label, so presence is
                // carried by the content (an empty label renders nothing).
                let has_save = !self.story.save_key.is_empty()
                    && read_save(&self.save_dir, &self.story.save_key).is_some();
                set_label(ctx, scaffold.continue_label, |l| {
                    if !has_save {
                        l.content.clear();
                    }
                });
            }
            _ => {
                tracing::warn!(
                    "StorySystem: story '{}' has no stage scaffold; story input is ignored",
                    self.story.title,
                );
            }
        }
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
                StoryCommand::Continue => self.continue_story(ctx),
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

// Mutate the first component with the given asset id; an unset reference or
// a missing component (a stage asset the world never declared) is a silent
// no-op.
fn set_label(ctx: &mut PipelineContext, id: Option<AssetId>, apply: impl FnOnce(&mut TextLabel)) {
    let Some(id) = id else { return };
    if let Some(label) = ctx.query_mut::<TextLabel>().find(|l| l.asset_id == id) {
        apply(label);
    }
}

fn set_sprite(ctx: &mut PipelineContext, id: Option<AssetId>, apply: impl FnOnce(&mut Sprite)) {
    let Some(id) = id else { return };
    if let Some(sprite) = ctx.query_mut::<Sprite>().find(|s| s.asset_id == id) {
        apply(sprite);
    }
}

// Apply a page's stage dressing: the backdrop keeps its dark fill when the
// page has no image; portraits hide when their slot is empty.
fn apply_stage(ctx: &mut PipelineContext, ids: &StageIds, stage: &StoryStage) {
    match &stage.bg {
        Some(image) => {
            let texture = image.texture;
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

fn apply_portrait(ctx: &mut PipelineContext, slot: Option<AssetId>, image: Option<&StoryImage>) {
    match image {
        Some(image) => {
            let (texture, x, y, w, h) =
                (image.texture, image.x, image.y, image.width, image.height);
            set_sprite(ctx, slot, |s| {
                s.visible = true;
                s.tint = [1.0, 1.0, 1.0, 1.0];
                s.texture = Some(texture);
                s.x = x;
                s.y = y;
                s.width = w;
                s.height = h;
            });
        }
        // An empty slot goes fully transparent rather than invisible: view
        // re-activation force-shows member sprites.
        None => set_sprite(ctx, slot, |s| s.tint = [1.0, 1.0, 1.0, 0.0]),
    }
}

fn play(ctx: &mut PipelineContext, clip: Option<AssetId>, kind: CueKind) {
    let Some(clip) = clip else { return };
    ctx.events_mut::<PlayCue>().send(PlayCue {
        clip,
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
    use crate::assets::{
        Story, StoryChoice, StoryNode, StoryPage, StoryScaffold, StorySpeaker, View,
    };
    use crate::ecs::World;
    use crate::ecs::asset_id::intern;

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
            panel: Some(intern("s_stage_panel")),
            options: vec![intern("s_stage_opt0_lbl")],
            continue_label: None,
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
        // Hidden furniture goes transparent (view re-activation force-shows
        // members, so `visible` cannot carry menu state).
        let alpha = world
            .query::<Sprite>()
            .find(|s| s.asset_id == panel)
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
                texture: intern("s_img0"),
                x: 0.0,
                y: 0.0,
                width: 1280.0,
                height: 720.0,
            }),
            center: Some(StoryImage {
                texture: intern("s_img1"),
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
        assert_eq!(sprite.tint[3], 0.0);
        let sprite = world.query::<Sprite>().find(|s| s.asset_id == bg).unwrap();
        assert_eq!(sprite.texture, None);
    }

    // Page audio is requested through PlayCue events the audio system plays.
    #[test]
    fn page_audio_sends_play_cues() {
        let mut story = two_page_story();
        story.nodes[0].pages[0].music = Some(intern("s_clip0"));
        story.nodes[0].pages[0].sounds = vec![intern("s_clip1")];
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
                                flag: "asked".to_string(),
                                clear: false,
                            }],
                            ..page("Intro.")
                        },
                        StoryPage {
                            gates: vec![StoryGate {
                                flag: "asked".to_string(),
                                negate: false,
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
                                flag: "secret".to_string(),
                                negate: false,
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
                            flag: "skip".to_string(),
                            clear: false,
                        }],
                        ..page("Once.")
                    }],
                    choices: vec![StoryChoice {
                        label: "Never shown".to_string(),
                        target: 0,
                        condition: None,
                    }],
                    choice_gates: vec![StoryGate {
                        flag: "skip".to_string(),
                        negate: false,
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

    // The save file round-trips position and flags; a missing file reads as
    // None and an unreadable one starts fresh.
    #[test]
    fn story_save_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read_save(dir.path(), "q").is_none());
        let save = StorySave {
            slug: "meadow".to_string(),
            page: 2,
            flags: vec!["asked".to_string(), "brave".to_string()],
        };
        write_save(dir.path(), "q", &save).unwrap();
        let back = read_save(dir.path(), "q").expect("save exists");
        assert_eq!(back.slug, "meadow");
        assert_eq!(back.page, 2);
        assert_eq!(back.flags, ["asked", "brave"]);
        // Corruption falls back to a fresh start rather than a panic.
        std::fs::write(save_file(dir.path(), "q"), b"not cbor").unwrap();
        assert!(read_save(dir.path(), "q").is_none());
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
}
