// src/story.rs
//
// Story playback: drives a compiled `Story` graph through the stage view its
// build expansion generated. An internal system (not a declarable asset):
// `World::start` constructs one whenever the world contains a `Story`. The
// whole story plays inside one view: this system fills the dialogue and
// name-plate labels (revealing text at the story's speed), swaps the backdrop
// and portrait sprite textures, shows the choice menu when a node ends in
// one, and asks the audio system to play page music and one-shots.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::assets::{
    CmpOp, CueKind, PlayCue, Sprite, Story, StoryCommand, StoryGate, StoryImage, StoryOp,
    StoryReload, StoryScaffold, StoryStage, TextLabel, ViewCommand, ViewShown,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, StepResult, System};

// The stage scaffolding's menu font size; choice labels center by the same
// glyph-width estimate the build uses (no font metrics at either stage).
const MENU_FONT_PX: f32 = 28.0;
const CHOICE_BUTTON_X: f32 = 280.0;
const CHOICE_BUTTON_W: f32 = 720.0;
// Shown tint of a choice option's box; matches the color the build authors
// (with zero alpha) on the generated `_opt<N>_box` sprites. An occupied menu
// slot re-tints its box to this; hiding is alpha zero, never `visible` (view
// re-activation force-shows every member).
const CHOICE_BOX_TINT: [f32; 4] = [0.16, 0.20, 0.35, 0.92];
// Quick-row toggle colors: an engaged mode reads gold, the rest gray.
const QUICK_ACTIVE: [f32; 3] = [1.0, 0.85, 0.3];
const QUICK_IDLE: [f32; 3] = [0.75, 0.75, 0.75];
// Shown alpha of the overlay dim behind the backlog and slot rows.
const OVERLAY_DIM_ALPHA: f32 = 0.85;
// Auto mode turns a fully revealed page after a base pause plus reading
// time per character; skip mode turns instantly revealed pages at a fixed
// rapid cadence.
const AUTO_BASE_SECS: f32 = 0.8;
const AUTO_PER_CHAR_SECS: f32 = 0.035;
const SKIP_PAGE_SECS: f32 = 0.15;
// Dialogue-history entries kept for the backlog overlay, and the most
// history lines its label shows at once.
const BACKLOG_ENTRIES: usize = 100;
const BACKLOG_LINES: usize = 20;

// The generated stage assets this system mutates, taken from the story's
// build-resolved scaffold references. An unset optional slot (e.g. a story
// with no dialog box) makes its mutations a no-op.
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
    // One box sprite per choice slot, shown behind its label.
    option_boxes: Vec<AssetId>,
    // One label per choice slot. The buttons' hit regions live inside
    // UiInputSystem and stay active the whole time; mode guards here make an
    // out-of-menu choose (or an in-menu advance) a no-op, so the overlap
    // with the full-canvas advance region resolves without touching them.
    options: Vec<AssetId>,
    // The title screen's Continue label, hidden while no save exists.
    continue_label: Option<AssetId>,
    // The title screen view (returned to when a load overlay opened from the
    // title is dismissed) and its Load label, hidden while no slot exists.
    title_view: Option<AssetId>,
    load_label: Option<AssetId>,
    // The pulsing waiting-for-input marker.
    marker: Option<AssetId>,
    // Quick-row control labels (Log / Auto / Skip / Save).
    log_label: Option<AssetId>,
    auto_label: Option<AssetId>,
    skip_label: Option<AssetId>,
    save_label: Option<AssetId>,
    // Overlay furniture: the shared dim, the backlog history text, and the
    // slot rows.
    overlay_dim: Option<AssetId>,
    backlog_label: Option<AssetId>,
    slot_title: Option<AssetId>,
    slot_boxes: Vec<AssetId>,
    slot_labels: Vec<AssetId>,
}

impl StageIds {
    // The stage references out of a build-resolved scaffold; `None` when the
    // scaffold is missing its stage or ending view (a story with no stage).
    fn from_scaffold(scaffold: &StoryScaffold) -> Option<Self> {
        Some(Self {
            view: scaffold.view?,
            ending_view: scaffold.ending?,
            bg: scaffold.bg,
            left: scaffold.left,
            center: scaffold.center,
            right: scaffold.right,
            dialog_box: scaffold.dialog_box,
            name: scaffold.name_label,
            text: scaffold.text_label,
            option_boxes: scaffold.option_boxes.clone(),
            options: scaffold.options.clone(),
            continue_label: scaffold.continue_label,
            title_view: scaffold.title,
            load_label: scaffold.load_label,
            marker: scaffold.advance_marker,
            log_label: scaffold.log_label,
            auto_label: scaffold.auto_label,
            skip_label: scaffold.skip_label,
            save_label: scaffold.save_label,
            overlay_dim: scaffold.overlay_dim,
            backlog_label: scaffold.backlog_label,
            slot_title: scaffold.slot_title,
            slot_boxes: scaffold.slot_boxes.clone(),
            slot_labels: scaffold.slot_labels.clone(),
        })
    }
}

// A story's persisted position and variables, auto-written page by page
// (resumed by `story:continue`) and written to numbered slots by the slot
// overlay. Position is kept by node slug (stable across story edits, unlike
// an index).
#[derive(serde::Serialize, serde::Deserialize)]
struct StorySave {
    slug: String,
    page: u32,
    #[serde(default)]
    vars: BTreeMap<String, i32>,
}

fn save_file(dir: &Path, key: &str) -> PathBuf {
    dir.join(format!("story_{}.bin", key))
}

fn slot_file(dir: &Path, key: &str, slot: usize) -> PathBuf {
    dir.join(format!("story_{}_slot{}.bin", key, slot))
}

fn read_save(path: &Path) -> Option<StorySave> {
    let bytes = std::fs::read(path).ok()?;
    match ciborium::from_reader(&bytes[..]) {
        Ok(save) => Some(save),
        Err(e) => {
            tracing::warn!("StorySystem: save unreadable, starting fresh: {e}");
            None
        }
    }
}

fn write_save(path: &Path, save: &StorySave) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut bytes = Vec::new();
    ciborium::into_writer(save, &mut bytes).map_err(std::io::Error::other)?;
    std::fs::write(path, bytes)
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

// A modal overlay drawn over the stage: the dialogue backlog or the save /
// load slot rows. Stage input is inert while one is up; any advance click
// dismisses it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Overlay {
    None,
    Backlog,
    SaveMenu,
    LoadMenu,
}

pub struct StorySystem {
    story: Story,
    ids: Option<StageIds>,
    // Current position in the graph.
    node: usize,
    page: usize,
    started: bool,
    in_choice: bool,
    // Story variables (a flag is a variable holding 1); reset on start,
    // mutated by page and menu ops. An unset variable reads as 0.
    vars: HashMap<String, i32>,
    // Where the saves live (the project data directory).
    save_dir: PathBuf,
    // The open menu's option indices into the node's choices (conditions
    // filter gated options out); button i picks menu[i].
    menu: Vec<usize>,
    typewriter: Typewriter,
    last_step: Option<Instant>,
    // Wall-clock seconds since construction, driving the marker pulse.
    elapsed: f32,
    // Reader-assist modes: auto turns fully revealed pages after a reading
    // pause; skip reveals instantly and turns pages rapidly until a menu.
    auto: bool,
    skip: bool,
    // Seconds since the current page finished revealing (auto) or turned
    // (skip).
    mode_timer: f32,
    overlay: Overlay,
    // Recent dialogue for the backlog overlay, one pre-wrapped entry per
    // page shown.
    history: Vec<String>,
    // The active view as announced by UiInputSystem; stage input (advance,
    // choose) is ignored while a menu or the title screen is up.
    active_view: Option<AssetId>,
    command_cursor: crate::ecs::EventCursor,
    view_shown_cursor: crate::ecs::EventCursor,
    reload_cursor: crate::ecs::EventCursor,
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
            vars: HashMap::new(),
            save_dir: concinnity_core::paths::data_dir(),
            menu: Vec::new(),
            typewriter: Typewriter::default(),
            last_step: None,
            elapsed: 0.0,
            auto: false,
            skip: false,
            mode_timer: 0.0,
            overlay: Overlay::None,
            history: Vec::new(),
            active_view: None,
            command_cursor: crate::ecs::EventCursor::default(),
            view_shown_cursor: crate::ecs::EventCursor::default(),
            reload_cursor: crate::ecs::EventCursor::default(),
        }
    }

    fn start(&mut self, ctx: &mut PipelineContext) {
        self.started = true;
        self.vars.clear();
        self.history.clear();
        self.auto = false;
        self.skip = false;
        self.overlay = Overlay::None;
        self.exit_choice_ui(ctx);
        let view = self.ids.as_ref().expect("resolved at init").view;
        if self.active_view != Some(view) {
            ctx.events_mut::<ViewCommand>()
                .send(ViewCommand::Show(view));
        }
        self.enter_node(0, ctx);
    }

    // Resume from the auto-saved position; a missing, stale, or unreadable
    // save starts fresh.
    fn continue_story(&mut self, ctx: &mut PipelineContext) {
        let save = if self.story.save_key.is_empty() {
            None
        } else {
            read_save(&save_file(&self.save_dir, &self.story.save_key))
        };
        match save {
            Some(save) => self.resume_from(save, ctx),
            None => self.start(ctx),
        }
    }

    // Put play at a saved position; an unknown slug or an empty node starts
    // fresh instead.
    fn resume_from(&mut self, save: StorySave, ctx: &mut PipelineContext) {
        let Some(node) = self.story.nodes.iter().position(|n| n.slug == save.slug) else {
            self.start(ctx);
            return;
        };
        if self.story.nodes[node].pages.is_empty() {
            self.start(ctx);
            return;
        }
        self.started = true;
        self.vars = save.vars.into_iter().collect();
        self.history.clear();
        self.overlay = Overlay::None;
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

    // The current position and variables as a save record.
    fn current_save(&self) -> StorySave {
        StorySave {
            slug: self.story.nodes[self.node].slug.clone(),
            page: self.page as u32,
            vars: self.vars.iter().map(|(k, v)| (k.clone(), *v)).collect(),
        }
    }

    // Auto-save the current position and variables; the title screen's
    // Continue lights up once one exists.
    fn persist_position(&mut self, ctx: &mut PipelineContext) {
        if self.story.save_key.is_empty() {
            return;
        }
        let save = self.current_save();
        if let Err(e) = write_save(&save_file(&self.save_dir, &self.story.save_key), &save) {
            tracing::warn!("StorySystem: save failed: {e}");
            return;
        }
        let continue_label = self.ids.as_ref().and_then(|i| i.continue_label);
        set_label(ctx, continue_label, |l| l.content = "Continue".to_string());
    }

    // A finished story starts fresh next time: drop the auto-save and dim
    // Continue. Slot saves stay.
    fn clear_save(&mut self, ctx: &mut PipelineContext) {
        if self.story.save_key.is_empty() {
            return;
        }
        let _ = std::fs::remove_file(save_file(&self.save_dir, &self.story.save_key));
        let continue_label = self.ids.as_ref().and_then(|i| i.continue_label);
        set_label(ctx, continue_label, |l| l.content.clear());
    }

    // Whether any manual slot save exists (the title's Load lights up).
    fn any_slot_save(&self) -> bool {
        !self.story.save_key.is_empty()
            && (0..self.ids.as_ref().map_or(0, |i| i.slot_boxes.len()))
                .any(|i| slot_file(&self.save_dir, &self.story.save_key, i).exists())
    }

    // Apply a run of variable operations.
    fn apply_ops(&mut self, ops: &[StoryOp]) {
        for op in ops {
            let slot = self.vars.entry(op.name.clone()).or_insert(0);
            if op.add {
                *slot = slot.saturating_add(op.value);
            } else {
                *slot = op.value;
            }
        }
    }

    fn cond_passes(&self, name: &str, op: CmpOp, value: i32) -> bool {
        op.eval(self.vars.get(name).copied().unwrap_or(0), value)
    }

    // The first gate whose condition passes, if any: its target node.
    fn passing_gate(&self, gates: &[StoryGate]) -> Option<usize> {
        gates
            .iter()
            .find(|g| self.cond_passes(&g.name, g.op, g.value))
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
                    .is_none_or(|cond| self.cond_passes(&cond.name, cond.op, cond.value))
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

    // Arrive on the current page: run its variable ops, fill the stage, fire
    // its one-shots, log it in the backlog, and auto-save.
    fn apply_page(&mut self, ctx: &mut PipelineContext) {
        let page = self.story.nodes[self.node].pages[self.page].clone();
        self.apply_ops(&page.ops);
        self.render_page(ctx);
        let entry = match &page.speaker {
            Some(s) => format!("{}: {}", s.name, page.text),
            None => page.text.clone(),
        };
        self.history.push(entry);
        if self.history.len() > BACKLOG_ENTRIES {
            self.history.remove(0);
        }
        for sound in &page.sounds {
            play(ctx, Some(*sound), CueKind::Sound);
        }
        self.persist_position(ctx);
    }

    // Fill the stage for the current page: name plate, dialogue reveal,
    // backdrop and portraits, page music. No arrival side effects (flag ops,
    // one-shots, the auto-save), so a hot-reload can re-render in place;
    // re-playing the page's music is a same-key no-op.
    fn render_page(&mut self, ctx: &mut PipelineContext) {
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
        play(ctx, page.music, CueKind::Music);

        // A skip run reveals instantly; otherwise the reveal restarts the
        // auto/skip pacing clock.
        if self.skip {
            self.typewriter.shown = self.typewriter.full.len();
            let text = self.typewriter.text();
            let text_id = self.ids.as_ref().expect("resolved at init").text;
            set_label(ctx, text_id, |l| l.content = text);
        }
        self.mode_timer = 0.0;
        self.render_quick_row(ctx);
    }

    // Fill (or clear) the quick-row control labels; engaged modes read gold.
    fn render_quick_row(&mut self, ctx: &mut PipelineContext) {
        let Some(ids) = self.ids.as_ref() else { return };
        let rows = [
            (ids.log_label, "Log", false),
            (ids.auto_label, "Auto", self.auto),
            (ids.skip_label, "Skip", self.skip),
            (ids.save_label, "Save", false),
        ];
        for (id, text, active) in rows {
            set_label(ctx, id, |l| {
                l.content = text.to_string();
                l.color = if active { QUICK_ACTIVE } else { QUICK_IDLE };
            });
        }
    }

    fn clear_quick_row(&mut self, ctx: &mut PipelineContext) {
        let Some(ids) = self.ids.as_ref() else { return };
        for id in [
            ids.log_label,
            ids.auto_label,
            ids.skip_label,
            ids.save_label,
        ] {
            set_label(ctx, id, |l| l.content.clear());
        }
        set_sprite(ctx, ids.marker, |s| s.tint[3] = 0.0);
    }

    // Arrive on the current node's choice menu: run its variable ops, show
    // the menu, and fire its one-shots. A skip run stops at a menu.
    fn enter_choice(&mut self, ctx: &mut PipelineContext) {
        let node = self.story.nodes[self.node].clone();
        self.apply_ops(&node.choice_ops);
        self.skip = false;
        self.render_choice(ctx);
        for sound in &node.choice_sounds {
            play(ctx, Some(*sound), CueKind::Sound);
        }
    }

    // Show the current node's choice menu over its stage dressing; stage
    // clicks are inert until an option is picked. Gated options are left off
    // the menu, and the button slots fill from the visible options in order.
    // No arrival side effects, so a hot-reload can re-render an open menu.
    fn render_choice(&mut self, ctx: &mut PipelineContext) {
        self.in_choice = true;
        let node = self.story.nodes[self.node].clone();
        self.menu = self.visible_choices(self.node);
        let menu = self.menu.clone();
        let ids = self.ids.as_ref().expect("resolved at init");

        apply_stage(ctx, ids, &node.choice_stage);
        play(ctx, node.choice_music, CueKind::Music);

        set_label(ctx, ids.name, |l| l.content.clear());
        set_label(ctx, ids.text, |l| l.content.clear());
        // Hidden stage furniture renders nothing (zero alpha, empty text)
        // rather than relying on `visible`: view re-activation (a pause
        // overlay dismissing back to the stage) force-shows every member.
        set_sprite(ctx, ids.dialog_box, |s| s.tint = [0.0, 0.0, 0.0, 0.0]);
        self.clear_quick_row(ctx);
        let ids = self.ids.as_ref().expect("resolved at init");
        let boxes = ids.option_boxes.clone();
        for (i, label_id) in ids.options.iter().enumerate() {
            let occupied = menu.get(i).is_some();
            set_sprite(ctx, boxes.get(i).copied(), |s| {
                s.visible = true;
                s.tint = if occupied {
                    CHOICE_BOX_TINT
                } else {
                    [
                        CHOICE_BOX_TINT[0],
                        CHOICE_BOX_TINT[1],
                        CHOICE_BOX_TINT[2],
                        0.0,
                    ]
                };
            });
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
        let boxes = ids.option_boxes.clone();
        for box_id in boxes {
            set_sprite(ctx, Some(box_id), |s| s.tint[3] = 0.0);
        }
        for label_id in &ids.options {
            set_label(ctx, Some(*label_id), |l| l.content.clear());
        }
    }

    // Swap in a freshly compiled graph (the editor re-expands the story when
    // its Markdown source is saved), keeping the current position and raised
    // flags so the edit lands in place in the running game. Position is
    // matched by node slug; a deleted node restarts the story.
    fn reload(&mut self, new: Story, ctx: &mut PipelineContext) {
        // A multi-story world reloads every story; only ours applies.
        if new.scaffold.view != self.story.scaffold.view {
            return;
        }
        // The re-rendered page would draw over an open overlay's dim.
        if self.overlay != Overlay::None {
            self.close_overlay(ctx);
        }
        let slug = self.story.nodes.get(self.node).map(|n| n.slug.clone());
        let was_in_choice = self.in_choice;
        self.story = new;
        // Refresh the stage references: the scaffold's option-slot count can
        // change with the story's widest menu (slots the running world never
        // declared are silent no-ops until a restart).
        if let Some(ids) = StageIds::from_scaffold(&self.story.scaffold) {
            self.ids = Some(ids);
        }
        if !self.started {
            return;
        }
        let node = slug.and_then(|s| self.story.nodes.iter().position(|n| n.slug == s));
        let Some(node) = node else {
            self.start(ctx);
            return;
        };
        self.node = node;
        let has_choices = !self.story.nodes[node].choices.is_empty();
        let page_count = self.story.nodes[node].pages.len();
        if was_in_choice && has_choices && !self.visible_choices(node).is_empty() {
            self.render_choice(ctx);
        } else if page_count > 0 {
            self.exit_choice_ui(ctx);
            self.page = self.page.min(page_count - 1);
            self.render_page(ctx);
            // Editing flow: show the whole revised page at once rather than
            // re-typing it out.
            self.typewriter.shown = self.typewriter.full.len();
            let text = self.typewriter.text();
            let text_id = self.ids.as_ref().expect("resolved at init").text;
            set_label(ctx, text_id, |l| l.content = text);
        } else {
            // The node lost its pages (and any open menu no longer applies):
            // re-enter it fresh so gates and fall-through resolve.
            self.exit_choice_ui(ctx);
            self.enter_node(node, ctx);
            return;
        }
        self.persist_position(ctx);
    }

    fn show_ending(&mut self, ctx: &mut PipelineContext) {
        self.clear_save(ctx);
        self.auto = false;
        self.skip = false;
        let ids = self.ids.as_ref().expect("resolved at init");
        ctx.events_mut::<ViewCommand>()
            .send(ViewCommand::Show(ids.ending_view));
    }

    // Ordinary page mode: mid-story, no menu, no overlay, stage in front.
    fn page_mode(&self) -> bool {
        self.started
            && !self.in_choice
            && self.overlay == Overlay::None
            && self.active_view == self.ids.as_ref().map(|i| i.view)
    }

    fn toggle_auto(&mut self, ctx: &mut PipelineContext) {
        if !self.page_mode() {
            return;
        }
        self.auto = !self.auto;
        self.mode_timer = 0.0;
        self.render_quick_row(ctx);
    }

    fn toggle_skip(&mut self, ctx: &mut PipelineContext) {
        if !self.page_mode() {
            return;
        }
        self.skip = !self.skip;
        self.mode_timer = 0.0;
        if self.skip && !self.typewriter.done() {
            self.typewriter.shown = self.typewriter.full.len();
            let text = self.typewriter.text();
            let text_id = self.ids.as_ref().expect("resolved at init").text;
            set_label(ctx, text_id, |l| l.content = text);
        }
        self.render_quick_row(ctx);
    }

    fn toggle_log(&mut self, ctx: &mut PipelineContext) {
        match self.overlay {
            Overlay::Backlog => self.close_overlay(ctx),
            Overlay::None if self.page_mode() => self.open_backlog(ctx),
            _ => {}
        }
    }

    // Clear the page furniture that draws over the overlay dim (labels
    // always render above sprites, so anything left filled would float on
    // top of it).
    fn hide_page_furniture(&mut self, ctx: &mut PipelineContext) {
        let (name, text) = {
            let ids = self.ids.as_ref().expect("resolved at init");
            (ids.name, ids.text)
        };
        set_label(ctx, name, |l| l.content.clear());
        set_label(ctx, text, |l| l.content.clear());
        self.clear_quick_row(ctx);
    }

    fn open_backlog(&mut self, ctx: &mut PipelineContext) {
        self.overlay = Overlay::Backlog;
        self.hide_page_furniture(ctx);
        // The most recent entries that fit the label, oldest first.
        let mut chosen: Vec<&str> = Vec::new();
        let mut lines = 0;
        for entry in self.history.iter().rev() {
            let n = entry.lines().count().max(1);
            if !chosen.is_empty() && lines + n > BACKLOG_LINES {
                break;
            }
            chosen.push(entry.as_str());
            lines += n;
            if lines >= BACKLOG_LINES {
                break;
            }
        }
        chosen.reverse();
        let text = chosen.join("\n");
        let ids = self.ids.as_ref().expect("resolved at init");
        set_sprite(ctx, ids.overlay_dim, |s| s.tint[3] = OVERLAY_DIM_ALPHA);
        set_label(ctx, ids.backlog_label, |l| l.content = text);
    }

    fn open_save(&mut self, ctx: &mut PipelineContext) {
        if self.page_mode() && !self.story.save_key.is_empty() {
            self.open_slots(ctx, false);
        }
    }

    // The title screen's Load: bring the (dimmed) stage up over whatever the
    // title left behind and offer the slots. Also reachable nowhere else;
    // mid-story loading goes through the same slots after a Save.
    fn open_load(&mut self, ctx: &mut PipelineContext) {
        if self.overlay != Overlay::None || self.story.save_key.is_empty() {
            return;
        }
        if self.started && !self.page_mode() {
            return;
        }
        if !self.started {
            // The overlay is set before the view change lands, so the
            // stage's ViewShown does not auto-start the story.
            let view = self.ids.as_ref().expect("resolved at init").view;
            ctx.events_mut::<ViewCommand>()
                .send(ViewCommand::Show(view));
        }
        self.open_slots(ctx, true);
    }

    fn open_slots(&mut self, ctx: &mut PipelineContext, load: bool) {
        self.overlay = if load {
            Overlay::LoadMenu
        } else {
            Overlay::SaveMenu
        };
        if self.started {
            self.hide_page_furniture(ctx);
        }
        let (dim, title, boxes, labels) = {
            let ids = self.ids.as_ref().expect("resolved at init");
            (
                ids.overlay_dim,
                ids.slot_title,
                ids.slot_boxes.clone(),
                ids.slot_labels.clone(),
            )
        };
        set_sprite(ctx, dim, |s| s.tint[3] = OVERLAY_DIM_ALPHA);
        set_label(ctx, title, |l| {
            l.content = (if load { "Load" } else { "Save" }).to_string();
        });
        for (i, box_id) in boxes.iter().enumerate() {
            let summary = self.slot_summary(i);
            set_sprite(ctx, Some(*box_id), |s| {
                s.visible = true;
                s.tint = CHOICE_BOX_TINT;
            });
            set_label(ctx, labels.get(i).copied(), |l| {
                l.content = summary;
                l.visible = true;
            });
        }
    }

    fn slot_summary(&self, slot: usize) -> String {
        match read_save(&slot_file(&self.save_dir, &self.story.save_key, slot)) {
            Some(save) => format!("Slot {}   {}, page {}", slot + 1, save.slug, save.page + 1),
            None => format!("Slot {}   (empty)", slot + 1),
        }
    }

    fn pick_slot(&mut self, slot: usize, ctx: &mut PipelineContext) {
        match self.overlay {
            Overlay::SaveMenu => {
                let save = self.current_save();
                let path = slot_file(&self.save_dir, &self.story.save_key, slot);
                if let Err(e) = write_save(&path, &save) {
                    tracing::warn!("StorySystem: slot save failed: {e}");
                }
                let load_label = self.ids.as_ref().and_then(|i| i.load_label);
                set_label(ctx, load_label, |l| l.content = "Load".to_string());
                self.close_overlay(ctx);
            }
            Overlay::LoadMenu => {
                // Picking an empty slot leaves the overlay up.
                let path = slot_file(&self.save_dir, &self.story.save_key, slot);
                let Some(save) = read_save(&path) else { return };
                self.hide_overlay_furniture(ctx);
                self.overlay = Overlay::None;
                self.resume_from(save, ctx);
            }
            _ => {}
        }
    }

    fn hide_overlay_furniture(&mut self, ctx: &mut PipelineContext) {
        let (dim, backlog, title, boxes, labels) = {
            let ids = self.ids.as_ref().expect("resolved at init");
            (
                ids.overlay_dim,
                ids.backlog_label,
                ids.slot_title,
                ids.slot_boxes.clone(),
                ids.slot_labels.clone(),
            )
        };
        set_sprite(ctx, dim, |s| s.tint[3] = 0.0);
        set_label(ctx, backlog, |l| l.content.clear());
        set_label(ctx, title, |l| l.content.clear());
        for box_id in boxes {
            set_sprite(ctx, Some(box_id), |s| s.tint[3] = 0.0);
        }
        for label_id in labels {
            set_label(ctx, Some(label_id), |l| l.content.clear());
        }
    }

    // Dismiss the open overlay: back to the page it covered, or back to the
    // title screen when the load overlay was opened from there.
    fn close_overlay(&mut self, ctx: &mut PipelineContext) {
        self.hide_overlay_furniture(ctx);
        self.overlay = Overlay::None;
        if !self.started {
            let title = self.ids.as_ref().and_then(|i| i.title_view);
            if let Some(title) = title {
                ctx.events_mut::<ViewCommand>()
                    .send(ViewCommand::Show(title));
            }
            return;
        }
        // Overlays open from page mode only; restore the covered page in
        // full (it was already read, so no re-typing).
        self.render_page(ctx);
        self.typewriter.shown = self.typewriter.full.len();
        let text = self.typewriter.text();
        let text_id = self.ids.as_ref().expect("resolved at init").text;
        set_label(ctx, text_id, |l| l.content = text);
    }

    // Per-frame reader-assist work: the waiting marker pulse and the auto /
    // skip page pacing.
    fn tick_modes(&mut self, ctx: &mut PipelineContext, dt: f32) {
        if !self.page_mode() {
            return;
        }
        let waiting = self.typewriter.done();
        let marker = self.ids.as_ref().and_then(|i| i.marker);
        let alpha = if waiting && !self.skip {
            0.35 + 0.3 * (self.elapsed * 5.0).sin()
        } else {
            0.0
        };
        set_sprite(ctx, marker, |s| s.tint[3] = alpha);
        if self.skip {
            if !self.typewriter.done() {
                self.typewriter.shown = self.typewriter.full.len();
                let text = self.typewriter.text();
                let text_id = self.ids.as_ref().expect("resolved at init").text;
                set_label(ctx, text_id, |l| l.content = text);
            }
            self.mode_timer += dt;
            if self.mode_timer >= SKIP_PAGE_SECS {
                self.mode_timer = 0.0;
                self.advance(ctx);
            }
        } else if self.auto && waiting {
            self.mode_timer += dt;
            let delay = AUTO_BASE_SECS + AUTO_PER_CHAR_SECS * self.typewriter.full.len() as f32;
            if self.mode_timer >= delay {
                self.advance(ctx);
            }
        }
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
        match StageIds::from_scaffold(&self.story.scaffold) {
            Some(ids) => {
                let continue_label = ids.continue_label;
                let load_label = ids.load_label;
                self.ids = Some(ids);
                // Continue / Load light up only when a resumable save exists.
                // View activation force-shows every member label, so presence
                // is carried by the content (an empty label renders nothing).
                let has_save = !self.story.save_key.is_empty()
                    && read_save(&save_file(&self.save_dir, &self.story.save_key)).is_some();
                set_label(ctx, continue_label, |l| {
                    if !has_save {
                        l.content.clear();
                    }
                });
                let has_slots = self.any_slot_save();
                set_label(ctx, load_label, |l| {
                    if !has_slots {
                        l.content.clear();
                    }
                });
            }
            None => {
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
        self.elapsed += dt;

        // Freshly re-compiled graphs from the editor's source hot-reload
        // swap in before any input is handled.
        let reloads: Vec<Story> = match ctx.events::<StoryReload>() {
            Some(events) => events
                .read(&mut self.reload_cursor)
                .into_iter()
                .map(|e| e.story.clone())
                .collect(),
            None => Vec::new(),
        };
        for story in reloads {
            self.reload(story, ctx);
        }

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
            // The load overlay shows the stage without starting play.
            if !self.started
                && self.overlay == Overlay::None
                && Some(view) == self.ids.as_ref().map(|i| i.view)
            {
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
        // One click over a button fires both the full-canvas advance region
        // and the button (every hovered region fires). The mode guards
        // absorb the wrong-mode half, but two same-tick rules remain: a
        // command that flips the page/menu mode must not let the click's
        // other half act in the new mode, and a click on a quick-row /
        // slot button must not also advance (or choose).
        let button_tick = commands.iter().any(|c| {
            matches!(
                c,
                StoryCommand::ToggleAuto
                    | StoryCommand::ToggleSkip
                    | StoryCommand::ToggleLog
                    | StoryCommand::OpenSave
                    | StoryCommand::OpenLoad
                    | StoryCommand::Slot(_)
            )
        });
        let mut mode_flipped = false;
        for command in commands {
            let was_in_choice = self.in_choice;
            match command {
                StoryCommand::Start => self.start(ctx),
                StoryCommand::Continue => self.continue_story(ctx),
                StoryCommand::ToggleAuto => self.toggle_auto(ctx),
                StoryCommand::ToggleSkip => self.toggle_skip(ctx),
                StoryCommand::ToggleLog => self.toggle_log(ctx),
                StoryCommand::OpenSave => self.open_save(ctx),
                StoryCommand::OpenLoad => self.open_load(ctx),
                StoryCommand::Slot(i) => self.pick_slot(i, ctx),
                StoryCommand::Advance if !mode_flipped && !button_tick => {
                    if self.overlay != Overlay::None {
                        // Any plain click dismisses an overlay.
                        self.close_overlay(ctx);
                    } else {
                        // Manual input ends a skip run.
                        if self.skip {
                            self.skip = false;
                            self.render_quick_row(ctx);
                        }
                        self.advance(ctx);
                    }
                }
                StoryCommand::Choose(i) if !mode_flipped && !button_tick => self.choose(i, ctx),
                StoryCommand::Advance | StoryCommand::Choose(_) => {}
            }
            mode_flipped |= self.in_choice != was_in_choice;
        }

        self.tick_typewriter(ctx, dt);
        self.tick_modes(ctx, dt);
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
            option_boxes: vec![intern("s_stage_opt0_box")],
            options: vec![intern("s_stage_opt0_lbl")],
            continue_label: None,
            title: None,
            load_label: None,
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
        let auto = save_file(dir.path(), "q");
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
        let slot = slot_file(dir.path(), "q", 1);
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
        write_save(&slot_file(dir.path(), "q", 0), &save).unwrap();
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
        let saved = read_save(&slot_file(dir.path(), "slot_test", 0)).expect("slot written");
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
}
