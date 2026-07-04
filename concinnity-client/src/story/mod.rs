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

mod graph;
mod input;
mod modes;
mod overlay;
mod reload;
mod render;
mod save;
mod title;

#[cfg(test)]
mod tests;

// Horizontal center of the choice / slot buttons (the build authors their
// labels as centered around this x, so the story only fills the text).
const MENU_CENTER_X: f32 = 640.0;
// Title menu layout: the story keeps only the applicable buttons (Start,
// Continue when an auto-save exists, Load when a slot exists, Quit) and stacks
// them contiguously and centered, so an absent button leaves no gap.
const TITLE_MENU_CENTER_Y: f32 = 500.0;
const TITLE_MENU_SPACING: f32 = 62.0;
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
    // The title screen's menu button labels. The story lays the title menu out
    // at runtime, keeping only the applicable buttons contiguous, so Continue
    // and Load appear only when a save exists.
    start_label: Option<AssetId>,
    quit_label: Option<AssetId>,
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
            start_label: scaffold.start_label,
            quit_label: scaffold.quit_label,
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
}

impl System for StorySystem {
    fn init(&mut self, _ctx: &mut PipelineContext) {
        // The scaffold references were resolved to ids at build time, like
        // every other cross-reference, so this works identically for a
        // compiled blob (`cn run`) and the interpreted debug path.
        match StageIds::from_scaffold(&self.story.scaffold) {
            Some(ids) => {
                self.ids = Some(ids);
                // The title menu is laid out on the first `ViewShown` (in
                // `step`), not here. Each menu button's hit region captures its
                // fixed vertical offset to its label in `UiInputSystem::init`,
                // which runs after this one. Relaying the labels now would move
                // them out from under that capture, so every region would bake
                // in the wrong offset and stick at its emitted position instead
                // of tracking the runtime layout. Leaving the emitted positions
                // untouched lets the capture read the authored region-to-label
                // gap; the first title `ViewShown` (announced by that same init)
                // then drives the real layout before the first frame renders.
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
            // Re-lay the title menu each time the title is shown: the save
            // state may have changed while a story played (a fresh auto-save,
            // a new slot, or a finished story clearing its auto-save).
            if Some(view) == self.ids.as_ref().and_then(|i| i.title_view) {
                self.layout_title_menu(ctx);
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
        // other half act in the new mode, and a click on a currently-shown
        // quick-row / slot button must not also advance (or choose).
        //
        // The suppression only counts a button that is actually shown right
        // now: the quick-row and slot regions stay hit-active the whole time
        // and overlap other furniture (the slot rows overlap the choice
        // buttons), so an out-of-mode button command is a no-op that must not
        // eat the click's advance/choose half.
        let button_tick = commands.iter().any(|c| match c {
            StoryCommand::ToggleAuto | StoryCommand::ToggleSkip | StoryCommand::OpenSave => {
                self.page_mode()
            }
            // Log opens from page mode and closes from the backlog; both are a
            // real click on the (shown or last-shown) Log button.
            StoryCommand::ToggleLog => self.page_mode() || self.overlay == Overlay::Backlog,
            // Slot rows are only shown while the slot overlay is up.
            StoryCommand::Slot(_) => {
                matches!(self.overlay, Overlay::SaveMenu | Overlay::LoadMenu)
            }
            // Load lives on the title screen, where no stage advance overlaps.
            _ => false,
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
