// src/story.rs
//
// Story playback: drives a compiled `Story` graph through the stage screen its
// build expansion generated. An internal system (not a declarable asset):
// `World::start` constructs one whenever the world contains a `Story`. The
// whole story plays inside one screen: this system fills the dialogue and
// name-plate labels (revealing text at the story's speed), swaps the backdrop
// and portrait sprite textures, shows the choice menu when a node ends in
// one, and asks the audio system to play page music and one-shots.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::assets::{
    CmpOp, CueKind, FrameInput, Key, PlayCue, ScreenCommand, ScreenShown, Sprite, Story,
    StoryCommand, StoryGate, StoryImage, StoryOp, StoryReload, StoryScaffold, StoryStage,
    TextLabel,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{AudioClipHandle, PipelineContext, StepResult, System};

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
// slot re-tints its box to this; hiding is alpha zero, never `visible` (screen
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
// Manual save slots the slot overlay scrolls through. More slots than the
// overlay shows at once (the build emits a fixed window of rows); the story
// scrolls the window over this many logical slots.
const SLOT_COUNT: usize = 10;
// Accumulated `scroll_delta` that advances the slot window by one row. Mirrors
// the settings list's feel (one wheel notch is one row).
const SLOT_SCROLL_UNIT: f32 = 20.0;

// The generated stage assets this system mutates, taken from the story's
// build-resolved scaffold references. An unset optional slot (e.g. a story
// with no dialog box) makes its mutations a no-op.
struct StageIds {
    screen: AssetId,
    ending_screen: AssetId,
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
    // The title screen screen (returned to when a load overlay opened from the
    // title is dismissed) and its Load label, hidden while no slot exists.
    title_screen_id: Option<AssetId>,
    load_label: Option<AssetId>,
    // The pause menu screen (the injected Escape overlay), the settings-screen
    // entry screen it and the title open, and the title's Settings label. All
    // unset when the world declares no pause menu.
    pause_view: Option<AssetId>,
    settings_screen: Option<AssetId>,
    settings_label: Option<AssetId>,
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
    // scaffold is missing its stage or ending screen (a story with no stage).
    fn from_scaffold(scaffold: &StoryScaffold) -> Option<Self> {
        Some(Self {
            screen: scaffold.screen?,
            ending_screen: scaffold.ending?,
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
            title_screen_id: scaffold.title,
            load_label: scaffold.load_label,
            pause_view: scaffold.pause,
            settings_screen: scaffold.settings,
            settings_label: scaffold.settings_label,
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

// The auto-save file (resumed by `story:continue`). One per game: v1 allows a
// single story per world, so saves are a flat list with no per-story key.
fn save_file(dir: &Path) -> PathBuf {
    dir.join("auto")
}

// A numbered manual save slot. Slots are 1-based on disk (`save1`..) to match
// the slot-menu labels; `slot` is the 0-based internal index.
fn slot_file(dir: &Path, slot: usize) -> PathBuf {
    dir.join(format!("save{}", slot + 1))
}

fn read_save(path: &Path) -> Option<StorySave> {
    crate::cbor_file::read(path, "StorySystem: save")
}

fn write_save(path: &Path, save: &StorySave) -> std::io::Result<()> {
    crate::cbor_file::write(path, save)
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
    // Momentary fast-forward while the skip modifier (Control) is held: acts
    // like `skip` but only for as long as the key is down, and only in page
    // mode (a choice or overlay stops it, like the toggle).
    hold_skip: bool,
    // Seconds since the current page finished revealing (auto) or turned
    // (skip).
    mode_timer: f32,
    overlay: Overlay,
    // Index of the first slot shown in the slot overlay's top row: the window
    // into the `SLOT_COUNT` logical slots. Scrolled by the wheel / arrow keys.
    slot_scroll: usize,
    // Fractional wheel travel toward the next row (reset when the overlay opens).
    slot_scroll_accum: f32,
    // Recent dialogue for the backlog overlay, one pre-wrapped entry per
    // page shown.
    history: Vec<String>,
    // The active screen as announced by UiInputSystem; stage input (advance,
    // choose) is ignored while a menu or the title screen is up.
    active_screen: Option<AssetId>,
    // The screen the settings screen returns to on Back (the pause menu or the
    // title, whichever opened it).
    settings_return: Option<AssetId>,
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
            save_dir: concinnity_store::paths::saves_dir(),
            menu: Vec::new(),
            typewriter: Typewriter::default(),
            last_step: None,
            elapsed: 0.0,
            auto: false,
            skip: false,
            hold_skip: false,
            mode_timer: 0.0,
            overlay: Overlay::None,
            slot_scroll: 0,
            slot_scroll_accum: 0.0,
            history: Vec::new(),
            active_screen: None,
            settings_return: None,
            command_cursor: crate::ecs::EventCursor::default(),
            view_shown_cursor: crate::ecs::EventCursor::default(),
            reload_cursor: crate::ecs::EventCursor::default(),
        }
    }
}

impl System for StorySystem {
    fn access(&self) -> crate::ecs::Access {
        crate::ecs::Access::new()
            .reads_components(crate::component_mask![crate::assets::FrameInput])
            .writes_components(crate::component_mask![
                crate::assets::TextLabel,
                crate::assets::Sprite,
            ])
            .reads_resources(crate::resource_mask![
                crate::assets::StoryReload,
                crate::assets::ScreenShown,
                crate::assets::StoryCommand,
            ])
            .writes_resources(crate::resource_mask![
                crate::assets::PlayCue,
                crate::assets::ScreenCommand,
            ])
    }

    fn init(&mut self, ctx: &mut PipelineContext) {
        // A preview session's saves land in a sandbox wiped here, so the save
        // UI works without touching the user's real files and every session
        // starts fresh.
        if ctx
            .resource::<crate::ecs::TransientSaves>()
            .is_some_and(|t| t.0)
        {
            self.save_dir = concinnity_store::paths::preview_saves_dir();
            let _ = std::fs::remove_dir_all(&self.save_dir);
        }
        // The scaffold references were resolved to ids at build time, like
        // every other cross-reference, so this works identically for a
        // compiled blob (`cn run`) and the interpreted debug path.
        match StageIds::from_scaffold(&self.story.scaffold) {
            Some(ids) => {
                self.ids = Some(ids);
                // The title menu is laid out on the first `ScreenShown` (in
                // `step`), not here. Each menu button's hit region captures its
                // fixed vertical offset to its label in `UiInputSystem::init`,
                // which runs after this one. Relaying the labels now would move
                // them out from under that capture, so every region would bake
                // in the wrong offset and stick at its emitted position instead
                // of tracking the runtime layout. Leaving the emitted positions
                // untouched lets the capture read the authored region-to-label
                // gap; the first title `ScreenShown` (announced by that same init)
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
                .map(|e| e.story.clone())
                .collect(),
            None => Vec::new(),
        };
        for story in reloads {
            self.reload(story, ctx);
        }

        // Track the active screen; auto-start when the stage itself is the
        // world's initial screen (no title screen).
        let shown: Vec<AssetId> = match ctx.events::<ScreenShown>() {
            Some(events) => events
                .read(&mut self.view_shown_cursor)
                .map(|e| e.screen)
                .collect(),
            None => Vec::new(),
        };
        for screen in shown {
            self.active_screen = Some(screen);
            // The load overlay shows the stage without starting play.
            if !self.started
                && self.overlay == Overlay::None
                && Some(screen) == self.ids.as_ref().map(|i| i.screen)
            {
                self.start(ctx);
            }
            // Re-lay the title menu each time the title is shown: the save
            // state may have changed while a story played (a fresh auto-save,
            // a new slot, or a finished story clearing its auto-save).
            if Some(screen) == self.ids.as_ref().and_then(|i| i.title_screen_id) {
                // Returning to the title (Quit-to-Title, or the ending's Back)
                // ends the playthrough, so Start / Continue / Load behave as a
                // fresh entry -- in particular Load routes through the
                // not-started path that raises the stage.
                self.started = false;
                self.layout_title_menu(ctx);
            }
        }

        let commands: Vec<StoryCommand> = match ctx.events::<StoryCommand>() {
            Some(events) => events.read(&mut self.command_cursor).cloned().collect(),
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
                StoryCommand::TogglePause => self.toggle_pause(ctx),
                StoryCommand::OpenSettings => self.open_settings(ctx),
                StoryCommand::CloseSettings => self.close_settings(ctx),
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

        // The per-frame input snapshot (queried, not drained, like every other
        // FrameInput reader). GraphicsSystem deposits it before this system runs,
        // so it is the current frame's input. Drives the held-Control fast-forward
        // and the slot-overlay scroll.
        let frame = ctx
            .query::<FrameInput>()
            .last()
            .cloned()
            .unwrap_or_default();
        self.update_hold_skip(&frame, ctx);
        self.handle_slot_scroll(&frame, ctx);

        self.tick_typewriter(ctx, dt);
        self.tick_modes(ctx, dt);
        StepResult::Continue
    }
}

impl StorySystem {
    // Finish the current page's reveal immediately and repaint its text.
    pub(super) fn reveal_all(&mut self, ctx: &mut PipelineContext) {
        self.typewriter.shown = self.typewriter.full.len();
        let text = self.typewriter.text();
        let text_id = self.ids.as_ref().expect("resolved at init").text;
        set_label(ctx, text_id, |l| l.content = text);
    }
}

// Mutate the stage component with the given asset id; an unset reference or a
// component the world never declared is a silent no-op.
fn set_label(ctx: &mut PipelineContext, id: Option<AssetId>, apply: impl FnOnce(&mut TextLabel)) {
    crate::ecs::by_asset_id::update(ctx, id, apply);
}

fn set_sprite(ctx: &mut PipelineContext, id: Option<AssetId>, apply: impl FnOnce(&mut Sprite)) {
    crate::ecs::by_asset_id::update(ctx, id, apply);
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
        // An empty slot goes fully transparent rather than invisible: screen
        // re-activation force-shows member sprites.
        None => set_sprite(ctx, slot, |s| s.tint = [1.0, 1.0, 1.0, 0.0]),
    }
}

fn play(ctx: &mut PipelineContext, clip: Option<AudioClipHandle>, kind: CueKind) {
    let Some(clip) = clip else { return };
    ctx.events_mut::<PlayCue>().send(PlayCue {
        clip,
        kind,
        volume: 1.0,
        priority: 0,
    });
}
