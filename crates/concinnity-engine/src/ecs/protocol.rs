//! Renderer-free protocol types: the resource singletons the runtime systems
//! publish and read to coordinate a tick, plus the world's cook-counted physics
//! reservation, published once at blob load. They name no graphics backend,
//! windowing, physics, or audio type, so they live in core where every subsystem
//! crate can reach them without depending on the renderer.

use alloc::string::String;
use alloc::vec::Vec;
pub use concinnity_core::fanout::ScheduleMode;

use crate::blob::PhysicsBudgetRecord;
use crate::ecs::FontHandle;
use crate::ecs::asset_id::AssetId;

/// The world's physics reservation as cook counted it, published at blob load.
/// Absent when the world declares no physics content, or when the world was
/// built in memory rather than loaded from a blob; the simulation then counts
/// the loaded components itself.
///
/// Lives here rather than with the engine's other blob resources because the
/// simulation driver reads it, and the engine depends on the driver.
pub struct WorldPhysicsBudget(pub PhysicsBudgetRecord);

/// The blob's baked per-scene exclusive content groups, published at blob load
/// for the streaming and scene-residency wiring to consume at graphics init.
pub struct BlobSceneGroups(pub Vec<crate::blob::SceneGroup>);

/// The blob's baked per-mesh geometry summaries (AABB + counts by mesh-source
/// handle), published at blob load so graphics init can build draw records for
/// deferred scene-owned meshes without decoding their payloads.
pub struct BlobMeshBounds(pub Vec<crate::blob::MeshBoundsRecord>);

/// Per-frame menu state, published as a resource by the overlay build (which runs
/// first in the schedule) and read by the simulation systems the same tick.
/// `true` while any world-pausing screen is open: physics and animation then freeze so they
/// stop consuming resources behind the menu. Each system keeps its own clock
/// aligned across the freeze, so resuming costs one normal frame -- no catch-up
/// burst, no pose jump.
#[derive(Debug, Clone, Copy, Default)]
pub struct MenuActive(pub bool);

/// Fixed-timestep budget for the current frame, published by the App-level
/// simulation clock before each world step. `ticks` is how many fixed steps the
/// simulation systems (physics, behavior) run this frame; `tick_dt` is the
/// seconds each step advances; `alpha` is the accumulator remainder as a
/// fraction of `tick_dt`, used to blend the previous and current simulated
/// states when writing render-facing transforms. Absent (a directly-stepped
/// world with no App), the default is exactly one tick per step with no
/// blending, which makes bare `World::step` loops deterministic.
#[derive(Debug, Clone, Copy)]
pub struct SimTiming {
    /// Fixed simulation steps to run this frame.
    pub ticks: u32,
    /// Seconds each fixed step advances.
    pub tick_dt: f32,
    /// Accumulator remainder as a fraction of `tick_dt`.
    pub alpha: f32,
    /// Seconds simulated once this frame's ticks have run: tick `i` of the
    /// frame starts at `time - (ticks - i) * tick_dt`.
    pub time: f64,
}

impl SimTiming {
    /// Seconds each fixed simulation step advances (60 Hz).
    pub const TICK_DT: f32 = 1.0 / 60.0;
    /// Most fixed steps one frame may run. Time past this is dropped, so a
    /// long hitch degrades to slow motion instead of a catch-up spiral.
    pub const MAX_TICKS_PER_FRAME: u32 = 5;

    /// The simulated instant tick `i` of this frame starts at.
    pub fn tick_start(&self, i: u32) -> f64 {
        self.time - f64::from(self.ticks.saturating_sub(i)) * f64::from(self.tick_dt)
    }

    /// The simulated instant the frame's blended poses show: `alpha` of the
    /// way from the state before the last tick to the state after it.
    pub fn drawn_time(&self) -> f64 {
        self.time - f64::from(1.0 - self.alpha) * f64::from(self.tick_dt)
    }
}

impl Default for SimTiming {
    fn default() -> Self {
        Self {
            ticks: 1,
            tick_dt: Self::TICK_DT,
            alpha: 1.0,
            time: 0.0,
        }
    }
}

/// Real time for the current frame, published by the App-level clock next to
/// [`SimTiming`]. `dt` is the seconds since the previous frame, still measured
/// while a menu pauses the world, and `elapsed` is the monotonic seconds since
/// the App started. Presentation systems (animation, cameras, HUDs, fades)
/// accumulate their own clocks from `dt`. Absent (a directly-stepped world with
/// no App), the default is one fixed tick of `dt` and an `elapsed` that stays
/// at zero, which makes bare `World::step` loops deterministic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameTime {
    /// Seconds since the previous frame.
    pub dt: f32,
    /// Seconds since the App started.
    pub elapsed: f32,
}

impl Default for FrameTime {
    fn default() -> Self {
        Self {
            dt: SimTiming::TICK_DT,
            elapsed: 0.0,
        }
    }
}

/// The live frame-rate cap in FPS (0 = unlimited), published by the renderer
/// (from `FramePacing` at init, refreshed by the settings row's live change)
/// and read by the runtime-level frame pacer before each world step. Independent of
/// the quality preset (a user/hardware preference, like vsync).
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameRateCap(pub u32);

/// Whether the window is minimized, published by the input poll from each frame's
/// sampled window state and read by the runtime-level frame pacer, which holds
/// a minimized run to a background rate.
#[derive(Debug, Clone, Copy, Default)]
pub struct WindowMinimized(pub bool);

/// An external per-frame driver (the `cn editor` HUD) can force the world's
/// "menu active" state through this resource: `Some(true)` frees the cursor and
/// freezes gameplay/physics/animation (edit mode), `Some(false)` captures the
/// cursor and lets the world run (play mode), both regardless of whether the
/// world has its own menu UI. The renderer also puts the backend in menu mode
/// while it is set, so a click frees to a UI action instead of re-capturing the
/// camera. `None` (the default absence) leaves the world's own menu logic in
/// charge; a shipped runtime never publishes it.
#[derive(Debug, Clone, Copy, Default)]
pub struct MenuOverride(pub Option<bool>);

/// Keeps a preview session out of the user's real save files: while present and
/// true, the systems that persist play state (behavior variables / once flags,
/// story position) neither read nor write their disk saves -- every session
/// starts fresh and leaves no trace. In-memory state is unaffected, so a `save`
/// node still works within the session. Published by the `cn editor` HUD
/// injection (sampled at each system's init); a shipped runtime never
/// publishes it.
#[derive(Debug, Clone, Copy, Default)]
pub struct TransientSaves(pub bool);

/// One hop of a behavior-node address, mirroring the world checker's fault
/// paths: object fields by key, list members by position. A node's path walks
/// from the behavior's args to the node (e.g. `do[1].if.then[0]` is
/// `[Field("do"), Index(1), Field("if"), Field("then"), Index(0)]` minus the
/// node's own trailing verb), so the editor can resolve a traced node to the
/// same outline row / chart card its checker faults land on. Field names are
/// the fixed authoring keys, so they borrow statically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceStep {
    /// An object field, by its fixed authoring key.
    Field(&'static str),
    /// A list member, by position.
    Index(u32),
}

/// A behavior node's address: the hops from the behavior's args down to it.
pub type TracePath = Vec<TraceStep>;

/// A behavior-body value in its cross-boundary form: what
/// [`Val`](crate::behavior::Val) publishes to an observer. Entities travel as
/// their id bits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TraceVal {
    /// A boolean value.
    Bool(bool),
    /// An integer value.
    Int(i32),
    /// A floating-point value.
    Float(f32),
    /// A 3-component vector value.
    Vec3([f32; 3]),
    /// An entity, as its id bits.
    Entity(u64),
}

/// One node execution: which behavior, and the node's compile-assigned
/// pre-order id (an index into that behavior's [`TracePaths`] entry).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceEvent {
    /// The behavior the node belongs to.
    pub behavior: AssetId,
    /// The node's compile-assigned pre-order id.
    pub node: u32,
}

/// An external observer's request for execution tracing, published per frame by
/// the `cn editor` HUD while its Behavior panel is open and removed when it
/// closes. While present, the behavior system records which nodes ran each
/// simulated tick and publishes [`ExecutionTrace`]; absent (the shipped runtime,
/// or the panel closed), the system does no recording work beyond noticing the
/// absence. `entity` selects whose per-entity locals to surface; `breakpoints`
/// are nodes whose execution should be reported as a [`ExecutionTrace::hit`] so
/// the observer can pause the simulation.
#[derive(Debug, Clone, Default)]
pub struct TraceRequest {
    /// Whose per-entity locals to surface, as entity id bits.
    pub entity: Option<u64>,
    /// Nodes whose execution should be reported as a hit.
    pub breakpoints: Vec<TraceEvent>,
}

/// What the behavior system observed over one simulated tick, published while a
/// [`TraceRequest`] stands. `frame` increments per published tick so the observer
/// can tell fresh data from the stale resource a paused world leaves behind.
/// `events` are the nodes that ran (deduplicated); `vars` the world variables
/// with their current values in slot order; `locals` the requested entity's
/// per-behavior locals; `hit` the first executed breakpoint, if any.
#[derive(Debug, Clone, Default)]
pub struct ExecutionTrace {
    /// Increments per published tick, so stale data is recognizable.
    pub frame: u64,
    /// The nodes that ran this tick, deduplicated.
    pub events: Vec<TraceEvent>,
    /// World variables with their current values, in slot order.
    pub vars: Vec<(String, TraceVal)>,
    /// The requested entity's per-behavior locals.
    pub locals: Vec<(AssetId, String, TraceVal)>,
    /// The first executed breakpoint, if any.
    pub hit: Option<TraceEvent>,
}

/// Each behavior's node paths, indexed by the node ids [`ExecutionTrace`] events
/// carry. Published once when tracing is first requested (the compile that
/// derives it runs at init either way; the publish just exposes it).
#[derive(Debug, Clone, Default)]
pub struct TracePaths(pub Vec<(AssetId, Vec<TracePath>)>);

/// The silhouette the in-engine cursor sprite should draw this frame. Published
/// by the `cn editor` HUD when the pointer is over a resizable panel's edge or
/// corner (or while a resize drag is in flight) and read by the overlay build,
/// which draws the matching shape at the pointer in place of the arrow. `Default`
/// is the plain arrow; the four resize shapes are double-headed arrows along a
/// window edge (east/west), edge (north/south), and the two diagonals. A shipped
/// runtime never publishes it, so the arrow always stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CursorShape {
    #[default]
    /// The plain arrow.
    Default,
    /// Double-headed arrow along the east/west edge.
    ResizeEW,
    /// Double-headed arrow along the north/south edge.
    ResizeNS,
    /// Double-headed diagonal arrow, north-west to south-east.
    ResizeNWSE,
    /// Double-headed diagonal arrow, north-east to south-west.
    ResizeNESW,
}

#[derive(Debug, Clone, Copy, Default)]
/// The silhouette the in-engine cursor sprite should draw this frame.
pub struct DesiredCursor(pub CursorShape);

/// Per-frame draw-layer overrides for HUD Sprites / `TextLabels` / `TextInputs`, keyed
/// by asset id and published by the `cn editor` HUD so its floating panels occlude
/// cleanly. Overlay draw calls render in two passes (all sprites, then all text),
/// so two overlapping panels' contents merge -- one panel's text draws over the
/// other's background. The renderer stable-sorts the overlay calls by this layer
/// (higher draws on top) when the map is non-empty, so the focused panel's whole
/// content sits above the others'. An id absent from the map is layer 0; an empty /
/// absent resource (the shipped runtime) leaves draw order at insertion order,
/// unchanged.
#[derive(Debug, Clone, Default)]
pub struct HudLayers(pub alloc::collections::BTreeMap<AssetId, i32>);

/// The active screen stack, published by `UiInputSystem` at init and whenever the
/// stack changes, and read a frame later (the same one-frame lag screen
/// visibility flips already have). `layers` maps each active Screen's id to its
/// computed draw layer (authored layer band + stack position; screen-less HUD
/// elements sit at 0); the overlay build spreads these onto the elements each
/// screen owns. `pauses_world` is true while any active screen pauses the
/// world; `captures_input` is true while any active screen captures input
/// (gameplay keys are suppressed even when the world keeps simulating).
/// Absent / empty in a world with no active screen.
#[derive(Debug, Clone, Default)]
pub struct ScreenStack {
    /// Each active screen's computed draw layer, keyed by screen id.
    pub layers: alloc::collections::BTreeMap<AssetId, i32>,
    /// `true` while any active screen pauses the world.
    pub pauses_world: bool,
    /// `true` while any active screen captures input.
    pub captures_input: bool,
}

/// World-space lines to draw this frame (trajectories, tethers, path previews,
/// the editor's origin axes), republished by their producer every frame:
/// the renderer expands whatever it finds into ribbon geometry and hands it
/// to the backend, so a stale list would keep drawing. Absent when nothing
/// draws lines, which keeps the line pass out of the frame graph.
#[derive(Debug, Clone, Default)]
pub struct WorldLines(pub Vec<crate::gfx::lines::Line>);

/// Device-memory pressure signal, published by the renderer whenever GPU
/// work fails for lack of device memory. Renderer-free counters so the
/// streaming valve can react (tighten budgets, evict) without naming the
/// renderer; nothing consumes it yet.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuMemoryPressure {
    /// Device-memory failures observed since startup.
    pub events: u64,
    /// Frame index of the most recent failure.
    pub last_frame: u64,
}

/// The editor's fly-camera state. While true (published only by the `cn
/// editor` HUD drive), the input poll keeps the navigation keys and mouse deltas
/// live and the renderer captures the cursor even though the world is frozen
/// behind the editor's menu override -- the editor integrates `Camera3D` itself,
/// so the viewport can be flown without running the simulation. Absent / false
/// in a shipped runtime.
#[derive(Debug, Clone, Copy, Default)]
pub struct FlyCam(pub bool);

/// Assets suppressed from rendering for this frame. The renderer collapses
/// each listed asset's draw slots to a degenerate transform (so it neither
/// rasterizes nor casts shadows) and drops it from the [`PickIndex`]. Authored
/// data is untouched, and the collapse is re-derived every frame, so clearing
/// an id restores the object immediately. Published by the `cn editor` HUD
/// drive; absent / empty otherwise.
#[derive(Debug, Clone, Default)]
pub struct HiddenAssets(pub alloc::collections::BTreeSet<AssetId>);

/// The viewport's view mode + show flags. The renderer forwards it to the
/// backend's `FrameParams`: the mode selects what the composite presents, the
/// flags skip feature passes for the frame. Published by the editor each frame
/// and by the dev tools' `view-set` and `show-set` verbs; absent otherwise,
/// which reads as the lit default.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ViewOverrides {
    /// What the composite presents.
    pub mode: crate::gfx::view_modes::ViewMode,
    /// Feature passes to skip for the frame.
    pub show: crate::gfx::view_modes::ShowFlags,
}

/// One pickable entity in the [`PickIndex`]: its asset id and current world-space
/// AABB. Ray-tested by the editor with `math::pick::ray_aabb`.
#[derive(Debug, Clone, Copy)]
pub struct PickEntry {
    /// The pickable entity's asset id.
    pub asset_id: AssetId,
    /// Lower corner of the world-space AABB.
    pub bb_min: [f32; 3],
    /// Upper corner of the world-space AABB.
    pub bb_max: [f32; 3],
}

/// The per-frame viewport-picking index: every renderable prop entity's asset id
/// and world-space AABB, refreshed by the renderer from the live transforms.
/// Opt-in: the renderer only builds it when the resource is already present at
/// init (the `cn editor` HUD injection inserts an empty one), so a shipped
/// runtime never pays for it. Rooms, instanced clusters, and voxel chunks are
/// not indexed; picking targets authored prop placements.
#[derive(Debug, Clone, Default)]
pub struct PickIndex {
    /// One entry per indexed prop entity.
    pub entries: Vec<PickEntry>,
}

/// One extra RGBA8 image for the sprite/text atlas pool, bound to a reserved
/// [`TextureHandle`](crate::components) the inserting tool chose. The handle space
/// must stay clear of the compiled world's dense texture handles (tools use a
/// high base).
#[derive(Debug, Clone)]
pub struct OverlayImage {
    /// The reserved handle a sprite names to sample this image.
    pub handle: crate::ecs::TextureHandle,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Row-major RGBA8 pixels.
    pub rgba: Vec<u8>,
}

impl OverlayImage {
    /// Whether the pixels fill exactly `width * height` RGBA8 texels.
    pub fn is_well_formed(&self) -> bool {
        self.rgba.len() as u64 == u64::from(self.width) * u64::from(self.height) * 4
    }
}

/// Extra images appended to the sprite/text atlas pool at graphics init: a
/// sprite whose `texture` names one of these handles samples the image like any
/// compiled texture. Opt-in like [`PickIndex`]: inserted before start (the `cn
/// editor` HUD injection adds baked asset thumbnails); absent everywhere else,
/// so a shipped runtime never pays for it. Read once at init -- images added to
/// the resource later join the pool on the next world rebuild.
#[derive(Debug, Clone, Default)]
pub struct OverlayImages(pub Vec<OverlayImage>);

/// The latest sampled cursor state (window pixels, top-left origin), published
/// by the input poll after each frame. The renderer reads it when building the
/// next frame's draw list: `follow_cursor` sprites are positioned a frame after
/// the input that moved them, and the in-engine cursor stops drawing once the
/// real cursor has left the window (`outside_window` is false in fullscreen,
/// where the backend confines the cursor, and on backends without window-bounds
/// tracking).
#[derive(Debug, Clone, Copy, Default)]
pub struct CursorState {
    /// Cursor position in window pixels, top-left origin.
    pub pos: (f32, f32),
    /// `true` once the real cursor has left the window.
    pub outside_window: bool,
}

/// Per-frame stats-HUD visibility, published as a resource by the renderer
/// (which runs first) and read by `StatHudSystem` the same tick. Each field is
/// the effective on/off for that chip: the master "Display performance stats"
/// toggle AND the per-readout toggle from the video settings. Absent (a HUD-only
/// unit test with no renderer) is treated as both shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HudPrefs {
    /// Whether the FPS chip is shown.
    pub show_fps: bool,
    /// Whether the VRAM chip is shown.
    pub show_vram: bool,
}

/// A settings dropdown's open floating option list, or `None` when none is open.
/// `UiInputSystem` owns the interaction state (open on a setting row's `open`
/// click, close on a pick / outside click / Escape / scroll) and publishes this
/// each frame; the renderer reads it the next tick to draw the list on top of
/// the menu. The renderer runs first, so the list appears one frame after the
/// row is clicked (the same lag the cursor + cycle labels already carry).
#[derive(Debug, Clone, Default)]
pub struct OpenDropdown(pub Option<DropdownView>);

/// What the renderer needs to draw an open dropdown list: the anchor control
/// rect (reference space), the option labels top-to-bottom, the selected +
/// hovered OPTION indices to highlight, the scroll position (`first`, the top
/// shown option of a list longer than the layout window), and the row value
/// label's font / scale / color so the list text matches the row it drops from.
#[derive(Debug, Clone)]
pub struct DropdownView {
    /// Anchor control rect in reference space.
    pub anchor: [f32; 4],
    /// Option labels, top to bottom.
    pub options: Vec<String>,
    /// Index of the selected option.
    pub selected: usize,
    /// Index of the top shown option, for a scrolled list.
    pub first: usize,
    /// Index of the hovered option, if any.
    pub hovered: Option<usize>,
    /// The screen the row belongs to, when it belongs to one.
    pub screen: Option<AssetId>,
    /// Font of the row's value label, so list text matches it.
    pub font: Option<FontHandle>,
    /// Text scale of the row's value label.
    pub scale: f32,
    /// Linear RGB text color of the row's value label.
    pub color: [f32; 3],
}

/// The [`ScheduleMode`] a world runs under: the published resource, or
/// `Serial` when nothing published one.
pub fn schedule_mode(resources: &crate::ecs::Resources) -> ScheduleMode {
    resources
        .get::<ScheduleMode>()
        .copied()
        .unwrap_or(ScheduleMode::Serial)
}

/// When the input this tick consumed was sampled, in microseconds on the host
/// clock. Published by the input system; it names the tick's frame, which
/// carries it to the backend and which the frame report pairs with its GPU
/// completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputSampledAt(pub u64);

/// A frame the report wants saved: the PNG path for the next frame drawn.
/// Written by the frame report; the graphics system moves it onto that frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameCaptureRequest(pub String);

/// A `screenshot` action fired: save this many consecutive frames, starting
/// with the next drawn. Written by the UI input system; the graphics system
/// schedules the burst.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScreenshotRequest(pub u32);

/// Setting rows disabled at runtime (e.g. `show_fps` while "Display
/// performance stats" is off), published each frame by the graphics system.
/// The UI input system makes a matching row inert (no hover, no click) while
/// its labels are grayed independently.
#[derive(Debug, Clone, Default)]
pub struct DisabledSettingRows(pub hashbrown::HashSet<crate::settings::SettingKey>);

/// The display modes the "Resolution" settings row offers, published once by
/// the graphics system at init and read by the UI input system to seed the
/// row's dropdown list. Ordered as displayed; a pick's `SetIndex` indexes it.
#[derive(Debug, Clone, Default)]
pub struct DisplayModes(pub Vec<crate::window::display_mode::DisplayMode>);

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn an_overlay_image_is_well_formed_only_at_its_exact_size() {
        let image = |rgba: Vec<u8>| OverlayImage {
            handle: crate::ecs::TextureHandle::new(0),
            width: 2,
            height: 1,
            rgba,
        };
        assert!(image(vec![0; 8]).is_well_formed());
        assert!(!image(vec![0; 7]).is_well_formed());
        assert!(!image(vec![0; 12]).is_well_formed());
    }
}
