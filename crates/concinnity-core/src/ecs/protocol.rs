// src/ecs/protocol.rs
//
// Renderer-free per-frame protocol types: the resource singletons the runtime
// systems publish and read to coordinate one tick. They name no graphics
// backend, windowing, physics, or audio type, so they live in core where every
// subsystem crate can reach them without depending on the renderer. The client
// `ecs` module re-exports them under the historical `crate::ecs::*` paths.

use crate::ecs::asset_id::AssetId;
use concinnity_asset::FontHandle;

// Per-frame menu state, published as a resource by the overlay build (which runs
// first in the schedule) and read by the simulation systems the same tick.
// `true` while any world-pausing screen is open: physics and animation then freeze so they
// stop consuming resources behind the menu. Each system keeps its own clock
// aligned across the freeze, so resuming costs one normal frame -- no catch-up
// burst, no pose jump.
#[derive(Debug, Clone, Copy, Default)]
pub struct MenuActive(pub bool);

// Fixed-timestep budget for the current frame, published by the App-level
// simulation clock before each world step. `ticks` is how many fixed steps the
// simulation systems (physics, behavior) run this frame; `tick_dt` is the
// seconds each step advances; `alpha` is the accumulator remainder as a
// fraction of `tick_dt`, used to blend the previous and current simulated
// states when writing render-facing transforms. Absent (a directly-stepped
// world with no App), the default is exactly one tick per step with no
// blending, which makes bare `World::step` loops deterministic.
#[derive(Debug, Clone, Copy)]
pub struct SimTiming {
    pub ticks: u32,
    pub tick_dt: f32,
    pub alpha: f32,
}

impl SimTiming {
    /// Seconds each fixed simulation step advances (60 Hz).
    pub const TICK_DT: f32 = 1.0 / 60.0;
}

impl Default for SimTiming {
    fn default() -> Self {
        Self {
            ticks: 1,
            tick_dt: Self::TICK_DT,
            alpha: 1.0,
        }
    }
}

// The live frame-rate cap in FPS (0 = unlimited), published by GraphicsSystem
// (from GraphicsConfig at init, refreshed by the settings row's live change)
// and read by the App-level frame pacer before each world step. Independent of
// the quality preset (a user/hardware preference, like vsync).
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameRateCap(pub u32);

// An external per-frame driver (the `cn editor` HUD) can force the world's
// "menu active" state through this resource: `Some(true)` frees the cursor and
// freezes gameplay/physics/animation (edit mode), `Some(false)` captures the
// cursor and lets the world run (play mode), both regardless of whether the
// world has its own menu UI. GraphicsSystem also puts the backend in menu mode
// while it is set, so a click frees to a UI action instead of re-capturing the
// camera. `None` (the default absence) leaves the world's own menu logic in
// charge; a shipped runtime never publishes it.
#[derive(Debug, Clone, Copy, Default)]
pub struct MenuOverride(pub Option<bool>);

// Keeps a preview session out of the user's real save files: while present and
// true, the systems that persist play state (behavior variables / once flags,
// story position) neither read nor write their disk saves -- every session
// starts fresh and leaves no trace. In-memory state is unaffected, so a `save`
// node still works within the session. Published by the `cn editor` HUD
// injection (sampled at each system's init); a shipped runtime never
// publishes it.
#[derive(Debug, Clone, Copy, Default)]
pub struct TransientSaves(pub bool);

// One hop of a behavior-node address, mirroring the world checker's fault
// paths: object fields by key, list members by position. A node's path walks
// from the behavior's args to the node (e.g. `do[1].if.then[0]` is
// `[Field("do"), Index(1), Field("if"), Field("then"), Index(0)]` minus the
// node's own trailing verb), so the editor can resolve a traced node to the
// same outline row / chart card its checker faults land on. Field names are
// the fixed authoring keys, so they borrow statically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceStep {
    Field(&'static str),
    Index(u32),
}

pub type TracePath = alloc::vec::Vec<TraceStep>;

// A behavior-body value in its cross-boundary form (the runtime's value type
// is private to the behavior system). Entities travel as their id bits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TraceVal {
    Bool(bool),
    Int(i32),
    Float(f32),
    Vec3([f32; 3]),
    Entity(u64),
}

// One node execution: which behavior, and the node's compile-assigned
// pre-order id (an index into that behavior's [TracePaths] entry).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceEvent {
    pub behavior: AssetId,
    pub node: u32,
}

// An external observer's request for execution tracing, published per frame by
// the `cn editor` HUD while its Behavior panel is open and removed when it
// closes. While present, the behavior system records which nodes ran each
// simulated tick and publishes [ExecutionTrace]; absent (the shipped runtime,
// or the panel closed), the system does no recording work beyond noticing the
// absence. `entity` selects whose per-entity locals to surface; `breakpoints`
// are nodes whose execution should be reported as a [ExecutionTrace::hit] so
// the observer can pause the simulation.
#[derive(Debug, Clone, Default)]
pub struct TraceRequest {
    pub entity: Option<u64>,
    pub breakpoints: alloc::vec::Vec<TraceEvent>,
}

// What the behavior system observed over one simulated tick, published while a
// [TraceRequest] stands. `frame` increments per published tick so the observer
// can tell fresh data from the stale resource a paused world leaves behind.
// `events` are the nodes that ran (deduplicated); `vars` the world variables
// with their current values in slot order; `locals` the requested entity's
// per-behavior locals; `hit` the first executed breakpoint, if any.
#[derive(Debug, Clone, Default)]
pub struct ExecutionTrace {
    pub frame: u64,
    pub events: alloc::vec::Vec<TraceEvent>,
    pub vars: alloc::vec::Vec<(String, TraceVal)>,
    pub locals: alloc::vec::Vec<(AssetId, String, TraceVal)>,
    pub hit: Option<TraceEvent>,
}

// Each behavior's node paths, indexed by the node ids [ExecutionTrace] events
// carry. Published once when tracing is first requested (the compile that
// derives it runs at init either way; the publish just exposes it).
#[derive(Debug, Clone, Default)]
pub struct TracePaths(pub alloc::vec::Vec<(AssetId, alloc::vec::Vec<TracePath>)>);

// The silhouette the in-engine cursor sprite should draw this frame. Published
// by the `cn editor` HUD when the pointer is over a resizable panel's edge or
// corner (or while a resize drag is in flight) and read by the overlay build,
// which draws the matching shape at the pointer in place of the arrow. `Default`
// is the plain arrow; the four resize shapes are double-headed arrows along a
// window edge (east/west), edge (north/south), and the two diagonals. A shipped
// runtime never publishes it, so the arrow always stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CursorShape {
    #[default]
    Default,
    ResizeEW,
    ResizeNS,
    ResizeNWSE,
    ResizeNESW,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DesiredCursor(pub CursorShape);

// Per-frame draw-layer overrides for HUD Sprites / TextLabels / TextInputs, keyed
// by asset id and published by the `cn editor` HUD so its floating panels occlude
// cleanly. Overlay draw calls render in two passes (all sprites, then all text),
// so two overlapping panels' contents merge -- one panel's text draws over the
// other's background. GraphicsSystem stable-sorts the overlay calls by this layer
// (higher draws on top) when the map is non-empty, so the focused panel's whole
// content sits above the others'. An id absent from the map is layer 0; an empty /
// absent resource (the shipped runtime) leaves draw order at insertion order,
// unchanged.
#[derive(Debug, Clone, Default)]
pub struct HudLayers(pub alloc::collections::BTreeMap<AssetId, i32>);

// The active screen stack, published by UiInputSystem at init and whenever the
// stack changes, and read a frame later (the same one-frame lag screen
// visibility flips already have). `layers` maps each active Screen's id to its
// computed draw layer (authored layer band + stack position; screen-less HUD
// elements sit at 0); the overlay build spreads these onto the elements each
// screen owns. `pauses_world` is true while any active screen pauses the
// world; `captures_input` is true while any active screen captures input
// (gameplay keys are suppressed even when the world keeps simulating).
// Absent / empty in a world with no active screen.
#[derive(Debug, Clone, Default)]
pub struct ScreenStack {
    pub layers: alloc::collections::BTreeMap<AssetId, i32>,
    pub pauses_world: bool,
    pub captures_input: bool,
}

// World-space lines to draw this frame (trajectories, tethers, path previews,
// the editor's origin axes), republished by their producer every frame:
// GraphicsSystem expands whatever it finds into ribbon geometry and hands it
// to the backend, so a stale list would keep drawing. Absent when nothing
// draws lines, which keeps the line pass out of the frame graph.
#[derive(Debug, Clone, Default)]
pub struct WorldLines(pub alloc::vec::Vec<crate::gfx::lines::Line>);

// Device-memory pressure signal, published by GraphicsSystem whenever GPU
// work fails for lack of device memory. Renderer-free counters so the
// streaming valve can react (tighten budgets, evict) without naming the
// renderer; nothing consumes it yet.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuMemoryPressure {
    // Device-memory failures observed since startup.
    pub events: u64,
    // Frame index of the most recent failure.
    pub last_frame: u64,
}

// The editor's fly-camera state. While true (published only by the `cn
// editor` HUD drive), InputSystem keeps the navigation keys and mouse deltas
// live and GraphicsSystem captures the cursor even though the world is frozen
// behind the editor's menu override -- the editor integrates Camera3D itself,
// so the viewport can be flown without running the simulation. Absent / false
// in a shipped runtime.
#[derive(Debug, Clone, Copy, Default)]
pub struct FlyCam(pub bool);

// Assets suppressed from rendering for this frame. GraphicsSystem collapses
// each listed asset's draw slots to a degenerate transform (so it neither
// rasterizes nor casts shadows) and drops it from the [PickIndex]. Authored
// data is untouched, and the collapse is re-derived every frame, so clearing
// an id restores the object immediately. Published by the `cn editor` HUD
// drive; absent / empty otherwise.
#[derive(Debug, Clone, Default)]
pub struct HiddenAssets(pub alloc::collections::BTreeSet<AssetId>);

// The viewport's view mode + show flags, published per frame by the editor.
// GraphicsSystem forwards it to the backend's FrameParams: the mode selects
// what the composite presents, the flags skip feature passes for the frame.
// Absent outside the editor, which reads as the lit default.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ViewOverrides {
    pub mode: crate::gfx::view_modes::ViewMode,
    pub show: crate::gfx::view_modes::ShowFlags,
}

// One pickable entity in the [PickIndex]: its asset id and current world-space
// AABB. Ray-tested by the editor with `gfx::pick::ray_aabb`.
#[derive(Debug, Clone, Copy)]
pub struct PickEntry {
    pub asset_id: AssetId,
    pub bb_min: [f32; 3],
    pub bb_max: [f32; 3],
}

// The per-frame viewport-picking index: every renderable prop entity's asset id
// and world-space AABB, refreshed by GraphicsSystem from the live transforms.
// Opt-in: GraphicsSystem only builds it when the resource is already present at
// init (the `cn editor` HUD injection inserts an empty one), so a shipped
// runtime never pays for it. Rooms, instanced clusters, and voxel chunks are
// not indexed; picking targets authored prop placements.
#[derive(Debug, Clone, Default)]
pub struct PickIndex {
    pub entries: Vec<PickEntry>,
}

// One extra RGBA8 image for the sprite/text atlas pool, bound to a reserved
// [TextureHandle](crate::assets) the inserting tool chose. The handle space
// must stay clear of the compiled world's dense texture handles (tools use a
// high base).
#[derive(Debug, Clone)]
pub struct OverlayImage {
    pub handle: crate::ecs::TextureHandle,
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

// Extra images appended to the sprite/text atlas pool at graphics init: a
// sprite whose `texture` names one of these handles samples the image like any
// compiled texture. Opt-in like [PickIndex]: inserted before start (the `cn
// editor` HUD injection adds baked asset thumbnails); absent everywhere else,
// so a shipped runtime never pays for it. Read once at init -- images added to
// the resource later join the pool on the next world rebuild.
#[derive(Debug, Clone, Default)]
pub struct OverlayImages(pub Vec<OverlayImage>);

// The latest sampled cursor state (window pixels, top-left origin), published
// by InputSystem after each poll. GraphicsSystem reads it when building the
// next frame's draw list: `follow_cursor` sprites are positioned a frame after
// the input that moved them, and the in-engine cursor stops drawing once the
// real cursor has left the window (`outside_window` is false in fullscreen,
// where the backend confines the cursor, and on backends without window-bounds
// tracking).
#[derive(Debug, Clone, Copy, Default)]
pub struct CursorState {
    pub pos: (f32, f32),
    pub outside_window: bool,
}

// Per-frame stats-HUD visibility, published as a resource by GraphicsSystem
// (which runs first) and read by `StatHudSystem` the same tick. Each field is
// the effective on/off for that chip: the master "Display performance stats"
// toggle AND the per-readout toggle from the video settings. Absent (a HUD-only
// unit test with no GraphicsSystem) is treated as both shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HudPrefs {
    pub show_fps: bool,
    pub show_vram: bool,
}

// A settings dropdown's open floating option list, or `None` when none is open.
// `UiInputSystem` owns the interaction state (open on a `setting:<key>:open`
// click, close on a pick / outside click / Escape / scroll) and publishes this
// each frame; GraphicsSystem reads it the next tick to draw the list on top of
// the menu. GraphicsSystem runs first, so the list appears one frame after the
// row is clicked (the same lag the cursor + cycle labels already carry).
#[derive(Debug, Clone, Default)]
pub struct OpenDropdown(pub Option<DropdownView>);

// What GraphicsSystem needs to draw an open dropdown list: the anchor control
// rect (reference space), the option labels top-to-bottom, the selected +
// hovered OPTION indices to highlight, the scroll position (`first`, the top
// shown option of a list longer than the layout window), and the row value
// label's font / scale / color so the list text matches the row it drops from.
#[derive(Debug, Clone)]
pub struct DropdownView {
    pub anchor: [f32; 4],
    pub options: Vec<String>,
    pub selected: usize,
    pub first: usize,
    pub hovered: Option<usize>,
    pub screen: Option<AssetId>,
    pub font: Option<FontHandle>,
    pub scale: f32,
    pub color: [f32; 3],
}
