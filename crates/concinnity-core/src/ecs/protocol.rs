// src/ecs/protocol.rs
//
// Renderer-free protocol types: the resource singletons the runtime systems
// publish and read to coordinate a tick, plus the world's cook-counted physics
// reservation, published once at blob load. They name no graphics backend,
// windowing, physics, or audio type, so they live in core where every subsystem
// crate can reach them without depending on the renderer. The client `ecs`
// module re-exports them under the historical `crate::ecs::*` paths.

use alloc::string::String;
use alloc::vec::Vec;

use crate::blob::PhysicsBudgetRecord;
use crate::ecs::asset_id::AssetId;
use concinnity_asset::FontHandle;

/// The world's physics reservation as cook counted it, published at blob load.
/// Absent when the world declares no physics content, or when the world was
/// built in memory rather than loaded from a blob; the simulation then counts
/// the loaded components itself.
///
/// Lives here rather than with the engine's other blob resources because the
/// simulation driver reads it, and the engine depends on the driver.
pub struct WorldPhysicsBudget(pub PhysicsBudgetRecord);

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

/// The live frame-rate cap in FPS (0 = unlimited), published by GraphicsSystem
/// (from GraphicsConfig at init, refreshed by the settings row's live change)
/// and read by the App-level frame pacer before each world step. Independent of
/// the quality preset (a user/hardware preference, like vsync).
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameRateCap(pub u32);

/// An external per-frame driver (the `cn editor` HUD) can force the world's
/// "menu active" state through this resource: `Some(true)` frees the cursor and
/// freezes gameplay/physics/animation (edit mode), `Some(false)` captures the
/// cursor and lets the world run (play mode), both regardless of whether the
/// world has its own menu UI. GraphicsSystem also puts the backend in menu mode
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
pub type TracePath = alloc::vec::Vec<TraceStep>;

/// A behavior-body value in its cross-boundary form (the runtime's value type
/// is private to the behavior system). Entities travel as their id bits.
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
/// pre-order id (an index into that behavior's [TracePaths] entry).
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
/// simulated tick and publishes [ExecutionTrace]; absent (the shipped runtime,
/// or the panel closed), the system does no recording work beyond noticing the
/// absence. `entity` selects whose per-entity locals to surface; `breakpoints`
/// are nodes whose execution should be reported as a [ExecutionTrace::hit] so
/// the observer can pause the simulation.
#[derive(Debug, Clone, Default)]
pub struct TraceRequest {
    /// Whose per-entity locals to surface, as entity id bits.
    pub entity: Option<u64>,
    /// Nodes whose execution should be reported as a hit.
    pub breakpoints: alloc::vec::Vec<TraceEvent>,
}

/// What the behavior system observed over one simulated tick, published while a
/// [TraceRequest] stands. `frame` increments per published tick so the observer
/// can tell fresh data from the stale resource a paused world leaves behind.
/// `events` are the nodes that ran (deduplicated); `vars` the world variables
/// with their current values in slot order; `locals` the requested entity's
/// per-behavior locals; `hit` the first executed breakpoint, if any.
#[derive(Debug, Clone, Default)]
pub struct ExecutionTrace {
    /// Increments per published tick, so stale data is recognisable.
    pub frame: u64,
    /// The nodes that ran this tick, deduplicated.
    pub events: alloc::vec::Vec<TraceEvent>,
    /// World variables with their current values, in slot order.
    pub vars: alloc::vec::Vec<(String, TraceVal)>,
    /// The requested entity's per-behavior locals.
    pub locals: alloc::vec::Vec<(AssetId, String, TraceVal)>,
    /// The first executed breakpoint, if any.
    pub hit: Option<TraceEvent>,
}

/// Each behavior's node paths, indexed by the node ids [ExecutionTrace] events
/// carry. Published once when tracing is first requested (the compile that
/// derives it runs at init either way; the publish just exposes it).
#[derive(Debug, Clone, Default)]
pub struct TracePaths(pub alloc::vec::Vec<(AssetId, alloc::vec::Vec<TracePath>)>);

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

/// Per-frame draw-layer overrides for HUD Sprites / TextLabels / TextInputs, keyed
/// by asset id and published by the `cn editor` HUD so its floating panels occlude
/// cleanly. Overlay draw calls render in two passes (all sprites, then all text),
/// so two overlapping panels' contents merge -- one panel's text draws over the
/// other's background. GraphicsSystem stable-sorts the overlay calls by this layer
/// (higher draws on top) when the map is non-empty, so the focused panel's whole
/// content sits above the others'. An id absent from the map is layer 0; an empty /
/// absent resource (the shipped runtime) leaves draw order at insertion order,
/// unchanged.
#[derive(Debug, Clone, Default)]
pub struct HudLayers(pub alloc::collections::BTreeMap<AssetId, i32>);

/// The active screen stack, published by UiInputSystem at init and whenever the
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
/// GraphicsSystem expands whatever it finds into ribbon geometry and hands it
/// to the backend, so a stale list would keep drawing. Absent when nothing
/// draws lines, which keeps the line pass out of the frame graph.
#[derive(Debug, Clone, Default)]
pub struct WorldLines(pub alloc::vec::Vec<crate::gfx::lines::Line>);

/// Device-memory pressure signal, published by GraphicsSystem whenever GPU
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
/// editor` HUD drive), InputSystem keeps the navigation keys and mouse deltas
/// live and GraphicsSystem captures the cursor even though the world is frozen
/// behind the editor's menu override -- the editor integrates Camera3D itself,
/// so the viewport can be flown without running the simulation. Absent / false
/// in a shipped runtime.
#[derive(Debug, Clone, Copy, Default)]
pub struct FlyCam(pub bool);

/// Assets suppressed from rendering for this frame. GraphicsSystem collapses
/// each listed asset's draw slots to a degenerate transform (so it neither
/// rasterizes nor casts shadows) and drops it from the [PickIndex]. Authored
/// data is untouched, and the collapse is re-derived every frame, so clearing
/// an id restores the object immediately. Published by the `cn editor` HUD
/// drive; absent / empty otherwise.
#[derive(Debug, Clone, Default)]
pub struct HiddenAssets(pub alloc::collections::BTreeSet<AssetId>);

/// The viewport's view mode + show flags, published per frame by the editor.
/// GraphicsSystem forwards it to the backend's FrameParams: the mode selects
/// what the composite presents, the flags skip feature passes for the frame.
/// Absent outside the editor, which reads as the lit default.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ViewOverrides {
    /// What the composite presents.
    pub mode: crate::gfx::view_modes::ViewMode,
    /// Feature passes to skip for the frame.
    pub show: crate::gfx::view_modes::ShowFlags,
}

/// One pickable entity in the [PickIndex]: its asset id and current world-space
/// AABB. Ray-tested by the editor with `gfx::pick::ray_aabb`.
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
/// and world-space AABB, refreshed by GraphicsSystem from the live transforms.
/// Opt-in: GraphicsSystem only builds it when the resource is already present at
/// init (the `cn editor` HUD injection inserts an empty one), so a shipped
/// runtime never pays for it. Rooms, instanced clusters, and voxel chunks are
/// not indexed; picking targets authored prop placements.
#[derive(Debug, Clone, Default)]
pub struct PickIndex {
    /// One entry per indexed prop entity.
    pub entries: Vec<PickEntry>,
}

/// One extra RGBA8 image for the sprite/text atlas pool, bound to a reserved
/// [TextureHandle](crate::components) the inserting tool chose. The handle space
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

/// Extra images appended to the sprite/text atlas pool at graphics init: a
/// sprite whose `texture` names one of these handles samples the image like any
/// compiled texture. Opt-in like [PickIndex]: inserted before start (the `cn
/// editor` HUD injection adds baked asset thumbnails); absent everywhere else,
/// so a shipped runtime never pays for it. Read once at init -- images added to
/// the resource later join the pool on the next world rebuild.
#[derive(Debug, Clone, Default)]
pub struct OverlayImages(pub Vec<OverlayImage>);

/// The latest sampled cursor state (window pixels, top-left origin), published
/// by InputSystem after each poll. GraphicsSystem reads it when building the
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

/// Per-frame stats-HUD visibility, published as a resource by GraphicsSystem
/// (which runs first) and read by `StatHudSystem` the same tick. Each field is
/// the effective on/off for that chip: the master "Display performance stats"
/// toggle AND the per-readout toggle from the video settings. Absent (a HUD-only
/// unit test with no GraphicsSystem) is treated as both shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HudPrefs {
    /// Whether the FPS chip is shown.
    pub show_fps: bool,
    /// Whether the VRAM chip is shown.
    pub show_vram: bool,
}

/// A settings dropdown's open floating option list, or `None` when none is open.
/// `UiInputSystem` owns the interaction state (open on a `setting:<key>:open`
/// click, close on a pick / outside click / Escape / scroll) and publishes this
/// each frame; GraphicsSystem reads it the next tick to draw the list on top of
/// the menu. GraphicsSystem runs first, so the list appears one frame after the
/// row is clicked (the same lag the cursor + cycle labels already carry).
#[derive(Debug, Clone, Default)]
pub struct OpenDropdown(pub Option<DropdownView>);

/// What GraphicsSystem needs to draw an open dropdown list: the anchor control
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
    /// Linear RGB text colour of the row's value label.
    pub color: [f32; 3],
}

/// How a tick's independent work executes. `Parallel` lets systems fan their
/// safe internal work across the job pool; `Serial` (or the resource being
/// absent, the editor's case) keeps every system's work on the stepping
/// thread -- the determinism oracle and the escape hatch
/// (`cn run --serial-schedule`). Both modes must produce identical world
/// state; the engine's schedule-determinism test is the gate on that claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScheduleMode {
    /// Keep every system's work on the stepping thread.
    Serial,
    #[default]
    /// Let systems fan safe internal work across the job pool.
    Parallel,
}

impl ScheduleMode {
    /// The mode a world runs under: the published resource, or `Serial` when
    /// nothing published one.
    pub fn current(resources: &crate::ecs::Resources) -> ScheduleMode {
        resources
            .get::<ScheduleMode>()
            .copied()
            .unwrap_or(ScheduleMode::Serial)
    }
}
