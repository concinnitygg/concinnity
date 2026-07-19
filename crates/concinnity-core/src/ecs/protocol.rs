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

// The editor's fly-camera state. While true (published only by the `cn
// editor` HUD drive), InputSystem keeps the navigation keys and mouse deltas
// live and GraphicsSystem captures the cursor even though the world is frozen
// behind the editor's menu override -- the editor integrates Camera3D itself,
// so the viewport can be flown without running the simulation. Absent / false
// in a shipped runtime.
#[derive(Debug, Clone, Copy, Default)]
pub struct FlyCam(pub bool);

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
