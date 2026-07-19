// src/gfx/input_system.rs
//
// Samples the window backend's input once per frame and publishes the
// `FrameInput` snapshot (as a resource and as the component column) plus the
// `CursorState` resource. Runs immediately after GraphicsSystem in the
// schedule: on Metal the OS event pump runs inside draw_frame, so sampling
// right after the draw snapshots every event that arrived up to and including
// this frame's pump -- the same freshness the sample had when it sat at the
// end of GraphicsSystem's own step. Every input consumer (camera controllers,
// UI, text input) runs later in the same tick.

use crate::assets::FrameInput;
use crate::ecs::{ActiveRenderBackend, PipelineContext, StepResult, System};

#[derive(Debug, Default)]
pub struct InputSystem;

impl InputSystem {
    pub fn new() -> Self {
        Self
    }
}

impl System for InputSystem {
    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        // No parked backend (graphics failed, or the editor transplanted it
        // away): nothing to sample; consumers keep the last snapshot.
        let Some(mut backend) = ActiveRenderBackend::take(ctx.resources) else {
            return StepResult::Continue;
        };
        let raw = backend.take_input();
        let cursor_outside = backend.cursor_outside_window();
        // Live viewport for UiInputSystem's overlay hit-testing, so a scaled
        // menu's HitRegions map back to the cursor consistently.
        let (vp_w, vp_h) = backend.logical_size();
        ActiveRenderBackend::put(ctx.resources, backend);

        // The cursor position + window-bounds state for next frame's draw
        // list (`follow_cursor` sprites are positioned a frame after the input
        // that moved them; the draw list is built before this poll).
        ctx.insert_resource(crate::ecs::CursorState {
            pos: (raw.mouse_x, raw.mouse_y),
            outside_window: cursor_outside,
        });

        // While a world-pausing screen is open (the overlay build published
        // the state earlier this tick), freeze gameplay input so the camera
        // does not drift behind the menu; the UI still gets the cursor
        // position, clicks, and Escape. A non-pausing screen that captures
        // input (a live console) also suppresses gameplay keys -- the world
        // keeps simulating, but keystrokes belong to the screen.
        let menu_active = ctx
            .resource::<crate::ecs::MenuActive>()
            .map(|m| m.0)
            .unwrap_or(false);
        let screen_captures = ctx
            .resource::<crate::ecs::ScreenStack>()
            .is_some_and(|s| s.captures_input);
        // The editor's fly camera keeps navigation live while its menu
        // override freezes the world (the editor integrates the camera
        // itself); a shipped runtime never publishes FlyCam.
        let fly = ctx.resource::<crate::ecs::FlyCam>().is_some_and(|f| f.0);
        let gameplay = (!menu_active && !screen_captures) || fly;

        // Both readers query (not drain) the snapshot, so clear the previous
        // frame's first.
        let _ = ctx.drain::<FrameInput>();
        let frame_input = FrameInput {
            forward: raw.forward && gameplay,
            backward: raw.backward && gameplay,
            left: raw.left && gameplay,
            right: raw.right && gameplay,
            sprint: raw.sprint && gameplay,
            interact: raw.interact && gameplay,
            jump: raw.jump && gameplay,
            mouse_dx: if gameplay { raw.mouse_dx } else { 0.0 },
            mouse_dy: if gameplay { raw.mouse_dy } else { 0.0 },
            // Not gated by `gameplay`: a scrollable menu still scrolls
            // while it is open (the camera is what freezes behind it).
            scroll_delta: raw.scroll_delta,
            mouse_x: raw.mouse_x,
            mouse_y: raw.mouse_y,
            left_click: raw.left_click,
            left_button_down: raw.left_button_down,
            viewport: [vp_w, vp_h],
            hud_toggle: raw.hud_toggle,
            escape: raw.escape,
            // Not gated by `gameplay`: a story's Ctrl fast-forward works
            // while its stage (a view) is up, like the rebind capture below.
            ctrl: raw.ctrl,
            // Not gated by `gameplay`: the rebind capture works while the
            // settings menu is open (the camera is what freezes behind it).
            captured_key: raw.captured_key,
            // Not gated by `gameplay`: text-input fields type while a menu
            // (or the in-engine editor) is up, like the rebind capture.
            typed_char: raw.typed_char,
        };
        // Publish the same snapshot two ways: the resource readers can
        // fetch by type, and the component column the camera and UI
        // systems still drain/query.
        ctx.insert_resource(frame_input.clone());
        ctx.push(frame_input);

        StepResult::Continue
    }
}
