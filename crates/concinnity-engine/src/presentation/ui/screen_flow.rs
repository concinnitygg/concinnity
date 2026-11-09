// The menu's screen stack: applying screen commands, the transitions between
// screens, and the visibility each change leaves.

use super::{UiInputSystem, screen};
use crate::components::{ScreenCommand, ScreenShown, Sprite, TextInput, TextLabel};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, by_asset_id};
use crate::protocol::ScreenStack;
use alloc::vec::Vec;

impl UiInputSystem {
    // Apply the ScreenCommands queued since last frame, ahead of this frame's
    // hit-testing. They are cloned out to release the ctx borrow for the apply.
    pub(super) fn apply_pending_screen_commands(&mut self, ctx: &mut PipelineContext<'_>) {
        let screen_cmds: Vec<ScreenCommand> = match ctx.events::<ScreenCommand>() {
            Some(events) => events.read(&mut self.screen_cmd_cursor).copied().collect(),
            None => Vec::new(),
        };
        for cmd in screen_cmds {
            self.apply_screen_command(cmd, ctx);
        }
    }

    fn apply_screen_command(&mut self, cmd: ScreenCommand, ctx: &mut PipelineContext<'_>) {
        let Some(transition) = self.screens.apply(cmd) else {
            return;
        };
        // A stack change dismisses any open dropdown and the focus cursor so
        // neither lingers over a different screen.
        self.open_dropdown = None;
        self.focus = None;
        self.apply_transition(&transition, ctx);
        self.publish_screen_stack(ctx);
    }

    // Enact one stack transition on the world: flip the entering / leaving
    // screens' element visibility, move keyboard focus to the new top's
    // `focus` field, and announce the newly-topmost screen (AudioCue).
    pub(super) fn apply_transition(
        &self,
        transition: &screen::ScreenTransition,
        ctx: &mut PipelineContext<'_>,
    ) {
        for id in &transition.hidden {
            self.set_screen_visibility(*id, false, ctx);
        }
        for id in &transition.shown {
            self.set_screen_visibility(*id, true, ctx);
        }
        if transition.top_changed {
            // Focus follows the top of the stack: blur every field, then focus
            // the new top's `focus` target (if it names one).
            let focus = transition
                .new_top
                .and_then(|id| self.screens.meta(id))
                .and_then(|m| m.focus);
            for ti in ctx.query_mut::<TextInput>() {
                ti.focused = false;
            }
            by_asset_id::update::<TextInput>(ctx, focus, |ti| ti.focused = true);
        }
        if let Some(top) = transition.new_top {
            ctx.events_mut::<ScreenShown>()
                .send(ScreenShown { screen: top });
        }
    }

    // Publish the stack's derived per-frame state (draw layers, world-pause,
    // input capture) for the overlay build and the input poll, which read it a
    // frame later, the same one-frame lag the visibility flips have.
    pub(super) fn publish_screen_stack(&self, ctx: &mut PipelineContext<'_>) {
        ctx.insert_resource(ScreenStack {
            layers: self.screens.layers(),
            pauses_world: self.screens.pauses_world(),
            captures_input: self.screens.captures_input(),
        });
    }

    fn set_screen_visibility(
        &self,
        screen_id: AssetId,
        visible: bool,
        ctx: &mut PipelineContext<'_>,
    ) {
        for &id in self.sprites_by_screen.get(&screen_id).into_iter().flatten() {
            by_asset_id::update::<Sprite>(ctx, Some(id), |s| s.visible = visible);
        }
        for &id in self.labels_by_screen.get(&screen_id).into_iter().flatten() {
            by_asset_id::update::<TextLabel>(ctx, Some(id), |l| l.visible = visible);
        }
        for &id in self
            .text_inputs_by_screen
            .get(&screen_id)
            .into_iter()
            .flatten()
        {
            by_asset_id::update::<TextInput>(ctx, Some(id), |ti| ti.visible = visible);
        }
    }
}
