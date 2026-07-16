use super::*;

impl StorySystem {
    // Ordinary page mode: mid-story, no menu, no overlay, stage in front.
    pub(super) fn page_mode(&self) -> bool {
        self.started
            && !self.in_choice
            && self.overlay == Overlay::None
            && self.active_screen == self.ids.as_ref().map(|i| i.screen)
    }

    pub(super) fn toggle_auto(&mut self, ctx: &mut PipelineContext) {
        if !self.page_mode() {
            return;
        }
        self.auto = !self.auto;
        self.mode_timer = 0.0;
        self.render_quick_row(ctx);
    }

    pub(super) fn toggle_skip(&mut self, ctx: &mut PipelineContext) {
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

    // Momentary fast-forward while the skip modifier (Control) is held in page
    // mode: behaves like the Skip toggle, but only for as long as the key is
    // down. A choice menu or overlay leaves page mode, so it stops there ("until
    // a choice") and resumes if the key is still held once page mode returns.
    pub(super) fn update_hold_skip(&mut self, frame: &FrameInput, ctx: &mut PipelineContext) {
        let want = frame.ctrl && self.page_mode();
        if want == self.hold_skip {
            return;
        }
        self.hold_skip = want;
        self.mode_timer = 0.0;
        // Only touch page furniture while page mode owns the screen: a
        // transition caused by leaving page mode (into a choice / overlay) must
        // not repaint the quick row over it.
        if self.page_mode() {
            if want && !self.typewriter.done() {
                self.typewriter.shown = self.typewriter.full.len();
                let text = self.typewriter.text();
                let text_id = self.ids.as_ref().expect("resolved at init").text;
                set_label(ctx, text_id, |l| l.content = text);
            }
            self.render_quick_row(ctx);
        }
    }

    // Whether a skip run is active (the toggle or the held modifier).
    pub(super) fn skipping(&self) -> bool {
        self.skip || self.hold_skip
    }

    // Per-frame reader-assist work: the waiting marker pulse and the auto /
    // skip page pacing.
    pub(super) fn tick_modes(&mut self, ctx: &mut PipelineContext, dt: f32) {
        if !self.page_mode() {
            return;
        }
        let skipping = self.skipping();
        let waiting = self.typewriter.done();
        let marker = self.ids.as_ref().and_then(|i| i.marker);
        let alpha = if waiting && !skipping {
            0.35 + 0.3 * (self.elapsed * 5.0).sin()
        } else {
            0.0
        };
        set_sprite(ctx, marker, |s| s.tint[3] = alpha);
        if skipping {
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
    pub(super) fn tick_typewriter(&mut self, ctx: &mut PipelineContext, dt: f32) {
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
