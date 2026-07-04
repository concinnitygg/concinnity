use super::*;

impl StorySystem {
    pub(super) fn start(&mut self, ctx: &mut PipelineContext) {
        self.started = true;
        self.vars.clear();
        self.history.clear();
        self.auto = false;
        self.skip = false;
        self.hold_skip = false;
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
    pub(super) fn continue_story(&mut self, ctx: &mut PipelineContext) {
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
    pub(super) fn resume_from(&mut self, save: StorySave, ctx: &mut PipelineContext) {
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
    pub(super) fn current_save(&self) -> StorySave {
        StorySave {
            slug: self.story.nodes[self.node].slug.clone(),
            page: self.page as u32,
            vars: self.vars.iter().map(|(k, v)| (k.clone(), *v)).collect(),
        }
    }

    // Auto-save the current position and variables. The title menu reads the
    // save state on disk when it is next shown, so Continue appears from here
    // on without touching its label mid-play.
    pub(super) fn persist_position(&mut self, _ctx: &mut PipelineContext) {
        if self.story.save_key.is_empty() {
            return;
        }
        let save = self.current_save();
        if let Err(e) = write_save(&save_file(&self.save_dir, &self.story.save_key), &save) {
            tracing::warn!("StorySystem: save failed: {e}");
        }
    }

    // A finished story starts fresh next time: drop the auto-save (the title
    // menu drops Continue when it next reads disk). Slot saves stay.
    pub(super) fn clear_save(&mut self, _ctx: &mut PipelineContext) {
        if self.story.save_key.is_empty() {
            return;
        }
        let _ = std::fs::remove_file(save_file(&self.save_dir, &self.story.save_key));
    }

    // Whether any manual slot save exists (the title's Load lights up). Scans
    // every logical slot, not just the overlay's visible window.
    pub(super) fn any_slot_save(&self) -> bool {
        !self.story.save_key.is_empty()
            && (0..SLOT_COUNT).any(|i| slot_file(&self.save_dir, &self.story.save_key, i).exists())
    }
}
