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
        let Some(screen) = self.ids.as_ref().map(|i| i.screen) else {
            return;
        };
        if self.active_screen != Some(screen) {
            ctx.events_mut::<ScreenCommand>()
                .send(ScreenCommand::Show(screen));
        }
        self.enter_node(0, ctx);
    }

    // Resume from the auto-saved position; a missing, stale, or unreadable
    // save starts fresh.
    pub(super) fn continue_story(&mut self, ctx: &mut PipelineContext) {
        match self.read_saved(SaveSlot::Auto) {
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
        self.vars = save.vars;
        self.history.clear();
        self.overlay = Overlay::None;
        self.exit_choice_ui(ctx);
        let Some(screen) = self.ids.as_ref().map(|i| i.screen) else {
            return;
        };
        if self.active_screen != Some(screen) {
            ctx.events_mut::<ScreenCommand>()
                .send(ScreenCommand::Show(screen));
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
            vars: self.vars.clone(),
        }
    }

    // Auto-save the current position and variables. The title menu reads the
    // save state when it is next shown, so Continue appears from here on
    // without touching its label mid-play.
    pub(super) fn persist_position(&self) {
        self.write_saved(SaveSlot::Auto);
    }

    // A finished story starts fresh next time: drop the auto-save (the title
    // menu drops Continue when it next reads the store). Slot saves stay.
    pub(super) fn clear_save(&self) {
        if let Some(store) = self.saving() {
            store.remove(SaveSlot::Auto);
        }
    }

    // Whether any manual slot save exists (the title's Load lights up). Scans
    // every logical slot, not just the overlay's visible window.
    pub(super) fn any_slot_save(&self) -> bool {
        self.saving()
            .is_some_and(|store| (0..SLOT_COUNT).any(|i| store.exists(SaveSlot::Manual(i))))
    }

    // Save the current position to `slot`.
    pub(super) fn write_saved(&self, slot: SaveSlot) {
        if let Some(store) = self.saving() {
            store.write(slot, &self.current_save());
        }
    }

    // The save in `slot`.
    pub(super) fn read_saved(&self, slot: SaveSlot) -> Option<StorySave> {
        self.saving()?.read(slot)
    }

    // The store, when this story saves at all (it declares a save key).
    fn saving(&self) -> Option<&dyn StoryStore> {
        if self.story.save_key.is_empty() {
            return None;
        }
        self.store.as_deref()
    }
}
