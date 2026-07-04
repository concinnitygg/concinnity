use super::*;

impl StorySystem {
    pub(super) fn toggle_log(&mut self, ctx: &mut PipelineContext) {
        match self.overlay {
            Overlay::Backlog => self.close_overlay(ctx),
            Overlay::None if self.page_mode() => self.open_backlog(ctx),
            _ => {}
        }
    }

    // Clear the page furniture that draws over the overlay dim (labels
    // always render above sprites, so anything left filled would float on
    // top of it).
    pub(super) fn hide_page_furniture(&mut self, ctx: &mut PipelineContext) {
        let (name, text) = {
            let ids = self.ids.as_ref().expect("resolved at init");
            (ids.name, ids.text)
        };
        set_label(ctx, name, |l| l.content.clear());
        set_label(ctx, text, |l| l.content.clear());
        self.clear_quick_row(ctx);
    }

    pub(super) fn open_backlog(&mut self, ctx: &mut PipelineContext) {
        self.overlay = Overlay::Backlog;
        self.hide_page_furniture(ctx);
        // The most recent entries that fit the label, oldest first.
        let mut chosen: Vec<&str> = Vec::new();
        let mut lines = 0;
        for entry in self.history.iter().rev() {
            let n = entry.lines().count().max(1);
            if !chosen.is_empty() && lines + n > BACKLOG_LINES {
                break;
            }
            chosen.push(entry.as_str());
            lines += n;
            if lines >= BACKLOG_LINES {
                break;
            }
        }
        chosen.reverse();
        let text = chosen.join("\n");
        let ids = self.ids.as_ref().expect("resolved at init");
        set_sprite(ctx, ids.overlay_dim, |s| s.tint[3] = OVERLAY_DIM_ALPHA);
        set_label(ctx, ids.backlog_label, |l| l.content = text);
    }

    pub(super) fn open_save(&mut self, ctx: &mut PipelineContext) {
        if self.story.save_key.is_empty() {
            return;
        }
        if self.page_mode() {
            self.open_slots(ctx, false);
        } else if self.menu_over_story() {
            // Saving from a menu shown over the story (the injected pause menu):
            // the stage is hidden behind the menu view, so raise it first, then
            // open the slot overlay (stage-view furniture) on top of it.
            self.raise_stage(ctx);
            self.open_slots(ctx, false);
        }
    }

    // Load offers the slot overlay from the title screen (before play), from
    // ordinary page mode, and from a menu shown over the story (the pause
    // menu). In the title and pause cases the stage is not the active view, so
    // it is raised first; the overlay is set (in `open_slots`) before that view
    // change lands, so the stage's `ViewShown` does not auto-start the story.
    pub(super) fn open_load(&mut self, ctx: &mut PipelineContext) {
        if self.overlay != Overlay::None || self.story.save_key.is_empty() {
            return;
        }
        if self.started {
            if self.page_mode() {
                // Already on the stage; the slots open in place.
            } else if self.menu_over_story() {
                self.raise_stage(ctx);
            } else {
                // A choice menu or the ending is up: loading is not offered.
                return;
            }
        } else {
            self.raise_stage(ctx);
        }
        self.open_slots(ctx, true);
    }

    // A view other than the story's own (the injected pause menu, say) is up
    // over a started, non-choice story with no overlay: the stage is hidden
    // behind it. Distinct from `page_mode`, which requires the stage itself to
    // be active.
    fn menu_over_story(&self) -> bool {
        let Some(ids) = self.ids.as_ref() else {
            return false;
        };
        self.started
            && !self.in_choice
            && self.overlay == Overlay::None
            && self.active_view.is_some()
            && self.active_view != Some(ids.view)
            && self.active_view != Some(ids.ending_view)
            && self.active_view != ids.title_view
    }

    // Bring the stage view forward (front and visible). The command lands next
    // frame; any furniture set this frame keeps its content when the view
    // re-activates.
    fn raise_stage(&mut self, ctx: &mut PipelineContext) {
        let view = self.ids.as_ref().expect("resolved at init").view;
        ctx.events_mut::<ViewCommand>()
            .send(ViewCommand::Show(view));
    }

    // Toggle the pause menu (the Escape binding and the Resume button). Views
    // are single-slot, so showing the pause hides the stage; closing must show
    // the stage explicitly rather than dismiss to no view (which renders
    // nothing). An open backlog / slot overlay is dismissed first, so Escape
    // steps back out one level at a time.
    pub(super) fn toggle_pause(&mut self, ctx: &mut PipelineContext) {
        if self.overlay != Overlay::None {
            self.close_overlay(ctx);
            return;
        }
        let (stage, pause) = {
            let Some(ids) = self.ids.as_ref() else {
                return;
            };
            (ids.view, ids.pause_view)
        };
        let Some(pause) = pause else { return };
        if self.active_view == Some(pause) {
            // Close: return to the stage the menu was opened over.
            ctx.events_mut::<ViewCommand>()
                .send(ViewCommand::Show(stage));
        } else if self.active_view == Some(stage) {
            // Open the pause menu over the stage.
            ctx.events_mut::<ViewCommand>()
                .send(ViewCommand::Show(pause));
        }
        // On the title / settings / ending screens Escape does nothing: those
        // navigate through their own buttons.
    }

    // Open the settings screen (the pause menu's or the title's Settings item),
    // remembering the menu that opened it so Back returns there.
    pub(super) fn open_settings(&mut self, ctx: &mut PipelineContext) {
        let settings = self.ids.as_ref().and_then(|i| i.settings_view);
        if let Some(settings) = settings {
            self.settings_return = self.active_view;
            ctx.events_mut::<ViewCommand>()
                .send(ViewCommand::Show(settings));
        }
    }

    // Close the settings screen, returning to the menu that opened it (falling
    // back to the title, then the stage).
    pub(super) fn close_settings(&mut self, ctx: &mut PipelineContext) {
        let target = self
            .settings_return
            .or_else(|| self.ids.as_ref().and_then(|i| i.title_view))
            .or_else(|| self.ids.as_ref().map(|i| i.view));
        if let Some(target) = target {
            ctx.events_mut::<ViewCommand>()
                .send(ViewCommand::Show(target));
        }
    }

    pub(super) fn open_slots(&mut self, ctx: &mut PipelineContext, load: bool) {
        self.overlay = if load {
            Overlay::LoadMenu
        } else {
            Overlay::SaveMenu
        };
        if self.started {
            self.hide_page_furniture(ctx);
        }
        let (dim, title, boxes, labels) = {
            let ids = self.ids.as_ref().expect("resolved at init");
            (
                ids.overlay_dim,
                ids.slot_title,
                ids.slot_boxes.clone(),
                ids.slot_labels.clone(),
            )
        };
        set_sprite(ctx, dim, |s| s.tint[3] = OVERLAY_DIM_ALPHA);
        set_label(ctx, title, |l| {
            l.content = (if load { "Load" } else { "Save" }).to_string();
        });
        for (i, box_id) in boxes.iter().enumerate() {
            let summary = self.slot_summary(i);
            set_sprite(ctx, Some(*box_id), |s| {
                s.visible = true;
                s.tint = CHOICE_BOX_TINT;
            });
            set_label(ctx, labels.get(i).copied(), |l| {
                l.content = summary;
                l.visible = true;
            });
        }
    }

    pub(super) fn slot_summary(&self, slot: usize) -> String {
        match read_save(&slot_file(&self.save_dir, &self.story.save_key, slot)) {
            Some(save) => format!("Slot {}   {}, page {}", slot + 1, save.slug, save.page + 1),
            None => format!("Slot {}   (empty)", slot + 1),
        }
    }

    pub(super) fn pick_slot(&mut self, slot: usize, ctx: &mut PipelineContext) {
        match self.overlay {
            Overlay::SaveMenu => {
                let save = self.current_save();
                let path = slot_file(&self.save_dir, &self.story.save_key, slot);
                if let Err(e) = write_save(&path, &save) {
                    tracing::warn!("StorySystem: slot save failed: {e}");
                }
                // The title menu picks up the new slot (its Load button) when
                // it is next shown.
                self.close_overlay(ctx);
            }
            Overlay::LoadMenu => {
                // Picking an empty slot leaves the overlay up.
                let path = slot_file(&self.save_dir, &self.story.save_key, slot);
                let Some(save) = read_save(&path) else { return };
                self.hide_overlay_furniture(ctx);
                self.overlay = Overlay::None;
                self.resume_from(save, ctx);
            }
            _ => {}
        }
    }

    pub(super) fn hide_overlay_furniture(&mut self, ctx: &mut PipelineContext) {
        let (dim, backlog, title, boxes, labels) = {
            let ids = self.ids.as_ref().expect("resolved at init");
            (
                ids.overlay_dim,
                ids.backlog_label,
                ids.slot_title,
                ids.slot_boxes.clone(),
                ids.slot_labels.clone(),
            )
        };
        set_sprite(ctx, dim, |s| s.tint[3] = 0.0);
        set_label(ctx, backlog, |l| l.content.clear());
        set_label(ctx, title, |l| l.content.clear());
        for box_id in boxes {
            set_sprite(ctx, Some(box_id), |s| s.tint[3] = 0.0);
        }
        for label_id in labels {
            set_label(ctx, Some(label_id), |l| l.content.clear());
        }
    }

    // Dismiss the open overlay: back to the page it covered, or back to the
    // title screen when the load overlay was opened from there.
    pub(super) fn close_overlay(&mut self, ctx: &mut PipelineContext) {
        self.hide_overlay_furniture(ctx);
        self.overlay = Overlay::None;
        if !self.started {
            let title = self.ids.as_ref().and_then(|i| i.title_view);
            if let Some(title) = title {
                ctx.events_mut::<ViewCommand>()
                    .send(ViewCommand::Show(title));
            }
            return;
        }
        // Overlays open from page mode only; restore the covered page in
        // full (it was already read, so no re-typing).
        self.render_page(ctx);
        self.typewriter.shown = self.typewriter.full.len();
        let text = self.typewriter.text();
        let text_id = self.ids.as_ref().expect("resolved at init").text;
        set_label(ctx, text_id, |l| l.content = text);
    }
}
