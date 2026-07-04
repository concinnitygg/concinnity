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
        if self.page_mode() && !self.story.save_key.is_empty() {
            self.open_slots(ctx, false);
        }
    }

    // The title screen's Load: bring the (dimmed) stage up over whatever the
    // title left behind and offer the slots. Also reachable nowhere else;
    // mid-story loading goes through the same slots after a Save.
    pub(super) fn open_load(&mut self, ctx: &mut PipelineContext) {
        if self.overlay != Overlay::None || self.story.save_key.is_empty() {
            return;
        }
        if self.started && !self.page_mode() {
            return;
        }
        if !self.started {
            // The overlay is set before the view change lands, so the
            // stage's ViewShown does not auto-start the story.
            let view = self.ids.as_ref().expect("resolved at init").view;
            ctx.events_mut::<ViewCommand>()
                .send(ViewCommand::Show(view));
        }
        self.open_slots(ctx, true);
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
