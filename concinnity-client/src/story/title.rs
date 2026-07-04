use super::*;

impl StorySystem {
    // Lay out the title screen menu from the save state on disk: Start and Quit
    // always, Continue only when an auto-save exists, Load only when a slot
    // does. The applicable buttons stack contiguously and centered, so an
    // absent one leaves no gap; each button's hit region follows its label
    // (and goes inert while the label is empty), so a hidden button neither
    // shows nor catches clicks. Runs at init and whenever the title is shown.
    pub(super) fn layout_title_menu(&mut self, ctx: &mut PipelineContext) {
        let (title_view, start, cont, load, quit) = match self.ids.as_ref() {
            Some(ids) => (
                ids.title_view,
                ids.start_label,
                ids.continue_label,
                ids.load_label,
                ids.quit_label,
            ),
            None => return,
        };
        if title_view.is_none() {
            return;
        }
        let has_save = !self.story.save_key.is_empty()
            && read_save(&save_file(&self.save_dir, &self.story.save_key)).is_some();
        let has_slots = self.any_slot_save();

        let mut buttons: Vec<(Option<AssetId>, &str)> = vec![(start, "Start")];
        if has_save {
            buttons.push((cont, "Continue"));
        }
        if has_slots {
            buttons.push((load, "Load"));
        }
        buttons.push((quit, "Quit"));

        let n = buttons.len() as f32;
        let top = TITLE_MENU_CENTER_Y - (n - 1.0) * TITLE_MENU_SPACING / 2.0;
        for (i, (id, text)) in buttons.into_iter().enumerate() {
            let y = top + i as f32 * TITLE_MENU_SPACING;
            let text = text.to_string();
            set_label(ctx, id, |l| {
                l.content = text;
                l.y = y;
            });
        }
        // Clear whichever optional buttons are absent this time so their
        // follow-regions go inert (an empty label renders nothing).
        if !has_save {
            set_label(ctx, cont, |l| l.content.clear());
        }
        if !has_slots {
            set_label(ctx, load, |l| l.content.clear());
        }
    }
}
