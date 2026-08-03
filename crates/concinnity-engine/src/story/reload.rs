use super::*;

impl StorySystem {
    // Swap in a freshly compiled graph (the editor re-expands the story when
    // its Markdown source is saved), keeping the current position and raised
    // flags so the edit lands in place in the running game. Position is
    // matched by node slug; a deleted node restarts the story.
    pub(super) fn reload(&mut self, new: Story, ctx: &mut PipelineContext) {
        // A multi-story world reloads every story; only ours applies.
        if new.scaffold.screen != self.story.scaffold.screen {
            return;
        }
        // The re-rendered page would draw over an open overlay's dim.
        if self.overlay != Overlay::None {
            self.close_overlay(ctx);
        }
        let slug = self.story.nodes.get(self.node).map(|n| n.slug.clone());
        let was_in_choice = self.in_choice;
        self.story = new;
        // Refresh the stage references: the scaffold's option-slot count can
        // change with the story's widest menu (slots the running world never
        // declared are silent no-ops until a restart).
        if let Some(ids) = StageIds::from_scaffold(&self.story.scaffold) {
            self.ids = Some(ids);
        }
        if !self.started {
            return;
        }
        let node = slug.and_then(|s| self.story.nodes.iter().position(|n| n.slug == s));
        let Some(node) = node else {
            self.start(ctx);
            return;
        };
        self.node = node;
        let has_choices = !self.story.nodes[node].choices.is_empty();
        let page_count = self.story.nodes[node].pages.len();
        if was_in_choice && has_choices && !self.visible_choices(node).is_empty() {
            self.render_choice(ctx);
        } else if page_count > 0 {
            self.exit_choice_ui(ctx);
            self.page = self.page.min(page_count - 1);
            self.render_page(ctx);
            // Editing flow: show the whole revised page at once rather than
            // re-typing it out.
            self.reveal_all(ctx);
        } else {
            // The node lost its pages (and any open menu no longer applies):
            // re-enter it fresh so gates and fall-through resolve.
            self.exit_choice_ui(ctx);
            self.enter_node(node, ctx);
            return;
        }
        self.persist_position(ctx);
    }

    pub(super) fn show_ending(&mut self, ctx: &mut PipelineContext) {
        self.clear_save(ctx);
        self.auto = false;
        self.skip = false;
        let ids = self.ids.as_ref().expect("resolved at init");
        ctx.events_mut::<ScreenCommand>()
            .send(ScreenCommand::Show(ids.ending_screen));
    }
}
