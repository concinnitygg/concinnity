use super::*;

impl StorySystem {
    pub(super) fn advance(&mut self, ctx: &mut PipelineContext) {
        let Some(screen) = self.ids.as_ref().map(|i| i.screen) else {
            return;
        };
        if !self.started || self.in_choice || self.active_screen != Some(screen) {
            return;
        }
        // A click mid-reveal completes the page instead of leaving it.
        if !self.typewriter.done() {
            self.reveal_all(ctx);
            return;
        }
        let node = &self.story.nodes[self.node];
        let jump = node.pages[self.page].jump;
        let more_pages = self.page + 1 < node.pages.len();
        let has_choices = !node.choices.is_empty();
        // Gates on whatever comes next redirect before it shows.
        let next_redirect = if jump.is_none() && more_pages {
            self.passing_gate(&node.pages[self.page + 1].gates)
        } else if jump.is_none() && has_choices {
            self.passing_gate(&node.choice_gates)
        } else {
            None
        };
        if let Some(jump) = jump {
            self.enter_node(jump as usize, ctx);
        } else if let Some(target) = next_redirect {
            self.enter_node(target, ctx);
        } else if more_pages {
            self.page += 1;
            self.apply_page(ctx);
        } else if has_choices && !self.visible_choices(self.node).is_empty() {
            self.enter_choice(ctx);
        } else {
            self.enter_node(self.node + 1, ctx);
        }
    }

    pub(super) fn choose(&mut self, option: usize, ctx: &mut PipelineContext) {
        let Some(screen) = self.ids.as_ref().map(|i| i.screen) else {
            return;
        };
        if !self.started || !self.in_choice || self.active_screen != Some(screen) {
            return;
        }
        let Some(choice) = self
            .menu
            .get(option)
            .and_then(|&i| self.story.nodes[self.node].choices.get(i))
        else {
            return;
        };
        let target = choice.target as usize;
        self.exit_choice_ui(ctx);
        self.enter_node(target, ctx);
    }
}
