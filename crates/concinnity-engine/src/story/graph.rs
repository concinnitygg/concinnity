use super::*;

impl StorySystem {
    // Apply a run of variable operations.
    pub(super) fn apply_ops(&mut self, ops: &[StoryOp]) {
        for op in ops {
            let slot = self.vars.entry(op.name.clone()).or_insert(0);
            if op.add {
                *slot = slot.saturating_add(op.value);
            } else {
                *slot = op.value;
            }
        }
    }

    pub(super) fn cond_passes(&self, name: &str, op: StoryCompareOp, value: i32) -> bool {
        op.eval(self.vars.get(name).copied().unwrap_or(0), value)
    }

    // The first gate whose condition passes, if any: its target node.
    pub(super) fn passing_gate(&self, gates: &[StoryGate]) -> Option<usize> {
        gates
            .iter()
            .find(|g| self.cond_passes(&g.name, g.op, g.value))
            .map(|g| g.target as usize)
    }

    // The node's choice indices whose conditions pass right now.
    pub(super) fn visible_choices(&self, node: usize) -> Vec<usize> {
        self.story.nodes[node]
            .choices
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                c.condition
                    .as_ref()
                    .is_none_or(|cond| self.cond_passes(&cond.name, cond.op, cond.value))
            })
            .map(|(i, _)| i)
            .collect()
    }

    // Move play to a node: its first page, or straight to its choice menu
    // when it has no pages. A node with neither falls through in document
    // order; running past the last node ends the story. Gates on the arrived
    // page (or menu) redirect first; the hop budget stops a gate cycle from
    // spinning forever.
    pub(super) fn enter_node(&mut self, index: usize, ctx: &mut PipelineContext) {
        let mut index = index;
        let mut hops = 0;
        loop {
            hops += 1;
            if hops > 64 {
                tracing::warn!("StorySystem: story gates form a loop; stopping");
                return;
            }
            let Some(node) = self.story.nodes.get(index) else {
                self.show_ending(ctx);
                return;
            };
            if !node.pages.is_empty() {
                if let Some(target) = self.passing_gate(&node.pages[0].gates) {
                    index = target;
                    continue;
                }
                self.node = index;
                self.page = 0;
                self.apply_page(ctx);
                return;
            }
            if !node.choices.is_empty() {
                if let Some(target) = self.passing_gate(&node.choice_gates) {
                    index = target;
                    continue;
                }
                if !self.visible_choices(index).is_empty() {
                    self.node = index;
                    self.enter_choice(ctx);
                    return;
                }
                // Every option is gated off: fall through like a menu-less
                // node.
            }
            index += 1;
        }
    }
}
