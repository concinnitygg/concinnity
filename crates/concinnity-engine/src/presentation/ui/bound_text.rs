// Labels that show a bound value, kept current as the value changes. A label
// is rewritten only when the state or variables it reads have moved on, and
// only when its text differs.

use alloc::string::String;
use alloc::vec::Vec;

use crate::components::{Binding, TextLabel, UiState, VariableValues};
use crate::ecs::{Entity, PipelineContext};

#[derive(Debug, Default)]
pub(super) struct BoundText {
    // Each label showing a `state:` or `var:` binding.
    labels: Vec<(Entity, Binding)>,
    // The state and variable revisions the labels last showed.
    seen: Option<(Option<u64>, Option<u64>)>,
    // A label's text, built here and compared before it is written.
    text: String,
}

impl BoundText {
    pub(super) fn collect(ctx: &PipelineContext<'_>) -> Self {
        let labels = ctx
            .query_with_entity::<TextLabel>()
            .filter_map(|(entity, label)| match &label.text_binding {
                Some(b @ (Binding::State(_) | Binding::Var(_))) => Some((entity, b.clone())),
                _ => None,
            })
            .collect();
        Self {
            labels,
            seen: None,
            text: String::new(),
        }
    }

    pub(super) fn refresh(&mut self, ctx: &mut PipelineContext<'_>) {
        if self.labels.is_empty() {
            return;
        }
        let revisions = (
            ctx.resource::<UiState>().map(UiState::revision),
            ctx.resource::<VariableValues>().map(|v| v.revision),
        );
        if self.seen == Some(revisions) {
            return;
        }
        self.seen = Some(revisions);
        for (entity, binding) in &self.labels {
            let value = match binding {
                Binding::State(name) => ctx.resource::<UiState>().and_then(|s| s.get(name)),
                Binding::Var(name) => ctx
                    .resource::<VariableValues>()
                    .and_then(|v| v.values.get(name)),
                Binding::Setting(_) => None,
            };
            let Some(value) = value else {
                continue;
            };
            value.write_display(&mut self.text);
            if let Some(label) = ctx.get_mut::<TextLabel>(*entity)
                && label.content != self.text
            {
                label.content.clear();
                label.content.push_str(&self.text);
            }
        }
    }
}
