// Labels and slider handles that show a bound value, kept current as the value
// changes. They are rewritten only when the state, variables or settings they
// read have moved on, and only where they differ.

use alloc::string::String;
use alloc::vec::Vec;

use crate::components::{Binding, HitRegion, Sprite, TextLabel, UiState, UiValue, VariableValues};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{Entity, PipelineContext, by_asset_id};
use crate::settings::SettingValues;

// A slider region's handle sprite, placed along the region by its bound value.
#[derive(Debug)]
pub(super) struct SliderHandle {
    handle: AssetId,
    binding: Binding,
    // The handle's x at 0, and how far it travels to 1.
    x: f32,
    travel: f32,
}

impl SliderHandle {
    // The handle a slider region places, when it names one.
    pub(super) fn of(region: &HitRegion, ctx: &PipelineContext<'_>) -> Option<Self> {
        let binding = region.slider.clone()?;
        let handle = region.drag_handle?.id();
        let width = ctx.get_by_id::<Sprite>(handle).map_or(0.0, |s| s.width);
        Some(Self {
            handle,
            binding,
            x: region.x,
            travel: (region.width - width).max(0.0),
        })
    }
}

// The UI state, variable and setting revisions bound elements last showed.
type Revisions = (Option<u64>, Option<u64>, Option<u64>);

#[derive(Debug, Default)]
pub(super) struct Bound {
    // Each label showing a binding.
    labels: Vec<(Entity, Binding)>,
    handles: Vec<SliderHandle>,
    seen: Option<Revisions>,
    // A label's text, built here and compared before it is written.
    text: String,
}

impl Bound {
    pub(super) fn collect(ctx: &PipelineContext<'_>, handles: Vec<SliderHandle>) -> Self {
        let labels = ctx
            .query_with_entity::<TextLabel>()
            .filter_map(|(entity, label)| Some((entity, label.text_binding.clone()?)))
            .collect();
        Self {
            labels,
            handles,
            seen: None,
            text: String::new(),
        }
    }

    pub(super) fn refresh(&mut self, ctx: &mut PipelineContext<'_>) {
        if self.labels.is_empty() && self.handles.is_empty() {
            return;
        }
        let revisions = (
            ctx.resource::<UiState>().map(UiState::revision),
            ctx.resource::<VariableValues>().map(|v| v.revision),
            ctx.resource::<SettingValues>().map(SettingValues::revision),
        );
        if self.seen == Some(revisions) {
            return;
        }
        self.seen = Some(revisions);
        for (entity, binding) in &self.labels {
            if !write_text(ctx, binding, &mut self.text) {
                continue;
            }
            if let Some(label) = ctx.get_mut::<TextLabel>(*entity)
                && label.content != self.text
            {
                label.content.clear();
                label.content.push_str(&self.text);
            }
        }
        for h in &self.handles {
            let Some(fraction) = fraction(ctx, &h.binding) else {
                continue;
            };
            let x = h.x + fraction * h.travel;
            by_asset_id::update::<Sprite>(ctx, Some(h.handle), |s| s.x = x);
        }
    }
}

// Write what `binding` shows into `text`; false when it has no value yet.
fn write_text(ctx: &PipelineContext<'_>, binding: &Binding, text: &mut String) -> bool {
    if let Binding::Setting(key) = binding {
        let Some(shown) = ctx.resource::<SettingValues>().and_then(|v| v.text(*key)) else {
            return false;
        };
        text.clear();
        text.push_str(shown);
        return true;
    }
    let Some(value) = value(ctx, binding) else {
        return false;
    };
    value.write_display(text);
    true
}

// The state or variable value `binding` names.
fn value<'a>(ctx: &'a PipelineContext<'_>, binding: &Binding) -> Option<&'a UiValue> {
    match binding {
        Binding::State(name) => ctx.resource::<UiState>()?.get(name),
        Binding::Var(name) => ctx.resource::<VariableValues>()?.values.get(name),
        Binding::Setting(_) => None,
    }
}

// Where `binding`'s value sits along a slider, 0 to 1.
pub(super) fn fraction(ctx: &PipelineContext<'_>, binding: &Binding) -> Option<f32> {
    let f = match binding {
        Binding::Setting(key) => ctx.resource::<SettingValues>()?.fraction(*key)?,
        _ => match value(ctx, binding)? {
            UiValue::Float(f) => *f,
            _ => return None,
        },
    };
    Some(f.clamp(0.0, 1.0))
}
