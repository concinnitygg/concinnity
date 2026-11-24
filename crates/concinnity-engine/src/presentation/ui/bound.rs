// Labels, buttons, and slider handles that show a bound value, kept current as the value
// changes. They are rewritten only when the state, variables or settings they
// read have moved on, and only where they differ. Labels and buttons added
// after the first frame are found the frame they appear.

use alloc::string::String;
use alloc::vec::Vec;

use crate::components::{
    Binding, Button, HitRegion, Sprite, TextLabel, UiState, UiValue, VariableValues,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{Entity, PipelineContext, Tick, by_asset_id};
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
    // Each button captioned by a binding.
    buttons: Vec<(Entity, Binding)>,
    handles: Vec<SliderHandle>,
    seen: Option<Revisions>,
    // The label and button columns' last additions when they were scanned.
    scanned: Option<(Tick, Tick)>,
    // A label's text, built here and compared before it is written.
    text: String,
}

impl Bound {
    pub(super) fn collect(ctx: &PipelineContext<'_>, handles: Vec<SliderHandle>) -> Self {
        let mut bound = Self {
            handles,
            ..Self::default()
        };
        bound.scan(ctx);
        bound
    }

    // The label and button columns' last additions.
    fn additions(ctx: &PipelineContext<'_>) -> (Tick, Tick) {
        (
            ctx.column_ticks::<TextLabel>().added,
            ctx.column_ticks::<Button>().added,
        )
    }

    // Find every label and button showing a binding.
    fn scan(&mut self, ctx: &PipelineContext<'_>) {
        self.labels = ctx
            .query_with_entity::<TextLabel>()
            .filter_map(|(entity, label)| Some((entity, label.text_binding.clone()?)))
            .collect();
        self.buttons = ctx
            .query_with_entity::<Button>()
            .filter_map(|(entity, button)| Some((entity, button.text_binding.clone()?)))
            .collect();
        self.scanned = Some(Self::additions(ctx));
        self.seen = None;
    }

    pub(super) fn refresh(&mut self, ctx: &mut PipelineContext<'_>) {
        if self.scanned.is_some_and(|s| s != Self::additions(ctx)) {
            self.scan(ctx);
        }
        if self.labels.is_empty() && self.buttons.is_empty() && self.handles.is_empty() {
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
            if !Values::of(ctx).write_text(binding, &mut self.text) {
                continue;
            }
            if let Some(label) = ctx.get_mut::<TextLabel>(*entity)
                && label.content != self.text
            {
                label.content.clear();
                label.content.push_str(&self.text);
            }
        }
        for (entity, binding) in &self.buttons {
            if !Values::of(ctx).write_text(binding, &mut self.text) {
                continue;
            }
            if let Some(button) = ctx.get_mut::<Button>(*entity)
                && button.label != self.text
            {
                button.label.clear();
                button.label.push_str(&self.text);
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

// The places a binding's value lives, read together: the app's UI state, the
// world's variables, and the settings' shown values.
#[derive(Clone, Copy, Default)]
pub(crate) struct Values<'a> {
    pub(crate) state: Option<&'a UiState>,
    pub(crate) vars: Option<&'a VariableValues>,
    pub(crate) settings: Option<&'a SettingValues>,
}

impl<'a> Values<'a> {
    pub(crate) fn of(ctx: &'a PipelineContext<'_>) -> Self {
        Self {
            state: ctx.resource::<UiState>(),
            vars: ctx.resource::<VariableValues>(),
            settings: ctx.resource::<SettingValues>(),
        }
    }

    // The state or variable value `binding` names.
    pub(crate) fn value(self, binding: &Binding) -> Option<&'a UiValue> {
        match binding {
            Binding::State(name) => self.state?.get(name),
            Binding::Var(name) => self.vars?.values.get(name),
            Binding::Setting(_) => None,
        }
    }

    // Write what `binding` shows into `text`; false when it has no value yet.
    pub(crate) fn write_text(self, binding: &Binding, text: &mut String) -> bool {
        if let Binding::Setting(key) = binding {
            let Some(shown) = self.settings.and_then(|v| v.text(*key)) else {
                return false;
            };
            text.clear();
            text.push_str(shown);
            return true;
        }
        let Some(value) = self.value(binding) else {
            return false;
        };
        value.write_display(text);
        true
    }

    // Where `binding`'s value sits along a slider, 0 to 1.
    pub(crate) fn fraction(self, binding: &Binding) -> Option<f32> {
        let f = match binding {
            Binding::Setting(key) => self.settings?.fraction(*key)?,
            _ => match self.value(binding)? {
                UiValue::Float(f) => *f,
                _ => return None,
            },
        };
        Some(f.clamp(0.0, 1.0))
    }

    // The number `binding` holds, whole or not, if it holds one.
    pub(crate) fn number(self, binding: &Binding) -> Option<f32> {
        match self.value(binding)? {
            UiValue::Float(f) => Some(*f),
            UiValue::Int(i) => Some(*i as f32),
            _ => None,
        }
    }

    // The whole number `binding` holds, if it holds one.
    pub(crate) fn int(self, binding: &Binding) -> Option<i32> {
        match self.value(binding)? {
            UiValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    // The boolean `binding` holds, if it holds one: a setting reads as true at
    // the second of its two options.
    pub(crate) fn bool(self, binding: &Binding) -> Option<bool> {
        if let Binding::Setting(key) = binding {
            let shown = self.settings?.text(*key)?;
            let options = crate::settings::options(*key)?;
            return options.iter().position(|o| *o == shown).map(|i| i == 1);
        }
        match self.value(binding)? {
            UiValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    // The text `binding` holds, if it holds text.
    pub(crate) fn text(self, binding: &Binding) -> Option<&'a str> {
        match self.value(binding)? {
            UiValue::Text(t) => Some(t),
            _ => None,
        }
    }
}

// Where `binding`'s value sits along a slider, 0 to 1.
pub(super) fn fraction(ctx: &PipelineContext<'_>, binding: &Binding) -> Option<f32> {
    Values::of(ctx).fraction(binding)
}

// The number `binding` holds, whole or not, if it holds one.
pub(crate) fn number_of(ctx: &PipelineContext<'_>, binding: &Binding) -> Option<f32> {
    Values::of(ctx).number(binding)
}

// The whole number `binding` holds, if it holds one.
pub(crate) fn int_of(ctx: &PipelineContext<'_>, binding: &Binding) -> Option<i32> {
    Values::of(ctx).int(binding)
}

// The boolean `binding` holds, if it holds one.
pub(crate) fn bool_of(ctx: &PipelineContext<'_>, binding: &Binding) -> Option<bool> {
    Values::of(ctx).bool(binding)
}
