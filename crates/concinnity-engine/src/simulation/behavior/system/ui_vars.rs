// The world variables as UI bindings see them: changes a `var:` action sends
// in, and the values a `var:` label reads back out.

use alloc::collections::BTreeMap;
use alloc::string::String;
use core::fmt::Write as _;

use super::BehaviorSystem;
use crate::components::{UiValue, VariableCommand, VariableValues};
use crate::ecs::PipelineContext;
use crate::simulation::behavior::Val;

// A variable's value as a binding holds it, `None` for an entity handle.
fn ui_value(val: Val) -> Option<UiValue> {
    match val {
        Val::Bool(b) => Some(UiValue::Bool(b)),
        Val::Int(i) => Some(UiValue::Int(i)),
        Val::Float(f) => Some(UiValue::Float(f)),
        Val::Vec3([x, y, z]) => Some(UiValue::Text(alloc::format!("{x}, {y}, {z}"))),
        Val::Entity(_) => None,
    }
}

// Rewrite `slot` to hold `val`, reusing a text slot's buffer for a vector.
// An entity handle leaves it as it is.
#[expect(
    clippy::let_underscore_must_use,
    reason = "writing to a String cannot fail"
)]
fn write_value(val: Val, slot: &mut UiValue) {
    match (val, slot) {
        (Val::Vec3([x, y, z]), UiValue::Text(text)) => {
            text.clear();
            let _ = write!(text, "{x}, {y}, {z}");
        }
        (Val::Entity(_), _) => {}
        (val, slot) => {
            if let Some(value) = ui_value(val) {
                *slot = value;
            }
        }
    }
}

// The variable value `value` sets a variable holding `current` to, when the
// two are the same type.
fn val_of(value: &UiValue, current: Val) -> Option<Val> {
    let next = match value {
        UiValue::Bool(b) => Val::Bool(*b),
        UiValue::Int(i) => Val::Int(*i),
        UiValue::Float(f) => Val::Float(*f),
        UiValue::Text(_) => return None,
    };
    current.same_type(next).then_some(next)
}

impl BehaviorSystem {
    // Apply the variable changes UI actions sent since the last step.
    pub(super) fn apply_variable_commands(&mut self, ctx: &PipelineContext<'_>) {
        let Some(events) = ctx.events::<VariableCommand>() else {
            return;
        };
        for cmd in events.read(&mut self.variable_cursor) {
            let Some(slot) = self.var_table.slot_of(&cmd.name) else {
                tracing::warn!("a UI action names the undeclared variable {:?}", cmd.name);
                continue;
            };
            let current = self.vars[slot as usize];
            let next = ui_value(current)
                .and_then(|value| cmd.op.apply(Some(&value)))
                .and_then(|value| val_of(&value, current));
            if let Some(next) = next {
                self.vars[slot as usize] = next;
            } else {
                tracing::warn!("{:?} does not apply to var:{}", cmd.op, cmd.name);
            }
        }
    }

    // Republish the variables' values when any changed since the last
    // publish. Once published, only the changed values are rewritten, in
    // place, so a variable a behavior moves every tick costs no allocation.
    pub(super) fn publish_variables(&mut self, ctx: &mut PipelineContext<'_>) {
        if self.published_vars == self.vars {
            return;
        }
        let names = self.var_table.names();
        let same_table = self.published_vars.len() == self.vars.len();
        if same_table && let Some(published) = ctx.resource_mut::<VariableValues>() {
            for (i, (name, val)) in names.iter().zip(&self.vars).enumerate() {
                if self.published_vars.get(i) == Some(val) {
                    continue;
                }
                match published.values.get_mut(name) {
                    Some(slot) => write_value(*val, slot),
                    None => {
                        if let Some(value) = ui_value(*val) {
                            published.values.insert(name.clone(), value);
                        }
                    }
                }
            }
            published.revision = published.revision.wrapping_add(1);
        } else {
            let values: BTreeMap<String, UiValue> = names
                .iter()
                .zip(&self.vars)
                .filter_map(|(name, val)| Some((name.clone(), ui_value(*val)?)))
                .collect();
            let revision = ctx
                .resource::<VariableValues>()
                .map_or(0, |v| v.revision.wrapping_add(1));
            ctx.insert_resource(VariableValues { values, revision });
        }
        self.published_vars.clone_from(&self.vars);
    }
}
