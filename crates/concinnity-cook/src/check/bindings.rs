// Bindings checked against the world: each one names a value the world has,
// and each use fits that value. A `var:` binding names a declared variable (or,
// in a world without a `Variables` table, one a behavior uses); a `state:`
// binding names a value a `UiStateValues` declares; a setting is stepped and
// set as its row allows. An action's op, a slider's value, and what a widget
// selects, shows, or edits must fit the value's type, and a slider region
// fires no action. A list a widget shows is one a `UiStateValues` names.

use std::collections::{HashMap, HashSet};

use concinnity_engine::components::Binding;
use concinnity_engine::settings::{SettingKey, SettingKind};
use serde_json::Value;

use super::behavior::DeclaredVars;
use crate::authoring::registry::RegisteredType;
use crate::authoring::world::WorldJsonlAsset;
use crate::world_builder::Diagnostic;

// The most option steps one `step` may take on a setting whose options are
// only known at runtime.
const MAX_SETTING_STEPS: u64 = 16;

// What an asset does with a binding.
enum Use<'a> {
    Set(&'a Value),
    Flip,
    Add,
    Step(i64),
    // A label showing the value.
    Show,
    // A slider region dragging it from 0 to 1.
    Slide,
    // A list or tabs selecting a row by its index.
    Index,
    // A checkbox showing and flipping it: a boolean, or a setting of two
    // options.
    Check,
    // A menu or dialog opened while it is true: a boolean value.
    Open,
    // A text field or area editing it.
    Edit,
    // A scroll group collapsing on it.
    Collapse,
}

// What the world declares that bindings name.
struct Declared<'a> {
    vars: &'a DeclaredVars,
    // Variables a behavior uses, for a world without a `Variables` table.
    used_vars: HashSet<String>,
    // Each declared UI state value's type.
    state: HashMap<String, &'static str>,
    // The lists the app fills.
    lists: HashSet<String>,
}

pub(crate) fn check_bindings(
    assets: &[WorldJsonlAsset],
    vars: &DeclaredVars,
    errors: &mut Vec<Diagnostic>,
) {
    let declared = Declared {
        vars,
        used_vars: behavior_vars(assets),
        state: declared_state(assets, errors),
        lists: declared_lists(assets),
    };
    for asset in assets {
        let fail =
            |message: String| Diagnostic::asset(asset.asset_type.as_str(), &asset.id, message);
        let set = |field: &str| asset.args.get(field).is_some_and(|v| !v.is_null());
        if asset.asset_type == RegisteredType::HitRegion && set("slider") && set("action") {
            errors.push(fail(
                "a slider region drags its value and fires no `action`; give the action its own region".into(),
            ));
        }
        let mut check = |text: &Value, field: &str, how: Use<'_>| {
            let Some(text) = text.as_str() else {
                return;
            };
            if let Err(e) = Binding::parse(text).and_then(|b| declared.fits(&b, &how)) {
                errors.push(fail(format!("`{field}`: {e}")));
            }
        };
        let arg = |field: &str| asset.args.get(field);
        match asset.asset_type {
            RegisteredType::HitRegion | RegisteredType::KeyBinding => {
                if let Some(b) = arg("slider") {
                    check(b, "slider", Use::Slide);
                }
                if let Some(action) = arg("action") {
                    check_action(action, "action", &mut check);
                }
            }
            RegisteredType::List | RegisteredType::Tabs => {
                if let Some(b) = arg("selected") {
                    check(b, "selected", Use::Index);
                }
            }
            RegisteredType::Checkbox => {
                if let Some(b) = arg("checked") {
                    check(b, "checked", Use::Check);
                }
            }
            RegisteredType::Button => {
                if let Some(b) = arg("text_binding") {
                    check(b, "text_binding", Use::Show);
                }
                if let Some(b) = arg("active") {
                    check(b, "active", Use::Check);
                }
                if let Some(action) = arg("action") {
                    check_action(action, "action", &mut check);
                }
            }
            RegisteredType::TextArea | RegisteredType::TextField => {
                if let Some(b) = arg("text") {
                    check(b, "text", Use::Edit);
                }
            }
            RegisteredType::Menu | RegisteredType::Modal => {
                if let Some(b) = arg("open") {
                    check(b, "open", Use::Open);
                }
                let entries = arg("entries").or_else(|| arg("buttons"));
                for entry in entries.and_then(Value::as_array).into_iter().flatten() {
                    if let Some(action) = entry.get("action") {
                        check_action(action, "action", &mut check);
                    }
                }
            }
            RegisteredType::TextLabel => {
                if let Some(b) = asset.args.get("text_binding") {
                    check(b, "text_binding", Use::Show);
                }
            }
            RegisteredType::ScrollPanel => {
                let groups = asset.args.get("groups").and_then(Value::as_array);
                for group in groups.into_iter().flatten() {
                    if let Some(b) = group.get("collapsed_binding") {
                        check(b, "collapsed_binding", Use::Collapse);
                    }
                }
            }
            _ => {}
        }
        if matches!(
            asset.asset_type,
            RegisteredType::List | RegisteredType::Menu
        ) && let Some(name) = arg("items").and_then(Value::as_str)
            && !name.is_empty()
            && !declared.lists.contains(name)
        {
            errors.push(fail(format!(
                "`items`: `{name}` names no list a UiStateValues declares"
            )));
        }
    }
}

// Check the bindings one action changes.
fn check_action(action: &Value, field: &str, check: &mut impl FnMut(&Value, &str, Use<'_>)) {
    let Some(action) = action.as_object() else {
        return;
    };
    if let Some(b) = action.get("flip") {
        check(b, field, Use::Flip);
    }
    for verb in ["set", "add", "step"] {
        let Some(op) = action.get(verb) else {
            continue;
        };
        let how = match verb {
            "set" => Use::Set(op.get("value").unwrap_or(&Value::Null)),
            "add" => Use::Add,
            _ => Use::Step(op.get("by").and_then(Value::as_i64).unwrap_or(0)),
        };
        if let Some(b) = op.get("binding") {
            check(b, field, how);
        }
    }
}

// The lists every UiStateValues names.
fn declared_lists(assets: &[WorldJsonlAsset]) -> HashSet<String> {
    assets
        .iter()
        .filter(|a| a.asset_type == RegisteredType::UiStateValues)
        .filter_map(|a| a.args.get("lists").and_then(Value::as_array))
        .flatten()
        .filter_map(|list| list.get("name").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

impl Declared<'_> {
    // Whether `how` fits the value `binding` names.
    fn fits(&self, binding: &Binding, how: &Use<'_>) -> Result<(), String> {
        let text = binding.text();
        match binding {
            Binding::Setting(key) => setting_fits(*key, how).map_err(|e| format!("{text}: {e}")),
            Binding::Var(name) => {
                if matches!(how, Use::Collapse) {
                    return Err(format!("{text}: a group collapses on a `state:` binding"));
                }
                if !self.vars.is_declared() {
                    // An undeclared variable's type is whatever its behaviors
                    // make of it, so only its name is checked.
                    return if self.used_vars.contains(name) {
                        Ok(())
                    } else {
                        Err(format!(
                            "{text} names a variable no Variables entry declares and no behavior uses"
                        ))
                    };
                }
                let ty = self.vars.type_name(name).ok_or_else(|| {
                    format!("{text} names no variable the world's Variables declares")
                })?;
                value_fits(ty, how).map_err(|e| format!("{text}: {e}"))
            }
            Binding::State(name) => {
                let ty = self
                    .state
                    .get(name)
                    .copied()
                    .ok_or_else(|| format!("{text} names a value no UiStateValues declares"))?;
                value_fits(ty, how).map_err(|e| format!("{text}: {e}"))
            }
        }
    }
}

// Whether `how` fits a setting: any is shown; a cycle row is flipped, stepped,
// or set to an option index; a slider is set to a fraction or dragged; nothing
// is added to.
fn setting_fits(key: SettingKey, how: &Use<'_>) -> Result<(), String> {
    let cycle = key.kind() == SettingKind::Cycle;
    let options = concinnity_engine::settings::options(key).map(<[_]>::len);
    match how {
        Use::Slide if key.kind() != SettingKind::Slider => {
            Err("only a slider setting is dragged".into())
        }
        Use::Collapse => Err("a group collapses on a `state:` binding".into()),
        Use::Index | Use::Edit | Use::Open => Err(
            "a widget selects, edits, or opens on a `state:` or `var:` value, not a setting".into(),
        ),
        Use::Check if options != Some(2) => {
            Err("a checkbox shows a setting of exactly two options".into())
        }
        Use::Add => Err("a setting is stepped or set, not added to".into()),
        Use::Flip | Use::Step(_) if !cycle => Err("only a setting with options is stepped".into()),
        Use::Show | Use::Slide | Use::Flip | Use::Check => Ok(()),
        Use::Step(by) => {
            let reach = options.map_or(MAX_SETTING_STEPS, |n| n.saturating_sub(1) as u64);
            if *by == 0 || by.unsigned_abs() > reach {
                Err(format!(
                    "a step is 1 to {reach} options either way, not {by}"
                ))
            } else {
                Ok(())
            }
        }
        Use::Set(value) => match (value_type(value), cycle) {
            (Some("int"), true) => {
                let index = value.as_i64().unwrap_or(-1);
                match (usize::try_from(index), options) {
                    (Err(_), _) => Err(format!("option {index} does not exist")),
                    (Ok(i), Some(n)) if i >= n => {
                        Err(format!("option {index} is past its {n} options"))
                    }
                    _ => Ok(()),
                }
            }
            (Some("bool"), true) if options == Some(2) => Ok(()),
            (Some("float"), false) => match value.as_f64() {
                Some(f) if (0.0..=1.0).contains(&f) => Ok(()),
                _ => Err("a slider is set to a fraction from 0 to 1".into()),
            },
            _ if cycle => Err("a setting with options is set to an option index".into()),
            _ => Err("a slider is set to a fraction from 0 to 1".into()),
        },
    }
}

// Whether `how` fits a value of type `ty`.
fn value_fits(ty: &str, how: &Use<'_>) -> Result<(), String> {
    let numeric = matches!(ty, "int" | "float");
    match how {
        Use::Check | Use::Open if ty != "bool" => {
            Err(format!("a widget shows a bool here, not a {ty}"))
        }
        Use::Show | Use::Check | Use::Open => Ok(()),
        Use::Index if ty == "int" => Ok(()),
        Use::Index => Err(format!("a row index is a whole number, not a {ty}")),
        Use::Edit if ty == "text" => Ok(()),
        Use::Edit => Err(format!("edited text is text, not a {ty}")),
        Use::Slide if ty == "float" => Ok(()),
        Use::Slide => Err(format!("a slider drags a float, not a {ty}")),
        Use::Flip | Use::Collapse if ty == "bool" => Ok(()),
        Use::Flip | Use::Collapse => Err(format!("a {ty} is not flipped; only a bool is")),
        Use::Add | Use::Step(_) if numeric => Ok(()),
        Use::Add | Use::Step(_) => Err(format!("a {ty} is not added to or stepped")),
        Use::Set(value) => match value_type(value) {
            Some(set) if set == ty => Ok(()),
            Some(set) => Err(format!("a {ty} is not set to a {set}")),
            None => Err(format!("a {ty} is set to a {ty} value")),
        },
    }
}

// The type a JSON value sets: "bool", "int", "float", or "text".
fn value_type(value: &Value) -> Option<&'static str> {
    match value {
        Value::Bool(_) => Some("bool"),
        Value::Number(n) if n.as_i64().is_some_and(|i| i32::try_from(i).is_ok()) => Some("int"),
        Value::Number(n) if n.is_f64() => Some("float"),
        Value::String(_) => Some("text"),
        _ => None,
    }
}

// Every UI state value the world declares, and its type. A name declared twice
// is reported.
fn declared_state(
    assets: &[WorldJsonlAsset],
    errors: &mut Vec<Diagnostic>,
) -> HashMap<String, &'static str> {
    let mut state = HashMap::new();
    for asset in assets
        .iter()
        .filter(|a| a.asset_type == RegisteredType::UiStateValues)
    {
        let values = asset.args.get("values").and_then(Value::as_array);
        for value in values.into_iter().flatten() {
            let Some(name) = value.get("name").and_then(Value::as_str) else {
                continue;
            };
            let Some(ty) = value.get("value").and_then(value_type) else {
                errors.push(Diagnostic::asset(
                    "UiStateValues",
                    &asset.id,
                    format!("`{name}` starts at no bool, number, or text"),
                ));
                continue;
            };
            if state.insert(name.to_string(), ty).is_some() {
                errors.push(Diagnostic::asset(
                    "UiStateValues",
                    &asset.id,
                    format!("`{name}` is declared more than once"),
                ));
            }
        }
    }
    state
}

// Every variable a behavior names: in a `set`, a `var` read, or its `on`.
fn behavior_vars(assets: &[WorldJsonlAsset]) -> HashSet<String> {
    fn walk(value: &Value, out: &mut HashSet<String>) {
        match value {
            Value::Object(map) => {
                for (key, inner) in map {
                    if matches!(key.as_str(), "var" | "variable")
                        && let Some(name) = inner.as_str()
                    {
                        out.insert(name.to_string());
                    }
                    walk(inner, out);
                }
            }
            Value::Array(items) => items.iter().for_each(|v| walk(v, out)),
            _ => {}
        }
    }
    let mut out = HashSet::new();
    for asset in assets
        .iter()
        .filter(|a| a.asset_type == RegisteredType::Behavior)
    {
        walk(&asset.args, &mut out);
    }
    out
}

#[cfg(test)]
mod tests;
